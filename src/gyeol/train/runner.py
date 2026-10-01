"""The training runner behind ``gyeol train <task> --config <yaml>`` (revision B2–B8).

One loop for every task:

* **data** — the prepared cache (``gyeol prepare``; synthetic data or
  manifests listed in the config are prepared on first use), split by
  singer into train / validation / test (saved to ``split.json``), streamed
  from disk in random-length crops (:mod:`gyeol.train.stream`);
* **device** — ``auto`` → CUDA if available, else CPU in float32, with a
  configurable thread count (:mod:`gyeol.train.device`);
* **components** — ``freeze | finetune | scratch`` per component, initial
  weights from local checkpoints through the license gate, optional
  gradient checkpointing (:mod:`gyeol.train.components`);
* **memory** — gradient accumulation over ``optim.grad_accum`` batches;
* **interrupt / resume** — the full training state (modules, optimizers,
  schedulers, RNG states, epoch / step / data position, early-stopping
  bookkeeping) is written atomically every ``run.save_every`` steps and on
  SIGINT / SIGTERM; ``--resume`` continues exactly from there;
* **validation** — every ``run.val_every`` steps on the singer-disjoint
  validation split, early stopping after ``run.patience`` validations
  without improvement, the best weights kept as a release checkpoint
  (license lineage embedded);
* **progress** — ETA on stdout, ``train_log.csv`` and ``val_log.csv``, audio
  samples for generative tasks;
* **report** — each run ends with ``report.json`` / ``report.md``: the best
  weights evaluated on the unseen (test) singers.
"""

from __future__ import annotations

import csv
import json
import math
import signal
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..core.license import Profile, lookup
from .components import Lineage, apply_mode, enable_gradient_checkpointing, keep_frozen_in_eval, load_initial_weights, param_groups
from .config import TrainConfig
from .device import resolve_device
from .state import RunPosition, atomic_write_text, load_training_state, save_training_state, seed_everything
from .stream import CropSpec, StreamingDataset, loader, split_by_singer
from .tasks import TASK_CLASSES, DataInfo, label_tasks

STATE_FILE = "state.pt"


@dataclass
class RunResult:
    status: str  # "finished" | "early_stopped" | "interrupted"
    out_dir: Path
    position: RunPosition
    best_checkpoint: Path | None
    report: dict | None


def prepare_data(cfg: TrainConfig, progress: Callable[[str], None] | None = None) -> Path:
    """Make sure the cache holds the configured data (resumable; already prepared items are skipped)."""
    from ..data.prepare import PrepareConfig, prepare, synthetic_manifest

    cache = Path(cfg.data.cache)
    pc = dict(cfg.data.prepare)
    if "features" in pc:
        pc["features"] = tuple(pc["features"])
    pcfg = PrepareConfig(**pc)
    manifests = list(cfg.data.manifests)
    if cfg.data.synthetic is not None:
        s = dict(cfg.data.synthetic)
        manifests.append(synthetic_manifest(cache / "synthetic_src", sr=pcfg.sr, **s))
    if manifests:
        rep = prepare(manifests, cache, Profile(cfg.profile), pcfg)
        if progress:
            progress(f"[gyeol prepare] {rep.summary()}")
    if not (cache / "index.jsonl").exists():
        raise FileNotFoundError(f"no prepared data in {cache}: run `gyeol prepare --manifest ... --out {cache}` or list manifests in the config")
    return cache


class _Stop:
    """SIGINT / SIGTERM → finish the current step, save, and stop."""

    def __init__(self):
        self.requested = False
        self._old = {}

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._old[sig] = signal.signal(sig, self._handler)
            except ValueError:  # not in the main thread
                pass
        return self

    def _handler(self, signum, frame):
        if self.requested:  # second signal: give up immediately
            raise KeyboardInterrupt
        self.requested = True
        print(f"\n[gyeol train] signal {signum}: finishing the current step, saving, stopping (send again to abort)", flush=True)

    def __exit__(self, *exc):
        for sig, h in self._old.items():
            signal.signal(sig, h)
        return False


class Trainer:
    def __init__(self, cfg: TrainConfig, *, resume: bool = False, out_dir: str | Path | None = None,
                 log: Callable[[str], None] | None = print, stop_after_steps: int | None = None):
        """``stop_after_steps``: behave as if interrupted after that many optimizer steps (tests, time-boxed runs)."""
        self.cfg = cfg
        self.out = Path(out_dir or cfg.run.out)
        self.resume = resume
        self.log = log or (lambda s: None)
        self.stop_after_steps = stop_after_steps

    # ------------------------------------------------------------ setup
    def _setup(self):
        cfg = self.cfg
        self.out.mkdir(parents=True, exist_ok=True)
        state_path = self.out / STATE_FILE
        if state_path.exists() and not self.resume:
            raise FileExistsError(f"{self.out} already holds a run; pass --resume to continue it or choose another run.out")
        if self.resume and not state_path.exists():
            raise FileNotFoundError(f"--resume: no training state at {state_path}")
        if self.resume:
            saved = json.loads((self.out / "config.json").read_text(encoding="utf-8"))
            if saved != json.loads(json.dumps(cfg.as_dict(), default=str)):
                diff = sorted(k for k in set(saved) | set(cfg.as_dict()) if saved.get(k) != json.loads(json.dumps(cfg.as_dict(), default=str)).get(k))
                self.log(f"[gyeol train] note: the config differs from the saved run in {diff}; resuming with the new values")
        atomic_write_text(json.dumps(cfg.as_dict(), indent=1, default=str), self.out / "config.json")
        self.spec = resolve_device(cfg.device, cfg.threads, cfg.mixed_precision)
        seed_everything(cfg.seed)
        torch.use_deterministic_algorithms(self.spec.is_cpu, warn_only=True)
        self.cache = prepare_data(cfg, self.log)
        from ..data.prepare import read_index

        rows = read_index(self.cache)
        if cfg.data.datasets:
            rows = [r for r in rows if r["dataset"] in cfg.data.datasets]
        for ds in sorted({r["dataset"] for r in rows}):
            from ..core.license import require_allowed

            require_allowed(lookup(ds), Profile(cfg.profile), announce=False)  # the cache may hold data of another profile
        split_file = self.out / "split.json"
        if split_file.exists():
            ids = json.loads(split_file.read_text(encoding="utf-8"))
            by_id = {r["id"]: r for r in rows}
            self.splits = {k: [by_id[i] for i in v if i in by_id] for k, v in ids.items()}
        else:
            self.splits = split_by_singer(rows, cfg.data.split, cfg.data.split_seed)
            atomic_write_text(json.dumps({k: [r["id"] for r in v] for k, v in self.splits.items()}, indent=1), split_file)
        tr = self.splits.get("train", [])
        if not tr:
            raise ValueError("the training split is empty")
        overlap = {r.get("singer") for r in tr} & {r.get("singer") for k in ("val", "test") for r in self.splits.get(k, [])} - {"", None}
        if overlap:
            raise ValueError(f"singers in both train and validation/test: {sorted(overlap)}")
        prep = json.loads((self.cache / "prepare.json").read_text(encoding="utf-8"))
        with np.load(self.cache / "items" / f"{tr[0]['id']}.npz") as z:
            feat_dims = {k.split("/", 1)[1]: int(z[k].shape[1]) for k in z.files if k.startswith("feat/")}
            n_ap = int(z["ap_bands"].shape[1])
        tasks, quals = label_tasks(tr)
        self.info = DataInfo(int(prep["sr"]), int(prep["hop"]), n_ap, feat_dims, sorted({r.get("singer") or "" for r in tr} - {""}),
                             tasks, quals, sorted({r["dataset"] for r in rows}), {r["dataset"]: r.get("license", "") for r in rows})
        self.task = TASK_CLASSES[cfg.task](cfg, self.info, self.spec.device)
        self.task.build()
        comps = self.task.component_modules()
        self.plan = cfg.plan()
        self.plan.validate(tuple(comps))
        self.lineage = Lineage(assets=list(self.info.datasets))
        for name, spec in self.plan.init.items():
            if self.plan.mode(name) == "scratch":
                continue
            load_initial_weights(comps[name], spec, Profile(cfg.profile), self.lineage,
                                 loader=lambda m, sd, n=name: self.task.load_init(n, m, sd))
        for name, m in comps.items():
            apply_mode(m, self.plan.mode(name))
        for name in self.plan.checkpointing:
            n = enable_gradient_checkpointing(comps[name])
            self.log(f"[gyeol train] gradient checkpointing on {name}: {n} blocks")
        for m in self.task.state_modules().values():
            m.to(self.spec.device)
        o = cfg.optim
        self.optimizers, self.schedulers = {}, {}
        for oname, names in self.task.optimizer_groups().items():
            lr = o.d_lr if (oname == "d" and o.d_lr) else o.lr
            groups = param_groups({n: comps[n] for n in names}, self.plan, lr)
            if not groups:
                continue
            opt = torch.optim.AdamW(groups, lr=lr, betas=tuple(o.betas), weight_decay=o.weight_decay)
            self.optimizers[oname] = opt
            self.schedulers[oname] = self._scheduler(opt)
        if not self.optimizers:
            raise ValueError("every component is frozen: nothing to train")
        self.position = RunPosition()
        if self.resume:
            self.position, _ = load_training_state(state_path, modules=self.task.state_modules(), optimizers=self.optimizers,
                                                   schedulers=self.schedulers, map_location=self.spec.device)
            self.log(f"[gyeol train] resumed at epoch {self.position.epoch}, step {self.position.step}, "
                     f"data position {self.position.data_position}")
        n_params = sum(p.numel() for m in self.task.state_modules().values() for p in m.parameters())
        n_train = sum(p.numel() for g in self.optimizers.values() for grp in g.param_groups for p in grp["params"])
        self.log(f"[gyeol train] {cfg.task}: {n_params / 1e6:.2f} M parameters ({n_train / 1e6:.2f} M trained) on "
                 f"{self.spec.describe()}; {len(tr)} train / {len(self.splits.get('val', []))} val / {len(self.splits.get('test', []))} "
                 f"test items, {len(self.info.train_singers)} train singers")

    def _scheduler(self, opt):
        o, total = self.cfg.optim, max(1, self.cfg.run.max_steps)

        def f(step: int) -> float:
            w = min(1.0, (step + 1) / o.warmup_steps) if o.warmup_steps else 1.0
            if o.schedule == "cosine":
                return w * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))
            if o.schedule == "exponential":
                return w * o.gamma**step
            return w

        if o.schedule not in ("constant", "cosine", "exponential"):
            raise ValueError(f"unknown schedule {o.schedule!r}")
        return torch.optim.lr_scheduler.LambdaLR(opt, f)

    # ------------------------------------------------------------ helpers
    def _dataset(self, split: str, epoch: int, start: int = 0, shuffle: bool = True) -> StreamingDataset:
        d = self.cfg.data
        feats = tuple(self.info.feature_dims) if self.cfg.task == "heads" else ()
        return StreamingDataset(self.cache, self.splits.get(split, []), d.batch_size, CropSpec(*d.crop_frames), seed=self.cfg.seed,
                                epoch=epoch, start_position=start, shuffle=shuffle, features=feats)

    def _val_batches(self, split: str = "val"):
        ds = self._dataset(split, epoch=0, shuffle=False)
        n = self.cfg.run.val_batches or ds.n_batches()
        return [ds.batch(b) for b in range(min(n, ds.n_batches()))]

    def _rng_devices(self) -> list:
        return [self.spec.device.index or 0] if self.spec.device.type == "cuda" else []

    def _set_train(self, flag: bool) -> None:
        for m in self.task.state_modules().values():
            m.train(flag)
        if flag:
            keep_frozen_in_eval(self.task.component_modules(), self.plan)

    def _save_state(self) -> None:
        save_training_state(self.out / STATE_FILE, modules=self.task.state_modules(), optimizers=self.optimizers,
                            schedulers=self.schedulers, position=self.position, config=self.cfg.as_dict(),
                            extra={"info": {"sr": self.info.sr, "hop": self.info.hop, "datasets": self.info.datasets}})

    def _release(self, name: str) -> Path:
        from .checkpoint import save_checkpoint

        path = self.out / f"{name}.pt"
        save_checkpoint(path, self.task.release_state(), name=f"{self.cfg.task}-{name}", sources=self.lineage.assets,
                        config=self.cfg.as_dict(), profile=Profile(self.cfg.profile), parents=self.lineage.parents,
                        extra={"task": self.cfg.task, "step": self.position.step, "components": dict(self.plan.modes)})
        return path

    def _csv(self, name: str, row: dict, fields: list[str]) -> None:
        path = self.out / name
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})

    def _validate(self) -> bool:
        """Validate, record, keep the best weights; returns True when early stopping triggers."""
        self._set_train(False)
        with torch.random.fork_rng(devices=self._rng_devices()):  # validation must not perturb the training RNG stream
            batches = self._val_batches()
            metrics = self.task.validate(batches) if batches else {"score": float("nan")}
            if self.task.generative and batches and not self.cfg.run.sample_every:
                self.task.samples(batches[0], self.out, self.position.step)
        self._set_train(True)
        p = self.position
        score = float(metrics.get("score", float("nan")))
        improved = math.isfinite(score) and (p.best_metric is None or score < p.best_metric - self.cfg.run.min_delta)
        if improved:
            p.best_metric, p.best_step, p.bad_validations = score, p.step, 0
            self._release("best")
        else:
            p.bad_validations += 1
        rec = {"step": p.step, "epoch": p.epoch, **{k: float(v) for k, v in metrics.items()}, "best": improved}
        p.history.append(rec)
        self._csv("val_log.csv", rec, ["step", "epoch", "score", "best"] + sorted(k for k in metrics if k != "score"))
        self.log(f"[val] step {p.step}: " + ", ".join(f"{k} {v:.4g}" for k, v in metrics.items()) + (" *best*" if improved else ""))
        return bool(self.cfg.run.patience) and p.bad_validations >= self.cfg.run.patience

    # ------------------------------------------------------------ main loop
    def run(self) -> RunResult:
        self._setup()
        cfg, p = self.cfg, self.position
        if p.finished:
            self.log("[gyeol train] this run has already finished")
            return RunResult("finished", self.out, p, self.out / "best.pt", self._read_report())
        accum = cfg.optim.grad_accum
        steps_per_epoch = max(1, self._dataset("train", 0).n_batches() // accum)
        total_steps = min(cfg.run.max_steps, cfg.run.max_epochs * steps_per_epoch)
        times: deque[float] = deque(maxlen=50)
        self._set_train(True)
        for o in self.optimizers.values():
            o.zero_grad(set_to_none=True)
        status = "finished"
        t_start, elapsed0 = time.time(), p.elapsed_s
        loss_fields = ["step", "epoch", "elapsed_s", "lr", "eta_s"] + list(self.task.loss_names)
        with _Stop() as stop:
            while p.step < total_steps and p.epoch < cfg.run.max_epochs and status == "finished":
                ds = self._dataset("train", p.epoch, start=p.data_position)
                t_last = time.time()
                for batch in loader(ds, cfg.data.num_workers):
                    losses = self.task.train_step(batch, 1.0 / accum, p.step)
                    p.micro_step += 1
                    p.data_position = batch["position_end"]
                    if p.micro_step % accum:
                        continue
                    for opt in self.optimizers.values():
                        params = [q for g in opt.param_groups for q in g["params"] if q.grad is not None]
                        if cfg.optim.grad_clip and params:
                            torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    for s in self.schedulers.values():
                        s.step()
                    p.step += 1
                    now = time.time()
                    times.append(now - t_last)
                    t_last = now
                    p.elapsed_s = elapsed0 + now - t_start
                    eta = float(np.mean(times)) * (total_steps - p.step)
                    if p.step % cfg.run.log_every == 0 or p.step == 1:
                        lr = next(iter(self.optimizers.values())).param_groups[0]["lr"]
                        row = {"step": p.step, "epoch": p.epoch, "elapsed_s": p.elapsed_s, "lr": lr, "eta_s": eta, **losses}
                        self._csv("train_log.csv", row, loss_fields)
                        self.log(f"[train] step {p.step}/{total_steps} epoch {p.epoch} loss {losses.get('total', float('nan')):.4g} "
                                 f"{times[-1]:.2f} s/step ETA {_hms(eta)}")
                    if self.task.generative and cfg.run.sample_every and p.step % cfg.run.sample_every == 0:
                        self._set_train(False)
                        with torch.random.fork_rng(devices=self._rng_devices()):
                            vb = self._val_batches()
                            if vb:
                                self.task.samples(vb[0], self.out, p.step)
                        self._set_train(True)
                    early = False
                    if cfg.run.val_every and p.step % cfg.run.val_every == 0:
                        early = self._validate()
                    if early:
                        status = "early_stopped"
                    elif self.stop_after_steps is not None and p.step >= self.stop_after_steps or stop.requested:
                        status = "interrupted"
                    if status != "finished" or p.step % cfg.run.save_every == 0 or p.step >= total_steps:
                        self._save_state()
                    if status != "finished" or p.step >= total_steps:
                        break
                else:
                    p.epoch += 1
                    p.data_position = 0
                    continue
                if status == "finished" and p.step >= total_steps:
                    # the epoch may have ended exactly here; the position is saved, nothing else to do
                    break
        if status == "interrupted":
            self._save_state()
            self.log(f"[gyeol train] interrupted at step {p.step}; continue with --resume")
            return RunResult(status, self.out, p, None, None)
        # end of run: final validation if the last step was not validated, then the report on unseen singers
        if not p.history or p.history[-1]["step"] != p.step:
            self._validate()
        report = self._final_report(status)
        p.finished = True
        self._save_state()
        return RunResult(status, self.out, p, self.out / "best.pt" if (self.out / "best.pt").exists() else None, report)

    def _read_report(self) -> dict | None:
        f = self.out / "report.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def _final_report(self, status: str) -> dict:
        from .checkpoint import load_checkpoint

        best = self.out / "best.pt"
        if best.exists():
            sd, _ = load_checkpoint(best, Profile(self.cfg.profile))
            for name, m in self.task.state_modules().items():
                prefix = f"{name}."
                m.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}, strict=False)
        self._set_train(False)
        with torch.random.fork_rng(devices=self._rng_devices()):
            test = self._val_batches("test")
            test_metrics = self.task.evaluate(test) if test else {}
            final = self.task.finalize(self.cache, self.splits, self.out)
        if final:
            self._release("best")  # calibration etc. become part of the released weights
        p = self.position
        report = {
            "task": self.cfg.task, "status": status, "steps": p.step, "epochs": p.epoch, "elapsed_s": p.elapsed_s,
            "best_step": p.best_step, "best_validation_score": p.best_metric,
            "test_singers": sorted({r.get("singer") for r in self.splits.get("test", [])} - {None, ""}),
            "validation_singers": sorted({r.get("singer") for r in self.splits.get("val", [])} - {None, ""}),
            "test": test_metrics, "finalize": final, "device": self.spec.describe(), "profile": self.cfg.profile,
            "license_lineage": {"datasets": self.lineage.assets, "parents": [x.name for x in self.lineage.parents]},
            "components": {n: self.plan.mode(n) for n in self.task.components},
        }
        atomic_write_text(json.dumps(report, indent=1, default=float), self.out / "report.json")
        lines = [f"# {self.cfg.task} — run report", "", f"- status: {status}; {p.step} steps, {p.epoch} epochs, {_hms(p.elapsed_s)}",
                 f"- device: {self.spec.describe()}", f"- best validation score {p.best_metric} at step {p.best_step}",
                 f"- evaluated on unseen singers: {', '.join(report['test_singers']) or '(none — no test split)'}", "",
                 "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v:.4g} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in test_metrics.items()]
        if final:
            lines += ["", "## finalize", "", "```json", json.dumps(final, indent=1, default=float), "```"]
        atomic_write_text("\n".join(lines) + "\n", self.out / "report.md")
        self.log(f"[gyeol train] report: {self.out / 'report.md'}")
        return report


def _hms(s: float) -> str:
    if not math.isfinite(s):
        return "?"
    s = int(s)
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def train(cfg: TrainConfig, *, resume: bool = False, out_dir: str | Path | None = None, log: Callable[[str], None] | None = print,
          stop_after_steps: int | None = None) -> RunResult:
    """Run (or resume) one training run; see the module docstring."""
    return Trainer(cfg, resume=resume, out_dir=out_dir, log=log, stop_after_steps=stop_after_steps).run()


__all__ = ["RunResult", "Trainer", "prepare_data", "train"]

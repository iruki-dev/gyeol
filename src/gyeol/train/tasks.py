"""Training tasks for ``gyeol train <task>`` (revision B2).

Each task builds its modules from the run config and the prepared data,
turns streamed batches (:mod:`gyeol.train.stream`) into the inputs of the
existing step functions and losses, and defines validation, the final
evaluation on unseen singers, and (for generative tasks) audio samples.

=============  =====================================================  =========================================
task           components                                             objective
=============  =====================================================  =========================================
heads          heads                                                  task losses of :mod:`gyeol.train.heads`, then
                                                                      temperature / OOD calibration on validation singers
autoencoder    singer_encoder, env_encoder, residual_encoder,         :func:`gyeol.train.autoencoder.generator_step`
               acoustic, vocoder, discriminators                      (+ :func:`discriminator_step` on vocoded steps)
vocoder        vocoder, discriminators                                mel L1 + multi-resolution STFT (+ LSGAN, FM)
pitch          pitch (RMVPE)                                          BCE on Gaussian-blurred 360-bin targets
ssl            ssl, heads                                             head losses back-propagated into the SSL encoder
=============  =====================================================  =========================================

Labels for the head tasks come from the manifests (adapters' vocabulary):
``register`` (or a register name under ``phonation``) → the register
softmax; a quality under ``phonation`` → that sigmoid positive, and every
other quality the run's data labels anywhere → negative (unlabelled
qualities stay unknown); ``laryngeal`` → the laryngeal softmax.  Labels are
broadcast over voiced frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attributes.heads import LARYNGEAL, PHONATION_QUALITIES, REGISTER, FrameHeads, TaskSpec
from .config import TrainConfig

# ---------------------------------------------------------------- data description


@dataclass
class DataInfo:
    sr: int
    hop: int
    n_ap: int
    feature_dims: dict[str, int]
    train_singers: list[str]
    tasks: dict[str, TaskSpec]
    labelled_qualities: tuple[str, ...]
    datasets: list[str]
    licenses: dict[str, str] = field(default_factory=dict)


def label_tasks(rows: list[dict]) -> tuple[dict[str, TaskSpec], tuple[str, ...]]:
    """Head tasks that the rows carry labels for, and the qualities labelled anywhere."""
    has_reg = any(_register(r.get("labels", {})) is not None for r in rows)
    quals = sorted({q for r in rows for q in [r.get("labels", {}).get("phonation")] if q in PHONATION_QUALITIES},
                   key=PHONATION_QUALITIES.index)
    has_lar = any(r.get("labels", {}).get("laryngeal") in LARYNGEAL for r in rows)
    tasks: dict[str, TaskSpec] = {}
    if has_reg:
        tasks["register"] = TaskSpec(3, "softmax", REGISTER)
    if quals:
        tasks["phonation"] = TaskSpec(len(PHONATION_QUALITIES), "sigmoid", PHONATION_QUALITIES)
    if has_lar:
        tasks["laryngeal"] = TaskSpec(3, "softmax", LARYNGEAL)
    return tasks, tuple(quals)


def _register(labels: dict) -> int | None:
    for v in (labels.get("register"), labels.get("phonation")):
        if v in REGISTER:
            return REGISTER.index(v)
    return None


def head_targets(batch: dict, tasks: dict[str, TaskSpec], labelled: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """Frame targets (B, T[, K]) from utterance labels, on voiced frames only."""
    voiced = (batch["conf"]["f0_cents"] > 0) & batch["mask"]
    B, T = voiced.shape
    out = {}
    if "register" in tasks:
        y = torch.full((B, T), -1, dtype=torch.long)
        for i, lab in enumerate(batch["labels"]):
            r = _register(lab)
            if r is not None:
                y[i, voiced[i]] = r
        out["register"] = y
    if "phonation" in tasks:
        y = torch.full((B, T, len(PHONATION_QUALITIES)), float("nan"))
        for i, lab in enumerate(batch["labels"]):
            if "phonation" not in lab:
                continue
            row = torch.full((len(PHONATION_QUALITIES),), float("nan"))
            for q in labelled:
                row[PHONATION_QUALITIES.index(q)] = 0.0
            if lab["phonation"] in PHONATION_QUALITIES:
                row[PHONATION_QUALITIES.index(lab["phonation"])] = 1.0
            y[i, voiced[i]] = row
        out["phonation"] = y
    if "laryngeal" in tasks:
        y = torch.full((B, T), -1, dtype=torch.long)
        for i, lab in enumerate(batch["labels"]):
            if lab.get("laryngeal") in LARYNGEAL:
                y[i, voiced[i]] = LARYNGEAL.index(lab["laryngeal"])
        out["laryngeal"] = y
    return out


def head_loss(logits: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], tasks: dict[str, TaskSpec]) -> tuple[torch.Tensor, dict]:
    total, parts = None, {}
    for k, spec in tasks.items():
        if k not in targets:
            continue
        z, t = logits[k], targets[k].to(logits[k].device)
        if spec.kind == "softmax":
            m = t >= 0
            if not m.any():
                continue
            loss = F.cross_entropy(z[m], t[m])
        else:
            m = ~torch.isnan(t)
            if not m.any():
                continue
            loss = F.binary_cross_entropy_with_logits(z[m], t[m])
        parts[k] = float(loss.detach())
        total = loss if total is None else total + loss
    if total is None:
        total = sum(z.sum() * 0.0 for z in logits.values())
    return total, parts


def head_metrics(logits: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], tasks: dict[str, TaskSpec]) -> dict[str, tuple[float, int]]:
    """(sum, count) accumulators: register / laryngeal accuracy, phonation accuracy at 0.5."""
    acc = {}
    for k, spec in tasks.items():
        if k not in targets:
            continue
        z, t = logits[k], targets[k].to(logits[k].device)
        if spec.kind == "softmax":
            m = t >= 0
            acc[f"{k}_acc"] = (float((z.argmax(-1)[m] == t[m]).sum()), int(m.sum()))
        else:
            m = ~torch.isnan(t)
            acc[f"{k}_acc"] = (float(((z[m] > 0).float() == t[m]).sum()), int(m.sum()))
    return acc


def f0_hz(batch: dict) -> torch.Tensor:
    c, conf = batch["curves"]["f0_cents"], batch["conf"]["f0_cents"]
    ok = (conf > 0) & torch.isfinite(c) & batch["mask"]
    return torch.where(ok, 440.0 * 2 ** (torch.nan_to_num(c) / 1200.0), torch.zeros_like(c))


class _Acc:
    def __init__(self):
        self.s: dict[str, list[float]] = {}

    def add(self, k: str, v: float) -> None:
        """One per-batch value (averaged over batches)."""
        a = self.s.setdefault(k, [0.0, 0])
        a[0] += v
        a[1] += 1

    def add_pair(self, k: str, pair: tuple[float, int]) -> None:
        a = self.s.setdefault(k, [0.0, 0])
        a[0] += pair[0]
        a[1] += pair[1]

    def means(self) -> dict[str, float]:
        return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in self.s.items()}


# ---------------------------------------------------------------- base


class Task:
    name = ""
    components: tuple[str, ...] = ()
    generative = False
    loss_names: tuple[str, ...] = ("total",)

    def __init__(self, cfg: TrainConfig, info: DataInfo, device: torch.device):
        self.cfg, self.info, self.device = cfg, info, device
        self.m = cfg.model
        self.modules: dict[str, nn.Module] = {}

    # building
    def build(self) -> dict[str, nn.Module]:
        raise NotImplementedError

    def component_modules(self) -> dict[str, nn.Module]:
        """Component name → module (may share submodules with :meth:`state_modules`)."""
        raise NotImplementedError

    def state_modules(self) -> dict[str, nn.Module]:
        return self.modules

    def optimizer_groups(self) -> dict[str, list[str]]:
        """Optimizer name → components it updates."""
        return {"main": list(self.components)}

    def load_init(self, component: str, module: nn.Module, sd: dict) -> None:
        res = module.load_state_dict(sd, strict=False)
        if res.missing_keys or res.unexpected_keys:
            raise ValueError(f"{component}: weights do not match: missing {res.missing_keys[:8]}, unexpected {res.unexpected_keys[:8]}")

    # steps
    def train_step(self, batch: dict, scale: float, step: int) -> dict[str, float]:
        raise NotImplementedError

    @torch.no_grad()
    def validate(self, batches) -> dict[str, float]:
        raise NotImplementedError

    def evaluate(self, batches) -> dict:
        return self.validate(batches)

    def finalize(self, cache: Path, splits: dict[str, list[dict]], out_dir: Path) -> dict:
        return {}

    def samples(self, batch: dict, out_dir: Path, step: int) -> list[Path]:
        return []

    def release_state(self) -> dict:
        return {f"{k}.{n}": v for k, m in self.state_modules().items() for n, v in m.state_dict().items()}


# ---------------------------------------------------------------- heads


class HeadsTask(Task):
    name = "heads"
    components = ("heads",)
    loss_names = ("total", "register", "phonation", "laryngeal")

    def build(self):
        if not self.info.tasks:
            raise ValueError("the training data carries no register / phonation / laryngeal labels")
        feat = self.m.get("features", "dsp")
        if feat not in self.info.feature_dims:
            raise ValueError(f"features {feat!r} are not in the prepared cache (have {sorted(self.info.feature_dims)})")
        self.feature = feat
        self.modules = {"heads": FrameHeads(self.info.feature_dims[feat], self.info.tasks, hidden=int(self.m.get("hidden", 128)),
                                            dropout=float(self.m.get("dropout", 0.1)))}
        return self.modules

    def component_modules(self):
        return {"heads": self.modules["heads"]}

    def _inputs(self, batch):
        return batch["feats"][self.feature].to(self.device)

    def train_step(self, batch, scale, step):
        logits, _ = self.modules["heads"](self._inputs(batch))
        loss, parts = head_loss(logits, head_targets(batch, self.info.tasks, self.info.labelled_qualities), self.info.tasks)
        (loss * scale).backward()
        return {"total": float(loss.detach()), **parts}

    @torch.no_grad()
    def validate(self, batches):
        acc = _Acc()
        for b in batches:
            logits, _ = self.modules["heads"](self._inputs(b))
            t = head_targets(b, self.info.tasks, self.info.labelled_qualities)
            loss, parts = head_loss(logits, t, self.info.tasks)
            acc.add("loss", float(loss))
            for k, v in head_metrics(logits, t, self.info.tasks).items():
                acc.add_pair(k, v)
        out = acc.means()
        out["score"] = out.get("loss", float("nan"))
        return out

    def _examples(self, cache: Path, rows: list[dict], limit: int = 400):
        from ..train.heads import FrameExample
        from .stream import CropSpec, StreamingDataset

        exs = []
        ds = StreamingDataset(cache, rows[:limit], 1, CropSpec(2, 1_000_000), shuffle=False, features=(self.feature,) if self.name == "heads" else ())
        for b in ds:
            feats = self._features_for_calibration(b)
            t = head_targets(b, self.info.tasks, self.info.labelled_qualities)
            targets = {k: v[0].numpy() for k, v in t.items()}
            exs.append(FrameExample(feats, targets, b["singer"][0]))
        return exs

    def _features_for_calibration(self, b) -> np.ndarray:
        return b["feats"][self.feature][0].numpy()

    def finalize(self, cache, splits, out_dir):
        """Calibrate on the validation singers; evaluate calibrated heads on the test singers."""
        from ..attributes.heads import expected_calibration_error
        from .heads import calibrate_heads

        train_ex = self._examples(cache, splits["train"], int(self.m.get("calibration_train_items", 200)))
        val_ex = self._examples(cache, splits.get("val", []))
        if not val_ex:
            return {"calibration": "skipped: no validation singers"}
        heads = calibrate_heads(self.modules["heads"], train_ex, val_ex, self.info.tasks, float(self.m.get("ood_quantile", 0.99)),
                                self.feature if self.name == "heads" else "ssl")
        self.calibrated = heads
        report = {"temperatures": heads.temperatures, "ood_threshold": None if heads.ood is None else float(heads.ood.threshold)}
        test_ex = self._examples(cache, splits.get("test", []))
        if test_ex:
            for k, spec in self.info.tasks.items():
                probs, ys = [], []
                for ex in test_ex:
                    if k not in ex.targets:
                        continue
                    out, unknown, _ = heads.predict(ex.features)
                    t = ex.targets[k]
                    m = (t >= 0) if spec.kind == "softmax" else ~np.isnan(t)
                    if spec.kind == "softmax":
                        probs.append(out[k][m])
                        ys.append(t[m])
                    else:
                        probs.append(out[k][m])
                        ys.append(t[m])
                if probs and sum(len(p) for p in probs):
                    p, y = np.concatenate(probs), np.concatenate(ys)
                    if spec.kind == "softmax":
                        report[f"test_{k}_acc"] = float(np.mean(p.argmax(-1) == y))
                        report[f"test_{k}_ece"] = float(expected_calibration_error(p, y.astype(int)))
                    else:
                        report[f"test_{k}_acc"] = float(np.mean((p > 0.5) == (y > 0.5)))
        (out_dir / "calibration.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
        return report

    def release_state(self):
        sd = super().release_state()
        cal = getattr(self, "calibrated", None)
        if cal is not None:
            sd["calibration.temperatures"] = torch.tensor([cal.temperatures.get(k, 1.0) for k in self.info.tasks])
        return sd


# ---------------------------------------------------------------- SSL fine-tuning


def build_ssl(model_cfg: dict) -> tuple[nn.Module, int]:
    """torchaudio HuBERT / WavLM base (weights from a fetched checkpoint via ``init.ssl``) or a tiny wav2vec2 for tests."""
    import torchaudio

    arch = model_cfg.get("arch", "tiny")
    if arch == "tiny":
        d = int(model_cfg.get("dim", 32))
        conv = [(d, 10, 5)] + [(d, 3, 2)] * 4 + [(d, 2, 2)] * 2
        m = torchaudio.models.wav2vec2_model(
            extractor_mode="group_norm", extractor_conv_layer_config=conv, extractor_conv_bias=False, encoder_embed_dim=d,
            encoder_projection_dropout=0.0, encoder_pos_conv_kernel=16, encoder_pos_conv_groups=4, encoder_num_layers=int(model_cfg.get("layers", 2)),
            encoder_num_heads=2, encoder_attention_dropout=0.0, encoder_ff_interm_features=2 * d, encoder_ff_interm_dropout=0.0,
            encoder_dropout=0.0, encoder_layer_norm_first=False, encoder_layer_drop=0.0, aux_num_out=None)
        return m, d
    builders = {"hubert_base": torchaudio.models.hubert_base, "wavlm_base": torchaudio.models.wavlm_base}
    if arch not in builders:
        raise ValueError(f"unknown SSL arch {arch!r}; one of tiny, {', '.join(builders)}")
    return builders[arch](), 768


class SSLTask(HeadsTask):
    name = "ssl"
    components = ("ssl", "heads")

    def build(self):
        if not self.info.tasks:
            raise ValueError("the training data carries no register / phonation / laryngeal labels")
        ssl, dim = build_ssl(self.m)
        self.layers = tuple(self.m.get("layers_used", (0, 1)))
        self.feature = "ssl"
        self.modules = {"ssl": ssl, "heads": FrameHeads(dim, self.info.tasks, hidden=int(self.m.get("hidden", 128)))}
        return self.modules

    def component_modules(self):
        return {"ssl": self.modules["ssl"], "heads": self.modules["heads"]}

    def load_init(self, component, module, sd):
        if component == "ssl":
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        super().load_init(component, module, sd)

    def _inputs(self, batch):
        from torchaudio.functional import resample

        wav = resample(batch["audio"].to(self.device), batch["sr"], 16000)
        feats, _ = self.modules["ssl"].extract_features(wav, num_layers=max(self.layers) + 1)
        h = torch.stack([feats[i] for i in self.layers]).mean(0)  # (B, T', D)
        return F.interpolate(h.transpose(1, 2), size=batch["frames"], mode="linear", align_corners=False).transpose(1, 2)

    @torch.no_grad()
    def _features_for_calibration(self, b) -> np.ndarray:
        return self._inputs(b)[0].cpu().numpy()


# ---------------------------------------------------------------- autoencoder


def _ae_config(info: DataInfo, m: dict, n_singers: int):
    from ..decoder.model import AutoencoderConfig

    kw = {k: v for k, v in m.items() if k in AutoencoderConfig.__dataclass_fields__}
    kw.update(sr=info.sr, hop=info.hop, n_ap=info.n_ap)
    kw["upsample_rates"] = tuple(kw["upsample_rates"]) if "upsample_rates" in kw else _rates_for(info.hop)
    if m.get("singer_adversary", True):
        kw["n_singers"] = n_singers
    return AutoencoderConfig.tiny(**kw) if m.get("size", "tiny") == "tiny" else AutoencoderConfig(**kw)


def _discriminators(m: dict) -> nn.ModuleList:
    from .losses import MultiPeriodDiscriminator, MultiScaleDiscriminator

    ch = int(m.get("disc_channels", 16))
    return nn.ModuleList([MultiPeriodDiscriminator(tuple(m.get("periods", (2, 3, 5, 7, 11))), ch=ch),
                          MultiScaleDiscriminator(int(m.get("scales", 3)), ch=ch)])


def _save_samples(out_dir: Path, step: int, ids: list[str], wavs: list[np.ndarray], refs: list[np.ndarray], sr: int, task: str,
                  profile: str) -> list[Path]:
    """Audio samples with a JSON sidecar marking them as AI-generated reconstructions of the training data."""
    from ..io import save_audio

    d = out_dir / "samples" / f"step_{step:07d}"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (iid, y, x) in enumerate(zip(ids, wavs, refs)):
        p = d / f"{i}_{iid}_generated.wav"
        save_audio(p, np.clip(y, -1, 1), sr)
        save_audio(d / f"{i}_{iid}_reference.wav", np.clip(x, -1, 1), sr)
        paths.append(p)
    (d / "samples.json").write_text(json.dumps({
        "ai_generated": True, "kind": f"{task} reconstruction of held-out validation audio (monitoring only, not for distribution)",
        "step": step, "items": ids, "profile": profile}, indent=1), encoding="utf-8")
    return paths


class AutoencoderTask(Task):
    name = "autoencoder"
    components = ("singer_encoder", "env_encoder", "residual_encoder", "acoustic", "vocoder", "discriminators")
    generative = True
    loss_names = ("total", "mel", "kl", "env", "singer", "leak", "stft", "reencode", "adv", "fm", "disc")

    def build(self):
        from ..decoder.model import GyeolAutoencoder

        self.singer_index = {s: i for i, s in enumerate(self.info.train_singers)}
        self.model = GyeolAutoencoder(_ae_config(self.info, self.m, len(self.singer_index)))
        self.discs = _discriminators(self.m)
        self.vocode_every = int(self.m.get("vocode_every", 4))
        self.adversarial = bool(self.m.get("adversarial", True))
        self.timbre_shift = bool(self.m.get("timbre_shift", True))
        self.modules = {"model": self.model, "discriminators": self.discs}
        return self.modules

    def component_modules(self):
        md = self.model
        return {"singer_encoder": nn.ModuleList([md.singer_enc, md.singer_env_adv]), "env_encoder": nn.ModuleList([md.env_enc]),
                "residual_encoder": nn.ModuleList([md.res_enc, md.leak]), "acoustic": nn.ModuleList([md.acoustic]),
                "vocoder": nn.ModuleList([md.vocoder]), "discriminators": self.discs}

    def optimizer_groups(self):
        return {"g": ["singer_encoder", "env_encoder", "residual_encoder", "acoustic", "vocoder"], "d": ["discriminators"]}

    def load_init(self, component, module, sd):
        # a component's weights may come from a full autoencoder checkpoint ("model.vocoder.…") or a bare module
        names = {"singer_encoder": ["singer_enc", "singer_env_adv"], "env_encoder": ["env_enc"], "residual_encoder": ["res_enc", "leak"],
                 "acoustic": ["acoustic"], "vocoder": ["vocoder"]}.get(component)
        if names is None:
            return super().load_init(component, module, sd)
        if any(k.startswith("model.") for k in sd):
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        if any(k.split(".")[0] in names for k in sd):
            sub = {f"{names.index(k.split('.')[0])}.{k.split('.', 1)[1]}": v for k, v in sd.items() if k.split(".")[0] in names}
        elif len(names) == 1:
            sub = {f"0.{k}": v for k, v in sd.items()}
        else:
            raise ValueError(f"{component}: cannot map checkpoint keys onto {names}")
        super().load_init(component, module, sub)

    def _batch(self, b: dict):
        from ..data.augment import timbre_shift
        from ..decoder.model import C_CHANNELS
        from .autoencoder import _SCALE, AEBatch
        from .losses import frame_weights

        mask = b["mask"]
        c = torch.stack([torch.nan_to_num(b["curves"][n]) * _SCALE.get(n, 1.0) for n in C_CHANNELS], -1)
        cm = torch.stack([(b["conf"][n] > 0) & mask for n in C_CHANNELS], -1)
        f0 = f0_hz(b)
        ap = torch.nan_to_num(b["curves"]["aperiodic_ratio"]) * 0.1
        voiced = (f0 > 0).numpy()
        w = torch.tensor(np.stack([frame_weights(b["curves"]["loudness_rel"][i].numpy(), voiced[i]) for i in range(len(voiced))]),
                         dtype=torch.float32) * mask.float()
        rough = (torch.nan_to_num(b["curves"]["subharmonic_ratio"]) / 0.5).clamp(0, 1)
        ids = [self.singer_index.get(s, -1) for s in b["singer"]]
        singer_id = torch.tensor(ids, dtype=torch.long) if all(i >= 0 for i in ids) and self.model.cfg.n_singers else None
        wr = None
        if self.timbre_shift and self.model.training:
            rng = np.random.default_rng([self.cfg.seed, b["epoch"], b["batch_index"], 7])
            n = b["audio"].shape[1]
            # the envelope warp needs a few STFT frames: pad short crops, cut back afterwards
            shifted = [timbre_shift(np.pad(x.numpy().astype(float), (0, max(0, 4096 - n))), b["sr"], rng)[0] for x in b["audio"]]
            wr = torch.tensor(np.stack([np.pad(y[:n], (0, max(0, n - len(y)))) for y in shifted]), dtype=torch.float32)
        ab = AEBatch(b["audio"], c, cm, f0, ap, w, wr, rough, singer_id, {})
        for k in ("wav", "c", "c_mask", "f0_hz", "aperiodic", "weights", "rough"):
            setattr(ab, k, getattr(ab, k).to(self.device))
        if ab.wav_for_residual is not None:
            ab.wav_for_residual = ab.wav_for_residual.to(self.device)
        if ab.singer_id is not None:
            ab.singer_id = ab.singer_id.to(self.device)
        return ab

    def _disc_trainable(self) -> bool:
        return any(p.requires_grad for p in self.discs.parameters())

    def train_step(self, batch, scale, step):
        from .autoencoder import discriminator_step, generator_step

        ab = self._batch(batch)
        vocode = self.vocode_every > 0 and step % self.vocode_every == 0
        use_adv = vocode and self.adversarial
        d_trainable = self._disc_trainable()
        for p in self.discs.parameters():
            p.requires_grad_(False)  # the generator's adversarial loss must not update the discriminators
        outs: dict = {}
        L = generator_step(self.model, ab, vocode=vocode, discriminators=list(self.discs) if use_adv else None, outputs=outs)
        (L["total"] * scale).backward()
        res = {k: float(v.detach()) for k, v in L.items()}
        if d_trainable:
            for p in self.discs.parameters():
                p.requires_grad_(True)
            if use_adv and "wav" in outs:
                d = discriminator_step(list(self.discs), outs["target"].detach(), outs["wav"].detach())
                (d * scale).backward()
                res["disc"] = float(d.detach())
        return res

    @torch.no_grad()
    def validate(self, batches):
        from .losses import weighted_mel_loss

        acc = _Acc()
        for b in batches:
            ab = self._batch(b)
            enc = self.model.encode(ab.wav, ab.c, ab.c_mask)
            T = enc["mel"].shape[1]
            dec = self.model.decode(enc["singer"], enc["env"], ab.c[:, :T], ab.c_mask[:, :T], enc["r"], ab.f0_hz[:, :T],
                                    ab.aperiodic[:, :T], ab.rough[:, :T], vocode=False)
            acc.add("mel_l1", float(weighted_mel_loss(dec["mel"], enc["mel"], ab.weights[:, :T])))
            acc.add("kl", float(enc["kl"].mean()))
        out = acc.means()
        out["score"] = out.get("mel_l1", float("nan"))
        return out

    @torch.no_grad()
    def evaluate(self, batches):
        from .losses import MultiResolutionSTFTLoss

        out = self.validate(batches)
        mr, acc = MultiResolutionSTFTLoss(), _Acc()
        for b in batches:
            y, x, ab = self._reconstruct(b)
            acc.add("mrstft", float(mr(y, x, ab.weights[:, : y.shape[1] // self.info.hop + 1], self.info.hop)))
        out.update(acc.means())
        return out

    @torch.no_grad()
    def _reconstruct(self, b):
        ab = self._batch(b)
        enc = self.model.encode(ab.wav, ab.c, ab.c_mask)
        T = enc["mel"].shape[1]
        y = self.model.decode(enc["singer"], enc["env"], ab.c[:, :T], ab.c_mask[:, :T], enc["r"], ab.f0_hz[:, :T], ab.aperiodic[:, :T],
                              ab.rough[:, :T], vocode=True, seed=0)["wav"]
        return y, ab.wav[:, : y.shape[1]], ab

    def samples(self, batch, out_dir, step):
        y, x, _ = self._reconstruct(batch)
        n = int(self.cfg.run.n_samples)
        return _save_samples(out_dir, step, batch["ids"][:n], [v.cpu().numpy() for v in y[:n]], [v.cpu().numpy() for v in x[:n]],
                             self.info.sr, self.name, self.cfg.profile)


# ---------------------------------------------------------------- vocoder


class VocoderTask(Task):
    name = "vocoder"
    components = ("vocoder", "discriminators")
    generative = True
    loss_names = ("total", "mel", "stft", "adv", "fm", "disc")

    def build(self):
        from ..decoder.vocoder import NSFVocoder
        from ..encoders.mel import LogMel

        size_tiny = self.m.get("size", "tiny") == "tiny"
        n_mels = int(self.m.get("n_mels", 32 if size_tiny else 80))
        ch = int(self.m.get("channels", 16 if size_tiny else 256))
        rates = tuple(self.m.get("upsample_rates", _rates_for(self.info.hop)))
        self.mel = LogMel(self.info.sr, self.info.hop, n_mels=n_mels).to(self.device)
        self.vocoder = NSFVocoder(n_mels, self.info.n_ap, self.info.sr, self.info.hop, ch, rates)
        self.discs = _discriminators(self.m)
        self.adversarial = bool(self.m.get("adversarial", True))
        self.w_mel, self.w_stft = float(self.m.get("w_mel", 45.0)), float(self.m.get("w_stft", 1.0))
        self.modules = {"vocoder": self.vocoder, "discriminators": self.discs}
        return self.modules

    def component_modules(self):
        return {"vocoder": self.vocoder, "discriminators": self.discs}

    def optimizer_groups(self):
        return {"g": ["vocoder"], "d": ["discriminators"]}

    def load_init(self, component, module, sd):
        if component == "vocoder" and any(k.startswith("model.vocoder.") for k in sd):
            sd = {k[len("model.vocoder."):]: v for k, v in sd.items() if k.startswith("model.vocoder.")}
        super().load_init(component, module, sd)

    def _inputs(self, b):
        x = b["audio"].to(self.device)
        mel = self.mel(x)
        T = min(mel.shape[1], b["frames"])
        rough = (torch.nan_to_num(b["curves"]["subharmonic_ratio"]) / 0.5).clamp(0, 1)
        return x, mel[:, :T], b["ap_bands"][:, :T].to(self.device), f0_hz(b)[:, :T].to(self.device), rough[:, :T].to(self.device), \
            b["mask"][:, :T].float().to(self.device)

    def _losses(self, x, y, w):
        from .losses import MultiResolutionSTFTLoss, weighted_mel_loss

        x = x[:, : y.shape[1]]
        T = w.shape[1]
        mel_loss = weighted_mel_loss(self.mel(y)[:, :T], self.mel(x)[:, :T], w)
        st = MultiResolutionSTFTLoss().to(y.device)(y, x, w, self.info.hop)
        return x, mel_loss, st

    def train_step(self, batch, scale, step):
        from .losses import discriminator_loss, feature_matching_loss, generator_adv_loss

        x, mel, ap, f0, rough, w = self._inputs(batch)
        d_trainable = any(p.requires_grad for p in self.discs.parameters())
        for p in self.discs.parameters():
            p.requires_grad_(False)
        y = self.vocoder(mel, ap, f0, rough)
        x, mel_loss, st = self._losses(x, y, w)
        res = {"mel": self.w_mel * mel_loss, "stft": self.w_stft * st}
        if self.adversarial:
            real = [o for d in self.discs for o in d(x)]
            fake = [o for d in self.discs for o in d(y)]
            res["adv"] = generator_adv_loss(fake)
            res["fm"] = 2.0 * feature_matching_loss(real, fake)
        total = sum(res.values())
        (total * scale).backward()
        out = {k: float(v.detach()) for k, v in res.items()}
        out["total"] = float(total.detach())
        if self.adversarial and d_trainable:
            for p in self.discs.parameters():
                p.requires_grad_(True)
            r = [o for d in self.discs for o in d(x)]
            f = [o for d in self.discs for o in d(y.detach())]
            dl = discriminator_loss(r, f)
            (dl * scale).backward()
            out["disc"] = float(dl.detach())
        return out

    @torch.no_grad()
    def validate(self, batches):
        acc = _Acc()
        for b in batches:
            x, mel, ap, f0, rough, w = self._inputs(b)
            y = self.vocoder(mel, ap, f0, rough, seed=0)
            _, mel_loss, st = self._losses(x, y, w)
            acc.add("mel_l1", float(mel_loss))
            acc.add("mrstft", float(st))
        out = acc.means()
        out["score"] = out.get("mel_l1", float("nan"))
        return out

    @torch.no_grad()
    def samples(self, batch, out_dir, step):
        x, mel, ap, f0, rough, _ = self._inputs(batch)
        y = self.vocoder(mel, ap, f0, rough, seed=0)
        n = int(self.cfg.run.n_samples)
        return _save_samples(out_dir, step, batch["ids"][:n], [v.cpu().numpy() for v in y[:n]], [v.cpu().numpy() for v in x[:n, : y.shape[1]]],
                             self.info.sr, self.name, self.cfg.profile)


def _rates_for(hop: int) -> tuple[int, ...]:
    rates, rem = [], hop
    for r in (8, 8, 4, 2, 2, 2):
        if rem % r == 0 and rem > 1:
            rates.append(r)
            rem //= r
    if rem != 1:
        raise ValueError(f"cannot factor hop {hop} into upsampling rates; set model.upsample_rates")
    return tuple(rates)


# ---------------------------------------------------------------- pitch (RMVPE)


class PitchTask(Task):
    name = "pitch"
    components = ("pitch",)
    loss_names = ("total",)

    def build(self):
        from ..pitch.rmvpe import RMVPE

        kw = {k: v for k, v in self.m.items() if k in ("n_blocks", "n_gru", "en_de_layers", "inter_layers", "en_out")}
        if self.m.get("size", "tiny") == "tiny":
            kw = {"n_blocks": 1, "inter_layers": 1, "en_out": 4, **kw}
        self.rmvpe = RMVPE(**kw)
        self.sigma = float(self.m.get("target_sigma_cents", 25.0))
        self.modules = {"pitch": self.rmvpe}
        return self.modules

    def component_modules(self):
        return {"pitch": self.rmvpe}

    def load_init(self, component, module, sd):
        # reference rmvpe.pt holds the E2E part; gyeol checkpoints hold "pitch.model.…"
        if any(k.startswith("pitch.") for k in sd):
            sd = {k[len("pitch."):]: v for k, v in sd.items() if k.startswith("pitch.")}
        if not any(k.startswith("model.") for k in sd):
            sd = {f"model.{k}": v for k, v in sd.items()}
        super().load_init(component, module, sd)

    def _targets(self, b) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """16 kHz audio, target salience (B, T16, 360), frame weights (B, T16), target cents re 10 Hz (0 = unvoiced)."""
        from ..dsp.base import resample
        from ..pitch.rmvpe import CENTS0, HOP, N_BINS, SR

        x16 = torch.tensor(np.stack([resample(a.numpy().astype(float), b["sr"], SR) for a in b["audio"]]), dtype=torch.float32)
        T16 = x16.shape[1] // HOP + 1
        t16 = np.arange(T16) * HOP / SR
        tg = np.arange(b["frames"]) * b["hop"] / b["sr"]
        c = b["curves"]["f0_cents"].numpy()
        conf = (b["conf"]["f0_cents"] * b["mask"]).numpy()
        cents10 = np.zeros((len(c), T16))
        w = np.zeros((len(c), T16))
        for i in range(len(c)):
            ok = conf[i] > 0
            near = np.clip(np.round(t16 * b["sr"] / b["hop"]).astype(int), 0, len(tg) - 1)
            vi = ok[near]
            ci = np.interp(t16, tg[ok], c[i][ok]) if ok.sum() >= 2 else np.zeros(T16)
            cents10[i] = np.where(vi, ci + 1200 * np.log2(440.0 / 10.0), 0.0)
            w[i] = np.where(near < b["frames"], 1.0, 0.0) * (t16 <= tg[-1] + 1e-9)
        bins = CENTS0 + 20.0 * np.arange(N_BINS)
        sal = np.exp(-((cents10[..., None] - bins) ** 2) / (2 * self.sigma**2)) * (cents10[..., None] > 0)
        return x16, torch.tensor(sal, dtype=torch.float32), torch.tensor(w, dtype=torch.float32), torch.tensor(cents10)

    def train_step(self, batch, scale, step):
        x16, sal, w, _ = self._targets(batch)
        pred = self.rmvpe(x16.to(self.device))
        T = min(pred.shape[1], sal.shape[1])
        bce = F.binary_cross_entropy(pred[:, :T].clamp(1e-6, 1 - 1e-6), sal[:, :T].to(self.device), reduction="none").mean(-1)
        loss = (bce * w[:, :T].to(self.device)).sum() / w[:, :T].sum().clamp_min(1.0)
        (loss * scale).backward()
        return {"total": float(loss.detach())}

    @torch.no_grad()
    def validate(self, batches):
        from ..pitch.rmvpe import salience_to_cents

        acc = _Acc()
        for b in batches:
            x16, sal, w, cents10 = self._targets(b)
            pred = self.rmvpe(x16.to(self.device))
            T = min(pred.shape[1], sal.shape[1])
            bce = F.binary_cross_entropy(pred[:, :T].clamp(1e-6, 1 - 1e-6), sal[:, :T].to(self.device), reduction="none").mean(-1)
            acc.add("loss", float((bce * w[:, :T]).sum() / w[:, :T].sum().clamp_min(1.0)))
            est = np.stack([salience_to_cents(p) for p in pred[:, :T].cpu().numpy()])
            ref = cents10[:, :T].numpy()
            vm = (ref > 0) & (w[:, :T].numpy() > 0)
            if vm.any():
                acc.add_pair("rpa", (float(((est > 0) & (np.abs(est - ref) <= 50))[vm].sum()), int(vm.sum())))
            um = (ref == 0) & (w[:, :T].numpy() > 0)
            if um.any():
                acc.add_pair("unvoiced_correct", (float((est[um] == 0).sum()), int(um.sum())))
        out = acc.means()
        out["score"] = out.get("loss", float("nan"))
        return out


TASK_CLASSES = {"heads": HeadsTask, "autoencoder": AutoencoderTask, "vocoder": VocoderTask, "pitch": PitchTask, "ssl": SSLTask}

__all__ = ["DataInfo", "TASK_CLASSES", "Task", "head_loss", "head_targets", "label_tasks"]

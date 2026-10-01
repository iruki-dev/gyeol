"""Revision B — CPU training: device handling (B1), the training runner (B2), interrupt / resume (B3),
memory (B4), per-component modes (B5), data preparation (B6), validation (B7), progress and presets (B8)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).parents[1]

# ================================================================ B1 device / every module on CPU


def test_device_auto_resolves_and_cpu_is_float32():
    from gyeol.train.device import autocast_context, resolve_device

    spec = resolve_device("auto", threads=2)
    assert spec.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert spec.dtype == torch.float32 and spec.threads == torch.get_num_threads() == 2
    cpu = resolve_device("cpu", mixed_precision=True)
    assert not cpu.autocast  # the CPU path never uses reduced precision
    with autocast_context(cpu):
        assert (torch.ones(2) @ torch.ones(2)).dtype == torch.float32
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda")


def _tiny_modules():
    """Every trainable module class in gyeol → (instance, forward that returns a scalar)."""
    from gyeol.attributes.heads import FrameHeads, default_tasks
    from gyeol.decoder.acoustic import AcousticModel
    from gyeol.decoder.model import AutoencoderConfig, GyeolAutoencoder
    from gyeol.decoder.vocoder import NSFVocoder
    from gyeol.discover.sae import TopKSAE
    from gyeol.encoders.latent import AttentiveStatsPool, EnvEncoder, LeakageHeads, ResidualEncoder, SingerEncoder
    from gyeol.frontend.roformer import BSRoFormer, tiny_config
    from gyeol.pitch.rmvpe import RMVPE
    from gyeol.train.losses import MultiPeriodDiscriminator, MultiResolutionSTFTLoss, MultiScaleDiscriminator
    from gyeol.train.tasks import build_ssl

    B, T, M = 2, 24, 16
    mel = torch.randn(B, T, M)
    out = {}

    def add(name, mod, fn):
        out[name] = (mod, fn)

    add("FrameHeads", FrameHeads(10, default_tasks(), hidden=8), lambda m: sum(v.sum() for v in m(torch.randn(B, T, 10))[0].values()))
    add("AttentiveStatsPool", AttentiveStatsPool(8), lambda m: m(torch.randn(B, 8, T)).pow(2).sum())
    add("SingerEncoder", SingerEncoder(M, 8, 4), lambda m: m(mel).sum())
    add("EnvEncoder", EnvEncoder(M, 8, 4), lambda m: m(mel)[0].sum() + sum(v.sum() for v in m(mel)[1].values()))
    add("ResidualEncoder", ResidualEncoder(M, 3, 8, 2), lambda m: sum(v.sum() for v in m(mel, torch.randn(B, T, 3))))
    add("LeakageHeads", LeakageHeads(2, {"f0": 1}, {"singer": 3}, hidden=8),
        lambda m: sum(v.sum() for d in m(torch.randn(B, T, 2)) for v in d.values()))
    add("AcousticModel", AcousticModel(3, 2, 4, 4, M, 5, 16, 1, 2),
        lambda m: sum(v.sum() for v in m(torch.randn(B, T, 3), torch.ones(B, T, 3), torch.randn(B, T, 2), torch.full((B, T), 220.0),
                                          torch.zeros(B, T), torch.randn(B, 4), torch.randn(B, 4))))
    add("NSFVocoder", NSFVocoder(M, 5, 16000, 64, 8, (8, 8)),
        lambda m: m(mel, -torch.rand(B, T, 5) * 20, torch.full((B, T), 220.0), torch.zeros(B, T)).pow(2).mean())
    add("GyeolAutoencoder", GyeolAutoencoder(AutoencoderConfig.tiny(sr=16000, hop=64, upsample_rates=(8, 8), n_singers=2)),
        lambda m: _ae_forward(m))
    add("MultiPeriodDiscriminator", MultiPeriodDiscriminator((2, 3), ch=4), lambda m: sum(o[0].sum() for o in m(torch.randn(B, 2000))))
    add("MultiScaleDiscriminator", MultiScaleDiscriminator(2, ch=4), lambda m: sum(o[0].sum() for o in m(torch.randn(B, 2000))))
    add("RMVPE", RMVPE(n_blocks=1, inter_layers=1, en_out=4), lambda m: m(torch.randn(1, 3200)).sum())
    add("BSRoFormer", BSRoFormer(**tiny_config()), lambda m: m(torch.randn(1, 2000)).pow(2).mean())
    add("TopKSAE", TopKSAE(8, 16, 2), lambda m: sum(v.sum() for v in m(torch.randn(B * T, 8))))
    add("SSL (wav2vec2 arch)", build_ssl({"arch": "tiny", "dim": 16, "layers": 1})[0],
        lambda m: m.extract_features(torch.randn(1, 4000))[0][-1].sum())
    stft = MultiResolutionSTFTLoss((64,), (16,))
    x = torch.randn(1, 2000, requires_grad=True)
    add("MultiResolutionSTFTLoss (input grad)", stft, lambda m: m(x * 1.0, torch.randn(1, 2000)))
    return out


def _ae_forward(m):
    """The full generator objective (encoders, leakage adversaries, acoustic model, vocoder, re-encoding)."""
    from gyeol.train.autoencoder import AEBatch, generator_step

    B, T = 4, 40
    wav = torch.randn(B, (T - 1) * 64) * 0.1
    c = torch.randn(B, T, m.cfg.c_dim)
    cm = torch.ones(B, T, m.cfg.c_dim, dtype=torch.bool)
    env = {k: torch.randint(0, n, (B,)) for k, n in m.cfg.env_heads.items()}
    b = AEBatch(wav, c, cm, torch.full((B, T), 220.0), torch.zeros(B, T), torch.ones(B, T), wav * 0.9, torch.zeros(B, T),
                torch.tensor([0, 0, 1, 1]), env)
    return generator_step(m, b, vocode=True)["total"]


@pytest.mark.parametrize("name", list(_tiny_modules()))
def test_every_module_runs_forward_and_backward_on_cpu(name):
    torch.manual_seed(0)
    mod, fn = _tiny_modules()[name]
    mod = mod.to("cpu").float().train()
    loss = fn(mod)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    params = [p for p in mod.parameters() if p.requires_grad]
    if params:
        with_grad = [p for p in params if p.grad is not None]
        assert len(with_grad) >= 0.8 * len(params), f"{name}: only {len(with_grad)}/{len(params)} parameters got gradients"
        assert all(torch.isfinite(p.grad).all() for p in with_grad)


def test_module_coverage_is_complete():
    """Every nn.Module subclass in gyeol is exercised above, directly or as part of a tested module."""
    import importlib
    import inspect
    import pkgutil

    import gyeol

    found = set()
    for info in pkgutil.walk_packages(gyeol.__path__, "gyeol."):
        if info.name.endswith(("cli", "__main__")):
            continue
        try:
            mod = importlib.import_module(info.name)
        except ImportError:
            continue
        for n, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, nn.Module) and obj.__module__ == mod.__name__:
                found.add(n)
    tested = {k.split(" ")[0] for k in _tiny_modules()}
    parts = {  # submodules covered through their parent
        "RMSNorm", "RotaryEmbedding", "FeedForward", "Attention", "Transformer", "BandSplit", "MaskEstimator",  # BSRoFormer
        "_PeriodDisc", "_ScaleDisc", "HarmonicSource", "_ResBlock", "NoiseBranch", "LogMel",  # discriminators, vocoder, AE
        "MelSpectrogram", "BiGRU", "ConvBlockRes", "ResEncoderBlock", "Encoder", "Intermediate", "ResDecoderBlock", "Decoder",
        "DeepUnet", "E2E",  # RMVPE
        "HeadsExport", "AcousticExport", "VocoderExport",  # ONNX wrappers around tested modules (M8 tests run them)
    }
    missing = found - tested - parts
    assert not missing, f"modules without a CPU forward/backward test: {sorted(missing)}"


# ================================================================ B3 state files


def test_atomic_save_never_leaves_a_partial_file(tmp_path, monkeypatch):
    from gyeol.train import state

    path = tmp_path / "s.pt"
    state.atomic_save({"a": torch.ones(3)}, path)
    real = torch.save

    def broken(obj, fh, *a, **k):
        fh.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(state.torch, "save", broken)
    with pytest.raises(OSError):
        state.atomic_save({"a": torch.zeros(3)}, path)
    monkeypatch.setattr(state.torch, "save", real)
    assert torch.equal(torch.load(path)["a"], torch.ones(3))  # the previous file is intact
    assert [p.name for p in tmp_path.iterdir()] == ["s.pt"]  # no temp file left behind


def test_rng_states_round_trip():
    import random

    from gyeol.train.state import rng_state, set_rng_state

    st = rng_state()
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_rng_state(st)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == a


# ================================================================ shared prepared data


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    from gyeol.data.prepare import PrepareConfig, prepare, synthetic_manifest

    root = tmp_path_factory.mktemp("prep")
    m = synthetic_manifest(root / "src", n_singers=5, seconds=0.8, sr=16000)
    rep = prepare([m], root / "cache", config=PrepareConfig(sr=16000, hop=128, separation="off"))
    assert rep.done == 45 and rep.failed == 0
    return root / "cache"


def _cfg(task, cache, out, **over):
    from gyeol.train.config import config_from_dict

    model = {"heads": {"hidden": 16}, "ssl": {"arch": "tiny", "dim": 16, "layers": 2, "layers_used": [0, 1], "hidden": 16},
             "autoencoder": {"size": "tiny", "vocode_every": 2, "disc_channels": 4, "periods": [2], "scales": 1},
             "vocoder": {"size": "tiny", "channels": 8, "disc_channels": 4, "periods": [2], "scales": 1},
             "pitch": {"size": "tiny"}}[task]
    d = {"task": task, "seed": 3, "device": "cpu", "threads": 1,
         "data": {"cache": str(cache), "split": {"train": 0.6, "val": 0.2, "test": 0.2}, "crop_frames": [12, 20], "batch_size": 2,
                  "prepare": {"sr": 16000, "hop": 128, "separation": "off"}},
         "optim": {"lr": 2e-3, "grad_accum": 2},
         "run": {"out": str(out), "max_steps": 6, "val_every": 3, "save_every": 2, "log_every": 1, "patience": 0, "val_batches": 1,
                 "n_samples": 1},
         "model": model}
    for k, v in over.items():
        node = d
        *path, last = k.split("__")
        for p in path:
            node = node.setdefault(p, {})
        node[last] = v
    return config_from_dict(d)


def _final_params(out: Path) -> dict:
    blob = torch.load(out / "state.pt", weights_only=False)
    return {f"{m}.{k}": v for m, sd in blob["modules"].items() for k, v in sd.items()}


# ================================================================ B3 interrupt and resume = uninterrupted


@pytest.mark.parametrize("task", ["heads", "autoencoder", "vocoder", "pitch", "ssl"])
def test_interrupted_and_resumed_run_matches_uninterrupted(task, cache, tmp_path):
    from gyeol.train.runner import train

    quiet = lambda s: None  # noqa: E731
    full = train(_cfg(task, cache, tmp_path / "full"), log=quiet)
    assert full.status == "finished" and full.position.step == 6
    part = train(_cfg(task, cache, tmp_path / "split"), log=quiet, stop_after_steps=4)  # interrupted mid-epoch, between saves
    assert part.status == "interrupted" and part.position.step == 4
    with pytest.raises(FileExistsError):
        train(_cfg(task, cache, tmp_path / "split"), log=quiet)  # a second start without --resume is refused
    rest = train(_cfg(task, cache, tmp_path / "split"), resume=True, log=quiet)
    assert rest.status == "finished" and rest.position.step == 6
    a, b = _final_params(tmp_path / "full"), _final_params(tmp_path / "split")
    assert a.keys() == b.keys()
    diff = [k for k in a if not torch.equal(a[k], b[k])]
    assert not diff, f"{task}: resumed run differs in {diff[:5]}"
    ja = json.loads((tmp_path / "full" / "report.json").read_text())
    jb = json.loads((tmp_path / "split" / "report.json").read_text())
    assert ja["best_step"] == jb["best_step"] and ja["test"] == jb["test"]


# ================================================================ B4 streaming, crops, accumulation, checkpointing


def test_stream_is_deterministic_across_workers_and_resumable(cache):
    from gyeol.data.prepare import read_index
    from gyeol.train.stream import CropSpec, StreamingDataset, loader

    rows = read_index(cache)
    ds = lambda start=0: StreamingDataset(cache, rows, 4, CropSpec(10, 30), seed=1, epoch=2, start_position=start)  # noqa: E731
    one = [(b["ids"], b["frames"]) for b in loader(ds(), 0)]
    two = [(b["ids"], b["frames"]) for b in loader(ds(), 2)]
    assert one == two
    assert len({f for _, f in one}) > 1 and all(10 <= f <= 30 for _, f in one)  # random lengths, one per batch
    assert [(b["ids"], b["frames"]) for b in ds(8)] == one[2:]
    with pytest.raises(ValueError):
        ds(3)
    b = next(iter(ds()))
    assert b["audio"].shape == (4, (b["frames"] - 1) * b["hop"]) and b["mask"].shape == (4, b["frames"])
    other_epoch = [x["ids"] for x in StreamingDataset(cache, rows, 4, CropSpec(10, 30), seed=1, epoch=3)]
    assert other_epoch != [i for i, _ in one]


def test_gradient_accumulation_counts_steps(cache, tmp_path):
    from gyeol.train.runner import train

    r = train(_cfg("heads", cache, tmp_path / "acc", optim__grad_accum=3, run__max_steps=4), log=lambda s: None)
    assert r.position.step == 4 and r.position.micro_step == 12


@pytest.mark.parametrize("which", ["acoustic", "rmvpe"])
def test_gradient_checkpointing_gives_the_same_gradients(which):
    from gyeol.train.components import enable_gradient_checkpointing

    def build():
        torch.manual_seed(0)
        if which == "acoustic":
            from gyeol.decoder.acoustic import AcousticModel

            m = AcousticModel(3, 2, 4, 4, 16, 5, 16, 2, 2, cond_dropout=0.0)
            args = (torch.randn(2, 20, 3), torch.ones(2, 20, 3), torch.randn(2, 20, 2), torch.full((2, 20), 200.0), torch.zeros(2, 20),
                    torch.randn(2, 4), torch.randn(2, 4))
            return m, lambda: m(*args)[0].pow(2).mean()
        from gyeol.pitch.rmvpe import RMVPE

        m = RMVPE(n_blocks=1, inter_layers=1, en_out=4)
        x = torch.randn(1, 3200)
        return m, lambda: m(x).sum()

    m1, f1 = build()
    m1.train()
    f1().backward()
    m2, f2 = build()
    m2.train()
    assert enable_gradient_checkpointing(m2) > 0
    f2().backward()
    for (n, p), q in zip(m1.named_parameters(), m2.parameters()):
        assert (p.grad is None) == (q.grad is None), n
        if p.grad is not None:
            assert torch.allclose(p.grad, q.grad, atol=1e-5), n


# ================================================================ B5 component modes and initial weights


def test_component_modes_and_lineage(cache, tmp_path):
    from gyeol.train.runner import train

    quiet = lambda s: None  # noqa: E731
    base = train(_cfg("autoencoder", cache, tmp_path / "base", run__max_steps=2), log=quiet)
    assert (tmp_path / "base" / "best.pt").exists()
    # freeze the vocoder, fine-tune the acoustic model from the first run's checkpoint
    cfg = _cfg("autoencoder", cache, tmp_path / "ft", run__max_steps=2, components={"vocoder": "freeze", "acoustic": "finetune"},
               init={"acoustic": {"path": str(base.best_checkpoint)}})
    from gyeol.train.runner import Trainer

    t = Trainer(cfg, log=quiet)
    t._setup()
    voc = t.task.model.vocoder
    assert not any(p.requires_grad for p in voc.parameters()) and not voc.training
    sd, _ = __import__("gyeol.train.checkpoint", fromlist=["x"]).load_checkpoint(base.best_checkpoint)
    w = next(k for k in sd if k.startswith("model.acoustic.") and k.endswith("weight"))
    assert torch.equal(dict(t.task.model.named_parameters())[w[len("model."):]].detach(), sd[w])
    lrs = {g["name"]: g["lr"] for g in t.optimizers["g"].param_groups}
    assert "vocoder" not in lrs and lrs["acoustic"] == pytest.approx(cfg.optim.lr * cfg.finetune_lr_scale)
    assert [p.name for p in t.lineage.parents] == ["autoencoder-best"]
    with pytest.raises(ValueError, match="finetune but has no init"):
        Trainer(_cfg("vocoder", cache, tmp_path / "x", components={"vocoder": "finetune"}), log=quiet)._setup()
    with pytest.raises(ValueError, match="not part of this task"):
        Trainer(_cfg("vocoder", cache, tmp_path / "y", components={"pitch": "freeze"}), log=quiet)._setup()


def test_initial_weights_are_recorded_in_the_lineage(cache, tmp_path):
    from gyeol.train.runner import Trainer

    scratch = Trainer(_cfg("vocoder", cache, tmp_path / "s"), log=lambda s: None)
    scratch._setup()
    voc = scratch.task.component_modules()["vocoder"]
    fake = tmp_path / "openvpi.ckpt"  # a third-party weight file (plain state dict) under a listed asset name
    torch.save(voc.state_dict(), fake)
    cfg = _cfg("vocoder", cache, tmp_path / "nc", components={"vocoder": "finetune"},
               init={"vocoder": {"path": str(fake), "asset": "openvpi_nsf_hifigan"}})
    t = Trainer(cfg, log=lambda s: None)
    t._setup()  # loads directly from the local path; the license is information, not a gate
    assert t.lineage.assets == ["gyeol_synthetic", "openvpi_nsf_hifigan"]  # training data, then the initial weights
    with pytest.raises(FileNotFoundError, match="gyeol fetch rmvpe"):
        Trainer(_cfg("pitch", cache, tmp_path / "rm", components={"pitch": "finetune"},
                     init={"pitch": {"path": str(tmp_path / "missing.pt"), "asset": "rmvpe"}}), log=lambda s: None)._setup()


# ================================================================ B6 preparation


def test_prepare_is_resumable_and_logs_failures(tmp_path):
    from gyeol.data.manifest import Manifest, ManifestItem
    from gyeol.data.prepare import PrepareConfig, prepare, read_index, synthetic_manifest

    m = synthetic_manifest(tmp_path / "src", n_singers=2, seconds=0.6, sr=16000)
    man = Manifest.read(m)
    (tmp_path / "src" / "broken.wav").write_bytes(b"not audio")
    man.items.append(ManifestItem("broken.wav", "s9", {}))
    man.items.append(ManifestItem("missing.wav", "s9", {}))
    cfg = PrepareConfig(sr=16000, hop=128, separation="off")
    r1 = prepare([man], tmp_path / "c", config=cfg, limit=5)
    assert r1.done == 5 and r1.failed == 0
    r2 = prepare([man], tmp_path / "c", config=cfg)
    assert r2.skipped == 5 and r2.done == 13 and r2.failed == 2
    lines = [json.loads(x) for x in (tmp_path / "c" / "failures.jsonl").read_text().splitlines()]
    assert {x["path"] for x in lines} == {"broken.wav", "missing.wav"} and all(x["reason"] for x in lines)
    assert len(read_index(tmp_path / "c")) == 18
    with pytest.raises(ValueError, match="different settings"):
        prepare([man], tmp_path / "c", config=PrepareConfig(sr=22050, hop=128, separation="off"))


def test_prepare_cli(tmp_path, capsys):
    from gyeol.cli import main

    assert main(["prepare", "--synthetic", "2", "--out", str(tmp_path / "c"), "--sr", "16000", "--hop", "128", "--separation", "off"]) == 0
    assert "prepared 18" in capsys.readouterr().out
    assert main(["prepare", "--out", str(tmp_path / "d")]) == 2


# ================================================================ B7 validation, early stopping, report on unseen singers


def test_early_stopping_best_checkpoint_and_unseen_singer_report(cache, tmp_path):
    from gyeol.train.checkpoint import load_checkpoint
    from gyeol.train.runner import train

    r = train(_cfg("heads", cache, tmp_path / "es", optim__lr=0.0, run__max_steps=40, run__val_every=2, run__patience=2), log=lambda s: None)
    assert r.status == "early_stopped" and r.position.step == 6  # best at step 2, then two validations without improvement
    _, info = load_checkpoint(r.best_checkpoint)
    assert info.source_names == ["gyeol_synthetic"] and r.report["provenance"]["sources"] == ["gyeol_synthetic"]
    rep = r.report
    split = json.loads((tmp_path / "es" / "split.json").read_text())
    index = {x["id"]: x["singer"] for x in map(json.loads, (cache / "index.jsonl").read_text().splitlines())}
    train_singers = {index[i] for i in split["train"]}
    assert rep["test_singers"] and not set(rep["test_singers"]) & train_singers
    assert not set(rep["validation_singers"]) & train_singers
    assert "register_acc" in rep["test"] and (tmp_path / "es" / "report.md").exists()
    assert "temperatures" in rep["finalize"]


# ================================================================ B8 progress, samples, presets


def test_logs_eta_and_samples(cache, tmp_path):
    import csv

    from gyeol.train.runner import train

    train(_cfg("vocoder", cache, tmp_path / "v", run__max_steps=4, run__val_every=2), log=lambda s: None)
    rows = list(csv.DictReader(open(tmp_path / "v" / "train_log.csv")))
    assert rows and {"eta_s", "elapsed_s", "lr", "total", "mel", "stft"} <= set(rows[0])
    assert float(rows[-1]["eta_s"]) == pytest.approx(0.0, abs=1e-6)
    vals = list(csv.DictReader(open(tmp_path / "v" / "val_log.csv")))
    assert [int(v["step"]) for v in vals] == [2, 4]
    side = json.loads(next((tmp_path / "v" / "samples").rglob("samples.json")).read_text())
    assert side["kind"].startswith("vocoder") and side["items"] and "ai_generated" not in side
    assert list((tmp_path / "v" / "samples").rglob("*_generated.wav"))


@pytest.mark.parametrize("preset", sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs").glob("cpu-*/*.yaml")))
def test_presets_load(preset):
    from gyeol.train.config import load_config

    path = ROOT / preset
    cfg = load_config(path)
    assert cfg.task == path.stem
    with pytest.raises(ValueError, match="unknown keys"):
        from gyeol.train.config import config_from_dict

        config_from_dict({**cfg.as_dict(), "optim": {**cfg.as_dict()["optim"], "lrr": 1}})


def test_all_tasks_have_smoke_and_full_presets():
    from gyeol.train.config import TASKS

    for kind in ("cpu-smoke", "cpu-full"):
        assert {p.stem for p in (ROOT / "configs" / kind).glob("*.yaml")} == set(TASKS)


def test_train_cli_interrupt_and_resume(cache, tmp_path):
    import yaml

    from gyeol.cli import main

    cfg = _cfg("heads", cache, tmp_path / "cli").as_dict()
    path = tmp_path / "heads.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert main(["train", "heads", "--config", str(path), "--stop-after", "2"]) == 3
    assert main(["train", "heads", "--config", str(path), "--resume", "--set", "run.log_every=2"]) == 0
    assert json.loads((tmp_path / "cli" / "report.json").read_text())["steps"] == 6
    with pytest.raises(ValueError, match="configures task"):
        main(["train", "pitch", "--config", str(path)])

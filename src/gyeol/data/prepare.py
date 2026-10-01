"""Batch data preparation for training (revision B6): ``gyeol prepare --manifest ...``.

For every item of one or more manifests (VocalSet, GTSinger,
AI Hub, own recordings — any :mod:`gyeol.data.adapters` output, or the
synthetic corpus below) this runs

1. loading and resampling to the training rate,
2. vocal separation (``analyze(separation=...)``, revision A1),
3. pitch (consensus) and the attribute curves,
4. optional feature caching: DSP frame features, and SSL features when an
   encoder is given,

and writes one ``items/<id>.npz`` per item, followed by a ``items/<id>.done``
completion marker.  Re-running skips every item that has a marker, so an
interrupted preparation resumes where it stopped.  Failures are appended to
``failures.jsonl`` with the reason and do not stop the batch; ``index.jsonl``
lists every prepared item (id, dataset, singer, labels, frames) for the
streaming loader (:mod:`gyeol.train.stream`).
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..core.status import Result
from .manifest import Manifest, ManifestItem, open_manifest

#: curves cached per item (values and confidence on the analysis grid)
CACHED_CURVES = ("f0_cents", "pitch_center", "vibrato_rate", "vibrato_extent", "loudness_rel", "aperiodic_ratio",
                 "subharmonic_ratio", "voicing")
FORMAT = "gyeol-prepared/1"


@dataclass
class PrepareConfig:
    sr: int = 44100
    hop: int = 512
    separation: str = "auto"  # analyze(separation=...)
    dsp_trackers_only: bool = True  # gyeol's DSP pitch trackers (neural trackers need fetched weights)
    features: tuple[str, ...] = ("dsp",)  # "dsp" and/or "ssl"
    max_seconds: float = 30.0  # longer files are cut (training crops are short anyway)
    min_voiced_frames: int = 10
    #: revision D3 — exact-f0 copies for pitch training: "hnm" (MDB-stem-synth-style harmonic-plus-noise
    #: resynthesis with the analysed f0) and/or "vocoder" (a gyeol-trained NSF vocoder, ``resynth_vocoder``).
    #: The f0 the copy is synthesised with is its ground truth (``f0_exact``).
    resynthesize: tuple[str, ...] = ()
    resynth_vocoder: str | None = None
    resynth_min_confidence: float = 0.5  # analysed frames below this confidence are synthesised unvoiced
    #: datasets that get no resynthesised copies (the application decides; by default its own users' recordings)
    resynth_skip_datasets: tuple[str, ...] = ("own_recordings",)


@dataclass
class PrepareReport:
    out_dir: Path
    done: int = 0
    skipped: int = 0  # already prepared (completion marker present)
    failed: int = 0
    failures: list[dict] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> str:
        return f"prepared {self.done}, already done {self.skipped}, failed {self.failed} ({self.seconds:.1f} s) → {self.out_dir}"


def item_id(dataset: str, item: ManifestItem) -> str:
    h = hashlib.sha1(f"{dataset}:{item.path}".encode()).hexdigest()[:16]
    return f"{dataset}-{h}"


def _prepare_one(audio: np.ndarray, sr: int, cfg: PrepareConfig, ssl_encoder=None) -> Result[dict]:
    from ..attributes.extract import AnalysisConfig, analyze
    from ..core.containers import Recording
    from ..dsp.base import resample
    from ..encoders.frame import DSPFrameFeatures
    from ..pitch.adapters import PyinTracker, SHSTracker, YinTracker

    x = resample(np.asarray(audio, float), sr, cfg.sr) if sr != cfg.sr else np.asarray(audio, float)
    x = x[: int(cfg.max_seconds * cfg.sr)]
    if len(x) < cfg.sr // 5:
        return Result.failure(f"too short ({len(x) / cfg.sr:.2f} s)")
    trackers = [PyinTracker(), YinTracker(), SHSTracker()] if cfg.dsp_trackers_only else None
    # dataset recordings are reference material, not app users
    r = analyze(Recording(x, cfg.sr), trackers=trackers, config=AnalysisConfig(hop=cfg.hop, keep_separated_audio=True), separation=cfg.separation)
    if not r.usable:
        return Result.failure(f"analysis failed: {r.reason}")
    rep = r.value
    voiced = np.isfinite(rep.curves["f0_cents"].values)
    if voiced.sum() < cfg.min_voiced_frames:
        return Result.failure(f"only {int(voiced.sum())} voiced frames")
    sep = rep.quality.get("separation", {})
    sig = x
    if sep.get("applied") and "separated_audio" in rep.meta:
        sig = np.asarray(rep.meta["separated_audio"], float)
    out = {"audio": sig.astype(np.float32), "sr": np.int64(cfg.sr), "hop": np.int64(cfg.hop), "n_frames": np.int64(rep.grid.n_frames)}
    for name in CACHED_CURVES:
        c = rep.curves[name]
        out[f"curve/{name}"] = c.values.astype(np.float32)
        out[f"conf/{name}"] = c.confidence.astype(np.float32)
    # band aperiodicity (dB ≤ 0; 0 = noise) for training the vocoder on its own: unvoiced frames are noise
    from ..dsp.spectral import band_aperiodicity, spectrogram

    f0_hz = np.where(voiced, 440.0 * 2 ** (np.nan_to_num(rep.curves["f0_cents"].values) / 1200.0), np.nan)
    ap, _, _ = band_aperiodicity(spectrogram(sig, cfg.sr, cfg.hop, rep.grid.n_frames), f0_hz)
    out["ap_bands"] = np.where(voiced[:, None], np.nan_to_num(np.clip(ap, -60.0, 0.0), nan=-20.0), 0.0).astype(np.float32)
    if "dsp" in cfg.features:
        out["feat/dsp"] = DSPFrameFeatures().from_representation(rep).astype(np.float32)
    if "ssl" in cfg.features:
        if ssl_encoder is None:
            return Result.failure("feature 'ssl' requested but no SSL encoder given")
        f = ssl_encoder.encode(sig, cfg.sr, rep.grid)
        if not f.usable:
            return Result.failure(f"SSL features failed: {f.reason}")
        out["feat/ssl"] = np.asarray(f.value, np.float32)
    out["quality"] = np.frombuffer(json.dumps({"flags": rep.quality.get("flags", {}), "separation": {
        k: v for k, v in sep.items() if isinstance(v, (str, int, float, bool, type(None)))}}, default=str).encode(), dtype=np.uint8)
    return Result.success(out)


def _save_npz_atomic(path: Path, arrays: dict) -> None:
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def prepare(manifests: Sequence[Manifest | str | Path], out_dir: str | Path,
            config: PrepareConfig | None = None, *, ssl_encoder=None, limit: int | None = None,
            progress: Callable[[str], None] | None = None) -> PrepareReport:
    """Prepare every item of ``manifests`` into ``out_dir`` (resumable; see module docstring)."""
    from ..io import load_audio

    cfg = config or PrepareConfig()
    out = Path(out_dir)
    items_dir = out / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rep = PrepareReport(out)
    meta_path = out / "prepare.json"
    settings = {"format": FORMAT, "sr": cfg.sr, "hop": cfg.hop, "separation": cfg.separation, "features": list(cfg.features),
                "curves": list(CACHED_CURVES)}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        clash = {k: (old.get(k), v) for k, v in settings.items() if old.get(k) != v}
        if clash:
            raise ValueError(f"{out} was prepared with different settings {clash}; use a new output folder")
    meta_path.write_text(json.dumps(settings, indent=1), encoding="utf-8")
    n = 0
    for m in manifests:
        ds = open_manifest(m)
        for item in ds:
            if limit is not None and n >= limit:
                break
            n += 1
            iid = item_id(ds.manifest.dataset, item)
            done = items_dir / f"{iid}.done"
            if done.exists():
                rep.skipped += 1
                continue
            if progress:
                progress(f"{iid} {item.path}")
            try:
                x, sr = load_audio(ds.resolve(item))
                r = _prepare_one(x, sr, cfg, ssl_encoder)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch; the reason is logged
                r = Result.failure(f"{type(exc).__name__}: {exc}")
                tb = traceback.format_exc(limit=3)
            else:
                tb = None
            if not r.usable:
                rec = {"id": iid, "dataset": ds.manifest.dataset, "path": item.path, "reason": r.reason,
                       "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
                if tb:
                    rec["traceback"] = tb
                with open(out / "failures.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rep.failed += 1
                rep.failures.append(rec)
                continue
            arrays = r.value
            _save_npz_atomic(items_dir / f"{iid}.npz", arrays)
            info = {"id": iid, "dataset": ds.manifest.dataset, "path": item.path, "singer": item.singer or "",
                    "labels": item.labels, "meta": item.meta, "n_frames": int(arrays["n_frames"]), "n_samples": int(len(arrays["audio"])),
                    "license": ds.info["license"]}
            done.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
            rep.done += 1
            for method in cfg.resynthesize:
                _resynth_item(items_dir, out, arrays, info, method, cfg, rep, progress)
    write_index(out)
    rep.seconds = time.time() - t0
    return rep


_VOCODERS: dict = {}


def resynthesize_exact_f0(arrays: dict, cfg: PrepareConfig, method: str) -> Result[tuple[np.ndarray, np.ndarray]]:
    """(audio, exact f0 per frame in Hz, NaN = unvoiced) synthesised with the analysed f0 of a prepared item."""
    from ..core.grid import FrameGrid

    sig = np.asarray(arrays["audio"], float)
    T = int(arrays["n_frames"])
    c, conf = arrays["curve/f0_cents"].astype(float), arrays["conf/f0_cents"].astype(float)
    f0 = np.where(np.isfinite(c) & (conf >= cfg.resynth_min_confidence), 440.0 * 2 ** (np.nan_to_num(c) / 1200.0), np.nan)
    if np.isfinite(f0).sum() < cfg.min_voiced_frames:
        return Result.failure("too few confident voiced frames to resynthesise")
    if method == "hnm":
        from ..demo.hnm import analyze_hnm, synthesize_hnm

        p = analyze_hnm(sig, FrameGrid(cfg.sr, cfg.hop, T), f0)
        y = synthesize_hnm(p, seed=0)
        return Result.success((np.pad(y, (0, max(0, len(sig) - len(y))))[: len(sig)], p.f0_hz))
    if method == "vocoder":
        import torch

        if not cfg.resynth_vocoder:
            return Result.failure("resynthesize 'vocoder' needs resynth_vocoder (a checkpoint from gyeol train vocoder)")
        key = cfg.resynth_vocoder
        if key not in _VOCODERS:
            from ..train.tasks import vocoder_from_checkpoint

            _VOCODERS[key] = vocoder_from_checkpoint(cfg.resynth_vocoder)
        voc, mel, _ = _VOCODERS[key]
        if voc.sr != cfg.sr or voc.hop != cfg.hop:
            return Result.failure(f"vocoder runs at {voc.sr} Hz / hop {voc.hop}, the cache at {cfg.sr} / {cfg.hop}")
        with torch.no_grad():
            x = torch.tensor(sig[: (T - 1) * cfg.hop], dtype=torch.float32)[None]
            m = mel(x)[:, :T]
            n = m.shape[1]
            f0_t = torch.tensor(np.nan_to_num(f0[:n]), dtype=torch.float32)[None]
            ap = torch.tensor(arrays["ap_bands"][:n], dtype=torch.float32)[None]
            rough = torch.tensor(np.clip(np.nan_to_num(arrays["curve/subharmonic_ratio"][:n]) / 0.5, 0, 1), dtype=torch.float32)[None]
            y = voc(m, ap, f0_t, rough, seed=0)[0].numpy().astype(float)
        truth = np.full(T, np.nan)
        truth[:n] = np.where(f0[:n] > 0, f0[:n], np.nan)
        return Result.success((np.pad(y, (0, max(0, len(sig) - len(y))))[: len(sig)], truth))
    return Result.failure(f"unknown resynthesis method {method!r} (hnm | vocoder)")


def _resynth_item(items_dir: Path, out: Path, arrays: dict, info: dict, method: str, cfg: PrepareConfig, rep: PrepareReport,
                  progress) -> None:
    rid = f"{info['id']}~{method}"
    if (items_dir / f"{rid}.done").exists():
        return
    fail = None
    if info["dataset"] in cfg.resynth_skip_datasets:
        fail = f"dataset {info['dataset']!r} is in resynth_skip_datasets"
    else:
        r = resynthesize_exact_f0(arrays, cfg, method)
        if not r.usable:
            fail = r.reason
        else:
            y, truth = r.value
            from dataclasses import replace

            p = _prepare_one(y, cfg.sr, replace(cfg, separation="off"))
            if not p.usable:
                fail = f"analysis of the resynthesised audio failed: {p.reason}"
    if fail is not None:
        rec = {"id": rid, "dataset": info["dataset"], "path": info["path"], "reason": fail, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(out / "failures.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        rep.failed += 1
        rep.failures.append(rec)
        return
    a2 = p.value
    T2 = int(a2["n_frames"])
    a2["f0_exact"] = np.pad(np.nan_to_num(truth, nan=0.0), (0, max(0, T2 - len(truth))))[:T2].astype(np.float32)
    _save_npz_atomic(items_dir / f"{rid}.npz", a2)
    meta = {**info.get("meta", {}), "resynth": method, "f0_truth": "exact", "source_id": info["id"]}
    row = {**info, "id": rid, "meta": meta, "n_frames": T2, "n_samples": int(len(a2["audio"]))}
    (items_dir / f"{rid}.done").write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    rep.done += 1
    if progress:
        progress(f"{rid} (exact-f0 {method} copy)")


def write_index(out_dir: str | Path) -> Path:
    """Rebuild ``index.jsonl`` from the completion markers (sorted by id: a stable order for the loader)."""
    out = Path(out_dir)
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "items").glob("*.done"))]
    path = out / "index.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def read_index(out_dir: str | Path) -> list[dict]:
    path = Path(out_dir) / "index.jsonl"
    if not path.exists():
        write_index(out_dir)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------- synthetic corpus (CI, smoke presets)

_REGISTERS = {"chest": ((110.0, 200.0), 0.5), "mixed": ((220.0, 330.0), 0.65), "falsetto": ((380.0, 560.0), 0.85)}
_QUALITIES = {"modal": (0.0, 0.02), "breathy": (0.15, 0.6), "pressed_belt": (-0.15, 0.0)}


def synthetic_manifest(out_dir: str | Path, n_singers: int = 6, takes_per_cell: int = 1, seconds: float = 1.2, sr: int = 22050,
                       seed: int = 0) -> Path:
    """Write a small synthetic singers × register × phonation corpus and its manifest (dataset ``gyeol_synthetic``).

    Each "singer" has its own vocal-tract scale; labels follow the adapters'
    vocabulary (``phonation`` ∈ register or quality labels, ``register`` explicit).
    Idempotent: existing files are kept.
    """
    from ..io import save_audio
    from ..synth import VOWELS, sung_vowel

    root = Path(out_dir)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    mpath = root / "manifest.json"
    if mpath.exists():
        return mpath
    rng = np.random.default_rng(seed)
    items = []
    for s in range(n_singers):
        scale = 0.88 + 0.24 * s / max(1, n_singers - 1)
        for reg, ((lo, hi), oq) in _REGISTERS.items():
            for q, (doq, asp) in _QUALITIES.items():
                for k in range(takes_per_cell):
                    vowel = "aeiou"[int(rng.integers(5))]
                    fv, bv = VOWELS[vowel]
                    v = sung_vowel(f0=float(rng.uniform(lo, hi)), duration=seconds, sr=sr, formants=tuple(f * scale for f in fv),
                                   bandwidths=bv, open_quotient=float(np.clip(oq + doq, 0.3, 0.95)), aspiration=asp,
                                   vibrato_rate=5.5, vibrato_extent_cents=float(rng.uniform(0, 40)), seed=int(rng.integers(1 << 30)))
                    x = np.r_[np.zeros(sr // 10), v.audio, np.zeros(sr // 10)] + rng.standard_normal(int(v.audio.size + sr // 5)) * 1e-4
                    rel = f"audio/s{s}_{reg}_{q}_{k}.wav"
                    save_audio(root / rel, x, sr)
                    labels = {"register": reg, "phonation": q if q != "modal" else None, "vowel": vowel}
                    items.append(ManifestItem(rel, f"s{s}", labels, {"synthetic": True}))
    Manifest("gyeol_synthetic", str(root), items).write(mpath)
    return mpath


__all__ = ["CACHED_CURVES", "PrepareConfig", "PrepareReport", "item_id", "prepare", "read_index", "synthetic_manifest", "write_index"]

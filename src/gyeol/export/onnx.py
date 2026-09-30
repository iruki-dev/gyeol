"""ONNX export of gyeol's neural components for server inference.

Exported graphs (opset 18, dynamo-based exporter, dynamic batch and time axes):

============================  =====================================================================
``attribute_heads``           features (B, T, D) → concatenated head outputs (B, T, Σ out) + embedding
``rmvpe``                     log-mel (B, 128, T), T a multiple of 32 → salience (B, T, 360)
``acoustic``                  c, c_mask, r, f0, aperiodic, singer, env → mel (B, T, M), ap (B, T, A)
``vocoder``                   mel, ap, f0, rough, jitter noise, source noise → waveform (harmonic path)
``singer_encoder``            log-mel (B, T, M) → singer embedding (B, S)
============================  =====================================================================

Two things deliberately stay outside the graphs:

* **front ends** (STFT/mel, RMVPE padding) — cheap, and ONNX STFT support
  across runtimes is uneven; :mod:`gyeol.pitch.rmvpe` / :mod:`gyeol.encoders.mel`
  compute them;
* **randomness and the vocoder's noise branch** — the vocoder graph takes
  its two standard-normal noise streams as inputs (so the server controls
  seeding and the graph is deterministic), and the aperiodic noise branch
  needs an inverse STFT, which ONNX lacks; run :class:`gyeol.decoder.vocoder.NoiseBranch`
  in PyTorch (or an equivalent numpy STFT) and add it.

Every export is checked against PyTorch with ONNX Runtime (:func:`verify`),
including a second input length to prove the time axis is really dynamic.
Checkpoints go through the license gate *before* export
(:func:`gyeol.train.checkpoint.load_checkpoint`); the exported file gets a
JSON sidecar with the checkpoint's license lineage.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OPSET = 18  # native opset of the dynamo exporter (no version down-conversion)


class HeadsExport(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        self.order = sorted(heads.tasks)

    def forward(self, feats: torch.Tensor):
        logits, emb = self.heads(feats)
        return torch.cat([logits[k] for k in self.order], -1), emb


class AcousticExport(nn.Module):
    def __init__(self, acoustic):
        super().__init__()
        self.m = acoustic

    def forward(self, c, c_mask, r, f0_hz, aperiodic, singer, env):
        return self.m(c, c_mask > 0.5, r, f0_hz, aperiodic, singer, env)


class VocoderExport(nn.Module):
    def __init__(self, vocoder):
        super().__init__()
        self.v = vocoder

    def forward(self, mel, ap_db, f0_hz, rough, jitter_noise, source_noise):
        return self.v.harmonic_path(mel, ap_db, f0_hz, rough, source_noise=(jitter_noise, source_noise))


@dataclass
class ExportSpec:
    name: str
    inputs: list[str]
    outputs: list[str]
    dynamic: dict[str, dict[int, str]]


@dataclass
class ExportResult:
    name: str
    path: Path
    max_abs_diff: float
    max_abs_diff_other_length: float
    passed: bool
    tolerance: float


def _dims(spec: ExportSpec, hop: int | None = None) -> tuple:
    """torch.export dynamic shapes from the spec's axis names ("batch", "time", "time32", "samples")."""
    from torch.export import Dim

    batch, time = Dim("batch", min=1, max=64), Dim("time", min=2, max=200_000)
    t32 = 32 * Dim("t32", min=1, max=10_000)
    named = {"batch": batch, "time": time, "time32": t32}
    if hop is not None:
        named["samples"] = hop * time - hop
    return tuple({ax: named[n] for ax, n in spec.dynamic.get(name, {}).items()} or None for name in spec.inputs)


def _export(module: nn.Module, args: tuple, path: Path, spec: ExportSpec, hop: int | None = None) -> None:
    """Export with the dynamo-based exporter (the TorchScript one bakes sequence lengths into attention
    reshapes).  Example inputs must use batch ≥ 2, since torch.export specialises size-1 dimensions."""
    import logging

    module.eval()
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(module, args, str(path), dynamo=True, dynamic_shapes=_dims(spec, hop), input_names=spec.inputs,
                          output_names=spec.outputs, opset_version=OPSET, verbose=False)


def _relax_output_shapes(path: Path, spec: ExportSpec) -> None:
    """Mark declared dynamic output axes as symbolic in the graph metadata (the exporter can leave the example's
    size there when a dimension passes through a recurrent layer; values are unaffected, runtimes only warn)."""
    import onnx

    model = onnx.load(str(path))
    changed = False
    for out in model.graph.output:
        for ax, name in spec.dynamic.get(out.name, {}).items():
            dims = out.type.tensor_type.shape.dim
            if ax < len(dims) and not dims[ax].dim_param:
                dims[ax].dim_param = name
                changed = True
    if changed:
        # intermediate shape hints can carry the same stale size; they are optional, so drop them
        del model.graph.value_info[:]
        onnx.save(model, str(path))


def verify(module: nn.Module, path: Path, args: tuple, input_names: list[str]) -> float:
    """Max |ONNX Runtime − PyTorch| over the first output."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {n: a.numpy() for n, a in zip(input_names, args)})[0]
    with torch.no_grad():
        ref = module(*args)
    ref = ref[0] if isinstance(ref, (tuple, list)) else ref
    return float(np.max(np.abs(out - ref.numpy())))


def export_component(name: str, module: nn.Module, args: tuple, other_args: tuple, spec: ExportSpec, out_dir: str | Path,
                     tolerance: float = 1e-4, provenance: dict | None = None, hop: int | None = None) -> ExportResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.onnx"
    _export(module, args, path, spec, hop)
    _relax_output_shapes(path, spec)
    d1 = verify(module, path, args, spec.inputs)
    d2 = verify(module, path, other_args, spec.inputs)
    res = ExportResult(name, path, d1, d2, bool(d1 <= tolerance and d2 <= tolerance), tolerance)
    meta = {"component": name, "opset": OPSET, "inputs": spec.inputs, "outputs": spec.outputs,
            "verification": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(res).items()},
            "provenance": provenance or {"license": "untrained / no checkpoint"}}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


# ---------------------------------------------------------------- per-component helpers


def export_heads(heads, in_dim: int, out_dir, provenance: dict | None = None, T: int = 50) -> ExportResult:
    m = HeadsExport(heads)
    spec = ExportSpec("attribute_heads", ["features"], ["outputs", "embedding"],
                      {"features": {0: "batch", 1: "time"}, "outputs": {0: "batch", 1: "time"}, "embedding": {0: "batch", 1: "time"}})
    g = torch.Generator().manual_seed(0)
    return export_component("attribute_heads", m, (torch.randn(2, T, in_dim, generator=g),), (torch.randn(1, T + 37, in_dim, generator=g),),
                            spec, out_dir, provenance=provenance)


def export_rmvpe(e2e, out_dir, provenance: dict | None = None) -> ExportResult:
    spec = ExportSpec("rmvpe", ["log_mel"], ["salience"], {"log_mel": {0: "batch", 2: "time32"}, "salience": {0: "batch", 1: "time"}})
    g = torch.Generator().manual_seed(0)
    return export_component("rmvpe", e2e, (torch.randn(2, 128, 64, generator=g),), (torch.randn(1, 128, 160, generator=g),), spec,
                            out_dir, tolerance=1e-4, provenance=provenance)


def _acoustic_args(cfg, B: int, T: int, g: torch.Generator) -> tuple:
    return (torch.randn(B, T, cfg.c_dim, generator=g), torch.ones(B, T, cfg.c_dim), torch.randn(B, T, cfg.r_dim, generator=g),
            torch.full((B, T), 220.0), torch.zeros(B, T), torch.randn(B, cfg.singer_dim, generator=g), torch.randn(B, cfg.env_dim, generator=g))


def export_acoustic(model, out_dir, provenance: dict | None = None) -> ExportResult:
    cfg = model.cfg
    names = ["c", "c_mask", "r", "f0_hz", "aperiodic", "singer", "env"]
    dyn = {n: {0: "batch", 1: "time"} for n in names[:5]} | {"singer": {0: "batch"}, "env": {0: "batch"}}
    dyn |= {"mel": {0: "batch", 1: "time"}, "ap": {0: "batch", 1: "time"}}
    g = torch.Generator().manual_seed(0)
    return export_component("acoustic", AcousticExport(model.acoustic), _acoustic_args(cfg, 2, 40, g), _acoustic_args(cfg, 1, 77, g),
                            ExportSpec("acoustic", names, ["mel", "ap"], dyn), out_dir, provenance=provenance)


def _vocoder_args(cfg, B: int, T: int, g: torch.Generator) -> tuple:
    n = (T - 1) * cfg.hop
    return (torch.randn(B, T, cfg.n_mels, generator=g), -torch.rand(B, T, cfg.n_ap, generator=g) * 20, torch.full((B, T), 220.0),
            torch.rand(B, T, generator=g) * 0.2, torch.randn(B, n, generator=g), torch.randn(B, n, generator=g))


def export_vocoder(model, out_dir, provenance: dict | None = None) -> ExportResult:
    cfg = model.cfg
    names = ["mel", "ap_db", "f0_hz", "rough", "jitter_noise", "source_noise"]
    dyn = {n: {0: "batch", 1: "time"} for n in names[:4]} | {n: {0: "batch", 1: "samples"} for n in names[4:]}
    dyn |= {"waveform": {0: "batch", 1: "samples"}}
    g = torch.Generator().manual_seed(0)
    return export_component("vocoder", VocoderExport(model.vocoder), _vocoder_args(cfg, 2, 12, g), _vocoder_args(cfg, 1, 23, g),
                            ExportSpec("vocoder", names, ["waveform"], dyn), out_dir, tolerance=1e-3, provenance=provenance, hop=cfg.hop)


def export_singer_encoder(model, out_dir, provenance: dict | None = None) -> ExportResult:
    cfg = model.cfg
    spec = ExportSpec("singer_encoder", ["log_mel"], ["singer"], {"log_mel": {0: "batch", 1: "time"}, "singer": {0: "batch"}})
    g = torch.Generator().manual_seed(0)
    return export_component("singer_encoder", model.singer_enc, (torch.randn(2, 60, cfg.n_mels, generator=g),),
                            (torch.randn(1, 131, cfg.n_mels, generator=g),), spec, out_dir, provenance=provenance)


def export_autoencoder(model, out_dir, provenance: dict | None = None) -> list[ExportResult]:
    return [export_acoustic(model, out_dir, provenance), export_vocoder(model, out_dir, provenance),
            export_singer_encoder(model, out_dir, provenance)]

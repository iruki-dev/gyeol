"""Frame-level feature encoders for the attribute heads and alignment.

* :class:`TorchSSLEncoder` — any frozen waveform model returning a list of
  layer outputs (torchaudio's ``extract_features`` convention); selected
  layers are averaged and projected onto the shared :class:`FrameGrid`.
  Builders load HuBERT / WavLM *architectures* from torchaudio and weights
  from a **local** checkpoint (e.g. one downloaded with ``gyeol fetch``).
  Nothing is downloaded here.
* :class:`DSPFrameFeatures` — a no-weights baseline made from the signal
  layer's curves (the v0.1-style baseline the learned heads must beat).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import torch

from ..core.containers import Representation
from ..core.grid import FrameGrid, project
from ..core.status import Result
from ..dsp.base import resample

SSL_SR = 16000


@runtime_checkable
class FrameEncoder(Protocol):
    name: str
    dim: int

    def encode(self, audio: np.ndarray, sr: int, grid: FrameGrid) -> Result[np.ndarray]: ...


class TorchSSLEncoder:
    """Frozen SSL model → (T, D) features on ``grid``.

    ``module(waveform[1, N]) -> (list[Tensor(1, T', D)], ...)`` as in
    torchaudio's ``Wav2Vec2Model.extract_features``; ``layers`` selects which
    outputs are averaged (low/mid layers carry phonation cues, upper layers
    phonetic content).
    """

    def __init__(self, module: torch.nn.Module, dim: int, layers: Sequence[int], frame_rate: float = 50.0,
                 name: str = "ssl", asset: str | None = None):
        self.module = module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.dim, self.layers, self.frame_rate, self.name, self.asset = dim, tuple(layers), frame_rate, name, asset

    @torch.no_grad()
    def encode(self, audio: np.ndarray, sr: int, grid: FrameGrid) -> Result[np.ndarray]:
        x = resample(np.asarray(audio, float), sr, SSL_SR)
        if len(x) < SSL_SR // 10:
            return Result.failure("audio shorter than 100 ms")
        wav = torch.tensor(x, dtype=torch.float32)[None]
        outs = self.module.extract_features(wav, num_layers=max(self.layers) + 1)[0] if hasattr(self.module, "extract_features") else self.module(wav)
        feats = torch.stack([outs[i][0] for i in self.layers]).mean(0).cpu().numpy()  # (T', D)
        times = (np.arange(len(feats)) + 0.5) / self.frame_rate
        return Result.success(project(times, feats, grid, max_gap=2.5 / self.frame_rate))


def _torchaudio_builder(kind: str):
    import torchaudio

    return {"hubert_base": torchaudio.models.hubert_base, "wavlm_base": torchaudio.models.wavlm_base}[kind]


def ssl_from_checkpoint(kind: str, checkpoint: str, layers: Sequence[int], asset: str | None = None) -> TorchSSLEncoder:
    """Build a torchaudio HuBERT/WavLM base model and load a local state dict.

    ``asset`` optionally names the weights in :mod:`gyeol.core.assets` (e.g. ``"hubert_fairseq"``), for provenance records.
    """
    model = _torchaudio_builder(kind)()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state)
    return TorchSSLEncoder(model, 768, layers, 50.0, kind, asset)


#: curves used by the DSP baseline encoder (masked values become 0, plus a validity flag each)
DSP_FEATURE_CURVES = ("pitch_center", "voicing", "aperiodic_ratio", "periodic_db", "subharmonic_ratio", "loudness_rel",
                      "vibrato_extent", "content")
#: fixed scales so feature magnitudes are comparable (cents → octaves, dB → tens of dB)
_SCALE = {"pitch_center": 1 / 1200.0, "aperiodic_ratio": 0.1, "periodic_db": 0.05, "loudness_rel": 0.1, "vibrato_extent": 0.02}


@dataclass
class DSPFrameFeatures:
    """Baseline features from an existing :class:`Representation` (no weights)."""

    curves: tuple[str, ...] = DSP_FEATURE_CURVES
    name: str = "dsp"

    def from_representation(self, rep: Representation) -> np.ndarray:
        cols = []
        for n in self.curves:
            c = rep.curves[n]
            v = (c.values if c.values.ndim == 2 else c.values[:, None]) * _SCALE.get(n, 1.0)
            ok = c.confidence > 0
            cols.append(np.where(ok[:, None], np.nan_to_num(v), 0.0))
            cols.append(ok[:, None].astype(float))
        return np.concatenate(cols, axis=1)

    def dim_for(self, rep: Representation) -> int:
        return self.from_representation(rep).shape[1]

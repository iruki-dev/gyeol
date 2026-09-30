"""Pitch-tracker adapters.

* DSP trackers (always available): pYIN, YIN and subharmonic summation,
  ported from v0.1 (:mod:`gyeol.pitch.dsp_trackers`).
* :class:`SwiftF0Tracker` – the ``swift-f0`` package (MIT, bundled ONNX).
* :class:`FCPETracker` – the ``torchfcpe`` package (MIT, bundled weights).
* :class:`RMVPETracker` – not implemented yet (needs a reimplementation that
  loads self-fetched Apache-2.0 weights).

Neural adapters check the license registry under the active profile when
constructed and return ``Status.UNAVAILABLE`` if their package is missing.
gyeol never downloads their weights; installing the package is the user's
explicit action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.license import Profile, lookup, require_allowed
from ..core.status import Result
from ..dsp.base import n_frames, resample
from . import dsp_trackers
from .base import PitchTrack

DSP_SR = 16000
DSP_HOP = 160  # 10 ms


def _prep(audio: np.ndarray, sr: int) -> Result[np.ndarray]:
    x = np.asarray(audio, dtype=float)
    if x.ndim != 1 or len(x) < int(0.05 * sr):
        return Result.failure("audio must be mono and at least 50 ms long")
    if not np.all(np.isfinite(x)):
        return Result.failure("audio contains NaN/inf")
    if np.max(np.abs(x)) == 0:
        return Result.failure("audio is digital silence")
    return Result.success(resample(x, sr, DSP_SR))


@dataclass
class _DSPTracker:
    impl: object
    name: str
    asset: str | None = None

    def track(self, audio: np.ndarray, sr: int) -> Result[PitchTrack]:
        pre = _prep(audio, sr)
        if not pre.ok:
            return Result.failure(pre.reason)
        x = pre.value
        n = n_frames(len(x), DSP_HOP)
        est = self.impl.track(x, DSP_SR, DSP_HOP, n)
        return Result.success(PitchTrack(self.name, np.arange(n) * DSP_HOP / DSP_SR, est.f0, np.clip(est.voiced_prob, 0, 1)))


def PyinTracker(fmin: float = 55.0, fmax: float = 1600.0) -> _DSPTracker:
    return _DSPTracker(dsp_trackers.PyinTracker(fmin=fmin, fmax=fmax), "pyin")


def YinTracker(fmin: float = 55.0, fmax: float = 1600.0) -> _DSPTracker:
    return _DSPTracker(dsp_trackers.YinTracker(fmin=fmin, fmax=fmax), "yin")


def SHSTracker(fmin: float = 55.0, fmax: float = 1600.0) -> _DSPTracker:
    return _DSPTracker(dsp_trackers.HarmonicSumTracker(fmin=fmin, fmax=fmax), "shs")


class SwiftF0Tracker:
    name = "swiftf0"
    asset = "swiftf0"

    def __init__(self, profile: Profile = Profile.COMMERCIAL, fmin: float | None = None, fmax: float | None = None):
        require_allowed(lookup(self.asset), profile, announce=False)
        self.fmin, self.fmax = fmin, fmax
        self._model = None

    def track(self, audio: np.ndarray, sr: int) -> Result[PitchTrack]:
        try:
            import swift_f0
        except ImportError:
            return Result.unavailable("swift-f0 is not installed (pip install swift-f0)")
        pre = _prep(audio, sr)
        if not pre.ok:
            return Result.failure(pre.reason)
        if self._model is None:
            self._model = swift_f0.SwiftF0()
        r = self._model.detect(pre.value, DSP_SR, fmin=self.fmin, fmax=self.fmax)
        voiced = r.confidence >= 0.5
        f0 = np.where(voiced & (r.pitch_hz > 0), r.pitch_hz, np.nan)
        return Result.success(PitchTrack(self.name, r.timestamps, f0, np.clip(r.confidence, 0, 1)))


class FCPETracker:
    name = "fcpe"
    asset = "fcpe"

    def __init__(self, profile: Profile = Profile.COMMERCIAL, fmin: float = 55.0, fmax: float = 1600.0, threshold: float = 0.006):
        require_allowed(lookup(self.asset), profile, announce=False)
        self.fmin, self.fmax, self.threshold = fmin, fmax, threshold
        self._model = None

    def track(self, audio: np.ndarray, sr: int) -> Result[PitchTrack]:
        try:
            import torch
            import torchfcpe
        except ImportError:
            return Result.unavailable("torchfcpe is not installed (pip install torchfcpe)")
        pre = _prep(audio, sr)
        if not pre.ok:
            return Result.failure(pre.reason)
        if self._model is None:
            self._model = torchfcpe.spawn_bundled_infer_model(device="cpu")
        wav = torch.tensor(pre.value, dtype=torch.float32)[None, :, None]
        with torch.no_grad():
            f0 = self._model.infer(wav, sr=DSP_SR, decoder_mode="local_argmax", threshold=self.threshold,
                                   f0_min=self.fmin, f0_max=self.fmax, interp_uv=False)
        f0 = f0.squeeze().cpu().numpy().astype(float)
        times = np.arange(len(f0)) * DSP_HOP / DSP_SR
        voiced = f0 > 0
        # FCPE gives no calibrated voicing probability; its mask is binary
        return Result.success(PitchTrack(self.name, times, np.where(voiced, f0, np.nan), voiced.astype(float)))


class RMVPETracker:
    name = "rmvpe"
    asset = "rmvpe"

    def __init__(self, weights_path: str, profile: Profile = Profile.COMMERCIAL):
        require_allowed(lookup(self.asset), profile, announce=False)
        raise NotImplementedError(
            "TODO(M8): reimplement the RMVPE network and verify it against weights fetched with `gyeol fetch rmvpe`; "
            "until then use SwiftF0Tracker / FCPETracker / DSP trackers"
        )

    def track(self, audio: np.ndarray, sr: int) -> Result[PitchTrack]:  # pragma: no cover
        raise NotImplementedError("TODO(M8)")


def default_trackers(profile: Profile = Profile.COMMERCIAL) -> list:
    """Neural trackers when installed, padded with DSP trackers to at least three."""
    out: list = []
    for cls in (SwiftF0Tracker, FCPETracker):
        pkg = "swift_f0" if cls is SwiftF0Tracker else "torchfcpe"
        try:
            __import__(pkg)
        except ImportError:
            continue
        out.append(cls(profile))
    for mk in (PyinTracker, SHSTracker, YinTracker):
        if len(out) >= 3:
            break
        out.append(mk())
    return out

"""Singing-voice separation adapters.

gyeol ships no separator weights.  Adapters wrap separators the user has
installed and fetched, and every adapter names its license-registry asset so
the active profile is enforced:

* :class:`CallableSeparator` – any ``f(audio, sr) -> vocals`` (e.g. a Mel- or
  BS-RoFormer inference function); pass the asset name for its weights
  (``"roformer_community"`` by default, whose training-data provenance is
  flagged as unclear).
* :class:`DemucsSeparator` – HTDemucs through the optional ``demucs`` package.
* :class:`BackingTrackCanceller` – the sing-along case: the backing track is
  known, so its leakage is cancelled by least squares per frequency.

Two separators can be compared with :func:`separation_agreement`
(symmetric SI-SDR), a cheap proxy for local separation reliability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import signal

from ..core.license import Profile, lookup, require_allowed
from ..core.status import Result
from ..dsp.base import resample, si_sdr, to_mono

EPS = 1e-12


class CallableSeparator:
    def __init__(self, fn: Callable[[np.ndarray, int], np.ndarray], name: str = "roformer", asset: str = "roformer_community",
                 profile: Profile = Profile.COMMERCIAL):
        require_allowed(lookup(asset), profile, announce=False)
        self.fn, self.name, self.asset = fn, name, asset

    def separate(self, audio: np.ndarray, sr: int) -> Result[np.ndarray]:
        try:
            out = to_mono(np.asarray(self.fn(audio, sr), dtype=float))
        except Exception as exc:  # noqa: BLE001 - surface separator failures as status
            return Result.failure(f"{self.name} failed: {exc}")
        if out.shape[0] != len(audio):
            return Result.failure(f"{self.name} returned {out.shape[0]} samples for {len(audio)} input samples")
        return Result.success(out)


class DemucsSeparator:
    name = "htdemucs"

    def __init__(self, model: str = "htdemucs", device: str = "cpu", asset: str | None = None, profile: Profile = Profile.COMMERCIAL):
        # HTDemucs is not yet in the registry; the caller must register its weights' license
        if asset is None:
            raise ValueError("register the Demucs weights in the license registry and pass asset=<name>")
        require_allowed(lookup(asset), profile, announce=False)
        self.model, self.device = model, device

    def separate(self, audio: np.ndarray, sr: int) -> Result[np.ndarray]:
        try:
            import torch
            from demucs.apply import apply_model
            from demucs.pretrained import get_model
        except ImportError:
            return Result.unavailable("demucs is not installed")
        m = get_model(self.model)
        m.eval()
        x = resample(to_mono(audio), sr, m.samplerate)
        wav = torch.tensor(np.stack([x, x]), dtype=torch.float32)[None]
        with torch.no_grad():
            out = apply_model(m, wav, device=self.device)[0]
        voc = out[m.sources.index("vocals")].mean(0).cpu().numpy()
        return Result.success(resample(voc, m.samplerate, sr)[: len(audio)])


@dataclass
class BackingTrackCanceller:
    backing: np.ndarray
    backing_sr: int
    nperseg: int = 2048
    max_lag_s: float = 1.0
    name: str = "backing-cancel"

    def separate(self, audio: np.ndarray, sr: int) -> Result[np.ndarray]:
        mix = to_mono(np.asarray(audio, float))
        b = resample(to_mono(self.backing), self.backing_sr, sr)
        n = min(len(mix), len(b))
        if n < 4 * self.nperseg:
            return Result.failure("signals too short for cancellation")
        max_lag = int(self.max_lag_s * sr)
        cc = signal.correlate(mix[:n], b[:n], mode="full", method="fft")
        lags = np.arange(-n + 1, n)
        sel = np.abs(lags) <= max_lag
        lag = int(lags[sel][np.argmax(np.abs(cc[sel]))])
        al = np.zeros(len(mix))
        if lag >= 0:
            seg = b[: len(mix) - lag]
            al[lag : lag + len(seg)] = seg
        else:
            seg = b[-lag : -lag + len(mix)]
            al[: len(seg)] = seg
        _, _, M = signal.stft(mix, sr, nperseg=self.nperseg)
        _, _, B = signal.stft(al, sr, nperseg=self.nperseg)
        H = np.sum(M * np.conj(B), axis=1) / (np.sum(np.abs(B) ** 2, axis=1) + EPS)
        _, v = signal.istft(M - H[:, None] * B, sr, nperseg=self.nperseg)
        return Result.success(v[: len(mix)])


def separation_agreement(stem_a: np.ndarray, stem_b: np.ndarray) -> float:
    """Symmetric SI-SDR (dB) between two separators' vocal stems."""
    return 0.5 * (si_sdr(stem_a, stem_b) + si_sdr(stem_b, stem_a))

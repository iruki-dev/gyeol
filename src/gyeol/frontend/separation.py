"""Singing-voice separation front-end.

The research recommends a Mel/BS-RoFormer vocal model as primary with
HTDemucs as a cross-check, using the disagreement between the two as a
cheap proxy for local separation unreliability (*UNVERIFIED HYPOTHESIS —
validate against ground-truth stems, research §2.2 / Test S*).

gyeol does not bundle model weights.  Adapters:

* :class:`CallableSeparator` – wrap any function ``f(audio, sr) -> vocals``
  (e.g. a BS-RoFormer / Mel-RoFormer inference script);
* :class:`DemucsSeparator`  – HTDemucs via the optional ``demucs`` package;
* :class:`BackingTrackCanceller` – karaoke case where the accompaniment is
  known: per-frequency least-squares cancellation of the backing track
  (*UNVERIFIED HYPOTHESIS that this beats blind separation on voice
  measures; test it*).

Enhancement / denoising is deliberately *not* offered: deep enhancement can
make acoustic features worse than the noisy input (research §2.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from scipy import signal

from .._dsp import EPS, resample, si_sdr, to_mono


@runtime_checkable
class Separator(Protocol):
    name: str

    def separate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Return the mono vocal stem at the input sample rate."""
        ...


@dataclass
class CallableSeparator:
    fn: Callable[[np.ndarray, int], np.ndarray]
    name: str = "callable"

    def separate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return to_mono(np.asarray(self.fn(audio, sr)))


@dataclass
class DemucsSeparator:
    """HTDemucs vocals (requires ``pip install demucs``)."""

    model: str = "htdemucs"
    device: str = "cpu"
    name: str = "htdemucs"

    def separate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        try:
            import torch
            from demucs.apply import apply_model
            from demucs.pretrained import get_model
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("DemucsSeparator needs the optional 'demucs' package") from exc
        model = get_model(self.model)
        model.eval()
        x = resample(to_mono(audio), sr, model.samplerate)
        wav = torch.tensor(np.stack([x, x]), dtype=torch.float32)[None]
        with torch.no_grad():
            out = apply_model(model, wav, device=self.device)[0]
        vocals = out[model.sources.index("vocals")].mean(0).cpu().numpy()
        return resample(vocals, model.samplerate, sr)


@dataclass
class BackingTrackCanceller:
    """Cancel a known backing track from a mixture.

    Estimates the per-frequency complex transfer H(f) from backing track to
    mixture (least squares over STFT frames, vocals treated as uncorrelated
    noise), subtracts H·B, then applies a mild spectral floor.
    """

    backing: np.ndarray
    backing_sr: int
    nperseg: int = 2048
    max_lag_s: float = 1.0
    name: str = "backing-cancel"

    def separate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        mix = to_mono(audio)
        b = resample(to_mono(self.backing), self.backing_sr, sr)
        b = _align(mix, b, int(self.max_lag_s * sr))
        _, _, M = signal.stft(mix, sr, nperseg=self.nperseg)
        _, _, B = signal.stft(b, sr, nperseg=self.nperseg)
        n = min(M.shape[1], B.shape[1])
        M, B = M[:, :n], B[:, :n]
        H = np.sum(M * np.conj(B), axis=1) / (np.sum(np.abs(B) ** 2, axis=1) + EPS)
        V = M - H[:, None] * B
        _, v = signal.istft(V, sr, nperseg=self.nperseg)
        return v[: len(mix)]


def _align(ref: np.ndarray, x: np.ndarray, max_lag: int) -> np.ndarray:
    n = min(len(ref), len(x), 30 * max(1, max_lag))
    corr = signal.correlate(ref[:n], x[:n], mode="full", method="fft")
    lags = np.arange(-n + 1, n)
    sel = np.abs(lags) <= max_lag
    lag = int(lags[sel][np.argmax(corr[sel])])
    out = np.zeros(len(ref))
    if lag >= 0:
        seg = x[: len(ref) - lag]
        out[lag : lag + len(seg)] = seg
    else:
        seg = x[-lag : -lag + len(ref)]
        out[: len(seg)] = seg
    return out


def separation_agreement(stem_a: np.ndarray, stem_b: np.ndarray) -> float:
    """Symmetric SI-SDR (dB) between two separators' vocal stems."""
    return 0.5 * (si_sdr(stem_a, stem_b) + si_sdr(stem_b, stem_a))

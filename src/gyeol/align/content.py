"""Content features for alignment.

Alignment must use content only, never voice-quality or pitch features;
otherwise the warp would absorb vibrato, scoops and timing-independent
differences.  M1 uses a DSP stand-in: liftered MFCCs (c1–c12, no energy) from
a 25 ms window plus a slow log-energy contour for silence/onset structure,
mean/variance-normalised per recording.  M3 replaces this with phonetic
posteriors from frozen SSL features behind the same protocol.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.fft import dct

from ..core.grid import FrameGrid
from ..dsp.base import resample

EPS = 1e-10
SR = 16000


class ContentFeatures(Protocol):
    name: str

    def extract(self, audio: np.ndarray, sr: int, grid: FrameGrid) -> np.ndarray: ...


def _mel_fb(n_mels: int, nfft: int, sr: int, fmin: float = 80.0, fmax: float = 7600.0) -> np.ndarray:
    mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, c, hi = pts[i : i + 3]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb


class MFCCContent:
    """Pitch-robust DSP content features (M1 stand-in for phonetic posteriors)."""

    name = "mfcc"

    def __init__(self, n_mels: int = 30, n_ceps: int = 12, win_seconds: float = 0.025, energy_weight: float = 1.0):
        self.n_mels, self.n_ceps, self.win_seconds, self.energy_weight = n_mels, n_ceps, win_seconds, energy_weight

    def extract(self, audio: np.ndarray, sr: int, grid: FrameGrid) -> np.ndarray:
        x = resample(np.asarray(audio, float), sr, SR)
        win = int(self.win_seconds * SR)
        nfft = 512
        centres = np.round(grid.times() * SR).astype(int)
        xp = np.pad(x, (win // 2, win))
        fr = xp[centres[:, None] + np.arange(win)[None, :]] * np.hamming(win)
        p = np.abs(np.fft.rfft(fr, nfft, axis=1)) ** 2
        mel = np.log(p @ _mel_fb(self.n_mels, nfft, SR).T + EPS)
        ceps = dct(mel, type=2, axis=1, norm="ortho")[:, 1 : self.n_ceps + 1]
        ceps *= 1 + 0.5 * 22 * np.sin(np.pi * np.arange(1, self.n_ceps + 1) / 22)  # sinusoidal lifter
        loge = np.log(p.sum(axis=1) + EPS)
        feats = np.column_stack([ceps, loge])
        feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)
        # weight after normalisation, otherwise CMVN undoes it
        feats[:, -1] *= self.energy_weight * np.sqrt(self.n_ceps)
        return feats

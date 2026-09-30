"""Loudness and periodic/aperiodic amplitudes (no learning).

* ``loudness``: A-weighted frame level in dB (relative to full scale), plus a
  gain-invariant ``loudness_rel`` (dB re the median voiced level).
* ``periodic_db`` / ``aperiodic_db``: harmonic-plus-noise split in the
  spirit of NANSY++'s periodic and aperiodic amplitudes.  Per frame the
  noise power density is estimated from inter-harmonic bins (more than a
  main-lobe away from every k·f0), extrapolated over the band, and the
  remainder is periodic.  ``aperiodic_ratio`` = aperiodic − periodic (dB).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..core.grid import FrameGrid
from ..dsp.base import resample

EPS = 1e-12
HN_SR = 22050


def _frames_at(x: np.ndarray, sr: int, grid: FrameGrid, win: int) -> np.ndarray:
    centres = np.round(grid.times() * sr).astype(int)
    xp = np.pad(x, (win // 2, win))
    return xp[centres[:, None] + np.arange(win)[None, :]]


def loudness(x: np.ndarray, sr: int, grid: FrameGrid, win_seconds: float = 0.046) -> np.ndarray:
    from ..io.loudness import a_weighting_sos

    y = signal.sosfilt(a_weighting_sos(sr), np.asarray(x, float))
    win = int(win_seconds * sr)
    fr = _frames_at(y, sr, grid, win) * np.hanning(win)
    return 10 * np.log10(np.mean(fr**2, axis=1) / np.mean(np.hanning(win) ** 2) + EPS)


def relative_loudness(loud_db: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    if not voiced.any():
        return np.full_like(loud_db, np.nan)
    return loud_db - np.median(loud_db[voiced])


def harmonic_noise(x: np.ndarray, sr: int, grid: FrameGrid, f0: np.ndarray, win_seconds: float = 0.064,
                   max_freq: float = 10000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(periodic_db, aperiodic_db, measurable) per grid frame."""
    xs = resample(np.asarray(x, float), sr, HN_SR)
    win = int(win_seconds * HN_SR)
    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    w = np.hanning(win)
    frames = _frames_at(xs, HN_SR, grid, win) * w
    freqs = np.fft.rfftfreq(nfft, 1 / HN_SR)
    band = (freqs >= 50) & (freqs <= max_freq)
    guard = 2.0 * HN_SR / win  # Hann main-lobe half width
    T = grid.n_frames
    per = np.full(T, np.nan)
    ape = np.full(T, np.nan)
    measurable = np.isfinite(f0) & (f0 > 2.2 * guard)
    rows = np.flatnonzero(measurable)
    norm = np.sum(w**2)
    for s in range(0, len(rows), 256):
        r = rows[s : s + 256]
        p = np.abs(np.fft.rfft(frames[r], nfft, axis=1)) ** 2 / norm
        f = f0[r][:, None]
        dist = np.abs(freqs[None, :] - np.round(freqs[None, :] / f) * f)
        noise_bin = (dist >= np.minimum(1.1 * guard, 0.45 * f)) & band[None, :]
        cnt = noise_bin.sum(axis=1)
        dens = (p * noise_bin).sum(axis=1) / np.maximum(cnt, 1)
        total = p[:, band].sum(axis=1) + EPS
        noise = np.minimum(dens * band.sum(), total)
        harm = np.maximum(total - noise, total * 1e-6)
        per[r] = 10 * np.log10(harm)
        ape[r] = 10 * np.log10(np.maximum(noise, EPS))
    return per, ape, measurable

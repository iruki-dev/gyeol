"""Smoothed cepstral peak prominence (CPPS).

Hillenbrand & Houde (1996)-type CPPS:

1. power spectrum in dB of each (Hann-windowed) frame,
2. power cepstrum in dB (|IFFT(dB spectrum)|²),
3. smoothing across time (``time_smooth_s``) and quefrency (``quef_smooth_s``),
4. a linear regression line over quefrency ≥ 1 ms,
5. peak prominence above that line in the pitch-period quefrency range.

The research requires an *f0-tracked* quefrency search, because at high f0
the peak at 1/f0 approaches the low-quefrency envelope region; when an f0
track is supplied the search is restricted to ±``track_tolerance`` around 1/f0.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .._dsp import EPS, frame


def cpps(
    x: np.ndarray,
    sr: int,
    hop: int,
    n_frames: int,
    f0: np.ndarray | None = None,
    fmin: float = 60.0,
    fmax: float = 1600.0,
    win_seconds: float = 0.048,
    time_smooth_s: float = 0.02,
    quef_smooth_s: float = 0.0005,
    regression_from_s: float = 0.001,
    track_tolerance: float = 0.15,
) -> np.ndarray:
    win = int(win_seconds * sr)
    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    frames = frame(x, win, hop, n_frames) * np.hanning(win)
    spec_db = 10 * np.log10(np.abs(np.fft.rfft(frames, nfft, axis=1)) ** 2 + EPS)
    ceps = np.abs(np.fft.irfft(spec_db, nfft, axis=1)[:, : nfft // 2]) ** 2
    quef = np.arange(nfft // 2) / sr
    # Smoothing is done on cepstral *power* (not dB) so a sharp, clean
    # rahmonic keeps its energy instead of being flattened by log-averaging.
    tw = max(1, int(round(time_smooth_s * sr / hop)))
    if tw > 1:
        ceps = ndimage.uniform_filter1d(ceps, tw, axis=0, mode="nearest")
    qw = max(1, int(round(quef_smooth_s * sr)))
    if qw > 1:
        ceps = ndimage.uniform_filter1d(ceps, qw, axis=1, mode="nearest")
    ceps_db = 10 * np.log10(np.maximum(ceps, 0.0) + EPS)
    # regression line over quefrency >= regression_from_s
    q_lo = int(np.ceil(regression_from_s * sr))
    q_hi = nfft // 2
    qx = quef[q_lo:q_hi]
    qy = ceps_db[:, q_lo:q_hi]
    qm = qx.mean()
    slope = ((qx - qm) * (qy - qy.mean(axis=1, keepdims=True))).sum(axis=1) / ((qx - qm) ** 2).sum()
    intercept = qy.mean(axis=1) - slope * qm
    # peak search range
    lo_q = np.full(n_frames, max(1.0 / fmax, 2.0 / sr))
    hi_q = np.full(n_frames, 1.0 / fmin)
    if f0 is not None:
        tracked = np.isfinite(f0)
        lo_q[tracked] = 1.0 / (f0[tracked] * (1 + track_tolerance))
        hi_q[tracked] = 1.0 / (f0[tracked] * (1 - track_tolerance))
    lo_i = np.clip(np.floor(lo_q * sr).astype(int), 1, q_hi - 2)
    hi_i = np.clip(np.ceil(hi_q * sr).astype(int), 2, q_hi - 1)
    idx = np.arange(q_hi)[None, :]
    in_range = (idx >= lo_i[:, None]) & (idx <= hi_i[:, None])
    masked = np.where(in_range, ceps_db, -np.inf)
    peak_i = masked.argmax(axis=1)
    peak = masked[np.arange(n_frames), peak_i]
    line = slope * quef[peak_i] + intercept
    return peak - line


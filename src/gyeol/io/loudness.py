"""Loudness: ITU-R BS.1770 integrated loudness / normalisation and A-weighting.

The K-weighting filters are derived from their analog prototypes for any
sample rate (the same approach as pyloudnorm), so 44.1 kHz input needs no
resampling.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..core.status import Result


def _k_weighting(sr: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-weighting biquads for any ``sr`` (De Man's derivation of BS.1770)."""
    # stage 1: high shelf
    G, Q, fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    K = np.tan(np.pi * fc / sr)
    Vh = 10 ** (G / 20)
    Vb = Vh**0.4996667741545416
    a0 = 1 + K / Q + K * K
    shelf = (np.array([(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]),
             np.array([1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]))
    # stage 2: RLB high-pass
    Q, fc = 0.5003270373238773, 38.13547087602444
    K = np.tan(np.pi * fc / sr)
    a0 = 1 + K / Q + K * K
    hp = (np.array([1.0, -2.0, 1.0]), np.array([1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]))
    return [shelf, hp]


def integrated_loudness(x: np.ndarray, sr: int) -> Result[float]:
    """BS.1770-4 gated integrated loudness in LUFS (mono)."""
    x = np.asarray(x, dtype=float)
    if len(x) < int(0.4 * sr):
        return Result.failure("need at least 400 ms of audio for BS.1770 gating")
    y = x
    for b, a in _k_weighting(sr):
        y = signal.lfilter(b, a, y)
    block, step = int(0.4 * sr), int(0.1 * sr)
    n = 1 + (len(y) - block) // step
    idx = np.arange(block)[None, :] + step * np.arange(n)[:, None]
    z = np.mean(y[idx] ** 2, axis=1)
    lk = -0.691 + 10 * np.log10(z + 1e-20)
    gated = z[lk > -70.0]
    if gated.size == 0:
        return Result.failure("signal below the -70 LUFS absolute gate")
    rel = -0.691 + 10 * np.log10(gated.mean()) - 10.0
    gated = z[(lk > -70.0) & (lk > rel)]
    return Result.success(float(-0.691 + 10 * np.log10(gated.mean())))


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float = -23.0, max_gain_db: float = 40.0) -> Result[tuple[np.ndarray, float]]:
    """Scale to ``target_lufs``.  Returns (audio, applied gain in dB)."""
    r = integrated_loudness(x, sr)
    if not r.ok:
        return Result.failure(r.reason)
    gain = float(np.clip(target_lufs - r.value, -max_gain_db, max_gain_db))
    y = np.asarray(x, float) * 10 ** (gain / 20)
    if np.max(np.abs(y)) > 1.0:
        return Result.unreliable((y, gain), "normalised signal exceeds full scale")
    return Result.success((y, gain))


def a_weighting_sos(sr: int) -> np.ndarray:
    """IEC 61672 A-weighting as a digital SOS filter (bilinear transform).

    Exact at low and mid frequencies; bilinear warping makes it ≈1.5 dB too
    low near 10 kHz at 44.1 kHz, which is irrelevant for a vocal loudness curve.
    """
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    a1000 = 1.9997
    z = [0, 0, 0, 0]
    p = [-2 * np.pi * f1, -2 * np.pi * f1, -2 * np.pi * f4, -2 * np.pi * f4, -2 * np.pi * f2, -2 * np.pi * f3]
    k = (2 * np.pi * f4) ** 2 * 10 ** (a1000 / 20)
    zd, pd, kd = signal.bilinear_zpk(z, p, k, sr)
    return signal.zpk2sos(zd, pd, kd)

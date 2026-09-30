"""Period-to-period perturbation (jitter / shimmer) for the irregularity index.

The research restricts jitter/shimmer to *sustained, non-vibrato* segments at
SNR ≥ 30 dB, because vibrato inflates both and lossy codecs raise both.  The
gating is applied by the validity layer; this module only measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import parabolic_peak


@dataclass
class Perturbation:
    jitter_local: float  # %
    shimmer_local: float  # %
    shimmer_db: float
    n_periods: int


def period_marks(x: np.ndarray, sr: int, f0_at: callable, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Peak-picking period marks guided by an f0 function of sample index.

    Returns (mark positions in samples, per-period peak-to-peak amplitudes).
    """
    seg = x[start:end]
    if len(seg) < 4:
        return np.empty(0), np.empty(0)
    polarity = 1.0 if np.max(seg) >= -np.min(seg) else -1.0
    y = polarity * x
    t0 = sr / f0_at(start)
    first = start + int(np.argmax(y[start : min(end, start + int(1.2 * t0))]))
    marks = [float(first)]
    amps = []
    pos = first
    while True:
        t0 = sr / f0_at(pos)
        lo, hi = int(pos + 0.8 * t0), int(pos + 1.2 * t0)
        if hi >= end:
            break
        i = lo + int(np.argmax(y[lo:hi]))
        frac, _ = parabolic_peak(y[None, i - 1 : i + 2], np.array([1]))
        marks.append(i - 1 + float(frac[0]))
        amps.append(float(np.ptp(x[pos:i])))
        pos = i
    return np.asarray(marks), np.asarray(amps)


def perturbation(x: np.ndarray, sr: int, f0_hz: np.ndarray, hop: int, start_frame: int, end_frame: int) -> Perturbation | None:
    start, end = start_frame * hop, min(len(x), end_frame * hop)

    def f0_at(sample: float) -> float:
        i = int(np.clip(round(sample / hop), start_frame, end_frame - 1))
        v = f0_hz[i]
        return float(v) if np.isfinite(v) else float(np.nanmedian(f0_hz[start_frame:end_frame]))

    marks, amps = period_marks(x, sr, f0_at, start, end)
    if len(marks) < 6:
        return None
    periods = np.diff(marks) / sr
    amps = amps[: len(periods)]
    jit = np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100
    shim = np.mean(np.abs(np.diff(amps))) / np.mean(amps) * 100
    with np.errstate(divide="ignore"):
        shim_db = float(np.mean(np.abs(20 * np.log10(amps[1:] / amps[:-1]))))
    return Perturbation(jitter_local=float(jit), shimmer_local=float(shim), shimmer_db=shim_db, n_periods=len(periods))

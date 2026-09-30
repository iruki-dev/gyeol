"""Note segmentation and vibrato descriptors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .._dsp import moving_average, runs


@dataclass
class NoteSpan:
    start: int  # frame index (inclusive)
    end: int  # frame index (exclusive)


def segment_notes(
    cents: np.ndarray,
    voiced: np.ndarray,
    hop_seconds: float,
    smooth_seconds: float = 0.25,
    jump_cents: float = 70.0,
    min_note_seconds: float = 0.1,
    max_gap_seconds: float = 0.03,
) -> list[NoteSpan]:
    """Split voiced runs into notes at steps of the vibrato-smoothed pitch.

    The smoothing window spans ≈1.5 vibrato cycles so vibrato itself does
    not create note boundaries.  External score / MIDI segmentation can be
    passed to the engine instead.
    """
    gap = int(round(max_gap_seconds / hop_seconds))
    v = voiced.copy()
    for s, e in runs(~voiced):
        if 0 < s and e < len(v) and e - s <= gap:
            v[s:e] = True  # bridge tiny dropouts
    width = max(3, int(round(smooth_seconds / hop_seconds)))
    min_len = max(2, int(round(min_note_seconds / hop_seconds)))
    notes: list[NoteSpan] = []
    for s, e in runs(v):
        if e - s < min_len:
            continue
        seg = cents[s:e]
        sm = moving_average(seg, width)
        lag = max(1, width // 2)
        step = np.zeros(e - s)
        step[lag:-lag] = sm[2 * lag :] - sm[: -2 * lag] if e - s > 2 * lag else 0
        peaks, _ = signal.find_peaks(np.abs(np.nan_to_num(step)), height=jump_cents, distance=min_len)
        bounds = [0] + [int(p) for p in peaks] + [e - s]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a >= min_len:
                notes.append(NoteSpan(s + a, s + b))
    return notes


@dataclass
class Vibrato:
    rate_hz: float
    extent_cents: float  # semi-extent (peak deviation)
    rate_cv: float
    extent_cv: float
    n_cycles: float


def vibrato(cents: np.ndarray, hop_seconds: float, min_cycles: float = 2.0, rate_range: tuple[float, float] = (3.0, 9.0), detrend_seconds: float = 0.4) -> Vibrato | None:
    """Vibrato rate / extent / regularity from a note's f0 contour (cents).

    The contour is detrended with a moving average, the rate comes from the
    autocorrelation peak within ``rate_range`` and a least-squares sinusoid
    fit gives the extent; regularity is the CV of per-half-cycle rates and
    extents.  Returns None if fewer than ``min_cycles`` cycles are present.
    """
    c = np.asarray(cents, dtype=float)
    if np.isnan(c).any():
        idx = np.arange(len(c))
        ok = np.isfinite(c)
        if ok.sum() < 4:
            return None
        c = np.interp(idx, idx[ok], c[ok])
    fs = 1.0 / hop_seconds
    dur = len(c) * hop_seconds
    if dur * rate_range[1] < min_cycles:
        return None  # too short for min_cycles even at the fastest rate
    trend = moving_average(c, max(3, int(round(detrend_seconds * fs))))
    d = c - trend
    d = d - d.mean()
    if np.std(d) < 1e-6:
        return None
    ac = np.correlate(d, d, mode="full")[len(d) - 1 :]
    ac /= ac[0]
    lo, hi = int(np.floor(fs / rate_range[1])), int(np.ceil(fs / rate_range[0]))
    hi = min(hi, len(ac) - 2)
    if hi <= lo + 1:
        return None
    lag = lo + int(np.argmax(ac[lo : hi + 1]))
    if ac[lag] < 0.2:
        return None
    rate = fs / lag
    n_cycles = dur * rate
    if n_cycles < min_cycles:
        return None
    # A moving average spanning exactly one vibrato period nulls the vibrato,
    # so re-detrend with it before fitting the extent.
    period = max(3, int(round(fs / rate)))
    d = c - moving_average(c, period)
    d = d - d.mean()
    t = np.arange(len(d)) / fs

    def fit(r: float) -> float:
        design = np.stack([np.sin(2 * np.pi * r * t), np.cos(2 * np.pi * r * t)], axis=1)
        coef, *_ = np.linalg.lstsq(design, d, rcond=None)
        return float(np.hypot(*coef))

    # refine the integer-lag rate estimate by maximising the fitted amplitude
    grid = rate * (1 + np.linspace(-0.1, 0.1, 41))
    amps = [fit(r) for r in grid]
    rate = float(grid[int(np.argmax(amps))])
    extent = float(np.max(amps))
    n_cycles = dur * rate
    # regularity from zero crossings / half-cycle extrema
    zc = np.flatnonzero(np.diff(np.signbit(d).astype(np.int8)) != 0)
    halves = np.diff(zc) * hop_seconds
    rate_cv = float(np.std(halves) / np.mean(halves)) if len(halves) >= 3 else float("nan")
    ext = [np.max(np.abs(d[a:b])) for a, b in zip(zc[:-1], zc[1:]) if b > a]
    extent_cv = float(np.std(ext) / np.mean(ext)) if len(ext) >= 3 and np.mean(ext) > 0 else float("nan")
    return Vibrato(rate_hz=float(rate), extent_cents=extent, rate_cv=rate_cv, extent_cv=extent_cv, n_cycles=float(n_cycles))

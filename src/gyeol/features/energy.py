"""Energy and onset group.

Absolute SPL is deliberately *omitted*: AGC and unknown gain make it
unrecoverable from consumer recordings.  Energy is stored relative to the
median of the voiced frames of its phrase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._dsp import frame, power_db, runs


def frame_energy_db(x: np.ndarray, hop: int, n_frames: int, win: int) -> np.ndarray:
    frames = frame(x, win, hop, n_frames)
    return power_db(np.mean(frames**2, axis=1), floor_db=-150.0)


def phrases(voiced: np.ndarray, hop_seconds: float, min_pause_seconds: float = 0.3) -> list[tuple[int, int]]:
    """Group voiced runs into phrases separated by pauses >= min_pause."""
    gap = int(round(min_pause_seconds / hop_seconds))
    out: list[tuple[int, int]] = []
    for s, e in runs(voiced):
        if out and s - out[-1][1] < gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def relative_energy(energy_db: np.ndarray, voiced: np.ndarray, hop_seconds: float) -> np.ndarray:
    rel = np.full_like(energy_db, np.nan)
    for s, e in phrases(voiced, hop_seconds):
        med = np.median(energy_db[s:e][voiced[s:e]])
        rel[s:e] = energy_db[s:e] - med
    return rel


@dataclass
class Onset:
    rise_time_ms: float  # 10–90 % energy rise
    f0_settle_ms: float  # voicing onset -> f0 within tolerance of note median
    pre_voicing_aperiodicity_db: float  # mean high-band aperiodicity, first 30 ms
    early_h1h2c_db: float  # median H1*–H2* in the first 50 ms


def onset_descriptors(
    start: int,
    end: int,
    energy_db: np.ndarray,
    cents: np.ndarray,
    aperiodicity_hi: np.ndarray,
    h1h2c: np.ndarray,
    hop_seconds: float,
    settle_tolerance_cents: float = 50.0,
    vibrato_extent_cents: float = 0.0,
) -> Onset:
    """Onset descriptors of one note (frame indices [start, end))."""
    fs = 1.0 / hop_seconds
    pre = max(0, start - int(0.05 * fs))
    win_end = min(end, start + int(0.3 * fs))
    e = 10 ** (energy_db[pre:win_end] / 20)
    rise = float("nan")
    if e.size >= 3:
        lo_e = e[: max(1, start - pre + 1)].min()
        pk = int(np.argmax(e))
        span = e[pk] - lo_e
        if span > 0:
            above10 = np.flatnonzero(e[: pk + 1] >= lo_e + 0.1 * span)
            above90 = np.flatnonzero(e[: pk + 1] >= lo_e + 0.9 * span)
            if above10.size and above90.size:
                rise = (above90[0] - above10[0]) * hop_seconds * 1000
    c = cents[start:end]
    settle = float("nan")
    if np.isfinite(c).sum() >= 3:
        target = np.nanmedian(c)
        tol = settle_tolerance_cents + vibrato_extent_cents
        inside = np.flatnonzero(np.abs(c - target) <= tol)
        if inside.size:
            settle = inside[0] * hop_seconds * 1000
    n30, n50 = max(1, int(0.03 * fs)), max(1, int(0.05 * fs))
    ap = aperiodicity_hi[start : start + n30]
    hh = h1h2c[start : start + n50]
    with np.errstate(all="ignore"):
        pre_ap = float(np.nanmean(ap)) if np.isfinite(ap).any() else float("nan")
        early = float(np.nanmedian(hh)) if np.isfinite(hh).any() else float("nan")
    return Onset(rise_time_ms=rise, f0_settle_ms=settle, pre_voicing_aperiodicity_db=pre_ap, early_h1h2c_db=early)

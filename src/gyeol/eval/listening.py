"""Calibrating the audibility score against listeners (protocol:
``docs/protocols/listening_test.md``).

Trials: a listener hears the unedited render and the render with one item
corrected (2-interval forced choice, or same/different) and answers whether
they differ.  :func:`calibrate_audibility` fits detection rate vs. the M5
audibility score with an isotonic (monotone) curve and returns the score at
which detection reaches the criterion (75 % for 2AFC, halfway between chance
and perfect).  That score is the **perceptual floor** a
:class:`gyeol.coach.ThresholdSet` can carry (``audibility_floor``): items the
renderer says are quieter than that are not shown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..verification.thresholds import _pava


@dataclass
class AudibilityCalibration:
    floor: float | None  # audibility score at the criterion; None = never reached in the tested range
    criterion: float
    levels: np.ndarray  # bin centres of the audibility score
    detection: np.ndarray  # isotonic detection rate per bin
    n_trials: int
    n_listeners: int


def calibrate_audibility(scores: np.ndarray, detected: np.ndarray, listeners: np.ndarray | None = None, criterion: float = 0.75,
                         n_bins: int = 8, min_per_bin: int = 10) -> AudibilityCalibration:
    s = np.asarray(scores, float)
    d = np.asarray(detected, float)
    ok = np.isfinite(s) & np.isfinite(d)
    s, d = s[ok], d[ok]
    if len(s) < n_bins * min_per_bin // 2:
        raise ValueError(f"too few trials ({len(s)}) for {n_bins} bins")
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1)
    keep = [b for b in range(n_bins) if (idx == b).sum() >= min_per_bin]
    lv = np.array([s[idx == b].mean() for b in keep])
    rate = np.array([d[idx == b].mean() for b in keep])
    w = np.array([(idx == b).sum() for b in keep], float)
    fit = _pava(rate, w, increasing=True)
    above = np.flatnonzero(fit >= criterion)
    floor = None
    if above.size:
        j = int(above[0])
        floor = float(lv[j]) if j == 0 or fit[j] == fit[j - 1] else float(np.interp(criterion, [fit[j - 1], fit[j]], [lv[j - 1], lv[j]]))
    nl = len(np.unique(np.asarray(listeners)[ok])) if listeners is not None else 0
    return AudibilityCalibration(floor, criterion, lv, fit, int(len(s)), int(nl))

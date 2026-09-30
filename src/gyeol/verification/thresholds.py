"""Validity-condition testing (research §5.iii).

For a dimension d and a nuisance axis (SNR, T60, codec bitrate, separation
SI-SDR, f0 height …) the *operating threshold* is the nuisance level at which
the upper 95 % prediction bound of |d_degraded − d_reference| crosses
MDC95(d).

The research suggests a GAM for the error curve.  To stay dependency-free
this module uses binned upper quantiles made monotone with the
pool-adjacent-violators algorithm — a conservative surrogate of the GAM
prediction bound.  Replace with a GAM (e.g. pyGAM) for publication-grade
thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pava(y: np.ndarray, w: np.ndarray, increasing: bool) -> np.ndarray:
    """Weighted isotonic regression (pool adjacent violators)."""
    y = np.asarray(y, float) if increasing else -np.asarray(y, float)
    vals, wts, sizes = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, s2 = vals.pop(), wts.pop(), sizes.pop()
            v1, w1, s1 = vals.pop(), wts.pop(), sizes.pop()
            wt = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / wt)
            wts.append(wt)
            sizes.append(s1 + s2)
    out = np.repeat(vals, sizes)
    return out if increasing else -out


@dataclass
class ThresholdResult:
    threshold: float | None  # nuisance level where the bound crosses the MDC
    levels: np.ndarray  # bin centres
    bound: np.ndarray  # monotone upper error bound per bin
    mdc: float
    higher_is_better: bool

    def is_valid(self, level: float) -> bool:
        if self.threshold is None:
            return False
        return level >= self.threshold if self.higher_is_better else level <= self.threshold


def operating_threshold(
    nuisance: np.ndarray,
    error: np.ndarray,
    mdc: float,
    higher_is_better: bool = True,
    n_bins: int = 10,
    quantile: float = 0.95,
    min_per_bin: int = 5,
) -> ThresholdResult:
    """Find where the error bound falls below ``mdc``.

    ``higher_is_better``: True for axes like SNR / SI-SDR / bitrate (error
    shrinks as the level grows), False for T60 or f0 height.
    """
    x = np.asarray(nuisance, float)
    e = np.abs(np.asarray(error, float))
    ok = np.isfinite(x) & np.isfinite(e)
    x, e = x[ok], e[ok]
    uniq = np.unique(x)
    if len(uniq) <= n_bins:
        edges_lv = uniq
        groups = [e[x == u] for u in uniq]
    else:
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        edges_lv = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
        groups = [e[idx == i] for i in range(n_bins)]
    keep = [i for i, g in enumerate(groups) if len(g) >= min_per_bin]
    levels = np.asarray(edges_lv)[keep]
    q = np.array([np.quantile(groups[i], quantile) for i in keep])
    w = np.array([len(groups[i]) for i in keep], float)
    if len(levels) == 0:
        return ThresholdResult(None, levels, q, mdc, higher_is_better)
    # error should decrease with level when higher is better
    bound = _pava(q, w, increasing=not higher_is_better)
    below = bound <= mdc
    thr: float | None = None
    if higher_is_better:
        # lowest level from which the bound stays below the MDC
        if below[-1]:
            j = len(below) - 1
            while j > 0 and below[j - 1]:
                j -= 1
            thr = float(levels[j]) if j == 0 else float(np.interp(mdc, [bound[j], bound[j - 1]], [levels[j], levels[j - 1]]))
    else:
        if below[0]:
            j = 0
            while j < len(below) - 1 and below[j + 1]:
                j += 1
            thr = float(levels[j]) if j == len(below) - 1 else float(np.interp(mdc, [bound[j], bound[j + 1]], [levels[j], levels[j + 1]]))
    return ThresholdResult(thr, levels, bound, mdc, higher_is_better)

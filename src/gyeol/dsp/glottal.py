"""Glottal source parameters by inverse filtering: NAQ, QOQ and Rd.

The research names QCP inverse filtering (Airaksinen et al., TASLP 2014) as
the primary estimator with IAIF (Alku, Speech Communication 1992) as the
fallback.  gyeol implements IAIF; QCP can be plugged in through
:class:`InverseFilter`.

Per glottal cycle (delimited by the negative peaks of the flow derivative):

* NAQ = f_ac / (d_peak · T)       (Alku, Bäckström & Vilkman, JASA 2002)
* QOQ = fraction of the cycle where flow > min + 50 % of the AC amplitude
* Rd  ≈ NAQ / 0.11                (Fant 1995: Rd = (1/0.11)·f0·U0/Ee, and
                                   NAQ = f0·U0/Ee when d_peak ≈ Ee)

*UNVERIFIED HYPOTHESIS (research §4.2): inverse-filtering validity above
≈500 Hz f0 must be validated against EGG-derived contact quotient.*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import signal

from .base import EPS, lpc


class InverseFilter(Protocol):
    def __call__(self, x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (glottal flow, glottal flow derivative) for a segment."""
        ...


def iaif(x: np.ndarray, sr: int, vt_order: int | None = None, gl_order: int = 4, leak: float = 0.99) -> tuple[np.ndarray, np.ndarray]:
    """Iterative Adaptive Inverse Filtering (Alku 1992)."""
    p = vt_order or int(round(sr / 1000)) + 2
    win = np.hanning(len(x))
    integ = lambda s: signal.lfilter([1.0], [1.0, -leak], s)  # noqa: E731

    def lp(s: np.ndarray, order: int) -> np.ndarray:
        return lpc((s * win)[None, :], order)[0]

    hg1 = lp(x, 1)
    y = signal.lfilter(hg1, [1.0], x)
    hvt1 = lp(y, p)
    g1 = integ(signal.lfilter(hvt1, [1.0], x))
    hg2 = lp(g1, gl_order)
    y = integ(signal.lfilter(hg2, [1.0], x))
    hvt2 = lp(y, p)
    dg = signal.lfilter(hvt2, [1.0], x)
    g = integ(dg)
    return g, dg


@dataclass
class GlottalParams:
    naq: float
    qoq: float
    rd: float
    n_cycles: int


def cycle_params(g: np.ndarray, dg: np.ndarray, sr: int, f0: float) -> GlottalParams | None:
    """NAQ / QOQ from inverse-filtered flow over the cycles of one segment."""
    period = sr / f0
    # negative peaks of the flow derivative mark glottal closure instants
    dist = max(1, int(0.7 * period))
    peaks, _ = signal.find_peaks(-dg, distance=dist)
    if len(peaks) < 3:
        return None
    naqs, qoqs = [], []
    for a, b in zip(peaks[:-1], peaks[1:]):
        t = b - a
        if not (0.7 * period <= t <= 1.3 * period):
            continue
        cyc = g[a:b]
        # remove the linear drift left by leaky integration
        cyc = cyc - np.linspace(cyc[0], g[b], len(cyc), endpoint=False)
        f_ac = cyc.max() - cyc.min()
        d_peak = -dg[b]
        if f_ac <= EPS or d_peak <= EPS:
            continue
        naqs.append(f_ac / (d_peak * t))
        qoqs.append(np.mean(cyc > cyc.min() + 0.5 * f_ac))
    if len(naqs) < 2:
        return None
    naq = float(np.median(naqs))
    return GlottalParams(naq=naq, qoq=float(np.median(qoqs)), rd=naq / 0.11, n_cycles=len(naqs))


def glottal_track(
    x: np.ndarray,
    sr: int,
    hop: int,
    f0: np.ndarray,
    stride: int = 2,
    min_periods: int = 4,
    min_seconds: float = 0.03,
    inverse_filter: InverseFilter | None = None,
) -> dict[str, np.ndarray]:
    """Frame-level NAQ / QOQ / Rd on voiced frames (every ``stride`` frames).

    Each analysis segment spans max(min_periods / f0, min_seconds) around the
    frame centre; values are forward-filled over the skipped frames.
    """
    inverse_filter = inverse_filter or iaif
    t_n = len(f0)
    out = {k: np.full(t_n, np.nan) for k in ("naq", "qoq", "rd")}
    last = -10
    for t in np.flatnonzero(np.isfinite(f0)):
        if t - last < stride and last >= 0 and np.isfinite(out["naq"][last]):
            for k in out:
                out[k][t] = out[k][last]
            continue
        last = t
        half = int(max(min_periods / f0[t], min_seconds) * sr / 2)
        c = t * hop
        lo, hi = c - half, c + half
        if lo < 0 or hi > len(x):
            continue
        seg = x[lo:hi]
        if np.max(np.abs(seg)) < EPS:
            continue
        g, dg = inverse_filter(seg, sr)
        pr = cycle_params(g, dg, sr, f0[t])
        if pr is not None:
            out["naq"][t], out["qoq"][t], out["rd"][t] = pr.naq, pr.qoq, pr.rd
    return out

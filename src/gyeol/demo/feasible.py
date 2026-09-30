"""The user's feasible range, and clamping of edits to it.

A demo must stay inside what *this* user can sing: edits that would push the
pitch, level or aperiodicity outside the range are clipped, and the clipped
fraction is reported.  The range comes from

* onboarding (range / tessitura measured by the coach layer, M6) via
  :meth:`FeasibleRange.from_onboarding`, or
* the user's own takes via :meth:`FeasibleRange.from_takes` (observed
  percentiles plus a margin).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..core.containers import Representation
from ..dsp.base import hz_to_cents
from .edits import Edit


@dataclass(frozen=True)
class FeasibleRange:
    f0_low_cents: float  # cents re A4
    f0_high_cents: float
    loudness_high_db: float = np.inf  # absolute frame level (dBFS, A-weighted)
    aperiodic_low_db: float = -np.inf
    aperiodic_high_db: float = np.inf
    source: str = "unspecified"

    @classmethod
    def from_onboarding(cls, low_hz: float, high_hz: float, **kw) -> "FeasibleRange":
        if not 0 < low_hz < high_hz:
            raise ValueError("need 0 < low_hz < high_hz")
        return cls(float(hz_to_cents(np.array([low_hz]))[0]), float(hz_to_cents(np.array([high_hz]))[0]),
                   source="onboarding", **kw)

    @classmethod
    def from_takes(cls, reps: list[Representation], margin_cents: float = 100.0, loudness_margin_db: float = 3.0,
                   aperiodic_margin_db: float = 6.0, min_confidence: float = 0.5) -> "FeasibleRange":
        f0, loud, ap = [], [], []
        for r in reps:
            c = r.curves["f0_cents"]
            ok = (c.confidence >= min_confidence) & np.isfinite(c.values)
            f0.append(c.values[ok])
            if "loudness" in r.curves:
                lc = r.curves["loudness"]
                loud.append(lc.values[ok & np.isfinite(lc.values)])
            if "aperiodic_ratio" in r.curves:
                a = r.curves["aperiodic_ratio"]
                ap.append(a.values[(a.confidence > 0) & np.isfinite(a.values)])
        f = np.concatenate(f0) if f0 else np.array([])
        if f.size < 5:
            raise ValueError("too few confident pitch frames to measure a range")
        lo, hi = np.percentile(f, [1, 99])
        la = np.concatenate(loud) if loud else np.array([])
        aa = np.concatenate(ap) if ap else np.array([])
        return cls(float(lo - margin_cents), float(hi + margin_cents),
                   float(np.percentile(la, 99) + loudness_margin_db) if la.size else np.inf,
                   float(np.percentile(aa, 1) - aperiodic_margin_db) if aa.size else -np.inf,
                   float(np.percentile(aa, 99) + aperiodic_margin_db) if aa.size else np.inf,
                   source="observed takes")


@dataclass
class ClampReport:
    clamped_fraction: float  # of the frames the edit acts on
    clamped_frames: int
    detail: dict


def clamp_edit(edit: Edit, user: Representation, rng: FeasibleRange) -> tuple[Edit, ClampReport]:
    """Clip the edit so that the edited curves stay inside ``rng``."""
    T = edit.n_frames
    active = edit.support(dilate=0)
    clamped = np.zeros(T, bool)
    detail = {}
    out = replace(edit, curves=dict(edit.curves))
    if edit.f0_cents is not None:
        f0 = user.curves["f0_cents"].values
        new = f0 + edit.f0_cents
        lim = np.clip(new, rng.f0_low_cents, rng.f0_high_cents)
        hit = np.isfinite(new) & (np.abs(lim - new) > 1e-9)
        out.f0_cents = np.where(hit, lim - f0, edit.f0_cents)
        clamped |= hit
        detail["f0"] = int(hit.sum())
    if edit.gain_db is not None and "loudness" in user.curves and np.isfinite(rng.loudness_high_db):
        loud = user.curves["loudness"].values
        new = loud + edit.gain_db
        hit = np.isfinite(new) & (new > rng.loudness_high_db) & (edit.gain_db > 0)
        out.gain_db = np.where(hit, np.maximum(rng.loudness_high_db - loud, 0.0), edit.gain_db)
        clamped |= hit
        detail["gain"] = int(hit.sum())
    if edit.aperiodic_db is not None and "aperiodic_ratio" in user.curves:
        a = user.curves["aperiodic_ratio"].values
        new = a + edit.aperiodic_db
        lim = np.clip(new, rng.aperiodic_low_db, rng.aperiodic_high_db)
        hit = np.isfinite(new) & (np.abs(lim - new) > 1e-9)
        out.aperiodic_db = np.where(hit, lim - a, edit.aperiodic_db)
        clamped |= hit
        detail["aperiodic"] = int(hit.sum())
    n_act = int(active.sum())
    frac = float((clamped & active).sum() / n_act) if n_act else 0.0
    return out, ClampReport(frac, int((clamped & active).sum()), detail)

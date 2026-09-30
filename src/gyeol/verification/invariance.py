"""Invariance study runner (ported from gyeol v0.1).

Input: a long table of note- or recording-level dimension values keyed by
(singer, condition) where *condition* is a device, room, codec, degradation
level or session.  Output per dimension: ICC(2,1) across conditions with a
bootstrap CI, Koo & Li band, Bland–Altman against the reference condition,
SEM / MDC95, within/between variance ratio and an acceptance verdict
(ICC ≥ 0.9 and |bias| < MDC).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from ..core.containers import Representation
from .stats import bland_altman, icc, icc_ci, koo_li, mdc95, sem, within_between_ratio


@dataclass
class Record:
    singer: str
    condition: str
    values: dict[str, float]


def recording_values(rep: Representation, names: Iterable[str] | None = None, min_confidence: float = 0.5) -> dict[str, float]:
    """Median of confident frames per 1-D attribute curve (NaN if none)."""
    out = {}
    for name, c in rep.curves.items():
        if names is not None and name not in names:
            continue
        if c.values.ndim != 1:
            continue
        v = c.values[(c.confidence >= min_confidence) & np.isfinite(c.values)]
        out[name] = float(np.median(v)) if v.size else float("nan")
    return out


@dataclass
class DimensionReport:
    dimension: str
    n_singers: int
    icc21: float
    icc_ci: tuple[float, float]
    band: str
    sd: float
    sem: float
    mdc95: float
    within_between: float
    bias_vs_reference: dict[str, float]
    loa_vs_reference: dict[str, tuple[float, float]]
    accepted: bool
    #: conditions entering the ICC; a condition where the dimension was never measured (e.g. an explanation
    #: item withheld as "cannot judge" under heavy noise) is left out instead of discarding every subject
    conditions_used: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def invariance_report(records: list[Record], reference: str, min_icc: float = 0.9, n_boot: int = 500) -> dict[str, DimensionReport]:
    singers = sorted({r.singer for r in records})
    conditions = sorted({r.condition for r in records})
    if reference not in conditions:
        raise ValueError(f"reference condition {reference!r} not in records")
    conditions = [reference] + [c for c in conditions if c != reference]
    dims = sorted({d for r in records for d in r.values})
    table = {(r.singer, r.condition): r.values for r in records}
    out: dict[str, DimensionReport] = {}
    for d in dims:
        y_all = np.array([[table.get((s, c), {}).get(d, np.nan) for c in conditions] for s in singers])
        measured = ~np.all(np.isnan(y_all), axis=0)
        if not measured[0]:
            continue  # never measured in the reference condition
        used = [c for c, m in zip(conditions, measured) if m]
        y = y_all[:, measured]
        complete = ~np.isnan(y).any(axis=1)
        yc = y[complete]
        if len(yc) < 3:
            continue
        val = icc(yc, "2,1")
        ci = icc_ci(yc, "2,1", n_boot=n_boot)
        sd = float(np.std(yc, ddof=1))
        s = sem(sd, val)
        m = mdc95(s)
        flat_v = yc.ravel()
        flat_s = np.repeat(np.arange(len(yc)), yc.shape[1])
        wb = within_between_ratio(flat_v, flat_s)
        bias, loa = {}, {}
        for j, c in enumerate(used[1:], start=1):
            ba = bland_altman(yc[:, 0], yc[:, j])
            bias[c] = ba.bias
            loa[c] = (ba.loa_low, ba.loa_high)
        accepted = bool(val >= min_icc and all(abs(b) < m for b in bias.values()))
        out[d] = DimensionReport(d, int(len(yc)), val, ci, koo_li(val), sd, s, m, wb, bias, loa, accepted, used)
    return out

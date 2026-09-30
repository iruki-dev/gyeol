"""Invariance study runner (research §5.ii).

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

from ..representation import VocalRepresentation
from .stats import bland_altman, icc, icc_ci, koo_li, mdc95, sem, within_between_ratio


@dataclass
class Record:
    singer: str
    condition: str
    values: dict[str, float]


def recording_values(rep: VocalRepresentation, dims: Iterable[str] | None = None) -> dict[str, float]:
    """Median of valid frames per 1-D dimension (NaN if none valid)."""
    out = {}
    for name, tr in rep.tracks.items():
        if dims is not None and name not in dims:
            continue
        if tr.values.ndim != 1:
            continue
        v = tr.valid_values()
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
        y = np.array([[table.get((s, c), {}).get(d, np.nan) for c in conditions] for s in singers])
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
        for j, c in enumerate(conditions[1:], start=1):
            ba = bland_altman(yc[:, 0], yc[:, j])
            bias[c] = ba.bias
            loa[c] = (ba.loa_low, ba.loa_high)
        accepted = bool(val >= min_icc and all(abs(b) < m for b in bias.values()))
        out[d] = DimensionReport(d, int(len(yc)), val, ci, koo_li(val), sd, s, m, wb, bias, loa, accepted)
    return out

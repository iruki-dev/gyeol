"""Residual-energy monitoring (brief §6: "always monitor residual energy").

The residual r(t) should carry only what the interpretable layer cannot
(reconstruction detail).  When a new kind of singing — a technique, genre or
recording condition the attributes do not cover — arrives, more of the
signal has to go through r, and its energy rises.  Rising residual energy
therefore means **coverage is missing**, and it points at where to run
discovery next.

* :func:`residual_energy` — mean squared norm of r per frame (voiced frames
  by default), one number per recording;
* :class:`ResidualMonitor` — a baseline (reference recordings), then
  per-slice comparison (e.g. per dataset, technique label, genre, app
  version) with a robust z-score against the baseline's recording-to-
  recording spread, and a time trend (Theil–Sen slope with its 95 % CI).

Flags are *where to look*, not verdicts; the z cut-off is configurable and
defaults to 3 robust SDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..core.containers import Representation


def residual_energy(r: np.ndarray, mask: np.ndarray | None = None) -> float:
    r = np.asarray(r, float)
    e = np.sum(r**2, axis=1)
    if mask is not None:
        m = np.asarray(mask, bool)[: len(e)]
        e = e[: len(m)][m]
    e = e[np.isfinite(e)]
    return float(np.mean(e)) if e.size else float("nan")


def representation_residual_energy(rep: Representation) -> float:
    if rep.residual is None:
        raise ValueError("representation has no residual (encode it with the M4 autoencoder first)")
    return residual_energy(rep.residual, np.nan_to_num(rep.curves["voicing"].values) > 0.5)


@dataclass
class SliceReport:
    slice: str
    n: int
    median: float
    ratio_to_baseline: float
    z: float
    flagged: bool


@dataclass
class TrendReport:
    slope_per_step: float
    ci_low: float
    ci_high: float
    rising: bool  # CI entirely above zero


@dataclass
class ResidualReport:
    baseline_median: float
    baseline_spread: float
    slices: list[SliceReport]
    trend: TrendReport | None
    flags: list[str]


@dataclass
class ResidualMonitor:
    z_threshold: float = 3.0
    baseline: list[float] = field(default_factory=list)
    records: list[tuple[str, float, float]] = field(default_factory=list)  # (slice, time, energy)

    def set_baseline(self, energies: list[float]) -> None:
        e = [float(v) for v in energies if np.isfinite(v)]
        if len(e) < 3:
            raise ValueError("need at least 3 baseline recordings")
        self.baseline = e

    def add(self, slice_label: str, energy: float, time: float | None = None) -> None:
        if not np.isfinite(energy):
            return
        t = float(len(self.records)) if time is None else float(time)
        self.records.append((str(slice_label), t, float(energy)))

    def report(self) -> ResidualReport:
        if not self.baseline:
            raise RuntimeError("set_baseline() first")
        b = np.log(np.asarray(self.baseline))  # energies are positive and skewed: compare in log
        med = float(np.median(b))
        spread = max(1.4826 * float(np.median(np.abs(b - med))), 1e-6)
        slices, flags = [], []
        for name in sorted({s for s, _, _ in self.records}):
            e = np.log(np.array([v for s, _, v in self.records if s == name]))
            m = float(np.median(e))
            z = (m - med) / (spread / np.sqrt(len(e)))
            flagged = bool(z > self.z_threshold)
            slices.append(SliceReport(name, len(e), float(np.exp(m)), float(np.exp(m - med)), float(z), flagged))
            if flagged:
                flags.append(f"coverage_gap:{name}")
        trend = None
        if len(self.records) >= 5:
            t = np.array([r[1] for r in self.records])
            e = np.log(np.array([r[2] for r in self.records]))
            res = stats.theilslopes(e, t, 0.95)
            trend = TrendReport(float(res.slope), float(res.low_slope), float(res.high_slope), bool(res.low_slope > 0))
            if trend.rising:
                flags.append("residual_rising")
        return ResidualReport(float(np.exp(med)), spread, slices, trend, flags)

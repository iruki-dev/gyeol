"""Operating thresholds for showing an explanation item — fitted, never hard-coded.

An item is shown only if its magnitude exceeds its own **measurement
uncertainty** at its confidence::

    |magnitude| ≥ U(confidence) = max(MDC95, E95(confidence))

* **MDC95** — minimal detectable change from repeated measurements of the
  same performance (``1.96 · SD(m₁ − m₂)``, Bland–Altman; = 1.96·√2·SEM);
* **E95(c)** — the 95 % upper bound of ``|measured − true|`` as a function of
  item confidence, from labelled validation items (knob recovery), binned and
  made monotone (non-increasing in confidence) with the pool-adjacent-violators
  surrogate of :func:`gyeol.verification.thresholds.operating_threshold`;
* below the lowest confidence seen in validation there is no evidence, so
  the item is not shown (``min_confidence``).

Both come from validation data (:func:`fit_attribute_threshold`).  An
attribute without a fitted threshold is **not shown** — the policy has no
numeric fallback.  Thresholds are stored as JSON with their provenance, and
every :class:`~gyeol.coach.session.Feedback` carries it, so thresholds fitted
on synthetic data cannot pass for validated ones silently.
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..core.containers import ExplanationItem
from ..verification.stats import bland_altman
from ..verification.thresholds import operating_threshold


@dataclass(frozen=True)
class AttributeThreshold:
    attribute: str  # exact attribute or an fnmatch pattern ("quality_*")
    mdc: float  # MDC95 in the item's unit
    conf_levels: tuple[float, ...]  # confidence bin centres (ascending)
    error_bound: tuple[float, ...]  # E95 per bin, non-increasing
    min_confidence: float | None  # lowest confidence with validation evidence; None = never shown
    unit: str = ""
    n_pairs: int = 0
    n_error: int = 0

    @property
    def usable(self) -> bool:
        return self.min_confidence is not None and len(self.conf_levels) > 0 and np.isfinite(self.mdc)

    def uncertainty(self, confidence: float) -> float:
        """U(c) = max(MDC95, E95(c)); +inf below ``min_confidence``."""
        if not self.usable or confidence < self.min_confidence:
            return float("inf")
        e = float(np.interp(confidence, self.conf_levels, self.error_bound))
        return max(self.mdc, e)

    @classmethod
    def flat(cls, attribute: str, threshold: float, min_confidence: float, unit: str = "") -> "AttributeThreshold":
        """A constant threshold (for tests and hand-entered values from an external validation)."""
        return cls(attribute, float(threshold), (float(min_confidence),), (float(threshold),), float(min_confidence), unit)


def fit_attribute_threshold(attribute: str, retest_a: np.ndarray, retest_b: np.ndarray, confidence: np.ndarray,
                            error: np.ndarray, unit: str = "", n_bins: int = 10, min_per_bin: int = 5) -> AttributeThreshold:
    """Fit one attribute's threshold from validation data.

    ``retest_a/b``: the item magnitude measured twice on the same performance
    (e.g. two devices or noise conditions).  ``confidence`` / ``error``: item
    confidence and ``measured − true`` magnitude on labelled items.
    """
    ba = bland_altman(np.asarray(retest_a, float), np.asarray(retest_b, float))
    if ba.n < 3:
        raise ValueError(f"{attribute}: need at least 3 retest pairs, got {ba.n}")
    mdc = 1.96 * ba.sd
    conf, err = np.asarray(confidence, float), np.asarray(error, float)
    ok = np.isfinite(conf) & np.isfinite(err)
    r = operating_threshold(conf[ok], err[ok], mdc, higher_is_better=True, n_bins=n_bins, min_per_bin=min_per_bin)
    if len(r.levels) == 0:
        return AttributeThreshold(attribute, float(mdc), (), (), None, unit, ba.n, int(ok.sum()))
    order = np.argsort(r.levels)
    lv, bd = np.asarray(r.levels)[order], np.asarray(r.bound)[order]
    return AttributeThreshold(attribute, float(mdc), tuple(map(float, lv)), tuple(map(float, bd)), float(np.min(conf[ok])),
                              unit, ba.n, int(ok.sum()))


@dataclass
class ThresholdSet:
    thresholds: dict[str, AttributeThreshold]
    provenance: dict = field(default_factory=dict)  # {"data": ..., "fitted_utc": ..., "synthetic": bool}

    def lookup(self, attribute: str) -> AttributeThreshold | None:
        if attribute in self.thresholds:
            return self.thresholds[attribute]
        for pat, t in self.thresholds.items():
            if any(c in pat for c in "*?[") and fnmatch.fnmatchcase(attribute, pat):
                return t
        return None

    def passes(self, item: ExplanationItem) -> tuple[bool, str]:
        """(shown?, reason) for one item."""
        t = self.lookup(item.attribute)
        if t is None:
            return False, "no_threshold"
        if not t.usable:
            return False, "attribute_not_reliable"
        if item.confidence < t.min_confidence:
            return False, "below_confidence"
        # status items (missing / extra ornaments, register class change) are categorical: magnitude is secondary
        if item.detail.get("status") in ("missing", "extra"):
            return True, "ok"
        if item.detail.get("status") == "same":
            return False, "no_difference"
        if abs(item.magnitude) < t.uncertainty(item.confidence):
            return False, "below_magnitude"
        return True, "ok"

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        data = {"provenance": self.provenance, "thresholds": {k: asdict(v) for k, v in self.thresholds.items()}}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "ThresholdSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "provenance" not in data or "thresholds" not in data:
            raise ValueError("threshold file needs 'provenance' and 'thresholds'")
        out = {}
        for k, v in data["thresholds"].items():
            v = dict(v, conf_levels=tuple(v["conf_levels"]), error_bound=tuple(v["error_bound"]))
            out[k] = AttributeThreshold(**v)
        return cls(out, data["provenance"])

    @classmethod
    def fitted(cls, thresholds: list[AttributeThreshold], data_description: str, synthetic: bool) -> "ThresholdSet":
        prov = {"data": data_description, "synthetic": bool(synthetic),
                "fitted_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
        return cls({t.attribute: t for t in thresholds}, prov)

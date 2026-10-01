"""Item priority: reliability × audibility, with a category order for ties.

``score = confidence × audibility`` (both from the explanation; confidence
is the calibrated item confidence, audibility the M5 render-based score).

When audibility is unavailable (it was not scored, or the
renderer cannot correct the item) the fallback is
``confidence × |magnitude| / U(confidence)`` — the size of the difference in
units of its own measurement uncertainty (:mod:`gyeol.coach.thresholds`), so
it stays comparable across units.
Scores of the two kinds are never mixed in one ranking: if any shown item
lacks audibility, the whole ranking uses the fallback.

Ties: items whose scores are within ``tie_tolerance`` (relative) of the best
remaining score are ordered by category tier (brief §7):

1. pitch offsets and interval compression
2. rhythm
3. pitch ornaments (scoop, fall, kkeokki, glide, vibrato)
4. dynamics (not in the brief's list; placed before phonation because it is
   not phrased tentatively)
5. register / phonation (phrased tentatively)
6. diction (phrase level)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.containers import ExplanationItem
from .thresholds import ThresholdSet

DEFAULT_TIERS: dict[str, int] = {"pitch": 0, "rhythm": 1, "ornament": 2, "dynamics": 3, "phonation": 4, "diction": 5}


@dataclass
class PriorityConfig:
    tiers: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TIERS))
    tie_tolerance: float = 0.1  # relative score difference treated as a tie


@dataclass
class RankedItem:
    item: ExplanationItem
    score: float
    basis: str  # "audibility" | "magnitude_over_uncertainty"
    tier: int


def score_items(items: list[ExplanationItem], thresholds: ThresholdSet) -> tuple[list[tuple[ExplanationItem, float]], str]:
    use_aud = all(it.audibility is not None for it in items)
    out = []
    for it in items:
        if use_aud:
            s = it.confidence * float(it.audibility)
        else:
            t = thresholds.lookup(it.attribute)
            u = t.uncertainty(it.confidence) if t is not None else float("inf")
            size = abs(it.magnitude) / u if 0 < u < float("inf") else 1.0
            if it.detail.get("status") in ("missing", "extra"):
                size = max(size, 1.0)
            s = it.confidence * size
        out.append((it, float(s)))
    return out, "audibility" if use_aud else "magnitude_over_uncertainty"


def rank(items: list[ExplanationItem], thresholds: ThresholdSet, config: PriorityConfig | None = None) -> list[RankedItem]:
    """Rank already-filtered items (see :meth:`ThresholdSet.passes`)."""
    cfg = config or PriorityConfig()
    scored, basis = score_items(items, thresholds)
    tier = lambda it: cfg.tiers.get(it.category, max(cfg.tiers.values(), default=0) + 1)  # noqa: E731
    remaining = list(range(len(scored)))  # indices: items hold arrays, so no equality tests on them
    out: list[RankedItem] = []
    while remaining:
        best = max(scored[i][1] for i in remaining)
        tied = [i for i in remaining if scored[i][1] >= best / (1.0 + cfg.tie_tolerance) or best <= 0]
        pick = min(tied, key=lambda i: (tier(scored[i][0]), -scored[i][1], scored[i][0].key))
        remaining.remove(pick)
        it, s = scored[pick]
        out.append(RankedItem(it, s, basis, tier(it)))
    return out

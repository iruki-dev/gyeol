"""Conditional directions from paired on/off data (brief §6, optional).

A *direction* is the average feature change when a technique is switched on
(the same phrase sung off and on).  It is fitted **per condition cell**
(f0 band × loudness band × vowel) because the same technique moves features
differently at different pitches, loudness and vowels.

Procedure:

1. **align** the pairs: each "on" frame is matched to the "off" frame its
   content-only warp maps to (:class:`gyeol.data.paired.PairedExample`);
2. **regress out f0 and loudness**: a linear model
   ``x ~ 1 + f0 + f0² + loudness`` is fitted on all off and on frames
   together and its prediction is removed, so that singing the "on" take a
   little higher or louder does not leak into the direction;
3. per cell: ``d = mean(r_on − r_off)``, its size, the sign consistency of
   the per-frame differences, and a **per-group** t statistic (groups =
   singers), so one singer cannot make a direction on their own;
4. a pooled direction serves cells with too few pairs.

:meth:`ConditionalDirections.score` projects new frames (residualised the
same way) onto their cell's unit direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class PairedFrames:
    """Aligned frame pairs: row i of ``off`` and ``on`` are the same moment of the phrase."""

    off: np.ndarray  # (N, D)
    on: np.ndarray  # (N, D)
    f0_off: np.ndarray  # (N,) cents
    f0_on: np.ndarray
    loud_off: np.ndarray  # (N,) dB
    loud_on: np.ndarray
    group: np.ndarray  # (N,) singer id
    vowel: np.ndarray | None = None  # (N,) labels; None = one vowel cell

    def __post_init__(self) -> None:
        n = len(self.off)
        for name in ("on", "f0_off", "f0_on", "loud_off", "loud_on", "group"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, expected {n}")
        if self.vowel is not None and len(self.vowel) != n:
            raise ValueError("vowel labels do not match the frame count")


def pair_frames(off_feats: np.ndarray, on_feats: np.ndarray, tau_on_to_off: np.ndarray, f0_off: np.ndarray, f0_on: np.ndarray,
                loud_off: np.ndarray, loud_on: np.ndarray, group, vowel_off: np.ndarray | None = None,
                valid_off: np.ndarray | None = None, valid_on: np.ndarray | None = None) -> PairedFrames:
    """Align one on/off pair with the warp (on frame t ↔ off frame round(τ(t)))."""
    j = np.clip(np.round(np.asarray(tau_on_to_off)).astype(int), 0, len(off_feats) - 1)
    t = np.arange(len(on_feats))
    ok = np.isfinite(f0_on) & np.isfinite(f0_off[j]) & np.isfinite(loud_on) & np.isfinite(loud_off[j])
    ok &= np.all(np.isfinite(on_feats), 1) & np.all(np.isfinite(off_feats[j]), 1)
    if valid_on is not None:
        ok &= valid_on
    if valid_off is not None:
        ok &= valid_off[j]
    t, j = t[ok], j[ok]
    return PairedFrames(off_feats[j], on_feats[t], f0_off[j], f0_on[t], loud_off[j], loud_on[t], np.full(len(t), group, dtype=object),
                        None if vowel_off is None else np.asarray(vowel_off)[j])


def pair_from_example(ex, features, group=None, min_confidence: float = 0.5) -> PairedFrames:
    """Aligned frames from a :class:`gyeol.data.paired.PairedExample`.

    ``features(rep) -> (T, D)`` gives the frames to contrast (phonation
    features, or the residual ``rep.residual``).  f0 is ``f0_cents``;
    loudness is the absolute A-weighted ``loudness`` curve.
    """
    a, b = ex.off, ex.on
    fa, fb = a.curves["f0_cents"], b.curves["f0_cents"]
    return pair_frames(np.asarray(features(a), float), np.asarray(features(b), float), ex.warp.tau,
                       np.where(fa.confidence >= min_confidence, fa.values, np.nan),
                       np.where(fb.confidence >= min_confidence, fb.values, np.nan),
                       a.curves["loudness"].values, b.curves["loudness"].values,
                       ex.labels_off.get("singer", ex.pair_id) if group is None else group)


def concat(parts: list[PairedFrames]) -> PairedFrames:
    vow = None if any(p.vowel is None for p in parts) else np.concatenate([p.vowel for p in parts])
    cat = lambda n: np.concatenate([getattr(p, n) for p in parts])  # noqa: E731
    return PairedFrames(cat("off"), cat("on"), cat("f0_off"), cat("f0_on"), cat("loud_off"), cat("loud_on"), cat("group"), vow)


@dataclass
class DirectionConfig:
    f0_edges_cents: tuple[float, ...] = ()  # band edges (cents re A4); () = one band
    loud_edges_db: tuple[float, ...] = ()
    min_pairs: int = 30
    regress_out: bool = True


@dataclass
class Direction:
    unit: np.ndarray  # (D,) unit vector
    size: float  # ||mean difference||
    sign_consistency: float  # fraction of frame differences with a positive projection
    group_t: float  # one-sample t over per-group mean projections (NaN with < 2 groups)
    n_pairs: int
    n_groups: int


Cell = tuple[int, int, object]


@dataclass
class ConditionalDirections:
    config: DirectionConfig
    coef: np.ndarray | None  # (4, D) covariate regression, None if not regressed out
    cells: dict[Cell, Direction]
    pooled: Direction
    meta: dict = field(default_factory=dict)

    @staticmethod
    def _design(f0: np.ndarray, loud: np.ndarray) -> np.ndarray:
        f = np.asarray(f0, float) / 1200.0
        return np.column_stack([np.ones_like(f), f, f**2, np.asarray(loud, float) / 10.0])

    def residualise(self, X: np.ndarray, f0: np.ndarray, loud: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        return X if self.coef is None else X - self._design(f0, loud) @ self.coef

    def cell_of(self, f0: np.ndarray, loud: np.ndarray, vowel: np.ndarray | None) -> list[Cell]:
        fb = np.searchsorted(self.config.f0_edges_cents, f0) if self.config.f0_edges_cents else np.zeros(len(f0), int)
        lb = np.searchsorted(self.config.loud_edges_db, loud) if self.config.loud_edges_db else np.zeros(len(loud), int)
        vw = vowel if vowel is not None else [None] * len(f0)
        return [(int(a), int(b), v) for a, b, v in zip(fb, lb, vw)]

    def direction_for(self, cell: Cell) -> Direction:
        return self.cells.get(cell, self.pooled)

    def score(self, X: np.ndarray, f0: np.ndarray, loud: np.ndarray, vowel: np.ndarray | None = None) -> np.ndarray:
        R = self.residualise(X, f0, loud)
        cells = self.cell_of(np.asarray(f0, float), np.asarray(loud, float), vowel)
        return np.array([R[i] @ self.direction_for(c).unit for i, c in enumerate(cells)])

    def similarity(self) -> dict[tuple[Cell, Cell], float]:
        """Cosine between the unit directions of every pair of fitted cells."""
        keys = sorted(self.cells, key=str)
        return {(a, b): float(self.cells[a].unit @ self.cells[b].unit) for i, a in enumerate(keys) for b in keys[i + 1 :]}


def _direction(diff: np.ndarray, group: np.ndarray) -> Direction:
    mean = diff.mean(0)
    size = float(np.linalg.norm(mean))
    unit = mean / size if size > 0 else np.zeros_like(mean)
    proj = diff @ unit
    per_group = [float(proj[group == g].mean()) for g in np.unique(group)]
    t = float(stats.ttest_1samp(per_group, 0.0).statistic) if len(per_group) >= 2 else float("nan")
    return Direction(unit, size, float(np.mean(proj > 0)), t, len(diff), len(per_group))


def fit_conditional_directions(pf: PairedFrames, config: DirectionConfig | None = None) -> ConditionalDirections:
    cfg = config or DirectionConfig()
    if len(pf.off) < cfg.min_pairs:
        raise ValueError(f"need at least {cfg.min_pairs} aligned pairs, got {len(pf.off)}")
    coef = None
    if cfg.regress_out:
        Z = np.vstack([ConditionalDirections._design(pf.f0_off, pf.loud_off), ConditionalDirections._design(pf.f0_on, pf.loud_on)])
        coef = np.linalg.lstsq(Z, np.vstack([pf.off, pf.on]), rcond=None)[0]
    cd = ConditionalDirections(cfg, coef, {}, None)  # type: ignore[arg-type]
    diff = cd.residualise(pf.on, pf.f0_on, pf.loud_on) - cd.residualise(pf.off, pf.f0_off, pf.loud_off)
    group = np.asarray(pf.group)
    cd.pooled = _direction(diff, group)
    # cells by the off frame's condition (both takes are the same moment of the same phrase)
    cells = cd.cell_of(0.5 * (pf.f0_off + pf.f0_on), 0.5 * (pf.loud_off + pf.loud_on), pf.vowel)
    keys = np.array([str(c) for c in cells])
    for c in set(cells):
        m = keys == str(c)
        if m.sum() >= cfg.min_pairs:
            cd.cells[c] = _direction(diff[m], group[m])
    cd.meta = {"n_pairs": len(diff), "n_groups": len(np.unique(group)), "regressed_out": cfg.regress_out}
    return cd

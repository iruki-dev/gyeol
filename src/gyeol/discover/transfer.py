"""Transfer tests and the promotion rule (brief §6, evaluation §3).

A discovered feature (an SAE latent, a conditional-direction score, or any
candidate frame feature) becomes a new attribute **only if** a linear readout
trained on some data still works on data that differ in a nuisance:

* ``singer`` — held-out singers (group K-fold over singers);
* ``pitch_range`` — held-out pitch bands (train on the other bands, test on
  one; every band must pass, including both extremes);
* ``language`` — held-out languages, required when language labels exist.

Each split is compared with an **in-distribution** baseline (K-fold over
recordings, all singers and pitches seen in training).

Scores are computed at the level of *units* (recordings; for pitch bands,
recording × band), not frames, because frames of one recording are not
independent:

* binary targets — gain = ``2·AUROC − 1`` of the readout over unit means,
  one-sided Mann–Whitney p-value;
* continuous targets — gain = Spearman ρ over unit means, one-sided p-value.

A split passes when ``p < alpha``, ``gain ≥ min_gain`` and
``gain ≥ (1 − max_drop) · in-distribution gain``.  The candidate is promoted
only if every required split passes; :class:`PromotionRegistry` refuses
anything else.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

from ..eval.probes import _fit_logreg


@dataclass
class TransferConfig:
    alpha: float = 0.05
    min_gain: float = 0.2
    max_drop: float = 0.5
    n_pitch_bins: int = 3
    k: int = 5
    l2: float = 1e-2
    seed: int = 0


@dataclass
class TransferResult:
    split: str
    gain: float  # held-out gain (worst fold / band)
    in_distribution_gain: float
    p_value: float
    n_units: int
    passed: bool
    reason: str
    per_fold: list[float] = field(default_factory=list)


@dataclass
class PromotionDecision:
    name: str
    kind: str  # "binary" | "continuous"
    passed: bool
    results: dict[str, TransferResult]
    required: tuple[str, ...]
    reasons: list[str]


class PromotionError(RuntimeError):
    pass


# ---------------------------------------------------------------- readouts


class _Readout:
    def __init__(self, X: np.ndarray, y: np.ndarray, kind: str, l2: float, seed: int):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        self.kind = kind
        if kind == "binary":
            lin = _fit_logreg(Xs, y.astype(int), 2, l2, seed)
            w = lin.weight.detach().numpy().astype(float)
            b = lin.bias.detach().numpy().astype(float)
            self.w, self.b = w[1] - w[0], float(b[1] - b[0])
        else:
            m = y.mean()
            self.w = np.linalg.solve(Xs.T @ Xs + l2 * len(Xs) * np.eye(X.shape[1]), Xs.T @ (y - m))
            self.b = float(m)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.w + self.b


def _unit_gain(score: np.ndarray, y: np.ndarray, units: np.ndarray, kind: str) -> tuple[float, float, int]:
    keys = np.array([str(u) for u in units])
    uniq = np.unique(keys)
    s = np.array([score[keys == u].mean() for u in uniq])
    t = np.array([y[keys == u].mean() for u in uniq])
    if kind == "binary":
        lab = t >= 0.5
        pos, neg = s[lab], s[~lab]
        if len(pos) < 2 or len(neg) < 2:
            return float("nan"), 1.0, len(uniq)
        u = stats.mannwhitneyu(pos, neg, alternative="greater")
        return float(2 * u.statistic / (len(pos) * len(neg)) - 1), float(u.pvalue), len(uniq)
    if len(uniq) < 4 or np.all(t == t[0]):
        return float("nan"), 1.0, len(uniq)
    r = stats.spearmanr(s, t, alternative="greater")
    return float(r.statistic), float(r.pvalue), len(uniq)


def _kfold_units(units: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    keys = np.array([str(u) for u in units])
    uniq = np.unique(keys)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    return [np.isin(keys, p) for p in np.array_split(uniq, min(k, len(uniq)))]


def _pooled_cv(X, y, units, test_masks, kind, cfg) -> tuple[np.ndarray, list[bool]]:
    """Out-of-fold readout scores; folds whose training set lacks a class are skipped."""
    score = np.full(len(y), np.nan)
    used = []
    for te in test_masks:
        tr = ~te
        ok = te.any() and tr.any() and (kind != "binary" or len(np.unique(y[tr])) == 2)
        used.append(bool(ok))
        if ok:
            score[te] = _Readout(X[tr], y[tr], kind, cfg.l2, cfg.seed)(X[te])
    return score, used


def _judge(split: str, gain: float, p: float, n: int, base: float, cfg: TransferConfig, per_fold: list[float]) -> TransferResult:
    if not np.isfinite(gain):
        return TransferResult(split, gain, base, p, n, False, "not enough units to test", per_fold)
    reasons = []
    if p >= cfg.alpha:
        reasons.append(f"not significant (p={p:.3g})")
    if gain < cfg.min_gain:
        reasons.append(f"gain {gain:.2f} < {cfg.min_gain}")
    if np.isfinite(base) and gain < (1 - cfg.max_drop) * base:
        reasons.append(f"drops from {base:.2f} in-distribution to {gain:.2f}")
    return TransferResult(split, gain, base, p, n, not reasons, "; ".join(reasons) or "ok", per_fold)


def in_distribution(X, y, units, kind, cfg) -> tuple[float, float, int]:
    score, _ = _pooled_cv(X, y, units, _kfold_units(units, cfg.k, cfg.seed), kind, cfg)
    ok = np.isfinite(score)
    return _unit_gain(score[ok], y[ok], units[ok], kind)


def held_out(split: str, X, y, units, groups, kind, cfg, base: float) -> TransferResult:
    """Group K-fold over ``groups`` (singers, languages); pooled out-of-fold scores."""
    g = np.array([str(v) for v in groups])
    uniq = np.unique(g)
    if len(uniq) < 2:
        return TransferResult(split, float("nan"), base, 1.0, 0, False, f"only one {split} group", [])
    masks = [g == v for v in uniq] if len(uniq) <= cfg.k else _kfold_units(g, cfg.k, cfg.seed)
    score, _ = _pooled_cv(X, y, units, masks, kind, cfg)
    ok = np.isfinite(score)
    gain, p, n = _unit_gain(score[ok], y[ok], units[ok], kind)
    per = [(_unit_gain(score[m & ok], y[m & ok], units[m & ok], kind)[0]) for m in masks]
    return _judge(split, gain, p, n, base, cfg, [float(v) for v in per])


def held_out_pitch(X, y, units, pitch_cents, kind, cfg, base: float) -> TransferResult:
    """Train on the other pitch bands, test on one; the worst band decides."""
    p = np.asarray(pitch_cents, float)
    ok = np.isfinite(p)
    edges = np.quantile(p[ok], np.linspace(0, 1, cfg.n_pitch_bins + 1)[1:-1])
    band = np.searchsorted(edges, p)
    gains, pvals, ns = [], [], []
    for b in range(cfg.n_pitch_bins):
        te, tr = ok & (band == b), ok & (band != b)
        if kind == "binary" and (len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2):
            gains.append(float("nan"))
            pvals.append(1.0)
            ns.append(0)
            continue
        s = _Readout(X[tr], y[tr], kind, cfg.l2, cfg.seed)(X[te])
        bu = np.array([f"{u}|{b}" for u in units[te]])
        g, pv, n = _unit_gain(s, y[te], bu, kind)
        gains.append(g)
        pvals.append(pv)
        ns.append(n)
    if not np.all(np.isfinite(gains)):
        return TransferResult("pitch_range", float("nan"), base, 1.0, int(sum(ns)), False,
                              "a pitch band lacks both classes or units (cannot test extrapolation)", [float(v) for v in gains])
    worst = int(np.argmin(gains))
    return _judge("pitch_range", float(gains[worst]), float(pvals[worst]), int(ns[worst]), base, cfg, [float(v) for v in gains])


def evaluate_promotion(name: str, X: np.ndarray, y: np.ndarray, units: np.ndarray, singers: np.ndarray, pitch_cents: np.ndarray,
                       language: np.ndarray | None = None, config: TransferConfig | None = None) -> PromotionDecision:
    """Run every required transfer test for one candidate.

    ``X`` (N, F) candidate features per frame (e.g. SAE codes of the matched
    latents, or a direction score); ``y`` binary (0/1) or continuous target;
    ``units`` recording ids; ``singers`` / ``pitch_cents`` / ``language`` per
    frame.
    """
    cfg = config or TransferConfig()
    X = np.asarray(X, float).reshape(len(y), -1)
    y = np.asarray(y, float)
    units, singers = np.asarray(units), np.asarray(singers)
    kind = "binary" if set(np.unique(y)) <= {0.0, 1.0} else "continuous"
    base, _, _ = in_distribution(X, y, units, kind, cfg)
    results = {"singer": held_out("singer", X, y, units, singers, kind, cfg, base),
               "pitch_range": held_out_pitch(X, y, units, pitch_cents, kind, cfg, base)}
    required = ("singer", "pitch_range")
    if language is not None and len(np.unique(np.asarray(language).astype(str))) >= 2:
        results["language"] = held_out("language", X, y, units, np.asarray(language), kind, cfg, base)
        required += ("language",)
    reasons = [f"{k}: {r.reason}" for k, r in results.items() if not r.passed]
    return PromotionDecision(name, kind, all(results[k].passed for k in required), results, required, reasons)


# ---------------------------------------------------------------- registry


@dataclass
class PromotedAttribute:
    name: str
    kind: str
    source: dict  # e.g. {"type": "sae_feature", "input": "residual", "latents": [12]}
    weights: list[float]
    bias: float
    mean: list[float]
    std: list[float]
    evidence: dict
    promoted_utc: str

    def score(self, X: np.ndarray) -> np.ndarray:
        """Readout value (logit for binary attributes) per frame."""
        X = np.asarray(X, float).reshape(len(X), -1)
        return ((X - np.asarray(self.mean)) / np.asarray(self.std)) @ np.asarray(self.weights) + self.bias

    def task_spec(self):
        """Head spec for training the promoted attribute as an M3 head."""
        from ..attributes.heads import TaskSpec

        if self.kind == "binary":
            return TaskSpec(1, "sigmoid", (self.name,))
        return TaskSpec(1, "regression", (self.name,), unit=str(self.source.get("unit", "")))


class PromotionRegistry:
    """JSON-backed list of promoted attributes; only passed decisions get in."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._items: dict[str, PromotedAttribute] = {}
        if self.path.exists():
            for d in json.loads(self.path.read_text(encoding="utf-8"))["promoted"]:
                self._items[d["name"]] = PromotedAttribute(**d)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def get(self, name: str) -> PromotedAttribute:
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)

    def register(self, decision: PromotionDecision, X: np.ndarray, y: np.ndarray, source: dict,
                 config: TransferConfig | None = None) -> PromotedAttribute:
        if not isinstance(decision, PromotionDecision) or not decision.passed:
            why = "; ".join(decision.reasons) if isinstance(decision, PromotionDecision) else "no decision"
            raise PromotionError(f"{getattr(decision, 'name', '?')} did not pass the transfer tests: {why}")
        missing = [s for s in ("singer", "pitch_range") if s not in decision.results or not decision.results[s].passed]
        if missing:
            raise PromotionError(f"required transfer tests missing or failed: {missing}")
        if decision.name in self._items:
            raise PromotionError(f"{decision.name!r} is already promoted")
        cfg = config or TransferConfig()
        X = np.asarray(X, float).reshape(len(y), -1)
        r = _Readout(X, np.asarray(y, float), decision.kind, cfg.l2, cfg.seed)
        attr = PromotedAttribute(
            decision.name, decision.kind, source, [float(v) for v in np.atleast_1d(r.w)], float(r.b),
            [float(v) for v in r.mu], [float(v) for v in r.sd],
            {k: asdict(v) for k, v in decision.results.items()},
            _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
        self._items[attr.name] = attr
        self.path.write_text(json.dumps({"promoted": [asdict(a) for a in self._items.values()]}, ensure_ascii=False, indent=2,
                                        default=float), encoding="utf-8")
        return attr


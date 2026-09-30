"""Linear probes and leakage tests (evaluation §1).

Probes decide what is coachable: an attribute is readable from a layer if a
*linear* probe beats chance on held-out singers.  Leakage tests invert the
question: probes on the residual ``r`` for attributes / singer id / env
type should sit near chance.

* :func:`probe_classify` — multinomial logistic regression, group K-fold
  (groups = singers), accuracy vs chance with a one-sided binomial test;
* :func:`probe_regress` — ridge regression, grouped CV, R²;
* :func:`leakage` — classify probe + verdict "near chance" when accuracy is
  not significantly above chance (and within ``margin`` of it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


@dataclass
class ProbeResult:
    target: str
    score: float  # accuracy (classify) or R² (regress)
    chance: float
    p_value: float | None
    n: int
    folds: list[float]


def _group_folds(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    parts = np.array_split(uniq, min(k, len(uniq)))
    return [np.isin(groups, p) for p in parts]


def _standardise(tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = tr.mean(0), tr.std(0) + 1e-8
    return (tr - mu) / sd, (te - mu) / sd


def _fit_logreg(X: np.ndarray, y: np.ndarray, n_classes: int, l2: float, seed: int, iters: int = 200) -> torch.nn.Linear:
    torch.manual_seed(seed)
    lin = torch.nn.Linear(X.shape[1], n_classes)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    opt = torch.optim.LBFGS(lin.parameters(), lr=0.5, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(Xt), yt) + l2 * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lin


def probe_classify(X: np.ndarray, y: np.ndarray, groups: np.ndarray, target: str = "", k: int = 5, l2: float = 1e-3, seed: int = 0) -> ProbeResult:
    X, y, groups = np.asarray(X, float), np.asarray(y), np.asarray(groups)
    classes, yi = np.unique(y, return_inverse=True)
    correct, total, folds = 0, 0, []
    for te in _group_folds(groups, k, seed):
        tr = ~te
        if len(np.unique(yi[tr])) < 2 or te.sum() == 0:
            continue
        Xtr, Xte = _standardise(X[tr], X[te])
        lin = _fit_logreg(Xtr, yi[tr], len(classes), l2, seed)
        with torch.no_grad():
            pred = lin(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()
        c = int((pred == yi[te]).sum())
        correct += c
        total += int(te.sum())
        folds.append(c / te.sum())
    chance = float(np.max(np.bincount(yi)) / len(yi))  # majority-class rate
    p = float(stats.binomtest(correct, total, chance, alternative="greater").pvalue) if total else None
    return ProbeResult(target, correct / max(total, 1), chance, p, total, folds)


def probe_regress(X: np.ndarray, y: np.ndarray, groups: np.ndarray, target: str = "", k: int = 5, l2: float = 1.0, seed: int = 0) -> ProbeResult:
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    preds = np.full(len(y), np.nan)
    folds = []
    for te in _group_folds(groups, k, seed):
        tr = ~te
        Xtr, Xte = _standardise(X[tr], X[te])
        mu = y[tr].mean()
        A = Xtr.T @ Xtr + l2 * np.eye(X.shape[1])
        w = np.linalg.solve(A, Xtr.T @ (y[tr] - mu))
        preds[te] = Xte @ w + mu
        ss = ((y[te] - preds[te]) ** 2).sum()
        folds.append(1 - ss / (((y[te] - y[te].mean()) ** 2).sum() + 1e-12))
    ok = np.isfinite(preds)
    r2 = 1 - ((y[ok] - preds[ok]) ** 2).sum() / (((y[ok] - y[ok].mean()) ** 2).sum() + 1e-12)
    return ProbeResult(target, float(r2), 0.0, None, int(ok.sum()), [float(f) for f in folds])


@dataclass
class LeakageResult:
    probe: ProbeResult
    near_chance: bool


def leakage(X: np.ndarray, y: np.ndarray, groups: np.ndarray, target: str = "", alpha: float = 0.05, margin: float = 0.05, **kw) -> LeakageResult:
    """Pass (near_chance=True) when the probe is not significantly above chance
    or within ``margin`` of it."""
    r = probe_classify(X, y, groups, target, **kw)
    near = (r.p_value is not None and r.p_value > alpha) or (r.score - r.chance) <= margin
    return LeakageResult(r, bool(near))


def probe_battery(layers: dict[str, np.ndarray], targets: dict[str, tuple[np.ndarray, str]], groups: np.ndarray, **kw) -> dict[tuple[str, str], ProbeResult]:
    """Every (layer, target) probe.  ``targets[name] = (values, "classify"|"regress")``."""
    out = {}
    for ln, X in layers.items():
        for tn, (y, kind) in targets.items():
            fn = probe_classify if kind == "classify" else probe_regress
            out[(ln, tn)] = fn(X, y, groups, tn, **kw)
    return out

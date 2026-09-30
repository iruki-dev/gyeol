"""Reliability and agreement statistics for the invariance program (§5.ii).

* ICC(1,1), ICC(2,1) absolute agreement, ICC(3,1) consistency
  (Shrout & Fleiss 1979; McGraw & Wong 1996) with bootstrap CIs
* Koo & Li (2016) interpretation bands
* Bland–Altman bias and 95 % limits of agreement
* SEM = SD·√(1 − ICC), MDC95 = 1.96·√2·SEM
* within-singer / between-singer variance ratio
* chance-level test for nuisance leakage (binomial)
* equal error rate for singer verification
* paired TOST equivalence test, Holm correction
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _anova(y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    n, k = y.shape
    gm = y.mean()
    ss_r = k * np.sum((y.mean(axis=1) - gm) ** 2)
    ss_c = n * np.sum((y.mean(axis=0) - gm) ** 2)
    ss_t = np.sum((y - gm) ** 2)
    ss_e = ss_t - ss_r - ss_c
    return {
        "n": n,
        "k": k,
        "msr": ss_r / (n - 1),
        "msc": ss_c / (k - 1),
        "mse": ss_e / ((n - 1) * (k - 1)),
        "msw": (ss_c + ss_e) / (n * (k - 1)),
    }


def icc(y: np.ndarray, kind: str = "2,1") -> float:
    """Intraclass correlation for an (n subjects × k raters/devices) matrix."""
    y = np.asarray(y, dtype=float)
    if np.isnan(y).any():
        y = y[~np.isnan(y).any(axis=1)]
    if y.shape[0] < 2 or y.shape[1] < 2:
        return float("nan")
    a = _anova(y)
    n, k = a["n"], a["k"]
    if kind == "1,1":
        return float((a["msr"] - a["msw"]) / (a["msr"] + (k - 1) * a["msw"]))
    if kind == "2,1":
        return float((a["msr"] - a["mse"]) / (a["msr"] + (k - 1) * a["mse"] + k * (a["msc"] - a["mse"]) / n))
    if kind == "3,1":
        return float((a["msr"] - a["mse"]) / (a["msr"] + (k - 1) * a["mse"]))
    raise ValueError(f"unknown ICC kind {kind!r}")


def icc_ci(y: np.ndarray, kind: str = "2,1", n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI over subjects."""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y).any(axis=1)]
    rng = np.random.default_rng(seed)
    vals = [icc(y[rng.integers(0, len(y), len(y))], kind) for _ in range(n_boot)]
    vals = np.asarray(vals)[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def koo_li(value: float) -> str:
    """Koo & Li (J Chiropr Med 2016) interpretation."""
    if not np.isfinite(value):
        return "undefined"
    if value < 0.5:
        return "poor"
    if value < 0.75:
        return "moderate"
    if value < 0.9:
        return "good"
    return "excellent"


@dataclass
class BlandAltman:
    bias: float
    sd: float
    loa_low: float
    loa_high: float
    n: int


def bland_altman(a: np.ndarray, b: np.ndarray) -> BlandAltman:
    """Bias (b − a) and 95 % limits of agreement (Bland & Altman, Lancet 1986)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = b[ok] - a[ok]
    bias, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    return BlandAltman(bias, sd, bias - 1.96 * sd, bias + 1.96 * sd, int(ok.sum()))


def sem(sd: float, icc_value: float) -> float:
    return float(sd * np.sqrt(max(0.0, 1.0 - icc_value)))


def mdc95(sem_value: float) -> float:
    return float(1.96 * np.sqrt(2.0) * sem_value)


def within_between_ratio(values: np.ndarray, subjects: np.ndarray) -> float:
    """Within-subject variance / between-subject variance (target < 0.1)."""
    values = np.asarray(values, float)
    subjects = np.asarray(subjects)
    ok = np.isfinite(values)
    values, subjects = values[ok], subjects[ok]
    groups = [values[subjects == s] for s in np.unique(subjects)]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan")
    within = np.mean([np.var(g, ddof=1) for g in groups])
    between = np.var([np.mean(g) for g in groups], ddof=1)
    return float(within / between) if between > 0 else float("inf")


def chance_test(n_correct: int, n_total: int, n_classes: int) -> float:
    """One-sided binomial p-value that accuracy exceeds chance (1/n_classes).

    Nuisance-leakage target: *not* significantly above chance.
    """
    return float(stats.binomtest(n_correct, n_total, 1.0 / n_classes, alternative="greater").pvalue)


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER for verification scores (higher = same speaker; labels 1 = target)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(~labels)
    fnr = 1 - tp / max(labels.sum(), 1)
    fpr = fp / max((~labels).sum(), 1)
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2)


def tost_paired(a: np.ndarray, b: np.ndarray, margin: float) -> float:
    """Two one-sided tests for equivalence of paired means within ±margin.

    Returns the TOST p-value (max of the two one-sided p-values).
    """
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d[np.isfinite(d)]
    n = len(d)
    se = np.std(d, ddof=1) / np.sqrt(n)
    t_lo = (np.mean(d) + margin) / se
    t_hi = (np.mean(d) - margin) / se
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    return float(max(p_lo, p_hi))


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down: which hypotheses are rejected at family-wise ``alpha``."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, bool] = {}
    still = True
    for i, (k, p) in enumerate(items):
        still = still and p <= alpha / (m - i)
        out[k] = still
    return out

"""Match SAE features to known labels and to v0.1 DSP features.

For every (feature, target) pair a *separation* score in [0, 1]:

* classification targets — one-vs-rest AUROC of the feature activation per
  class; score = max over classes of ``|2·AUROC − 1|`` (the class and sign
  are kept);
* continuous targets (labels or DSP features such as H1*–H2*, CPPS) —
  ``|Spearman ρ|``.

Frames where a target is unknown (NaN) are skipped for that target.
Features that match nothing above ``max_known_score`` are the **novel
candidates** — the discovery output, ranked by the reconstruction energy
they carry.  Scores are descriptive; a candidate becomes an attribute only
through the transfer tests in :mod:`gyeol.discover.transfer`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUROC (Mann–Whitney), ties averaged; NaN if one class is empty."""
    s, p = np.asarray(scores, float), np.asarray(positive, bool)
    n1, n0 = int(p.sum()), int((~p).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@dataclass(frozen=True)
class FeatureMatch:
    feature: int
    target: str
    score: float  # separation in [0, 1]
    kind: str  # "classify" | "regress"
    detail: str = ""  # class (and sign) for classify; sign for regress


@dataclass
class MatchTable:
    matches: list[FeatureMatch]
    n_features: int

    def best_for_target(self, target: str, n: int = 3) -> list[FeatureMatch]:
        return sorted((m for m in self.matches if m.target == target), key=lambda m: -m.score)[:n]

    def best_for_feature(self, feature: int) -> FeatureMatch | None:
        ms = [m for m in self.matches if m.feature == feature]
        return max(ms, key=lambda m: m.score) if ms else None

    def novel(self, max_known_score: float, energy_share: np.ndarray | None = None, active: np.ndarray | None = None) -> list[int]:
        """Features whose best match to any known target stays ≤ ``max_known_score``."""
        out = []
        for f in range(self.n_features):
            if active is not None and not active[f]:
                continue
            b = self.best_for_feature(f)
            if b is None or b.score <= max_known_score:
                out.append(f)
        if energy_share is not None:
            out.sort(key=lambda f: -energy_share[f])
        return out


def match_features(codes: np.ndarray, targets: dict[str, tuple[np.ndarray, str]], min_frames: int = 20) -> MatchTable:
    """``targets[name] = (values per frame, "classify" | "regress")``."""
    Z = np.asarray(codes, float)
    out: list[FeatureMatch] = []
    for name, (vals, kind) in targets.items():
        v = np.asarray(vals)
        if kind == "regress" or np.issubdtype(v.dtype, np.floating):
            known = np.isfinite(v.astype(float))
        else:  # categorical labels: None marks unknown
            known = np.array([x is not None for x in v], bool)
        if known.sum() < min_frames:
            continue
        Zk, vk = Z[known], v[known]
        for f in range(Z.shape[1]):
            z = Zk[:, f]
            if np.all(z == z[0]):
                out.append(FeatureMatch(f, name, 0.0, kind))
                continue
            if kind == "classify":
                best, det = 0.0, ""
                for c in np.unique(vk):
                    a = auroc(z, vk == c)
                    if np.isfinite(a) and abs(2 * a - 1) > best:
                        best, det = abs(2 * a - 1), f"{c}{'+' if a >= 0.5 else '-'}"
                out.append(FeatureMatch(f, name, float(best), kind, det))
            else:
                rho = stats.spearmanr(z, vk.astype(float)).statistic
                rho = 0.0 if not np.isfinite(rho) else float(rho)
                out.append(FeatureMatch(f, name, abs(rho), kind, "+" if rho >= 0 else "-"))
    return MatchTable(out, Z.shape[1])

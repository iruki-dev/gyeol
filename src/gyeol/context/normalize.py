"""Singer-specific context normalisation (research §6).

Stores, for a dimension d, the residual ``d − E[d | laryngeal class,
time-since-onset bin]`` where the conditional expectation is estimated per
singer from their own modal singing.  Raw values are kept as well, so
stylistic pressing (trot, 통성) stays visible in the raw channel.

*UNVERIFIED HYPOTHESIS: that this separates phonetic from stylistic
pressing; test with coach probes on "intentional vs phonetic pressing".*
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..representation import VocalRepresentation
from .korean import LARYNGEAL_CLASSES


@dataclass
class ContextNormalizer:
    dimensions: tuple[str, ...] = ("h1h2c", "h1a1c", "cpps", "f0_cents")
    time_bins_ms: tuple[float, ...] = (0.0, 50.0, 100.0, 200.0, np.inf)
    min_count: int = 20
    table: dict[str, dict[tuple[int, int], float]] = field(default_factory=dict)
    global_mean: dict[str, float] = field(default_factory=dict)

    def _keys(self, rep: VocalRepresentation) -> tuple[np.ndarray, np.ndarray]:
        c = rep.context
        if c is None:
            raise ValueError("representation has no context tokens")
        tb = np.digitize(np.nan_to_num(c.time_since_onset_ms, nan=-1.0), self.time_bins_ms) - 1
        return c.laryngeal_class.astype(int), tb

    def fit(self, reps: list[VocalRepresentation]) -> "ContextNormalizer":
        for d in self.dimensions:
            buckets: dict[tuple[int, int], list[np.ndarray]] = {}
            all_vals = []
            for rep in reps:
                if d not in rep.tracks or rep.context is None:
                    continue
                tr = rep.tracks[d]
                lar, tb = self._keys(rep)
                n = min(len(tr.values), len(lar))
                ok = tr.valid[:n] & np.isfinite(tr.values[:n]) & (tb[:n] >= 0)
                all_vals.append(tr.values[:n][ok])
                for key in set(zip(lar[:n][ok].tolist(), tb[:n][ok].tolist())):
                    sel = ok & (lar[:n] == key[0]) & (tb[:n] == key[1])
                    buckets.setdefault(key, []).append(tr.values[:n][sel])
            if not all_vals:
                continue
            gm = float(np.median(np.concatenate(all_vals)))
            self.global_mean[d] = gm
            self.table[d] = {k: float(np.median(np.concatenate(v))) for k, v in buckets.items() if sum(len(x) for x in v) >= self.min_count}
        return self

    def transform(self, rep: VocalRepresentation) -> dict[str, np.ndarray]:
        """Context-normalised residual tracks ``<dim>_ctx`` (NaN where unknown)."""
        lar, tb = self._keys(rep)
        out = {}
        for d, tab in self.table.items():
            if d not in rep.tracks:
                continue
            tr = rep.tracks[d]
            n = min(len(tr.values), len(lar))
            expect = np.array([tab.get((l, b), np.nan) for l, b in zip(lar[:n], tb[:n])])
            res = np.full(len(tr.values), np.nan)
            res[:n] = np.where(tr.valid[:n], tr.values[:n] - expect, np.nan)
            out[f"{d}_ctx"] = res
        return out

    def describe(self) -> dict[str, dict[str, float]]:
        return {
            d: {f"{LARYNGEAL_CLASSES[l]}@{self.time_bins_ms[b]:g}ms": v for (l, b), v in sorted(tab.items())}
            for d, tab in self.table.items()
        }

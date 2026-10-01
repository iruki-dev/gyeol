"""Pitch evaluation on human-annotated real singing (revision D3).

Synthetic exact-f0 data trains the pitch model; *real* singing with human f0 annotations (Vocadito, MIR-1K —
:mod:`gyeol.data.pitch_sets`) measures it.  Metrics follow the usual melody-extraction definitions (mir_eval style, 50-cent tolerance):
raw pitch accuracy (RPA), raw chroma accuracy (RCA), voicing recall and voicing false alarm, and overall accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.status import Result
from ..data.manifest import Manifest, open_manifest
from ..data.pitch_sets import load_item_audio, read_reference_f0


@dataclass
class PitchScores:
    rpa: float
    rca: float
    voicing_recall: float
    voicing_false_alarm: float
    overall_accuracy: float
    n_voiced: int
    n_frames: int


def score_pitch(ref_t: np.ndarray, ref_hz: np.ndarray, est_t: np.ndarray, est_hz: np.ndarray, tol_cents: float = 50.0) -> PitchScores:
    """Scores of an estimate (NaN or ≤ 0 = unvoiced) sampled at the reference times (nearest estimate frame)."""
    est_hz = np.where(np.isfinite(est_hz) & (est_hz > 0), est_hz, 0.0)
    idx = np.clip(np.searchsorted(est_t, ref_t), 1, max(1, len(est_t) - 1))
    if len(est_t) > 1:
        idx = np.where(np.abs(est_t[idx - 1] - ref_t) <= np.abs(est_t[np.minimum(idx, len(est_t) - 1)] - ref_t), idx - 1, idx)
    e = est_hz[np.clip(idx, 0, len(est_hz) - 1)] if len(est_hz) else np.zeros_like(ref_hz)
    rv, ev = ref_hz > 0, e > 0
    d = 1200 * np.log2(np.where(rv & ev, e, 1.0) / np.where(rv & ev, ref_hz, 1.0))
    ok = rv & ev & (np.abs(d) <= tol_cents)
    chroma = rv & ev & (np.abs((d + 600) % 1200 - 600) <= tol_cents)
    n_v, n_u = int(rv.sum()), int((~rv).sum())
    return PitchScores(float(ok.sum() / max(n_v, 1)), float(chroma.sum() / max(n_v, 1)), float((rv & ev).sum() / max(n_v, 1)),
                       float((~rv & ev).sum() / max(n_u, 1)), float((ok.sum() + (~rv & ~ev).sum()) / max(len(ref_hz), 1)), n_v, len(ref_hz))


@dataclass
class PitchEvalReport:
    dataset: str
    tracker: str
    items: dict[str, PitchScores] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.items:
            return {"dataset": self.dataset, "tracker": self.tracker, "n_items": 0, "failures": self.failures}
        s = list(self.items.values())
        w = np.array([x.n_voiced for x in s], float)
        mean = lambda k: float(np.mean([getattr(x, k) for x in s]))  # noqa: E731
        return {"dataset": self.dataset, "tracker": self.tracker, "n_items": len(s),
                "rpa": mean("rpa"), "rca": mean("rca"), "voicing_recall": mean("voicing_recall"),
                "voicing_false_alarm": mean("voicing_false_alarm"), "overall_accuracy": mean("overall_accuracy"),
                "rpa_frame_weighted": float(np.sum([x.rpa for x in s] * w) / max(w.sum(), 1)), "failures": self.failures}


def evaluate_pitch(manifest: Manifest | str | Path, tracker, limit: int | None = None
                   ) -> Result[PitchEvalReport]:
    """Run ``tracker`` (any :class:`~gyeol.pitch.base.PitchTracker`) on every annotated item of ``manifest``."""
    ds = open_manifest(manifest)
    rep = PitchEvalReport(ds.manifest.dataset, getattr(tracker, "name", type(tracker).__name__))
    for i, item in enumerate(ds):
        if limit is not None and i >= limit:
            break
        try:
            x, sr = load_item_audio(ds.resolve(item), item)
            rt, rhz = read_reference_f0(ds.manifest.root, item)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            rep.failures.append(f"{item.path}: {type(exc).__name__}: {exc}")
            continue
        tr = tracker.track(x, sr)
        if not tr.usable:
            rep.failures.append(f"{item.path}: {tr.reason}")
            continue
        rep.items[item.path] = score_pitch(rt, rhz, tr.value.times, tr.value.f0_hz)
    if not rep.items:
        return Result.failure(f"no item could be evaluated ({len(rep.failures)} failures)")
    return Result.success(rep)


__all__ = ["PitchEvalReport", "PitchScores", "evaluate_pitch", "score_pitch"]

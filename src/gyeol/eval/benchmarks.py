"""Coaching-validity benchmarks (evaluation §6): expert-annotated external sets.

The first target is **VocalCoachBench** (arXiv:2609.04241): 515 recordings,
18 professional vocal trainers, segment-grounded feedback tagged with a
7-label issue taxonomy under 3 parents — Technical Production (Breath,
Vocalization, Technique), Musical Accuracy (Pitch, Rhythm) and Delivery
(Diction, Expression).  Its audio keeps each source corpus's license (DAMP
under the Smule Research Data License; several CC BY-NC-SA sources; see
:mod:`gyeol.core.assets`).

The release's exact JSON layout is not given in the paper, so the loader uses
a :class:`BenchmarkFieldMap` (dotted paths, like the AI Hub adapter): check it
against the files with :func:`gyeol.data.adapters.inspect_json_keys` after
download; clips whose fields are missing are reported, never guessed.

What gyeol can be scored on (and what not):

* gyeol explains a take **against a target** (sing-along).  Clips with a
  reference performance (the same-song subset) can be explained; clips
  without one are skipped with a reason.
* **Top-k issue-label identification**: gyeol's items, ranked by the coach's
  priority, are mapped to the benchmark taxonomy (:data:`GYEOL_TO_LABELS`)
  and compared with the experts' labels for the clip, next to the
  label-prior baseline (always the k most frequent labels).
* **Segment grounding**: an expert segment is recalled when a gyeol item with
  a compatible label overlaps it in time; precision counts gyeol items that
  overlap any compatible expert segment.
* Free-form claims (diagnosis / correction text) are outside gyeol's
  structured output and are not scored.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.containers import Explanation
from ..core.status import Result

LABELS = ("breath", "vocalization", "technique", "pitch", "rhythm", "diction", "expression")

#: gyeol explanation categories / attributes → benchmark labels they can speak to
GYEOL_TO_LABELS: dict[str, tuple[str, ...]] = {
    "pitch": ("pitch",),
    "rhythm": ("rhythm",),
    "ornament": ("technique", "expression"),
    "dynamics": ("expression",),
    "phonation": ("vocalization", "breath"),
    "diction": ("diction",),
    "phonation/breathiness": ("breath",),
    "phonation/register": ("vocalization", "technique"),
}


def labels_for(category: str, attribute: str = "") -> tuple[str, ...]:
    return GYEOL_TO_LABELS.get(f"{category}/{attribute}", GYEOL_TO_LABELS.get(category, ()))


@dataclass(frozen=True)
class ExpertSegment:
    start_s: float
    end_s: float
    label: str
    expert: str = ""


@dataclass
class BenchmarkClip:
    clip_id: str
    audio: Path
    reference: Path | None
    segments: list[ExpertSegment]
    meta: dict = field(default_factory=dict)

    @property
    def labels(self) -> set[str]:
        return {s.label for s in self.segments}


@dataclass
class BenchmarkFieldMap:
    """Dotted paths into one clip record.  Placeholders until checked against the release."""

    clips: str = "clips"  # path to the list of clip records in the index file
    clip_id: str = "id"
    audio: str = "audio_path"
    reference: str | None = "reference_path"
    segments: str = "segments"
    seg_start: str = "start"
    seg_end: str = "end"
    seg_label: str = "issue"
    seg_expert: str | None = "annotator"


def _get(d: Any, path: str | None) -> Any:
    if path is None:
        return None
    cur = d
    for part in path.split("."):
        if isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


@dataclass
class LoadReport:
    clips: list[BenchmarkClip]
    skipped: dict[str, str]  # clip id (or index) → reason


def load_expert_benchmark(index_json: str | Path, field_map: BenchmarkFieldMap | None = None) -> Result[LoadReport]:
    """Load a locally downloaded benchmark index."""
    fm = field_map or BenchmarkFieldMap()
    path = Path(index_json)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Result.failure(f"cannot read {path}: {exc}")
    recs = _get(data, fm.clips) if fm.clips else data
    if not isinstance(recs, list):
        return Result.failure(f"no clip list at {fm.clips!r} in {path} (check the field map with inspect_json_keys)")
    clips, skipped = [], {}
    for i, r in enumerate(recs):
        cid = _get(r, fm.clip_id)
        audio = _get(r, fm.audio)
        segs = _get(r, fm.segments)
        if cid is None or audio is None or not isinstance(segs, list):
            skipped[str(cid if cid is not None else i)] = "missing id, audio or segments (field map?)"
            continue
        out = []
        for s in segs:
            lab = str(_get(s, fm.seg_label) or "").strip().lower()
            st, en = _get(s, fm.seg_start), _get(s, fm.seg_end)
            if lab not in LABELS or st is None or en is None:
                continue
            out.append(ExpertSegment(float(st), float(en), lab, str(_get(s, fm.seg_expert) or "")))
        ref = _get(r, fm.reference)
        clips.append(BenchmarkClip(str(cid), path.parent / str(audio), path.parent / str(ref) if ref else None, out))
    return Result.success(LoadReport(clips, skipped))


# ---------------------------------------------------------------- scoring


@dataclass(frozen=True)
class Prediction:
    category: str
    attribute: str
    start_s: float
    end_s: float
    score: float  # priority (higher = more important)


def predictions_from_explanation(exp: Explanation, scores: dict[tuple, float] | None = None) -> list[Prediction]:
    """gyeol items as timed predictions; ``scores`` (item key → priority) default to the item confidence."""
    out = []
    for it in exp.items:
        for sp in it.spans or []:
            a, b = sp.seconds(exp.grid)
            out.append(Prediction(it.category, it.attribute, a, max(b, a + exp.grid.hop_seconds),
                                  float((scores or {}).get(it.key, it.confidence))))
    return out


@dataclass
class BenchmarkReport:
    n_clips: int
    topk: int
    topk_hit_rate: float  # fraction of clips where ≥ 1 of gyeol's top-k labels is among the experts' labels
    prior_hit_rate: float  # same for the label-prior baseline
    segment_recall: dict[str, float]  # per label
    segment_precision: float
    label_counts: dict[str, int]
    skipped: dict[str, str]


def _overlap(a0, a1, b0, b1, tol) -> bool:
    return min(a1, b1) - max(a0, b0) > -tol


def score_benchmark(predictions: dict[str, list[Prediction]], clips: list[BenchmarkClip], k: int = 3, tolerance_s: float = 0.1) -> BenchmarkReport:
    counts = Counter(s.label for c in clips for s in c.segments)
    prior = [lab for lab, _ in counts.most_common(k)]
    hits, prior_hits, n = 0, 0, 0
    rec_hit: Counter = Counter()
    rec_all: Counter = Counter()
    prec_ok, prec_all = 0, 0
    skipped = {}
    for c in clips:
        if c.clip_id not in predictions:
            skipped[c.clip_id] = "no gyeol prediction (e.g. no reference performance to explain against)"
            continue
        if not c.segments:
            skipped[c.clip_id] = "no expert segments with known labels"
            continue
        preds = predictions[c.clip_id]
        n += 1
        ranked: list[str] = []
        for p in sorted(preds, key=lambda p: -p.score):
            for lab in labels_for(p.category, p.attribute):
                if lab not in ranked:
                    ranked.append(lab)
        hits += bool(set(ranked[:k]) & c.labels)
        prior_hits += bool(set(prior) & c.labels)
        for s in c.segments:
            rec_all[s.label] += 1
            rec_hit[s.label] += any(s.label in labels_for(p.category, p.attribute) and _overlap(p.start_s, p.end_s, s.start_s, s.end_s, tolerance_s)
                                    for p in preds)
        for p in preds:
            prec_all += 1
            prec_ok += any(s.label in labels_for(p.category, p.attribute) and _overlap(p.start_s, p.end_s, s.start_s, s.end_s, tolerance_s)
                           for s in c.segments)
    rate = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return BenchmarkReport(n, k, rate(hits, n), rate(prior_hits, n), {lab: rate(rec_hit[lab], rec_all[lab]) for lab in rec_all},
                           rate(prec_ok, prec_all), dict(counts), skipped)


# ---------------------------------------------------------------- coach agreement


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' κ for an (items × categories) matrix of rater counts (equal raters per item)."""
    r = np.asarray(ratings, float)
    n = r.sum(1)
    if np.any(n != n[0]) or n[0] < 2:
        raise ValueError("every item needs the same number (≥ 2) of ratings")
    n = n[0]
    p_j = r.sum(0) / r.sum()
    P_i = (np.sum(r**2, 1) - n) / (n * (n - 1))
    Pe = float(np.sum(p_j**2))
    return float((P_i.mean() - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


@dataclass
class AgreementReport:
    fleiss_per_label: dict[str, float]  # coaches among themselves (label present / absent per clip)
    gyeol_vs_consensus: dict[str, float]  # Cohen's κ of gyeol against the coaches' majority
    n_clips: int
    n_raters: int


def coach_agreement(coach_labels: dict[str, dict[str, set[str]]], gyeol_labels: dict[str, set[str]] | None = None,
                    labels: tuple[str, ...] = LABELS) -> AgreementReport:
    """Internal coach-agreement protocol (docs/protocols/coach_agreement.md).

    ``coach_labels[clip][rater]`` = labels that rater flagged on the clip.
    """
    clips = sorted(coach_labels)
    raters = sorted({r for c in clips for r in coach_labels[c]})
    clips = [c for c in clips if set(coach_labels[c]) == set(raters)]  # complete ratings only
    fl, gv = {}, {}
    for lab in labels:
        m = np.array([[sum(lab in coach_labels[c][r] for r in raters), sum(lab not in coach_labels[c][r] for r in raters)] for c in clips])
        if len(clips) and m[:, 0].sum() not in (0, m.sum()):
            fl[lab] = fleiss_kappa(m)
        if gyeol_labels is not None and len(clips):
            cons = np.array([m_[0] * 2 > len(raters) for m_ in m])
            gy = np.array([lab in gyeol_labels.get(c, set()) for c in clips])
            if len(set(cons)) > 1 or len(set(gy)) > 1:
                gv[lab] = cohen_kappa(cons, gy)
    return AgreementReport(fl, gv, len(clips), len(raters))

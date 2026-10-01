"""Vocal-health guard (brief §7) — pure logic over measured numbers.

* **Range and tessitura** (:class:`VoiceRange`, from onboarding) and
  :func:`check_phrase`: is the target phrase comfortable?  If not, suggest an
  octave choice and/or a transposition in semitones.
* **Beginner restrictions** (:func:`restricted_for_level`): rough voice, fry
  and pressed/belt qualities, and chest-register targets above the user's
  tessitura (high belting), are never coached for beginners.
* **Phonation time** (:func:`phonation_warnings`): warnings from the voiced
  time accumulated in a session and a day (:func:`voiced_seconds` measures one
  attempt; the accumulation is user state and lives in the service layer —
  ``reference_service/gyeol_service/wellbeing.py``).
* **Fatigue within a session** (:func:`fatigue_flags`): rising f0
  instability, a falling top of range, increasing breath noise across a
  sequence of :class:`AttemptMetrics` — each judged against the session's own
  attempt-to-attempt noise.

Revision C1: every function here is stateless; attempt histories, phonation
logs and the medical-referral notice text belong to the service.

Limits and defaults come from ``resources/coach/norms.json`` (policy data,
provisional until reviewed); nothing here diagnoses anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Sequence

import numpy as np

from ..core.containers import ExplanationItem, Representation


@lru_cache(maxsize=2)
def load_norms() -> dict:
    return json.loads(resources.files("gyeol").joinpath("resources/coach/norms.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- range


@dataclass(frozen=True)
class VoiceRange:
    """Pitches in cents re A4.  ``tess_*`` is the comfortable singing zone inside the range."""

    low_cents: float
    high_cents: float
    tess_low_cents: float
    tess_high_cents: float
    source: str = "onboarding"

    def __post_init__(self) -> None:
        if not (self.low_cents <= self.tess_low_cents < self.tess_high_cents <= self.high_cents):
            raise ValueError("need low ≤ tess_low < tess_high ≤ high")

    @classmethod
    def from_samples(cls, glide_cents: np.ndarray, comfortable_cents: np.ndarray, source: str = "onboarding") -> "VoiceRange":
        """Range from siren/glide pitch samples (2nd–98th percentile), tessitura from
        comfortably sung notes (10th–90th percentile)."""
        g = np.asarray(glide_cents, float)
        c = np.asarray(comfortable_cents, float)
        g, c = g[np.isfinite(g)], c[np.isfinite(c)]
        if g.size < 10 or c.size < 5:
            raise ValueError("too few pitch samples for a range")
        lo, hi = np.percentile(g, [2, 98])
        tl, th = np.percentile(c, [10, 90])
        tl, th = max(tl, lo), min(th, hi)
        if th <= tl:
            raise ValueError("comfortable zone is empty")
        return cls(float(lo), float(hi), float(tl), float(th), source)


@dataclass
class PhraseCheck:
    status: str  # comfortable | above_tessitura | below_tessitura | out_of_range | no_notes
    octave_shift_cents: float  # applied to the target notes to fit the user's register
    transpose_semitones: int  # suggested extra transposition (0 = none)
    frac_outside_tessitura: float
    frac_outside_range: float


def _fracs(notes: np.ndarray, vr: VoiceRange) -> tuple[float, float, float, float]:
    out_r = float(np.mean((notes < vr.low_cents) | (notes > vr.high_cents)))
    above = float(np.mean(notes > vr.tess_high_cents))
    below = float(np.mean(notes < vr.tess_low_cents))
    return out_r, above + below, above, below


def check_phrase(target_note_cents: list[float] | np.ndarray, vr: VoiceRange, allow_octave: bool = True,
                 max_transpose: int = 6) -> PhraseCheck:
    n = np.asarray(target_note_cents, float)
    n = n[np.isfinite(n)]
    if n.size == 0:
        return PhraseCheck("no_notes", 0.0, 0, 0.0, 0.0)
    shifts = [k * 1200.0 for k in (0, -1, 1, -2, 2)] if allow_octave else [0.0]
    best = min(shifts, key=lambda s: (_fracs(n + s, vr)[:2], abs(s)))
    fr, ft, _, _ = _fracs(n + best, vr)
    semis = 0
    if ft > 0:
        cand = min(range(-max_transpose, max_transpose + 1), key=lambda k: (_fracs(n + best + 100 * k, vr)[:2], abs(k)))
        if _fracs(n + best + 100 * cand, vr)[:2] < (fr, ft):
            semis = cand
    fr0, ft0, above, below = _fracs(n + best, vr)
    if fr0 > 0:
        status = "out_of_range"
    elif above > 0:
        status = "above_tessitura"
    elif below > 0:
        status = "below_tessitura"
    else:
        status = "comfortable"
    return PhraseCheck(status, best, semis, ft0, fr0)


def note_centres(rep: Representation) -> list[float]:
    """Median pitch centre of each note (cents re A4), NaN when unknown."""
    pc = rep.curves["pitch_center"].values
    out = []
    for s, e in rep.meta.get("notes", []):
        v = pc[s:e][np.isfinite(pc[s:e])]
        out.append(float(np.median(v)) if v.size else float("nan"))
    return out


# ---------------------------------------------------------------- beginner restrictions


def restricted_for_level(item: ExplanationItem, level: str, vr: VoiceRange | None = None,
                         target_note_cents: list[float] | None = None, octave_shift_cents: float = 0.0) -> str | None:
    """Reason string if this item must not be coached at ``level``, else None."""
    if level != "beginner":
        return None
    h = load_norms()["health"]
    attr = item.attribute
    if attr.startswith("quality_") and attr[len("quality_"):] in h["beginner_restricted_qualities"] and item.magnitude < 0:
        return f"beginner_restricted:{attr[len('quality_'):]}"  # coaching would push toward rough / fry / belt
    if attr == "register" and h.get("beginner_no_chest_above_tessitura") and item.detail.get("target") == "chest":
        k = item.detail.get("target_note", -1)
        if vr is not None and target_note_cents is not None and 0 <= k < len(target_note_cents):
            if np.isfinite(target_note_cents[k]) and target_note_cents[k] + octave_shift_cents > vr.tess_high_cents:
                return "beginner_restricted:high_belt"
    return None


# ---------------------------------------------------------------- phonation time


def phonation_warnings(session_s: float, day_s: float) -> list[str]:
    """``phonation_session`` / ``phonation_daily`` when the accumulated voiced time reaches the norms."""
    for v in (session_s, day_s):
        if v < 0 or not np.isfinite(v):
            raise ValueError("voiced time must be a finite, non-negative number")
    h = load_norms()["health"]
    out = []
    if session_s >= h["session_phonation_warn_s"]:
        out.append("phonation_session")
    if day_s >= h["daily_phonation_warn_s"]:
        out.append("phonation_daily")
    return out


def voiced_seconds(rep: Representation) -> float:
    v = rep.curves["voicing"].values
    return float(np.sum(np.nan_to_num(v) > 0.5) * rep.grid.hop_seconds)


# ---------------------------------------------------------------- fatigue


@dataclass(frozen=True)
class AttemptMetrics:
    instability_cents: float  # frame-to-frame f0 jitter outside vibrato (robust SD)
    top_cents: float  # 95th percentile of the sung pitch (cents re A4)
    breath_db: float  # median aperiodic-to-periodic ratio over voiced frames


def attempt_metrics(rep: Representation, min_confidence: float = 0.5) -> AttemptMetrics:
    f0 = rep.curves["f0_cents"]
    ok = (f0.confidence >= min_confidence) & np.isfinite(f0.values)
    if "vibrato_extent" in rep.curves:
        ok &= rep.curves["vibrato_extent"].confidence < 0.5
    d = np.diff(f0.values)
    both = ok[1:] & ok[:-1]
    inst = float(1.4826 * np.median(np.abs(d[both] - np.median(d[both])))) if both.sum() > 5 else float("nan")
    top = float(np.percentile(f0.values[ok], 95)) if ok.sum() > 5 else float("nan")
    ap = rep.curves["aperiodic_ratio"]
    aok = (ap.confidence > 0) & np.isfinite(ap.values)
    breath = float(np.median(ap.values[aok])) if aok.sum() > 5 else float("nan")
    return AttemptMetrics(inst, top, breath)


def _trend_z(x: np.ndarray, window: int) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2 * window:
        return 0.0
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]  # insensitive to a linear trend
    noise = max(1.4826 * float(np.median(np.abs(d2))) / np.sqrt(6.0), 1e-9 * (1.0 + float(np.max(np.abs(x)))))
    return float((np.median(x[-window:]) - np.median(x[:window])) / noise)


def fatigue_flags(history: Sequence[AttemptMetrics]) -> list[str]:
    """Fatigue signs across a session's attempts (oldest first): ``fatigue_instability``, ``fatigue_top_range``,
    ``fatigue_breath``.  Each compares the last and first ``fatigue_window`` attempts against the session's own noise."""
    h = load_norms()["health"]
    if len(history) < h["fatigue_min_attempts"]:
        return []
    w, zc = int(h["fatigue_window"]), float(h["fatigue_z"])
    col = lambda name: np.array([getattr(m, name) for m in history], float)  # noqa: E731
    out = []
    if _trend_z(col("instability_cents"), w) > zc:
        out.append("fatigue_instability")
    if _trend_z(col("top_cents"), w) < -zc:
        out.append("fatigue_top_range")
    if _trend_z(col("breath_db"), w) > zc:
        out.append("fatigue_breath")
    return out

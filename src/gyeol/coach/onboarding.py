"""Onboarding screen in the style of SSAP / SPB (Seattle Singing Accuracy
Protocol; Sung Performance Battery).

Tasks (the app plays stimuli and measures the sung pitch; this module scores
the numbers):

* single-pitch matching — :class:`PitchMatchTrial`
* intervals — :class:`IntervalTrial`
* a short melody — :class:`MelodyTrial`
* pitch discrimination — :class:`DiscriminationTrial`

The profile keeps three things separate:

* **production accuracy** — mean absolute error (cents); pitch matching is
  octave-folded (singing a target in one's own octave is correct);
* **precision** — consistency: pooled SD of signed errors over repeated
  trials of the same target (bias removed);
* **perception** — discrimination threshold: the smallest pitch difference
  answered correctly at the criterion rate (isotonic fit of accuracy vs Δ).

Routing: poor production with good perception → own-voice imitation and
wide-range pitch matching (the brief's rule); poor perception → listening
practice.  Band cut-offs come from ``resources/coach/norms.json``
(literature-based, provisional).  The profile never labels anyone: band
names are ``on_target`` / ``developing``, and no user-facing string says
"tone-deaf" (a test enforces this over every resource file).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..verification.thresholds import _pava
from .health import VoiceRange, load_norms


def fold_octave(cents: float | np.ndarray) -> float | np.ndarray:
    """Wrap a pitch error into [−600, 600) cents."""
    return (np.asarray(cents, float) + 600.0) % 1200.0 - 600.0


@dataclass(frozen=True)
class PitchMatchTrial:
    target_cents: float
    sung_cents: float  # median stable sung pitch; NaN = nothing usable was sung


@dataclass(frozen=True)
class IntervalTrial:
    interval_cents: float  # target interval (signed)
    sung_first_cents: float
    sung_second_cents: float


@dataclass(frozen=True)
class MelodyTrial:
    target_notes_cents: tuple[float, ...]
    sung_notes_cents: tuple[float, ...]  # same length; NaN for missed notes


@dataclass(frozen=True)
class DiscriminationTrial:
    delta_cents: float  # |pitch difference| presented
    correct: bool


@dataclass
class OnboardingProfile:
    production_mae_cents: float | None
    precision_sd_cents: float | None
    perception_threshold_cents: float | None  # None: not reached within the tested Δ, or not measured
    production_band: str | None  # on_target | developing
    precision_band: str | None  # consistent | variable
    perception_band: str | None  # fine | developing
    routes: list[str]
    voice_range: VoiceRange | None = None
    per_task_mae: dict[str, float] = field(default_factory=dict)
    status: dict[str, str] = field(default_factory=dict)  # per dimension: ok | not_enough_trials | not_reached


def _pitch_errors(trials: list[PitchMatchTrial]) -> tuple[np.ndarray, np.ndarray]:
    err = np.array([fold_octave(t.sung_cents - t.target_cents) for t in trials], float)
    tgt = np.array([t.target_cents for t in trials], float)
    ok = np.isfinite(err)
    return err[ok], tgt[ok]


def perception_threshold(trials: list[DiscriminationTrial], criterion: float) -> tuple[float | None, str]:
    """Smallest Δ with (isotonic) accuracy ≥ criterion, linearly interpolated."""
    by = defaultdict(list)
    for t in trials:
        by[float(abs(t.delta_cents))].append(bool(t.correct))
    deltas = np.array(sorted(by))
    acc = np.array([np.mean(by[d]) for d in deltas])
    w = np.array([len(by[d]) for d in deltas], float)
    fit = _pava(acc, w, increasing=True)
    above = np.flatnonzero(fit >= criterion)
    if above.size == 0:
        return None, "not_reached"
    j = int(above[0])
    if j == 0 or fit[j] == fit[j - 1]:
        return float(deltas[j]), "ok"
    return float(np.interp(criterion, [fit[j - 1], fit[j]], [deltas[j - 1], deltas[j]])), "ok"


def score_onboarding(pitch: list[PitchMatchTrial], intervals: list[IntervalTrial], melodies: list[MelodyTrial],
                     discrimination: list[DiscriminationTrial], voice_range: VoiceRange | None = None,
                     norms: dict | None = None) -> OnboardingProfile:
    nm = (norms or load_norms())["onboarding"]
    mins = nm["min_trials"]
    status: dict[str, str] = {}
    per_task: dict[str, float] = {}
    abs_errors: list[float] = []

    pe, ptgt = _pitch_errors(pitch)
    if len(pe) >= mins["pitch_match"]:
        per_task["pitch_match"] = float(np.mean(np.abs(pe)))
        abs_errors += list(np.abs(pe))
    ie = np.array([t.sung_second_cents - t.sung_first_cents - t.interval_cents for t in intervals], float)
    ie = ie[np.isfinite(ie)]
    if len(ie) >= mins["interval"]:
        per_task["interval"] = float(np.mean(np.abs(ie)))
        abs_errors += list(np.abs(ie))
    me: list[float] = []
    for m in melodies:
        if len(m.target_notes_cents) != len(m.sung_notes_cents):
            raise ValueError("melody trial: target and sung note counts differ")
        ti, si = np.diff(m.target_notes_cents), np.diff(m.sung_notes_cents)
        d = si - ti
        me += list(np.abs(d[np.isfinite(d)]))
    if len(melodies) >= mins["melody"] and me:
        per_task["melody"] = float(np.mean(me))
        abs_errors += me
    mae = float(np.mean(abs_errors)) if per_task else None
    status["production"] = "ok" if mae is not None else "not_enough_trials"

    # precision: pooled within-target SD (bias per target removed); needs repeats
    groups = defaultdict(list)
    for e, t in zip(pe, ptgt):
        groups[round(float(t), 1)].append(e)
    reps = [np.asarray(g) for g in groups.values() if len(g) >= 2]
    if reps:
        ss = sum(float(np.sum((g - g.mean()) ** 2)) for g in reps)
        dof = sum(len(g) - 1 for g in reps)
        sd = float(np.sqrt(ss / dof))
        status["precision"] = "ok"
    else:
        sd = None
        status["precision"] = "not_enough_trials"

    thr: float | None = None
    if len(discrimination) >= mins["discrimination"]:
        thr, st = perception_threshold(discrimination, nm["perception_criterion"])
        status["perception"] = st
    else:
        status["perception"] = "not_enough_trials"

    prod_band = None if mae is None else ("on_target" if mae <= nm["poor_production_mae_cents"] else "developing")
    prec_band = None if sd is None else ("consistent" if sd <= nm["imprecise_sd_cents"] else "variable")
    if status["perception"] == "not_reached":
        perc_band = "developing"
    elif thr is None:
        perc_band = None
    else:
        perc_band = "fine" if thr <= nm["good_perception_threshold_cents"] else "developing"

    routes: list[str] = []
    if prod_band == "developing" and perc_band == "fine":
        routes += ["own_voice_imitation", "wide_range_pitch_matching"]
    elif prod_band == "developing":
        routes += ["wide_range_pitch_matching"]
    if perc_band == "developing":
        routes.append("perception_training")
    if not routes and prod_band is not None:
        routes.append("standard")
    return OnboardingProfile(mae, sd, thr, prod_band, prec_band, perc_band, routes, voice_range, per_task, status)

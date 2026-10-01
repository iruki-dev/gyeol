"""Pedagogy policy layer (M6) — pure logic over explanations and measured numbers, no ML.

* :mod:`~gyeol.coach.thresholds` — fitted operating thresholds (never hard-coded);
* :mod:`~gyeol.coach.priority` — reliability × audibility ranking with a category tie order;
* :mod:`~gyeol.coach.practice` — practice suggestions from a coach-authored data file;
* :mod:`~gyeol.coach.onboarding` — SSAP/SPB-style screen: production accuracy, precision, perception, routing;
* :mod:`~gyeol.coach.health` — range/tessitura checks, beginner restrictions, phonation-time and fatigue measures.

Revision C1: the library is stateless.  The coaching *session* (feedback
fading schedule, attempt history, self-assessment flow, summaries) and the
consent / storage / deletion store live in the reference service package,
``reference_service/gyeol_service`` — one way to hold user state on top of
these functions, not part of the library.
"""

from .health import (
    AttemptMetrics,
    PhraseCheck,
    VoiceRange,
    attempt_metrics,
    check_phrase,
    fatigue_flags,
    note_centres,
    phonation_warnings,
    restricted_for_level,
    voiced_seconds,
)
from .onboarding import (
    DiscriminationTrial,
    IntervalTrial,
    MelodyTrial,
    OnboardingProfile,
    PitchMatchTrial,
    fold_octave,
    perception_threshold,
    score_onboarding,
)
from .practice import Exercise, PracticeMap
from .priority import DEFAULT_TIERS, PriorityConfig, RankedItem, rank
from .thresholds import AttributeThreshold, ThresholdSet, fit_attribute_threshold

__all__ = [
    "AttemptMetrics", "AttributeThreshold", "DEFAULT_TIERS", "DiscriminationTrial", "Exercise", "IntervalTrial", "MelodyTrial",
    "OnboardingProfile", "PhraseCheck", "PitchMatchTrial", "PracticeMap", "PriorityConfig", "RankedItem", "ThresholdSet",
    "VoiceRange", "attempt_metrics", "check_phrase", "fatigue_flags", "fit_attribute_threshold", "fold_octave", "note_centres",
    "perception_threshold", "phonation_warnings", "rank", "restricted_for_level", "score_onboarding", "voiced_seconds",
]

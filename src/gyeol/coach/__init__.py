"""Pedagogy policy layer (M6) — pure logic over explanations and measured numbers, no ML.

* :mod:`~gyeol.coach.thresholds` — fitted operating thresholds (never hard-coded);
* :mod:`~gyeol.coach.priority` — reliability × audibility ranking with a category tie order;
* :mod:`~gyeol.coach.session` — feedback volume, fading schedule, self-assessment first, summaries;
* :mod:`~gyeol.coach.practice` — practice suggestions from a coach-authored data file;
* :mod:`~gyeol.coach.onboarding` — SSAP/SPB-style screen: production accuracy, precision, perception, routing;
* :mod:`~gyeol.coach.health` — range/tessitura checks, beginner restrictions, phonation time, fatigue, referral notice.
"""

from .health import (
    AttemptMetrics,
    FatigueMonitor,
    PhonationLog,
    PhraseCheck,
    VoiceRange,
    attempt_metrics,
    check_phrase,
    note_centres,
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
from .session import Attempt, CoachConfig, CoachSession, Feedback, FeedbackEntry, SelfAssessmentPrompt, SessionSummary
from .thresholds import AttributeThreshold, ThresholdSet, fit_attribute_threshold

__all__ = [
    "Attempt", "AttemptMetrics", "AttributeThreshold", "CoachConfig", "CoachSession", "DEFAULT_TIERS", "DiscriminationTrial",
    "Exercise", "FatigueMonitor", "Feedback", "FeedbackEntry", "IntervalTrial", "MelodyTrial", "OnboardingProfile",
    "PhonationLog", "PhraseCheck", "PitchMatchTrial", "PracticeMap", "PriorityConfig", "RankedItem", "SelfAssessmentPrompt",
    "SessionSummary", "ThresholdSet", "VoiceRange", "attempt_metrics", "check_phrase", "fit_attribute_threshold",
    "fold_octave", "note_centres", "perception_threshold", "rank", "restricted_for_level", "score_onboarding",
    "voiced_seconds",
]

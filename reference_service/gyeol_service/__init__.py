"""gyeol reference service — user state on top of the stateless gyeol library (revision C1).

* :mod:`gyeol_service.store` — consent records, feature / raw-audio storage with retention, user deletion;
* :mod:`gyeol_service.session` — coaching session: feedback volume, fading schedule, self-assessment, attempt
  history and summaries;
* :mod:`gyeol_service.wellbeing` — running phonation time and fatigue history.

This package shows one way to hold that state (local files, in-process objects).  A production service would
replace the storage with its own database while calling the same library functions.  It is not part of the
``gyeol`` wheel: install it with ``pip install -e reference_service``.
"""

from .session import Attempt, CoachConfig, CoachSession, Feedback, FeedbackEntry, SelfAssessmentPrompt, SessionSummary, coach_strings
from .store import ConsentStore, FeatureStore, RawAudioRetention, RawAudioStore, RetentionPolicy, delete_user
from .wellbeing import FatigueMonitor, PhonationLog

__all__ = ["Attempt", "CoachConfig", "CoachSession", "ConsentStore", "FatigueMonitor", "FeatureStore", "Feedback", "FeedbackEntry",
           "PhonationLog", "RawAudioRetention", "RawAudioStore", "RetentionPolicy", "SelfAssessmentPrompt", "SessionSummary",
           "coach_strings", "delete_user"]

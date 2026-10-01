"""Per-user accumulation for the health guard (reference service layer, revision C1).

The measurements are stateless library functions (:func:`gyeol.coach.voiced_seconds`,
:func:`gyeol.coach.attempt_metrics`, :func:`gyeol.coach.phonation_warnings`,
:func:`gyeol.coach.fatigue_flags`); keeping the running totals and the attempt
history is user state and belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gyeol.coach.health import AttemptMetrics, fatigue_flags, phonation_warnings


@dataclass
class PhonationLog:
    """Accumulated voiced seconds per session and per day (``day`` is any label, e.g. an ISO date)."""

    session_s: float = 0.0
    by_day: dict[str, float] = field(default_factory=dict)

    def add(self, voiced_seconds: float, day: str) -> None:
        if voiced_seconds < 0 or not np.isfinite(voiced_seconds):
            raise ValueError("voiced_seconds must be a finite, non-negative number")
        self.session_s += voiced_seconds
        self.by_day[day] = self.by_day.get(day, 0.0) + voiced_seconds

    def warnings(self, day: str) -> list[str]:
        return phonation_warnings(self.session_s, self.by_day.get(day, 0.0))


@dataclass
class FatigueMonitor:
    history: list[AttemptMetrics] = field(default_factory=list)

    def add(self, m: AttemptMetrics) -> None:
        self.history.append(m)

    def flags(self) -> list[str]:
        return fatigue_flags(self.history)


__all__ = ["FatigueMonitor", "PhonationLog"]

"""Pitch-tracker protocol and the per-tracker result type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.grid import FrameGrid, project
from ..core.status import Result


@dataclass
class PitchTrack:
    """Raw output of one tracker at its own frame times."""

    name: str
    times: np.ndarray  # seconds
    f0_hz: np.ndarray  # NaN where the tracker says unvoiced
    voiced_prob: np.ndarray  # [0, 1]

    def on_grid(self, grid: FrameGrid) -> tuple[np.ndarray, np.ndarray]:
        """(f0_hz, voiced_prob) projected onto ``grid`` (NaN / 0 outside)."""
        cents = 1200 * np.log2(np.where(self.f0_hz > 0, self.f0_hz, np.nan) / 440.0)
        c = project(self.times, cents, grid)
        vp = project(self.times, self.voiced_prob, grid, kind="nearest")
        return 440.0 * 2 ** (c / 1200), np.nan_to_num(vp)


@runtime_checkable
class PitchTracker(Protocol):
    """Anything that turns mono audio into a :class:`PitchTrack`.

    ``asset`` names the tracker's entry in the license registry (``None`` for
    gyeol's own DSP trackers).  Implementations must return a failed /
    unavailable :class:`Result` instead of raising on bad input.
    """

    name: str
    asset: str | None

    def track(self, audio: np.ndarray, sr: int) -> Result[PitchTrack]: ...

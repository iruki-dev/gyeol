"""Frame-level Korean phonetic context tokens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ContextTokens:
    """Frame-level Korean phonetic context (T_context)."""

    rate: float
    syllable_index: np.ndarray  # (T,) int, -1 = none
    laryngeal_class: np.ndarray  # (T,) int index into LARYNGEAL_CLASSES
    time_since_onset_ms: np.ndarray  # (T,) float, NaN = none
    syllable_position: np.ndarray  # (T,) int index into SYLLABLE_POSITIONS
    phrase_initial: np.ndarray  # (T,) bool
    syllables: list[dict[str, Any]] = field(default_factory=list)
    alignment_source: str = "none"

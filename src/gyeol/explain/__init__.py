"""Δc computation, key/octave invariance, take consistency, remainder,
"cannot judge" and audibility (M1 pitch/rhythm/ornaments; M5 all attributes)."""

from .audibility import perceptual_distance, score_audibility
from .explain import EVENT_KINDS, ExplainConfig, explain

__all__ = ["EVENT_KINDS", "ExplainConfig", "explain", "perceptual_distance", "score_audibility"]

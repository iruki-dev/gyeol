"""Attribute curves c(t): signal-derived curves and events (M1); learned heads (M3)."""

from .extract import AnalysisConfig, analyze
from .pitch_curves import EventConfig, detect_events, notes_from_pitch, pitch_center, vibrato_curves
from .signal import harmonic_noise, loudness, relative_loudness

__all__ = ["AnalysisConfig", "EventConfig", "analyze", "detect_events", "harmonic_noise", "loudness", "notes_from_pitch",
           "pitch_center", "relative_loudness", "vibrato_curves"]

"""Core types: frame grid, status, licensing, consent and containers."""

from .consent import ConsentedVoice, ConsentError, ConsentToken, Provenance, Purpose, SingerVector, require_consented_voice
from .containers import (
    AttributeCurve,
    AttributeCurves,
    Consistency,
    Event,
    Explanation,
    ExplanationItem,
    Recording,
    Representation,
    Span,
)
from .grid import DEFAULT_HOP, DEFAULT_SR, FrameGrid, GridMismatchError
from .license import REGISTRY, AssetKind, LicensedAsset, LicenseError, LicenseTag, Profile, decide, lookup, require_allowed
from .status import Result, ResultError, Status

__all__ = [
    "AssetKind", "AttributeCurve", "AttributeCurves", "ConsentError", "ConsentToken", "ConsentedVoice", "Consistency",
    "DEFAULT_HOP", "DEFAULT_SR", "Event", "Explanation", "ExplanationItem", "FrameGrid", "GridMismatchError",
    "LicenseError", "LicenseTag", "LicensedAsset", "Profile", "Provenance", "Purpose", "REGISTRY", "Recording",
    "Representation", "Result", "ResultError", "SingerVector", "Span", "Status", "decide", "lookup",
    "require_allowed", "require_consented_voice",
]

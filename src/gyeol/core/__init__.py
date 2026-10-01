"""Core types: frame grid, status, the third-party asset list and containers."""

from .assets import ASSETS, Asset, add_asset, asset, describe
from .containers import (
    AttributeCurve,
    AttributeCurves,
    Consistency,
    Event,
    Explanation,
    ExplanationItem,
    Premise,
    Recording,
    Representation,
    SingerVector,
    Span,
    WithheldItem,
)
from .grid import DEFAULT_HOP, DEFAULT_SR, FrameGrid, GridMismatchError
from .status import Result, ResultError, Status

__all__ = [
    "ASSETS", "Asset", "AttributeCurve", "AttributeCurves", "Consistency", "DEFAULT_HOP", "DEFAULT_SR", "Event", "Explanation",
    "ExplanationItem", "FrameGrid", "GridMismatchError", "Premise", "Recording", "Representation", "Result", "ResultError",
    "SingerVector", "Span", "Status", "WithheldItem", "add_asset", "asset", "describe",
]

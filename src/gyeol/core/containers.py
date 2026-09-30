"""Typed containers shared by every gyeol module."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Mapping

import numpy as np

from .consent import Provenance, SingerVector
from .grid import FrameGrid, GridMismatchError
from .license import LicensedAsset


@dataclass
class Recording:
    """Mono audio plus who it belongs to and under which license."""

    audio: np.ndarray
    sr: int
    provenance: Provenance = Provenance.UNKNOWN
    owner_id: str | None = None
    recording_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    license: LicensedAsset | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        a = np.asarray(self.audio, dtype=np.float64)
        if a.ndim != 1:
            raise ValueError(f"Recording expects mono audio, got shape {a.shape}")
        self.audio = a
        if self.provenance is Provenance.USER and not self.owner_id:
            raise ValueError("a USER recording needs owner_id")

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sr

    def grid(self, hop: int | None = None) -> FrameGrid:
        from .grid import DEFAULT_HOP

        return FrameGrid.for_samples(len(self.audio), self.sr, hop or DEFAULT_HOP)


@dataclass
class AttributeCurve:
    """One attribute ``c_k(t)`` with a per-frame confidence in [0, 1].

    ``values`` is (T,) or (T, D) (e.g. posteriors).  Frames where the value is
    unknown hold NaN *and* confidence 0.
    """

    name: str
    values: np.ndarray
    confidence: np.ndarray
    grid: FrameGrid
    unit: str = ""
    labels: tuple[str, ...] = ()  # names of the D columns for posteriors
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        self.confidence = np.clip(np.nan_to_num(np.asarray(self.confidence, dtype=np.float64)), 0.0, 1.0)
        self.grid.check_array(self.values, f"{self.name}.values")
        self.grid.check_array(self.confidence, f"{self.name}.confidence")
        unknown = ~np.isfinite(self.values) if self.values.ndim == 1 else ~np.all(np.isfinite(self.values), axis=1)
        self.confidence[unknown] = 0.0

    def masked(self, min_confidence: float = 0.0) -> np.ndarray:
        out = self.values.copy()
        out[self.confidence <= min_confidence] = np.nan
        return out


class AttributeCurves(Mapping[str, AttributeCurve]):
    """Named curves that all share one :class:`FrameGrid`."""

    def __init__(self, grid: FrameGrid, curves: Mapping[str, AttributeCurve] | None = None):
        self.grid = grid
        self._c: dict[str, AttributeCurve] = {}
        for c in (curves or {}).values():
            self.add(c)

    def add(self, curve: AttributeCurve) -> None:
        if curve.grid != self.grid:
            raise GridMismatchError(f"curve {curve.name!r} is on {curve.grid}, collection on {self.grid}")
        self._c[curve.name] = curve

    def __getitem__(self, k: str) -> AttributeCurve:
        return self._c[k]

    def __iter__(self) -> Iterator[str]:
        return iter(self._c)

    def __len__(self) -> int:
        return len(self._c)


@dataclass
class Event:
    """A discrete ornament / onset event on the frame grid."""

    kind: str  # e.g. "scoop", "fall", "kkeokki", "glide", "onset"
    start: int
    end: int
    magnitude: float  # cents, or event-specific unit
    confidence: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Representation:
    """The three-layer representation of one sung phrase."""

    grid: FrameGrid
    curves: AttributeCurves
    recording_id: str
    provenance: Provenance
    events: list[Event] = field(default_factory=list)
    singer: SingerVector | None = None  # M4
    env: np.ndarray | None = None  # M4
    residual: np.ndarray | None = None  # (T, R), M4
    quality: dict[str, Any] = field(default_factory=dict)  # frontend flags
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.grid.require_same(self.curves.grid)
        if self.residual is not None:
            self.grid.check_array(self.residual, "residual")
        if self.singer is not None and self.singer.provenance is not self.provenance:
            raise ValueError("singer vector provenance must match the representation's provenance")


class Consistency(str, Enum):
    STYLE_OR_HABIT = "style_or_habit"  # consistent across takes
    ERROR = "error"  # inconsistent across takes
    UNDETERMINED = "undetermined"  # fewer than two takes / not enough data


@dataclass
class Span:
    start: int  # user frame index (inclusive)
    end: int  # exclusive
    syllables: tuple[str, ...] = ()
    reason: str = ""  # for "cannot judge" spans: why

    def seconds(self, grid: FrameGrid) -> tuple[float, float]:
        return float(grid.time_of(self.start)), float(grid.time_of(self.end))


@dataclass
class ExplanationItem:
    category: str  # "pitch" | "rhythm" | "ornament" | "phonation" | "diction"
    attribute: str  # e.g. "intonation_offset", "onset_timing", "vibrato_extent", "scoop"
    spans: list[Span]
    magnitude: float  # signed, in ``unit``
    unit: str
    confidence: float
    consistency: Consistency = Consistency.UNDETERMINED
    delta: np.ndarray | None = None  # per-frame Δ on the user grid (NaN outside spans)
    audibility: float | None = None  # M5; None = not computed
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Explanation:
    grid: FrameGrid  # user grid
    warp: np.ndarray  # τ(t): target frame (float) for each user frame
    transposition_cents: float
    items: list[ExplanationItem]
    cannot_judge: list[Span]
    n_takes: int
    meta: dict[str, Any] = field(default_factory=dict)

    def by_category(self, category: str) -> list[ExplanationItem]:
        return [i for i in self.items if i.category == category]

"""The shared time axis.

Every frame-rate quantity in gyeol lives on a :class:`FrameGrid`.  Frame
``i`` is *centred* on sample ``i * hop``.  Containers carry their grid and
operations that combine two curves call :meth:`FrameGrid.require_same`, so
time axes can never silently drift (e.g. 16 kHz/160 vs 44.1 kHz/512).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_SR = 44100
DEFAULT_HOP = 512


class GridMismatchError(ValueError):
    """Raised when two objects on different frame grids are combined."""


@dataclass(frozen=True)
class FrameGrid:
    sr: int = DEFAULT_SR
    hop: int = DEFAULT_HOP
    n_frames: int = 0

    def __post_init__(self) -> None:
        if self.sr <= 0 or self.hop <= 0 or self.n_frames < 0:
            raise ValueError(f"invalid FrameGrid({self.sr}, {self.hop}, {self.n_frames})")

    # -- construction --------------------------------------------------------
    @classmethod
    def for_samples(cls, n_samples: int, sr: int = DEFAULT_SR, hop: int = DEFAULT_HOP) -> "FrameGrid":
        """Grid covering ``n_samples`` (frame centres 0, hop, 2·hop, …)."""
        return cls(sr=sr, hop=hop, n_frames=1 + max(0, n_samples) // hop)

    @classmethod
    def for_duration(cls, seconds: float, sr: int = DEFAULT_SR, hop: int = DEFAULT_HOP) -> "FrameGrid":
        return cls.for_samples(int(round(seconds * sr)), sr, hop)

    # -- geometry ------------------------------------------------------------
    @property
    def rate(self) -> float:
        """Frames per second."""
        return self.sr / self.hop

    @property
    def hop_seconds(self) -> float:
        return self.hop / self.sr

    @property
    def duration(self) -> float:
        return (self.n_frames - 1) * self.hop_seconds if self.n_frames else 0.0

    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) * self.hop_seconds

    def frame_of(self, t: float | np.ndarray) -> np.ndarray | int:
        """Nearest frame index for time(s) in seconds (clipped to the grid)."""
        idx = np.clip(np.round(np.asarray(t) / self.hop_seconds).astype(int), 0, max(0, self.n_frames - 1))
        return int(idx) if idx.ndim == 0 else idx

    def time_of(self, i: int | np.ndarray) -> float | np.ndarray:
        return np.asarray(i) * self.hop_seconds

    def with_frames(self, n_frames: int) -> "FrameGrid":
        return FrameGrid(self.sr, self.hop, n_frames)

    # -- safety --------------------------------------------------------------
    def same_axis(self, other: "FrameGrid") -> bool:
        """Same sample rate and hop (frame counts may differ)."""
        return self.sr == other.sr and self.hop == other.hop

    def require_same(self, other: "FrameGrid", *, frames: bool = True) -> None:
        if not self.same_axis(other) or (frames and self.n_frames != other.n_frames):
            raise GridMismatchError(f"frame grids differ: {self} vs {other}")

    def check_array(self, a: np.ndarray, name: str = "array") -> None:
        if a.shape[0] != self.n_frames:
            raise GridMismatchError(f"{name} has {a.shape[0]} frames, grid has {self.n_frames}")


def project(src_times: np.ndarray, values: np.ndarray, grid: FrameGrid, *, kind: str = "linear", max_gap: float | None = None) -> np.ndarray:
    """Project a curve sampled at ``src_times`` (seconds) onto ``grid``.

    ``values`` is (N,) or (N, D).  NaN source samples are not bridged:
    ``kind="linear"`` interpolates only between finite neighbours that are at
    most ``max_gap`` seconds apart (default: 2 source periods); other grid
    frames get NaN.  ``kind="nearest"`` takes the nearest source sample.
    This is the only sanctioned way for an estimator with its own hop to put
    results on the shared grid.
    """
    src_times = np.asarray(src_times, dtype=float)
    v = np.asarray(values, dtype=float)
    t = grid.times()
    if v.ndim == 1:
        return _project_1d(src_times, v, t, kind, max_gap)
    return np.stack([_project_1d(src_times, v[:, j], t, kind, max_gap) for j in range(v.shape[1])], axis=1)


def _project_1d(ts: np.ndarray, v: np.ndarray, t: np.ndarray, kind: str, max_gap: float | None) -> np.ndarray:
    out = np.full(len(t), np.nan)
    if len(ts) == 0:
        return out
    step = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0
    gap = 2.0 * step if max_gap is None else max_gap
    if kind == "nearest":
        idx = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1) if len(ts) > 1 else np.zeros(len(t), int)
        if len(ts) > 1:
            left = idx - 1
            idx = np.where(np.abs(t - ts[left]) <= np.abs(ts[idx] - t), left, idx)
        near = np.abs(ts[idx] - t) <= gap
        out[near] = v[idx][near]
        return out
    if kind != "linear":
        raise ValueError(f"unknown kind {kind!r}")
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return out
    tf, vf = ts[ok], v[ok]
    j = np.searchsorted(tf, t)
    lo = np.clip(j - 1, 0, len(tf) - 1)
    hi = np.clip(j, 0, len(tf) - 1)
    exact = np.abs(tf[lo] - t) < 1e-9
    span = tf[hi] - tf[lo]
    w = np.where(span > 0, (t - tf[lo]) / np.where(span > 0, span, 1.0), 0.0)
    inside = (t >= tf[lo] - 1e-9) & (t <= tf[hi] + 1e-9) & (span <= gap)
    val = vf[lo] + np.clip(w, 0, 1) * (vf[hi] - vf[lo])
    # at the ends of a finite run, hold the edge value for half a source step
    edge = (lo == hi) & (np.abs(t - tf[lo]) <= 0.5 * step + 1e-9)
    sel = inside | exact | edge
    out[sel] = val[sel]
    return out

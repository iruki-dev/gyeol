"""The vocal representation container.

A :class:`VocalRepresentation` is the output of the engine.  It holds

* ``tracks``   – frame-level dimensions of T_voice, each with its own validity
  mask and confidence (the *validity mask is mandatory*: a value outside its
  validity domain is missing, never a number to be trusted);
* ``notes``    – note-level descriptors (vibrato, glottal parameters,
  irregularity, onset) and aggregates;
* ``context``  – Korean phonetic-context tokens (T_context);
* ``residual`` – the optional channel-adversarial residual embedding z;
* ``nuisance`` – the nuisance side channel N̂.  It is stored *outside* T and is
  only used to drive the validity masks.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Track:
    """One representation dimension sampled on a regular time grid."""

    name: str
    values: np.ndarray  # (T,) or (T, D)
    valid: np.ndarray  # (T,) bool
    rate: float  # frames per second
    unit: str
    confidence: np.ndarray | None = None  # (T,) in [0, 1]
    #: fraction of frames failing each validity condition (explainability)
    invalid_reasons: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        self.valid = np.asarray(self.valid, dtype=bool)
        if self.valid.shape[0] != self.values.shape[0]:
            raise ValueError(f"{self.name}: valid mask length does not match values")
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=float)

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.values.shape[0]) / self.rate

    def masked(self) -> np.ndarray:
        """Values with invalid frames replaced by NaN."""
        out = self.values.copy()
        out[~self.valid] = np.nan
        return out

    def valid_values(self, start: float | None = None, end: float | None = None) -> np.ndarray:
        t = self.times
        sel = self.valid.copy()
        if start is not None:
            sel &= t >= start
        if end is not None:
            sel &= t < end
        v = self.values[sel]
        finite = np.isfinite(v) if v.ndim == 1 else np.all(np.isfinite(v), axis=1)
        return v[finite]

    def coverage(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0


@dataclass
class Note:
    """A note / sustained segment with its note-level descriptors."""

    start: float
    end: float
    features: dict[str, float] = field(default_factory=dict)
    valid: dict[str, bool] = field(default_factory=dict)
    syllable: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


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


@dataclass
class NuisanceReport:
    """Nuisance side channel N̂ (never part of T_voice)."""

    snr_db: float | None = None
    noise_floor_db: float | None = None
    effective_bandwidth_hz: float | None = None
    native_sample_rate: int | None = None
    codec_suspected: bool = False
    codec_evidence: list[str] = field(default_factory=list)
    clipping_fraction: float = 0.0
    agc_suspected: bool = False
    noise_gate_suspected: bool = False
    t60_s: float | None = None
    drr_db: float | None = None
    separation_used: str | None = None
    separation_agreement_db: float | None = None
    device_eq_applied: bool = False
    device_highpass_hz: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VocalRepresentation:
    sample_rate: int
    hop_seconds: float
    duration: float
    tracks: dict[str, Track]
    notes: list[Note]
    nuisance: NuisanceReport
    context: ContextTokens | None = None
    residual: Track | None = None
    config: dict[str, Any] = field(default_factory=dict)
    version: str = "0"

    # -- convenience ---------------------------------------------------------

    def __getitem__(self, name: str) -> Track:
        return self.tracks[name]

    @property
    def times(self) -> np.ndarray:
        return np.arange(round(self.duration / self.hop_seconds) + 1) * self.hop_seconds

    def coverage(self) -> dict[str, float]:
        """Fraction of frames where each dimension is valid."""
        return {k: t.coverage() for k, t in self.tracks.items()}

    def summary(self) -> dict[str, Any]:
        """Compact JSON-friendly overview: medians of valid values + coverage."""
        dims: dict[str, Any] = {}
        for name, tr in self.tracks.items():
            if tr.values.ndim != 1:
                dims[name] = {"coverage": round(tr.coverage(), 3), "unit": tr.unit}
                continue
            v = tr.valid_values()
            entry: dict[str, Any] = {"unit": tr.unit, "coverage": round(tr.coverage(), 3)}
            if v.size:
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                entry.update(median=round(float(med), 3), iqr=round(float(q3 - q1), 3))
            if tr.invalid_reasons:
                entry["invalid_reasons"] = {k: round(r, 3) for k, r in tr.invalid_reasons.items() if r > 0}
            dims[name] = entry
        return {
            "duration_s": round(self.duration, 3),
            "n_notes": len(self.notes),
            "dimensions": dims,
            "nuisance": self.nuisance.to_dict(),
            "context": None if self.context is None else self.context.alignment_source,
            "residual": None if self.residual is None else list(self.residual.values.shape),
        }

    # -- serialisation -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save as a compressed ``.npz`` (arrays) with a JSON metadata entry."""
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, Any] = {
            "sample_rate": self.sample_rate,
            "hop_seconds": self.hop_seconds,
            "duration": self.duration,
            "version": self.version,
            "config": self.config,
            "nuisance": self.nuisance.to_dict(),
            "notes": [asdict(n) for n in self.notes],
            "tracks": {},
        }
        for name, tr in list(self.tracks.items()) + ([("__residual__", self.residual)] if self.residual else []):
            arrays[f"{name}/values"] = tr.values
            arrays[f"{name}/valid"] = tr.valid
            if tr.confidence is not None:
                arrays[f"{name}/confidence"] = tr.confidence
            meta["tracks"][name] = {
                "rate": tr.rate,
                "unit": tr.unit,
                "invalid_reasons": tr.invalid_reasons,
                "has_confidence": tr.confidence is not None,
            }
        if self.context is not None:
            c = self.context
            for key in ("syllable_index", "laryngeal_class", "time_since_onset_ms", "syllable_position", "phrase_initial"):
                arrays[f"__context__/{key}"] = getattr(c, key)
            meta["context"] = {"rate": c.rate, "syllables": c.syllables, "alignment_source": c.alignment_source}
        buf = json.dumps(meta, ensure_ascii=False, default=_json_default).encode("utf-8")
        arrays["__meta__"] = np.frombuffer(buf, dtype=np.uint8)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "VocalRepresentation":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["__meta__"]).decode("utf-8"))
            tracks: dict[str, Track] = {}
            residual = None
            for name, info in meta["tracks"].items():
                tr = Track(
                    name=name,
                    values=z[f"{name}/values"],
                    valid=z[f"{name}/valid"],
                    rate=info["rate"],
                    unit=info["unit"],
                    confidence=z[f"{name}/confidence"] if info["has_confidence"] else None,
                    invalid_reasons=info.get("invalid_reasons", {}),
                )
                if name == "__residual__":
                    tr.name = "residual"
                    residual = tr
                else:
                    tracks[name] = tr
            context = None
            if "context" in meta:
                cm = meta["context"]
                context = ContextTokens(
                    rate=cm["rate"],
                    syllables=cm["syllables"],
                    alignment_source=cm["alignment_source"],
                    **{k: z[f"__context__/{k}"] for k in ("syllable_index", "laryngeal_class", "time_since_onset_ms", "syllable_position", "phrase_initial")},
                )
        return cls(
            sample_rate=meta["sample_rate"],
            hop_seconds=meta["hop_seconds"],
            duration=meta["duration"],
            tracks=tracks,
            notes=[Note(**n) for n in meta["notes"]],
            nuisance=NuisanceReport(**meta["nuisance"]),
            context=context,
            residual=residual,
            config=meta.get("config", {}),
            version=meta.get("version", "0"),
        )

    def to_json(self) -> str:
        return json.dumps(self.summary(), ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, io.IOBase):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")

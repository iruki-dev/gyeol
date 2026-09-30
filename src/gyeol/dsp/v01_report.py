"""v0.1 nuisance report container (kept for the v0.1 baseline estimators)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any



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

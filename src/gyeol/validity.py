"""Per-dimension validity masks.

"The same values for the same person regardless of recording conditions"
only holds *inside a measured validity domain*.  Every dimension therefore
carries a boolean mask; values outside it are missing, never numbers.

:class:`ValidityPolicy` holds every threshold in one place.  Thresholds that
come from the literature cite it; those the research marks as UNVERIFIED
HYPOTHESIS are labelled *provisional* and are expected to be replaced by
operating thresholds measured with :mod:`gyeol.verification.thresholds`
(error-vs-nuisance curves crossing the MDC, research §5.iii).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class ValidityPolicy:
    # --- literature-backed -------------------------------------------------
    #: CPPS / aperiodicity / perturbation need SNR ≥ 30 dB (Deliyski, Shaw &
    #: Evans, J Voice 2005: acceptable > 30 dB, recommended > 42 dB)
    min_snr_db: float = 30.0
    #: jitter/shimmer need ≥ 19 kHz sampling (Deliyski et al., LPV 2005)
    perturbation_min_sr: int = 19000
    #: SPR / tilt / formants need ≥ 5 kHz bandwidth; 16 kHz analysis of
    #: aperiodicity needs ≥ 8 kHz (spec "global conventions")
    min_bandwidth_spectral_hz: float = 5000.0
    min_bandwidth_aperiodicity_hz: float = 7500.0
    #: vibrato needs ≥ 2 cycles
    vibrato_min_cycles: float = 2.0
    #: HNR / aperiodicity / perturbation are codec-fragile (J Voice 2020 pilot;
    #: Weerathunge et al. 2021) — reject when a lossy codec is suspected
    reject_lossy_for_noise_measures: bool = True
    # --- provisional (UNVERIFIED HYPOTHESIS; calibrate per §5.iii) -----------
    cpps_max_f0: float = 700.0
    glottal_max_f0: float = 500.0
    #: a formant is treated as resolvable only if F_n ≥ ratio · f0
    formant_min_f0_ratio: float = 1.5
    #: H1*–H2* invalid when F1 lies within this many B1 of f0 or 2·f0
    h1h2_f1_bandwidths: float = 1.0
    min_f0_confidence: float = 0.3
    min_separation_agreement_db: float = 5.0
    max_clipping_fraction: float = 1e-3
    perturbation_max_vibrato_cents: float = 15.0
    perturbation_min_seconds: float = 0.25
    nasality_min_p0_index: int = 3
    #: frames after an aspirated / fortis onset excluded from pitch-accuracy
    #: and phonation aggregates (window to be set by §5.iii)
    context_exclusion_ms: dict[str, float] = field(default_factory=lambda: {"aspirated": 60.0, "fortis": 80.0})
    #: confidence multiplier for EQ-sensitive dimensions without device EQ
    no_eq_confidence: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


class MaskBuilder:
    """AND-combines named conditions and records how often each one fails."""

    def __init__(self, n: int):
        self.valid = np.ones(n, dtype=bool)
        self.reasons: dict[str, float] = {}
        self.n = n

    def require(self, name: str, ok: np.ndarray | bool) -> "MaskBuilder":
        ok_arr = np.broadcast_to(np.asarray(ok, dtype=bool), (self.n,))
        fail = ~ok_arr
        if fail.any():
            self.reasons[name] = self.reasons.get(name, 0.0) + float(fail.mean())
        self.valid &= ok_arr
        return self

    def build(self) -> tuple[np.ndarray, dict[str, float]]:
        return self.valid.copy(), dict(self.reasons)


def finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.isfinite(x) if x.ndim == 1 else np.all(np.isfinite(x), axis=1)

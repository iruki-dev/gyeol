"""Formant-corrected harmonic differences and A1–P0 nasality."""

from __future__ import annotations

import numpy as np

from .formants import hawks_miller_bandwidth, iseli_alwan_correction
from .spectral import Harmonics, harmonic_near


def corrected_harmonic_measures(
    harm: Harmonics,
    f0: np.ndarray,
    formants: np.ndarray,
    sr: int,
    measured_bw: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """H1*–H2*, H2*–H4*, H1*–A1*, H1*–A3* (dB) with Iseli–Alwan correction.

    Correction sets follow VoiceSauce: H1, H2, H4, A1 and A2 are corrected for
    F1 and F2; A3 for F1–F3.  Bandwidths come from the Hawks & Miller model
    unless ``measured_bw`` is given.
    """
    t_n = len(f0)
    nan = np.full(t_n, np.nan)
    if not np.isfinite(f0).any():
        return {k: nan.copy() for k in ("h1h2c", "h2h4c", "h1a1c", "h1a3c", "h1h2", "a1", "a3")}
    F = formants[:, :3]
    B = measured_bw[:, :3] if measured_bw is not None else hawks_miller_bandwidth(F, f0[:, None])

    def corr(freq: np.ndarray, n_formants: int) -> np.ndarray:
        total = np.zeros(t_n)
        for i in range(n_formants):
            total += iseli_alwan_correction(freq, F[:, i], B[:, i], sr)
        return total

    h1 = harm.amp_db[:, 0]
    h2 = harm.amp_db[:, 1]
    h4 = harm.amp_db[:, 3]
    a1, _ = harmonic_near(harm, F[:, 0], f0)
    a3, _ = harmonic_near(harm, F[:, 2], f0)
    a1_f = F[:, 0]
    a3_f = F[:, 2]
    h1c = h1 - corr(f0, 2)
    h2c = h2 - corr(2 * f0, 2)
    h4c = h4 - corr(4 * f0, 2)
    a1c = a1 - corr(a1_f, 2)
    a3c = a3 - corr(a3_f, 3)
    return {
        "h1h2c": h1c - h2c,
        "h2h4c": h2c - h4c,
        "h1a1c": h1c - a1c,
        "h1a3c": h1c - a3c,
        "h1h2": h1 - h2,  # uncorrected, kept for diagnostics
        "a1": a1,
        "a3": a3,
    }


def a1_p0(harm: Harmonics, f0: np.ndarray, f1: np.ndarray, p0_hz: float = 250.0) -> tuple[np.ndarray, np.ndarray]:
    """A1–P0 nasality (Chen, JASA 1997).

    A1 is the strongest harmonic near F1; P0 the strongest harmonic near the
    ~250 Hz nasal pole.  Returns (a1_p0 dB, P0 harmonic index) so the
    validity layer can reject frames where P0 falls on H1/H2.
    """
    t_n = len(f0)
    a1, _ = harmonic_near(harm, f1, f0)
    p0, p0_idx = harmonic_near(harm, np.full(t_n, p0_hz), f0, tolerance=0.4)
    return a1 - p0, p0_idx


def resonance_tuning(f0: np.ndarray, f1: np.ndarray, b1: np.ndarray) -> dict[str, np.ndarray]:
    """R1:f0 and R1:2f0 tuning indices |F1 − k·f0| / B1 (smaller = tuned)."""
    b1 = np.maximum(b1, 1.0)
    return {"r1_f0": np.abs(f1 - f0) / b1, "r1_2f0": np.abs(f1 - 2 * f0) / b1}

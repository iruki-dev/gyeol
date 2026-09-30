"""v0.1 DSP features as *gated weak labels* on the shared frame grid.

These are auxiliary targets / baselines for the learned heads (M3), never
coaching signals by themselves.  Two v0.1 defects are fixed here:

* **Formants swap under noise** (defect 7): formants are estimated with
  several LPC orders; a formant is reported only where the orders agree
  (relative spread ≤ ``max_order_spread``) and the local SNR is at least
  ``min_snr_db``.  Otherwise it is *missing*, never a swapped value.
* **High-f0 F1 mis-tracking inflates H1*–H2*** (defect 3): a formant is
  resolvable only when F_n ≥ ``min_harmonics`` · f0 (two harmonics below it,
  v0.1 used 1.5), and H1*–H2* additionally requires F1 to be ≥ one bandwidth
  from f0 and 2·f0.  Close vowels at moderate f0 therefore become invalid
  instead of inflated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.containers import AttributeCurve
from ..core.grid import FrameGrid, project
from .base import n_frames, resample
from .cepstral import cpps
from .formants import hawks_miller_bandwidth, lpc_formants
from .source import corrected_harmonic_measures
from .spectral import harmonic_peaks, spectrogram

WL_SR = 16000
WL_HOP = 160


@dataclass
class WeakLabelConfig:
    lpc_orders: tuple[int, ...] = (10, 12, 14)
    max_order_spread: float = 0.06
    min_snr_db: float = 30.0
    min_harmonics: float = 2.0
    f1_bandwidths: float = 1.0


def robust_formants(x16: np.ndarray, f0: np.ndarray, frame_snr_db: np.ndarray, cfg: WeakLabelConfig) -> tuple[np.ndarray, np.ndarray]:
    """(F1–F3 in Hz with NaN where unreliable, validity mask) at 16 kHz / 10 ms."""
    n = len(f0)
    voiced = np.isfinite(f0)
    est = np.stack([lpc_formants(x16, WL_SR, WL_HOP, n, voiced, n_poles=p, method="wlp")[0][:, :3] for p in cfg.lpc_orders])
    with np.errstate(all="ignore"):
        med = np.nanmedian(est, axis=0)
        spread = (np.nanmax(est, axis=0) - np.nanmin(est, axis=0)) / med
    ok = np.isfinite(med) & (spread <= cfg.max_order_spread)
    ok &= (frame_snr_db >= cfg.min_snr_db)[:, None]
    ok &= med >= cfg.min_harmonics * np.where(voiced, f0, np.inf)[:, None]
    return np.where(ok, med, np.nan), ok


def weak_labels(x: np.ndarray, sr: int, grid: FrameGrid, f0_hz: np.ndarray, frame_snr_db: np.ndarray,
                config: WeakLabelConfig | None = None) -> dict[str, AttributeCurve]:
    """Gated v0.1 labels (F1–F3, H1*–H2*, CPPS) projected onto ``grid``.

    ``f0_hz`` and ``frame_snr_db`` are on ``grid`` (from the pitch consensus and
    the frontend).  Values outside their validity conditions are NaN with
    confidence 0.
    """
    cfg = config or WeakLabelConfig()
    x16 = resample(np.asarray(x, float), sr, WL_SR)
    n = n_frames(len(x16), WL_HOP)
    t16 = np.arange(n) * WL_HOP / WL_SR
    gt = grid.times()
    f0 = np.interp(t16, gt, np.nan_to_num(f0_hz, nan=0.0))
    near_unvoiced = np.interp(t16, gt, (~np.isfinite(f0_hz)).astype(float)) > 0.0
    f0 = np.where((f0 > 0) & ~near_unvoiced, f0, np.nan)
    snr = np.interp(t16, gt, np.nan_to_num(frame_snr_db, nan=-99.0))
    F, ok = robust_formants(x16, f0, snr, cfg)
    spec = spectrogram(x16, WL_SR, WL_HOP, n)
    harm = harmonic_peaks(spec, f0)
    hm = corrected_harmonic_measures(harm, f0, np.column_stack([F, np.full(n, np.nan)]), WL_SR)
    B1 = hawks_miller_bandwidth(F[:, 0], f0)
    near = (np.abs(F[:, 0] - f0) <= cfg.f1_bandwidths * B1) | (np.abs(F[:, 0] - 2 * f0) <= cfg.f1_bandwidths * B1)
    h_ok = ok[:, 0] & ok[:, 1] & ~near & np.isfinite(hm["h1h2c"])
    h1h2c = np.where(h_ok, hm["h1h2c"], np.nan)
    cp = cpps(x16, WL_SR, WL_HOP, n, f0)
    cp = np.where(np.isfinite(f0) & (snr >= cfg.min_snr_db), cp, np.nan)
    out: dict[str, AttributeCurve] = {}
    for name, vals, unit in (("f1_hz", F[:, 0], "Hz"), ("f2_hz", F[:, 1], "Hz"), ("f3_hz", F[:, 2], "Hz"),
                             ("h1h2c_db", h1h2c, "dB"), ("cpps_db", cp, "dB")):
        v = project(t16, vals, grid)
        out[name] = AttributeCurve(f"weak_{name}", v, np.isfinite(v).astype(float), grid, unit, meta={"source": "v0.1 DSP (gated)"})
    return out

"""Environment / nuisance estimators for the side channel.

v0.1's T60 heuristic silently returned ``None`` or wrong values under noise
(defect 5).  Here every estimate returns a :class:`Result`: a value is only
``OK`` when enough clean free decays were observed, otherwise the status is
``UNRELIABLE`` / ``FAILED`` with the reason.  Learned env encoders replace
these in M4; DRR / C50 need a trained estimator and are reported as
unavailable until then.
"""

from __future__ import annotations

import numpy as np

from ..core.status import Result

EPS = 1e-12


def energy_envelope_db(x: np.ndarray, sr: int, hop_s: float = 0.005, win_s: float = 0.02) -> np.ndarray:
    hop, win = int(hop_s * sr), int(win_s * sr)
    n = 1 + max(0, len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return 10 * np.log10(np.mean(np.asarray(x, float)[idx] ** 2, axis=1) + EPS)


def estimate_t60(x: np.ndarray, sr: int, *, min_decays: int = 2, min_dynamic_db: float = 40.0, hop_s: float = 0.005) -> Result[float]:
    """T60 from free decays after offsets (line fit from −10 to −30 dB, past the direct sound).

    A decay counts only if the level falls at least ``min_dynamic_db`` from
    the offset peak to the stationary floor, so the −30 dB end of the fit is
    still ≥ 10 dB above noise.  Too few such decays → not OK.

    This is a heuristic: on synthetic rooms it over-estimates by roughly
    15–30 % (the source's own release adds to the decay).  M4's learned env
    encoder is meant to replace it.
    """
    db = energy_envelope_db(x, sr, hop_s)
    if len(db) < 50:
        return Result.failure("recording too short for decay analysis")
    floor = float(np.percentile(db, 10))
    peak = float(np.percentile(db, 99))
    if peak - floor < min_dynamic_db:
        return Result.unreliable(float("nan"), f"dynamic range {peak - floor:.0f} dB < {min_dynamic_db:.0f} dB: noise masks the decays")
    fs = 1.0 / hop_s
    estimates = []
    half = int(0.05 * fs)
    i = half
    while i < len(db) - half:
        ref = db[i]
        is_peak = ref >= db[i - half : i + half + 1].max() and ref - floor >= min_dynamic_db
        if not is_peak:
            i += 1
            continue
        # follow the free decay while the level does not re-rise by > 3 dB
        j, run_min = i, ref
        limit = min(len(db), i + int(1.5 * fs))
        while j + 1 < limit and db[j + 1] <= run_min + 3.0 and db[j] > ref - 31:
            j += 1
            run_min = min(run_min, db[j])
        seg = db[i : j + 1]
        if seg.min() <= ref - 30:
            sel = np.flatnonzero((seg <= ref - 10) & (seg >= ref - 30))
            if sel.size >= 5:
                slope = np.polyfit(sel / fs, seg[sel], 1)[0]
                if slope < -5:
                    estimates.append(-60.0 / slope)
        i = max(j, i + 1)
    if len(estimates) < min_decays:
        return Result.unreliable(float("nan"), f"only {len(estimates)} clean free decays (need {min_decays})")
    return Result.success(float(np.median(estimates)))


def estimate_drr(x: np.ndarray, sr: int) -> Result[float]:
    return Result.unavailable("blind DRR needs a trained estimator (TODO(M4): env encoder)")

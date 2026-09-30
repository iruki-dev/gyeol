"""Time warp τ(t) between a user take and the target, from content only.

User and target share a clock (the user sings along with the track) up to
the route latency, which is removed beforehand (:mod:`gyeol.io.latency`).
The warp is therefore searched in a **band around identity**, and then
projected onto a **smooth, monotone** function:

1. banded DTW on content features (cosine distance, symmetric steps with a
   small penalty on non-diagonal moves);
2. per user frame, the mean matched target frame;
3. smoothing spline on τ(t) − t, then slope clipping to
   [``min_slope``, ``max_slope``] (monotone by construction) and band
   clipping.

Soft-DTW refinement is optional in the design brief and not implemented.

τ maps each *user* frame to a (fractional) *target* frame.  τ'(t) > 1 means
the user moves through the target faster than written (rushing); < 1 means
dragging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_smoothing_spline

from ..core.grid import FrameGrid
from ..core.status import Result


@dataclass
class WarpConfig:
    band_seconds: float = 0.6
    off_diagonal_penalty: float = 0.15  # × median local cost
    smoothing_seconds: float = 0.04  # spline smoothness scale
    #: silence carries no content, so slopes across gaps may be extreme
    min_slope: float = 0.25
    max_slope: float = 4.0
    contrast_for_full_confidence: float = 0.4


@dataclass
class Warp:
    tau: np.ndarray  # (T_user,) target frame index (float)
    confidence: np.ndarray  # (T_user,) 0..1: match contrast against the band
    local_cost: np.ndarray  # (T_user,)
    band_frames: int

    def slope(self) -> np.ndarray:
        return np.gradient(self.tau)

    def inverse(self, target_frame: float) -> float:
        """User frame at which the target frame is reached (first crossing)."""
        idx = np.flatnonzero(self.tau >= target_frame)
        if idx.size == 0:
            return float(len(self.tau) - 1)
        i = int(idx[0])
        if i == 0:
            return 0.0
        t0, t1 = self.tau[i - 1], self.tau[i]
        return float(i - 1 + (target_frame - t0) / max(t1 - t0, 1e-9))


def _cosine_cost(U: np.ndarray, T: np.ndarray) -> np.ndarray:
    Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    return 1.0 - Un @ Tn.T


def banded_dtw(U: np.ndarray, T: np.ndarray, band: int, penalty: float) -> tuple[np.ndarray, np.ndarray]:
    """Path (list of (u, t)) and local cost along it; |u − t| ≤ band."""
    nu, nt = len(U), len(T)
    C = np.full((nu, nt), np.inf)
    for u in range(nu):
        lo, hi = max(0, u - band), min(nt, u + band + 1)
        if lo < hi:
            C[u, lo:hi] = _cosine_cost(U[u : u + 1], T[lo:hi])[0]
    med = np.median(C[np.isfinite(C)]) if np.isfinite(C).any() else 1.0
    pen = penalty * med
    D = np.full((nu + 1, nt + 1), np.inf)
    D[0, 0] = 0.0
    move = np.zeros((nu, nt), dtype=np.int8)  # 0 diag, 1 up (u-1), 2 left (t-1)
    for u in range(1, nu + 1):
        lo, hi = max(1, u - band), min(nt, u + band)
        for t in range(lo, hi + 1):
            c = C[u - 1, t - 1]
            if not np.isfinite(c):
                continue
            d, up, left = D[u - 1, t - 1], D[u - 1, t] + pen, D[u, t - 1] + pen
            if d <= up and d <= left:
                D[u, t], move[u - 1, t - 1] = c + d, 0
            elif up <= left:
                D[u, t], move[u - 1, t - 1] = c + up, 1
            else:
                D[u, t], move[u - 1, t - 1] = c + left, 2
    # end at the last user frame, best reachable target frame inside the band
    tail = D[nu, max(1, nu - band) : min(nt, nu + band) + 1]
    t = max(1, nu - band) + int(np.argmin(tail))
    u = nu
    path = []
    while u > 0 and t > 0:
        path.append((u - 1, t - 1))
        m = move[u - 1, t - 1]
        if m == 0:
            u, t = u - 1, t - 1
        elif m == 1:
            u -= 1
        else:
            t -= 1
    path.reverse()
    p = np.array(path)
    return p, C[p[:, 0], p[:, 1]]


def band_mean_cost(U: np.ndarray, T: np.ndarray, band: int) -> np.ndarray:
    """Mean cost of each user frame against every target frame in its band."""
    nu, nt = len(U), len(T)
    out = np.zeros(nu)
    for u in range(nu):
        lo, hi = max(0, u - band), min(nt, u + band + 1)
        out[u] = _cosine_cost(U[u : u + 1], T[lo:hi])[0].mean() if lo < hi else 1.0
    return out


def estimate_warp(user_content: np.ndarray, target_content: np.ndarray, grid: FrameGrid, config: WarpConfig | None = None) -> Result[Warp]:
    """Warp from user frames to target frames.  Both feature matrices must be on
    grids with the same sample rate and hop (checked by the caller)."""
    cfg = config or WarpConfig()
    nu = len(user_content)
    if nu < 5 or len(target_content) < 5:
        return Result.failure("sequences too short to align")
    band = max(2, int(round(cfg.band_seconds * grid.rate)))
    if abs(nu - len(target_content)) > band:
        return Result.failure(f"length difference exceeds the alignment band ({band} frames); remove latency first")
    path, cost = banded_dtw(np.asarray(user_content), np.asarray(target_content), band, cfg.off_diagonal_penalty)
    tau_raw = np.zeros(nu)
    lc = np.zeros(nu)
    cnt = np.zeros(nu)
    np.add.at(tau_raw, path[:, 0], path[:, 1])
    np.add.at(lc, path[:, 0], cost)
    np.add.at(cnt, path[:, 0], 1)
    covered = cnt > 0
    tau_raw[covered] /= cnt[covered]
    lc[covered] /= cnt[covered]
    u = np.arange(nu, dtype=float)
    if not covered.all():
        tau_raw = np.interp(u, u[covered], tau_raw[covered])
        lc = np.interp(u, u[covered], lc[covered])
    # smoothing spline on the deviation from identity
    lam = (cfg.smoothing_seconds * grid.rate) ** 3
    dev = make_smoothing_spline(u, tau_raw - u, lam=lam)(u)
    tau = u + dev
    # monotone, slope-bounded, inside the band
    out = np.empty(nu)
    out[0] = np.clip(tau[0], 0, band)
    for i in range(1, nu):
        step = np.clip(tau[i] - out[i - 1], cfg.min_slope, cfg.max_slope)
        out[i] = np.clip(out[i - 1] + step, i - band, i + band)
    out = np.clip(out, 0, len(target_content) - 1)
    # confidence = contrast of the chosen match against the alternatives in
    # the band: a frame is reliably aligned when its match is clearly better
    # than a random target frame nearby, even if the absolute cost is high
    # (e.g. the same vowel sung an octave lower)
    bm = band_mean_cost(np.asarray(user_content), np.asarray(target_content), band)
    contrast = (bm - lc) / (bm + 1e-9)
    conf = np.clip(contrast / cfg.contrast_for_full_confidence, 0.0, 1.0)
    return Result.success(Warp(out, conf, lc, band))


@dataclass
class OnsetDeviation:
    target_frame: int
    user_frame: float
    deviation_s: float  # + = user late (dragging), − = early (rushing)


def onset_deviations(warp: Warp, target_onsets: list[int], grid: FrameGrid) -> list[OnsetDeviation]:
    out = []
    for o in target_onsets:
        uf = warp.inverse(o)
        out.append(OnsetDeviation(o, uf, (uf - o) * grid.hop_seconds))
    return out


def tempo_ratio(warp: Warp, window_frames: int = 9) -> np.ndarray:
    """Smoothed local slope τ'(t): > 1 rushing, < 1 dragging."""
    s = warp.slope()
    if window_frames > 1:
        k = np.ones(window_frames) / window_frames
        s = np.convolve(np.pad(s, window_frames // 2, mode="edge"), k, mode="valid")[: len(s)]
    return s

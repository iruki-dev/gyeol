"""Low-level signal-processing helpers shared by every feature extractor.

All frame-based analyses in gyeol use *centred* frames: frame ``i`` is centred
on sample ``i * hop``.  Different estimators use different window lengths but
share the same frame centres, so their outputs line up on one time axis.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy import linalg, signal

EPS = 1e-12
A4_HZ = 440.0


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def hz_to_cents(f0: np.ndarray, ref: float = A4_HZ) -> np.ndarray:
    """Cents relative to ``ref`` (A4 by default).  NaN stays NaN."""
    f0 = np.asarray(f0, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1200.0 * np.log2(f0 / ref)


def cents_to_hz(cents: np.ndarray, ref: float = A4_HZ) -> np.ndarray:
    return ref * 2.0 ** (np.asarray(cents, dtype=float) / 1200.0)


def power_db(x: np.ndarray, floor_db: float = -200.0) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, 10.0 ** (floor_db / 10.0)))



# ---------------------------------------------------------------------------
# Basic signal handling
# ---------------------------------------------------------------------------


def to_mono(x: np.ndarray) -> np.ndarray:
    """Average channels.  Accepts (n,), (n, ch) or (ch, n) with ch <= 8."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x
    if x.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {x.shape}")
    if x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
        return x.mean(axis=0)
    return x.mean(axis=1)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(x, dtype=float)
    g = gcd(int(sr_in), int(sr_out))
    return signal.resample_poly(x, sr_out // g, sr_in // g)


def n_frames(n_samples: int, hop: int) -> int:
    return 1 + n_samples // hop


def frame(x: np.ndarray, win: int, hop: int, count: int | None = None) -> np.ndarray:
    """Return centred frames as a (count, win) view-backed array (zero padded)."""
    x = np.asarray(x, dtype=float)
    count = n_frames(len(x), hop) if count is None else count
    pad_left = win // 2
    needed = (count - 1) * hop + win
    pad_right = max(0, needed - pad_left - len(x))
    xp = np.pad(x, (pad_left, pad_right))
    frames = np.lib.stride_tricks.sliding_window_view(xp, win)[::hop]
    return frames[:count]


def frame_times(count: int, hop: int, sr: int) -> np.ndarray:
    return np.arange(count) * hop / sr


def parabolic_peak(y: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parabolic interpolation around integer peaks along the last axis.

    Returns (fractional index, interpolated value).
    """
    y = np.asarray(y)
    idx = np.asarray(idx)
    n = y.shape[-1]
    single = idx.ndim == y.ndim - 1
    if single:
        idx = idx[..., None]
    i0 = np.clip(idx - 1, 0, n - 1)
    i2 = np.clip(idx + 1, 0, n - 1)
    a = np.take_along_axis(y, i0, axis=-1)
    b = np.take_along_axis(y, idx, axis=-1)
    c = np.take_along_axis(y, i2, axis=-1)
    if single:
        a, b, c, idx = a[..., 0], b[..., 0], c[..., 0], idx[..., 0]
    denom = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(np.abs(denom) > EPS, 0.5 * (a - c) / denom, 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    value = b - 0.25 * (a - c) * delta
    return idx + delta, value


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """NaN-aware centred moving average."""
    x = np.asarray(x, dtype=float)
    width = min(int(width), len(x))
    if width <= 1:
        return x.copy()
    valid = np.isfinite(x)
    kernel = np.ones(width)
    num = np.convolve(np.where(valid, x, 0.0), kernel, mode="same")
    den = np.convolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index runs where ``mask`` is True."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------------------
# Linear prediction
# ---------------------------------------------------------------------------


def levinson(r: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Levinson–Durbin recursion, vectorised over leading axes.

    ``r`` has shape (..., >= order + 1).  Returns (a, err) with ``a`` of shape
    (..., order + 1), a[..., 0] == 1.
    """
    r = np.asarray(r, dtype=float)
    lead = r.shape[:-1]
    a = np.zeros(lead + (order + 1,))
    a[..., 0] = 1.0
    err = r[..., 0].copy() + EPS
    for i in range(1, order + 1):
        acc = r[..., i] + np.sum(a[..., 1:i] * r[..., i - 1 : 0 : -1], axis=-1)
        k = -acc / err
        a_prev = a.copy()
        a[..., i] = k
        a[..., 1:i] = a_prev[..., 1:i] + k[..., None] * a_prev[..., i - 1 : 0 : -1]
        err = err * (1.0 - k**2)
        err = np.maximum(err, EPS)
    return a, err


def autocorr(frames: np.ndarray, max_lag: int) -> np.ndarray:
    n = frames.shape[-1]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(frames, nfft, axis=-1)
    r = np.fft.irfft(np.abs(spec) ** 2, nfft, axis=-1)
    return r[..., : max_lag + 1]


def lpc(frames: np.ndarray, order: int) -> np.ndarray:
    """Autocorrelation-method LPC (frames should already be windowed)."""
    frames = np.atleast_2d(frames)
    r = autocorr(frames, order)
    r[..., 0] *= 1.0 + 1e-9  # white-noise correction for stability
    a, _ = levinson(r, order)
    return a


def burg(frames: np.ndarray, order: int) -> np.ndarray:
    """Burg's method, vectorised over frames.  Returns (n_frames, order+1)."""
    x = np.atleast_2d(np.asarray(frames, dtype=float))
    nf, n = x.shape
    a = np.zeros((nf, order + 1))
    a[:, 0] = 1.0
    f = x[:, 1:].copy()
    b = x[:, :-1].copy()
    for m in range(order):
        num = -2.0 * np.sum(f * b, axis=1)
        den = np.sum(f * f, axis=1) + np.sum(b * b, axis=1) + EPS
        k = num / den
        a_prev = a.copy()
        a[:, 1 : m + 2] = a_prev[:, 1 : m + 2] + k[:, None] * a_prev[:, m::-1][:, : m + 1]
        f_new = f + k[:, None] * b
        b_new = b + k[:, None] * f
        f = f_new[:, 1:]
        b = b_new[:, :-1]
        if f.shape[1] == 0:
            break
    return a


def weighted_lp(x: np.ndarray, order: int, weight: np.ndarray) -> np.ndarray:
    """Weighted linear prediction (covariance form) for one frame.

    Minimises sum_n w[n] * e[n]^2 (Ma, Kamp & Willems 1993; Magi et al. 2009).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    rows = np.stack([x[order - k : n - k] for k in range(order + 1)])  # (p+1, n-p)
    w = weight[order:n]
    cov = (rows * w) @ rows.T
    cov_pp = cov[1:, 1:] + np.eye(order) * (1e-9 * np.trace(cov[1:, 1:]) + EPS)
    try:
        coef = linalg.solve(cov_pp, -cov[1:, 0], assume_a="pos")
    except linalg.LinAlgError:
        coef = np.linalg.lstsq(cov_pp, -cov[1:, 0], rcond=None)[0]
    return np.concatenate([[1.0], coef])


def lpc_to_resonances(a: np.ndarray, sr: float) -> tuple[np.ndarray, np.ndarray]:
    """Pole frequencies (Hz) and 3-dB bandwidths (Hz) for one LPC polynomial."""
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]
    freqs = np.angle(roots) * sr / (2 * np.pi)
    bws = -np.log(np.maximum(np.abs(roots), EPS)) * sr / np.pi
    order = np.argsort(freqs)
    return freqs[order], bws[order]


def minimum_phase_fir(magnitude: np.ndarray, n_taps: int) -> np.ndarray:
    """Minimum-phase FIR from a one-sided magnitude response (homomorphic method).

    ``magnitude`` has length nfft//2 + 1 on a linear frequency grid.
    """
    nfft = 2 * (len(magnitude) - 1)
    log_mag = np.log(np.maximum(magnitude, 1e-8))
    cep = np.fft.irfft(log_mag, nfft)
    fold = np.zeros(nfft)
    fold[0] = cep[0]
    fold[1 : nfft // 2] = 2 * cep[1 : nfft // 2]
    fold[nfft // 2] = cep[nfft // 2]
    h = np.fft.irfft(np.exp(np.fft.rfft(fold, nfft)), nfft)
    h = h[:n_taps].copy()
    # Taper the last quarter so truncation does not ring.
    tail = n_taps // 4
    if tail > 1:
        h[-tail:] *= np.hanning(2 * tail)[tail:]
    return h


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant SDR in dB (Le Roux et al. 2019)."""
    n = min(len(estimate), len(reference))
    est = np.asarray(estimate[:n], dtype=float)
    ref = np.asarray(reference[:n], dtype=float)
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + EPS)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.dot(target, target) + EPS) / (np.dot(noise, noise) + EPS)))

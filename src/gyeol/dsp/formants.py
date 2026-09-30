"""Filter (resonance) group: formants, bandwidths and spectral envelope.

Formant estimators
------------------
``burg``   Burg LPC after resampling to 2 × max-formant (Praat-style).
``wlp``    Weighted linear prediction with short-time-energy weighting
           (Ma, Kamp & Willems 1993; Magi et al. 2009).  Down-weights the
           main-excitation instants that bias LPC toward harmonics.
``sweep``  High-f0 estimator: harmonic peak samples (freq, amplitude) are
           pooled over a window spanning at least one vibrato cycle, so the
           vocal-tract transfer function is sampled more densely than one
           frame's harmonic comb.  The pooled samples are interpolated to a
           dense envelope and fitted with an all-pole model.
           *UNVERIFIED HYPOTHESIS (research §4.3): the extension of the
           sweep-tone principle to sung vibrato must be validated against EGG
           / synthetic stimuli before its values are trusted.*

The research recommends QCP-FB + WLP-AME below ≈350 Hz and a vibrato-sweep
estimate above; ``method="auto"`` switches per frame at ``sweep_above_hz``.
QCP-FB is not implemented; plug an external tracker via
:class:`FormantTracker` if needed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import signal

from .base import EPS, burg, frame, levinson, lpc_to_resonances, resample, runs, weighted_lp
from .spectral import Harmonics

N_FORMANTS = 4


@dataclass
class FormantTrack:
    freq: np.ndarray  # (T, 4) Hz
    bw: np.ndarray  # (T, 4) Hz
    method: np.ndarray  # (T,) str-coded: 0 none, 1 lpc/wlp, 2 sweep


class FormantTracker(Protocol):
    def track(self, x: np.ndarray, sr: int, hop: int, n_frames: int, f0: np.ndarray) -> FormantTrack: ...


def _pick_formants(a: np.ndarray, sr: float, max_formant: float) -> tuple[np.ndarray, np.ndarray]:
    f, b = lpc_to_resonances(a, sr)
    keep = (f > 90) & (f < max_formant - 50) & (b < 700) & (b > 0)
    f, b = f[keep], b[keep]
    out_f = np.full(N_FORMANTS, np.nan)
    out_b = np.full(N_FORMANTS, np.nan)
    n = min(N_FORMANTS, len(f))
    out_f[:n], out_b[:n] = f[:n], b[:n]
    return out_f, out_b


def lpc_formants(
    x: np.ndarray,
    sr: int,
    hop: int,
    n_frames: int,
    frames_mask: np.ndarray,
    max_formant: float = 5500.0,
    n_poles: int = 10,
    win_seconds: float = 0.03,
    method: str = "burg",
) -> tuple[np.ndarray, np.ndarray]:
    sr_lp = int(2 * max_formant)
    y = resample(x, sr, sr_lp)
    alpha = np.exp(-2 * np.pi * 50 / sr_lp)
    y = signal.lfilter([1, -alpha], [1], y)
    hop_lp = hop * sr_lp / sr
    win = int(round(win_seconds * sr_lp))
    # frame centres must match the core grid even though hop_lp is fractional
    centres = np.round(np.arange(n_frames) * hop_lp).astype(int)
    pad = win // 2
    yp = np.pad(y, (pad, pad + win))
    frames = np.stack([yp[c : c + win] for c in centres])
    freq = np.full((n_frames, N_FORMANTS), np.nan)
    bw = np.full((n_frames, N_FORMANTS), np.nan)
    rows = np.flatnonzero(frames_mask)
    if rows.size == 0:
        return freq, bw
    w = np.hamming(win)
    if method == "burg":
        coefs = burg(frames[rows] * w, n_poles)
    elif method == "wlp":
        coefs = np.stack([_ste_wlp(fr * w, n_poles) for fr in frames[rows]])
    else:
        raise ValueError(f"unknown LPC method {method!r}")
    for r, a in zip(rows, coefs):
        freq[r], bw[r] = _pick_formants(a, sr_lp, max_formant)
    return freq, bw


def _ste_wlp(x: np.ndarray, order: int, lag: int = 12) -> np.ndarray:
    """STE-weighted LP: w[n] = sum_{k=1..M} x[n-k]^2 (Magi et al. 2009)."""
    ste = np.convolve(x**2, np.r_[0.0, np.ones(lag)])[: len(x)]
    return weighted_lp(x, order, ste + EPS * (np.max(ste) + EPS))


def sweep_formants(
    harm: Harmonics,
    f0: np.ndarray,
    frames_mask: np.ndarray,
    hop_seconds: float,
    pool_seconds: float = 0.25,
    max_formant: float = 5500.0,
    n_poles: int = 10,
    bin_hz: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Formants from harmonic samples pooled across (at least) a vibrato cycle."""
    t_n = len(f0)
    freq = np.full((t_n, N_FORMANTS), np.nan)
    bw = np.full((t_n, N_FORMANTS), np.nan)
    half = max(1, int(round(pool_seconds / hop_seconds / 2)))
    sr_lp = 2 * max_formant
    nfft = 1024
    grid = np.linspace(0, max_formant, nfft // 2 + 1)
    alpha = np.exp(-2 * np.pi * 50 / sr_lp)
    pre_emph_db = 20 * np.log10(np.abs(1 - alpha * np.exp(-2j * np.pi * grid / sr_lp)) + EPS)
    voiced = np.isfinite(f0)
    for s, e in runs(voiced):
        for t in range(s, e):
            if not frames_mask[t]:
                continue
            lo, hi = max(s, t - half), min(e, t + half + 1)
            hf = harm.freq[lo:hi].ravel()
            ha = harm.amp_db[lo:hi].ravel()
            ok = np.isfinite(hf) & np.isfinite(ha) & (hf < max_formant)
            if ok.sum() < 6:
                continue
            hf, ha = hf[ok], ha[ok]
            # frame-level normalisation removes vibrato-synchronous level changes
            bins = np.floor(hf / bin_hz).astype(int)
            ub, inv = np.unique(bins, return_inverse=True)
            env_pts = np.array([np.max(ha[inv == i]) for i in range(len(ub))])
            env_f = (ub + 0.5) * bin_hz
            env_db = np.interp(grid, env_f, env_pts, left=env_pts[0], right=env_pts[-1])
            # same 50 Hz pre-emphasis as the LPC path, so source tilt does not consume poles
            env_db = env_db + pre_emph_db
            power = 10 ** (env_db / 10)
            r = np.fft.irfft(power, nfft)[: n_poles + 1]
            a, _ = levinson(r[None, :], n_poles)
            freq[t], bw[t] = _pick_formants(a[0], sr_lp, max_formant)
    return freq, bw


def track_formants(
    x: np.ndarray,
    sr: int,
    hop: int,
    n_frames: int,
    f0: np.ndarray,
    harm: Harmonics,
    method: str = "auto",
    lpc_method: str = "wlp",
    sweep_above_hz: float = 350.0,
    max_formant: float = 5500.0,
    smooth_frames: int = 5,
) -> FormantTrack:
    voiced = np.isfinite(f0)
    use_sweep = voiced & (f0 >= sweep_above_hz) if method == "auto" else (voiced if method == "sweep" else np.zeros_like(voiced))
    use_lpc = voiced & ~use_sweep
    freq = np.full((n_frames, N_FORMANTS), np.nan)
    bw = np.full((n_frames, N_FORMANTS), np.nan)
    if use_lpc.any():
        f1, b1 = lpc_formants(x, sr, hop, n_frames, use_lpc, max_formant=max_formant, method=lpc_method)
        freq[use_lpc], bw[use_lpc] = f1[use_lpc], b1[use_lpc]
    if use_sweep.any():
        f2, b2 = sweep_formants(harm, f0, use_sweep, hop / sr, max_formant=max_formant)
        freq[use_sweep], bw[use_sweep] = f2[use_sweep], b2[use_sweep]
    if smooth_frames > 1:
        freq = _median_smooth(freq, voiced, smooth_frames)
        bw = _median_smooth(bw, voiced, smooth_frames)
    method_code = np.where(use_sweep, 2, np.where(use_lpc, 1, 0))
    return FormantTrack(freq=freq, bw=bw, method=method_code)


def _median_smooth(v: np.ndarray, voiced: np.ndarray, width: int) -> np.ndarray:
    """NaN-aware running median within each voiced run (never across runs)."""
    out = v.copy()
    half = width // 2
    for s, e in runs(voiced):
        seg = np.pad(v[s:e], ((half, half), (0, 0)), constant_values=np.nan)
        win = np.lib.stride_tricks.sliding_window_view(seg, width, axis=0)  # (n, cols, width)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[s:e] = np.nanmedian(win, axis=-1)
    return out


# ---------------------------------------------------------------------------
# Bandwidth model and Iseli–Alwan correction
# ---------------------------------------------------------------------------


def hawks_miller_bandwidth(f: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """Formant bandwidth predicted from frequency and f0 (Hawks & Miller 1995).

    Used (as in VoiceSauce) for the Iseli–Alwan correction because measured
    LPC bandwidths are noisy.
    """
    f = np.asarray(f, dtype=float)
    s = 1 + 0.25 * (np.asarray(f0, dtype=float) - 132) / 88
    lo = 165.327516 - 6.73636734e-1 * f + 1.80874446e-3 * f**2 - 4.52201682e-6 * f**3 + 7.49514000e-9 * f**4 - 4.70219241e-12 * f**5
    hi = 15.8146139 + 8.10159009e-2 * f - 9.79728215e-5 * f**2 + 5.28725064e-8 * f**3 - 1.07099364e-11 * f**4 + 7.91528509e-16 * f**5
    return np.clip(s * np.where(f < 500, lo, hi), 20.0, 500.0)


def iseli_alwan_correction(freq_hz: np.ndarray, formant_hz: np.ndarray, bw_hz: np.ndarray, sr: float) -> np.ndarray:
    """dB boost that formant (F, B) applies at ``freq_hz`` relative to DC.

    |H(ω)|² = (1 − 2r cos θ + r²)² / [(1 − 2r cos(θ−ω) + r²)(1 − 2r cos(θ+ω) + r²)],
    r = exp(−πB/fs), θ = 2πF/fs (Iseli & Alwan, ICASSP 2004 / JASA 2007).
    Subtracting it from a harmonic amplitude gives the "starred" measure.
    """
    r = np.exp(-np.pi * bw_hz / sr)
    th = 2 * np.pi * formant_hz / sr
    w = 2 * np.pi * freq_hz / sr
    num = (1 - 2 * r * np.cos(th) + r**2) ** 2
    den = (1 - 2 * r * np.cos(th - w) + r**2) * (1 - 2 * r * np.cos(th + w) + r**2)
    return 10 * np.log10(num / np.maximum(den, EPS))


# ---------------------------------------------------------------------------
# True envelope
# ---------------------------------------------------------------------------


def true_envelope_cepstrum(
    x: np.ndarray,
    sr: int,
    hop: int,
    n_frames: int,
    f0: np.ndarray,
    n_coeffs: int = 24,
    win_seconds: float = 0.064,
    max_iter: int = 60,
    tol_db: float = 1.0,
) -> np.ndarray:
    """True-envelope cepstrum (Röbel & Rodet, DAFx 2005), first ``n_coeffs``.

    The cepstral order per frame is fs / (2 f0) (the TE optimum), capped at
    ``4 * n_coeffs``; the stored coefficients are the first ``n_coeffs`` of the
    converged envelope's real cepstrum (natural-log amplitude units).
    """
    out = np.full((n_frames, n_coeffs), np.nan)
    rows = np.flatnonzero(np.isfinite(f0))
    if rows.size == 0:
        return out
    win = int(win_seconds * sr)
    nfft = 1 << int(np.ceil(np.log2(win)))
    frames = frame(x, win, hop, n_frames)[rows] * np.hanning(win)
    spec = np.log(np.abs(np.fft.rfft(frames, nfft, axis=1)) + 1e-9)
    order = np.clip(np.round(sr / (2 * f0[rows])).astype(int), n_coeffs, 4 * n_coeffs)
    q = np.arange(nfft)
    qq = np.minimum(q, nfft - q)[None, :]
    lifter = (qq < order[:, None]).astype(float)
    lifter[:, 0] = 1
    target = spec.copy()
    tol = tol_db / 20 * np.log(10)
    for _ in range(max_iter):
        cep = np.fft.irfft(target, nfft, axis=1) * lifter
        env = np.fft.rfft(cep, nfft, axis=1).real
        if np.max(spec - env) < tol:
            break
        target = np.maximum(spec, env)
    cep = np.fft.irfft(env, nfft, axis=1)
    out[rows] = cep[:, :n_coeffs]
    return out

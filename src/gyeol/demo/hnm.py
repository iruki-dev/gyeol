"""A small harmonic-plus-noise analysis/synthesis vocoder on the shared FrameGrid.

This is the **DSP renderer's** engine: no training, no weights, so it is the
baseline for demos and audibility scoring until the M4 decoder is trained.

Analysis (per grid frame, centred on ``i·hop``, Hann window):

* ``harm_env``: log amplitude of each harmonic peak ``k·f0`` (largest bin
  within a main lobe), interpolated over frequency.  Pitch edits read new
  harmonic amplitudes from this envelope, so formants stay in place;
* ``noise_env``: log of the locally averaged power spectrum (≈400 Hz
  moving average), the level the noise is shaped to;
* ``aperiodicity``: per-band noise power fraction from inter-harmonic bins
  (voiced frames; unvoiced frames are all noise).

Synthesis: harmonics from a per-sample f0 phase, plus seeded white noise shaped
per frame and overlap-added.  Edits are c(t) terms on the source grid: f0
(cents), gain (dB), aperiodic-ratio change (dB) and a time map (source frame
for every output frame).  Equal seeds give equal noise, so two renders that
differ in one edit differ only where that edit acts.

Known limits: plosive bursts are smeared into frame-length noise, f0 changes
faster than a frame are smoothed and reverb tails are rendered as noise.  The
neural renderer replaces this engine once trained weights exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.grid import FrameGrid

EPS = 1e-12
BAND_EDGES_HZ = (0.0, 1000.0, 2000.0, 4000.0, 8000.0)


def _frames(x: np.ndarray, grid: FrameGrid, win: int) -> np.ndarray:
    centres = np.arange(grid.n_frames) * grid.hop
    xp = np.pad(np.asarray(x, float), (win // 2, win))
    return xp[centres[:, None] + np.arange(win)[None, :]]


def _band_edges(sr: int) -> list[float]:
    return [e for e in BAND_EDGES_HZ if e < sr / 2] + [sr / 2 + 1]


@dataclass
class HNMParams:
    grid: FrameGrid
    f0_hz: np.ndarray  # (T,), NaN = unvoiced
    harm_env: np.ndarray  # (T, F) log amplitude of the harmonic peaks (voiced frames), -inf-ish elsewhere
    noise_env: np.ndarray  # (T, F) log of the smoothed frame power spectrum
    aperiodicity: np.ndarray  # (T, B) noise power fraction per band in [0, 1]
    win: int
    freqs: np.ndarray  # (F,)


def analyze_hnm(x: np.ndarray, grid: FrameGrid, f0_hz: np.ndarray, win: int = 2048) -> HNMParams:
    sr = grid.sr
    w = np.hanning(win)
    spec = np.fft.rfft(_frames(x, grid, win) * w, axis=1)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(win, 1 / sr)
    df = sr / win
    T, F = power.shape
    voiced = np.isfinite(f0_hz) & (f0_hz > 2.5 * df)

    width = max(3, int(round(400.0 / df)) | 1)
    kern = np.ones(width) / width
    smooth = np.apply_along_axis(lambda r: np.convolve(r, kern, mode="same"), 1, power)
    noise_env = 0.5 * np.log(smooth + EPS)

    harm_env = np.full((T, F), np.log(EPS))
    mag = np.sqrt(power)
    guard = 2.0 * df  # Hann main-lobe half width
    for t in np.flatnonzero(voiced):
        f = f0_hz[t]
        ks = np.arange(1, int((sr / 2 - guard) / f) + 1)
        centre = ks * f / df
        half = max(1, int(min(0.3 * f, guard) / df))
        lo = np.clip(np.round(centre).astype(int) - half, 0, F - 1)
        idx = lo[:, None] + np.arange(2 * half + 1)[None, :]
        peak = mag[t][np.clip(idx, 0, F - 1)].max(axis=1)
        harm_env[t] = np.interp(freqs, ks * f, np.log(peak + EPS))

    edges = _band_edges(sr)
    ap = np.ones((T, len(edges) - 1))
    rows = np.flatnonzero(voiced)
    for s in range(0, len(rows), 256):
        r = rows[s : s + 256]
        f = f0_hz[r][:, None]
        dist = np.abs(freqs[None, :] - np.round(freqs[None, :] / f) * f)
        noise_bin = dist >= np.minimum(1.1 * guard, 0.45 * f)
        for b in range(len(edges) - 1):
            band = (freqs >= edges[b]) & (freqs < edges[b + 1])
            nb = noise_bin & band[None, :]
            if not nb.any():
                continue
            # median / ln 2 = mean of exponentially distributed noise power, robust to sidelobe leakage
            dens = np.nanmedian(np.where(nb, power[r], np.nan), axis=1) / np.log(2)
            total = power[r][:, band].sum(1) + EPS
            ap[r, b] = np.clip(dens * band.sum() / total, 0.0, 1.0)
    return HNMParams(grid, np.where(voiced, f0_hz, np.nan), harm_env, noise_env, ap, win, freqs)


def _band_curve(ap: np.ndarray, freqs: np.ndarray, sr: int) -> np.ndarray:
    edges = _band_edges(sr)
    centres = np.array([(edges[b] + min(edges[b + 1], sr / 2)) / 2 for b in range(len(edges) - 1)])
    return np.stack([np.interp(freqs, centres, row) for row in ap])


def _interp_rows(a: np.ndarray, pos: np.ndarray) -> np.ndarray:
    lo = np.clip(np.floor(pos).astype(int), 0, len(a) - 1)
    hi = np.clip(lo + 1, 0, len(a) - 1)
    w = np.clip(pos - lo, 0, 1)
    if a.ndim == 1:
        return a[lo] * (1 - w) + a[hi] * w
    return a[lo] * (1 - w)[:, None] + a[hi] * w[:, None]


def synthesize_hnm(p: HNMParams, *, f0_cents_delta: np.ndarray | None = None, gain_db: np.ndarray | None = None,
                   aperiodic_db: np.ndarray | None = None, time_map: np.ndarray | None = None, seed: int = 0,
                   frames: tuple[int, int] | None = None) -> np.ndarray:
    """Render ``p`` with optional edits.

    Edit curves live on the **source** grid; ``time_map[o]`` is the
    (fractional) source frame shown at output frame ``o``.  Output length is
    ``(n_out − 1)·hop + 1`` samples.  ``frames=(a, b)`` renders only output
    frames ``a..b-1``; the noise of output frame ``o`` is seeded by
    ``(seed, o)``, so a partial render matches the full one there (up to
    harmonic phase).
    """
    g = p.grid
    T, sr, hop, win = g.n_frames, g.sr, g.hop, p.win
    zeros = np.zeros(T)
    f0 = p.f0_hz * 2 ** (np.nan_to_num(zeros if f0_cents_delta is None else f0_cents_delta) / 1200)
    gain = np.nan_to_num(zeros if gain_db is None else gain_db)
    apd = np.nan_to_num(zeros if aperiodic_db is None else aperiodic_db)
    tm = np.arange(T, dtype=float) if time_map is None else np.clip(np.asarray(time_map, float), 0, T - 1)
    first = 0
    if frames is not None:
        first, last = max(0, frames[0]), min(len(tm), frames[1])
        if last - first < 2:
            raise ValueError(f"frames {frames} select fewer than two output frames")
        tm = tm[first:last]
    n_out = len(tm)

    near = np.clip(np.round(tm).astype(int), 0, T - 1)
    voiced = np.isfinite(f0[near])
    f0_o = np.where(voiced, _interp_rows(np.nan_to_num(f0, nan=0.0), tm), 0.0)
    # next to a voicing boundary the linear interpolation mixes in 0 Hz: use the voiced neighbour
    f0_o = np.where(voiced & (f0_o < 0.9 * np.nan_to_num(f0[near])), np.nan_to_num(f0[near]), f0_o)
    g_lin = 10 ** (_interp_rows(gain, tm) / 20)[:, None]
    ap = _band_curve(_interp_rows(p.aperiodicity, tm), p.freqs, sr)
    d = _interp_rows(apd, tm)[:, None]
    ratio = ap / np.maximum(1 - ap, 1e-6) * 10 ** (d / 10)
    ap_new = ratio / (1 + ratio)
    h_gain = np.sqrt(np.maximum(1 - ap_new, 0) / np.maximum(1 - ap, 1e-6))
    n_gain = np.sqrt(ap_new / np.maximum(ap, 1e-6))
    n_gain[~voiced] = 1.0

    n = (n_out - 1) * hop + 1
    w = np.hanning(win)
    ts = np.arange(n)
    t_frames = np.arange(n_out) * hop
    # --- harmonics
    harm = np.zeros(n)
    if voiced.any():
        f0_s = np.interp(ts, t_frames, f0_o)
        v_s = np.interp(ts, t_frames, voiced.astype(float))
        phase = 2 * np.pi * np.cumsum(f0_s) / sr
        henv = np.exp(_interp_rows(p.harm_env, tm)) * h_gain * g_lin * (2.0 / w.sum())
        kmax = int(sr / 2 / max(float(np.min(f0_o[voiced])), 50.0))
        rows = np.arange(n_out)
        # per-sample linear interpolation weights between frames (shared by all harmonics)
        i0 = np.minimum(ts // hop, n_out - 1)
        i1 = np.minimum(i0 + 1, n_out - 1)
        fr = (ts - i0 * hop) / hop
        # sin(kφ) by the Chebyshev recurrence: sin(kφ) = 2cos(φ)·sin((k−1)φ) − sin((k−2)φ)
        two_cos = 2 * np.cos(phase)
        s_prev, s_cur = np.zeros(n), np.sin(phase)
        for k in range(1, kmax + 1):
            fk = k * f0_o
            ok = voiced & (fk < sr / 2 - 2 * sr / win)
            if not ok.any():
                break
            idx = np.clip(np.round(fk / (sr / win)).astype(int), 0, len(p.freqs) - 1)
            a = np.where(ok, henv[rows, idx], 0.0)
            harm += (a[i0] * (1 - fr) + a[i1] * fr) * s_cur
            s_prev, s_cur = s_cur, two_cos * s_cur - s_prev
        harm *= v_s
    # --- noise
    nmag = np.exp(_interp_rows(p.noise_env, tm)) * np.sqrt(np.where(voiced[:, None], ap, 1.0)) * n_gain * g_lin / np.sqrt(np.sum(w**2))
    out = np.zeros(n + win)
    norm = np.zeros(n + win)
    for o in range(n_out):
        white = np.random.default_rng((seed, first + o)).standard_normal(win)
        seg = np.fft.irfft(np.fft.rfft(white) * nmag[o], win) * w
        out[o * hop : o * hop + win] += seg
        norm[o * hop : o * hop + win] += w**2
    # independent noise per frame: normalise by sqrt(Σw²) to keep the variance
    noise = out[win // 2 : win // 2 + n] / np.sqrt(np.maximum(norm[win // 2 : win // 2 + n], 1e-3))
    return harm + noise

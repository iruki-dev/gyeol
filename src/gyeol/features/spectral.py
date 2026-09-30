"""Harmonic and band-spectral analysis.

One short-time spectrum per core frame (64 ms Hann window, 10 ms hop) feeds

* harmonic amplitudes H_k (dB) and frequencies,
* band aperiodicity / band HNR (5 bands up to 8 kHz),
* spectral tilt set: alpha ratio, Hammarberg index, L/H ratio,
* singing power ratio (SPR),
* subharmonic-to-harmonic ratio (SHR),
* the harmonic picks A1, A2, A3 and P0 used by H1*–A_n* and A1–P0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._dsp import EPS, frame, parabolic_peak, power_db

#: default aperiodicity bands (Hz); five bands up to 8 kHz as in the spec
APERIODICITY_BANDS: tuple[tuple[float, float], ...] = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000))


@dataclass
class Spectrogram:
    power: np.ndarray  # (T, F) linear power
    freqs: np.ndarray  # (F,)
    sr: int
    win: int

    @property
    def df(self) -> float:
        return float(self.freqs[1] - self.freqs[0])

    @property
    def mainlobe_hz(self) -> float:
        """Half-width of the Hann main lobe."""
        return 2.0 * self.sr / self.win


def spectrogram(x: np.ndarray, sr: int, hop: int, n_frames: int, win_seconds: float = 0.064, oversample: int = 2) -> Spectrogram:
    win = int(round(win_seconds * sr))
    nfft = 1 << int(np.ceil(np.log2(win * oversample)))
    w = np.hanning(win)
    frames = frame(x, win, hop, n_frames) * w
    power = np.abs(np.fft.rfft(frames, nfft, axis=1)) ** 2 / np.sum(w**2)
    return Spectrogram(power=power, freqs=np.fft.rfftfreq(nfft, 1 / sr), sr=sr, win=win)


# ---------------------------------------------------------------------------
# Harmonics
# ---------------------------------------------------------------------------


@dataclass
class Harmonics:
    freq: np.ndarray  # (T, K) Hz, NaN if not measurable
    amp_db: np.ndarray  # (T, K) dB


def harmonic_peaks(spec: Spectrogram, f0: np.ndarray, max_harmonics: int = 40, search: float = 0.15) -> Harmonics:
    """Peak amplitude and frequency of each harmonic near k*f0.

    The search half-width is ``search * f0`` (enough for vibrato within a
    frame without wandering into the neighbouring harmonic).
    """
    t_n = len(f0)
    k = np.arange(1, max_harmonics + 1)
    freq = np.full((t_n, max_harmonics), np.nan)
    amp = np.full((t_n, max_harmonics), np.nan)
    voiced = np.flatnonzero(np.isfinite(f0))
    if voiced.size == 0:
        return Harmonics(freq, amp)
    db = power_db(spec.power)
    n_bins = db.shape[1]
    df = spec.df
    m_max = int(np.ceil(search * np.nanmax(f0) / df))
    offs = np.arange(-m_max, m_max + 1)
    for s in range(0, voiced.size, 256):
        rows = voiced[s : s + 256]
        f = f0[rows][:, None]
        centre = np.round(f * k / df).astype(int)  # (n, K)
        m = np.ceil(search * f / df).astype(int)  # (n, 1)
        idx = centre[..., None] + offs  # (n, K, O)
        ok = (np.abs(offs) <= m[..., None]) & (idx > 0) & (idx < n_bins - 1)
        idx_c = np.clip(idx, 1, n_bins - 2)
        sub = db[rows]
        r3 = np.arange(len(rows))[:, None, None]
        vals = np.where(ok, sub[r3, idx_c], -np.inf)
        best = np.take_along_axis(idx_c, vals.argmax(axis=2)[..., None], axis=2)[..., 0]  # (n, K)
        pos, val = parabolic_peak(sub, best)
        good = (centre < n_bins - 1) & np.isfinite(vals.max(axis=2))
        freq[rows] = np.where(good, pos * df, np.nan)
        amp[rows] = np.where(good, val, np.nan)
    return Harmonics(freq, amp)


def harmonic_near(h: Harmonics, target_hz: np.ndarray, f0: np.ndarray, tolerance: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Strongest harmonic within ±tolerance*target (at least ±f0/2) of target.

    Returns (amplitude dB, harmonic index 1-based).  Used for A1, A2, A3, P0.
    """
    tol = np.maximum(tolerance * target_hz, 0.5 * f0)[:, None]
    near = np.abs(h.freq - target_hz[:, None]) <= tol
    vals = np.where(near, h.amp_db, -np.inf)
    arg = vals.argmax(axis=1)
    amp = vals[np.arange(len(arg)), arg]
    amp = np.where(np.isfinite(amp), amp, np.nan)
    return amp, np.where(np.isfinite(amp), arg + 1, 0)


# ---------------------------------------------------------------------------
# Aperiodicity
# ---------------------------------------------------------------------------


def band_aperiodicity(spec: Spectrogram, f0: np.ndarray, bands=APERIODICITY_BANDS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Band aperiodicity (dB, <= 0) and band HNR (dB) from a harmonic comb.

    In each band the noise power *density* is estimated from inter-harmonic
    bins (more than one main-lobe width away from every harmonic) and
    extrapolated across the band; harmonic power is the remainder.  This is a
    D4C-inspired, deterministic estimator rather than WORLD's D4C itself.

    Returns (aperiodicity (T, B), hnr (T, B), measurable (T,) bool).
    """
    t_n = len(f0)
    nb = len(bands)
    ap = np.full((t_n, nb), np.nan)
    hnr = np.full((t_n, nb), np.nan)
    freqs = spec.freqs
    guard = spec.mainlobe_hz
    measurable = np.isfinite(f0) & (f0 > 2.2 * guard)
    rows = np.flatnonzero(measurable)
    if rows.size == 0:
        return ap, hnr, measurable
    for s in range(0, rows.size, 512):
        r = rows[s : s + 512]
        f = f0[r][:, None]
        dist = np.abs(freqs[None, :] - np.round(freqs[None, :] / f) * f)
        noise_bin = dist >= np.minimum(guard * 1.1, 0.45 * f)
        p = spec.power[r]
        for j, (lo, hi) in enumerate(bands):
            inb = (freqs >= lo) & (freqs < min(hi, spec.sr / 2))
            if not inb.any():
                continue
            nb_mask = noise_bin & inb
            cnt = nb_mask.sum(axis=1)
            dens = np.where(cnt > 0, (p * nb_mask).sum(axis=1) / np.maximum(cnt, 1), np.nan)
            total = p[:, inb].sum(axis=1) + EPS
            noise = np.minimum(dens * inb.sum(), total)
            harm = np.maximum(total - noise, total * 1e-6)
            ap[r, j] = 10 * np.log10(np.maximum(noise, EPS) / total)
            hnr[r, j] = 10 * np.log10(harm / np.maximum(noise, EPS))
    return np.clip(ap, -60.0, 0.0), np.clip(hnr, -20.0, 60.0), measurable


# ---------------------------------------------------------------------------
# Tilt, SPR, SHR
# ---------------------------------------------------------------------------


def _band_energy(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sel = (freqs >= lo) & (freqs < hi)
    return power[:, sel].sum(axis=1) + EPS


def _band_max_db(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sel = (freqs >= lo) & (freqs < hi)
    return power_db(power[:, sel].max(axis=1))


def tilt_measures(spec: Spectrogram) -> dict[str, np.ndarray]:
    """Alpha ratio, Hammarberg index, L/H ratio and SPR (all dB).

    * alpha ratio  = 10 log10(E[1–5 kHz] / E[50 Hz–1 kHz])  (eGeMAPS sign)
    * Hammarberg   = max dB[0–2 kHz] − max dB[2–5 kHz]
    * L/H ratio    = 10 log10(E[<4 kHz] / E[>=4 kHz])  (ADSV convention)
    * SPR          = max dB[2–4 kHz] − max dB[0–2 kHz]  (Omori et al. 1996)
    """
    p, f = spec.power, spec.freqs
    nyq = spec.sr / 2
    return {
        "alpha_ratio": 10 * np.log10(_band_energy(p, f, 1000, min(5000, nyq)) / _band_energy(p, f, 50, 1000)),
        "hammarberg": _band_max_db(p, f, 0, 2000) - _band_max_db(p, f, 2000, min(5000, nyq)),
        "lh_ratio": 10 * np.log10(_band_energy(p, f, 0, 4000) / _band_energy(p, f, 4000, nyq)),
        "spr": _band_max_db(p, f, 2000, 4000) - _band_max_db(p, f, 0, 2000),
    }


def subharmonic_ratio(spec: Spectrogram, f0: np.ndarray, max_freq: float = 1250.0) -> np.ndarray:
    """Simplified subharmonic-to-harmonic ratio at the tracked f0.

    Sun (ICASSP 2002) defines SHR = SS/SH, where SH sums spectral amplitude
    at harmonics and SS at the half-integer subharmonics.  Here f0 comes from
    the consensus tracker, so SHR is evaluated directly at (k − ½)·f0 vs k·f0
    on the linear amplitude spectrum below ``max_freq``.
    """
    out = np.full(len(f0), np.nan)
    rows = np.flatnonzero(np.isfinite(f0))
    if rows.size == 0:
        return out
    amp = np.sqrt(spec.power[rows])
    f = f0[rows][:, None]
    k = np.arange(1, 64)[None, :]
    harm = k * f
    sub = (k - 0.5) * f
    use = harm <= max_freq
    df = spec.df

    def sample(freq: np.ndarray) -> np.ndarray:
        # peak within ±main-lobe to tolerate vibrato and bin quantisation
        i = np.round(freq / df).astype(int)
        w = max(1, int(round(spec.mainlobe_hz / df / 2)))
        stack = [np.take_along_axis(amp, np.clip(i + o, 0, amp.shape[1] - 1), axis=1) for o in range(-w, w + 1)]
        return np.max(stack, axis=0)

    sh = np.sum(sample(harm) * use, axis=1)
    ss = np.sum(sample(sub) * use, axis=1)
    out[rows] = ss / np.maximum(sh, EPS)
    return out

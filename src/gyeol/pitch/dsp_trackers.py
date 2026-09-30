"""Pitch group: f0, voicing probability and periodicity confidence.

The research specifies a *consensus* of several trackers (RMVPE, pYIN,
SwiftF0) where the median is the estimate and the inter-tracker spread is the
confidence.  gyeol ships three dependency-free trackers:

* :class:`YinTracker`         – YIN (de Cheveigné & Kawahara, JASA 2002)
* :class:`PyinTracker`        – probabilistic YIN + HMM/Viterbi decoding
                                 (Mauch & Dixon, ICASSP 2014)
* :class:`HarmonicSumTracker` – subharmonic summation (Hermes, JASA 1988)

Neural trackers (RMVPE, SwiftF0, CREPE …) plug in through the
:class:`PitchTracker` protocol: any object with a ``name`` attribute and a
``track(x, sr, hop, n_frames)`` method that returns a :class:`PitchEstimate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from scipy import special

from ..dsp.base import EPS, cents_to_hz, frame, hz_to_cents, parabolic_peak


@dataclass
class PitchEstimate:
    f0: np.ndarray  # Hz, NaN where unvoiced
    voiced_prob: np.ndarray  # [0, 1]
    voiced: np.ndarray  # bool decision
    name: str = ""


@runtime_checkable
class PitchTracker(Protocol):
    name: str

    def track(self, x: np.ndarray, sr: int, hop: int, n_frames: int) -> PitchEstimate: ...


# ---------------------------------------------------------------------------
# YIN core
# ---------------------------------------------------------------------------


_CMNDF_CACHE: dict = {}


def cmndf(x: np.ndarray, sr: int, hop: int, count: int, fmin: float, fmax: float) -> tuple[np.ndarray, int, int]:
    """Cumulative mean normalised difference function per frame.

    Returns (d', tau_min, tau_max) with d' of shape (count, tau_max + 1).
    The last result is cached so YIN and pYIN on the same signal share it.
    """
    key = (id(x), x.ctypes.data, len(x), float(np.sum(x[::997])), sr, hop, count, fmin, fmax)
    if key in _CMNDF_CACHE:
        return _CMNDF_CACHE[key]
    result = _cmndf(x, sr, hop, count, fmin, fmax)
    _CMNDF_CACHE.clear()
    _CMNDF_CACHE[key] = result
    return result


def _cmndf(x: np.ndarray, sr: int, hop: int, count: int, fmin: float, fmax: float) -> tuple[np.ndarray, int, int]:
    tau_min = max(2, int(np.floor(sr / fmax)))
    tau_max = int(np.ceil(sr / fmin))
    w = max(tau_max, int(0.025 * sr))
    frames = frame(x, w + tau_max + 1, hop, count)
    nfft = 1 << int(np.ceil(np.log2(2 * frames.shape[1])))
    head = frames[:, :w]
    spec_full = np.fft.rfft(frames, nfft, axis=1)
    spec_head = np.fft.rfft(head, nfft, axis=1)
    r = np.fft.irfft(np.conj(spec_head) * spec_full, nfft, axis=1)[:, : tau_max + 1]
    sq = frames**2
    csum = np.concatenate([np.zeros((count, 1)), np.cumsum(sq, axis=1)], axis=1)
    e0 = csum[:, w][:, None]
    taus = np.arange(tau_max + 1)
    e_tau = csum[:, taus + w] - csum[:, taus]
    d = np.maximum(e0 + e_tau - 2 * r, 0.0)
    cum = np.cumsum(d[:, 1:], axis=1)
    dn = np.ones_like(d)
    dn[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cum, EPS)
    # silent frames (absolute, or > 80 dB below the loudest frame) are
    # aperiodic by definition — a DC or near-zero residue would otherwise
    # give d'(tau) ≈ 0, i.e. "perfect periodicity"
    silent = (e0[:, 0] < EPS * w) | (e0[:, 0] < 1e-8 * np.max(e0))
    dn[silent] = 1.0
    return dn, tau_min, tau_max


def yin_time_shift(tau: np.ndarray, sr: int, fmin: float, fmax: float) -> np.ndarray:
    """Offset (samples) between a YIN frame centre and its effective analysis centre.

    The difference function compares x[j] with x[j + tau] for j < W, so the
    analysed span is centred (W + tau) / 2 samples after the frame start.
    """
    tau_max = int(np.ceil(sr / fmin))
    w = max(tau_max, int(0.025 * sr))
    length = w + tau_max + 1
    return (w + tau) / 2.0 - length // 2


def realign(f0: np.ndarray, shift_samples: np.ndarray, hop: int) -> np.ndarray:
    """Resample an f0 track whose frame i really describes time i*hop + shift."""
    out = f0.copy()
    from ..dsp.base import runs

    for s, e in runs(np.isfinite(f0)):
        if e - s < 2:
            continue
        idx = np.arange(s, e)
        t_eff = idx * hop + shift_samples[s:e]
        order = np.argsort(t_eff)
        c = hz_to_cents(f0[s:e])[order]
        out[s:e] = cents_to_hz(np.interp(idx * hop, t_eff[order], c))
    return out


def _local_minima(row: np.ndarray, lo: int, hi: int) -> np.ndarray:
    seg = row[lo - 1 : hi + 2]
    idx = np.flatnonzero((seg[1:-1] < seg[:-2]) & (seg[1:-1] <= seg[2:])) + lo
    return idx


@dataclass
class YinTracker:
    fmin: float = 60.0
    fmax: float = 1600.0
    threshold: float = 0.15
    name: str = "yin"

    def track(self, x: np.ndarray, sr: int, hop: int, n_frames: int) -> PitchEstimate:
        dn, lo, hi = cmndf(x, sr, hop, n_frames, self.fmin, self.fmax)
        sub = dn[:, lo : hi + 1]
        below = sub < self.threshold
        # first dip under threshold, then walk down to its local minimum
        first = np.where(below.any(axis=1), below.argmax(axis=1), sub.argmin(axis=1))
        tau = first.copy()
        n = sub.shape[1]
        for _ in range(n):
            nxt = np.minimum(tau + 1, n - 1)
            step = sub[np.arange(len(tau)), nxt] < sub[np.arange(len(tau)), tau]
            if not step.any():
                break
            tau = np.where(step, nxt, tau)
        idx = tau + lo
        frac, val = parabolic_peak(-dn, idx)
        val = -val
        f0 = sr / np.maximum(frac, 1.0)
        periodicity = np.clip(1.0 - val, 0.0, 1.0)
        voiced = val < self.threshold
        f0 = np.where(voiced, f0, np.nan)
        f0 = realign(f0, yin_time_shift(frac, sr, self.fmin, self.fmax), hop)
        return PitchEstimate(f0=f0, voiced_prob=periodicity, voiced=voiced, name=self.name)


# ---------------------------------------------------------------------------
# pYIN
# ---------------------------------------------------------------------------


@dataclass
class PyinTracker:
    """Probabilistic YIN with Viterbi decoding over (pitch bin × voicing)."""

    fmin: float = 60.0
    fmax: float = 1600.0
    beta_a: float = 2.0
    beta_b: float = 18.0
    bin_cents: float = 20.0
    max_jump_cents: float = 400.0  # per 10 ms frame
    switch_prob: float = 0.01
    no_trough_prob: float = 0.01
    name: str = "pyin"

    def candidates(self, x: np.ndarray, sr: int, hop: int, n_frames: int) -> list[tuple[np.ndarray, np.ndarray]]:
        dn, lo, hi = cmndf(x, sr, hop, n_frames, self.fmin, self.fmax)
        out: list[tuple[np.ndarray, np.ndarray]] = []
        cdf = lambda s: special.betainc(self.beta_a, self.beta_b, np.clip(s, 0.0, 1.0))  # noqa: E731
        for row in dn:
            troughs = _local_minima(row, lo, hi)
            if troughs.size == 0:
                out.append((np.empty(0), np.empty(0)))
                continue
            vals = row[troughs]
            prev_min = np.minimum.accumulate(np.concatenate([[1.0], vals[:-1]]))
            probs = np.maximum(cdf(prev_min) - cdf(vals), 0.0)
            # thresholds below every trough fall back to the global minimum
            gmin = int(np.argmin(vals))
            probs[gmin] += self.no_trough_prob * cdf(vals[gmin])
            frac, _ = parabolic_peak(-row[None, :], troughs[None, :])
            f0 = sr / np.maximum(frac[0], 1.0)
            keep = probs > 1e-6
            out.append((f0[keep], probs[keep]))
        return out

    def track(self, x: np.ndarray, sr: int, hop: int, n_frames: int) -> PitchEstimate:
        cands = self.candidates(x, sr, hop, n_frames)
        c_lo = hz_to_cents(self.fmin)
        n_bins = int(np.ceil((hz_to_cents(self.fmax) - c_lo) / self.bin_cents)) + 1
        obs_v = np.full((n_frames, n_bins), EPS)
        vprob = np.zeros(n_frames)
        for t, (f, p) in enumerate(cands):
            if f.size:
                b = np.clip(np.round((hz_to_cents(f) - c_lo) / self.bin_cents).astype(int), 0, n_bins - 1)
                np.add.at(obs_v[t], b, p)
                vprob[t] = min(p.sum(), 1.0)
        obs_u = np.maximum(1.0 - vprob, EPS)[:, None] / n_bins
        path_bin, path_voiced = self._viterbi(np.log(obs_v), np.log(np.broadcast_to(obs_u, obs_v.shape)))
        f0 = np.full(n_frames, np.nan)
        for t in np.flatnonzero(path_voiced):
            f, p = cands[t]
            target = cents_to_hz(c_lo + path_bin[t] * self.bin_cents)
            if f.size:
                j = np.argmin(np.abs(hz_to_cents(f) - hz_to_cents(target)))
                if abs(hz_to_cents(f[j]) - hz_to_cents(target)) <= 1.5 * self.bin_cents:
                    target = f[j]
            f0[t] = target
        tau = np.where(np.isfinite(f0), sr / np.where(np.isfinite(f0), f0, 1.0), 0.0)
        f0 = realign(f0, yin_time_shift(tau, sr, self.fmin, self.fmax), hop)
        return PitchEstimate(f0=f0, voiced_prob=vprob, voiced=path_voiced, name=self.name)

    def _viterbi(self, log_ov: np.ndarray, log_ou: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_t, n_b = log_ov.shape
        w = max(1, int(round(self.max_jump_cents / self.bin_cents)))
        offsets = np.arange(-w, w + 1)
        tri = (w + 1 - np.abs(offsets)).astype(float)
        log_tri = np.log(tri / tri.sum())
        ls, lk = np.log(self.switch_prob), np.log1p(-self.switch_prob)
        delta_v = log_ov[0] + np.log(0.5)
        delta_u = log_ou[0] + np.log(0.5)
        back_off = np.zeros((n_t, 2, n_b), dtype=np.int16)
        back_src = np.zeros((n_t, 2, n_b), dtype=np.int8)

        def spread(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            padded = np.pad(d, w, constant_values=-np.inf)
            win = np.lib.stride_tricks.sliding_window_view(padded, 2 * w + 1) + log_tri[::-1]
            arg = win.argmax(axis=1)
            return win[np.arange(n_b), arg], (arg - w).astype(np.int16)  # source = b + offset

        for t in range(1, n_t):
            mv, ov = spread(delta_v)
            mu, ou = spread(delta_u)
            # into voiced
            a, b = mv + lk, mu + ls
            src_v = (b > a).astype(np.int8)
            new_v = np.maximum(a, b) + log_ov[t]
            # into unvoiced
            a2, b2 = mv + ls, mu + lk
            src_u = (b2 > a2).astype(np.int8)
            new_u = np.maximum(a2, b2) + log_ou[t]
            back_src[t, 0], back_src[t, 1] = src_v, src_u
            back_off[t, 0] = np.where(src_v == 1, ou, ov)
            back_off[t, 1] = np.where(src_u == 1, ou, ov)
            delta_v, delta_u = new_v, new_u
        state = 0 if delta_v.max() >= delta_u.max() else 1
        b = int(np.argmax(delta_v if state == 0 else delta_u))
        bins = np.zeros(n_t, dtype=int)
        voiced = np.zeros(n_t, dtype=bool)
        for t in range(n_t - 1, -1, -1):
            bins[t], voiced[t] = b, state == 0
            if t == 0:
                break
            src = int(back_src[t, state, b])
            b = int(np.clip(b + back_off[t, state, b], 0, n_b - 1))
            state = src
        return bins, voiced


# ---------------------------------------------------------------------------
# Subharmonic summation
# ---------------------------------------------------------------------------


@dataclass
class HarmonicSumTracker:
    fmin: float = 60.0
    fmax: float = 1600.0
    n_harmonics: int = 10
    decay: float = 0.84
    grid_cents: float = 10.0
    win_seconds: float = 0.064
    salience_threshold: float = 0.35
    name: str = "shs"

    def track(self, x: np.ndarray, sr: int, hop: int, n_frames: int) -> PitchEstimate:
        win = int(self.win_seconds * sr)
        nfft = 1 << int(np.ceil(np.log2(4 * win)))
        frames = frame(x, win, hop, n_frames) * np.hanning(win)
        grid = cents_to_hz(np.arange(hz_to_cents(self.fmin), hz_to_cents(self.fmax), self.grid_cents))
        h = np.arange(1, self.n_harmonics + 1)
        weights = self.decay ** (h - 1)
        fbin = np.outer(grid, h) * nfft / sr  # (C, H)
        in_range = fbin < nfft // 2 - 1
        fbin = np.minimum(fbin, nfft // 2 - 2)
        i0 = np.floor(fbin).astype(int)
        frac = fbin - i0
        f0 = np.full(n_frames, np.nan)
        sal = np.zeros(n_frames)
        chunk = 512
        for s in range(0, n_frames, chunk):
            mag = np.abs(np.fft.rfft(frames[s : s + chunk], nfft, axis=1))
            ref = np.percentile(mag, 99, axis=1, keepdims=True) + EPS
            comp = np.log1p(mag / (0.01 * ref))  # level-invariant compression
            vals = comp[:, i0] * (1 - frac) + comp[:, i0 + 1] * frac  # (n, C, H)
            score = np.sum(vals * weights * in_range, axis=2) / np.sum(weights * in_range, axis=1)
            best = score.argmax(axis=1)
            pos, peak = parabolic_peak(score, best)
            f0[s : s + chunk] = cents_to_hz(hz_to_cents(self.fmin) + pos * self.grid_cents)
            mean = score.mean(axis=1) + EPS
            sal[s : s + chunk] = np.clip((peak - mean) / (peak + EPS), 0.0, 1.0)
        rms = np.sqrt(np.mean(frames**2, axis=1))
        silent = rms < 1e-4 * (np.max(rms) + EPS)
        sal[silent] = 0.0
        voiced = sal > self.salience_threshold
        return PitchEstimate(f0=np.where(voiced, f0, np.nan), voiced_prob=sal, voiced=voiced, name=self.name)


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


@dataclass
class PitchConsensus:
    f0: np.ndarray  # Hz, NaN unvoiced
    cents: np.ndarray  # re A4
    voiced: np.ndarray
    voiced_prob: np.ndarray
    confidence: np.ndarray  # periodicity confidence from inter-tracker spread
    spread_cents: np.ndarray
    estimates: list[PitchEstimate]


def default_trackers(fmin: float = 60.0, fmax: float = 1600.0) -> list[PitchTracker]:
    return [PyinTracker(fmin=fmin, fmax=fmax), YinTracker(fmin=fmin, fmax=fmax), HarmonicSumTracker(fmin=fmin, fmax=fmax)]


def consensus(
    x: np.ndarray,
    sr: int,
    hop: int,
    n_frames: int,
    trackers: Sequence[PitchTracker] | None = None,
    spread_scale_cents: float = 100.0,
    min_run_frames: int = 3,
) -> PitchConsensus:
    """Median-of-trackers f0 with inter-tracker spread as confidence."""
    trackers = list(trackers) if trackers else default_trackers()
    ests = [t.track(x, sr, hop, n_frames) for t in trackers]
    voiced_votes = np.stack([e.voiced for e in ests]).astype(float)
    vprob = np.mean(np.stack([e.voiced_prob for e in ests]), axis=0)
    voiced = voiced_votes.mean(axis=0) > 0.5
    cents_all = np.stack([hz_to_cents(e.f0) for e in ests])
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(cents_all, axis=0)
            spread = np.nanmax(np.abs(cents_all - med), axis=0)
    n_agree = np.sum(np.isfinite(cents_all), axis=0)
    voiced &= np.isfinite(med) & (n_agree >= min(2, len(ests)))
    voiced = _remove_short_runs(voiced, min_run_frames)
    spread = np.where(voiced, np.nan_to_num(spread, nan=0.0), np.nan)
    conf = np.where(voiced, np.exp(-spread / spread_scale_cents), 0.0)
    cents = np.where(voiced, med, np.nan)
    return PitchConsensus(
        f0=cents_to_hz(cents),
        cents=cents,
        voiced=voiced,
        voiced_prob=vprob,
        confidence=conf,
        spread_cents=spread,
        estimates=ests,
    )


def _remove_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    from ..dsp.base import runs

    out = mask.copy()
    for s, e in runs(mask):
        if e - s < min_len:
            out[s:e] = False
    return out

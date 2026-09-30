"""Latency calibration for sing-along recording.

The user hears the track on headphones and sings along, so user and target
share a clock *up to the audio-route latency* (output + input buffers, plus
Bluetooth codec delay).  That offset must be removed before alignment.

* :func:`chirp` + :func:`loopback_latency` – wired routes: play a chirp,
  record it through a loopback (or the mic near the earpiece) and
  cross-correlate.
* :func:`tap_along_latency` – Bluetooth / no loopback: the user taps along
  with a click track; the median tap offset minus a human reaction bias is
  the latency, the MAD is the jitter.
* :func:`refine_offset` – offline refinement: cross-correlate onset-strength
  envelopes of the user's take and the target guide vocal within ±max_lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..core.status import Result


@dataclass
class LatencyEstimate:
    latency_s: float
    jitter_s: float
    confidence: float  # 0..1
    method: str


def chirp(sr: int, duration: float = 1.0, f0: float = 100.0, f1: float = 8000.0, level: float = 0.5) -> np.ndarray:
    """Exponential sine sweep with raised-cosine fades."""
    t = np.arange(int(duration * sr)) / sr
    y = level * signal.chirp(t, f0, duration, f1, method="logarithmic")
    fade = int(0.01 * sr)
    w = np.ones_like(y)
    w[:fade] = np.hanning(2 * fade)[:fade]
    w[-fade:] = np.hanning(2 * fade)[fade:]
    return y * w


def _xcorr_lag(ref: np.ndarray, rec: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (samples) such that rec[n] ≈ ref[n - lag], plus peak-to-sidelobe ratio."""
    n = len(ref) + len(rec) - 1
    nfft = 1 << int(np.ceil(np.log2(n)))
    R = np.fft.rfft(rec, nfft) * np.conj(np.fft.rfft(ref, nfft))
    cc = np.fft.irfft(R / (np.abs(R) + 1e-12), nfft)  # PHAT weighting
    lags = np.r_[np.arange(0, max_lag + 1), np.arange(-max_lag, 0)]
    vals = np.r_[cc[: max_lag + 1], cc[-max_lag:]]
    i = int(np.argmax(vals))
    peak = vals[i]
    side = np.sort(np.abs(np.delete(vals, np.arange(max(0, i - 5), min(len(vals), i + 6)))))[-10:].mean() if len(vals) > 20 else 1e-9
    return int(lags[i]), float(peak / (side + 1e-12))


def loopback_latency(played: np.ndarray, recorded: np.ndarray, sr: int, max_latency_s: float = 1.0) -> Result[LatencyEstimate]:
    if len(recorded) < len(played) // 2:
        return Result.failure("recording is shorter than half the calibration signal")
    lag, psr = _xcorr_lag(np.asarray(played, float), np.asarray(recorded, float), int(max_latency_s * sr))
    conf = float(np.clip((psr - 3.0) / 10.0, 0.0, 1.0))
    est = LatencyEstimate(lag / sr, 0.0, conf, "loopback")
    if lag < 0:
        return Result.unreliable(est, "negative latency: calibration recording precedes playback")
    if psr < 4.0:
        return Result.unreliable(est, f"weak correlation peak (peak/sidelobe {psr:.1f}); repeat calibration closer to the source")
    return Result.success(est)


def tap_along_latency(tap_times: np.ndarray, beat_times: np.ndarray, reaction_bias_s: float = 0.0, max_offset_s: float = 0.6) -> Result[LatencyEstimate]:
    """Latency from taps on a click track: median(tap − nearest beat) − bias.

    Taps farther than ``max_offset_s`` from any beat are discarded.  At least
    8 usable taps are required.
    """
    taps = np.sort(np.asarray(tap_times, float))
    beats = np.sort(np.asarray(beat_times, float))
    if len(beats) < 2 or len(taps) == 0:
        return Result.failure("need at least two beats and one tap")
    j = np.clip(np.searchsorted(beats, taps), 1, len(beats) - 1)
    prev, nxt = beats[j - 1], beats[j]
    # taps come *after* the beat they answer (route latency ≥ 0)
    d = np.where(taps - prev <= max_offset_s, taps - prev, taps - nxt)
    d = d[np.abs(d) <= max_offset_s]
    if len(d) < 8:
        return Result.failure(f"only {len(d)} usable taps (need ≥ 8)")
    med = float(np.median(d))
    mad = float(1.4826 * np.median(np.abs(d - med)))
    conf = float(np.clip(1.0 - mad / 0.05, 0.0, 1.0))
    est = LatencyEstimate(med - reaction_bias_s, mad, conf, "tap-along")
    if mad > 0.04:
        return Result.unreliable(est, f"tap jitter {mad * 1000:.0f} ms is too large for a stable estimate")
    return Result.success(est)


def onset_envelope(x: np.ndarray, sr: int, hop: int = 256, win: int = 1024) -> np.ndarray:
    """Half-wave-rectified log-spectral flux."""
    x = np.asarray(x, float)
    n = 1 + max(0, len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * np.hanning(win), axis=1))
    lm = np.log1p(100 * mag / (mag.max() + 1e-12))
    flux = np.maximum(np.diff(lm, axis=0), 0).sum(axis=1)
    return np.r_[0.0, flux]


def refine_offset(user: np.ndarray, guide: np.ndarray, sr: int, max_lag_s: float = 0.3, hop: int = 256) -> Result[LatencyEstimate]:
    """Residual offset of ``user`` relative to the target ``guide`` vocal.

    Positive = the user is late.  Uses onset-strength envelopes so it does not
    depend on timbre or key.
    """
    eu = onset_envelope(user, sr, hop)
    eg = onset_envelope(guide, sr, hop)
    if eu.std() < 1e-9 or eg.std() < 1e-9:
        return Result.failure("no onsets to correlate")
    eu = (eu - eu.mean()) / eu.std()
    eg = (eg - eg.mean()) / eg.std()
    max_lag = int(max_lag_s * sr / hop)
    n = min(len(eu), len(eg))
    cc = signal.correlate(eu[:n], eg[:n], mode="full", method="fft") / n
    lags = np.arange(-n + 1, n)
    sel = np.abs(lags) <= max_lag
    i = int(np.argmax(cc[sel]))
    lag = int(lags[sel][i])
    peak = float(cc[sel][i])
    # sub-frame refinement
    c = cc[sel]
    if 0 < i < len(c) - 1:
        den = c[i - 1] - 2 * c[i] + c[i + 1]
        frac = 0.5 * (c[i - 1] - c[i + 1]) / den if abs(den) > 1e-12 else 0.0
    else:
        frac = 0.0
    est = LatencyEstimate((lag + frac) * hop / sr, 0.0, float(np.clip(peak / 0.5, 0, 1)), "onset-xcorr")
    if peak < 0.2:
        return Result.unreliable(est, f"weak onset correlation ({peak:.2f})")
    return Result.success(est)


def shift(x: np.ndarray, seconds: float, sr: int) -> np.ndarray:
    """Advance (positive seconds) or delay (negative) a signal, zero-padding."""
    n = int(round(seconds * sr))
    if n > 0:
        return np.r_[x[n:], np.zeros(n)]
    if n < 0:
        return np.r_[np.zeros(-n), x[:n]]
    return x.copy()

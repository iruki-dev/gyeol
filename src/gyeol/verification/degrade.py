"""Controlled-degradation benchmark (research §5.ii, secondary leg).

Applies known nuisances to clean (reference-mic or clean-stem) audio:
additive noise at a target SNR, reverberation with a target T60/DRR,
band-limiting, lossy codecs (via ffmpeg when available), AGC compression,
clipping, device frequency responses and accompaniment remixing.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from scipy import signal

from ..dsp.base import EPS, resample


def add_noise(x: np.ndarray, snr_db: float, noise: str | np.ndarray = "white", seed: int = 0, active_only: bool = True) -> np.ndarray:
    """Add noise so that (active) signal power / noise power = snr_db."""
    rng = np.random.default_rng(seed)
    if isinstance(noise, np.ndarray):
        nz = np.resize(noise, len(x)).astype(float)
    elif noise == "white":
        nz = rng.standard_normal(len(x))
    elif noise == "pink":
        spec = np.fft.rfft(rng.standard_normal(len(x)))
        f = np.arange(len(spec))
        spec[1:] /= np.sqrt(f[1:])
        nz = np.fft.irfft(spec, len(x))
    else:
        raise ValueError(f"unknown noise {noise!r}")
    if active_only:
        env = np.abs(x)
        thr = 0.05 * np.max(env)
        sig = x[env > thr] if (env > thr).any() else x
    else:
        sig = x
    p_sig = np.mean(sig**2)
    p_n = np.mean(nz**2) + EPS
    return x + nz * np.sqrt(p_sig / (p_n * 10 ** (snr_db / 10)))


def synthetic_rir(sr: int, t60: float, drr_db: float = 0.0, length_s: float | None = None, seed: int = 0) -> np.ndarray:
    """Exponentially decaying noise tail with a direct impulse at a given DRR."""
    rng = np.random.default_rng(seed)
    length_s = length_s or max(0.1, 1.2 * t60)
    n = int(length_s * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9078 * t / max(t60, 1e-3))
    tail[: int(0.0025 * sr)] = 0.0  # small gap after the direct path
    tail_e = np.sum(tail**2)
    direct_e = tail_e * 10 ** (drr_db / 10)
    h = tail.copy()
    h[0] = np.sqrt(direct_e)
    return h / np.max(np.abs(h))


def reverberate(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    y = signal.fftconvolve(x, rir)[: len(x)]
    return y * (np.std(x) / (np.std(y) + EPS))


def bandlimit(x: np.ndarray, sr: int, cutoff_hz: float, order: int = 10) -> np.ndarray:
    sos = signal.butter(order, cutoff_hz, "low", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x)


def device_response(x: np.ndarray, sr: int, freqs_hz: list[float], gains_db: list[float], n_taps: int = 513) -> np.ndarray:
    """Apply a device magnitude response given by breakpoints (linear phase FIR)."""
    f = np.clip(np.asarray(freqs_hz, float) / (sr / 2), 0, 1)
    g = 10 ** (np.asarray(gains_db, float) / 20)
    if f[0] > 0:
        f, g = np.r_[0.0, f], np.r_[g[0], g]
    if f[-1] < 1:
        f, g = np.r_[f, 1.0], np.r_[g, g[-1]]
    h = signal.firwin2(n_taps, f, g)
    return signal.fftconvolve(x, h, mode="same")


def agc(x: np.ndarray, sr: int, target_db: float = -20.0, attack_s: float = 0.05, release_s: float = 1.0, max_gain_db: float = 30.0) -> np.ndarray:
    """Simple feed-forward automatic gain control (level follower)."""
    a_att = np.exp(-1 / (attack_s * sr))
    a_rel = np.exp(-1 / (release_s * sr))
    level = np.empty_like(x)
    env = 1e-3
    for i, v in enumerate(np.abs(x)):
        a = a_att if v > env else a_rel
        env = a * env + (1 - a) * v
        level[i] = env
    gain_db = np.clip(target_db - 20 * np.log10(level + EPS), -max_gain_db, max_gain_db)
    return x * 10 ** (gain_db / 20)


def clip(x: np.ndarray, level: float = 0.5) -> np.ndarray:
    peak = np.max(np.abs(x)) + EPS
    return np.clip(x, -level * peak, level * peak)


def mix_accompaniment(voice: np.ndarray, accompaniment: np.ndarray, var_db: float) -> np.ndarray:
    """Mix at a voice-to-accompaniment ratio of ``var_db``."""
    acc = np.resize(accompaniment, len(voice))
    g = np.sqrt(np.mean(voice**2) / (np.mean(acc**2) + EPS) / 10 ** (var_db / 10))
    return voice + g * acc


def codec(x: np.ndarray, sr: int, fmt: str = "mp3", bitrate_kbps: int = 64) -> np.ndarray:
    """Round-trip through a lossy codec with ffmpeg (mp3 / aac / opus)."""
    ff = shutil.which("ffmpeg")
    if ff is None:
        raise RuntimeError("codec() needs ffmpeg on PATH")
    from ..io import load_audio, save_audio

    ext, enc = {"mp3": (".mp3", "libmp3lame"), "aac": (".m4a", "aac"), "opus": (".opus", "libopus")}[fmt]
    with tempfile.TemporaryDirectory() as d:
        src, mid, dst = Path(d, "in.wav"), Path(d, "mid" + ext), Path(d, "out.wav")
        save_audio(src, x, sr)
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-c:a", enc, "-b:a", f"{bitrate_kbps}k", str(mid)], check=True)
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mid), "-ar", str(sr), "-ac", "1", str(dst)], check=True)
        y, sr2 = load_audio(dst)
    y = resample(y, sr2, sr)
    # codecs add priming delay; realign to the input
    n = min(len(x), len(y))
    lag = int(np.argmax(signal.correlate(y[:n], x[:n], mode="full", method="fft")) - (n - 1))
    y = np.roll(y, -lag)[: len(x)]
    return np.pad(y, (0, len(x) - len(y)))


@dataclass
class Condition:
    name: str
    level: float
    apply: Callable[[np.ndarray, int], np.ndarray]


def _reverb_t60(x: np.ndarray, sr: int, t60: float) -> np.ndarray:
    return reverberate(x, synthetic_rir(sr, t60))


def standard_grid(accompaniment: np.ndarray | None = None) -> list[Condition]:
    """The degradation axes of research §5.ii (codecs only if ffmpeg exists)."""
    conds: list[Condition] = [Condition("clean", 0.0, lambda x, sr: x)]
    conds += [Condition("snr_db", s, lambda x, sr, s=s: add_noise(x, s)) for s in range(0, 55, 5)]
    conds += [Condition("t60_s", t, partial(_reverb_t60, t60=t)) for t in (0.2, 0.4, 0.8, 1.2)]
    conds += [Condition("bandwidth_hz", b, partial(bandlimit, cutoff_hz=b)) for b in (3400, 5000, 7000)]
    if shutil.which("ffmpeg"):
        for fmt, rates in (("mp3", (32, 64, 128)), ("aac", (64, 128)), ("opus", (16, 32))):
            conds += [Condition(f"{fmt}_kbps", r, partial(codec, fmt=fmt, bitrate_kbps=r)) for r in rates]
    if accompaniment is not None:
        conds += [Condition("var_db", v, lambda x, sr, v=v: mix_accompaniment(x, accompaniment, v)) for v in (10, 5, 0, -5, -10)]
    return conds


def run_grid(x: np.ndarray, sr: int, conditions: list[Condition]) -> Iterator[tuple[Condition, np.ndarray]]:
    for c in conditions:
        yield c, c.apply(x, sr)


__all__ = [
    "add_noise", "synthetic_rir", "reverberate", "bandlimit", "device_response", "agc", "clip",
    "mix_accompaniment", "codec", "Condition", "standard_grid", "run_grid",
]

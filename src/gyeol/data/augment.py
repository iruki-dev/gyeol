"""Augmentation suite with labels (training data for env/residual encoders, M4).

Every transform is ``f(x, sr, rng) -> (y, labels)``; ``labels`` records what
was applied so the env encoder can be trained on it and the adversarial
heads can be supervised.  :class:`AugmentationPipeline` samples a random
chain deterministically from a seed.

Transforms

* channel / room: :func:`noise`, :func:`reverb`, :func:`device_eq`,
  :func:`codec`, :func:`compression`, :func:`agc`
* production: :func:`separation_artifacts` (mask musical noise, spectral
  holes, residual accompaniment)
* transport: :func:`bluetooth_jitter` (time-varying delay + packet loss)
* voice: :func:`timbre_shift` (formant warp, f0 kept) and
  :func:`pitch_shift` (f0 shift, formants kept) — the Seed-VC / YingMusic
  style timbre augmentation used for the residual encoder.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import signal

from ..verification import degrade

EPS = 1e-12
Labels = dict[str, object]
Transform = Callable[[np.ndarray, int, np.random.Generator], tuple[np.ndarray, Labels]]


# ---------------------------------------------------------------------------
# channel and room
# ---------------------------------------------------------------------------


def noise(x: np.ndarray, sr: int, rng: np.random.Generator, snr_range=(0.0, 40.0), kinds=("white", "pink", "babble")) -> tuple[np.ndarray, Labels]:
    kind = str(rng.choice(kinds))
    snr = float(rng.uniform(*snr_range))
    if kind == "babble":
        n = np.zeros(len(x))
        for _ in range(6):  # modulated band-limited noise streams ≈ distant speech-like babble
            b = signal.sosfilt(signal.butter(2, [200, 3500], "band", fs=sr, output="sos"), rng.standard_normal(len(x)))
            env = np.abs(signal.sosfilt(signal.butter(1, 4, "low", fs=sr, output="sos"), rng.standard_normal(len(x))))
            n += b * env
        y = degrade.add_noise(x, snr, noise=n, seed=int(rng.integers(1 << 31)))
    else:
        y = degrade.add_noise(x, snr, noise=kind, seed=int(rng.integers(1 << 31)))
    return y, {"noise": kind, "snr_db": snr}


def reverb(x: np.ndarray, sr: int, rng: np.random.Generator, t60_range=(0.15, 1.2), drr_range=(-3.0, 12.0),
           rirs: list[np.ndarray] | None = None) -> tuple[np.ndarray, Labels]:
    if rirs:
        k = int(rng.integers(len(rirs)))
        return degrade.reverberate(x, rirs[k]), {"room": "measured", "rir_index": k}
    t60, drr = float(rng.uniform(*t60_range)), float(rng.uniform(*drr_range))
    return degrade.reverberate(x, degrade.synthetic_rir(sr, t60, drr, seed=int(rng.integers(1 << 31)))), {"room": "synthetic", "t60_s": t60, "drr_db": drr}


def device_eq(x: np.ndarray, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, Labels]:
    """Random consumer-device response: high-pass corner, low-pass, tilt and peaks."""
    hp = float(rng.uniform(40, 400))
    lp = float(min(rng.uniform(4000, 20000), sr / 2 * 0.95))
    tilt = float(rng.uniform(-6, 6))  # dB per octave around 1 kHz
    freqs = np.geomspace(20, sr / 2 * 0.99, 64)
    gains = tilt * np.log2(freqs / 1000.0)
    for _ in range(int(rng.integers(0, 4))):
        fc, g, q = rng.uniform(300, 8000), rng.uniform(-6, 6), rng.uniform(1, 4)
        gains += g / (1 + (q * np.log2(freqs / fc)) ** 2)
    gains -= 40 * np.clip(np.log2(hp / freqs), 0, None)  # 12 dB/oct-ish below the corner
    gains -= 40 * np.clip(np.log2(freqs / lp), 0, None)
    gains -= np.interp(1000, freqs, gains)
    y = degrade.device_response(x, sr, list(freqs), list(gains), n_taps=1025)
    return y, {"eq_highpass_hz": hp, "eq_lowpass_hz": lp, "eq_tilt_db_oct": tilt}


def simulated_codec(x: np.ndarray, sr: int, cutoff_hz: float, step_db: float, rng: np.random.Generator) -> np.ndarray:
    """Codec-like degradation without ffmpeg: band limit + per-frame spectral
    quantisation relative to the frame peak (creates holes and musical noise)."""
    y = degrade.bandlimit(x, sr, min(cutoff_hz, sr / 2 * 0.95))
    f, t, Z = signal.stft(y, sr, nperseg=1024)
    mag = np.abs(Z)
    ref = mag.max(axis=0, keepdims=True) + EPS
    db = 20 * np.log10(mag / ref + EPS)
    floor = -step_db * 6
    q = np.where(db < floor, 0.0, 10 ** ((np.round(db / step_db) * step_db) / 20)) * ref
    _, y2 = signal.istft(q * np.exp(1j * np.angle(Z)), sr, nperseg=1024)
    return y2[: len(x)]


def codec(x: np.ndarray, sr: int, rng: np.random.Generator, use_ffmpeg: bool = True) -> tuple[np.ndarray, Labels]:
    if use_ffmpeg and shutil.which("ffmpeg"):
        fmt = str(rng.choice(["mp3", "aac", "opus"]))
        kbps = int(rng.choice({"mp3": [32, 64, 128], "aac": [48, 96, 128], "opus": [16, 24, 32]}[fmt]))
        return degrade.codec(x, sr, fmt, kbps), {"codec": fmt, "bitrate_kbps": kbps}
    cutoff = float(rng.choice([4000, 7000, 11000, 16000]))
    step = float(rng.uniform(1.5, 6.0))
    return simulated_codec(x, sr, cutoff, step, rng), {"codec": "simulated", "cutoff_hz": cutoff, "quant_step_db": step}


def compression(x: np.ndarray, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, Labels]:
    """Feed-forward RMS compressor (threshold re signal peak, ratio, attack/release)."""
    thr = float(rng.uniform(-30, -10))
    ratio = float(rng.uniform(2, 10))
    att, rel = float(rng.uniform(0.002, 0.02)), float(rng.uniform(0.05, 0.3))
    peak_db = 20 * np.log10(np.max(np.abs(x)) + EPS)
    a_a, a_r = np.exp(-1 / (att * sr)), np.exp(-1 / (rel * sr))
    lvl = signal.lfilter([1 - a_r], [1, -a_r], x**2)  # smoothed power (release-dominated)
    fast = signal.lfilter([1 - a_a], [1, -a_a], x**2)
    env_db = 10 * np.log10(np.maximum(lvl, fast) + EPS)
    over = np.maximum(env_db - (peak_db + thr), 0.0)
    gain_db = -over * (1 - 1 / ratio)
    y = x * 10 ** (gain_db / 20)
    y *= (np.max(np.abs(x)) + EPS) / (np.max(np.abs(y)) + EPS)  # make-up gain to the original peak
    return y, {"compression": True, "threshold_db": thr, "ratio": ratio}


def agc(x: np.ndarray, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, Labels]:
    rel = float(rng.uniform(0.3, 2.0))
    return degrade.agc(x, sr, target_db=-20.0, release_s=rel), {"agc": True, "agc_release_s": rel}


# ---------------------------------------------------------------------------
# production and transport
# ---------------------------------------------------------------------------


def separation_artifacts(x: np.ndarray, sr: int, rng: np.random.Generator, accompaniment: np.ndarray | None = None) -> tuple[np.ndarray, Labels]:
    """Mask-based separation artefacts.

    A soft mask is estimated as if the vocal had been separated from a mix
    (random per-bin reliability), producing musical noise and spectral
    holes; optionally residual accompaniment leaks through.
    """
    f, t, Z = signal.stft(x, sr, nperseg=2048)
    mag = np.abs(Z)
    strength = float(rng.uniform(0.1, 0.6))
    mask = np.clip(1 - strength * rng.random(Z.shape) ** 2, 0, 1)
    rel = 20 * np.log10(mag / (mag.max(axis=0, keepdims=True) + EPS) + EPS)
    holes = rel < -float(rng.uniform(25, 45))  # low-level bins (breath, inter-harmonic noise) get carved out
    mask[holes] *= 0.3
    _, y = signal.istft(Z * mask, sr, nperseg=2048)
    y = y[: len(x)]
    labels: Labels = {"separation": "mask", "mask_strength": strength}
    if accompaniment is not None:
        leak_db = float(rng.uniform(-30, -10))
        acc = np.resize(accompaniment, len(x))
        y = degrade.mix_accompaniment(y, acc, -leak_db)
        labels.update(separation="mask+bleed", bleed_db=leak_db)
    return y, labels


def bluetooth_jitter(x: np.ndarray, sr: int, rng: np.random.Generator, base_latency_s=(0.1, 0.3), wander_ms=(5.0, 40.0),
                     loss_rate=(0.0, 0.02)) -> tuple[np.ndarray, Labels]:
    """Constant route latency + slowly wandering delay + packet loss (10 ms frames)."""
    n = len(x)
    lat = float(rng.uniform(*base_latency_s))
    wander = float(rng.uniform(*wander_ms)) / 1000.0
    period = float(rng.uniform(2.0, 8.0))
    t = np.arange(n) / sr
    delay = lat + wander * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
    src = t - delay
    y = np.interp(src, t, x, left=0.0, right=0.0)
    p = float(rng.uniform(*loss_rate))
    pkt = int(0.01 * sr)
    lost = rng.random(n // pkt + 1) < p
    for i in np.flatnonzero(lost):
        y[i * pkt : (i + 1) * pkt] = 0.0
    return y, {"bt_latency_s": lat, "bt_wander_s": wander, "bt_loss_rate": p}


# ---------------------------------------------------------------------------
# voice: timbre and pitch
# ---------------------------------------------------------------------------


def _envelope(logmag: np.ndarray, n_ceps: int, iterations: int = 20) -> np.ndarray:
    """True envelope (Röbel & Rodet) of the log-magnitude per frame (rows = freq bins):
    iterative cepstral smoothing that rises to the harmonic peaks instead of
    averaging peaks and valleys."""
    nfft = 2 * (logmag.shape[0] - 1)
    target = logmag.copy()
    env = logmag
    for _ in range(iterations):
        c = np.fft.irfft(target, nfft, axis=0)
        c[n_ceps : nfft - n_ceps + 1] = 0
        env = np.fft.rfft(c, nfft, axis=0).real
        target = np.maximum(logmag, env)
    return env


def _f0_ceiling(x: np.ndarray, sr: int, default: float = 500.0) -> float:
    from ..pitch.adapters import YinTracker

    r = YinTracker().track(x, sr)
    if not r.ok or not np.isfinite(r.value.f0_hz).any():
        return default
    return float(np.nanpercentile(r.value.f0_hz, 95))


def warp_envelope(x: np.ndarray, sr: int, alpha: float, f0_max: float | None = None, nperseg: int = 2048) -> np.ndarray:
    """Scale the spectral envelope along frequency by ``alpha`` (> 1 = brighter /
    shorter vocal tract), keeping the harmonic fine structure (and f0).

    The true-envelope order is 0.5·sr/f0_max (its optimum): high enough to
    carry sharp formant peaks, low enough not to follow the harmonics.  When
    ``f0_max`` is not given it is the 95th percentile of a YIN track.
    """
    f0_max = f0_max or _f0_ceiling(x, sr)
    f, t, Z = signal.stft(x, sr, nperseg=nperseg)
    lm = np.log(np.abs(Z) + 1e-9)
    env = _envelope(lm, max(8, int(0.5 * sr / f0_max)))
    src = np.clip(f / alpha, 0, f[-1])
    warped = np.stack([np.interp(src, f, env[:, j]) for j in range(env.shape[1])], axis=1)
    _, y = signal.istft(Z * np.exp(warped - env), sr, nperseg=nperseg)
    y = y[: len(x)]
    return y * (np.std(x) / (np.std(y) + EPS))


def timbre_shift(x: np.ndarray, sr: int, rng: np.random.Generator, alpha_range=(0.85, 1.18)) -> tuple[np.ndarray, Labels]:
    a = float(rng.uniform(*alpha_range))
    return warp_envelope(x, sr, a), {"timbre_alpha": a}


def _phase_vocoder(x: np.ndarray, rate: float, nperseg: int = 2048, hop: int = 512) -> np.ndarray:
    """Time-stretch by ``rate`` (> 1 = shorter) with a standard phase vocoder."""
    win = np.hanning(nperseg)
    n_frames = 1 + max(0, len(x) - nperseg) // hop
    frames = np.stack([np.fft.rfft(win * x[i * hop : i * hop + nperseg]) for i in range(n_frames)])
    steps = np.arange(0, n_frames - 1, rate)
    omega = 2 * np.pi * hop * np.arange(nperseg // 2 + 1) / nperseg
    phase = np.angle(frames[0])
    out = np.zeros(int(len(steps) * hop + nperseg))
    for k, s in enumerate(steps):
        i = int(s)
        frac = s - i
        mag = (1 - frac) * np.abs(frames[i]) + frac * np.abs(frames[i + 1])
        seg = np.fft.irfft(mag * np.exp(1j * phase), nperseg) * win
        out[k * hop : k * hop + nperseg] += seg
        dphi = np.angle(frames[i + 1]) - np.angle(frames[i]) - omega
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        phase += omega + dphi
    return out / (np.sum(win**2) / hop)


def pitch_shift(x: np.ndarray, sr: int, semitones: float, keep_formants: bool = True) -> np.ndarray:
    """Shift f0 by ``semitones`` keeping duration (and formants, by default)."""
    ratio = 2 ** (semitones / 12)
    stretched = _phase_vocoder(np.asarray(x, float), 1.0 / ratio)
    idx = np.arange(0, len(stretched), ratio)
    y = np.interp(idx, np.arange(len(stretched)), stretched)
    y = np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]
    if keep_formants:
        y = warp_envelope(y, sr, 1.0 / ratio, f0_max=_f0_ceiling(y, sr))
    return y * (np.std(x) / (np.std(y) + EPS))


def pitch_shift_random(x: np.ndarray, sr: int, rng: np.random.Generator, semitone_range=(-3.0, 3.0)) -> tuple[np.ndarray, Labels]:
    s = float(rng.uniform(*semitone_range))
    return pitch_shift(x, sr, s), {"pitch_shift_semitones": s}


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

DEFAULT_CHAIN: tuple[tuple[str, Transform, float], ...] = (
    ("timbre", timbre_shift, 0.3),
    ("pitch", pitch_shift_random, 0.2),
    ("eq", device_eq, 0.6),
    ("reverb", reverb, 0.5),
    ("noise", noise, 0.6),
    ("compression", compression, 0.3),
    ("codec", codec, 0.4),
    ("separation", separation_artifacts, 0.3),
)


@dataclass
class AugmentationPipeline:
    """Random chain of transforms, each applied with its probability."""

    chain: tuple[tuple[str, Transform, float], ...] = DEFAULT_CHAIN
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, x: np.ndarray, sr: int) -> tuple[np.ndarray, Labels]:
        y = np.asarray(x, float)
        labels: Labels = {"applied": []}
        for name, fn, p in self.chain:
            if self._rng.random() < p:
                y, lab = fn(y, sr, self._rng)
                labels.update(lab)
                labels["applied"].append(name)  # type: ignore[union-attr]
        peak = np.max(np.abs(y)) + EPS
        if peak > 0.99:
            y = y * 0.99 / peak
        return y, labels


#: coarse class labels for the env encoder's adversarial / supervised heads
def env_classes(labels: Labels) -> dict[str, int]:
    t60 = labels.get("t60_s")
    room = 0 if "room" not in labels else (1 if (t60 or 0) < 0.4 else 2 if (t60 or 0) < 0.8 else 3)
    codec_names = ["none", "mp3", "aac", "opus", "simulated"]
    return {
        "noise": ["none", "white", "pink", "babble"].index(labels.get("noise", "none")),
        "room": room,
        "eq": int("eq_highpass_hz" in labels),
        "codec": codec_names.index(labels.get("codec", "none")),
        "compression": int(bool(labels.get("compression"))),
        "separation": int("separation" in labels),
    }

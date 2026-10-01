"""Input quality checks.

Fixes relative to v0.1:

* **Clipping is measured on the raw input** (defect 4).  v0.1 high-passed
  the signal first, which turns flat clipped runs into slopes and hides them.
* **Effective bandwidth** uses a *minimum-statistics* noise floor per
  frequency bin (a low percentile over time) instead of the mean spectrum of
  "pauses" (defect 1).  Breath, consonants and reverb tails are transient,
  so they no longer masquerade as a noise floor that swallows the voice band.
* **SNR** compares voiced-frame level with the same stationary floor.
* **Bleed**: headphone leakage of the backing track into the microphone,
  estimated by least-squares prediction of the mic signal from the track.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from ..core.grid import FrameGrid
from ..core.status import Result

EPS = 1e-12


@dataclass
class ClippingReport:
    fraction: float  # fraction of samples inside clipped runs
    n_runs: int
    longest_run: int
    at_full_scale: bool


def detect_clipping(raw: np.ndarray, min_run: int = 3, tolerance: float = 1e-4) -> ClippingReport:
    """Flat runs of ≥ ``min_run`` samples at the positive or negative extreme.

    Must be given the *raw* signal (no filtering, no DC removal): clipping
    happens at the converter, so it shows up as runs of identical extreme
    values in the file.
    """
    x = np.asarray(raw, dtype=float)
    if x.size == 0:
        return ClippingReport(0.0, 0, 0, False)
    hi, lo = x.max(), x.min()
    span = max(hi - lo, EPS)
    hot = (x >= hi - tolerance * span) | (x <= lo + tolerance * span)
    d = np.diff(np.r_[0, hot.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    lengths = ends - starts
    keep = lengths >= min_run
    n_clipped = int(lengths[keep].sum())
    return ClippingReport(
        fraction=n_clipped / len(x),
        n_runs=int(keep.sum()),
        longest_run=int(lengths.max()) if len(lengths) else 0,
        at_full_scale=bool(max(abs(hi), abs(lo)) >= 0.999),
    )


@dataclass
class Spectrogram:
    power: np.ndarray  # (F, N)
    freqs: np.ndarray
    times: np.ndarray


def stft_power(x: np.ndarray, sr: int, nperseg: int = 2048) -> Spectrogram:
    f, t, z = signal.stft(np.asarray(x, float), sr, nperseg=nperseg, noverlap=nperseg // 2, boundary=None, padded=False)
    return Spectrogram(np.abs(z) ** 2, f, t)


def noise_floor(spec: Spectrogram, percentile: float = 10.0) -> np.ndarray:
    """Stationary noise power per bin: low percentile over time (minimum statistics)."""
    return np.percentile(spec.power, percentile, axis=1) + EPS


def _smooth_sixth(f: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Mean over ±1/12 octave around each bin."""
    lo = np.searchsorted(f, f * 2 ** (-1 / 12))
    hi = np.maximum(np.searchsorted(f, f * 2 ** (1 / 12), side="right"), lo + 1)
    c = np.r_[0.0, np.cumsum(v)]
    return (c[hi] - c[lo]) / (hi - lo)


@dataclass
class BandwidthReport:
    bandwidth_hz: float
    cliff: bool  # steep low-pass (codec / resampling evidence)


def effective_bandwidth(x: np.ndarray, sr: int, margin_db: float = 10.0, percentile: float = 90.0) -> Result[BandwidthReport]:
    """Highest frequency the *channel* carries above its stationary floor.

    Bandwidth is a property of the recording chain, so every active frame
    counts as evidence — voiced notes, consonants and breath alike (v0.1
    compared voiced frames against "pauses" and so mistook consonants,
    breath and reverb for noise).  Per bin, the ``percentile``-th power over
    active frames is compared with the minimum-statistics floor.
    """
    spec = stft_power(x, sr)
    if spec.power.shape[1] < 10:
        return Result.failure("recording too short for a bandwidth estimate (< 10 STFT frames)")
    floor = noise_floor(spec)
    e_db = 10 * np.log10(spec.power.sum(axis=0) + EPS)
    active = e_db >= np.percentile(e_db, 10) + 10.0
    if active.sum() < 3:
        return Result.unreliable(BandwidthReport(0.0, False), "fewer than 3 frames rise 10 dB above the floor")
    voice = np.percentile(spec.power[:, active], percentile, axis=1)
    f = spec.freqs
    # smooth in the power domain: harmonic peaks, not inter-harmonic valleys, carry the band
    v_db = 10 * np.log10(_smooth_sixth(f, voice) + EPS)
    n_db = 10 * np.log10(_smooth_sixth(f, floor) + EPS)
    cliff = False
    bw = float(sr / 2)
    for i in np.flatnonzero(f > 3000):
        j = np.searchsorted(f, f[i] * 2 ** (1 / 3))
        if j < len(f) - 3 and v_db[i] - v_db[j] > 30 and v_db[j:].max() < v_db[i] - 25:
            # the steep drop starts at i; report where it has fallen 6 dB
            k = i + int(np.argmax(v_db[i:] < v_db[i] - 6.0))
            bw, cliff = float(f[k]), True
            break
    above = np.flatnonzero((v_db > n_db + margin_db) & (f > 0))
    if above.size == 0:
        return Result.unreliable(BandwidthReport(0.0, cliff), "voice never exceeds the stationary floor")
    bw = min(bw, float(f[above[-1]]))
    return Result.success(BandwidthReport(bw, cliff))


@dataclass
class SNRReport:
    snr_db: float
    floor_db: float
    voice_db: float
    frame_snr_db: np.ndarray  # on the analysis grid


def estimate_snr(x: np.ndarray, sr: int, grid: FrameGrid, voiced: np.ndarray, win_seconds: float = 0.04) -> Result[SNRReport]:
    """Voiced level vs stationary floor (both from frame energies on ``grid``)."""
    x = np.asarray(x, float)
    win = int(win_seconds * sr)
    centres = np.round(grid.times() * sr).astype(int)
    xp = np.pad(x, (win // 2, win))
    e = np.mean(xp[centres[:, None] + np.arange(win)[None, :]] ** 2, axis=1)
    db = 10 * np.log10(e + EPS)
    if not voiced.any():
        return Result.failure("no voiced frames")
    floor = float(np.percentile(db, 5)) if (~voiced).sum() < 5 else float(np.percentile(db[~voiced], 10))
    floor = max(floor, float(np.percentile(db, 1)))
    voice = float(np.median(db[voiced]))
    # digital silence would make the ratio meaningless; cap it
    return Result.success(SNRReport(min(voice - floor, 100.0), floor, voice, db - floor))


@dataclass
class BleedReport:
    bleed_db: float  # predicted-leakage energy relative to the mic signal
    lag_s: float
    coherence: float


def detect_bleed(mic: np.ndarray, backing: np.ndarray, sr: int, max_lag_s: float = 0.5, nperseg: int = 2048) -> Result[BleedReport]:
    """How much of the backing track leaks into the microphone.

    The lag comes from PHAT-weighted cross-correlation.  The leaked fraction
    of mic power is the power-weighted magnitude-squared coherence γ², minus
    the same statistic against a decorrelated (circularly shifted) copy of
    the backing — an empirical estimate of the finite-sample bias — so an
    uncorrelated voice does not read as leakage.  Leakage below ≈ −30 dB
    cannot be resolved (it reads as ≈ −30 dB), which is well under the
    default flag threshold of −20 dB.
    """
    mic = np.asarray(mic, float)
    b = np.asarray(backing, float)
    n = min(len(mic), len(b))
    if n < nperseg * 8:
        return Result.failure("signals too short for bleed estimation")
    mic, b = mic[:n], b[:n]
    max_lag = int(max_lag_s * sr)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    R = np.fft.rfft(mic, nfft) * np.conj(np.fft.rfft(b, nfft))
    cc = np.fft.irfft(R / (np.abs(R) + EPS), nfft)
    lag = int(np.argmax(cc[: max_lag + 1]))
    b_al = np.r_[np.zeros(lag), b[: n - lag]]
    _, pmm = signal.welch(mic, sr, nperseg=nperseg)

    def leaked(ref: np.ndarray) -> tuple[float, np.ndarray]:
        _, coh = signal.coherence(mic, ref, sr, nperseg=nperseg)
        return float(np.sum(coh * pmm) / (np.sum(pmm) + EPS)), coh

    frac, coh = leaked(b_al)
    # empirical bias floor: the same statistic against a decorrelated copy
    null, _ = leaked(np.roll(b_al, n // 2))
    frac = max(frac - null, 0.0)
    return Result.success(BleedReport(10 * np.log10(frac + 1e-9), lag / sr, float(np.mean(coh))))


@dataclass
class QualityPolicy:
    """Thresholds for input flags.  Defaults are conservative starting points;
    replace them with operating thresholds from :mod:`gyeol.verification`."""

    max_clipping_fraction: float = 1e-3
    min_snr_db: float = 30.0
    min_bandwidth_hz: float = 7000.0
    max_bleed_db: float = -20.0


@dataclass
class QualityReport:
    clipping: ClippingReport
    snr: SNRReport | None
    bandwidth: BandwidthReport | None
    bleed: BleedReport | None = None
    flags: dict[str, str] = field(default_factory=dict)  # flag -> reason
    #: per-frame multiplier (0..1) applied to noise-sensitive attribute confidences
    frame_factor: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return not self.flags


def assess(raw: np.ndarray, sr: int, grid: FrameGrid, voiced: np.ndarray, *, backing: np.ndarray | None = None,
           policy: QualityPolicy | None = None, analysis: np.ndarray | None = None) -> QualityReport:
    """Run every check and collect flags.

    ``raw`` must be the unprocessed input: clipping is always measured on it.
    ``analysis`` is the signal that is actually analysed (e.g. the separated
    vocal stem); SNR, bandwidth and backing-track bleed are measured on it.
    """
    pol = policy or QualityPolicy()
    sig = raw if analysis is None else np.asarray(analysis, float)
    clip = detect_clipping(raw)
    snr_r = estimate_snr(sig - np.mean(sig), sr, grid, voiced)
    bw_r = effective_bandwidth(sig, sr)
    bleed_r = detect_bleed(sig, backing, sr) if backing is not None else None
    rep = QualityReport(clip, snr_r.value if snr_r.usable else None, bw_r.value if bw_r.usable else None,
                        bleed_r.value if (bleed_r is not None and bleed_r.usable) else None)
    if clip.fraction > pol.max_clipping_fraction:
        rep.flags["clipping"] = f"{clip.fraction * 100:.2f}% of samples in clipped runs"
    if rep.snr is None:
        rep.flags["snr_unknown"] = snr_r.reason
    elif rep.snr.snr_db < pol.min_snr_db:
        rep.flags["low_snr"] = f"SNR {rep.snr.snr_db:.1f} dB < {pol.min_snr_db:.0f} dB"
    if rep.bandwidth is None:
        rep.flags["bandwidth_unknown"] = bw_r.reason
    else:
        if rep.bandwidth.bandwidth_hz < pol.min_bandwidth_hz:
            rep.flags["narrow_bandwidth"] = f"effective bandwidth {rep.bandwidth.bandwidth_hz:.0f} Hz"
        if rep.bandwidth.cliff:
            rep.flags["codec_suspected"] = "steep low-pass cliff"
    if rep.bleed is not None and rep.bleed.bleed_db > pol.max_bleed_db:
        rep.flags["bleed"] = f"backing-track leakage {rep.bleed.bleed_db:.1f} dB"
    # frame factor: 1 at ≥ min_snr, falling linearly to 0 at min_snr − 20 dB
    if rep.snr is not None:
        rep.frame_factor = np.clip((rep.snr.frame_snr_db - (pol.min_snr_db - 20.0)) / 20.0, 0.0, 1.0)
    else:
        rep.frame_factor = np.zeros(grid.n_frames)
    return rep


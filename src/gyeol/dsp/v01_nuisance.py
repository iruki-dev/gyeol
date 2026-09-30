"""Blind nuisance estimation: the side channel N̂.

These estimates never enter T_voice; they only drive validity masks.

Implemented estimators (all deliberately simple and replaceable):

* SNR: voiced-frame level vs noise floor from pauses (Deliyski-style VNR),
  plus a per-frame SNR for local gating.
* Effective bandwidth: highest frequency where the long-term average
  spectrum stays above the recording's noise floor; a steep cliff below
  Nyquist is evidence of a lossy codec / resampling low-pass.
* Clipping fraction.
* AGC pumping: noise-floor level in pauses anti-correlated with the vocal
  level that preceded them (gain recovering after loud singing).
* Noise-gate / spectral-gating suspicion: pauses at digital silence.
* T60: slope of free energy decays after phrase offsets.
  *Heuristic — replace with an ACE-benchmarked estimator for production.*

DRR / C50 need a trained blind estimator (ACE Challenge, Eaton et al. 2016);
they are left ``None`` unless supplied via :class:`NuisanceEstimator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy import signal

from .base import EPS, frame, power_db, runs
from .v01_report import NuisanceReport

LOSSY_EXTENSIONS = {".mp3", ".aac", ".m4a", ".ogg", ".opus", ".webm", ".amr", ".3gp", ".wma"}


class NuisanceEstimator(Protocol):
    """Hook for trained estimators (T60/DRR/C50, codec classifier, SDR predictor)."""

    def __call__(self, x: np.ndarray, sr: int, report: NuisanceReport) -> None: ...


@dataclass
class LevelStats:
    frame_db: np.ndarray
    noise_floor_db: float
    signal_db: float


def level_stats(x: np.ndarray, sr: int, hop: int, n_frames: int, voiced: np.ndarray) -> LevelStats:
    win = int(0.04 * sr)
    db = power_db(np.mean(frame(x, win, hop, n_frames) ** 2, axis=1), floor_db=-150.0)
    unvoiced = ~voiced
    # pauses: unvoiced frames not adjacent to voiced ones (avoid consonants / tails)
    pad = int(round(0.05 * sr / hop))
    near_voice = np.convolve(voiced.astype(float), np.ones(2 * pad + 1), mode="same") > 0
    pause = unvoiced & ~near_voice
    pool = db[pause] if pause.sum() >= 5 else db[unvoiced] if unvoiced.sum() >= 5 else db
    # guard against digitally silent padding dominating the floor
    pool = pool[pool > -140] if (pool > -140).sum() >= 5 else pool
    floor = float(np.percentile(pool, 20))
    sig = float(np.median(db[voiced])) if voiced.any() else float(np.max(db))
    return LevelStats(frame_db=db, noise_floor_db=floor, signal_db=sig)


def effective_bandwidth(
    x: np.ndarray,
    sr: int,
    voiced_times: np.ndarray | None = None,
    hop_seconds: float = 0.01,
    margin_db: float = 10.0,
) -> tuple[float, bool]:
    """(bandwidth Hz, low-pass cliff detected).

    Bandwidth is the highest frequency at which the voiced-frame LTAS stays
    ``margin_db`` above the pause-frame (noise) LTAS, both smoothed over
    1/6 octave.  Independently, a drop of more than 30 dB within 1/3 octave
    above 3 kHz with nothing recovering above it is flagged as a low-pass
    cliff (codec / resampling evidence) — far steeper than any natural
    voice roll-off.
    """
    nper = 2048 if sr > 24000 else 1024
    f, t, z = signal.stft(x, sr, nperseg=nper, noverlap=nper // 2, boundary=None, padded=False)
    power = np.abs(z) ** 2
    if power.shape[1] == 0:
        return float(sr / 2), False
    if voiced_times is not None and len(voiced_times):
        idx = np.clip(np.round(t / hop_seconds).astype(int), 0, len(voiced_times) - 1)
        v = voiced_times[idx]
    else:
        v = np.ones(power.shape[1], bool)
    voice = power[:, v].mean(axis=1) if v.any() else power.mean(axis=1)
    ltas = _smooth_sixth(f, power_db(voice))
    cliff = False
    bw = float(sr / 2)
    for i in np.flatnonzero(f > 3000):
        j = np.searchsorted(f, f[i] * 2 ** (1 / 3))
        if j < len(f) - 3 and ltas[i] - ltas[j] > 30 and ltas[j:].max() < ltas[i] - 25:
            # report the −10 dB point relative to the level just below the cliff
            lo = np.searchsorted(f, f[i] * 2 ** (-1 / 3))
            ref = ltas[lo : i + 1].max()
            k = lo + int(np.argmax(ltas[lo:] < ref - 10))
            bw, cliff = float(f[k]), True
            break
    if (~v).sum() >= 3:
        noise = _smooth_sixth(f, power_db(power[:, ~v].mean(axis=1)))
        above = np.flatnonzero((ltas > noise + margin_db) & (f > 0))
        bw = min(bw, float(f[above[-1]])) if above.size else 0.0
    return bw, cliff


def _smooth_sixth(f: np.ndarray, db: np.ndarray) -> np.ndarray:
    out = np.empty_like(db)
    lo = np.searchsorted(f, f * 2 ** (-1 / 12))
    hi = np.searchsorted(f, f * 2 ** (1 / 12), side="right")
    c = np.concatenate([[0.0], np.cumsum(db)])
    for i in range(len(f)):
        a, b = lo[i], max(hi[i], lo[i] + 1)
        out[i] = (c[b] - c[a]) / (b - a)
    return out


def clipping_fraction(x: np.ndarray, rel: float = 0.9999, min_run: int = 3) -> float:
    """Fraction of samples in flat runs (>= min_run) at the peak level."""
    peak = np.max(np.abs(x)) + EPS
    hot = np.abs(x) >= rel * peak
    n = sum(e - s for s, e in runs(hot) if e - s >= min_run)
    return float(n / max(len(x), 1))


def agc_suspected(db: np.ndarray, voiced: np.ndarray, hop_seconds: float) -> bool:
    """Noise floor rising in pauses that follow loud phrases -> gain pumping."""
    pairs = []
    fs = 1.0 / hop_seconds
    for s, e in runs(~voiced):
        if e - s < int(0.3 * fs) or s < int(0.5 * fs):
            continue
        prev = db[max(0, s - int(1.0 * fs)) : s][voiced[max(0, s - int(1.0 * fs)) : s]]
        if prev.size < 5:
            continue
        pause = db[s:e]
        early = np.median(pause[: len(pause) // 3 + 1])
        late = np.median(pause[-(len(pause) // 3 + 1) :])
        pairs.append((np.median(prev), late - early))
    if len(pairs) < 3:
        return False
    lvl, recovery = np.array(pairs).T
    if np.std(lvl) < 1e-6 or np.std(recovery) < 1e-6:
        return False
    r = np.corrcoef(lvl, recovery)[0, 1]
    return bool(r > 0.6 and np.max(recovery) > 6.0)


def noise_gate_suspected(db: np.ndarray, voiced: np.ndarray) -> bool:
    pauses = db[~voiced]
    if pauses.size < 10 or not voiced.any():
        return False
    return bool(np.mean(pauses < -100) > 0.3 and np.median(db[voiced]) > -60)


def blind_t60(db: np.ndarray, voiced: np.ndarray, hop_seconds: float, min_decay_db: float = 15.0) -> float | None:
    """T60 from free decays after phrase offsets (heuristic).

    For each voiced→unvoiced transition, the energy decay over the next
    ≤ 400 ms is fitted with a line between −5 dB and −(5+min_decay) dB below
    the offset level (a T-style evaluation range); T60 = −60 / slope.
    """
    fs = 1.0 / hop_seconds
    estimates = []
    for _, e in runs(voiced):
        if e >= len(db) - 3:
            continue
        seg = db[e - 1 : min(len(db), e + int(0.4 * fs))]
        if len(seg) < 5:
            continue
        ref = seg[0]
        rng = (seg <= ref - 5) & (seg >= ref - 5 - min_decay_db)
        idx = np.flatnonzero(rng)
        if idx.size < 3:
            continue
        slope = np.polyfit(idx * hop_seconds, seg[idx], 1)[0]
        if slope < -1:
            estimates.append(-60.0 / slope)
    if len(estimates) < 2:
        return None
    return float(np.median(estimates))


def estimate(
    x_native: np.ndarray,
    sr_native: int,
    x_work: np.ndarray,
    sr_work: int,
    hop: int,
    n_frames: int,
    voiced: np.ndarray,
    source_path: str | Path | None = None,
    extra: list[NuisanceEstimator] | None = None,
) -> tuple[NuisanceReport, np.ndarray]:
    """Build the side channel.  Returns (report, per-frame SNR in dB)."""
    rep = NuisanceReport(native_sample_rate=int(sr_native))
    lv = level_stats(x_work, sr_work, hop, n_frames, voiced)
    rep.noise_floor_db = lv.noise_floor_db
    rep.snr_db = lv.signal_db - lv.noise_floor_db
    frame_snr = lv.frame_db - lv.noise_floor_db
    bw, cliff = effective_bandwidth(x_native, sr_native, voiced, hop / sr_work)
    rep.effective_bandwidth_hz = bw
    if cliff:
        rep.codec_suspected = True
        rep.codec_evidence.append(f"low-pass cliff at {bw:.0f} Hz")
    if source_path is not None and Path(source_path).suffix.lower() in LOSSY_EXTENSIONS:
        rep.codec_suspected = True
        rep.codec_evidence.append(f"lossy container {Path(source_path).suffix.lower()}")
    rep.clipping_fraction = clipping_fraction(x_native)
    hop_s = hop / sr_work
    rep.agc_suspected = agc_suspected(lv.frame_db, voiced, hop_s)
    rep.noise_gate_suspected = noise_gate_suspected(lv.frame_db, voiced)
    rep.t60_s = blind_t60(lv.frame_db, voiced, hop_s)
    rep.notes.append("T60 is a heuristic free-decay estimate; DRR/C50 require a trained estimator")
    for fn in extra or []:
        fn(x_native, sr_native, rep)
    return rep, frame_snr

"""Tracker consensus with octave-error repair.

v0.1 took the median of the trackers and turned their spread into confidence,
so a single tracker's octave error collapsed confidence (defect 2), and
period-doubled "rasp" voices were followed an octave low (defect 6).

v2 works on candidates:

1. every tracker's f0, *and its octave neighbours* (×½, ×2), is a candidate;
2. each candidate f gets a harmonic salience from the spectrum: the mean
   log-magnitude at k·f minus the mean at the half-integer positions
   (k − ½)·f.  For a true f0 the harmonics stand above the in-between
   positions; for f/2 of a rasp voice the "odd harmonics" are only weak
   subharmonics, so f wins (the subharmonic energy is reported separately);
3. a Viterbi pass over candidates adds temporal continuity (octave jumps
   are expensive) and tracker votes (octave-folded agreement counts);
4. confidence = voicing × octave-folded agreement × salience.  A tracker
   that is an octave off still supports the chosen pitch class, so one
   octave error no longer collapses confidence.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..core.grid import FrameGrid
from ..core.status import Result, Status
from ..dsp.base import EPS, resample
from .base import PitchTrack, PitchTracker

SPEC_SR = 16000


@dataclass
class ConsensusConfig:
    fmin: float = 55.0
    fmax: float = 1600.0
    agree_cents: float = 50.0
    #: tracker votes that agree only after octave folding count this much
    folded_vote: float = 0.8
    n_harmonics: int = 8
    #: 100 ms keeps quarter-harmonic positions outside the Hann main lobe down to ~60 Hz
    win_seconds: float = 0.1
    #: Viterbi cost per 100 cents of frame-to-frame change
    jump_cost: float = 1.0
    #: extra Viterbi cost for an octave jump between neighbouring frames
    octave_jump_cost: float = 4.0
    #: even-over-odd log-magnitude gap (natural log; ln 2 ≈ −6 dB) above which
    #: a candidate is treated as a subharmonic of its octave
    octave_gap: float = 0.7
    octave_gap_weight: float = 2.0
    #: subharmonic ratio above which the candidate's lower octave is the pitch
    max_subharmonic_ratio: float = 0.6
    voicing_threshold: float = 0.5
    min_salience: float = 0.05
    min_run_frames: int = 3


@dataclass
class PitchResult:
    grid: FrameGrid
    f0_hz: np.ndarray  # NaN where unvoiced
    voiced_prob: np.ndarray
    f0_conf: np.ndarray
    salience: np.ndarray
    subharmonic_ratio: np.ndarray  # energy at (k-½)f0 relative to k·f0, NaN unvoiced
    octave_repaired: np.ndarray  # bool: some tracker disagreed by an octave here
    tracks: list[PitchTrack] = field(default_factory=list)
    failed_trackers: dict[str, str] = field(default_factory=dict)

    @property
    def voiced(self) -> np.ndarray:
        return np.isfinite(self.f0_hz)

    @property
    def cents(self) -> np.ndarray:
        return 1200 * np.log2(self.f0_hz / 440.0)


def _spectra(audio: np.ndarray, sr: int, grid: FrameGrid, win_seconds: float) -> tuple[np.ndarray, float]:
    """Log-magnitude spectra (T, F) at 16 kHz with frames centred on grid times."""
    x = resample(np.asarray(audio, float), sr, SPEC_SR)
    win = int(win_seconds * SPEC_SR)
    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    centres = np.round(grid.times() * SPEC_SR).astype(int)
    xp = np.pad(x, (win // 2, win))
    idx = centres[:, None] + np.arange(win)[None, :]
    frames = xp[idx] * np.hanning(win)
    mag = np.abs(np.fft.rfft(frames, nfft, axis=1))
    ref = np.percentile(mag, 99.5) + EPS
    return np.log1p(mag / (1e-3 * ref)), SPEC_SR / nfft


def _salience(logmag: np.ndarray, df: float, f: np.ndarray, n_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per candidate f (T, C): (salience, subharmonic ratio, even-over-odd gap).

    * salience: harmonics k·f against quarter-offset positions (k − ¼)·f,
      which stay clear of both harmonics and half-integer subharmonics;
    * subharmonic ratio: linear magnitude at (k − ½)·f over k·f (roughness);
    * even-over-odd gap: mean log-magnitude of even minus odd multiples of f.
      A large gap means the odd multiples are only subharmonics, i.e. f is
      an octave-down artefact of 2f.
    """
    T, F = logmag.shape
    k = np.arange(1, n_h + 1)
    ok = np.isfinite(f)
    fs = np.where(ok, f, 100.0)

    def sample(freqs: np.ndarray) -> np.ndarray:  # (T, C, K), max over ±1 bin
        b = np.round(freqs / df).astype(int)
        rows = np.arange(T)[:, None, None]
        vals = [logmag[rows, np.clip(b + o, 0, F - 1)] for o in (-1, 0, 1)]
        return np.max(vals, axis=0)

    below = ((fs[..., None] * (k + 0.5)) < SPEC_SR / 2 - 200).astype(float)
    h = sample(fs[..., None] * k)
    q = sample(fs[..., None] * (k - 0.25))
    s = sample(fs[..., None] * (k - 0.5))
    nh = np.maximum(below.sum(-1), 1)
    mh = (h * below).sum(-1) / nh
    mq = (q * below).sum(-1) / nh
    ms = (s * below).sum(-1) / nh
    odd = below * (k % 2 == 1)
    even = below * (k % 2 == 0)
    m_odd = (h * odd).sum(-1) / np.maximum(odd.sum(-1), 1)
    m_even = (h * even).sum(-1) / np.maximum(even.sum(-1), 1)
    sal = np.where(ok, (mh - mq) / (mh + EPS), np.nan)
    sub = np.where(ok, np.expm1(ms) / (np.expm1(mh) + EPS), np.nan)
    gap = np.where(ok, m_even - m_odd, np.nan)
    return sal, sub, gap


def consensus(audio: np.ndarray, sr: int, grid: FrameGrid, trackers: Sequence[PitchTracker], config: ConsensusConfig | None = None) -> Result[PitchResult]:
    cfg = config or ConsensusConfig()
    tracks: list[PitchTrack] = []
    failed: dict[str, str] = {}
    for t in trackers:
        r = t.track(audio, sr)
        if r.ok:
            tracks.append(r.value)
        else:
            failed[t.name] = f"{r.status.value}: {r.reason}"
    if not tracks:
        return Result.failure(f"no pitch tracker succeeded: {failed}")
    T = grid.n_frames
    f0s, vps = zip(*(tr.on_grid(grid) for tr in tracks))
    F0 = np.stack(f0s, axis=1)  # (T, N)
    VP = np.stack(vps, axis=1)
    F0[(F0 < cfg.fmin * 0.9) | (F0 > cfg.fmax * 1.1)] = np.nan
    N = F0.shape[1]
    # candidates: each tracker estimate ×{½, 1, 2}
    cand = np.concatenate([F0 * 0.5, F0, F0 * 2.0], axis=1)  # (T, 3N)
    cand[(cand < cfg.fmin) | (cand > cfg.fmax)] = np.nan
    logmag, df = _spectra(audio, sr, grid, cfg.win_seconds)
    sal, sub, gap = _salience(logmag, df, cand, cfg.n_harmonics)
    cents = 1200 * np.log2(cand / 440.0)
    tc = 1200 * np.log2(F0 / 440.0)
    # tracker votes per candidate
    diff = np.abs(cents[:, :, None] - tc[:, None, :])  # (T, C, N)
    direct = (diff <= cfg.agree_cents).astype(float)
    folded = (np.abs(diff - 1200) <= cfg.agree_cents).astype(float)
    w = VP[:, None, :]
    votes = (np.nansum(direct * w, axis=2) + cfg.folded_vote * np.nansum(folded * w, axis=2)) / N
    # octave-down artefacts: odd multiples much weaker than even ones
    octave_penalty = cfg.octave_gap_weight * np.maximum(0.0, np.nan_to_num(gap) - cfg.octave_gap)
    # octave-up artefacts: the half-integer positions carry near-harmonic
    # energy, i.e. the odd harmonics of f/2 are real
    octave_penalty += cfg.octave_gap_weight * np.maximum(0.0, np.nan_to_num(sub) - cfg.max_subharmonic_ratio) / (1.0 - cfg.max_subharmonic_ratio)
    score = np.where(np.isfinite(cand), 2.0 * np.nan_to_num(sal) + votes - octave_penalty, -np.inf)
    # Viterbi over candidates within voiced runs
    choice = _viterbi(score, cents, cfg)
    rows = np.arange(T)
    has = choice >= 0
    ch = np.where(has, choice, 0)
    f0 = np.where(has, cand[rows, ch], np.nan)
    s_ch = np.where(has, sal[rows, ch], np.nan)
    sub_ch = np.where(has, sub[rows, ch], np.nan)
    # refine: mean of tracker estimates agreeing with the chosen candidate
    c0 = 1200 * np.log2(f0 / 440.0)
    agree = np.abs(tc - c0[:, None]) <= cfg.agree_cents
    agree_fold = np.abs(np.abs(tc - c0[:, None]) - 1200) <= cfg.agree_cents
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        refined = np.where(agree.any(1), np.nanmean(np.where(agree, tc, np.nan), axis=1), c0) if N else c0
    f0 = 440.0 * 2 ** (refined / 1200)
    voiced_prob = VP.mean(axis=1)
    voiced = has & (voiced_prob >= cfg.voicing_threshold) & (np.nan_to_num(s_ch) >= cfg.min_salience)
    voiced = _drop_short(voiced, cfg.min_run_frames)
    support = (agree.sum(1) + cfg.folded_vote * agree_fold.sum(1)) / max(N, 1)
    conf = np.where(voiced, np.clip(voiced_prob, 0, 1) * np.clip(support, 0, 1) * np.clip(np.nan_to_num(s_ch) / 0.2, 0, 1), 0.0)
    repaired = voiced & agree_fold.any(1)
    out = PitchResult(
        grid=grid,
        f0_hz=np.where(voiced, f0, np.nan),
        voiced_prob=voiced_prob,
        f0_conf=conf,
        salience=np.where(voiced, s_ch, np.nan),
        subharmonic_ratio=np.where(voiced, sub_ch, np.nan),
        octave_repaired=repaired,
        tracks=tracks,
        failed_trackers=failed,
    )
    if failed:
        return Result(Status.OK, out, "", [f"tracker failures: {failed}"])
    return Result.success(out)


def _viterbi(score: np.ndarray, cents: np.ndarray, cfg: ConsensusConfig) -> np.ndarray:
    """Best candidate index per frame (−1 where no finite candidate)."""
    T, C = score.shape
    choice = np.full(T, -1, dtype=int)
    valid = np.isfinite(score).any(axis=1)
    t = 0
    while t < T:
        if not valid[t]:
            t += 1
            continue
        s = t
        while t < T and valid[t]:
            t += 1
        seg = slice(s, t)
        sc, ce = score[seg], cents[seg]
        n = t - s
        acc = sc[0].copy()
        back = np.zeros((n, C), dtype=int)
        for i in range(1, n):
            d = np.abs(ce[i][None, :] - ce[i - 1][:, None])  # (prev, cur)
            d = np.where(np.isfinite(d), d, 1e4)
            cost = cfg.jump_cost * d / 100.0 + cfg.octave_jump_cost * (d > 600)
            tot = acc[:, None] - cost
            back[i] = np.argmax(tot, axis=0)
            acc = tot[back[i], np.arange(C)] + sc[i]
        path = np.zeros(n, dtype=int)
        path[-1] = int(np.argmax(acc))
        for i in range(n - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
        choice[seg] = path
    return choice


def _drop_short(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = mask.copy()
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s < min_len:
            out[s:e] = False
    return out

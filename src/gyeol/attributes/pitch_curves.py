"""Pitch-derived attribute curves and ornament events (no learning).

Curves (all on the shared grid, each with confidence):

* ``pitch_center``: f0 in cents with vibrato and fast ornaments removed
  (250 ms running median, then zero-phase 3 Hz low-pass, within each note).  The slow intonation offset
  is computed from it *against the target* in :mod:`gyeol.explain`.
* ``vibrato_rate`` (Hz) and ``vibrato_extent`` (cents, semi-extent): local
  sinusoid fit over a 0.5 s window; confidence = fraction of variance the
  sinusoid explains × pitch confidence.

Events (:class:`~gyeol.core.containers.Event`):

* ``scoop``: a note entered from below (≥ 60 cents) that rises into pitch
  within 300 ms;
* ``fall``: a note released downwards (≥ 80 cents) into silence in its last 250 ms;
* ``kkeokki`` (꺾기): a single excursion of ≥ 80 cents lasting 50–250 ms
  inside a note with no opposite-sign excursion of half its size within
  100 ms (vibrato's opposite half-cycle always falls inside that window);
* ``glide``: a continuous transition of ≥ 150 cents lasting ≥ 60 ms between
  two notes of one voiced run.

Thresholds are detection defaults, not coaching thresholds: the coach layer
chooses what to report using operating thresholds from validation data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from ..core.containers import Event
from ..core.grid import FrameGrid
from ..dsp.base import runs
from ..dsp.notes import NoteSpan, segment_notes


@dataclass
class EventConfig:
    scoop_min_cents: float = 60.0
    scoop_max_s: float = 0.3
    fall_min_cents: float = 80.0
    fall_window_s: float = 0.25
    fall_min_s: float = 0.05
    scoop_min_s: float = 0.04
    kkeokki_min_cents: float = 80.0
    kkeokki_min_s: float = 0.05
    kkeokki_max_s: float = 0.25
    glide_min_cents: float = 150.0
    glide_min_s: float = 0.06
    glide_max_gap_s: float = 0.06


def pitch_center(cents: np.ndarray, voiced: np.ndarray, grid: FrameGrid, cutoff_hz: float = 3.0,
                 segments: list[NoteSpan] | None = None, median_s: float = 0.25) -> np.ndarray:
    """Running median (``median_s``) + zero-phase low-pass of the f0 contour, per segment.

    With ``segments`` (notes) the filter never crosses a note boundary, so a
    glide or leap does not bend the centre of the neighbouring notes; voiced
    frames outside every segment fall back to per-voiced-run filtering.
    """
    out = np.full(len(cents), np.nan)
    sos = signal.butter(2, cutoff_hz, "low", fs=grid.rate, output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    med_frames = max(3, int(round(median_s * grid.rate)))
    ok = voiced & np.isfinite(cents)

    def fill(s: int, e: int) -> None:
        seg = cents[s:e]
        good = np.isfinite(seg)
        if good.sum() == 0:
            return
        seg = np.interp(np.arange(e - s), np.flatnonzero(good), seg[good])
        # running median first: it ignores excursions shorter than half the
        # window (꺾기, scoops), which a plain low-pass would partly follow
        k = min(e - s, med_frames) | 1
        seg = ndimage.median_filter(seg, size=k, mode="nearest")
        out[s:e] = signal.sosfiltfilt(sos, seg) if e - s > padlen + 1 else np.mean(seg)
        out[s:e][~ok[s:e]] = np.nan

    covered = np.zeros(len(cents), bool)
    for n in segments or []:
        fill(n.start, n.end)
        covered[n.start : n.end] = True
    for s, e in runs(ok & ~covered):
        fill(s, e)
    return out


def vibrato_curves(cents: np.ndarray, center: np.ndarray, voiced: np.ndarray, grid: FrameGrid, conf: np.ndarray,
                   window_s: float = 0.5, rates: tuple[float, float] = (3.5, 8.5)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rate_hz, extent_cents, confidence) per frame."""
    T = len(cents)
    W = max(8, int(round(window_s * grid.rate)))
    half = W // 2
    freqs = np.arange(rates[0], rates[1] + 1e-9, 0.1)
    t = np.arange(W) / grid.rate
    S = np.sin(2 * np.pi * np.outer(t, freqs))
    C = np.cos(2 * np.pi * np.outer(t, freqs))
    rate = np.full(T, np.nan)
    extent = np.full(T, np.nan)
    vconf = np.zeros(T)
    d = np.where(voiced, cents - center, np.nan)
    ok_rows = [i for i in range(half, T - half) if np.all(np.isfinite(d[i - half : i - half + W]))]
    if not ok_rows:
        return rate, extent, vconf
    idx = np.array(ok_rows)
    D = d[idx[:, None] - half + np.arange(W)[None, :]]
    D = D - D.mean(axis=1, keepdims=True)
    a = D @ S * (2.0 / W)
    b = D @ C * (2.0 / W)
    amp = np.hypot(a, b)
    best = np.argmax(amp, axis=1)
    A = amp[np.arange(len(idx)), best]
    var = D.var(axis=1) + 1e-9
    r2 = np.clip(0.5 * A**2 / var, 0.0, 1.0)
    rate[idx] = freqs[best]
    extent[idx] = A
    vconf[idx] = r2 * conf[idx]
    return rate, extent, vconf


def notes_from_pitch(cents: np.ndarray, voiced: np.ndarray, grid: FrameGrid, center: np.ndarray | None = None,
                     min_stable_s: float = 0.2, max_unstable_range: float = 100.0) -> list[NoteSpan]:
    """Notes = pitch-stable segments.

    A segment shorter than ``min_stable_s`` whose raw pitch *sweeps* (range >
    ``max_unstable_range`` and mostly monotonic: |end − start| > 0.6·range)
    is a scoop / fall / glide transition, not a note: it is merged into the
    contiguous previous note (tail) or, failing that, the next one (head).
    Oscillating segments (vibrato) and stable short notes are kept.
    """
    raw = segment_notes(cents, voiced, grid.hop_seconds)
    if not raw:
        return raw
    min_len = int(min_stable_s * grid.rate)

    def sweep(n: NoteSpan) -> bool:
        seg = cents[n.start : n.end]
        seg = seg[np.isfinite(seg)]
        if (n.end - n.start) >= min_len or seg.size < 3:
            return False
        rng = seg.max() - seg.min()
        return rng > max_unstable_range and abs(seg[-1] - seg[0]) > 0.6 * rng

    out: list[NoteSpan] = []
    pending_head: NoteSpan | None = None
    for n in raw:
        if sweep(n):
            if out and n.start - out[-1].end <= 1:
                out[-1] = NoteSpan(out[-1].start, n.end)  # tail of the previous note
            else:
                pending_head = n  # head of the next note
            continue
        if pending_head is not None and n.start - pending_head.end <= 1:
            n = NoteSpan(pending_head.start, n.end)
        pending_head = None
        out.append(n)
    return out


def detect_events(cents: np.ndarray, center: np.ndarray, voiced: np.ndarray, conf: np.ndarray, vib_conf: np.ndarray,
                  notes: list[NoteSpan], grid: FrameGrid, cfg: EventConfig | None = None) -> list[Event]:
    cfg = cfg or EventConfig()
    fps = grid.rate
    ev: list[Event] = []
    for n in notes:
        s, e = n.start, n.end
        L = e - s
        if L < 4:
            continue
        core = center[s + L // 4 : e - L // 4] if L >= 8 else center[s:e]
        target = float(np.nanmedian(core))
        seg = cents[s:e]
        # scoop
        k = min(L, int(cfg.scoop_max_s * fps))
        head = seg[:k]
        if np.isfinite(head).sum() >= 3:
            start_dev = target - np.nanmin(head[: max(2, k // 3)])
            arrived = np.flatnonzero(np.abs(head - target) <= 30)
            if start_dev >= cfg.scoop_min_cents and arrived.size and arrived[0] / fps >= cfg.scoop_min_s:
                j = int(arrived[0])
                ev.append(Event("scoop", s, s + max(j, 1), float(start_dev), float(np.mean(conf[s : s + max(j, 1)])), {"note_start": s}))
        # fall: only when the note is released into silence (a legato drop into
        # the next note is a glide, handled below)
        k = min(L, int(cfg.fall_window_s * fps))
        tail = seg[-k:]
        released = e >= len(voiced) or not voiced[e : min(len(voiced), e + int(0.05 * fps))].any()
        if released and np.isfinite(tail).sum() >= 3:
            drop = target - np.nanmin(tail[-max(2, k // 3) :])
            left = np.flatnonzero(np.abs(tail - target) <= 30)
            j = int(left[-1]) if left.size else 0
            if drop >= cfg.fall_min_cents and (k - j) / fps >= cfg.fall_min_s:
                ev.append(Event("fall", e - k + j, e, float(drop), float(np.mean(conf[e - k + j : e])), {"note_end": e}))
        # kkeokki: a single excursion from the local centre, isolated in time
        # (vibrato produces a train of alternating excursions instead)
        margin = int(0.1 * fps)
        iso = int(0.1 * fps)
        if L > 2 * margin + 3:
            inner = np.arange(s + margin, e - margin)
            dev = cents[inner] - center[inner]
            big = np.abs(dev) >= cfg.kkeokki_min_cents * 0.5
            for a, b in runs(big):
                dur = (b - a) / fps
                peak = dev[a:b][np.argmax(np.abs(dev[a:b]))]
                if not (cfg.kkeokki_min_s <= dur <= cfg.kkeokki_max_s and abs(peak) >= cfg.kkeokki_min_cents):
                    continue
                ga, gb = int(inner[a]), int(inner[b - 1]) + 1
                around = np.r_[cents[max(s, ga - iso) : ga] - center[max(s, ga - iso) : ga], cents[gb : min(e, gb + iso)] - center[gb : min(e, gb + iso)]]
                # vibrato would show an opposite-sign excursion within half a period
                if around.size and np.nanmin(around * np.sign(peak)) > -0.5 * abs(peak):
                    ev.append(Event("kkeokki", ga, gb, float(peak), float(np.mean(conf[ga:gb])), {"direction": "up" if peak > 0 else "down"}))
    # glides between adjacent notes of one voiced run
    max_gap = int(cfg.glide_max_gap_s * fps)
    for n1, n2 in zip(notes[:-1], notes[1:]):
        if n2.start - n1.end > max_gap:
            continue
        c1 = float(np.nanmedian(center[n1.start : n1.end]))
        c2 = float(np.nanmedian(center[n2.start : n2.end]))
        interval = c2 - c1
        if abs(interval) < cfg.glide_min_cents:
            continue
        lo, hi = min(c1, c2) + 0.15 * abs(interval), max(c1, c2) - 0.15 * abs(interval)
        win = np.arange(max(n1.start, n1.end - int(0.4 * fps)), min(n2.end, n2.start + int(0.4 * fps)))
        between = win[(cents[win] > lo) & (cents[win] < hi)]
        # fast glides lose voicing in their steepest part: count the dropout as part of the glide
        if between.size and (n2.start - n1.end) > 0:
            between = np.arange(min(between[0], n1.end), max(between[-1] + 1, n2.start))
        if between.size and (between[-1] - between[0] + 1) / fps >= cfg.glide_min_s:
            ev.append(Event("glide", int(between[0]), int(between[-1]) + 1, float(interval),
                            float(np.mean(conf[between])), {"from_note": n1.start, "to_note": n2.start}))
    return sorted(ev, key=lambda x: x.start)

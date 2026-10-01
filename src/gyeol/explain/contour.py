"""Whole-contour pitch comparison (revision A5).

Note-level items compare pitch *centres*; this compares the aligned f0
contours frame by frame along the whole phrase — sustained parts, attacks,
releases and the transitions between notes — and reports spans where

* both sides are voiced and confident (``min_confidence`` on both tracks and
  on the alignment), and
* the difference stays beyond a *detection* floor with one sign for at least
  ``min_span_s``.

The floor only forms candidate spans; whether a span is *shown* is decided by
the coach with the fitted thresholds for ``contour_deviation`` /
``transition_deviation`` (:mod:`gyeol.coach.thresholds`).

Each span is located on the target's notes through τ: inside one note →
``contour_deviation`` at that note (its syllable); across a note boundary or
in the gap between notes → ``transition_deviation`` between notes k and k+1
(both syllables).  The event detectors label what the span is about: a
scoop / fall / 꺾기 / glide of the user or the target overlapping the span is
named in ``detail["event"]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.containers import ExplanationItem, Representation, Span
from ..dsp.base import runs


@dataclass
class ContourConfig:
    detection_floor_cents: float = 20.0  # candidate spans only; display uses the fitted coach thresholds
    noise_multiplier: float = 3.0  # … or this many robust SDs of the frame-to-frame difference noise, if larger
    min_confidence: float = 0.5
    min_span_s: float = 0.06
    merge_gap_frames: int = 2


def _syllables(syllables: list[tuple[int, int, str]], start: float, end: float) -> tuple[str, ...]:
    return tuple(t for s, e, t in syllables if s < end and e > start)


def contour_items(user: Representation, target: Representation, tau: np.ndarray, diff: np.ndarray, u_conf: np.ndarray,
                  t_conf: np.ndarray, warp_conf: np.ndarray, notes: list[tuple[int, int]], syllables: list[tuple[int, int, str]],
                  cfg: ContourConfig) -> dict[tuple, ExplanationItem]:
    """``diff`` is the per-frame f0 difference user − target(τ) with the octave/key relation removed (or folded)."""
    g = user.grid
    u_f0, t_f0 = user.curves["f0_cents"].values, np.interp(tau, np.arange(target.grid.n_frames), target.curves["f0_cents"].values)
    valid = (np.isfinite(diff) & np.isfinite(u_f0) & np.isfinite(t_f0) & (u_conf >= cfg.min_confidence)
             & (t_conf >= cfg.min_confidence) & (warp_conf >= cfg.min_confidence))
    if valid.sum() < 5:
        return {}
    d = np.where(valid, diff, 0.0)
    dv = np.diff(d[valid])
    noise = 1.4826 * float(np.median(np.abs(dv - np.median(dv)))) / np.sqrt(2) if dv.size > 4 else 0.0
    floor = max(cfg.detection_floor_cents, cfg.noise_multiplier * noise)
    min_len = max(2, int(round(cfg.min_span_s * g.rate)))
    out: dict[tuple, ExplanationItem] = {}
    for sign in (1, -1):
        mask = valid & (sign * d >= floor)
        # bridge tiny gaps (a frame or two of lost confidence) inside one deviation
        for s, e in runs(~mask):
            if 0 < s and e < len(mask) and e - s <= cfg.merge_gap_frames and mask[s - 1] and mask[e]:
                mask[s:e] = True
        for s, e in runs(mask):
            if e - s < min_len:
                continue
            seg = slice(s, e)
            ok = valid[seg]
            mag = float(np.mean(d[seg][ok]))
            conf = float(np.mean(np.minimum(u_conf[seg], t_conf[seg])[ok] * warp_conf[seg][ok]))
            t0, t1 = float(tau[s]), float(tau[e - 1]) + 1.0
            inside = [k for k, (ns, ne) in enumerate(notes) if ns <= t0 and t1 <= ne + 1]
            if inside:
                k = inside[0]
                attr, key_note, loc = "contour_deviation", k, {"location": "note", "target_note": k}
                syl = _syllables(syllables, *notes[k])
            else:
                before = [k for k, (ns, ne) in enumerate(notes) if ns <= t0]
                k = before[-1] if before else 0
                k2 = min(k + 1, len(notes) - 1) if notes else 0
                attr, key_note = "transition_deviation", k
                loc = {"location": "transition", "target_note": k, "between_notes": [k, k2]}
                syl = _syllables(syllables, *notes[k]) + _syllables(syllables, *notes[k2]) if notes else ()
            events = sorted({ev.kind for ev in user.events if ev.kind != "onset" and ev.start < e and ev.end > s}
                            | {ev.kind for ev in target.events if ev.kind != "onset" and ev.start < t1 and ev.end > t0})
            delta = np.full(len(d), np.nan)
            delta[seg] = np.where(ok, d[seg], np.nan)
            item = ExplanationItem("pitch", attr, [Span(s, e, syl)], mag, "cents", conf, delta=delta, detail={
                **loc, "event": events[0] if events else None, "events": events, "max_abs_cents": float(np.max(np.abs(d[seg][ok]))),
                "duration_s": (e - s) * g.hop_seconds, "detection_floor_cents": floor})
            key = ("pitch", attr, key_note)
            prev = out.get(key)
            # one item per note / transition: keep the span with the larger area
            if prev is None or abs(item.magnitude) * item.detail["duration_s"] > abs(prev.magnitude) * prev.detail["duration_s"]:
                out[key] = item
    return out

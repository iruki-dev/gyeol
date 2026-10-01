"""Audibility score per explanation item (brief §5.5).

For each item: render the user's phrase with **only that item corrected**
(:func:`gyeol.demo.edits.edit_for_item`) and compute a perceptual distance to
the **unedited render of the same renderer** — so renderer artefacts cancel and
only the correction is measured.

The distance (:func:`perceptual_distance`) is a specific-loudness proxy:
80 mel bands up to 16 kHz, power compressed with exponent 0.23
(Zwicker-style), then the mean absolute difference per frame relative to the
mean of both frames' specific loudness, averaged over the frames the edit acts
on.  It is **uncalibrated**: use it to rank items, not as a JND.  A listening
test against it is part of the M8 evaluation.

Items that are not scored keep ``audibility = None``; the coach then falls
back to confidence × size over measurement uncertainty (:mod:`gyeol.coach.priority`).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..core.containers import Explanation, Representation
from ..core.status import Result

EPS = 1e-12


@lru_cache(maxsize=4)
def _mel_fb(sr: int, n_fft: int, n_mels: int, fmax: float) -> np.ndarray:
    mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    pts = imel(np.linspace(mel(40.0), mel(min(fmax, sr / 2)), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, c, hi = pts[i : i + 3]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb


def _specific_loudness(x: np.ndarray, sr: int, hop: int, n_fft: int = 2048, n_mels: int = 80) -> np.ndarray:
    n = len(x)
    xp = np.pad(np.asarray(x, float), (n_fft // 2, n_fft))
    idx = np.arange(1 + (n - 1) // hop)[:, None] * hop + np.arange(n_fft)[None, :]
    p = np.abs(np.fft.rfft(xp[idx] * np.hanning(n_fft), axis=1)) ** 2
    return (p @ _mel_fb(sr, n_fft, n_mels, 16000.0).T + EPS) ** 0.23


def perceptual_distance(ref: np.ndarray, test: np.ndarray, sr: int, hop: int = 512, frames: np.ndarray | None = None) -> float:
    """Mean relative specific-loudness difference over ``frames`` (bool mask; all if None)."""
    n = min(len(ref), len(test))
    a, b = _specific_loudness(ref[:n], sr, hop), _specific_loudness(test[:n], sr, hop)
    d = np.mean(np.abs(a - b), axis=1) / (0.5 * (np.mean(a, axis=1) + np.mean(b, axis=1)) + EPS)
    if frames is not None:
        m = np.zeros(len(d), bool)
        f = np.asarray(frames, bool)[: len(d)]
        m[: len(f)] = f
        if not m.any():
            return 0.0
        d = d[m]
    return float(np.mean(d))


def score_audibility(exp: Explanation, take, target: Representation, renderer, *,
                     feasible=None, seed: int = 0, context: int = 8) -> Result[Explanation]:
    """Fill ``item.audibility`` for every item the renderer can correct.

    Items it cannot render keep ``audibility = None`` with
    ``detail["audibility_status"]`` saying why.  Edits are clamped to the
    user's feasible range as in the demo, so audibility measures what the user
    would actually hear.
    """
    from ..demo.edits import edit_for_item
    from ..demo.feasible import FeasibleRange, clamp_edit
    from ..demo.renderers import _check

    _check(take)
    try:
        rng = feasible or FeasibleRange.from_takes([take.rep])
    except ValueError as exc:
        return Result.failure(f"feasible range: {exc}")
    sr, hop = take.recording.sr, take.rep.grid.hop
    n_scored = 0
    for it in exp.items:
        e = edit_for_item(it, take.rep, target, exp)
        if not e.ok:
            it.audibility, it.detail["audibility_status"] = None, f"no edit: {e.reason}"
            continue
        if e.value.requires - renderer.capabilities:
            it.audibility = None
            it.detail["audibility_status"] = f"{renderer.name} cannot render {sorted(e.value.requires - renderer.capabilities)}"
            continue
        edit, _ = clamp_edit(e.value, take.rep, rng)
        support = edit.support(dilate=1)
        if not support.any():
            it.audibility, it.detail["audibility_status"] = 0.0, "ok (edit is empty)"
            n_scored += 1
            continue
        # render only the affected region (plus context) for the baseline and the edit
        idx = np.flatnonzero(support)
        a, b = max(0, idx[0] - context), min(len(support), idx[-1] + 1 + context)
        base = renderer.render(take, None, seed=seed, frames=(a, b))
        r = renderer.render(take, edit, seed=seed, frames=(a, b))
        if not (base.ok and r.ok):
            it.audibility, it.detail["audibility_status"] = None, f"render failed: {base.reason or r.reason}"
            continue
        it.audibility = perceptual_distance(base.value, r.value, sr, hop, support[a:b])
        it.detail["audibility_status"] = f"ok ({renderer.name}, uncalibrated)"
        n_scored += 1
    exp.meta["audibility"] = {"renderer": renderer.name, "n_scored": n_scored, "n_items": len(exp.items),
                              "metric": "relative specific-loudness distance (uncalibrated)"}
    return Result.success(exp)

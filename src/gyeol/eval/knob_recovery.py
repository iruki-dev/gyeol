"""Knob recovery (evaluation §5) and the data for fitting coach thresholds.

Synthetic user takes are rendered from a target melody with **known** knobs:
per-note detune (cents), onset shift (ms), vibrato presence and attack scoops /
release falls.  Whole-contour spans (``contour_deviation`` /
``transition_deviation``, revision A5) are scored against the synthesiser's
exact f0 tracks over the span the item reports.  Each take is
measured twice — clean, and through a degradation (noise at ``snr_db``) —
and explained against the target.  The result gives, per attribute,

* ``true`` and ``measured`` magnitudes and item ``confidence`` (for the
  error-vs-confidence operating threshold), and
* the two conditions' magnitudes for the same item (retest pairs for the MDC),

which is exactly what :func:`gyeol.coach.thresholds.fit_attribute_threshold`
needs.  With synthetic data the fitted thresholds are marked
``synthetic: True``; real validation needs real paired recordings (M8).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..core.consent import Provenance
from ..core.containers import Recording
from ..synth import SynthNote, melody
from ..verification.degrade import add_noise

BASE_NOTES = ((262, "a", "s"), (294, "o", "t"), (330, "i", "k"), (349, "e", "h"), (392, "a", "s"), (330, "u", "t"))


@dataclass
class KnobData:
    true: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    measured: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    confidence: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    retest_a: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    retest_b: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    n_takes: int = 0

    def arrays(self, attr: str) -> dict[str, np.ndarray]:
        return {k: np.asarray(getattr(self, k)[attr], float) for k in ("true", "measured", "confidence", "retest_a", "retest_b")}


def _notes(detune, shifts, vib, dur, gap, scoop=None, fall=None):
    n = len(BASE_NOTES)
    scoop = np.zeros(n) if scoop is None else scoop
    fall = np.zeros(n) if fall is None else fall
    return [SynthNote(f * 2 ** (d / 1200), dur, gap_after=gap, vowel=v, consonant=c, onset_shift_s=s,
                      vibrato_rate=5.5 if e else 0.0, vibrato_extent_cents=e, scoop_cents=sc, fall_cents=fa)
            for (f, v, c), d, s, e, sc, fa in zip(BASE_NOTES, detune, shifts, vib, scoop, fall)]


def _true_cents(f0_track: np.ndarray, sr: int, times: np.ndarray) -> np.ndarray:
    idx = np.clip(np.round(times * sr).astype(int), 0, len(f0_track) - 1)
    f = f0_track[idx]
    return np.where(f > 0, 1200 * np.log2(np.maximum(f, 1e-9) / 440.0), np.nan)


def _contour_truth(item, u_true: np.ndarray, t_true: np.ndarray, tau: np.ndarray) -> float | None:
    """Mean true user − target difference (cents) over the frames the item measured."""
    if item.delta is None:
        return None
    sel = np.flatnonzero(np.isfinite(item.delta))
    if sel.size == 0:
        return None
    tt = np.interp(tau[sel], np.arange(len(t_true)), t_true)
    d = u_true[sel] - tt
    d = d[np.isfinite(d)]
    return float(np.mean(d)) if d.size else None


def knob_recovery(n_takes: int = 8, seed: int = 0, snr_db: float = 25.0, dur: float = 0.9, gap: float = 0.2,
                  max_detune: float = 60.0, max_shift_s: float = 0.08, trackers=None) -> KnobData:
    from ..attributes.extract import analyze
    from ..explain import explain

    rng = np.random.default_rng(seed)
    n = len(BASE_NOTES)
    sr = 44100
    tvib = tuple(60.0 if k in (2, 4) else 0.0 for k in range(n))
    target_m = melody(_notes((0,) * n, (0,) * n, tvib, dur, gap), sr=sr)
    target_audio = target_m.audio
    data = KnobData()
    an = lambda x, prov: analyze(Recording(x, sr, prov, owner_id="knob" if prov is Provenance.USER else None), trackers=trackers)  # noqa: E731
    for i in range(n_takes):
        detune = rng.uniform(-max_detune, max_detune, n)
        shifts = rng.uniform(-max_shift_s, max_shift_s, n)
        shifts[0] = 0.0
        vib = np.where(rng.random(n) < 0.5, 60.0, 0.0)
        scoop = np.where(rng.random(n) < 0.5, rng.uniform(80.0, 200.0, n), 0.0)
        fall = np.where(rng.random(n) < 0.5, rng.uniform(80.0, 200.0, n), 0.0)
        user_m = melody(_notes(detune, shifts, vib, dur, gap, scoop, fall), sr=sr, seed=int(rng.integers(1 << 30)))
        user = user_m.audio
        m = max(len(user), len(target_audio))
        t_pad, u_pad = np.pad(target_audio, (0, m - len(target_audio))), np.pad(user, (0, m - len(user)))
        target = an(t_pad, Provenance.REFERENCE)
        if not target.usable:
            continue
        per_cond, contour_truth = [], {}
        t_true = _true_cents(target_m.truth["f0_track"], sr, target.value.grid.times())
        for x in (u_pad, add_noise(u_pad, snr_db, "pink", seed=i)):
            rep = an(x, Provenance.USER)
            ex = explain([rep.value], target.value) if rep.usable else None
            items = {it.key: it for it in ex.value.items} if ex is not None and ex.usable else {}
            per_cond.append(items)
            if items and not contour_truth:  # truth over the spans found in the clean condition
                u_true = _true_cents(np.pad(user_m.truth["f0_track"], (0, m - len(user))), sr, rep.value.grid.times())
                for key, it in items.items():
                    if key[1] in ("contour_deviation", "transition_deviation"):
                        tv = _contour_truth(it, u_true, t_true, ex.value.warp)
                        if tv is not None:
                            contour_truth[key] = tv
        truth = {("pitch", "intonation_offset", k): detune[k] for k in range(n)}
        truth |= {("rhythm", "onset_timing", k): shifts[k] * 1000.0 for k in range(n)}
        truth |= {("ornament", "vibrato_extent", k): vib[k] - tvib[k] for k in range(n) if vib[k] or tvib[k]}
        truth[("pitch", "global_offset", -1)] = float(np.median(detune))
        truth |= contour_truth
        for key, tv in truth.items():
            attr = key[1]
            for cond in per_cond:
                if key in cond:
                    data.true[attr].append(float(tv))
                    data.measured[attr].append(cond[key].magnitude)
                    data.confidence[attr].append(cond[key].confidence)
            if all(key in c for c in per_cond) and len(per_cond) == 2:
                data.retest_a[attr].append(per_cond[0][key].magnitude)
                data.retest_b[attr].append(per_cond[1][key].magnitude)
        data.n_takes += 1
    return data


def fit_from_knob_data(data: KnobData, attributes: tuple[str, ...] = ("intonation_offset", "onset_timing", "vibrato_extent", "global_offset",
                                                                       "contour_deviation", "transition_deviation"),
                       units: dict[str, str] | None = None, n_bins: int = 5, min_per_bin: int = 5):
    """Fit coach thresholds for the attributes knob recovery has truth for."""
    from ..coach.thresholds import ThresholdSet, fit_attribute_threshold

    units = units or {"intonation_offset": "cents", "onset_timing": "ms", "vibrato_extent": "cents", "global_offset": "cents",
                      "contour_deviation": "cents", "transition_deviation": "cents"}
    out = []
    for attr in attributes:
        a = data.arrays(attr)
        if len(a["retest_a"]) < 3:
            continue
        out.append(fit_attribute_threshold(attr, a["retest_a"], a["retest_b"], a["confidence"], a["measured"] - a["true"],
                                           units.get(attr, ""), n_bins=n_bins, min_per_bin=min_per_bin))
    return ThresholdSet.fitted(out, f"synthetic knob recovery: {data.n_takes} takes, clean vs pink-noise condition", synthetic=True)

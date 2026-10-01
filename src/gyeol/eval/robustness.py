"""Robustness grid (evaluation §4): ICC / MDC of c(t) and of explanation items.

Every recording is passed through each degradation — noise, reverberation,
band limiting, lossy codecs (when ffmpeg exists), separation artefacts,
Bluetooth latency jitter, backing-track bleed — then analysed (and, with a
target, explained) exactly as the app would, including the offline latency
refinement against the guide.  Two reports come out:

* **curves** — per attribute curve, the recording-level median of confident
  frames (:func:`gyeol.verification.invariance.recording_values`);
* **items** — per explanation item (category/attribute/note), its magnitude,
  plus how often the item is found at all under each condition.

For both: ICC(2,1) across conditions with CI, SEM, MDC95, Bland–Altman bias
per condition (:func:`gyeol.verification.invariance.invariance_report`), the
median absolute deviation from the clean condition per condition, and — for
axes with several levels (SNR, T60, bitrate) — the operating threshold where
the error bound crosses the MDC (:func:`gyeol.verification.thresholds.operating_threshold`).
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Sequence

import numpy as np

from ..core.containers import Recording, Representation
from ..core.status import Result
from ..verification import degrade
from ..verification.invariance import DimensionReport, Record, invariance_report, recording_values
from ..verification.thresholds import ThresholdResult, operating_threshold


@dataclass
class GridCondition:
    name: str  # axis name, e.g. "snr_db"
    level: float
    apply: Callable[[np.ndarray, int], np.ndarray]
    latency: bool = False  # the transport shifts time: refine the offset like the app does

    @property
    def key(self) -> str:
        return "clean" if self.name == "clean" else f"{self.name}={self.level:g}"


def _sep(x, sr, seed):
    from ..data.augment import separation_artifacts

    return separation_artifacts(x, sr, np.random.default_rng(seed))[0]


def _bt(x, sr, seed):
    from ..data.augment import bluetooth_jitter

    return bluetooth_jitter(x, sr, np.random.default_rng(seed))[0]


def robustness_grid(levels: str = "short", seed: int = 0, accompaniment: np.ndarray | None = None,
                    codecs: bool | None = None) -> list[GridCondition]:
    """``levels="short"`` (one or two levels per axis, for CI) or ``"full"`` (the research grid)."""
    full = levels == "full"
    g = [GridCondition("clean", 0.0, lambda x, sr: x)]
    g += [GridCondition("snr_db", s, lambda x, sr, s=s: degrade.add_noise(x, s, "pink", seed=seed))
          for s in ((40, 30, 20, 10, 5, 0) if full else (30, 10))]
    g += [GridCondition("t60_s", t, lambda x, sr, t=t: degrade.reverberate(x, degrade.synthetic_rir(sr, t, seed=seed)))
          for t in ((0.2, 0.4, 0.8, 1.2) if full else (0.5,))]
    g += [GridCondition("bandwidth_hz", b, partial(degrade.bandlimit, cutoff_hz=b)) for b in ((3400, 5000, 7000) if full else (5000,))]
    if codecs if codecs is not None else bool(shutil.which("ffmpeg")):
        for fmt, rates in (("mp3", (32, 64, 128)), ("opus", (16, 32))) if full else (("mp3", (64,)),):
            g += [GridCondition(f"{fmt}_kbps", r, partial(degrade.codec, fmt=fmt, bitrate_kbps=r)) for r in rates]
    g += [GridCondition("separation", float(i), lambda x, sr, i=i: _sep(x, sr, seed + i)) for i in range(3 if full else 1)]
    g += [GridCondition("bluetooth", float(i), lambda x, sr, i=i: _bt(x, sr, seed + i), latency=True) for i in range(3 if full else 1)]
    if accompaniment is not None:
        g += [GridCondition("var_db", v, lambda x, sr, v=v: degrade.mix_accompaniment(x, accompaniment, v))
              for v in ((10, 5, 0, -5) if full else (5,))]
    return g


@dataclass
class GridItem:
    item_id: str  # the "subject" of the ICC (one performance)
    audio: np.ndarray
    sr: int
    target: Representation | None = None  # explain against this when given
    target_audio: np.ndarray | None = None  # guide vocal for latency refinement


@dataclass
class AxisThreshold:
    dimension: str
    axis: str
    result: ThresholdResult


@dataclass
class RobustnessReport:
    curves: dict[str, DimensionReport]
    items: dict[str, DimensionReport]
    deviation: dict[str, dict[str, float]]  # dimension → condition → median |value − clean|
    presence: dict[str, dict[str, float]]  # item → condition → fraction of performances where it was found
    thresholds: list[AxisThreshold]
    failures: list[str] = field(default_factory=list)

    def summary(self) -> list[str]:
        out = []
        for kind, reps in (("curve", self.curves), ("item", self.items)):
            for d, r in sorted(reps.items()):
                worst = max(self.deviation.get(d, {}).items(), key=lambda kv: kv[1], default=("-", float("nan")))
                out.append(f"{kind:5s} {d:40s} ICC {r.icc21:5.2f} [{r.band:9s}] MDC95 {r.mdc95:8.2f}  worst: {worst[0]} "
                           f"(|Δ| {worst[1]:.2f}){'  ACCEPTED' if r.accepted else ''}")
        return out


def _item_key(it) -> str:
    return f"{it.category}/{it.attribute}/{it.detail.get('target_note', -1)}"


def run_robustness(items: Sequence[GridItem], conditions: Sequence[GridCondition], *,
                   analyzer: Callable[[Recording], Result[Representation]] | None = None,
                   curve_names: Sequence[str] = ("pitch_center", "loudness_rel", "aperiodic_ratio", "vibrato_extent", "vibrato_rate",
                                                 "subharmonic_ratio"),
                   min_icc: float = 0.9, n_boot: int = 200) -> RobustnessReport:
    from ..attributes.extract import analyze
    from ..explain import explain
    from ..io import refine_offset, shift

    analyzer = analyzer or analyze
    if not any(c.key == "clean" for c in conditions):
        raise ValueError("the grid needs a 'clean' reference condition")
    curve_recs: list[Record] = []
    item_recs: list[Record] = []
    found: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    failures: list[str] = []
    for it in items:
        per_cond_items: dict[str, dict[str, float]] = {}
        for c in conditions:
            y = np.asarray(c.apply(np.asarray(it.audio, float), it.sr), float)[: len(it.audio)]
            y = np.pad(y, (0, len(it.audio) - len(y)))
            if c.latency:
                ref = it.target_audio if it.target_audio is not None else it.audio
                off = refine_offset(y, ref, it.sr, max_lag_s=0.5)
                if off.usable:
                    y = shift(y, off.value.latency_s, it.sr)
            r = analyzer(Recording(y, it.sr))
            if not r.usable:
                failures.append(f"{it.item_id} @ {c.key}: {r.reason}")
                continue
            curve_recs.append(Record(it.item_id, c.key, recording_values(r.value, curve_names)))
            if it.target is not None:
                ex = explain([r.value], it.target)
                vals = {_item_key(i): i.magnitude for i in ex.value.items} if ex.usable else {}
                if not ex.usable:
                    failures.append(f"{it.item_id} @ {c.key}: explain: {ex.reason}")
                per_cond_items[c.key] = vals
        if per_cond_items:
            keys = set().union(*[set(v) for v in per_cond_items.values()])
            for ck, vals in per_cond_items.items():
                item_recs.append(Record(it.item_id, ck, {k: vals.get(k, np.nan) for k in keys}))
                for k in keys:
                    found[k][ck].append(k in vals)
    curves = invariance_report(curve_recs, "clean", min_icc, n_boot) if curve_recs else {}
    items_rep = invariance_report(item_recs, "clean", min_icc, n_boot) if item_recs else {}
    deviation: dict[str, dict[str, float]] = defaultdict(dict)
    thresholds: list[AxisThreshold] = []
    for recs, reps in ((curve_recs, curves), (item_recs, items_rep)):
        table = {(r.singer, r.condition): r.values for r in recs}
        subjects = sorted({r.singer for r in recs})
        for d, rep in reps.items():
            by_axis: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
            for c in conditions:
                if c.key == "clean":
                    continue
                errs = [abs(table[(s, c.key)].get(d, np.nan) - table[(s, "clean")].get(d, np.nan))
                        for s in subjects if (s, c.key) in table and (s, "clean") in table]
                errs = [e for e in errs if np.isfinite(e)]
                if errs:
                    deviation[d][c.key] = float(np.median(errs))
                    by_axis[c.name][0].extend([c.level] * len(errs))
                    by_axis[c.name][1].extend(errs)
            for axis, (lv, er) in by_axis.items():
                if len(set(lv)) >= 2 and (axis in ("snr_db", "t60_s", "bandwidth_hz") or axis.endswith("_kbps")):
                    hib = axis != "t60_s"
                    thresholds.append(AxisThreshold(d, axis, operating_threshold(np.array(lv), np.array(er), rep.mdc95, hib,
                                                                                 n_bins=len(set(lv)), min_per_bin=2)))
    presence = {k: {ck: float(np.mean(v)) for ck, v in cond.items()} for k, cond in found.items()}
    return RobustnessReport(curves, items_rep, dict(deviation), presence, thresholds, failures)

"""Premise checks: state what a judgement assumes, check it, withhold the judgement if it fails.

=================  ==============================================================  =========================================
premise            statement                                                       judgements that depend on it
=================  ==============================================================  =========================================
shared_clock       user and target recordings run on one clock (sing-along,        tempo; absolute onset timing (otherwise
                   latency removed): similar length and start, confident          onsets are compared note-to-note)
                   alignment that stays inside its band
octave_relation    both pitch tracks are confident enough to decide the octave /    absolute pitch comparison (otherwise
                   key relation between user and target                            octave-invariant)
level_chain        both level chains are comparable (no clipping, no unseparated   loudness, dynamic range
                   or leaking accompaniment)
noise_floor        both recordings are clean enough to measure aperiodicity         breathiness
                   (SNR, no accompaniment, no codec)
interval_set       at least three confident note intervals of ≥ 1 semitone          interval compression
=================  ==============================================================  =========================================

Thresholds live in :class:`PremiseConfig`; the defaults are conservative
starting points to be re-fitted on the real-recording set (``gyeol eval realset``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.containers import ExplanationItem, Premise, Representation, WithheldItem

SHARED_CLOCK = "shared_clock"
OCTAVE_RELATION = "octave_relation"
LEVEL_CHAIN = "level_chain"
NOISE_FLOOR = "noise_floor"
INTERVAL_SET = "interval_set"

#: attribute → premises it needs (onset timing is handled by the comparison mode, not withheld)
REQUIRES: dict[str, tuple[str, ...]] = {
    "tempo": (SHARED_CLOCK,),
    "loudness": (LEVEL_CHAIN,),
    "dynamic_range": (LEVEL_CHAIN,),
    "breathiness": (NOISE_FLOOR,),
    "interval_compression": (INTERVAL_SET,),
}


@dataclass
class PremiseConfig:
    # shared clock
    max_duration_diff_s: float = 1.0
    max_start_offset_s: float = 0.25
    min_alignment_confidence: float = 0.4
    max_band_saturation: float = 0.05
    min_latency_confidence: float = 0.5
    # octave / key relation
    octave_min_confidence: float = 0.6
    octave_min_coverage: float = 0.25
    octave_min_consistency: float = 0.8
    # level chain and noise floor
    level_blocking_flags: tuple[str, ...] = ("clipping", "accompaniment_unseparated", "residual_accompaniment")
    noise_min_snr_db: float = 30.0
    noise_blocking_flags: tuple[str, ...] = ("clipping", "codec_suspected", "accompaniment_unseparated", "residual_accompaniment")
    min_intervals: int = 3


def check_shared_clock(user: Representation, target: Representation, warp, warp_failure: str | None, sung: np.ndarray,
                       cfg: PremiseConfig) -> Premise:
    g = user.grid
    m: dict = {"duration_diff_s": abs(user.grid.n_frames - target.grid.n_frames) * g.hop_seconds}
    reasons = []
    if m["duration_diff_s"] > cfg.max_duration_diff_s:
        reasons.append(f"durations differ by {m['duration_diff_s']:.2f} s (> {cfg.max_duration_diff_s} s)")
    lat = user.meta.get("latency") or {}
    if "confidence" in lat:
        m["latency_offset_s"] = lat.get("offset_s")
        m["latency_confidence"] = lat["confidence"]
        if lat["confidence"] < cfg.min_latency_confidence:
            reasons.append(f"latency refinement confidence {lat['confidence']:.2f} (< {cfg.min_latency_confidence})")
    else:
        m["latency_confidence"] = None  # not measured: the other checks decide
    if warp is None:
        reasons.append(f"alignment around a shared clock failed: {warp_failure}")
    else:
        sel = sung & (warp.confidence >= cfg.min_alignment_confidence)
        dev = warp.tau - np.arange(len(warp.tau))
        m["start_offset_s"] = float(np.median(dev[sel]) * g.hop_seconds) if sel.any() else float("nan")
        m["alignment_confidence"] = float(np.mean(warp.confidence[sung])) if sung.any() else 0.0
        m["band_saturation"] = float(np.mean(np.abs(dev[sung]) >= 0.9 * warp.band_frames)) if sung.any() else 1.0
        if not np.isfinite(m["start_offset_s"]) or abs(m["start_offset_s"]) > cfg.max_start_offset_s:
            reasons.append(f"start offset {m['start_offset_s']:.2f} s (> {cfg.max_start_offset_s} s)")
        if m["alignment_confidence"] < cfg.min_alignment_confidence:
            reasons.append(f"alignment confidence {m['alignment_confidence']:.2f} (< {cfg.min_alignment_confidence})")
        if m["band_saturation"] > cfg.max_band_saturation:
            reasons.append(f"the time warp sits at its band limit on {m['band_saturation'] * 100:.0f}% of sung frames")
    return Premise(SHARED_CLOCK, "user and target recordings share a clock (sing-along with latency removed)",
                   not reasons, "; ".join(reasons), m)


def check_octave_relation(raw_diff: np.ndarray, u_conf: np.ndarray, t_conf: np.ndarray, both: np.ndarray, sung: np.ndarray,
                          step: float, cfg: PremiseConfig) -> Premise:
    m: dict = {"step_cents": step}
    reasons = []
    n_sung = max(int(sung.sum()), 1)
    m["coverage"] = float(both.sum() / n_sung)
    if both.sum() < 5:
        return Premise(OCTAVE_RELATION, "both pitch tracks are confident enough to decide the octave/key relation", False,
                       "too few frames where both pitches are reliable", m)
    m["user_confidence"] = float(np.median(u_conf[both]))
    m["target_confidence"] = float(np.median(t_conf[both]))
    d0 = float(np.median(raw_diff[both]))
    trans = step * round(d0 / step)
    m["transposition_cents"] = trans
    m["consistency"] = float(np.mean(np.abs(raw_diff[both] - trans) < step / 2))
    if m["coverage"] < cfg.octave_min_coverage:
        reasons.append(f"only {m['coverage'] * 100:.0f}% of sung frames have reliable pitch on both sides")
    if min(m["user_confidence"], m["target_confidence"]) < cfg.octave_min_confidence:
        reasons.append(f"pitch confidence user {m['user_confidence']:.2f} / target {m['target_confidence']:.2f} (< {cfg.octave_min_confidence})")
    if m["consistency"] < cfg.octave_min_consistency:
        reasons.append(f"only {m['consistency'] * 100:.0f}% of frames agree on one {'octave' if step == 1200 else 'key'} relation")
    return Premise(OCTAVE_RELATION, "both pitch tracks are confident enough to decide the octave/key relation", not reasons,
                   "; ".join(reasons), m)


def _flags(rep: Representation) -> dict:
    return dict((rep.quality or {}).get("flags", {}))


def check_level_chain(user: Representation, target: Representation, cfg: PremiseConfig) -> Premise:
    bad = {f"{side}:{k}": v for side, rep in (("user", user), ("target", target)) for k, v in _flags(rep).items()
           if k in cfg.level_blocking_flags}
    return Premise(LEVEL_CHAIN, "both level chains are comparable (no clipping, no unseparated or leaking accompaniment)",
                   not bad, "; ".join(f"{k} ({v})" for k, v in bad.items()), {"blocking_flags": sorted(bad)})


def check_noise_floor(user: Representation, target: Representation, cfg: PremiseConfig) -> Premise:
    reasons, m = [], {}
    for side, rep in (("user", user), ("target", target)):
        snr = (rep.quality or {}).get("snr_db")
        m[f"{side}_snr_db"] = snr
        if snr is None:
            reasons.append(f"{side} SNR unknown")
        elif snr < cfg.noise_min_snr_db:
            reasons.append(f"{side} SNR {snr:.1f} dB (< {cfg.noise_min_snr_db})")
        for k in _flags(rep):
            if k in cfg.noise_blocking_flags:
                reasons.append(f"{side}: {k}")
    return Premise(NOISE_FLOOR, "both recordings are clean enough to measure aperiodicity", not reasons, "; ".join(reasons), m)


def check_interval_set(n_intervals: int, cfg: PremiseConfig) -> Premise:
    ok = n_intervals >= cfg.min_intervals
    return Premise(INTERVAL_SET, "at least three confident note intervals of ≥ 1 semitone", ok,
                   "" if ok else f"only {n_intervals} usable intervals", {"n_intervals": n_intervals})


def apply_premises(items: dict[tuple, ExplanationItem], premises: dict[str, Premise]) -> tuple[dict[tuple, ExplanationItem], list[WithheldItem]]:
    """Remove items whose premises do not hold; return (kept, withheld)."""
    kept, withheld = {}, []
    for key, it in items.items():
        failed = [premises[p] for p in REQUIRES.get(it.attribute, ()) if p in premises and not premises[p].holds]
        if failed:
            p = failed[0]
            withheld.append(WithheldItem(it.category, it.attribute, int(it.detail.get("target_note", -1)), p.name, p.reason))
        else:
            kept[key] = it
    return kept, withheld

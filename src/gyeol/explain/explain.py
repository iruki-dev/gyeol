"""Explain a user take as the target transformed by τ(t) and Δc(t).

For every user take:

1. **τ(t)** from content features only (:mod:`gyeol.align`);
2. **key/octave invariance**: the median aligned difference of the pitch
   centres is rounded to whole octaves (``allow_transposition`` rounds to
   semitones instead, for transposed backing tracks) and removed;
3. per target note: intonation offset (pitch-centre Δ), vibrato extent /
   rate Δ, onset timing (from τ), and ornament events matched by kind
   (missing / extra / different size);
4. per phrase: overall intonation offset, interval compression (slope of
   user intervals vs target intervals) and tempo drift (mean τ′);
5. phonation, dynamics and phrase-level diction items
   (:mod:`gyeol.explain.attributes`);
6. **cannot judge**: frames where either side's pitch confidence or the
   alignment confidence is too low, items whose own confidence is, and
   frames whose residual difference is unexplained (remainder).

Audibility is filled in afterwards by :func:`gyeol.explain.audibility.score_audibility`.

With several takes, items are keyed by (category, attribute, target note).
A difference with the same sign in every take and |mean| > SD is labelled
``style_or_habit``; otherwise ``error`` (fewer than two takes →
``undetermined``).  Magnitudes and spans refer to the **last** take.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..align.warp import Warp, WarpConfig, estimate_warp, onset_deviations, tempo_ratio
from ..core.containers import Consistency, Explanation, ExplanationItem, Premise, Representation, Span, WithheldItem
from ..core.status import Result
from ..dsp.base import runs
from .attributes import diction_items, phonation_dynamics_items, remainder_spans
from .contour import ContourConfig, contour_items
from .premises import (
    PremiseConfig,
    apply_premises,
    check_interval_set,
    check_level_chain,
    check_noise_floor,
    check_octave_relation,
    check_shared_clock,
)

EVENT_KINDS = ("scoop", "fall", "kkeokki", "glide")


@dataclass
class ExplainConfig:
    warp: WarpConfig = field(default_factory=WarpConfig)
    allow_transposition: bool = False
    min_confidence: float = 0.3
    min_warp_confidence: float = 0.3
    min_note_frames: int = 5
    vibrato_presence_conf: float = 0.5
    cannot_judge_min_s: float = 0.1
    #: robust z-score of the residual difference above which frames are unexplained
    remainder_z: float = 4.0
    #: revision A2/A3: premise checks; A5: whole-contour comparison
    premises: PremiseConfig = field(default_factory=PremiseConfig)
    contour: ContourConfig = field(default_factory=ContourConfig)


def _fold_octave(raw_diff: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """Octave-invariant difference: fold onto the 1200-cent circle around the circular mean of the reliable frames."""
    r = raw_diff[ref_mask & np.isfinite(raw_diff)]
    if r.size == 0:
        return (raw_diff + 600.0) % 1200.0 - 600.0
    ang = 2 * np.pi * r / 1200.0
    centre = 1200.0 / (2 * np.pi) * np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
    return centre + ((raw_diff - centre + 600.0) % 1200.0 - 600.0)


def _at(values: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Target curve sampled at fractional target frames τ (NaN-aware linear)."""
    n = len(values)
    lo = np.clip(np.floor(tau).astype(int), 0, n - 1)
    hi = np.clip(lo + 1, 0, n - 1)
    w = np.clip(tau - lo, 0, 1)
    a, b = values[lo], values[hi]
    out = a * (1 - w) + b * w
    out = np.where(np.isfinite(a) & ~np.isfinite(b), a, out)
    return np.where(~np.isfinite(a) & np.isfinite(b), b, out)


@dataclass
class _Take:
    rep: Representation
    warp: Warp
    items: dict[tuple, ExplanationItem]
    cannot: list[Span]
    transposition: float | None
    premises: dict[str, Premise] = field(default_factory=dict)
    withheld: list[WithheldItem] = field(default_factory=list)
    modes: dict[str, str] = field(default_factory=dict)


def _syllables_for(syllables: list, start: int, end: int) -> tuple[str, ...]:
    out = []
    for s, e, text in syllables:
        if s < end and e > start:
            out.append(text)
    return tuple(out)


def _explain_take(user: Representation, target: Representation, cfg: ExplainConfig, syllables: list | None = None) -> Result[_Take]:
    syllables = list(target.meta.get("syllables", [])) if syllables is None else syllables
    if not user.grid.same_axis(target.grid):
        return Result.failure(f"user grid {user.grid} and target grid {target.grid} differ")
    for need in ("content", "pitch_center", "f0_cents"):
        if need not in user.curves or need not in target.curves:
            return Result.failure(f"missing curve {need!r}")
    g = user.grid
    uc, tc = user.curves, target.curves
    premises: dict[str, Premise] = {}
    # A3 — premise: one clock.  Align around identity first; if the clock is not shared, align on content alone.
    wr = estimate_warp(uc["content"].values, tc["content"].values, g, cfg.warp)
    u_sung = np.isfinite(uc["pitch_center"].values) | (np.nan_to_num(uc["voicing"].values) > 0.5)
    clock = check_shared_clock(user, target, wr.value if wr.ok else None, None if wr.ok else wr.reason, u_sung, cfg.premises)
    premises[clock.name] = clock
    if clock.holds:
        warp, timing_mode = wr.value, "shared_clock"
    else:
        wide = replace(cfg.warp, band_seconds=max(user.grid.n_frames, target.grid.n_frames) * g.hop_seconds, open_begin=True)
        wr2 = estimate_warp(uc["content"].values, tc["content"].values, g, wide)
        if not wr2.ok:
            return Result.failure(f"alignment failed: {wr2.reason}")
        warp, timing_mode = wr2.value, "content_aligned"
    tau = warp.tau
    u_center, u_conf = uc["pitch_center"].values, uc["f0_cents"].confidence
    t_center = _at(tc["pitch_center"].values, tau)
    t_conf = _at(tc["f0_cents"].confidence, tau)
    both = (u_conf >= cfg.min_confidence) & (t_conf >= cfg.min_confidence) & (warp.confidence >= cfg.min_warp_confidence)
    both &= np.isfinite(u_center) & np.isfinite(t_center)
    if both.sum() < cfg.min_note_frames:
        return Result.failure("too few frames where both pitches and the alignment are reliable")
    raw_diff = u_center - t_center
    # A2 — premise: the octave / key relation is decided only when both pitch tracks are confident
    step = 100.0 if cfg.allow_transposition else 1200.0
    sung = u_sung | (_at(np.nan_to_num(tc["voicing"].values), tau) > 0.5)
    rel = check_octave_relation(raw_diff, u_conf, t_conf, both, sung, step, cfg.premises)
    premises[rel.name] = rel
    if rel.holds:
        transposition: float | None = float(rel.measures["transposition_cents"])
        diff = raw_diff - transposition
        pitch_mode = "absolute"
    else:  # octave-invariant: fold every difference onto the octave circle around its circular mean
        transposition = None
        diff = _fold_octave(raw_diff, both)
        pitch_mode = "octave_invariant"
    frame_conf = np.minimum(u_conf, t_conf) * warp.confidence

    items: dict[tuple, ExplanationItem] = {}
    notes = target.meta.get("notes", [])
    t_onsets = [s for s, _ in notes]
    devs = onset_deviations(warp, t_onsets, g) if notes else []
    u_vconf, t_vconf = uc["vibrato_extent"].confidence, _at(tc["vibrato_extent"].confidence, tau)
    u_ext, t_ext = uc["vibrato_extent"].values, _at(tc["vibrato_extent"].values, tau)
    u_rate, t_rate = uc["vibrato_rate"].values, _at(tc["vibrato_rate"].values, tau)
    note_centres: list[tuple[float, float]] = []
    note_spans: dict[int, Span] = {}

    for k, (ns, ne) in enumerate(notes):
        sel = np.flatnonzero((tau >= ns) & (tau < ne))
        if sel.size < cfg.min_note_frames:
            continue
        span = Span(int(sel[0]), int(sel[-1]) + 1, _syllables_for(syllables, ns, ne))
        note_spans[k] = span
        # intonation: middle 60 % of the note (ornaments live at the edges)
        a, b = int(sel[0] + 0.2 * sel.size), int(sel[0] + 0.8 * sel.size)
        core = np.arange(a, max(a + 1, b))
        good = core[both[core]]
        if good.size >= 3:
            d = float(np.median(diff[good]))
            delta = np.full(g.n_frames, np.nan)
            delta[span.start : span.end] = diff[span.start : span.end]
            items[("pitch", "intonation_offset", k)] = ExplanationItem(
                "pitch", "intonation_offset", [span], d, "cents", float(np.mean(frame_conf[good])),
                delta=delta, detail={"target_note": k})
            note_centres.append((float(np.median(t_center[good]) + d), float(np.median(t_center[good]))))
        # vibrato
        fc = both[sel]
        if fc.mean() > 0.5:
            u_has = (u_vconf[sel] >= cfg.vibrato_presence_conf).mean() > 0.3
            t_has = (t_vconf[sel] >= cfg.vibrato_presence_conf).mean() > 0.3
            if u_has or t_has:
                ue = float(np.nanmedian(u_ext[sel][u_vconf[sel] >= cfg.vibrato_presence_conf])) if u_has else 0.0
                te = float(np.nanmedian(t_ext[sel][t_vconf[sel] >= cfg.vibrato_presence_conf])) if t_has else 0.0
                conf = float(np.mean(frame_conf[sel]))
                items[("ornament", "vibrato_extent", k)] = ExplanationItem(
                    "ornament", "vibrato_extent", [span], ue - te, "cents", conf,
                    detail={"target_note": k, "user": ue, "target": te})
                if u_has and t_has:
                    ur = float(np.nanmedian(u_rate[sel][u_vconf[sel] >= cfg.vibrato_presence_conf]))
                    trr = float(np.nanmedian(t_rate[sel][t_vconf[sel] >= cfg.vibrato_presence_conf]))
                    items[("ornament", "vibrato_rate", k)] = ExplanationItem(
                        "ornament", "vibrato_rate", [span], ur - trr, "Hz", conf, detail={"target_note": k, "user": ur, "target": trr})
        # onset timing: absolute on a shared clock, else relative to the previous note (content-aligned)
        # the onset must lie inside the stretch of the target the take covers (a trimmed clip may start mid-note)
        if k < len(devs) and tau[0] + 2 <= ns <= tau[-1] - 2:
            dv = devs[k]
            uf = int(np.clip(round(dv.user_frame), 0, g.n_frames - 1))
            conf = float(np.mean(warp.confidence[max(0, uf - 3) : uf + 4]))
            if timing_mode == "shared_clock":
                items[("rhythm", "onset_timing", k)] = ExplanationItem(
                    "rhythm", "onset_timing", [Span(uf, uf + 1, span.syllables)], dv.deviation_s * 1000.0, "ms", conf,
                    detail={"target_note": k, "reference": "shared clock"})
            elif k > 0 and k - 1 < len(devs) and tau[0] + 2 <= notes[k - 1][0]:
                rel_ms = (dv.deviation_s - devs[k - 1].deviation_s) * 1000.0
                items[("rhythm", "onset_timing", k)] = ExplanationItem(
                    "rhythm", "onset_timing", [Span(uf, uf + 1, span.syllables)], rel_ms, "ms", conf,
                    detail={"target_note": k, "reference": "previous note"})
        # ornaments: match events of each kind inside the note
        for kind in EVENT_KINDS:
            t_ev = [e for e in target.events if e.kind == kind and ns <= e.start < ne]
            u_ev = [e for e in user.events if e.kind == kind and span.start <= e.start < span.end]
            if not t_ev and not u_ev:
                continue
            tm = max((e.magnitude for e in t_ev), key=abs, default=0.0)
            um = max((e.magnitude for e in u_ev), key=abs, default=0.0)
            evs = u_ev or []
            spans = [Span(e.start, e.end, span.syllables) for e in evs] or [span]
            confs = [e.confidence for e in t_ev + u_ev]
            status = "missing" if t_ev and not u_ev else "extra" if u_ev and not t_ev else "different"
            items[("ornament", kind, k)] = ExplanationItem(
                "ornament", kind, spans, um - tm, "cents", float(np.mean(confs)) * float(np.mean(warp.confidence[sel])),
                detail={"target_note": k, "status": status, "user": um, "target": tm})

    # phrase level
    all_span = [Span(int(np.flatnonzero(both)[0]), int(np.flatnonzero(both)[-1]) + 1)]
    if len(note_centres) >= 1:
        offs = [u - t for u, t in note_centres]
        items[("pitch", "global_offset", -1)] = ExplanationItem(
            "pitch", "global_offset", all_span, float(np.median(offs)), "cents", float(np.mean(frame_conf[both])),
            detail={"n_notes": len(offs)})
    n_intervals = 0
    if len(note_centres) >= 2:
        uc_, tc_ = np.array(note_centres).T
        ti, ui = np.diff(tc_), np.diff(uc_)
        use = np.abs(ti) >= 100
        n_intervals = int(use.sum())
        if use.sum() >= 3:
            slope = float(np.dot(ui[use], ti[use]) / np.dot(ti[use], ti[use]))
            resid = ui[use] - slope * ti[use]
            fit = 1.0 - float(np.var(resid) / (np.var(ui[use]) + 1e-9))
            items[("pitch", "interval_compression", -1)] = ExplanationItem(
                "pitch", "interval_compression", all_span, (slope - 1.0) * 100.0, "%", float(np.clip(fit, 0, 1)),
                detail={"slope": slope, "n_intervals": int(use.sum())})
    voiced_any = np.isfinite(u_center) | np.isfinite(t_center)
    if voiced_any.sum() > 10:
        tr = tempo_ratio(warp)
        items[("rhythm", "tempo", -1)] = ExplanationItem(
            "rhythm", "tempo", all_span, float((np.mean(tr[voiced_any]) - 1.0) * 100.0), "%", float(np.mean(warp.confidence[voiced_any])),
            detail={"meaning": "+ rushing, − dragging"})

    # M5: phonation, dynamics, diction
    items.update(phonation_dynamics_items(user, target, tau, warp.confidence, notes, note_spans, cfg.min_confidence))
    items.update(diction_items(user, target, tau, warp.confidence, cfg.min_confidence))
    # A5: whole-contour comparison (attacks, releases, transitions)
    for key, it in contour_items(user, target, tau, diff, u_conf, t_conf, warp.confidence, notes, syllables, cfg.contour).items():
        # a span that only restates the note's own centre offset adds nothing to the intonation item
        note = items.get(("pitch", "intonation_offset", key[2]))
        if (it.attribute == "contour_deviation" and note is not None and np.sign(note.magnitude) == np.sign(it.magnitude)
                and abs(it.magnitude - note.magnitude) < it.detail["detection_floor_cents"]):
            continue
        items[key] = it

    # A3: every premise-dependent judgement is withheld when its premise fails
    for pr in (check_level_chain(user, target, cfg.premises), check_noise_floor(user, target, cfg.premises),
               check_interval_set(n_intervals, cfg.premises)):
        premises[pr.name] = pr
    items, withheld = apply_premises(items, premises)

    # cannot judge: voiced somewhere but not reliably comparable, or an unexplained residual remainder
    sung = np.isfinite(u_center) | (_at(tc["voicing"].values, tau) > 0.5)
    bad = sung & ~both
    min_len = int(cfg.cannot_judge_min_s * g.rate)
    cannot = [Span(s, e, reason="low_confidence") for s, e in runs(bad) if e - s >= min_len]
    cannot += remainder_spans(user, target, tau, sung, cfg.remainder_z, min_len)
    for key in [k for k, it in items.items() if it.confidence < cfg.min_confidence]:
        cannot.extend(Span(sp.start, sp.end, sp.syllables, f"item_confidence:{key[1]}") for sp in items.pop(key).spans)
    return Result.success(_Take(user, warp, items, cannot, transposition, premises, withheld,
                                {"timing": timing_mode, "pitch": pitch_mode}))


def explain(user_takes: list[Representation], target: Representation, config: ExplainConfig | None = None,
            lyrics: str | None = None) -> Result[Explanation]:
    """Explain the user's take(s) against the target phrase.

    ``lyrics`` (optional) maps syllables onto the target's notes with
    :func:`gyeol.context.assign_syllables` (Hangul split independent of spaces
    and punctuation); otherwise ``target.meta["syllables"]`` is used if present.
    """
    cfg = config or ExplainConfig()
    if not user_takes:
        return Result.failure("no user takes")
    if lyrics is not None:
        from ..context import assign_syllables

        syllables = assign_syllables(target.meta.get("notes", []), lyrics)
    else:
        syllables = list(target.meta.get("syllables", []))
    takes: list[_Take] = []
    failures: list[str] = []
    for i, rep in enumerate(user_takes):
        r = _explain_take(rep, target, cfg, syllables)
        if r.ok:
            takes.append(r.value)
        else:
            failures.append(f"take {i}: {r.reason}")
    if not takes:
        return Result.failure("; ".join(failures))
    last = takes[-1]
    items: list[ExplanationItem] = []
    for key, it in last.items.items():
        vals = [t.items[key].magnitude for t in takes if key in t.items]
        if len(vals) < 2:
            it.consistency = Consistency.UNDETERMINED
        else:
            v = np.array(vals)
            same_sign = np.all(v > 0) or np.all(v < 0)
            it.consistency = Consistency.STYLE_OR_HABIT if same_sign and abs(v.mean()) > v.std() else Consistency.ERROR
        it.detail["per_take"] = vals
        items.append(it)
    items.sort(key=lambda it: (it.spans[0].start if it.spans else 0, it.category))
    exp = Explanation(
        grid=last.rep.grid, warp=last.warp.tau, transposition_cents=last.transposition, items=items,
        cannot_judge=sorted(last.cannot, key=lambda s: s.start), n_takes=len(takes),
        meta={"failed_takes": failures, "warp_confidence": last.warp.confidence, "user_quality": last.rep.quality,
              "target_quality": target.quality, "syllables": syllables},
        premises=last.premises, withheld=last.withheld, comparison_mode=last.modes,
    )
    return Result(Result.success(exp).status, exp, "", failures)

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

from dataclasses import dataclass, field

import numpy as np

from ..align.warp import Warp, WarpConfig, estimate_warp, onset_deviations, tempo_ratio
from ..core.containers import Consistency, Explanation, ExplanationItem, Representation, Span
from ..core.status import Result
from ..dsp.base import runs
from .attributes import diction_items, phonation_dynamics_items, remainder_spans

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
    transposition: float


def _syllables_for(target: Representation, start: int, end: int) -> tuple[str, ...]:
    out = []
    for s, e, text in target.meta.get("syllables", []):
        if s < end and e > start:
            out.append(text)
    return tuple(out)


def _explain_take(user: Representation, target: Representation, cfg: ExplainConfig) -> Result[_Take]:
    if not user.grid.same_axis(target.grid):
        return Result.failure(f"user grid {user.grid} and target grid {target.grid} differ")
    for need in ("content", "pitch_center", "f0_cents"):
        if need not in user.curves or need not in target.curves:
            return Result.failure(f"missing curve {need!r}")
    wr = estimate_warp(user.curves["content"].values, target.curves["content"].values, user.grid, cfg.warp)
    if not wr.ok:
        return Result.failure(f"alignment failed: {wr.reason}")
    warp = wr.value
    tau = warp.tau
    g = user.grid
    uc, tc = user.curves, target.curves
    u_center, u_conf = uc["pitch_center"].values, uc["f0_cents"].confidence
    t_center = _at(tc["pitch_center"].values, tau)
    t_conf = _at(tc["f0_cents"].confidence, tau)
    both = (u_conf >= cfg.min_confidence) & (t_conf >= cfg.min_confidence) & (warp.confidence >= cfg.min_warp_confidence)
    both &= np.isfinite(u_center) & np.isfinite(t_center)
    if both.sum() < cfg.min_note_frames:
        return Result.failure("too few frames where both pitches and the alignment are reliable")
    raw_diff = u_center - t_center
    d0 = float(np.median(raw_diff[both]))
    step = 100.0 if cfg.allow_transposition else 1200.0
    transposition = step * round(d0 / step)
    diff = raw_diff - transposition
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
        span = Span(int(sel[0]), int(sel[-1]) + 1, _syllables_for(target, ns, ne))
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
            note_centres.append((float(np.median(u_center[good]) - transposition), float(np.median(t_center[good]))))
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
        # onset timing
        if k < len(devs):
            dv = devs[k]
            uf = int(np.clip(round(dv.user_frame), 0, g.n_frames - 1))
            conf = float(np.mean(warp.confidence[max(0, uf - 3) : uf + 4]))
            items[("rhythm", "onset_timing", k)] = ExplanationItem(
                "rhythm", "onset_timing", [Span(uf, uf + 1, span.syllables)], dv.deviation_s * 1000.0, "ms", conf,
                detail={"target_note": k})
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
    if len(note_centres) >= 4:
        uc_, tc_ = np.array(note_centres).T
        ti, ui = np.diff(tc_), np.diff(uc_)
        use = np.abs(ti) >= 100
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

    # cannot judge: voiced somewhere but not reliably comparable, or an unexplained residual remainder
    sung = np.isfinite(u_center) | (_at(tc["voicing"].values, tau) > 0.5)
    bad = sung & ~both
    min_len = int(cfg.cannot_judge_min_s * g.rate)
    cannot = [Span(s, e, reason="low_confidence") for s, e in runs(bad) if e - s >= min_len]
    cannot += remainder_spans(user, target, tau, sung, cfg.remainder_z, min_len)
    for key in [k for k, it in items.items() if it.confidence < cfg.min_confidence]:
        cannot.extend(Span(sp.start, sp.end, sp.syllables, f"item_confidence:{key[1]}") for sp in items.pop(key).spans)
    return Result.success(_Take(user, warp, items, cannot, transposition))


def explain(user_takes: list[Representation], target: Representation, config: ExplainConfig | None = None) -> Result[Explanation]:
    """Explain the user's take(s) against the target phrase."""
    cfg = config or ExplainConfig()
    if not user_takes:
        return Result.failure("no user takes")
    takes: list[_Take] = []
    failures: list[str] = []
    for i, rep in enumerate(user_takes):
        r = _explain_take(rep, target, cfg)
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
              "target_quality": target.quality},
    )
    return Result(Result.success(exp).status, exp, "", failures)

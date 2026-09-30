"""Edits of the **user's** c(t) that correct one explanation item.

An :class:`Edit` lives on the user's grid and is expressed in attribute terms:

* ``f0_cents`` — additive pitch change (cents), applied where voiced;
* ``gain_db`` — additive loudness change (dB);
* ``aperiodic_db`` — change of the aperiodic-to-periodic ratio (dB);
* ``time_map`` — for every output frame, the (fractional) user frame shown there;
* ``curves`` — deltas of learned posterior curves (register, phonation, …),
  usable only by a renderer whose decoder is conditioned on them.

``requires`` names the capabilities a renderer needs (``f0``, ``gain``,
``aperiodic``, ``timing``, ``curve:<name>``).  Edits scale (``alpha`` for the
stepwise schedule) and add (several items at once).

How each item is corrected (user → target, nothing else touched):

======================  ==========================================================
intonation_offset       constant −Δ over the note (raised-cosine edges)
global_offset           constant −Δ over the phrase
interval_compression    each note centre moved so user intervals scale by 1/slope
onset_timing            piecewise-linear retiming of the onset, neighbours fixed
tempo                   linear retiming at the target pace
vibrato_extent          user oscillation scaled to the target extent (or the
                        target's oscillation transplanted when the user has none)
vibrato_rate            target's oscillation transplanted at the user's extent
scoop/fall/kkeokki/glide  the target's contour relative to its note centre
                        transplanted over the event region (user's and
                        target's events) onto the user's note centre
breathiness             aperiodic ratio −Δ over the note
loudness                gain −Δ over the note
dynamic_range           loudness deviations scaled by target / user range
register, quality_*,    learned posterior moved to the target's (needs a
laryngeal_*, phone_match  decoder conditioned on that curve)
======================  ==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.containers import Explanation, ExplanationItem, Representation
from ..core.status import Result

ItemKey = tuple[str, str, int]


def item_key(it: ExplanationItem) -> ItemKey:
    return (it.category, it.attribute, int(it.detail.get("target_note", -1)))


@dataclass
class Edit:
    n_frames: int
    f0_cents: np.ndarray | None = None
    gain_db: np.ndarray | None = None
    aperiodic_db: np.ndarray | None = None
    time_map: np.ndarray | None = None
    curves: dict[str, np.ndarray] = field(default_factory=dict)
    requires: frozenset[str] = frozenset()
    items: tuple[ItemKey, ...] = ()

    @classmethod
    def identity(cls, n_frames: int) -> "Edit":
        return cls(n_frames)

    def _arr(self, a: np.ndarray | None) -> np.ndarray:
        return np.zeros(self.n_frames) if a is None else a

    def scaled(self, alpha: float) -> "Edit":
        s = lambda a: None if a is None else a * alpha  # noqa: E731
        tm = None
        if self.time_map is not None:
            ident = np.arange(self.n_frames, dtype=float)
            tm = ident + alpha * (self.time_map - ident)
        return Edit(self.n_frames, s(self.f0_cents), s(self.gain_db), s(self.aperiodic_db), tm,
                    {k: v * alpha for k, v in self.curves.items()}, self.requires, self.items)

    def __add__(self, other: "Edit") -> "Edit":
        if other.n_frames != self.n_frames:
            raise ValueError("edits on different grids")
        add = lambda a, b: None if a is None and b is None else self._arr(a) + self._arr(b)  # noqa: E731
        tm = None
        if self.time_map is not None or other.time_map is not None:
            ident = np.arange(self.n_frames, dtype=float)
            tm = ident + (self._tm() - ident) + (other._tm() - ident)
            tm = np.maximum.accumulate(np.clip(tm, 0, self.n_frames - 1))
        curves = dict(self.curves)
        for k, v in other.curves.items():
            curves[k] = curves[k] + v if k in curves else v
        return Edit(self.n_frames, add(self.f0_cents, other.f0_cents), add(self.gain_db, other.gain_db),
                    add(self.aperiodic_db, other.aperiodic_db), tm, curves, self.requires | other.requires,
                    self.items + other.items)

    def _tm(self) -> np.ndarray:
        return np.arange(self.n_frames, dtype=float) if self.time_map is None else self.time_map

    def support(self, dilate: int = 3) -> np.ndarray:
        """Frames the edit acts on (bool), dilated by ``dilate`` frames."""
        m = np.zeros(self.n_frames, bool)
        for a in (self.f0_cents, self.gain_db, self.aperiodic_db):
            if a is not None:
                m |= np.abs(np.nan_to_num(a)) > 1e-6
        if self.time_map is not None:
            m |= np.abs(self.time_map - np.arange(self.n_frames)) > 1e-3
        for v in self.curves.values():
            m |= np.any(np.abs(np.nan_to_num(v.reshape(self.n_frames, -1))) > 1e-6, axis=1)
        if dilate and m.any():
            m = np.convolve(m.astype(float), np.ones(2 * dilate + 1), "same") > 0
        return m


@dataclass
class EditConfig:
    ramp_s: float = 0.03  # raised-cosine edges of box-shaped edits
    event_margin_frames: int = 2


def _ramp_box(T: int, start: int, end: int, ramp: int) -> np.ndarray:
    m = np.zeros(T)
    m[max(0, start) : min(T, end)] = 1.0
    if ramp > 0:
        k = np.hanning(2 * ramp + 1)
        m = np.convolve(m, k / k.sum(), "same")
    return m


def _smooth_mask(mask: np.ndarray, ramp: int) -> np.ndarray:
    if ramp <= 0:
        return mask.astype(float)
    k = np.hanning(2 * ramp + 1)
    return np.minimum(np.convolve(mask.astype(float), k / k.sum(), "same") * 1.0, 1.0) * mask


def _at(values: np.ndarray, tau: np.ndarray) -> np.ndarray:
    from ..explain.explain import _at as at

    return at(values, tau)


def _deviation(rep: Representation) -> np.ndarray:
    return rep.curves["f0_cents"].values - rep.curves["pitch_center"].values


def _core_median(values: np.ndarray, start: int, end: int) -> float:
    """Median over the middle 60 % of [start, end) (ornaments live at the edges)."""
    n = end - start
    v = values[start + int(0.2 * n) : start + max(int(0.8 * n), int(0.2 * n) + 1)]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _note_spans(exp: Explanation) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for it in exp.items:
        k = it.detail.get("target_note", -1)
        if k is not None and k >= 0 and it.category in ("pitch", "phonation", "dynamics") and it.spans:
            out.setdefault(int(k), (it.spans[0].start, it.spans[-1].end))
    for it in exp.items:  # fall back to any note-level item
        k = it.detail.get("target_note", -1)
        if k is not None and k >= 0 and it.spans and it.attribute != "onset_timing":
            out.setdefault(int(k), (it.spans[0].start, it.spans[-1].end))
    return out


def _onsets(exp: Explanation) -> dict[int, int]:
    return {int(it.detail["target_note"]): it.spans[0].start for it in exp.items
            if it.attribute == "onset_timing" and it.spans and it.detail.get("target_note", -1) >= 0}


def edit_for_item(item: ExplanationItem, user: Representation, target: Representation, exp: Explanation,
                  config: EditConfig | None = None) -> Result[Edit]:
    """Build the edit that moves the user's c(t) to the target for this one item."""
    cfg = config or EditConfig()
    T = user.grid.n_frames
    if exp.grid != user.grid:
        return Result.failure("explanation and user representation are on different grids")
    tau = exp.warp
    ramp = max(1, int(round(cfg.ramp_s * user.grid.rate)))
    attr, key = item.attribute, item_key(item)
    span = (item.spans[0].start, item.spans[-1].end) if item.spans else (0, T)
    mk = lambda **kw: Result.success(Edit(T, items=(key,), **kw))  # noqa: E731

    if attr in ("intonation_offset", "global_offset"):
        return mk(f0_cents=-item.magnitude * _ramp_box(T, *span, ramp), requires=frozenset({"f0"}))

    if attr == "interval_compression":
        slope = 1.0 + item.magnitude / 100.0
        if slope <= 0.05:
            return Result.failure(f"implausible interval slope {slope:.2f}")
        pc = user.curves["pitch_center"].values
        centres = {}
        for k, (s, e) in _note_spans(exp).items():
            v = pc[s:e][np.isfinite(pc[s:e])]
            if v.size:
                centres[k] = (s, e, float(np.median(v)))
        if len(centres) < 2:
            return Result.failure("fewer than two notes with a pitch centre")
        mean = float(np.mean([c for _, _, c in centres.values()]))
        d = np.zeros(T)
        for s, e, c in centres.values():
            d += (c - mean) * (1.0 / slope - 1.0) * _ramp_box(T, s, e, ramp)
        return mk(f0_cents=d, requires=frozenset({"f0"}))

    if attr == "onset_timing":
        u = span[0]
        dev = item.magnitude / 1000.0 * user.grid.rate  # + late
        others = sorted(v for k, v in _onsets(exp).items() if v != u)
        prev = max([o for o in others if o < u], default=0)
        nxt = min([o for o in others if o > u], default=T - 1)
        new_u = float(np.clip(u - dev, prev + 1, nxt - 1))
        if nxt - prev < 3:
            return Result.failure("no room to retime this onset")
        o = np.arange(T, dtype=float)
        tm = o.copy()
        sel = (o >= prev) & (o <= nxt)
        tm[sel] = np.interp(o[sel], [prev, new_u, nxt], [prev, u, nxt])
        return mk(time_map=tm, requires=frozenset({"timing"}))

    if attr == "tempo":
        rate = 1.0 + item.magnitude / 100.0  # target frames per user frame
        a = span[0]
        o = np.arange(T, dtype=float)
        tm = np.where(o >= a, a + (o - a) * rate, o)
        return mk(time_map=np.clip(tm, 0, T - 1), requires=frozenset({"timing"}))

    if attr in ("vibrato_extent", "vibrato_rate") or attr in ("scoop", "fall", "kkeokki", "glide"):
        if "pitch_center" not in target.curves:
            return Result.failure("target has no pitch centre")
        udev = _deviation(user)
        tdev = _at(_deviation(target), tau)
        region = np.zeros(T, bool)
        if attr.startswith("vibrato"):
            region[span[0] : span[1]] = True
            # leave ornament events alone: they are separate items
            for e in user.events:
                if e.kind != "onset":
                    region[e.start : e.end] = False
            ue, te = float(item.detail.get("user", 0.0)), float(item.detail.get("target", 0.0))
            if attr == "vibrato_extent" and ue > 1.0:
                d = np.where(region & np.isfinite(udev), (te / ue - 1.0) * np.nan_to_num(udev), 0.0)
            else:
                scale = 1.0 if attr == "vibrato_extent" or te <= 1.0 else ue / te
                ok = region & np.isfinite(udev) & np.isfinite(tdev)
                d = np.where(ok, scale * np.nan_to_num(tdev) - np.nan_to_num(udev), 0.0)
        else:
            m = cfg.event_margin_frames
            k = item.detail.get("target_note", -1)
            for e in user.events:
                if e.kind == attr and span[0] - m <= e.start < span[1] + m:
                    region[max(0, e.start - m) : e.end + m] = True
            t_events = [e for e in target.events if e.kind == attr]
            note_range = target.meta.get("notes", [])
            if 0 <= k < len(note_range):
                ns, ne = note_range[k]
                t_events = [e for e in t_events if ns <= e.start < ne]
            for e in t_events:
                region |= (tau >= e.start - m) & (tau < e.end + m)
            if not region.any():
                return Result.failure(f"no {attr} region found for this item")
            # events sit at note edges, where the running-median centre bends with them:
            # measure both contours against their *note* centre instead
            u_note = _note_spans(exp).get(k, span)
            uc = _core_median(user.curves["pitch_center"].values, *u_note)
            tc = _core_median(target.curves["pitch_center"].values, *note_range[k]) if 0 <= k < len(note_range) else np.nan
            if not (np.isfinite(uc) and np.isfinite(tc)):
                return Result.failure("no pitch centre for this note")
            uf = user.curves["f0_cents"].values - uc
            tf = _at(target.curves["f0_cents"].values, tau) - tc
            ok = region & np.isfinite(uf) & np.isfinite(tf)
            d = np.where(ok, np.nan_to_num(tf) - np.nan_to_num(uf), 0.0)
        d = d * _smooth_mask(np.abs(d) > 0, 1)
        return mk(f0_cents=d, requires=frozenset({"f0"}))

    if attr == "breathiness":
        return mk(aperiodic_db=-item.magnitude * _ramp_box(T, *span, ramp), requires=frozenset({"aperiodic"}))

    if attr == "loudness":
        return mk(gain_db=-item.magnitude * _ramp_box(T, *span, ramp), requires=frozenset({"gain"}))

    if attr == "dynamic_range":
        ur, tr = item.detail.get("user_range_db"), item.detail.get("target_range_db")
        if not ur or ur <= 0 or tr is None:
            return Result.failure("dynamic range item without ranges")
        lr = user.curves["loudness_rel"]
        ok = (lr.confidence > 0) & np.isfinite(lr.values)
        if ok.sum() < 3:
            return Result.failure("too few frames with loudness")
        dev = lr.values - np.median(lr.values[ok])
        g = np.interp(np.arange(T), np.flatnonzero(ok), (tr / ur - 1.0) * dev[ok])
        g = np.convolve(g, np.ones(5) / 5, "same")
        return mk(gain_db=g, requires=frozenset({"gain"}))

    learned = {"register": "register", "phone_match": "phones"}
    curve, col = learned.get(attr), None
    if attr.startswith("quality_"):
        curve = "phonation"
        labels = user.curves["phonation"].labels if "phonation" in user.curves else ()
        col = labels.index(attr[len("quality_"):]) if attr[len("quality_"):] in labels else None
    elif attr.startswith("laryngeal_"):
        curve = "laryngeal"
    if curve is not None:
        if curve not in user.curves or curve not in target.curves:
            return Result.failure(f"curve {curve!r} missing on one side")
        up = user.curves[curve].values
        idx = np.clip(np.round(tau).astype(int), 0, len(target.curves[curve].values) - 1)
        tp = target.curves[curve].values[idx]
        mask = np.zeros(T, bool)
        for sp in item.spans:
            mask[sp.start : sp.end] = True
        d = np.where(mask[:, None] & np.isfinite(up) & np.isfinite(tp), np.nan_to_num(tp) - np.nan_to_num(up), 0.0)
        if col is not None:
            keep = np.zeros(d.shape[1], bool)
            keep[col] = True
            d[:, ~keep] = 0.0
        return mk(curves={curve: d}, requires=frozenset({f"curve:{curve}"}))

    return Result.unavailable(f"no edit defined for attribute {attr!r}")

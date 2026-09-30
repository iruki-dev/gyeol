"""M5 explanation items beyond pitch / rhythm / ornaments.

All comparisons are made on aligned frames (user frame t ↔ target frame τ(t))
and only where both sides are confident.

* **phonation**: breathiness via the aperiodic-to-periodic ratio (signal
  layer, always available); register and phonation-quality posteriors when
  learned heads (M3) produced them.  Register items are flagged
  ``tentative`` — coaches phrase them as possibilities.
* **dynamics**: per-note loudness Δ and the phrase dynamic range.
* **diction**: phrase level only, compared with the *target singer's
  actual realisation* (their laryngeal / phone posteriors), never with a
  dictionary pronunciation.
* **remainder**: frames where the residual r differs far more than usual
  after τ and Δc are accounted for → "cannot judge".
"""

from __future__ import annotations

import numpy as np

from ..core.containers import ExplanationItem, Representation, Span
from ..dsp.base import runs


def _at(values: np.ndarray, tau: np.ndarray) -> np.ndarray:
    from .explain import _at as at

    return at(values, tau)


def _at_matrix(values: np.ndarray, tau: np.ndarray) -> np.ndarray:
    idx = np.clip(np.round(tau).astype(int), 0, len(values) - 1)
    return values[idx]


def phonation_dynamics_items(user: Representation, target: Representation, tau: np.ndarray, warp_conf: np.ndarray,
                             notes: list[tuple[int, int]], note_spans: dict[int, Span], min_conf: float) -> dict[tuple, ExplanationItem]:
    items: dict[tuple, ExplanationItem] = {}
    uc, tc = user.curves, target.curves
    T = user.grid.n_frames

    def note_frames(k: int, ns: int, ne: int) -> np.ndarray:
        return np.flatnonzero((tau >= ns) & (tau < ne))

    for name, attr, cat, unit in (("aperiodic_ratio", "breathiness", "phonation", "dB"), ("loudness_rel", "loudness", "dynamics", "dB")):
        if name not in uc or name not in tc:
            continue
        uv, ucf = uc[name].values, uc[name].confidence
        tv, tcf = _at(tc[name].values, tau), _at(tc[name].confidence, tau)
        ok = (ucf >= min_conf) & (tcf >= min_conf) & np.isfinite(uv) & np.isfinite(tv)
        for k, (ns, ne) in enumerate(notes):
            sel = note_frames(k, ns, ne)
            good = sel[ok[sel]]
            if good.size < 3 or k not in note_spans:
                continue
            d = uv - tv
            delta = np.full(T, np.nan)
            delta[good] = d[good]
            conf = float(np.mean(np.minimum(ucf[good], tcf[good]) * warp_conf[good]))
            items[(cat, attr, k)] = ExplanationItem(cat, attr, [note_spans[k]], float(np.median(d[good])), unit, conf,
                                                    delta=delta, detail={"target_note": k})
    # phrase dynamic range
    if "loudness_rel" in uc and "loudness_rel" in tc:
        uv = uc["loudness_rel"].masked(min_conf)
        tv = tc["loudness_rel"].masked(min_conf)
        uv, tv = uv[np.isfinite(uv)], tv[np.isfinite(tv)]
        if uv.size > 20 and tv.size > 20:
            ur, tr = np.subtract(*np.percentile(uv, [90, 10])), np.subtract(*np.percentile(tv, [90, 10]))
            items[("dynamics", "dynamic_range", -1)] = ExplanationItem(
                "dynamics", "dynamic_range", [Span(0, T)], float(ur - tr), "dB", float(min(1.0, uv.size / 100)),
                detail={"user_range_db": float(ur), "target_range_db": float(tr)})
    # learned posteriors (M3 heads), when both sides have them
    for curve, cat in (("register", "phonation"), ("phonation", "phonation")):
        if curve not in uc or curve not in tc:
            continue
        up, upc = uc[curve].values, uc[curve].confidence
        tp, tpc = _at_matrix(tc[curve].values, tau), _at(tc[curve].confidence, tau)
        labels = uc[curve].labels
        ok = (upc >= min_conf) & (tpc >= min_conf) & np.all(np.isfinite(up), 1) & np.all(np.isfinite(tp), 1)
        for k, (ns, ne) in enumerate(notes):
            sel = note_frames(k, ns, ne)
            good = sel[ok[sel]]
            if good.size < 3 or k not in note_spans:
                continue
            u_mean, t_mean = up[good].mean(0), tp[good].mean(0)
            conf = float(np.mean(np.minimum(upc[good], tpc[good]) * warp_conf[good]))
            if curve == "register":
                ti, ui = int(np.argmax(t_mean)), int(np.argmax(u_mean))
                items[(cat, "register", k)] = ExplanationItem(
                    cat, "register", [note_spans[k]], float(u_mean[ti] - t_mean[ti]), "probability", conf,
                    detail={"target_note": k, "target": labels[ti] if labels else ti, "user": labels[ui] if labels else ui,
                            "status": "different" if ti != ui else "same", "tentative": True})
            else:
                for j, lab in enumerate(labels):
                    items[(cat, f"quality_{lab}", k)] = ExplanationItem(
                        cat, f"quality_{lab}", [note_spans[k]], float(u_mean[j] - t_mean[j]), "probability", conf,
                        detail={"target_note": k, "tentative": True})
    return items


def diction_items(user: Representation, target: Representation, tau: np.ndarray, warp_conf: np.ndarray, min_conf: float) -> dict[tuple, ExplanationItem]:
    """Phrase-level diction against the target's realisation (needs learned 'laryngeal' / 'phones' curves)."""
    items: dict[tuple, ExplanationItem] = {}
    uc, tc = user.curves, target.curves
    T = user.grid.n_frames
    if "laryngeal" in uc and "laryngeal" in tc:
        up, upc = uc["laryngeal"].values, uc["laryngeal"].confidence
        tp, tpc = _at_matrix(tc["laryngeal"].values, tau), _at(tc["laryngeal"].confidence, tau)
        ok = (upc >= min_conf) & (tpc >= min_conf) & np.all(np.isfinite(up), 1) & np.all(np.isfinite(tp), 1)
        labels = uc["laryngeal"].labels or ("lenis", "aspirated", "fortis")
        dom = np.argmax(np.nan_to_num(tp), 1)
        for j, lab in enumerate(labels):
            sel = ok & (dom == j) & (np.nan_to_num(tp[:, j]) > 0.5)
            if sel.sum() < 3:
                continue
            spans = [Span(s, e) for s, e in runs(sel)]
            items[("diction", f"laryngeal_{lab}", -1)] = ExplanationItem(
                "diction", f"laryngeal_{lab}", spans, float(np.mean(up[sel, j] - tp[sel, j])), "probability",
                float(np.mean(np.minimum(upc[sel], tpc[sel]) * warp_conf[sel])),
                detail={"n_frames": int(sel.sum()), "phrase_level": True, "reference": "target singer's realisation"})
    if "phones" in uc and "phones" in tc:
        up, tp = uc["phones"].values, _at_matrix(tc["phones"].values, tau)
        ok = (uc["phones"].confidence >= min_conf) & (_at(tc["phones"].confidence, tau) >= min_conf)
        ok &= np.all(np.isfinite(up), 1) & np.all(np.isfinite(tp), 1)
        if ok.sum() >= 5:
            tv = 0.5 * np.abs(up[ok] - tp[ok]).sum(1)  # total-variation distance per frame
            items[("diction", "phone_match", -1)] = ExplanationItem(
                "diction", "phone_match", [Span(0, T)], float(1.0 - tv.mean()), "similarity", float(np.mean(warp_conf[ok])),
                detail={"phrase_level": True, "reference": "target singer's realisation"})
    return items


def remainder_spans(user: Representation, target: Representation, tau: np.ndarray, voiced: np.ndarray, z_threshold: float,
                    min_frames: int) -> list[Span]:
    """Frames where the residual difference is an outlier (robust z) → 'cannot judge'."""
    if user.residual is None or target.residual is None:
        return []
    ru = user.residual
    rt = target.residual[np.clip(np.round(tau).astype(int), 0, len(target.residual) - 1)]
    d = np.linalg.norm(ru - rt, axis=1)
    med = np.median(d[voiced]) if voiced.any() else np.median(d)
    mad = 1.4826 * np.median(np.abs(d[voiced] - med)) + 1e-9 if voiced.any() else 1.0
    bad = voiced & ((d - med) / mad > z_threshold)
    return [Span(s, e, reason="remainder") for s, e in runs(bad) if e - s >= min_frames]

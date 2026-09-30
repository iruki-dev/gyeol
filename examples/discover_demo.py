"""Discovery workflow demo (M7) on synthetic singing.

    python examples/discover_demo.py --out /tmp/gyeol_discover

1. Render 4 synthetic "singers" (different glottal open quotient) × 2 keys ×
   {normal, breathy}, and compute a low-level phonation feature (frame-
   normalised log-mel shape).  Breathiness is *not* given to the model.
2. Train a TopK SAE on the features; match latents to known labels (pitch)
   and to a v0.1-style DSP feature (aperiodic ratio); list novel latents.
3. Fit a conditional direction (f0 band cells, f0 and loudness regressed out)
   from the on/off pairs of two "discovery" singers.
4. Transfer tests on the two unseen singers and across pitch bands; promote
   if they pass (registry JSON), and show a control (raw pitch) being rejected.

Residual-energy monitoring (:class:`gyeol.discover.ResidualMonitor`) needs
``rep.residual`` from a trained M4 autoencoder; it is not run here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from gyeol.attributes.extract import analyze
from gyeol.core import Provenance, Recording
from gyeol.discover import (
    DirectionConfig,
    PromotionRegistry,
    SAEConfig,
    concat,
    evaluate_promotion,
    feature_stats,
    fit_conditional_directions,
    match_features,
    pair_frames,
    train_sae,
)
from gyeol.encoders.mel import LogMel
from gyeol.pitch.adapters import PyinTracker, SHSTracker, YinTracker
from gyeol.synth import SynthNote, melody

NOTES = [(262, "a", "s"), (294, "o", "t"), (330, "i", "k"), (349, "e", "h"), (392, "a", "s"), (330, "u", "t")]


def render(oq: float, transpose: float, breathy: bool, seed: int):
    notes = [SynthNote(f, 0.3, gap_after=0.1, vowel=v, consonant=c) for f, v, c in NOTES]
    m = melody(notes, sr=44100, transpose_cents=transpose, seed=seed, open_quotient=oq, aspiration=0.35 if breathy else 0.02)
    rep = analyze(Recording(m.audio, 44100, Provenance.SYNTHETIC), trackers=[PyinTracker(), YinTracker(), SHSTracker()]).unwrap()
    L = LogMel(n_mels=24)(torch.tensor(m.audio, dtype=torch.float32)[None])[0].numpy()[: rep.grid.n_frames]
    f0 = np.where(rep.curves["f0_cents"].confidence > 0.5, rep.curves["f0_cents"].values, np.nan)
    return dict(F=L - L.mean(1, keepdims=True), f0=f0, loud=rep.curves["loudness"].values, notes=rep.meta["notes"],
                ap=rep.curves["aperiodic_ratio"].values)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("gyeol_discover_out"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    print("1) rendering and analysing 16 synthetic recordings …")
    recs = {(s, tr, lab): render(oq, tr, bool(lab), 100 * s + 10 * (tr > 0) + lab)
            for s, oq in enumerate((0.45, 0.55, 0.62, 0.7)) for tr in (-500, 400) for lab in (0, 1)}

    print("2) TopK SAE on the phonation features")
    F = np.vstack([r["F"][np.isfinite(r["f0"])] for r in recs.values()])
    P = np.concatenate([r["f0"][np.isfinite(r["f0"])] for r in recs.values()])
    A = np.concatenate([r["ap"][np.isfinite(r["f0"])] for r in recs.values()])
    fit = train_sae(F, SAEConfig(n_latents=32, k=4, steps=1500))
    st = feature_stats(fit, F)
    print(f"   frames {len(F)}, FVU {fit.fvu(F):.3f}, active latents {(st.frequency > 0).sum()}/32")
    mt = match_features(fit.codes(F), {"pitch (known attribute)": (P, "regress"), "aperiodic ratio (v0.1 DSP)": (A, "regress")})
    for t in ("pitch (known attribute)", "aperiodic ratio (v0.1 DSP)"):
        print(f"   best latents for {t}: " + ", ".join(f"#{m.feature} ({m.score:.2f}{m.detail})" for m in mt.best_for_target(t, 3)))
    print(f"   novel latents (no match > 0.3), by energy: {mt.novel(0.3, st.energy_share, st.frequency > 0)[:6]}")

    print("3) conditional direction from on/off pairs of singers 0–1")
    parts = []
    for s in (0, 1):
        for tr in (-500, 400):
            a, b = recs[(s, tr, 0)], recs[(s, tr, 1)]
            n = min(len(a["F"]), len(b["F"]))
            parts.append(pair_frames(a["F"][:n], b["F"][:n], np.arange(n, dtype=float), a["f0"][:n], b["f0"][:n],
                                     a["loud"][:n], b["loud"][:n], s))
    cd = fit_conditional_directions(concat(parts), DirectionConfig(f0_edges_cents=(-700.0,)))
    for cell, d in sorted(cd.cells.items(), key=str):
        print(f"   cell f0-band {cell[0]}: {d.n_pairs} pairs, size {d.size:.2f}, sign consistency {d.sign_consistency:.2f}")
    for (a, b), c in cd.similarity().items():
        print(f"   cosine between f0 bands {a[0]} and {b[0]}: {c:.2f} (the direction depends on pitch)")

    print("4) transfer tests on unseen singers 2–3 and across pitch bands")
    X, y, units, singers, pitch = [], [], [], [], []
    for (s, tr, lab), r in recs.items():
        if s < 2:
            continue
        v = np.isfinite(r["f0"])
        note = np.full(len(r["F"]), -1)
        for k, (a, b) in enumerate(r["notes"]):
            note[a:b] = k
        X.append(cd.score(r["F"][v], r["f0"][v], r["loud"][v]))
        y += [lab] * int(v.sum())
        units += [f"{s}{tr}{lab}n{k}" for k in note[v]]
        singers += [s] * int(v.sum())
        pitch.append(r["f0"][v])
    X, y, units, singers, pitch = np.concatenate(X), np.array(y, float), np.array(units), np.array(singers), np.concatenate(pitch)
    reg = PromotionRegistry(args.out / "promoted.json")
    for name, cand in (("breathy_direction", X), ("control: raw pitch", pitch)):
        d = evaluate_promotion(name, cand, y, units, singers, pitch)
        for split, r in d.results.items():
            print(f"   {name:18s} {split:12s} gain {r.gain:+.2f} (in-distribution {r.in_distribution_gain:+.2f}), p={r.p_value:.2g} → "
                  f"{'pass' if r.passed else 'FAIL: ' + r.reason}")
        if d.passed and name not in reg:
            reg.register(d, cand, y, {"type": "conditional_direction", "input": "log-mel shape", "cells": len(cd.cells)})
            print(f"   → promoted '{name}' (registry: {reg.path})")
        elif not d.passed:
            print(f"   → '{name}' not promoted")
    return 0


if __name__ == "__main__":
    sys.exit(main())

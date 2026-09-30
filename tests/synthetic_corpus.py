"""Tiny synthetic 'singers × register × phonation' corpus for head / probe tests."""

from __future__ import annotations

import numpy as np

from gyeol.attributes.extract import analyze
from gyeol.core import Provenance, Recording
from gyeol.encoders import DSPFrameFeatures
from gyeol.synth import VOWELS, sung_vowel
from gyeol.train.heads import FrameExample

from .helpers import dsp_trackers

SR = 22050
REGISTERS = {"chest": ((110, 200), 0.5), "mixed": ((220, 330), 0.65), "falsetto": ((380, 600), 0.85)}
PHONATIONS = {"modal": (0.0, 0.02), "breathy": (0.15, 0.6), "pressed": (-0.15, 0.0)}  # (ΔOQ, aspiration)


def corpus(n_singers: int = 6, seed: int = 0, dur: float = 0.6):
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_singers):
        scale = 0.88 + 0.24 * s / max(1, n_singers - 1)  # vocal-tract size per singer
        for ri, (_reg, ((lo, hi), oq)) in enumerate(REGISTERS.items()):
            for pi, (ph, (doq, asp)) in enumerate(PHONATIONS.items()):
                vowel = "aeiou"[(s + ri + pi) % 5]
                fv, bv = VOWELS[vowel]
                v = sung_vowel(f0=float(rng.uniform(lo, hi)), duration=dur, sr=SR, formants=tuple(f * scale for f in fv),
                               bandwidths=bv, open_quotient=float(np.clip(oq + doq, 0.3, 0.95)), aspiration=asp,
                               vibrato_rate=5.5, vibrato_extent_cents=float(rng.uniform(0, 40)), seed=int(rng.integers(1 << 30)))
                x = np.r_[np.zeros(SR // 10), v.audio, np.zeros(SR // 10)]
                x = x + rng.standard_normal(len(x)) * 1e-4
                out.append({"audio": x, "singer": f"s{s}", "register": ri, "phonation": ph})
    return out


def examples(items, encoder=None):
    enc = encoder or DSPFrameFeatures()
    exs, reps = [], []
    for it in items:
        rep = analyze(Recording(it["audio"], SR, Provenance.SYNTHETIC), trackers=dsp_trackers()).unwrap()
        f = enc.from_representation(rep)
        voiced = np.isfinite(rep.curves["f0_cents"].values)
        reg = np.where(voiced, it["register"], -1)
        ph = np.full((len(f), 5), np.nan)
        ph[voiced] = 0.0
        if it["phonation"] == "breathy":
            ph[voiced, 0] = 1.0
        if it["phonation"] == "pressed":
            ph[voiced, 1] = 1.0
        ph[voiced, 2:] = np.nan  # twang / fry / rough not represented in this corpus
        exs.append(FrameExample(f, {"register": reg, "phonation": ph}, it["singer"]))
        reps.append(rep)
    return exs, reps

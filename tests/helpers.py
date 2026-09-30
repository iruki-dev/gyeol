"""Shared helpers for the test-suite (tiny, deterministic, no downloads)."""

from __future__ import annotations

import numpy as np

from gyeol.attributes.extract import analyze
from gyeol.core import Provenance, Recording
from gyeol.pitch.adapters import PyinTracker, SHSTracker, YinTracker
from gyeol.synth import SynthNote, melody

SR = 44100


def dsp_trackers():
    return [PyinTracker(), YinTracker(), SHSTracker()]


BASE = [(262, "a", "s"), (294, "o", "t"), (330, "i", "k"), (349, "e", "h"), (392, "a", "s"), (330, "u", "t")]


def make_melody(detune=(0,) * 6, shifts=(0,) * 6, vib=(0,) * 6, scoop=(0,) * 6, fall=(0,) * 6, dur=0.45, gap=0.2,
                transpose=0.0, seed=0, **kw):
    notes = [SynthNote(f * 2 ** (d / 1200), dur, gap_after=gap, vowel=v, consonant=c, onset_shift_s=s,
                       vibrato_rate=5.5 if e else 0.0, vibrato_extent_cents=e, scoop_cents=sc, fall_cents=fa)
             for (f, v, c), d, s, e, sc, fa in zip(BASE, detune, shifts, vib, scoop, fall)]
    return melody(notes, sr=SR, transpose_cents=transpose, seed=seed, **kw)


def rep_of(audio, provenance=Provenance.SYNTHETIC, owner=None, sr=SR, **kw):
    if provenance is Provenance.USER and owner is None:
        owner = "tester"
    return analyze(Recording(np.asarray(audio, float), sr, provenance, owner_id=owner), trackers=dsp_trackers(), **kw)


def pad_to(*arrays):
    n = max(len(a) for a in arrays)
    return [np.pad(a, (0, n - len(a))) for a in arrays]

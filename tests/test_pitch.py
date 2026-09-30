import numpy as np
import pytest

from gyeol._dsp import hz_to_cents
from gyeol.features.notes import segment_notes, vibrato
from gyeol.synth import phrase, sung_vowel

from .conftest import HOP, SR


@pytest.mark.parametrize("f0", [90, 220, 440, 700, 1000])
def test_consensus_f0_accuracy_with_vibrato(f0, analyse_pitch):
    v = sung_vowel(f0=f0, duration=1.0, vibrato_rate=5.5, vibrato_extent_cents=50)
    n, pc = analyse_pitch(v.audio)
    truth = v.truth["f0_track"][np.minimum(np.arange(n) * HOP, len(v.audio) - 1)]
    err = np.abs(pc.cents - hz_to_cents(truth))
    assert pc.voiced.mean() > 0.9
    assert np.nanmedian(err) < 5.0
    assert np.nanpercentile(err, 95) < 15.0
    # trackers agree → high confidence
    assert np.median(pc.confidence[pc.voiced]) > 0.8


def test_silence_is_unvoiced(analyse_pitch):
    x = np.zeros(SR)
    _, pc = analyse_pitch(x)
    assert not pc.voiced.any()
    assert np.isnan(pc.f0).all()


def test_note_segmentation_legato_and_gaps(analyse_pitch):
    p = phrase([(220, 0.6), (247, 0.6), (294, 0.8)], vibrato_rate=5.5, vibrato_extent_cents=60, gap_s=0.0)
    _, pc = analyse_pitch(p.audio)
    notes = segment_notes(pc.cents, pc.voiced, HOP / SR)
    assert len(notes) == 3
    bounds = [n.start * HOP / SR for n in notes[1:]]
    assert bounds == pytest.approx([0.6, 1.2], abs=0.05)


@pytest.mark.parametrize("rate,extent", [(5.0, 30.0), (6.0, 80.0), (7.0, 50.0)])
def test_vibrato_rate_and_extent(rate, extent, analyse_pitch):
    v = sung_vowel(f0=330, duration=1.5, vibrato_rate=rate, vibrato_extent_cents=extent)
    _, pc = analyse_pitch(v.audio)
    vb = vibrato(pc.cents[pc.voiced], HOP / SR)
    assert vb is not None
    assert vb.rate_hz == pytest.approx(rate, abs=0.2)
    assert vb.extent_cents == pytest.approx(extent, rel=0.2)
    assert vb.n_cycles >= 2


def test_no_vibrato_detected_on_steady_tone(analyse_pitch):
    v = sung_vowel(f0=330, duration=1.5)
    _, pc = analyse_pitch(v.audio)
    vb = vibrato(pc.cents[pc.voiced], HOP / SR)
    assert vb is None or vb.extent_cents < 5

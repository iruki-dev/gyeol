import numpy as np
import pytest
from scipy import signal

from gyeol.core import FrameGrid, Provenance, Status
from gyeol.frontend import BackingTrackCanceller, CallableSeparator, assess, detect_bleed, estimate_snr
from gyeol.dsp.base import si_sdr
from gyeol.synth import SynthNote, melody, phrase, sung_vowel
from gyeol.verification import degrade

from .helpers import SR, make_melody, rep_of


# --- frontend -----------------------------------------------------------------

def test_snr_tracks_added_noise():
    p = phrase([(220, 0.8), (262, 0.8), (294, 0.8)], sr=SR, gap_s=0.4)
    g = FrameGrid.for_samples(len(p.audio), SR)
    voiced = np.zeros(g.n_frames, bool)
    for a, b, _ in p.truth["notes"]:
        voiced[g.frame_of(a) : g.frame_of(b)] = True
    est = [estimate_snr(degrade.add_noise(p.audio, s), SR, g, voiced).unwrap().snr_db for s in (50, 35, 20)]
    assert est[0] > est[1] > est[2] and est[2] == pytest.approx(20, abs=6)


def test_bleed_detection():
    rng = np.random.default_rng(0)
    voice = phrase([(220, 1.0), (330, 1.0)], sr=SR, gap_s=0.2).audio
    backing = signal.lfilter([1], [1, -0.9], rng.standard_normal(len(voice))) * 0.02
    leak = np.r_[np.zeros(300), backing[:-300]]
    quiet = detect_bleed(voice + 0.001 * leak, backing, SR).unwrap()  # true leak ≈ −68 dB
    medium = detect_bleed(voice + 0.1 * leak, backing, SR).unwrap()  # ≈ −28 dB
    loud = detect_bleed(voice + 1.0 * leak, backing, SR).unwrap()  # ≈ −8.5 dB
    assert quiet.bleed_db < -20 < loud.bleed_db  # the default policy threshold separates them
    assert loud.bleed_db == pytest.approx(-8.5, abs=2.0) and medium.bleed_db == pytest.approx(-28, abs=5.0)
    assert loud.lag_s == pytest.approx(300 / SR, abs=1e-3)


def test_assess_flags_and_frame_factor():
    m = make_melody()
    x = m.audio
    g = FrameGrid.for_samples(len(x), SR)
    voiced = np.zeros(g.n_frames, bool)
    for a, b, _ in m.truth["notes"]:
        voiced[g.frame_of(a) : g.frame_of(b)] = True
    clean = assess(x, SR, g, voiced)
    assert clean.ok, clean.flags
    noisy = assess(degrade.add_noise(x, 12), SR, g, voiced)
    assert "low_snr" in noisy.flags and noisy.frame_factor[voiced].mean() < clean.frame_factor[voiced].mean()
    clipped = assess(np.clip(x * 20, -1, 1), SR, g, voiced)
    assert "clipping" in clipped.flags
    lp = assess(degrade.bandlimit(x, SR, 3400), SR, g, voiced)
    assert "narrow_bandwidth" in lp.flags


def test_separation_adapters():
    voice = phrase([(220, 1.0), (330, 1.0)], sr=SR, gap_s=0.2).audio
    rng = np.random.default_rng(3)
    backing = signal.lfilter([1], [1, -0.95], rng.standard_normal(len(voice) + 4000)) * 0.02
    mix = voice + 0.7 * np.r_[np.zeros(123), backing[: len(voice) - 123]]
    out = BackingTrackCanceller(backing, SR).separate(mix, SR).unwrap()
    assert si_sdr(out, voice) > si_sdr(mix, voice) + 8
    bad = CallableSeparator(lambda a, sr: a[:10])
    assert bad.separate(mix, SR).status is Status.FAILED


# --- attribute curves & events -----------------------------------------------

@pytest.fixture(scope="module")
def ornamented():
    notes = [SynthNote(262, 0.5, consonant="s"), SynthNote(294, 0.5, scoop_cents=150, vowel="o"),
             SynthNote(330, 0.9, vibrato_rate=5.5, vibrato_extent_cents=60, vowel="i", consonant="t"),
             SynthNote(392, 0.6, kkeokki_cents=150, gap_after=0, glide_to_next=True, glide_s=0.2),
             SynthNote(262, 0.7, fall_cents=200, vowel="u")]
    m = melody(notes, sr=SR)
    return m, rep_of(m.audio).unwrap()


def test_notes_and_events(ornamented):
    m, rep = ornamented
    h = rep.grid.hop_seconds
    assert len(rep.meta["notes"]) == 5
    for i, ((a, _b), (ta, _tb, _)) in enumerate(zip(rep.meta["notes"], m.truth["notes"])):
        # after a 200 ms glide the boundary is legitimately anywhere inside the glide
        assert a * h == pytest.approx(ta, abs=0.2 if i == 4 else 0.06)
    found = {e.kind: e for e in rep.events}
    assert set(found) == {"scoop", "kkeokki", "glide", "fall"}
    truth = {k: (a, b, mag) for k, a, b, mag in m.truth["events"]}
    for kind, e in found.items():
        a, b, mag = truth[kind]
        assert e.start * h == pytest.approx(a, abs=0.12), kind
        assert abs(e.magnitude) == pytest.approx(abs(mag), rel=0.25), kind


def test_vibrato_curves(ornamented):
    m, rep = ornamented
    c = rep.curves
    note = slice(rep.grid.frame_of(1.6), rep.grid.frame_of(2.0))
    conf = c["vibrato_extent"].confidence[note]
    assert (conf > 0.5).mean() > 0.5
    assert np.nanmedian(c["vibrato_rate"].values[note][conf > 0.5]) == pytest.approx(5.5, abs=0.3)
    assert np.nanmedian(c["vibrato_extent"].values[note][conf > 0.5]) == pytest.approx(60, rel=0.25)
    # no vibrato on the steady first note
    first = slice(rep.grid.frame_of(0.3), rep.grid.frame_of(0.6))
    assert np.nanmax(c["vibrato_extent"].confidence[first]) < 0.5


def test_vibrato_is_not_mistaken_for_kkeokki():
    m = melody([SynthNote(330, 1.5, vibrato_rate=5.5, vibrato_extent_cents=100)], sr=SR)
    rep = rep_of(m.audio).unwrap()
    assert not [e for e in rep.events if e.kind == "kkeokki"]


def test_loudness_is_gain_invariant_and_aperiodicity_tracks_breathiness():
    m = make_melody()
    a = rep_of(m.audio).unwrap()
    b = rep_of(0.2 * m.audio).unwrap()
    va, vb = a.curves["loudness_rel"].masked(0.5), b.curves["loudness_rel"].masked(0.5)
    ok = np.isfinite(va) & np.isfinite(vb)
    assert np.median(np.abs(va[ok] - vb[ok])) < 0.5
    ratios = []
    for asp in (0.0, 0.3, 0.6):
        v = sung_vowel(f0=220, duration=1.0, sr=SR, aspiration=asp)
        r = rep_of(np.r_[np.zeros(SR // 4), v.audio, np.zeros(SR // 4)]).unwrap()
        ratios.append(np.nanmedian(r.curves["aperiodic_ratio"].masked(0.3)))
    assert ratios[0] < ratios[1] < ratios[2]


def test_quality_lowers_confidence_and_bad_input_fails():
    m = make_melody()
    clean = rep_of(m.audio).unwrap()
    noisy = rep_of(degrade.add_noise(m.audio, 15)).unwrap()
    assert "low_snr" in noisy.quality["flags"]
    assert noisy.curves["aperiodic_ratio"].confidence.mean() < 0.5 * clean.curves["aperiodic_ratio"].confidence.mean()
    assert rep_of(np.zeros(SR)).status is Status.FAILED
    assert rep_of(np.zeros(1000) + 0.1).status is Status.FAILED
    noise_only = rep_of(np.random.default_rng(0).standard_normal(SR) * 0.1)
    assert noise_only.status is Status.FAILED and "voiced" in noise_only.reason


def test_representation_provenance_and_grid():
    m = make_melody()
    r = rep_of(m.audio, Provenance.USER, owner="u1").unwrap()
    assert r.provenance is Provenance.USER and r.grid.hop == 512 and r.grid.sr == SR
    for c in r.curves.values():
        assert c.grid == r.grid

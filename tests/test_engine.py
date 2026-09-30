import numpy as np
import pytest

import gyeol
from gyeol import Engine, EngineConfig, ValidityPolicy, VocalRepresentation
from gyeol.synth import phrase, sung_vowel
from gyeol.verification import degrade

from .conftest import add_white_noise


@pytest.fixture(scope="module")
def phrase_rep():
    p = phrase([(220, 0.6), (262, 0.6), (294, 0.8), (330, 0.9)], vibrato_rate=5.5, vibrato_extent_cents=40, gap_s=0.2, aspiration=0.05)
    x = add_white_noise(p.audio, 70)
    return gyeol.analyze(x, 16000, lyrics="국밥 좋다")


def test_representation_structure(phrase_rep):
    rep = phrase_rep
    assert len(rep.notes) == 4
    for name in gyeol.DIMENSIONS:
        assert name in rep.tracks, name
    assert rep.tracks["f1"].rate == pytest.approx(50.0)
    assert rep.tracks["alpha_ratio"].rate == pytest.approx(25.0)
    assert rep.tracks["envelope"].values.shape[1] == 24
    assert rep.tracks["band_aperiodicity"].values.shape[1] == 5
    # the nuisance side channel is separate from T_voice
    assert "snr_db" not in rep.tracks
    assert rep.nuisance.snr_db > 40
    assert rep.nuisance.effective_bandwidth_hz >= 7500


def test_note_descriptors(phrase_rep):
    medians = [n.features["f0_cents_median"] for n in phrase_rep.notes]
    expected = [1200 * np.log2(f / 440) for f in (220, 262, 294, 330)]
    assert medians == pytest.approx(expected, abs=10)
    for n in phrase_rep.notes:
        assert n.valid["vibrato_rate_hz"]
        assert n.features["vibrato_rate_hz"] == pytest.approx(5.5, abs=0.3)
        # 16 kHz input < 19 kHz: perturbation must be declared invalid
        assert not n.valid["jitter_local"]
    assert [n.syllable for n in phrase_rep.notes] == ["국", "밥", "좋", "다"]


def test_invalid_values_are_reported_with_reasons(phrase_rep):
    tr = phrase_rep.tracks["cpps"]
    assert 0.5 < tr.coverage() < 1.0
    assert "unvoiced" in tr.invalid_reasons
    assert np.isnan(tr.masked()[~tr.valid]).all()


def test_save_load_roundtrip(phrase_rep, tmp_path):
    path = tmp_path / "rep.npz"
    phrase_rep.save(path)
    back = VocalRepresentation.load(path)
    assert back.summary() == phrase_rep.summary()
    np.testing.assert_array_equal(back.tracks["h1h2c"].valid, phrase_rep.tracks["h1h2c"].valid)
    assert back.context.syllables[0]["text"] == "국"


def test_low_snr_invalidates_noise_measures_but_not_f0():
    v = sung_vowel(f0=220, duration=1.5, vibrato_rate=5.5, vibrato_extent_cents=40)
    x = np.concatenate([np.zeros(8000), v.audio, np.zeros(8000)])
    clean = gyeol.analyze(add_white_noise(x, 60), 16000)
    noisy = gyeol.analyze(add_white_noise(x, 15), 16000)
    assert clean.tracks["cpps"].coverage() > 0.4
    assert noisy.tracks["cpps"].coverage() == 0.0
    assert noisy.tracks["cpps"].invalid_reasons["low_snr"] > 0.5
    assert noisy.tracks["f0_hz"].coverage() > 0.4


def test_bandlimited_input_invalidates_high_band_dimensions():
    sr = 44100
    v = sung_vowel(f0=220, duration=1.5, sr=sr, aspiration=0.1)
    x = np.concatenate([np.zeros(sr // 2), v.audio, np.zeros(sr // 2)])
    x = degrade.add_noise(x, 60, active_only=True)
    lp = degrade.bandlimit(x, sr, 3400)
    rep = gyeol.analyze(lp, sr)
    assert rep.nuisance.effective_bandwidth_hz < 4500
    assert rep.nuisance.codec_suspected
    for name in ("band_aperiodicity", "spr", "alpha_ratio", "f1"):
        assert rep.tracks[name].coverage() == 0.0
        assert "bandwidth" in rep.tracks[name].invalid_reasons
    assert rep.tracks["f0_hz"].coverage() > 0.4


def test_high_f0_gates_h1h2c_and_f1():
    rep = gyeol.analyze(sung_vowel(f0=500, duration=1.2, vowel="i", vibrato_rate=5.5, vibrato_extent_cents=40).audio, 16000)
    # /i/ F1 = 300 Hz < f0: F1 cannot be resolved, so H1*–H2* must not be reported
    assert rep.tracks["f1"].coverage() == 0.0
    assert rep.tracks["h1h2c"].coverage() == 0.0
    assert rep.tracks["f2"].coverage() > 0.5


def test_lossy_container_flag():
    v = sung_vowel(f0=220, duration=1.0)
    rep = Engine().analyze(v.audio, 16000, source_path="take.mp3")
    assert rep.nuisance.codec_suspected
    assert rep.tracks["band_aperiodicity"].coverage() == 0.0
    assert rep.tracks["band_aperiodicity"].invalid_reasons["codec"] == 1.0


def test_policy_is_configurable():
    v = sung_vowel(f0=220, duration=1.0)
    x = add_white_noise(np.concatenate([np.zeros(8000), v.audio, np.zeros(8000)]), 25)
    strict = gyeol.analyze(x, 16000)
    lax = Engine(EngineConfig(policy=ValidityPolicy(min_snr_db=10.0))).analyze(x, 16000)
    assert strict.tracks["cpps"].coverage() == 0.0
    assert lax.tracks["cpps"].coverage() > 0.3


def test_external_notes_and_file_input(tmp_path):
    from gyeol.io import save_audio

    p = phrase([(262, 0.5), (330, 0.5)], gap_s=0.1)
    path = tmp_path / "take.wav"
    save_audio(path, p.audio, 16000)
    rep = gyeol.analyze(path, notes=[(0.0, 0.5), (0.6, 1.1)], lyrics="도레")
    assert len(rep.notes) == 2
    assert rep.context.alignment_source == "heuristic-note"
    assert rep.notes[1].syllable == "레"

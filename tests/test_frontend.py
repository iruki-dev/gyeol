import numpy as np
import pytest
from scipy import signal

from gyeol._dsp import n_frames, si_sdr
from gyeol.features.pitch import consensus
from gyeol.frontend import BackingTrackCanceller, CallableSeparator, DeviceProfile, LTASNormalizer, separation_agreement
from gyeol.frontend.nuisance import clipping_fraction, effective_bandwidth, estimate
from gyeol.synth import phrase, sung_vowel
from gyeol.verification import degrade


def _voiced(x, sr=16000):
    n = n_frames(len(x), 160)
    return n, consensus(x, sr, 160, n).voiced


def test_snr_estimate_tracks_added_noise():
    p = phrase([(220, 0.8), (262, 0.8), (294, 0.8)], gap_s=0.4)
    est = []
    for snr in (50, 35, 20):
        x = degrade.add_noise(p.audio, snr)
        n, voiced = _voiced(x)
        rep, frame_snr = estimate(x, 16000, x, 16000, 160, n, voiced)
        est.append(rep.snr_db)
        assert frame_snr.shape == (n,)
    assert est[0] > est[1] > est[2]
    assert est[2] == pytest.approx(20, abs=6)


def test_bandwidth_and_cliff_detection():
    sr = 44100
    v = sung_vowel(f0=220, duration=1.5, sr=sr, aspiration=0.1)
    x = degrade.add_noise(np.concatenate([np.zeros(sr // 2), v.audio, np.zeros(sr // 2)]), 60)
    voiced_t = np.zeros(n_frames(len(x), sr // 100), bool)
    voiced_t[55:195] = True
    bw_full, cliff_full = effective_bandwidth(x, sr, voiced_t)
    bw_lp, cliff_lp = effective_bandwidth(degrade.bandlimit(x, sr, 5000), sr, voiced_t)
    assert bw_full > 10000 and not cliff_full
    assert 4000 < bw_lp < 5500 and cliff_lp


def test_clipping_fraction():
    x = np.sin(np.linspace(0, 200 * np.pi, 16000))
    assert clipping_fraction(x) < 1e-3
    assert clipping_fraction(degrade.clip(x, 0.5)) > 0.2


def test_device_profile_restores_reference_spectrum():
    sr = 16000
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(sr * 4)
    # consumer device: high-pass ~200 Hz, −10 dB above 3 kHz
    dev = degrade.device_response(ref, sr, [0, 100, 250, 2500, 3500, 8000], [-30, -20, 0, 0, -10, -10])
    prof = DeviceProfile.fit(ref, dev, sr)
    assert prof.highpass_hz is not None and 150 < prof.highpass_hz < 400
    fixed = prof.apply(dev, sr)
    f, p_ref = signal.welch(ref, sr, nperseg=1024)
    _, p_fix = signal.welch(fixed, sr, nperseg=1024)
    _, p_dev = signal.welch(dev, sr, nperseg=1024)
    band = (f > 500) & (f < 7000)
    err_fixed = np.abs(10 * np.log10(p_fix[band] / p_ref[band]))
    err_dev = np.abs(10 * np.log10(p_dev[band] / p_ref[band]))
    assert np.median(err_fixed) < 1.5 < np.median(err_dev)


def test_device_profile_json_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(16000 * 2)
    prof = DeviceProfile.fit(ref, 0.5 * ref, 16000, name="x")
    prof.save(tmp_path / "p.json")
    back = DeviceProfile.load(tmp_path / "p.json")
    assert back.name == "x" and back.band == prof.band
    np.testing.assert_allclose(back.fir(16000), prof.fir(16000))


def test_ltas_normalizer_moves_toward_reference():
    sr = 16000
    rng = np.random.default_rng(2)
    ref = rng.standard_normal(sr * 3)
    dev = degrade.device_response(rng.standard_normal(sr * 3), sr, [0, 2000, 4000, 8000], [0, 0, -12, -12])
    norm = LTASNormalizer.from_reference([ref], sr)
    out = norm.apply(dev, sr)
    f, p_ref = signal.welch(ref, sr, nperseg=1024)
    _, p_out = signal.welch(out, sr, nperseg=1024)
    _, p_dev = signal.welch(dev, sr, nperseg=1024)
    hi = (f > 4500) & (f < 7500)
    tilt = lambda p: np.median(10 * np.log10(p[hi])) - np.median(10 * np.log10(p[(f > 500) & (f < 1500)]))  # noqa: E731
    assert abs(tilt(p_out) - tilt(p_ref)) < abs(tilt(p_dev) - tilt(p_ref)) - 6


def test_backing_track_canceller():
    sr = 16000
    voice = phrase([(220, 1.0), (330, 1.0)], gap_s=0.2).audio
    rng = np.random.default_rng(3)
    backing = signal.lfilter([1], [1, -0.95], rng.standard_normal(len(voice) + 4000)) * 0.02
    delay = 123
    mix = voice + 0.7 * np.r_[np.zeros(delay), backing[: len(voice) - delay]]
    out = BackingTrackCanceller(backing, sr).separate(mix, sr)
    assert si_sdr(out, voice) > 15
    assert si_sdr(out, voice) > si_sdr(mix, voice) + 8


def test_callable_separator_and_agreement():
    x = np.random.default_rng(0).standard_normal(16000)
    sep = CallableSeparator(lambda a, sr: a * 0.5, name="half")
    assert np.allclose(sep.separate(x, 16000), 0.5 * x)
    assert separation_agreement(x, x + 1e-3 * np.random.default_rng(1).standard_normal(16000)) > 40

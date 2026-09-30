import numpy as np
import pytest

from gyeol.features.cepstral import cpps
from gyeol.features.formants import iseli_alwan_correction, track_formants
from gyeol.features.glottal import glottal_track
from gyeol.features.perturbation import perturbation
from gyeol.features.source import corrected_harmonic_measures
from gyeol.features.spectral import band_aperiodicity, harmonic_peaks, spectrogram, subharmonic_ratio
from gyeol.synth import sung_vowel

from .conftest import HOP, SR, add_white_noise


def _core(x, analyse_pitch):
    n, pc = analyse_pitch(x)
    spec = spectrogram(x, SR, HOP, n)
    harm = harmonic_peaks(spec, pc.f0)
    return n, pc, spec, harm


@pytest.mark.parametrize("f0", [110, 220, 440])
def test_cpps_decreases_with_noise(f0, analyse_pitch):
    v = sung_vowel(f0=f0, duration=1.0, vibrato_rate=5.5, vibrato_extent_cents=40)
    n, pc = analyse_pitch(v.audio)
    vals = []
    for snr in (60, 30, 10, 0):
        x = add_white_noise(v.audio, snr, seed=1)
        vals.append(np.nanmedian(cpps(x, SR, HOP, n, pc.f0)[pc.voiced]))
    assert all(a > b for a, b in zip(vals, vals[1:])), vals


def test_cpps_decreases_with_breathiness(analyse_pitch):
    vals = []
    for asp in (0.0, 0.3, 0.6):
        v = sung_vowel(f0=220, duration=1.0, aspiration=asp)
        n, pc = analyse_pitch(v.audio)
        assert pc.voiced.mean() > 0.9
        vals.append(np.nanmedian(cpps(v.audio, SR, HOP, n, pc.f0)[pc.voiced]))
    assert vals[0] > vals[1] > vals[2]


def test_band_aperiodicity_tracks_aspiration(analyse_pitch):
    mids = []
    for asp in (0.0, 0.3, 0.6):
        v = sung_vowel(f0=220, duration=1.0, aspiration=asp)
        n, pc, spec, _ = _core(v.audio, analyse_pitch)
        ap, hnr, ok = band_aperiodicity(spec, pc.f0)
        assert ok[pc.voiced].mean() > 0.9
        mids.append(np.nanmedian(ap[pc.voiced, 1:3]))
    assert mids[0] < mids[1] < mids[2]
    assert np.all(ap[np.isfinite(ap)] <= 0.0)


@pytest.mark.parametrize("f0", [110, 220])
@pytest.mark.parametrize("vowel", ["a", "i", "u"])
def test_formants_low_f0(f0, vowel, analyse_pitch):
    v = sung_vowel(f0=f0, duration=1.2, vowel=vowel, vibrato_rate=5.5, vibrato_extent_cents=40, aspiration=0.05)
    n, pc, _, harm = _core(v.audio, analyse_pitch)
    ft = track_formants(v.audio, SR, HOP, n, pc.f0, harm)
    med = np.nanmedian(ft.freq[20:-20], axis=0)[:3]
    truth = np.array(v.truth["formants"][:3])
    rel = np.abs(med - truth) / truth
    # F2/F3 tight; F1 of /u/ at 220 Hz is known to be biased toward 2·f0
    assert rel[1] < 0.06 and rel[2] < 0.06
    assert rel[0] < (0.2 if (vowel == "u" and f0 == 220) else 0.06)


def test_sweep_estimator_used_at_high_f0(analyse_pitch):
    v = sung_vowel(f0=660, duration=1.2, vowel="a", vibrato_rate=5.5, vibrato_extent_cents=60)
    n, pc, _, harm = _core(v.audio, analyse_pitch)
    ft = track_formants(v.audio, SR, HOP, n, pc.f0, harm)
    assert (ft.method[pc.voiced] == 2).all()
    med = np.nanmedian(ft.freq[20:-20], axis=0)
    truth = np.array(v.truth["formants"][:3])
    assert np.all(np.abs(med[1:3] - truth[1:3]) / truth[1:3] < 0.08)


def test_iseli_alwan_correction_is_zero_at_dc_and_peaks_at_formant():
    assert iseli_alwan_correction(np.array(0.0), np.array(700.0), np.array(80.0), SR) == pytest.approx(0.0, abs=1e-9)
    f = np.linspace(100, 2000, 400)
    c = iseli_alwan_correction(f, np.full_like(f, 700.0), np.full_like(f, 80.0), SR)
    assert f[np.argmax(c)] == pytest.approx(700, abs=10)


@pytest.mark.parametrize("f0", [120, 220])
def test_h1h2c_orders_open_quotient_and_is_vowel_independent(f0, analyse_pitch):
    res = {}
    for vowel in ("a", "i") if f0 == 120 else ("a",):
        for oq in (0.4, 0.6, 0.8):
            v = sung_vowel(f0=f0, duration=1.0, vowel=vowel, open_quotient=oq, vibrato_rate=5.5, vibrato_extent_cents=30)
            n, pc, _, harm = _core(v.audio, analyse_pitch)
            ft = track_formants(v.audio, SR, HOP, n, pc.f0, harm)
            hm = corrected_harmonic_measures(harm, pc.f0, ft.freq, SR)
            res[(vowel, oq)] = np.nanmedian(hm["h1h2c"][20:-20])
    assert res[("a", 0.4)] < res[("a", 0.6)] < res[("a", 0.8)]
    if f0 == 120:
        # the formant correction removes the vowel effect (uncorrected H1–H2 differs by > 5 dB)
        for oq in (0.4, 0.6, 0.8):
            assert abs(res[("a", oq)] - res[("i", oq)]) < 2.0


def test_naq_increases_with_open_quotient(analyse_pitch):
    naq = []
    for oq in (0.4, 0.6, 0.8):
        v = sung_vowel(f0=150, duration=1.0, open_quotient=oq)
        _, pc = analyse_pitch(v.audio)
        g = glottal_track(v.audio, SR, HOP, pc.f0)
        naq.append(np.nanmedian(g["naq"]))
    assert naq[0] < naq[1] < naq[2]
    assert 0.05 < naq[0] and naq[2] < 0.3


def test_shr_detects_period_doubling(analyse_pitch):
    out = []
    for sub in (0.0, 0.3):
        v = sung_vowel(f0=220, duration=1.0, subharmonic=sub)
        n, pc, spec, _ = _core(v.audio, analyse_pitch)
        # evaluate at the *perceived* 220 Hz so subharmonics fall between harmonics
        f0 = np.where(pc.voiced, 220.0, np.nan)
        out.append(np.nanmedian(subharmonic_ratio(spec, f0)))
    assert out[1] > 5 * out[0]


def test_jitter_increases_with_synthetic_jitter():
    vals = []
    for j in (0.0, 0.005, 0.02):
        sr = 44100
        v = sung_vowel(f0=200, duration=1.0, sr=sr, jitter=j, seed=3)
        hop = sr // 100
        n = len(v.audio) // hop
        pt = perturbation(v.audio, sr, np.full(n, 200.0), hop, 10, n - 10)
        vals.append(pt.jitter_local)
    assert vals[0] < 0.05 < vals[1] < vals[2]

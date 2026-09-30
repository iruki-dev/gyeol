"""Regression tests for the eight known v0.1 defects (see the v2 brief §3).

Each test reproduces the v0.1 failure mode on the ported baseline where that
is still possible, and checks that the v2 code path does not have it.
"""

import numpy as np
import pytest
from scipy import signal

from gyeol.core import FrameGrid, Result, Status
from gyeol.dsp import v01_nuisance
from gyeol.dsp.weak_labels import weak_labels
from gyeol.frontend import detect_clipping, effective_bandwidth, estimate_snr, estimate_t60
from gyeol.pitch.adapters import PyinTracker, SHSTracker, YinTracker
from gyeol.pitch.base import PitchTrack
from gyeol.pitch.consensus import consensus
from gyeol.synth import VOWELS, formant_filter, higher_pole_correction, phrase, pole_set, sung_vowel
from gyeol.verification import degrade

from .helpers import SR, dsp_trackers


def _breathy_phrase():
    p = phrase([(220, 0.8), (262, 0.8)], sr=SR, gap_s=0.4)
    x = degrade.add_noise(p.audio, 60)
    rng = np.random.default_rng(0)
    m = np.zeros(len(x))
    m[int(0.85 * SR) : int(1.15 * SR)] = 1  # breath noise filling the pause
    return x, x + rng.standard_normal(len(x)) * 0.01 * m


def test_defect1_bandwidth_not_fooled_by_breath_consonants_reverb():
    x, xb = _breathy_phrase()
    voiced = np.zeros(1 + len(x) // 441, bool)
    voiced[5:80] = True
    voiced[120:195] = True
    # v0.1 baseline collapses to ~1 kHz
    assert v01_nuisance.effective_bandwidth(xb, SR, voiced, 0.01)[0] < 2000
    for sig in (x, xb, degrade.reverberate(x, degrade.synthetic_rir(SR, 0.5))):
        assert effective_bandwidth(sig, SR).unwrap().bandwidth_hz > 10000
    assert effective_bandwidth(degrade.bandlimit(x, SR, 5000), SR).unwrap().bandwidth_hz < 5500


class _OctaveOff:
    name, asset = "octave-off", None

    def track(self, audio, sr):
        t = YinTracker().track(audio, sr).value
        return Result.success(PitchTrack(self.name, t.times, t.f0_hz * 2, t.voiced_prob))


def test_defect2_single_octave_error_does_not_collapse_confidence():
    v = sung_vowel(f0=220, duration=1.0, sr=SR)
    g = FrameGrid.for_samples(len(v.audio), SR)
    r = consensus(v.audio, SR, g, [PyinTracker(), _OctaveOff(), SHSTracker()]).unwrap()
    assert np.nanmedian(r.f0_hz) == pytest.approx(220, rel=0.01)
    assert np.median(r.f0_conf[r.voiced]) > 0.7
    assert r.octave_repaired[r.voiced].mean() > 0.8


def _h1h2c_weak(f0, vowel, oq=0.6):
    v = sung_vowel(f0=f0, duration=1.0, sr=SR, vowel=vowel, open_quotient=oq, vibrato_rate=5.5, vibrato_extent_cents=30)
    x = degrade.add_noise(np.r_[np.zeros(SR // 4), v.audio, np.zeros(SR // 4)], 70)
    g = FrameGrid.for_samples(len(x), SR)
    p = consensus(x, SR, g, dsp_trackers()).unwrap()
    snr = estimate_snr(x, SR, g, p.voiced).unwrap()
    return weak_labels(x, SR, g, p.f0_hz, snr.frame_snr_db)


def test_defect3_close_vowel_f1_does_not_inflate_h1h2c():
    ref_curve = _h1h2c_weak(120, "a")["h1h2c_db"]
    assert np.isfinite(ref_curve.values).mean() > 0.3
    ref = np.nanmedian(ref_curve.values)
    for vowel in ("i", "u"):
        for f0 in (150, 180, 200, 240):
            h = _h1h2c_weak(f0, vowel)["h1h2c_db"].values
            if np.isfinite(h).sum() >= 5:  # reported values must be sane …
                assert abs(np.nanmedian(h) - ref) < 3.0, (vowel, f0)
            # … otherwise the label is (correctly) missing


def test_defect4_clipping_measured_on_raw_input():
    t = np.arange(SR) / SR
    y = np.clip(3 * (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2), -1, 0.9)  # clipped with a DC offset
    hp = signal.sosfiltfilt(signal.butter(2, 20, "high", fs=SR, output="sos"), y)
    assert v01_nuisance.clipping_fraction(hp) < 1e-3  # v0.1: hidden by the high-pass
    rep = detect_clipping(y)
    assert rep.fraction > 0.2 and rep.n_runs > 100
    assert detect_clipping(0.5 * np.sin(2 * np.pi * 220 * t)).fraction < 1e-3


def test_defect5_t60_reports_unreliable_under_noise():
    p = phrase([(220, 0.8), (262, 0.8)], sr=SR, gap_s=0.4)
    sig = np.concatenate([p.audio, np.zeros(SR // 2), p.audio, np.zeros(SR // 2)])
    clean = estimate_t60(degrade.add_noise(degrade.reverberate(sig, degrade.synthetic_rir(SR, 0.6)), 60), SR)
    # heuristic: known +15–30 % bias on synthetic rooms (documented in estimate_t60)
    assert clean.ok and clean.value == pytest.approx(0.6, rel=0.35)
    noisy = estimate_t60(degrade.add_noise(degrade.reverberate(sig, degrade.synthetic_rir(SR, 0.6)), 20), SR)
    assert noisy.status is Status.UNRELIABLE and noisy.reason and not np.isfinite(noisy.value)


@pytest.mark.parametrize("sub", [0.1, 0.2, 0.3])
def test_defect6_rasp_voice_tracked_at_nominal_pitch(sub):
    v = sung_vowel(f0=220, duration=1.0, sr=SR, subharmonic=sub)
    g = FrameGrid.for_samples(len(v.audio), SR)
    r = consensus(v.audio, SR, g, dsp_trackers()).unwrap()
    f = r.f0_hz[r.voiced]
    assert np.mean(np.abs(1200 * np.log2(f / 220)) < 50) > 0.8
    # roughness is reported, not folded into pitch
    assert np.nanmedian(r.subharmonic_ratio) > 0.08


def test_defect7_formants_do_not_swap_under_noise():
    for f0 in (110, 150, 200):
        for snr in (40, 20, 10):
            for vowel in ("a", "u", "i", "o"):
                v = sung_vowel(f0=f0, duration=1.0, sr=SR, vowel=vowel, vibrato_rate=5.5, vibrato_extent_cents=30)
                x = degrade.add_noise(np.r_[np.zeros(SR // 4), v.audio, np.zeros(SR // 4)], snr)
                g = FrameGrid.for_samples(len(x), SR)
                p = consensus(x, SR, g, dsp_trackers()).unwrap()
                wl = weak_labels(x, SR, g, p.f0_hz, estimate_snr(x, SR, g, p.voiced).unwrap().frame_snr_db)
                for k, name in enumerate(("f1_hz", "f2_hz", "f3_hz")):
                    vals = wl[name].values[np.isfinite(wl[name].values)]
                    if vals.size >= 5:
                        truth = v.truth["formants"][k]
                        assert np.median(np.abs(vals - truth) / truth) < 0.12, (f0, snr, vowel, name)
    # clean /a/ must still be measured
    v = sung_vowel(f0=150, duration=1.0, sr=SR, vowel="a")
    x = degrade.add_noise(np.r_[np.zeros(SR // 4), v.audio, np.zeros(SR // 4)], 60)
    g = FrameGrid.for_samples(len(x), SR)
    p = consensus(x, SR, g, dsp_trackers()).unwrap()
    wl = weak_labels(x, SR, g, p.f0_hz, estimate_snr(x, SR, g, p.voiced).unwrap().frame_snr_db)
    assert all(np.isfinite(wl[n].values).sum() > 20 for n in ("f1_hz", "f2_hz", "f3_hz"))


@pytest.mark.parametrize("sr", [16000, 44100, 48000])
def test_defect8_no_synth_poles_near_nyquist_and_smooth_top_end(sr):
    fs, _ = pole_set(*VOWELS["a"], sr)
    assert max(fs) <= sr / 2 - 1000
    imp = np.zeros(16384)
    imp[0] = 1
    fv, bv = VOWELS["a"]
    h = signal.fftconvolve(formant_filter(imp, sr, *pole_set(fv, bv, sr)), higher_pole_correction(sr, fv, bv))[:16384]
    H = 20 * np.log10(np.abs(np.fft.rfft(h)) + 1e-20)
    f = np.fft.rfftfreq(16384, 1 / sr)
    # the same analog response at every sample rate below 7 kHz …
    ref = {750: 25.6, 2600: 17.7, 5600: -7.0}
    for fr, db in ref.items():
        assert H[np.argmin(np.abs(f - fr))] == pytest.approx(db, abs=1.0)
    # … and no resonance boost near Nyquist
    top = (f > 0.8 * sr / 2)
    assert H[top].max() < H[np.argmin(np.abs(f - 5600))]

import numpy as np
import pytest

from gyeol.core import FrameGrid, LicenseError, Profile, Provenance, Result, Status
from gyeol.io import chirp, integrated_loudness, load_recording, loopback_latency, normalize_loudness, refine_offset, save_audio, tap_along_latency
from gyeol.io.loudness import a_weighting_sos
from gyeol.pitch.adapters import FCPETracker, PyinTracker, RMVPETracker, SHSTracker, SwiftF0Tracker, YinTracker
from gyeol.pitch.base import PitchTrack
from gyeol.pitch.consensus import consensus
from gyeol.synth import phrase, sung_vowel

from .helpers import SR, dsp_trackers


# --- io ----------------------------------------------------------------------

@pytest.mark.parametrize("sr", [44100, 48000])
def test_bs1770_reference_tone(sr):
    t = np.arange(3 * sr) / sr
    assert integrated_loudness(0.1 * np.sin(2 * np.pi * 997 * t), sr).value == pytest.approx(-23.01, abs=0.05)
    y, gain = normalize_loudness(0.1 * np.sin(2 * np.pi * 997 * t), sr, -30.0).unwrap()
    assert integrated_loudness(y, sr).value == pytest.approx(-30.0, abs=0.05)
    assert integrated_loudness(np.zeros(sr // 10), sr).status is Status.FAILED


def test_a_weighting_reference_points():
    from scipy import signal

    _, h = signal.sosfreqz(a_weighting_sos(44100), [100, 1000], fs=44100)
    assert 20 * np.log10(np.abs(h)) == pytest.approx([-19.1, 0.0], abs=0.2)


def test_latency_tools():
    rng = np.random.default_rng(0)
    c = chirp(SR)
    rec = np.r_[np.zeros(int(0.137 * SR)), 0.3 * c, np.zeros(SR)]
    rec += 0.01 * rng.standard_normal(len(rec))
    assert loopback_latency(c, rec, SR).unwrap().latency_s == pytest.approx(0.137, abs=0.001)
    beats = np.arange(0, 20, 0.5)
    taps = beats + 0.21 + 0.01 * rng.standard_normal(len(beats))
    est = tap_along_latency(taps, beats).unwrap()
    assert est.latency_s == pytest.approx(0.21, abs=0.01)
    assert tap_along_latency(taps[:3], beats).status is Status.FAILED
    g = phrase([(220, 0.4), (262, 0.3), (294, 0.5), (330, 0.3)], sr=SR, gap_s=0.1).audio
    u = np.r_[np.zeros(int(0.08 * SR)), phrase([(233, 0.4), (262, 0.3), (294, 0.5), (330, 0.3)], sr=SR, gap_s=0.1,
                                               vibrato_rate=5, vibrato_extent_cents=30).audio]
    assert refine_offset(u, g, SR).unwrap().latency_s == pytest.approx(0.08, abs=0.01)


def test_load_recording_reports_failures(tmp_path):
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"not audio")
    r = load_recording(bad, Provenance.USER, owner_id="u")
    assert r.status is Status.FAILED and "cannot read" in r.reason
    ok = tmp_path / "ok.wav"
    save_audio(ok, np.zeros(1000) + 0.1, SR, metadata={"comment": "test"})
    assert load_recording(ok, Provenance.USER, owner_id="u").ok


# --- pitch -------------------------------------------------------------------

def _run(x, trackers=None):
    g = FrameGrid.for_samples(len(x), SR)
    return g, consensus(x, SR, g, trackers or dsp_trackers()).unwrap()


def _err_cents(v, g, r):
    truth = v.truth["f0_track"][np.minimum((g.times() * SR).astype(int), len(v.audio) - 1)]
    return np.abs(1200 * np.log2(r.f0_hz / truth))


@pytest.mark.parametrize("f0", [70, 110, 220, 440, 1000])
def test_consensus_accuracy(f0):
    v = sung_vowel(f0=f0, duration=1.0, sr=SR, vibrato_rate=5.5, vibrato_extent_cents=50)
    g, r = _run(v.audio)
    assert r.voiced.mean() > 0.75
    assert np.nanmedian(_err_cents(v, g, r)) < 5.0
    assert np.median(r.f0_conf[r.voiced]) > 0.5


class _OctaveOff:
    name, asset = "octave-off", None

    def __init__(self, base):
        self.base = base

    def track(self, audio, sr):
        t = self.base.track(audio, sr).value
        return Result.success(PitchTrack(self.name, t.times, t.f0_hz * 2, t.voiced_prob))


class _Broken:
    name, asset = "broken", None

    def track(self, audio, sr):
        return Result.failure("simulated failure")


def test_consensus_reports_tracker_failures():
    v = sung_vowel(f0=220, duration=0.8, sr=SR)
    g = FrameGrid.for_samples(len(v.audio), SR)
    r = consensus(v.audio, SR, g, [PyinTracker(), _Broken(), SHSTracker()])
    assert r.ok and "broken" in r.value.failed_trackers and r.warnings
    assert consensus(v.audio, SR, g, [_Broken()]).status is Status.FAILED


def test_neural_adapters_are_optional_and_license_gated():
    with pytest.raises(FileNotFoundError, match="gyeol fetch rmvpe"):  # reimplemented in M8; weights are never bundled
        RMVPETracker("weights.pt")
    for cls, pkg in ((SwiftF0Tracker, "swift_f0"), (FCPETracker, "torchfcpe")):
        tr = cls(Profile.COMMERCIAL)
        try:
            __import__(pkg)
        except ImportError:
            assert tr.track(np.zeros(SR), SR).status is Status.UNAVAILABLE
            continue
        v = sung_vowel(f0=220, duration=0.6, sr=SR)
        res = tr.track(v.audio, SR)
        assert res.ok and np.nanmedian(res.value.f0_hz) == pytest.approx(220, rel=0.02)


def test_adapter_license_check_runs_at_construction(monkeypatch):
    import gyeol.pitch.adapters as ad
    from gyeol.core.license import AssetKind, LicensedAsset, LicenseTag

    monkeypatch.setattr(ad, "lookup", lambda n: LicensedAsset(n, AssetKind.CHECKPOINT, LicenseTag.NONCOMMERCIAL, "NC"))
    with pytest.raises(LicenseError):
        ad.SwiftF0Tracker(Profile.COMMERCIAL)


def test_yin_and_pyin_are_distinct_trackers():
    v = sung_vowel(f0=330, duration=0.5, sr=SR)
    assert YinTracker().track(v.audio, SR).ok and PyinTracker().track(v.audio, SR).ok
    assert YinTracker().track(np.zeros(SR), SR).status is Status.FAILED

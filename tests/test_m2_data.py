import json
import warnings

import numpy as np
import pytest

from gyeol.core import FrameGrid, LicenseError, Profile, Purpose, Status
from gyeol.data import (
    AIHubFieldMap,
    PairedLoader,
    inspect_json_keys,
    open_manifest,
    pairs,
    scan_aihub,
    scan_gtsinger,
    scan_own,
    scan_vocalset,
    singer_split,
)
from gyeol.data import augment as A
from gyeol.dsp.base import n_frames, resample
from gyeol.dsp.weak_labels import WeakLabelConfig, robust_formants
from gyeol.frontend import effective_bandwidth, estimate_t60
from gyeol.io import save_audio
from gyeol.pitch.consensus import consensus
from gyeol_service.store import ConsentStore
from gyeol.synth import SynthNote, melody, sung_vowel

from .helpers import SR, dsp_trackers


def _wav(path, f0=220.0, dur=0.3, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(path, sung_vowel(f0=f0, duration=dur, sr=16000, **kw).audio, 16000)


# --- adapters ------------------------------------------------------------------

def test_vocalset_adapter(tmp_path):
    root = tmp_path / "VocalSet"
    _wav(root / "FULL/female1/arpeggios/belt/f1_arpeggios_belt_a.wav")
    _wav(root / "FULL/male3/long_tones/breathy/m3_long_breathy_e.wav")
    _wav(root / "FULL/male3/long_tones/vibrato/m3_long_vibrato_o.wav")
    _wav(root / "FULL/male3/excerpts/spoken/m3_spoken.wav")
    _wav(root / "misc/readme_tone.wav")
    rep = scan_vocalset(root).unwrap()
    items = {i.path: i for i in rep.manifest.items}
    assert len(items) == 3 and len(rep.skipped) == 2
    belt = next(i for i in items.values() if "belt" in i.path)
    assert belt.singer == "female1" and belt.labels["phonation"] == "pressed_belt" and belt.labels["vowel"] == "a"
    assert next(i for i in items.values() if "vibrato" in i.path).labels["vibrato"] is True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(open_manifest(rep.manifest, Profile.COMMERCIAL)) == 3
    assert scan_vocalset(tmp_path / "missing").status is Status.FAILED


def test_aihub_adapter_prints_conditions_and_uses_explicit_map(tmp_path, capsys):
    root = tmp_path / "aihub"
    _wav(root / "원천데이터/s01/song1.wav")
    _wav(root / "원천데이터/s02/song2.wav")
    (root / "라벨링데이터/s01").mkdir(parents=True)
    (root / "라벨링데이터/s01/song1.json").write_text(json.dumps(
        {"meta": {"singer_id": "S01", "gender": "F"}, "tech": {"vibrato": True, "bending": 2}}), encoding="utf-8")
    fm = AIHubFieldMap(singer="meta.singer_id", gender="meta.gender", genre=None, lyrics=None,
                       attributes={"vibrato": "tech.vibrato", "kkeokki": "tech.bending"})
    rep = scan_aihub(root, "aihub_465_multi_singer", fm).unwrap()
    err = capsys.readouterr().err
    assert "never be redistributed" in err and "domestically" in err
    assert len(rep.manifest.items) == 1 and rep.manifest.items[0].labels == {"vibrato": True, "kkeokki": 2}
    assert rep.skipped[0][1].startswith("no label JSON")
    # the default map is a placeholder: files it cannot map are reported, not guessed
    rep2 = scan_aihub(root, "aihub_465_multi_singer").unwrap()
    assert not rep2.manifest.items and "check AIHubFieldMap" in rep2.skipped[0][1]
    assert "meta.singer_id" in inspect_json_keys(root)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        open_manifest(rep.manifest, Profile.COMMERCIAL)  # allowed under conditions


def test_gtsinger_adapter_is_research_only_and_pairs(tmp_path):
    root = tmp_path / "GTSinger/Korean"
    for grp, f in (("Control_Group", 220), ("Breathy_Group", 220)):
        _wav(root / f"KO-Tenor-1/Breathy/song_a/{grp}/0001.wav", f0=f)
    _wav(root / "KO-Tenor-1/Breathy/song_a/Control_Group/0002.wav")  # unpaired
    rep = scan_gtsinger(root).unwrap()
    ps = pairs(rep.manifest)
    assert len(ps) == 1 and ps[0][0].meta["pair_role"] == "off" and ps[0][1].labels["phonation"] == "breathy"
    with pytest.raises(LicenseError):
        open_manifest(rep.manifest, Profile.COMMERCIAL)


def test_own_recordings_require_training_consent(tmp_path):
    cs = ConsentStore(tmp_path / "store")
    cs.grant("u1", {Purpose.ANALYSIS, Purpose.TRAINING})
    cs.grant("u2", {Purpose.ANALYSIS})
    (tmp_path / "recordings.json").write_text(json.dumps([
        {"path": "a.wav", "user_id": "u1", "labels": {"phonation": "breathy"}, "pair_id": "p1", "pair_role": "on"},
        {"path": "b.wav", "user_id": "u2", "labels": {}},
        {"path": "c.wav", "labels": {}},
    ]))
    r = scan_own(tmp_path, cs.allows)
    assert [i.singer for i in r.value.manifest.items] == ["u1"] and len(r.value.skipped) == 2
    assert r.value.manifest.items[0].meta["pair_id"] == "p1"


def test_singer_split_is_disjoint_and_stable(tmp_path):
    from gyeol.data import Manifest, ManifestItem

    items = [ManifestItem(f"{s}_{k}.wav", f"singer{s}") for s in range(40) for k in range(3)]
    m = Manifest("vocalset", "", items)
    a, b = singer_split(m, seed=1), singer_split(m, seed=1)
    assert {k: [i.path for i in v] for k, v in a.items()} == {k: [i.path for i in v] for k, v in b.items()}
    singers = {k: {i.singer for i in v} for k, v in a.items()}
    assert not (singers["train"] & singers["test"]) and not (singers["train"] & singers["calib"])
    assert len(singers["train"]) > len(singers["test"])


# --- augmentation --------------------------------------------------------------

@pytest.fixture(scope="module")
def vowel():
    v = sung_vowel(f0=150, duration=1.2, sr=SR, vowel="a", vibrato_rate=5.5, vibrato_extent_cents=30)
    return np.r_[np.zeros(SR // 5), v.audio, np.zeros(SR // 5)]


def _f0(y):
    g = FrameGrid.for_samples(len(y), SR)
    return np.nanmedian(consensus(y, SR, g, dsp_trackers()).unwrap().f0_hz)


def _formants(y, f0):
    x16 = resample(y, SR, 16000)
    n = n_frames(len(x16), 160)
    F, _ = robust_formants(x16, np.full(n, float(f0)), np.full(n, 60.0), WeakLabelConfig(max_order_spread=1.0))
    return np.nanmedian(F, axis=0)


def test_pitch_shift_keeps_duration_and_formants(vowel):
    for s in (-3, 4):
        y = A.pitch_shift(vowel, SR, s)
        assert len(y) == len(vowel)
        assert _f0(y) == pytest.approx(150 * 2 ** (s / 12), rel=0.01)
    base = _formants(vowel, 150)
    assert np.all(np.abs(_formants(A.pitch_shift(vowel, SR, 4), 150 * 2 ** (4 / 12)) / base - 1) < 0.05)


def test_timbre_shift_moves_formants_keeps_f0(vowel):
    base = _formants(vowel, 150)
    lo, hi = _formants(A.warp_envelope(vowel, SR, 0.85), 150) / base, _formants(A.warp_envelope(vowel, SR, 1.15), 150) / base
    assert np.all((lo > 0.8) & (lo < 0.95)) and np.all((hi > 1.05) & (hi < 1.2))
    assert _f0(A.warp_envelope(vowel, SR, 1.15)) == pytest.approx(150, rel=0.01)


def test_channel_augmentations_have_measurable_effects(vowel):
    rng = np.random.default_rng(0)
    y, lab = A.noise(vowel, SR, rng, snr_range=(10, 10))
    assert lab["snr_db"] == 10
    m = melody([SynthNote(262, 0.6), SynthNote(330, 0.6)], sr=SR)
    sig = np.r_[m.audio, np.zeros(SR // 2), m.audio]
    y, lab = A.reverb(sig, SR, rng, t60_range=(0.8, 0.8), drr_range=(0, 0))
    est = estimate_t60(y, SR)
    assert lab["t60_s"] == 0.8 and est.ok and est.value > 0.5
    y, lab = A.codec(m.audio, SR, np.random.default_rng(2), use_ffmpeg=False)
    if lab["cutoff_hz"] < 12000:
        assert effective_bandwidth(y, SR).unwrap().bandwidth_hz < lab["cutoff_hz"] * 1.2
    from scipy import signal as sps

    # band-limited noise: its correlation survives the slowly drifting delay
    wn = sps.sosfilt(sps.butter(4, [50, 1000], "band", fs=SR, output="sos"), np.random.default_rng(9).standard_normal(SR))

    def local_lag(y, a, b, look=16000):
        c = sps.correlate(wn[a - look : b], y[a:b], "valid", method="fft")  # index k ↔ delay look − k
        return (look - int(np.argmax(c))) / SR

    for seed in (3, 4, 5):
        y, lab = A.bluetooth_jitter(wn, SR, np.random.default_rng(seed), wander_ms=(0.0, 0.0), loss_rate=(0.0, 0.0))
        assert local_lag(y, 20000, 24000) == pytest.approx(lab["bt_latency_s"], abs=2 / SR)
        y, lab = A.bluetooth_jitter(wn, SR, np.random.default_rng(seed), wander_ms=(20.0, 20.0), loss_rate=(0.0, 0.0))
        lags = [local_lag(y, a, a + 2000) for a in range(18000, 42000, 6000)]
        assert all(abs(v - lab["bt_latency_s"]) <= 0.021 for v in lags)
    y, lab = A.bluetooth_jitter(wn, SR, np.random.default_rng(3), loss_rate=(0.05, 0.05))
    assert (np.abs(y) == 0).mean() > 0.02
    for fn in (A.device_eq, A.compression, A.agc, A.separation_artifacts, A.timbre_shift):
        y, lab = fn(m.audio, SR, np.random.default_rng(4))
        assert y.shape == m.audio.shape and np.all(np.isfinite(y)) and lab


def test_pipeline_is_deterministic_and_labels_classes(vowel):
    y1, l1 = A.AugmentationPipeline(seed=7)(vowel, SR)
    y2, l2 = A.AugmentationPipeline(seed=7)(vowel, SR)
    assert np.array_equal(y1, y2) and l1 == l2
    assert np.max(np.abs(y1)) <= 0.99 + 1e-9
    cls = A.env_classes(l1)
    assert set(cls) == {"noise", "room", "eq", "codec", "compression", "separation"}
    assert A.env_classes({"applied": []}) == dict.fromkeys(cls, 0)


# --- paired loader ----------------------------------------------------------------

def test_paired_loader_aligns_on_to_off(tmp_path):
    root = tmp_path / "GTSinger/Korean"
    base = [SynthNote(262, 0.5, consonant="s"), SynthNote(330, 0.5, vowel="i", consonant="t"), SynthNote(294, 0.6, vowel="o")]
    off = melody(base, sr=SR)
    on_notes = [SynthNote(n.f0, n.dur, vowel=n.vowel, consonant=n.consonant, onset_shift_s=0.15) for n in base]
    on = melody(on_notes, sr=SR, aspiration=0.3, seed=2)  # breathier and 150 ms later
    for grp, m in (("Control_Group", off), ("Breathy_Group", on)):
        p = root / f"KO-Alto-1/Breathy/song_b/{grp}/0001.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        save_audio(p, m.audio, SR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = open_manifest(scan_gtsinger(root).unwrap().manifest, Profile.RESEARCH)
    from gyeol.attributes.extract import analyze

    loader = PairedLoader(ds, analyzer=lambda rec: analyze(rec, trackers=dsp_trackers()))
    assert len(loader) == 1
    ex = loader.load(0).unwrap()
    g = ex.on.grid
    # the "on" onset of note 2 (at 0.2 + 0.6 + 0.15 s) maps to the "off" onset (0.2 + 0.6 s)
    u = g.frame_of(0.2 + 0.6 + 0.15 + 0.02)
    assert ex.warp.tau[u] * g.hop_seconds == pytest.approx(0.2 + 0.6 + 0.02, abs=0.04)
    assert ex.labels_on["phonation"] == "breathy"

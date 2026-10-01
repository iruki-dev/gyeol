"""Revision D — follow-up decisions: target separation once with a content-hash cache and bleed-gated takes (D1),
checksum-verified separator weights (D2, loader side), exact-f0 pitch ground truth and evaluation on human-annotated
sets (D3)."""

import json
from pathlib import Path

import numpy as np
import pytest

from gyeol import api
from gyeol.frontend.separation import (
    CallableSeparator,
    SeparationCache,
    SeparationQueue,
    content_key,
    make_separator,
    separate_target,
)
from gyeol.synth import accompaniment

from .helpers import SR, dsp_trackers, make_melody


def _rms(x):
    return float(np.sqrt(np.mean(x**2)))


@pytest.fixture(scope="module")
def song():
    voc = make_melody(seed=3, vib=(25,) * 6).audio
    back = accompaniment(len(voc) / SR + 0.1, sr=SR, seed=7)[: len(voc)]
    back = np.pad(back, (0, len(voc) - len(back)))
    back = back / _rms(back) * _rms(voc) * 0.7
    return voc, back, voc + back


class _Counting:
    """An oracle 'heavy' separator that returns the true vocal and counts its calls."""

    def __init__(self, vocal):
        self.vocal, self.calls, self.name = vocal, 0, "oracle"

    def separate(self, audio, sr):
        self.calls += 1
        return CallableSeparator(lambda x, s: self.vocal[: len(x)], name="oracle").separate(audio, sr)


# ================================================================ D1 target once, cached by content


def test_content_key_and_cache_round_trip(song, tmp_path):
    voc, back, mix = song
    k = content_key(mix, SR, "bs_roformer")
    assert k == content_key(mix.copy(), SR, "bs_roformer") and k != content_key(mix, SR, "other") and k != content_key(mix * 0.5, SR, "bs_roformer")
    sep = _Counting(voc)
    cache = SeparationCache(tmp_path)
    first = separate_target(mix, SR, cache=cache, separator=sep, separator_id="oracle").unwrap()
    again = separate_target(mix, SR, cache=cache, separator=sep, separator_id="oracle").unwrap()
    assert sep.calls == 1 and first.meta["cache"] == "miss" and again.meta["cache"] == "hit"
    assert np.allclose(again.vocal, voc, atol=1e-6) and np.allclose(again.accompaniment, back, atol=1e-5)
    assert list(tmp_path.rglob("*.npz")) and not list(tmp_path.rglob("*.tmp"))


def test_background_queue_separates_once(song, tmp_path):
    voc, _, mix = song
    sep = _Counting(voc)
    q = SeparationQueue(SeparationCache(tmp_path))
    f1 = q.submit(mix, SR, separator=sep, separator_id="oracle")
    f2 = q.submit(mix, SR, separator=sep, separator_id="oracle")
    assert f1 is f2 and f1.result(timeout=60).ok and sep.calls == 1
    q.shutdown()
    fut = api.separate_target(mix, SR, cache_dir=tmp_path, separator=sep, separator_id="oracle", background=True)
    assert fut.result(timeout=60).value.meta["cache"] == "hit" and sep.calls == 1


def test_target_analysis_uses_the_cached_vocal(song, tmp_path):
    voc, _, mix = song
    ts = api.separate_target(mix, SR, cache_dir=tmp_path, separator=_Counting(voc), separator_id="oracle").unwrap()
    t = api.analyze(mix, SR, separated=ts, dsp_only=True).unwrap()
    q = t.quality["separation"]
    assert q["applied"] and "(cached)" in q["separator"]
    clean = api.analyze(voc, SR, dsp_only=True).unwrap()
    both = np.isfinite(t.curves["f0_cents"].values) & np.isfinite(clean.curves["f0_cents"].values)
    assert np.median(np.abs(t.curves["f0_cents"].values[both] - clean.curves["f0_cents"].values[both])) < 5


def test_headphone_takes_are_separated_only_when_bleed_is_detected(song, tmp_path):
    voc, back, mix = song
    ts = api.separate_target(mix, SR, cache_dir=tmp_path, separator=_Counting(voc), separator_id="oracle").unwrap()
    take = make_melody(seed=4, vib=(25,) * 6, detune=(0, 0, 30, 0, 0, 0)).audio
    clean = api.analyze(take, SR, target=ts, dsp_only=True).unwrap()
    q = clean.quality["separation"]
    assert not q["applied"] and q["bleed"]["detected"] is False and q["reason"] == "no accompaniment bleed detected"
    lag = 300
    bled = take + np.r_[np.zeros(lag), back[:-lag]] / _rms(back) * _rms(take) * 10 ** (-15 / 20)
    r = api.analyze(bled, SR, target=ts, dsp_only=True).unwrap()
    q = r.quality["separation"]
    assert q["applied"] and q["separator"] == "backing-cancel" and q["bleed"]["detected"] is True
    assert q["bleed"]["lag_s"] == pytest.approx(lag / SR, abs=2e-3)


def test_bleed_detection_survives_a_repeating_accompaniment():
    """A circular-shift null cancels leakage of looped music; the phase-randomised null does not."""
    from gyeol.frontend.quality import detect_bleed

    rng = np.random.default_rng(0)
    bar = accompaniment(2.0, sr=SR, seed=1)
    loop = np.tile(bar, 4)
    voice = make_melody(seed=5, detune=(7, -5, 9, -8, 4, 6)).audio
    n = min(len(loop), len(voice))
    loop, voice = loop[:n], voice[:n]
    mic = voice + loop / _rms(loop) * _rms(voice) * 10 ** (-18 / 20) + rng.standard_normal(n) * 1e-4
    b = detect_bleed(mic, loop, SR).unwrap()
    assert b.bleed_db > -24 and b.lag_prominence > 16
    quiet = detect_bleed(voice + rng.standard_normal(n) * 1e-4, loop, SR).unwrap()
    assert quiet.lag_prominence < 16


def test_separator_choice():
    with pytest.raises(ValueError, match="unknown separator"):
        make_separator("magic")
    assert make_separator("backing").status.name == "UNAVAILABLE"
    assert make_separator("roformer_light").status.name == "UNAVAILABLE"
    assert make_separator("backing", accompaniment=np.zeros(10), accompaniment_sr=SR).ok
    obj = _Counting(np.zeros(10))
    assert make_separator(obj).value is obj


# ================================================================ D2 checksum verified on every load


def test_roformer_weights_are_checksum_verified(tmp_path, monkeypatch):
    import torch

    from gyeol.frontend import roformer
    from gyeol.frontend.roformer import BSRoFormer, RoFormerSeparator, file_sha256, tiny_config

    path = tmp_path / "w.ckpt"
    torch.save(BSRoFormer(**tiny_config()).state_dict(), path)
    digest = file_sha256(path)
    monkeypatch.setattr(roformer, "pinned_sha256", lambda name: digest)
    sep = RoFormerSeparator.from_checkpoint(path, tiny_config())
    assert sep.weights_sha256 == digest
    monkeypatch.setattr(roformer, "pinned_sha256", lambda name: "0" * 64)
    with pytest.raises(ValueError, match="checksum mismatch"):
        RoFormerSeparator.from_checkpoint(path, tiny_config())


# ================================================================ D3 exact-f0 ground truth


@pytest.fixture(scope="module")
def resynth_cache(tmp_path_factory):
    from gyeol.data.prepare import PrepareConfig, prepare, synthetic_manifest

    root = tmp_path_factory.mktemp("resynth")
    m = synthetic_manifest(root / "src", n_singers=2, seconds=0.8, sr=16000)
    rep = prepare([m], root / "cache", config=PrepareConfig(sr=16000, hop=128, separation="off", resynthesize=("hnm",)))
    assert rep.failed == 0 and rep.done == 36
    return root / "cache"


def test_resynthesised_copies_carry_exact_f0(resynth_cache):
    from gyeol.data.prepare import read_index
    from gyeol.eval.pitch_eval import score_pitch

    rows = read_index(resynth_cache)
    copies = [r for r in rows if r["meta"].get("f0_truth") == "exact"]
    assert len(copies) == 18 and all(r["id"].endswith("~hnm") and r["meta"]["source_id"] for r in copies)
    tr = dsp_trackers()[0]
    for r in copies[:3]:
        with np.load(resynth_cache / "items" / f"{r['id']}.npz") as z:
            x, f0, hop, sr = z["audio"].astype(float), z["f0_exact"], int(z["hop"]), int(z["sr"])
        assert (f0 > 0).mean() > 0.4
        est = tr.track(x, sr).unwrap()
        s = score_pitch(np.arange(len(f0)) * hop / sr, f0, est.times, est.f0_hz)
        assert s.rpa > 0.9  # the copy really sounds at its stated f0


def test_resynthesis_skip_list_is_a_configurable_default(tmp_path):
    from gyeol.data.manifest import Manifest, ManifestItem
    from gyeol.data.prepare import PrepareConfig, prepare
    from gyeol.io import save_audio

    save_audio(tmp_path / "a.wav", make_melody(seed=1).audio, SR)
    m = Manifest("own_recordings", str(tmp_path), [ManifestItem("a.wav", "user-1", {})])
    rep = prepare([m], tmp_path / "c", config=PrepareConfig(sr=16000, hop=128, separation="off", resynthesize=("hnm",)))
    assert rep.done == 1 and rep.failed == 1 and "resynth_skip_datasets" in rep.failures[0]["reason"]
    # the application decides: an empty skip list resynthesises its own recordings too
    rep = prepare([m], tmp_path / "d", config=PrepareConfig(sr=16000, hop=128, separation="off", resynthesize=("hnm",),
                                                            resynth_skip_datasets=()))
    assert rep.done == 2 and rep.failed == 0


def test_pitch_targets_weight_exact_over_consensus(resynth_cache):
    from gyeol.data.prepare import read_index
    from gyeol.train.config import config_from_dict
    from gyeol.train.stream import CropSpec, StreamingDataset
    from gyeol.train.tasks import DataInfo, PitchTask

    rows = read_index(resynth_cache)
    ds = StreamingDataset(resynth_cache, rows, len(rows), CropSpec(20, 30), shuffle=False)
    b = ds.batch(0)
    cfg = config_from_dict({"task": "pitch", "model": {"size": "tiny", "weak_weight": 0.3, "weak_min_confidence": 0.8}})
    t = PitchTask(cfg, DataInfo(16000, 128, 5, {}, [], {}, (), []), "cpu")
    t.build()
    _, _, w, cents, exact = t._targets(b)
    is_copy = np.array([i.endswith("~hnm") for i in b["ids"]])
    w, exact = w.numpy(), exact.numpy()
    assert exact[is_copy].any() and not exact[~is_copy].any()
    assert np.all(np.isclose(w[is_copy], 0.0) | np.isclose(w[is_copy], 1.0)) and np.isclose(w[is_copy], 1.0).mean() > 0.9  # every frame
    assert np.all(np.isclose(w[~is_copy], 0.0) | np.isclose(w[~is_copy], 0.3))  # consensus: weak, only on confident frames
    assert np.isclose(w[~is_copy], 0.3).any() and np.isclose(w[~is_copy], 0.0).any()
    assert ((cents.numpy()[is_copy] == 0) & (w[is_copy] == 1)).any()  # exact unvoiced frames are trained as such


def _fake_vocadito(root: Path, n=2):
    from gyeol.io import save_audio

    (root / "Audio").mkdir(parents=True)
    (root / "Annotations" / "F0").mkdir(parents=True)
    for i in (1, 10)[:n]:
        m = make_melody(seed=i, dur=0.4)
        save_audio(root / "Audio" / f"vocadito_{i}.wav", m.audio, SR)
        f0 = m.truth["f0_track"]
        t = np.arange(0, len(f0) / SR, 0.0058)
        hz = np.interp(t, np.arange(len(f0)) / SR, f0)
        (root / "Annotations" / "F0" / f"vocadito_{i}_f0.csv").write_text("\n".join(f"{a:.4f},{b:.2f}" for a, b in zip(t, hz)))


def test_annotated_sets_are_scored(tmp_path):
    from gyeol.data.pitch_sets import read_reference_f0, scan_mir1k, scan_vocadito
    from gyeol.eval.pitch_eval import evaluate_pitch

    _fake_vocadito(tmp_path / "voc")
    man = scan_vocadito(tmp_path / "voc").unwrap()
    pairs = {i.path: i.meta["f0"] for i in man.items}
    assert pairs["Audio/vocadito_1.wav"].endswith("vocadito_1_f0.csv") and pairs["Audio/vocadito_10.wav"].endswith("vocadito_10_f0.csv")
    rep = evaluate_pitch(man, dsp_trackers()[0]).unwrap().summary()
    assert rep["n_items"] == 2 and rep["rpa"] > 0.85 and rep["voicing_recall"] > 0.85
    # MIR-1K: stereo (accompaniment left, voice right), semitones per 20 ms
    import soundfile as sf

    (tmp_path / "mir" / "Wavfile").mkdir(parents=True)
    (tmp_path / "mir" / "PitchLabel").mkdir(parents=True)
    from gyeol.dsp.base import resample

    m = make_melody(seed=2, dur=0.4)
    x16 = resample(m.audio, SR, 16000)
    sf.write(tmp_path / "mir" / "Wavfile" / "abjones_1_01.wav", np.stack([np.random.default_rng(0).standard_normal(len(x16)) * 0.01, x16], 1), 16000)
    f0 = m.truth["f0_track"]
    t = 0.02 * (np.arange(int(len(x16) / 16000 / 0.02)) + 1)
    hz = np.interp(t, np.arange(len(f0)) / SR, f0)
    semis = np.where(hz > 0, 69 + 12 * np.log2(np.maximum(hz, 1) / 440), 0)
    (tmp_path / "mir" / "PitchLabel" / "abjones_1_01.pv").write_text("\n".join(f"{s:.4f}" for s in semis))
    mm = scan_mir1k(tmp_path / "mir").unwrap()
    assert mm.items[0].singer == "abjones" and mm.items[0].meta["channel"] == 1
    rt, rhz = read_reference_f0(mm.root, mm.items[0])
    assert rt[0] == pytest.approx(0.02) and np.allclose(rhz[hz > 0], hz[hz > 0], rtol=1e-3)
    assert evaluate_pitch(mm, dsp_trackers()[0]).unwrap().summary()["rpa"] > 0.8


def test_eval_pitch_cli(tmp_path, capsys):
    from gyeol.cli import main
    from gyeol.data.pitch_sets import scan_vocadito

    _fake_vocadito(tmp_path / "voc", n=1)
    scan_vocadito(tmp_path / "voc").unwrap().write(tmp_path / "voc.json")
    assert main(["eval", "pitch", str(tmp_path / "voc.json")]) == 0
    out = capsys.readouterr().out
    assert "vocadito / pyin" in out and "RPA" in out


def test_pitch_run_reports_annotated_evaluation(resynth_cache, tmp_path):
    from gyeol.data.pitch_sets import scan_vocadito
    from gyeol.train.config import config_from_dict
    from gyeol.train.runner import train

    _fake_vocadito(tmp_path / "voc", n=1)
    scan_vocadito(tmp_path / "voc").unwrap().write(tmp_path / "voc.json")
    cfg = config_from_dict({"task": "pitch", "device": "cpu", "threads": 1,
                            "data": {"cache": str(resynth_cache), "split": {"train": 0.5, "val": 0.5, "test": 0.0},
                                     "crop_frames": [16, 24], "batch_size": 2, "prepare": {"sr": 16000, "hop": 128, "separation": "off"}},
                            "run": {"out": str(tmp_path / "run"), "max_steps": 2, "val_every": 2, "patience": 0},
                            "model": {"size": "tiny", "eval_sets": [str(tmp_path / "voc.json")]}})
    r = train(cfg, log=lambda s: None)
    ev = r.report["finalize"]["annotated_eval"][str(tmp_path / "voc.json")]
    assert ev["dataset"] == "vocadito" and ev["n_items"] == 1 and "rpa" in ev
    assert json.loads((tmp_path / "run" / "pitch_eval.json").read_text())
    assert any(k.startswith("rpa_") for k in r.position.history[-1])

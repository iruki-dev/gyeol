"""M8: RMVPE reimplementation, regression heads, robustness grid, expert benchmarks and coach
agreement, listening calibration, model cards, ONNX export and latency profiles."""

import json
import os

import numpy as np
import pytest
import torch

from gyeol.core import FrameGrid, LicenseError, Profile, Provenance, Status

# ================================================================ RMVPE


TINY = dict(n_blocks=1, inter_layers=1, en_out=4)


def test_rmvpe_architecture_matches_the_reference_layout():
    from gyeol.pitch.rmvpe import N_BINS, RMVPE

    full = RMVPE()
    keys = set(full.model.state_dict())
    for k in ("unet.encoder.bn.weight", "unet.encoder.layers.0.conv.0.conv.0.weight", "unet.encoder.layers.0.conv.0.shortcut.weight",
              "unet.intermediate.layers.3.conv.3.conv.4.running_var", "unet.decoder.layers.4.conv1.0.weight", "cnn.weight",
              "fc.0.gru.weight_ih_l0_reverse", "fc.1.weight"):
        assert k in keys, k
    assert full.model.fc[1].out_features == N_BINS and full.model.fc[0].gru.input_size == 3 * 128
    tiny = RMVPE(**TINY).eval()
    with torch.no_grad():
        for n in (16000, 16000 * 2 + 77):
            assert tiny(torch.randn(1, n)).shape == (1, n // 160 + 1, N_BINS)


def test_rmvpe_decoding_and_weight_loading(tmp_path):
    from gyeol.pitch.rmvpe import CENTS0, RMVPE, RMVPETracker, cents_to_hz, salience_to_cents

    sal = np.zeros((3, 360))
    sal[0, 100] = sal[0, 101] = 1.0  # halfway between two bins
    sal[1, 200] = 0.02  # below the 0.03 threshold → unvoiced
    c = salience_to_cents(sal)
    assert c[0] == pytest.approx(CENTS0 + 20 * 100.5) and c[1] == 0 and cents_to_hz(c)[1] == 0
    assert cents_to_hz(np.array([1200.0 * np.log2(220 / 10)]))[0] == pytest.approx(220.0)
    m = RMVPE(**TINY)
    torch.save(m.model.state_dict(), tmp_path / "ok.pt")
    RMVPE(**TINY).load_reference_weights(tmp_path / "ok.pt")  # strict round trip
    sd = m.model.state_dict()
    sd.pop("fc.1.bias")
    sd["extra.weight"] = torch.zeros(1)
    torch.save(sd, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="fc.1.bias"):
        RMVPE(**TINY).load_reference_weights(tmp_path / "bad.pt")
    with pytest.raises(FileNotFoundError, match="gyeol fetch rmvpe"):
        RMVPETracker(tmp_path / "missing.pt")
    with pytest.raises(ValueError):
        RMVPETracker()


class _OracleRMVPE(torch.nn.Module):
    """Stands in for trained weights: puts a salience bump at the true f0 of a known sine."""

    def __init__(self, f0: float):
        super().__init__()
        self.f0 = f0

    def forward(self, audio):
        from gyeol.pitch.rmvpe import CENTS0

        T = audio.shape[-1] // 160 + 1
        b = (1200 * np.log2(self.f0 / 10) - CENTS0) / 20
        bins = torch.arange(360, dtype=torch.float32)
        return torch.exp(-0.5 * ((bins - b) / 1.2) ** 2).expand(1, T, 360).clone()


def test_rmvpe_tracker_timing_and_chunking():
    from gyeol.pitch.consensus import consensus
    from gyeol.pitch.rmvpe import RMVPETracker

    sr = 44100
    x = 0.3 * np.sin(2 * np.pi * 247.0 * np.arange(int(2.5 * sr)) / sr)
    whole = RMVPETracker(model=_OracleRMVPE(247.0)).track(x, sr).unwrap()
    chunked = RMVPETracker(model=_OracleRMVPE(247.0), max_chunk_s=0.7).track(x, sr).unwrap()
    assert len(whole.f0_hz) == len(chunked.f0_hz) == int(2.5 * 16000) // 160 + 1
    assert np.allclose(whole.times[1] - whole.times[0], 0.01)
    assert np.nanmedian(np.abs(1200 * np.log2(whole.f0_hz / 247.0))) < 2
    assert np.allclose(whole.f0_hz, chunked.f0_hz, equal_nan=True)
    assert RMVPETracker(model=_OracleRMVPE(247.0)).track(np.zeros(10), sr).status is Status.FAILED
    # it plugs into the consensus like any other tracker
    from gyeol.pitch.adapters import PyinTracker, SHSTracker

    g = FrameGrid.for_samples(len(x), sr)
    r = consensus(x, sr, g, [RMVPETracker(model=_OracleRMVPE(247.0)), PyinTracker(), SHSTracker()])
    assert r.ok and "rmvpe" in [t.name for t in r.value.tracks]


@pytest.mark.skipif(not os.environ.get("GYEOL_RMVPE_WEIGHTS"), reason="set GYEOL_RMVPE_WEIGHTS to fetched rmvpe.pt to run")
def test_rmvpe_with_reference_weights():  # pragma: no cover - needs user-fetched weights
    from gyeol.pitch.rmvpe import RMVPETracker
    from gyeol.synth import sung_vowel

    v = sung_vowel(f0=220, duration=1.5, sr=44100)
    tr = RMVPETracker(os.environ["GYEOL_RMVPE_WEIGHTS"]).track(v.audio, 44100).unwrap()
    f = tr.f0_hz[np.isfinite(tr.f0_hz)]
    assert np.median(np.abs(1200 * np.log2(f / 220))) < 10


# ================================================================ regression heads


def test_regression_heads_are_calibrated_on_held_out_singers():
    from gyeol.attributes.heads import TaskSpec
    from gyeol.train.heads import FrameExample, HeadTrainConfig, train_heads

    rng = np.random.default_rng(0)
    D = 6
    w = rng.standard_normal(D)

    def ex(s):
        X = rng.standard_normal((80, D))
        y = X @ w + rng.standard_normal(80) * (0.2 + 0.8 * (X[:, 0] > 0))
        return FrameExample(X, {"level": y[:, None]}, str(s))

    tr, ca, te = [ex(s) for s in range(12)], [ex(s) for s in range(12, 16)], [ex(s) for s in range(16, 20)]
    h = train_heads(tr, ca, {"level": TaskSpec(1, "regression", ("level",), "dB")}, HeadTrainConfig(epochs=40, hidden=32))
    assert "level" in h.variance_scale and "level" in h.sigma_ref
    cov, hi, lo = [], [], []
    for e in te:
        p = h.predict(e.features)[0]["level"]
        mu, sig = p[:, 0], p[:, 1]
        cov.append(np.abs(e.targets["level"][:, 0] - mu) <= 1.96 * sig)
        hi.append(sig[e.features[:, 0] > 0].mean())
        lo.append(sig[e.features[:, 0] <= 0].mean())
    assert 0.9 < np.concatenate(cov).mean() < 0.995  # ≈ 95 % after held-out variance scaling
    assert np.mean(hi) > 1.3 * np.mean(lo)  # heteroscedastic: larger σ where the data are noisier
    c = h.curves(te[0].features, FrameGrid(44100, 512, 80))["level"]
    assert c.values.shape == (80,) and c.unit == "dB" and np.all((c.confidence >= 0) & (c.confidence <= 1))
    with pytest.raises(ValueError):
        TaskSpec(1, "ordinal")


# ================================================================ robustness grid


def test_bluetooth_jitter_does_not_shift_pitch():
    """Regression (found by the M8 grid): a smoothly varying delay resampled the audio and shifted
    pitch by ~65 cents; re-sync jumps + ppm drift must leave pitch alone."""
    from gyeol.data.augment import bluetooth_jitter
    from gyeol.pitch.adapters import YinTracker

    sr = 44100
    x = 0.3 * np.sin(2 * np.pi * 220.0 * np.arange(6 * sr) / sr)
    for seed in range(3):
        y, lab = bluetooth_jitter(x, sr, np.random.default_rng(seed), wander_ms=(40.0, 40.0), loss_rate=(0.0, 0.0))
        f = YinTracker().track(y, sr).unwrap().f0_hz
        assert abs(np.nanmedian(1200 * np.log2(f[np.isfinite(f)] / 220.0))) < 2 and abs(lab["bt_drift_ppm"]) <= 50


def test_robustness_grid_reports_icc_mdc_and_item_presence():
    from gyeol.attributes.extract import analyze
    from gyeol.eval import GridItem, robustness_grid, run_robustness

    from .helpers import dsp_trackers, make_melody, pad_to, rep_of

    names = [c.key for c in robustness_grid("full", codecs=False)]
    assert {"clean", "snr_db=0", "t60_s=1.2", "separation=2", "bluetooth=2"} <= set(names)
    conds = [c for c in robustness_grid("short", codecs=False) if c.key in ("clean", "snr_db=30", "separation=0", "bluetooth=0")]
    tgt = make_melody(dur=0.4, gap=0.15)
    items = []
    for i, (d, sh) in enumerate([(-40, -0.06), (-15, 0.0), (10, 0.05), (35, 0.09)]):
        u = make_melody(detune=(0, d, 0, 0, 0, 0), shifts=(0, 0, sh, 0, 0, 0), dur=0.4, gap=0.15, seed=i + 1)
        t, uu = pad_to(tgt.audio, u.audio)
        items.append(GridItem(f"p{i}", uu, 44100, rep_of(t, Provenance.REFERENCE).unwrap(), t))
    rep = run_robustness(items, conds, analyzer=lambda r: analyze(r, trackers=dsp_trackers()), n_boot=50)
    assert not rep.failures
    ino = rep.items["pitch/intonation_offset/1"]
    assert ino.icc21 > 0.9 and ino.n_singers == 4 and ino.mdc95 < 20
    ons = rep.items["rhythm/onset_timing/2"]
    assert ons.icc21 > 0.8
    assert rep.presence["pitch/intonation_offset/1"]["clean"] == 1.0
    assert ino.conditions_used[0] == "clean"
    assert set(rep.deviation["pitch/intonation_offset/1"]) == {"snr_db=30", "separation=0", "bluetooth=0"}
    assert any("intonation_offset/1" in line for line in rep.summary())
    with pytest.raises(ValueError, match="clean"):
        run_robustness(items[:1], conds[1:])


# ================================================================ expert benchmarks and agreement


def _bench_index(tmp_path):
    clips = [
        {"id": "c1", "audio_path": "c1.wav", "reference_path": "ref1.wav",
         "segments": [{"start": 1.0, "end": 2.0, "issue": "Pitch", "annotator": "a"},
                      {"start": 3.0, "end": 3.5, "issue": "Rhythm", "annotator": "b"}]},
        {"id": "c2", "audio_path": "c2.wav", "segments": [{"start": 0.5, "end": 1.5, "issue": "Breath"}]},
        {"id": "c3", "audio_path": "c3.wav", "segments": [{"start": 0.0, "end": 1.0, "issue": "Diction"}]},
        {"audio_path": "broken.wav"},
    ]
    p = tmp_path / "index.json"
    p.write_text(json.dumps({"clips": clips}), encoding="utf-8")
    return p


def test_expert_benchmark_is_license_gated_and_scored(tmp_path):
    from gyeol.eval import Prediction, load_expert_benchmark, score_benchmark

    p = _bench_index(tmp_path)
    with pytest.raises(LicenseError):
        load_expert_benchmark(p, Profile.COMMERCIAL)  # research / non-commercial sources
    rep = load_expert_benchmark(p, Profile.RESEARCH).unwrap()
    assert [c.clip_id for c in rep.clips] == ["c1", "c2", "c3"] and "3" in rep.skipped
    assert rep.clips[0].labels == {"pitch", "rhythm"} and rep.clips[0].reference is not None and rep.clips[1].reference is None
    preds = {"c1": [Prediction("pitch", "intonation_offset", 1.2, 1.6, 0.9), Prediction("dynamics", "loudness", 5.0, 6.0, 0.8)],
             "c2": [Prediction("phonation", "breathiness", 0.6, 1.0, 0.7)]}
    s = score_benchmark(preds, rep.clips, k=1)
    assert s.n_clips == 2 and s.topk_hit_rate == 1.0 and s.skipped == {"c3": s.skipped["c3"]}
    assert s.segment_recall == {"pitch": 1.0, "rhythm": 0.0, "breath": 1.0}
    assert s.segment_precision == pytest.approx(2 / 3)
    assert 0 <= s.prior_hit_rate <= 1


def test_agreement_statistics():
    from gyeol.eval import coach_agreement, cohen_kappa, fleiss_kappa

    wiki = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
                     [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])
    assert fleiss_kappa(wiki) == pytest.approx(0.210, abs=0.001)  # Fleiss (1971) worked example
    assert cohen_kappa(np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0])) == 1.0
    assert cohen_kappa(np.array([1, 0, 1, 0]), np.array([0, 1, 0, 1])) == -1.0
    with pytest.raises(ValueError):
        fleiss_kappa(np.array([[1, 0], [2, 0]]))
    coaches = {f"clip{i}": {r: ({"pitch"} if i % 2 else {"rhythm"}) for r in ("r1", "r2", "r3")} for i in range(8)}
    gy = {f"clip{i}": ({"pitch"} if i % 2 else {"rhythm"}) for i in range(8)}
    ag = coach_agreement(coaches, gy)
    assert ag.n_raters == 3 and ag.n_clips == 8
    assert ag.fleiss_per_label["pitch"] == pytest.approx(1.0) and ag.gyeol_vs_consensus["pitch"] == pytest.approx(1.0)


# ================================================================ listening calibration and the perceptual floor


def test_audibility_calibration_and_floor(tmp_path):
    from gyeol.coach import AttributeThreshold, ThresholdSet
    from gyeol.core import ExplanationItem, Span
    from gyeol.eval import calibrate_audibility

    rng = np.random.default_rng(0)
    s = rng.uniform(0, 0.3, 4000)
    p = 0.5 + 0.5 / (1 + np.exp(-(s - 0.1) / 0.02))  # 2AFC: 75 % detection at 0.1
    cal = calibrate_audibility(s, rng.random(4000) < p, listeners=rng.integers(0, 12, 4000))
    assert cal.floor == pytest.approx(0.1, abs=0.02) and cal.n_listeners == 12 and np.all(np.diff(cal.detection) >= 0)
    assert calibrate_audibility(s, rng.random(4000) < 0.6).floor is None  # never reaches 75 %
    ts = ThresholdSet({"intonation_offset": AttributeThreshold.flat("intonation_offset", 5.0, 0.3)}, {"data": "t"}, cal.floor)
    it = lambda aud: ExplanationItem("pitch", "intonation_offset", [Span(0, 5)], -30.0, "cents", 0.9, audibility=aud,  # noqa: E731
                                     detail={"target_note": 1})
    assert ts.passes(it(0.02)) == (False, "below_audibility")
    assert ts.passes(it(0.5)) == (True, "ok") and ts.passes(it(None)) == (True, "ok")
    back = ThresholdSet.from_json(ts.to_json(tmp_path / "t.json"))
    assert back.audibility_floor == pytest.approx(cal.floor)


# ================================================================ model cards


def test_model_card_from_checkpoint(tmp_path):
    from gyeol.attributes.heads import FrameHeads, default_tasks
    from gyeol.eval import REQUIRED_OUT_OF_SCOPE, EvalTable, ModelCard, card_from_checkpoint
    from gyeol.train.checkpoint import save_checkpoint

    heads = FrameHeads(10, default_tasks(), hidden=16)
    save_checkpoint(tmp_path / "h.pt", heads.state_dict(), name="heads-demo", sources=["gyeol_synthetic", "vocalset"],
                    config={"hidden": 16}, profile=Profile.COMMERCIAL)
    ev = [EvalTable("probe accuracy", {"register": 0.91, "ece": 0.03}, synthetic=True, data="synthetic melodies")]
    lim = ["All evaluation so far is on synthetic data.", "License tags were not re-verified upstream by this code."]
    card = card_from_checkpoint(tmp_path / "h.pt", component="attribute heads", architecture="temporal-conv trunk + linear heads",
                                intended_use=["Phonation posteriors for coaching explanations."], evaluation=ev, limitations=lim,
                                profile=Profile.COMMERCIAL)
    assert card.validate() == []
    assert card.license == "commercial_ok" and {s["name"] for s in card.sources} == {"gyeol_synthetic", "vocalset"}
    assert card.parameters == sum(p.numel() for p in heads.parameters())
    md = card.to_markdown()
    assert "| vocalset | dataset | `commercial_ok` | CC-BY-4.0 |" in md and "(synthetic data)" in md and "target-singer" in md
    assert json.loads(card.to_json(tmp_path / "card.json").read_text())["name"] == "heads-demo"
    # a card that hides problems is flagged
    bad = ModelCard("x", "c", "a", 1, "noncommercial", "commercial", "h",
                    [{"name": "gtsinger", "kind": "dataset", "tag": "noncommercial", "license": "NC", "verified": False}], [],
                    ["use"], list(REQUIRED_OUT_OF_SCOPE[:1]), ev, [])
    problems = " | ".join(bad.validate())
    assert "non-commercial sources" in problems and "missing out-of-scope" in problems and "synthetic" in problems
    assert "Card validation problems" in bad.to_markdown()


# ================================================================ ONNX export and latency


def test_onnx_export_matches_pytorch_with_dynamic_shapes(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from gyeol.attributes.heads import FrameHeads, TaskSpec, default_tasks
    from gyeol.decoder import AutoencoderConfig, GyeolAutoencoder
    from gyeol.export import export_acoustic, export_heads, export_rmvpe, export_vocoder
    from gyeol.pitch.rmvpe import E2E

    torch.manual_seed(0)
    tasks = default_tasks() | {"level": TaskSpec(1, "regression")}
    ae = GyeolAutoencoder(AutoencoderConfig.tiny())
    results = [export_heads(FrameHeads(20, tasks, hidden=32), 20, tmp_path, provenance={"license": "commercial_ok"}),
               export_rmvpe(E2E(**TINY), tmp_path), export_acoustic(ae, tmp_path), export_vocoder(ae, tmp_path)]
    for r in results:
        assert r.passed, (r.name, r.max_abs_diff, r.max_abs_diff_other_length)
        meta = json.loads(r.path.with_suffix(".json").read_text())
        assert meta["verification"]["passed"] and "provenance" in meta
    assert json.loads((tmp_path / "attribute_heads.json").read_text())["provenance"]["license"] == "commercial_ok"


def test_latency_profiles_and_cli(tmp_path, capsys):
    from gyeol.cli import main
    from gyeol.export import pipeline_profile, profile

    from .helpers import dsp_trackers, make_melody

    prof = profile("sum", lambda d: (lambda x=np.ones(int(d * 16000)): x.sum()), durations_s=(0.5, 1.0), n_runs=3)
    assert [r.audio_seconds for r in prof.rows] == [0.5, 1.0] and all(r.rtf >= 0 for r in prof.rows)
    assert "median ms" in prof.table() and "cpus" in prof.environment
    m = make_melody(dur=0.3, gap=0.1)
    stages = pipeline_profile(m.audio, 44100, trackers=dsp_trackers(), n_runs=1)
    assert {"pitch", "harmonic_noise", "content", "events", "analyze_total"} <= set(stages)
    assert stages["pitch"] <= stages["analyze_total"] and stages["audio_seconds"] > 2
    assert main(["profile", "--dsp-only", "--seconds", "3", "--runs", "1"]) == 0
    assert "RTF" in capsys.readouterr().out

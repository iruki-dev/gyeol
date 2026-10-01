"""Revision A — analysis path: separation by default (A1), confidence-gated octave decisions (A2),
premise checks before judgements (A3), Korean syllabification (A4), whole-contour comparison (A5)
and the real-recording evaluation set (A6, exercised on synthetic audio)."""

import json

import numpy as np
import pytest
import torch

from gyeol.core import Recording
from gyeol.explain import explain
from gyeol.synth import accompaniment

from .helpers import SR, dsp_trackers, make_melody, rep_of


@pytest.fixture(scope="module")
def target_rep():
    return rep_of(make_melody(seed=1).audio).unwrap()


def _with_backing(voc, level_db, seed=7):
    back = accompaniment(len(voc) / SR + 0.1, sr=SR, seed=seed)[: len(voc)]
    back = np.pad(back, (0, len(voc) - len(back)))
    rms = lambda x: np.sqrt(np.mean(x**2)) + 1e-12  # noqa: E731
    return voc + back / rms(back) * rms(voc) * 10 ** (level_db / 20), back


# ================================================================ A4 syllables


def test_syllabify_splits_hangul_regardless_of_spaces_and_punctuation():
    from gyeol.context.korean import syllabify

    sy = syllabify("사랑해, 요!  그대여")
    assert [s.text for s in sy] == list("사랑해요그대여")
    assert [s.word_initial for s in sy] == [True, False, False, True, True, False, False]
    # no spaces at all and decomposed (NFD) input give the same split
    import unicodedata

    assert [s.text for s in syllabify(unicodedata.normalize("NFD", "사랑해요그대여"))] == list("사랑해요그대여")


def test_assign_syllables_handles_melisma_and_extra_syllables():
    from gyeol.context import assign_syllables

    notes = [(0, 10), (10, 20), (20, 60)]
    assert [t for *_, t in assign_syllables(notes, "사랑해")] == ["사", "랑", "해"]
    mel = assign_syllables(notes + [(60, 70)], "사랑해")  # more notes than syllables: the last syllable holds
    assert [t for *_, t in mel] == ["사", "랑", "해", "해"]
    extra = assign_syllables(notes, "사랑해요")  # more syllables than notes: the longest note takes two
    assert [t for *_, t in extra] == ["사", "랑", "해요"]


def test_explain_maps_lyrics_onto_notes(target_rep):
    u = rep_of(make_melody(detune=(0, 0, 40, 0, 0, 0), seed=2).audio).unwrap()
    e = explain([u], target_rep, lyrics="사랑 해요, 그대!").unwrap()
    it = next(i for i in e.items if i.attribute == "intonation_offset" and i.detail["target_note"] == 2)
    assert it.spans[0].syllables == ("해",)


# ================================================================ A1 separation


def test_accompaniment_estimate_tells_voice_from_mixture():
    from gyeol.frontend.separation import estimate_accompaniment

    voc = make_melody(seed=3).audio
    assert estimate_accompaniment(voc, SR).unwrap().may_contain is False
    vib = make_melody(vib=(60,) * 6, seed=4).audio
    assert estimate_accompaniment(vib, SR).unwrap().may_contain is False
    mix, _ = _with_backing(voc, -6)
    a = estimate_accompaniment(mix, SR).unwrap()
    assert a.may_contain is True and a.reasons
    short = estimate_accompaniment(voc[: SR // 10], SR)
    assert not short.usable  # too short to tell: an explicit status, not a guess


def test_unseparated_penalty_is_graded_by_residual_level():
    from gyeol.frontend.separation import SeparationPolicy, unseparated_factor

    pol = SeparationPolicy()
    assert unseparated_factor(-10.0, pol) == pytest.approx(pol.unseparated_factor)
    assert unseparated_factor(-40.0, pol) == pytest.approx(1.0)
    assert pol.unseparated_factor < unseparated_factor(-22.0, pol) < 1.0
    assert unseparated_factor(None, pol) == pol.unseparated_factor


def test_analyze_separation_modes():
    voc = make_melody(seed=5).audio
    mix, back = _with_backing(voc, -6)
    rec = Recording(mix, SR)
    off = rep_of(mix, separation="off").unwrap()
    assert off.quality["separation"]["applied"] is False and off.meta["analysis_signal"] == "input"
    # a known backing track → cancelled first, quality measured against it
    from gyeol.attributes.extract import analyze

    auto = analyze(rec, trackers=dsp_trackers(), backing=back).unwrap()
    sq = auto.quality["separation"]
    assert sq["applied"] and sq["separator"] == "backing-cancel" and sq["residual_reference"] == "backing"
    assert "separation" in auto.meta["timings_s"]
    # an oracle separator recovers the clean pitch
    from gyeol.frontend.separation import CallableSeparator

    oracle = CallableSeparator(lambda x, sr: voc[: len(x)], name="oracle")
    sep = analyze(rec, trackers=dsp_trackers(), separator=oracle).unwrap()
    clean = rep_of(voc).unwrap()
    both = np.isfinite(sep.curves["f0_cents"].values) & np.isfinite(clean.curves["f0_cents"].values)
    assert np.median(np.abs(sep.curves["f0_cents"].values[both] - clean.curves["f0_cents"].values[both])) < 5


def test_suspected_accompaniment_without_separator_lowers_confidence(monkeypatch):
    import gyeol.attributes.extract as ex
    from gyeol.core.status import Result

    monkeypatch.setattr(ex, "default_separator", lambda *a, **k: Result.unavailable("no weights in this test"))
    voc = make_melody(seed=6).audio
    mix, _ = _with_backing(voc, -6)
    r = rep_of(mix).unwrap()
    assert "accompaniment_unseparated" in r.quality["flags"]
    assert r.quality["separation"]["unseparated_factor"] < 1.0
    plain = rep_of(mix, separation="off").unwrap()
    v = np.isfinite(r.curves["f0_cents"].values) & np.isfinite(plain.curves["f0_cents"].values)
    assert np.mean(r.curves["f0_cents"].confidence[v]) < np.mean(plain.curves["f0_cents"].confidence[v])
    # "always" refuses to analyse without a separator
    always = rep_of(mix, separation="always")
    assert not always.ok and "separation required" in always.reason
    with pytest.raises(ValueError):
        rep_of(mix, separation="sometimes")


# ---------------------------------------------------------------- BS-RoFormer


def test_roformer_shapes_and_strict_loading(tmp_path):
    from gyeol.frontend.roformer import BSRoFormer, tiny_config

    m = BSRoFormer(**tiny_config()).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 4000)).shape == (2, 1, 4000)
        assert m(torch.randn(1, 1, 3001)).shape == (1, 1, 3001)
    torch.save({"state_dict": {"model." + k: v for k, v in m.state_dict().items()}}, tmp_path / "wrapped.ckpt")
    BSRoFormer(**tiny_config()).load_reference_weights(tmp_path / "wrapped.ckpt")
    bad = dict(m.state_dict())
    bad.pop(next(iter(bad)))
    torch.save(bad, tmp_path / "bad.ckpt")
    with pytest.raises(ValueError, match="does not match"):
        BSRoFormer(**tiny_config()).load_reference_weights(tmp_path / "bad.ckpt")


def test_roformer_forward_backward_on_cpu():
    from gyeol.frontend.roformer import BSRoFormer, tiny_config

    m = BSRoFormer(**tiny_config())
    m.checkpointing = True
    m.train()
    y = m(torch.randn(1, 4000))
    y.pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)


class _Identity(torch.nn.Module):
    audio_channels = 1

    def forward(self, x):
        return x


def test_roformer_separator_chunking_reconstructs_and_gates(tmp_path, monkeypatch):
    from gyeol.frontend.roformer import RoFormerSeparator

    x = np.random.default_rng(0).standard_normal(SR * 3) * 0.1
    sep = RoFormerSeparator(_Identity(), sr=SR, chunk=SR // 2)
    y = sep.separate(x, SR).unwrap()
    assert y.shape == x.shape and np.max(np.abs(y - x)) < 1e-5  # cross-faded chunks add back to the input
    assert not sep.separate(x[:100], SR).ok
    with pytest.raises(FileNotFoundError, match="gyeol fetch bs_roformer_viperx_ep317"):
        RoFormerSeparator.from_checkpoint(tmp_path / "missing.ckpt")
    import gyeol.cli as cli

    monkeypatch.setattr(cli, "CACHE", tmp_path)
    assert RoFormerSeparator.from_cache().status.name == "UNAVAILABLE"  # never downloads


def test_roformer_asset_is_listed_with_url_and_checksum(capsys):
    from gyeol.cli import main
    from gyeol.core import asset

    a = asset("bs_roformer_viperx_ep317")
    assert a.url and a.url.endswith(".ckpt") and len(a.sha256) == 64
    assert any("provenance" in n.lower() for n in a.notes)
    assert main(["licenses"]) == 0 and "bs_roformer_viperx_ep317" in capsys.readouterr().out


def test_roformer_matches_reference_implementation():
    """Numerical check against the MIT reference package, when it is installed."""
    import sys
    import types

    sys.modules.setdefault("librosa", types.ModuleType("librosa"))
    ref = pytest.importorskip("bs_roformer")
    from gyeol.frontend.roformer import BSRoFormer, tiny_config

    cfg = tiny_config()
    theirs = ref.BSRoFormer(**cfg).eval()
    ours = BSRoFormer(**cfg).eval()
    ours.load_state_dict(theirs.state_dict())
    x = torch.randn(1, 4096)
    with torch.no_grad():
        a, b = ours(x)[..., :4000], theirs(x)
    n = min(a.shape[-1], b.shape[-1])
    assert torch.allclose(a.reshape(-1)[:n], b.reshape(-1)[:n], atol=1e-4)


# ================================================================ A2 octave decisions


def test_octave_relation_is_decided_only_with_confidence(target_rep):
    from dataclasses import replace

    from gyeol.explain import ExplainConfig

    low = rep_of(make_melody(transpose=-1200, detune=(0, 0, 30, 0, 0, 0), seed=8).audio).unwrap()
    e = explain([low], target_rep).unwrap()
    assert e.premises["octave_relation"].holds and e.transposition_cents == -1200
    assert e.comparison_mode["pitch"] == "absolute"
    strict = ExplainConfig()
    strict = replace(strict, premises=replace(strict.premises, octave_min_confidence=1.01))
    w = explain([low], target_rep, strict).unwrap()
    p = w.premises["octave_relation"]
    assert not p.holds and "confidence" in p.reason
    assert w.transposition_cents is None and w.comparison_mode["pitch"] == "octave_invariant"
    # the octave-invariant comparison still finds the detuned note
    it = next(i for i in w.items if i.attribute == "intonation_offset" and i.detail["target_note"] == 2)
    assert it.magnitude == pytest.approx(30, abs=10)
    from gyeol.explain.render_text import explanation_notes

    assert any("옥타브" in n for n in explanation_notes(w))


def test_fold_octave_is_circular():
    from gyeol.explain.explain import _fold_octave

    d = np.array([1190.0, -1195.0, 5.0, 1210.0, np.nan])
    f = _fold_octave(d, np.isfinite(d))
    assert np.allclose(f[:4], [-10, 5, 5, 10], atol=1e-6) and np.isnan(f[4])


# ================================================================ A3 premises


def test_trimmed_clip_uses_content_alignment_and_withholds_tempo(target_rep):
    shifts = (0, 0, 0, 0.08, 0, 0)
    m = make_melody(shifts=shifts, seed=9)
    start = int(0.9 * SR)
    u = rep_of(m.audio[start:]).unwrap()
    e = explain([u], target_rep).unwrap()
    clock = e.premises["shared_clock"]
    assert not clock.holds and clock.reason
    assert e.comparison_mode["timing"] == "content_aligned"
    assert not any(i.attribute == "tempo" for i in e.items)
    assert any(w.attribute == "tempo" and w.premise == "shared_clock" for w in e.withheld)
    onsets = {i.detail["target_note"]: i for i in e.items if i.attribute == "onset_timing"}
    assert 0 not in onsets and 1 not in onsets  # notes the clip does not cover get no verdict
    assert onsets[3].detail["reference"] == "previous note" and onsets[3].magnitude == pytest.approx(80, abs=25)
    assert e.warp[0] > 0.5 * u.grid.rate  # the take starts inside the target


def test_shared_clock_take_keeps_absolute_timing(target_rep):
    u = rep_of(make_melody(shifts=(0, 0, 0.07, 0, 0, 0), seed=10).audio).unwrap()
    e = explain([u], target_rep).unwrap()
    assert e.premises["shared_clock"].holds and e.comparison_mode["timing"] == "shared_clock"
    it = next(i for i in e.items if i.attribute == "onset_timing" and i.detail["target_note"] == 2)
    assert it.detail["reference"] == "shared clock" and it.magnitude == pytest.approx(70, abs=20)


def test_apply_premises_withholds_dependent_items():
    from gyeol.core.containers import ExplanationItem, Premise, Span
    from gyeol.explain.premises import LEVEL_CHAIN, NOISE_FLOOR, apply_premises

    items = {("dynamics", "loudness", 0): ExplanationItem("dynamics", "loudness", [Span(0, 5)], 3.0, "dB", 0.9, detail={"target_note": 0}),
             ("phonation", "breathiness", 0): ExplanationItem("phonation", "breathiness", [Span(0, 5)], 2.0, "dB", 0.9, detail={"target_note": 0}),
             ("pitch", "intonation_offset", 0): ExplanationItem("pitch", "intonation_offset", [Span(0, 5)], 20.0, "cents", 0.9, detail={"target_note": 0})}
    prem = {LEVEL_CHAIN: Premise(LEVEL_CHAIN, "s", False, "user:clipping", {}), NOISE_FLOOR: Premise(NOISE_FLOOR, "s", True, "", {})}
    kept, withheld = apply_premises(items, prem)
    assert set(k[1] for k in kept) == {"breathiness", "intonation_offset"}
    assert [(w.attribute, w.premise, w.reason) for w in withheld] == [("loudness", LEVEL_CHAIN, "user:clipping")]


def test_explanation_records_premises_and_modes(target_rep):
    u = rep_of(make_melody(seed=11).audio).unwrap()
    e = explain([u], target_rep).unwrap()
    for name in ("shared_clock", "octave_relation", "level_chain", "noise_floor", "interval_set"):
        assert name in e.premises and e.premises[name].statement
    assert set(e.comparison_mode) == {"timing", "pitch"}


# ================================================================ A5 whole contour


def test_contour_spans_label_events_and_skip_restated_offsets(target_rep):
    u = rep_of(make_melody(scoop=(0, 0, 0, 0, 200, 0), detune=(0, 45, 0, 0, 0, 0), seed=12).audio).unwrap()
    e = explain([u], target_rep).unwrap()
    contour = [i for i in e.items if i.attribute in ("contour_deviation", "transition_deviation")]
    assert contour, "the scoop should leave a contour span"
    sc = [i for i in contour if "scoop" in i.detail["events"]]
    assert sc and sc[0].detail["event"] == "scoop" and sc[0].magnitude < 0
    assert sc[0].delta is not None and np.isfinite(sc[0].delta).sum() >= 3
    # the +45-cent note is an intonation item, not restated as a contour span
    assert any(i.attribute == "intonation_offset" and i.detail["target_note"] == 1 for i in e.items)
    assert not any(i.attribute == "contour_deviation" and i.detail["target_note"] == 1 for i in e.items)
    from gyeol.explain.render_text import item_text

    assert "스쿱" in item_text(sc[0])


def test_contour_items_find_transition_spans():
    from gyeol.core import FrameGrid
    from gyeol.core.containers import AttributeCurve, AttributeCurves, Representation
    from gyeol.explain.contour import ContourConfig, contour_items

    g = FrameGrid(SR, 441, 100)
    f0 = np.full(100, 0.0)

    def rep(v):
        curves = AttributeCurves(g, {"f0_cents": AttributeCurve("f0_cents", v.copy(), np.ones(100), g, "cents")})
        return Representation(g, curves, "r")

    diff = np.zeros(100)
    diff[48:58] = 80.0  # across the boundary between note 0 (0..50) and note 1 (50..100)
    diff[20:30] = -60.0  # inside note 0
    out = contour_items(rep(f0), rep(f0), np.arange(100.0), diff, np.ones(100), np.ones(100), np.ones(100),
                        [(0, 50), (50, 100)], [(0, 50, "사"), (50, 100, "랑")], ContourConfig())
    kinds = {(k[1], it.detail["location"]) for k, it in out.items()}
    assert ("transition_deviation", "transition") in kinds and ("contour_deviation", "note") in kinds
    tr = next(it for it in out.values() if it.attribute == "transition_deviation")
    assert tr.spans[0].syllables == ("사", "랑") and tr.magnitude == pytest.approx(80)


def test_contour_items_have_demo_edits(target_rep):
    from gyeol.demo.edits import edit_for_item

    u = rep_of(make_melody(scoop=(0, 0, 0, 0, 200, 0), seed=12).audio).unwrap()
    e = explain([u], target_rep).unwrap()
    it = next(i for i in e.items if i.attribute in ("contour_deviation", "transition_deviation"))
    ed = edit_for_item(it, u, target_rep, e).unwrap()
    sp = it.spans[0]
    assert np.abs(ed.f0_cents[sp.start : sp.end]).max() > 20 and np.all(ed.f0_cents[: max(0, sp.start - 3)] == 0)


# ================================================================ A6 real-recording set


def test_realset_manifest_validation(tmp_path):
    from gyeol.eval.realset import load_realset

    assert not load_realset(tmp_path).ok  # no manifest
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"id": "a", "audio": "missing.wav", "role": "user", "condition": "clean"}) + "\n")
    r = load_realset(tmp_path)
    assert not r.ok and "missing.wav" in r.reason
    (tmp_path / "x.wav").write_bytes(b"")
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"id": "a", "audio": "x.wav", "role": "user", "condition": "studio",
                                                         "singer": "s", "target": "t"}) + "\n")
    assert "condition" in load_realset(tmp_path).reason


@pytest.fixture(scope="module")
def synthetic_realset(tmp_path_factory):
    from gyeol.eval.realset_synth import make_synthetic_realset

    return make_synthetic_realset(tmp_path_factory.mktemp("realset"), n_singers=3, conditions=("clean", "mixture_karaoke", "trimmed"))


def test_realset_split_is_singer_disjoint(synthetic_realset):
    from gyeol.eval.realset import load_realset, split_realset

    rs = load_realset(synthetic_realset).unwrap()
    sides = split_realset(rs, 0.34, seed=0)
    by_singer = {}
    for u in rs.users:
        by_singer.setdefault(u.singer, set()).add(sides[u.id])
    assert all(len(v) == 1 for v in by_singer.values()) and {"tuning", "held_out"} <= set(sides.values())
    assert split_realset(rs, 0.34, seed=0) == sides and (synthetic_realset / "split.json").exists()


def test_eval_realset_cli_on_a_folder(synthetic_realset, capsys):
    from gyeol.cli import main

    out = synthetic_realset / "report.json"
    assert main(["eval", "realset", str(synthetic_realset), "--dsp-only", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "octave_errors" in text and "mixture_karaoke" in text
    rep = json.loads(out.read_text())
    conds = rep["conditions"]
    assert set(conds) == {"clean", "mixture_karaoke", "trimmed"}
    for c in conds.values():
        assert c["octave_errors"] == 0 and c["false_rhythm_verdicts"] == 0
        assert c["withheld_rate"] is not None and c["rpa"] is not None
    assert conds["clean"]["rpa"] > 0.9 and conds["mixture_karaoke"]["separated"] == 3
    assert conds["trimmed"]["premise_failures"].get("shared_clock") == 3


def test_tracker_breakdown(synthetic_realset):
    from gyeol.eval.realset import load_realset, tracker_breakdown

    bd = tracker_breakdown(load_realset(synthetic_realset).unwrap(), dsp_trackers())
    assert set(bd) == {"clean", "mixture_karaoke", "trimmed"}
    assert bd["clean"]["pyin"]["rpa"] > 0.9 and bd["clean"]["pyin"]["n"] == 3
    assert bd["mixture_karaoke"]["pyin"]["rpa"] < bd["clean"]["pyin"]["rpa"]


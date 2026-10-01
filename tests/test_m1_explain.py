import numpy as np
import pytest

from gyeol.align import estimate_warp, onset_deviations
from gyeol.core import Consistency, Status
from gyeol.explain import ExplainConfig, explain
from gyeol.explain.render_text import item_text, load_strings, particles

from .helpers import SR, make_melody, pad_to, rep_of


@pytest.fixture(scope="module")
def target():
    m = make_melody(vib=(0, 0, 60, 0, 60, 0), scoop=(0, 120, 0, 0, 0, 0), fall=(0, 0, 0, 0, 0, 150), dur=0.9)
    return m


def _pair(target_m, user_m):
    t, u = pad_to(target_m.audio, user_m.audio)
    return rep_of(t).unwrap(), rep_of(u).unwrap()


def _items(exp):
    return {(i.category, i.attribute, i.detail.get("target_note", -1)): i for i in exp.items}


def test_warp_recovers_onsets_with_octave_and_ornaments():
    shifts = (0, 0.06, -0.05, 0.1, 0.0, -0.08)
    tgt = make_melody()
    usr = make_melody(shifts=shifts, vib=(60,) * 6, scoop=(100,) * 6, transpose=-1200, seed=5)
    T, U = _pair(tgt, usr)
    w = estimate_warp(U.curves["content"].values, T.curves["content"].values, U.grid).unwrap()
    onsets = [T.grid.frame_of(a) for a, _, _ in tgt.truth["notes"]]
    devs = [d.deviation_s for d in onset_deviations(w, onsets, U.grid)]
    assert devs == pytest.approx(list(shifts), abs=0.015)
    assert np.all(np.diff(w.tau) > 0)


def test_explain_recovers_known_knobs(target):
    usr = make_melody(detune=(0, -40, 0, 30, 0, 0), shifts=(0, 0.06, 0, -0.05, 0, 0), vib=(0, 0, 0, 0, 60, 0),
                      fall=(0, 0, 0, 0, 0, 150), dur=0.9, transpose=-1200, seed=3)
    T, U = _pair(target, usr)
    exp = explain([U], T).unwrap()
    assert exp.transposition_cents == -1200
    it = _items(exp)
    assert it[("pitch", "intonation_offset", 1)].magnitude == pytest.approx(-40, abs=8)
    assert it[("pitch", "intonation_offset", 3)].magnitude == pytest.approx(30, abs=8)
    assert abs(it[("pitch", "intonation_offset", 0)].magnitude) < 5
    assert it[("rhythm", "onset_timing", 1)].magnitude == pytest.approx(60, abs=15)
    assert it[("rhythm", "onset_timing", 3)].magnitude == pytest.approx(-50, abs=15)
    assert it[("ornament", "scoop", 1)].detail["status"] == "missing"
    assert it[("ornament", "vibrato_extent", 2)].magnitude == pytest.approx(-60, rel=0.3)  # vibrato missing
    assert abs(it[("ornament", "vibrato_extent", 4)].magnitude) < 15  # both have it
    assert it[("ornament", "fall", 5)].detail["status"] == "different" and abs(it[("ornament", "fall", 5)].magnitude) < 40
    assert exp.n_takes == 1 and all(i.consistency is Consistency.UNDETERMINED for i in exp.items)
    assert all(i.audibility is None for i in exp.items)  # filled only by explain.audibility.score_audibility (M5)


def test_take_consistency(target):
    takes = []
    for seed, (d, s) in enumerate([(-45, 0.0), (-40, 0.09)]):
        takes.append(make_melody(detune=(0, 0, d, 0, 0, 0), shifts=(0, 0, 0, s, 0, 0), vib=(0, 0, 60, 0, 60, 0),
                                 scoop=(0, 120, 0, 0, 0, 0), fall=(0, 0, 0, 0, 0, 150), dur=0.9, seed=seed + 1))
    arrays = pad_to(target.audio, *[t.audio for t in takes])
    T = rep_of(arrays[0]).unwrap()
    U = [rep_of(a).unwrap() for a in arrays[1:]]
    it = _items(explain(U, T).unwrap())
    flat = it[("pitch", "intonation_offset", 2)]
    assert flat.consistency is Consistency.STYLE_OR_HABIT and flat.magnitude == pytest.approx(-40, abs=8)
    late = it[("rhythm", "onset_timing", 3)]
    assert late.consistency is Consistency.ERROR and late.magnitude == pytest.approx(90, abs=15)


def test_unreliable_regions_become_cannot_judge(target):
    usr = make_melody(vib=(0, 0, 60, 0, 60, 0), scoop=(0, 120, 0, 0, 0, 0), fall=(0, 0, 0, 0, 0, 150), dur=0.9, seed=4)
    x = usr.audio.copy()
    a, b = int(2.4 * SR), int(3.2 * SR)  # replace the 3rd–4th notes by loud noise
    x[a:b] = np.random.default_rng(1).standard_normal(b - a) * 0.2
    T, U = _pair(target, type(usr)(x, SR, usr.truth))
    exp = explain([U], T).unwrap()
    h = exp.grid.hop_seconds
    covered = sum(min(s.end * h, 3.2) - max(s.start * h, 2.4) for s in exp.cannot_judge if s.end * h > 2.4 and s.start * h < 3.2)
    assert covered > 0.4
    notes_in_noise = {i.detail.get("target_note") for i in exp.items if i.category == "pitch" and i.spans and 2.5 < i.spans[0].start * h < 3.1}
    assert not notes_in_noise


def test_semitone_transposition_is_opt_in(target):
    usr = make_melody(vib=(0, 0, 60, 0, 60, 0), scoop=(0, 120, 0, 0, 0, 0), fall=(0, 0, 0, 0, 0, 150), dur=0.9, transpose=200, seed=6)
    T, U = _pair(target, usr)
    octave_only = _items(explain([U], T).unwrap())
    assert octave_only[("pitch", "global_offset", -1)].magnitude == pytest.approx(200, abs=10)
    semis = explain([U], T, ExplainConfig(allow_transposition=True)).unwrap()
    assert semis.transposition_cents == 200
    assert abs(_items(semis)[("pitch", "global_offset", -1)].magnitude) < 10


def test_explain_failure_modes(target):
    T = rep_of(target.audio).unwrap()
    assert explain([], T).status is Status.FAILED
    # a hand-trimmed 1 s clip does not share the target's clock: explained on content alone, timing withheld (revision A3)
    short = rep_of(make_melody().audio[: SR]).unwrap()
    r = explain([short], T)
    assert r.ok and r.value.comparison_mode["timing"] == "content_aligned"
    assert not r.value.premises["shared_clock"].holds and "band" in r.value.premises["shared_clock"].reason
    assert not any(i.attribute == "tempo" for i in r.value.items)


def test_korean_rendering():
    s = load_strings("ko")
    assert {"pitch", "rhythm", "ornament", "phonation", "diction"} <= set(s["category"])
    for attr in ("intonation_offset", "global_offset", "interval_compression", "onset_timing", "tempo", "vibrato_extent",
                 "vibrato_rate", "scoop", "fall", "kkeokki", "glide"):
        assert attr in s["attribute"]
    assert particles("'랑'")["i_ga"] == "이" and particles("'요'")["i_ga"] == "가" and particles("'달'")["euro"] == "로"
    from gyeol.core import ExplanationItem, Span

    it = ExplanationItem("pitch", "intonation_offset", [Span(0, 1, ("해",))], -40.0, "cents", 0.9, detail={"target_note": 2})
    assert item_text(it) == "'해'가 목표보다 40센트 낮아요"


def test_demo_runs_end_to_end(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("coach_demo_v2", Path(__file__).parents[1] / "examples" / "coach_demo_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--synthetic", "--out", str(tmp_path), "--audibility", "--render-demo"]) == 0
    out = capsys.readouterr().out
    assert "40센트 낮아요" in out and "스쿱" in out and "늦게 들어갔어요" in out
    # M5: audibility per item and a stepwise demo in the user's voice (plain WAVs + demo.json)
    import json

    assert "들리는 차이" in out
    demos = sorted(tmp_path.glob("demo_*.wav"))
    assert len(demos) >= 2
    meta = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))
    assert meta["schema"] == "gyeol.demo" and meta["version"] == 2 and "ai_generated" not in meta

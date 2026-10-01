"""M5 explanation items: phonation, dynamics, diction, remainder, Korean text."""

import numpy as np
import pytest

from gyeol.core import AttributeCurve, ExplanationItem, Span
from gyeol.explain import explain
from gyeol.explain.render_text import explanation_notes, item_text, load_strings

from .helpers import SR, make_melody, pad_to, rep_of


def _items(exp):
    return {(i.category, i.attribute, i.detail.get("target_note", -1)): i for i in exp.items}


@pytest.fixture(scope="module")
def base():
    tgt = make_melody()
    return tgt, rep_of(tgt.audio).unwrap()


def _pair(tgt_audio, usr_audio):
    t, u = pad_to(tgt_audio, usr_audio)
    return rep_of(t).unwrap(), rep_of(u).unwrap()


def test_breathiness_and_loudness_items(base):
    tgt, _ = base
    usr = make_melody(seed=2, aspiration=0.4)
    x = usr.audio.copy()
    s, e = usr.truth["notes"][3][:2]
    x[int(s * SR) : int(e * SR)] *= 10 ** (6 / 20)  # note 3 six dB louder
    T, U = _pair(tgt.audio, x)
    it = _items(explain([U], T).unwrap())
    breath = [it[("phonation", "breathiness", k)].magnitude for k in range(6) if ("phonation", "breathiness", k) in it]
    assert len(breath) >= 4 and np.median(breath) > 3  # consistently breathier
    loud = it[("dynamics", "loudness", 3)]
    others = [it[("dynamics", "loudness", k)].magnitude for k in (0, 1, 2, 4, 5) if ("dynamics", "loudness", k) in it]
    # loudness_rel is re the median voiced level (which the louder note shifts), so compare with the other notes
    assert len(others) >= 4 and loud.magnitude - np.median(others) == pytest.approx(6, abs=1)
    rng = it[("dynamics", "dynamic_range", -1)]
    assert rng.unit == "dB" and {"user_range_db", "target_range_db"} <= set(rng.detail)


def _posterior(rep, name, labels, rows):
    T = rep.grid.n_frames
    v = np.full((T, len(labels)), np.nan)
    conf = np.zeros(T)
    for (s, e), probs in rows:
        v[s:e] = probs
        conf[s:e] = 0.9
    rep.curves.add(AttributeCurve(name, v, conf, rep.grid, "probability", labels=labels))


def test_register_quality_and_diction_items_from_learned_curves(base):
    tgt, _ = base
    T, U = _pair(tgt.audio, make_melody(seed=4).audio)
    notes = T.meta["notes"]
    chest, falsetto = [0.8, 0.15, 0.05], [0.05, 0.15, 0.8]
    _posterior(T, "register", ("chest", "mixed", "falsetto"), [(n, chest) for n in notes])
    _posterior(U, "register", ("chest", "mixed", "falsetto"), [(n, falsetto if k == 2 else chest) for k, n in enumerate(U.meta["notes"])])
    q = ("breathy", "pressed_belt", "pharyngeal_twang", "fry", "rough")
    _posterior(T, "phonation", q, [(n, [0.1] * 5) for n in notes])
    _posterior(U, "phonation", q, [(n, [0.7, 0.1, 0.1, 0.1, 0.1]) for n in U.meta["notes"]])
    lar = ("lenis", "aspirated", "fortis")
    _posterior(T, "laryngeal", lar, [((n[0], n[0] + 6), [0.1, 0.8, 0.1]) for n in notes])
    _posterior(U, "laryngeal", lar, [((n[0], n[0] + 6), [0.1, 0.4, 0.5]) for n in U.meta["notes"]])
    it = _items(explain([U], T).unwrap())
    reg = it[("phonation", "register", 2)]
    assert reg.detail["status"] == "different" and reg.detail["target"] == "chest" and reg.detail["user"] == "falsetto"
    assert reg.detail["tentative"] and reg.magnitude < -0.5
    assert it[("phonation", "register", 1)].detail["status"] == "same"
    assert it[("phonation", "quality_breathy", 3)].magnitude == pytest.approx(0.6, abs=0.05)
    asp = it[("diction", "laryngeal_aspirated", -1)]
    assert asp.magnitude == pytest.approx(-0.4, abs=0.1) and asp.detail["phrase_level"]
    assert asp.detail["reference"] == "target singer's realisation"
    assert all(i.detail.get("target_note", -1) == -1 for i in it.values() if i.category == "diction")  # phrase level only
    txt = item_text(reg)
    assert "가성" in txt and "흉성" in txt and "수 있어요" in txt  # tentative wording
    assert "숨 섞인 소리" in item_text(it[("phonation", "quality_breathy", 3)])
    assert "거센소리" in item_text(asp)


def test_remainder_becomes_cannot_judge(base):
    tgt, _ = base
    T, U = _pair(tgt.audio, make_melody(seed=5).audio)
    rng = np.random.default_rng(0)
    n = U.grid.n_frames
    T.residual = rng.standard_normal((n, 4)) * 0.1
    U.residual = T.residual + rng.standard_normal((n, 4)) * 0.1
    s = T.meta["notes"][3][0] + 5
    U.residual[s : s + 20] += 3.0  # something the curves don't explain
    exp = explain([U], T).unwrap()
    rem = [sp for sp in exp.cannot_judge if sp.reason == "remainder"]
    assert rem and any(sp.start <= s + 10 < sp.end for sp in rem)
    assert all(sp.reason for sp in exp.cannot_judge)
    notes = explanation_notes(exp)
    assert any("설명할 수 없어요" in line for line in notes)


def test_korean_strings_cover_every_item_kind():
    s = load_strings("ko")
    assert s["category"]["dynamics"] == "강약"
    for attr in ("breathiness", "loudness", "dynamic_range", "register", "quality", "laryngeal", "phone_match"):
        assert attr in s["attribute"]
    cases = [
        ExplanationItem("phonation", "breathiness", [Span(0, 1, ("해",))], 4.0, "dB", 0.9, detail={"target_note": 1}),
        ExplanationItem("dynamics", "loudness", [Span(0, 1, ("달",))], -3.0, "dB", 0.9, detail={"target_note": 1}),
        ExplanationItem("dynamics", "dynamic_range", [Span(0, 9)], -5.0, "dB", 0.9),
        ExplanationItem("diction", "phone_match", [Span(0, 9)], 0.82, "similarity", 0.9),
    ]
    texts = [item_text(i) for i in cases]
    assert texts[0] == "'해'에서 목표보다 숨소리가 4.0dB 더 섞였어요"
    assert texts[1] == "'달'이 목표보다 3.0dB 작게 들려요"
    assert "좁아요" in texts[2] and "82%" in texts[3]

"""M6: coaching policy layer — thresholds, priority, volume, fading, self-assessment,
summary, practice mapping, onboarding and the vocal-health guard."""

import json
import re
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from gyeol.coach import (
    AttemptMetrics,
    AttributeThreshold,
    CoachConfig,
    CoachSession,
    DiscriminationTrial,
    FatigueMonitor,
    IntervalTrial,
    MelodyTrial,
    PhonationLog,
    PitchMatchTrial,
    PracticeMap,
    PriorityConfig,
    ThresholdSet,
    VoiceRange,
    attempt_metrics,
    check_phrase,
    fit_attribute_threshold,
    fold_octave,
    perception_threshold,
    rank,
    restricted_for_level,
    score_onboarding,
)
from gyeol.coach.health import load_norms
from gyeol.core import Consistency, Explanation, ExplanationItem, FrameGrid, Provenance, Span

GRID = FrameGrid(44100, 512, 400)


def item(cat, attr, mag, conf=0.9, note=1, aud=None, status=None, cons=Consistency.UNDETERMINED, **detail):
    d = {"target_note": note, **detail}
    if status:
        d["status"] = status
    return ExplanationItem(cat, attr, [Span(10 * max(note, 0), 10 * max(note, 0) + 8)], mag, "u", conf, cons, None, aud, d)


def exp_of(*items):
    return Explanation(GRID, np.arange(GRID.n_frames, dtype=float), 0.0, list(items), [], 1)


def thresholds(**overrides):
    base = {
        "intonation_offset": AttributeThreshold.flat("intonation_offset", 10.0, 0.3, "cents"),
        "global_offset": AttributeThreshold.flat("global_offset", 10.0, 0.3, "cents"),
        "interval_compression": AttributeThreshold.flat("interval_compression", 3.0, 0.3, "%"),
        "onset_timing": AttributeThreshold.flat("onset_timing", 30.0, 0.3, "ms"),
        "tempo": AttributeThreshold.flat("tempo", 3.0, 0.3, "%"),
        "scoop": AttributeThreshold.flat("scoop", 30.0, 0.3, "cents"),
        "vibrato_extent": AttributeThreshold.flat("vibrato_extent", 15.0, 0.3, "cents"),
        "loudness": AttributeThreshold.flat("loudness", 1.5, 0.3, "dB"),
        "breathiness": AttributeThreshold.flat("breathiness", 2.0, 0.3, "dB"),
        "register": AttributeThreshold.flat("register", 0.2, 0.3, "probability"),
        "quality_*": AttributeThreshold.flat("quality_*", 0.2, 0.3, "probability"),
        "laryngeal_*": AttributeThreshold.flat("laryngeal_*", 0.2, 0.3, "probability"),
    }
    base.update(overrides)
    return ThresholdSet(base, {"data": "unit-test numbers", "synthetic": True})


# ================================================================ thresholds


def test_fit_attribute_threshold_from_validation_data():
    rng = np.random.default_rng(0)
    truth = rng.uniform(-80, 80, 200)
    a, b = truth + rng.normal(0, 4, 200), truth + rng.normal(0, 4, 200)  # two conditions, SD 4 cents each
    conf = rng.uniform(0, 1, 2000)
    err = rng.normal(0, 1, 2000) * (2 + 30 * (1 - conf) ** 2)  # error shrinks with confidence
    t = fit_attribute_threshold("intonation_offset", a, b, conf, err, "cents")
    assert t.mdc == pytest.approx(1.96 * np.sqrt(2) * 4, rel=0.15)  # MDC95
    assert t.usable and t.min_confidence < 0.01
    assert np.all(np.diff(t.error_bound) <= 1e-12)  # E95 non-increasing in confidence
    # uncertainty: large at low confidence, the MDC floor at high confidence, infinite below the evidence
    assert t.uncertainty(0.05) > 3 * t.uncertainty(0.95) and t.uncertainty(0.99) >= t.mdc
    assert t.uncertainty(0.99) == pytest.approx(max(t.mdc, 1.96 * 2.0), rel=0.35)
    assert t.uncertainty(-1.0) == float("inf")
    few = fit_attribute_threshold("x", a, b, conf[:3], err[:3])
    assert not few.usable and few.uncertainty(0.9) == float("inf")
    with pytest.raises(ValueError):
        fit_attribute_threshold("x", [1, 2], [1, 2], conf, err)


def test_threshold_set_rules_and_roundtrip(tmp_path):
    ts = thresholds(tempo=AttributeThreshold("tempo", 3.0, (), (), None))
    ok = lambda it: ts.passes(it)  # noqa: E731
    assert ok(item("pitch", "intonation_offset", -40)) == (True, "ok")
    assert ok(item("pitch", "intonation_offset", -5)) == (False, "below_magnitude")
    assert ok(item("pitch", "intonation_offset", -40, conf=0.1)) == (False, "below_confidence")
    assert ok(item("ornament", "scoop", -5, status="missing")) == (True, "ok")  # categorical: magnitude secondary
    assert ok(item("phonation", "register", 0.5, status="same")) == (False, "no_difference")
    assert ok(item("phonation", "quality_rough", 0.5)) == (True, "ok")  # pattern
    assert ok(item("ornament", "kkeokki", 100)) == (False, "no_threshold")  # never shown without a fitted threshold
    assert ok(item("rhythm", "tempo", 10)) == (False, "attribute_not_reliable")
    p = ts.to_json(tmp_path / "t.json")
    back = ThresholdSet.from_json(p)
    assert back.provenance == ts.provenance and back.lookup("quality_fry").uncertainty(0.9) == 0.2
    assert back.lookup("intonation_offset") == ts.lookup("intonation_offset")
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(ValueError):
        ThresholdSet.from_json(tmp_path / "bad.json")
    fitted = ThresholdSet.fitted([AttributeThreshold.flat("a", 1.0, 0.5)], "synthetic knob recovery", synthetic=True)
    assert fitted.provenance["synthetic"] is True and "fitted_utc" in fitted.provenance


def test_session_requires_fitted_thresholds_and_shows_nothing_without_them():
    with pytest.raises(TypeError):
        CoachSession(None)
    s = CoachSession(ThresholdSet({}, {"data": "none"}))
    fb = s.new_attempt(exp_of(item("pitch", "intonation_offset", -80))).reveal()
    assert fb.primary is None and fb.withheld == {"no_threshold": 1} and fb.reason == "no_items"


def test_no_korean_text_in_coach_code():
    src = Path(__file__).parents[1] / "src" / "gyeol" / "coach"
    for f in src.glob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"[가-힣]", text), f"Korean user-facing text in {f.name}: belongs in resources"


# ================================================================ priority


def test_priority_reliability_times_audibility():
    items = [item("pitch", "intonation_offset", -40, conf=0.9, aud=0.1, note=1),
             item("rhythm", "onset_timing", 80, conf=0.5, aud=0.4, note=2),
             item("ornament", "scoop", -100, conf=0.8, aud=0.05, note=3, status="missing")]
    r = rank(items, thresholds())
    assert [x.item.attribute for x in r] == ["onset_timing", "intonation_offset", "scoop"]
    assert r[0].score == pytest.approx(0.2) and r[0].basis == "audibility"


def test_priority_fallback_without_audibility_is_size_over_mdc():
    items = [item("pitch", "intonation_offset", -20, aud=0.5, note=1), item("rhythm", "onset_timing", 150, aud=None, note=2)]
    r = rank(items, thresholds())
    assert all(x.basis == "magnitude_over_uncertainty" for x in r)
    assert r[0].item.attribute == "onset_timing" and r[0].score == pytest.approx(0.9 * 150 / 30)
    assert r[0].basis == "magnitude_over_uncertainty"


def test_priority_tie_order_follows_the_brief():
    cats = [("diction", "laryngeal_fortis"), ("phonation", "breathiness"), ("dynamics", "loudness"),
            ("ornament", "scoop"), ("rhythm", "onset_timing"), ("pitch", "intonation_offset")]
    items = [item(c, a, 1.0, conf=0.8, aud=0.5, note=i) for i, (c, a) in enumerate(cats)]
    order = [x.item.category for x in rank(items, thresholds())]
    assert order == ["pitch", "rhythm", "ornament", "dynamics", "phonation", "diction"]
    # outside the tie tolerance the score wins
    items[0].audibility = 0.9
    assert rank(items, thresholds(), PriorityConfig(tie_tolerance=0.1))[0].item.category == "diction"


# ================================================================ session: volume, self-assessment, fading


def _many():
    return exp_of(*[item("pitch", "intonation_offset", -(20 + 10 * k), note=k) for k in range(6)])


def test_one_primary_and_at_most_two_secondary_items():
    s = CoachSession(thresholds())
    fb = s.new_attempt(_many()).reveal()
    assert fb.given and fb.primary.item.magnitude == -70 and len(fb.secondary) == 2
    assert "먼저 들어 볼 부분" in fb.lines[0] and any("더 보기" in line for line in fb.lines)
    assert CoachSession(thresholds(), CoachConfig(n_secondary=0)).new_attempt(_many()).reveal().secondary == []
    assert fb.thresholds_provenance["synthetic"] is True


def test_self_assessment_first():
    s = CoachSession(thresholds(), CoachConfig(require_self_assessment=True, schedule=(1.0,)))
    a = s.new_attempt(_many())
    prompt = a.self_assessment_prompt()
    assert "느낀" in prompt.question and {k for k, _ in prompt.options} >= {"pitch", "rhythm", "diction", "nothing"}
    with pytest.raises(RuntimeError):
        a.reveal()
    with pytest.raises(ValueError):
        a.record_self_assessment({"tone"})
    a.record_self_assessment({"pitch"})
    assert a.reveal().self_assessment == "matched"
    b = s.new_attempt(_many())
    b.record_self_assessment({"rhythm"})
    fb = b.reveal()
    assert fb.self_assessment == "missed" and fb.primary is not None
    c = s.new_attempt(_many())
    c.record_self_assessment({"nothing"})
    assert c.reveal().self_assessment == "nothing"
    assert CoachSession(thresholds()).new_attempt(_many()).reveal().self_assessment is None  # optional by default


def test_feedback_fades_as_performance_stabilises_and_returns_when_it_worsens():
    s = CoachSession(thresholds(), CoachConfig(schedule=(1.0, 0.5, 0.25), stability_window=3, stability_tolerance=0.25))
    given = [s.new_attempt(_many()).reveal().given for _ in range(30)]
    assert given[0] and given[1]
    assert s.records[-1].stage == 2
    assert 0.2 < np.mean(given[-12:]) <= 0.34  # quarter frequency at the last stage
    withheld = next(fb for fb in [s.new_attempt(_many()).reveal() for _ in range(4)] if not fb.given)
    assert withheld.reason == "fading" and "스스로" in withheld.lines[0] and withheld.notices
    worse = exp_of(*[item("pitch", "intonation_offset", -(80 + 20 * k), note=k) for k in range(6)])
    fb = s.new_attempt(worse).reveal()
    assert fb.given and fb.stage == 0 and fb.frequency == 1.0
    # an unstable performer keeps full feedback
    s2 = CoachSession(thresholds())
    rng = np.random.default_rng(0)
    for _ in range(12):
        k = int(rng.integers(1, 6))
        assert s2.new_attempt(exp_of(*[item("pitch", "intonation_offset", -30 * (j + 1), note=j) for j in range(k)])).reveal().given or s2.records[-1].stage > 0
    with pytest.raises(ValueError):
        CoachConfig(schedule=(1.0, 0.0))


def test_consistency_and_tentative_wording():
    s = CoachSession(thresholds())
    fb = s.new_attempt(exp_of(item("phonation", "register", -0.6, aud=None, target="chest", user="falsetto", status="different",
                                   tentative=True, cons=Consistency.STYLE_OR_HABIT))).reveal()
    assert fb.primary.tentative and "가능성" in " ".join(fb.lines)
    assert "습관" in fb.primary.consistency


def test_session_summary():
    s = CoachSession(thresholds(), CoachConfig(level="intermediate"))
    flat = lambda: item("pitch", "intonation_offset", -40, note=1, cons=Consistency.STYLE_OR_HABIT)  # noqa: E731
    late = lambda: item("rhythm", "onset_timing", 90, note=3)  # noqa: E731
    s.new_attempt(exp_of(flat(), late()))
    s.new_attempt(exp_of(flat(), late()))
    s.new_attempt(exp_of(flat()))
    sm = s.summary()
    assert sm.n_attempts == 3
    assert [i.attribute for i in sm.persistent] == ["intonation_offset", "onset_timing"]  # priority tier order
    assert [i.attribute for i in sm.improved] == ["onset_timing"]
    assert [i.attribute for i in sm.habits][0] == "intonation_offset"
    text = "\n".join(sm.lines)
    assert "3번" in text and "좋아진 부분" in text and "이비인후과" in text


# ================================================================ practice mapping


def test_practice_map_lookup_and_filters(tmp_path):
    pm = PracticeMap.load()
    assert pm.for_item(item("ornament", "scoop", -100, status="missing"))[0].id == "scoop_imitation"
    assert pm.for_item(item("ornament", "scoop", 100, status="extra"))[0].id == "straight_tone"
    assert pm.for_item(item("ornament", "vibrato_extent", -40))[0].id == "vibrato_pulses"
    assert pm.for_item(item("ornament", "vibrato_extent", 40))[0].id == "straight_tone"
    assert pm.for_item(item("phonation", "quality_breathy", 0.5))[0].id == "register_sirens"
    assert pm.for_item(item("ornament", "scoop", -100, status="missing"), level="beginner") == []  # intermediate exercise
    assert all(e.level == "beginner" for e in pm.for_item(item("pitch", "intonation_offset", -30), level="beginner"))
    for route in ("own_voice_imitation", "wide_range_pitch_matching", "perception_training", "standard"):
        assert pm.for_route(route), route
    data = json.loads(resources.files("gyeol").joinpath("resources/ko/practice.json").read_text(encoding="utf-8"))
    data["exercises"]["drone_match"]["avoid_when"] = ["fatigue"]
    data["mapping"]["mystery"] = ["does_not_exist"]
    p = tmp_path / "practice.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown exercises"):
        PracticeMap.load(p)
    del data["mapping"]["mystery"]
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    custom = PracticeMap.load(p)
    it = item("pitch", "intonation_offset", -30)
    assert [e.id for e in custom.for_item(it)] == ["drone_match", "slide_into_pitch"]
    assert [e.id for e in custom.for_item(it, avoid={"fatigue"})] == ["slide_into_pitch"]


# ================================================================ onboarding


def _discrimination(threshold_cents, rng):
    trials = []
    for d in (5, 10, 20, 35, 50, 75, 100, 150):
        p = 0.5 + 0.5 / (1 + np.exp(-(d - threshold_cents) / 5))
        trials += [DiscriminationTrial(d, bool(rng.random() < p)) for _ in range(20)]
    return trials


def test_onboarding_separates_production_precision_perception():
    rng = np.random.default_rng(1)
    targets = [-900, -700, -500] * 2
    acc = score_onboarding(
        [PitchMatchTrial(t, t - 1200 + rng.normal(0, 10)) for t in targets],  # an octave low = on target
        [IntervalTrial(700, -900, -200 + rng.normal(0, 15)) for _ in range(3)],
        [MelodyTrial((0, 200, 400, 200), (-300, -100, 105, -95))],  # transposed melody: interval errors only
        _discrimination(15, rng))
    assert acc.production_band == "on_target" and acc.production_mae_cents < 20
    assert acc.precision_band == "consistent" and acc.perception_band == "fine"
    assert acc.routes == ["standard"] and acc.status == {"production": "ok", "precision": "ok", "perception": "ok"}
    assert acc.per_task_mae["melody"] == pytest.approx(5 / 3, abs=0.01)

    poor = score_onboarding([PitchMatchTrial(t, t + rng.choice([-1, 1]) * rng.uniform(150, 350)) for t in targets], [], [],
                            _discrimination(15, rng))
    assert poor.production_band == "developing" and poor.perception_band == "fine"
    assert poor.routes == ["own_voice_imitation", "wide_range_pitch_matching"]
    assert poor.precision_band == "variable"

    deaf_free = score_onboarding([PitchMatchTrial(t, t + 5) for t in targets], [], [],
                                 [DiscriminationTrial(d, i % 2 == 0) for d in (25, 50, 100) for i in range(6)])
    assert deaf_free.perception_band == "developing" and deaf_free.status["perception"] == "not_reached"
    assert deaf_free.perception_threshold_cents is None and "perception_training" in deaf_free.routes

    few = score_onboarding([PitchMatchTrial(-900, -890)], [], [], [])
    assert few.production_mae_cents is None and few.routes == [] and few.status["production"] == "not_enough_trials"
    with pytest.raises(ValueError):
        score_onboarding([], [], [MelodyTrial((0, 100), (0,))], [])


def test_perception_threshold_and_octave_folding():
    trials = [DiscriminationTrial(10, c) for c in [True] * 5 + [False] * 5] + \
             [DiscriminationTrial(30, c) for c in [True] * 9 + [False]]
    thr, st = perception_threshold(trials, 0.75)
    assert st == "ok" and thr == pytest.approx(10 + 20 * (0.25 / 0.4))
    assert fold_octave(1190) == pytest.approx(-10) and fold_octave(-1210) == pytest.approx(-10) and fold_octave(599) == 599


def test_user_facing_strings_never_label_the_user():
    forbidden = ("음치", "박치", "tone-deaf", "tone deaf", "amusic", "amusia")
    base = resources.files("gyeol").joinpath("resources/ko")
    for f in base.iterdir():
        if f.name.endswith(".json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            data.pop("_comment", None)
            text = json.dumps(data, ensure_ascii=False).lower()
            assert not any(w in text for w in forbidden), f.name


# ================================================================ health guard


VR = VoiceRange(-1500, 300, -1200, -300)  # cents re A4 (≈ C3..C5 range, A3..F#4 comfortable)


def test_voice_range_and_phrase_check():
    with pytest.raises(ValueError):
        VoiceRange(0, 100, 50, 20)
    vr = VoiceRange.from_samples(np.linspace(-1500, 300, 200), np.linspace(-1200, -300, 50))
    assert vr.low_cents < vr.tess_low_cents < vr.tess_high_cents < vr.high_cents
    assert check_phrase([-1000, -800, -600], VR).status == "comfortable"
    up = check_phrase([200, 400, 600], VR)  # an octave too high for this voice
    assert up.octave_shift_cents == -1200 and up.status == "comfortable"
    no_oct = check_phrase([-400, -200, 0], VR, allow_octave=False)
    assert no_oct.status == "above_tessitura" and no_oct.transpose_semitones < 0
    assert check_phrase([-400, -200, 0], VR, allow_octave=False).frac_outside_range == 0
    wide = check_phrase([-2000, -1000, 1000], VR, allow_octave=False)
    assert wide.status == "out_of_range"
    assert check_phrase([np.nan], VR).status == "no_notes"


def test_beginner_restrictions():
    rough = item("phonation", "quality_rough", -0.5)  # target rougher: coaching would push toward roughness
    assert restricted_for_level(rough, "beginner") == "beginner_restricted:rough"
    assert restricted_for_level(rough, "advanced") is None
    assert restricted_for_level(item("phonation", "quality_rough", 0.5), "beginner") is None  # less roughness is fine
    belt = item("phonation", "register", -0.6, note=1, target="chest", status="different")
    notes = [-900, -100, -800]
    assert restricted_for_level(belt, "beginner", VR, notes) == "beginner_restricted:high_belt"
    assert restricted_for_level(belt, "beginner", VR, [-900, -800, -800]) is None
    s = CoachSession(thresholds(), CoachConfig(level="beginner"), voice_range=VR, target_notes_cents=notes)
    fb = s.new_attempt(exp_of(rough, belt, item("pitch", "intonation_offset", -30, note=0))).reveal()
    assert fb.withheld.get("beginner_restricted") == 2 and fb.primary.item.attribute == "intonation_offset"


def test_phonation_time_and_fatigue():
    h = load_norms()["health"]
    log = PhonationLog()
    log.add(h["session_phonation_warn_s"] - 1, "d1")
    assert log.warnings("d1") == []
    log.add(2, "d1")
    assert log.warnings("d1") == ["phonation_session"]
    with pytest.raises(ValueError):
        log.add(-1, "d1")
    rng = np.random.default_rng(0)
    steady = FatigueMonitor()
    for _ in range(12):
        steady.add(AttemptMetrics(8 + rng.normal(0, 1), -300 + rng.normal(0, 10), -30 + rng.normal(0, 1)))
    assert steady.flags() == []
    tired = FatigueMonitor()
    for i in range(12):
        tired.add(AttemptMetrics(8 + 1.5 * i + rng.normal(0, 0.5), -300 - 25 * i + rng.normal(0, 5), -30 + 1.0 * i + rng.normal(0, 0.3)))
    assert set(tired.flags()) == {"fatigue_instability", "fatigue_top_range", "fatigue_breath"}
    short = FatigueMonitor(tired.history[: h["fatigue_min_attempts"] - 1])
    assert short.flags() == []
    # the session surfaces the flags, keeps the referral notice and avoids fatigue-tagged practice
    s = CoachSession(thresholds())
    for m in tired.history:
        fb = s.new_attempt(_many(), metrics=m, voiced_s=10.0).reveal()
    assert any("쉬어" in n for n in fb.notices) and "이비인후과" in fb.notices[0]
    assert s.summary().phonation_s == pytest.approx(120.0)


def test_attempt_metrics_from_a_representation():
    from .helpers import make_melody, rep_of

    clean = attempt_metrics(rep_of(make_melody(dur=0.4, gap=0.15).audio, Provenance.USER).unwrap())
    breathy = attempt_metrics(rep_of(make_melody(dur=0.4, gap=0.15, aspiration=0.4).audio, Provenance.USER).unwrap())
    assert np.isfinite([clean.instability_cents, clean.top_cents, clean.breath_db]).all()
    assert breathy.breath_db > clean.breath_db + 3
    assert clean.top_cents == pytest.approx(1200 * np.log2(392 / 440), abs=30)  # highest note G4


# ================================================================ integration with a real explanation


def test_coach_on_a_real_explanation():
    from gyeol.coach import note_centres
    from gyeol.explain import explain

    from .helpers import make_melody, pad_to, rep_of

    t, u = pad_to(make_melody().audio, make_melody(detune=(0, -45, 0, 0, 0, 0), shifts=(0, 0, 0, 0.09, 0, 0), seed=2).audio)
    target = rep_of(t, Provenance.REFERENCE).unwrap()
    exp = explain([rep_of(u, Provenance.USER).unwrap()], target).unwrap()
    notes = note_centres(target)
    s = CoachSession(thresholds(), voice_range=VR, target_notes_cents=notes)
    fb = s.new_attempt(exp).reveal()
    assert fb.primary.item.key in {("pitch", "intonation_offset", 1), ("rhythm", "onset_timing", 3)}
    shown = {fb.primary.item.key} | {e.item.key for e in fb.secondary}
    assert {("pitch", "intonation_offset", 1), ("rhythm", "onset_timing", 3)} <= shown
    assert fb.primary.practice and "이비인후과" in fb.notices[0]
    assert sum(fb.withheld.values()) + 1 + len(fb.secondary) <= len(exp.items)


def test_knob_recovery_fit_and_coach_example(tmp_path, capsys):
    import importlib.util

    def load(name):
        spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "examples" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    th = tmp_path / "th.json"
    assert load("fit_thresholds").main(["--synthetic", "--takes", "2", "--out", str(th)]) == 0
    ts = ThresholdSet.from_json(th)
    assert ts.provenance["synthetic"] is True and ts.lookup("intonation_offset").usable
    assert ts.lookup("onset_timing").uncertainty(1.0) > 0
    assert load("coach_demo_v2").main(["--synthetic", "--out", str(tmp_path), "--coach", str(th), "--noticed", "pitch"]) == 0
    out = capsys.readouterr().out
    assert "먼저 들어 볼 부분" in out and "센트 낮아요" in out and "합성 데이터" in out and "이비인후과" in out
    assert "스스로 느낀 부분이 맞아요" in out and "2번 부른 결과" in out

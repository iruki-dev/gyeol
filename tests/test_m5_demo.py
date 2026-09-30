"""M5: edits, renderers, consent gating, feasible range, stepwise demos, AI labelling, audibility."""

import json

import numpy as np
import pytest
import soundfile as sf

from gyeol.attributes.extract import analyze
from gyeol.core import ConsentedVoice, ConsentToken, FrameGrid, Provenance, Purpose, Recording, Status
from gyeol.core.consent import ConsentError, SingerVector
from gyeol.demo import (
    DSPRenderer,
    Edit,
    FeasibleRange,
    NeuralRenderer,
    SpreadSpectrumWatermark,
    UserTake,
    clamp_edit,
    edit_for_item,
    item_key,
    read_label,
    render_demo,
    save_labelled,
    stepwise_schedule,
)
from gyeol.demo.hnm import analyze_hnm, synthesize_hnm
from gyeol.encoders.latent import ltas_singer_vector
from gyeol.explain import explain, perceptual_distance, score_audibility

from .helpers import SR, dsp_trackers, make_melody, pad_to, rep_of


def _analyze(rec):
    return analyze(rec, trackers=dsp_trackers()).unwrap()


def _items(exp):
    return {item_key(i): i for i in exp.items}


def _voice(rec, user="alice"):
    tok = ConsentToken(user, frozenset({Purpose.ANALYSIS, Purpose.VOICE_SYNTHESIS}))
    return ConsentedVoice.create(ltas_singer_vector(rec), rec, tok)


@pytest.fixture(scope="module")
def scene():
    tgt = make_melody(vib=(0, 0, 60, 0, 0, 0), scoop=(0, 0, 0, 0, 120, 0), dur=0.7)
    usr = make_melody(detune=(0, -50, 0, 20, 0, 0), shifts=(0, 0, 0.07, 0, 0, 0), dur=0.7, seed=3)
    t, u = pad_to(tgt.audio, usr.audio)
    target = rep_of(t, Provenance.REFERENCE).unwrap()
    rec = Recording(u, SR, Provenance.USER, owner_id="alice")
    take = UserTake(rec, _analyze(rec))
    exp = explain([take.rep], target).unwrap()
    return {"target": target, "target_audio": t, "take": take, "exp": exp, "voice": _voice(rec)}


def _reexplain(audio, target):
    rec = Recording(audio, SR, Provenance.USER, owner_id="alice")
    return _items(explain([_analyze(rec)], target).unwrap())


# ---------------------------------------------------------------- HNM engine


def test_hnm_resynthesis_and_parameter_edits():
    m = make_melody(dur=0.4, gap=0.15)
    rep = rep_of(m.audio).unwrap()
    f0 = 440 * 2 ** (rep.curves["f0_cents"].values / 1200)
    p = analyze_hnm(m.audio, rep.grid, f0)
    T = rep.grid.n_frames
    y = synthesize_hnm(p)
    assert abs(20 * np.log10(np.std(y) / np.std(m.audio))) < 2.0
    re = rep_of(np.pad(y, (0, len(m.audio) - len(y)))).unwrap()
    a, b = rep.curves["f0_cents"].values, re.curves["f0_cents"].values
    ok = np.isfinite(a) & np.isfinite(b)
    assert np.median(np.abs(a[ok] - b[ok])) < 2.0
    up = rep_of(np.pad(synthesize_hnm(p, f0_cents_delta=np.full(T, 100.0)), (0, len(m.audio) - len(y)))).unwrap()
    c = up.curves["f0_cents"].values
    ok = np.isfinite(a) & np.isfinite(c)
    assert np.median(c[ok] - a[ok]) == pytest.approx(100, abs=5)  # pitch edit
    assert 20 * np.log10(np.std(synthesize_hnm(p, gain_db=np.full(T, -6.0))) / np.std(y)) == pytest.approx(-6, abs=0.01)
    breathy = rep_of(np.pad(synthesize_hnm(p, aperiodic_db=np.full(T, 10.0)), (0, len(m.audio) - len(y)))).unwrap()
    da = np.nanmedian(breathy.curves["aperiodic_ratio"].values - re.curves["aperiodic_ratio"].values)
    assert da > 3.0  # more noise (the harmonic-modulation floor makes the measured change < 10 dB)
    # a partial render reproduces the full render there (same per-frame noise seeds)
    part = synthesize_hnm(p, frames=(100, 140))
    assert len(part) == 39 * rep.grid.hop + 1
    assert perceptual_distance(y[100 * 512 : 100 * 512 + len(part)], part, SR) < 0.05


# ---------------------------------------------------------------- consent gating


def test_renderers_refuse_anything_but_the_users_consented_voice(scene):
    take, voice = scene["take"], scene["voice"]
    ren = DSPRenderer()
    with pytest.raises(ConsentError):
        ren.render(voice.singer, take)  # a bare singer vector is not consent
    with pytest.raises(ConsentError):
        ren.render(None, take)
    # a reference / target recording can neither become a voice nor a take
    ref = Recording(scene["target_audio"], SR, Provenance.REFERENCE)
    with pytest.raises(ConsentError):
        ConsentedVoice.create(ltas_singer_vector(ref), ref, ConsentToken("alice", frozenset({Purpose.VOICE_SYNTHESIS})))
    with pytest.raises(ConsentError):
        UserTake(ref, scene["target"])
    # another user's take with alice's voice
    bob = Recording(take.recording.audio, SR, Provenance.USER, owner_id="bob")
    with pytest.raises(ConsentError):
        ren.render(voice, UserTake(bob, _analyze(bob)))
    # revoked consent / analysis-only consent cannot create a voice
    rec = take.recording
    for tok in (ConsentToken("alice", frozenset({Purpose.VOICE_SYNTHESIS}), revoked=True), ConsentToken("alice", frozenset({Purpose.ANALYSIS}))):
        with pytest.raises(ConsentError):
            ConsentedVoice.create(ltas_singer_vector(rec), rec, tok)
    with pytest.raises(TypeError):
        ConsentedVoice(ltas_singer_vector(rec), "alice", "t", rec.recording_id)
    # the whole demo pipeline is gated too
    with pytest.raises(ConsentError):
        render_demo(voice.singer, take, scene["exp"], scene["target"], ("pitch", "intonation_offset", 1), ren)
    with pytest.raises(ConsentError):
        score_audibility(scene["exp"], voice.singer, take, scene["target"], ren)


def test_user_take_requires_matching_representation(scene):
    other = Recording(scene["take"].recording.audio, SR, Provenance.USER, owner_id="alice")
    with pytest.raises(ValueError, match="not analysed"):
        UserTake(other, scene["take"].rep)


# ---------------------------------------------------------------- edits


def test_single_item_edit_corrects_only_that_item(scene):
    take, voice, target, exp = scene["take"], scene["voice"], scene["target"], scene["exp"]
    it = _items(exp)
    e = edit_for_item(it[("pitch", "intonation_offset", 1)], take.rep, target, exp).unwrap()
    assert e.requires == {"f0"} and np.nanmax(e.f0_cents) == pytest.approx(-it[("pitch", "intonation_offset", 1)].magnitude, abs=0.5)
    y = DSPRenderer().render(voice, take, e).unwrap()
    after = _reexplain(y, target)
    assert abs(after[("pitch", "intonation_offset", 1)].magnitude) < 6  # was −50
    assert after[("pitch", "intonation_offset", 3)].magnitude == pytest.approx(20, abs=6)  # untouched
    assert after[("rhythm", "onset_timing", 2)].magnitude == pytest.approx(70, abs=20)  # untouched


def test_onset_retiming_edit(scene):
    take, voice, target, exp = scene["take"], scene["voice"], scene["target"], scene["exp"]
    item = _items(exp)[("rhythm", "onset_timing", 2)]
    e = edit_for_item(item, take.rep, target, exp).unwrap()
    assert e.requires == {"timing"} and np.all(np.diff(e.time_map) >= 0)
    after = _reexplain(DSPRenderer().render(voice, take, e).unwrap(), target)
    assert abs(after[("rhythm", "onset_timing", 2)].magnitude) < 25  # was +70 ms
    assert after[("pitch", "intonation_offset", 1)].magnitude == pytest.approx(-50, abs=8)


def test_missing_ornaments_are_transplanted_from_the_target(scene):
    take, voice, target, exp = scene["take"], scene["voice"], scene["target"], scene["exp"]
    it = _items(exp)
    vib = it[("ornament", "vibrato_extent", 2)]
    scoop = it[("ornament", "scoop", 4)]
    assert scoop.detail["status"] == "missing"
    assert vib.detail["user"] == 0.0 and vib.detail["target"] > 30
    e = edit_for_item(vib, take.rep, target, exp).unwrap() + edit_for_item(scoop, take.rep, target, exp).unwrap()
    y = DSPRenderer().render(voice, take, e).unwrap()
    # the rendered pitch now oscillates like the target's over that note
    rep = _analyze(Recording(y, SR, Provenance.USER, owner_id="alice"))
    s0, s1 = vib.spans[0].start, vib.spans[0].end
    dev = (rep.curves["f0_cents"].values - rep.curves["pitch_center"].values)[s0:s1]
    tdev = np.interp(exp.warp[s0:s1], np.arange(target.grid.n_frames),
                     target.curves["f0_cents"].values - target.curves["pitch_center"].values)
    ok = np.isfinite(dev) & np.isfinite(tdev)
    assert ok.sum() > 0.8 * (s1 - s0) and np.corrcoef(dev[ok], tdev[ok])[0, 1] > 0.9
    assert 0.7 < np.std(dev[ok]) / np.std(tdev[ok]) < 1.3
    after = _reexplain(y, target)
    assert after.get(("ornament", "scoop", 4)) is None or after[("ornament", "scoop", 4)].detail["status"] != "missing"


def test_edit_algebra():
    T = 50
    a = Edit(T, f0_cents=np.ones(T), items=(("pitch", "x", 0),), requires=frozenset({"f0"}))
    tm = np.clip(np.arange(T) + 2.0, 0, T - 1)
    b = Edit(T, time_map=tm, requires=frozenset({"timing"}))
    c = a + b.scaled(0.5)
    assert c.requires == {"f0", "timing"} and np.allclose(c.f0_cents, 1)
    assert np.allclose(c.time_map[5:40], np.arange(5, 40) + 1.0) and np.all(np.diff(c.time_map) >= 0)
    assert not Edit.identity(T).support().any() and a.support().all()
    with pytest.raises(ValueError):
        a + Edit(T + 1)


def test_learned_curve_items_need_a_capable_renderer(scene):
    exp, target, take = scene["exp"], scene["target"], scene["take"]
    from gyeol.core import ExplanationItem, Span

    it = ExplanationItem("phonation", "register", [Span(10, 30)], -0.5, "probability", 0.9, detail={"target_note": 1})
    r = edit_for_item(it, take.rep, target, exp)
    assert r.status is Status.FAILED and "register" in r.reason  # no learned curves on these representations
    e = Edit(take.rep.grid.n_frames, requires=frozenset({"curve:register"}))
    assert DSPRenderer().render(scene["voice"], take, e).status is Status.UNAVAILABLE


# ---------------------------------------------------------------- feasible range


def test_feasible_range_clamps_edits(scene):
    rep = scene["take"].rep
    T = rep.grid.n_frames
    f0 = rep.curves["f0_cents"].values
    top = float(np.nanmax(f0))
    rng = FeasibleRange(float(np.nanmin(f0)) - 50, top + 50, source="test")
    e, report = clamp_edit(Edit(T, f0_cents=np.full(T, 300.0)), rep, rng)
    assert np.nanmax(f0 + e.f0_cents) <= top + 50 + 1e-6
    assert 0 < report.clamped_fraction <= 1 and report.detail["f0"] > 0
    e2, r2 = clamp_edit(Edit(T, f0_cents=np.full(T, 10.0)), rep, rng)
    assert r2.clamped_fraction == 0 and np.allclose(e2.f0_cents, 10.0)
    obs = FeasibleRange.from_takes([rep])
    assert obs.f0_low_cents < np.nanmin(f0) + 100 and obs.f0_high_cents > np.nanmax(f0) - 100 and obs.source == "observed takes"
    onb = FeasibleRange.from_onboarding(110.0, 440.0)
    assert onb.f0_low_cents == pytest.approx(-2400) and onb.f0_high_cents == pytest.approx(0)
    with pytest.raises(ValueError):
        FeasibleRange.from_onboarding(440.0, 110.0)


# ---------------------------------------------------------------- stepwise demo + labelling


def test_stepwise_demo_is_labelled_and_moves_toward_the_target(scene, tmp_path):
    take, voice, target, exp = scene["take"], scene["voice"], scene["target"], scene["exp"]
    sel = ("pitch", "intonation_offset", 1)
    steps = stepwise_schedule(exp, sel)
    assert steps[0].label == "selected" and not steps[0].others
    assert [s.alpha for s in steps[1:]] == [0.5, 1.0] and all(sel not in s.others for s in steps[1:])
    with pytest.raises(KeyError):
        stepwise_schedule(exp, ("pitch", "nope", 0))
    d = render_demo(voice, take, exp, target, sel, DSPRenderer()).unwrap()
    assert len(d.steps) == 3
    base = d.baseline.audio
    dist = [perceptual_distance(base, s.audio.audio, SR) for s in d.steps]
    assert dist[0] > 0 and dist[0] <= dist[1] <= dist[2] + 1e-9
    wm = SpreadSpectrumWatermark()
    for la in [d.baseline] + [s.audio for s in d.steps]:
        assert la.metadata["ai_generated"] is True and la.metadata["consent_token_id"] == voice.token_id
        assert wm.detect(la.audio, SR).value.detected
    assert d.steps[0].audio.metadata["edits"]["items"] == [list(sel)]
    path = save_labelled(tmp_path / "demo.wav", d.steps[0].audio)
    meta = read_label(path)
    assert meta["ai_generated"] and "AI" in meta["notice"]["ko"] and meta["renderer"] == "dsp-hnm"
    with sf.SoundFile(str(path)) as f:
        assert "AI-generated" in f.title and json.loads(f.comment)["ai_generated"]
    raw = path.read_bytes() + path.with_suffix(".ai.json").read_bytes()
    assert b"alice" not in raw  # the user id is only stored hashed
    # a label survives in the sidecar even if the WAV's INFO chunk is dropped
    y, _ = sf.read(str(path))
    sf.write(str(tmp_path / "stripped.wav"), y, SR)
    assert read_label(tmp_path / "stripped.wav") is None
    (tmp_path / "stripped.ai.json").write_text(path.with_suffix(".ai.json").read_text(encoding="utf-8"), encoding="utf-8")
    assert read_label(tmp_path / "stripped.wav")["ai_generated"]
    assert render_demo(voice, take, exp, target, ("pitch", "nope", 0), DSPRenderer()).status is Status.FAILED


# ---------------------------------------------------------------- watermark


def test_watermark_detection():
    x = make_melody(dur=0.3, gap=0.1).audio
    wm = SpreadSpectrumWatermark()
    y = wm.embed(x, SR)
    snr = 10 * np.log10(np.sum(x**2) / np.sum((y - x) ** 2))
    assert snr > 30  # low-level mark
    assert wm.detect(y, SR).value.detected and wm.detect(0.3 * y, SR).value.detected
    q = np.round(y * 0.5 * 32767) / 32767  # 16-bit quantisation
    assert wm.detect(q, SR).value.detected
    assert not wm.detect(x, SR).value.detected
    assert not SpreadSpectrumWatermark(key="other").detect(y, SR).value.detected
    assert wm.detect(y[:8000], SR).status is Status.FAILED
    assert set(wm.describe()) >= {"scheme", "key_id"} and wm.key not in json.dumps(wm.describe())


# ---------------------------------------------------------------- audibility


def test_audibility_ranks_larger_corrections_higher(scene):
    take, voice, target = scene["take"], scene["voice"], scene["target"]
    exp = explain([take.rep], target).unwrap()
    out = score_audibility(exp, voice, take, target, DSPRenderer()).unwrap()
    it = _items(out)
    big, small = it[("pitch", "intonation_offset", 1)], it[("pitch", "intonation_offset", 3)]  # −50 vs +20 cents
    tiny = it[("pitch", "intonation_offset", 0)]
    assert big.audibility > small.audibility > tiny.audibility
    assert it[("rhythm", "onset_timing", 2)].audibility > it[("rhythm", "onset_timing", 0)].audibility
    assert out.meta["audibility"]["n_scored"] == sum(i.audibility is not None for i in out.items) > 10
    assert all("uncalibrated" in i.detail["audibility_status"] or "empty" in i.detail["audibility_status"]
               for i in out.items if i.audibility is not None)


def test_perceptual_distance_basics():
    x = make_melody(dur=0.3, gap=0.1).audio
    assert perceptual_distance(x, x, SR) == 0.0
    assert perceptual_distance(x, 0.5 * x, SR) > perceptual_distance(x, 0.9 * x, SR) > 0
    none = np.zeros(1 + (len(x) - 1) // 512, bool)
    assert perceptual_distance(x, 0.5 * x, SR, frames=none) == 0.0


# ---------------------------------------------------------------- neural renderer


def test_neural_renderer_uses_the_consented_voice(scene):
    import torch

    from gyeol.decoder import AutoencoderConfig, GyeolAutoencoder

    torch.manual_seed(0)
    take = scene["take"]
    cfg = AutoencoderConfig.tiny()
    model = GyeolAutoencoder(cfg)
    rec = take.recording
    tok = ConsentToken("alice", frozenset({Purpose.VOICE_SYNTHESIS}))
    sv = SingerVector(np.random.default_rng(0).standard_normal(cfg.singer_dim), Provenance.USER, rec.recording_id, "alice")
    voice = ConsentedVoice.create(sv, rec, tok)
    ren = NeuralRenderer(model)
    y = ren.render(voice, take, seed=0).unwrap()
    assert len(y) == len(rec.audio) and np.all(np.isfinite(y))
    T = take.rep.grid.n_frames
    y2 = ren.render(voice, take, Edit(T, f0_cents=np.full(T, 100.0), requires=frozenset({"f0"})), seed=0).unwrap()
    assert not np.allclose(y, y2)
    seg = ren.render(voice, take, None, seed=0, frames=(10, 20)).unwrap()
    assert len(seg) == 9 * 512 + 1
    assert ren.render(voice, take, Edit(T, requires=frozenset({"curve:register"}))).status is Status.UNAVAILABLE
    wrong = ConsentedVoice.create(ltas_singer_vector(rec, n_bands=5), rec, tok)
    assert ren.render(wrong, take).status is Status.FAILED
    with pytest.raises(ConsentError):
        ren.render(sv, take)


def test_ltas_vector_inherits_provenance():
    x = make_melody(dur=0.3, gap=0.1).audio
    ref = Recording(x, SR, Provenance.REFERENCE)
    v = ltas_singer_vector(ref)
    assert v.provenance is Provenance.REFERENCE and v.source_recording_id == ref.recording_id
    assert v.vector.shape == (32,) and abs(float(v.vector.mean())) < 1e-5
    assert FrameGrid.for_samples(len(x), SR).n_frames > 0

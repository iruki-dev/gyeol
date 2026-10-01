"""Revision C — library boundary: user state moved to the reference service (C1), the public API and versioned
JSON (C2), and the personal non-commercial license profile (C3)."""

import ast
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from gyeol import api
from gyeol.core.license import LicenseError, Profile, decide, lookup

from .helpers import SR, make_melody

ROOT = Path(__file__).parents[1]


# ================================================================ C1 user state lives in the service


def test_library_has_no_user_state_modules():
    for mod in ("gyeol.store", "gyeol.coach.session"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)
    import gyeol.coach as coach

    for name in ("CoachSession", "CoachConfig", "Attempt", "Feedback", "PhonationLog", "FatigueMonitor", "ConsentStore"):
        assert not hasattr(coach, name)
    # stateless parts stay: consent types and guards, thresholds, priority, health measures, onboarding scoring
    from gyeol.coach import attempt_metrics, check_phrase, fatigue_flags, phonation_warnings, rank, score_onboarding  # noqa: F401
    from gyeol.core import ConsentedVoice, require_consented_voice  # noqa: F401


def test_library_never_imports_the_service():
    offenders = []
    for p in (ROOT / "src" / "gyeol").rglob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            if any(n.split(".")[0] == "gyeol_service" for n in names):
                offenders.append(str(p))
    assert not offenders


def test_service_state_matches_stateless_measures():
    from gyeol.coach import AttemptMetrics, fatigue_flags, phonation_warnings
    from gyeol.coach.health import load_norms
    from gyeol_service import FatigueMonitor, PhonationLog

    h = load_norms()["health"]
    log = PhonationLog()
    log.add(h["session_phonation_warn_s"] - 1, "d1")
    assert log.warnings("d1") == phonation_warnings(log.session_s, log.by_day["d1"]) == []
    log.add(2.0, "d1")
    assert log.warnings("d1") == ["phonation_session"] == phonation_warnings(log.session_s, log.by_day["d1"])[:1]
    with pytest.raises(ValueError):
        phonation_warnings(-1.0, 0.0)
    tired = [AttemptMetrics(10.0 + 3 * i + (i % 2), 600.0 - 20 * i, -20.0 + i) for i in range(12)]
    mon = FatigueMonitor(list(tired))
    assert mon.flags() == fatigue_flags(tired) and "fatigue_instability" in mon.flags()
    assert fatigue_flags(tired[:2]) == []


def test_reference_service_package_is_separate():
    py = (ROOT / "pyproject.toml").read_text()
    assert "gyeol_service" not in py and "reference_service" not in py
    svc = (ROOT / "reference_service" / "pyproject.toml").read_text()
    assert 'packages = ["gyeol_service"]' in svc
    from gyeol_service import CoachSession, ConsentStore, delete_user  # noqa: F401


# ================================================================ C2 public API and JSON


@pytest.fixture(scope="module")
def analysed():
    target = make_melody(seed=1, vib=(0, 0, 50, 0, 0, 0))
    take = make_melody(detune=(0, 0, 40, 0, 0, 0), shifts=(0, 0, 0, 0.07, 0, 0), seed=2)
    t = api.analyze(target.audio, SR, role="reference", lyrics="사랑해요 그대", dsp_only=True).unwrap()
    u = api.analyze(take.audio, SR, owner_id="u1", reference=(target.audio, SR), dsp_only=True).unwrap()
    return target, take, t, u


def test_api_surface():
    for name in ("analyze", "compare", "render_demo", "train", "to_json", "from_json", "json_schema", "OwnVoice"):
        assert name in api.__all__ and callable(getattr(api, name)) or isinstance(getattr(api, name), type)


def test_api_analyze_and_compare(analysed):
    _, _, t, u = analysed
    assert u.meta["latency"]["confidence"] > 0.5 and t.meta["syllables"]
    assert u.meta["profile"] == "commercial"
    e = api.compare(u, t).unwrap()
    it = next(i for i in e.items if i.attribute == "intonation_offset" and i.detail["target_note"] == 2)
    assert it.magnitude == pytest.approx(40, abs=10) and it.spans[0].syllables == ("해",)
    with pytest.raises(ValueError, match="owner_id"):
        api.analyze(np.zeros(SR), SR)
    bad = api.analyze(np.zeros(SR // 50), SR, role="synthetic", dsp_only=True)
    assert not bad.ok and bad.reason  # explicit status, no exception


def _validate(doc, schema, root=None, where="$"):
    """A small JSON-Schema subset (type, const, enum, required, properties, additionalProperties, items, anyOf, $ref)."""
    root = root or schema
    if "$ref" in schema:
        node = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return _validate(doc, node, root, where)
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                _validate(doc, branch, root, where)
                return None
            except AssertionError:
                continue
        raise AssertionError(f"{where}: no anyOf branch matches")
    if "const" in schema:
        assert doc == schema["const"], f"{where}: {doc!r} != {schema['const']!r}"
    if "enum" in schema:
        assert doc in schema["enum"], f"{where}: {doc!r} not in {schema['enum']}"
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        py = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool, "null": type(None)}
        ok = any(isinstance(doc, py[x]) and not (x in ("integer", "number") and isinstance(doc, bool)) for x in types)
        if not ok:
            raise AssertionError(f"{where}: {type(doc).__name__} is not {types}")
    if isinstance(doc, dict):
        for k in schema.get("required", []):
            assert k in doc, f"{where}: missing {k}"
        for k, v in doc.items():
            if k in schema.get("properties", {}):
                _validate(v, schema["properties"][k], root, f"{where}.{k}")
            elif isinstance(schema.get("additionalProperties"), dict):
                _validate(v, schema["additionalProperties"], root, f"{where}.{k}")
    if isinstance(doc, list) and "items" in schema:
        for i, v in enumerate(doc[:50]):
            _validate(v, schema["items"], root, f"{where}[{i}]")
    return None


def test_versioned_json_round_trip_and_schema(analysed, tmp_path):
    from gyeol.schema import SchemaError

    _, _, t, u = analysed
    e = api.compare(u, t).unwrap()
    for obj, name in ((u, "gyeol.representation"), (e, "gyeol.explanation")):
        text = api.to_json(obj, tmp_path / f"{name}.json")
        doc = json.loads(text)
        assert doc["schema"] == name and doc["version"] == 1
        _validate(doc, api.json_schema(name))
        back = api.from_json(tmp_path / f"{name}.json")
        assert json.loads(api.to_json(back)) == doc  # lossless for everything the schema carries
    r2 = api.from_json(tmp_path / "gyeol.representation.json")
    a, b = u.curves["f0_cents"].values, r2.curves["f0_cents"].values
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)])
    e2 = api.from_json(tmp_path / "gyeol.explanation.json")
    assert [i.key for i in e2.items] == [i.key for i in e.items] and e2.premises.keys() == e.premises.keys()
    assert e2.comparison_mode == e.comparison_mode
    doc = json.loads(api.to_json(e))
    with pytest.raises(SchemaError, match="version"):
        api.from_json(json.dumps({**doc, "version": 99}))
    with pytest.raises(SchemaError):
        api.from_json(json.dumps({"schema": "something.else", "version": 1}))


def test_json_leaves_out_biometrics_by_default(analysed):
    from gyeol.core.consent import SingerVector
    from gyeol.schema import representation_to_dict

    _, _, _, u = analysed
    u.singer = SingerVector(np.ones(4), u.provenance, u.recording_id, "u1")
    u.meta["separated_audio"] = np.zeros(10)
    d = representation_to_dict(u)
    assert "singer" not in d and "residual" not in d and "separated_audio" not in d["meta"]
    assert representation_to_dict(u, include_biometric=True)["singer"]["vector"] == [1.0] * 4
    u.singer = None
    u.meta.pop("separated_audio")


def test_render_demo_needs_the_owners_voice_synthesis_consent(analysed, tmp_path):
    from gyeol.demo import read_label

    target, take, t, u = analysed
    e = api.compare(u, t).unwrap()
    no_synth = api.ConsentToken("u1", frozenset({api.Purpose.ANALYSIS}))
    with pytest.raises(api.ConsentError):
        api.render_demo(api.OwnVoice(take.audio, SR, u, no_synth), e, t)
    other = api.ConsentToken("someone-else", frozenset({api.Purpose.VOICE_SYNTHESIS}))
    with pytest.raises(api.ConsentError):
        api.render_demo(api.OwnVoice(take.audio, SR, u, other), e, t)
    ok = api.ConsentToken("u1", frozenset({api.Purpose.VOICE_SYNTHESIS}))
    d = api.render_demo(api.OwnVoice(take.audio, SR, u, ok), e, t, out_dir=tmp_path).unwrap()
    meta = json.loads((tmp_path / "demo.json").read_text())
    _validate(meta, api.json_schema("gyeol.demo"))
    assert meta["ai_generated"] and meta["profile"] == "commercial" and len(meta["steps"]) == len(d.files) - 1
    assert all(read_label(p)["ai_generated"] for p in d.files.values())
    # a reference (another person's) recording can never be the "own voice"
    with pytest.raises((api.ConsentError, ValueError)):
        api.render_demo(api.OwnVoice(target.audio, SR, t, ok), e, t)


def test_coach_demo_uses_the_public_api_only():
    tree = ast.parse((ROOT / "examples" / "coach_demo_v2.py").read_text(encoding="utf-8"))
    gyeol_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            gyeol_imports |= {a.name for a in node.names if a.name.startswith("gyeol")}
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("gyeol"):
            gyeol_imports |= {f"{node.module}.{a.name}" for a in node.names}
    assert gyeol_imports == {"gyeol.api"}


def test_api_train(tmp_path):
    from gyeol.train.config import load_config

    cfg = load_config(ROOT / "configs" / "cpu-smoke" / "heads.yaml", overrides=[
        f"data.cache={tmp_path / 'cache'}", f"run.out={tmp_path / 'run'}", "run.max_steps=4", "run.val_every=2",
        "data.synthetic={n_singers: 3, seconds: 0.6}", "data.split={train: 0.34, val: 0.33, test: 0.33}"])
    r = api.train(cfg, log=lambda s: None)
    assert r.status == "finished" and r.report["profile"] == "commercial"


# ================================================================ C3 personal profile


def test_personal_profile_decisions():
    for name in ("openvpi_nsf_hifigan", "gtsinger"):
        a = lookup(name)
        assert not decide(a, Profile.COMMERCIAL).allowed  # the commercial profile is unchanged
        d = decide(a, Profile.PERSONAL)
        assert d.allowed and any("non-commercial" in n for n in d.notices)
    assert decide(lookup("vocalset"), Profile.PERSONAL).allowed
    from gyeol.core.license import AssetKind, unknown_asset

    assert not decide(unknown_asset("mystery", AssetKind.DATASET), Profile.PERSONAL).allowed
    assert decide(unknown_asset("mystery", AssetKind.DATASET), Profile.RESEARCH).allowed


def test_personal_checkpoints_carry_the_profile(tmp_path):
    import torch

    from gyeol.train.checkpoint import load_checkpoint, save_checkpoint

    sd = {"w": torch.ones(2)}
    with pytest.raises(LicenseError):
        save_checkpoint(tmp_path / "c.pt", sd, name="v", sources=["openvpi_nsf_hifigan"], config={}, profile=Profile.COMMERCIAL)
    info = save_checkpoint(tmp_path / "p.pt", sd, name="v", sources=["openvpi_nsf_hifigan", "vocalset"], config={}, profile=Profile.PERSONAL)
    assert info.profile is Profile.PERSONAL and info.license.value == "noncommercial"
    _, back = load_checkpoint(tmp_path / "p.pt", Profile.PERSONAL)
    assert back.profile is Profile.PERSONAL
    with pytest.raises(LicenseError):
        load_checkpoint(tmp_path / "p.pt", Profile.COMMERCIAL)
    # even from commercially clean sources, a personal-profile checkpoint stays out of commercial use
    save_checkpoint(tmp_path / "q.pt", sd, name="q", sources=["vocalset"], config={}, profile=Profile.PERSONAL)
    with pytest.raises(LicenseError, match="personal profile"):
        load_checkpoint(tmp_path / "q.pt", Profile.COMMERCIAL)
    save_checkpoint(tmp_path / "r.pt", sd, name="r", sources=["vocalset"], config={}, profile=Profile.RESEARCH)
    load_checkpoint(tmp_path / "r.pt", Profile.COMMERCIAL)  # unchanged behaviour for other profiles


def test_outputs_made_under_personal_carry_the_tag(analysed, tmp_path):
    target, take, _, _ = analysed
    t = api.analyze(target.audio, SR, role="reference", dsp_only=True, profile="personal").unwrap()
    u = api.analyze(take.audio, SR, owner_id="u1", dsp_only=True).unwrap()
    assert t.meta["profile"] == "personal" and u.meta["profile"] == "commercial"
    e = api.compare(u, t).unwrap()
    assert e.meta["profile"] == "personal"  # the most restrictive input wins
    assert json.loads(api.to_json(e))["meta"]["profile"] == "personal"
    tok = api.ConsentToken("u1", frozenset({api.Purpose.VOICE_SYNTHESIS}))
    d = api.render_demo(api.OwnVoice(take.audio, SR, u, tok), e, t, out_dir=tmp_path).unwrap()
    assert d.metadata["profile"] == "personal" and all(s["metadata"]["profile"] == "personal" for s in d.metadata["steps"])


def test_personal_training_run_and_noncommercial_data(tmp_path):
    from gyeol.data.manifest import Manifest, ManifestItem, open_manifest
    from gyeol.train.checkpoint import load_checkpoint
    from gyeol.train.config import load_config

    m = Manifest("gtsinger", str(tmp_path), [ManifestItem("a.wav", "s1", {})])
    with pytest.raises(LicenseError):
        open_manifest(m, Profile.COMMERCIAL)
    assert open_manifest(m, Profile.PERSONAL).profile is Profile.PERSONAL
    cfg = load_config(ROOT / "configs" / "cpu-smoke" / "heads.yaml", overrides=[
        "profile=personal", f"data.cache={tmp_path / 'cache'}", f"run.out={tmp_path / 'run'}", "run.max_steps=2", "run.val_every=2",
        "data.synthetic={n_singers: 3, seconds: 0.6}", "data.split={train: 0.34, val: 0.33, test: 0.33}"])
    r = api.train(cfg, log=lambda s: None)
    assert r.report["profile"] == "personal"
    _, info = load_checkpoint(r.best_checkpoint, Profile.PERSONAL)
    assert info.profile is Profile.PERSONAL
    with pytest.raises(LicenseError):
        load_checkpoint(r.best_checkpoint, Profile.COMMERCIAL)


def test_cli_lists_the_personal_profile(capsys):
    from gyeol.cli import main

    assert main(["licenses", "--profile", "personal"]) == 0
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if x.startswith("openvpi_nsf_hifigan"))
    assert "allowed" in line
    unknownish = [x for x in out.splitlines() if x.split()[0] in ("so_vits_svc",)]
    assert unknownish and "allowed" in unknownish[0]  # copyleft code may be used personally (never vendored)

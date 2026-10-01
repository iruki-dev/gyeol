"""Revision C — library boundary: user state moved to the reference service (C1), the public API and versioned
JSON (C2), and the personal non-commercial license profile (C3)."""

import ast
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from gyeol import api

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
    # stateless parts stay: thresholds, priority, health measures, onboarding scoring
    from gyeol.coach import attempt_metrics, check_phrase, fatigue_flags, phonation_warnings, rank, score_onboarding  # noqa: F401
    # consent, AI labelling and license policy belong to the application (revision E)
    for mod in ("gyeol.core.consent", "gyeol.core.license", "gyeol.demo.label", "gyeol.demo.watermark"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)


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
    t = api.analyze(target.audio, SR, lyrics="사랑해요 그대", dsp_only=True).unwrap()
    u = api.analyze(take.audio, SR, reference=(target.audio, SR), dsp_only=True).unwrap()
    return target, take, t, u


def test_api_surface():
    for name in ("analyze", "compare", "render_demo", "train", "to_json", "from_json", "json_schema", "take"):
        assert name in api.__all__ and callable(getattr(api, name)) or isinstance(getattr(api, name), type)


def test_api_analyze_and_compare(analysed):
    _, _, t, u = analysed
    assert u.meta["latency"]["confidence"] > 0.5 and t.meta["syllables"]
    e = api.compare(u, t).unwrap()
    it = next(i for i in e.items if i.attribute == "intonation_offset" and i.detail["target_note"] == 2)
    assert it.magnitude == pytest.approx(40, abs=10) and it.spans[0].syllables == ("해",)
    bad = api.analyze(np.zeros(SR // 50), SR, dsp_only=True)
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
        assert doc["schema"] == name and doc["version"] == {"gyeol.representation": 2, "gyeol.explanation": 1}[name]
        assert "provenance" not in doc
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
    from gyeol.core import SingerVector
    from gyeol.schema import representation_to_dict

    _, _, _, u = analysed
    u.singer = SingerVector(np.ones(4), u.recording_id)
    u.meta["separated_audio"] = np.zeros(10)
    d = representation_to_dict(u)
    assert "singer" not in d and "residual" not in d and "separated_audio" not in d["meta"]
    assert representation_to_dict(u, include_biometric=True)["singer"]["vector"] == [1.0] * 4
    u.singer = None
    u.meta.pop("separated_audio")


def test_render_demo_returns_plain_audio(analysed, tmp_path):
    _, take, t, u = analysed
    e = api.compare(u, t).unwrap()
    d = api.render_demo(take.audio, SR, u, e, t, out_dir=tmp_path).unwrap()
    assert isinstance(d.baseline, np.ndarray) and all(isinstance(y, np.ndarray) for y in d.steps) and d.sr == SR
    meta = json.loads((tmp_path / "demo.json").read_text())
    _validate(meta, api.json_schema("gyeol.demo"))
    assert meta["version"] == 2 and len(meta["steps"]) == len(d.files) - 1 == len(d.steps)
    assert not {"ai_generated", "profile", "provenance"} & set(meta)
    assert all(p.exists() and p.suffix == ".wav" and not p.with_suffix(".ai.json").exists() for p in d.files.values())
    # without out_dir nothing is written
    d2 = api.render_demo(take.audio, SR, u, e, t).unwrap()
    assert d2.files == {} and np.allclose(d2.baseline, d.baseline)
    # the audio must be the audio the representation was analysed from
    with pytest.raises(ValueError, match="sample rates differ"):
        api.render_demo(take.audio, SR // 2, u, e, t)


def test_old_v1_documents_are_still_read(analysed):
    _, _, _, u = analysed
    doc = json.loads(api.to_json(u))
    doc.update(version=1, provenance="user")
    back = api.from_json(json.dumps(doc))
    assert back.recording_id == u.recording_id


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
    assert r.status == "finished" and r.report["provenance"]["sources"] == ["gyeol_synthetic"]

"""Versioned JSON serialisation of gyeol's outputs (revision C2).

Three documents, each tagged with ``{"schema": <name>, "version": <int>}``:

* ``gyeol.representation`` — the interpretable layer of one recording: grid,
  attribute curves (values + confidence), events, quality report, metadata;
* ``gyeol.explanation`` — the comparison of user take(s) with a target: items
  with spans / syllables / magnitudes / confidences, cannot-judge spans,
  premises and withheld judgements, comparison modes, the time warp;
* ``gyeol.demo`` — metadata of a rendered demo (renderer, steps, the items
  and clamping each step applied); the audio itself is not included.

Version history: representation v2 and demo v2 (revision E) dropped the
``provenance`` field and the AI-label metadata; v1 documents are still read.

JSON Schemas for the three live in ``gyeol/resources/schema/`` (draft 2020-12).
Arrays are plain lists with ``null`` for NaN.  Readers accept any document
whose major ``version`` they know and ignore fields they do not know; writers
bump the version when a field changes meaning or is removed.

Privacy: singer vectors and residual latents are voice-derived biometric data.
They are **left out** unless ``include_biometric=True``; the separated audio a
data-preparation run may keep in ``meta`` is never serialised.
"""

from __future__ import annotations

import dataclasses
import json
import math
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from .core.containers import (
    AttributeCurve,
    AttributeCurves,
    Consistency,
    Event,
    Explanation,
    ExplanationItem,
    Premise,
    Representation,
    Span,
    WithheldItem,
)
from .core.grid import FrameGrid

VERSIONS = {"gyeol.representation": 2, "gyeol.explanation": 1, "gyeol.demo": 2}
_DROP_META = {"separated_audio"}


class SchemaError(ValueError):
    pass


def jsonable(x: Any) -> Any:
    """Plain JSON types: arrays → lists (NaN/inf → null), numpy scalars → Python, enums → values, tuples → lists."""
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "fc":
            return [jsonable(v) for v in x.tolist()] if x.ndim else jsonable(float(x))
        return x.tolist()
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items() if k not in _DROP_META}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in x]
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    return str(x)


def _arr(v, dtype=float) -> np.ndarray:
    return np.array([np.nan if e is None else e for e in v] if (v and not isinstance(v[0], list)) else
                    [[np.nan if e is None else e for e in row] for row in v], dtype=dtype)


def _header(name: str) -> dict:
    return {"schema": name, "version": VERSIONS[name]}


def _check(doc: dict, name: str) -> None:
    if not isinstance(doc, dict) or doc.get("schema") != name:
        raise SchemaError(f"not a {name} document (schema={doc.get('schema') if isinstance(doc, dict) else type(doc).__name__})")
    v = doc.get("version")
    if not isinstance(v, int) or v > VERSIONS[name] or v < 1:
        raise SchemaError(f"{name} version {v!r} is not supported (this gyeol reads up to {VERSIONS[name]})")


def _grid(g: FrameGrid) -> dict:
    return {"sr": g.sr, "hop": g.hop, "n_frames": g.n_frames}


# ---------------------------------------------------------------- representation


def representation_to_dict(rep: Representation, *, include_biometric: bool = False) -> dict:
    d = _header("gyeol.representation")
    d.update(grid=_grid(rep.grid), recording_id=rep.recording_id,
             curves={n: {"unit": c.unit, "labels": list(c.labels), "values": jsonable(c.values), "confidence": jsonable(c.confidence),
                         "meta": jsonable(c.meta)} for n, c in rep.curves.items()},
             events=[jsonable(e) for e in rep.events], quality=jsonable(rep.quality), meta=jsonable(rep.meta),
             env=None if rep.env is None else jsonable(rep.env))
    if include_biometric:
        d["singer"] = None if rep.singer is None else {"vector": jsonable(rep.singer.vector),
                                                       "source_recording_id": rep.singer.source_recording_id}
        d["residual"] = None if rep.residual is None else jsonable(rep.residual)
    return d


def representation_from_dict(d: dict) -> Representation:
    _check(d, "gyeol.representation")
    g = FrameGrid(**d["grid"])
    curves = AttributeCurves(g)
    for name, c in d["curves"].items():
        curves.add(AttributeCurve(name, _arr(c["values"]), _arr(c["confidence"]), g, c.get("unit", ""), tuple(c.get("labels", ())),
                                  dict(c.get("meta", {}))))
    events = [Event(**e) for e in d.get("events", [])]
    env = None if d.get("env") is None else _arr(d["env"])
    return Representation(g, curves, d["recording_id"], events, None, env, None,
                          dict(d.get("quality", {})), dict(d.get("meta", {})))


# ---------------------------------------------------------------- explanation


def _item_to_dict(it: ExplanationItem) -> dict:
    return {"category": it.category, "attribute": it.attribute, "spans": [jsonable(s) for s in it.spans], "magnitude": jsonable(it.magnitude),
            "unit": it.unit, "confidence": jsonable(it.confidence), "consistency": it.consistency.value,
            "delta": None if it.delta is None else jsonable(it.delta), "audibility": jsonable(it.audibility), "detail": jsonable(it.detail)}


def _span(s: dict) -> Span:
    return Span(int(s["start"]), int(s["end"]), tuple(s.get("syllables", ())), s.get("reason", ""))


def explanation_to_dict(exp: Explanation) -> dict:
    d = _header("gyeol.explanation")
    d.update(grid=_grid(exp.grid), warp=jsonable(exp.warp), transposition_cents=jsonable(exp.transposition_cents),
             items=[_item_to_dict(it) for it in exp.items], cannot_judge=[jsonable(s) for s in exp.cannot_judge], n_takes=exp.n_takes,
             premises={k: jsonable(p) for k, p in exp.premises.items()}, withheld=[jsonable(w) for w in exp.withheld],
             comparison_mode=dict(exp.comparison_mode), meta=jsonable(exp.meta))
    return d


def explanation_from_dict(d: dict) -> Explanation:
    _check(d, "gyeol.explanation")
    items = []
    for i in d["items"]:
        items.append(ExplanationItem(i["category"], i["attribute"], [_span(s) for s in i["spans"]],
                                     float("nan") if i["magnitude"] is None else float(i["magnitude"]), i["unit"],
                                     float("nan") if i["confidence"] is None else float(i["confidence"]), Consistency(i["consistency"]),
                                     None if i.get("delta") is None else _arr(i["delta"]), i.get("audibility"), dict(i.get("detail", {}))))
    return Explanation(FrameGrid(**d["grid"]), _arr(d["warp"]), d.get("transposition_cents"), items,
                       [_span(s) for s in d.get("cannot_judge", [])], int(d["n_takes"]), dict(d.get("meta", {})),
                       {k: Premise(**p) for k, p in d.get("premises", {}).items()}, [WithheldItem(**w) for w in d.get("withheld", [])],
                       dict(d.get("comparison_mode", {})))


# ---------------------------------------------------------------- demo


def demo_to_dict(demo, item_key=None, files: dict[str, str] | None = None) -> dict:
    """Metadata of a :class:`gyeol.demo.Demo` (no audio): renderer, and what every step applied."""
    d = _header("gyeol.demo")
    d.update(selected_item=None if item_key is None else list(item_key), sr=int(demo.sr), renderer=demo.renderer,
             steps=[{"label": st.step.label, "alpha": jsonable(st.step.alpha), "selected": list(st.step.selected),
                     "others": [list(k) for k in st.step.others], "clamped_fraction": jsonable(st.clamp.clamped_fraction),
                     "applied": jsonable(st.applied)} for st in demo.steps],
             files=dict(files or {}))
    return d


# ---------------------------------------------------------------- generic


def to_dict(obj, **kw) -> dict:
    if isinstance(obj, Representation):
        return representation_to_dict(obj, **kw)
    if isinstance(obj, Explanation):
        return explanation_to_dict(obj)
    if hasattr(obj, "baseline") and hasattr(obj, "steps"):
        return demo_to_dict(obj, **kw)
    raise TypeError(f"no gyeol schema for {type(obj).__name__}")


def from_dict(d: dict):
    name = d.get("schema") if isinstance(d, dict) else None
    if name == "gyeol.representation":
        return representation_from_dict(d)
    if name == "gyeol.explanation":
        return explanation_from_dict(d)
    if name == "gyeol.demo":
        _check(d, name)
        return d
    raise SchemaError(f"unknown document schema {name!r}")


def to_json(obj, path: str | Path | None = None, **kw) -> str:
    text = json.dumps(to_dict(obj, **kw), ensure_ascii=False, allow_nan=False)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def from_json(text_or_path: str | Path):
    p = Path(text_or_path) if not str(text_or_path).lstrip().startswith("{") else None
    text = p.read_text(encoding="utf-8") if p is not None else str(text_or_path)
    return from_dict(json.loads(text))


def json_schema(name: str) -> dict:
    """The JSON Schema document for ``gyeol.representation`` / ``gyeol.explanation`` / ``gyeol.demo``."""
    short = name.split(".", 1)[-1]
    return json.loads(resources.files("gyeol").joinpath(f"resources/schema/{short}.v{VERSIONS[name]}.json").read_text(encoding="utf-8"))


__all__ = ["SchemaError", "VERSIONS", "demo_to_dict", "explanation_from_dict", "explanation_to_dict", "from_dict", "from_json",
           "json_schema", "jsonable", "representation_from_dict", "representation_to_dict", "to_dict", "to_json"]

"""Practice suggestions from a coach-authored data file.

All exercise text lives in ``resources/<lang>/practice.json``; this module only
looks keys up.  Lookup order for an item:

1. ``<attribute>:<status>`` (``missing`` / ``extra`` / ``different``) or
   ``<attribute>:<pos|neg>`` (sign of the magnitude);
2. ``<attribute>``;
3. fnmatch patterns such as ``quality_*``.

Onboarding routes use ``route:<id>``.  Exercises can be filtered by level
(``beginner`` < ``intermediate`` < ``advanced``) and by ``avoid_when`` tags
(e.g. ``fatigue``) set by the health guard.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from ..core.containers import ExplanationItem

LEVELS = ("beginner", "intermediate", "advanced")


@dataclass(frozen=True)
class Exercise:
    id: str
    title: str
    instructions: str
    level: str = "beginner"
    minutes: float = 3.0
    avoid_when: tuple[str, ...] = ()


@dataclass
class PracticeMap:
    exercises: dict[str, Exercise]
    mapping: dict[str, list[str]]
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None, lang: str = "ko") -> "PracticeMap":
        if path is None:
            text = resources.files("gyeol").joinpath(f"resources/{lang}/practice.json").read_text(encoding="utf-8")
        else:
            text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text)
        ex = {k: Exercise(k, v["title"], v["instructions"], v.get("level", "beginner"), float(v.get("minutes", 3.0)),
                          tuple(v.get("avoid_when", ()))) for k, v in data["exercises"].items()}
        mapping = {k: list(v) for k, v in data["mapping"].items()}
        unknown = sorted({e for ids in mapping.values() for e in ids} - set(ex))
        if unknown:
            raise ValueError(f"practice mapping refers to unknown exercises: {unknown}")
        bad = sorted(e.id for e in ex.values() if e.level not in LEVELS)
        if bad:
            raise ValueError(f"unknown level for exercises {bad}")
        return cls(ex, mapping, {k: v for k, v in data.items() if k not in ("exercises", "mapping")})

    def _ids(self, key: str) -> list[str] | None:
        if key in self.mapping:
            return self.mapping[key]
        for pat, ids in self.mapping.items():
            if any(c in pat for c in "*?[") and fnmatch.fnmatchcase(key, pat):
                return ids
        return None

    def _filter(self, ids: list[str], level: str, avoid: set[str]) -> list[Exercise]:
        cap = LEVELS.index(level) if level in LEVELS else len(LEVELS) - 1
        return [self.exercises[i] for i in ids if LEVELS.index(self.exercises[i].level) <= cap and not avoid & set(self.exercises[i].avoid_when)]

    def for_item(self, item: ExplanationItem, level: str = "advanced", avoid: set[str] | frozenset[str] = frozenset()) -> list[Exercise]:
        attr = item.attribute
        keys = []
        if item.detail.get("status"):
            keys.append(f"{attr}:{item.detail['status']}")
        keys += [f"{attr}:{'pos' if item.magnitude >= 0 else 'neg'}", attr]
        for k in keys:
            ids = self._ids(k)
            if ids is not None:
                return self._filter(ids, level, set(avoid))
        return []

    def for_route(self, route: str, level: str = "advanced") -> list[Exercise]:
        return self._filter(self._ids(f"route:{route}") or [], level, set())

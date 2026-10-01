"""Dataset adapters: local dataset trees → :class:`Manifest` objects.

Adapters never download anything.  Each one scans a directory the user has
obtained themselves and emits a manifest whose ``dataset`` is the dataset's
name in :mod:`gyeol.core.assets` (its listed license is recorded with
anything built from it).

Label vocabularies are mapped onto gyeol's attribute names
(:data:`PHONATION_LABELS`).  Where a dataset's on-disk schema could not be
verified from its documentation here, the adapter takes an explicit mapping
(``field_map`` / ``technique_map``) and reports every file it could not map
instead of guessing silently.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.status import Result, Status
from .manifest import Manifest, ManifestItem

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg"}

#: gyeol's register / phonation vocabulary (targets of the M3 heads)
PHONATION_LABELS = ("chest", "mixed", "falsetto", "breathy", "pressed_belt", "pharyngeal_twang", "fry", "rough")


@dataclass
class ScanReport:
    manifest: Manifest
    n_files: int
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)

    def summary(self) -> str:
        return f"{self.manifest.dataset}: {len(self.manifest.items)} items from {self.n_files} files, {len(self.skipped)} skipped"


def _audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT)


# ---------------------------------------------------------------------------
# VocalSet (CC BY 4.0)
# ---------------------------------------------------------------------------

VOCALSET_TECHNIQUES = {
    # VocalSet technique folder → gyeol labels (phonation label or None, plus flags)
    "belt": {"phonation": "pressed_belt"},
    "breathy": {"phonation": "breathy"},
    "vocal_fry": {"phonation": "fry"},
    "straight": {"phonation": None, "vibrato": False},
    "vibrato": {"phonation": None, "vibrato": True},
    "fast_forte": {}, "fast_piano": {}, "forte": {}, "pp": {}, "slow_forte": {}, "slow_piano": {},
    "messa": {}, "trill": {}, "trillo": {}, "lip_trill": {}, "inhaled": {"exclude": True}, "spoken": {"exclude": True},
}
VOCALSET_CONTEXTS = {"arpeggios", "scales", "long_tones", "excerpts"}


def scan_vocalset(root: str | Path) -> Result[ScanReport]:
    """VocalSet: ``<root>/**/<singer>/<context>/<technique>/<file>.wav``.

    Singer, context and technique are read from path components (singer =
    ``female<N>``/``male<N>``); the vowel from a trailing ``_[aeiou]`` in the
    file name when present.
    """
    root = Path(root)
    if not root.is_dir():
        return Result.failure(f"{root} is not a directory")
    files = _audio_files(root)
    items, skipped = [], []
    for p in files:
        parts = [x.lower() for x in p.relative_to(root).parts]
        singer = next((x for x in parts if re.fullmatch(r"(fe)?male\d+", x)), None)
        tech = next((x for x in parts[:-1] if x in VOCALSET_TECHNIQUES), None)
        ctx = next((x for x in parts[:-1] if x in VOCALSET_CONTEXTS), None)
        if singer is None or tech is None:
            skipped.append((str(p), "singer or technique folder not recognised"))
            continue
        info = VOCALSET_TECHNIQUES[tech]
        if info.get("exclude"):
            skipped.append((str(p), f"technique {tech!r} is not sung phonation"))
            continue
        m = re.search(r"_([aeiou])$", p.stem.lower())
        labels: dict[str, Any] = {"technique": tech, "context": ctx}
        if info.get("phonation"):
            labels["phonation"] = info["phonation"]
        if "vibrato" in info:
            labels["vibrato"] = info["vibrato"]
        if m:
            labels["vowel"] = m.group(1)
        items.append(ManifestItem(str(p.relative_to(root)), singer, labels))
    return Result.success(ScanReport(Manifest("vocalset", str(root), items), len(files), skipped))


# ---------------------------------------------------------------------------
# AI Hub #465 / #473 (commercial use under conditions)
# ---------------------------------------------------------------------------


@dataclass
class AIHubFieldMap:
    """Where to find labels in an AI Hub JSON label file.

    Keys are dotted paths into the JSON (``"a.b.0.c"``).  The defaults are
    *placeholders* — AI Hub's label schema must be checked against the
    dataset's own documentation after download (use :func:`inspect_json_keys`)
    and the map adjusted; unmapped files are reported, never guessed.
    """

    singer: str = "singer.id"
    gender: str | None = "singer.gender"
    genre: str | None = "song.genre"
    lyrics: str | None = "song.lyrics"
    #: label name → dotted path; values are coerced to bool/float/str as found
    attributes: dict[str, str] = field(default_factory=lambda: {
        "vibrato": "technique.vibrato", "kkeokki": "technique.bending", "breath": "technique.breath", "husky": "tone.husky",
    })


def _get(d: Any, path: str) -> Any:
    cur = d
    for part in path.split("."):
        if isinstance(cur, list):
            if not part.isdigit() or int(part) >= len(cur):
                return None
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur


def inspect_json_keys(root: str | Path, limit: int = 20) -> dict[str, int]:
    """Count dotted key paths across up to ``limit`` JSON files (schema discovery)."""
    counts: dict[str, int] = {}

    def walk(d: Any, prefix: str) -> None:
        if isinstance(d, dict):
            for k, v in d.items():
                walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(d, list) and d:
            walk(d[0], f"{prefix}.0")
        else:
            counts[prefix] = counts.get(prefix, 0) + 1

    for p in sorted(Path(root).rglob("*.json"))[:limit]:
        walk(json.loads(p.read_text(encoding="utf-8")), "")
    return counts


def scan_aihub(root: str | Path, dataset: str, field_map: AIHubFieldMap | None = None) -> Result[ScanReport]:
    """AI Hub singing datasets: audio files with a same-stem ``.json`` label file anywhere under ``root``.

    ``dataset`` is ``"aihub_465_multi_singer"`` or ``"aihub_473_guide_vocal"``
    (AI Hub's usage terms are listed in :mod:`gyeol.core.assets`).
    """
    if dataset not in ("aihub_465_multi_singer", "aihub_473_guide_vocal"):
        return Result.failure(f"unknown AI Hub dataset {dataset!r}")
    root = Path(root)
    if not root.is_dir():
        return Result.failure(f"{root} is not a directory")
    fm = field_map or AIHubFieldMap()
    labels_by_stem = {p.stem: p for p in root.rglob("*.json")}
    files = _audio_files(root)
    items, skipped = [], []
    for p in files:
        lp = labels_by_stem.get(p.stem)
        if lp is None:
            skipped.append((str(p), "no label JSON with the same stem"))
            continue
        try:
            d = json.loads(lp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append((str(p), f"unreadable label file: {exc}"))
            continue
        singer = _get(d, fm.singer)
        if singer is None:
            skipped.append((str(p), f"singer field {fm.singer!r} not found (check AIHubFieldMap)"))
            continue
        labels: dict[str, Any] = {}
        for name, path in fm.attributes.items():
            v = _get(d, path)
            if v is not None:
                labels[name] = v
        meta = {k: _get(d, getattr(fm, k)) for k in ("gender", "genre", "lyrics") if getattr(fm, k)}
        meta["label_file"] = str(lp.relative_to(root))
        items.append(ManifestItem(str(p.relative_to(root)), str(singer), labels, {k: v for k, v in meta.items() if v is not None}))
    return Result.success(ScanReport(Manifest(dataset, str(root), items), len(files), skipped))


# ---------------------------------------------------------------------------
# GTSinger (CC BY-NC-SA)
# ---------------------------------------------------------------------------

GTSINGER_TECHNIQUES = {
    "mixed_voice": "mixed", "falsetto": "falsetto", "breathy": "breathy", "pharyngeal": "pharyngeal_twang",
    "vibrato": None, "glissando": None,
}


def scan_gtsinger(root: str | Path, technique_map: dict[str, str | None] | None = None) -> Result[ScanReport]:
    """GTSinger-style paired technique data.

    Expects ``.../<singer>/<technique>/<song>/<Group>/<file>.wav`` where
    ``<Group>`` is ``Control_Group`` (technique off) or ``<Technique>_Group``
    (technique on).  Items of the same song and technique form a pair
    (``meta.pair_id``, ``meta.pair_role`` ∈ {off, on}); pairing is by file
    stem within the song folder.
    """
    root = Path(root)
    if not root.is_dir():
        return Result.failure(f"{root} is not a directory")
    tmap = technique_map or GTSINGER_TECHNIQUES
    files = _audio_files(root)
    items, skipped = [], []
    for p in files:
        rel = p.relative_to(root)
        parts = rel.parts
        group_i = next((i for i, x in enumerate(parts) if x.endswith("_Group")), None)
        if group_i is None or group_i < 3:
            skipped.append((str(p), "no <Group> folder at depth ≥ 3"))
            continue
        singer, technique, song, group = parts[group_i - 3], parts[group_i - 2], parts[group_i - 1], parts[group_i]
        role = "off" if group.lower().startswith("control") else "on"
        tech_key = technique.lower()
        labels: dict[str, Any] = {"technique": tech_key, "technique_on": role == "on"}
        mapped = tmap.get(tech_key)
        if role == "on" and mapped:
            labels["phonation"] = mapped
        pair_id = f"{singer}/{technique}/{song}/{p.stem}"
        items.append(ManifestItem(str(rel), singer, labels, {"pair_id": pair_id, "pair_role": role}))
    return Result.success(ScanReport(Manifest("gtsinger", str(root), items), len(files), skipped))


# ---------------------------------------------------------------------------
# Own recordings
# ---------------------------------------------------------------------------


def scan_own(root: str | Path, include: Callable[[str], bool] | None = None, dataset: str = "own_recordings") -> Result[ScanReport]:
    """In-house recordings described by ``<root>/recordings.json``.

    Each entry: ``{"path": ..., "user_id": ..., "labels": {...}, "pair_id"?: ..., "pair_role"?: ...}``.
    ``include(user_id)`` (optional) lets the application choose which users' recordings are used.
    """
    root = Path(root)
    desc = root / "recordings.json"
    if not desc.exists():
        return Result.failure(f"{desc} not found")
    entries = json.loads(desc.read_text(encoding="utf-8"))
    items, skipped = [], []
    for e in entries:
        uid = e.get("user_id")
        if not uid:
            skipped.append((e.get("path", "?"), "no user_id"))
            continue
        if include is not None and not include(uid):
            skipped.append((e.get("path", "?"), "excluded by the include filter"))
            continue
        meta = {k: e[k] for k in ("pair_id", "pair_role", "session", "device") if k in e}
        items.append(ManifestItem(e["path"], uid, dict(e.get("labels", {})), meta))
    report = ScanReport(Manifest(dataset, str(root), items), len(entries), skipped)
    return Result(Status.OK, report, "", [f"{len(skipped)} recordings skipped"] if skipped else [])

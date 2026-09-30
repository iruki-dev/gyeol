"""Singer-disjoint splits (held-out singers for calibration, probing and transfer)."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from .manifest import Manifest, ManifestItem


def _bucket(key: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def singer_split(manifest: Manifest, fractions: dict[str, float] | None = None, seed: int = 0) -> dict[str, list[ManifestItem]]:
    """Assign whole singers to splits by a seeded hash (stable as data grows).

    Items without a singer id go to the split named ``"unassigned"``.
    """
    fractions = fractions or {"train": 0.8, "calib": 0.1, "test": 0.1}
    total = sum(fractions.values())
    edges, acc = [], 0.0
    for name, f in fractions.items():
        acc += f / total
        edges.append((acc, name))
    out: dict[str, list[ManifestItem]] = defaultdict(list)
    for it in manifest.items:
        if not it.singer:
            out["unassigned"].append(it)
            continue
        b = _bucket(it.singer, seed)
        out[next(name for edge, name in edges if b <= edge)].append(it)
    return dict(out)


def pairs(manifest: Manifest) -> list[tuple[ManifestItem, ManifestItem]]:
    """(off, on) item pairs by ``meta.pair_id`` / ``meta.pair_role``."""
    groups: dict[str, dict[str, ManifestItem]] = defaultdict(dict)
    for it in manifest.items:
        pid, role = it.meta.get("pair_id"), it.meta.get("pair_role")
        if pid and role in ("off", "on"):
            groups[pid][role] = it
    return [(g["off"], g["on"]) for g in groups.values() if "off" in g and "on" in g]

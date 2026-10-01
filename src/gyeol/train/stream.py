"""Streaming training data from a prepared cache (revision B4).

:class:`StreamingDataset` is a :class:`torch.utils.data.IterableDataset` that
reads ``items/<id>.npz`` files written by :func:`gyeol.data.prepare.prepare`
one at a time from disk (nothing is held in memory beyond the current
batch) and yields **collated batches** of random-length crops.

Determinism and exact resume
    The item order of epoch *e* is a permutation drawn from
    ``default_rng([seed, e])``; batch *b* of that order uses its own
    ``default_rng([seed, e, b])`` for the crop length (shared by the batch, so
    no padding is wasted) and the crop offsets.  Nothing depends on global RNG
    state or on how many workers there are: worker *w* of *W* produces the
    batches ``b ≡ w (mod W)``, and the DataLoader's in-order round robin
    restores the global order.  Resuming at ``data_position`` (items already
    consumed in the epoch, a multiple of the batch size) therefore replays
    exactly the batches an uninterrupted run would have seen.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..data.prepare import CACHED_CURVES


@dataclass
class CropSpec:
    min_frames: int = 32
    max_frames: int = 96

    def __post_init__(self) -> None:
        if not 2 <= self.min_frames <= self.max_frames:
            raise ValueError(f"invalid crop range [{self.min_frames}, {self.max_frames}]")


def split_by_singer(rows: list[dict], fractions: dict[str, float], seed: int = 0) -> dict[str, list[dict]]:
    """Singer-disjoint split of index rows (seeded hash of the singer id; rows without a singer go to train)."""
    total = sum(fractions.values())
    edges, acc = [], 0.0
    for name, f in fractions.items():
        acc += f / total
        edges.append((acc, name))
    out: dict[str, list[dict]] = {name: [] for name in fractions}
    singers = sorted({r.get("singer") or "" for r in rows} - {""})
    # make sure every non-empty split gets at least one singer when there are enough singers
    assign: dict[str, str] = {}
    for s in singers:
        b = int.from_bytes(hashlib.sha256(f"{seed}:{s}".encode()).digest()[:8], "big") / 2**64
        assign[s] = next(name for edge, name in edges if b <= edge)
    names = [n for n, f in fractions.items() if f > 0]
    if len(singers) >= len(names):
        ordered = sorted(singers, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
        for name in names:
            if name not in assign.values():
                donor = max(names, key=lambda n: sum(v == n for v in assign.values()))
                s = next(x for x in ordered if assign[x] == donor)
                assign[s] = name
    for r in rows:
        out[assign.get(r.get("singer") or "", names[0])].append(r)
    return out


def _load(cache: Path, row: dict) -> dict[str, np.ndarray]:
    with np.load(cache / "items" / f"{row['id']}.npz") as z:
        return {k: z[k] for k in z.files if k != "quality"}


class StreamingDataset(IterableDataset):
    """Batches of random-length crops streamed from a prepared cache (see module docstring)."""

    def __init__(self, cache: str | Path, rows: list[dict], batch_size: int, crop: CropSpec, *, seed: int = 0, epoch: int = 0,
                 start_position: int = 0, shuffle: bool = True, drop_last: bool = False, features: tuple[str, ...] = ()):
        if start_position % batch_size:
            raise ValueError("start_position must be a multiple of the batch size")
        self.cache, self.rows, self.batch_size, self.crop = Path(cache), list(rows), int(batch_size), crop
        self.seed, self.epoch, self.start_position, self.shuffle, self.drop_last = seed, epoch, start_position, shuffle, drop_last
        self.features = tuple(features)

    def order(self) -> np.ndarray:
        n = len(self.rows)
        return np.random.default_rng([self.seed, self.epoch]).permutation(n) if self.shuffle else np.arange(n)

    def n_batches(self) -> int:
        n = len(self.rows)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __len__(self) -> int:
        return max(0, self.n_batches() - self.start_position // self.batch_size)

    def batch(self, b: int, order: np.ndarray | None = None) -> dict:
        order = self.order() if order is None else order
        idx = order[b * self.batch_size : (b + 1) * self.batch_size]
        rng = np.random.default_rng([self.seed, self.epoch, b])
        items = [_load(self.cache, self.rows[i]) for i in idx]
        lengths = [int(it["n_frames"]) for it in items]
        L = int(rng.integers(self.crop.min_frames, self.crop.max_frames + 1))
        L = max(2, min(L, max(lengths)))
        starts = [int(rng.integers(0, max(1, n - L + 1))) for n in lengths]
        hop = int(items[0]["hop"])
        B = len(items)
        out: dict = {"ids": [self.rows[i]["id"] for i in idx], "singer": [self.rows[i].get("singer", "") for i in idx],
                     "labels": [self.rows[i].get("labels", {}) for i in idx], "frames": L, "hop": hop, "sr": int(items[0]["sr"]),
                     "position_end": (b + 1) * self.batch_size, "batch_index": b, "epoch": self.epoch}
        mask = np.zeros((B, L), bool)
        N = (L - 1) * hop
        audio = np.zeros((B, N), np.float32)
        curves = {n: np.full((B, L), np.nan, np.float32) for n in CACHED_CURVES}
        confs = {n: np.zeros((B, L), np.float32) for n in CACHED_CURVES}
        ap = np.zeros((B, L, items[0]["ap_bands"].shape[1]), np.float32)
        feats = {f: np.zeros((B, L, items[0][f"feat/{f}"].shape[1]), np.float32) for f in self.features}
        f0_exact = np.full((B, L), np.nan, np.float32)  # Hz, 0 = exactly unvoiced, NaN = no exact truth (revision D3)
        for i, (it, s) in enumerate(zip(items, starts)):
            n = min(L, lengths[i] - s)
            mask[i, :n] = True
            seg = it["audio"][s * hop : s * hop + N]
            audio[i, : len(seg)] = seg
            for name in CACHED_CURVES:
                curves[name][i, :n] = it[f"curve/{name}"][s : s + n]
                confs[name][i, :n] = it[f"conf/{name}"][s : s + n]
            ap[i, :n] = it["ap_bands"][s : s + n]
            if "f0_exact" in it:
                f0_exact[i, :n] = it["f0_exact"][s : s + n]
            for f in self.features:
                feats[f][i, :n] = it[f"feat/{f}"][s : s + n]
        out["mask"] = torch.from_numpy(mask)
        out["audio"] = torch.from_numpy(audio)
        out["curves"] = {k: torch.from_numpy(v) for k, v in curves.items()}
        out["conf"] = {k: torch.from_numpy(v) for k, v in confs.items()}
        out["ap_bands"] = torch.from_numpy(ap)
        out["f0_exact"] = torch.from_numpy(f0_exact)
        out["feats"] = {k: torch.from_numpy(v) for k, v in feats.items()}
        return out

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        w, W = (0, 1) if info is None else (info.id, info.num_workers)
        order = self.order()
        first = self.start_position // self.batch_size
        for b in range(first, self.n_batches()):
            if (b - first) % W == w:
                yield self.batch(b, order)


def loader(ds: StreamingDataset, num_workers: int = 0) -> torch.utils.data.DataLoader:
    """A DataLoader over pre-collated batches (``batch_size=None``), in order.

    The loader gets its own generator: creating an iterator would otherwise
    draw a seed from the *global* torch RNG, and a resumed run (which creates
    its iterator mid-epoch) would then see different dropout masks.
    """
    g = torch.Generator()
    g.manual_seed(int(np.random.default_rng([ds.seed, ds.epoch, 99]).integers(1 << 62)))
    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=num_workers, persistent_workers=False, generator=g)


__all__ = ["CropSpec", "StreamingDataset", "loader", "split_by_singer"]

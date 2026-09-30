"""Consent-gated renderers: user take + edit → audio in the user's own voice.

Every renderer checks, before doing anything:

1. ``voice`` is a :class:`~gyeol.core.consent.ConsentedVoice`
   (:func:`~gyeol.core.consent.require_consented_voice`);
2. the take is the user's own: the recording's provenance is ``USER``, its
   owner is the voice's user, and the representation was analysed from that
   recording (:class:`UserTake`).

A reference / target recording can therefore never be rendered, whatever the
edit: there is no code path that takes the target's audio or singer vector.

* :class:`DSPRenderer` — harmonic-plus-noise resynthesis of the user's own
  recording (:mod:`gyeol.demo.hnm`); needs no weights.  Supports f0, gain,
  aperiodicity and timing edits.
* :class:`NeuralRenderer` — the M4 decoder with the **consented** singer
  vector and a neutral env preset.  Supports the same edits through the
  decoder's c(t) channels.  Edits of learned posteriors (register,
  phonation, diction) need a decoder conditioned on those curves; none is
  trained yet, so they report UNAVAILABLE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.consent import ConsentedVoice, ConsentError, Provenance, require_consented_voice
from ..core.containers import Recording, Representation
from ..core.status import Result
from .edits import Edit
from .hnm import HNMParams, analyze_hnm, synthesize_hnm


@dataclass(frozen=True)
class UserTake:
    """A user's own recording and its representation."""

    recording: Recording
    rep: Representation

    def __post_init__(self) -> None:
        if self.recording.provenance is not Provenance.USER or self.rep.provenance is not Provenance.USER:
            raise ConsentError("only the user's own recordings can be rendered (provenance must be USER)")
        if self.rep.recording_id != self.recording.recording_id:
            raise ValueError("representation was not analysed from this recording")
        if self.rep.grid.sr != self.recording.sr:
            raise ValueError("representation and recording sample rates differ")


def check_take(voice: object, take: UserTake) -> ConsentedVoice:
    v = require_consented_voice(voice)
    if not isinstance(take, UserTake):
        raise TypeError(f"expected a UserTake, got {type(take).__name__}")
    if take.recording.owner_id != v.user_id:
        raise ConsentError("the take belongs to a different user than the consented voice")
    return v


@runtime_checkable
class Renderer(Protocol):
    name: str
    capabilities: frozenset[str]

    def render(self, voice: ConsentedVoice, take: UserTake, edit: Edit | None = None, seed: int = 0,
               frames: tuple[int, int] | None = None) -> Result[np.ndarray]:
        """Audio of the whole take (length of the recording), or of output frames
        ``frames=(a, b)`` only (``(b − a − 1)·hop + 1`` samples from ``a·hop``)."""
        ...


def _unsupported(renderer: "Renderer", edit: Edit | None) -> str | None:
    if edit is None:
        return None
    missing = sorted(edit.requires - renderer.capabilities)
    return f"{renderer.name} cannot render edits of {missing}" if missing else None


class DSPRenderer:
    """Harmonic-plus-noise resynthesis of the user's own recording."""

    name = "dsp-hnm"
    capabilities = frozenset({"f0", "gain", "aperiodic", "timing"})

    def __init__(self, win: int = 2048):
        self.win = win
        self._cache: dict[str, HNMParams] = {}

    def params(self, take: UserTake) -> HNMParams:
        key = take.recording.recording_id
        if key not in self._cache:
            f0 = 440.0 * 2 ** (take.rep.curves["f0_cents"].values / 1200.0)
            self._cache = {key: analyze_hnm(take.recording.audio, take.rep.grid, f0, self.win)}
        return self._cache[key]

    def render(self, voice: ConsentedVoice, take: UserTake, edit: Edit | None = None, seed: int = 0,
               frames: tuple[int, int] | None = None) -> Result[np.ndarray]:
        check_take(voice, take)
        why = _unsupported(self, edit)
        if why:
            return Result.unavailable(why)
        e = edit or Edit.identity(take.rep.grid.n_frames)
        if e.n_frames != take.rep.grid.n_frames:
            return Result.failure("edit and take are on different grids")
        y = synthesize_hnm(self.params(take), f0_cents_delta=e.f0_cents, gain_db=e.gain_db, aperiodic_db=e.aperiodic_db,
                           time_map=e.time_map, seed=seed, frames=frames)
        if frames is None:
            n = len(take.recording.audio)
            y = np.pad(y, (0, max(0, n - len(y))))[:n]
        if not np.all(np.isfinite(y)):
            return Result.failure("resynthesis produced non-finite samples")
        return Result.success(y)


class NeuralRenderer:
    """The M4 decoder, decoding with the consented singer vector and a neutral env."""

    name = "neural-m4"

    def __init__(self, model, neutral_env: np.ndarray | None = None):
        from ..decoder.model import C_CHANNELS

        self.model = model
        self.neutral_env = neutral_env
        self.channels = C_CHANNELS
        self.capabilities = frozenset({"f0", "gain", "aperiodic", "timing"})

    def render(self, voice: ConsentedVoice, take: UserTake, edit: Edit | None = None, seed: int = 0,
               frames: tuple[int, int] | None = None) -> Result[np.ndarray]:
        import torch

        from ..train.autoencoder import _SCALE, batch_from_representations

        v = check_take(voice, take)
        why = _unsupported(self, edit)
        if why:
            return Result.unavailable(why)
        cfg = self.model.cfg
        if take.recording.sr != cfg.sr or take.rep.grid.hop != cfg.hop:
            return Result.failure(f"model expects sr={cfg.sr}, hop={cfg.hop}")
        sv = np.asarray(v.singer.vector, np.float32).reshape(-1)
        if sv.shape[0] != cfg.singer_dim:
            return Result.failure(f"singer vector has {sv.shape[0]} dims, the decoder expects {cfg.singer_dim}")
        b = batch_from_representations([take.rep], [take.recording.audio])
        T = take.rep.grid.n_frames
        e = edit or Edit.identity(T)
        c, f0, ap = b.c.clone(), b.f0_hz.clone(), b.aperiodic.clone()
        ch = {n: i for i, n in enumerate(self.channels)}
        tt = lambda a: torch.tensor(np.nan_to_num(a), dtype=torch.float32)  # noqa: E731
        if e.f0_cents is not None:
            f0[0, :T] = f0[0, :T] * 2 ** (tt(e.f0_cents) / 1200)
            c[0, :T, ch["pitch_center"]] += tt(e.f0_cents) * _SCALE["pitch_center"]
        if e.gain_db is not None:
            c[0, :T, ch["loudness_rel"]] += tt(e.gain_db) * _SCALE["loudness_rel"]
        if e.aperiodic_db is not None:
            c[0, :T, ch["aperiodic_ratio"]] += tt(e.aperiodic_db) * _SCALE["aperiodic_ratio"]
            ap[0, :T] += tt(e.aperiodic_db) * 0.1
        self.model.eval()
        with torch.no_grad():
            enc = self.model.encode(b.wav, b.c, b.c_mask)
            Tm = enc["r"].shape[1]
            idx = np.arange(Tm) if e.time_map is None else np.clip(np.round(e.time_map[:Tm]).astype(int), 0, Tm - 1)
            ti = torch.tensor(idx, dtype=torch.long)
            env = torch.zeros(1, cfg.env_dim) if self.neutral_env is None else torch.tensor(self.neutral_env, dtype=torch.float32).reshape(1, -1)
            out = self.model.decode(torch.tensor(sv).reshape(1, -1), env, c[:, :Tm][:, ti], b.c_mask[:, :Tm][:, ti],
                                    enc["r"][:, ti], f0[:, :Tm][:, ti], ap[:, :Tm][:, ti], b.rough[:, :Tm][:, ti] if b.rough is not None else None,
                                    seed=seed)
        y = out["wav"][0].numpy().astype(float)
        n = len(take.recording.audio)
        y = np.pad(y, (0, max(0, n - len(y))))[:n]
        if frames is not None:
            hop = take.rep.grid.hop
            y = y[frames[0] * hop : (frames[1] - 1) * hop + 1]
        if not np.all(np.isfinite(y)):
            return Result.failure("decoder produced non-finite samples")
        return Result.success(y)

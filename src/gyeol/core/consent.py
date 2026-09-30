"""Provenance and consent types that gate voice rendering.

Rendering a voice requires a :class:`ConsentedVoice`.  It can only be built by
:meth:`ConsentedVoice.create` from

* a :class:`SingerVector` whose provenance is ``USER``,
* the user's own :class:`~gyeol.core.containers.Recording` it was derived from,
* a :class:`ConsentToken` of the same user that includes ``voice_synthesis``.

A vector derived from a reference / target track has provenance
``REFERENCE`` and can never become a ``ConsentedVoice``: gyeol has no
target-singer synthesis path.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Provenance(str, Enum):
    USER = "user"  # the app user's own voice, recorded in-app
    REFERENCE = "reference"  # target / reference track (another person's voice)
    SYNTHETIC = "synthetic"  # generated test signal, no person
    UNKNOWN = "unknown"


class Purpose(str, Enum):
    ANALYSIS = "analysis"
    TRAINING = "training"
    STORAGE = "storage"
    #: rendering demos in the user's own voice (added beyond the three purposes
    #: of the brief so synthesis is never implied by analysis consent)
    VOICE_SYNTHESIS = "voice_synthesis"


class ConsentError(PermissionError):
    pass


@dataclass(frozen=True)
class ConsentToken:
    user_id: str
    purposes: frozenset[Purpose]
    token_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    issued_at: float = field(default_factory=time.time)
    revoked: bool = False

    def allows(self, purpose: Purpose) -> bool:
        return not self.revoked and Purpose(purpose) in self.purposes


@dataclass(frozen=True, eq=False)
class SingerVector:
    """Utterance-level singer embedding with its provenance."""

    vector: np.ndarray
    provenance: Provenance
    source_recording_id: str
    owner_id: str | None = None  # user id when provenance is USER

    def __post_init__(self) -> None:
        v = np.asarray(self.vector, dtype=np.float32)
        v.setflags(write=False)
        object.__setattr__(self, "vector", v)
        if self.provenance is Provenance.USER and not self.owner_id:
            raise ValueError("a USER singer vector needs owner_id")


_KEY = object()


class ConsentedVoice:
    """A user's own voice, cleared for synthesis.  Build with :meth:`create`."""

    __slots__ = ("singer", "user_id", "token_id", "recording_id")

    def __init__(self, singer: SingerVector, user_id: str, token_id: str, recording_id: str, *, _key: object = None):
        if _key is not _KEY:
            raise TypeError("ConsentedVoice cannot be constructed directly; use ConsentedVoice.create()")
        self.singer = singer
        self.user_id = user_id
        self.token_id = token_id
        self.recording_id = recording_id

    @classmethod
    def create(cls, singer: SingerVector, recording: "Recording", token: ConsentToken) -> "ConsentedVoice":  # noqa: F821
        from .containers import Recording

        if not isinstance(singer, SingerVector) or not isinstance(recording, Recording):
            raise TypeError("create() needs a SingerVector and a Recording")
        if singer.provenance is not Provenance.USER:
            raise ConsentError(f"singer vector provenance is {singer.provenance.value!r}; only the user's own voice may be rendered")
        if recording.provenance is not Provenance.USER:
            raise ConsentError("the source recording is not the user's own recording")
        if singer.source_recording_id != recording.recording_id:
            raise ConsentError("singer vector was not derived from the given recording")
        if not (singer.owner_id == recording.owner_id == token.user_id):
            raise ConsentError("user id mismatch between singer vector, recording and consent token")
        if not token.allows(Purpose.VOICE_SYNTHESIS):
            raise ConsentError("consent token does not include voice_synthesis (or was revoked)")
        return cls(singer, token.user_id, token.token_id, recording.recording_id, _key=_KEY)

    def __setattr__(self, name, value):  # immutable after construction
        if hasattr(self, "recording_id"):
            raise AttributeError("ConsentedVoice is immutable")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return f"ConsentedVoice(user_id={self.user_id!r}, recording_id={self.recording_id!r})"


def require_consented_voice(voice: object) -> ConsentedVoice:
    """Runtime guard used by every renderer."""
    if not isinstance(voice, ConsentedVoice):
        raise ConsentError(f"rendering requires a ConsentedVoice, got {type(voice).__name__}")
    if voice.singer.provenance is not Provenance.USER:  # defence in depth
        raise ConsentError("ConsentedVoice with non-user provenance")
    return voice

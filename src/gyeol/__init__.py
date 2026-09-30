"""gyeol (결) — a condition-aware intermediate representation of the singing voice.

Audio → a hybrid representation:

1. an interpretable, frame-level source–filter core (pitch, source, filter,
   energy groups),
2. explicit Korean phonetic-context tokens,
3. an optional channel-adversarial residual embedding z,
4. a mandatory per-dimension validity mask driven by a separate nuisance
   side channel (SNR, bandwidth/codec, clipping, AGC, reverberation,
   separation quality).

Quick start::

    import gyeol
    rep = gyeol.analyze("take.wav", lyrics="사랑해")
    rep.summary()
"""

__version__ = "0.1.0"

from .engine import Engine, EngineConfig, analyze  # noqa: E402
from .representation import ContextTokens, Note, NuisanceReport, Track, VocalRepresentation  # noqa: E402
from .spec import DIMENSIONS, NOTE_DIMENSIONS, OMISSIONS  # noqa: E402
from .validity import ValidityPolicy  # noqa: E402

__all__ = [
    "__version__",
    "Engine",
    "EngineConfig",
    "analyze",
    "VocalRepresentation",
    "Track",
    "Note",
    "NuisanceReport",
    "ContextTokens",
    "ValidityPolicy",
    "DIMENSIONS",
    "NOTE_DIMENSIONS",
    "OMISSIONS",
]

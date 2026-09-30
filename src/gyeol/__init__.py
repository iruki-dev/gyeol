"""gyeol (결) v2 — an interpretable singing-voice model for vocal coaching.

A sung phrase is represented by three layers:

* global: ``singer`` and ``env`` vectors (M4),
* interpretable: attribute curves ``c(t)`` with per-frame confidence,
* residual: a low-dimensional ``r(t)`` for reconstruction only (M4).

Coaching explains a user's take as the target phrase transformed by a time
warp ``τ(t)`` and attribute differences ``Δc(t)``; whatever is left over is
reported as "cannot judge".  Demos are rendered only in the user's own,
consented voice.

See ``docs/milestones`` for what each milestone delivers.
"""

__version__ = "2.0.0.dev0"

from .core import (  # noqa: E402
    AttributeCurve,
    AttributeCurves,
    ConsentedVoice,
    Explanation,
    FrameGrid,
    LicenseTag,
    Profile,
    Provenance,
    Recording,
    Representation,
    Result,
    Status,
)

__all__ = [
    "__version__",
    "AttributeCurve",
    "AttributeCurves",
    "ConsentedVoice",
    "Explanation",
    "FrameGrid",
    "LicenseTag",
    "Profile",
    "Provenance",
    "Recording",
    "Representation",
    "Result",
    "Status",
]

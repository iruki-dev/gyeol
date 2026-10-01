"""gyeol (결) v2 — an interpretable singing-voice model for vocal coaching.

A sung phrase is represented by three layers:

* global: ``singer`` and ``env`` vectors (M4),
* interpretable: attribute curves ``c(t)`` with per-frame confidence,
* residual: a low-dimensional ``r(t)`` for reconstruction only (M4).

Coaching explains a user's take as the target phrase transformed by a time
warp ``τ(t)`` and attribute differences ``Δc(t)``; whatever is left over is
reported as "cannot judge".  Demos re-render a recording with chosen edits.

Third-party models and datasets are listed in :mod:`gyeol.core.assets`;
complying with their licenses is up to the user (see the README).

See ``docs/milestones`` for what each milestone delivers.
"""

__version__ = "2.0.0.dev0"

from .core import (  # noqa: E402
    AttributeCurve,
    AttributeCurves,
    Explanation,
    FrameGrid,
    Recording,
    Representation,
    Result,
    Status,
)

__all__ = ["__version__", "AttributeCurve", "AttributeCurves", "Explanation", "FrameGrid", "Recording", "Representation", "Result", "Status"]

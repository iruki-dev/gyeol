"""Residual embedding z: interface.

z carries only what the interpretable core misses (idiosyncratic timbre,
수리성 texture, ornament idiosyncrasies).  It is produced by a trained encoder;
gyeol ships the architecture and losses (:mod:`gyeol.residual.torch_model`)
but no trained weights, because training needs the multi-device corpus of
research §5.ii.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..representation import Track

#: core tracks (100 Hz, 1-D) fed to the residual encoder / decoder as conditioning
CORE_INPUTS: tuple[str, ...] = (
    "voicing_prob", "f0_cents", "energy_rel_db", "cpps", "h1h2c", "h2h4c", "h1a1c", "h1a3c", "naq", "shr",
)


@runtime_checkable
class ResidualEncoder(Protocol):
    def encode(self, audio: np.ndarray, sr: int, core: dict[str, Track]) -> Track:
        """Return the residual track (T_z, D) at its own frame rate."""
        ...


def core_matrix(core: dict[str, Track], names: tuple[str, ...] = CORE_INPUTS) -> np.ndarray:
    """Stack core tracks into (T, 2·len(names)): values (0 where invalid) + validity flags.

    Validity flags are part of the input so the encoder can learn to put
    nothing into z that the core already carries when the core is valid.
    """
    cols = []
    n = max(len(core[k].values) for k in names if k in core)
    for k in names:
        if k not in core:
            cols += [np.zeros(n), np.zeros(n)]
            continue
        tr = core[k]
        v = np.where(tr.valid, np.nan_to_num(tr.values), 0.0)[:n]
        cols += [v, tr.valid[:n].astype(float)]
    return np.stack(cols, axis=1)

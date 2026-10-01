"""Audio loading and saving.

Loading never modifies samples beyond channel down-mixing: quality checks
such as clipping must see the *raw* signal (v0.1 defect 4 measured clipping
after a high-pass filter).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.containers import Recording
from ..core.status import Result
from ..dsp.base import resample as _resample
from ..dsp.base import to_mono


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling (exact rational ratio)."""
    return _resample(np.asarray(x, dtype=float), int(sr_in), int(sr_out))


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a file as mono float64 without any processing."""
    import soundfile as sf

    x, sr = sf.read(str(path), always_2d=False, dtype="float64")
    return to_mono(x), int(sr)


def save_audio(path: str | Path, x: np.ndarray, sr: int, *, metadata: dict[str, str] | None = None) -> None:
    """Write float audio; ``metadata`` is stored as file tags where the format supports it."""
    import soundfile as sf

    with sf.SoundFile(str(path), "w", samplerate=int(sr), channels=1, subtype="FLOAT" if str(path).endswith(".wav") else None) as f:
        for k, v in (metadata or {}).items():
            try:
                setattr(f, k, v)
            except (AttributeError, RuntimeError):
                pass  # tag not supported by this container
        f.write(np.asarray(x, dtype=np.float32))


def load_recording(path: str | Path) -> Result[Recording]:
    """Load a file into a :class:`Recording`, with an explicit status."""
    try:
        x, sr = load_audio(path)
    except Exception as exc:  # noqa: BLE001 - report any decoder failure as a status
        return Result.failure(f"cannot read {path}: {exc}")
    if len(x) == 0:
        return Result.failure(f"{path} is empty")
    if not np.all(np.isfinite(x)):
        return Result.failure(f"{path} contains NaN/inf samples")
    return Result.success(Recording(x, sr, meta={"path": str(path)}))

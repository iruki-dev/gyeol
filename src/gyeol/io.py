"""Audio loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._dsp import to_mono


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a file as mono float64.  Uses ``soundfile`` when available
    (WAV/FLAC/OGG/MP3 via libsndfile), otherwise ``scipy.io.wavfile``."""
    path = Path(path)
    try:
        import soundfile as sf

        x, sr = sf.read(str(path), always_2d=False, dtype="float64")
        return to_mono(x), int(sr)
    except ImportError:
        from scipy.io import wavfile

        sr, x = wavfile.read(str(path))
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float64) / np.iinfo(x.dtype).max
        return to_mono(x.astype(np.float64)), int(sr)


def save_audio(path: str | Path, x: np.ndarray, sr: int) -> None:
    try:
        import soundfile as sf

        sf.write(str(path), np.asarray(x, dtype=np.float32), sr)
    except ImportError:
        from scipy.io import wavfile

        wavfile.write(str(path), sr, np.asarray(x, dtype=np.float32))

"""AI-generated labelling of rendered audio (Korean AI Basic Act).

Every waveform produced by :mod:`gyeol.demo` is a :class:`LabelledAudio`:
audio that has passed through the watermark hook, plus metadata that says it
is AI-generated, which renderer produced it, from which consent, and with
which edits.  :func:`save_labelled` writes the metadata into the WAV's INFO
tags (``title``, ``software``, ``comment`` = JSON) **and** a JSON sidecar
(``<name>.ai.json``) so the label survives tools that drop INFO chunks.

The user id is never written in clear: only a SHA-256 digest (PIPA).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.consent import ConsentedVoice
from .watermark import WatermarkHook


def _notice() -> dict:
    from importlib import resources

    s = json.loads(resources.files("gyeol").joinpath("resources/ko/demo.json").read_text(encoding="utf-8"))
    return {"ko": s["ai_notice"], "en": "AI-generated audio (gyeol demo rendered in the user's own consented voice)"}


@dataclass(frozen=True)
class LabelledAudio:
    audio: np.ndarray
    sr: int
    metadata: dict = field(default_factory=dict)


def label_ai_generated(audio: np.ndarray, sr: int, *, voice: ConsentedVoice, renderer: str, description: dict,
                       watermark: WatermarkHook, profile: str = "commercial") -> LabelledAudio:
    from .. import __version__

    marked = watermark.embed(np.asarray(audio, float), sr)
    meta = {
        "ai_generated": True,
        "notice": _notice(),
        "generator": f"gyeol {__version__}",
        "renderer": renderer,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "consent_token_id": voice.token_id,
        "voice_owner_sha256": hashlib.sha256(voice.user_id.encode()).hexdigest(),
        "source_recording_id": voice.recording_id,
        "watermark": watermark.describe(),
        "edits": description,
        "profile": profile,  # revision C3: license profile the demo was produced under
    }
    return LabelledAudio(marked, sr, meta)


def save_labelled(path: str | Path, la: LabelledAudio, subtype: str = "PCM_16") -> Path:
    import soundfile as sf

    path = Path(path)
    peak = float(np.max(np.abs(la.audio))) if len(la.audio) else 0.0
    audio = la.audio / peak * 0.98 if peak > 0.98 else la.audio
    with sf.SoundFile(str(path), "w", la.sr, 1, subtype) as f:
        f.title = "AI-generated (gyeol demo)"
        f.software = la.metadata.get("generator", "gyeol")
        f.comment = json.dumps(la.metadata, ensure_ascii=True, default=str)
        f.write(audio)
    path.with_suffix(".ai.json").write_text(json.dumps(la.metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def read_label(path: str | Path) -> dict | None:
    """The AI label of a saved file (INFO comment first, then the sidecar); None if unlabelled."""
    import soundfile as sf

    path = Path(path)
    try:
        with sf.SoundFile(str(path)) as f:
            meta = json.loads(f.comment) if f.comment else None
        if isinstance(meta, dict) and meta.get("ai_generated"):
            return meta
    except (RuntimeError, ValueError):
        pass
    side = path.with_suffix(".ai.json")
    if side.exists():
        meta = json.loads(side.read_text(encoding="utf-8"))
        return meta if meta.get("ai_generated") else None
    return None

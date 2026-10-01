"""gyeol's public API (revision C2).

Four entry points cover what an application needs; everything else in the
package is implementation detail that may change between versions::

    from gyeol import api

    target = api.analyze(guide_audio, sr, role="reference", lyrics="사랑해요").unwrap()
    take = api.analyze(take_audio, sr, owner_id="user-42", reference=(guide_audio, sr)).unwrap()
    exp = api.compare([take], target).unwrap()
    api.to_json(exp, "explanation.json")                       # versioned JSON (gyeol.explanation v1)

    own = api.OwnVoice(take_audio, sr, take, consent_token)      # the user's own take + their consent
    demo = api.render_demo(own, exp, target, out_dir="demo/").unwrap()

    run = api.train("configs/cpu-smoke/heads.yaml")

* :func:`analyze` — one recording → :class:`~gyeol.core.Representation`
  (separation by default, optional latency refinement against the guide);
* :func:`compare` — user take(s) vs a target → :class:`~gyeol.core.Explanation`
  (optionally with audibility, which renders the user's *own* consented voice);
* :func:`render_demo` — a stepwise own-voice demo of one item, AI-labelled and
  watermarked, with ``gyeol.demo`` JSON metadata;
* :func:`train` — a training run from a YAML config (revision B).

Results that can fail on bad audio come back as :class:`~gyeol.core.Result`
with an explicit status.  User state (consent records, storage, coaching
sessions) is not part of the library; see ``reference_service/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .core.consent import ConsentedVoice, ConsentError, ConsentToken, Provenance, Purpose
from .core.containers import Explanation, ExplanationItem, Recording, Representation
from .core.license import Profile
from .core.status import Result, Status
from .explain.render_text import explanation_notes, item_text
from .explain.render_text import load_strings as _explain_strings
from .io import load_audio, save_audio
from .schema import from_dict, from_json, json_schema, to_dict, to_json
from .synth import SynthNote, melody

ROLES = {"user": Provenance.USER, "reference": Provenance.REFERENCE, "synthetic": Provenance.SYNTHETIC}


def _trackers(dsp_only: bool, profile: Profile):
    from .pitch.adapters import PyinTracker, SHSTracker, YinTracker, default_trackers

    return [PyinTracker(), YinTracker(), SHSTracker()] if dsp_only else default_trackers(profile)


def _audio(audio, sr):
    if isinstance(audio, (str, Path)):
        x, file_sr = load_audio(audio)
        return x, file_sr
    if sr is None:
        raise ValueError("sr is required when audio is an array")
    return np.asarray(audio, float), int(sr)


def analyze(audio: np.ndarray | str | Path, sr: int | None = None, *, role: str = "user", owner_id: str | None = None,
            reference: tuple[np.ndarray, int] | str | Path | None = None, backing: np.ndarray | None = None,
            separation: str = "auto", profile: Profile | str = Profile.COMMERCIAL, dsp_only: bool = False,
            lyrics: str | None = None, recording_id: str | None = None) -> Result[Representation]:
    """Analyse one recording.

    ``role``: ``"user"`` (the app user's own take; ``owner_id`` required),
    ``"reference"`` (a target / guide vocal — another person's voice) or
    ``"synthetic"``.  ``reference``: the guide vocal (array + sr, or a path)
    the user sang along with; the take is shifted by the refined latency and
    the estimate is kept in ``rep.meta["latency"]`` (the shared-clock premise
    reads its confidence).  ``lyrics`` maps syllables onto the notes of a
    reference.  ``separation``: ``auto`` | ``always`` | ``off`` (revision A1).
    """
    from .attributes.extract import AnalysisConfig
    from .attributes.extract import analyze as _analyze
    from .context import assign_syllables
    from .io import refine_offset, shift

    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    prof = Profile(profile)
    try:
        x, sr = _audio(audio, sr)
    except (OSError, RuntimeError) as exc:
        return Result.failure(f"cannot read audio: {exc}")
    prov = ROLES[role]
    if prov is Provenance.USER and not owner_id:
        raise ValueError("a user take needs owner_id")
    latency = None
    if reference is not None:
        gx, gsr = _audio(*reference) if isinstance(reference, tuple) else _audio(reference, None)
        if gsr != sr:
            from .dsp.base import resample

            gx = resample(gx, gsr, sr)
        off = refine_offset(x, gx, sr)
        if off.usable:
            x = shift(x, off.value.latency_s, sr)
            latency = {"offset_s": off.value.latency_s, "confidence": off.value.confidence, "method": off.value.method,
                       "status": off.status.value}
        else:
            latency = {"offset_s": 0.0, "confidence": 0.0, "method": "onset-xcorr", "status": off.status.value, "reason": off.reason}
    kw = {"recording_id": recording_id} if recording_id else {}
    rec = Recording(x, sr, prov, owner_id=owner_id if prov is Provenance.USER else None, **kw)
    r = _analyze(rec, trackers=_trackers(dsp_only, prof), backing=backing, separation=separation, config=AnalysisConfig(profile=prof))
    if r.usable:
        if latency is not None:
            r.value.meta["latency"] = latency
        if lyrics:
            r.value.meta["syllables"] = assign_syllables(r.value.meta.get("notes", []), lyrics)
    return r


@dataclass
class OwnVoice:
    """The app user's own take (as passed to :func:`analyze`) plus their consent token.

    Rendering (audibility, demos) is only ever done in this voice: the token
    must include ``voice_synthesis`` and belong to the take's owner, else
    :class:`~gyeol.core.ConsentError` is raised before anything is rendered.
    """

    audio: np.ndarray
    sr: int
    rep: Representation
    consent: ConsentToken
    _cache: dict = field(default_factory=dict, repr=False)

    def take(self):
        from .demo import UserTake
        from .io import shift

        if "take" not in self._cache:
            import hashlib

            digest = self.rep.meta.get("owner_sha256")
            if self.rep.provenance is not Provenance.USER or digest is None:
                raise ConsentError("only the user's own analysed take (role='user') can be rendered")
            if hashlib.sha256(self.consent.user_id.encode()).hexdigest() != digest:
                raise ConsentError("the consent token belongs to a different user than this take")
            x = np.asarray(self.audio, float)
            lat = self.rep.meta.get("latency") or {}
            if lat.get("offset_s"):
                x = shift(x, float(lat["offset_s"]), self.sr)  # the same shift analyze() applied
            rec = Recording(x, self.sr, Provenance.USER, owner_id=self.consent.user_id, recording_id=self.rep.recording_id)
            self._cache["take"] = UserTake(rec, self.rep)
        return self._cache["take"]

    def voice(self) -> ConsentedVoice:
        from .encoders.latent import ltas_singer_vector

        if "voice" not in self._cache:
            t = self.take()
            self._cache["voice"] = ConsentedVoice.create(ltas_singer_vector(t.recording), t.recording, self.consent)
        return self._cache["voice"]


def compare(user_takes: Representation | Sequence[Representation], target: Representation, *, lyrics: str | None = None,
            audibility: OwnVoice | None = None, config=None) -> Result[Explanation]:
    """Explain the user's take(s) against ``target`` (habit vs error across takes, premises, withheld items).

    ``audibility``: the user's own last take and consent; when given, every
    item gets an audibility score by re-rendering that take with the item
    corrected (DSP renderer).
    """
    from .explain import explain, score_audibility

    takes = [user_takes] if isinstance(user_takes, Representation) else list(user_takes)
    ex = explain(takes, target, config, lyrics=lyrics)
    if not ex.usable or audibility is None:
        return ex
    from .demo import DSPRenderer

    aud = score_audibility(ex.value, audibility.voice(), audibility.take(), target, DSPRenderer())
    if not aud.ok:
        return Result(Status.UNRELIABLE, ex.value, f"audibility not scored: {aud.reason}", ex.warnings)
    return ex


@dataclass
class DemoResult:
    files: dict[str, Path]  # "baseline", "step_1", … → written WAV (each with an .ai.json sidecar)
    metadata: dict  # gyeol.demo document
    item: ExplanationItem


def _default_item(exp: Explanation) -> ExplanationItem | None:
    cands = [it for it in exp.items if it.category != "diction" and it.confidence >= 0.5]
    if not cands:
        return None
    return max(cands, key=lambda it: it.confidence * (it.audibility if it.audibility is not None else abs(it.magnitude)))


def render_demo(own: OwnVoice, explanation: Explanation, target: Representation, *, item: ExplanationItem | tuple | None = None,
                out_dir: str | Path | None = None, renderer=None) -> Result[DemoResult]:
    """Stepwise demo of one item in the user's own consented voice: the item alone, then 50 % / 100 % toward the target.

    Every waveform is watermarked and AI-labelled (WAV INFO tags + ``.ai.json``
    sidecar); ``out_dir`` receives the files plus ``demo.json`` (``gyeol.demo``).
    """
    from .demo import DSPRenderer, item_key, save_labelled
    from .demo import render_demo as _render

    if item is None:
        item = _default_item(explanation)
        if item is None:
            return Result.unavailable("no confident item to demonstrate")
    if isinstance(item, tuple):
        match = [it for it in explanation.items if item_key(it) == item]
        if not match:
            return Result.failure(f"item {item} is not in the explanation")
        item = match[0]
    d = _render(own.voice(), own.take(), explanation, target, item_key(item), renderer or DSPRenderer())
    if not d.ok:
        return Result(d.status, None, d.reason)
    files: dict[str, Path] = {}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files["baseline"] = save_labelled(out / "demo_0_baseline.wav", d.value.baseline)
        for i, st in enumerate(d.value.steps, 1):
            files[f"step_{i}"] = save_labelled(out / f"demo_{i}_{st.step.label}.wav", st.audio)
    meta = to_dict(d.value, item_key=item_key(item), files={k: str(v) for k, v in files.items()})
    if out_dir is not None:
        (Path(out_dir) / "demo.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return Result.success(DemoResult(files, meta, item))


def train(config, *, resume: bool = False, overrides: list[str] | None = None, task: str | None = None, out_dir: str | Path | None = None,
          log=print):
    """Run (or resume) a training run; ``config`` is a YAML path or a :class:`gyeol.train.config.TrainConfig`."""
    from .train.config import TrainConfig, load_config
    from .train.runner import train as _train

    cfg = config if isinstance(config, TrainConfig) else load_config(config, task, overrides)
    return _train(cfg, resume=resume, out_dir=out_dir, log=log)


def load_strings(lang: str = "ko", resource: str = "explain") -> dict:
    """User-facing strings from ``gyeol/resources/<lang>/<resource>.json`` (``explain`` or ``demo``)."""
    if resource == "explain":
        return _explain_strings(lang)
    from importlib import resources

    return json.loads(resources.files("gyeol").joinpath(f"resources/{lang}/{resource}.json").read_text(encoding="utf-8"))


def text(explanation: Explanation, lang: str = "ko") -> tuple[list[str], list[tuple[ExplanationItem, str]]]:
    """User-facing text from the resource files: (context notes, [(item, sentence)])."""
    return explanation_notes(explanation, lang), [(it, item_text(it, lang)) for it in explanation.items]


__all__ = ["ConsentError", "ConsentToken", "DemoResult", "OwnVoice", "Profile", "Purpose", "Result", "Status", "SynthNote", "analyze",
           "compare", "explanation_notes", "from_dict", "from_json", "item_text", "json_schema", "load_audio", "load_strings", "melody",
           "render_demo", "save_audio", "text", "to_dict", "to_json", "train"]

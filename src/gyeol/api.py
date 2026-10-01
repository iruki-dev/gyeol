"""gyeol's public API (revision C2).

Four entry points cover what an application needs; everything else in the
package is implementation detail that may change between versions::

    from gyeol import api

    target = api.analyze(guide_audio, sr, lyrics="사랑해요").unwrap()
    take = api.analyze(take_audio, sr, reference=(guide_audio, sr)).unwrap()
    exp = api.compare([take], target).unwrap()
    api.to_json(exp, "explanation.json")                       # versioned JSON (gyeol.explanation v1)

    demo = api.render_demo(take_audio, sr, take, exp, target, out_dir="demo/").unwrap()

    run = api.train("configs/cpu-smoke/heads.yaml")

* :func:`analyze` — one recording → :class:`~gyeol.core.Representation`
  (separation by default, optional latency refinement against the guide);
* :func:`compare` — user take(s) vs a target → :class:`~gyeol.core.Explanation`
  (optionally with audibility, which re-renders the last take with each item corrected);
* :func:`render_demo` — the take re-rendered with one item corrected, then step by
  step toward the target; plain audio plus ``gyeol.demo`` JSON metadata;
* :func:`train` — a training run from a YAML config (revision B).

Results that can fail on bad audio come back as :class:`~gyeol.core.Result`
with an explicit status.  Application policy — consent, labelling of rendered
audio, license compliance — is up to the application; user state (storage,
coaching sessions) lives in ``reference_service/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .core.containers import Explanation, ExplanationItem, Recording, Representation
from .core.status import Result, Status
from .explain.render_text import explanation_notes, item_text
from .explain.render_text import load_strings as _explain_strings
from .io import load_audio, save_audio
from .schema import from_dict, from_json, json_schema, to_dict, to_json
from .synth import SynthNote, melody

def _trackers(dsp_only: bool):
    from .pitch.adapters import PyinTracker, SHSTracker, YinTracker, default_trackers

    return [PyinTracker(), YinTracker(), SHSTracker()] if dsp_only else default_trackers()


def _audio(audio, sr):
    if isinstance(audio, (str, Path)):
        x, file_sr = load_audio(audio)
        return x, file_sr
    if sr is None:
        raise ValueError("sr is required when audio is an array")
    return np.asarray(audio, float), int(sr)


def analyze(audio: np.ndarray | str | Path, sr: int | None = None, *,
            reference: tuple[np.ndarray, int] | str | Path | None = None, backing: np.ndarray | None = None,
            separation: str = "auto", dsp_only: bool = False, lyrics: str | None = None, recording_id: str | None = None, separated=None, target=None,
            take_separator=None) -> Result[Representation]:
    """Analyse one recording.

    ``reference``: the guide vocal (array + sr, or a path) a take was sung
    along with; the take is shifted by the refined latency and the estimate is
    kept in ``rep.meta["latency"]`` (the shared-clock premise reads its
    confidence).  ``lyrics`` maps syllables onto the notes (for a target).  ``separation``: ``auto`` | ``always`` | ``off`` (revision A1).

    Revision D1:

    * ``separated`` (a target song): the song's :class:`~gyeol.frontend.separation.TargetSeparation`
      from :func:`separate_target` — its cached vocal is analysed instead of separating again;
    * ``target`` (a take of that song): the target song's ``TargetSeparation``; the take (assumed recorded on
      headphones) is separated only when its accompaniment bleeds into the microphone, with
      ``take_separator`` (a name from ``SEPARATORS`` — default ``"backing"``, which subtracts the cached
      accompaniment — or a separator object).
    """
    from .attributes.extract import AnalysisConfig
    from .attributes.extract import analyze as _analyze
    from .context import assign_syllables
    from .io import refine_offset, shift

    try:
        x, sr = _audio(audio, sr)
    except (OSError, RuntimeError) as exc:
        return Result.failure(f"cannot read audio: {exc}")
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
    rec = Recording(x, sr, **kw)
    separator, acc_ref = take_separator, None
    if separated is not None:
        from .frontend.separation import PrecomputedSeparator

        separator, separation = PrecomputedSeparator(separated), "always"
    if target is not None:
        acc_ref = np.asarray(target.accompaniment, float)
        if target.sr != sr:
            from .dsp.base import resample

            acc_ref = resample(acc_ref, target.sr, sr)
    r = _analyze(rec, trackers=_trackers(dsp_only), backing=backing, separation=separation, config=AnalysisConfig(),
                 separator=separator, accompaniment_ref=acc_ref)
    if r.usable:
        if latency is not None:
            r.value.meta["latency"] = latency
        if lyrics:
            r.value.meta["syllables"] = assign_syllables(r.value.meta.get("notes", []), lyrics)
    return r


def separate_target(audio: np.ndarray | str | Path, sr: int | None = None, *, cache_dir: str | Path | None = None,
                    separator="bs_roformer", background: bool = False, **kw):
    """Separate an uploaded target song once (heavy BS-RoFormer by default) and cache the stems by content hash.

    ``background=True`` returns a :class:`concurrent.futures.Future` from a per-cache background queue, so the
    upload request can return at once; the result (``Result[TargetSeparation]``) is cached, and later calls with
    the same audio return it immediately.  Pass the result to :func:`analyze` as ``separated=`` (the song) or
    ``target=`` (user takes).
    """
    from .frontend.separation import SeparationCache, SeparationQueue
    from .frontend.separation import separate_target as _separate_target

    x, sr = _audio(audio, sr)
    cache = SeparationCache(cache_dir) if cache_dir is not None else None
    if not background:
        return _separate_target(x, sr, cache=cache, separator=separator, **kw)
    key = str(Path(cache_dir).resolve()) if cache_dir is not None else None
    q = _QUEUES.get(key)
    if q is None:
        q = _QUEUES[key] = SeparationQueue(cache)
    return q.submit(x, sr, separator=separator, **kw)


_QUEUES: dict = {}


def take(audio: np.ndarray | str | Path, sr: int | None, rep: Representation):
    """The :class:`~gyeol.demo.Take` for audio passed to :func:`analyze` (re-applies the latency shift analyze applied)."""
    from .demo import Take
    from .io import shift

    x, sr = _audio(audio, sr)
    lat = rep.meta.get("latency") or {}
    if lat.get("offset_s"):
        x = shift(x, float(lat["offset_s"]), sr)
    return Take(Recording(x, sr, recording_id=rep.recording_id), rep)


def compare(user_takes: Representation | Sequence[Representation], target: Representation, *, lyrics: str | None = None,
            audibility: tuple[np.ndarray, int] | None = None, config=None) -> Result[Explanation]:
    """Explain the take(s) against ``target`` (habit vs error across takes, premises, withheld items).

    ``audibility``: the last take's audio as ``(audio, sr)``; when given, every item gets an audibility score by
    re-rendering that take with the item corrected (DSP renderer).
    """
    from .explain import explain, score_audibility

    takes = [user_takes] if isinstance(user_takes, Representation) else list(user_takes)
    ex = explain(takes, target, config, lyrics=lyrics)
    if not ex.usable or audibility is None:
        return ex
    from .demo import DSPRenderer

    aud = score_audibility(ex.value, take(audibility[0], audibility[1], takes[-1]), target, DSPRenderer())
    if not aud.ok:
        return Result(Status.UNRELIABLE, ex.value, f"audibility not scored: {aud.reason}", ex.warnings)
    return ex


@dataclass
class DemoResult:
    baseline: np.ndarray  # the unedited re-render (for A/B listening)
    steps: list[np.ndarray]  # the item alone, then step by step toward the target
    sr: int
    metadata: dict  # gyeol.demo document
    item: ExplanationItem
    files: dict[str, Path]  # "baseline", "step_1", … → written WAV (when out_dir is given)


def _default_item(exp: Explanation) -> ExplanationItem | None:
    cands = [it for it in exp.items if it.category != "diction" and it.confidence >= 0.5]
    if not cands:
        return None
    return max(cands, key=lambda it: it.confidence * (it.audibility if it.audibility is not None else abs(it.magnitude)))


def render_demo(audio: np.ndarray | str | Path, sr: int | None, rep: Representation, explanation: Explanation, target: Representation, *,
                item: ExplanationItem | tuple | None = None, out_dir: str | Path | None = None, renderer=None) -> Result[DemoResult]:
    """Re-render a take with one item corrected, then 50 % / 100 % of the way toward the target.

    ``audio``/``sr`` are what was passed to :func:`analyze` for ``rep``.  Returns plain audio arrays; with
    ``out_dir`` the WAVs and ``demo.json`` (``gyeol.demo``) are also written.
    """
    from .demo import DSPRenderer, item_key
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
    t = take(audio, sr, rep)
    d = _render(t, explanation, target, item_key(item), renderer or DSPRenderer())
    if not d.ok:
        return Result(d.status, None, d.reason)
    files: dict[str, Path] = {}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files["baseline"] = out / "demo_0_baseline.wav"
        save_audio(files["baseline"], d.value.baseline, d.value.sr)
        for i, st in enumerate(d.value.steps, 1):
            files[f"step_{i}"] = out / f"demo_{i}_{st.step.label}.wav"
            save_audio(files[f"step_{i}"], st.audio, d.value.sr)
    meta = to_dict(d.value, item_key=item_key(item), files={k: str(v) for k, v in files.items()})
    if out_dir is not None:
        (Path(out_dir) / "demo.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return Result.success(DemoResult(d.value.baseline, [st.audio for st in d.value.steps], d.value.sr, meta, item, files))


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


__all__ = ["DemoResult", "Result", "Status", "SynthNote", "analyze", "compare", "separate_target", "take", "explanation_notes", "from_dict", "from_json", "item_text", "json_schema", "load_audio", "load_strings", "melody",
           "render_demo", "save_audio", "text", "to_dict", "to_json", "train"]

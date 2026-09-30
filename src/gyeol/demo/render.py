"""Stepwise own-voice demos.

:func:`stepwise_schedule` builds the steps (brief §5.6):

1. ``selected`` — only the chosen item corrected, fully;
2. ``toward_target`` — the chosen item plus every other confident, renderable
   item, at increasing fractions ``partial`` of the way to the target style.

:func:`render_demo` renders an unedited baseline (the same renderer with no
edit, for fair A/B listening) and each step, clamps every edit to the user's
feasible range, and returns only :class:`~gyeol.demo.label.LabelledAudio`
(AI-labelled and watermarked).  It requires a
:class:`~gyeol.core.consent.ConsentedVoice` and the user's own take.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.consent import ConsentedVoice
from ..core.containers import Explanation, ExplanationItem, Representation
from ..core.status import Result
from .edits import Edit, EditConfig, ItemKey, edit_for_item, item_key
from .feasible import ClampReport, FeasibleRange, clamp_edit
from .label import LabelledAudio, label_ai_generated
from .renderers import Renderer, UserTake, check_take
from .watermark import SpreadSpectrumWatermark, WatermarkHook


@dataclass
class DemoStep:
    label: str  # "selected" | "toward_target"
    alpha: float  # fraction applied to the non-selected items (selected item is always 1)
    selected: ItemKey
    others: list[ItemKey] = field(default_factory=list)


def stepwise_schedule(exp: Explanation, selected: ItemKey, partial: tuple[float, ...] = (0.5, 1.0),
                      min_confidence: float = 0.5, renderable: set[ItemKey] | None = None) -> list[DemoStep]:
    keys = [item_key(it) for it in exp.items]
    if selected not in keys:
        raise KeyError(f"item {selected} is not in the explanation")
    others = [item_key(it) for it in exp.items if item_key(it) != selected and it.confidence >= min_confidence
              and (renderable is None or item_key(it) in renderable)]
    steps = [DemoStep("selected", 0.0, selected)]
    if others:
        steps += [DemoStep("toward_target", float(a), selected, others) for a in partial if a > 0]
    return steps


@dataclass
class RenderedStep:
    step: DemoStep
    audio: LabelledAudio
    clamp: ClampReport
    skipped: dict[ItemKey, str]


@dataclass
class Demo:
    baseline: LabelledAudio
    steps: list[RenderedStep]
    feasible: FeasibleRange


def _items_by_key(exp: Explanation) -> dict[ItemKey, ExplanationItem]:
    return {item_key(it): it for it in exp.items}


def render_demo(voice: ConsentedVoice, take: UserTake, exp: Explanation, target: Representation, selected: ItemKey,
                renderer: Renderer, *, feasible: FeasibleRange | None = None, schedule: list[DemoStep] | None = None,
                watermark: WatermarkHook | None = None, edit_config: EditConfig | None = None, seed: int = 0) -> Result[Demo]:
    """Render the stepwise demo for ``selected`` in the user's own consented voice."""
    check_take(voice, take)  # raises ConsentError before any rendering
    wm = watermark or SpreadSpectrumWatermark()
    items = _items_by_key(exp)
    if selected not in items:
        return Result.failure(f"item {selected} is not in the explanation")
    try:
        rng = feasible or FeasibleRange.from_takes([take.rep])
    except ValueError as exc:
        return Result.failure(f"feasible range: {exc}")
    edits: dict[ItemKey, Edit] = {}
    skipped: dict[ItemKey, str] = {}
    for k, it in items.items():
        r = edit_for_item(it, take.rep, target, exp, edit_config)
        if not r.ok:
            skipped[k] = r.reason
        elif r.value.requires - renderer.capabilities:
            skipped[k] = f"{renderer.name} cannot render {sorted(r.value.requires - renderer.capabilities)}"
        else:
            edits[k] = r.value
    if selected not in edits:
        return Result.unavailable(f"cannot render a demo for {selected}: {skipped.get(selected, 'no edit')}")
    steps = schedule or stepwise_schedule(exp, selected, renderable=set(edits))

    base = renderer.render(voice, take, None, seed=seed)
    if not base.ok:
        return Result.failure(f"baseline render failed: {base.reason}")
    lab = lambda y, d: label_ai_generated(y, take.recording.sr, voice=voice, renderer=renderer.name, description=d, watermark=wm)  # noqa: E731
    out = []
    for st in steps:
        e = edits[st.selected]
        used = [st.selected]
        for k in st.others:
            if k in edits:
                e = e + edits[k].scaled(st.alpha)
                used.append(k)
        e, rep = clamp_edit(e, take.rep, rng)
        r = renderer.render(voice, take, e, seed=seed)
        if not r.ok:
            return Result.failure(f"step {st.label} failed: {r.reason}")
        desc = {"step": st.label, "alpha": st.alpha, "items": [list(k) for k in used], "clamped_fraction": rep.clamped_fraction,
                "feasible_range_source": rng.source}
        out.append(RenderedStep(st, lab(r.value, desc), rep, {k: v for k, v in skipped.items() if k in st.others}))
    return Result.success(Demo(lab(base.value, {"step": "baseline", "items": []}), out, rng))

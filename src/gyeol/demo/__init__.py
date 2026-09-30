"""Consent-gated own-voice demo rendering with AI labelling (M5).

* :mod:`~gyeol.demo.edits` — explanation item → edit of the user's c(t);
* :mod:`~gyeol.demo.feasible` — the user's feasible range and clamping;
* :mod:`~gyeol.demo.renderers` — DSP (harmonic-plus-noise) and neural (M4)
  renderers; both require a :class:`~gyeol.core.consent.ConsentedVoice`;
* :mod:`~gyeol.demo.render` — stepwise schedule and :func:`render_demo`;
* :mod:`~gyeol.demo.label` / :mod:`~gyeol.demo.watermark` — AI-generated
  metadata and the pluggable watermark hook.
"""

from .edits import Edit, EditConfig, edit_for_item, item_key
from .feasible import ClampReport, FeasibleRange, clamp_edit
from .label import LabelledAudio, label_ai_generated, read_label, save_labelled
from .render import Demo, DemoStep, RenderedStep, render_demo, stepwise_schedule
from .renderers import DSPRenderer, NeuralRenderer, Renderer, UserTake, check_take
from .watermark import SpreadSpectrumWatermark, WatermarkDetection, WatermarkHook

__all__ = [
    "ClampReport", "DSPRenderer", "Demo", "DemoStep", "Edit", "EditConfig", "FeasibleRange", "LabelledAudio",
    "NeuralRenderer", "RenderedStep", "Renderer", "SpreadSpectrumWatermark", "UserTake", "WatermarkDetection",
    "WatermarkHook", "check_take", "clamp_edit", "edit_for_item", "item_key", "label_ai_generated", "read_label",
    "render_demo", "save_labelled", "stepwise_schedule",
]

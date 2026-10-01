"""Demo rendering (M5): a take re-rendered with explanation items corrected.

* :mod:`~gyeol.demo.edits` — explanation item → edit of the take's c(t);
* :mod:`~gyeol.demo.feasible` — the singer's feasible range and clamping;
* :mod:`~gyeol.demo.renderers` — DSP (harmonic-plus-noise) and neural (M4) renderers;
* :mod:`~gyeol.demo.render` — stepwise schedule and :func:`render_demo`.

Renderers return plain audio.  Labelling rendered audio as AI-generated, and any consent the application needs,
are up to the application.
"""

from .edits import Edit, EditConfig, edit_for_item, item_key
from .feasible import ClampReport, FeasibleRange, clamp_edit
from .render import Demo, DemoStep, RenderedStep, render_demo, stepwise_schedule
from .renderers import DSPRenderer, NeuralRenderer, Renderer, Take

__all__ = [
    "ClampReport", "DSPRenderer", "Demo", "DemoStep", "Edit", "EditConfig", "FeasibleRange", "NeuralRenderer", "RenderedStep",
    "Renderer", "Take", "clamp_edit", "edit_for_item", "item_key", "render_demo", "stepwise_schedule",
]

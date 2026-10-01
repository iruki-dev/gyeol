"""Model cards with data provenance (M8).

A card is built from a gyeol checkpoint's embedded metadata, so its provenance is not typed in by hand: every
dataset and weight file it was built from (with the license and source listed in :mod:`gyeol.core.assets`), the
gyeol checkpoints it was initialised from and the config hash.  Evaluation results are attached as named tables
with a ``synthetic`` flag; :meth:`ModelCard.validate` reports cards that

* have no evaluation, or only synthetic evaluation, without saying so in the limitations,
* are missing the standard out-of-scope uses (voice imitation without permission, medical diagnosis, singer
  identification).

The licenses are recorded as information; complying with them is up to whoever uses the model.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_OUT_OF_SCOPE = (
    "Synthesising or imitating a person's voice without their permission.",
    "Medical diagnosis of voice disorders; the coach shows a referral notice instead.",
    "Identifying or verifying singers (voice features are biometric data in many jurisdictions).",
)

DEFAULT_ETHICS = (
    "Voice-derived features (singer vectors, embeddings) can be biometric data; how they are stored and deleted is "
    "decided by the application that uses the model.",
    "Coaching text never labels users (e.g. 'tone-deaf'); register and phonation feedback is phrased tentatively.",
)


@dataclass
class EvalTable:
    name: str
    metrics: dict[str, Any]
    synthetic: bool
    data: str = ""


@dataclass
class ModelCard:
    name: str
    component: str
    architecture: str
    parameters: int
    config_hash: str
    sources: list[dict]  # describe(name) of every dataset / weight file it was built from
    parents: list[dict]  # gyeol checkpoints it was initialised from
    intended_use: list[str]
    out_of_scope: list[str]
    evaluation: list[EvalTable]
    limitations: list[str]
    ethics: list[str] = field(default_factory=lambda: list(DEFAULT_ETHICS))
    created_utc: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))

    def validate(self) -> list[str]:
        problems = []
        for req in REQUIRED_OUT_OF_SCOPE:
            if req not in self.out_of_scope:
                problems.append(f"missing out-of-scope statement: {req!r}")
        if not self.evaluation:
            problems.append("no evaluation results")
        elif all(t.synthetic for t in self.evaluation) and not any("synthetic" in lim.lower() for lim in self.limitations):
            problems.append("all evaluation is synthetic but the limitations do not say so")
        return problems

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def to_markdown(self) -> str:
        L = [f"# Model card — {self.name}", "",
             f"*Component:* {self.component}  ", f"*Architecture:* {self.architecture}  ", f"*Parameters:* {self.parameters:,}  ",
             f"*Created:* {self.created_utc}", "", "## Provenance", "",
             f"- **Config hash:** `{self.config_hash}`", "",
             "Built from (licenses as listed upstream; complying with them is up to the user of this model):", "",
             "| Source | Kind | License | Notes |", "|---|---|---|---|"]
        for s in self.sources:
            notes = "; ".join(s.get("notes", [])) or "—"
            L.append(f"| {s['name']} | {s.get('kind', '')} | {s.get('license', '')} | {notes} |")
        if self.parents:
            L += ["", "Initialised from:", ""] + [f"- {p['name']}" for p in self.parents]
        L += ["", "## Intended use", ""] + [f"- {u}" for u in self.intended_use]
        L += ["", "## Out of scope", ""] + [f"- {u}" for u in self.out_of_scope]
        L += ["", "## Evaluation", ""]
        for t in self.evaluation:
            L += [f"### {t.name}{' (synthetic data)' if t.synthetic else ''}", ""]
            if t.data:
                L += [f"Data: {t.data}", ""]
            L += ["| Metric | Value |", "|---|---|"] + [f"| {k} | {v if not isinstance(v, float) else f'{v:.4g}'} |" for k, v in t.metrics.items()]
            L.append("")
        L += ["## Limitations", ""] + [f"- {x}" for x in self.limitations]
        L += ["", "## Ethics, privacy and rights", ""] + [f"- {x}" for x in self.ethics]
        problems = self.validate()
        if problems:
            L += ["", "## ⚠ Card validation problems", ""] + [f"- {p}" for p in problems]
        return "\n".join(L) + "\n"


def card_from_checkpoint(path: str | Path, *, component: str, architecture: str, intended_use: list[str],
                         evaluation: list[EvalTable], limitations: list[str], extra_out_of_scope: list[str] | None = None) -> ModelCard:
    """Build a card from a gyeol checkpoint's provenance."""
    from ..train.checkpoint import load_checkpoint

    state, info = load_checkpoint(path)
    n_params = int(sum(v.numel() for k, v in state.items() if hasattr(v, "numel") and not k.endswith("num_batches_tracked")))
    return ModelCard(info.name, component, architecture, n_params, info.config_hash,
                     [dict(s) for s in info.sources], list(info.parents), list(intended_use),
                     list(REQUIRED_OUT_OF_SCOPE) + list(extra_out_of_scope or []), list(evaluation), list(limitations))

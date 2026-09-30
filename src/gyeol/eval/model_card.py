"""Model cards with licenses and data provenance (M8).

A card is built from a gyeol checkpoint's embedded metadata, so the license
lineage cannot be typed in by hand: the effective tag, every source asset
(with its registry entry: license, conditions, caveats, whether the tag was
re-verified upstream), the profile it was trained under and the config hash.
Evaluation results are attached as named tables with a ``synthetic`` flag;
:meth:`ModelCard.validate` refuses cards that

* claim the commercial profile with a non-commercial source,
* have no evaluation, or only synthetic evaluation, without saying so in the
  limitations,
* are missing the out-of-scope uses that gyeol requires (no target-singer
  synthesis, no medical diagnosis, no singer identification).
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.license import LicenseTag, Profile, lookup

REQUIRED_OUT_OF_SCOPE = (
    "Synthesising any voice other than the consenting user's own (there is no target-singer synthesis path).",
    "Medical diagnosis of voice disorders; the coach shows a referral notice instead.",
    "Identifying or verifying singers (voice features are sensitive biometric data under PIPA).",
)

DEFAULT_ETHICS = (
    "Voice-derived features (singer vectors, embeddings) are sensitive biometric information under PIPA: stored only with "
    "per-purpose consent, deletable by user id, raw audio deleted after feature extraction by default.",
    "Every generated waveform is labelled AI-generated (metadata + sidecar) and passes a watermark hook (Korean AI Basic Act).",
    "Rendering requires a ConsentedVoice built from the user's own recording and consent token.",
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
    license: str  # effective tag
    profile: str
    config_hash: str
    sources: list[dict]  # registry entries of every training source
    conditions: list[str]
    intended_use: list[str]
    out_of_scope: list[str]
    evaluation: list[EvalTable]
    limitations: list[str]
    ethics: list[str] = field(default_factory=lambda: list(DEFAULT_ETHICS))
    created_utc: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))

    def validate(self) -> list[str]:
        problems = []
        if self.profile == Profile.COMMERCIAL.value:
            bad = [s["name"] for s in self.sources if s["tag"] not in (LicenseTag.COMMERCIAL_OK.value, LicenseTag.COMMERCIAL_OK_CONDITIONAL.value)]
            if bad:
                problems.append(f"commercial profile but non-commercial sources: {bad}")
        for req in REQUIRED_OUT_OF_SCOPE:
            if req not in self.out_of_scope:
                problems.append(f"missing out-of-scope statement: {req!r}")
        if not self.evaluation:
            problems.append("no evaluation results")
        elif all(t.synthetic for t in self.evaluation) and not any("synthetic" in lim.lower() for lim in self.limitations):
            problems.append("all evaluation is synthetic but the limitations do not say so")
        unverified = [s["name"] for s in self.sources if not s.get("verified")]
        if unverified and not any("verif" in lim.lower() for lim in self.limitations):
            problems.append(f"license tags not re-verified upstream for {unverified}; say so in the limitations")
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
             f"*Created:* {self.created_utc}", "", "## License and provenance", "",
             f"- **Effective license tag:** `{self.license}` (most restrictive of all sources)",
             f"- **Training profile:** `{self.profile}`", f"- **Config hash:** `{self.config_hash}`", "",
             "| Source | Kind | Tag | License | Re-verified | Conditions / caveats |", "|---|---|---|---|---|---|"]
        for s in self.sources:
            notes = "; ".join(list(s.get("conditions", [])) + list(s.get("caveats", []))) or "—"
            L.append(f"| {s['name']} | {s['kind']} | `{s['tag']}` | {s['license']} | {'yes' if s.get('verified') else 'no'} | {notes} |")
        if self.conditions:
            L += ["", "Conditions that travel with this model:", ""] + [f"- {c}" for c in self.conditions]
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


def _source_entry(name: str) -> dict:
    a = lookup(name)
    return {"name": a.name, "kind": a.kind.value, "tag": a.tag.value, "license": a.license, "source": a.source,
            "conditions": list(a.conditions), "caveats": list(a.caveats), "verified": bool(a.verified)}


def card_from_checkpoint(path: str | Path, *, component: str, architecture: str, intended_use: list[str],
                         evaluation: list[EvalTable], limitations: list[str], profile: Profile = Profile.RESEARCH,
                         extra_out_of_scope: list[str] | None = None) -> ModelCard:
    """Build a card from a gyeol checkpoint (license gate applies to ``profile``)."""
    from ..train.checkpoint import load_checkpoint

    state, info = load_checkpoint(path, profile)
    n_params = int(sum(v.numel() for k, v in state.items() if hasattr(v, "numel") and not k.endswith("num_batches_tracked")))
    return ModelCard(info.name, component, architecture, n_params, info.license.value, info.profile.value, info.config_hash,
                     [_source_entry(s) for s in info.sources], list(info.conditions), list(intended_use),
                     list(REQUIRED_OUT_OF_SCOPE) + list(extra_out_of_scope or []), list(evaluation), list(limitations))

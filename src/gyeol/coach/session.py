"""Coaching session: what to say after each attempt, and when to stay quiet.

Per attempt (:meth:`CoachSession.new_attempt` → :class:`Attempt`):

1. **filter** — every item must pass its fitted operating threshold
   (:class:`~gyeol.coach.thresholds.ThresholdSet`; no threshold → not shown)
   and the health guard (beginner restrictions);
2. **rank** — reliability × audibility with the category tie order
   (:mod:`gyeol.coach.priority`);
3. **volume** — one primary item plus at most ``n_secondary`` (default 2)
   expandable secondary items;
4. **fading** — feedback frequency follows ``schedule`` (default full →
   half → quarter).  The stage advances when the attempt's *error load*
   (sum of the eligible items' scores) has been stable for
   ``stability_window`` attempts (relative spread ≤ ``stability_tolerance``)
   and drops back to full feedback when the load rises again;
5. **self-assessment first** — :meth:`Attempt.self_assessment_prompt` asks
   what the user noticed; with ``require_self_assessment`` the feedback is
   only revealed after an answer.  The feedback says whether the user's
   observation matched the primary item.

Every :class:`Feedback` carries the persistent medical-referral notice,
current health notices, practice suggestions from the coach-authored data
file, and the provenance of the thresholds it used.  :meth:`CoachSession.summary`
summarises a set of attempts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Iterable

import numpy as np

from ..core.containers import Consistency, Explanation, ExplanationItem
from ..explain.render_text import explanation_notes, item_text, load_strings
from .health import AttemptMetrics, FatigueMonitor, PhonationLog, PhraseCheck, VoiceRange, check_phrase, restricted_for_level
from .practice import Exercise, PracticeMap
from .priority import PriorityConfig, RankedItem, rank
from .thresholds import ThresholdSet

SELF_ASSESSMENT_OPTIONS = ("pitch", "rhythm", "ornament", "dynamics", "phonation", "diction", "nothing")


@lru_cache(maxsize=4)
def coach_strings(lang: str = "ko") -> dict:
    return json.loads(resources.files("gyeol").joinpath(f"resources/{lang}/coach.json").read_text(encoding="utf-8"))


@dataclass
class CoachConfig:
    n_secondary: int = 2
    schedule: tuple[float, ...] = (1.0, 0.5, 0.25)
    stability_window: int = 3
    stability_tolerance: float = 0.25
    require_self_assessment: bool = False
    level: str = "beginner"  # beginner | intermediate | advanced
    priority: PriorityConfig = field(default_factory=PriorityConfig)
    lang: str = "ko"

    def __post_init__(self) -> None:
        if not self.schedule or any(not 0 < f <= 1 for f in self.schedule):
            raise ValueError("schedule frequencies must be in (0, 1]")
        if self.n_secondary < 0 or self.stability_window < 2:
            raise ValueError("n_secondary ≥ 0 and stability_window ≥ 2 required")


@dataclass
class FeedbackEntry:
    item: ExplanationItem
    score: float
    text: str
    tentative: bool
    consistency: str
    practice: list[Exercise]


@dataclass
class Feedback:
    given: bool
    reason: str  # "ok" | "fading" | "no_items"
    primary: FeedbackEntry | None
    secondary: list[FeedbackEntry]
    withheld: dict[str, int]  # reason → count of items not shown
    self_assessment: str | None  # matched | missed | nothing | None (not asked)
    lines: list[str]  # user-facing text, in order
    notices: list[str]  # health notices (always includes the medical referral)
    context: list[str]  # transposition, input quality, cannot-judge spans
    stage: int
    frequency: float
    thresholds_provenance: dict


@dataclass
class AttemptRecord:
    ranked: list[RankedItem]
    load: float
    given: bool
    stage: int
    withheld: dict[str, int]


@dataclass
class SelfAssessmentPrompt:
    question: str
    options: list[tuple[str, str]]  # (id, Korean label)


class Attempt:
    def __init__(self, session: "CoachSession", exp: Explanation, record: AttemptRecord, notices: list[str]):
        self._s, self.explanation, self.record, self._notices = session, exp, record, notices
        self._noticed: set[str] | None = None

    def self_assessment_prompt(self) -> SelfAssessmentPrompt:
        s = coach_strings(self._s.config.lang)["self_assessment"]
        return SelfAssessmentPrompt(s["question"], [(k, s["options"][k]) for k in SELF_ASSESSMENT_OPTIONS])

    def record_self_assessment(self, noticed: Iterable[str]) -> None:
        n = set(noticed)
        unknown = n - set(SELF_ASSESSMENT_OPTIONS)
        if unknown:
            raise ValueError(f"unknown self-assessment options {sorted(unknown)}")
        self._noticed = n

    def _entry(self, r: RankedItem, avoid: set[str]) -> FeedbackEntry:
        it = r.item
        cfg = self._s.config
        cons = load_strings(cfg.lang)["consistency"][it.consistency.value]
        tentative = it.category == "phonation" or bool(it.detail.get("tentative"))
        return FeedbackEntry(it, r.score, item_text(it, cfg.lang), tentative, cons,
                             self._s.practice.for_item(it, cfg.level, avoid) if self._s.practice else [])

    def reveal(self) -> Feedback:
        cfg = self._s.config
        if cfg.require_self_assessment and self._noticed is None:
            raise RuntimeError("record_self_assessment() must be called before reveal() (require_self_assessment=True)")
        s = coach_strings(cfg.lang)
        rec = self.record
        context = explanation_notes(self.explanation, cfg.lang)
        freq = cfg.schedule[rec.stage]
        prov = self._s.thresholds.provenance
        if not rec.given:
            return Feedback(False, "fading", None, [], rec.withheld, None, [s["feedback"]["withheld_fading"]], self._notices,
                            context, rec.stage, freq, prov)
        if not rec.ranked:
            return Feedback(True, "no_items", None, [], rec.withheld, None, [s["feedback"]["none"]], self._notices, context,
                            rec.stage, freq, prov)
        avoid = {"fatigue"} if any(n.startswith("fatigue") for n in self._s.health_flags) else set()
        primary = self._entry(rec.ranked[0], avoid)
        secondary = [self._entry(r, avoid) for r in rec.ranked[1 : 1 + cfg.n_secondary]]
        sa = None
        lines: list[str] = []
        if self._noticed is not None:
            sa = "nothing" if self._noticed <= {"nothing"} else ("matched" if primary.item.category in self._noticed else "missed")
            lines.append(s["self_assessment"][sa])
        lines.append(f"{s['feedback']['primary']}: {primary.text}")
        if primary.tentative:
            lines.append(s["feedback"]["tentative"])
        lines.append(primary.consistency)
        for ex in primary.practice[:1]:
            lines.append(s["feedback"]["practice"].format(title=ex.title, minutes=f"{ex.minutes:g}", instructions=ex.instructions))
        if secondary:
            lines.append(f"{s['feedback']['secondary']}: " + " / ".join(e.text for e in secondary))
        return Feedback(True, "ok", primary, secondary, rec.withheld, sa, lines, self._notices, context, rec.stage, freq, prov)


@dataclass
class SessionSummary:
    n_attempts: int
    persistent: list[ExplanationItem]
    improved: list[ExplanationItem]
    habits: list[ExplanationItem]
    health_flags: list[str]
    phonation_s: float
    lines: list[str]


class CoachSession:
    def __init__(self, thresholds: ThresholdSet, config: CoachConfig | None = None, practice: PracticeMap | None = None,
                 voice_range: VoiceRange | None = None, target_notes_cents: list[float] | None = None):
        if not isinstance(thresholds, ThresholdSet):
            raise TypeError("a fitted ThresholdSet is required: the coach has no built-in display thresholds")
        self.thresholds = thresholds
        self.config = config or CoachConfig()
        self.practice = practice if practice is not None else PracticeMap.load(lang=self.config.lang)
        self.voice_range = voice_range
        self.target_notes = target_notes_cents
        self.phrase: PhraseCheck | None = None
        if voice_range is not None and target_notes_cents is not None:
            self.phrase = check_phrase(target_notes_cents, voice_range)
        self.records: list[AttemptRecord] = []
        self.items_by_attempt: list[dict[tuple, ExplanationItem]] = []
        self.fatigue = FatigueMonitor()
        self.phonation = PhonationLog()
        self.health_flags: list[str] = []
        self._stage, self._since, self._credit = 0, 0, 0.0

    # ------------------------------------------------------------ phrase / health

    def phrase_notices(self) -> list[str]:
        if self.phrase is None:
            return []
        h = coach_strings(self.config.lang)["health"]
        out = []
        if self.phrase.status in ("above_tessitura", "below_tessitura", "out_of_range"):
            out.append(h[self.phrase.status])
        if self.phrase.transpose_semitones:
            k = self.phrase.transpose_semitones
            out.append(h["transpose"].format(direction=h["transpose_up" if k > 0 else "transpose_down"], semitones=abs(k)))
        if self.phrase.octave_shift_cents and self.phrase.status != "comfortable":
            out.append(h["octave"])
        return out

    def _notices(self) -> list[str]:
        h = coach_strings(self.config.lang)["health"]
        return [h["referral"]] + [h[f] for f in self.health_flags] + self.phrase_notices()

    # ------------------------------------------------------------ attempts

    def _filter(self, exp: Explanation) -> tuple[list[ExplanationItem], dict[str, int]]:
        kept, withheld = [], {}
        shift = self.phrase.octave_shift_cents if self.phrase else 0.0
        for it in exp.items:
            ok, why = self.thresholds.passes(it)
            if ok:
                r = restricted_for_level(it, self.config.level, self.voice_range, self.target_notes, shift)
                if r is not None:
                    ok, why = False, r.split(":")[0]
            if ok:
                kept.append(it)
            else:
                withheld[why] = withheld.get(why, 0) + 1
        return kept, withheld

    def _update_stage(self, load: float) -> None:
        cfg = self.config
        loads = [r.load for r in self.records]  # previous attempts
        w, tol = cfg.stability_window, cfg.stability_tolerance
        if len(loads) >= w and load > (1 + tol) * float(np.median(loads[-w:])) + 1e-12 and self._stage > 0:
            self._stage, self._since, self._credit = 0, 0, 0.0  # performance got worse: back to full feedback
            return
        self._since += 1
        window = (loads + [load])[-w:]
        if self._since >= w and len(window) == w:
            spread = (max(window) - min(window)) / max(float(np.mean(window)), 1e-12)
            if spread <= tol or max(window) == 0:
                if self._stage < len(cfg.schedule) - 1:
                    self._stage += 1
                self._since = 0

    def new_attempt(self, exp: Explanation, *, metrics: AttemptMetrics | None = None, voiced_s: float | None = None,
                    day: str = "session") -> Attempt:
        if metrics is not None:
            self.fatigue.add(metrics)
        if voiced_s is not None:
            self.phonation.add(voiced_s, day)
        self.health_flags = self.fatigue.flags() + self.phonation.warnings(day)
        items, withheld = self._filter(exp)
        ranked = rank(items, self.thresholds, self.config.priority)
        load = float(sum(r.score for r in ranked))
        self._update_stage(load)
        self._credit += self.config.schedule[self._stage]
        given = self._credit >= 1.0 - 1e-9
        if given:
            self._credit -= 1.0
        rec = AttemptRecord(ranked, load, given, self._stage, withheld)
        self.records.append(rec)
        self.items_by_attempt.append({it.key: it for it in items})
        return Attempt(self, exp, rec, self._notices())

    # ------------------------------------------------------------ summary

    def summary(self) -> SessionSummary:
        n = len(self.records)
        s = coach_strings(self.config.lang)["summary"]
        keys = {k for d in self.items_by_attempt for k in d}
        persistent, improved, habits = [], [], []
        for k in sorted(keys):
            seen = [i for i, d in enumerate(self.items_by_attempt) if k in d]
            latest = self.items_by_attempt[seen[-1]][k]
            if len(seen) >= 2 and len(seen) >= n / 2:
                persistent.append(latest)
                if latest.consistency is Consistency.STYLE_OR_HABIT or all(
                        np.sign(self.items_by_attempt[i][k].magnitude) == np.sign(latest.magnitude) for i in seen):
                    habits.append(latest)
            if n >= 2 and seen[0] < n - 1 and k not in self.items_by_attempt[-1]:
                improved.append(self.items_by_attempt[seen[0]][k])
        tiers = self.config.priority.tiers
        order = lambda i: (tiers.get(i.category, len(tiers)), -abs(i.magnitude))  # noqa: E731
        persistent.sort(key=order)
        improved.sort(key=order)
        habits.sort(key=order)
        lang = self.config.lang
        lines = [s["header"].format(n=n)]
        lines += [s["persistent"].format(text=item_text(i, lang)) for i in persistent]
        lines += [s["improved"].format(text=item_text(i, lang)) for i in improved]
        lines += [s["habit"].format(text=item_text(i, lang)) for i in habits]
        if not (persistent or improved):
            lines.append(s["none"])
        lines += self._notices()
        return SessionSummary(n, persistent, improved, habits, list(self.health_flags), self.phonation.session_s, lines)

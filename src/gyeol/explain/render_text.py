"""Turn :class:`ExplanationItem` objects into user-facing text from resource files.

All wording lives in ``gyeol/resources/<lang>/explain.json``; this module only
selects a template and fills placeholders.  It is a thin presentation helper
for demos — prioritisation and wording policy belong to :mod:`gyeol.coach` (M6).
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from ..core.containers import Explanation, ExplanationItem


@lru_cache(maxsize=4)
def load_strings(lang: str = "ko") -> dict:
    return json.loads(resources.files("gyeol").joinpath(f"resources/{lang}/explain.json").read_text(encoding="utf-8"))


def _where(it: ExplanationItem, strings: dict) -> str:
    note = it.detail.get("target_note", -1)
    if note is None or note < 0:
        return strings["where"]["phrase"]
    syl = next((s.syllables for s in it.spans if s.syllables), ())
    if syl:
        return strings["where"]["note"].format(syllable="'" + "".join(syl) + "'")
    return strings["where"]["note_unknown"].format(index=note + 1)


def _final_consonant(word: str) -> int | None:
    """Index of the final consonant (0 = none) of the last Hangul syllable, None if not Hangul."""
    for ch in reversed(word):
        if "가" <= ch <= "힣":
            return (ord(ch) - 0xAC00) % 28
        if ch.isalnum():
            return None
    return None


def particles(word: str) -> dict[str, str]:
    """Korean particles agreeing with ``word`` (이/가, 은/는, 을/를, 으로/로)."""
    fc = _final_consonant(word)
    if fc is None:
        return {"i_ga": "이(가)", "eun_neun": "은(는)", "eul_reul": "을(를)", "euro": "(으)로"}
    has = fc != 0
    return {"i_ga": "이" if has else "가", "eun_neun": "은" if has else "는", "eul_reul": "을" if has else "를",
            "euro": "으로" if has and fc != 8 else "로"}  # ㄹ-final takes 로


def _template_key(attr: str) -> tuple[str, str | None]:
    """(template name, label) — ``quality_breathy`` → (``quality``, ``breathy``)."""
    for prefix in ("quality_", "laryngeal_"):
        if attr.startswith(prefix):
            return prefix[:-1], attr[len(prefix):]
    return attr, None


def item_text(it: ExplanationItem, lang: str = "ko") -> str:
    s = load_strings(lang)
    name, label = _template_key(it.attribute)
    tmpl = s["attribute"].get(name)
    if tmpl is None:
        return f"{it.category}/{it.attribute}: {it.magnitude:+.1f} {it.unit}"
    mag = it.magnitude * 100.0 if it.unit in ("similarity", "probability") else it.magnitude
    key = it.detail.get("status") or ("pos" if mag >= 0 else "neg")
    if key not in tmpl:
        key = next(iter(tmpl))
    value = f"{abs(mag):.0f}" if abs(mag) >= 10 else f"{abs(mag):.1f}"
    where = _where(it, s)
    names = s.get("labels", {})
    extra = {"label": names.get(label, label or ""), "target": names.get(str(it.detail.get("target", "")), it.detail.get("target", "")),
             "user": names.get(str(it.detail.get("user", "")), it.detail.get("user", ""))}
    return tmpl[key].format(where=where, value=value, unit=it.unit, **particles(where), **extra)


def explanation_notes(exp: Explanation, lang: str = "ko") -> list[str]:
    """Context lines: transposition, input-quality flags, cannot-judge spans."""
    s = load_strings(lang)
    out = []
    tr = exp.transposition_cents
    if tr <= -1200:
        out.append(s["transposition"]["octave_down"])
    elif tr >= 1200:
        out.append(s["transposition"]["octave_up"])
    elif tr != 0:
        out.append(s["transposition"]["semitones"].format(value=round(tr / 100)))
    for flag in (exp.meta.get("user_quality") or {}).get("flags", {}):
        out.append(s["quality_flag"].get(flag, flag))
    for sp in exp.cannot_judge:
        a, b = sp.seconds(exp.grid)
        reason = sp.reason.split(":", 1)[0] if sp.reason else "low_confidence"
        tmpl = s["cannot_judge"].get(reason, s["cannot_judge"]["low_confidence"])
        out.append(tmpl.format(start=f"{a:.1f}", end=f"{b:.1f}"))
    return out

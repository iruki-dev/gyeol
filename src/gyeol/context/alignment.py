"""Lyric alignment → frame-level context tokens.

Preferred source: a forced alignment (e.g. Montreal Forced Aligner with a
Korean model adapted on sung data, research §6) read from a Praat TextGrid.
Fallback: a heuristic one-syllable-per-note mapping (common in Korean
singing, where each syllable usually carries one note).  The alignment
source is recorded in the representation so downstream users know how much
to trust the tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tokens import ContextTokens
from .korean import LARYNGEAL_CLASSES, SYLLABLE_POSITIONS, Syllable, is_hangul, lyrics_to_syllables


@dataclass
class Interval:
    start: float
    end: float
    label: str


def read_textgrid(path: str | Path) -> dict[str, list[Interval]]:
    """Minimal Praat TextGrid reader (long and short text formats, interval tiers)."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    tiers: dict[str, list[Interval]] = {}
    if 'item [' in text:  # long format
        for block in re.split(r"item \[\d+\]:", text)[1:]:
            m = re.search(r'name = "(.*?)"', block)
            if not m or 'class = "IntervalTier"' not in block:
                continue
            ivs = re.findall(r'xmin = ([\d.eE+-]+)\s*xmax = ([\d.eE+-]+)\s*text = "(.*?)"', block, flags=re.S)
            tiers[m.group(1)] = [Interval(float(a), float(b), c) for a, b, c in ivs]
        return tiers
    tokens = re.findall(r'"(?:[^"]|"")*"|[^\s"]+', text)
    i = 0
    while i < len(tokens):
        if tokens[i] == '"IntervalTier"':
            name = tokens[i + 1].strip('"')
            n = int(tokens[i + 4])
            j = i + 5
            ivs = []
            for _ in range(n):
                ivs.append(Interval(float(tokens[j]), float(tokens[j + 1]), tokens[j + 2][1:-1].replace('""', '"')))
                j += 3
            tiers[name] = ivs
            i = j
        else:
            i += 1
    return tiers


@dataclass
class AlignedSyllable:
    syllable: Syllable
    start: float
    end: float
    vowel_onset: float  # time the vowel starts (== start if unknown)


def align_to_notes(syllables: list[Syllable], notes: list[tuple[float, float]]) -> list[AlignedSyllable]:
    """Heuristic: syllable i ↔ note i; extra syllables split the longest notes,
    extra notes are treated as melismas of the previous syllable."""
    if not syllables or not notes:
        return []
    spans = [list(n) for n in notes]
    while len(spans) < len(syllables):
        k = int(np.argmax([b - a for a, b in spans]))
        a, b = spans[k]
        mid = 0.5 * (a + b)
        spans[k : k + 1] = [[a, mid], [mid, b]]
    out: list[AlignedSyllable] = []
    # melismas: surplus notes stay attached to the last syllable
    groups: list[list[list[float]]] = [[s] for s in spans[: len(syllables)]]
    for s in spans[len(syllables) :]:
        groups[-1].append(s)
    for syl, g in zip(syllables, groups):
        out.append(AlignedSyllable(syl, g[0][0], g[-1][1], g[0][0]))
    return out


def align_from_textgrid(syllables: list[Syllable], tiers: dict[str, list[Interval]], vowel_labels: set[str] | None = None) -> list[AlignedSyllable]:
    """Use a words/syllables tier (Hangul labels) and, if present, a phones
    tier to locate each vowel onset."""
    word_tier = next((tiers[k] for k in ("syllables", "words", "word") if k in tiers), None)
    if word_tier is None:
        raise ValueError(f"TextGrid needs a 'words' or 'syllables' tier, found {list(tiers)}")
    phone_tier = next((tiers[k] for k in ("phones", "phone") if k in tiers), [])
    # distribute word intervals over their Hangul syllables
    spans: list[tuple[float, float]] = []
    for iv in word_tier:
        chars = [c for c in iv.label if is_hangul(c)]
        if not chars:
            continue
        step = (iv.end - iv.start) / len(chars)
        spans.extend((iv.start + k * step, iv.start + (k + 1) * step) for k in range(len(chars)))
    n = min(len(spans), len(syllables))
    vowel_labels = vowel_labels or _default_vowel_labels()
    out = []
    for syl, (a, b) in zip(syllables[:n], spans[:n]):
        onset = a
        for ph in phone_tier:
            if a - 1e-3 <= ph.start < b and ph.label.strip() in vowel_labels:
                onset = ph.start
                break
        out.append(AlignedSyllable(syl, a, b, onset))
    return out


def _default_vowel_labels() -> set[str]:
    from .korean import MEDIALS

    # MFA Korean models use Hangul jamo or romanised IPA for vowels
    ipa = {"a", "e", "i", "o", "u", "ɛ", "ʌ", "ɯ", "ø", "y", "ja", "jʌ", "jo", "ju", "je", "jɛ", "wa", "wʌ", "we", "wi", "wɛ", "ɰi", "ɐ", "ɨ"}
    return set(MEDIALS) | ipa


def context_tokens(
    aligned: list[AlignedSyllable],
    n_frames: int,
    rate: float,
    source: str,
    phrase_pause_s: float = 0.3,
) -> ContextTokens:
    t = np.arange(n_frames) / rate
    syl_idx = np.full(n_frames, -1, dtype=np.int32)
    lar = np.zeros(n_frames, dtype=np.int8)
    since = np.full(n_frames, np.nan)
    pos = np.zeros(n_frames, dtype=np.int8)
    phrase_init = np.zeros(n_frames, dtype=bool)
    prev_end = -np.inf
    for k, a in enumerate(aligned):
        sel = (t >= a.start) & (t < a.end)
        syl_idx[sel] = k
        lar[sel] = LARYNGEAL_CLASSES.index(a.syllable.laryngeal_class)
        since[sel] = (t[sel] - a.vowel_onset) * 1000.0
        p = np.where(t[sel] < a.vowel_onset, SYLLABLE_POSITIONS.index("onset"), SYLLABLE_POSITIONS.index("nucleus"))
        pos[sel] = p
        if a.start - prev_end >= phrase_pause_s:
            phrase_init[sel] = True
        prev_end = a.end
    syll_meta = [dict(a.syllable.to_dict(), start=a.start, end=a.end, vowel_onset=a.vowel_onset) for a in aligned]
    return ContextTokens(
        rate=rate,
        syllable_index=syl_idx,
        laryngeal_class=lar,
        time_since_onset_ms=since,
        syllable_position=pos,
        phrase_initial=phrase_init,
        syllables=syll_meta,
        alignment_source=source,
    )


def build_context(
    n_frames: int,
    rate: float,
    notes: list[tuple[float, float]],
    lyrics: str | None = None,
    textgrid: str | Path | None = None,
    use_g2pk: bool = False,
) -> ContextTokens | None:
    if lyrics is None and textgrid is None:
        return None
    if textgrid is not None:
        tiers = read_textgrid(textgrid)
        if lyrics is None:
            word_tier = next((tiers[k] for k in ("syllables", "words", "word") if k in tiers), [])
            lyrics = " ".join(iv.label for iv in word_tier if iv.label.strip())
        sylls = lyrics_to_syllables(lyrics, use_g2pk=use_g2pk)
        return context_tokens(align_from_textgrid(sylls, tiers), n_frames, rate, source="textgrid")
    sylls = lyrics_to_syllables(lyrics or "", use_g2pk=use_g2pk)
    return context_tokens(align_to_notes(sylls, notes), n_frames, rate, source="heuristic-note")

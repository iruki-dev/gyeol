"""Korean phonetic context: Hangul decomposition, surface pronunciation and
the laryngeal class of each syllable's onset.

Why it matters (research §4.5, §6): vowel voice quality after lenis (평음),
aspirated (격음) and fortis (경음) stops differs systematically — H1–H2 is
highest after lenis, intermediate after aspirated, lowest after fortis
(Cho, Jun & Ladefoged 2002) — and aspirated/fortis onsets raise f0 in Seoul
Korean (Kang 2014).  Without these tokens, phonetic pressing is mistaken for
strain and consonant f0 perturbation for pitch error.

The laryngeal class must be taken from the *surface* pronunciation: 국밥 is
pronounced [국빱], so the second syllable has a fortis onset.  A small
rule-based G2P covers the rules that change onset class (연음 liaison,
경음화 tensification, 격음화 aspiration, ㅎ deletion) plus 비음화.  If the
optional ``g2pk`` package is installed it can be used instead
(``use_g2pk=True``).  *g2pK's accuracy on sung lyrics is undocumented
(UNVERIFIED HYPOTHESIS, research §6).*
"""

from __future__ import annotations

from dataclasses import dataclass

HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3

INITIALS = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
MEDIALS = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
FINALS = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

#: laryngeal classes used by the context tokens (index = stored integer)
LARYNGEAL_CLASSES = ("none", "lenis", "aspirated", "fortis", "sonorant", "nasal")
#: syllable-position classes (index = stored integer)
SYLLABLE_POSITIONS = ("none", "onset", "nucleus", "coda")

_LARYNGEAL = {
    "ㄱ": "lenis", "ㄷ": "lenis", "ㅂ": "lenis", "ㅈ": "lenis", "ㅅ": "lenis",
    "ㅋ": "aspirated", "ㅌ": "aspirated", "ㅍ": "aspirated", "ㅊ": "aspirated",
    # ㅎ patterns with the aspirated series in onset f0 raising (Kang 2014)
    "ㅎ": "aspirated",
    "ㄲ": "fortis", "ㄸ": "fortis", "ㅃ": "fortis", "ㅆ": "fortis", "ㅉ": "fortis",
    "ㄴ": "nasal", "ㅁ": "nasal",
    "ㄹ": "sonorant",
    "ㅇ": "none",
}

_TENSE = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
_ASPIRATE = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ", "ㅅ": "ㅆ"}
# representative (neutralised) coda sounds — 7 대표음
_CODA_NEUTRAL = {
    "ㄱ": "ㄱ", "ㄲ": "ㄱ", "ㅋ": "ㄱ", "ㄳ": "ㄱ", "ㄺ": "ㄱ",
    "ㄴ": "ㄴ", "ㄵ": "ㄴ", "ㄶ": "ㄴ",
    "ㄷ": "ㄷ", "ㅅ": "ㄷ", "ㅆ": "ㄷ", "ㅈ": "ㄷ", "ㅊ": "ㄷ", "ㅌ": "ㄷ", "ㅎ": "ㄷ",
    "ㄹ": "ㄹ", "ㄼ": "ㄹ", "ㄽ": "ㄹ", "ㄾ": "ㄹ", "ㅀ": "ㄹ",
    "ㅁ": "ㅁ", "ㄻ": "ㅁ",
    "ㅂ": "ㅂ", "ㅍ": "ㅂ", "ㄿ": "ㅂ", "ㅄ": "ㅂ",
    "ㅇ": "ㅇ", "": "",
}
_OBSTRUENT_NEUTRAL = {"ㄱ", "ㄷ", "ㅂ"}
_NASALISE = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}
# double codas: (remaining coda, consonant that moves to the next onset)
_SPLIT = {
    "ㄳ": ("ㄱ", "ㅅ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("ㄴ", "ㅎ"), "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"),
    "ㄼ": ("ㄹ", "ㅂ"), "ㄽ": ("ㄹ", "ㅅ"), "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"), "ㅀ": ("ㄹ", "ㅎ"), "ㅄ": ("ㅂ", "ㅅ"),
}


@dataclass
class Syllable:
    text: str  # orthographic syllable
    initial: str  # orthographic onset jamo ("ㅇ" = empty onset)
    medial: str
    final: str
    surface_initial: str = ""
    surface_final: str = ""
    word_initial: bool = False

    @property
    def laryngeal_class(self) -> str:
        return _LARYNGEAL.get(self.surface_initial or self.initial, "none")

    @property
    def nasal_coda(self) -> bool:
        return (self.surface_final or self.final) in ("ㄴ", "ㅁ", "ㅇ")

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "initial": self.initial,
            "surface_initial": self.surface_initial,
            "medial": self.medial,
            "final": self.final,
            "surface_final": self.surface_final,
            "laryngeal_class": self.laryngeal_class,
            "nasal_coda": self.nasal_coda,
            "word_initial": self.word_initial,
        }


def is_hangul(ch: str) -> bool:
    return len(ch) == 1 and HANGUL_BASE <= ord(ch) <= HANGUL_END


def decompose(ch: str) -> tuple[str, str, str]:
    code = ord(ch) - HANGUL_BASE
    return INITIALS[code // 588], MEDIALS[(code % 588) // 28], FINALS[code % 28]


def syllabify(text: str) -> list[Syllable]:
    """Hangul syllables of ``text`` (non-Hangul characters are skipped)."""
    out: list[Syllable] = []
    new_word = True
    for ch in text:
        if is_hangul(ch):
            i, m, f = decompose(ch)
            out.append(Syllable(text=ch, initial=i, medial=m, final=f, word_initial=new_word))
            new_word = False
        elif ch.isspace() or ch in ",.!?~-":
            new_word = True
    return out


def surface_pronunciation(syllables: list[Syllable], across_words: bool = True) -> list[Syllable]:
    """Apply the onset-changing phonological rules in place and return the list.

    Rules (applied left to right at each syllable boundary):
    1. ㅎ-coda + lenis onset → aspirated onset; ㅎ-coda + ㅇ → ㅎ deleted
    2. obstruent coda + ㅎ onset → aspirated onset (격음화)
    3. coda + ㅇ onset → coda moves to onset (연음; double codas split)
    4. obstruent coda + lenis onset → fortis onset (경음화)
    5. obstruent coda + nasal onset → nasalised coda (비음화)
    In singing, words are usually sung connected, so rules apply across word
    boundaries unless ``across_words`` is False.
    """
    for s in syllables:
        s.surface_initial = s.initial
        s.surface_final = s.final
    for a, b in zip(syllables[:-1], syllables[1:]):
        if b.word_initial and not across_words:
            a.surface_final = _CODA_NEUTRAL.get(a.surface_final, a.surface_final)
            continue
        coda, onset = a.surface_final, b.surface_initial
        if coda in ("ㅎ", "ㄶ", "ㅀ"):
            keep = {"ㅎ": "", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}[coda]
            if onset in _ASPIRATE:
                b.surface_initial = _ASPIRATE[onset]
                a.surface_final = keep
                continue
            if onset == "ㅇ":
                a.surface_final = keep
                if keep:
                    a.surface_final, b.surface_initial = "", keep
                continue
            a.surface_final = keep or "ㄷ"
            coda = a.surface_final
        if onset == "ㅎ" and _CODA_NEUTRAL.get(coda, "") in _OBSTRUENT_NEUTRAL | {"ㄷ"} and coda:
            if coda in _SPLIT:
                a.surface_final, moved = _SPLIT[coda]
            else:
                a.surface_final, moved = "", _CODA_NEUTRAL[coda]
            b.surface_initial = _ASPIRATE.get(moved, onset)
            continue
        if onset == "ㅇ" and coda and coda != "ㅇ":
            if coda in _SPLIT:
                a.surface_final, moved = _SPLIT[coda]
                # 값이 → [갑씨]: the moved lenis tenses after an obstruent coda
                if _CODA_NEUTRAL.get(a.surface_final) in _OBSTRUENT_NEUTRAL and moved in _TENSE:
                    moved = _TENSE[moved]
                b.surface_initial = moved
            else:
                a.surface_final, b.surface_initial = "", coda
            continue
        neutral = _CODA_NEUTRAL.get(coda, coda)
        if neutral in _OBSTRUENT_NEUTRAL and onset in _TENSE:
            b.surface_initial = _TENSE[onset]
        if neutral in _OBSTRUENT_NEUTRAL and onset in ("ㄴ", "ㅁ"):
            neutral = _NASALISE[neutral]
        a.surface_final = neutral
    if syllables:
        last = syllables[-1]
        last.surface_final = _CODA_NEUTRAL.get(last.surface_final, last.surface_final)
    return syllables


def lyrics_to_syllables(text: str, use_g2pk: bool = False) -> list[Syllable]:
    """Syllables with surface onsets for a Korean lyric line."""
    sylls = syllabify(text)
    if use_g2pk:
        try:
            from g2pk import G2p  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("use_g2pk=True needs the optional 'g2pk' package") from exc
        surface = syllabify(G2p()(text))
        if len(surface) == len(sylls):
            for s, p in zip(sylls, surface):
                s.surface_initial, s.surface_final = p.initial, p.final
            return sylls
    return surface_pronunciation(sylls)

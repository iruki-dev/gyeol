"""Korean G2P context (surface_initial / surface_final) and phone set."""

from .korean import LARYNGEAL_CLASSES, SYLLABLE_POSITIONS, Syllable, lyrics_to_syllables, syllabify, surface_pronunciation

__all__ = ["LARYNGEAL_CLASSES", "SYLLABLE_POSITIONS", "Syllable", "assign_syllables", "lyrics_to_syllables", "surface_pronunciation", "syllabify"]


def assign_syllables(notes: list[tuple[int, int]], lyrics: str) -> list[tuple[int, int, str]]:
    """Heuristic one-syllable-per-note mapping (frame spans → Hangul syllables).

    Extra syllables share the last note; extra notes (melismas) keep the
    previous syllable.  Replace with forced alignment (M3) when available.
    """
    sylls = [s.text for s in lyrics_to_syllables(lyrics)]
    if not sylls or not notes:
        return []
    out = []
    for i, (s, e) in enumerate(notes):
        k = min(i, len(sylls) - 1)
        text = "".join(sylls[k:]) if i == len(notes) - 1 and len(sylls) > len(notes) else sylls[k]
        out.append((s, e, text))
    return out

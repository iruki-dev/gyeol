"""Korean G2P context (surface_initial / surface_final) and phone set."""

from .korean import LARYNGEAL_CLASSES, SYLLABLE_POSITIONS, Syllable, lyrics_to_syllables, syllabify, surface_pronunciation

__all__ = ["LARYNGEAL_CLASSES", "SYLLABLE_POSITIONS", "Syllable", "assign_syllables", "lyrics_to_syllables", "surface_pronunciation", "syllabify"]


def assign_syllables(notes: list[tuple[int, int]], lyrics: str) -> list[tuple[int, int, str]]:
    """Map lyric syllables onto note spans (frame indices) — one syllable per note when the counts match.

    The lyric is split with :func:`syllabify` (independent of spaces and
    punctuation).  When there are more syllables than notes, notes take
    several syllables in proportion to their length (longest notes first,
    in sung order); when there are fewer, the extra notes are melismas and
    keep the previous syllable.  Replace with forced alignment when available.
    """
    sylls = [s.text for s in lyrics_to_syllables(lyrics)]
    if not sylls or not notes:
        return []
    n_notes = len(notes)
    if len(sylls) <= n_notes:
        return [(s, e, sylls[min(i, len(sylls) - 1)]) for i, (s, e) in enumerate(notes)]
    # more syllables than notes: give each note ≥ 1 syllable, distribute the rest by note length
    lengths = [max(e - s, 1) for s, e in notes]
    counts = [1] * n_notes
    for _ in range(len(sylls) - n_notes):
        k = max(range(n_notes), key=lambda j: (lengths[j] / (counts[j] + 1), -j))
        counts[k] += 1
    out, pos = [], 0
    for (s, e), c in zip(notes, counts):
        out.append((s, e, "".join(sylls[pos : pos + c])))
        pos += c
    return out

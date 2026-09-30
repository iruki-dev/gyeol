import numpy as np
import pytest

from gyeol.context.alignment import align_to_notes, build_context, read_textgrid
from gyeol.context.korean import LARYNGEAL_CLASSES, decompose, lyrics_to_syllables


def surface(text):
    return "|".join(s.surface_initial + s.medial + s.surface_final for s in lyrics_to_syllables(text))


def test_decompose():
    assert decompose("한") == ("ㅎ", "ㅏ", "ㄴ")
    assert decompose("꿈") == ("ㄲ", "ㅜ", "ㅁ")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("국밥", "ㄱㅜㄱ|ㅃㅏㅂ"),  # 경음화
        ("학교", "ㅎㅏㄱ|ㄲㅛ"),
        ("좋다", "ㅈㅗ|ㅌㅏ"),  # ㅎ + lenis → aspirated
        ("축하", "ㅊㅜ|ㅋㅏ"),  # obstruent + ㅎ → aspirated
        ("입학", "ㅇㅣ|ㅍㅏㄱ"),
        ("좋아", "ㅈㅗ|ㅇㅏ"),  # ㅎ deletion
        ("않아", "ㅇㅏ|ㄴㅏ"),
        ("음악", "ㅇㅡ|ㅁㅏㄱ"),  # 연음
        ("꽃이", "ㄲㅗ|ㅊㅣ"),
        ("닭이", "ㄷㅏㄹ|ㄱㅣ"),  # double coda split
        ("없어", "ㅇㅓㅂ|ㅆㅓ"),  # split + tensification
        ("밥 먹어", "ㅂㅏㅁ|ㅁㅓ|ㄱㅓ"),  # 비음화 + 연음
        ("놓는", "ㄴㅗㄴ|ㄴㅡㄴ"),
    ],
)
def test_surface_pronunciation(text, expected):
    assert surface(text) == expected


def test_laryngeal_classes_follow_surface_form():
    s = lyrics_to_syllables("국밥 좋다 사랑해")
    assert [x.laryngeal_class for x in s] == ["lenis", "fortis", "fortis", "aspirated", "lenis", "sonorant", "aspirated"]
    assert s[5].nasal_coda  # 랑


def test_align_to_notes_handles_count_mismatch():
    sylls = lyrics_to_syllables("가나다")
    a = align_to_notes(sylls, [(0.0, 1.0), (1.0, 3.0)])  # fewer notes than syllables → split longest
    assert len(a) == 3 and a[0].start == 0.0 and a[-1].end == 3.0
    b = align_to_notes(lyrics_to_syllables("가"), [(0.0, 1.0), (1.0, 2.0)])  # melisma
    assert len(b) == 1 and b[0].end == 2.0


def test_context_tokens_and_textgrid(tmp_path):
    tg = tmp_path / "a.TextGrid"
    tg.write_text(
        'File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = 1.0\ntiers? <exists>\nsize = 2\nitem []:\n'
        '    item [1]:\n        class = "IntervalTier"\n        name = "words"\n        xmin = 0\n        xmax = 1.0\n        intervals: size = 2\n'
        '        intervals [1]:\n            xmin = 0\n            xmax = 0.1\n            text = ""\n'
        '        intervals [2]:\n            xmin = 0.1\n            xmax = 1.0\n            text = "타요"\n'
        '    item [2]:\n        class = "IntervalTier"\n        name = "phones"\n        xmin = 0\n        xmax = 1.0\n        intervals: size = 3\n'
        '        intervals [1]:\n            xmin = 0.1\n            xmax = 0.2\n            text = "tʰ"\n'
        '        intervals [2]:\n            xmin = 0.2\n            xmax = 0.55\n            text = "a"\n'
        '        intervals [3]:\n            xmin = 0.55\n            xmax = 1.0\n            text = "jo"\n',
        encoding="utf-8",
    )
    tiers = read_textgrid(tg)
    assert [iv.label for iv in tiers["words"]] == ["", "타요"]
    ctx = build_context(100, 100.0, [], textgrid=tg)
    assert ctx.alignment_source == "textgrid"
    assert ctx.syllables[0]["vowel_onset"] == pytest.approx(0.2)
    assert LARYNGEAL_CLASSES[ctx.laryngeal_class[15]] == "aspirated"
    # 10 ms after the vowel onset
    assert ctx.time_since_onset_ms[21] == pytest.approx(10.0, abs=0.5)
    assert np.isnan(ctx.time_since_onset_ms[5])
    assert ctx.phrase_initial[15]

"""A synthetic stand-in for the real-recording set (tests, CI and a worked manifest example).

``make_synthetic_realset(folder)`` writes WAVs, annotation files and a
``manifest.jsonl`` in exactly the format :mod:`gyeol.eval.realset` reads, with
every condition:

* ``clean`` — the user alone;
* ``mixture_karaoke`` — user + backing track, the backing file is listed (sing-along);
* ``mixture_phone`` — user + backing at a louder level, band-limited and noisy, no backing file;
* ``separated`` — user + a little residual backing, as if separated beforehand;
* ``trimmed`` — a hand-cut clip that starts later and ends earlier than the target;
* ``multi_singer`` — user + a second voice a third above.

Annotations come from the synthesiser's ground truth: f0 track, note onsets,
octave relation and per-note onset deviations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..io import save_audio
from ..synth import SynthNote, accompaniment, melody

NOTES = ((262, "a", "s"), (294, "o", "t"), (330, "i", "k"), (349, "e", "h"), (392, "a", "s"), (330, "u", "t"))
SR = 44100


def _render(detune, shifts, transpose, seed, dur=0.5, gap=0.18, vib=0.0, oq=0.6):
    notes = [SynthNote(f * 2 ** (d / 1200), dur, gap_after=gap, vowel=v, consonant=c, onset_shift_s=s,
                       vibrato_rate=5.5 if vib else 0.0, vibrato_extent_cents=vib)
             for (f, v, c), d, s in zip(NOTES, detune, shifts)]
    return melody(notes, sr=SR, transpose_cents=transpose, seed=seed, open_quotient=oq)


def _norm(x, db):
    return x / (np.sqrt(np.mean(x**2)) + 1e-12) * 10 ** (db / 20)


def make_synthetic_realset(folder: str | Path, n_singers: int = 2, seed: int = 0,
                           conditions: tuple[str, ...] = ("clean", "mixture_karaoke", "mixture_phone", "separated", "trimmed", "multi_singer")) -> Path:
    from scipy import signal

    root = Path(folder)
    for d in ("audio", "ann", "backing"):
        (root / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    guide = _render((0,) * 6, (0,) * 6, 0.0, seed=1000)
    n = len(guide.audio) + int(0.3 * SR)
    pad = lambda x: np.pad(x, (0, max(0, n - len(x))))[:n]  # noqa: E731
    save_audio(root / "audio" / "song01_guide.wav", _norm(pad(guide.audio), -20), SR)
    back = pad(accompaniment(n / SR + 0.1, seed=7))
    save_audio(root / "backing" / "song01.wav", back, SR)
    lines = [{"id": "song01", "audio": "audio/song01_guide.wav", "role": "target", "condition": "clean", "lyrics": "사랑해요 그대여"}]
    for s in range(n_singers):
        octave = int(rng.choice([-1, 0]))
        for ci, cond in enumerate(conditions):
            detune = rng.uniform(-40, 40, 6).round(1)
            shifts = np.r_[0.0, rng.uniform(-0.07, 0.07, 5)].round(3)
            on_time = rng.random(6) < 0.5
            shifts[on_time] = 0.0
            m = _render(detune, shifts, 1200.0 * octave, seed=100 * s + ci, oq=0.5 + 0.1 * s)
            voc = _norm(pad(m.audio), -20)
            rid = f"s{s:02d}_{cond}"
            rec: dict = {"device": "synthetic", "route": "wired"}
            start = 0.0
            if cond == "mixture_karaoke":
                x = voc + _norm(back, -26)
                rec["backing"] = "backing/song01.wav"
            elif cond == "mixture_phone":
                x = signal.sosfilt(signal.butter(4, [200, 3400], btype="bandpass", fs=SR, output="sos"), voc + _norm(back, -23))
                x = x + _norm(rng.standard_normal(n), -50)
                rec.update(device="phone", route="speaker")
            elif cond == "separated":
                x = voc + _norm(back, -45)
            elif cond == "trimmed":
                start = 0.9
                x = voc[int(start * SR) : int((n / SR - 0.6) * SR)]
            elif cond == "multi_singer":
                second = _render(detune * 0, shifts, 1200.0 * octave + 400, seed=999 + s, oq=0.7)
                x = voc + _norm(pad(second.audio), -23)
            else:
                x = voc
            save_audio(root / "audio" / f"{rid}.wav", x, SR)
            f0 = m.truth["f0_track"]
            t = np.arange(0, len(f0) / SR, 0.01)
            f0s = np.interp(t, np.arange(len(f0)) / SR, f0)
            keep = (t >= start) & (t < start + len(x) / SR)
            with open(root / "ann" / f"{rid}.f0.csv", "w", encoding="utf-8") as fh:
                fh.write("# time_s,f0_hz\n")
                for ti, fi in zip(t[keep] - start, f0s[keep]):
                    fh.write(f"{ti:.3f},{fi:.2f}\n")
            onsets = [round(a - start, 3) for a, _, _ in m.truth["notes"] if start <= a < start + len(x) / SR]
            lines.append({"id": rid, "audio": f"audio/{rid}.wav", "role": "user", "target": "song01", "singer": f"s{s:02d}",
                          "session": f"s{s:02d}-session1", "condition": cond, "lyrics": "사랑해요 그대여",
                          "annotations": {"f0": f"ann/{rid}.f0.csv", "onsets": onsets, "octave_relation": octave,
                                          "onset_deviation_ms": {str(k): float(sh * 1000) for k, sh in enumerate(shifts)}},
                          "recording": rec})
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
    return root

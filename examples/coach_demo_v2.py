"""gyeol v2 coaching demo (M1: pitch, rhythm and ornaments).

    # synthetic target + two user takes with known deviations
    python examples/coach_demo_v2.py --synthetic --out /tmp/gyeol_demo

    # real recordings: target guide vocal + one or more sing-along takes
    python examples/coach_demo_v2.py --target guide.wav --user take1.wav take2.wav --lyrics "사랑해 너를"

Steps: load → offline latency refinement against the guide vocal →
signal-layer analysis → explanation → Korean text from resource files.
Prioritisation, feedback volume and practice suggestions are M6 (coach);
audibility scores and own-voice demos are M5.  This demo lists every item.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gyeol.attributes.extract import analyze
from gyeol.context import assign_syllables
from gyeol.core import Provenance, Recording
from gyeol.explain import explain
from gyeol.explain.render_text import explanation_notes, item_text, load_strings
from gyeol.io import load_recording, refine_offset, save_audio, shift
from gyeol.pitch.adapters import PyinTracker, SHSTracker, YinTracker, default_trackers


def synthesize(out: Path) -> tuple[Path, list[Path], str]:
    from gyeol.synth import SynthNote as N
    from gyeol.synth import melody

    sr = 44100
    lyrics = "사랑 해요 그대"
    base = [(262, "a", "s"), (294, "a", None), (330, "e", "h"), (349, "o", None), (392, "e", "k"), (330, "e", "t")]

    def notes(detune=(0,) * 6, shifts=(0,) * 6, vib=(0,) * 6, scoop=(0,) * 6, fall=(0,) * 6):
        return [N(f * 2 ** (d / 1200), 0.8, gap_after=0.15, vowel=v, consonant=c, onset_shift_s=s,
                  vibrato_rate=5.5 if e else 0, vibrato_extent_cents=e, scoop_cents=sc, fall_cents=fa)
                for (f, v, c), d, s, e, sc, fa in zip(base, detune, shifts, vib, scoop, fall)]

    out.mkdir(parents=True, exist_ok=True)
    target = melody(notes(vib=(0, 0, 50, 0, 50, 0), scoop=(0, 120, 0, 0, 0, 0), fall=(0, 0, 0, 0, 0, 150)), sr=sr)
    # take 1 and 2: consistently flat on the 3rd syllable, no scoop (habit);
    # a late entry only in take 2 (inconsistent → error); an octave lower
    t1 = melody(notes(detune=(0, 0, -45, 0, 0, 0), vib=(0, 0, 50, 0, 0, 0), fall=(0, 0, 0, 0, 0, 150)), sr=sr, transpose_cents=-1200, seed=1)
    t2 = melody(notes(detune=(0, 0, -40, 0, 0, 0), shifts=(0, 0, 0, 0.08, 0, 0), vib=(0, 0, 50, 0, 0, 0), fall=(0, 0, 0, 0, 0, 150)),
                sr=sr, transpose_cents=-1200, seed=2)
    n = max(len(target.audio), len(t1.audio), len(t2.audio)) + int(0.1 * sr)
    paths = []
    for name, m, lat in (("target", target, 0.0), ("take1", t1, 0.035), ("take2", t2, 0.035)):
        audio = np.pad(np.r_[np.zeros(int(lat * sr)), m.audio], (0, 0))[:n]
        audio = np.pad(audio, (0, n - len(audio)))
        p = out / f"{name}.wav"
        save_audio(p, audio, sr)
        paths.append(p)
    return paths[0], paths[1:], lyrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path)
    ap.add_argument("--user", type=Path, nargs="+")
    ap.add_argument("--lyrics", default="")
    ap.add_argument("--synthetic", action="store_true", help="generate a synthetic target and two takes")
    ap.add_argument("--out", type=Path, default=Path("gyeol_demo_out"))
    ap.add_argument("--dsp-only", action="store_true", help="use only gyeol's DSP pitch trackers")
    ap.add_argument("--show-all", action="store_true", help="list every item, ignoring the demo display filter")
    args = ap.parse_args(argv)

    if args.synthetic:
        target_path, user_paths, lyrics = synthesize(args.out)
        print(f"synthetic files written to {args.out}")
    elif args.target and args.user:
        target_path, user_paths, lyrics = args.target, args.user, args.lyrics
    else:
        ap.error("give --synthetic or --target and --user")

    trackers = [PyinTracker(), YinTracker(), SHSTracker()] if (args.dsp_only or args.synthetic) else default_trackers()
    tr = load_recording(target_path, Provenance.REFERENCE)
    if not tr.ok:
        print(f"target: {tr.reason}", file=sys.stderr)
        return 2
    target_rec = tr.value
    t_rep = analyze(target_rec, trackers=trackers)
    if not t_rep.usable:
        print(f"target analysis failed: {t_rep.reason}", file=sys.stderr)
        return 2
    target = t_rep.value
    if lyrics:
        target.meta["syllables"] = assign_syllables(target.meta["notes"], lyrics)

    takes = []
    for p in user_paths:
        ur = load_recording(p, Provenance.USER, owner_id="demo-user")
        if not ur.ok:
            print(f"{p}: {ur.reason}", file=sys.stderr)
            continue
        rec = ur.value
        off = refine_offset(rec.audio, target_rec.audio, rec.sr)
        audio = rec.audio
        if off.usable:
            audio = shift(rec.audio, off.value.latency_s, rec.sr)
            print(f"{p.name}: latency refined by {off.value.latency_s * 1000:.0f} ms ({off.status.value})")
        rep = analyze(Recording(audio, rec.sr, Provenance.USER, owner_id=rec.owner_id, recording_id=rec.recording_id), trackers=trackers)
        if not rep.usable:
            print(f"{p.name}: analysis failed: {rep.reason}", file=sys.stderr)
            continue
        takes.append(rep.value)
    if not takes:
        return 2
    ex = explain(takes, target)
    if not ex.usable:
        print(f"explanation failed: {ex.reason}", file=sys.stderr)
        return 2
    e = ex.value
    s = load_strings("ko")
    print(f"\n=== 설명 ({e.n_takes}회 녹음, 마지막 녹음 기준) ===")
    for line in explanation_notes(e):
        print(f"  · {line}")
    display = json.loads((Path(__file__).parent / "demo_display.json").read_text(encoding="utf-8"))
    hidden = 0
    for it in e.items:
        small = abs(it.magnitude) < display["min_abs_magnitude"].get(it.unit, 0.0) and it.detail.get("status") in (None, "different")
        if not args.show_all and (small or it.confidence < display["min_confidence"]):
            hidden += 1
            continue
        print(f"[{s['category'][it.category]}] {item_text(it)}  "
              f"(신뢰도 {it.confidence:.2f}; {s['consistency'][it.consistency.value]})")
    if hidden:
        print(f"  (작은 차이 {hidden}개는 숨겼어요 — --show-all 로 모두 보기)")
    print(f"\n  · {s['audibility_pending']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

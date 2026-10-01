"""gyeol v2 coaching demo — uses the public API (``gyeol.api``) only.

    # synthetic target + two user takes with known deviations
    python examples/coach_demo_v2.py --synthetic --out /tmp/gyeol_demo

    # real recordings: target guide vocal + one or more sing-along takes
    python examples/coach_demo_v2.py --target guide.wav --user take1.wav take2.wav --lyrics "사랑해 너를"

    # audibility per item, and a stepwise demo of the top item in the user's voice
    python examples/coach_demo_v2.py --synthetic --audibility --render-demo

Steps: analyse the guide → analyse each take with latency refinement against
the guide → compare → Korean text from resource files → versioned JSON
(``explanation.json``, ``gyeol.explanation`` v1).  ``--audibility`` and
``--render-demo`` re-render the user's last take with the edits applied.
Consent, labelling and storage of rendered audio are the application's
policy (see the reference service).

The coaching session (feedback fading, attempt history, self-assessment) is
user state and lives in the reference service: see
``reference_service/examples/coach_session_demo.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gyeol import api


def synthesize(out: Path) -> tuple[Path, list[Path], str]:
    sr = 44100
    lyrics = "사랑 해요 그대"
    base = [(262, "a", "s"), (294, "a", None), (330, "e", "h"), (349, "o", None), (392, "e", "k"), (330, "e", "t")]

    def notes(detune=(0,) * 6, shifts=(0,) * 6, vib=(0,) * 6, scoop=(0,) * 6, fall=(0,) * 6):
        return [api.SynthNote(f * 2 ** (d / 1200), 0.8, gap_after=0.15, vowel=v, consonant=c, onset_shift_s=s,
                              vibrato_rate=5.5 if e else 0, vibrato_extent_cents=e, scoop_cents=sc, fall_cents=fa)
                for (f, v, c), d, s, e, sc, fa in zip(base, detune, shifts, vib, scoop, fall)]

    out.mkdir(parents=True, exist_ok=True)
    target = api.melody(notes(vib=(0, 0, 50, 0, 50, 0), scoop=(0, 120, 0, 0, 0, 0), fall=(0, 0, 0, 0, 0, 150)), sr=sr)
    # take 1 and 2: consistently flat on the 3rd syllable, no scoop (habit);
    # a late entry only in take 2 (inconsistent → error); an octave lower
    t1 = api.melody(notes(detune=(0, 0, -45, 0, 0, 0), vib=(0, 0, 50, 0, 0, 0), fall=(0, 0, 0, 0, 0, 150)), sr=sr, transpose_cents=-1200, seed=1)
    t2 = api.melody(notes(detune=(0, 0, -40, 0, 0, 0), shifts=(0, 0, 0, 0.08, 0, 0), vib=(0, 0, 50, 0, 0, 0), fall=(0, 0, 0, 0, 0, 150)),
                    sr=sr, transpose_cents=-1200, seed=2)
    n = max(len(target.audio), len(t1.audio), len(t2.audio)) + int(0.1 * sr)
    paths = []
    for name, m, lat in (("target", target, 0.0), ("take1", t1, 0.035), ("take2", t2, 0.035)):
        audio = np.r_[np.zeros(int(lat * sr)), m.audio][:n]
        p = out / f"{name}.wav"
        api.save_audio(p, np.pad(audio, (0, n - len(audio))), sr)
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
    ap.add_argument("--audibility", action="store_true", help="score audibility per item (renders your own voice)")
    ap.add_argument("--render-demo", action="store_true", help="render a stepwise own-voice demo of the top item")
    args = ap.parse_args(argv)

    if args.synthetic:
        target_path, user_paths, lyrics = synthesize(args.out)
        print(f"synthetic files written to {args.out}")
    elif args.target and args.user:
        target_path, user_paths, lyrics = args.target, args.user, args.lyrics
    else:
        ap.error("give --synthetic or --target and --user")

    dsp_only = args.dsp_only or args.synthetic
    guide, gsr = api.load_audio(target_path)
    t = api.analyze(guide, gsr, lyrics=lyrics or None, dsp_only=dsp_only)
    if not t.usable:
        print(f"target analysis failed: {t.reason}", file=sys.stderr)
        return 2
    target = t.value

    takes, last = [], None
    for p in user_paths:
        x, sr = api.load_audio(p)
        r = api.analyze(x, sr, reference=(guide, gsr), dsp_only=dsp_only)
        if not r.usable:
            print(f"{p.name}: analysis failed: {r.reason}", file=sys.stderr)
            continue
        lat = r.value.meta.get("latency", {})
        print(f"{p.name}: latency refined by {lat.get('offset_s', 0.0) * 1000:.0f} ms ({lat.get('status', '-')})")
        takes.append(r.value)
        last = (x, sr, r.value)
    if not takes:
        return 2

    ex = api.compare(takes, target, audibility=(last[0], last[1]) if args.audibility else None)
    if not ex.usable:
        print(f"explanation failed: {ex.reason}", file=sys.stderr)
        return 2
    if ex.reason:
        print(f"note: {ex.reason}", file=sys.stderr)
    e = ex.value
    api.to_json(e, args.out / "explanation.json")

    s = api.load_strings("ko")
    notes, sentences = api.text(e)
    print(f"\n=== 설명 ({e.n_takes}회 녹음, 마지막 녹음 기준) ===")
    for line in notes:
        print(f"  · {line}")
    display = json.loads((Path(__file__).parent / "demo_display.json").read_text(encoding="utf-8"))
    hidden = 0
    for it, sentence in sentences:
        small = abs(it.magnitude) < display["min_abs_magnitude"].get(it.unit, 0.0) and it.detail.get("status") in (None, "different")
        if not args.show_all and (small or it.confidence < display["min_confidence"]):
            hidden += 1
            continue
        aud = "" if it.audibility is None else f"; {s['audibility']['label']} {it.audibility:.3f}"
        print(f"[{s['category'][it.category]}] {sentence}  (신뢰도 {it.confidence:.2f}{aud}; {s['consistency'][it.consistency.value]})")
    if hidden:
        print(f"  (작은 차이 {hidden}개는 숨겼어요 — --show-all 로 모두 보기)")
    if not args.audibility:
        print(f"\n  · {s['audibility']['unavailable']}")
    print(f"\n  explanation.json → {args.out / 'explanation.json'}")
    if args.render_demo:
        return _render(last, e, target, args.out)
    return 0


def _render(last, e, target, out: Path) -> int:
    x, sr, rep = last
    d = api.render_demo(x, sr, rep, e, target, out_dir=out)
    if not d.usable:
        print(f"demo not rendered: {d.reason}", file=sys.stderr)
        return 0
    ds = api.load_strings("ko", "demo")
    print(f"\n=== 내 목소리 시범: {api.item_text(d.value.item)} ===")
    print(f"  {d.value.files['baseline'].name} — {ds['baseline']}")
    for i, st in enumerate(d.value.metadata["steps"], 1):
        label = ds["step"]["selected"] if st["label"] == "selected" else ds["step"]["toward_target"].format(percent=round(100 * st["alpha"]))
        print(f"  {d.value.files[f'step_{i}'].name} — {label}")
        if round(100 * st["clamped_fraction"]) >= 1:
            print(f"     {ds['clamped'].format(percent=round(100 * st['clamped_fraction']))}")
    print(f"  demo.json (gyeol.demo v{d.value.metadata['version']}) → {out / 'demo.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

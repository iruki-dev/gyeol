"""gyeol v2 coaching demo (M1 explanation; M5 audibility and own-voice demos).

    # synthetic target + two user takes with known deviations
    python examples/coach_demo_v2.py --synthetic --out /tmp/gyeol_demo

    # real recordings: target guide vocal + one or more sing-along takes
    python examples/coach_demo_v2.py --target guide.wav --user take1.wav take2.wav --lyrics "사랑해 너를"

    # M5: audibility per item, and a stepwise own-voice demo of the top item
    python examples/coach_demo_v2.py --synthetic --audibility --render-demo

    # M6: coaching policy (thresholds fitted by examples/fit_thresholds.py)
    python examples/fit_thresholds.py --synthetic --out thresholds.json
    python examples/coach_demo_v2.py --synthetic --coach thresholds.json --level beginner

Steps: load → offline latency refinement against the guide vocal →
signal-layer analysis → explanation → Korean text from resource files.
With ``--audibility`` / ``--render-demo`` the demo user grants
``voice_synthesis`` consent in a local :class:`~gyeol.store.ConsentStore`, a
:class:`~gyeol.core.ConsentedVoice` is built from their **own** last take, and
the DSP renderer resynthesises that take (never the target).  Demo WAVs are
AI-labelled (INFO tags + ``.ai.json`` sidecar) and watermarked.
Prioritisation, feedback volume and practice suggestions are M6 (coach); here
the demo item is simply the one with the largest confidence × audibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gyeol.attributes.extract import analyze
from gyeol.context import assign_syllables
from gyeol.core import ConsentedVoice, Provenance, Purpose, Recording
from gyeol.demo import DSPRenderer, UserTake, item_key, render_demo, save_labelled
from gyeol.encoders.latent import ltas_singer_vector
from gyeol.explain import explain, score_audibility
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
    ap.add_argument("--audibility", action="store_true", help="score audibility per item (renders your own voice)")
    ap.add_argument("--render-demo", action="store_true", help="render a stepwise own-voice demo of the top item")
    ap.add_argument("--coach", type=Path, help="coach each take with thresholds from this JSON (see fit_thresholds.py)")
    ap.add_argument("--level", default="beginner", choices=["beginner", "intermediate", "advanced"])
    ap.add_argument("--noticed", nargs="*", help="self-assessment answer, e.g. --noticed pitch rhythm")
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
        aligned = Recording(audio, rec.sr, Provenance.USER, owner_id=rec.owner_id, recording_id=rec.recording_id)
        rep = analyze(aligned, trackers=trackers)
        if not rep.usable:
            print(f"{p.name}: analysis failed: {rep.reason}", file=sys.stderr)
            continue
        takes.append(rep.value)
        last_take = UserTake(aligned, rep.value)
    if not takes:
        return 2
    ex = explain(takes, target)
    if not ex.usable:
        print(f"explanation failed: {ex.reason}", file=sys.stderr)
        return 2
    e = ex.value
    s = load_strings("ko")
    voice = None
    if args.audibility or args.render_demo:
        from gyeol.store import ConsentStore

        # in the app the user grants this on a consent screen; here the demo user does
        token = ConsentStore(args.out / "consent").grant(last_take.recording.owner_id, {Purpose.ANALYSIS, Purpose.VOICE_SYNTHESIS})
        voice = ConsentedVoice.create(ltas_singer_vector(last_take.recording), last_take.recording, token)
        renderer = DSPRenderer()
        if args.audibility:
            aud = score_audibility(e, voice, last_take, target, renderer)
            if not aud.ok:
                print(f"audibility failed: {aud.reason}", file=sys.stderr)
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
        aud = "" if it.audibility is None else f"; {s['audibility']['label']} {it.audibility:.3f}"
        print(f"[{s['category'][it.category]}] {item_text(it)}  "
              f"(신뢰도 {it.confidence:.2f}{aud}; {s['consistency'][it.consistency.value]})")
    if hidden:
        print(f"  (작은 차이 {hidden}개는 숨겼어요 — --show-all 로 모두 보기)")
    if not args.audibility:
        print(f"\n  · {s['audibility']['unavailable']}")
    if args.coach:
        _coach(takes, target, args)
    if args.render_demo:
        return _render(e, voice, last_take, target, renderer, args.out)
    return 0


def _coach(takes, target, args) -> None:
    """Treat each take as one attempt of a coaching session (M6)."""
    import numpy as np

    from gyeol.coach import CoachConfig, CoachSession, ThresholdSet, VoiceRange, attempt_metrics, note_centres, voiced_seconds
    from gyeol.coach.session import coach_strings

    cs = coach_strings("ko")
    ts = ThresholdSet.from_json(args.coach)
    sung = np.concatenate([t.curves["f0_cents"].values for t in takes])
    try:  # demo only: range from the takes themselves; the app measures it at onboarding
        # (assume a little headroom beyond what was sung: glides are not recorded in this demo)
        vr = VoiceRange.from_samples(np.concatenate([sung - 300, sung + 300]), sung, source="observed takes (demo)")
    except ValueError:
        vr = None
    session = CoachSession(ts, CoachConfig(level=args.level), voice_range=vr, target_notes_cents=note_centres(target))
    print("\n=== 코칭 ===")
    if ts.provenance.get("synthetic"):
        print(f"  {cs['feedback']['thresholds_synthetic']}")
    for i, rep in enumerate(takes, 1):
        ex = explain(takes[:i], target)  # all takes so far: habit vs error across attempts
        if not ex.usable:
            continue
        attempt = session.new_attempt(ex.value, metrics=attempt_metrics(rep), voiced_s=voiced_seconds(rep))
        if args.noticed is not None:
            print(f"  ? {attempt.self_assessment_prompt().question}  → {', '.join(args.noticed) or '-'}")
            attempt.record_self_assessment(args.noticed or ["nothing"])
        fb = attempt.reveal()
        print(f"\n[{cs['feedback']['attempt'].format(n=i)}]")
        for line in fb.lines:
            print(f"  {line}")
    print()
    for line in session.summary().lines:
        print(f"  · {line}")


def _render(e, voice, take, target, renderer, out: Path) -> int:
    from importlib import resources

    ds = json.loads(resources.files("gyeol").joinpath("resources/ko/demo.json").read_text(encoding="utf-8"))
    cands = [it for it in e.items if it.category != "diction" and it.confidence >= 0.5]
    if not cands:
        print("no confident item to demonstrate")
        return 0
    top = max(cands, key=lambda it: it.confidence * (it.audibility if it.audibility is not None else abs(it.magnitude)))
    d = render_demo(voice, take, e, target, item_key(top), renderer)
    if not d.ok:
        print(f"demo not rendered: {d.reason}", file=sys.stderr)
        return 0
    print(f"\n=== 내 목소리 시범: {item_text(top)} ===")
    print(f"  · {ds['ai_notice']}")
    save_labelled(out / "demo_0_baseline.wav", d.value.baseline)
    print(f"  demo_0_baseline.wav — {ds['baseline']}")
    for i, st in enumerate(d.value.steps, 1):
        name = f"demo_{i}_{st.step.label}.wav"
        save_labelled(out / name, st.audio)
        label = ds["step"]["selected"] if st.step.label == "selected" else ds["step"]["toward_target"].format(percent=round(100 * st.step.alpha))
        print(f"  {name} — {label}")
        if round(100 * st.clamp.clamped_fraction) >= 1:
            print(f"     {ds['clamped'].format(percent=round(100 * st.clamp.clamped_fraction))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

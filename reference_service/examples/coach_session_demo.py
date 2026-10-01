"""Coaching-session demo on top of the stateless library (revision C1).

    python examples/fit_thresholds.py --synthetic --out thresholds.json
    python reference_service/examples/coach_session_demo.py --synthetic --coach thresholds.json --level beginner --noticed pitch

The library (``gyeol.api``, ``gyeol.coach``) measures and explains; this
service-side session keeps the user state — attempt history, the feedback
fading schedule, self-assessment and the session summary — and the consent
store issues the consent token.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

from gyeol import api
from gyeol.coach import ThresholdSet, VoiceRange, attempt_metrics, note_centres, voiced_seconds
from gyeol_service import CoachConfig, CoachSession, ConsentStore, coach_strings

ROOT = Path(__file__).resolve().parents[2]


def _synthesize(out: Path):
    spec = importlib.util.spec_from_file_location("coach_demo_v2", ROOT / "examples" / "coach_demo_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.synthesize(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path)
    ap.add_argument("--user", type=Path, nargs="+")
    ap.add_argument("--lyrics", default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("gyeol_session_out"))
    ap.add_argument("--coach", type=Path, required=True, help="thresholds JSON (examples/fit_thresholds.py)")
    ap.add_argument("--level", default="beginner", choices=["beginner", "intermediate", "advanced"])
    ap.add_argument("--noticed", nargs="*", help="self-assessment answer, e.g. --noticed pitch rhythm")
    args = ap.parse_args(argv)
    if args.synthetic:
        target_path, user_paths, lyrics = _synthesize(args.out)
    elif args.target and args.user:
        target_path, user_paths, lyrics = args.target, args.user, args.lyrics
    else:
        ap.error("give --synthetic or --target and --user")

    owner = "demo-user"
    # the service records the user's consent; the library only checks the token it is given
    ConsentStore(args.out / "consent").grant(owner, {api.Purpose.ANALYSIS})
    guide, gsr = api.load_audio(target_path)
    target = api.analyze(guide, gsr, role="reference", lyrics=lyrics or None, dsp_only=True).unwrap()
    takes = []
    for p in user_paths:
        x, sr = api.load_audio(p)
        r = api.analyze(x, sr, owner_id=owner, reference=(guide, gsr), dsp_only=True)
        if r.usable:
            takes.append(r.value)
    if not takes:
        return 2

    cs = coach_strings("ko")
    ts = ThresholdSet.from_json(args.coach)
    sung = np.concatenate([t.curves["f0_cents"].values for t in takes])
    try:  # demo only: range from the takes themselves; the app measures it at onboarding
        vr = VoiceRange.from_samples(np.concatenate([sung - 300, sung + 300]), sung, source="observed takes (demo)")
    except ValueError:
        vr = None
    session = CoachSession(ts, CoachConfig(level=args.level), voice_range=vr, target_notes_cents=note_centres(target))
    print("=== 코칭 ===")
    if ts.provenance.get("synthetic"):
        print(f"  {cs['feedback']['thresholds_synthetic']}")
    for i, rep in enumerate(takes, 1):
        ex = api.compare(takes[:i], target)  # all takes so far: habit vs error across attempts
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
    return 0


if __name__ == "__main__":
    sys.exit(main())

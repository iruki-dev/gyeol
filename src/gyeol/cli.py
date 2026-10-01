"""Command-line interface.

    gyeol licenses                 list registered assets and their tags
    gyeol fetch <name> [--yes]     show the license, ask, then download
    gyeol profile [wav]            per-stage latency of the analysis pipeline
    gyeol eval realset <folder>    evaluate on user-supplied real recordings (manifest.jsonl)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from . import __version__
from .core.license import REGISTRY, Profile, decide, lookup

CACHE = Path.home() / ".cache" / "gyeol"


def _licenses(args: argparse.Namespace) -> int:
    prof = Profile(args.profile)
    for a in REGISTRY.values():
        d = decide(a, prof)
        mark = "allowed" if d.allowed else "REFUSED"
        ver = "" if a.verified else " (unverified)"
        print(f"{a.name:26s} {a.kind.value:10s} {a.tag.value:26s} {a.license:18s} {mark}{ver}")
    return 0


def _fetch(args: argparse.Namespace, stdin=None) -> int:
    asset = lookup(args.name)
    prof = Profile(args.profile)
    d = decide(asset, prof)
    print(f"asset:    {asset.name} ({asset.kind.value})")
    print(f"license:  {asset.license}  [{asset.tag.value}]{'' if asset.verified else '  (tag not re-verified upstream)'}")
    print(f"source:   {asset.source or '-'}")
    for line in list(asset.conditions) + list(asset.caveats):
        print(f"note:     {line}")
    if not d.allowed:
        print(f"refused:  {d.reason}", file=sys.stderr)
        return 2
    if asset.url is None:
        print("This asset has no registered download URL; obtain it manually from the source above and place it in", CACHE / asset.name)
        return 3
    if not args.yes:
        stream = stdin or sys.stdin
        print(f"Download {asset.url} to {CACHE / asset.name}? Type 'yes' to accept the license: ", end="", flush=True)
        if stream.readline().strip().lower() != "yes":
            print("aborted")
            return 1
    dest = CACHE / asset.name / Path(asset.url).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(asset.url, dest)  # noqa: S310 - explicit, user-confirmed download
    if asset.sha256:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest != asset.sha256:
            dest.unlink()
            print(f"checksum mismatch for {dest}", file=sys.stderr)
            return 4
    print(f"saved {dest}")
    return 0


def _profile(args: argparse.Namespace) -> int:
    """Per-stage wall time of the signal-layer analysis (and explanation against itself)."""
    from .export.latency import environment, pipeline_profile
    from .pitch.adapters import PyinTracker, SHSTracker, YinTracker

    if args.wav:
        from .io import load_audio

        x, sr = load_audio(args.wav)
    else:
        from .synth import SynthNote, melody

        notes = [SynthNote(f, 0.6, gap_after=0.15) for f in (262, 294, 330, 349, 392, 330) * max(1, int(args.seconds // 4.5))]
        m = melody(notes, sr=44100)
        x, sr = m.audio, 44100
    trackers = [PyinTracker(), YinTracker(), SHSTracker()] if args.dsp_only else None
    target = None
    if args.explain:
        from .attributes.extract import analyze
        from .core.consent import Provenance
        from .core.containers import Recording

        target = analyze(Recording(x, sr, Provenance.SYNTHETIC), trackers=trackers).unwrap()
    prof = pipeline_profile(x, sr, target, trackers, n_runs=args.runs)
    dur = prof.pop("audio_seconds")
    print(f"audio {dur:.2f} s; environment {environment()}")
    for k, v in sorted(prof.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {v * 1000:9.1f} ms   RTF {v / dur:6.3f}")
    return 0


def _eval_realset(args: argparse.Namespace) -> int:
    """Evaluate the analysis path on a user-supplied folder of real recordings (revision A6)."""
    from .eval.realset import evaluate_realset, load_realset
    from .pitch.adapters import PyinTracker, SHSTracker, YinTracker

    rs = load_realset(args.folder)
    if not rs.ok:
        print(rs.reason, file=sys.stderr)
        return 2
    trackers = [PyinTracker(), YinTracker(), SHSTracker()] if args.dsp_only else None
    rep = evaluate_realset(rs.value, split=args.split, held_out_fraction=args.held_out_fraction, seed=args.seed,
                           separation=args.separation, trackers=trackers, rhythm_verdict_ms=args.rhythm_verdict_ms,
                           progress=(lambda i: print(f"  {i}", file=sys.stderr)) if args.verbose else None)
    print(f"realset {args.folder} — split: {args.split}, separation: {args.separation}")
    print(rep.table())
    for f in rep.failures:
        print(f"  failed: {f}")
    out = Path(args.out) if args.out else Path(args.folder) / f"realset_report_{args.split}.json"
    rep.to_json(out)
    if args.per_tracker:
        import json

        from .eval.realset import tracker_breakdown
        from .pitch.adapters import default_trackers

        bd = tracker_breakdown(rs.value, trackers or default_trackers(), split=args.split, held_out_fraction=args.held_out_fraction, seed=args.seed)
        print("per-tracker pitch agreement on the raw input (RPA / RCA / voicing recall):")
        for cond, by in sorted(bd.items()):
            for name, m in sorted(by.items()):
                print(f"  {cond:18s} {name:10s} {m['rpa']:.3f} / {m['rca']:.3f} / {m['voicing_recall']:.3f}  (n={m['n']})")
        out.with_name(out.stem + "_trackers.json").write_text(json.dumps(bd, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gyeol", description="gyeol v2 — interpretable singing-voice model")
    p.add_argument("--version", action="version", version=f"gyeol {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    lic = sub.add_parser("licenses", help="list registered assets and license decisions")
    lic.add_argument("--profile", default="commercial", choices=[x.value for x in Profile])
    lic.set_defaults(fn=_licenses)
    f = sub.add_parser("fetch", help="show a license, ask for confirmation, then download")
    f.add_argument("name")
    f.add_argument("--profile", default="commercial", choices=[x.value for x in Profile])
    f.add_argument("--yes", action="store_true", help="accept the shown license non-interactively")
    f.set_defaults(fn=_fetch)
    pr = sub.add_parser("profile", help="per-stage latency of the analysis pipeline on this machine")
    pr.add_argument("wav", nargs="?", help="audio file (default: a synthetic melody)")
    pr.add_argument("--seconds", type=float, default=9.0, help="length of the synthetic melody")
    pr.add_argument("--runs", type=int, default=3)
    pr.add_argument("--dsp-only", action="store_true", help="use only gyeol's DSP pitch trackers")
    pr.add_argument("--explain", action="store_true", help="also time the explanation (against the same recording)")
    pr.set_defaults(fn=_profile)
    ev = sub.add_parser("eval", help="evaluations")
    evs = ev.add_subparsers(dest="eval_cmd", required=True)
    rs = evs.add_parser("realset", help="evaluate on a folder of real recordings with manifest.jsonl")
    rs.add_argument("folder")
    rs.add_argument("--split", default="all", choices=["all", "tuning", "held_out"])
    rs.add_argument("--held-out-fraction", type=float, default=0.3)
    rs.add_argument("--seed", type=int, default=0)
    rs.add_argument("--separation", default="auto", choices=["auto", "always", "off"])
    rs.add_argument("--rhythm-verdict-ms", type=float, default=50.0)
    rs.add_argument("--dsp-only", action="store_true", help="use only gyeol's DSP pitch trackers")
    rs.add_argument("--out", help="JSON report path (default: <folder>/realset_report_<split>.json)")
    rs.add_argument("--per-tracker", action="store_true", help="also report each pitch tracker alone on the raw input")
    rs.add_argument("--verbose", action="store_true")
    rs.set_defaults(fn=_eval_realset)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

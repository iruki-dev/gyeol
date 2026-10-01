"""Command-line interface.

    gyeol licenses                 list the third-party models and datasets gyeol knows, with their licenses
    gyeol fetch <name>             download a listed asset and verify its SHA-256
    gyeol profile [wav]            per-stage latency of the analysis pipeline
    gyeol eval realset <folder>    evaluate on user-supplied real recordings (manifest.jsonl)
    gyeol eval pitch <manifest>    pitch accuracy on a human-annotated set
    gyeol prepare --manifest m.json --out cache     separation, pitch, curves, features (resumable)
    gyeol train <task> --config run.yaml [--resume] heads | autoencoder | vocoder | pitch | ssl
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from . import __version__
from .core.assets import ASSETS, asset

CACHE = Path.home() / ".cache" / "gyeol"


def _licenses(args: argparse.Namespace) -> int:
    for a in ASSETS.values():
        pin = " sha256 pinned" if a.sha256 else ""
        print(f"{a.name:26s} {a.kind:8s} {a.license:40s} {a.source}{pin}")
    print("\nYou are responsible for complying with these licenses and with applicable law.")
    return 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(args: argparse.Namespace) -> int:
    try:
        a = asset(args.name)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2
    print(f"asset:    {a.name} ({a.kind})")
    print(f"license:  {a.license}")
    print(f"source:   {a.source}")
    for line in a.notes:
        print(f"note:     {line}")
    if a.url is None:
        print(f"no download URL is listed; obtain it from the source above and place it in {CACHE / a.name}")
        return 3
    dest = Path(args.dest) if args.dest else CACHE / a.name / Path(a.url).name
    if dest.exists() and a.sha256 and _sha256(dest) == a.sha256:
        print(f"already present and verified: {dest}")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(a.url, tmp)  # noqa: S310 - explicit download requested on the command line
    if a.sha256:
        digest = _sha256(tmp)
        if digest != a.sha256:
            tmp.unlink()
            print(f"checksum mismatch for {a.url}: got {digest}, expected {a.sha256}; nothing saved", file=sys.stderr)
            return 4
    tmp.replace(dest)
    print(f"saved {dest}" + (" (sha256 verified)" if a.sha256 else " (no checksum listed)"))
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
        from .core.containers import Recording

        target = analyze(Recording(x, sr), trackers=trackers).unwrap()
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


def _eval_pitch(args: argparse.Namespace) -> int:
    """Pitch accuracy on a human-annotated set (revision D3)."""
    from .eval.pitch_eval import evaluate_pitch

    if args.checkpoint:
        from .pitch.rmvpe import RMVPE, RMVPETracker
        from .train.checkpoint import load_checkpoint

        sd, _ = load_checkpoint(args.checkpoint)
        model = RMVPE()
        model.load_state_dict({k[len("pitch."):]: v for k, v in sd.items() if k.startswith("pitch.")})
        trackers = [RMVPETracker(model=model)]
    elif args.rmvpe_weights:
        from .pitch.rmvpe import RMVPETracker

        trackers = [RMVPETracker(args.rmvpe_weights)]
    else:
        from .pitch.adapters import PyinTracker, SHSTracker, YinTracker

        trackers = [PyinTracker(), YinTracker(), SHSTracker()]
    for tr in trackers:
        r = evaluate_pitch(args.manifest, tr, limit=args.limit)
        if not r.ok:
            print(f"{tr.name}: {r.reason}", file=sys.stderr)
            continue
        s = r.value.summary()
        print(f"{s['dataset']} / {s['tracker']}: n={s['n_items']} RPA {s['rpa']:.3f} RCA {s['rca']:.3f} "
              f"VR {s['voicing_recall']:.3f} VFA {s['voicing_false_alarm']:.3f} OA {s['overall_accuracy']:.3f}")
    return 0


def _prepare(args: argparse.Namespace) -> int:
    """Batch preparation of training data (revision B6)."""
    from .data.prepare import PrepareConfig, prepare, synthetic_manifest

    manifests = list(args.manifest or [])
    cfg = PrepareConfig(sr=args.sr, hop=args.hop, separation=args.separation, dsp_trackers_only=not args.neural_trackers,
                        features=tuple(args.features), max_seconds=args.max_seconds, resynthesize=tuple(args.resynthesize or ()),
                        resynth_vocoder=args.resynth_vocoder)
    if args.synthetic:
        manifests.append(synthetic_manifest(Path(args.out) / "synthetic_src", n_singers=args.synthetic, sr=args.sr))
    if not manifests:
        print("nothing to prepare: give --manifest (one or more) or --synthetic N", file=sys.stderr)
        return 2
    ssl = None
    if "ssl" in cfg.features:
        if not args.ssl_checkpoint:
            print("--features ssl needs --ssl-checkpoint (local HuBERT / WavLM weights)", file=sys.stderr)
            return 2
        from .encoders.frame import ssl_from_checkpoint

        ssl = ssl_from_checkpoint(args.ssl_arch, args.ssl_checkpoint, args.ssl_layers, args.ssl_asset)
    rep = prepare(manifests, args.out, cfg, ssl_encoder=ssl, limit=args.limit,
                  progress=(lambda s: print(f"  {s}", file=sys.stderr)) if args.verbose else None)
    print(rep.summary())
    for f in rep.failures[:20]:
        print(f"  failed: {f['path']}: {f['reason']}")
    if rep.failed:
        print(f"  all failures with reasons: {Path(args.out) / 'failures.jsonl'}")
    return 0 if rep.done + rep.skipped > 0 else 1


def _train(args: argparse.Namespace) -> int:
    """Training runner (revision B2)."""
    from .train.config import load_config
    from .train.runner import train

    overrides = list(args.set or [])
    for flag, key in ((args.device, "device"), (args.threads, "threads"), (args.out, "run.out"), (args.max_steps, "run.max_steps")):
        if flag is not None:
            overrides.append(f"{key}={flag}")
    cfg = load_config(args.config, args.task, overrides)
    res = train(cfg, resume=args.resume, stop_after_steps=args.stop_after)
    print(f"{res.status}: {res.position.step} steps → {res.out_dir}")
    if res.best_checkpoint:
        print(f"best weights: {res.best_checkpoint}")
    return 0 if res.status in ("finished", "early_stopped") else 3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gyeol", description="gyeol v2 — interpretable singing-voice model")
    p.add_argument("--version", action="version", version=f"gyeol {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    lic = sub.add_parser("licenses", help="list third-party models and datasets with their licenses")
    lic.set_defaults(fn=_licenses)
    f = sub.add_parser("fetch", help="download a listed asset and verify its SHA-256")
    f.add_argument("name")
    f.add_argument("--dest", help="file to write (default: ~/.cache/gyeol/<name>/<file>)")
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
    ep = evs.add_parser("pitch", help="pitch accuracy on a human-annotated set (Vocadito, MIR-1K manifests)")
    ep.add_argument("manifest")
    ep.add_argument("--checkpoint", help="a `gyeol train pitch` checkpoint (full-size RMVPE)")
    ep.add_argument("--rmvpe-weights", help="fetched reference RMVPE weights")
    ep.add_argument("--limit", type=int)
    ep.set_defaults(fn=_eval_pitch)
    pp = sub.add_parser("prepare", help="prepare training data: separation, pitch, curves, features (resumable)")
    pp.add_argument("--manifest", action="append", help="gyeol manifest JSON (repeatable)")
    pp.add_argument("--out", required=True, help="cache folder")
    pp.add_argument("--sr", type=int, default=44100)
    pp.add_argument("--hop", type=int, default=512)
    pp.add_argument("--separation", default="auto", choices=["auto", "always", "off"])
    pp.add_argument("--features", nargs="+", default=["dsp"], choices=["dsp", "ssl"])
    pp.add_argument("--ssl-checkpoint")
    pp.add_argument("--ssl-asset", help="name of the SSL weights in the asset list (recorded as provenance), e.g. hubert_fairseq")
    pp.add_argument("--ssl-arch", default="hubert_base", choices=["hubert_base", "wavlm_base"])
    pp.add_argument("--ssl-layers", type=int, nargs="+", default=[3, 4, 5])
    pp.add_argument("--neural-trackers", action="store_true", help="use the default tracker set (neural trackers need fetched weights)")
    pp.add_argument("--max-seconds", type=float, default=30.0)
    pp.add_argument("--resynthesize", nargs="+", choices=["hnm", "vocoder"],
                    help="also write exact-f0 resynthesised copies for pitch training (never of app users' recordings)")
    pp.add_argument("--resynth-vocoder", help="checkpoint from `gyeol train vocoder` for --resynthesize vocoder")
    pp.add_argument("--synthetic", type=int, default=0, metavar="N_SINGERS", help="also generate a synthetic corpus with N singers")
    pp.add_argument("--limit", type=int)
    pp.add_argument("--verbose", action="store_true")
    pp.set_defaults(fn=_prepare)
    tr = sub.add_parser("train", help="train a task from a YAML config (CPU or CUDA; interruptible, resumable)")
    tr.add_argument("task", choices=["heads", "autoencoder", "vocoder", "pitch", "ssl"])
    tr.add_argument("--config", required=True)
    tr.add_argument("--resume", action="store_true", help="continue the run in run.out exactly where it stopped")
    tr.add_argument("--device", help="auto | cpu | cuda (overrides the config)")
    tr.add_argument("--threads", type=int)
    tr.add_argument("--out", help="run folder (overrides run.out)")
    tr.add_argument("--max-steps", type=int)
    tr.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value, e.g. optim.lr=1e-3 (repeatable)")
    tr.add_argument("--stop-after", type=int, help=argparse.SUPPRESS)  # simulate an interruption (tests / CI)
    tr.set_defaults(fn=_train)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Command-line interface.

    gyeol analyze take.wav -o take.npz --lyrics "사랑해" --summary
    gyeol spec
    gyeol fit-device reference.wav phone.wav -o phone.json
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .engine import Engine, EngineConfig
from .spec import DIMENSIONS, NOTE_DIMENSIONS, OMISSIONS


def _analyze(args: argparse.Namespace) -> int:
    from .frontend.equalization import DeviceProfile

    profile = DeviceProfile.load(args.device_profile) if args.device_profile else None
    notes = None
    if args.notes:
        notes = [tuple(map(float, line.split()[:2])) for line in open(args.notes, encoding="utf-8") if line.strip()]
    engine = Engine(EngineConfig(use_g2pk=args.g2pk), device_profile=profile)
    rep = engine.analyze(args.input, lyrics=args.lyrics, textgrid=args.textgrid, notes=notes)
    if args.output:
        rep.save(args.output)
    if args.summary or not args.output:
        print(rep.to_json())
    return 0


def _spec(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps({"frame": {k: v.to_dict() for k, v in DIMENSIONS.items()},
                          "note": {k: v.to_dict() for k, v in NOTE_DIMENSIONS.items()},
                          "omissions": OMISSIONS}, ensure_ascii=False, indent=2))
        return 0
    for title, table in (("frame-level", DIMENSIONS), ("note-level", NOTE_DIMENSIONS)):
        print(f"## {title}")
        for d in table.values():
            flag = "  [hypothesis]" if d.hypothesis else ""
            print(f"{d.name:24s} {d.unit:22s} {d.rate:14s} {d.validity}{flag}")
        print()
    print("## declared omissions")
    for k, v in OMISSIONS.items():
        print(f"{k:24s} {v}")
    return 0


def _fit_device(args: argparse.Namespace) -> int:
    from .frontend.equalization import DeviceProfile
    from .io import load_audio

    ref, sr_r = load_audio(args.reference)
    dev, sr_d = load_audio(args.device)
    if sr_r != sr_d:
        from ._dsp import resample

        dev = resample(dev, sr_d, sr_r)
    prof = DeviceProfile.fit(ref, dev, sr_r, name=args.name or args.device)
    prof.save(args.output)
    print(json.dumps({"name": prof.name, "highpass_hz": prof.highpass_hz}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gyeol", description="Singing-voice intermediate representation engine")
    p.add_argument("--version", action="version", version=f"gyeol {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyse an audio file")
    a.add_argument("input")
    a.add_argument("-o", "--output", help="write the representation (.npz)")
    a.add_argument("--lyrics", help="Korean lyrics for context tokens")
    a.add_argument("--textgrid", help="forced-alignment TextGrid (MFA)")
    a.add_argument("--notes", help="text file with 'start end' seconds per line")
    a.add_argument("--device-profile", help="DeviceProfile JSON for device EQ")
    a.add_argument("--g2pk", action="store_true", help="use g2pK for surface pronunciation")
    a.add_argument("--summary", action="store_true", help="print a JSON summary")
    a.set_defaults(fn=_analyze)

    s = sub.add_parser("spec", help="print the dimension specification")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_spec)

    f = sub.add_parser("fit-device", help="fit a DeviceProfile from simultaneous recordings")
    f.add_argument("reference")
    f.add_argument("device")
    f.add_argument("-o", "--output", required=True)
    f.add_argument("--name")
    f.set_defaults(fn=_fit_device)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

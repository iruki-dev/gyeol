"""Command-line interface.

    gyeol licenses                 list registered assets and their tags
    gyeol fetch <name> [--yes]     show the license, ask, then download
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
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

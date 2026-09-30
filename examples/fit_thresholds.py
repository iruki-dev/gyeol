"""Fit coach display thresholds from knob-recovery data (M6).

    # synthetic knob recovery (known detune / timing / vibrato), clean vs pink-noise retest
    python examples/fit_thresholds.py --synthetic --takes 8 --out thresholds.json

The output is a :class:`gyeol.coach.ThresholdSet` JSON with its provenance.
Thresholds fitted on synthetic singing are marked ``"synthetic": true`` and
are for demos and tests only; production thresholds need real paired
recordings (M8).  The coach never falls back to built-in numbers: attributes
without a fitted threshold are not shown.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gyeol.eval.knob_recovery import fit_from_knob_data, knob_recovery
from gyeol.pitch.adapters import PyinTracker, SHSTracker, YinTracker


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="generate synthetic knob-recovery data")
    ap.add_argument("--takes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("thresholds.json"))
    args = ap.parse_args(argv)
    if not args.synthetic:
        ap.error("only --synthetic is available until real paired validation recordings exist (M8)")
    data = knob_recovery(n_takes=args.takes, seed=args.seed, trackers=[PyinTracker(), YinTracker(), SHSTracker()])
    ts = fit_from_knob_data(data)
    for name, t in ts.thresholds.items():
        state = f"U(0.9) = {t.uncertainty(0.9):.2f} {t.unit}, MDC95 = {t.mdc:.2f}" if t.usable else "not reliable → never shown"
        print(f"{name:18s} {state}  (pairs {t.n_pairs}, labelled items {t.n_error})")
    ts.to_json(args.out)
    print(f"wrote {args.out} (provenance: {ts.provenance['data']}; synthetic={ts.provenance['synthetic']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

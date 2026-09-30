"""M8 walkthrough: robustness grid, ONNX export with a latency profile, and a model card.

    python examples/evaluate_and_export.py --out /tmp/gyeol_m8          # production-size (untrained) networks
    python examples/evaluate_and_export.py --out /tmp/gyeol_m8 --tiny   # quick run with CPU-test-sized networks

1. **Robustness** — four synthetic performances that differ in detune and timing, explained against their
   target under clean, noise, reverb, band-limit, separation-artefact and Bluetooth conditions (codecs too when
   ffmpeg exists): ICC(2,1), MDC95 and worst-case deviation per explanation item.
2. **Export** — attribute heads, RMVPE network, acoustic model, vocoder harmonic path and singer encoder to
   ONNX, each verified against PyTorch at two input sizes; then latency (PyTorch eager vs ONNX Runtime) for
   1, 5 and 10 s of audio, plus the signal-layer stage profile.
3. **Model card** — a (synthetic-data) heads checkpoint with its license lineage, carrying the robustness
   results as its evaluation table.

The networks are **untrained**: the latency numbers are real, the outputs are not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from gyeol.attributes.extract import analyze
from gyeol.attributes.heads import FrameHeads, default_tasks
from gyeol.core import Profile, Provenance, Recording
from gyeol.decoder import AutoencoderConfig, GyeolAutoencoder
from gyeol.eval import EvalTable, GridItem, card_from_checkpoint, robustness_grid, run_robustness
from gyeol.export import export_autoencoder, export_heads, export_rmvpe, pipeline_profile, profile, profile_onnx
from gyeol.pitch.adapters import PyinTracker, SHSTracker, YinTracker
from gyeol.pitch.rmvpe import E2E
from gyeol.synth import SynthNote, melody
from gyeol.train.checkpoint import save_checkpoint

NOTES = [(262, "a", "s"), (294, "o", "t"), (330, "i", "k"), (349, "e", "h"), (392, "a", "s"), (330, "u", "t")]
TRACKERS = [PyinTracker(), YinTracker(), SHSTracker()]


def _melody(detune=0.0, shift=0.0, seed=0):
    notes = [SynthNote(f * (2 ** (detune / 1200) if k == 1 else 1), 0.45, gap_after=0.15, vowel=v, consonant=c,
                       onset_shift_s=shift if k == 2 else 0.0) for k, (f, v, c) in enumerate(NOTES)]
    return melody(notes, sr=44100, seed=seed).audio


def robustness(out: Path) -> dict:
    print("1) robustness grid")
    tgt = _melody()
    items = []
    for i, (d, sh) in enumerate([(-40, -0.06), (-15, 0.0), (10, 0.05), (35, 0.09)]):
        u = _melody(d, sh, seed=i + 1)
        n = max(len(tgt), len(u))
        t, uu = np.pad(tgt, (0, n - len(tgt))), np.pad(u, (0, n - len(u)))
        target = analyze(Recording(t, 44100, Provenance.REFERENCE), trackers=TRACKERS).unwrap()
        items.append(GridItem(f"perf{i}", uu, 44100, target, t))
    rep = run_robustness(items, robustness_grid("short"), analyzer=lambda r: analyze(r, trackers=TRACKERS), n_boot=100)
    lines = [ln for ln in rep.summary() if "intonation_offset/1" in ln or "onset_timing/2" in ln]
    for ln in lines:
        print("   " + ln)
    (out / "robustness.txt").write_text("\n".join(rep.summary()) + "\n", encoding="utf-8")
    ino, ons = rep.items["pitch/intonation_offset/1"], rep.items["rhythm/onset_timing/2"]
    return {"intonation ICC(2,1)": ino.icc21, "intonation MDC95 (cents)": ino.mdc95, "onset ICC(2,1)": ons.icc21,
            "onset MDC95 (ms)": ons.mdc95, "conditions": len(robustness_grid("short"))}


def export_and_profile(out: Path, tiny: bool) -> None:
    print("2) ONNX export (verified against PyTorch at two input sizes)")
    cfg = AutoencoderConfig.tiny() if tiny else AutoencoderConfig()
    torch.manual_seed(0)
    ae = GyeolAutoencoder(cfg).eval()
    heads = FrameHeads(40, default_tasks(), hidden=32 if tiny else 128)
    e2e = E2E(n_blocks=1, inter_layers=1, en_out=4) if tiny else E2E()
    results = [export_heads(heads, 40, out / "onnx"), export_rmvpe(e2e.eval(), out / "onnx")] + export_autoencoder(ae, out / "onnx")
    for r in results:
        print(f"   {r.name:16s} {'ok ' if r.passed else 'FAIL'} max |Δ| {max(r.max_abs_diff, r.max_abs_diff_other_length):.1e}")

    print("   latency (median of 3 runs; RTF = compute s / audio s)")
    frames = lambda d: int(d * cfg.sr / cfg.hop) + 1  # noqa: E731
    tables = []
    g = torch.Generator().manual_seed(0)

    def voc_inputs(d):
        T = frames(d)
        n = (T - 1) * cfg.hop
        return (torch.randn(1, T, cfg.n_mels, generator=g), -torch.rand(1, T, cfg.n_ap, generator=g) * 20, torch.full((1, T), 220.0),
                torch.zeros(1, T), torch.randn(1, n, generator=g), torch.randn(1, n, generator=g))

    names = ["mel", "ap_db", "f0_hz", "rough", "jitter_noise", "source_noise"]
    tables.append(profile("vocoder torch", lambda d, a=None: (lambda a=voc_inputs(d): ae.vocoder.harmonic_path(*a[:4], source_noise=a[4:])),
                          n_runs=3))
    tables.append(profile_onnx(str(out / "onnx" / "vocoder.onnx"), lambda d: {k: v.numpy() for k, v in zip(names, voc_inputs(d))}, n_runs=3))
    mel_frames = lambda d: 32 * ((int(d * 100) + 1 + 31) // 32)  # noqa: E731
    with torch.no_grad():
        tables.append(profile("rmvpe torch", lambda d: (lambda m=torch.randn(1, 128, mel_frames(d)): e2e(m)), n_runs=3))
    tables.append(profile_onnx(str(out / "onnx" / "rmvpe.onnx"), lambda d: {"log_mel": np.random.randn(1, 128, mel_frames(d)).astype(np.float32)},
                               n_runs=3))
    stages = pipeline_profile(_melody(), 44100, trackers=TRACKERS, n_runs=2)
    dur = stages.pop("audio_seconds")
    text = "\n\n".join(t.table() for t in tables)
    text += f"\n\n# signal layer, {dur:.1f} s of audio (DSP trackers)\n" + "\n".join(
        f"{k:16s} {v * 1000:9.1f} ms  RTF {v / dur:.3f}" for k, v in sorted(stages.items(), key=lambda kv: -kv[1]))
    text += f"\n\nenvironment: {tables[0].environment}\n"
    (out / "latency.md").write_text(text, encoding="utf-8")
    for t in tables:
        r10 = t.rows[-1]
        print(f"   {t.name:34s} 10 s audio: {r10.median_ms:8.1f} ms (RTF {r10.rtf:.3f})")


def model_card(out: Path, metrics: dict) -> None:
    print("3) model card")
    heads = FrameHeads(40, default_tasks(), hidden=64)
    save_checkpoint(out / "heads_demo.pt", heads.state_dict(), name="gyeol-heads-demo", sources=["gyeol_synthetic"],
                    config={"hidden": 64, "tasks": sorted(default_tasks())}, profile=Profile.COMMERCIAL)
    card = card_from_checkpoint(
        out / "heads_demo.pt", component="attribute heads (register / phonation / laryngeal)",
        architecture="temporal-conv trunk + linear heads, temperature scaling, Mahalanobis OOD", profile=Profile.COMMERCIAL,
        intended_use=["Frame-level phonation posteriors that feed coaching explanations (tentative wording)."],
        evaluation=[EvalTable("explanation robustness (short grid)", metrics, synthetic=True, data="4 synthetic performances")],
        limitations=["Untrained demo checkpoint; all evaluation shown is on synthetic data.",
                     "License tags come from gyeol's registry and were not re-verified upstream by this code."])
    (out / "model_card.md").write_text(card.to_markdown(), encoding="utf-8")
    print(f"   wrote {out / 'model_card.md'} (validation problems: {card.validate() or 'none'})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("gyeol_m8_out"))
    ap.add_argument("--tiny", action="store_true", help="CPU-test-sized networks (fast)")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = robustness(args.out)
    if not args.skip_export:
        export_and_profile(args.out, args.tiny)
    model_card(args.out, metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())

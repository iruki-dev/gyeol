# gyeol (결) v2

An interpretable singing-voice model for Korean vocal coaching.

A user sings a phrase along with a target song. gyeol explains how the take differs from the target as:

- a **time warp** `τ(t)`, which becomes rhythm feedback;
- **attribute differences** `Δc(t)`, which become pitch, ornament, phonation and diction feedback;
- a remainder it reports as **"cannot judge"**, rather than guessing.

Practice demos are rendered only in the **user's own, consented voice**.

> v2 is a rewrite in progress. v0.1 (a DSP feature engine) is in git history, and its features survive as weak labels
> in `gyeol.dsp`.

## Representation

| Layer | Content | Rate |
|---|---|---|
| Global | `singer` (timbre) and `env` (mic, room, mix) vectors | per recording (M4) |
| Interpretable | attribute curves `c(t)` with per-frame confidence: pitch, loudness, periodic/aperiodic balance, vibrato, glides, register/phonation and phonetic posteriors | frame |
| Residual | low-dimensional `r(t)` for reconstruction only, with attribute leakage suppressed | frame (M4) |

## Milestones

| | Scope | Status |
|---|---|---|
| M0 | skeleton, `FrameGrid`, containers, license enforcement, consent types, store, verification toolkit, CI | done |
| M1 | signal layer: IO, latency, quality checks, pitch consensus, loudness/aperiodicity, pitch-derived curves, alignment, pitch/rhythm/ornament explanations, demo CLI | done |
| M2 | dataset adapters, augmentation, paired loader | done |
| M3 | phonation/register/diction heads, calibration, probing (machinery; trained heads need real data) | done |
| M4 | encoders, acoustic model, source-filter vocoder, losses, leakage/benchmark harness (untrained: needs data + GPU) | done |
| M5 | full explanation, audibility, consent-gated own-voice demos with AI labelling (DSP renderer; neural renderer needs M4 weights) | done |
| M6 | coaching policy (fitted thresholds, priority, fading, self-assessment), onboarding, health guard, practice mapping | done |
| M7 | discovery: TopK SAE, feature matching, conditional directions, transfer tests + promotion registry, residual monitor | done |
| M8 | RMVPE reimplementation, regression heads, robustness grid, expert-benchmark adapter + agreement/retention protocols, listening calibration, model cards, ONNX export + latency profile | done |

See `docs/milestones/`.

## Try it

```bash
python examples/coach_demo_v2.py --synthetic --out /tmp/gyeol_demo
# with audibility per item and an AI-labelled stepwise demo in the (synthetic) user's own voice
python examples/coach_demo_v2.py --synthetic --audibility --render-demo --out /tmp/gyeol_demo
# coaching: fit display thresholds (synthetic knob recovery), then coach each take as an attempt
python examples/fit_thresholds.py --synthetic --out /tmp/gyeol_demo/thresholds.json
python examples/coach_demo_v2.py --synthetic --coach /tmp/gyeol_demo/thresholds.json --noticed pitch
# discovery: SAE, conditional directions, transfer tests and promotion
python examples/discover_demo.py --out /tmp/gyeol_discover
# evaluation and hardening: robustness grid, ONNX export + latency, model card (needs the [onnx] extra)
python examples/evaluate_and_export.py --out /tmp/gyeol_m8
gyeol profile --dsp-only --explain
# real-recording evaluation on your own folder (manifest.jsonl; see gyeol.eval.realset for the format)
gyeol eval realset /path/to/realset --split held_out --per-tracker
# optional: vocal separation weights for analyze(separation="auto") (shows the license, asks first)
gyeol fetch bs_roformer_viperx_ep317
```

## Train on a CPU (or a GPU)

```bash
pip install -e ".[train]"            # PyYAML + torchaudio
# prepare data once (separation, pitch, curves, features; resumable, failures logged)
gyeol prepare --manifest data/manifests/vocalset.json --out runs/cache
# every task has a smoke preset (minutes on a laptop CPU, runs in CI) and a long-run preset
gyeol train heads --config configs/cpu-smoke/heads.yaml
gyeol train autoencoder --config configs/cpu-full/autoencoder.yaml   # days; Ctrl-C saves, then:
gyeol train autoencoder --config configs/cpu-full/autoencoder.yaml --resume
```

Tasks: `heads`, `autoencoder`, `vocoder`, `pitch` (RMVPE), `ssl` (fine-tuning).
- Each component is `freeze`, `finetune` or `scratch`. Initial weights load from local checkpoints through the license
  gate.
- `device: auto` picks CUDA when present, otherwise CPU in float32.
- Runs log CSV with an ETA, validate on held-out singers, stop early, keep the best checkpoint, and end with a report
  on unseen singers.

Revision notes: `docs/revisions/` (A: analysis path, B: CPU training).

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"        # add ",onnx" for ONNX export
pytest
```

## Licensing and rights

- Every dataset and checkpoint carries a `LicenseTag`. Under the commercial profile, non-commercial, copyleft and
  unknown assets are refused.
- Weights are never downloaded automatically. `gyeol fetch <name>` shows the license and asks for confirmation first.
- There is no API that synthesises a target singer's voice. Rendering requires a `ConsentedVoice` built from the
  user's own recording and consent token. Renderers also check that the take belongs to that user.
- Every generated waveform is labelled as AI-generated (metadata tags and a JSON sidecar) and passes through a
  pluggable watermark hook.
- Voice-derived data is treated as sensitive biometric information. Storage needs consent, raw audio is deleted after
  feature extraction by default, and `store.delete_user` removes everything.

## License

MIT (code). Third-party data and weights keep their own licenses.

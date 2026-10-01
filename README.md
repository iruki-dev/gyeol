# gyeol (결) v2

An interpretable singing-voice model for Korean vocal coaching.

A user sings a phrase along with a target song. gyeol explains how the take differs from the target as:

- a **time warp** `τ(t)`, which becomes rhythm feedback;
- **attribute differences** `Δc(t)`, which become pitch, ornament, phonation and diction feedback;
- a remainder it reports as **"cannot judge"**, rather than guessing.

Practice demos re-render the **user's own take** with the explained differences corrected.

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
| M0 | skeleton, `FrameGrid`, containers, asset list, store, verification toolkit, CI | done |
| M1 | signal layer: IO, latency, quality checks, pitch consensus, loudness/aperiodicity, pitch-derived curves, alignment, pitch/rhythm/ornament explanations, demo CLI | done |
| M2 | dataset adapters, augmentation, paired loader | done |
| M3 | phonation/register/diction heads, calibration, probing (machinery; trained heads need real data) | done |
| M4 | encoders, acoustic model, source-filter vocoder, losses, leakage/benchmark harness (untrained: needs data + GPU) | done |
| M5 | full explanation, audibility, stepwise demos in the user's voice (DSP renderer; neural renderer needs M4 weights) | done |
| M6 | coaching policy (fitted thresholds, priority, fading, self-assessment), onboarding, health guard, practice mapping | done |
| M7 | discovery: TopK SAE, feature matching, conditional directions, transfer tests + promotion registry, residual monitor | done |
| M8 | RMVPE reimplementation, regression heads, robustness grid, expert-benchmark adapter + agreement/retention protocols, listening calibration, model cards, ONNX export + latency profile | done |

See `docs/milestones/`.

## Try it

```bash
python examples/coach_demo_v2.py --synthetic --out /tmp/gyeol_demo
# with audibility per item and a stepwise demo re-rendered from the (synthetic) user's take
python examples/coach_demo_v2.py --synthetic --audibility --render-demo --out /tmp/gyeol_demo
# coaching: fit display thresholds (synthetic knob recovery), then coach each take as an attempt
python examples/fit_thresholds.py --synthetic --out /tmp/gyeol_demo/thresholds.json
python reference_service/examples/coach_session_demo.py --synthetic --coach /tmp/gyeol_demo/thresholds.json --noticed pitch
# discovery: SAE, conditional directions, transfer tests and promotion
python examples/discover_demo.py --out /tmp/gyeol_discover
# evaluation and hardening: robustness grid, ONNX export + latency, model card (needs the [onnx] extra)
python examples/evaluate_and_export.py --out /tmp/gyeol_m8
gyeol profile --dsp-only --explain
# real-recording evaluation on your own folder (manifest.jsonl; see gyeol.eval.realset for the format)
gyeol eval realset /path/to/realset --split held_out --per-tracker
# optional: vocal separation weights for analyze(separation="auto") (downloads and verifies the sha256)
gyeol fetch bs_roformer_viperx_ep317
```

## Public API

```python
from gyeol import api

target = api.analyze("guide.wav", lyrics="사랑해요 그대").unwrap()
x, sr = api.load_audio("take.wav")
take = api.analyze(x, sr, reference="guide.wav").unwrap()   # latency refined against the guide
exp = api.compare([take], target).unwrap()
api.to_json(exp, "explanation.json")          # versioned JSON: gyeol.explanation v1 (schemas in gyeol/resources/schema)
# stepwise demo: the take re-rendered with one item corrected, then 50 % / 100 % toward the target (plain audio)
demo = api.render_demo(x, sr, take, exp, target, out_dir="demo/").unwrap()
run = api.train("configs/cpu-smoke/heads.yaml")
```

`gyeol` is stateless and policy-free: it analyses, compares and renders the audio it is given. Consent,
labelling of generated audio, storage and deletion, and coaching sessions (attempt history, feedback fading) are
the application's decisions; the reference service package `reference_service/` (`gyeol_service`) shows one way
to implement them.

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
- Each component is `freeze`, `finetune` or `scratch`. Initial weights load directly from local files; checkpoints
  and model cards record the datasets and weights they were built from.
- `device: auto` picks CUDA when present, otherwise CPU in float32.
- Runs log CSV with an ETA, validate on held-out singers, stop early, keep the best checkpoint, and end with a report
  on unseen singers.

Separating songs: separate each target song once at upload and cache it by content
(`api.separate_target(song, sr, cache_dir=..., background=True)`). Then analyse the song with `separated=` and user
takes with `target=`. Headphone takes are separated only when bleed is detected.

Revision notes: `docs/revisions/` (A: analysis path, B: CPU training, C: library boundary, D: follow-up decisions,
E: licensing and consent handling).

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"        # add ",onnx" for ONNX export
pytest
```

## Licenses

gyeol's code is MIT-licensed. The third-party models and datasets it can use keep their own licenses, listed below
as stated upstream (`gyeol licenses` prints the same list from `gyeol.core.assets`). Nothing is downloaded
automatically; `gyeol fetch <name>` downloads an asset that has a URL and verifies its SHA-256.

Models and weights:

| Name | License | Source |
|---|---|---|
| `bigvgan_v2` | MIT | NVIDIA BigVGAN |
| `vocos` | MIT | Vocos |
| `dac` | MIT | Descript Audio Codec |
| `hubert_fairseq` | MIT | fairseq HuBERT |
| `contentvec` | MIT | ContentVec |
| `rmvpe` | Apache-2.0 | RMVPE (Wei et al., Interspeech 2023) |
| `fcpe` | MIT | torchfcpe |
| `swiftf0` | MIT | SwiftF0 |
| `roformer_community` | MIT (as listed) | community Mel/BS-RoFormer weights |
| `bs_roformer_viperx_ep317` | not stated upstream (listed with community RoFormer weights as MIT) | viperx BS-RoFormer vocal model, UVR public model repository (sha256 pinned) |
| `openvpi_nsf_hifigan` | CC-BY-NC-SA-4.0 | openvpi vocoders |
| `openvpi_pc_nsf_hifigan` | CC-BY-NC-SA-4.0 | openvpi vocoders |

Datasets:

| Name | License | Source |
|---|---|---|
| `vocalset` | CC-BY-4.0 | VocalSet (Wilkins et al., ISMIR 2018) |
| `gtsinger` | CC-BY-NC-SA-4.0 | GTSinger (NeurIPS 2024 Datasets and Benchmarks) |
| `popbutfy` | CC-BY-NC-SA | PopBuTFy |
| `csd` | CC-BY-NC-SA-4.0 | CSD |
| `opencpop` | CC-BY-NC | Opencpop |
| `vocalcoachbench` | mixed per source (Smule Research Data License, CC-BY-NC-SA-4.0, CC-BY(-SA)-4.0) | VocalCoachBench |
| `aihub_473_guide_vocal` | AI Hub terms | AI Hub 다음색 가이드보컬 #473 |
| `aihub_465_multi_singer` | AI Hub terms | AI Hub 다화자 가창 #465 |
| `vocadito` | CC-BY-4.0 | vocadito (Bittner et al., 2021) |
| `mir1k` | not stated in the distribution | MIR-1K (Hsu & Jang, 2010) |

Code gyeol does not include or require: `so_vits_svc` (AGPL-3.0), `pesto` (LGPL-3.0).

You are responsible for complying with these licenses and with applicable law.

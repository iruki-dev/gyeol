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
| M5 | full explanation, audibility, consent-gated demos | planned |
| M6 | coaching policy, onboarding, health guard | planned |
| M7 | discovery on the residual | planned |
| M8 | benchmarks, robustness, model cards, ONNX | planned |

See `docs/milestones/`.

## Try it

```bash
python examples/coach_demo_v2.py --synthetic --out /tmp/gyeol_demo
```

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
pytest
```

## Licensing and rights

- Every dataset and checkpoint carries a `LicenseTag`. Under the commercial profile, non-commercial, copyleft and
  unknown assets are refused.
- Weights are never downloaded automatically. `gyeol fetch <name>` shows the license and asks for confirmation first.
- There is no API that synthesises a target singer's voice. Rendering requires a `ConsentedVoice` built from the
  user's own recording and consent token.
- Voice-derived data is treated as sensitive biometric information. Storage needs consent, raw audio is deleted after
  feature extraction by default, and `store.delete_user` removes everything.

## License

MIT (code). Third-party data and weights keep their own licenses.

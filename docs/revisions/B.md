# Revision B — CPU training

Scope: `gyeol.train` (runner, tasks, device, state, stream, components, config), `gyeol.data.prepare`, `gyeol train`
and `gyeol prepare`, and `configs/cpu-smoke` / `configs/cpu-full`. Tests: `tests/test_rev_b.py` (48 tests, ~50 s on 4
CPU threads). CI runs every `cpu-smoke` preset plus an interrupted and resumed run.

## Changes

### B1 Device handling
- `train.device.resolve_device("auto" | "cpu" | "cuda", threads, mixed_precision)`:
  - `auto` resolves to CUDA when available, otherwise CPU;
  - the CPU path is always float32 (autocast only on CUDA, and only when asked);
  - `threads` sets `torch.set_num_threads`.
- Every `nn.Module` class in gyeol has a CPU forward/backward test (`test_every_module_runs_forward_and_backward_on_cpu`):
  - covered: heads, encoders, acoustic model, vocoder, autoencoder (full generator objective), discriminators,
    RMVPE, BS-RoFormer, SAE and the SSL architecture;
  - each test checks float32, finite loss and finite gradients on ≥ 80 % of parameters;
  - `test_module_coverage_is_complete` fails when a new module class has no such test.

### B2 Training runner
- `gyeol train <task> --config <yaml> [--resume] [--device] [--threads] [--out] [--max-steps] [--set key=value]` runs
  five tasks, each built on the existing steps and losses:

  | task | components | objective |
  |---|---|---|
  | `heads` | heads | `train.heads` task losses; afterwards temperatures and OOD detectors fitted on the validation singers (`calibrate_heads`, factored out of `train_heads`) |
  | `autoencoder` | singer / env / residual encoders, acoustic model, vocoder, discriminators | `generator_step` (+ `discriminator_step` on vocoded steps); timbre-shifted copy for the residual encoder |
  | `vocoder` | vocoder, discriminators | mel L1 + multi-resolution STFT + LSGAN + feature matching on cached band aperiodicity |
  | `pitch` | pitch (RMVPE) | BCE on Gaussian-blurred 360-bin targets from the cached consensus pitch |
  | `ssl` | ssl, heads | head losses back-propagated into a torchaudio HuBERT/WavLM base (fetched weights) or a tiny wav2vec2 for tests |

- Configs are YAML, and unknown keys are errors.
- During the generator step, the discriminators' parameters are excluded from gradients, so the adversarial loss
  never updates them.

### B3 Interrupt and resume
- `train.state.save_training_state` writes one file atomically (temp file, fsync, `os.replace`). It holds:
  - every module, optimizer and LR scheduler;
  - the Python, NumPy and Torch (and CUDA) RNG states;
  - the run position: epoch, optimizer step, micro-step, data position in the epoch, best score and step,
    early-stopping counter, elapsed time and validation history.
- When the state is written:
  - every `run.save_every` steps;
  - when the run ends;
  - on SIGINT/SIGTERM, after the current step finishes (a second signal aborts).

  `--resume` continues from it. Starting again on an existing run folder without `--resume` is refused.
- Exactness, as implemented:
  - The data order and crops depend only on `(seed, epoch, batch index)`.
  - Validation and sample rendering run under `torch.random.fork_rng`.
  - The DataLoader gets its own generator. Otherwise, creating the iterator draws from the global torch RNG, and a
    resumed run, which creates its iterator mid-epoch, would see different dropout masks. This bug was found by the
    test.
  - The state is only written at optimizer-step boundaries, so no partial gradient is lost.
- Test: for every task, a run interrupted after 4 of 6 steps (between periodic saves) and then resumed ends with
  bit-identical parameters, the same best step and the same test metrics as an uninterrupted run with the same seed
  (`torch.equal` on every tensor).

### B4 Memory
- Gradient accumulation: `optim.grad_accum`; the loss is scaled by `1/accum`.
- Random-length crops: `data.crop_frames: [min, max]`. One length is drawn per batch, so no padding is wasted.
- Gradient checkpointing: `checkpointing: [component, ...]`.
  - Each block of a component's `ModuleList`s is recomputed in backward. The bound `forward` is wrapped, so state
    dicts are unchanged.
  - BS-RoFormer uses its own switch.
  - Tested to give the same gradients as without checkpointing (acoustic model, RMVPE).
- `train.stream.StreamingDataset` (an `IterableDataset`) streams prepared items from disk:
  - it reads one `.npz` per item per batch and never holds the dataset in memory;
  - it yields collated batches, and worker *w* of *W* produces batches *b ≡ w (mod W)*, so the order is identical
    for any worker count (tested with 0 and 2).

### B5 Per-component training mode
- `components: {name: freeze | finetune | scratch}` applies to `ssl`, `pitch`, `residual_encoder`,
  `singer_encoder`, `env_encoder`, `acoustic`, `vocoder`, `discriminators` and `heads`:
  - frozen components get no gradients, stay in eval mode and are left out of the optimizer;
  - `finetune` needs an `init` and trains at `finetune_lr_scale` × lr (default 0.1);
  - unknown or mismatched component names are errors.
- `init: {name: {path, asset?, prefix?}}` loads initial weights from local files only, after the license gate:
  - a gyeol checkpoint goes through `load_checkpoint` under the run's profile, and its lineage is recorded as a
    parent;
  - a registered third-party checkpoint (`asset: rmvpe`, `hubert_fairseq`, …) goes through `load_third_party`;
  - a missing file names the `gyeol fetch` command.
- Release checkpoints (`best.pt`) embed:
  - the most restrictive license of the datasets and parents (parents now feed into the tag);
  - the parents' names and sources;
  - the task, step and component modes.

### B6 Data preparation
- `gyeol prepare --manifest m.json [--manifest …] --out cache` (and `--synthetic N`) runs, per item:
  - load and resample;
  - separation (`analyze(separation=...)`; the separated vocal is cached);
  - consensus pitch and attribute curves;
  - band aperiodicity (for the vocoder);
  - DSP frame features, plus SSL features with `--features ssl --ssl-checkpoint --ssl-asset`.
- It works over any adapter manifest (VocalSet, GTSinger, AI Hub, own recordings), and the license gate runs before
  any item is read.
- Resumability and logging:
  - one `items/<id>.npz` (written atomically) plus an `items/<id>.done` marker per item; re-runs skip finished
    items;
  - `failures.jsonl` records path, reason and a short traceback for each failure, and a failure never stops the
    batch;
  - `prepare.json` pins the settings, so mixing rates in one cache is refused;
  - `index.jsonl` lists the prepared items.
- The runner prepares configured manifests or synthetic data on first use. It re-checks every dataset's license
  under the run's profile, since a cache can be shared between profiles.

### B7 Validation
- The split is singer-disjoint: train / val / test by a seeded hash of the singer id. Every non-empty split gets at
  least one singer, the split is saved to `split.json`, and overlap is an error.
- Validation runs every `run.val_every` steps. Early stopping triggers after `run.patience` validations without
  improvement. The best weights are released as `best.pt`.
- At the end of a run:
  - the best weights are evaluated on the unseen (test) singers;
  - the head tasks are then calibrated on the validation singers and evaluated (accuracy, ECE) on the test singers;
  - results go to `report.json` and `report.md`.

### B8 Progress and presets
- Progress output:
  - an ETA (moving average of step time) on stdout;
  - `train_log.csv`: step, epoch, elapsed, lr, ETA and every loss term;
  - `val_log.csv`.
- Generative tasks write audio samples (generated and reference) with a `samples.json` sidecar. It marks them as
  AI-generated reconstructions of validation audio and records the profile.
- `configs/cpu-smoke/<task>.yaml` uses synthetic data (no people) and finishes in seconds to a minute per task on a
  laptop CPU. CI runs all five, then interrupts and resumes one.
- `configs/cpu-full/<task>.yaml` is sized for about a week of unattended CPU time per task, from the measurements
  below.

## Measurements

Machine: 4 CPU threads, float32, PyTorch 2.x CPU build.

`cpu-smoke`:
- Synthetic data: 54 items, 6 singers, 22.05 kHz, hop 256. Preparing the cache takes 12 s once.
- Per-task results:

  | task | parameters | steps | wall time (incl. start-up) | first → last validation |
  |---|---|---|---|---|
  | heads | 0.01 M | 40 | 21 s (incl. preparation) | loss 1.41 → 0.75; register acc 0.75 → 0.85 |
  | vocoder | 0.02 M | 12 | 12 s | mel L1 4.52 → 4.27 |
  | pitch | 1.74 M | 10 | 12 s | BCE 0.35 → 0.06 |
  | autoencoder | tiny | 16 | 17 s | mel L1 3.47 → 2.52 |
  | ssl | 0.05 M | 12 | 10 s | loss 1.77 → 1.72 |

- The smoke heads run, calibrated on its validation singer and tested on an unseen one, reaches:
  - register accuracy 0.82, ECE 0.059;
  - phonation accuracy 0.74.

  These are synthetic numbers; they show that the pipeline works, not model quality.

`cpu-full` model sizes, steps 1–4 on a synthetic 44.1 kHz / hop 512 cache. One measurement ran while another
process was busy, so treat these as rough:

| task | parameters | s / optimizer step | batch × accum, crop |
|---|---|---|---|
| heads | 0.10 M | ~0.1 | 16 × 1, 128–384 frames |
| vocoder (256 ch) | 2.31 M | ~30 (contended) | 8 × 2, 24–48 frames |
| pitch (RMVPE) | 90.4 M | ~9 | 8 × 2, 96–192 frames |
| autoencoder | 4.97 M | ~17–31 vocoded, ~3 otherwise | 8 × 4, 64–128 frames |
| ssl (HuBERT base) | 95.0 M | ~8 | 4 × 4, 96–192 frames |

## Open questions
1. **Vocoder on a CPU.** Even at 128 channels, a week of CPU time is a small fraction of the steps a HiFi-GAN-class
   vocoder normally gets. Options:
   - fine-tune from existing weights. The openvpi weights are CC BY-NC-SA, so they are only usable under the
     personal profile (revision C3);
   - accept a weaker vocoder for demos;
   - use a GPU for this one task.
2. **Pitch targets.** The `pitch` task learns from the cached consensus pitch, which is itself DSP. Fine-tuning RMVPE
   from its reference weights on these pseudo-labels may teach it the DSP trackers' errors. Two options:
   - add annotated f0 from the realset annotations (A6) or from a dataset with ground-truth f0;
   - restrict the targets to high-confidence frames.

   Which f0 source do you want treated as truth?
3. **SSL fine-tuning objective.** The `ssl` task fine-tunes through the attribute heads only. With few labelled
   singers this can overfit the encoder. The preset freezes nothing and uses 0.1× lr for the encoder. Freezing the
   lower layers is a possible next step.
4. **Environment heads.** The autoencoder task trains without the environment-class supervision of M4: the prepared
   cache holds clean data and no augmentation labels. On-the-fly augmentation with labels, seeded per batch like the
   crops, is the natural extension.
5. **Workers and exactness.** The data order is exact for any worker count. Resume exactness is tested with
   `num_workers: 0`. With workers it holds as long as nothing in a worker touches global RNG state; nothing does
   today.

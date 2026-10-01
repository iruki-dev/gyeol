# Changelog

## 2.0.0.dev0 (unreleased)

gyeol v2 is a rewrite: an interpretable encoder–decoder singing-voice model for vocal coaching. It replaces the
v0.1 DSP feature engine.

### Revision E — licensing and consent handling (see `docs/revisions/E.md`)
- Breaking:
  - The license gate is replaced by a plain asset list, `gyeol.core.assets` (name, license, source, URL,
    sha256).
  - `Profile`, `LicenseTag`, `require_allowed` and every `profile=` parameter are removed; assets load directly
    from local paths.
  - `gyeol fetch` downloads and verifies the SHA-256 (no prompt); `gyeol licenses` lists the assets.
- Checkpoints and model cards record the datasets and weights they were built from (`describe()`), as
  information.
- Breaking:
  - `ConsentToken`, `ConsentedVoice`, `Provenance`, `Purpose`, `OwnVoice`, the AI-label metadata
    (`demo.label`) and the watermark module are removed.
  - Renderers take a `demo.Take(recording, rep)` and return plain audio.
  - `api.render_demo(audio, sr, rep, explanation, target, …)`; `api.compare(audibility=(audio, sr))`.
- Schemas: `gyeol.representation` v2 and `gyeol.demo` v2 drop provenance and AI-label fields; v1 is still read.
- README: a "Licenses" section; consent and labelling are documented as application policy (the reference
  service keeps its own consent store).
- Supersedes revision D's personal-only and copyleft opt-in items.

### Revision D — follow-up decisions (see `docs/revisions/D.md`)
- D1:
  - Target songs are separated once with BS-RoFormer, in the background, and cached by content hash
    (`separate_target`, `SeparationCache`, `SeparationQueue`, `api.separate_target(background=True)`).
  - Headphone takes are separated only when accompaniment bleed is detected against the cached accompaniment,
    with a selectable light separator (`SEPARATORS`, default `backing`).
  - `detect_bleed` uses a phase-randomised null and reports delay-peak prominence.
- D2: BS-RoFormer weights are SHA-256-verified on every load against the registry's pinned hash (hash computed:
  `5b84f37e…15aa`). Pinning it in the registry and the personal-only restriction are pending (see the note).
- D3:
  - Exact-f0 resynthesised copies (`gyeol prepare --resynthesize hnm|vocoder`) are the pitch model's primary
    ground truth; DSP consensus is a weak label on confident frames only.
  - Vocadito and MIR-1K adapters, `gyeol eval pitch`, and `model.eval_sets` for pitch runs, all license-gated.
- D4 (copyleft opt-in): pending (see the note).

### Revision C — library boundary (see `docs/revisions/C.md`)
- C1:
  - User state moved out of `src/gyeol` into `reference_service/` (`gyeol_service`): the consent / storage /
    deletion store, the coaching session (fading schedule, attempt history, self-assessment, summaries), and the
    running phonation-time and fatigue history.
  - The library keeps the stateless parts:
    - consent types and guards;
    - `coach.thresholds`, `coach.priority` and the practice map;
    - health measures, now pure functions `phonation_warnings` and `fatigue_flags`;
    - onboarding scoring.
  - **Breaking:** `gyeol.store` and `gyeol.coach.session` are gone; import them from `gyeol_service`.
- C2:
  - `gyeol.api`: `analyze`, `compare`, `render_demo` and `train`, plus `OwnVoice`.
  - Versioned JSON (`gyeol.representation`, `gyeol.explanation`, `gyeol.demo` v1) with JSON Schemas; biometrics are
    left out by default.
  - `examples/coach_demo_v2.py` uses `gyeol.api` only; the session demo moved to
    `reference_service/examples/coach_session_demo.py`.
  - Rendering checks the consent token against the take's owner digest.
- C3: `Profile.PERSONAL` allows non-commercial assets. Analyses, explanations, demos, training samples, reports and
  checkpoints carry the profile, and personal-profile checkpoints are refused under `commercial`. The commercial
  profile is unchanged.

### Revision B — CPU training (see `docs/revisions/B.md`)
- B1:
  - `device="auto"` resolves to CUDA when available, otherwise CPU in float32; thread count configurable.
  - A CPU forward/backward test for every module, plus a coverage check that new modules get one.
- B2: `gyeol train <task> --config <yaml>` for `heads`, `autoencoder`, `vocoder`, `pitch` (RMVPE) and `ssl`
  (fine-tuning), built on the existing step functions and losses.
- B3:
  - Atomic training state (temp file + `os.replace`): modules, optimizers, schedulers, RNG states, epoch, step and
    data position.
  - Periodic saves, and saves on SIGINT/SIGTERM.
  - `--resume` reproduces an uninterrupted run bit for bit (tested for every task).
- B4:
  - Gradient accumulation.
  - Random-length crops.
  - Optional gradient checkpointing (`checkpointing: [component]`).
  - An `IterableDataset` that streams prepared items from disk and is deterministic across worker counts.
- B5: per-component `freeze | finetune | scratch`; initial weights from local gyeol or registered third-party
  checkpoints through the license gate; lineage recorded in released checkpoints (`parents`).
- B6: `gyeol prepare --manifest ...` runs separation, pitch, curves, band aperiodicity and DSP/SSL feature caching.
  It is resumable through per-item completion markers, and failures are logged with reasons.
- B7: singer-disjoint train/val/test split, early stopping, best-checkpoint release, and a final report on unseen
  singers. The heads and ssl tasks are also calibrated on the validation singers.
- B8:
  - ETA, `train_log.csv` and `val_log.csv`.
  - AI-labelled audio samples for the generative tasks.
  - `configs/cpu-smoke/` (run in CI) and `configs/cpu-full/`.

### Revision A — analysis path (see `docs/revisions/A.md`)
- A1:
  - `analyze(separation="auto" | "always" | "off")`, with `auto` as the default.
  - Pitch-informed accompaniment estimator.
  - BS-RoFormer local-checkpoint loader (`frontend.roformer`); `bs_roformer_viperx_ep317` registered for
    `gyeol fetch`.
  - Separation quality and residual accompaniment feed per-frame confidence; a graded penalty applies when
    accompaniment is suspected but nothing was separated.
- A2:
  - The octave/key relation is decided only with confident pitch on both sides; otherwise it is `None` with a
    reason, and the comparison is octave-invariant.
  - `gyeol eval realset --per-tracker`.
- A3:
  - Premise framework (`Premise`, `WithheldItem`, `Explanation.premises/withheld/comparison_mode`) for shared clock,
    octave relation, level chain, noise floor and interval set.
  - Content-based subsequence alignment (`WarpConfig.open_begin`) and relative onset timing when the clock is not
    shared.
- A4: Hangul syllabification independent of spaces and punctuation; `assign_syllables`; `explain(lyrics=...)`.
- A5:
  - Whole-contour comparison: `contour_deviation` and `transition_deviation`, labelled with detected events.
  - Demo edits for them.
  - Knob-recovery truth for fitting their coach thresholds.
- A6:
  - Real-recording manifest, singer-disjoint split and `gyeol eval realset`.
  - Synthetic stand-in set (`eval.realset_synth`).

### M8 — evaluation and hardening
- RMVPE reimplemented (reference module layout, strict key-by-key weight loading, chunked tracker in the
  consensus); weights are never bundled or downloaded.
- Regression heads: heteroscedastic Gaussian with held-out variance scaling; promoted continuous attributes train
  as heads.
- Robustness grid: noise, reverb, band-limit, codecs, separation artefacts, Bluetooth and bleed. ICC, MDC95, bias,
  per-condition |Δ|, item presence and operating thresholds for curves and explanation items.
- Expert-benchmark adapter (VocalCoachBench, research profile only; field-mapped loader; top-k label and segment
  scoring against a label-prior baseline); Fleiss'/Cohen's κ and coach-agreement analysis; protocols for coach
  agreement, the retention study and audibility calibration.
- Listening calibration of the audibility score and an optional perceptual floor (`ThresholdSet.audibility_floor`).
- Model cards built from checkpoint license lineage, with validation.
- ONNX export (opset 18, dynamo exporter, dynamic batch/time, ORT parity check at a second shape, provenance
  sidecar) for heads, RMVPE, acoustic model, vocoder harmonic path and singer encoder; latency profiles;
  per-stage analysis timings; `gyeol profile`; `[onnx]` extra (also installed in CI).
- Fixes:
  - Bluetooth augmentation no longer shifts pitch (piecewise-constant re-sync delays plus ppm drift instead of a
    resampled wandering delay).
  - Vocoder phase is accumulated in float64 and wrapped.
  - The invariance report ignores conditions where a dimension was never measured instead of dropping the
    dimension.
- `gyeol_synthetic` registry entry; `examples/evaluate_and_export.py`.

### M7 — discovery
- TopK sparse autoencoder with AuxK dead-latent revival, deterministic training and per-latent statistics.
- Matching of SAE latents to labels (AUROC) and to v0.1 DSP features (Spearman); novel-candidate list.
- Conditional directions from aligned on/off pairs per f0 × loudness × vowel cell, with f0 and loudness regressed
  out and per-singer statistics.
- Transfer tests on held-out singers, pitch bands and languages at the unit level; a promotion rule and a
  JSON-backed `PromotionRegistry` that refuses anything that did not pass.
- Residual-energy monitor: slice z-scores against a baseline and a Theil–Sen trend, flagging coverage gaps.
- `examples/discover_demo.py`: breathiness is rediscovered from frame-normalised log-mel shape, transfers to
  unseen singers and pitch bands, and is promoted; a raw-pitch control is rejected.

### M6 — coaching policy
- `gyeol.coach` (pure logic):
  - fitted display thresholds, where an item is shown only above max(MDC95, E95(confidence)), with provenance
    and no numeric fallback;
  - priority by confidence × audibility (or size over uncertainty) with the brief's tie order;
  - one primary and at most two secondary items, a fading schedule (full → half → quarter) driven by the
    stability of the error load, self-assessment before reveal, and session summaries.
- Practice mapping from a coach-authored data file (`resources/ko/practice.json`).
- SSAP/SPB-style onboarding scoring: production accuracy, precision and perception, routed without labelling
  anyone.
- Vocal-health guard: range/tessitura phrase check with transposition, beginner restrictions (rough, fry, belt,
  high chest), phonation time, within-session fatigue flags and a persistent medical-referral notice.
- `gyeol.eval.knob_recovery` and `examples/fit_thresholds.py`; `coach_demo_v2.py --coach`.
- `ExplanationItem.key`.

### M5 — full explanation and demo
- Explanation items for phonation (breathiness from the aperiodic ratio; tentative register and phonation-quality
  items from learned heads), dynamics (per-note loudness, phrase dynamic range) and phrase-level diction against the
  target singer's own realisation (laryngeal contrast, phone match).
- "Cannot judge" spans carry a reason: low confidence, low item confidence, or an unexplained residual remainder.
- `gyeol.demo`:
  - edits of the user's c(t) for every item kind, with scaling and composition;
  - feasible-range clamping (from onboarding or the user's own takes);
  - stepwise schedule (the selected item first, then partial steps toward the target style);
  - a harmonic-plus-noise DSP renderer and an M4 neural renderer. Both refuse anything but a `ConsentedVoice` and
    the same user's own take.
- AI labelling: every rendered waveform is watermarked through a pluggable hook (default: documented
  spread-spectrum mark with a detector) and carries AI-generated metadata (WAV INFO tags plus a JSON sidecar; user
  id stored only as a hash).
- Audibility per item: render with only that item corrected, then take an uncalibrated specific-loudness distance
  to the unedited render.
- `ltas_singer_vector`: a DSP timbre descriptor, so DSP rendering is consent-gated before a singer encoder is trained.
- Korean strings for the new categories and attributes and for the demo notices; the example gains `--audibility`
  and `--render-demo`.

### M4 — autoencoder
- Grid-aligned log-mel.
- Singer encoder (SupCon, provenance-carrying `SingerVector`), env encoder with augmentation-label heads, and a VIB
  residual encoder with gradient-reversal leakage adversaries (on r and on the singer vector).
- Conv-transformer acoustic model with condition dropout.
- Self-implemented NSF source-filter vocoder (harmonic, subharmonic/jitter and noise branches) and an optional
  BigVGAN adapter.
- Losses: MR-STFT and mel with consonant/low-energy weighting, MPD/MSD discriminators, feature matching,
  r re-encoding consistency.
- Vocoder benchmark per technique and consonant class; listening-test protocol template.

### M3 — attribute heads
- License-gated frozen SSL encoder adapters (local checkpoints only) and a DSP baseline encoder.
- Multi-task frame heads (register, phonation qualities, laryngeal contrast, phones) and a deterministic training
  loop.
- Temperature scaling on held-out singers; Mahalanobis OOD on the embedding and the inputs, reported as "unknown".
- Quality-factor confidence; learned curves integrated into `analyze`.
- Probing suite: grouped logistic/ridge probes, leakage tests, probe battery.
- Pitch consensus accepts strong harmonic salience as voicing evidence (breathy voices).
- Note segmentation merges only monotonic pitch sweeps; minimum durations for falls and scoops.

### M2 — data
- Dataset adapters (VocalSet, AI Hub #465/#473 with an explicit field map and printed conditions, GTSinger
  research-only with on/off pairing, consent-gated own recordings).
- Singer-disjoint splits.
- Labelled augmentation suite: noise, reverb, EQ, codec, compression, AGC, separation artefacts, Bluetooth jitter,
  timbre shift, pitch shift.
- Paired on/off loader with content-only alignment.
- `register()` for new license assets; `own_recordings` registered.

### M1 — signal layer
- IO: BS.1770 loudness, A-weighting, and latency calibration (loopback, tap-along, offline refinement).
- Frontend: raw-input clipping, SNR, channel bandwidth and codec cliff, backing-track bleed, T60 with status, and
  separation adapters.
- Pitch: tracker protocol, DSP trackers, optional SwiftF0/FCPE adapters, and an octave-aware consensus.
- Attribute curves (pitch centre, vibrato, loudness, periodic/aperiodic, subharmonics, content) and ornament events
  (scoop, fall, 꺾기, glide).
- Content-only banded DTW alignment and an explanation layer for pitch/rhythm/ornaments, with take consistency and
  "cannot judge".
- Korean resource strings and `examples/coach_demo_v2.py`.
- Regression tests for the eight v0.1 defects, all fixed.

### M0 — skeleton
- New package layout, `FrameGrid`, `Result`/`Status` and typed containers.
- `LicenseTag`/`Profile` registry, enforced in the dataset-manifest and checkpoint loaders.
- Consent types (`ConsentedVoice` is gated by provenance and consent) and a consent-aware store with deletion and
  raw-audio retention.
- Verification toolkit ported from v0.1.
- `gyeol licenses`, `gyeol fetch` (license shown, explicit confirmation), CI.

### Removed from v0.1
- The rule-based comparison thresholds, formant-ratio vocal-tract scaling and the untrained residual stub.
- The v0.1 DSP features survive only as weak labels and baselines in `gyeol.dsp`.

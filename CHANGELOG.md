# Changelog

## 2.0.0.dev0 (unreleased)

gyeol v2 is a rewrite: an interpretable encoder–decoder singing-voice model for vocal coaching. It replaces the
v0.1 DSP feature engine.

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

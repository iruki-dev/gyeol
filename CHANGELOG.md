# Changelog

## 2.0.0.dev0 (unreleased)

gyeol v2 is a rewrite: an interpretable encoder–decoder singing-voice model for vocal coaching. It replaces the
v0.1 DSP feature engine.

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

# Changelog

## 2.0.0.dev0 (unreleased)

gyeol v2 is a rewrite: an interpretable encoder–decoder singing-voice model for vocal coaching. It replaces the
v0.1 DSP feature engine.

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

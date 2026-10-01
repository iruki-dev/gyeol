# Revision E — licensing and consent handling

Brings the library in line with standard practice: license information is documented, and policy decisions
(consent, labelling of generated audio, license compliance) belong to the application that uses gyeol.

| item | status |
|---|---|
| 1. Replace the license gate with a plain asset list; `gyeol fetch` downloads and verifies checksums; assets load directly from local paths everywhere; remove `Profile`, `LicenseTag`, `require_allowed`; checkpoints and model cards record their sources as information | **done** |
| 2. `render_demo` and the renderers take the user's audio, its analysis and the edits, and return plain audio; remove `ConsentToken`, `ConsentedVoice`, `Provenance`, `Purpose`, the `OwnVoice` checks, the AI-label metadata and the watermark module | **done** |
| 3. README "Licenses" section with each third-party model and dataset and its license, plus the responsibility line | **done** |
| 4. `gyeol.api`, CLI, examples and tests updated; gate/consent-only tests removed; this note; full test suite and ruff pass | **done** |

## Changes

### 1. Licensing
- **Removed** `gyeol.core.license` (`LicenseTag`, `Profile`, `LicensedAsset`, `REGISTRY`, `lookup`, `decide`,
  `require_allowed`, `LicenseError`, `most_restrictive`, the AI Hub condition printing and the copyleft opt-in).
- **Added** `gyeol.core.assets`, a plain list:
  - `Asset(name, kind, license, source, url, sha256, notes)`, with `kind` one of `dataset`, `weights`, `code`;
  - `ASSETS`, `asset(name)` (`KeyError` for unlisted names), `add_asset(...)` to list your own models or data;
  - `describe(name)`, the provenance record written into checkpoints and model cards. Unlisted names are
    recorded as `kind: "unlisted"` rather than refused.
- `gyeol licenses` prints the list and the responsibility line. `gyeol fetch <name> [--dest PATH]` downloads an
  asset that has a URL to `<file>.part`, verifies the listed SHA-256, and only then moves it into place:
  - checksum mismatch → the file is deleted, exit code 4;
  - no URL listed → prints where to get it, exit code 3;
  - already present and verified → nothing is downloaded.

  There is no confirmation prompt and no `--profile` / `--yes` option.
- The `profile` parameter and every `require_allowed` call are gone from separation (`make_separator`,
  `default_separator`, RoFormer), pitch trackers (RMVPE, FCPE, SwiftF0), SSL encoders (`ssl_from_checkpoint`),
  the vocoder (BigVGAN), data (`open_manifest`, adapters, `prepare`), training (`TrainConfig`, `Trainer`,
  `load_initial_weights`, `save_checkpoint` / `load_checkpoint`) and evaluation (`evaluate_pitch`,
  `load_expert_benchmark`, robustness, realset, knob recovery, reconstruction). Every one of them loads from the
  local path it is given.
- BS-RoFormer: the pinned SHA-256 (`5b84f37e…15aa`) is in the asset list, and `RoFormerSeparator.from_checkpoint`
  still verifies it on every load. That is an integrity check, not a license check.
- Checkpoints: `CheckpointInfo(name, sources, config_hash, parents)`.
  - `sources` holds `describe()` of every dataset and weight file the checkpoint was built from (name, license,
    source, …).
  - `parents` lists the gyeol checkpoints it was initialised from, with their sources.
  - `load_checkpoint` reads plain state dicts too, with empty provenance.
- Model cards list the same sources in a "Provenance" table. A card is no longer validated against a license
  profile. `validate()` still checks the required out-of-scope statements (now including "synthesising or
  imitating a person's voice without their permission") and that synthetic-only evaluation is disclosed.
- Training reports carry `provenance: {sources, parents}` instead of `profile`.

### 2. Rendering
- **Removed** `gyeol.core.consent` (`Provenance`, `Purpose`, `ConsentToken`, `ConsentedVoice`,
  `require_consented_voice`, `ConsentError`), `gyeol.demo.label` (`save_labelled`, `read_label`, the AI-label
  sidecars) and `gyeol.demo.watermark` (`SpreadSpectrumWatermark`, `WatermarkHook`), plus `api.OwnVoice`,
  `api.ConsentToken` and `api.Purpose`.
- Containers no longer carry provenance or owners:
  - `Recording(audio, sr, recording_id, meta)`;
  - `SingerVector(vector, source_recording_id)`;
  - `Representation` without a `provenance` field.
  - `analyze` no longer takes `role` / `owner_id`.
- `demo.Take(recording, rep)` replaces `UserTake`; it only checks that the representation was analysed from that
  recording.
  - `DSPRenderer.render(take, edit, seed, frames)` and `NeuralRenderer.render(take, …)` return `Result[np.ndarray]`.
    The neural renderer decodes the take's own encoded singer vector.
  - `render_demo(take, explanation, target, selected, renderer, …)` returns a `Demo` whose `baseline` and
    `steps[i].audio` are plain arrays, each step with an `applied` description.
- `score_audibility(exp, take, target, renderer, …)`.
- `api.render_demo(audio, sr, rep, explanation, target, *, item, out_dir, renderer)` returns plain audio. With
  `out_dir` it writes plain WAVs and `demo.json`. `api.compare(..., audibility=(audio, sr))`.
  `api.take(audio, sr, rep)` builds the `Take`, re-applying the latency shift `analyze` applied.
- Schema versions:
  - `gyeol.representation` **v2** drops `provenance`;
  - `gyeol.demo` **v2** drops `ai_generated`, `profile`, the notice and the watermark description;
  - v1 documents are still read; `gyeol.explanation` stays v1.
- Training samples (`samples/…/samples.json`) no longer carry an AI-generated flag or profile.
- `PrepareConfig.resynth_skip_datasets` (default `("own_recordings",)`) stays as a configurable default: an
  application that wants exact-f0 copies of its own recordings sets it to `()`.
- The Korean resource `demo.json` no longer has `ai_notice`.

### 3. Documentation
- README: a "Licenses" section lists every third-party model and dataset with its license (the same entries as
  `gyeol licenses`; a test keeps the two in sync), followed by "You are responsible for complying with these
  licenses and with applicable law." The profile and consent wording is gone from the API section.

### 4. API, CLI, examples, tests
- The `--profile` options are gone from `gyeol prepare`, `train`, `eval` and `licenses`, and from the config
  presets.
- Examples:
  - `coach_demo_v2.py` uses `api.analyze` / `compare(audibility=(x, sr))` / `render_demo(x, sr, rep, …)` without
    tokens or labels;
  - `discover_demo.py` and `evaluate_and_export.py` no longer pass provenance or profiles.
- Reference service: `gyeol_service.store` now defines its own `Purpose`, `ConsentToken` and `ConsentError`. Its
  `ConsentStore`, `FeatureStore`, `RawAudioStore` and `delete_user` are unchanged. They are one example of
  application policy, and the library never imports them.
- Tests:
  - removed: those that only covered the license gate (profiles, refusals, AI Hub notices, adapter / checkpoint /
    SSL / benchmark / pitch-set gating, the personal profile), consent (`ConsentedVoice`, renderer refusals,
    `OwnVoice`), labels and the watermark;
  - added: the asset list, `add_asset` / `describe`, checkpoint provenance, `gyeol fetch` checksum verification
    (match, already present, mismatch, no URL, unknown name), the `licenses` output, README ↔ asset-list
    consistency, plain-audio rendering through the renderers and `api.render_demo`, reading v1 documents, and
    the removed modules staying removed.

## What moved to the application

| concern | before | now |
|---|---|---|
| Which licenses are acceptable | `Profile` + `require_allowed` in every loader | the application decides; gyeol documents licenses (`gyeol licenses`, README, checkpoint and model-card provenance) |
| Whose voice may be re-rendered | `ConsentedVoice` / `OwnVoice` checks in the renderers | the application decides what audio it passes in (the reference service shows a consent store) |
| Disclosure of generated audio | AI-label metadata, sidecars, watermark | the application labels or watermarks its outputs as its jurisdiction and product require |

## Superseded decisions
Revision D's item 2 ("allow the BS-RoFormer weights only under `personal`") and item 4 (copyleft opt-in via
`allow_copyleft=True`) no longer apply, because there are no profiles. The checksum pin from item 2 remains.
Copyleft code is still neither vendored nor a dependency. The asset list names `so_vits_svc` and `pesto` only to
document that.

## Open questions
- The upstream license of the viperx BS-RoFormer weights is still not stated. The asset list says so rather than
  guessing.
- The MIR-1K distribution states no license. The list records "not stated in the distribution".
- The milestone notes in `docs/milestones/` describe the mechanisms as they were built and are left as history.
  This note supersedes their licensing and consent parts.

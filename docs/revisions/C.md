# Revision C — library boundary

Scope: what lives in `src/gyeol` (stateless library) versus `reference_service/` (user state); the public API and
its JSON; the personal license profile. Tests: `tests/test_rev_c.py` (16 tests), plus the moved store and session
tests in `test_m0_core`, `test_m2_data` and `test_m6_coach`, which now import `gyeol_service`.

## Changes

### C1 User state out of the library
- Moved to `reference_service/gyeol_service/` (its own `pyproject.toml`, not in the `gyeol` wheel):
  - `store.py`: `ConsentStore`, `FeatureStore`, `RawAudioStore`, `RetentionPolicy` and `delete_user`. It was
    `gyeol.store` and is unchanged apart from imports.
  - `session.py`: `CoachSession`, `Attempt`, `Feedback` and the summaries, i.e. the feedback fading schedule,
    attempt history and self-assessment. It was `gyeol.coach.session`. Its Korean strings (`coach.json`) moved
    with it.
  - `wellbeing.py`: `PhonationLog` and `FatigueMonitor`, the running totals and history. They were classes in
    `gyeol.coach.health`.
- Kept in the library, all stateless:
  - `core`: `ConsentToken`, `ConsentedVoice`, `require_consented_voice`.
  - `coach.thresholds` and `coach.priority`; the practice map (read-only data).
  - `coach.onboarding`: the scoring functions.
  - `coach.health`: the measurement functions `check_phrase`, `restricted_for_level`, `voiced_seconds`,
    `attempt_metrics`, and new pure functions `phonation_warnings(session_s, day_s)` and `fatigue_flags(history)`.
    The service classes delegate to these, which is tested.
- Guards:
  - a test fails if any module under `src/gyeol` imports `gyeol_service`;
  - another fails if `gyeol.store` or `gyeol.coach.session` becomes importable again.
- Breaking change: `from gyeol.store import …` and `from gyeol.coach import CoachSession, …` become
  `from gyeol_service import …`.
- Install the service with `pip install -e reference_service`. The test-suite finds it through the root
  `conftest.py`.

### C2 Public API and versioned JSON
- `gyeol.api` exposes four functions:
  - `analyze(audio | path, sr, role=user|reference|synthetic, owner_id, reference=guide, backing, separation, profile, dsp_only, lyrics) → Result[Representation]`.
    It refines latency against the guide and keeps the estimate in `meta["latency"]`, which the shared-clock premise
    reads. It also maps lyrics to notes.
  - `compare(takes, target, lyrics, audibility=OwnVoice) → Result[Explanation]`.
  - `render_demo(OwnVoice, explanation, target, item=None, out_dir) → Result[DemoResult]`. It writes AI-labelled,
    watermarked WAVs and `demo.json`.
  - `train(config | path, resume, overrides) → RunResult`.

  Also exported: `OwnVoice`, `ConsentToken`/`Purpose`/`ConsentError`, `to_json`/`from_json`/`json_schema`, text
  helpers (`text`, `item_text`, `load_strings`), and `load_audio`, `save_audio`, `melody`, `SynthNote` for the
  demo.
- `OwnVoice(audio, sr, rep, consent)` is the only way to render through the API. Rendering is refused unless:
  - the representation is a user take;
  - the token's user matches the take's owner. Analysis now records `meta["owner_sha256"]`, a SHA-256 of the owner
    id, never the id in clear;
  - the token includes `voice_synthesis`.

  The test caught that the first version took the owner from the token, which would have let one user's token
  render another user's take. That is fixed.
- `gyeol.schema` handles the JSON:
  - three documents: `gyeol.representation`, `gyeol.explanation` and `gyeol.demo`, each tagged
    `{"schema", "version": 1}`;
  - `to_json`/`from_json` round-trip losslessly (NaN ↔ null);
  - readers reject unknown schemas and newer versions;
  - JSON Schemas (draft 2020-12) live in `gyeol/resources/schema/`, and the tests validate real outputs against
    them;
  - singer vectors and residual latents (biometric) are left out unless `include_biometric=True`; separated audio
    kept in `meta` is never serialised.
- `examples/coach_demo_v2.py` imports `gyeol.api` only (checked by an AST test). It writes `explanation.json` and,
  with `--render-demo`, `demo.json`. The coaching-session part moved to
  `reference_service/examples/coach_session_demo.py`, which combines `gyeol.api`, the stateless `gyeol.coach`
  functions and `gyeol_service`.

### C3 Personal non-commercial profile
- `Profile.PERSONAL` decisions by license tag:

  | tag | personal | commercial (unchanged) | research (unchanged) |
  |---|---|---|---|
  | commercial_ok | allowed | allowed | allowed |
  | commercial_ok_conditional | allowed + conditions | allowed + conditions | allowed + conditions |
  | noncommercial (openvpi vocoders, GTSinger, CSD, …) | **allowed**, with a "personal, non-commercial only" notice | refused | allowed |
  | copyleft | allowed, never vendored | refused | allowed |
  | unknown | **refused** | refused | allowed (research only) |

- Outputs carry the profile:
  - `Representation.meta["profile"]`;
  - `Explanation.meta["profile"]`: the most restrictive of the inputs;
  - each demo file's AI label and `demo.json`;
  - training samples, `report.json` and the prepared cache (`prepare.json`);
  - checkpoints, as `gyeol.profile`.
- Personal-profile checkpoints are refused under the commercial profile even when every source is commercially
  clean. Checkpoints from other profiles behave as before.

## Measurements
- Full test suite, ruff, and the API-only demo (`--audibility --render-demo`) pass.
- The API demo reproduces the M5/M6 output:
  - "'해'가 목표보다 40센트 낮아요", the missing scoop, "'요'가 83ms 늦게 들어갔어요";
  - audibility per item;
  - four AI-labelled demo files.
- JSON sizes for the synthetic demo take (6 notes, 6.5 s, 44.1 kHz, hop 512 → 560 frames):
  - representation: 329 kB for 12 curves, of which the alignment `content` features are 153 kB;
  - explanation: 118 kB, two takes with per-item `delta` arrays.

## Open questions
1. **Content features in JSON.** `curves.content` (the alignment features) makes up most of a serialised
   representation. Should the app-facing JSON leave it out by default (keeping it for server-side re-alignment
   only), or keep the lossless default?
2. **Owner digest.** `meta["owner_sha256"]` is an unsalted SHA-256 of the user id, the same convention as the demo
   labels. If user ids are guessable (e.g. phone numbers), a keyed hash (HMAC with a service secret) is better; that
   key would live in the service.
3. **Personal profile and copyleft code.** Copyleft code (so-vits-svc, PESTO) is allowed under `personal` because it
   restricts distribution, not use. It is still never vendored or linked into gyeol. Confirm this is the intent, or
   refuse it like `commercial` does.
4. **Milestone docs.** `docs/milestones/M2.md` and `M6.md` still describe `ConsentStore` and `coach.session` at
   their old paths. They are left as historical records; this note and the CHANGELOG describe the move.

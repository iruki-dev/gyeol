# Revision A — analysis path

Scope: `src/gyeol` analysis and explanation (A1–A6). Tests: `tests/test_rev_a.py` (25 tests; 1 skipped when the MIT
`bs_roformer` reference package is not installed).

## Changes

### A1 Vocal separation as default preprocessing
- `analyze(..., separation="auto" | "always" | "off", separator=None)`. The default is `auto`.
  - `auto` separates whenever a backing track is given, or when the input may contain accompaniment.
  - `always` fails with an explicit status if no separator can run.
  - `off` analyses the input as given. The caller then vouches that it is a clean vocal, and no penalty is applied.
- `frontend.separation.estimate_accompaniment` decides "may contain accompaniment". It is pitch-informed and cheap
  (YIN at 16 kHz plus one STFT). Its cues:
  - energy outside the voice's harmonic comb, in stable voiced frames (dB relative to the comb), and the spectral
    flatness of that residual (tonal residual = instruments, flat residual = breath or noise);
  - the share of loud frames that are tonal but have no voice ("tonal gaps");
  - energy below 90 Hz.

  Clips too short to judge return `unavailable`. `analyze` then separates if it can and applies no penalty.
- `frontend.roformer`: a BS-RoFormer implementation that loads local checkpoints (no new dependency; written from the
  MIT reference).
  - It loads reference checkpoints strictly, key by key, and accepts `state_dict`/`model` wrappers and a `model.`
    prefix.
  - Inference runs in chunks with cross-fades. Optional gradient checkpointing (`checkpointing=True`).
  - `RoFormerSeparator.from_cache()` only reads `~/.cache/gyeol/<asset>/` and never downloads.
- `core.license`: registers `bs_roformer_viperx_ep317` with its URL. `gyeol fetch bs_roformer_viperx_ep317` shows the
  license and caveats, then asks before downloading. The caveats say that provenance is unclear and the weights'
  license is unstated.
- The default separator is chosen as follows:
  - a backing track is given → `BackingTrackCanceller`;
  - otherwise → fetched BS-RoFormer weights, if present;
  - otherwise → none. This gives a warning and a confidence penalty.
- Separation quality feeds per-frame confidence. `separation_quality` uses `frontend.quality.detect_bleed`:
  - Residual is measured by coherence of the stem against the known backing, or against the estimated accompaniment
    (mixture − stem).
  - Per-frame stem-to-accompaniment ratio maps to a factor of 0–1.
  - The factor multiplies every pitch-derived confidence (f0, pitch centre, vibrato, events) and the frame
    confidence.
  - Flags: `residual_accompaniment` when leakage is above −20 dB; `accompaniment_unseparated` when accompaniment is
    suspected and nothing was separated.
- If accompaniment is suspected but could not be separated, the confidence penalty is graded by the estimated
  residual level (`unseparated_factor`).
  - A full penalty of 0.5 applies at a residual of −15 dB or more.
  - No penalty applies at −30 dB or less.
  - An already-separated stem with a faint residual is therefore not treated like a full mixture.
- `frontend.quality.assess` measures clipping on the raw input, and SNR, bandwidth and bleed on the analysed signal.
- `rep.quality["separation"]` records mode, separator, residual, flags and the accompaniment estimate.
  `rep.meta["analysis_signal"]` is `"input"` or `"separated vocal"`.

### A2 Confidence-based pitch decisions
- The octave/key relation is a premise (`octave_relation`). It is decided only when all of these hold:
  - the median consensus confidence on both sides is ≥ 0.6;
  - ≥ 25 % of sung frames are reliable on both sides;
  - ≥ 80 % of reliable frames agree on one relation.
- Otherwise `Explanation.transposition_cents` is `None`, the premise carries the reason, and pitch is compared
  octave-invariantly: each frame difference is folded onto the 1200-cent circle around its circular mean
  (`comparison_mode["pitch"] == "octave_invariant"`).
- `gyeol eval realset --per-tracker` (`eval.realset.tracker_breakdown`) reports each tracker alone on the raw input,
  per condition. This is the tool for tuning the consensus on real mixtures versus separated vocals.

### A3 Premise checks before judgements
- `explain.premises` introduces the pattern: state the premise → check it → withhold dependent items when it fails.
  - `Premise(name, statement, passed, reason, measures)`.
  - `WithheldItem`.
  - `Explanation.premises`, `.withheld` and `.comparison_mode`.
- The premises and the items that depend on them:

  | premise | checks | withheld when it fails |
  |---|---|---|
  | `shared_clock` | duration difference ≤ 1 s, start offset ≤ 0.25 s, latency-refinement confidence ≥ 0.5, alignment confidence ≥ 0.4, warp at its band limit on ≤ 5 % of sung frames | `tempo`; onset timing switches from absolute to *relative to the previous note* |
  | `octave_relation` | see A2 | absolute pitch comparison (switches to octave-invariant) |
  | `level_chain` | no clipping / unseparated / leaking accompaniment on either side | `loudness`, `dynamic_range` |
  | `noise_floor` | SNR ≥ 30 dB on both sides, no clipping, codec or accompaniment flags | `breathiness` |
  | `interval_set` | ≥ 3 confident intervals of ≥ 1 semitone | `interval_compression` |

- When the clock is not shared, local comparisons use content-based alignment.
  - The DTW runs with a whole-take band and an open beginning (subsequence DTW, new `WarpConfig.open_begin`), so a
    hand-trimmed clip can start inside the target.
  - Target notes whose onset lies outside the stretch the take covers get no onset verdict.
- Each mode is recorded in `Explanation.comparison_mode` (`timing`: `shared_clock` | `content_aligned`; `pitch`:
  `absolute` | `octave_invariant`). The text renderer adds a context line for each mode and for each withheld
  premise. The Korean strings are in `resources/ko/explain.json`.

### A4 Korean syllabification
- `context.korean.syllabify` NFC-normalises its input and splits Hangul into syllables regardless of spaces and
  punctuation. Any non-Hangul character starts a new word, for the `word_initial` flag.
- `context.assign_syllables(notes, lyrics)` maps syllables to notes:
  - one syllable per note when the counts match;
  - extra notes are melismas that hold the previous syllable;
  - extra syllables are distributed by note length.
- `explain(..., lyrics=...)` uses it.

### A5 Whole-contour pitch comparison
- `explain.contour.contour_items` compares the aligned f0 contours frame by frame, including attacks, releases and
  the transitions between notes.
  - It reports spans where both pitch tracks and the alignment have confidence ≥ 0.5.
  - The difference must keep one sign beyond a detection floor for ≥ 60 ms. The floor is max(20 ¢, 3 × the robust
    frame-difference noise).
- Each span is placed on the target notes through τ:
  - inside one note → `contour_deviation` at that note's syllable;
  - across a boundary or in a gap → `transition_deviation` between notes k and k+1 ("'랑'→'해'").
- The existing event detectors label spans: `detail["event"]` and `detail["events"]` hold the scoop, fall, 꺾기 or
  glide that overlaps the span on either side. The rendered text names it, e.g. "(스쿱 구간)".
- A `contour_deviation` that only restates the note's own centre offset is dropped, so the same error is not
  reported twice. A span counts as a restatement when it has the same sign and lies within the detection floor of
  the `intonation_offset` item.
- Display uses the fitted coach thresholds. The detection floor only forms candidates, and `coach.priority` applies
  `coach.thresholds` to both new attributes like any other item.
- `eval.knob_recovery` adds known attack scoops and release falls to the synthetic takes. It scores each contour
  span against the synthesiser's exact f0 tracks over the frames the item measured, so `fit_from_knob_data` fits
  `contour_deviation` (and `transition_deviation`, when legato material produces it) like the other attributes.
- Items carry the per-frame `delta`. `demo.edits` turns it into an f0 edit with soft edges, so contour items can be
  demonstrated in the user's own voice.

### A6 Real-recording evaluation set
- `eval.realset` defines the format: a folder with `manifest.jsonl`, one JSON object per recording. The full format
  is in the module docstring.
  - Required: `id`, `audio`, `role` (`target` | `user`), `condition`; for users, also `target` and `singer`.
  - Optional: `session`, `lyrics`, `recording` (device, route, `backing` file), and `annotations`:
    - `f0` (CSV or `.npy`);
    - `onsets` (list or file);
    - `octave_relation` or `transposition_semitones`;
    - `onset_deviation_ms` per target note, or `rhythm_ok`.
- Conditions: `mixture_phone`, `mixture_karaoke`, `separated`, `clean`, `trimmed`, `multi_singer`.
- `split_realset` splits by singer into tuning and held-out sets.
  - It checks that no recording session is shared across the two sides.
  - It is deterministic per seed and is saved to `split.json`, so later runs reuse it.
- `gyeol eval realset <folder> [--split all|tuning|held_out] [--separation ...] [--per-tracker] [--dsp-only]`
  reports, per condition:
  - octave decisions made, withheld and wrong;
  - rhythm verdicts and false rhythm verdicts. A false verdict is a verdict of ≥ 50 ms on a note annotated as on
    time within ±30 ms. Content-aligned verdicts are relative to the previous note, so they are scored against the
    relative truth;
  - withheld rate and premise failures;
  - RPA, RCA and voicing recall against the annotated f0;
  - onset F-measure (±50 ms).

  It writes a JSON report.
- `eval.realset_synth.make_synthetic_realset` writes a synthetic folder in exactly this format, with every
  condition. It is used by the tests and serves as a worked example of the manifest.

## Measurements

All numbers are on **synthetic** audio (`make_synthetic_realset`, 3 singers × 6 conditions, DSP trackers only). They
check that the machinery works and that the premises behave as intended. They are **not** accuracy claims; real
numbers come from the user's set.

`gyeol eval realset … --dsp-only`, separation `auto` (no BS-RoFormer weights fetched, so only the karaoke condition,
which lists its backing file, is separated):

| condition | n | analysis/explain failed | octave decided / withheld / wrong | rhythm verdicts / false | withheld rate | RPA | onset F |
|---|---|---|---|---|---|---|---|
| clean | 3 | 0 | 3 / 0 / 0 | 1 / 0 | 0.00 | 0.988 | 1.00 |
| mixture_karaoke (backing cancelled) | 3 | 0 | 3 / 0 / 0 | 2 / 0 | 0.00 | 0.994 | 0.86 |
| mixture_phone (not separable here) | 3 | 0 | 0 / 3 / 0 | 2 / 0 | 0.00 | 0.680 | 0.33 |
| separated (residual −25 dB) | 3 | 0 | 3 / 0 / 0 | 2 / 0 | 0.44 | 0.988 | 0.71 |
| trimmed | 3 | 0 | 3 / 0 / 0 | 4 / 0 | 0.05 | 0.987 | 0.93 |
| multi_singer | 3 | 2 | 0 / 1 / 0 | 0 / 0 | 0.00 | 0.647 | 0.94 |

The same set with separation `off`:

| condition | octave decided / withheld / wrong | withheld rate | RPA | onset F |
|---|---|---|---|---|
| mixture_karaoke | 1 / 2 / 0 | 0.04 | 0.639 | 0.38 |
| mixture_phone | 1 / 2 / 0 | 0.02 | 0.680 | 0.33 |
| separated | 3 / 0 / 0 | 0.20 | 0.988 | 0.71 |
| multi_singer | 0 / 3 / 0 | 0.00 | 0.647 | 0.94 |

Readings:
- **No octave error in any condition.** Where pitch is unreliable (unseparated phone mixtures, two singers), the
  octave is withheld instead of guessed.
- **Separating a karaoke take with its backing track** lifts RPA from 0.64 to 0.99 and onset F from 0.38 to 0.86.
- **Trimmed clips.** Before A3, trimmed clips produced 6 false rhythm verdicts in 9: the forced start alignment and
  absolute timing on a clock that was not shared. After A3 there are none. `shared_clock` fails on all three clips,
  tempo is withheld, and the uncovered notes get no verdict.
- **The `separated` withheld rate (0.44 under `auto`)** comes from `level_chain` and `noise_floor` failing. The −25 dB
  residual is detected and flagged, so loudness, dynamic range and breathiness are withheld. With `off` the caller
  vouches for the stem, and only breathiness is withheld (SNR < 30 dB).

Per-tracker RPA on the raw input (`--per-tracker`):

| condition | pYIN | YIN | SHS |
|---|---|---|---|
| clean | 1.000 | 0.946 | 1.000 |
| mixture_karaoke | 0.451 | 0.527 | 0.703 |
| mixture_phone | 0.434 | 0.460 | 0.759 |
| separated | 0.990 | 0.952 | 0.996 |
| multi_singer | 0.218 (RCA 0.876) | 0.037 | 0.668 |

On unseparated mixtures, SHS alone (0.76) beats the consensus (0.68). I did **not** retune the consensus defaults on
synthetic accompaniment: the gap is specific to how the mixture was synthesised. This table is the tool for
tuning them on the tuning split of the real set.

Accompaniment estimator, synthetic cases:
- Classified correctly (12/12): plain, vibrato, breathy, low, short, noisy, the demo files, and backing at −20 and
  −10 dB.
- Realset estimates:

  | condition | residual | flatness | tonal gaps | verdict |
  |---|---|---|---|---|
  | clean | −40 to −43 dB | — | — | no accompaniment |
  | separated | −26 dB | 0.015 | — | may contain |
  | mixtures | −11 to −14 dB | — | 50–59 % | may contain |

Contour thresholds (`knob_recovery`, 8 takes, clean vs pink noise at 25 dB):
- `contour_deviation`: 21 items, 10 retest pairs.
- Median |measured − true| is 24 ¢ and p95 is 54 ¢. The trackers smooth fast attack scoops and release falls.
- Fitted values: MDC95 = 22 ¢, E95 = 45 ¢ at confidence ≥ 0.62, so a span is shown only when its mean deviation
  exceeds ≈ 45 ¢.
- For comparison, `intonation_offset` (a note centre) is measured to 0.5 ¢ median and 2 ¢ E95.

Separator speed: BS-RoFormer with the viperx config has 159.8 M parameters. It runs at **RTF 20.8** on 4 CPU threads
(10 s of audio in 208 s; 8 s chunks with 25 % overlap).

## Open questions
1. **Separation cost on CPU.** At RTF ≈ 21, `auto` with BS-RoFormer is not interactive on a laptop. Options:
   - a smaller RoFormer (e.g. dim 256, depth 6), trained with revision B;
   - a GPU;
   - separating only when the estimator is confident, and otherwise analysing unseparated audio with the graded
     penalty.

   Which trade-off do you want for the app?
2. **Weights licence.** `bs_roformer_viperx_ep317` is tagged "MIT (as listed)", but the weights' own licence is not
   stated upstream and the checksum is not pinned. Please confirm before relying on it commercially, or point me to
   weights with a clear licence.
3. **Thresholds to fit on the real set** (tuning split):
   - accompaniment cues: residual −28 dB / flatness 0.02 / tonal gaps 0.3 / LF 5·10⁻⁴;
   - separation SIR mapping: −10…5 dB;
   - graded penalty: −15…−30 dB;
   - octave premise: 0.6 / 0.25 / 0.8;
   - shared-clock limits: 1 s / 0.25 s / 0.4 / 5 %.

   All of these are calibrated on synthetic audio only.
4. **Two singers.** Under `auto`, 2 of 3 multi-singer takes fail with "too few reliable frames": the second voice
   trips the accompaniment estimator, and the penalty pushes confidences below the explain threshold. The result is
   an explicit failure rather than a wrong answer. A dedicated "second voice" cue, which would withhold instead of
   penalising, may be better. This needs real examples.
5. **Content-aligned DTW cost.** The whole-take band makes the DTW O(T_user × T_target) in Python, which is fine for
   phrases but slow for whole songs. Vectorising or banding around a coarse alignment is a follow-up.
6. **Contour thresholds.**
   - `contour_deviation` now has a synthetic fit. `transition_deviation` has none yet: the synthetic takes are
     detached notes with gaps, so transitions are unvoiced. Until it is fitted, the coach does not show it (the
     policy has no numeric fallback).
   - Legato and glide takes in knob recovery, or annotated real transitions, are needed.
   - Neither attribute is in the listening calibration yet.

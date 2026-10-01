# Revision D — follow-up decisions

Implements the decisions taken after revisions A–C. Tests: `tests/test_rev_d.py` (13 tests).

| decision | status |
|---|---|
| 1. Separate the target once (async, BS-RoFormer, cached by content hash); separate headphone takes only on detected bleed, with a selectable light separator | **done** |
| 2. Pin the BS-RoFormer weights' sha256, verify it on every load, keep the license unverified, allow it only under `personal` | **partly done**: the hash is computed and the loader verifies any pinned hash. The registry edit (pin + personal-only) is **pending** — see below |
| 3. Exact-f0 synthetic ground truth, fine-tune from pretrained RMVPE, consensus only as a weak label, evaluate on human-annotated real singing behind the license gate | **done** (the evaluation sets still need registry entries with verified licenses) |
| 4. Copyleft off by default even under `personal`; explicit `allow_copyleft=True` opt-in with a warning; out of the default install | **pending** — see below |

## Changes

### D1 Target separation once, cached; takes separated only on bleed
- `frontend.separation`:
  - `SEPARATORS` registry and `make_separator(name, profile, …)` for the separator choice:

    | name | kind | what it is |
    |---|---|---|
    | `bs_roformer` | heavy | fetched viperx weights |
    | `roformer_light` | light | a smaller BS-RoFormer from a local checkpoint with its config and registry asset |
    | `htdemucs` | light | needs `demucs` and its weights registered |
    | `backing` | light | subtracts a known accompaniment |

  - `separate_target(audio, sr, cache=SeparationCache(dir), separator=…)` keys the cache by SHA-256 of the mono
    float32 samples, the sample rate and the separator identity. It stores the vocal and the accompaniment
    (mixture − vocal) atomically and returns a cache hit without loading any model.
  - `SeparationQueue` runs separations in a background worker, so the upload request returns at once. Duplicate
    submissions of the same song share one future.
  - `PrecomputedSeparator` replays a cached vocal.
- `analyze(…, accompaniment_ref=…)` handles user takes, which are assumed to be recorded on headphones:
  - in `auto` mode, `detect_bleed` runs against the target's cached accompaniment;
  - the take is separated only when bleed ≥ −25 dB **and** the delay peak is prominent (`TakePolicy`);
  - the separator is the selectable light one, by default `backing`, which subtracts the known accompaniment;
  - the report records bleed level, delay and prominence.
- `api.separate_target(…, cache_dir=, background=True)` returns a `Future`. `api.analyze(…, separated=ts)`
  analyses the song from its cached vocal, and `api.analyze(…, target=ts, take_separator=…)` analyses a take.
- `frontend.quality.detect_bleed` gets two fixes, both found while testing D1:
  - **Bias null.** The old null (coherence against a circularly shifted copy of the backing) cancels genuine leakage
    of repetitive music, because looped bars stay coherent with themselves. It now uses a phase-randomised
    surrogate with the same spectrum.
  - **Averaged null.** For a few seconds of audio the coherence bias is about as large as a −20 dB leak, so the null
    is averaged over 6 surrogates. The M1 bleed test reads −26 dB for its −28 dB case and −8.5 dB for −8.5 dB.
  - **Delay peak prominence.** A voice on the same pitches as the accompaniment is coherent with it without any
    leakage: a clean take read −22 dB. Leakage has one sharp, consistent delay across frequencies, so
    `BleedReport.lag_prominence` (robust z of the GCC-PHAT peak) gates both the take decision and the
    post-separation residual flag.

### D2 Weights checksum
- The viperx BS-RoFormer file was streamed through SHA-256 without being stored:

  ```
  sha256 5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa
  size   639331213 bytes
  url    https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt
  ```

  This pins what the URL served on 2026-10-01. It does not prove provenance.
- `RoFormerSeparator.from_checkpoint` hashes the file on every load and refuses a file whose hash differs from the
  registry's pinned `sha256`. The verified hash is kept on the separator (`weights_sha256`) and in the target
  cache's metadata.
- **Pending:** writing the hash into `core/license.py` and restricting the asset to `personal`. See "Not done".

### D3 Pitch ground truth
- `gyeol prepare --resynthesize hnm [vocoder --resynth-vocoder ckpt]` (or `prepare.resynthesize` in a run config)
  adds exact-f0 copies:
  - `hnm` is MDB-stem-synth-style. The item is analysed with the harmonic-plus-noise engine at its analysed f0 and
    resynthesised, so the f0 it is synthesised with is exact.
  - `vocoder` resynthesises through a gyeol-trained NSF vocoder (`train.tasks.vocoder_from_checkpoint`). Release
    checkpoints now record their model and data settings.
  - Each copy (`<id>~hnm`) stores `f0_exact` per frame and keeps its source's dataset, license and singer, so splits
    stay singer-disjoint.
  - Copies are never made of `own_recordings`: synthesising an app user's voice needs their separate consent.
  - Tested: a DSP tracker run on the copies agrees with `f0_exact` (RPA > 0.9).
- The `pitch` task uses the copies as primary truth:
  - every frame of an exact copy, voiced or not, is trained at `exact_weight`;
  - original items contribute DSP consensus only as a weak label (`weak_weight` 0.3) on frames with consensus
    confidence ≥ `weak_min_confidence` (0.8); their other frames are ignored;
  - validation reports `rpa_exact` and `rpa_weak` separately.

  `cpu-full/pitch.yaml` fine-tunes from the pretrained reference RMVPE (`init.pitch`, asset `rmvpe`). Other tasks
  skip the resynthesised copies.
- Evaluation on human-annotated real singing:
  - `data.pitch_sets`: `scan_vocadito` (CSV f0) and `scan_mir1k` (`.pv` semitones per 20 ms; the voice channel by
    default);
  - `eval.pitch_eval.evaluate_pitch`: RPA, RCA, voicing recall, voicing false alarm and overall accuracy, and
    `gyeol eval pitch <manifest> [--checkpoint | --rmvpe-weights] --profile`;
  - `model.eval_sets` in a pitch run evaluates the best weights at the end (`pitch_eval.json`);
  - every set goes through `open_manifest`, so an unregistered dataset is refused ("not in the license registry"),
    and a registered one is allowed only under profiles its tag permits.

## Measurements
- Bleed decision on synthetic takes: a voice with 25 ¢ vibrato over a chord accompaniment, delayed 9 ms (400
  samples at 44.1 kHz):

  | bleed | coherence estimate | delay prominence | separated? | residual after the backing canceller |
  |---|---|---|---|---|
  | none | −21.0 dB | 9.6 | no | — |
  | −30 dB | −24.9 dB | 13.2 | no (below the resolution of the check) | — |
  | −25 dB | −23.1 dB | 21.0 | yes | −18.1 dB (flagged) |
  | −20 dB | −19.7 dB | 33.2 | yes | −16.5 dB (flagged) |
  | −12 dB | −12.4 dB | 69.5 | yes | at the −30 dB floor |

    Without the prominence gate, every clean take on these pitches would have been separated. At −20 and −25 dB the
  canceller leaves a measurable residual, which is flagged and lowers confidence. A better light separator for
  takes (`roformer_light` with trained weights) would reduce it.
- Cache: a repeated `separate_target` call is a file read, and the separator is not invoked (tested with a counting
  separator).
- `cpu-smoke/pitch` with resynthesis: 45 s in total on 4 threads, including preparing 54 copies on a fresh cache.

## Not done — needs your decision on how to proceed
While implementing decisions 2 and 4, my read of the license-gate module (`src/gyeol/core/license.py`) and its call
sites was **blocked by the session's automatic permission classifier** (it flagged the action as weakening security).
Both decisions *tighten* the gate, but I did not work around the block. Still to do:
1. **D2 registry edit.**
   - Set `sha256="5b84f37e…15aa"` on `bs_roformer_viperx_ep317` (the loader already enforces it once set; `gyeol
     fetch` already checks it after download).
   - Keep `verified=False`.
   - Add a per-asset profile restriction so the asset is allowed only under `personal`. Commercial and research are
     then refused until the license is confirmed.
2. **D4 copyleft.**
   - In `decide()`, refuse copyleft under `personal` by default.
   - Add an explicit `allow_copyleft=True` opt-in (also on `require_allowed`) that prints a warning.
   - Test that no copyleft package is in any install extra. None is today: `pyproject.toml` lists none.
3. **Register the evaluation sets.** `vocadito` and `mir1k` need registry entries with their *verified* tags. MIR-1K
   states no clear license, so it would be `unknown` (research only).

If you allow edits to `core/license.py` (or make them yourself), items 1 and 2 are about 30 lines plus tests.

# Protocol — internal coach agreement

Purpose: to learn how much trained Korean vocal coaches agree with each other about what is wrong in a take, and
how gyeol's explanation compares with their consensus. Coach-to-coach agreement is the ceiling. gyeol is not
expected to beat it, and a gyeol item that coaches do not agree on is not a coaching target. Pre-register this
protocol before collecting any ratings.

## Panel and material
- **Panel:** ≥ 5 vocal coaches (≥ 3 years teaching popular or trot singing), and 2 speech-language pathologists
  for health-related flags.
- **Material:** ≥ 60 sing-along takes from consenting users (`Purpose.TRAINING` + `STORAGE`), each with its target
  phrase. Stratify by level (onboarding band), genre, device route (wired or Bluetooth) and phrase difficulty.
- **Duplicates:** 20 % of takes appear twice, randomly ordered, to measure each coach's own test–retest agreement.

## Rating task
- Blind to gyeol's output and to other coaches' ratings.
- For each take, the coach marks every issue they would mention, using the same 7-label taxonomy as the external
  benchmark (breath, vocalization, technique, pitch, rhythm, diction, expression). For each issue they add a time
  span (tap on the waveform), a severity from 1 to 3, and whether they would raise it *first*.
- Free-text notes are optional. They are not scored, but they feed the practice-mapping file (`practice.json`).

## Analysis (`gyeol.eval.benchmarks`)
1. **Coach–coach:** Fleiss' κ per label, from label presence per take (`coach_agreement`). Report κ with a
   bootstrap CI over takes, and each coach's test–retest Cohen's κ.
2. **gyeol vs consensus:** the consensus is the majority of coaches per take and label. Compute Cohen's κ between
   gyeol's shown items (after the coach policy: thresholds, priority, volume), mapped with `GYEOL_TO_LABELS`, and
   the consensus.
3. **First issue:** the share of takes where gyeol's primary item falls in the label most coaches would raise
   first. Compare with a label-prior baseline.
4. **Segments:** `score_benchmark` recall and precision against the coach spans (±100 ms tolerance).
5. **Acceptance:** gyeol's κ against consensus for a label must lie within a pre-registered TOST margin of the
   median coach-vs-rest κ for that label. Labels that fail are withheld or phrased more tentatively in the app.

## Ethics
Takes are sensitive biometric data. Coaches see audio only on a controlled device, and no identifiers are shown.
Every rating is deleted when the user deletes their data (`store.delete_user`).

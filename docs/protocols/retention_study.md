# Protocol — pre/post retention study (does the coaching produce learning?)

Performance *during* feedback is not learning. This design measures what remains when feedback is removed: the
retention and transfer tests of the motor-learning literature, where frequent feedback can help during practice
and still hurt retention (the guidance hypothesis). It is the evidence that should decide the coach's fading
schedule and feedback volume. Pre-register it.

## Design
- **Participants:** adult learners recruited through the app, stratified by onboarding production band and
  perception band (never by a "tone-deaf" label; `coach.onboarding`). Consent covers analysis, storage and
  research use.
- **Arms** (randomised within strata):
  1. **gyeol coach** (default policy: one primary item plus ≤ 2 secondary items, fading full → half → quarter,
     self-assessment first);
  2. **full feedback** (every item after every attempt; no fading, no self-assessment);
  3. **control** (same practice time and the same target phrases, but only the singer's own recording played
     back).
- **Schedule:** pre-test (day 0) → 6 practice sessions over 2 weeks → post-test (day 15) → retention test
  (day 22). No practice between day 15 and day 22.
- **Tests without feedback:** each test has the practised phrases (retention) and unpractised phrases of similar
  range and tempo (transfer). The same recording route is used at every test (check with `io.latency`).

## Outcomes
- **Primary:** the change from pre-test to the day-22 retention test in note-level intonation error (median
  |intonation_offset| over notes, cents) on practised phrases, measured by gyeol's signal layer with thresholds
  frozen before the study.
- **Secondary:**
  - onset-timing error (ms);
  - the same metrics on transfer phrases;
  - self-assessment accuracy (how often the user's self-assessment matched the primary item);
  - coach-rated overall improvement (blind raters, `coach_agreement.md`);
  - vocal-health flags per session (`coach.health`).
- **Measurement check:** for every outcome, report the robustness-grid MDC95 for the recording routes actually
  used (`eval.robustness`). Changes smaller than the MDC are not interpreted.

## Analysis
- Mixed-effects model: outcome ~ arm × time + stratum + (1 | participant) + (1 | phrase). The planned contrasts
  are gyeol vs full feedback at retention (the guidance hypothesis) and gyeol vs control.
- **Sample size:** from a pilot estimate of the between-participant SD (σ) and the smallest effect of interest
  (Δ, at least the MDC). With two-sided α = 0.05 and 80 % power, n per arm ≈ 2·(1.96 + 0.84)²·σ²/Δ², inflated
  for clustering and a pre-registered dropout rate.
- **Missing data:** multiple imputation; report per-protocol and intention-to-treat results.

## Stopping and safety
Stop a participant's practice if the health guard flags fatigue on two consecutive sessions, or if they report
pain or hoarseness. The referral notice stays visible throughout.

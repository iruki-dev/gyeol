# Listening-test protocol template (reconstruction and judgement preservation)

A template to fill in before running a test. Pre-register it, including the margins and the analysis.

## 1. MUSHRA fidelity (ITU-R BS.1534-3)
- **Stimuli:** 10–12 s sung Korean phrases, stratified by technique (modal, breathy, rough/수리성, falsetto,
  belt/통성, trot 꺾기) and by consonant class (lenis / aspirated / fortis onsets).
- **Conditions per trial:**
  - the hidden reference;
  - a 3.5 kHz low-pass anchor;
  - a 7 kHz low-pass mid anchor;
  - the systems under test: gyeol vocoder, gyeol full autoencoder, and the BigVGAN fallback.
- **Listeners:** ≥ 20 expert listeners. Apply BS.1534-3 post-screening: exclude a listener who rates the hidden
  reference below 90 in more than 15 % of trials.
- **Analysis:** mixed-effects model with fixed effects system × technique and random effects listener and item.
  Report means with 95 % CIs, per technique and per consonant class.

## 2. Judgement preservation (coach agreement)
- **Panel:** ≥ 5 Korean vocal coaches, plus 2 SLPs for health-related probes.
- **Stimuli:** originals and resyntheses, presented blind in random order. 20 % of stimuli are duplicated to measure
  intra-rater reliability.
- **Probes:**
  - pitch accuracy
  - vibrato
  - onset (glottal / breathy / balanced)
  - phonation (pressed–modal–breathy)
  - breathiness
  - register
  - resonance / nasality
  - strain
  - style
  - diction (ㄷ/ㄸ/ㅌ)
- **Criterion:** original–resynthesis agreement (weighted κ or ICC) is at least the same coach's test–retest agreement
  on the originals, within a pre-registered TOST margin.
- **ABX:** discrimination of ±1 breathiness step, original vs resynthesised.

## 3. Reporting
- Report every metric by technique and consonant class. Never report a single pooled number only.
- Record audio settings, playback device, room and listener hearing screening.

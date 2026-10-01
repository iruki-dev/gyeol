# gyeol reference service

User state on top of the stateless `gyeol` library (revision C1). It is not part of the `gyeol` wheel.

| module | holds |
|---|---|
| `gyeol_service.store` | consent records per user and purpose; feature and raw-audio storage with retention; user deletion |
| `gyeol_service.session` | coaching session: feedback volume, fading schedule, self-assessment, attempt history, summaries |
| `gyeol_service.wellbeing` | running phonation time and fatigue history (the measures themselves are `gyeol.coach` functions) |

The library keeps everything stateless:
- the consent types and guards: `gyeol.core.ConsentToken`, `ConsentedVoice` and `require_consented_voice`;
- `gyeol.coach.thresholds`, `priority`, `practice` and `onboarding` scoring;
- the measurement functions in `gyeol.coach.health`;
- the public API `gyeol.api`.

```bash
pip install -e .                    # gyeol
pip install -e reference_service    # this package
python examples/fit_thresholds.py --synthetic --out /tmp/th.json
python reference_service/examples/coach_session_demo.py --synthetic --coach /tmp/th.json --noticed pitch
```

A production service would keep the same calls into `gyeol` and replace the file-based storage here with its own
database.

# Screening and case report form (CRF)

Participant code: **P____**   Date: ________   Operator: ______   Rater 1: ______   Rater 2: ______

## Part A — before recording

| Field (`participants.csv` column) | Entry |
|---|---|
| Age, years (`age`) | |
| Sex (`sex`: F / M / other / prefer not to say) | |
| Pacemaker or implanted cardiac device? (**exclude if yes**) | Y / N |
| Known atrial fibrillation or other sustained arrhythmia? (**exclude if yes**) | Y / N |
| Medicines that lower heart rate, e.g. beta-blockers (`hr_lowering_medication`: 0/1, name) | |
| Caffeine or nicotine in the last 2 hours? (`caffeine_2h`: 0/1) | |
| Beard or moustache covering cheeks (`beard`: 0 none, 1 partial, 2 full) | |
| Spectacles worn during recording (`spectacles`: 0/1) | |
| Bindi / tilak (`bindi`: 0/1) | |
| Facial make-up (`makeup`: 0/1) | |
| Head covering (`head_covering`: 0/1) | |
| Self-rated Monk shade 1–10 (`monk_self`) | |
| Fitzpatrick self-report (`fitzpatrick_self`, I–VI as 1–6), from the two questions below | |
| Consent tier (`consent_tier`: research_only / data_use_agreement) | |

Fitzpatrick questions (read aloud): "If you stay in strong sun for an hour without protection, does
your skin (a) always burn, never tan; (b) usually burn, tan a little; (c) sometimes burn, tan
gradually; (d) rarely burn, tan well; (e) very rarely burn, tan very easily; (f) never burn?"
→ (a)=1 … (f)=6.

## Part B — during and after recording

| Field | Entry |
|---|---|
| Light 1 lux at the face (also in session.json `lux.light1_500lux`) | |
| Light 2 lux at the face | |
| Webcam pixel format actually used (YUYV / MJPG) and resolution | |
| Camera auto-exposure: locked / auto | |
| Card used and its reflectance (grey 0.18 / white ~0.9) | |
| Oximeter model and finger | |
| Phone recorded? model | |
| Deviations from protocol (blocks skipped, shortened, repeated) | |
| Notable events (cough, talking in rest, left the box) with approximate time | |
| Participant stopped early? reason | |
| Session file checked with `python -m sama qc` (usable Y/N, reasons) | |

Copy Part A into `pilot_data/participants.csv` (template in `pilot/templates/`).

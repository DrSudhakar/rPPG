---
title: "SAMA proposal: what changed from the 28 October draft, and why"
author: "Working document for the PI"
---

Each change below is in `proposal/SAMA_proposal.md`. The reason is given so you can defend it, or undo it. Two passes produced these changes: a rewrite for the call and the pilot (9 November), then a hostile review of that rewrite (10 November), whose findings are marked **[review]**.

# Call compliance (must not be skipped)

| Change | Why |
|---|---|
| Duration 24 → **36 months**; timeline, milestones, Figure 3 and Figure 8 redrawn | The PM-ECRG page on anrfonline.in states a three-year grant with no extension. Recording now runs months 5–16 (main study) after a protocol pilot in months 3–4. |
| Budget ₹18.04 lakh → **₹33,67,250** (+ overheads) | Three years of fellowship (JRF from month 3, **SRF from month 27** after two years as per norms **[review]**), a Project Associate for the 12 recording months, 175 paid sessions (150 + 10 protocol pilot + 15 safeguard), two logging oximeters, **about 20 TB of storage (4 × 18 TB, primary + encrypted backup)** **[review: the earlier 2 × 4 TB could not hold the lossless video]**, and a developer workshop. Well under the ₹60 lakh cap; travel ₹2.70 lakh and contingency ₹2.40 lakh are each under the ₹3 lakh cap. The build checks that the table adds up. |
| Eligibility items added as **[TO FILL]** fields | The call requires application within two years of joining, age ≤ 42 (+3 where applicable), and no earlier SERB/ANRF project grant. The build now refuses to report the proposal as ready while any [TO FILL] field remains **[review]**. |
| A named possible reviewer (Dr Gupta) removed from section 10 **[review]** | Suggesting a reviewer whose group works on the closest topic invites a conflict-of-interest objection, and the call template does not ask for it. |

# Sample size: SAMA-120 → SAMA-150 **[review]**

| Change | Why |
|---|---|
| **50 recruited, 45 analysable per stratum** (150 adults); safeguard up to 55 per stratum | The October H1 effect (3.94 → 6.20 bpm, Chari et al. 2020) spans Fitzpatrick I–II to V–VI, a range of at least 75° of ITA°; the Indian ITA° range is narrower. Re-anchored conservatively (ratio 1.57 from ITA° 50 to −25, i.e. 2.87 → 4.5 bpm), H1 power at 36 per stratum is 0.74. At 45 it is 0.83 (0.64 if the between-person CV is 0.8). Every confirmatory criterion now has power ≥ 0.83 at 45. |
| Screening of up to 390 volunteers; PA workload and payments re-costed | The lighter stratum (ITA° > 28) is the scarce one in Indian colorimeter data (Pawar et al. 2024). |
| Milestones at 75 and 150 participants; the month-10 power re-check uses the protocol pilot's spread | A committee wants to see when the safeguard is triggered and by what rule. |

# Hypotheses and statistics

| Change | Why |
|---|---|
| H1 endpoint: raw MAE → **log of per-participant MAE** (estimand: ratio of errors per 10° ITA°) | Errors are right-skewed; raw-MAE power collapses if the between-person SD is large. Type-I error of the log test simulated at 0.053. |
| H2 decision rule: "a clear answer either way" → **A (widening) / B (equivalence, 90% CI inside ±1.0 bpm per 75°) / inconclusive** | With the October ±0.5 margin, a null almost never established B (probability 0.002 at 36); the margin is now the same smallest effect as A **[review]**, so B is reachable (0.86 at 45 if there is no widening). |
| H3: **standard light** is primary (dim light and pooled are secondary) **[review]** | At 100 lux the 1/30 s exposure cap leaves only sensor gain, which adds no photons; pooling would dilute the effect. Power 0.96 at 1.0 bpm; 0.50 at 0.5 bpm (stated). |
| H4: five joint criteria → **one primary** (lower 95% bound of the darker-stratum reduction ≥ 10%, target 30%; power 0.92), error on **all** windows; coverage, gap, abstention and useful yield as pre-registered secondary outcomes | A point estimate of "≥ 30% lower" succeeds half the time when the truth is exactly 30%; the five-criterion conjunction failed in every simulated study; error on shown windows falls mechanically with abstention. |
| H5: reactivity per stratum → **resting ln RMSSD from the SAMA method, strata pooled; lower 95% bound of ICC(2,1) ≥ 0.60** **[review: the earlier point-estimate rule succeeded half the time at the anchor]**; per-stratum bounds and reactivity secondary | The 0.88 anchor (Pstras et al. 2026) is resting-level agreement in 35 white participants, a weak anchor, so power is also shown at true ICC 0.75 (0.92) and 0.70 (0.50). |
| Multiplicity paragraph | Each hypothesis is a separate paper's primary claim; Holm-adjusted p-values reported alongside. |
| New Figure 7 and power table from `analysis/proposal_power.py` (each criterion as worded; markers at 45) | The October table gave power for criteria that were not the ones written. |
| Cross-conformal calibration of the gate (out-of-fold scores, guarantee ≥ 1 − 2α); Mondrian only at ≥ 30 per stratum | A plain split inside five-fold CV leaves about 7 calibration participants per stratum; 9 are needed for a finite 90% interval. |
| Pilot power re-check with the pilot's own spread and its confidence interval; the H1 re-check uses **the same simulation model** as section 6 (`sama/power_check.py`) **[review]** | Two different power models for the same hypothesis would be indefensible. |

# Claims and references (from the reference audit)

| Change | Why |
|---|---|
| Chari et al. no longer "the largest skin-tone-labelled study"; figures attributed to the facial-averaging baseline; Vital Videos (893 people) cited | Factual corrections |
| Dasari et al. recorded **indoors** | Factual correction |
| QUMPHY guide "describes" (not "recommends") conformal prediction | Factual correction |
| rPPG-Toolbox: **11** methods; MMPD 9–18 (trained) and 12–22 (unsupervised) bpm | Factual correction |
| POS chosen as best on two of three toolbox test sets "in our reading" | The toolbox paper does not call POS "most consistent" |
| Saikia et al. record respiratory rate **with** a finger-oximeter pulse channel | Factual correction |
| Ref 31 → Pstras et al. 2026 with DOI; Tian et al. author list; UBFC-Phys full citation; Barber et al. 2021, McGraw & Wong 1996, Pawar et al. 2024, Monk scale added | Completeness |
| "Worst-group-trained extraction (new)" → the *combination* is claimed as new, each step tested by ablation | The in-silico rehearsal found no benefit of FiLM or group-DRO; a reviewer will probe this |
| "Equal reliability" → **reliability audited per stratum** (coverage, abstention, useful yield) | A guarantee of equal coverage per group needs Mondrian calibration with ≥ 30 per stratum; with fewer, only pooled coverage is guaranteed, and the proposal now says so |

# Design and methods

| Change | Why |
|---|---|
| Skin tone measured three ways: colorimeter (reference), camera ITA° against a grey card and a video-to-ITA° model (validated against it), Monk ratings by one or two raters | Turns the pilot's camera/Monk work into a contribution and answers "how good is a phone at skin tone?" |
| Pre-registered stratum rules: tertile fallback for recruitment, and a Monk fallback for a participant without a usable ITA° | Every participant has a stratum fixed before analysis |
| Fingertip oximeter added to the main study | Comparability with the pilot and public datasets |
| ECG-derived waveform target for SAMA-150 training | SAMA-150 has no contact-PPG target otherwise |
| Synchronisation: shared clock refined by two-pass cross-correlation; automatic session QC | Both were built and used in the pilot |
| Failed windows scored, not dropped **[review]** | Dropping a window where a method gave no estimate favours the methods, and the skin tones, that fail most |
| New risks: compression erases the pulse; H4 margin missed; storage fills **[review]** | Honest risk register |

# Section 5 (preliminary work)

| Change | Why |
|---|---|
| 5.1: the open toolkit and the public-data training run (numbers filled by the build from the trained bundle) | Shows capability and gives real numbers even before the pilot |
| 5.2: the November pilot, every number filled from `results/pilot/pilot_summary.json`; SAMA trained only on public data and tested untouched on Indian faces; the run stops if the pilot was used for training **[review]** | Replaces the bracket template; the build refuses to finish while anything is pending |
| The pilot is described as pre-funding feasibility, **not** the ten instrumented protocol-pilot sessions of WP1 **[review]** | It used a webcam, a fingertip oximeter and a grey card, not the colorimeter, ECG and four cameras |

# SAP_SIM — frozen in-silico analysis plan (C3, tag `sap-v1`)

Frozen 2026-11-02 before any confirmatory in-silico analysis was run. Anything not listed here is
exploratory. Deviations go to DECISIONS.md with reasons. Parameters come from `spec/spec.yaml`.

## 0. Status of the data
No REAL-PUBLIC data could be acquired (ISSUES I-22). Every analysis below runs on SYNTHETIC data
(SAMA-Sim) or, where noted, on SEMI-SYNTHETIC data that do not exist in this run. **Claims ceiling:**
results speak to pipeline correctness, the behaviour of the proposal's statistics under stated
assumptions, and computational feasibility; never to Indian skin, never to whether compression
widens the gap in people, never to whether camera HRV is valid.

The simulator's skin-tone gradient is **calibrated to the proposal's own H1 assumption** (POS MAE
2.5 → 4.5 bpm across ITA 50 → −25; `configs/sim_calibrated.json`). Any H1s result is therefore
ASSUMPTION-DRIVEN by construction; the rehearsal tests the *design*, not the effect.

## 1. Data-generating designs
| Run | Design | Use |
|---|---|---|
| `pool_A`, `pool_B` | 90 per stratum (270), World A / World B, reference camera 720p30, lossless + H.264-500 surrogate, both lights, 8 seated blocks at proposal durations | H1s, H2s power, type-I, discriminability |
| `pool_melanin` | 60 per stratum, counter-world: relative pulse amplitude falls with melanin (I-02) | sensitivity |
| `cohort_A_s7` | one virtual SAMA-120: 40 recruited per stratum, 20 F / 20 M, ages 18–60; full SAMA features; phones A/B, webcam, codec 500/250 on rest + teleconsultation; HRV at 60 fps, 30 fps, 30 fps + 500 kbps | H3s, H4s, ablations, coverage, H5s, E1, error budget |

Session loss: 10 % of participants per stratum dropped at random in each simulated study
(40 → 36). ECG artefact bursts flag beats; a window with > 10 % flagged beats is unusable.

## 2. Confirmatory analogues
Each is reported with an interval and one evidence label.

**H1s.** Endpoint: per-participant mean |HR_POS − HR_ECG| over all 30 s windows (15 s step) of the
seated blocks, both lights, lossless 720p30 reference video. Test: OLS slope of MAE on ITA°,
one-sided (slope < 0 means error rises as ITA falls), α = 0.05. Power: share of 1,000 simulated
studies (36 analysable per stratum drawn without replacement from the pool) with p < 0.05.
Type-I: same, with ITA° permuted across participants within each study. Sensitivities: ITA°
measured with error SD ∈ {0, 3, 6, 10}° (attenuation of the slope, I-03); mixed model
log(1 + e) ~ ITA + light + block + (1 | participant) on windows (Wald z), reported for 200 studies.
Label: SYNTHETIC, ASSUMPTION-DRIVEN.

**H2s.** Endpoint: per-participant penalty = MAE(H.264 500 kbps surrogate) − MAE(lossless).
Test: OLS slope of penalty on ITA°, two-sided. Pre-specified decision rule (proposal s.3):
- "supports A" if slope < 0 with p < 0.05 (penalty larger at lower ITA°);
- "supports B" if not "A" **and** the SNR penalty (SNR_lossless − SNR_500, dB) has slope < 0 with p < 0.05;
- otherwise "inconclusive".
Discriminability: P(A | World A), P(B | World B), P(A | World B), P(B | World A) over 1,000 studies
per world. Label: SYNTHETIC; identifiability only.

**H3s.** Darker stratum, rest + teleconsultation blocks, phone B (adaptive) vs phone A (auto):
per-participant MAE difference (A − B), paired one-sided t, per light and pooled. Also the photon
budget (electrons per pixel, gain) per stratum × light × controller. Label: ASSUMPTION-DRIVEN.

**H4s.** Nested participant-level cross-validation: 5 outer folds stratified by stratum; inside the
outer training set, a stratified 75/25 split into train-proper (network, σ̂ model) and
calibration (conformal). No inner tuning loop beyond fixed defaults (D-09); baselines get the same
folds and the same gate. "Best baseline" = the unsupervised method with the lowest darker-stratum
MAE on the **outer training** participants of that fold (never the test fold). Primary MAE is on
**all windows** (abstained windows scored with the point estimate). Five criteria, each with a
participant-bootstrap 95 % interval (2,000 resamples), then their conjunction:
1. darker-stratum MAE ≥ 30 % lower than best baseline;
2. best-to-worst stratum gap ≥ 50 % smaller;
3. lighter-stratum MAE not raised by more than 10 %;
4. 90 % interval coverage ≥ 88 % in every stratum (on all windows; shown-only reported too, I-27);
5. abstention rates within 10 percentage points across strata.
Also: retained-only MAE; baseline (POS + same σ̂ + same gate) at matched abstention; coverage under
shift (dim light, phone, webcam, 500 and 250 kbps, ITA° estimated with SD 6° error); useful yield.
Ablations: −step 2, −step 3, −CNN, −FiLM, −group-DRO (ERM), −gate. Label: SYNTHETIC; the
neural baselines are missing (I-24).

**H4 design power.** (a) Analytic + simulated distribution of each criterion at n = 36 per stratum
under the proposal's assumptions (I-05, I-06, I-07); (b) participant bootstrap of the cohort's
outer-fold predictions at 36 per stratum: P(each criterion), P(all five).

**H5s.** Endpoint: reactivity = ln RMSSD(arithmetic) − ln RMSSD(rest), standard light, camera
(weighted-patch POS, parabolic peaks) vs ECG. ICC(2,1) absolute agreement with McGraw–Wong 95 %
interval, overall and per stratum, at 60 fps lossless (primary), 30 fps, and 30 fps + 500 kbps.
Bland–Altman per stratum. Power by participant bootstrap at 36 per stratum: P(overall ICC ≥ 0.75
and every stratum's lower bound ≥ 0.60). Design check I-25 (reactivity vs level reliability) and
I-26 (joint per-stratum criterion; Fisher-z vs F-based interval). Label: ASSUMPTION-DRIVEN.

**E1 (exploratory).** Leave-one-subject-out logistic regression, arithmetic vs rest, features HR
and ln RMSSD from ECG vs from camera; accuracy per stratum with bootstrap intervals.

**Recruitment (I-18).** Probability of filling 40 per stratum within a fixed number of screened
volunteers, under ITA° distributions from the reference audit (REFS_AUDIT.md "Indian ITA ranges").

## 3. Error budget (WP2)
Mixed model of log(1 + e) on windows with fixed effects ITA (centred, per 10°), dim light, device
(webcam, phone A, phone B vs reference), codec (500, 250 vs lossless), ITA × each, and random
intercepts for participant and block (REML, `sama.stats.lmm_crossed`). Planted-parameter recovery
check: on SAMA-Sim data where the simulator's true ITA effect is known only implicitly, recovery is
checked instead on a synthetic linear-model dataset with planted γ, δ (tolerance: 95 % Wald
intervals cover the planted value in ≥ 90 % of 200 replicates). Convergence failures and boundary
fits are counted and reported.

## 4. Seeds, multiplicity, tolerances
Master seed 20261102; pools seed 1 (A), 2 (B), 3 (melanin); cohort seed 7; study resampling seed
2026. One primary test per hypothesis at α = 0.05; secondaries Holm-corrected within hypothesis.
C2 implementation tolerance: Pearson r > 1 − 10⁻⁶ between our and rPPG-Toolbox outputs for GREEN,
POS, CHROM, LGI, PBV (ICA excluded: different algorithm, D-08). Simulator calibration tolerance:
±0.5 bpm of the anchor at ITA 45° and −20°.

## 5. What a pass or a fail may mean
| Result | May say | May not say |
|---|---|---|
| H1s power ≥ 0.8 | "Under the assumed gradient and the simulator's error distribution, n = 36 per stratum detects it" | anything about the size of the gradient in people |
| H2s discriminability high | "The H2 analysis can tell World A from World B as simulated" | which world is true |
| H3s gain | "The photon budget bounds the benefit at X under the simulator" | that adaptive exposure works on phones |
| H4s criteria | "The SAMA pipeline meets / misses criterion k on SAMA-Sim" | "SAMA achieves" without the label |
| H5s ICC | "Timing quantisation at 30 fps costs X ICC under the simulator" | that camera HRV is valid |

# GO / NO-GO — SAMA in silico (C10)

## Verdict: **CONDITIONAL GO — revise the design text before submission**

"Conditional go" here means that the proposal's design text should be revised before the
17 November upload. It is **not** evidence that SAMA works. No real recording was analysed, so
Q-A (feasibility on real data) has no evidence: `make realdata` now runs the loaders, which were
tested only on synthetic files in the UBFC layout.

The firm findings hold under the proposal's own assumptions, without the simulator. Read as
written, five things in the design are weaker than the power table says:
- H4's criterion 1, read as a point estimate, is met only about half the time when the true effect is exactly the target;
- H5's joint per-stratum criterion has power 0.74;
- a plain conformal split inside 5-fold cross-validation leaves too few calibration participants;
- the coverage criterion is fragile;
- the H2 rule counts a null as support for B.

The simulator results (H1 heavy tails, H3 size, H4 abstention imbalance, H5 ICC, compression)
only illustrate these risks. The C5 gate failed: no real dataset could be matched. So these are
not findings about the method or about people.

**Every recommendation below that comes from the rehearsal was chosen after seeing simulated
results.** Each must be pre-registered before any data are collected. Changing endpoints, for
example to log MAE, changes the estimand. Type-I error for the alternative endpoints was not
estimated (R10-05).

Evidence labels used below: **LIT** = LITERATURE; **AD** = ASSUMPTION-DRIVEN (follows from the
proposal's assumptions or the simulator's premises); **SYN** = SYNTHETIC rehearsal;
**SW** = software test. No row below is REAL-PUBLIC.

## Hypothesis by hypothesis

| Element | Evidence (label) | Status | Risk | Change to the proposal (exact wording) |
|---|---|---|---|---|
| **H1** | Independent rerun under the proposal's assumptions: {{pr_h1:.2f}} (AD). On full sessions, the simulator's gradient was {{c6_grad_A:.2f}} bpm per 75° ITA° (95 % CI {{c6_grad_A_lo:.2f}} to {{c6_grad_A_hi:.2f}}). That is below the 2.0 assumed, and it explains much of the power loss. Per-participant MAE was heavy-tailed (SD {{c6_sd_lighter:.1f}}–{{c6_sd_darker:.1f}} bpm; medians {{c6_med_light_A:.2f}}–{{c6_med_dark_A:.2f}}). Like-for-like resampling of the pool: MAE {{c6_h1_cond:.2f}}, log MAE {{c6_h1_logmae:.2f}}, rank test {{c6_h1_rank:.2f}}. Type-I error was checked for MAE only ({{c6_h1_typeI:.3f}}; {{c6_h1_err6:.2f}} power and 0.061 type-I with ITA° error SD 6°) (SYN). | Fragile | Depends on the effect size and the spread. Neither is known. | *Post-hoc suggestion, to be pre-registered:* "log of per-participant MAE (primary; estimand: ratio of errors per 10° ITA°), rank correlation as sensitivity; type-I error to be verified by simulation before registration". s.6: "between-person SD to be estimated from the pilot; raw-MAE power falls to 0.43 at SD 4 bpm". |
| **H2** | The rerun reproduces 0.80 (AD). In the rehearsal, the frozen rule named World A when A was true in {{c6_h2_correct_A:.0%}} of studies. It **never** named World B when B was true ({{c6_h2_B_inconcl:.0%}} inconclusive; the equivalence rule {{c6_h2_B_equiv:.1%}}): at 36 per stratum, the penalty slope's CI is too wide to show "no widening". Type-I error {{c6_h2_typeI:.3f}}. On rendered tiles, real H.264 kept about {{codec_small_500:.0%}} of sub-0.3 DN modulations at 500 kbps. This is a noisy, non-monotone estimate, and the stand-in was fitted to the same tiles, without held-out validation (SYN rendering, real codec). | Asymmetric | H2 can confirm widening but cannot confirm its absence | s.3 H2 criterion → "Estimate the 500 kbps penalty slope on ITA° with a 95 % CI. A slope below zero (p < 0.05) supports prediction A. A CI inside ±0.5 bpm per 75° ITA° (equivalence) supports B. Otherwise the result is inconclusive. At 36 per stratum, power is adequate for A and not for B." |
| **H3** | The rerun reproduces 0.90 at an assumed 1.0 bpm (AD). Rehearsal, darker stratum: gain {{h3_dark_pooled:.2f}} bpm (95 % CI {{h3_dark_pooled_lo:.2f}} to {{h3_dark_pooled_hi:.2f}}), {{h3_dark_std:.2f}} in standard light and {{h3_dark_dim:.2f}} in dim light (p = {{h3_dark_dim_p:.2f}}). At the 1/30 s cap, extra gain cannot add photons (AD). | Effect likely smaller than powered | Under-powered if the true gain is about 0.5 bpm | s.3 H3: "1.0 bpm is the smallest reduction that matters" → "the primary analysis pools both lights. A gain is expected mainly in good light, because at 100 lux the 1/30 s exposure cap leaves only sensor gain, which adds no photons." Methods: "Both phones run Open Camera; phone A uses its automatic exposure and phone B the adaptive rule" (I-13). |
| **H4 criterion 1** (≥ 30 % lower) | Read as a point estimate, the criterion has a {{h4rule_point30:.0%}} chance of being met when the true effect is exactly 30 % (AD). Rehearsal: {{h4_dark_red:.0%}} lower (95 % CI {{h4_dark_red_lo:.0%}} to {{h4_dark_red_hi:.0%}}) against POS (SYN, no neural baselines). | Wording | The stated 0.99 is power for "any reduction". | "Darker-stratum MAE lower than the best baseline, with the lower 95 % confidence bound of the relative reduction at least 10 %; target reduction 30 % (power 0.83 under s.6 assumptions)." |
| **H4 criteria 2 and 5** | Rehearsal: the gap shrank {{h4_gap_red:.0%}} and SAMA's abstention differed by {{h4_abst_spread:.0f}} pp across strata. All five criteria were met in {{h4_p_all:.2f}} of participant resamples, a resampling probability, not power (SYN). This imbalance is specific to SAMA's σ̂: POS with the same gate abstained **least** in the darker stratum (64 / 68 / 39 %). So it is a risk, not a law. | Fragile | H4 may fail as written | *Post-hoc suggestion:* report the gap and the useful yield per stratum with CIs, instead of hard thresholds. Note that the alternative "yield ratio ≥ 0.8" also failed in the rehearsal (0.66/0.96 = 0.69). Any threshold must be pre-registered and powered before data collection. |
| **H4 criterion 4** (coverage ≥ 88 % in every stratum) | With 36 calibration participants, P(all three strata ≥ 0.88) = {{cov36_all:.2f}} for a valid 90 % method (AD). The proposal does not specify the calibration split inside nested CV. Any plain split that leaves ≤ 8 participants per stratum gives infinite 90 % intervals (n ≥ 9 needed). The in-silico plan's 75/25 split left 7 (SYN, I-30). | Fragile | The gate cannot be evaluated with a small calibration set | Methods: "cross-conformal (CV+) on all outer-training participants; the deployed gate is calibrated on ≥ 30 participants per stratum" (the shipped demonstrator uses 18; R10-21). Criterion: coverage reported per stratum with participant-bootstrap CIs, plus a non-inferiority margin to be fixed before data (e.g. ≥ 85 %). |
| **H5** | The proposal's 0.90 is power for one stratum. The joint criterion gives {{pr_h5_joint:.2f}} at 36 and {{pr_h5_joint40:.2f}} at 40 per stratum (AD). The reliability of a reactivity score is 0.29–0.85 when the level's reliability is 0.88, depending on the assumed SD of true change (0.47–0.64 at SD 0.22) (AD). The anchor r = 0.88 is camera-vs-ECG validity, not test–retest reliability. Rehearsal: reactivity ICC {{h5_icc60:.2f}} vs level ICC {{h5_level60:.2f}} (SYN; depends on our pulse model and beat detector). | Fragile | The endpoint is mis-anchored | *Suggestion:* primary = resting ln RMSSD agreement (ICC; strata pooled for the bound, each stratum reported); reactivity secondary; joint power stated. |
| **E1** | LOSO accuracy, rest vs arithmetic: ECG {{e1_ecg:.2f}}, camera {{e1_cam:.2f}} (SYN) | Exploratory | — | none |
| **Sample size 36/stratum** | Adequate under the proposal's assumptions for H1 (if SD ≈ 2), H3 (if the gain is ≈ 1 bpm) and for H4's "any reduction". Not adequate for H4's point-estimate reading (0.51) or for H5's joint criterion (0.74) (AD) | — | — | Keep 40/36. State the full-criterion powers. |
| **Recruitment** | About {{rec_lighter:.0%}} of Indian adults have ITA > 28° and {{rec_darker:.0%}} have ITA ≤ 10°, so a median of {{rec_screen:.0f}} people must be screened to fill 40 lighter participants (LIT, Pawar et al. 2024, not Surat) | Risk reversed | Lighter stratum fills slowly | s.9 risk register: "the darker stratum may fill slowly" → "the lighter stratum (ITA > 28°) is likely the scarce one in India (≈15 % of adults in one colorimeter study); screening of ≈260–310 volunteers is budgeted; strata may be redefined by tertiles if the pilot confirms this." |
| **Reliability gate under shift** | Coverage held at 0.94–1.00 under device, light and ITA°-estimate shifts on the test folds. The codec-shift rows used the unvalidated stand-in v0 and are not evidence (SYN). The acceptance test still gave 7 confident wrong readings: lighter stratum, phone + 250 kbps, refitted codec (SW/SYN). | Conditional | The narrowest (lighter) intervals fail first under unseen compression | Methods: "the gate is calibrated on the codec and device mix used in deployment; below 500 kbps the app abstains." |
| **Method novelty claims** | Ablation: FiLM and group-DRO add nothing on SAMA-Sim. The CNN refinement drives the gain. At matched abstention, darker retained MAE does not improve (SYN). | Unsupported in silico | Over-claim | Reword "worst-group-trained extraction (new)" → "extraction trained with group-DRO; whether worst-group training and skin-tone conditioning help is tested by ablation (WP3)". |
| **Deployment** | ONNX fp32/fp16/int8 parity ≤ {{parity_int8:.3f}} bpm; {{mem_mb:.0f}} MB; first reading at 30 s; no files written. **Two acceptance failures:** steady-state real-time factor {{rtf_onnx:.2f}} on one core (threshold 0.5; real time but over budget; a 2 s update interval would halve it), and {{faults_wrong:.0f}} of {{faults_shown_250:.0f}} shown readings wrong at phone + 250 kbps (lighter stratum). Android source NOT COMPILED; the step-2/3/FiLM inputs used simulator truth (true pixel counts, true ITA°) (SW/SYN) | Conditional | Not ready for live use beyond a demo | s.4: "(source and exported model ready; device latency and battery to be measured)". |

## Updated risk register (additions)
1. No public data analysed. Section 5 must not cite public-data numbers until `make realdata` has run.
2. H4 and H5 as written are likely to fail even if the method works (above).
3. Compression may erase most of the pulse at teleconsultation bitrates. Real H.264 kept under 10 % of small modulations on rendered tiles. This raises the value of "measure before you compress".
4. Lighter-stratum recruitment.
5. The gate needs per-device calibration and a bitrate floor.
6. Heavy-tailed per-participant errors: power depends on the spread, which the pilot must estimate.

## Statements the proposal must not make
- That Chari et al. (2020) is "the largest skin-tone-labelled rPPG study" (VitalVideos, 893 participants), or that it used an ECG reference. Its 3.9→6.2 bpm figures are the facial-aggregation baseline, and "+2.06 bpm" is a derived difference, not a published figure.
- That Dasari et al. (2021) recorded outdoors: they recorded indoors.
- That the QUMPHY guide "recommends" conformal prediction: it describes it.
- That rPPG-Toolbox benchmarks 12 methods (it benchmarks 11), or that it calls POS the "most consistent".
- Any public-data or pilot number in s.5 that has not been measured.
- That SAMA "achieves" anything, that its FiLM/group-DRO steps are known to help, or any device latency or battery figure.
- "First Indian rPPG dataset with …", unless the C1 audit is extended to every Indian dataset. Keep "to our knowledge".

## Diff-ready replacement text
See `proposal_inserts/proposal_change_list.md` (sections 3, 5, 6 and 9.3).

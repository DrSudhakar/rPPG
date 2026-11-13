# Proposal change list — diff-ready text for the 17 November upload

Each item gives the current text, the replacement, and the evidence behind it. **Items 1, 4, 5 and 13 were chosen after seeing simulated results. They are suggestions to pre-register before data collection, not pre-specified findings.** Type-I error for the alternative endpoints was not estimated. Evidence labels:
AD = assumption-driven (the proposal's own assumptions); SYN = SAMA-Sim rehearsal;
LIT = literature (REFS_AUDIT.md). No item rests on real participant data.

## Section 3 (hypotheses table)

1. **H1 endpoint**
   - Current: "Heart-rate MAE, POS, lossless reference video, both lights".
   - Replace with: "Log of per-participant heart-rate MAE (POS, lossless reference video, both lights). Sensitivity analysis: Spearman rank correlation with ITA°."
   - Why (SYN): heavy-tailed errors. Power is {{c6_h1_cond:.2f}} for raw MAE, {{c6_h1_logmae:.2f}} for log MAE and {{c6_h1_rank:.2f}} for the rank test.
2. **H2 success criterion**
   - Current: "A clear answer either way: widening supports prediction A; a null or narrowing, with a widening signal-to-noise gap, supports B".
   - Replace with: "The slope of the 500 kbps penalty on ITA°, with a 95 % CI. A slope below zero (p < 0.05) supports A. A CI inside ±0.5 bpm per 75° of ITA° (two one-sided tests) supports B. Anything else is reported as inconclusive; the study is powered to detect A, not to establish B."
   - Why (SYN): in World B, B was named in 0 % of 1,000 simulated studies.
3. **H3**
   - Current: "Mean reduction at p < 0.05; 1.0 bpm is the smallest reduction that matters".
   - Replace with: "Mean reduction, pooled over both lights, at p < 0.05. Per-light results are reported with Holm correction. Both phones run Open Camera: phone A with automatic exposure, phone B with the adaptive rule."
   - Why: AD/SYN, I-13 and I-14. The gain is expected mainly in good light, and the rehearsal gave {{h3_dark_pooled:.2f}} bpm.
4. **H4**
   - Current: "At least 30% lower; best-to-worst gap at least 50% smaller; lighter-stratum MAE not raised by more than 10%; coverage of 90% intervals at least 88% in every stratum; abstention rates within 10 percentage points across strata".
   - Replace with: "Primary: darker-stratum MAE over all windows (abstained windows scored with the point estimate) lower than the best baseline, with the lower 95 % bound of the relative reduction at least 10 % (target 30 %). Secondary, each with participant-bootstrap CIs:
     - lighter-stratum MAE not raised by more than 10 %;
     - coverage of 90 % intervals at least 85 % in every stratum, on all windows and on shown windows;
     - best-to-worst MAE gap;
     - useful yield (readings shown and within ±5 bpm) per stratum, with a darker-to-lighter yield ratio of at least 0.8."
   - Why: the criterion read as a point estimate succeeds {{h4rule_point30:.0%}} of the time when the true effect is exactly 30 % (AD). All five criteria together succeeded {{h4_p_all:.0%}} of the time (SYN), because the gate abstains {{abst_darker:.0%}} vs {{abst_lighter:.0%}} (darker vs lighter). This change was chosen after seeing simulated results, so pre-register it before any data are collected (R7-04). Note: the yield-ratio threshold 0.8 itself failed in the rehearsal (0.69). Fix any threshold only after powering it.
5. **H5**
   - Current: "RMSSD reactivity (arithmetic minus rest) … Overall ICC at least 0.75; lower 95% bound at least 0.60 in each stratum".
   - Replace with: "Resting ln RMSSD, ICC(2,1) absolute agreement: overall at least 0.75, with the lower 95 % bound at least 0.60 for the strata pooled. Each stratum's ICC is reported with its CI. Reactivity (arithmetic minus rest) is secondary."
   - Why: a reactivity score's reliability is 0.29–0.85 (0.47–0.64 at SD of change 0.22) when the level's reliability is 0.88; the proposal's 0.88 is camera-vs-ECG validity, not test–retest reliability (AD). The joint per-stratum criterion has power {{pr_h5_joint:.2f}}, not 0.90. The rehearsal gave a reactivity ICC of {{h5_icc60:.2f}} vs a resting-level ICC of {{h5_level60:.2f}} (SYN).

## Section 4 (methods)

6. **WP3 step 4**
   - Current: "Physics-guided, worst-group-trained extraction (new)".
   - Replace with: "Physics-guided extraction with group-DRO training (whether worst-group training and skin-tone conditioning help is tested by ablation)."
   - Why (SYN, I-32): FiLM and group-DRO added nothing in the rehearsal.
7. **WP3 step 6** — add: "Inside nested cross-validation the gate is calibrated by cross-conformal prediction (CV+; Barber et al. 2021) on all outer-training participants. A plain split leaves about 7 calibration participants per stratum, too few for a finite 90 % interval. The deployed gate is calibrated on at least 30 participants per stratum, using the codec and device mix of deployment. The 200 one-window draws are combined by their median; the rehearsal used the mean, which over-covered." (I-30, I-31, I-29, I-33)
8. **WP3 training** — add: "SAMA-120 records ECG but no contact PPG. The network's waveform target is a pulse template placed on the ECG R peaks; its heart-rate target comes from the ECG." (R3-13)
9. **Section 4, recording**
   - Current: phone B uses "open-source Open Camera app".
   - Replace with: "both phones run Open Camera".

## Section 5

10. Replace the bracketed pilot paragraph with real pilot results when available. Until then, use `proposal_inserts/section5_public_data.md`, which reports a design rehearsal and no data. Do not use `figure6_SYNTHETIC_NOT_FOR_SUBMISSION.png` in the proposal. It only shows that `pilot_figure.py` runs end to end.

## Section 6

11. **Power table**
    - Replace the H4 and H5 rows' power with the full-criterion values: H4 primary (lower bound ≥ 10 % at a true 30 %) 0.83; H5 joint {{pr_h5_joint:.2f}} at 36 and {{pr_h5_joint40:.2f}} at 40. Alternatively, adopt items 4–5 and state the new powers.
    - H1 spread: "Between-person SD 2.0, right-skewed" → "log-MAE SD to be estimated from the pilot; raw-MAE power falls to 0.43 at SD 4 bpm".
    - H2 spread label: "Within-person SD 1.0" → "Between-participant SD of the penalty 1.0" (I-28).
12. Add: "Each criterion was simulated as written; the code is released with the protocol."

## Section 9.3 (risks)

13. **First risk row**
    - Current: "The darker stratum (ITA° ≤ 10) fills slowly".
    - Replace with: "The lighter stratum (ITA° > 28) fills slowly. In a colorimeter study of 368 Indian adults, about {{rec_lighter:.0%}} had ITA° > 28° and {{rec_darker:.0%}} had ITA° ≤ 10°. Mitigation: screen about {{rec_screen:.0f}}–{{rec_screen90:.0f}} volunteers, with outreach for the lighter quota. If the pilot confirms the imbalance, strata are redefined by local tertiles before recruitment."
14. **New risk:** "Compression may remove most of the pulse at call bitrates. Real H.264 kept about {{codec_small_500:.0%}} of pulse-sized modulations at 500 kbps on rendered test patches. Mitigation: measure on the phone before compression; the gate abstains below 500 kbps."
15. **New risk:** "H4/H5 criteria fail although the method helps." Mitigation: the revised criteria (items 4–5).

## Sections 1–2 and references (LIT, REFS_AUDIT.md)

16. **Chari et al. (2020)**:
    - not "the largest skin-tone-labelled rPPG study" (VitalVideos has 893 participants);
    - 3.9→6.2 bpm is the facial-aggregation baseline, with a PPG reference rather than ECG;
    - "+2.06 bpm" is our subtraction, not a published figure.
17. **Dasari et al. (2021)**: the 44 Indian adults were recorded *indoors*, not outdoors.
18. **Harris et al. (2026)**: "recommends conformal prediction" → "describes conformal prediction among uncertainty methods for PPG".
19. **rPPG-Toolbox**: 11 methods benchmarked, not 12; on MMPD, MAE is 9–18 bpm for supervised and 12–22 bpm for unsupervised methods.
20. **Saikia et al. (2026)**: "respiration only" → "respiratory rate, with a finger-oximeter pulse channel".
21. **Ref 31**: add authors: Pstras L, Okupnik T, Ponikowska B, Paleczny B. Sci Rep 2026, doi:10.1038/s41598-026-70292-0.
22. **Nowara 2020**: title uses "Skin Tone", not "Skin Type".
23. **UBFC-Phys**: full citation in REFS_AUDIT.md item 20.
24. **ANSI/AAMI EC13:2002**: add the year.

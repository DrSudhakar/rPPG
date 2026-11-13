
## 3. Results

### 3.1 R1–R2: data, provenance and replication

No REAL-PUBLIC or SEMI-SYNTHETIC result exists in this report (Table 2), so Q-A is answered only for software. The implementation check passed for all five deterministic methods (Table 3). Replicating the published rPPG-Toolbox error on UBFC-rPPG and PURE, which is the C2 gate, could not be attempted. Everything downstream is therefore provisional in the sense the protocol defines.

Table 3. Implementation check against rPPG-Toolbox (commit b7500b8), identical synthetic RGB input (SW).

| Method | GREEN | POS | CHROM | LGI | PBV | ICA |
|---|---|---|---|---|---|---|
| Pearson r with toolbox output | > 1 − 10⁻⁶ | > 1 − 10⁻⁶ | > 1 − 10⁻⁶ | > 1 − 10⁻⁶ | > 1 − 10⁻⁶ | not compared (FastICA vs JADE) |

![](results/figures/fig_sim_vs_reality.png){width=100%}

<p class="caption">Figure 2. (A) Median per-participant POS error by stratum in the simulator pools, against the proposal's assumed line (SYNTHETIC vs LITERATURE). (B) Real-codec gain curves, as in Figure 1 (SYNTHETIC rendering, real codec).</p>

### 3.2 R4: does the simulator behave like reality?

The simulator reproduced the two qualitative patterns required: error rises as light falls, and compression degrades error (Figures 2 and 6). It matched the LITERATURE anchor only at the calibration points. On full-length sessions, the World A pool's gradient came out at {{c6_grad_A:.2f}} bpm per 75° of ITA° (95 % CI {{c6_grad_A_lo:.2f}} to {{c6_grad_A_hi:.2f}}), smaller than the 2.0 it was tuned to on short blocks. In the counter-world, where melanin lowers relative pulse amplitude, the gradient was {{c6_grad_mel:.1f}} bpm. Per-participant errors were heavy-tailed. The mean MAE was {{c6_mae_light_A:.2f}} bpm for the lighter and {{c6_mae_dark_A:.2f}} for the darker stratum, the medians {{c6_med_light_A:.2f}} and {{c6_med_dark_A:.2f}}, and the SDs {{c6_sd_lighter:.1f}}–{{c6_sd_darker:.1f}} bpm, against the 2.0 the proposal assumes. The C5 gate's requirement to reproduce one real dataset quantitatively could not be met. Simulator-based claims are therefore restricted to pipeline correctness and to how the design behaves under stated assumptions (ISSUES I-02).

### 3.3 R5: the rehearsal for H1 and H2

**H1s (SYNTHETIC, ASSUMPTION-DRIVEN).** At 36 analysable participants per stratum, the one-sided slope test detected the simulator's gradient in {{c6_h1_cond:.0%}} of 1,000 resampled studies and in {{c6_h1_param:.0%}} with fresh parametric participants. Type-I error under ITA° permutation was {{c6_h1_typeI:.3f}}. With ITA° measured with error SD 6° or 10°, power fell to {{c6_h1_err6:.2f}} and {{c6_h1_err10:.2f}}. The window-level mixed model had power {{c6_h1_mm:.2f}}, with {{c6_h1_mm_nonconv:.0f}} non-converged fits in 100. Endpoints robust to heavy tails did much better: log MAE {{c6_h1_logmae:.2f}}, Spearman rank {{c6_h1_rank:.2f}}. The per-participant median error did not ({{c6_h1_medae:.2f}}), because the median throws away the failing windows that carry the skin-tone effect. Under its own assumptions the proposal's H1 power reruns at {{pr_h1:.2f}}. It holds if the between-person SD is about 2 bpm, and drops to 0.43 at 4 bpm.

![](results/figures/fig_h1_rehearsal.png){width=100%}

<p class="caption">Figure 3. H1s. (A) Per-participant POS MAE against ITA° in the World A pool, on a log scale, with the proposal's assumed line. (B) Power against sample size in the rehearsal (parametric) and under the proposal's assumptions (SYNTHETIC, ASSUMPTION-DRIVEN).</p>

**H2s (SYNTHETIC, identifiability only).** The pooled penalty slope was {{c6_h2_slopeA:.1f}} bpm per 75° of ITA° in World A, where small pulses are removed first. In World B it was {{c6_h2_slopeB:.2f}} (95 % CI {{c6_h2_slopeB_lo:.2f}} to {{c6_h2_slopeB_hi:.2f}}). The pre-specified rule named World A in {{c6_h2_correct_A:.0%}} of World A studies. It never named World B in World B studies: {{c6_h2_B_inconcl:.0%}} were "inconclusive", because the signal-to-noise gap does not widen when noise is relative. The amended equivalence rule named B in {{c6_h2_B_equiv:.1%}}. Type-I error was {{c6_h2_typeI:.3f}}. At this sample size H2 can confirm widening but not its absence (Figure 4).

![](results/figures/fig_h2_discriminability.png){width=90%}

<p class="caption">Figure 4. H2s discriminability over 1,000 simulated studies per world, frozen and amended decision rules (SYNTHETIC).</p>

**Error budget (WP2; SYNTHETIC).** The crossed random-intercept mixed model converged on {{budget_nwin:.0f}} windows. In a planted-parameter check on the same design, every 95 % interval covered its planted value in at least {{budget_planted:.0%}} of 100 replicates. In the simulator, dim light and heavy compression added error mainly in the darker stratum. Phones with auto-exposure did slightly better than the fixed-exposure reference, because they brightened darker faces (Figure 5).

![](results/figures/fig_error_budget.png){width=100%}

<p class="caption">Figure 5. Error budget by stratum: predicted window error with each component, against the lossless good-light reference (black tick), with participant-bootstrap 95 % CIs (SYNTHETIC; codec rows use the unvalidated stand-in v0, D-23).</p>


Table 4. H1s–H5s and E1 at a glance (36 analysable per stratum; intervals are 95 %).

| Analogue | Estimate | Interval | Test / probability | Label |
|---|---|---|---|---|
| H1s gradient (pool A) | {{c6_grad_A:.2f}} bpm per 75° | {{c6_grad_A_lo:.2f}} to {{c6_grad_A_hi:.2f}} | power {{c6_h1_param:.2f}} (MAE), {{c6_h1_logmae:.2f}} (log MAE); type-I {{c6_h1_typeI:.3f}} | SYNTHETIC, AD |
| H2s penalty slope, World A / B | {{c6_h2_slopeA:.1f}} / {{c6_h2_slopeB:.2f}} bpm per 75° | B: {{c6_h2_slopeB_lo:.2f}} to {{c6_h2_slopeB_hi:.2f}} | A named {{c6_h2_correct_A:.0%}}; B named 0 %; type-I {{c6_h2_typeI:.3f}} | SYNTHETIC |
| H3s darker, B − A | {{h3_dark_pooled:.2f}} bpm | {{h3_dark_pooled_lo:.2f}} to {{h3_dark_pooled_hi:.2f}} | p = {{h3_dark_pooled_p:.3f}} | AD |
| H4s darker reduction | {{h4_dark_red:.0%}} | {{h4_dark_red_lo:.0%}} to {{h4_dark_red_hi:.0%}} | paired p = {{h4_paired_p:.3f}}; all five criteria {{h4_p_all:.2f}} | SYNTHETIC |
| H5s reactivity ICC (60 fps) | {{h5_icc60:.2f}} | {{h5_icc60_lo:.2f}} to {{h5_icc60_hi:.2f}} | criterion met {{h5_ppass:.2f}} | AD |
| E1 accuracy, ECG / camera | {{e1_ecg:.2f}} / {{e1_cam:.2f}} | — | exploratory | SYNTHETIC |

### 3.4 R6: the SAMA method — ablation, H3s and H4s

**Frozen plan.** The pre-specified 75/25 split gave seven calibration participants per stratum in every fold. Every conformal quantile was infinite, so the gate would have abstained on every window. This is the outcome of the plan as frozen. Amendment A1 (a 50/50 split, about 14 per stratum) is the operative analysis.

**Ablation and H4s (SYNTHETIC; no neural baselines).** POS was the best baseline in every fold. Measured over all windows, SAMA's darker-stratum MAE was {{abl_SAMA_darker:.2f}} bpm against {{abl_POS_gate_darker:.2f}} for POS, a reduction of {{h4_dark_red:.0%}} (95 % CI {{h4_dark_red_lo:.0%}} to {{h4_dark_red_hi:.0%}}). The gain came from the CNN refinement, not from the parts claimed as new (Figure 7):

- removing FiLM gave a darker MAE of {{abl_noFiLM_darker:.2f}};
- removing group-DRO gave {{abl_nogroupDRO_darker:.2f}};

- steps 2–3 without the network gave {{abl_noCNN_darker:.2f}}.

Coverage of the 90 % intervals was {{cov_lighter:.2f}} / {{cov_middle:.2f}} / {{cov_darker:.2f}} (lighter / middle / darker), above nominal. Abstention was {{abst_lighter:.0%}} / {{abst_middle:.0%}} / {{abst_darker:.0%}} and useful yield {{yield_lighter:.0%}} / {{yield_middle:.0%}} / {{yield_darker:.0%}} (Figure 6). At matched abstention and participant-level averaging, SAMA's retained darker-stratum MAE was no better than POS's: POS minus SAMA {{matched_dark_diff:.2f}} bpm (95 % CI {{matched_dark_lo:.2f}} to {{matched_dark_hi:.2f}}). Most of the darker-stratum gain therefore lies in windows that the gate withholds anyway. Table 5 lists the five H4 criteria; all five were met together in {{h4_p_all:.0%}} of participant resamples. That figure is the probability of meeting a criterion given this cohort's fitted models, not design power (R7-03).

![](results/figures/fig_coverage_abstention.png){width=100%}

<p class="caption">Figure 6. Equal reliability is not equal availability. (A) Coverage of 90 % intervals on all windows (bars) and on shown windows (dots); (B) abstention; (C) useful yield, by stratum, for SAMA and for POS with the same gate (SYNTHETIC).</p>

![](results/figures/fig_ablation.png){width=100%}

<p class="caption">Figure 7. Ablation: participant-averaged MAE on all windows by stratum (SYNTHETIC; neural baselines absent).</p>

Table 5. H4s criteria, one by one (SYNTHETIC; 5-fold participant CV; 95 % participant-bootstrap CI).

| Criterion | Required | Point estimate | 95 % CI | P(met), resampling |
|---|---|---|---|---|
| 1. Darker MAE reduction vs best baseline | ≥ 30 % | {{h4_dark_red:.0%}} | {{h4_dark_red_lo:.0%}} to {{h4_dark_red_hi:.0%}} | {{h4_p_c1:.2f}} |
| 2. Best-to-worst gap reduction | ≥ 50 % | {{h4_gap_red:.0%}} | wide (see c7_h4_criteria.csv) | {{h4_p_c2:.2f}} |
| 3. Lighter MAE change | ≤ +10 % | {{h4_light_inc:.0%}} | below 0 | 1.00 |
| 4. Coverage, every stratum | ≥ 88 % | {{h4_mincov:.0%}} (minimum) | ≥ 92 % | 1.00 |
| 5. Abstention spread | ≤ 10 pp | {{h4_abst_spread:.0f}} pp | 23–41 pp | {{h4_p_c5:.2f}} |
| All five | — | — | — | {{h4_p_all:.2f}} |

**H3s (ASSUMPTION-DRIVEN).** In the darker stratum, adaptive exposure lowered POS error by {{h3_dark_pooled:.2f}} bpm pooled over lights (95 % CI {{h3_dark_pooled_lo:.2f}} to {{h3_dark_pooled_hi:.2f}}; p = {{h3_dark_pooled_p:.3f}}). The gain was {{h3_dark_std:.2f}} in good light and {{h3_dark_dim:.2f}} in dim light (p = {{h3_dark_dim_p:.2f}}). This is the photon budget at work. At 100 lux both controllers hit the 1/30 s cap and differ only in gain, which adds no photons (`h3_photon_budget.csv`). The rehearsal's gain is about half the 1.0 bpm that the proposal powers for.

### 3.5 R7: HRV and arousal

Camera RMSSD reactivity agreed poorly with ECG: ICC {{h5_icc60:.2f}} (95 % CI {{h5_icc60_lo:.2f}} to {{h5_icc60_hi:.2f}}) at 60 fps lossless. It fell to {{h5_icc30hq:.2f}} at 30 fps with the same sensor, {{h5_icc30:.2f}} at 720p30 and {{h5_icc500:.2f}} at 500 kbps (Figure 8). The resting level agreed better (ICC {{h5_level60:.2f}}), as the reliability of a difference score predicts. On the proposal's own anchor, a resting-level r of 0.88 implies a reactivity reliability of 0.45–0.65 (`h5_reactivity_icc.csv`). P(H5 criterion met) at 36 per stratum was {{h5_ppass:.2f}} in the rehearsal. Under the proposal's assumptions the joint criterion has power {{pr_h5_joint:.2f}} (Fisher-z single-stratum {{pr_h5_one:.2f}}). E1 leave-one-subject-out accuracy (rest vs arithmetic) was {{e1_ecg:.2f}} from ECG features and {{e1_cam:.2f}} from camera features. These HRV numbers depend on the simulator's pulse model and on our beat detector, and are ASSUMPTION-DRIVEN.

![](results/figures/fig_h5_icc.png){width=88%}

<p class="caption">Figure 8. H5s: ICC(2,1) of ln RMSSD reactivity, camera vs ECG, overall (95 % CI) and by stratum, for four capture conditions (SYNTHETIC, ASSUMPTION-DRIVEN).</p>

### 3.6 R8: deployment benchmarks

Table 6 summarises the C9 acceptance tests, whose thresholds were fixed beforehand. The streaming pipeline ran at a real-time factor of {{rtf_onnx:.2f}} (onnxruntime) and {{rtf_numpy:.2f}} (numpy) on one 2.1 GHz Xeon core. These are CPU-proxy figures: no phone was available. It used {{mem_mb:.0f}} MB of Python memory and produced its first reading after 30 s, then one per second (median compute {{update_ms:.0f}} ms per update). Weight-only fp16 and int8 exports matched numpy within {{parity_fp16:.4f}} and {{parity_int8:.3f}} bpm. The fault-injection test **failed**. Dropped frames, frame-rate changes, exposure flips, face loss, heavy motion, lighting changes and dim light produced correct readings or coded abstentions. Heavy compression (phone, 250 kbps) produced {{faults_wrong:.0f}} confident wrong readings, all in the lighter stratum, whose narrow intervals were calibrated without such compression. No file was written during streaming. The 56 unit tests cover {{coverage_pct:.0f}} % of core lines. The web demonstrator sends only patch colour means.

![](results/figures/app_simulator.png){width=62%}

<p class="caption">Figure 10. The web demonstrator (simulator tab), streaming a SAMA-Sim participant: "No reliable reading" until 30 s are buffered, then readings with 90 % intervals beside the simulated true heart rate (SYNTHETIC; research-use label shown on every page).</p>

Table 6. Deployment acceptance tests (SW; SYNTHETIC inputs; CPU proxy).

| Test | Threshold | Result | Pass |
|---|---|---|---|
| Export parity, fp16 / int8 | < 0.1 bpm | {{parity_fp16:.4f}} / {{parity_int8:.3f}} bpm | yes |
| Real-time factor, 1 core | ≤ 0.5 | {{rtf_onnx:.2f}} | yes (marginal) |
| Peak memory | < 200 MB | {{mem_mb:.0f}} MB | yes |
| Latency | report | first reading 30 s; update 1 s | — |
| Fault injection (9 faults × 3 strata) | 0 confident wrong | {{faults_wrong:.0f}} (250 kbps, lighter) | **no** |
| Privacy: files written; network permission | 0; none | 0; none | yes |
| TFLite export; Android build | — | not produced; NOT COMPILED | — |

### 3.7 R9: the proposal's power table re-checked

The independent rerun reproduced every entry: H1 {{pr_h1:.2f}}, H2 {{pr_h2:.2f}}, H3 {{pr_h3:.2f}}, H4 {{pr_h4:.2f}} and H5 {{pr_h5_one:.2f}}. Three entries answer a different question from the criterion they support:

- **H4.** 0.99 is the power for "any reduction". The "≥ 30 % lower" criterion read as a point estimate succeeds {{h4rule_point30:.0%}} of the time at a true 30 %; a lower-bound version (≥ 10 %) succeeds {{h4rule_lb10_30:.0%}} of the time.
- **H5.** 0.90 is the power for one stratum; the joint criterion has {{pr_h5_joint:.2f}}.

- **Coverage.** With 36 calibration participants, a valid 90 % gate has all three strata at ≥ 88 % only {{cov36_all:.0%}} of the time.

A further design issue is recruitment. In the one Indian colorimeter study found, about {{rec_lighter:.0%}} of adults had ITA° > 28° and {{rec_darker:.0%}} had ITA° ≤ 10° [Pawar 2024]. Filling 40 lighter participants would need a median of {{rec_screen:.0f}} people screened (90th percentile {{rec_screen90:.0f}}). The proposal's risk register expects the darker stratum to fill slowly.


![](results/figures/fig_power_vs_proposal.png){width=100%}

<p class="caption">Figure 9. The proposal's power table, an independent rerun under the same assumptions, a rerun of each full success criterion, and the rehearsal (ASSUMPTION-DRIVEN and SYNTHETIC).</p>

## 4. Discussion

**What was learned.** The in-silico study did its job, which was to find problems cheaply. The software works and is fast enough. Four of the five confirmatory hypotheses, however, have success criteria that a working method could fail.

H4 asks for equal abstention from a gate whose purpose is to abstain more where the signal is weaker. It also asks for a ≥ 30 % point estimate that has even odds when the true effect is exactly 30 %, and for per-stratum coverage that a valid method misses in at least one stratum about half the time. Its calibration split cannot produce a finite interval inside nested cross-validation.

H5 tests a difference score whose reliability is roughly half that of the level its power is anchored on.

H2 can confirm widening but not its absence.

H1 rests on a between-person SD that heavy-tailed rPPG errors are unlikely to respect. A log-transformed endpoint fixes it at no cost.

These are design findings that hold under the proposal's own assumptions. They do not depend on the simulator.

**What cannot be claimed.** Nothing here shows that SAMA works on Indian skin, that compression widens the gap in people, or that camera HRV is valid. The simulator's skin-tone gradient was set to the proposal's assumption. Its codec stand-in comes from rendered tiles, not faces. Its HRV results depend on our pulse model and beat detector. The method's apparent gain over POS reflects a network trained and tested on the same simulator. Its new components (FiLM, group-DRO) showed no benefit, and at matched abstention its darker-stratum advantage disappeared.

**Threats to validity.**

- *Simulator validity:* no real dataset could be matched quantitatively.
- *ITA° estimation:* modelled only as Gaussian error. ITA° errors of 6–10° cut H1 power by 0.02–0.05.

- *Dataset bias:* none of the planned public datasets was analysed.
- *Leakage:* participant-disjoint folds were checked by an independent reviewer, who found none. An early reference leak in beat detection was fixed before analysis.

- *Multiple comparisons:* one primary test per hypothesis; secondaries are Holm-corrected or labelled exploratory.
- *Circularity:* every simulator-dependent result is labelled ASSUMPTION-DRIVEN, and a counter-world is reported.

- *Conditional power:* rehearsal "power" is conditional on one pool or one cohort, and the parametric version helps only partly.
- *Bootstrap:* the H4 resampling probability is not power.

Two fresh-context reviews found 3 critical and 1 critical issue respectively. All were fixed and filed unedited in ISSUES.md.

**What the November pilot must measure.** The rehearsal turns each fragile assumption into a number the pilot can estimate. Each comes with the value that would change the design:

- **Spread of log participant MAE.** If its SD exceeds about 0.9 (natural log), H1 at 36 per stratum falls below 0.8 power even with the log endpoint.
- **Resting ln RMSSD agreement with ECG.** An ICC below 0.75 means H5 should be stated as exploratory.
- **Share of windows the gate withholds, by skin tone.** If the darker group's useful yield is below about 0.6, SAMA's benefit will not reach users, whatever its accuracy.
- **ITA° distribution among volunteers.** If fewer than 1 in 6 exceed 28°, the strata should be redefined by local tertiles before recruitment begins.
- **Pulse amplitude after the call codec.** Record one pilot session through Jitsi at 500 kbps. If the pulse peak is lost in most windows, "measure before you compress" becomes the main recommendation of WP2 rather than a secondary analysis.

**Equity.** A reliability gate can deliver equal coverage while giving darker-skinned users far fewer readings. In the rehearsal they got a usable reading in {{yield_darker:.0%}} of windows against {{yield_lighter:.0%}} for lighter. A system that is honest about uncertainty can still be unequal in availability. The proposal should report useful yield per stratum as an outcome in its own right, and should not promise equal abstention.

**Specific changes to the proposal.** The changes are listed in GO_NO_GO.md and `proposal_inserts/proposal_change_list.md`:

1. Log-MAE primary endpoint for H1.

2. Estimation plus equivalence for H2.

3. Pooled lights for H3, with Open Camera on both phones.

4. For H4: a lower-bound primary criterion; cross-conformal calibration; coverage ≥ 85 %; useful yield in place of equal abstention.

5. For H5: resting ln RMSSD as primary, with the joint power stated.

6. A reversed recruitment risk.

7. A compression risk.

8. Reference corrections.

9. Section 5 with no unmeasured numbers.

## 5. Conclusion

**Verdict: conditional go.** The proposal is worth funding, and the software needed to run it is built and tested. As written, however, it promises results its own design is unlikely to deliver, even if SAMA works.

The conditions:

- rewrite the H1, H2, H4 and H5 criteria and their powers as listed;
- pre-register the revised criteria before any data are collected;

- correct the reference statements;
- describe section 5 as a design rehearsal until public data have been analysed.

Before the upload, the PI should download UBFC-rPPG, PURE and MMPD and run `make realdata`. That answers Q-A on real recordings, which no amount of simulation can.

What this study cannot replace is the pilot. Only the pilot can estimate the between-person spread of camera error, the reliability of camera HRV, and the ITA° distribution in Surat. The rehearsal shows that each of these decides whether SAMA-120 succeeds.

<!--END-MAIN-->

<h2 class="pb">References</h2>

1. Barber RF, Candès EJ, Ramdas A, Tibshirani RJ. Predictive inference with the jackknife+. Ann Stat 2021;49(1). doi:10.1214/20-AOS1965.

2. Bondarenko M, Menon C, Elgendi M. Demographic bias in public remote photoplethysmography datasets. npj Digit Med 2025;8. doi:10.1038/s41746-025-01973-9.

3. Chardon A, Cretois I, Hourseau C. Skin colour typology and suntanning pathways. Int J Cosmet Sci 1991. doi:10.1111/j.1467-2494.1991.tb00561.x.

4. Chari P, Kabra K, Karinca D, et al. Diverse R-PPG: camera-based heart rate estimation for diverse subject skin-tones and scenes. arXiv:2010.12769, 2020.

5. Del Bino S, Bernerd F. Variations in skin colour and the biological consequences of ultraviolet radiation exposure. Br J Dermatol 2013;169 Suppl 3:33–40.

6. de Haan G, Jeanne V. Robust pulse rate from chrominance-based rPPG. IEEE Trans Biomed Eng 2013;60(10):2878–2886.

7. Liu X, Narayanswamy G, Paruchuri A, et al. rPPG-Toolbox: Deep remote PPG toolbox. NeurIPS 2023 Datasets and Benchmarks. arXiv:2210.00716.

8. McDuff DJ, Blackford EB, Estepp JR. The impact of video compression on remote cardiac pulse measurement using imaging photoplethysmography. IEEE FG 2017.

9. McGraw KO, Wong SP. Forming inferences about some intraclass correlation coefficients. Psychol Methods 1996;1(1):30–46.

10. Nowara EM, McDuff D, Veeraraghavan A. A meta-analysis of the impact of skin tone and gender on non-contact photoplethysmography measurements. CVPR Workshops 2020:284–285.

11. Pawar et al. Facial skin color assessment and proposed skin shade guide for maxillofacial prosthesis using cluster analysis in a cohort of Indian adults. J Indian Prosthodont Soc 2024;24(4):366–373. doi:10.4103/jips.jips_194_24.

12. Perez E, Strub F, de Vries H, Dumoulin V, Courville A. FiLM: visual reasoning with a general conditioning layer. AAAI 2018.

13. Pstras L, Okupnik T, Ponikowska B, Paleczny B. Estimating heart rate variability using facial video photoplethysmography: a pilot validation study. Sci Rep 2026. doi:10.1038/s41598-026-70292-0.

14. Romano Y, Barber RF, Sabatti C, Candès EJ. With malice toward none: assessing uncertainty via equalized coverage. Harvard Data Sci Rev 2020;2(2). doi:10.1162/99608f92.03f00592.

15. Sagawa S, Koh PW, Hashimoto TB, Liang P. Distributionally robust neural networks for group shifts. ICLR 2020.

16. Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater reliability. Psychol Bull 1979;86(2):420–428.

17. Sjoding MW, Dickson RP, Iwashyna TJ, Gay SE, Valley TS. Racial bias in pulse oximetry measurement. N Engl J Med 2020;383(25):2477–2478.

18. Tang J, Chen K, Wang Y, et al. MMPD: multi-domain mobile video physiology dataset. IEEE EMBC 2023. arXiv:2302.03840.

19. Vovk V, Gammerman A, Shafer G. Algorithmic learning in a random world. Springer, 2005.

20. Wang W, den Brinker AC, Stuijk S, de Haan G. Algorithmic principles of remote PPG. IEEE Trans Biomed Eng 2017;64(7):1479–1491.

Reference details were checked at C1 (data/REFS_AUDIT.md); author lists abbreviated where the audit did not record them.

## Appendix A. Evidence ledger

| Claim in this report | Label | Strength |
|---|---|---|
| Pipeline runs end to end, real time on one core | SW, SYNTHETIC inputs | strong for software; no device |
| Five methods equal rPPG-Toolbox code | SW | strong (implementation only) |
| Published toolbox errors replicated | — | **not tested** (no data) |
| Proposal power table reproduced | ASSUMPTION-DRIVEN | strong (two independent codes) |
| H4 point criterion ≈ coin flip at true 30 % | ASSUMPTION-DRIVEN | strong |
| H5 joint power ≈ 0.74 | ASSUMPTION-DRIVEN | strong |
| Reactivity reliability ≪ level reliability | ASSUMPTION-DRIVEN + LITERATURE | moderate (depends on SD of change) |
| 75/25 split gives infinite intervals | arithmetic + SYNTHETIC | strong |
| Coverage criterion fragile | ASSUMPTION-DRIVEN | moderate |
| Heavy-tailed errors cut H1 power; log endpoint restores it | SYNTHETIC, ASSUMPTION-DRIVEN | moderate (simulator tails) |
| H2 cannot establish World B at n = 36 | SYNTHETIC | moderate |
| Real H.264 removes small modulations | SYNTHETIC rendering, real codec | moderate (tiles, not faces) |
| SAMA lowers darker MAE vs POS | SYNTHETIC | weak (same-simulator train/test; no neural baselines) |
| FiLM and group-DRO add nothing | SYNTHETIC | weak |
| Abstention unequal across strata | SYNTHETIC | moderate (mechanism general) |
| Adaptive exposure gain < 1 bpm, mostly in good light | ASSUMPTION-DRIVEN | moderate (photon budget) |
| Lighter stratum scarce in India | LITERATURE (one study, not Surat) | moderate |
| Anything about Indian skin | — | **none** |

## Appendix B. Issue register, analysis plan, parameters and reproducibility

The full issue register with final statuses is `ISSUES.md`: I-01 to I-34, plus the two fresh-context reviews filed unedited (R3-01 to R3-30 and R7-01 to R7-20). The frozen plan is `spec/SAP_SIM.md` (tag `sap-v1`), with amendment 1 in `spec/SAP_SIM_amendment_1.md`. Simulator parameters and their sources are in `sim/samasim.py` (`PARAM_SOURCES`), `configs/sim_calibrated.json` and `configs/codec_surrogate_v1.json`.

**Reproducibility.** Commit @@COMMIT@@. Environment: Linux 6.18, Python 3.13.16, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, onnxruntime 1.29.0, ffmpeg with libx264/libvpx; 2 × Xeon 2.1 GHz cores, 7 GB RAM, no GPU (ENV.md). Seeds: master 20261102; pools 1, 2 and 3; cohort 7; resampling 2026. Run times on this machine:

- virtual cohort: 63 min;
- pools: about 49 min each for A and B;

- method evaluation: 30 min;
- rehearsal analysis: 1 min.

`make all` regenerates every table and figure, and `results/manifest.json` maps every number in this report to its source file.

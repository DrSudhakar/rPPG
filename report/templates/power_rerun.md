# Power re-run (C4/C6) — for proposal section 6

Labels: AD = ASSUMPTION-DRIVEN (the proposal's own assumptions); SYN = SAMA-Sim rehearsal.
Independent code: `analysis/design_checks.py` (not `power.py`), 3,000 simulated studies per cell.

| Hypothesis | Proposal (36 / 40) | Independent rerun, same assumptions (AD) | Rerun of the full success criterion (AD) | Rehearsal on SAMA-Sim (SYN) |
|---|---|---|---|---|
| H1 | 0.87 / 0.90 | {{pr_h1:.2f}} | same test | {{c6_h1_param:.2f}} (MAE); {{c6_h1_logmae:.2f}} (log MAE) |
| H2 | 0.80 / 0.85 | {{pr_h2:.2f}} | "clear answer either way" not powered for B | A named {{c6_h2_correct_A:.0%}}; B named 0 % |
| H3 | 0.90 / 0.92 | {{pr_h3:.2f}} | same | gain {{h3_dark_pooled:.2f}} bpm (CI {{h3_dark_pooled_lo:.2f}}–{{h3_dark_pooled_hi:.2f}}) vs 1.0 assumed |
| H4 | 0.99 / 0.99 | {{pr_h4:.2f}} (any reduction) | {{h4rule_point30:.2f}} for "≥ 30 % lower" as a point estimate when the true effect is 30 % | all five criteria: {{h4_p_all:.2f}} |
| H5 | 0.90 / 0.93 | {{pr_h5_one:.2f}} (one stratum) | {{pr_h5_joint:.2f}} / {{pr_h5_joint40:.2f}} (all strata + overall, F-based CI) | reactivity ICC {{h5_icc60:.2f}}; P(pass) {{h5_ppass:.2f}} |

**Empirical spreads.** No public data could be analysed (I-22). In the simulator, per-participant
MAE had SD {{c6_sd_lighter:.1f}}–{{c6_sd_darker:.1f}} bpm against medians of {{c6_med_light_A:.2f}}–{{c6_med_dark_A:.2f}} bpm.
The errors are heavy-tailed, so the 2.0 bpm SD assumed in s.6 is optimistic for untransformed MAE.
At SD 4 bpm the rerun gives H1 power 0.43 (AD). The pilot must estimate the SD of log MAE.

**Recommended text for s.6:** "Power was estimated by simulation (3,000 studies per sample size)
under the assumptions below. Each criterion was simulated as written, including H4's relative
threshold and H5's per-stratum bound. Where a criterion as written has power below 0.8, the table says so."

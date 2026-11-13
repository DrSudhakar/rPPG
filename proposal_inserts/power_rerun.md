# Power re-run (C4/C6) — for proposal section 6

Labels: AD = ASSUMPTION-DRIVEN (the proposal's own assumptions); SYN = SAMA-Sim rehearsal.
Independent code: `analysis/design_checks.py` (not `power.py`), 3,000 simulated studies per cell.

| Hypothesis | Proposal (36 / 40) | Independent rerun, same assumptions (AD) | Rerun of the full success criterion (AD) | Rehearsal on SAMA-Sim (SYN) |
|---|---|---|---|---|
| H1 | 0.87 / 0.90 | 0.88 | same test | 0.35 (MAE); 0.78 (log MAE) |
| H2 | 0.80 / 0.85 | 0.80 | "clear answer either way" not powered for B | A named 95%; B named 0 % |
| H3 | 0.90 / 0.92 | 0.90 | same | gain 0.48 bpm (CI 0.01–0.95) vs 1.0 assumed |
| H4 | 0.99 / 0.99 | 0.99 (any reduction) | 0.51 for "≥ 30 % lower" as a point estimate when the true effect is 30 % | all five criteria: 0.00 |
| H5 | 0.90 / 0.93 | 0.91 (one stratum) | 0.74 / 0.81 (all strata + overall, F-based CI) | reactivity ICC 0.15; P(pass) 0.00 |

**Empirical spreads.** No public data could be analysed (I-22). In the simulator, per-participant
MAE had SD 3.5–3.7 bpm against medians of 0.68–1.15 bpm.
The errors are heavy-tailed, so the 2.0 bpm SD assumed in s.6 is optimistic for untransformed MAE.
At SD 4 bpm the rerun gives H1 power 0.43 (AD). The pilot must estimate the SD of log MAE.

**Recommended text for s.6:** "Power was estimated by simulation (3,000 studies per sample size)
under the assumptions below. Each criterion was simulated as written, including H4's relative
threshold and H5's per-stratum bound. Where a criterion as written has power below 0.8, the table says so."

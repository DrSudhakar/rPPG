# SAP_SIM amendment 1 (2026-11-02, before any confirmatory in-silico analysis)

Triggered by the fresh-context review (results/logs/review_C3.md). The frozen plan (`sap-v1`) is
kept; every change below is a logged deviation (DECISIONS D-16 to D-22) and the frozen version's
outcome is reported alongside wherever it differs.

| # | Frozen (sap-v1) | Amended | Why |
|---|---|---|---|
| A1 | H4: 75/25 train-proper/calibration inside each outer training set | Report the frozen split's outcome (≈7 calibration participants per stratum → infinite conformal quantile → 100 % abstention); operative analysis uses a **50/50** split (≈14 per stratum) | R3-01; I-30. A pre-specified design that cannot produce a finite interval is itself a finding |
| A2 | H2 "supports B" = MAE slope not significant + SNR-penalty slope < 0 | Frozen rule reported; added: estimate of the MAE-penalty slope with 95 % CI and an equivalence test (TOST, margin ±0.5 bpm per 75° of ITA) as the recommended rule | R3-05: absence of evidence counted as evidence |
| A3 | World B = amplitude-independent additive codec noise | World B = DC-proportional (relative) codec noise | R3-06: additive DN noise still hits smaller (darker) pulses harder, so it was not a distinct world |
| A4 | H3: three tests, no primary | Primary: per-participant MAE difference averaged over both lights, paired one-sided; per-light tests are Holm-corrected secondaries | R3-18 |
| A5 | H4 criteria: pass rule unspecified | Primary: point estimates (proposal wording); secondary: each criterion on its 95 % bootstrap bound | R3-17 |
| A6 | H1s/H2s power: resampling one pool | Also a parametric version: fit the pool's per-participant distribution (slope + stratum-specific residual quantiles) and draw fresh participants | R3-04, R3-16 |
| A7 | Network target = simulator's noise-free pulse | ECG-derived template pulse (R-peaks + population PTT 0.22 s); simulator now has participant PTT (SD 30 ms) and respiratory PTT modulation | R3-13, R3-15 |
| A8 | One RNG for every device | Physiology and motion shared by simultaneous streams; camera noise drawn per device and codec | R3-11 |
| A9 | H5 60 fps vs 30 fps contrast confounded with precision | Added `reference30hq` (same sensor and precision as the 60 fps reference, 30 fps) | R3-12 |
| A10 | Pools 90 per stratum | World A and B pools 90 per stratum; counter-world 40 per stratum | CPU budget |
| A11 | H1s endpoint per-participant MAE only | Added exploratory alternatives: log MAE, median absolute error, Spearman rank test | heavy-tailed per-participant errors seen during simulator calibration (between-participant SD ≈ 9 bpm vs 2.0 assumed) |

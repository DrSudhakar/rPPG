# Section 5 insert — "Preliminary analysis" (≤ 350 words)

> **No public dataset could be analysed before this draft (download access pending), so this section reports a design rehearsal, not data.** Replace it with real results once the downloads in DATA_INVENTORY.md are in hand. Never add numbers that have not been measured.

**Design rehearsal (in silico).** Before recording anyone, we rehearsed SAMA-120 in full on simulated data (SYNTHETIC): 120 virtual participants with the protocol's strata, lights, devices and codec levels, the complete analysis and the six-step method. The rehearsal checks the design, not the size of any effect. It led to five changes in this proposal:
1. **H1 endpoint.** Simulated camera errors were heavy-tailed. When the pool was resampled, a log-transformed participant MAE detected the simulated gradient more often than raw MAE ({{c6_h1_logmae:.2f}} vs {{c6_h1_cond:.2f}}). This is a post-hoc choice; its type-I error is still to be checked.
2. **H4 calibration.** Inside five-fold cross-validation, a 75/25 split leaves about 7 calibration participants per stratum. That is too few for a finite 90 % conformal interval, so we use cross-conformal calibration.
3. **H4 criteria.** A reliability gate abstains more where signals are weaker. We therefore report useful yield per stratum rather than requiring equal abstention.
4. **H5 endpoint.** The primary endpoint is now resting ln RMSSD. Reactivity is less reliable than the resting level.
5. **Compression.** Real H.264 encoding of rendered skin-like patches kept only {{codec_small_500:.0%}} of pulse-sized colour changes at 500 kbps, which motivates measuring on the phone before the video is compressed.

**Software readiness.** The pipeline runs end to end, from patch colours to a gated reading. It is published as code with unit tests ({{coverage_pct:.0f}} % line coverage of core modules), ONNX models (fp32/fp16/int8, parity within {{parity_int8:.3f}} bpm) and a browser demonstrator that sends no images. It runs in real time on one CPU core, but above its compute budget. Android source is written but not yet compiled.

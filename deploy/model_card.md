# Model card — SAMA step-4 network and gate, version 0.1.0-insilico

**Not a medical device. Research use only. No diagnostic use.**

- **What it is:** a 1D convolutional network of 9625 parameters with FiLM conditioning on ITA° and brightness. It refines a POS pulse from 30 weighted skin patches (30 s at 15 Hz) into a heart-rate estimate. Around it sit a ridge σ̂(x) model and a Mondrian conformal gate. Files: `deploy/model/` (numpy and JSON) and `deploy/onnx/` (fp32, fp16-weight and int8-weight formats).
- **Intended use:** research on camera heart-rate measurement and its skin-tone equity; running the SAMA-120 protocol; teaching.
- **Out of scope:**
  - any clinical, diagnostic, screening or triage use;
  - use on people without consent;
  - covert measurement;
  - inferring emotion, deception or health status;
  - children;
  - use with arrhythmia, a pacemaker or tremor.
- **Training data provenance:** SYNTHETIC only. The virtual SAMA-120 (SAMA-Sim, World A, seed 7) contains no real person. Training used half of the 108 analysable virtual participants; the other half calibrated the gate. The target is an ECG-derived pulse template.
- **Performance by stratum:** SYNTHETIC, 5-fold participant cross-validation, lossless reference video, both lights. All-window MAE (lighter / middle / darker) was 0.65 / 0.88 / 1.83 bpm, against 2.09 / 3.33 / 3.22 for POS.

  | Metric | Lighter | Middle | Darker |
  |---|---|---|---|
  | Coverage of 90 % intervals | 0.96 | 0.96 | 0.96 |
  | Abstention | 2% | 12% | 34% |
  | Useful yield | 96% | 86% | 66% |

  **These numbers say nothing about real faces.**
- **Failure modes seen:**
  1. Confident wrong readings for the lighter stratum under unseen heavy compression (phone + 250 kbps).
  2. The darker stratum receives a reading much less often.
  3. In the darker stratum the network's gain is mostly in windows that the gate withholds.
  4. FiLM and group-DRO showed no benefit in ablation.
  5. HRV from 30 fps video is unreliable for reactivity.
- **Gate calibration:** 18 calibration participants per stratum, below the 30 recommended (R10-21).
- **Compute:** steady state 0.80 s of compute per second of video on one core, above the 0.5 budget.
- **Inputs:** evaluation used simulator truth for pixel counts and ITA°.
- **Export parity:** largest heart-rate difference against numpy was 0.0006 bpm for fp16 and 0.014 bpm for int8. TFLite was not produced.
- **Fairness note:** equal coverage does not mean equal availability. Report useful yield per stratum alongside accuracy.
- **Monitoring:** see `deploy/LIMITATIONS.md` items 4–7, covering drift proxies and recalibration triggers.

# SPEC — SAMA design parameters (locked at C0)

The machine-readable source of truth is `spec/spec.yaml`; every value carries the proposal section
it came from. This page lists the parameters in plain words and the ambiguities found while
encoding them (each logged in ISSUES.md rather than silently resolved).

- **Strata** by ITA°: lighter > 28, middle 10–28, darker ≤ 10. 40 recruited and 36 analysable per
  stratum, 20 women and 20 men each, ages 18–60 (s.4 WP1, s.6).
- **Lights** ≈500 and ≈100 lux. **Streams**: lossless 60 fps machine-vision reference; 1080p30
  webcam; phone A (auto-exposure); phone B (adaptive exposure); live call capped at 500 kbps; 1 kHz
  ECG; Polar H10; skin conductance; respiration.
- **Codec ladder**: lossless, H.264 2000/1000/500/250 kbps, VP8 500/250 kbps, after resizing to
  720p30 (`transcode_ladder.sh`).
- **Blocks**: paced breathing, rest, mock teleconsultation, two neutral and two high-arousal clips,
  arithmetic, recovery, step test (durations inferred, D-13).
- **Windows**: HR 30 s, step 15 s; HRV 60 s; tolerance ±5 bpm or ±10 % (greater); band 0.7–3.0 Hz.
- **Methods**: POS primary; Green, ICA, CHROM, POS, LGI, PBV, DeepPhys, TS-CAN, PhysNet, EfficientPhys.
- **SAMA**: six steps as in s.4 WP3 (normalised Mondrian conformal, 5 bpm abstention).
- **Hypotheses** H1–H5 and E1 with the success criteria and claimed powers in spec.yaml.

Ambiguities logged: I-29 (how the 200 calibration draws are combined); "best baseline … in the same
outer folds" (D-11); H5's per-stratum criterion vs single-stratum power (I-26); H2 spread label (I-28).

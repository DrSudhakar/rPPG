# Datasheet — SAMA-Sim pools (World A, World B, counter-world)

- `pool_A_s1` and `pool_B_s2`: 90 participants per stratum each; `pool_melanin_s3`: 40 per
  stratum. POS heart-rate windows on the 720p30 reference stream, lossless and H.264 500 kbps
  (surrogate v1), both lights, 8 seated blocks. SYNTHETIC.
- World A: the codec keeps a smaller share of small (absolute) pulses (fitted to real ffmpeg on
  rendered tiles, D-23). World B: codec noise proportional to brightness (same relative cost for
  every tone). Counter-world: relative pulse amplitude falls with melanin (I-02).
- Use: power, type-I error and identifiability of H1/H2 by resampling (D-14) and by a fitted
  parametric model (amendment A6).

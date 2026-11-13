---
title: "SAMA technical report v2: from simulation to real recordings"
subtitle: "Real-data training, skin tone without a colorimeter, the live interface, and the pilot pipeline"
author: "SAMA project, Department of Artificial Intelligence, SVNIT Surat · research use only, not a medical device"
---

<div class="note">

**Provenance.** Section 6 reports the verification run on SYNTHETIC sessions (labelled V). Section 7 reports real
results: the public-data training run (T) and the Indian pilot (P). Numbers marked **[pending]** have not been
measured yet. The in-silico design rehearsal is reported separately in *SAMA in-silico report v1*; its findings on the
design are summarised in section 2 and are not repeated here.

</div>

# 1. Summary

The first SAMA report was a design rehearsal on simulated data: no real recording could be analysed. This second
report describes what was built to move SAMA onto real recordings, and how it was checked:

- **One set of knobs** (`configs/sama.yaml`) controls every option, from where skin tone comes from to how the
  reliability gate is calibrated; every run records the switches it used.
- **Training without skin-tone labels.** Public rPPG datasets have no instrument skin-tone labels, and the pilot has no
  colorimeter. SAMA can now be trained skin-tone blind (no conditioning, group-DRO over datasets, a pooled gate), or with
  a camera estimate, Monk ratings, a learned video-to-skin-tone model, or colorimeter values once they exist.
- **Skin tone without a colorimeter.** ITA° is estimated from video against a grey card held beside the cheek; Monk
  ratings by one or two raters are collected and their agreement reported; a video-to-skin-tone model is trained and
  cross-validated on whatever labels exist.
- **A live interface** shows heart rate, HRV, a dot at every beat, and warnings when movement, light, face position or
  frame rate would corrupt the readings; a lab-recorder mode writes lossless video for the pilot at the same time.
- **The pilot pipeline** turns recorded sessions into the four results promised in proposal section 5 and Figure 6, with
  the SAMA method tested on Indian faces it never saw.

# 2. What the in-silico rehearsal decided

The rehearsal (report v1, `GO_NO_GO.md`) found that several of the October proposal's criteria would likely fail even if
the method worked, and that a plain conformal split inside five-fold cross-validation leaves too few calibration
participants. These findings fixed four design choices carried into this work: cross-conformal calibration of the gate
(section 4.6), all-window scoring of accuracy (to avoid selective-prediction bias), revised confirmatory criteria
(proposal sections 3 and 6), and the warning that FiLM conditioning and group-DRO are untested assumptions to be
decided by ablation. None of these is evidence about people.

# 3. Architecture

![Figure 1. Data flow from recording to the proposal.](docs/pipeline.png){width=100%}

| Command | What it does |
|---|---|
| `python -m sama record` | Lab recorder: lossless webcam video via ffmpeg, live analysis, block markers, box drawing |
| `python -m sama serve` | Live interface with the browser camera; only colour averages leave the page |
| `python -m sama prepare` | Per session: signals, compression ladder, camera ITA°, synchronisation, quality control |
| `python -m sama train` | SAMA on UBFC-rPPG / PURE (and optionally the pilot), participant-level cross-validation |
| `python -m sama train-ita` | Video-to-skin-tone model on Monk or colorimeter labels |
| `python -m sama pilot` | Section-5 results, Figure 6, pilot report |
| `python proposal/build_proposal.py` | Proposal with the measured numbers filled in |

**Knobs.** Every option is a key in `configs/sama.yaml`, overridable with `--set key=value`. Unknown keys and values are
rejected; when one switch makes another impossible (for example `ita.source=none` with a per-stratum gate), the
dependent switch is changed, a message is printed, and the change is written at the top of `config_used.yaml`. The full
list is in `docs/KNOBS.md`.

# 4. Methods

## 4.1 From video to signals
Video is decoded only through the ffmpeg command-line tools. On the PI's machine OpenCV's bundled FFmpeg corrupted
memory while decoding UBFC-rPPG files; the command-line route avoids that library entirely. Each frame is reduced, after
area-averaging to 320 pixels wide, to 30 skin-patch colour means inside the face box (forehead 2 × 6, cheeks 3 × 3 each),
using only pixels that pass a wide YCbCr skin rule and are neither clipped nor black; plus the skin fraction of the box,
its brightness, and a **motion index** (mean absolute luma change between consecutive frames on a 48 × 48 thumbnail).
No frame is stored. The face box comes from OpenCV's Haar detector when it is installed, else from the box the operator
drew in the recorder, else from a fixed guide box (`face.tracker`). Every compression level is measured on the boxes
tracked on the lossless video.

## 4.2 Skin tone
**Camera ITA°.** For each of up to 150 frames from the 30-second card block, skin pixels from the cheek patches (darkest
and brightest 5 % by luma dropped symmetrically, so the trim does not push ITA° lighter; `ita.camera.trim_pct`) are converted to linear RGB, divided channel by channel by the card's linear RGB and
multiplied by the card's reflectance as entered in the recorder for that session (von Kries white balance; a session
without a stored reflectance gets no estimate, flag `CARD_REFLECTANCE_MISSING`), then converted to CIELAB (D65) and ITA°. The estimate is
the median over frames; the inter-quartile range is its stability. Flags: `WHITE_CLIPPED`, `WHITE_TOO_DARK`,
`ITA_UNSTABLE`, `L_OUT_OF_SKIN_RANGE`, `ITA_OUT_OF_HUMAN_RANGE`, `NO_WHITE_REF`. A grey card (reflectance 0.18) is the
default because a white card saturates when auto-exposure brightens a darker face; the recorder shows the card's clipping
live. With a 24-patch chart, a 3 × 3 colour-correction matrix is fitted instead. Both `ita_capture`
blocks (two lights) are used and their estimates averaged. A participant whose estimate carries a blocking flag gets no
ITA° and is placed in a stratum by Monk rating; if fewer than two analysed participants fall in the lighter or darker ITA°
group, every participant is stratified by Monk rating (both rules pre-specified and reported in `strata_note`).

**Monk ratings.** One or two raters (`raters.n`). With two, the report gives quadratic-weighted kappa, ICC(2,1) with the
McGraw–Wong interval, and the share of exact and within-one-shade matches; with one, agreement is reported as not
estimable. Strata from Monk (1–4, 5–7, 8–10) are a stated convention, not a validated mapping.

**Video-to-skin-tone model.** Ridge regression on per-participant colour features (raw and white-balanced cheek and
forehead colour, chromaticities, CIELAB, luma percentiles, the card's own colour), trained on Monk ratings now and on
colorimeter ITA° once it exists, evaluated by participant-level cross-validation, and saved with its CV metrics.

## 4.3 Reference and synchronisation
The reference CSV's columns are mapped by configuration. With a pleth waveform, beats are systolic peaks with parabolic
timing; with only a displayed rate, window means are used and HRV is unavailable. Camera and reference are aligned from
the PC clock and refined by cross-correlating the camera POS pulse with the oximeter wave in each 30-second window: a
wide search (±3 s) gives a median, a narrow search (±0.3 s) around it gives per-window lags. Both signals are oriented by
the skew of their derivative so that a polarity difference cannot shift the alignment by half a beat. The robust spread
(1.4826 × MAD) of the per-window lags is the synchronisation error.

## 4.4 Windows and methods
30-second windows with a 15-second step, never crossing a protocol block; the card and motion-check blocks are excluded
from accuracy. Per window: reference heart rate (60 / mean clean inter-beat interval, with a coverage rule), and camera
heart rate from POS, CHROM, GREEN, ICA, LGI, PBV (spectral peak with parabolic refinement) and SAMA. HRV in 60-second
windows of the other blocks, from camera beats (POS and the noise-weighted SAMA pulse) and from the oximeter,
excluding windows with motion above the warning threshold.

## 4.5 Training SAMA on real recordings
Recordings are resampled to 30 frames a second and cut into 30-second windows every 5 seconds for training. The network
target is the contact-PPG waveform (UBFC-rPPG, PURE, or the synchronised oximeter wave); with an HR-only reference the
spectral loss alone is used. Participant-level K-fold cross-validation trains the network and the difficulty model σ̂ on
K − 1 folds and predicts the held-out fold. With no skin-tone source, FiLM conditioning is off and group-DRO runs over
datasets (worst-dataset training).

## 4.6 Reliability gate
Normalised scores |ĥ − h| / σ̂ of every participant come from a model that never saw that participant (cross-conformal,
CV+). The gate quantile is the ⌈0.9(n+1)⌉-th smallest score with one window per participant, averaged over 200 draws;
pooled by default, per stratum when strata exist for all training participants and are large enough. Coverage is checked
fold by fold with quantiles from the other folds only. A reading is shown when the interval half-width is at most
5 bpm and the motion index is below the abstention threshold. Cross-conformal intervals guarantee coverage of at least
1 − 2α (80 % for nominal 90 %) in the worst case; in practice they are close to nominal, which is why coverage is
measured rather than assumed. Equal coverage in every stratum is guaranteed only by the per-stratum (Mondrian) gate,
which needs about 30 calibration participants per stratum; with fewer, coverage is reported per stratum, not promised.

## 4.7 Live analysis
Every second the analyser computes from the last 20 seconds the heart rate (POS, CHROM, or SAMA when 30 seconds are
buffered), the band-passed pulse of the last 10 seconds with its beat times, and quality levels; and from the last
60 seconds the HRV, only if no movement exceeded 1.5 × the warning threshold. Abstention codes: WARMING_UP, FACE_LOST,
FRAME_RATE, LOW_LIGHT, MOTION, LOW_SIGNAL, IMPLAUSIBLE, INTERVAL_TOO_WIDE. A withheld reading never carries a number.

## 4.8 Pilot statistics
Section-5 results are computed exactly as the proposal states them: (1) synchronisation spread, session minutes, usable
sessions by automatic QC; (2) the change in per-participant POS MAE per 10° fall in ITA° with a participant-bootstrap
CI (2,000 replicates); (3) the 500 kbps H.264 penalty per stratum with bootstrap CIs; (4) Bland–Altman bias and limits,
widened for repeated windows per participant by a one-way random-effects decomposition. A window in which a method gave
no estimate is scored as a failure (against the participant's median estimate) rather than dropped, and failure rates are
reported. The power re-check replaces the assumed spread with the pilot's (for H1, the residual SD of log MAE on ITA°,
using the same simulation model as the proposal's section 6, `sama/power_check.py`), and also uses the upper 95 % bound of
that spread (chi-square interval), because SDs from small samples are imprecise; power is reported at 36, 45 and 55 per
stratum. SAMA-150 plans 50 recruited and 45 analysable per stratum.

**Leakage guard.** `python -m sama pilot` stops if the trained bundle lists PILOT among its training datasets
(`pilot.role=test_only`, the default). With `pilot.role=calibrate`, only the gate's quantile is recalibrated on the
pilot, leave-one-participant-out, and the report says so.

# 5. Software verification

The test suite (`make test`) covers: configuration validation and reconciliation; camera ITA° recovery under a coloured
light with a grey card (within 4°), bias without a reference, and rejection of a clipped card; Monk consensus and
agreement with one and two raters; oximeter beat detection and window heart rate; synchronisation recovering a clock
offset with either waveform polarity; protocol blocks from events; live heart rate within 4 bpm of truth, beats, and
motion abstention; the skin-tone model's save/load and cross-validation; clustered Bland–Altman; the power helpers; the
web interface end to end (session, frames, knobs, saved readings, no pixels sent); and the lab recorder writing lossless
video, session metadata and events from a test video.

# 6. End-to-end rehearsal on synthetic sessions (SYNTHETIC)

{{V:n_public:d}} synthetic recordings in the UBFC-rPPG layout and {{V:n_pilot:d}} synthetic pilot sessions (lossless
FFV1 video painted from SAMA-Sim patch colours, a grey card, an oximeter file with a +0.35 s clock offset, events, and
two Monk raters) were run through `train`, `prepare`, `pilot` and `train-ita` with the default knobs. This checks that
the chain runs and that its numbers can be traced; it says nothing about people.

| Check | Result |
|---|---|
| Camera ITA° vs the simulator's true ITA° (grey card), mean absolute error | {{V:ita_abs_error_mean:.2f}}° (max {{V:ita_abs_error_max:.2f}}°) |
| Synchronisation: recovered lag vs the true finger-minus-face delay, mean absolute error | {{V:sync_abs_error_ms:.0f}} ms; median per-session spread {{V:sync_spread_ms:.0f}} ms |
| Sessions passing automatic QC | {{V:n_usable:d}} of {{V:n_pilot:d}} |
| Public-layout training, cross-validated MAE: SAMA vs POS | {{V:train_mae_sama:.2f}} vs {{V:train_mae_pos:.2f}} bpm |
| Pilot, POS MAE (lossless) | {{V:pilot_mae_pos:.2f}} bpm |
| Pilot, SAMA MAE (all windows) / abstention / coverage of shown intervals | {{V:pilot_mae_sama:.2f}} bpm / {{V:pilot_abstention:pct}} / {{V:pilot_coverage:pct}} |
| Video-to-Monk model, cross-validated MAE | {{V:ita_model_mae:.2f}} shades |

# 7. Real data

## 7.1 Public datasets (REAL-PUBLIC)
Trained on {{T:n_recordings:d}} recordings from {{T:n_participants:d}} participants ({{T:trained_on:list}}), skin-tone
blind. Cross-validated heart-rate MAE {{T:cv.all.mae_sama:.2f}} bpm for SAMA against {{T:cv.all.mae_pos:.2f}} for POS on
the same windows; within ±5 bpm {{T:cv.all.within5_sama:pct}} vs {{T:cv.all.within5_pos:pct}}; the gate withheld
{{T:cv.all.abstention:pct}} of windows, and its shown intervals covered the reference in {{T:cv.all.coverage_shown:pct}}.

## 7.2 Indian pilot (REAL-PILOT)
{{P:participants.n:d}} participants analysed of {{P:feasibility.n_recorded:d}} recorded. Synchronisation spread
{{P:feasibility.sync_spread_ms_median:.0f}} ms. POS MAE change per 10° fall in ITA°: {{P:skin_tone.POS.rise_per_10deg_fall:+.2f}}
bpm (95 % CI {{P:skin_tone.POS.lo95:+.2f}} to {{P:skin_tone.POS.hi95:+.2f}}). 500 kbps penalty: darker
{{P:compression.POS.h264.by_stratum.darker.penalty:+.2f}}, lighter {{P:compression.POS.h264.by_stratum.lighter.penalty:+.2f}} bpm.
Bland–Altman bias {{P:agreement.POS.bias:+.2f}} bpm (limits {{P:agreement.POS.loa_lo:+.1f}} to {{P:agreement.POS.loa_hi:+.1f}}).
SAMA, untouched by the pilot: MAE {{P:sama.lossless.all.mae_sama_all:.2f}} vs {{P:sama.lossless.all.mae_primary_all:.2f}} bpm
for POS; abstention {{P:sama.lossless.all.abstention:pct}}; coverage of shown intervals {{P:sama.lossless.all.coverage_shown:pct}}.
Camera ITA° vs Monk: Spearman ρ = {{P:ita_vs_monk.spearman_rho:.2f}}. The full tables are in `results/pilot/pilot_report.md`.

# 8. Limitations and the claims ceiling

- The pilot's skin tone is a camera estimate against a grey card, cross-checked against Monk ratings; it is not the
  colorimeter ITA° the main study uses. Pilot results stratified by skin tone are descriptive.
- The pilot reference is a finger PPG; its beat-to-beat variability is pulse-rate variability, not ECG HRV.
- SAMA as tested on the pilot was trained on public datasets that are mostly lighter-skinned and have no skin-tone
  labels; a failure on darker faces would be a finding about training data as much as about the method.
- With 10–20 participants, every interval is wide. No result here tests H1–H5.
- The motion index thresholds are camera-specific and must be tuned from the motion-check block.
- The browser mode uses a fixed face box; the lab recorder uses the operator's box or the Haar detector. A face-mesh
  tracker is future work.
- Nothing here is a medical device or a diagnosis.

# 9. Reproducibility

Every command writes `config_used.yaml`; the trained bundle stores the list of training recordings and CV metrics; the
pilot summary stores the provenance label; the proposal build refuses SYNTHETIC inputs unless asked for a watermarked
test build. Seeds: `project.seed`. Tests: `make test`. Synthetic rehearsal: `make pilot-synthetic`.

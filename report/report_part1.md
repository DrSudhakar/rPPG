# SAMA in silico: a simulation-based design rehearsal and deployable research software for skin-tone-equitable camera heart rate

<p class="meta">Authors: [PI to complete]. Code and drafting were AI-assisted (Claude); every number was produced by repository code from logged outputs. Commit @@COMMIT@@. Date: 2 November 2026.</p>

<div class="abstract">

**Background.** Camera heart rate (remote photoplethysmography, rPPG) is less accurate for darker skin, and teleconsultation video is compressed. SAMA proposes a 120-person Indian study (SAMA-120) and a method that adapts exposure, weights skin patches by measured noise, trains for the worst-served group and withholds unreliable readings. **Objective.** Before funding, test what a cheap in-silico study can and cannot tell: whether the pipeline runs end to end (Q-A), whether the proposal's design numbers hold (Q-B), and which claims need rewording (Q-C). **Methods.** Three evidence tiers were planned: REAL-PUBLIC recordings, SYNTHETIC data from a physics-based simulator (SAMA-Sim), and SEMI-SYNTHETIC degradations of real video. No public dataset could be downloaded, so only the SYNTHETIC tier and LITERATURE inputs were used. No new human data were collected, and no result here is evidence about Indian skin. We rehearsed SAMA-120 in full (virtual participants, nested cross-validation, conformal gate), re-ran the power analysis independently, validated the codec stand-in against real ffmpeg, and built a tested web demonstrator, ONNX exports and Android source. **Results.** The pipeline ran end to end, at about half real time on one CPU core. Five baseline methods reproduced rPPG-Toolbox outputs exactly. Re-running the proposal's power analysis reproduced its table, but its own success criteria need more than the table assumes. H4's point-estimate criterion is met only {{h4rule_point30:.0%}} of the time at a true 30 % effect. The joint H5 criterion has power {{pr_h5_joint:.2f}}, not 0.90. Inside five-fold cross-validation, the planned calibration split leaves too few participants for a finite conformal interval. In the rehearsal, SAMA lowered darker-stratum error by {{h4_dark_red:.0%}} (95 % CI {{h4_dark_red_lo:.0%}}–{{h4_dark_red_hi:.0%}}), but it abstained {{abst_darker:.0%}} vs {{abst_lighter:.0%}} of the time (darker vs lighter), so all five H4 criteria were met in {{h4_p_all:.0%}} of resamples. Heavy-tailed per-participant errors cut H1 power to {{c6_h1_param:.2f}}; a log endpoint restored {{c6_h1_logmae:.2f}}. Real H.264 kept {{codec_small_500:.0%}} of pulse-sized modulations at 500 kbps. **Conclusions.** Conditional go. The pipeline is ready, but H1, H2, H4 and H5 need rewording and Q-A remains open until public data are analysed.

</div>

## 1. Introduction

Optical sensors read light that has passed through skin, and melanin absorbs part of it. Pulse oximeters missed occult hypoxaemia almost three times as often in Black as in White patients [Sjoding 2020]. Camera rPPG shows the same pattern: errors rise for darker skin [Nowara 2020; Chari 2020], and an audit of 100 rPPG studies found darker skin tones under-represented [Bondarenko 2025]. Chari et al. traced much of the gap to a pulse signal that is too small relative to camera noise rather than to a different pulse; in their data, mean absolute error (MAE) of a facial-aggregation baseline rose from 3.94 to 6.20 beats per minute (bpm) between lighter and darker Fitzpatrick groups, with a PPG reference [Chari 2020]. Video compression also removes small colour changes [McDuff 2017]. India's teleconsultation services add both problems at scale: a wide range of skin tones and low-bandwidth calls.

SAMA (Sanskrit: *equal*) asks four things of a camera heart-rate system: measure the gap with instrument-measured skin tone (Individual Typology Angle, ITA° [Chardon 1991; Del Bino 2013]) and ECG; explain it with an error budget over skin, light, device and codec; fix it at capture and in the algorithm; and guarantee equal reliability, so that every skin-tone group gets intervals with the same coverage and the system says "no reliable reading" rather than a wrong number. The proposal plans SAMA-120 (40 recruited, 36 analysable per ITA° stratum) and five pre-registered hypotheses (Table 1).

A study like SAMA-120 costs a year of recording. Several of its design numbers can be checked without any new data: the power of each test under its own assumptions, whether each success criterion can be met by a method that works, whether the analysis can tell competing explanations apart, and whether the software runs in real time. Other questions cannot be answered in silico at all, above all how large the skin-tone gap is in Indian participants and whether compression widens it in people. This report separates the two. It answers three questions: **Q-A (feasibility)**, does the pipeline from video to a calibrated, gated heart rate and HRV run end to end and replicate known results on public data; **Q-B (design)**, do the power table, the H4 criteria, the 5 bpm abstention rule and 36 analysable participants per stratum hold; and **Q-C (claims)**, which statements can be supported by preliminary evidence before the 17 November upload.

The contribution is a design audit with code, not a demonstration that SAMA works. Every number carries one provenance label: REAL-PUBLIC, SYNTHETIC, SEMI-SYNTHETIC or LITERATURE; results that follow from a premise written into the simulator are also marked ASSUMPTION-DRIVEN.

## 2. Methodology

### 2.1 Data tiers and what was obtained

The plan had three tiers (Table 2). Every dataset host (Google Drive, TU Ilmenau, IEEE DataPort, MAHNOB, Zenodo, Figshare, Hugging Face, PhysioNet, Kaggle) refused connections from the build environment, and PyPI was also blocked, so PyTorch, statsmodels and TensorFlow could not be installed (ENV.md). GitHub was reachable, which allowed an implementation check against rPPG-Toolbox [Liu 2023]. The REAL-PUBLIC and SEMI-SYNTHETIC tiers therefore produced nothing; the C1 gate failed and the project was downgraded to a design rehearsal, as the protocol prescribes. The download list for the PI is in `data/DATA_INVENTORY.md`.

Table 2. Data tiers: planned and obtained.

| Tier | Planned sources | Obtained | Used for |
|---|---|---|---|
| REAL-PUBLIC | UBFC-rPPG, PURE, MMPD, UBFC-Phys, MAHNOB-HCI, OBF, others | **none** (hosts refused; licence forms needed) | — |
| SEMI-SYNTHETIC | codec ladder, dimming, darkening of real uncompressed video | **none** (no source video) | — |
| SYNTHETIC | SAMA-Sim pools (270 + 270 + 120), virtual SAMA-120 (120), rendered codec tiles | yes | H1s–H5s, E1, method, deployment tests |
| LITERATURE | 44 proposal references audited; 5 more for this report | 31 verified, 8 corrected, 5 claim-mismatch | anchors, recruitment |


### 2.2 SAMA-Sim

SAMA-Sim simulates 30 skin patches per face rather than images. Physiology: beat times with respiratory sinus arrhythmia, a 0.1 Hz component and white variability scaled to a target RMSSD, block-specific heart rate and HRV shifts (rest, paced breathing, mock teleconsultation, film clips, mental arithmetic, recovery, post-exercise), occasional premature beats and ECG artefact bursts; pulse arrival 0.22 s after the R peak with a participant-level spread (SD 30 ms) and respiratory modulation. Skin optics: ITA° sets CIELAB L* at fixed b*, converted to linear reflectance; the pulse's *relative* amplitude follows a fixed blood-volume signature and is independent of melanin (the Chari premise), so its *absolute* amplitude scales with reflected light. A counter-world lowers the relative amplitude with melanin (I-02). Camera: illuminance, exposure time capped at 1/30 s, sensor gain, photon shot noise, read and ADC noise, a tone curve and the exact expectation of 8-bit rounding of noisy pixels. Two exposure controllers: centre-weighted auto-exposure to mid-grey, and SAMA's step 1 (face to a target share of full scale, exposure first, then gain, under 1 % of skin pixels clipped). Motion and region-of-interest jitter are mostly colour-neutral.

Two free signal parameters (relative pulse amplitude, effective pixels per patch) were calibrated so that POS MAE on lossless reference video follows the proposal's own H1 assumption: 2.5 bpm at ITA 50° rising to 4.5 bpm at ITA −25° (achieved 2.79 and 4.38 bpm at ITA 45° and −20° on four 90 s blocks per light, 16 participants per point). **The simulator's skin-tone gradient is therefore assumed, not discovered**; H1s tests the design, not the effect.

The codec is an analytic stand-in fitted to real ffmpeg. We rendered 30 tiles carrying pulses of known amplitude into 720p30 video, encoded it at every ladder level, decoded it, and measured the share of each pulse that survived (Figure 1). Figure 1 shows the result. The first stand-in kept 90 % or more of the pulse at every bitrate; real H.264 kept 7–10 % of modulations under 0.3 grey levels even at 1–2 Mbps and 74–86 % of large ones. The stand-in was re-fitted (amplitude kept = floor + (top − floor)·A²/(A² + τ²), plus noise) and is "World A". "World B" is the alternative explanation in the proposal: codec noise proportional to brightness, with no amplitude-dependent loss, so every skin tone pays the same relative cost.

### 2.3 The virtual SAMA-120 and the rehearsal

The rehearsal copies the proposal's design: three ITA° strata (lighter > 28°, middle 10–28°, darker ≤ 10°), 40 recruited and 36 analysable per stratum (10 % random session loss), 20 women and 20 men, eight seated blocks under ≈500 and ≈100 lux (47 recorded minutes), a lossless reference camera (60 fps; 720p30 for the ladder), a webcam, phone A with auto-exposure, phone B with adaptive exposure, and H.264 at 500 and 250 kbps. Heart rate is estimated in 30 s windows every 15 s and HRV in 60 s windows. Six unsupervised baselines (Green, ICA, CHROM, POS, LGI, PBV) run on every window. GREEN, POS, CHROM, LGI and PBV reproduce the rPPG-Toolbox code on identical inputs (Pearson r > 1 − 10⁻⁶, Table 3); ICA uses FastICA rather than JADE. The four neural baselines could not run without PyTorch (I-24).

For H1 and H2, pools of 270 virtual participants in World A and World B, and 120 in the counter-world, were simulated once; 1,000 SAMA-120 studies were drawn from each pool without replacement within a study, with a permutation null for type-I error. Because resampling one pool gives power conditional on that pool, a parametric version drew fresh participants from the pool's fitted trend and stratum-specific residuals.

### 2.4 The SAMA method in six steps

(1) Adaptive exposure (simulated controller; Android code). (2) Linearisation with a 24-patch chart and a photon-transfer noise model fitted to static chart frames. (3) Each patch weighted by pulse-band power over expected noise variance. (4) POS on the weighted relative RGB gives a candidate; a 1D convolutional network (9,625 parameters; three dilated layers with feature-wise linear modulation by ITA° and brightness [Perez 2018]) refines it, trained with negative Pearson plus a spectral heart-rate loss and group distributionally robust optimisation over six groups, three strata by two lights [Sagawa 2020]. Because SAMA-120 records ECG but no contact PPG, the training target is a pulse template placed on the ECG R peaks. (5) Beats by parabolic interpolation to 1 ms after a 0.7–3.0 Hz band-pass; the peak-spacing hint comes from the camera signal's autocorrelation, never from the reference. (6) A difficulty score σ̂(x) (ridge regression of log absolute error on 12 quality features) and Mondrian split conformal prediction with normalised scores |ĥ − h|/σ̂ per stratum [Vovk 2005; Romano 2020], calibrated on one window per participant drawn 200 times; a reading is shown only if the 90 % half-width is at most 5 bpm.

H4 uses nested participant-level cross-validation: five outer folds stratified by stratum. The frozen plan split each outer training set 75/25 into training and calibration participants; amendment A1 changed this to 50/50 before any analysis (Section 3.5). The best baseline is chosen on the outer training participants. Primary MAE scores every window, including abstained ones, with the point estimate. Baselines get the same folds and the same gate (POS + σ̂ + conformal).

### 2.5 Frozen analysis plan, review and deployment build

The analysis plan was frozen and tagged before any confirmatory analysis (`spec/SAP_SIM.md`, tag `sap-v1`). A reviewer with fresh context, given only the specification, code and outputs, found three critical flaws: the calibration-set size, reference leakage in beat detection, and noisy simulator calibration. One amendment was adopted before analysis (`SAP_SIM_amendment_1.md`), and the frozen versions' outcomes are reported where they differ. Design checks re-implemented the proposal's power analysis independently of its script and simulated each success criterion as written. The deployment build comprises:

- an installable package;
- a frame-by-frame streaming estimator with coded abstention reasons;

- a web application in which the browser sends only patch colour means;
- ONNX exports with weights in fp32, fp16 and int8, written without the onnx package and validated in onnxruntime;

- Kotlin source for an Android demonstrator, which could not be compiled here (no Android SDK).

Acceptance thresholds were fixed before measurement (DECISIONS D-24).

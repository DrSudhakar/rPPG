# Knobs: every switch in the SAMA pipeline

All knobs live in `configs/sama.yaml`. Change them there, in your own YAML passed with
`--config my.yaml` (only the keys you change), or for one run with `--set key=value`:

```
python -m sama pilot --set ita.source=monk --set raters.n=2
python -m sama train --set training.group_dro=false --set gate.type=pooled
python -m sama config --set ita.source=none          # print the effective configuration and check it
```

Every run writes `config_used.yaml` next to its outputs. Unknown keys and values are rejected.
When one switch makes another impossible, the pipeline changes the dependent switch, prints a
`[config]` line and records it at the top of `config_used.yaml` (rules in `sama/config.py: reconcile`).

## Ready-made combinations

| Goal | Overrides |
|---|---|
| **Skin-tone blind training and testing** (no ITA anywhere) | `ita.source=none` → FiLM off, strata off, pooled gate, group-DRO over datasets |
| Train on public data that has no ITA, test on the pilot with camera ITA for reporting | defaults (`training.public_ita=none`, `ita.source=camera`) |
| Strata from Monk ratings instead of ITA | `strata.by=monk` (or `ita.source=monk`) |
| Two Monk raters, use their mean | `raters.n=2 raters.consensus=mean` |
| Two raters, but strata from rater 1 only | `raters.n=2 raters.consensus=rater1` |
| Colorimeter values once the instrument is bought | `ita.source=measured` (file `pilot_data/ita_measured.csv`) |
| Use the trained video→ITA° model | `ita.source=predicted` (after `python -m sama train-ita --target ita`) |
| Oximeter that logs only pulse rate | `reference.type=oximeter_hr` → wall-clock sync, no reference HRV |
| Ablation: no network (steps 3+POS only) | `training.use_network=false` |
| Ablation: no noise weighting | `training.noise_weighting=false` |
| Ablation: no group-DRO | `training.group_dro=false` |
| Per-stratum (Mondrian) gate | `gate.type=mondrian` (needs strata for every training participant) |
| No gate (always show a number) | `gate.type=off` in analysis; `live.gate=false` in the interface |
| Recalibrate SAMA's gate on the pilot (leave-one-participant-out) | `pilot.role=calibrate` |
| Faster analysis on a slow PC | `signal.analysis_width=240 pilot.ladder.vp8_kbps=[]` |

## Reference

### project
| Key | Values | Meaning |
|---|---|---|
| `provenance_label` | REAL-PILOT, REAL-PUBLIC, SYNTHETIC | Stamped on every output. Use SYNTHETIC for `synth-pilot` sessions. |
| `seed` | integer | All random choices (folds, bootstrap, initial weights). |

### signal
| Key | Default | Meaning |
|---|---|---|
| `window_s`, `step_s` | 30, 15 | Heart-rate windows (proposal). |
| `band_hz` | [0.7, 3.0] | Pulse band (42–180 bpm). |
| `methods` | POS CHROM GREEN ICA LGI PBV SAMA | Methods computed by `python -m sama pilot`. Drop SAMA to run baselines only. |
| `primary_method` | POS | The baseline quoted in section 5 (proposal: POS). |
| `windows_within_blocks` | true | Windows never straddle two protocol blocks. |
| `analysis_width` | 320 | Frames are area-averaged to this width before measuring (patch means do not change; decoding is faster). 0 = native. |

### face
| Key | Default | Meaning |
|---|---|---|
| `tracker` | auto | `auto`: Haar detector if OpenCV's model file is installed, else the box drawn in the recorder, else the fixed box. `manual_box`: always the drawn box. `fixed_box`: always `fixed_box`. `haar`: require the detector. |
| `fixed_box` | [0.30, 0.15, 0.40, 0.60] | x, y, w, h as fractions of the frame. |
| `min_skin_fraction` | 0.35 | Below this share of skin-coloured pixels the face counts as lost. |

### ita (skin tone)
| Key | Default | Meaning |
|---|---|---|
| `source` | camera | measured / camera / predicted / monk / none (see table above). |
| `conditioning` | true | FiLM conditioning of the network on ITA°. Forced off when the source gives no ITA°, and when any training recording lacks ITA°. |
| `camera.white_reference` | card | card (grey or white card in view), chart (24-patch chart), none (flagged unreliable). |
| `camera.white_reflectance` | 0.18 | Reflectance of the card (18 % grey card). The recorder stores the value entered for each session in `session.json`; card mode refuses to estimate ITA° for a session that lacks it (flag CARD_REFLECTANCE_MISSING). |
| `camera.regions` | [cheeks] | cheeks and/or forehead. |
| `camera.frames`, `camera.segment` | 150, ita_capture | Frames sampled from the named block. |
| `camera.max_iqr_deg` | 6 | Frame-to-frame IQR above this flags ITA_UNSTABLE. |
| `camera.trim_pct` | 5 | Darkest and brightest 5 % of skin pixels dropped symmetrically. |

### strata, raters
| Key | Default | Meaning |
|---|---|---|
| `strata.by` | ita | ita / monk / none. Used for reporting, group-DRO (`dro_groups=stratum`) and the Mondrian gate. |
| `strata.ita_cuts` | [10, 28] | darker ≤ 10 < middle ≤ 28 < lighter (Del Bino & Bernerd 2013). |
| `strata.monk_cuts` | [4, 7] | Monk 1–4 lighter, 5–7 middle, 8–10 darker. A convention, not a validated mapping. |
| `raters.n` | 1 | 1 or 2 raters. With 1, agreement is reported as not estimable. |
| `raters.consensus` | mean | mean of two raters, or rater1. |

### reference, sync
| Key | Default | Meaning |
|---|---|---|
| `reference.type` | oximeter_pleth | oximeter_pleth / oximeter_hr / polar_rr / ecg. |
| `reference.columns` | time_s, pleth, hr_bpm, spo2, rr_ms, ecg | Column names in the export. |
| `reference.time_format` | unix_s | unix_s / unix_ms / relative_s. |
| `sync.method` | xcorr | wallclock, or wall clock refined by cross-correlation with the reference wave. |
| `sync.max_lag_s`, `phone_max_lag_s` | 3, 30 | Search ranges. |

### quality, hrv
| Key | Default | Meaning |
|---|---|---|
| `quality.motion_warn` | 3.0 | Amber warning; HRV paused. Units: mean absolute luma change per frame in the face box (8-bit). Tune from the `motion_check` block: `qc.json` suggests values. |
| `quality.motion_abstain` | 6.0 | Red warning; no reading shown; windows above it are withheld by SAMA. |
| `quality.low_light_dn`, `min_fps`, `max_gap_s`, `min_snr_db` | 25, 12, 0.5, −3 | Abstention rules. |
| `quality.qc_min_reference_valid`, `qc_min_face_frac` | 0.8, 0.9 | Session-level QC. |
| `hrv.window_s`, `min_beats`, `max_edited_frac` | 60, 40, 0.05 | HRV windows and artefact rules (proposal). |

### training
| Key | Default | Meaning |
|---|---|---|
| `bundle` | models/sama_real | Where the trained model is written and read. |
| `datasets` | UBFC-rPPG, PURE | Any subset of UBFC-rPPG, PURE, VIPL-HR, iBVP, SUMS, LADH, M3PD, RLAP, MMPD, PILOT. Each public dataset lives in its own folder named after it under `paths.public_root`; `python -m sama datasets` shows what was found. Add PILOT to train on the pilot (then it is no longer a test set). |
| `public_ita` | none | none: skin-tone blind on public data; camera_uncalibrated: use the unreliable estimate anyway. |
| `folds`, `epochs`, `lr`, `channels`, `window_step_s` | 5, 25, 0.002, 24, 5 | Cross-validation and network. |
| `group_dro`, `dro_groups` | true, dataset | Worst-group training over datasets, strata, or both. |
| `loss` | pearson_spectral | spectral if the reference has no waveform. |
| `noise_weighting`, `use_network`, `chart_calibration` | true, true, false | Method steps 3, 4 and 2. |
| `max_recordings` (per dataset) | 0 | Limit for a quick smoke run. |

### gate
| Key | Default | Meaning |
|---|---|---|
| `type` | pooled | pooled (one interval rule for all), mondrian (per stratum), off. |
| `alpha`, `halfwidth_max_bpm` | 0.10, 5 | 90 % intervals; no reading if the half-width exceeds 5 bpm. |
| `calibration` | cross | Out-of-fold scores of every training participant (CV+). |
| `min_calibration_participants`, `draws` | 9, 200 | Below 9 participants a 90 % interval is infinite. One window per participant per draw. |

### pilot
| Key | Default | Meaning |
|---|---|---|
| `role` | test_only | test_only: the pilot never trains or calibrates anything. calibrate: SAMA's gate is recalibrated on the pilot, leave-one-participant-out. Training on the pilot (`training.datasets` with PILOT) while analysing it stops the run. |
| `blocks`, `lights` | see file | Buttons shown in the recorder. |
| `exclude_blocks_from_accuracy` | ita_capture, motion_check | Blocks left out of HR and HRV accuracy. |
| `ladder.*` | H.264 2000/1000/500/250, VP8 500/250 | Compression ladder (proposal). |
| `figure_codec`, `compression_kbps` | h264, 500 | What Figure 6B and section 5 result 3 show. |
| `bootstrap` | 2000 | Participant-bootstrap replicates for every CI. |

### live (interface)
| Key | Default | Meaning |
|---|---|---|
| `method` | SAMA | POS, CHROM or SAMA (falls back to POS without a trained bundle). |
| `update_s`, `hr_window_s`, `display_s` | 1, 20, 10 | Refresh, heart-rate window and trace length. All but `display_s` can be changed live in the Knobs panel. |
| `show_hrv`, `show_beats`, `gate`, `require_still_for_hrv` | true | Display switches, also live. |

### recorder, privacy
| Key | Default | Meaning |
|---|---|---|
| `recorder.input_format`, `device` | v4l2, /dev/video0 | dshow + "video=Camera name" on Windows; avfoundation + "0" on macOS; file + path for testing. |
| `recorder.size`, `fps`, `pixel_format` | 640x480, 30, "" | Ask the camera for this mode; set `pixel_format: yuyv422` to force uncompressed. |
| `recorder.codec` | ffv1 | ffv1 (lossless) or h264_lossless (libx264rgb -qp 0, smaller files). |
| `privacy.save_video` | true | Recorder writes the lossless video (needed for the ladder). |
| `privacy.save_signals` | true | Live readings saved (recorder; browser mode only when a participant id is given). |

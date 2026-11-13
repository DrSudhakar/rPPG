# DATA_INVENTORY (C1)

Status values: **acquired**, **needs-PI** (licence form, account or host the session cannot reach), **unavailable**.
Probe: 2026-11-02 from the cloud session; every dataset host returned no connection (egress policy), see ENV.md.
No raw video is stored in this repository; only derived signals would be (none exist in this run).

| Dataset | Status | Access route (for the PI) | Licence / terms | What SAMA would use it for | Verified properties |
|---|---|---|---|---|---|
| UBFC-rPPG | needs-PI | Dataset page of Bobbia et al. (2019) at Univ. Bourgogne Franche-Comté; Google Drive links | research use; cite Bobbia et al. 2019 | uncompressed 30 fps video for the codec ladder; POS/CHROM replication | 640×480, 30 fps, uncompressed, CMS50E finger oximeter (rPPG-Toolbox paper, LITERATURE) |
| PURE | needs-PI | request to TU Ilmenau (Stricker et al. 2014) | research use agreement | ladder source; motion | 10 participants (8 M, 2 F), 6 tasks, 640×480, 30 fps, CMS50E at 60 Hz (LITERATURE) |
| MMPD | needs-PI | signed request form to the authors (Tang et al. 2023), GitHub page | research licence | real darker-skin errors; ITA° estimator vs Fitzpatrick labels | 33 participants, 660 one-minute videos, Galaxy S22 Ultra 1280×720 → 320×240, Fitzpatrick 3–6, 4 lights, HKG-07C+ oximeter (LITERATURE) |
| UBFC-Phys | needs-PI | IEEE DataPort (account) | CC BY 4.0 per DataPort page (verify) | rest/speech/arithmetic, HRV proxy | 56 participants, 1024×1024 at 35 Hz, Empatica E4 BVP (LITERATURE) |
| MAHNOB-HCI | needs-PI | EULA at mahnob-db.eu | academic EULA | ECG-referenced HRV | ECG at 256 Hz (verify) |
| OBF | needs-PI | Univ. Oulu request | EULA | ECG-referenced HRV | not verified |
| COHFACE, MR-NIRP, VIPL-HR, LGI-PPGI, MMSE-HR, VITAL | needs-PI | individual request pages | various | skin-tone labels, compression | not verified this run |
| SCAMPS | needs-PI | download link in Liu et al. (2022) | CC (verify) | synthetic pretraining | 2,800 synthetic clips (LITERATURE, rPPG-Toolbox paper) |
| rPPG-Toolbox code and weights | **acquired** (code); weights unusable | `git clone` github.com/ubicomplab/rPPG-Toolbox @ b7500b8 | Responsible AI License v1.1 | implementation equivalence (C2) | `.pth` weights need PyTorch (I-23, I-24) |

**C1 gate: FAILED** (no REAL-PUBLIC dataset acquired). Per prompt s.7 C1: continue on SYNTHETIC data,
project downgraded to a design rehearsal, Q-A provisional; download list sent to the PI as a blocker.
`make realdata` (Makefile) documents the loaders to run once the files are placed under `data/raw/`.

---

## Supported datasets and where each one goes on disk

The code (`sama/datasets.py`) reads nine public datasets plus the pilot's own sessions. Put each
one in its own folder under `paths.public_root` (default `data/raw/`), **named after the dataset**:
the folder name is what distinguishes SUMS, LADH, M3PD and RLAP, which all come from the same
PhysRecorder tool and therefore share a layout. Then check what was recognised before training:

```bash
python -m sama datasets                 # what is on disk, per dataset
python -m sama train  --set 'training.datasets=[RLAP, PURE, MMPD]'
python -m sama test   --bundle models/sama_real --datasets MMPD,iBVP   # cross-dataset test
```

| Dataset | Folder | Expected layout | Reference signal | Notes |
|---|---|---|---|---|
| UBFC-rPPG | `data/raw/UBFC-rPPG/` | `subject*/{vid.avi, ground_truth.txt}` | PPG wave in line 1, times in line 3 | the ladder source |
| PURE | `data/raw/PURE/` | `NN-NN/Image*.png` + `NN-NN.json` | `/FullPackage` waveform, 60 Hz | PNG sequence, ns filenames |
| VIPL-HR | `data/raw/VIPL-HR/` | `data/pNN/vN/sourceN/{video.avi, wave.csv, gt_HR.csv, time.txt}` | `wave.csv`; falls back to `gt_HR.csv` | variable fps, so `time.txt` is used as the clock; NIR sources skipped |
| iBVP | `data/raw/iBVP/` | `pNN_x/{pNN_x_rgb/, pNN_x_t/, pNN_x_bvp.csv}` | `*_bvp.csv` | RGB only; paired thermal (`*_t/`) unused |
| SUMS | `data/raw/SUMS/` | `pNNN/vNN/{video_*.avi, BVP.csv, HR.csv, SpO2.csv, frames_timestamp.csv}` | `BVP.csv` on a UNIX clock | face camera only (fingertip camera skipped); high-altitude SpO₂ range |
| LADH | `data/raw/LADH/` | `<MM_DD>/p_<MM_DD>_<name>/vNN/{video_RGB_H264.avi, BVP.csv, frames_timestamp_RGB.csv}` | `BVP.csv`, 20 Hz | RGB only (IR skipped); the same person recurs across days and is kept as **one** participant so CV folds stay honest |
| M3PD | `data/raw/M3PD/` | `<subject>/{dual_camera_session_*/front_camera_*.mp4, vNN/BVP.csv}` | `BVP.csv`, 20 Hz | front (face) camera only; rear fingertip view skipped |
| RLAP | `data/raw/RLAP/` | `pNNN/<task>/{video_ZIP_H264.mp4 \| pictures_ZIP_RAW_RGB/, BVP.csv, frames_timestamp.csv}` | `BVP.csv` on a UNIX clock | prefers the lossless PNG sequence when present |
| MMPD | `data/raw/MMPD/` | `subject*/pN_M.mat` | `GT_ppg` inside the `.mat` | per-video labels (light, motion, skin_color, …) carried into the recording's meta; v7.3 `.mat` files need `h5py` |
| PILOT | `pilot_data/` | `sessions/<id>/…` | the pilot's own synchronised oximeter | loaded by `sama.pilotitems`, not by the registry |

Column layouts of the `BVP.csv`/`wave.csv` files are not documented by most of these datasets, so
`sama.datasets.read_signal_csv` infers them (header or not, with or without a UNIX timestamp
column, seconds/ms/ns) and reports what it inferred in each recording's `meta["ref"]`. If a
dataset's HR comes out wrong, that field is the first place to look.

**What none of these datasets can support.** None has a white reference or colour chart in view,
so any ITA° estimated from them is uncalibrated (`training.public_ita`, I-03) — use Monk/Fitzpatrick
labels (MMPD has them) or the pilot for skin-tone strata. None is recorded on the proposal's
compression ladder, so the bitrate analysis stays a pilot-only result.

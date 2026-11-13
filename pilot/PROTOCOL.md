# SAMA pilot protocol (November 2026)

**Purpose.** To show that the SAMA recording set-up, synchronisation, compression ladder and
analysis work on Indian volunteers, and to measure the between-person spread that the power
analysis needs. The pilot does not test H1–H5. It is a pre-funding feasibility study that uses the
cheaper equipment available now (webcam, fingertip oximeter, grey card); it does not replace the
ten instrumented pilot sessions planned in WP1 (colorimeter, 1 kHz ECG, four cameras).

**Scope.** Healthy adult volunteers, about 25 minutes each, non-invasive (a camera and a fingertip
pulse oximeter), minimal risk. No diagnosis is made and no reading is given to the participant as
health information.

**Ethics first.** Do not record anyone until the SVNIT Institutional Ethics Committee (or a
DHR-registered committee) has approved the pilot. `forms/06_ethics_pilot_application.md` is a
ready-to-adapt application. Use `forms/01` and `forms/02` for information and consent.

---

## 1. Sample

| Item | Plan | Why |
|---|---|---|
| Number | 12–20 volunteers (minimum 10) | Enough to estimate the between-person spread and show the pipeline works; at least 9 are needed before a pooled 90 % reliability interval is finite |
| Skin tone | Spread across the whole range: aim for at least 4 with Monk 1–4 (lighter), 4 with Monk 5–7, 4 with Monk 8–10 | Section 5 result 2 needs a range of skin tone; result 3 needs a darker and a lighter group |
| Sex | Roughly balanced | Matches the main study design |
| Age | 18–60 | As in the proposal |
| Recruitment | SVNIT students and staff, by notice and word of mouth | Pan-India campus; see the recruitment note in section 9 |

Because no colorimeter is available yet, recruit by a quick Monk rating at screening (section 4.3).
Indian colorimeter data suggest lighter skin (ITA° > 28) is the scarce group, so look for lighter
volunteers early.

**Exclusions** (same rules as the proposal, limited to what invalidates the reference): pacemaker
or other implanted cardiac device; known atrial fibrillation or other sustained arrhythmia.
**Recorded, not excluded:** beta-blockers or other heart-rate-lowering medicines, beard,
spectacles, bindi, make-up, head covering, nail polish on the oximeter finger (remove if possible,
otherwise use another finger and note it).

---

## 2. Equipment

| Item | Notes |
|---|---|
| USB webcam, 640×480 at 30 fps, **uncompressed (YUYV) output** | Check with `v4l2-ctl --list-formats-ext -d /dev/video0` (Linux). If the camera offers only MJPG at 30 fps, it is already compressing: record at a lower resolution that offers YUYV at 30 fps, or note it. Turn off auto-focus hunting if the driver allows (`v4l2-ctl -c focus_auto=0`). |
| Recording PC (Linux, Windows or macOS) | Python 3.11+, ffmpeg with libx264 and libvpx, this repository. **Disk: lossless 640×480 video is about 0.5–1 GB per minute**, i.e. 15–25 GB per session; keep 300 GB free for 15 sessions and back up to an external drive the same day. |
| Fingertip pulse oximeter **that logs its waveform to the PC** | e.g. a CMS50D+/CMS50E-type unit with USB cable and PC software that saves the pleth wave (about 60 Hz) with PC time stamps. A unit that only shows numbers on its screen cannot give per-window reference values; see `OXIMETER.md`. |
| Grey card (photographic 18 % grey) or a matte white card | The skin-tone reference. A grey card is preferred because a white card often clips (saturates) when the camera brightens a darker face. Note its reflectance (18 % grey: 0.18; matte white paper: about 0.85–0.90). |
| Two light conditions | Light 1: desk LED lamp in front of the face, about 500 lux at the face. Light 2: ordinary room light only, about 100 lux. A lux-meter phone app is enough; write the readings in the CRF. |
| Optional: a phone on a stand | Its own camera app, front camera, 30 fps. Adds the "phone" stream; it is synchronised afterwards by cross-correlation. |
| Optional: 24-patch colour chart | Better than a card for colour calibration (`ita.camera.white_reference=chart`). |
| Printed or on-screen Monk Skin Tone scale | From the official source (skintone.google). Use the same copy and the same light for every rating. |
| Chair with backrest, plain background, a table | The participant sits about 60–80 cm from the webcam, face filling about a third of the frame width. |
| LED or torch for sync marks (optional) | A short flash in front of the camera while the operator presses "Sync mark" in the recorder. |

---

## 3. Set-up (once, then 5 minutes before each session)

1. Place the webcam at eye level, 60–80 cm from the face. Lock the camera's exposure and white
   balance if the driver allows it; if not, leave it on auto and note that in the CRF (auto-exposure
   is what consumer devices do, and the grey card corrects colour for the skin-tone estimate).
2. Start the oximeter PC software on the **same PC** as the recorder (one clock for both). Check that
   it writes PC time stamps.
3. Start the recorder: `python -m sama record --participant P01` and open http://127.0.0.1:8000.
4. On the preview, **draw the face box** (forehead to chin, cheek to cheek) and the **card box**
   (inside the grey card, away from its edges). Enter the card's reflectance. The card indicator must
   read "card ok"; if it reads "clipped", dim the lamp or move the card slightly away from the light.
   The card box is a fixed rectangle in the frame, so the card must be **in that rectangle during
   both `ita_capture` blocks**. Easiest: clip the card to a small stand or the chair's headrest
   beside the cheek, at face distance, and leave it there for the whole session. If the participant
   holds the card instead, ask them to hold it in the same place both times and redraw the card box
   at the start of each `ita_capture` block. Record both `ita_capture` blocks: the estimate is the
   average of the two lights, and a session whose card was never in the box gets no camera ITA°
   (it then falls back to the Monk stratum).
5. Measure the lux at the face for light 1 and light 2 and write both in the CRF.

---

## 4. Session (about 25 minutes)

Talk the participant through each step. They may stop at any time.

### 4.1 Before recording (5 min)
- Information sheet, questions, written consent (`forms/01`, `forms/02`).
- Screening and CRF part A (`forms/03`): age, sex, exclusions, medicines, beard, spectacles, etc.
- Monk rating by rater 1 (and rater 2 if available), **before** any camera result is seen
  (`forms/04`). The participant also gives their own Monk rating and Fitzpatrick answers.
- Oximeter on the left index finger (or the finger noted in the CRF), hand resting on the table
  below heart level and still. Start oximeter logging.

### 4.2 Recording (about 18 minutes)
Press **Start recording**. For each block below: choose the light and block in the recorder, press
**Block start**, run the block, press **Block end**.

| # | Block (`block` name) | Light | Duration | Instructions to the participant |
|---|---|---|---|---|
| 0 | `ita_capture` | light 1 | 30 s | Hold the grey card flat beside the cheek, facing the camera, at the same distance as the face. Look at the camera. |
| 1 | `motion_check` | light 1 | 30 s | First 20 s completely still; then 10 s of gentle nodding. (Tunes the motion warning.) |
| 2 | `rest` | light 1 | 3 min | Sit quietly, breathe normally, look at the screen. |
| 3 | `consultation` | light 1 | 3 min | The operator asks the ten scripted questions below; answer naturally. |
| 4 | `arithmetic` | light 1 | 2 min | Count backwards from 1,000 in steps of 7, aloud; the operator says "mistake, start again from 1,000" after errors. |
|   | *switch to light 2* | | 1 min | Turn the lamp off; check the face is still visible; press Sync mark. |
| 5 | `ita_capture` | light 2 | 30 s | As block 0 (gives a second skin-tone estimate under a different light). |
| 6 | `rest` | light 2 | 3 min | As block 2. |
| 7 | `consultation` | light 2 | 3 min | Different ten questions (list B). |
| 8 | `arithmetic` | light 2 | 2 min | Count backwards from 1,022 in steps of 13. |

Press **Stop recording**, then stop the oximeter logging and save its file.

Watch the interface during the session: the **motion warning** (amber = HRV paused; red = readings
withheld), the face, light and frame-rate indicators. If the face leaves the box for more than a few
seconds, add a note ("participant turned away at 2:10"). Never tell the participant the heart rate
or HRV values: they are research signals, not health information.

**Mock consultation questions, list A:** (1) What brings you here today? (2) When did it start?
(3) Any fever, cough or cold in the last week? (4) Are you taking any medicines? (5) Any allergies?
(6) How have you been sleeping? (7) How is your appetite? (8) Do you exercise regularly? (9) Any
headaches recently? (10) Anything else you would like to tell the doctor?
**List B:** (1) How are you feeling today? (2) Any pain anywhere? (3) Any breathing difficulty on
climbing stairs? (4) How many hours do you sleep? (5) Do you drink tea or coffee, how many cups?
(6) Any recent travel? (7) Anyone at home unwell? (8) Do you feel stressed at work or study?
(9) Any dizziness? (10) Any questions for the doctor?

### 4.3 After recording (2 min)
- Remove the oximeter. Thank and pay/thank the participant as approved by the ethics committee.
- Copy the oximeter file into the session folder as `reference.csv` (see `OXIMETER.md`).
- Fill CRF part B (lux, any problems, deviations).
- Run the session check: `python -m sama prepare --session P01` then `python -m sama qc`. Preparing one
  25-minute session (seven decodes and the compression ladder) takes roughly 20–40 minutes on a
  desktop PC; start it after the participant leaves, or run all sessions overnight.
  Re-record only if the participant agrees and QC failed for a fixable reason.

---

## 5. Files per participant

```
pilot_data/
  participants.csv            from the CRF (one row per participant; templates/participants.csv)
  monk_ratings.csv            one row per rater per participant (templates/monk_ratings.csv)
  sessions/P01/
    session.json              written by the recorder (boxes, card reflectance, start time)
    events.csv                block markers, sync marks, notes (recorder)
    webcam.mkv                lossless video (recorder)
    live_readings.csv         what the live interface computed (recorder)
    reference.csv             oximeter export (you copy it here)
    phone.mp4                 optional phone recording (you copy it here)
```

Participant ids are codes (P01, P02, ...). Names and contact details stay only on the paper
consent forms, in a locked cupboard, never in these folders.

---

## 6. Analysis

```
python -m sama prepare            # every session: signals, compression ladder, camera ITA, sync, QC
python -m sama train              # SAMA on public data (UBFC-rPPG, PURE) unless already trained
python -m sama pilot              # section-5 numbers, Figure 6, pilot report
python -m sama train-ita --target monk   # optional: video -> Monk model, cross-validated
```

Outputs in `results/pilot/`: `section5_filled.md` (the proposal paragraph with every bracket
filled), `figure6_POS.png` and `figure6_SAMA.png`, `pilot_report.md`, `pilot_summary.json`,
`pilot_windows.csv`. Every knob is in `configs/sama.yaml`; see `docs/KNOBS.md`.

---

## 7. What can and cannot be claimed from this pilot

| Can say | Cannot say |
|---|---|
| The set-up recorded n volunteers; k were usable; streams aligned within x ms | That SAMA reduces the skin-tone gap |
| Error moved by a bpm per 10° of (camera-estimated) ITA° in this small sample, with its CI | That error rises with skin tone in Indians (the CI will be wide) |
| The compression ladder runs and changes error by b and c bpm in the two groups | That compression widens the gap |
| Camera ITA° and Monk ratings agreed with ρ = ... | That camera ITA° equals colorimeter ITA° |
| SAMA, trained only on public data, gave ... on the pilot with ...% abstention | Anything about clinical use, diagnosis or a medical device |

## 8. Safety and data protection
- Stop immediately if the participant feels unwell or wishes to stop. Arithmetic stress is mild
  and brief; debrief afterwards.
- Video is stored only on the encrypted recording PC and one encrypted backup drive. No cloud.
- Faces are personal data under the DPDP Act 2023: use the tiered consent in `forms/02`.

## 9. Recruitment note
The recruitment notice should say: "healthy adult volunteers of every skin tone, 25 minutes, a
camera and a finger clip, no needles". Track the Monk distribution after each session (the pilot
report lists it) and actively seek the group that is short.

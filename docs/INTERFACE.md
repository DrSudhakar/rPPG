# The live interface

Two modes, one page (`app/static/`), served by `app/server.py`.

| | Browser camera (`python -m sama serve`) | Lab recorder (`python -m sama record`) |
|---|---|---|
| Camera | Opened by the browser | Opened by ffmpeg on the PC |
| What leaves the camera | Only 30 colour averages, a motion index, the skin fraction and the brightness per frame, computed in the page | Lossless video to `webcam.mkv` on this PC; a reduced copy to the live analysis |
| Saved | Nothing, unless a participant id is entered (then readings only) | Video, session metadata, block markers, live readings |
| Use | Demonstrations; checking a set-up | The pilot and the main study |

## What the screen shows

- **Heart rate** in large digits, with its 90 % interval when the SAMA model and gate are on. When a reading is withheld the number is replaced by the reason ("Moving too much: please hold still", "Face not in the box", "Too dark", "No reliable reading").
- **Beat dots**: a red dot on the pulse trace at every detected beat, and a dot that blinks on the video and beside the heart rate with the rhythm of the detected beats (about one second behind real time, because beats are confirmed on the signal before they are shown).
- **HRV**: RMSSD, SDNN and pNN50 over the last 60 seconds, only when that whole minute was still enough; otherwise "Paused: movement in the last minute" or "Collecting 34 of 60 s".
- **Signal quality**: motion, face, light and frame-rate indicators, green / amber / red.
- **Movement warning**: an amber banner on the video when motion may affect the readings (HRV paused), and a flashing red banner when readings are withheld.
- **Knobs (live)**: method (POS, CHROM, SAMA), heart-rate window, motion thresholds, update interval, gate on/off, HRV and beat dots on/off, HRV only when still.
- **Lab recorder controls**: participant id, operator, start/stop recording, light and block selection with block start/end buttons, sync mark, notes, the event log, a recording timer, and the card indicator (ok / clipped / too dark).

## Drawing boxes
Choose *Face box*, *Grey/white card* or *Chart corners* and drag (or click four corners) on the video.
The face box should run from the forehead to the chin and from cheek to cheek; the card box should sit
inside the card, away from its edges. Enter the card's reflectance (18 % grey card: 0.18).

## Motion index
Mean absolute change of the face region's brightness between consecutive frames, in 8-bit units, on a
48 × 48 thumbnail. Typical still values are below 1–2; nodding gives 5–15, depending on the camera and
light. Set `quality.motion_warn` and `quality.motion_abstain` from the `motion_check` block of your own
sessions (`qc.json` suggests values).

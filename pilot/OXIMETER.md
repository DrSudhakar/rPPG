# Getting reference data from a fingertip pulse oximeter

The pilot's reference is a fingertip pulse oximeter (contact photoplethysmography at the finger).
What the analysis can do depends on what the oximeter can export:

| What the oximeter gives | Set `reference.type` to | HR per 30 s window | HRV (pulse-rate variability) | Cross-correlation sync |
|---|---|---|---|---|
| Pulse waveform ("pleth", ~60 Hz) with PC time stamps | `oximeter_pleth` | yes, from beats | yes | yes |
| Displayed pulse rate only (about 1 value per second) with PC time stamps | `oximeter_hr` | yes, mean of readings (lags by a few seconds) | no | no (wall clock only) |
| Only a number on its screen, nothing saved | cannot be used for per-window accuracy | — | — | — |

**Buy or borrow a model that logs the waveform to a PC** (USB cable plus PC software that saves the
pleth trace). Units of the CMS50D+/CMS50E family are widely used for this (UBFC-rPPG and PURE used
CMS50E units) and cost little; check before buying that the PC software for your unit exports the
waveform with time stamps. Bluetooth units with phone apps often export only the pulse rate.

## File format

Each session needs one CSV in the session folder whose name matches `reference.file_glob`
(default `reference*.csv`). The column names are set in `configs/sama.yaml`, so you can usually keep
the export as it is and just tell the pipeline which column is which:

```yaml
reference:
  type: oximeter_pleth
  columns: {time: time_s, pleth: pleth, hr: hr_bpm, spo2: spo2}
  time_format: unix_s        # unix_s | unix_ms | relative_s
  fs_hz: 60                  # only used when there is no time column
```

If the export's times are clock times such as `2026-11-12 10:31:05.250` rather than unix seconds,
convert them with the helper:

```
python pilot/tools/convert_reference.py export.csv pilot_data/sessions/P01/reference.csv \
    --time-col "Time" --time-format "%Y-%m-%d %H:%M:%S.%f" --pleth-col "Pleth" --hr-col "PR" --spo2-col "SpO2"
```

It writes the canonical columns `time_s` (unix seconds), `pleth`, `hr_bpm`, `spo2`, and prints the
duration, the sampling rate and the share of missing samples so you can check the file at once.

## Clocks

Run the oximeter software on the recording PC. Then both files carry the same PC clock and the
pipeline aligns them directly, then refines the alignment by cross-correlating the camera pulse with
the oximeter wave (`sync.method: xcorr`). The finger pulse arrives about 0.1–0.2 s after the face
pulse; that constant offset is absorbed by the alignment and does not change heart rate.

If the oximeter writes times relative to its own start (`relative_s`), press **Sync mark** in the
recorder at the moment you start the oximeter logging, and the cross-correlation does the rest.

## What the reference is, and is not
A finger PPG measures the pulse wave, not the heart's electrical beats. Its beat-to-beat variability
(pulse-rate variability) follows ECG HRV closely at rest but not exactly, because pulse transit time
varies with breathing and posture. The pilot report therefore calls it PRV and the proposal keeps
1 kHz ECG as the main study's reference.

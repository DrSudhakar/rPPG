# SAMA pilot kit

Everything needed to record the SAMA pilot on volunteers and turn it into the results promised in
section 5 of the proposal.

| File | Use |
|---|---|
| `PROTOCOL.md` | The session, minute by minute; equipment; what can and cannot be claimed |
| `OXIMETER.md` | Getting reference data out of a fingertip pulse oximeter; file format; clocks |
| `forms/01_participant_information_sheet.md` | Give to each volunteer |
| `forms/02_consent_form.md` | Written, tiered consent |
| `forms/03_screening_and_case_report_form.md` | Screening, participant details, session log (CRF) |
| `forms/04_monk_rating_sheet.md` | Monk Skin Tone rating, one or two raters |
| `forms/05_session_checklist.md` | One per session |
| `forms/06_ethics_pilot_application.md` | Template for the SVNIT IEC (expedited review) |
| `forms/print/*.docx` | The forms as Word files for printing (`make forms`) |
| `templates/*` | Example `participants.csv`, `monk_ratings.csv`, `ita_measured.csv`, `session.json`, `events.csv`, `reference.csv` |
| `tools/convert_reference.py` | Convert an oximeter export to the reference CSV |
| `tools/import_remote.py` | Convert downloaded remote/browser sessions (see below) into this layout |

## Remote arm (no lab visit, runs in the participant's browser)

`../webexp/` is a separate, no-install experiment participants run from a link in their own
browser — no lab visit, their own webcam, works offline and syncs to Firebase when connected. See
`../webexp/REMOTE_PILOT_GUIDE.md` for full setup, testing, troubleshooting, and the participant
FAQ. In short:

```bash
# after downloading each participant's webcam.webm + metadata.json from Firebase Storage into
# downloads/<pid>/...
python pilot/tools/import_remote.py downloads --out pilot_data_REMOTE

# analyze the remote arm SEPARATELY from the lab pilot_data/ -- no grey card (skip the compression
# ladder) and no external rater (self-report Monk/Fitzpatrick only, via ita.source=monk)
python -m sama prepare --set paths.pilot_root=pilot_data_REMOTE
python -m sama qc      --set paths.pilot_root=pilot_data_REMOTE
python -m sama pilot   --set paths.pilot_root=pilot_data_REMOTE --set paths.outputs=results/pilot_remote \
                       --set pilot.ladder.enabled=false --set ita.source=monk --set raters.n=1
```

Ethics approval for this arm must be in hand (likely an amendment to
`forms/06_ethics_pilot_application.md`) before sending the link to any real participant.

## The pilot in eight commands

```bash
python -m sama config                                  # check the configuration (knobs: docs/KNOBS.md)
python -m sama record --participant P01                # record a session (open http://127.0.0.1:8000)
python pilot/tools/convert_reference.py ...            # if the oximeter export needs converting
python -m sama prepare --session P01 && python -m sama qc   # right after each session
python -m sama train                                   # SAMA on UBFC-rPPG + PURE (once; needs data/raw/)
python -m sama pilot                                   # section 5 numbers, Figure 6, pilot report
python -m sama train-ita --target monk                 # optional: video -> Monk model
python -m sama serve                                   # live demo with the browser camera (no recording)
```

## Try it first on synthetic sessions

```bash
python -m sama synth-pilot --n 8 --out pilot_data_SYNTHETIC
python -m sama all --set paths.pilot_root=pilot_data_SYNTHETIC --set project.provenance_label=SYNTHETIC \
                   --set paths.outputs=results/pilot_SYNTHETIC
```

The synthetic sessions have exactly the recorder's layout (lossless video with a grey card,
oximeter CSV with a clock offset, events, Monk ratings by two raters). Their results are labelled
SYNTHETIC and must never be quoted.

## What the pilot produces (`results/pilot/`)

| File | Content |
|---|---|
| `section5_filled.md` | The proposal's section 5 pilot paragraph with every bracket filled from the measured numbers, or "[not measured]" |
| `figure6_POS.png`, `figure6_SAMA.png` | Figure 6 (A: error vs ITA°; B: error by bitrate per stratum; C: Bland–Altman), drawn by the proposal's own `scripts/pilot_figure.py` |
| `pilot_report.md` | All results with CIs: feasibility, skin tone, accuracy by method and level, compression, agreement, SAMA gate per stratum, HRV, camera ITA° vs Monk, rater agreement, power re-check, configuration |
| `pilot_summary.json` | Every number (the proposal build reads these) |
| `pilot_windows.csv`, `pilot_hrv_windows.csv`, `pilot_participant_mae.csv`, `pilot_skin_tone.csv` | Window- and participant-level tables |
| `pilot_hr.csv` | Input for `pilot_figure.py` in the proposal's schema |
| `figure_sama_gate.png`, `figure_ita_vs_monk.png`, `figure_hrv_ba.png` | Supporting figures |
| `config_used.yaml` | The switches that produced all of the above |

# STATUS (10 November 2026)

## Where things stand

| Area | State | Where |
|---|---|---|
| Real-data pipeline | Built and tested on synthetic sessions; never yet run on a real pilot session | `sama/`, `docs/PIPELINE.md` |
| Knobs | One configuration file, CLI overrides, reconciliation logged | `configs/sama.yaml`, `docs/KNOBS.md` |
| Skin tone without a colorimeter | Camera ITA° against a grey card; Monk ratings (1 or 2 raters); video-to-skin-tone model | `sama/ita_camera.py`, `sama/monk.py`, `sama/ita_model.py` |
| SAMA training on real data | Skin-tone blind by default; cross-conformal gate; CV metrics in the bundle | `sama/realtrain.py`; run `python -m sama train` on UBFC-rPPG + PURE |
| Live interface | BPM, HRV, beat dots, motion/face/light/frame-rate warnings, knobs; browser and lab-recorder modes | `app/`, `docs/INTERFACE.md` |
| Pilot kit | Protocol, oximeter guide, forms (also as Word), templates | `pilot/` |
| Pilot results | Script ready; produces section 5 numbers, Figure 6, pilot report | `python -m sama pilot` |
| Proposal | Rewritten for 36 months, SAMA-150 (45 analysable per stratum), ₹33,67,250; revised criteria with simulated power; section 5 filled automatically; build refuses pending numbers and [TO FILL] fields | `proposal/`, `python proposal/build_proposal.py` |
| Technical report v2 | Written; real-data sections fill when results exist | `report/SAMA_technical_report_v2.md` |
| In-silico rehearsal (v1) | Unchanged | `report/SAMA_insilico_report.pdf`, `GO_NO_GO.md` |

## Blockers needing the PI

1. **Ethics approval for the pilot** (`pilot/forms/06_ethics_pilot_application.md`). No recording before approval.
2. **A fingertip oximeter that logs its waveform to the PC**, a grey card, one or two Monk raters.
3. **Call terms and eligibility** confirmed on anrfonline.in (36 months; ₹60 lakh cap; within two years of joining).
4. **Public data**: UBFC-rPPG is on the PI's machine (30 subjects); PURE if access arrives. Then `python -m sama train`.
5. **Quotations** for the budget lines.

## Next steps
1. `make test`; `make pilot-synthetic` to rehearse the chain on your machine.
2. `python -m sama train` on UBFC-rPPG.
3. Record the pilot (`python -m sama record --participant P01`), `prepare`, `qc`, then `python -m sama pilot`.
4. `python proposal/build_proposal.py` until nothing is pending; upload.

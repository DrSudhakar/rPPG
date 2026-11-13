---
title: "SAMA: pilot and submission plan (not for submission)"
subtitle: "10 to 17 November 2026"
author: "Working document for the PI"
---

# The plan in one paragraph

Seven days remain. Run the pilot on 12–20 Indian volunteers with the lab recorder, a fingertip oximeter that logs its waveform to the PC, an 18% grey card and one or two Monk raters; train SAMA once on UBFC-rPPG (and PURE if available); run `python -m sama pilot`; build the proposal with `python proposal/build_proposal.py`, which fills section 5 from the measured numbers and exits with status 2 while any number is pending or any **[TO FILL]** field is left. In parallel, settle the administrative items that can sink the application regardless of its science (call terms, eligibility, ethics, quotations). **Never put a number in the proposal that the build did not fill from measured data, and never submit a build that used SYNTHETIC inputs** (every page of such a build is watermarked).

# Day-by-day

| Day | Pilot | Public data and software | Proposal and admin |
|---|---|---|---|
| **Tue 10** | Submit the pilot application (`pilot/forms/06`) to the SVNIT IEC for expedited review; book the room; recruitment notice out (aim for spread on the Monk scale, including lighter volunteers) | `make test`; `make pilot-synthetic` to rehearse the whole chain on your machine | Download the call document and template from anrfonline.in; confirm 36 months, ₹60 lakh cap, travel and contingency ≤ ₹3 lakh each, overhead rate; **eligibility: within two years of joining SVNIT; age ≤ 42 (+3 where applicable); no earlier SERB/ANRF project grant** |
| **Wed 11** | Get the oximeter (PC waveform logging, CMS50D+/CMS50E type), grey card and a desk LED lamp; check the webcam's uncompressed mode (`ffmpeg -f v4l2 -list_formats all -i /dev/video0`); dry run on yourself: `python -m sama record --participant P00`, then `prepare`, `qc` | `python -m sama train` on UBFC-rPPG (+ PURE); check `models/sama_real/cv_metrics.json` | Request written quotations for every equipment line |
| **Thu 12** | If approved: P01–P05. After each session: `python -m sama prepare --session <dir>` and `python -m sama qc`. After P02, tune `quality.motion_warn`/`motion_abstain` from the suggestions in `qc.json` | Fix anything QC flags (face box, card clipping or out of its box, oximeter time format) | Draft the endorsement letter (lab space, GPU cluster, existing equipment) |
| **Fri 13** | P06–P12, filling the scarce Monk groups (check `results/pilot/participants.csv` after each run) | Back up sessions to the encrypted drive | Fill section 9.2 and 5.2 ethics reference; check references |
| **Sat 14** | P13–P20 if volunteers are available; second rater rates if possible (`raters.n=2`) | `python -m sama pilot` (full run); read `results/pilot/pilot_report.md` | `python proposal/build_proposal.py`; read section 5 against the pilot report |
| **Sun 15** | Spare day for re-recording failed sessions | `python -m sama train-ita --target monk` (optional; feasibility only) | Give the built PDF and the stress test to one colleague to attack |
| **Mon 16** | — | Final pilot run with the frozen configuration; keep `config_used.yaml` | Revise; paste quotations; re-build until the build prints no PENDING and no TO FILL; fill the portal fields |
| **Tue 17** | — | — | Upload: technical document PDF, biodata, PI certificate, endorsement, age proof, plagiarism undertaking. Deadline. |

# If ethics approval does not arrive in time

Use the public-data results only and say so plainly. Retitle section 5.2 "Preliminary analysis of public data", keep 5.1 (filled from the trained bundle), and state that the volunteer pilot is scheduled for months 3–4 as the WP1 protocol pilot. Do **not** record volunteers without approval: a reviewer who sees "approval pending" next to human data will stop reading. Build with `--pilot` pointing to an empty folder: every pilot number then shows **[pending]**, and those sentences must be deleted from `SAMA_proposal.md` rather than filled by hand.

# What the pilot must show to persuade a reviewer

1. **The pipeline runs end to end on Indian faces** (feasibility: k of n usable, synchronisation spread in ms, minutes per session).
2. **The measurements behave sensibly:** error moves with skin tone in the expected direction or the CI is honestly wide; compression changes error; Bland–Altman bias near zero.
3. **The reliability gate does what it says on people it never saw:** coverage of shown intervals near 90%, abstention reported by group.
4. **The team can recruit across the skin-tone range**, including the scarce lighter group.
5. **The power assumptions hold:** the re-check reports H1 power at 45 per stratum using the pilot's own spread. If it falls below 0.80, the proposal already states the safeguard (up to 55 recruited per stratum).

It does not need significance, and it must not claim it.

# Pilot sample: why 12–20 and how to spread it

Twelve usable sessions are the minimum for the re-check to give a usable SD estimate and for the gate's coverage to mean anything (nine participants are needed for a finite 90% interval). Aim for at least four volunteers in each Monk group (1–4, 5–7, 8–10). If fewer than two land in the lighter or darker ITA° group, the analysis switches to Monk strata for everyone (pre-specified, and the report says so); a single participant whose card was out of its box is placed by their Monk rating.

# Equipment for the pilot (all low cost)

| Item | Why | Approximate cost |
|---|---|---|
| Fingertip pulse oximeter with USB/PC software that logs the pleth waveform | Reference HR and pulse-rate variability; cross-correlation sync | ₹2,500–5,000 |
| Photographic 18% grey card (preferred) or matte white card; a clip or stand to hold it beside the cheek | White reference for camera ITA°; must stay inside its drawn box | ₹300–800 |
| Desk LED lamp | Light 1 (about 500 lux) | existing |
| USB webcam with uncompressed mode at 640×480, 30 fps | Lossless source video for the ladder | existing or ₹2,000–4,000 |
| External drive, 1 TB, encrypted | Lossless video is about 0.3–1 GB a minute | existing |
| Printed Monk Skin Tone scale (official file) | Ratings by one or two raters | printing |

# Submission checklist

- [ ] Call terms confirmed: 36 months; ≤ ₹60 lakh excluding overheads; travel ≤ ₹3 lakh; contingency ≤ ₹3 lakh
- [ ] Eligibility confirmed: within two years of joining SVNIT; age; no earlier SERB/ANRF project grant
- [ ] Ethics: pilot approval reference in sections 5.2 and 9.2; main-study application plan stated
- [ ] Every **[TO FILL]** field replaced (ethics reference, date of joining, existing equipment)
- [ ] Written quotations attached; budget table rebuilt (the build checks the sums)
- [ ] `python proposal/build_proposal.py` finishes with exit status 0: no PENDING, no TO FILL, limits met
- [ ] Figure 6 is `results/pilot/figure6_POS.png` from real pilot data (never the synthetic one)
- [ ] The pilot summary says `provenance: REAL-PILOT` and `trained_on` does not include PILOT
- [ ] Portal fields within limits (the build prints the counts)
- [ ] References verified against the publishers' pages
- [ ] Endorsement letter lists lab space, GPU cluster and existing equipment
- [ ] Subject area: Engineering Sciences
- [ ] One PM-ECRG proposal only
- [ ] A colleague has read the draft as a hostile reviewer, using the stress test

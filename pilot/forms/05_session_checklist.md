# Session checklist (print one per participant)

Participant code: **P____**   Date: ______   Operator: ______

**Before the participant arrives**
- [ ] Ethics approval on file; information sheets and consent forms printed
- [ ] Disk space ≥ 30 GB free on the recording PC; backup drive connected
- [ ] Webcam in YUYV 640×480 at 30 fps (`v4l2-ctl --list-formats-ext`); focus fixed if possible
- [ ] Oximeter charged; PC software logs waveform with PC time stamps (test 30 s)
- [ ] `python -m sama record --participant P__` running; page open at http://127.0.0.1:8000
- [ ] Desk lamp and room light checked; lux written in CRF
- [ ] Grey/white card and Monk scale ready

**With the participant**
- [ ] Information sheet read; questions answered; consent signed (tier A or B)
- [ ] Screening: no pacemaker, no known arrhythmia (else thank and stop)
- [ ] CRF Part A completed; Monk ratings by rater 1 (and rater 2) done independently
- [ ] Oximeter on finger; hand resting still; logging started
- [ ] Face box drawn; card box drawn; card indicator "ok"
- [ ] Start recording
- [ ] Light 1: ita_capture 30 s → motion_check 30 s → rest 3 min → consultation 3 min → arithmetic 2 min (Block start/end each)
- [ ] Switch to light 2; Sync mark
- [ ] Light 2: ita_capture 30 s → rest 3 min → consultation 3 min → arithmetic 2 min
- [ ] Stop recording; stop oximeter logging
- [ ] Debrief; payment as approved

**After**
- [ ] Oximeter file copied to `sessions/P__/reference.csv` (convert if needed)
- [ ] CRF Part B completed; rows added to participants.csv and monk_ratings.csv
- [ ] `python -m sama prepare --session P__` and `python -m sama qc` → usable?
- [ ] Session folder backed up to the encrypted drive; consent form filed

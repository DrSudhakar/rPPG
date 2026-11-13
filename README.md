# SAMA

Skin-tone-equitable camera heart rate and HRV: the research software, pilot kit and proposal for
the SAMA PM-ECRG project. Research use only. Not a medical device. No diagnostic output.

## What is here

| Folder | What it is |
|---|---|
| `sama/` | The Python package. `python -m sama <command>`: config, record, serve, prepare, qc, train, train-ita, pilot, synth-pilot, all |
| `configs/sama.yaml` | **Every knob** in one file (skin-tone source, conditioning, group-DRO, gate type, raters, reference type, motion thresholds, ...). Explained in `docs/KNOBS.md` |
| `app/` | The live interface (BPM, HRV, beat dots, motion/face/light warnings) in browser-camera mode and lab-recorder mode |
| `pilot/` | Pilot kit: protocol, oximeter guide, information sheet, consent, CRF, Monk rating sheet (1 or 2 raters), checklist, ethics application, templates |
| `proposal/` | The rewritten proposal, pilot and submission plan, reviewer stress test and change log, as Markdown sources with a build script that fills section 5 from the pilot results |
| `report/` | Technical reports: the in-silico design rehearsal (v1) and the real-data and pilot pipeline (v2) |
| `analysis/` | Design checks, the in-silico rehearsal, proposal power, figures |
| `sim/` | SAMA-Sim, the physics-based simulator (synthetic data and tests only) |
| `tests/` | Unit and integration tests (`make test`) |
| `docs/` | `KNOBS.md`, `PIPELINE.md`, `INTERFACE.md`, pipeline diagram |

## Quick start

```bash
pip install -r requirements.txt           # Python 3.11+; also needs ffmpeg with libx264 and libvpx
python -m sama config                     # check the configuration
make test                                 # run the tests

# rehearse the whole pilot pipeline on SYNTHETIC sessions (never quote these numbers)
python -m sama synth-pilot --n 8 --out pilot_data_SYNTHETIC
python -m sama all --set paths.pilot_root=pilot_data_SYNTHETIC --set project.provenance_label=SYNTHETIC \
                   --set paths.outputs=results/pilot_SYNTHETIC

# the real pilot
python -m sama datasets                   # which public datasets are on disk and recognised
python -m sama train                      # SAMA on the datasets in training.datasets (data/raw/)
python -m sama test --datasets MMPD       # test a trained bundle on datasets it never saw
python -m sama record --participant P01   # record a session; open http://127.0.0.1:8000
python -m sama prepare && python -m sama qc
python -m sama pilot                      # section-5 results, Figure 6, pilot report
python proposal/build_proposal.py         # proposal PDF/DOCX with the measured numbers filled in
```

Start with `pilot/README.md` and `pilot/PROTOCOL.md`. Where things stand: `STATUS.md`. Every
non-trivial choice: `DECISIONS.md`. Every known problem: `ISSUES.md`.

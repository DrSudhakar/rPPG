# REPRODUCE

From a clean checkout on Linux with Python 3.13, ffmpeg (libx264, libvpx) and the packages in
`requirements.lock`:

```bash
make env          # check the environment
make all          # third_party, tests, design checks, simulator calibration, real-codec check,
                  # cohort + pools, C6 rehearsal, C7/C8 evaluation, ONNX export, acceptance, figures, manifest
make coverage     # unit-test line coverage of core modules (stdlib trace)
make app          # web demonstrator at http://127.0.0.1:8000
make report       # report/SAMA_insilico_report.pdf and .docx
```

Seeds are fixed (spec/SAP_SIM.md s.4). Run times on the build machine (2 × Xeon 2.1 GHz cores,
7 GB RAM, no GPU) are in `results/logs/`. Large intermediates go to `results/cache/` (not
committed). Every number in the report is looked up through `results/manifest.json`.

Tolerance for reproduction: tables regenerated on another machine should match within Monte
Carlo error (power and coverage ±0.02; MAE ±0.05 bpm) because numpy/BLAS reductions can differ
in the last bits across CPUs.

# SAMA in-silico: one command regenerates every table and figure from a clean checkout.
# `make all` = test + data + analyses + export + acceptance + manifest (≈3-4 h on 2 CPU cores).
PY ?= python3
TOOLBOX_COMMIT = b7500b848f84ad7f86e277b4612563b69f4f88f9

# Pilot / real-data pipeline (see pilot/README.md):  make pilot-synthetic | pilot | proposal | forms
.PHONY: pilot pilot-synthetic proposal forms power-proposal all env third_party test coverage calibrate design data cohort pools codec rehearsal method export acceptance figures manifest app report realdata realdata_sama clean

all: third_party test design calibrate codec data rehearsal method export acceptance figures manifest

env:
	$(PY) -c "import numpy, scipy, pandas, sklearn, matplotlib, cv2, onnxruntime, google.protobuf, starlette, uvicorn, yaml; print('environment OK')"
	ffmpeg -hide_banner -encoders | grep -E "libx264|libvpx" >/dev/null && echo "ffmpeg OK"

third_party:
	@test -d third_party/rPPG-Toolbox || git clone https://github.com/ubicomplab/rPPG-Toolbox.git third_party/rPPG-Toolbox
	cd third_party/rPPG-Toolbox && git checkout -q $(TOOLBOX_COMMIT)

test:
	$(PY) -m unittest discover -s tests

coverage:
	$(PY) tools/coverage.py

design:            # proposal's own numbers under its own assumptions (Q-B)
	$(PY) analysis/design_checks.py

calibrate:         # simulator calibration grid (LITERATURE anchor)
	$(PY) sim/calibrate.py

codec:             # real ffmpeg check of the codec surrogate
	$(PY) sim/codec_real.py

cohort:
	$(PY) sim/generate.py cohort --world A --per-stratum 40 --seed 7
pools:
	$(PY) sim/generate.py pool --world A --per-stratum 90 --seed 1
	$(PY) sim/generate.py pool --world B --per-stratum 90 --seed 2
	$(PY) sim/generate.py pool --world melanin --per-stratum 40 --seed 3
data: cohort pools

rehearsal:         # C6
	$(PY) analysis/rehearsal.py
method:            # C7, C8, WP2 budget, final deployment model
	$(PY) analysis/method_eval.py --epochs 5 --folds 5
export:            # ONNX fp32/fp16/int8 + parity
	$(PY) analysis/export_models.py
acceptance:        # C9 acceptance tests
	$(PY) analysis/acceptance.py
figures:
	$(PY) analysis/figures.py
	$(PY) analysis/pilot_figure_public.py
manifest:
	$(PY) tools/manifest.py

app:               # web application on http://127.0.0.1:8000
	$(PY) -m app.server

report:
	$(PY) report/build_report.py

realdata:          # REAL-PUBLIC: needs the PI's downloads under data/raw/ (DATA_INVENTORY.md); loaders tested on synthetic files only
	$(PY) analysis/realdata.py data/raw

realdata_sama:     # EXPLORATORY (I-35/D-26): the trained SAMA bundle, uncalibrated, on data/raw/; not sap-v1, Q-A feasibility only
	$(PY) analysis/realdata_sama.py data/raw

clean:
	rm -rf results/cache results/tables/*.csv results/figures/*.png

# ----------------------------------------------------------------- real data, pilot and proposal
pilot:             # REAL-PILOT: prepare every session under pilot_data/, train SAMA if needed, section-5 results
	$(PY) -m sama all

pilot-synthetic:   # SYNTHETIC rehearsal of the pilot pipeline (never quote its numbers)
	$(PY) -m sama synth-pilot --n 8 --out pilot_data_SYNTHETIC
	$(PY) -m sama all --set paths.pilot_root=pilot_data_SYNTHETIC --set project.provenance_label=SYNTHETIC --set paths.outputs=results/pilot_SYNTHETIC

power-proposal:    # power of the revised criteria (Figure 7 of the proposal)
	$(PY) analysis/proposal_power.py
	$(PY) analysis/proposal_figures.py

proposal:          # proposal, pilot plan, stress test, change log -> proposal/build/*.pdf|docx (fails while numbers are pending)
	$(PY) proposal/build_proposal.py

forms:             # pilot forms as Word files for printing
	mkdir -p pilot/forms/print
	for f in pilot/forms/*.md; do pandoc "$$f" -o "pilot/forms/print/$$(basename $${f%.md}).docx"; done

"""Collect the SYNTHETIC end-to-end verification numbers for the technical report (section 6).

    python analysis/verify_synthetic.py --pilot-root pilot_data_SYNTHETIC --results results/pilot_SYNTHETIC \
        --bundle models/sama_SYNTHETIC --ita-model models/ita_SYNTHETIC/ita_model.json --public-root data_SYNTHETIC/raw
Writes results/verification/verification_synthetic.json (label: SYNTHETIC).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-root", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--ita-model")
    ap.add_argument("--public-root")
    ap.add_argument("--out", default=str(ROOT / "results/verification/verification_synthetic.json"))
    a = ap.parse_args(argv)
    V = {"provenance": "SYNTHETIC"}
    ita_err, lag_err, spreads, usable = [], [], [], 0
    sessions = sorted((Path(a.pilot_root) / "sessions").glob("*"))
    for sd in sessions:
        meta = json.loads((sd / "session.json").read_text())
        d = sd / "derived"
        if (d / "ita_camera.json").exists():
            ita_err.append(abs(json.loads((d / "ita_camera.json").read_text())["ita"] - meta["true_ita_SYNTHETIC"]))
        if (d / "sync.json").exists():
            w = json.loads((d / "sync.json").read_text()).get("webcam", {})
            true_lag = meta.get("true_lag_SYNTHETIC", 0.08)   # finger PTT 0.30 s minus mean face PTT 0.22 s
            lag_err.append(abs(w.get("lag_s", np.nan) - true_lag))
            if w.get("spread_ms") is not None:
                spreads.append(w["spread_ms"])
        if (d / "qc.json").exists():
            usable += json.loads((d / "qc.json").read_text())["usable"]
    V.update(n_pilot=len(sessions), n_usable=usable, ita_abs_error_mean=float(np.mean(ita_err)), ita_abs_error_max=float(np.max(ita_err)),
             sync_abs_error_ms=float(1000 * np.nanmean(lag_err)), sync_spread_ms=float(np.median(spreads)) if spreads else None)
    if a.public_root:
        V["n_public"] = len(list(Path(a.public_root).glob("**/ground_truth.txt")))
    meta = json.loads((Path(a.bundle) / "sama_model.json").read_text())["meta"]
    V.update(train_mae_sama=meta["cv"]["all"]["mae_sama"], train_mae_pos=meta["cv"]["all"]["mae_pos"])
    S = json.loads((Path(a.results) / "pilot_summary.json").read_text())
    V["pilot_mae_pos"] = S["accuracy"]["POS"]["lossless"]["mae"]
    if "sama" in S:
        al = S["sama"]["lossless"]["all"]
        V.update(pilot_mae_sama=al["mae_sama_all"], pilot_abstention=al["abstention"], pilot_coverage=al["coverage_shown"])
    if a.ita_model and Path(a.ita_model).exists():
        V["ita_model_mae"] = json.loads(Path(a.ita_model).read_text())["meta"]["cv"]["mae"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(V, indent=1, default=float))
    print(json.dumps(V, indent=1, default=float))


if __name__ == "__main__":
    main()

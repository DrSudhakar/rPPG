"""SUPERSEDED (11-09) by `python -m sama train` and `python -m sama pilot`; kept for the record.
EXPLORATORY, not sap-v1 (ISSUES I-35, DECISIONS D-26): run the trained SAMA method (deploy/model/,
fit entirely on SYNTHETIC SAMA-Sim data) on REAL-PUBLIC recordings under data/raw/ (UBFC-rPPG, PURE).

This answers only "does the trained network + conformal gate behave sensibly on a real face" (Q-A
feasibility). It is NOT evidence on SAMA's skin-tone-equity claim and is NOT a confirmatory analysis:
- step 2 (colour-chart calibration) cannot run: neither dataset filmed a chart. Uncalibrated, same as
  the web demo path (D-25) -- a materially different pipeline from the one evaluated in the report.
- ITA deg is the same uncalibrated, no-white-reference estimate used in analysis/realdata.py (I-03).

Output: results/tables/sama_realdata.csv (one row per 1 s update per recording) and a per-dataset
summary (MAE on shown windows, abstention rate, coverage) printed and written to
results/tables/sama_realdata_summary.csv, for comparison with realdata_replication.csv's POS row.
Never run on real data in this build -- see tests/test_loaders.py for the only testing performed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sama import ita as ITA, loaders, model_io  # noqa: E402
from sama.calib import ToneCurve  # noqa: E402
from sama.streaming import StreamingEstimator  # noqa: E402


def approx_ita(patches: np.ndarray) -> float:
    """Same no-white-reference approximation as analysis/realdata.py (I-03): cheek patches only,
    assumed sRGB gamma, no chart to correct for the camera's actual tone curve."""
    srgb = ToneCurve(gamma=2.2, scale=255.0)
    cheek = srgb.linearise(patches[:, 12:].mean((0, 1))) / 255.0
    return float(ITA.ita_from_camera(cheek))


def run_recording(bundle, rec) -> list:
    ita_deg = approx_ita(rec.patches)
    est = StreamingEstimator(gate=bundle["gate"], sigma_model=bundle["sigma"], params=bundle["params"],
                              onnx_path=None, ita=ita_deg)   # tone=noise=None: uncalibrated, no chart (D-25)
    out = []
    for i in range(rec.patches.shape[0]):
        t = i / rec.fps
        r = est.push(t, rec.patches[i], None)
        if r is None or r.reason == "WARMING_UP":
            continue
        beats = rec.beats[(rec.beats >= t - 30) & (rec.beats < t)]
        hr_true = float(60 / np.mean(np.diff(beats))) if beats.size > 2 else np.nan
        out.append({"dataset": rec.source, "recording": rec.name, "t": t, "ita_est_deg": ita_deg,
                    **r.to_dict(), "hr_true": hr_true})
    return out


def main(root=ROOT / "data/raw"):
    bundle = model_io.load_bundle(ROOT / "deploy/model")
    if bundle is None:
        sys.exit("no trained model bundle at deploy/model/ -- run `make method` first")
    recs = loaders.discover(root)
    if not recs:
        sys.exit(f"no UBFC-rPPG or PURE recordings found under {root} (see data/DATA_INVENTORY.md)")
    rows = []
    for kind, path in recs:
        rows += run_recording(bundle, loaders.load(kind, path))
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results/tables/sama_realdata.csv", index=False)
    summ = []
    for g, sub in d.groupby("dataset"):
        shown = sub[sub.shown]
        err = (shown.hr - shown.hr_true).abs().dropna()
        cov = ((shown.hr_true >= shown.lo) & (shown.hr_true <= shown.hi)).mean() if len(shown) else np.nan
        summ.append({"dataset": g, "n_updates": len(sub), "n_shown": len(shown),
                     "abstention_rate": 1 - len(shown) / max(len(sub), 1),
                     "mae_shown_bpm": float(err.mean()) if len(err) else np.nan,
                     "coverage_shown": float(cov)})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(ROOT / "results/tables/sama_realdata_summary.csv", index=False)
    print("EXPLORATORY (I-35): SAMA method, uncalibrated (no colour chart), on REAL-PUBLIC data.")
    print("Not sap-v1. Not evidence on the skin-tone-equity claim. Compare against the POS row in")
    print("realdata_replication.csv for the same recordings.\n")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))

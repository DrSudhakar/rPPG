"""C9: export the deployment network to ONNX (fp32, fp16-weights, int8-weights) and test parity on
SYNTHETIC windows (cohort participants, including the final model's training participants: a numerical parity check, not an accuracy test, R7-05): largest HR difference vs numpy must be < 0.1 bpm; int8 loss per stratum."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sama import export, model_io, net  # noqa: E402
from sama.sama_method import hr_from_net_output  # noqa: E402


def main():
    b = model_io.load_bundle(ROOT / "deploy/model")
    P = b["params"]
    assert net.n_params(P) < 500_000, "network over the 0.5M-parameter budget"
    paths = export.save_all(P, ROOT / "deploy/onnx", meta=json.dumps(b["meta"]))
    files = sorted((ROOT / "results/cache/cohort_A_s7").glob("*.pkl"))
    X, C, ref, strata = [], [], [], []
    for f in files[::4]:  # every 4th participant, lossless, both lights
        d = pickle.load(open(f, "rb"))
        for (light, blk), (wf, _) in d["lossless"].items():
            if blk in ("rest", "mock_teleconsultation"):
                X.append(wf.X); C.append(wf.cond); ref.append(wf.hr_ref); strata += [d["part"]["stratum"]] * len(wf.hr_ref)
    X, C, ref, strata = np.concatenate(X), np.concatenate(C), np.concatenate(ref), np.array(strata)
    h_np = hr_from_net_output(net.predict(P, X, C))[0]
    rows = []
    for w, pth in paths.items():
        y = np.concatenate([export.run_onnx(pth, X[i:i + 256], C[i:i + 256]) for i in range(0, len(X), 256)])
        h = hr_from_net_output(y)[0]
        diff = np.abs(h - h_np)
        r = {"format": w, "bytes": Path(pth).stat().st_size, "max_hr_diff_bpm": float(np.nanmax(diff)),
             "share_windows_diff_gt_0.1": float(np.nanmean(diff > 0.1)), "windows": len(h)}
        for s in ("lighter", "middle", "darker"):
            m = (strata == s) & np.isfinite(ref)
            r[f"mae_{s}"] = float(np.nanmean(np.abs(h[m] - ref[m])))
            r[f"mae_{s}_numpy"] = float(np.nanmean(np.abs(h_np[m] - ref[m])))
        r["pass_parity_0.1bpm"] = r["max_hr_diff_bpm"] < 0.1
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results/tables/export_parity.csv", index=False)
    (ROOT / "deploy/tflite/NOT_PRODUCED.md").write_text(
        "TFLite export not produced: no TensorFlow / onnx2tf in this environment (ISSUES I-23, DECISIONS D-07).\n"
        "Steps: pip install onnx onnx2tf tensorflow; onnx2tf -i deploy/onnx/sama_net_fp32.onnx -o deploy/tflite/;\n"
        "then re-run analysis/export_models.py with a TFLite interpreter branch and require max HR difference < 0.1 bpm.\n")
    print(d.to_string())


if __name__ == "__main__":
    main()

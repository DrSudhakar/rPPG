"""C4 substitute: no REAL-PUBLIC or SEMI-SYNTHETIC data exist in this run (I-22), so Figure 6 cannot
be rebuilt from public data. This script exercises the PI's own scripts/pilot_figure.py end to end
on the virtual cohort (SYNTHETIC), writes pilot_hr.csv in the exact pilot schema, and produces
figure6_SYNTHETIC_NOT_FOR_SUBMISSION.png. Use it to check the pilot workflow, never as evidence.
"""
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "proposal_inserts"


def main():
    rows = []
    for f in sorted((ROOT / "results/cache/cohort_A_s7").glob("*.pkl"))[::3]:   # 40 virtual participants
        d = pickle.load(open(f, "rb"))
        pid, ita = d["part"]["pid"], d["part"]["ita"]
        for (light, blk), (wf, bl) in d["lossless"].items():
            if light != "standard" or blk not in ("rest", "mock_teleconsultation"):
                continue
            for m in ("POS", "CHROM"):
                rows += [{"participant_id": pid, "ita_deg": round(ita, 1), "method": m, "bitrate_kbps": 0, "hr_ref_bpm": round(r, 2),
                          "hr_cam_bpm": round(c, 2)} for r, c in zip(wf.hr_ref, bl[m]) if np.isfinite(r) and np.isfinite(c)]
        for key, (wf, bl) in d["shift"].items():
            dev, light, blk, codec = key
            if dev == "reference30" and codec in ("500", "250"):
                rows += [{"participant_id": pid, "ita_deg": round(ita, 1), "method": "POS", "bitrate_kbps": int(codec), "hr_ref_bpm": round(r, 2),
                          "hr_cam_bpm": round(c, 2)} for r, c in zip(wf.hr_ref, bl["POS"]) if np.isfinite(r) and np.isfinite(c)]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pilot_hr_SYNTHETIC.csv", index=False)
    script = ROOT / "scripts/pilot_figure.py"
    subprocess.run([sys.executable, str(script), str(OUT / "pilot_hr_SYNTHETIC.csv"), "--method", "POS", "--out",
                    str(OUT / "figure6_SYNTHETIC_NOT_FOR_SUBMISSION.png"), "--ref", "SIMULATED ECG (SAMA-Sim) - SYNTHETIC, NOT FOR SUBMISSION"], check=True)
    print(len(df), "rows")


if __name__ == "__main__":
    main()

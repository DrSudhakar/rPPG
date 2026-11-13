"""`make realdata`: run the six unsupervised methods on every recording under data/raw/ (UBFC-rPPG and
PURE layouts), compare with the contact-PPG reference in 30 s / 15 s windows, and write
results/tables/realdata_replication.csv (per dataset x method MAE, for comparison with the
rPPG-Toolbox paper) and proposal_inserts/pilot_hr_REALPUBLIC.csv (pilot schema; ITA° approximate,
from the cheek patches without a white reference). Label: REAL-PUBLIC. Never run in this build.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sama import ita, loaders, methods, pipeline  # noqa: E402
from sama.calib import ToneCurve  # noqa: E402


def main(root=ROOT / "data/raw"):
    recs = loaders.discover(root)
    if not recs:
        sys.exit(f"no UBFC-rPPG or PURE recordings found under {root} (see data/DATA_INVENTORY.md)")
    rows, pilot = [], []
    srgb = ToneCurve(gamma=2.2, scale=255.0)
    for kind, path in recs:
        r = loaders.load(kind, path)
        face = r.patches.mean(1)
        cheek = srgb.linearise(r.patches[:, 12:].mean((0, 1))) / 255.0
        ita_est = ita.ita_from_camera(cheek)  # approximate: no white patch (I-03)
        res = pipeline.run_methods(face, r.fps, r.beats, None)
        for m, w in res.items():
            for x in w:
                rows.append({"dataset": r.source, "recording": r.name, "method": m, "hr_ref": x["hr_ref"], "hr_cam": x["hr_cam"]})
                pilot.append({"participant_id": f"{r.source}:{r.name}", "ita_deg": round(ita_est, 1), "method": m, "bitrate_kbps": 0,
                              "hr_ref_bpm": x["hr_ref"], "hr_cam_bpm": x["hr_cam"]})
    d = pd.DataFrame(rows)
    d["err"] = (d.hr_cam - d.hr_ref).abs()
    tab = d.dropna().groupby(["dataset", "method"]).err.mean().unstack()
    tab.to_csv(ROOT / "results/tables/realdata_replication.csv")
    pd.DataFrame(pilot).dropna().to_csv(ROOT / "proposal_inserts/pilot_hr_REALPUBLIC.csv", index=False)
    print("REAL-PUBLIC window MAE (bpm):\n", tab.round(2))
    print("rPPG-Toolbox (video-level, LITERATURE): POS 3.67 PURE / 4.00 UBFC-rPPG; CHROM 5.77 / 3.98. Windowed values are not identical by design.")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))

"""R7-02: matched-abstention comparison at participant level (the H4 averaging unit), with a
participant bootstrap interval of the difference. SYNTHETIC."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(11)


def main():
    d = pd.read_csv(ROOT / "results/cache/c7_test_predictions.csv.gz")
    d = d[np.isfinite(d.hr_ref)]
    rows = []
    for s, q in d.groupby("stratum"):
        a = 1 - q.SAMA_shown.mean()
        thr = np.quantile(q["POS+gate_sigma"], 1 - a) if a > 0 else np.inf
        q = q.assign(pos_keep=q["POS+gate_sigma"] <= thr, e_s=(q.SAMA_hr - q.hr_ref).abs(), e_p=(q.hr_POS - q.hr_ref).abs())
        ps = q[q.SAMA_shown.astype(bool)].groupby("pid").e_s.mean()
        pp = q[q.pos_keep].groupby("pid").e_p.mean()
        both = ps.index.intersection(pp.index)
        diff = (pp[both] - ps[both]).to_numpy()
        boots = [RNG.choice(diff, diff.size).mean() for _ in range(2000)]
        rows.append({"stratum": s, "abstention": a, "SAMA_retained_mae_participant": ps.mean(), "POS_matched_mae_participant": pp.mean(),
                     "SAMA_retained_mae_window": q.e_s[q.SAMA_shown.astype(bool)].mean(), "POS_matched_mae_window": q.e_p[q.pos_keep].mean(),
                     "diff_POS_minus_SAMA_paired": diff.mean(), "lo95": np.quantile(boots, 0.025), "hi95": np.quantile(boots, 0.975),
                     "n_participants": both.size})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/tables/c7_matched_abstention.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()

"""Calibrate SAMA-Sim's two free signal parameters (rel_amp_green, pix_per_patch) so that POS
heart-rate MAE on lossless reference video (both lights, seated blocks) follows the proposal's
H1 anchor: 2.5 bpm at ITA 50 rising linearly to 4.5 bpm at ITA -25 (proposal s.6, from Chari et
al. 2020). Output: results/tables/sim_calibration.csv and configs/sim_calibrated.yaml.

This makes the simulator's skin-tone gradient ASSUMPTION-DRIVEN by construction: it is matched
to the proposal's own assumption so that the rehearsal tests the *design*, not nature.
"""
from __future__ import annotations

import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sama import pipeline as P  # noqa: E402
from sim import samasim as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ITAS = (45.0, -20.0)
TARGET = {ita: 2.5 + 2.0 * (50 - ita) / 75 for ita in ITAS}
BLOCKS = ("rest", "mock_teleconsultation", "mental_arithmetic", "high_arousal_clips")
N_PART = 16


def one(args):
    amp, pix, ita, i = args
    p = S.with_params(rel_amp_green=amp, pix_per_patch=pix)
    part = S.make_participant(f"c{i}", S.stratum_of(ita), np.random.default_rng(5000 + i), ita=ita)
    errs = []
    for blk in BLOCKS:
        for light in ("standard", "dim"):
            rec = S.simulate_block(part, blk, light, "reference30", p, dur_s=90.0)
            rows = P.run_methods(rec.rgb_face, rec.fps, rec.beats, rec.beats_valid, methods=("POS",))["POS"]
            errs += [abs(r["hr_cam"] - r["hr_ref"]) for r in rows if np.isfinite(r["hr_ref"])]
    return amp, pix, ita, float(np.mean(errs))


def main(grid_amp=(0.0015, 0.002, 0.0025), grid_pix=(60.0, 100.0, 160.0)):
    jobs = [(a, p, ita, i) for a, p in itertools.product(grid_amp, grid_pix) for ita in ITAS for i in range(N_PART)]
    with Pool(2) as pool:
        res = pool.map(one, jobs)
    rows = {}
    for a, p, ita, m in res:
        rows.setdefault((a, p, ita), []).append(m)
    table = []
    for (a, p, ita), v in sorted(rows.items()):
        table.append({"rel_amp_green": a, "pix_per_patch": p, "ita": ita, "pos_mae": float(np.mean(v)),
                      "sd_between": float(np.std(v, ddof=1)), "target": TARGET[ita]})
    best, score = None, 1e9
    for a, p in itertools.product(grid_amp, grid_pix):
        sc = sum((r["pos_mae"] - r["target"]) ** 2 for r in table if r["rel_amp_green"] == a and r["pix_per_patch"] == p)
        if sc < score:
            best, score = (a, p), sc
    out = ROOT / "results/tables/sim_calibration.csv"
    import csv
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    (ROOT / "configs/sim_calibrated.json").write_text(json.dumps(
        {"rel_amp_green": best[0], "pix_per_patch": best[1], "score": score, "targets": TARGET,
         "provenance": "SYNTHETIC; calibrated to proposal H1 assumption (LITERATURE anchor: Chari et al. 2020 via proposal s.6)"}, indent=1))
    print("best", best, "score", round(score, 3))
    for r in table:
        if (r["rel_amp_green"], r["pix_per_patch"]) == best:
            print(r)


if __name__ == "__main__":
    main()

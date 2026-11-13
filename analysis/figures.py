"""Remaining report figures from logged outputs (each caption carries its provenance label)."""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.figstyle import COL, setup  # noqa: E402

T, F = ROOT / "results/tables", ROOT / "results/figures"
STRATA = ("lighter", "middle", "darker")


def tag(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=8.5, color="#555")


def budget():
    d = pd.read_csv(T / "budget_by_stratum.csv")
    comps = ["dim", "webcam", "phoneA", "codec500", "codec250"]
    names = {"dim": "dim light (100 lux)", "webcam": "webcam", "phoneA": "phone, auto-exposure", "codec500": "H.264 500 kbps", "codec250": "H.264 250 kbps"}
    fig, axes = plt.subplots(1, len(comps), figsize=(12, 3.6), sharey=True, constrained_layout=True)
    for ax, c in zip(axes, comps):
        y = d[c].to_numpy()
        err = np.vstack([y - d[f"{c}_lo"], d[f"{c}_hi"] - y])
        tot = d["base"].to_numpy() + y
        ax.bar(STRATA, tot, color=[COL[s] for s in STRATA], yerr=err, capsize=3, label="with component")
        ax.scatter(STRATA, d["base"], marker="_", s=900, color="k", zorder=4, label="lossless reference, good light")
        ax.set_title(names[c], fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Window error, bpm (back-transformed)")
    axes[0].legend(frameon=False, fontsize=8)
    tag(fig, "SYNTHETIC (virtual SAMA-120, POS) · mixed model, participant-bootstrap 95% CI · codec rows use surrogate v0 (D-23)")
    fig.savefig(F / "fig_error_budget.png", dpi=180)


def coverage():
    c = pd.read_csv(T / "c7_ablation.csv").set_index("variant")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    x = np.arange(3)
    for i, v in enumerate(["SAMA", "POS+gate"]):
        ax[0].bar(x + i * 0.38, [c.loc[v, f"cov_all_{s}"] for s in STRATA], 0.36, label=f"{v}: all windows", color=[COL["SAMA"], COL["POS"]][i])
        ax[0].scatter(x + i * 0.38, [c.loc[v, f"cov_shown_{s}"] for s in STRATA], color="k", zorder=3, s=14, label="shown windows only" if i == 0 else None)
        ax[1].bar(x + i * 0.38, [c.loc[v, f"abst_{s}"] for s in STRATA], 0.36, color=[COL["SAMA"], COL["POS"]][i])
        ax[2].bar(x + i * 0.38, [c.loc[v, f"yield_{s}"] for s in STRATA], 0.36, color=[COL["SAMA"], COL["POS"]][i])
    ax[0].axhline(0.88, ls="--", color="#888")
    ax[0].axhline(0.90, ls=":", color="#888")
    ax[0].set_ylim(0.5, 1.0)
    for a, t in zip(ax, ("A  Coverage of 90% intervals", "B  Abstention rate", "C  Useful yield (shown, within ±5 bpm)")):
        a.set_xticks(x + 0.19, STRATA)
        a.set_title(t)
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    tag(fig, "SYNTHETIC (virtual SAMA-120, 5-fold nested CV, 50/50 calibration split per amendment A1)")
    fig.savefig(F / "fig_coverage_abstention.png", dpi=180)


def ablation():
    c = pd.read_csv(T / "c7_ablation.csv")
    keep = ["SAMA", "-FiLM", "-groupDRO", "-CNN", "POS+gate", "best baseline", "POS", "CHROM", "GREEN"]
    c = c[c.variant.isin(keep)].set_index("variant").loc[[k for k in keep if k in set(c.variant)]]
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    x = np.arange(len(c))
    for i, s in enumerate(STRATA):
        ax.bar(x + (i - 1) * 0.27, c[f"mae_{s}"], 0.26, color=COL[s], label=s)
    ax.set_xticks(x, c.index, rotation=20)
    ax.set_ylabel("MAE, all windows (bpm)")
    ax.set_title("Ablation: mean of participant MAE by stratum (abstained windows scored with the point estimate)")
    ax.legend(frameon=False)
    tag(fig, "SYNTHETIC · neural baselines absent (I-24) · step 2 and step 3 not separable here (−CNN keeps both)")
    fig.savefig(F / "fig_ablation.png", dpi=180)


def h5():
    d = pd.read_csv(T / "c8_h5_icc.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    x = np.arange(len(d))
    ax.errorbar(x - 0.15, d.icc_overall, yerr=[d.icc_overall - d.lo95, d.hi95 - d.icc_overall], fmt="o", color="k", label="overall (95% CI)")
    for i, s in enumerate(STRATA):
        ax.scatter(x + 0.05 * (i + 1), d[f"icc_{s}"], color=COL[s], label=s, zorder=3)
    ax.axhline(0.75, ls="--", color="#888")
    ax.axhline(0.60, ls=":", color="#888")
    ax.set_xticks(x, ["60 fps\nlossless", "30 fps\nsame sensor", "30 fps\n720p", "30 fps\n500 kbps"])
    ax.set_ylabel("ICC(2,1), RMSSD reactivity")
    ax.set_title("H5s: camera vs ECG agreement for ln RMSSD reactivity")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    tag(fig, "SYNTHETIC · ASSUMPTION-DRIVEN (HRV generator and pulse model)")
    fig.savefig(F / "fig_h5_icc.png", dpi=180)


def power():
    pr = json.load(open(T / "design_checks.json"))
    rows = pr["power_rerun"]
    c6 = json.load(open(ROOT / "results/c6_summary.json"))
    c7 = json.load(open(ROOT / "results/c7_summary.json"))
    labels = ["H1", "H2", "H3", "H4", "H5"]
    claimed = [0.87, 0.80, 0.90, 0.99, 0.90]
    rerun = [rows[0]["H1"], rows[0]["H2"], rows[0]["H3"], rows[0]["H4_any_reduction"], rows[0]["H5_fisher_one"]]
    criterion = [rows[0]["H1"], np.nan, rows[0]["H3"], pr["h4_rules"][1]["point_estimate_ge30pct"], rows[0]["H5_joint_f"]]
    reh = [next(r["power"] for r in c6["h1"] if r["method"].startswith("parametric")), np.nan, np.nan, c7["h4"]["p_all_five"],
           c7["h5"]["reference"]["p_pass_36"]]
    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    x = np.arange(5)
    for i, (v, lab, col) in enumerate([(claimed, "proposal table", "#B8C0C8"), (rerun, "independent rerun, same assumptions", "#7F93CF"),
                                       (criterion, "rerun, full success criterion", "#2F4BA0"), (reh, "rehearsal on SAMA-Sim", COL["SAMA"])]):
        ax.bar(x + (i - 1.5) * 0.2, v, 0.19, label=lab, color=col)
        for xx, vv in zip(x + (i - 1.5) * 0.2, v):
            if np.isfinite(vv):
                ax.text(xx, vv + 0.01, f"{vv:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.8, ls="--", color="#888")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Power / probability of success at 36 per stratum")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    ax.set_title("The proposal's power table against what its own success criteria need")
    tag(fig, "ASSUMPTION-DRIVEN (proposal assumptions) and SYNTHETIC (rehearsal); H2 rehearsal in fig_h2_discriminability")
    fig.savefig(F / "fig_power_vs_proposal.png", dpi=180)


def sim_vs_literature():
    g = pd.read_csv(T / "c6_realised_gradient.csv")
    codec = pd.read_csv(T / "codec_real_gain.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for _, r in g.iterrows():
        ax[0].plot(STRATA, [r[f"median_{s}"] for s in STRATA], marker="o", label=f"World {r.world} (median)")
    ax[0].plot(STRATA, [2.5 + 2 * (50 - m) / 75 for m in (39, 19, -7.5)], ls="--", color="#888", label="proposal assumption (mean)")
    ax[0].set_ylabel("Participant POS MAE (bpm)")
    ax[0].set_title("A  Simulator vs the LITERATURE anchor")
    ax[0].legend(frameon=False, fontsize=8)
    for name, q in codec.groupby("level"):
        q = q.sort_values("amp_dn")
        ax[1].plot(q.amp_dn, q.gain, marker=".", lw=1.2, label=name)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Pulse amplitude in tile (DN)")
    ax[1].set_ylabel("Share kept after encoding")
    ax[1].set_title("B  Real ffmpeg on rendered tiles")
    ax[1].legend(frameon=False, fontsize=7, ncol=2)
    tag(fig, "A: SYNTHETIC vs LITERATURE (proposal s.6) · B: SYNTHETIC rendering, real codec")
    fig.savefig(F / "fig_sim_vs_reality.png", dpi=180)


if __name__ == "__main__":
    setup()
    for f in (budget, coverage, ablation, h5, power, sim_vs_literature):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:  # noqa: BLE001
            print("FAILED", f.__name__, e)

"""Make the SAMA pilot figure (Figure 6 of the proposal) from your pilot results.

Usage
  python pilot_figure.py pilot_hr.csv               -> pilot_figure.png
  python pilot_figure.py --template                 -> pilot_figure_template.png (empty axes)
  python pilot_figure.py pilot_hr.csv --method POS --out fig6.png

Input CSV: one row per analysis window (e.g. 30 s) per participant, method and bitrate.
  participant_id  any id (P01 ...)
  ita_deg         participant's ITA in degrees (colorimeter or calibrated image)
  method          e.g. POS, CHROM, TS-CAN
  bitrate_kbps    0 for the uncompressed/lossless video, otherwise the encoded bitrate
                  (use one codec, e.g. H.264, for the pilot; rows are grouped by bitrate only)
  hr_ref_bpm      reference heart rate in the window (ECG or Polar H10)
  hr_cam_bpm      camera heart rate in the window

Panels
  A  per-participant mean absolute error (lossless video) against ITA, one line per method
  B  mean absolute error by bitrate for each skin-tone stratum (chosen method)
  C  Bland-Altman plot, camera minus reference (chosen method, lossless video)

Strata follow Del Bino & Bernerd (2013) ITA categories, merged into three:
  lighter: ITA > 28, middle: 10 < ITA <= 28, darker: ITA <= 10.
Requires numpy, pandas, matplotlib.
"""
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATA = [("Lighter (ITA > 28°)", 28, 999, "#D9A066"),
          ("Middle (10–28°)", 10, 28, "#A0612B"),
          ("Darker (ITA ≤ 10°)", -999, 10, "#4F2B12")]
METHOD_COLOURS = ["#2F4BA0", "#0B6E6E", "#6B7480", "#A64B00"]
INK, MUTED, GRID = "#14202B", "#46535F", "#D5DBE3"


def stratum(ita):
    for name, lo, hi, _ in STRATA:
        if lo < ita <= hi:
            return name
    return STRATA[0][0]


def style():
    for f in ("Inter", "DejaVu Sans"):
        if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 12,
                         "axes.titlelocation": "left"})


def frame(axes):
    a, b, c = axes
    a.set_title("A  Error rises as skin tone darkens?")
    a.set_xlabel("ITA (°)  ←  lighter      darker  →")
    a.set_ylabel("Heart-rate error, MAE (bpm)")
    a.invert_xaxis()
    b.set_title("B  Does compression widen the gap?")
    b.set_xlabel("Video bitrate (kbps)")
    b.set_ylabel("Heart-rate error, MAE (bpm)")
    c.set_title("C  Agreement with the reference")
    c.set_xlabel("Mean of camera and reference HR (bpm)")
    c.set_ylabel("Camera − reference (bpm)")
    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def template(out):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    frame(axes)
    axes[0].set_xlim(55, -25); axes[0].set_ylim(0, 12)
    axes[1].set_xlim(-0.3, 4.3); axes[1].set_xticks(range(5)); axes[1].set_xticklabels(["250", "500", "1000", "2000", "lossless"])
    axes[1].set_ylim(0, 12)
    axes[2].set_xlim(55, 130); axes[2].set_ylim(-20, 20)
    for ax in axes:
        ax.text(0.5, 0.5, "PILOT DATA\nGOES HERE", transform=ax.transAxes, ha="center", va="center",
                fontsize=17, color="#B8C0C8", fontweight="bold")
    for name, _, _, col in STRATA:
        axes[1].plot([], [], color=col, linewidth=2.5, label=name)
    axes[1].legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.suptitle("Figure 6 (template). Replace with the output of pilot_figure.py run on your pilot CSV.",
                 x=0.01, ha="left", fontsize=11, color=MUTED)
    fig.savefig(out, dpi=200)
    print("wrote", out)


def make(csv, method, out, ref):
    df = pd.read_csv(csv)
    need = {"participant_id", "ita_deg", "method", "bitrate_kbps", "hr_ref_bpm", "hr_cam_bpm"}
    miss = need - set(df.columns)
    if miss:
        sys.exit(f"missing columns: {sorted(miss)}")
    df["abs_err"] = (df.hr_cam_bpm - df.hr_ref_bpm).abs()
    df["stratum"] = df.ita_deg.map(stratum)
    methods = list(dict.fromkeys(df.method))
    if method not in methods:
        method = methods[0]
    style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    frame(axes)

    # A: per-participant MAE on lossless video vs ITA, per method, with least-squares line
    lossless = df[df.bitrate_kbps == 0]
    per = lossless.groupby(["method", "participant_id"]).agg(ita=("ita_deg", "first"), mae=("abs_err", "mean")).reset_index()
    for i, m in enumerate(methods):
        d = per[per.method == m]
        col = METHOD_COLOURS[i % len(METHOD_COLOURS)]
        axes[0].scatter(d.ita, d.mae, s=34, color=col, alpha=0.85, label=f"{m} (n={len(d)})", zorder=3)
        if len(d) >= 3:
            k, c0 = np.polyfit(d.ita, d.mae, 1)
            xs = np.linspace(d.ita.min(), d.ita.max(), 50)
            axes[0].plot(xs, k * xs + c0, color=col, linewidth=1.8)
    axes[0].legend(frameon=False, fontsize=9.5)

    # B: MAE by bitrate and stratum for the chosen method (participant means, then stratum mean ± SE)
    d = df[df.method == method]
    rates = sorted(r for r in d.bitrate_kbps.unique() if r > 0)
    order = rates + [0]
    xpos = {r: i for i, r in enumerate(order)}
    pp = d.groupby(["stratum", "participant_id", "bitrate_kbps"]).abs_err.mean().reset_index()
    for name, _, _, col in STRATA:
        s = pp[pp.stratum == name]
        if s.empty:
            continue
        g = s.groupby("bitrate_kbps").abs_err.agg(["mean", "sem", "count"]).reindex(order)
        x = [xpos[r] for r in order]
        axes[1].errorbar(x, g["mean"], yerr=g["sem"].fillna(0), color=col, linewidth=2.2, marker="o",
                         capsize=3, label=f"{name}, n={int(g['count'].max())}")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels([str(int(r)) for r in rates] + ["lossless"])
    axes[1].legend(frameon=False, fontsize=9.5)
    axes[1].set_title(f"B  Does compression widen the gap? ({method})")

    # C: Bland-Altman, chosen method, lossless video
    b = lossless[lossless.method == method]
    mean = (b.hr_cam_bpm + b.hr_ref_bpm) / 2
    diff = b.hr_cam_bpm - b.hr_ref_bpm
    for name, _, _, col in STRATA:
        sel = b.stratum == name
        axes[2].scatter(mean[sel], diff[sel], s=16, color=col, alpha=0.75, label=name)
    bias, sd = diff.mean(), diff.std(ddof=1)
    for y, lab in ((bias, f"bias {bias:+.1f}"), (bias + 1.96 * sd, f"+1.96 SD {bias + 1.96 * sd:+.1f}"),
                   (bias - 1.96 * sd, f"−1.96 SD {bias - 1.96 * sd:+.1f}")):
        axes[2].axhline(y, color=MUTED, linewidth=1.2, linestyle="-" if lab.startswith("bias") else "--")
        axes[2].text(axes[2].get_xlim()[1], y, " " + lab, va="center", fontsize=9, color=MUTED, clip_on=False)
    axes[2].set_title(f"C  Agreement with the reference ({method}, lossless)")
    n_people = df.participant_id.nunique()
    fig.suptitle(f"Pilot, {n_people} participants. Reference: {ref}.",
                 x=0.01, ha="left", fontsize=11, color=MUTED)
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--method", default="POS")
    ap.add_argument("--out")
    ap.add_argument("--ref", default="Polar H10 chest-strap ECG")
    a = ap.parse_args()
    if a.template or not a.csv:
        template(a.out or "pilot_figure_template.png")
    else:
        make(a.csv, a.method, a.out or "pilot_figure.png", a.ref)

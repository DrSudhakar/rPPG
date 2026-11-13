"""Figures for the rewritten 36-month proposal: work packages (Figure 3) and timeline (Figure 8)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "proposal/figures"
INK, MUTED = "#14202B", "#46535F"
WP = [("WP1 · MEASURE", "SAMA-150 dataset", "Months 1–20", "#0B6E6E",
       ["Set-up: ethics, OSF pre-registration,\nrig, synchronisation test (M1–3)",
        "Protocol pilot, 10 sessions (M3–4)",
        "Record 150 adults, 50 per ITA° stratum,\n25 women + 25 men each (M4–16)",
        "Ground truth: 1 kHz ECG; HR, RMSSD,\nSDNN; codec ladder 2,000–250 kbps"],
       "D1 data release and descriptor (M20)"),
      ("WP2 · EXPLAIN", "Bias audit and error budget", "Months 8–22", "#2F4BA0",
       ["Benchmark 10 methods, pretrained\nand fine-tuned by participant-wise CV",
        "Skin-tone gradient in HR and\nHRV error  [H1]",
        "Error budget: optics, light, device,\ncodec; does compression widen\nthe gap?  [H2]",
        "Interim audit of first 45 and\npower re-check (M10)"],
       "D2 audit paper, open error tables (M22)"),
      ("WP3 · FIX AND GUARANTEE", "The SAMA method", "Months 10–30", "#A64B00",
       ["Adaptive exposure vs auto-exposure\non one phone model  [H3]",
        "Calibrated, noise-weighted extraction;\nablations of every step",
        "Reliability gate: 90 % intervals, cross-\nconformal; audited per stratum  [H4]",
        "Measure-before-you-compress\nanalysis per stratum"],
       "D3 method paper, open model\nand code (M30)"),
      ("WP4 · VALIDATE AND RELEASE", "HRV, stress and tools", "Months 18–36", "#14202B",
       ["HRV agreement with ECG (SAMA),\nstrata pooled and per stratum  [H5]",
        "Exploratory: stress and arousal,\ncamera vs ECG features  [E1]",
        "On-device demonstrator: latency,\nbattery, abstention on 3 phones",
        "Developer workshop (M34)"],
       "D4 HRV paper (M32), toolkit v1.0,\nfinal report (M36)")]


def work_packages():
    fig = plt.figure(figsize=(15, 8.0))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.86])
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.06, 0.92)
    ax.axis("off")
    fig.text(0.015, 0.955, "Four work packages over 36 months: measure, explain, fix and guarantee, validate and release",
             fontsize=16, fontweight="bold", color=INK)
    fig.text(0.015, 0.915, "Hypotheses are pre-registered on OSF before the first participant. Each package closes with a deliverable that stands on its own.",
             fontsize=11, color=MUTED)
    for i, (tag, title, months, col, items, deliv) in enumerate(WP):
        x = i + 0.04
        ax.add_patch(FancyBboxPatch((x, 0.74), 0.9, 0.16, boxstyle="round,pad=0.005,rounding_size=0.02", fc=col, ec=col))
        ax.text(x + 0.03, 0.865, tag, color="white", fontsize=9.5, fontweight="bold")
        ax.text(x + 0.03, 0.81, title, color="white", fontsize=12.5, fontweight="bold")
        ax.text(x + 0.03, 0.765, months, color="white", fontsize=10)
        for j, it in enumerate(items):
            y = 0.56 - j * 0.145
            ax.add_patch(FancyBboxPatch((x, y), 0.9, 0.13, boxstyle="round,pad=0.005,rounding_size=0.015", fc="white", ec="#C9D1DA"))
            ax.text(x + 0.03, y + 0.065, it, fontsize=8.8, color=INK, va="center")
        ax.add_patch(FancyBboxPatch((x, -0.035), 0.9, 0.08, boxstyle="round,pad=0.005,rounding_size=0.015", fc="#EEF1F5", ec="#EEF1F5"))
        ax.text(x + 0.03, 0.005, deliv, fontsize=8.8, color=INK, va="center", fontweight="bold")
        if i < 3:
            ax.text(i + 0.97, 0.45, "›", fontsize=26, color=MUTED, va="center")
    fig.savefig(OUT / "Fig3_work_packages_36m.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def timeline():
    rows = [("Ethics, OSF, procurement, JRF, rig and sync test", 1, 3, "#7A8691"),
            ("Protocol pilot (10 sessions)", 3, 4, "#0B6E6E"),
            ("WP1 recording, 150 participants (~3 a week)", 4, 16, "#0B6E6E"),
            ("WP1 curation, release, descriptor", 14, 20, "#0B6E6E"),
            ("WP2 benchmark and error budget (H1, H2)", 8, 22, "#2F4BA0"),
            ("WP2 interim audit + power re-check", 10, 10, "#2F4BA0"),
            ("WP3 adaptive exposure (H3)", 10, 18, "#A64B00"),
            ("WP3 extraction, gate, ablations (H4)", 12, 30, "#A64B00"),
            ("WP4 HRV and stress (H5, E1)", 18, 32, "#14202B"),
            ("WP4 on-device demonstrator", 24, 34, "#14202B"),
            ("Workshop, toolkit v1.0, final report", 33, 36, "#14202B")]
    miles = [(3, "D1.1"), (10, "D1.2/D2.1"), (16, "D1.3"), (20, "D1"), (22, "D2"), (30, "D3"), (32, "D4.1"), (34, "D4.2"), (36, "D4.3")]
    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    for k, (lab, a, b, c) in enumerate(rows[::-1]):
        ax.barh(k, b - a + 1, left=a - 0.5, color=c, height=0.6)
        ax.text(0, k, lab + "  ", ha="right", va="center", fontsize=9.5, color=INK)
    for m, lab in miles:
        ax.axvline(m + 0.5, color="#C9D1DA", lw=0.8, zorder=0)
        ax.text(m + 0.5, len(rows) - 0.35, lab, rotation=90, fontsize=8, color=MUTED, va="bottom", ha="center")
    ax.set_ylim(-0.6, len(rows) + 0.9)
    ax.set_xlim(0.5, 36.5)
    ax.set_xticks(range(1, 37, 1))
    ax.set_xticklabels([str(i) if i % 3 == 0 or i == 1 else "" for i in range(1, 37)])
    ax.set_yticks([])
    ax.set_xlabel("Month")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Figure 8. Activities by month over 36 months; bars include their start and end months.", loc="left", fontsize=11, pad=12)
    fig.savefig(OUT / "Fig8_timeline_36m.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    work_packages()
    timeline()
    print("figures in", OUT)


def _card(ax, x, y, w, h, fc="white", ec="#C9D1DA", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=ec, lw=lw))


def overview():
    fig = plt.figure(figsize=(15, 8.6))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.955, "SAMA: measure, explain, fix and guarantee camera-based pulse measurement across Indian skin tones",
            fontsize=16, fontweight="bold", color=INK, va="top")
    ax.text(0.03, 0.885, "Four steps, four work packages; each step ends in an output that stands on its own.", fontsize=11.5, color=MUTED)
    _card(ax, 0.03, 0.69, 3.94, 0.16, fc="#EEF1F5", ec="#EEF1F5")
    ax.text(0.07, 0.815, "THE PROBLEM", fontsize=10.5, fontweight="bold", color="#A64B00")
    ax.text(0.07, 0.775, "A webcam or phone can read the pulse from tiny colour changes in facial skin, adding heart rate and HRV to video consultations\n"
                         "at no hardware cost. That signal weakens in darker skin, dim light and compressed video, and no Indian data with ECG ground\n"
                         "truth and instrument-measured skin tone shows by how much.", fontsize=10.8, color=INK, va="top", linespacing=1.45)
    cards = [("1  MEASURE · WP1", "SAMA-150 dataset", "#0B6E6E",
              "150 adults in three skin-tone\nstrata measured with a\ncolorimeter (ITA°), 1 kHz ECG,\nfour cameras, two light levels\nand a seven-level video-\ncompression ladder.", "H1 · H2"),
             ("2  EXPLAIN · WP2", "Error budget", "#2F4BA0",
              "Split each person's error into\nskin optics, light, device and\ncompression. Test whether\ncompression widens the\nskin-tone gap.", "H1 · H2"),
             ("3  FIX · WP3", "Capture and model", "#A64B00",
              "Skin-tone-adaptive camera\nexposure, then calibrated,\nnoise-weighted pulse\nextraction; every step tested\nby ablation.", "H3 · H4"),
             ("4  GUARANTEE · WP3–4", "Audited reliability", "#14202B",
              "90 % intervals; no reading is\nshown when the interval is too\nwide. Coverage and withheld\nreadings reported for every\nskin-tone stratum. HRV vs ECG.", "H4 · H5")]
    for i, (tag, title, col, body, hyp) in enumerate(cards):
        x = 0.03 + i * 0.99
        _card(ax, x, 0.17, 0.92, 0.47, ec=col, lw=2.2)
        ax.text(x + 0.04, 0.595, tag, fontsize=10.5, fontweight="bold", color=col)
        ax.text(x + 0.04, 0.535, title, fontsize=14, fontweight="bold", color=INK)
        ax.text(x + 0.04, 0.49, body, fontsize=10.6, color=INK, va="top", linespacing=1.45)
        ax.text(x + 0.04, 0.2, hyp, fontsize=10.5, fontweight="bold", color=col)
        if i < 3:
            ax.text(x + 0.945, 0.40, "›", fontsize=28, color=MUTED, va="center")
    _card(ax, 0.03, 0.02, 3.94, 0.11, ec=INK, lw=1.6)
    ax.text(0.07, 0.075, "OUTPUTS", fontsize=10.5, fontweight="bold", color=INK, va="center")
    ax.text(0.42, 0.075, "Open dataset (controlled access for faces) · reference error tables by skin tone, light, device and bitrate · open model and toolkit\n"
                         "· on-device Android research demonstrator. Not a medical device; no deployment is claimed.", fontsize=10.5, color=INK, va="center",
            linespacing=1.4)
    fig.savefig(OUT / "Fig1_SAMA_overview_v2.png", dpi=160)
    plt.close(fig)


def method():
    fig = plt.figure(figsize=(15, 8.0))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.955, "The SAMA method: fix the signal at capture, extract it carefully, and show a reading only when it can be trusted",
            fontsize=15.5, fontweight="bold", color=INK, va="top")
    ax.text(0.04, 0.89, "Steps 1, 4 and 6 combine established tools in a way not yet tested with measured skin tone; steps 2, 3 and 5 adapt established methods.\n"
                        "Whether each step helps is decided by ablation. In the demonstrator all steps run on the phone before any compression.",
            fontsize=10.8, color=MUTED, va="top", linespacing=1.4)
    steps = [("1 · CAPTURE", "Skin-tone-\nadaptive\nexposure", "Exposure set so the\nface sits at a target\nbrightness without\nclipping; white\nbalance locked.", True),
             ("2 · CALIBRATE", "Linear colour\nand a noise\nmodel", "Colour chart gives\nlinear RGB and each\ncamera's noise at\nevery brightness.", False),
             ("3 · SKIN PATCHES", "Weighted by\nsignal-to-noise", "About 30 patches;\nbeard, bindi,\nhighlights masked;\nweight = pulse power\n/ expected noise.", False),
             ("4 · PULSE", "Physics-guided\nrefinement", "POS projection\nrefined by a small\nnetwork; conditioning\non ITA° and group-\nDRO tested by ablation.", True),
             ("5 · BEATS", "Sub-frame\nbeat timing", "Peaks to 1 ms;\nheart rate, inter-\nbeat intervals,\nRMSSD, SDNN.", False),
             ("6 · GATE", "Reliability\ngate with\naudit", "90 % interval;\nshown only if half-\nwidth ≤ 5 bpm, else\n\"no reliable reading\";\naudited per stratum.", True)]
    for i, (tag, title, body, new) in enumerate(steps):
        x = 0.04 + i * 0.99
        col = "#A64B00" if new else "#C9D1DA"
        _card(ax, x, 0.33, 0.9, 0.47, fc="#FCEFE3" if new else "white", ec=col, lw=2.2 if new else 1.4)
        ax.text(x + 0.05, 0.765, tag, fontsize=10, fontweight="bold", color=MUTED)
        ax.text(x + 0.05, 0.735, title, fontsize=12.5, fontweight="bold", color=INK, va="top", linespacing=1.15)
        ax.text(x + 0.05, 0.575, body, fontsize=10, color=INK, va="top", linespacing=1.4)
        if i < 5:
            ax.text(x + 0.925, 0.56, "›", fontsize=24, color=MUTED, va="center")
    _card(ax, 0.04, 0.04, 2.9, 0.24)
    ax.text(0.09, 0.235, "COMPARED IN WP2 AND WP3", fontsize=10.5, fontweight="bold", color="#2F4BA0")
    ax.text(0.09, 0.2, "The same steps 2–6 run on video received after compression at\n2,000 to 250 kbps. The difference is what \"measure before you\ncompress\" saves, per skin-tone stratum.",
            fontsize=10.8, color=INK, va="top", linespacing=1.4)
    _card(ax, 3.06, 0.04, 2.9, 0.24)
    ax.text(3.11, 0.235, "TRAINING AND TESTING", fontsize=10.5, fontweight="bold", color="#0B6E6E")
    ax.text(3.11, 0.2, "Pre-trained on public datasets (already done for the pilot), fine-\ntuned on SAMA-150 by nested participant-wise cross-validation;\ncross-conformal gate; external tests on UBFC-Phys and MMPD.",
            fontsize=10.8, color=INK, va="top", linespacing=1.4)
    fig.savefig(OUT / "Fig5_SAMA_method_v2.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    overview()
    method()

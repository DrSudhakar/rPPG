"""Power for the revised confirmatory criteria of the rewritten proposal (section 6, Figure 7).

ASSUMPTION-DRIVEN: same assumptions as the original proposal (scripts/power.py), each criterion
simulated exactly as it is now worded:

H1  one-sided slope of log per-participant MAE on ITA° (sama/power_check.py, shared with the pilot
    re-check). Effect: ratio 1.57 across ITA° 50 -> -25 (2.87 -> 4.5 bpm; Chari et al. 2020's
    3.94 -> 6.20 bpm spans at least that range); tau = 0.55 (raw-scale CV ~0.6). The October draft's
    larger effect (2.5 -> 4.5, ratio 1.8) and tau = 0.7 are sensitivities. Type-I error at zero slope.
H2  two-sided slope of the 500 kbps penalty on ITA° (+1.0 bpm across the range, SD 1.0): power to name
    prediction A; expected 95 % CI half-width; probability that TOST with +/-1.0 bpm per 75° (the same
    smallest effect of interest as A) establishes B when the true slope is 0.
H3  one-sided paired t, darker stratum, standard light: 1.0 bpm (minimum important) and 0.5 bpm.
H4  darker stratum: lower 95 % bound of the mean reduction >= 10 % of the baseline mean, true 30 %
    (analysis/design_checks.h4_rules, unchanged model).
H5  resting ln RMSSD ICC(2,1), strata pooled (3n participants): lower 95 % bound (McGraw-Wong) >= 0.60,
    at true ICC 0.85, 0.80, 0.75, 0.70; the October draft's extra "point ICC >= 0.75" and "every stratum"
    requirements are reported for comparison.
Coverage: probability that the pooled gate's coverage in one held-out stratum is >= 0.85 (the
non-inferiority margin), calibration set of 3n participants.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from design_checks import h4_rules  # noqa: E402
from sama.power_check import h1_power  # noqa: E402
from sama.stats import icc21  # noqa: E402

SIMS = 2000
NS = [30, 36, 40, 45, 50]


def ita(rng, n):
    return np.concatenate([rng.uniform(28, 50, n), rng.uniform(10, 28, n), rng.uniform(-25, 10, n)])


def h1(n, tau=0.55, anchor=(2.87, 4.5)):
    return h1_power(n, tau, anchor, SIMS, 1, "log"), h1_power(n, tau, anchor, SIMS, 1, "raw")


def h2(n, sd=1.0, effect=1.0, margin=1.0, sims=SIMS, seed=2):
    rng = np.random.default_rng(seed)
    hit = 0
    hw, eq = [], 0
    for _ in range(sims):
        x = ita(rng, n)
        y = 1.0 + effect * (50 - x) / 75 + rng.normal(0, sd, x.size)
        r = stats.linregress(x, y)
        hit += 2 * stats.t.sf(abs(r.slope / r.stderr), x.size - 2) < 0.05
        hw.append(1.96 * r.stderr * 75)
        y0 = 1.0 + rng.normal(0, sd, x.size)
        r0 = stats.linregress(x, y0)
        t90 = stats.t.ppf(0.95, x.size - 2)
        lo, hi = (r0.slope - t90 * r0.stderr) * 75, (r0.slope + t90 * r0.stderr) * 75
        eq += (lo > -margin) and (hi < margin)
    return hit / sims, float(np.median(hw)), eq / sims


def h3(n, delta, sd=2.0, sims=SIMS, seed=3):
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(sims):
        d = rng.normal(delta, sd, n)
        hit += stats.t.sf(d.mean() / (d.std(ddof=1) / np.sqrt(n)), n - 1) < 0.05
    return hit / sims


def h5(n, rho=0.85, sims=600, seed=5):
    """Returns P(pooled lower 95 % bound >= 0.60) [the primary criterion], P(that and pooled ICC >= 0.75),
    and P(every stratum's lower bound >= 0.60)."""
    rng = np.random.default_rng(seed)
    e = np.sqrt((1 - rho) / rho)
    lb = both = per = 0
    for _ in range(sims):
        xs, lows = [], []
        for _s in range(3):
            t = rng.normal(0, 1, n)
            x = np.column_stack([t + rng.normal(0, e, n), t + 0.05 + rng.normal(0, e, n)])
            xs.append(x)
            lows.append(icc21(x)["lo95"])
        r = icc21(np.vstack(xs))
        lb += r["lo95"] >= 0.60
        both += (r["icc"] >= 0.75) and (r["lo95"] >= 0.60)
        per += all(v >= 0.60 for v in lows)
    return lb / sims, both / sims, per / sims


def coverage_ge(n_cal, margin=0.85, n_test=36, win=100, tau_sd=0.3, alpha=0.10, sims=1500, seed=6):
    """P(coverage in one held-out stratum >= margin) for a pooled normalised-score conformal gate
    with n_cal calibration participants (one window each), heterogeneous participant difficulty."""
    import math
    rng = np.random.default_rng(seed)
    k = math.ceil((1 - alpha) * (n_cal + 1))
    hits = 0
    for _ in range(sims):
        sc = np.abs(rng.normal(0, np.exp(rng.normal(0, tau_sd, n_cal))))
        q = np.sort(sc)[k - 1]
        st = np.abs(rng.normal(0, np.repeat(np.exp(rng.normal(0, tau_sd, n_test)), win)))
        hits += (st <= q).mean() >= margin
    return hits / sims


def type1_h1(n=36, tau=0.55, sims=4000, seed=9):
    rng = np.random.default_rng(seed)
    a = b = 0
    for _ in range(sims):
        x = ita(rng, n)
        y = 3.6 * np.exp(rng.normal(-tau ** 2 / 2, tau, x.size))
        r1, r2 = stats.linregress(x, np.log(y)), stats.linregress(x, y)
        a += stats.t.cdf(r1.slope / r1.stderr, x.size - 2) < 0.05
        b += stats.t.cdf(r2.slope / r2.stderr, x.size - 2) < 0.05
    return a / sims, b / sims


def main():
    out = {"assumptions": __doc__, "sims": SIMS, "rows": []}
    for n in NS:
        l, r = h1(n)
        l7, r7 = h1(n, tau=0.7)
        lo18, _ = h1(n, anchor=(2.5, 4.5))
        p2, hw, eq = h2(n)
        h4 = h4_rules(n=n, rng=np.random.default_rng(4), sims=SIMS)
        p5, p5b, p5s = h5(n)
        p5_80, _, _ = h5(n, rho=0.80)
        p5_75, p5b_75, _ = h5(n, rho=0.75)
        p5_70, _, _ = h5(n, rho=0.70)
        row = {"n_per_stratum": n, "H1_log": l, "H1_raw": r, "H1_log_tau07": l7, "H1_raw_tau07": r7, "H1_log_october_effect": lo18, "H2_A": p2,
               "H2_slope_halfwidth_bpm_per75": hw, "H2_B_equivalence_when_null": eq, "H3_1.0": h3(n, 1.0), "H3_0.5": h3(n, 0.5),
               "H4_lower95_ge10pct": h4["lower95_ge_10pct"], "H4_point_ge30pct": h4["point_estimate_ge30pct"],
               "H5_pooled_lb": p5, "H5_pooled_lb_and_icc075": p5b, "H5_all_strata": p5s, "H5_pooled_lb_icc080": p5_80,
               "H5_pooled_lb_icc075": p5_75, "H5_pooled_lb_and_icc075_at_icc075": p5b_75, "H5_pooled_lb_icc070": p5_70,
               "coverage_ge085": coverage_ge(3 * n)}
        out["rows"].append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}))
    out["type1_H1_log"], out["type1_H1_raw"] = type1_h1()
    print("type I (n=36):", out["type1_H1_log"], out["type1_H1_raw"])
    (ROOT / "results/tables/proposal_power.json").write_text(json.dumps(out, indent=1))
    figure(out)


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = out["rows"]
    n = [r["n_per_stratum"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
    series = [("H1: log-MAE slope on ITA°", "H1_log", "#2F4BA0", "-"), ("H2: penalty slope (prediction A)", "H2_A", "#0B6E6E", "-"),
              ("H3: adaptive exposure (good light), 1.0 bpm", "H3_1.0", "#A64B00", "-"), ("H3 at 0.5 bpm", "H3_0.5", "#A64B00", ":"),
              ("H4: lower bound ≥ 10 % (true 30 %)", "H4_lower95_ge10pct", "#6B2FA0", "-"),
              ("H5: pooled lower bound ≥ 0.60 (true ICC 0.85)", "H5_pooled_lb", "#1D7A46", "-"),
              ("H5 if true ICC were 0.75", "H5_pooled_lb_icc075", "#1D7A46", "--"),
              ("H5 if every stratum had to pass", "H5_all_strata", "#1D7A46", ":")]
    for lab, k, c, ls in series:
        ax.plot(n, [r[k] for r in rows], ls, color=c, marker="o", ms=4, lw=2, label=lab)
    ax.axhline(0.8, color="#46535F", lw=1, ls="--")
    ax.axvline(45, color="#B8C0C8", lw=1)
    ax.text(45.3, 0.32, "45 analysable\nper stratum\n(50 recruited)", fontsize=8.5, color="#46535F")
    ax.set_xlabel("Analysable participants per skin-tone stratum")
    ax.set_ylabel("Power (probability the criterion is met)")
    ax.set_ylim(0.3, 1.01)
    ax.grid(True, color="#E3E7EC")
    ax.legend(fontsize=8.6, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_xticks(n)
    ax.set_title("Figure 7. Power of each confirmatory criterion as worded (2,000 simulated studies per point)", loc="left", fontsize=10)
    fig.savefig(ROOT / "results/figures/fig_proposal_power.png", dpi=200)


if __name__ == "__main__":
    if sys.argv[1:] == ["--figure-only"]:
        figure(json.loads((ROOT / "results/tables/proposal_power.json").read_text()))
    else:
        main()

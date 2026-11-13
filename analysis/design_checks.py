"""Design checks on the proposal's own numbers (Q-B), independent of power.py.

Everything here is a statistical simulation under the proposal's stated assumptions (labelled
ASSUMPTION-DRIVEN) or LITERATURE inputs (labelled). Outputs: results/tables/*.csv,
proposal_inserts/power_rerun.md.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sama.stats import icc21  # noqa: E402

T = ROOT / "results/tables"
T.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261102)
SIMS = 3000
ALPHA = 0.05
N36 = 36


def lognormal_noise(n, sd, rng):
    s = 0.6
    z = rng.lognormal(0, s, n)
    z = (z - np.exp(s ** 2 / 2)) / np.sqrt((np.exp(s ** 2) - 1) * np.exp(s ** 2))
    return z * sd


def ita_proposal(n, rng):
    return np.concatenate([rng.uniform(28, 50, n), rng.uniform(10, 28, n), rng.uniform(-25, 10, n)])


def ita_india(n, rng):
    """Strata filled from the Pawar et al. 2024 mixture (LITERATURE): clusters -47±14.8 (24 %),
    4.3±11.3 (61 %), 34.7±4.3 (15 %); each stratum gets n draws restricted to its range."""
    comp = [(-47.0, 14.8, 0.239), (4.3, 11.3, 0.609), (34.7, 4.3, 0.152)]
    out = {"lighter": [], "middle": [], "darker": []}
    while min(len(v) for v in out.values()) < n:
        k = rng.choice(3, p=[c[2] for c in comp])
        x = rng.normal(comp[k][0], comp[k][1])
        s = "lighter" if x > 28 else ("middle" if x > 10 else "darker")
        if len(out[s]) < n:
            out[s].append(x)
    return np.concatenate([out["lighter"], out["middle"], out["darker"]])


def slope_p(x, y, one_sided=True):
    r = stats.linregress(x, y)
    t = r.slope / r.stderr
    return stats.t.cdf(t, len(x) - 2) if one_sided else 2 * stats.t.sf(abs(t), len(x) - 2)


# ---------------------------------------------------------------- H1 / H2 (independent rerun)

def power_h1(n, ita_fn=ita_proposal, sd=2.0, gap=2.0, ita_err=0.0, rng=RNG, sims=SIMS):
    hits = 0
    for _ in range(sims):
        ita = ita_fn(n, rng)
        mae = 2.5 + gap * (50 - ita) / 75 + lognormal_noise(3 * n, sd, rng)
        obs = ita + rng.normal(0, ita_err, ita.size) if ita_err else ita
        hits += slope_p(obs, mae) < ALPHA
    return hits / sims


def power_h2(n, sd=1.0, rng=RNG, sims=SIMS):
    hits = 0
    for _ in range(sims):
        ita = ita_proposal(n, rng)
        pen = 1.0 + 1.0 * (50 - ita) / 75 + rng.normal(0, sd, 3 * n)
        hits += slope_p(ita, pen, False) < ALPHA
    return hits / sims


# ---------------------------------------------------------------- H3 / H4

def power_paired(n, delta, sd, rng=RNG, sims=SIMS):
    d = rng.normal(delta, sd, (sims, n))
    t = d.mean(1) / (d.std(1, ddof=1) / np.sqrt(n))
    return float((stats.t.sf(t, n - 1) < ALPHA).mean())


def h4_rules(n=N36, base=4.5, rel_true=0.30, sd_diff=2.0, sd_base=2.0, rng=RNG, sims=SIMS):
    """P(success) of the darker-stratum MAE criterion under three readings (I-05)."""
    out = {"any_reduction_p05": 0, "point_estimate_ge30pct": 0, "lower95_gt_0": 0, "lower95_ge_10pct": 0}
    for _ in range(sims):
        b = np.clip(base + lognormal_noise(n, sd_base, rng), 0.3, None)
        d = rng.normal(rel_true * base, sd_diff, n)
        s = b - d
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
        p = stats.t.sf(m / se, n - 1)
        out["any_reduction_p05"] += p < ALPHA
        out["point_estimate_ge30pct"] += (1 - s.mean() / b.mean()) >= 0.30
        lo = m - stats.t.ppf(0.95, n - 1) * se
        out["lower95_gt_0"] += lo > 0
        out["lower95_ge_10pct"] += lo >= 0.10 * b.mean()
    return {k: v / sims for k, v in out.items()}


# ---------------------------------------------------------------- coverage criterion (I-06, I-30)

def coverage_criterion(n_cal, n_test=N36, win_per_test=100, tau_sd=0.3, alpha=0.10, rng=RNG, sims=2000):
    """Normalised-score split conformal with one window per calibration participant. Participant
    difficulty tau_i (sigma_hat misspecified by a lognormal factor) makes windows correlated.
    Returns P(empirical coverage >= 0.88) in one stratum and P(all 3 strata), plus P(q = inf)."""
    k = math.ceil((1 - alpha) * (n_cal + 1))
    one, inf = [], 0
    for _ in range(sims * 3):
        if k > n_cal:
            inf += 1
            one.append(True)  # infinite interval covers everything but abstains always
            continue
        tc = np.exp(rng.normal(0, tau_sd, n_cal))
        sc = np.abs(rng.normal(0, tc))
        q = np.sort(sc)[k - 1]
        tt = np.exp(rng.normal(0, tau_sd, n_test))
        st = np.abs(rng.normal(0, np.repeat(tt, win_per_test)))
        one.append((st <= q).mean() >= 0.88)
    one = np.array(one).reshape(sims, 3)
    return {"n_cal": n_cal, "k": k, "finite_quantile": k <= n_cal,
            "p_one_stratum": float(one[:, 0].mean()), "p_all_three": float(one.all(1).mean()),
            "beta_p_below_088": float(stats.beta.cdf(0.88, k, n_cal + 1 - k)) if k <= n_cal else float("nan")}


# ---------------------------------------------------------------- H5 (I-25, I-26)

def h5_power(n=N36, rho=0.85, rng=RNG, sims=2000):
    """Per-stratum lower bound >= 0.60 by Fisher z (power.py) vs McGraw-Wong F interval, and the
    joint criterion (overall ICC >= 0.75 and all three strata lower bound >= 0.60)."""
    res = {"fisher_one": 0, "f_one": 0, "joint_fisher": 0, "joint_f": 0}
    e = np.sqrt((1 - rho) / rho)
    for _ in range(sims):
        lows_f, lows_z, alls = [], [], []
        for _s in range(3):
            true = rng.normal(0, 1, n)
            x = np.column_stack([true + rng.normal(0, e, n), true + 0.05 + rng.normal(0, e, n)])
            r = icc21(x)
            rz = min(max(r["icc"], -0.99), 0.999)
            lows_z.append(np.tanh(np.arctanh(rz) - 1.959964 / np.sqrt(n - 2)))
            lows_f.append(r["lo95"])
            alls.append(x)
        overall = icc21(np.vstack(alls))["icc"]
        res["fisher_one"] += lows_z[0] >= 0.60
        res["f_one"] += lows_f[0] >= 0.60
        res["joint_fisher"] += (overall >= 0.75) and all(v >= 0.60 for v in lows_z)
        res["joint_f"] += (overall >= 0.75) and all(v >= 0.60 for v in lows_f)
    return {k: v / sims for k, v in res.items()}


def reactivity_reliability(r_level, sd_between=0.45, sd_change=0.22, rho_err=0.0):
    """Reliability (≈ ICC) of a difference score given the reliability of the level (I-25).
    Level: r = sb^2/(sb^2+se^2). Difference error var = 2 se^2 (1 - rho_err)."""
    se2 = sd_between ** 2 * (1 / r_level - 1)
    return sd_change ** 2 / (sd_change ** 2 + 2 * se2 * (1 - rho_err))


def main():
    out = {}
    # independent power rerun at 36 and 40
    rows = []
    for n in (36, 40):
        rows.append({"n_per_stratum": n, "H1": power_h1(n), "H2": power_h2(n), "H3": power_paired(n, 1.0, 2.0),
                     "H4_any_reduction": power_paired(n, 1.35, 2.0), **{f"H5_{k}": v for k, v in h5_power(n).items()}})
    pr = pd.DataFrame(rows)
    pr.to_csv(T / "power_rerun.csv", index=False)
    out["power_rerun"] = rows
    # H1 sensitivity: ITA distribution, between-person SD, ITA measurement error
    sens = []
    for label, fn in (("proposal ranges", ita_proposal), ("Indian mixture (Pawar 2024)", ita_india)):
        for sd in (2.0, 4.0, 6.0):
            for err in (0.0, 6.0):
                sens.append({"ita_distribution": label, "between_sd": sd, "ita_error_sd": err,
                             "power_n36": power_h1(36, fn, sd=sd, ita_err=err, sims=1500)})
    pd.DataFrame(sens).to_csv(T / "h1_power_sensitivity.csv", index=False)
    # H4 rules
    h4 = []
    for rel in (0.20, 0.30, 0.40, 0.50):
        h4.append({"true_reduction": rel, **h4_rules(rel_true=rel)})
    pd.DataFrame(h4).to_csv(T / "h4_power_by_rule.csv", index=False)
    # coverage criterion vs calibration size (I-06, I-30)
    cov = [coverage_criterion(n) for n in (7, 8, 9, 14, 20, 36, 72)]
    pd.DataFrame(cov).to_csv(T / "coverage_criterion.csv", index=False)
    # reactivity reliability (I-25)
    rr = []
    for r_level in (0.80, 0.88, 0.95):
        for sd_change in (0.15, 0.22, 0.30, 0.40):
            for rho in (0.0, 0.3, 0.5):
                rr.append({"level_reliability": r_level, "sd_true_change": sd_change, "error_corr_between_blocks": rho,
                           "reactivity_reliability": reactivity_reliability(r_level, sd_change=sd_change, rho_err=rho)})
    pd.DataFrame(rr).to_csv(T / "h5_reactivity_icc.csv", index=False)
    # recruitment (I-18): screened volunteers needed to fill 40 per stratum from the Indian mixture
    p_strata = {}
    draws = np.concatenate([RNG.normal(m, s, int(200000 * w)) for m, s, w in ((-47, 14.8, .239), (4.3, 11.3, .609), (34.7, 4.3, .152))])
    p_strata = {"lighter": float((draws > 28).mean()), "middle": float(((draws > 10) & (draws <= 28)).mean()), "darker": float((draws <= 10).mean())}
    need = []
    for _ in range(4000):
        cnt = {"lighter": 0, "middle": 0, "darker": 0}
        n = 0
        while min(cnt.values()) < 40:
            u = RNG.random()
            s = "lighter" if u < p_strata["lighter"] else ("middle" if u < p_strata["lighter"] + p_strata["middle"] else "darker")
            cnt[s] += 1
            n += 1
        need.append(n)
    need = np.array(need)
    rec = {"p_lighter": p_strata["lighter"], "p_middle": p_strata["middle"], "p_darker": p_strata["darker"],
           "screened_median": float(np.median(need)), "screened_p90": float(np.quantile(need, 0.9)),
           "source": "LITERATURE: Pawar et al. 2024 (n=368 Indian adults, cheek ITA clusters); city not Surat"}
    pd.DataFrame([rec]).to_csv(T / "recruitment_sim.csv", index=False)
    out.update({"h4_rules": h4, "coverage": cov, "recruitment": rec, "h1_sensitivity": sens})
    (T / "design_checks.json").write_text(json.dumps(out, indent=1, default=float))
    print(pr.round(3).to_string())
    print(pd.DataFrame(h4).round(3).to_string())
    print(pd.DataFrame(cov).round(3).to_string())
    print(rec)


if __name__ == "__main__":
    main()

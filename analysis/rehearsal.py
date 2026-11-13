"""C6: SAMA-120 rehearsal for H1s and H2s on the SAMA-Sim pools (SYNTHETIC, ASSUMPTION-DRIVEN).

Reads results/cache/pool_{A,B,melanin}_s*.csv.gz; writes results/tables/c6_*.csv, figures and
results/c6_summary.json. Plan: spec/SAP_SIM.md s.2 with amendment 1 (A2, A6).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sama.stats import lmm_crossed, slope_test  # noqa: E402

T, F = ROOT / "results/tables", ROOT / "results/figures"
STUDIES = 1000
RNG = np.random.default_rng(2026)
STRATA = ("lighter", "middle", "darker")


def load_pool(world):
    f = sorted(glob.glob(str(ROOT / f"results/cache/pool_{world}_s*.csv.gz")))
    if not f:
        return None
    d = pd.read_csv(f[0])
    d["err"] = (d.hr_cam - d.hr_ref).abs()
    return d


def per_participant(d):
    g = d[np.isfinite(d.hr_ref)].groupby(["pid", "codec"]).agg(ita=("ita", "first"), stratum=("stratum", "first"),
                                                                  mae=("err", "mean"), snr=("snr_db", "mean")).reset_index()
    w = g.pivot(index="pid", columns="codec", values=["mae", "snr"])
    med = d[np.isfinite(d.hr_ref) & (d.codec == "lossless")].groupby("pid").err.median()
    out = g[g.codec == "lossless"].set_index("pid")[["ita", "stratum"]].copy()
    out["medae"] = med
    out["mae"] = w[("mae", "lossless")]
    out["mae500"] = w[("mae", "500")]
    out["penalty"] = out.mae500 - out.mae
    out["snr_pen"] = w[("snr", "lossless")] - w[("snr", "500")]
    return out.reset_index()


def draw_study(pp, rng, n_rec=40, n_an=36):
    parts = []
    for s in STRATA:
        ids = pp.index[pp.stratum == s].to_numpy()
        pick = rng.choice(ids, size=min(n_rec, ids.size), replace=False)
        parts.append(rng.choice(pick, size=min(n_an, pick.size), replace=False))  # 10 % session loss
    return pp.loc[np.concatenate(parts)]


def tost_slope(x, y, margin_per75=0.5):
    r = slope_test(x, y, "two-sided")
    m = margin_per75 / 75.0
    df = r["n"] - 2
    p_lo = stats.t.sf((r["slope"] + m) / r["se"], df)
    p_hi = stats.t.cdf((r["slope"] - m) / r["se"], df)
    return max(p_lo, p_hi) < 0.05


def h2_decision(st):
    a = slope_test(st.ita, st.penalty, "two-sided")
    snr = slope_test(st.ita, st.snr_pen, "two-sided")
    if a["p"] < 0.05 and a["slope"] < 0:
        frozen = "A"
    elif snr["p"] < 0.05 and snr["slope"] < 0:
        frozen = "B"
    else:
        frozen = "inconclusive"
    eq = tost_slope(st.ita, st.penalty)
    amended = "A" if frozen == "A" else ("B-equivalent" if eq else "inconclusive")
    return frozen, amended, a["slope"] * 75


def parametric_sampler(pp, col="mae"):
    """A6: fresh participants from a fitted model: linear trend in ITA + stratum-specific
    empirical residuals (residual bootstrap), ITA uniform within the pool's stratum range."""
    r = slope_test(pp.ita, pp[col], "two-sided")
    resid = pp[col] - (pp[col].mean() + r["slope"] * (pp.ita - pp.ita.mean()))
    rng_ = {s: (pp.ita[pp.stratum == s].min(), pp.ita[pp.stratum == s].max()) for s in STRATA}
    res = {s: resid[pp.stratum == s].to_numpy() for s in STRATA}

    def draw(rng, n):
        rows = []
        for s in STRATA:
            ita = rng.uniform(*rng_[s], n)
            y = pp[col].mean() + r["slope"] * (ita - pp.ita.mean()) + rng.choice(res[s], n)
            rows.append(pd.DataFrame({"ita": ita, col: y, "stratum": s}))
        return pd.concat(rows)
    return draw


def main():
    T.mkdir(parents=True, exist_ok=True)
    F.mkdir(parents=True, exist_ok=True)
    summ = {"provenance": "SYNTHETIC (SAMA-Sim pools); ASSUMPTION-DRIVEN", "studies": STUDIES}
    pools = {w: load_pool(w) for w in ("A", "B", "melanin")}
    pps = {w: per_participant(d) for w, d in pools.items() if d is not None}
    # realised gradient in each pool (whole pool)
    grad = []
    for w, pp in pps.items():
        r = slope_test(pp.ita, pp.mae, "two-sided")
        q = {s: pp.mae[pp.stratum == s] for s in STRATA}
        grad.append({"world": w, "n": len(pp), "slope_per75": -r["slope"] * 75, "lo95": -r["hi95"] * 75, "hi95": -r["lo95"] * 75,
                     **{f"mae_{s}": float(q[s].mean()) for s in STRATA}, **{f"sd_{s}": float(q[s].std(ddof=1)) for s in STRATA},
                     **{f"median_{s}": float(q[s].median()) for s in STRATA}})
    pd.DataFrame(grad).to_csv(T / "c6_realised_gradient.csv", index=False)
    summ["realised_gradient"] = grad
    pp = pps["A"]
    # H1 power, type-I, ITA error, conditional (pool) and parametric
    rows = []
    for err in (0.0, 3.0, 6.0, 10.0):
        hit = hit0 = 0
        for _ in range(STUDIES):
            st = draw_study(pp, RNG)
            ita = st.ita + RNG.normal(0, err, len(st)) if err else st.ita
            hit += slope_test(ita, st.mae, "less")["p"] < 0.05
            hit0 += slope_test(RNG.permutation(ita.to_numpy()), st.mae.to_numpy(), "less")["p"] < 0.05
        rows.append({"method": "pool resampling (conditional)", "ita_error_sd": err, "power": hit / STUDIES, "type_I": hit0 / STUDIES})
    draw = parametric_sampler(pp, "mae")
    curve = []
    for n in (20, 28, 36, 44, 52, 60):
        hit = sum(slope_test((s := draw(RNG, n)).ita, s.mae, "less")["p"] < 0.05 for _ in range(STUDIES))
        curve.append({"n_per_stratum": n, "power_parametric": hit / STUDIES})
    rows.append({"method": "parametric (fresh participants)", "ita_error_sd": 0.0,
                 "power": [c for c in curve if c["n_per_stratum"] == 36][0]["power_parametric"], "type_I": np.nan})
    # A11: endpoints robust to heavy tails (exploratory alternatives for the proposal)
    for name, col, tf in (("log MAE", "mae", np.log), ("median AE", "medae", lambda v: v), ("rank (Spearman)", "mae", None)):
        hit = 0
        for _ in range(STUDIES):
            st = draw_study(pp, RNG)
            if tf is None:
                rho, p2 = stats.spearmanr(st.ita, st[col])
                hit += (rho < 0) and (p2 / 2 < 0.05)
            else:
                hit += slope_test(st.ita, tf(st[col]), "less")["p"] < 0.05
        rows.append({"method": f"pool resampling, endpoint {name}", "ita_error_sd": 0.0, "power": hit / STUDIES, "type_I": np.nan})
    pd.DataFrame(rows).to_csv(T / "c6_h1_power.csv", index=False)
    pd.DataFrame(curve).to_csv(T / "c6_h1_power_curve.csv", index=False)
    summ["h1"] = rows
    # mixed-model sensitivity on 100 studies (windows)
    d = pools["A"]
    d = d[(d.codec == "lossless") & np.isfinite(d.hr_ref) & np.isfinite(d.err)]
    mm = []
    for i in range(100):
        st = draw_study(pp, RNG)
        w = d[d.pid.isin(st.pid)]
        X = np.column_stack([np.ones(len(w)), (w.ita - 15) / 10, (w.light == "dim").astype(float)])
        fit = lmm_crossed(np.log1p(w.err.to_numpy()), X, [w.pid.to_numpy(), w.block.to_numpy()], ["icpt", "ita10", "dim"])
        b, se = fit.beta[1], fit.se[1]
        mm.append({"study": i, "ita10": b, "se": se, "p_one_sided": stats.norm.cdf(b / se), "converged": fit.converged,
                   "singular": "singular" in fit.message})
    mm = pd.DataFrame(mm)
    mm.to_csv(T / "c6_h1_mixed_model.csv", index=False)
    summ["h1_mixed_model"] = {"power": float((mm.p_one_sided < 0.05).mean()), "nonconverged": int((~mm.converged).sum()),
                              "singular": int(mm.singular.sum()), "studies": len(mm)}
    # H2 discriminability (frozen and amended rules)
    disc = []
    for w in ("A", "B"):
        if w not in pps:
            continue
        cnt = {}
        for _ in range(STUDIES):
            fr, am, sl = h2_decision(draw_study(pps[w], RNG))
            cnt[("frozen", fr)] = cnt.get(("frozen", fr), 0) + 1
            cnt[("amended", am)] = cnt.get(("amended", am), 0) + 1
        for (rule, dec), c in cnt.items():
            disc.append({"true_world": w, "rule": rule, "decision": dec, "share": c / STUDIES})
        r = slope_test(pps[w].ita, pps[w].penalty, "two-sided")
        summ[f"h2_pool_slope_{w}"] = {"penalty_slope_per75": -r["slope"] * 75, "lo95": -r["hi95"] * 75, "hi95": -r["lo95"] * 75}
    pd.DataFrame(disc).to_csv(T / "c6_h2_discriminability.csv", index=False)
    summ["h2"] = disc
    # H2 type-I (penalty unrelated to ITA): permutation within World A studies
    t1 = sum(slope_test(RNG.permutation((st := draw_study(pp, RNG)).ita.to_numpy()), st.penalty.to_numpy(), "two-sided")["p"] < 0.05
             for _ in range(STUDIES)) / STUDIES
    summ["h2_type_I_permutation"] = t1
    (ROOT / "results/c6_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    figures(pps, curve)
    print(json.dumps({k: summ[k] for k in ("realised_gradient", "h1", "h1_mixed_model", "h2_type_I_permutation")}, indent=1, default=float))


def figures(pps, curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.figstyle import COL, setup
    setup()
    pp = pps["A"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for s in STRATA:
        q = pp[pp.stratum == s]
        ax[0].scatter(q.ita, q.mae, s=14, color=COL[s], alpha=0.8, label=s)
    r = slope_test(pp.ita, pp.mae, "two-sided")
    xs = np.linspace(pp.ita.min(), pp.ita.max(), 50)
    ax[0].plot(xs, pp.mae.mean() + r["slope"] * (xs - pp.ita.mean()), color="#14202B", lw=2)
    ax[0].plot([50, -25], [2.5, 4.5], ls="--", color="#888", lw=1.5, label="proposal assumption")
    ax[0].invert_xaxis()
    ax[0].set_yscale("log")
    ax[0].set_xlabel("ITA° (lighter → darker)")
    ax[0].set_ylabel("Participant POS MAE, bpm (log scale)")
    ax[0].set_title("A  Per-participant error is heavy-tailed")
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].plot([c["n_per_stratum"] for c in curve], [c["power_parametric"] for c in curve], marker="o", color=COL["darker"], lw=2.2, label="rehearsal (SAMA-Sim, World A)")
    prop = json.load(open(ROOT / "results/tables/design_checks.json"))["power_rerun"]
    ax[1].scatter([36, 40], [p["H1"] for p in prop], color="#888", zorder=3, label="proposal assumptions (rerun)")
    ax[1].axhline(0.8, ls="--", color="#888", lw=1)
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("Analysable participants per stratum")
    ax[1].set_ylabel("Power, H1 (one-sided slope, α 0.05)")
    ax[1].set_title("B  H1 power under the simulator's error spread")
    ax[1].legend(frameon=False, fontsize=9)
    fig.suptitle("SYNTHETIC · ASSUMPTION-DRIVEN (gradient calibrated to the proposal's H1 assumption)", x=0.01, ha="left", fontsize=9, color="#555")
    fig.savefig(F / "fig_h1_rehearsal.png", dpi=180)
    d = pd.read_csv(T / "c6_h2_discriminability.csv")
    fig, ax = plt.subplots(figsize=(8.5, 3.6), constrained_layout=True)
    labs, vals, cols = [], [], []
    for (w, rule), g in d.groupby(["true_world", "rule"]):
        for _, r in g.iterrows():
            labs.append(f"World {w} · {rule} · {r.decision}")
            vals.append(r.share)
            cols.append(COL["darker"] if (r.decision.startswith(w)) else "#B8C0C8")
    ax.barh(labs, vals, color=cols)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of 1,000 simulated studies (36 per stratum)")
    ax.set_title("H2: how often the pre-specified rule names the true world (dark bars = correct)", loc="left")
    fig.suptitle("SYNTHETIC · identifiability only", x=0.01, ha="left", fontsize=9, color="#555")
    fig.savefig(F / "fig_h2_discriminability.png", dpi=180)


if __name__ == "__main__":
    main()

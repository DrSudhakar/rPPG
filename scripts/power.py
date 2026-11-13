"""Monte Carlo power for SAMA's confirmatory hypotheses (assumptions stated in ASSUME)."""
import json
import numpy as np
from scipy import stats

rng = np.random.default_rng(20261028)
SIMS = 3000
ALPHA = 0.05  # one pre-specified primary test per hypothesis

ASSUME = {
    "ita_strata": {"A (ITA > 28)": [28, 50], "B (10 to 28)": [10, 28], "C (<= 10)": [-25, 10]},
    "H1": "Per-participant HR MAE on lossless video, standard light: 2.5 bpm at ITA 50 rising linearly to 4.5 bpm at ITA -25 (+2.0 bpm; Chari et al. 2020 report +2.06 bpm from Fitzpatrick 3-4 to 5-6); between-participant SD 2.0 bpm (log-normal noise). Test: one-sided slope of MAE on ITA.",
    "H2": "Compression penalty at 500 kbps (MAE minus lossless MAE): +1.0 bpm at ITA 50 rising to +2.0 bpm at ITA -25; within-participant SD 1.0 bpm (to be re-estimated from the pilot). Test: two-sided slope of penalty on ITA.",
    "H3": "Darkest stratum, same phone model: adaptive exposure lowers MAE by 1.0 bpm vs auto-exposure; SD of paired differences 2.0 bpm. Test: one-sided paired t.",
    "H4": "Darkest stratum: SAMA method lowers MAE by 1.35 bpm (30% of 4.5) vs best baseline; SD of paired differences 2.0 bpm. Test: one-sided paired t.",
    "H5": "True ICC(2,1) of RMSSD reactivity 0.85 in each stratum; criterion: lower 95% bound >= 0.60 (Fisher z approximation).",
}


def ita_draw(n):
    a = rng.uniform(28, 50, n); b = rng.uniform(10, 28, n); c = rng.uniform(-25, 10, n)
    return np.concatenate([a, b, c])


def slope_p(x, y, one_sided):
    res = stats.linregress(x, y)
    t = res.slope / res.stderr
    df = len(x) - 2
    if one_sided:  # expect negative slope (error rises as ITA falls)
        return stats.t.cdf(t, df)
    return 2 * stats.t.sf(abs(t), df)


def lognormal_noise(n, sd):
    # zero-mean, right-skewed noise with the given SD
    sigma = 0.6
    z = rng.lognormal(0, sigma, n)
    z = (z - np.exp(sigma**2 / 2)) / np.sqrt((np.exp(sigma**2) - 1) * np.exp(sigma**2))
    return z * sd


def power_h1(n):
    hits = 0
    for _ in range(SIMS):
        ita = ita_draw(n)
        mae = 2.5 + 2.0 * (50 - ita) / 75 + lognormal_noise(3 * n, 2.0)
        hits += slope_p(ita, mae, True) < ALPHA
    return hits / SIMS


def power_h2(n):
    hits = 0
    for _ in range(SIMS):
        ita = ita_draw(n)
        pen = 1.0 + 1.0 * (50 - ita) / 75 + rng.normal(0, 1.0, 3 * n)
        hits += slope_p(ita, pen, False) < ALPHA
    return hits / SIMS


def power_paired(n, delta, sd):
    hits = 0
    for _ in range(SIMS):
        d = rng.normal(delta, sd, n)
        t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
        hits += stats.t.sf(t, n - 1) < ALPHA
    return hits / SIMS


def icc21(x):
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def power_h5(n, rho=0.85, thr=0.60):
    hits = 0; lows = []
    for _ in range(SIMS):
        true = rng.normal(0, 1, n)
        e = np.sqrt((1 - rho) / rho)
        x = np.column_stack([true + rng.normal(0, e, n), true + 0.05 + rng.normal(0, e, n)])
        r = icc21(x)
        r = min(max(r, -0.99), 0.999)
        lo = np.tanh(np.arctanh(r) - 1.959964 / np.sqrt(n - 2))
        lows.append(lo); hits += lo >= thr
    return hits / SIMS, float(np.median(lows))


if __name__ == "__main__":
    ns = list(range(20, 62, 4))
    curves = {"H1": [], "H2": [], "H3": [], "H4": [], "H5": []}
    for n in ns:
        curves["H1"].append(power_h1(n))
        curves["H2"].append(power_h2(n))
        curves["H3"].append(power_paired(n, 1.0, 2.0))
        curves["H4"].append(power_paired(n, 1.35, 2.0))
        curves["H5"].append(power_h5(n)[0])
    at36 = {"H1": power_h1(36), "H2": power_h2(36), "H3": power_paired(36, 1.0, 2.0),
            "H4": power_paired(36, 1.35, 2.0), "H5": power_h5(36)}
    at40 = {"H1": power_h1(40), "H2": power_h2(40), "H3": power_paired(40, 1.0, 2.0),
            "H4": power_paired(40, 1.35, 2.0), "H5": power_h5(40)}
    out = {"alpha": ALPHA, "sims": SIMS, "n_per_stratum": ns, "curves": curves, "at36": at36, "at40": at40, "assume": ASSUME}
    json.dump(out, open("power.json", "w"), indent=1)
    print(json.dumps({"at36": at36, "at40": at40}, indent=1))
    for k, v in curves.items():
        print(k, [round(p, 2) for p in v])

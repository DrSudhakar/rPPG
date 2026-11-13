"""Metrics and statistics: error metrics, fairness summaries, slope tests, ICC(2,1) with
McGraw-Wong intervals, Bland-Altman, participant bootstrap and a REML linear mixed model with
crossed random intercepts (statsmodels is unavailable here; see DECISIONS.md D-05)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

# ---------------------------------------------------------------- error metrics


def hr_metrics(h_cam: np.ndarray, h_ref: np.ndarray) -> dict:
    """MAE, RMSE, Pearson r, share within ANSI/AAMI EC13 tolerance (±5 bpm or ±10 %)."""
    c, r = np.asarray(h_cam, float), np.asarray(h_ref, float)
    ok = np.isfinite(c) & np.isfinite(r)
    c, r = c[ok], r[ok]
    if c.size == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r": np.nan, "within_tol": np.nan}
    e = c - r
    tol = np.maximum(5.0, 0.10 * r)
    return {
        "n": int(c.size),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "r": float(np.corrcoef(c, r)[0, 1]) if c.size > 2 and c.std() > 0 and r.std() > 0 else np.nan,
        "within_tol": float((np.abs(e) <= tol).mean()),
    }


def fairness(mae_by_stratum: dict) -> dict:
    """Worst-stratum MAE, best-to-worst gap and ratio."""
    v = {k: x for k, x in mae_by_stratum.items() if np.isfinite(x)}
    if not v:
        return {"worst": np.nan, "gap": np.nan, "ratio": np.nan}
    w, b = max(v.values()), min(v.values())
    return {"worst": w, "gap": w - b, "ratio": w / b if b > 0 else np.nan}


# ---------------------------------------------------------------- tests


def slope_test(x, y, alternative: str = "less") -> dict:
    """OLS slope of y on x with t-test. alternative: 'less' (slope<0), 'greater' or 'two-sided'."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    res = stats.linregress(x[ok], y[ok])
    df = ok.sum() - 2
    t = res.slope / res.stderr if res.stderr > 0 else 0.0
    if alternative == "less":
        p = stats.t.cdf(t, df)
    elif alternative == "greater":
        p = stats.t.sf(t, df)
    else:
        p = 2 * stats.t.sf(abs(t), df)
    q = stats.t.ppf(0.975, df)
    return {"slope": res.slope, "se": res.stderr, "lo95": res.slope - q * res.stderr,
            "hi95": res.slope + q * res.stderr, "t": t, "p": float(p), "n": int(ok.sum())}


def paired_test(diff, alternative: str = "greater") -> dict:
    """One-sample t on paired differences (baseline minus new, so 'greater' = improvement)."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    n = d.size
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    t = m / se if se > 0 else 0.0
    p = stats.t.sf(t, n - 1) if alternative == "greater" else (stats.t.cdf(t, n - 1) if alternative == "less" else 2 * stats.t.sf(abs(t), n - 1))
    q = stats.t.ppf(0.975, n - 1)
    return {"mean": m, "se": se, "lo95": m - q * se, "hi95": m + q * se, "t": t, "p": float(p), "n": n}


def holm(pvals: dict) -> dict:
    """Holm-adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


# ---------------------------------------------------------------- agreement


def icc21(x: np.ndarray, alpha: float = 0.05) -> dict:
    """ICC(2,1), two-way random, absolute agreement, single measures (Shrout & Fleiss 1979),
    with the McGraw & Wong (1996, case 2A) F-based confidence interval. x: (n subjects, k raters)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x).all(1)]
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    icc = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    a = k * icc / (n * (1 - icc)) if icc < 1 else np.inf
    b = 1 + k * icc * (n - 1) / (n * (1 - icc)) if icc < 1 else np.inf
    v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    fl = stats.f.ppf(1 - alpha / 2, n - 1, v)
    fu = stats.f.ppf(1 - alpha / 2, v, n - 1)
    lo = n * (msr - fl * mse) / (fl * (k * msc + (k * n - k - n) * mse) + n * msr)
    hi = n * (fu * msr - mse) / (k * msc + (k * n - k - n) * mse + n * fu * msr)
    return {"icc": float(icc), "lo95": float(lo), "hi95": float(hi), "n": int(n)}


def bland_altman(a, b) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    m, s = d.mean(), d.std(ddof=1)
    return {"bias": float(m), "loa_lo": float(m - 1.96 * s), "loa_hi": float(m + 1.96 * s), "n": int(d.size)}


def bootstrap_ci(groups: list, fn, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple:
    """Participant-level bootstrap: resample `groups` (list of per-participant items) with
    replacement and apply fn(list) -> float. Returns (estimate, lo, hi)."""
    rng = np.random.default_rng(seed)
    est = fn(groups)
    n = len(groups)
    vals = np.array([fn([groups[i] for i in rng.integers(0, n, n)]) for _ in range(n_boot)])
    vals = vals[np.isfinite(vals)]
    q = (1 - level) / 2
    return float(est), float(np.quantile(vals, q)), float(np.quantile(vals, 1 - q))


# ---------------------------------------------------------------- mixed model


@dataclass
class LMMResult:
    names: list
    beta: np.ndarray
    se: np.ndarray
    var_components: dict
    converged: bool
    loglik_reml: float
    message: str

    def table(self) -> list[dict]:
        z = self.beta / self.se
        p = 2 * stats.norm.sf(np.abs(z))
        return [{"term": n, "est": b, "se": s, "lo95": b - 1.96 * s, "hi95": b + 1.96 * s, "p": pp}
                for n, b, s, pp in zip(self.names, self.beta, self.se, p)]


def lmm_crossed(y: np.ndarray, X: np.ndarray, groups: list[np.ndarray], names: list[str] | None = None) -> LMMResult:
    """REML fit of y = X b + sum_g Z_g u_g + e with independent random intercepts for each
    grouping factor in `groups` (e.g. participant and block type), via Henderson's mixed-model
    equations with variance ratios optimised on the log scale. Wald z intervals for b."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, p = X.shape
    Zs, qs = [], []
    for g in groups:
        _, inv = np.unique(g, return_inverse=True)
        Z = np.zeros((n, inv.max() + 1))
        Z[np.arange(n), inv] = 1.0
        Zs.append(Z)
        qs.append(Z.shape[1])
    Z = np.hstack(Zs)
    W = np.hstack([X, Z])
    WtW, Wty, yty = W.T @ W, W.T @ y, y @ y

    def build(logg):
        ginv = np.concatenate([np.full(q, np.exp(-lg)) for q, lg in zip(qs, logg)])
        C = WtW.copy()
        C[p:, p:] += np.diag(ginv)
        return C

    def neg2ll(logg):
        C = build(logg)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return 1e12
        theta = np.linalg.solve(C, Wty)
        s2 = (yty - theta @ Wty) / (n - p)
        if s2 <= 0:
            return 1e12
        logdetC = 2 * np.log(np.diag(L)).sum()
        logdetG = sum(q * lg for q, lg in zip(qs, logg))
        return (n - p) * np.log(s2) + logdetG + logdetC + (n - p)

    best = None
    for start in ([0.0] * len(qs), [-2.0] * len(qs), [2.0] * len(qs)):
        r = optimize.minimize(neg2ll, np.array(start), method="Nelder-Mead",
                              options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    logg = np.clip(best.x, -20, 20)
    C = build(logg)
    theta = np.linalg.solve(C, Wty)
    s2 = (yty - theta @ Wty) / (n - p)
    Cinv = np.linalg.inv(C)
    beta = theta[:p]
    se = np.sqrt(np.clip(np.diag(Cinv)[:p] * s2, 0, None))
    vc = {"residual": float(s2)}
    for i, lg in enumerate(logg):
        vc[f"group{i}"] = float(np.exp(lg) * s2)
    boundary = bool(np.any(best.x < -15))
    msg = "converged" if best.success else f"optimizer: {best.message}"
    if boundary:
        msg += "; variance component at boundary (singular fit)"
    return LMMResult(names or [f"b{i}" for i in range(p)], beta, se, vc, bool(best.success), -0.5 * best.fun, msg)

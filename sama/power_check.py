"""Power models shared by the proposal's power analysis (analysis/proposal_power.py) and the pilot's
power re-check (sama/pilot.py), so both use exactly the same data-generating assumptions.

H1 (primary endpoint: log of per-participant MAE, one-sided slope on ITA°). Multiplicative model:
    MAE_i = m(ITA_i) * exp(e_i),  e_i ~ N(-tau^2/2, tau^2)
m(ITA) rises geometrically from `lo` at ITA° 50 to `hi` at ITA° -25. Default anchor: ratio 1.57 over
the range (Chari et al. 2020: 3.94 -> 6.20 bpm between Fitzpatrick I-II and V-VI, which spans at least
75° of ITA°), i.e. 2.87 -> 4.5 bpm. tau is the SD of log MAE between people; the pilot estimates it
as the SD of the residuals of log MAE on ITA°.
H2 (penalty slope on ITA°, two-sided): additive normal model, +effect bpm across the range, residual
SD sd; equivalence margin +/- margin bpm per 75°.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

SIMS = 2000
H1_ANCHOR = (2.87, 4.5)          # bpm at ITA° 50 and -25 (ratio 1.57)


def ita_sample(rng, n):
    return np.concatenate([rng.uniform(28, 50, n), rng.uniform(10, 28, n), rng.uniform(-25, 10, n)])


def h1_power(n_per_stratum: int, tau: float = 0.55, anchor=H1_ANCHOR, sims: int = SIMS, seed: int = 1,
             endpoint: str = "log") -> float:
    rng = np.random.default_rng(seed)
    lo, hi = anchor
    hits = 0
    for _ in range(sims):
        x = ita_sample(rng, n_per_stratum)
        m = lo * (hi / lo) ** ((50 - x) / 75)
        y = m * np.exp(rng.normal(-tau ** 2 / 2, tau, x.size))
        r = stats.linregress(x, np.log(y) if endpoint == "log" else y)
        hits += stats.t.cdf(r.slope / r.stderr, x.size - 2) < 0.05
    return hits / sims


def h2_power(n_per_stratum: int, sd: float = 1.0, effect: float = 1.0, sims: int = SIMS, seed: int = 2) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(sims):
        x = ita_sample(rng, n_per_stratum)
        y = 1.0 + effect * (50 - x) / 75 + rng.normal(0, sd, x.size)
        r = stats.linregress(x, y)
        hits += 2 * stats.t.sf(abs(r.slope / r.stderr), x.size - 2) < 0.05
    return hits / sims


def sd_ci(sd: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Chi-square confidence interval for a standard deviation (assumes normal residuals; apply it
    to log-scale residuals, not to skewed raw MAE)."""
    if n is None or n < 3 or not np.isfinite(sd):
        return float("nan"), float("nan")
    df = n - 1
    a = (1 - level) / 2
    return float(sd * np.sqrt(df / stats.chi2.ppf(1 - a, df))), float(sd * np.sqrt(df / stats.chi2.ppf(a, df)))

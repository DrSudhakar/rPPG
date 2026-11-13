"""Monk Skin Tone (MST) ratings: one or two raters, consensus, agreement and use as strata.

The MST scale (Monk 2019; released by Google under CC BY 4.0) has 10 shades, 1 (lightest) to 10
(darkest). Raters compare the participant's cheek, under the session light, with the official scale
(https://skintone.google). A printed or on-screen copy changes colour with the printer or screen, so
use the same copy and the same light for every participant.

Ratings file (pilot_data/monk_ratings.csv), one row per rater per participant:
    participant_id, rater_id, monk, light, notes
Participant self-ratings (and Fitzpatrick self-report) are in pilot_data/participants.csv.

With raters.n = 1 there is no agreement statistic: the report says so instead of inventing one.
With raters.n = 2 the report gives quadratic-weighted kappa, ICC(2,1) with its 95 % CI, and the
share of exact and within-one-shade matches; the consensus is the mean of the two (or rater 1).

Monk shades are not ITA° values. Mapping them to the three ITA strata (strata.monk_cuts) is a
convention for the pilot only; the analysis reports the camera-ITA vs Monk rank correlation so the
reader can judge whether the two orderings agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sst

from .stats import icc21

# Hex colours of the 10 MST shades as distributed with the scale (for display only; rating must use
# the official scale). Verify against skintone.google before printing.
MST_HEX = ["#f6ede4", "#f3e7db", "#f7ead0", "#eadaba", "#d7bd96", "#a07e56", "#825c43", "#604134", "#3a312a", "#292420"]


def weighted_kappa(a, b, k: int = 10) -> float:
    """Cohen's kappa with quadratic weights for ordinal 1..k ratings."""
    a, b = np.asarray(a, int) - 1, np.asarray(b, int) - 1
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float("nan")


def load_ratings(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"participant_id", "rater_id", "monk"}
    if need - set(df.columns):
        raise ValueError(f"{path}: missing columns {sorted(need - set(df.columns))}")
    df["monk"] = pd.to_numeric(df["monk"], errors="coerce")
    bad = df[(df.monk < 1) | (df.monk > 10) | df.monk.isna()]
    if len(bad):
        raise ValueError(f"{path}: Monk ratings must be integers 1-10; bad rows:\n{bad}")
    return df


def consensus(df: pd.DataFrame, n_raters: int = 1, rule: str = "mean") -> pd.DataFrame:
    """One row per participant: monk (consensus), monk_r1, monk_r2 (if any), n_raters_found."""
    rows = []
    for pid, g in df.groupby("participant_id"):
        g = g.sort_values("rater_id")
        vals = g.monk.to_numpy(float)
        r1 = vals[0]
        r2 = vals[1] if len(vals) > 1 else np.nan
        if n_raters == 2 and len(vals) >= 2:
            c = r1 if rule == "rater1" else float(np.mean(vals[:2]))
        else:
            c = r1
        rows.append({"participant_id": pid, "monk": c, "monk_r1": r1, "monk_r2": r2, "n_raters_found": len(vals)})
    return pd.DataFrame(rows)


def agreement(cons: pd.DataFrame, n_raters: int) -> dict:
    if n_raters < 2:
        return {"n_raters": 1, "note": "one rater per participant: inter-rater agreement cannot be estimated"}
    both = cons.dropna(subset=["monk_r1", "monk_r2"])
    if len(both) < 3:
        return {"n_raters": 2, "n": int(len(both)), "note": "fewer than 3 participants with two ratings"}
    a, b = both.monk_r1.to_numpy(int), both.monk_r2.to_numpy(int)
    ic = icc21(np.column_stack([a, b]).astype(float))
    return {"n_raters": 2, "n": int(len(both)), "kappa_quadratic": weighted_kappa(a, b), "icc21": ic["icc"],
            "icc21_lo95": ic["lo95"], "icc21_hi95": ic["hi95"], "exact": float(np.mean(a == b)),
            "within_one": float(np.mean(np.abs(a - b) <= 1)), "mean_abs_diff": float(np.mean(np.abs(a - b)))}


def stratum_from_monk(m: float, cuts=(4, 7)) -> str:
    if not np.isfinite(m):
        return "unknown"
    lo, hi = cuts
    r = int(round(m))
    return "lighter" if r <= lo else ("middle" if r <= hi else "darker")


def ita_vs_monk(ita: np.ndarray, monk: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Spearman correlation between camera ITA° and Monk shade; expected negative (darker = higher
    Monk = lower ITA°). Participant bootstrap CI."""
    ita, monk = np.asarray(ita, float), np.asarray(monk, float)
    ok = np.isfinite(ita) & np.isfinite(monk)
    ita, monk = ita[ok], monk[ok]
    n = len(ita)
    if n < 4:
        return {"n": int(n), "note": "fewer than 4 participants with both values"}
    rho = sst.spearmanr(ita, monk).statistic
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.unique(monk[i]).size > 1 and np.unique(ita[i]).size > 1:
            bs.append(sst.spearmanr(ita[i], monk[i]).statistic)
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return {"n": int(n), "spearman_rho": float(rho), "lo95": float(lo), "hi95": float(hi)}

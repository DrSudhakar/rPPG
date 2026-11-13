"""SAMA step 6: equal-reliability gate.

Group-conditional (Mondrian) split conformal prediction with normalised scores
s = |h_hat - h| / sigma_hat(x) (proposal s.4 WP3 step 6). The quantile per stratum g is the
ceil(0.9 (n_g + 1))-th smallest score; calibration uses one window per participant, drawn 200
times; the 200 quantiles are averaged (ambiguity logged as ISSUES I-29, DECISIONS D-10).
A reading is shown only if the half-width q_g * sigma_hat(x) <= 5 bpm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ABSTAIN_HALFWIDTH = 5.0   # spec.yaml step6_gate.abstain_halfwidth_bpm
ALPHA = 0.10


def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    s = np.sort(np.asarray(scores, float)[np.isfinite(scores)])
    n = s.size
    k = math.ceil((1 - alpha) * (n + 1))
    return float(s[k - 1]) if k <= n else float("inf")


@dataclass
class MondrianGate:
    alpha: float = ALPHA
    halfwidth_max: float = ABSTAIN_HALFWIDTH
    draws: int = 200
    q: dict = field(default_factory=dict)
    q_draws: dict = field(default_factory=dict)

    def fit(self, h_hat, h_ref, sigma, group, participant, seed: int = 0, aggregate: str = "mean"):
        h_hat, h_ref, sigma = (np.asarray(v, float) for v in (h_hat, h_ref, sigma))
        group, participant = np.asarray(group), np.asarray(participant)
        s = np.abs(h_hat - h_ref) / np.maximum(sigma, 1e-6)
        ok = np.isfinite(s)
        rng = np.random.default_rng(seed)
        for g in np.unique(group[ok]):
            m = ok & (group == g)
            pids = np.unique(participant[m])
            idx_by_p = [np.where(m & (participant == p))[0] for p in pids]
            qs = []
            for _ in range(self.draws):
                pick = np.array([ix[rng.integers(0, ix.size)] for ix in idx_by_p])
                qs.append(conformal_quantile(s[pick], self.alpha))
            qs = np.array(qs)
            self.q_draws[str(g)] = qs
            if not np.isfinite(qs).all():      # too few calibration participants (n_g < 9 at alpha 0.1)
                self.q[str(g)] = float("inf")  # infinite interval: covers, but the gate always abstains
            else:
                self.q[str(g)] = float(np.mean(qs)) if aggregate == "mean" else float(np.median(qs))
        return self

    def interval(self, h_hat, sigma, group):
        h_hat, sigma = np.asarray(h_hat, float), np.asarray(sigma, float)
        q = np.array([self.q.get(str(g), np.inf) for g in np.asarray(group)])
        hw = q * sigma
        return h_hat - hw, h_hat + hw, hw

    def decide(self, h_hat, sigma, group):
        """Returns (lo, hi, halfwidth, shown) where shown is False for 'no reliable reading'."""
        lo, hi, hw = self.interval(h_hat, sigma, group)
        shown = np.isfinite(hw) & (hw <= self.halfwidth_max) & np.isfinite(h_hat)
        return lo, hi, hw, shown

    def to_dict(self):
        return {"alpha": self.alpha, "halfwidth_max": self.halfwidth_max, "draws": self.draws, "q": self.q}

    @classmethod
    def from_dict(cls, d):
        g = cls(alpha=d["alpha"], halfwidth_max=d["halfwidth_max"], draws=d["draws"])
        g.q = dict(d["q"])
        return g


def coverage_report(h_ref, lo, hi, shown, group, participant) -> dict:
    """Coverage on all windows and on shown windows, abstention rate, useful yield
    (shown and within ±5 bpm), per group."""
    h_ref, lo, hi = (np.asarray(v, float) for v in (h_ref, lo, hi))
    shown, group = np.asarray(shown, bool), np.asarray(group)
    out = {}
    for g in np.unique(group):
        m = (group == g) & np.isfinite(h_ref)
        cov_all = ((h_ref >= lo) & (h_ref <= hi))[m]
        cov_shown = ((h_ref >= lo) & (h_ref <= hi))[m & shown]
        with np.errstate(invalid="ignore"):
            mid = (lo + hi) / 2
        useful = (shown & (np.abs(mid - h_ref) <= 5.0))[m]
        out[str(g)] = {"n": int(m.sum()), "coverage_all": float(cov_all.mean()) if cov_all.size else np.nan,
                       "coverage_shown": float(cov_shown.mean()) if cov_shown.size else np.nan,
                       "abstention": float(1 - shown[m].mean()) if m.any() else np.nan,
                       "useful_yield": float(useful.mean()) if useful.size else np.nan,
                       "median_halfwidth": float(np.nanmedian(((hi - lo) / 2)[m]))}
    return out

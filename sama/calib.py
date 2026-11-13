"""SAMA step 2: colour-chart linearisation and a measured photon-transfer noise model.

- fit_tone_curve: from the chart's known reflectances and their camera DN, fit DN = s * R^(1/g)
  in log space and invert to linear units.
- photon_transfer: from static chart frames, fit per-pixel temporal variance = a * mean + b
  (shot noise slope a = DN per electron, intercept b = read + ADC noise), in linear DN.
- patch_noise_var: expected variance of a skin patch's mean, the step 3 weight denominator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToneCurve:
    gamma: float
    scale: float

    def linearise(self, dn: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(dn, float) / self.scale, 1e-6, None)
        return 255.0 * x ** self.gamma


def fit_tone_curve(chart_dn: np.ndarray, chart_reflectance: np.ndarray) -> ToneCurve:
    """chart_dn, chart_reflectance: (K,) green-channel means of the 24 patches (K >= 3).
    Fits log DN = log s + (1/g) log R on unclipped patches."""
    dn, r = np.asarray(chart_dn, float), np.asarray(chart_reflectance, float)
    ok = (dn > 3) & (dn < 250) & (r > 0)
    k, c = np.polyfit(np.log(r[ok]), np.log(dn[ok]), 1)
    return ToneCurve(gamma=float(1.0 / k), scale=float(np.exp(c)))


@dataclass
class NoiseModel:
    a: float  # DN^2 per DN (shot-noise slope, linear DN)
    b: float  # DN^2 (read + ADC)

    def pixel_var(self, mean_lin_dn):
        return self.a * np.asarray(mean_lin_dn, float) + self.b

    def patch_var(self, mean_lin_dn, n_pix):
        return self.pixel_var(mean_lin_dn) / np.asarray(n_pix, float)


def photon_transfer(static_patch_traces: np.ndarray, n_pix_per_patch: float) -> NoiseModel:
    """static_patch_traces: (T, K) linear-DN patch means of a static chart at one setting.
    Temporal variance of a patch mean x n_pix gives per-pixel variance; regress on the mean."""
    x = np.asarray(static_patch_traces, float)
    mu = x.mean(0)
    var_pix = x.var(0, ddof=1) * n_pix_per_patch
    A = np.column_stack([mu, np.ones_like(mu)])
    (a, b), *_ = np.linalg.lstsq(A, var_pix, rcond=None)
    return NoiseModel(float(max(a, 1e-9)), float(max(b, 0.0)))

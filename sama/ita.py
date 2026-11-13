"""Individual Typology Angle from colorimeter CIELAB or from calibrated camera RGB.

ITA° = arctan((L* - 50) / b*) * 180 / pi (Chardon et al. 1991). Strata (spec.yaml):
lighter > 28, middle 10-28, darker <= 10. Camera estimates are approximate (ISSUES I-03).
"""
from __future__ import annotations

import math

import numpy as np

M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def ita_from_lab(L: float, b: float) -> float:
    return math.degrees(math.atan2(L - 50.0, b)) if b != 0 else float("nan")


def lab_from_linear_rgb(rgb_lin) -> tuple[float, float, float]:
    xyz = M_RGB2XYZ @ np.asarray(rgb_lin, float) / WHITE_D65
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return float(116 * f[1] - 16), float(500 * (f[0] - f[1])), float(200 * (f[1] - f[2]))


def ita_from_camera(skin_lin_rgb, white_lin_rgb=None, white_reflectance: float = 0.90) -> float:
    """ITA° from a skin patch's mean linear RGB, white-balanced to a neutral chart patch in the
    same frame (as scripts/ita_from_patch.py, which this generalises)."""
    s = np.asarray(skin_lin_rgb, float)
    if white_lin_rgb is not None:
        s = s / np.asarray(white_lin_rgb, float) * white_reflectance
    L, _, b = lab_from_linear_rgb(s)
    return ita_from_lab(L, b)


def stratum(ita: float) -> str:
    if not np.isfinite(ita):
        return "unknown"
    return "lighter" if ita > 28 else ("middle" if ita > 10 else "darker")


STRATA = ("lighter", "middle", "darker")

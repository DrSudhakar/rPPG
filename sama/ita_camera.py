"""Skin tone (ITA°) estimated from video, for when no colorimeter is available.

ITA° = arctan((L* - 50) / b*) x 180/pi (Chardon et al. 1991), with L* and b* in CIELAB (D65).
A camera does not report CIELAB: its colours depend on the light and on its own white balance.
So the estimate is only as good as the reference in the frame:

  card   a matte white or mid-grey card beside the cheek, at the same distance and angle. Each
         channel of the skin colour is divided by the card's colour and multiplied by the card's
         reflectance (von Kries white balance). Recommended: costs nothing and removes most of
         the light and white-balance error.
  chart  a 24-patch colour chart. A 3x3 colour-correction matrix is fitted from the 24 patches
         to their nominal sRGB values (replace NOMINAL_CHART_SRGB with your chart's data sheet).
  none   no reference: the camera's automatic white balance is trusted. Flagged NO_WHITE_REF.
         The UBFC-rPPG values of 0-118° seen earlier show what this does (I-03).

Robustness: skin pixels come from the cheek (and optionally forehead) patches only; a YCbCr skin
rule masks hair, eyebrows and background; clipped pixels are dropped; the darkest and brightest
`trim_pct` per cent are dropped symmetrically (shadow, stubble, highlights). Each sampled frame gives one ITA°;
the estimate is the median over frames, and the interquartile range shows its stability.

This is an approximation of the colorimeter reading the proposal specifies. Every value it
produces is labelled ita_source=camera, and the pilot reports how it agrees with Monk ratings.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .ita import lab_from_linear_rgb, stratum
from .model_io import LAYOUT
from .video import CHEEKS, FOREHEAD, crop, skin_mask

# Nominal sRGB (8-bit) of the 24 ColorChecker Classic patches, row by row from "dark skin" to
# "black". Values as commonly published for the post-2014 chart. ASSUMPTION: replace with the
# values printed on your own chart's data sheet before relying on the chart mode.
NOMINAL_CHART_SRGB = np.array([
    [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170],
    [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
    [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31], [187, 86, 149], [8, 133, 161],
    [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52]], float)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def ita_from_linear(rgb_lin) -> tuple[float, float, float, float]:
    L, a, b = lab_from_linear_rgb(np.clip(rgb_lin, 1e-6, None))
    ita = math.degrees(math.atan2(L - 50.0, b)) if b != 0 else float("nan")
    return ita, L, a, b


def skin_pixels(frame: np.ndarray, box, regions=("cheeks",), shadow_pct: float = 5.0, specular: bool = True) -> np.ndarray:
    """(N, 3) uint8-scale skin pixels from the chosen patch groups of the face box. The darkest and (if
    `specular`) the brightest `shadow_pct` per cent by luma are dropped symmetrically, so the median
    used for ITA° is not pushed lighter (trimming only the dark end biases ITA° upwards)."""
    roi = crop(frame, box).astype(float)
    H, W, _ = roi.shape
    idx = (CHEEKS if "cheeks" in regions else []) + (FOREHEAD if "forehead" in regions else [])
    out = []
    for i in idx:
        x, y, w, h = LAYOUT[i]
        p = roi[int(y * H):max(int((y + h) * H), int(y * H) + 1), int(x * W):max(int((x + w) * W), int(x * W) + 1)]
        m = skin_mask(p) & (p.max(-1) < 250) & (p.min(-1) > 3)
        out.append(p[m])
    px = np.concatenate(out) if out else np.zeros((0, 3))
    if len(px) < 20:
        return px
    luma = px @ np.array([0.299, 0.587, 0.114])
    keep = luma >= np.percentile(luma, shadow_pct)
    if specular:
        keep &= luma <= np.percentile(luma, 100 - shadow_pct)
    return px[keep]


def chart_patch_means(frame: np.ndarray, corners) -> np.ndarray:
    """24 patch colours (8-bit RGB) from the chart's four corners [(x, y) tl, tr, br, bl] in
    frame fractions; samples the central 40 % of each patch on a 6 x 4 grid."""
    H, W = frame.shape[:2]
    c = np.asarray(corners, float) * [W, H]
    out = []
    for r in range(4):
        for k in range(6):
            u, v = (k + 0.5) / 6, (r + 0.5) / 4
            top = c[0] + u * (c[1] - c[0])
            bot = c[3] + u * (c[2] - c[3])
            cx, cy = top + v * (bot - top)
            hw, hh = 0.2 * np.linalg.norm(c[1] - c[0]) / 6, 0.2 * np.linalg.norm(c[3] - c[0]) / 4
            p = frame[int(max(0, cy - hh)):int(cy + hh) + 1, int(max(0, cx - hw)):int(cx + hw) + 1].reshape(-1, 3)
            out.append(np.median(p, 0))
    return np.array(out, float)


def fit_ccm(measured_srgb: np.ndarray, reference_srgb: np.ndarray = NOMINAL_CHART_SRGB) -> np.ndarray:
    """3x3 matrix M with ref_lin ≈ meas_lin @ M.T, least squares on unclipped patches."""
    m, r = srgb_to_linear(measured_srgb), srgb_to_linear(reference_srgb)
    ok = (measured_srgb.max(1) < 250) & (measured_srgb.min(1) > 3)
    X, *_ = np.linalg.lstsq(m[ok], r[ok], rcond=None)
    return X.T


@dataclass
class ItaEstimate:
    ita: float
    iqr: float
    L: float
    a: float
    b: float
    n_frames: int
    method: str
    flags: list = field(default_factory=list)
    stratum: str = "unknown"
    per_frame: list = field(default_factory=list)

    def to_dict(self, per_frame: bool = False):
        d = asdict(self)
        if not per_frame:
            d.pop("per_frame")
        return d


def estimate(frames, boxes, white_box=None, chart_corners=None, method: str = "card", reflectance: float = 0.18,
             regions=("cheeks",), shadow_pct: float = 5.0, specular: bool = True, max_iqr: float = 6.0,
             cuts=(10, 28)) -> ItaEstimate:
    """frames: iterable of RGB uint8 frames; boxes: one face box per frame (or a single box)."""
    flags, vals = [], []
    ccm = None
    boxes = np.asarray(boxes, float)
    for i, fr in enumerate(frames):
        box = boxes if boxes.ndim == 1 else boxes[min(i, len(boxes) - 1)]
        px = skin_pixels(fr, box, regions, shadow_pct, specular)
        if len(px) < 50:
            continue
        s = np.median(srgb_to_linear(px), 0)
        if method == "card":
            if white_box is None:
                raise ValueError("ita.camera.white_reference=card needs a white-card box (draw it in the recorder)")
            wp = crop(fr, white_box).reshape(-1, 3).astype(float)
            if (wp.max(1) >= 250).mean() > 0.05:
                flags.append("WHITE_CLIPPED")
                continue
            w = np.median(srgb_to_linear(wp), 0)
            if w.min() < 0.03:
                flags.append("WHITE_TOO_DARK")
                continue
            s = s / w * reflectance
        elif method == "chart":
            if chart_corners is None:
                raise ValueError("ita.camera.white_reference=chart needs the chart corners (draw them in the recorder)")
            if ccm is None:
                ccm = fit_ccm(chart_patch_means(fr, chart_corners))
            s = ccm @ s
        elif method != "none":
            raise ValueError(method)
        ita, L, a, b = ita_from_linear(s)
        if np.isfinite(ita):
            vals.append((ita, L, a, b))
    if method == "none":
        flags.append("NO_WHITE_REF")
    if not vals:
        return ItaEstimate(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0, method,
                           sorted(set(flags + ["NO_SKIN_PIXELS"])))
    v = np.array(vals)
    q1, q3 = np.percentile(v[:, 0], [25, 75])
    est = ItaEstimate(float(np.median(v[:, 0])), float(q3 - q1), float(np.median(v[:, 1])), float(np.median(v[:, 2])),
                      float(np.median(v[:, 3])), len(v), method, sorted(set(flags)), per_frame=v[:, 0].round(2).tolist())
    if est.iqr > max_iqr:
        est.flags.append("ITA_UNSTABLE")
    if est.L > 85 or est.L < 15:
        est.flags.append("L_OUT_OF_SKIN_RANGE")
    if not -60 <= est.ita <= 75:
        est.flags.append("ITA_OUT_OF_HUMAN_RANGE")
    est.stratum = stratum_from(est.ita, cuts)
    return est


def stratum_from(ita: float, cuts=(10, 28)) -> str:
    if not np.isfinite(ita):
        return "unknown"
    lo, hi = cuts
    return "lighter" if ita > hi else ("middle" if ita > lo else "darker")


__all__ = ["estimate", "ItaEstimate", "fit_ccm", "chart_patch_means", "skin_pixels", "srgb_to_linear",
           "ita_from_linear", "stratum_from", "NOMINAL_CHART_SRGB", "stratum"]

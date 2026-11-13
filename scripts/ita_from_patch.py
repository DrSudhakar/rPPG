"""Approximate ITA (Individual Typology Angle) from a cropped skin patch in a photo or video frame.

For the pilot only: a colorimeter is the reference. Colour-correct with a white or grey patch
from the colour chart in the same frame, so the camera's white balance does not shift the result.

Usage
  python ita_from_patch.py skin.png --white white_patch.png
  python ita_from_patch.py skin.png --white white_patch.png --white-reflectance 0.9

ITA = arctan((L* - 50) / b*) * 180 / pi   (Chardon et al. 1991), L*, b* in CIELAB (D65).
Requires numpy and Pillow.
"""
import argparse
import math
import numpy as np
from PIL import Image


def srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


M = np.array([[0.4124564, 0.3575761, 0.1804375],
              [0.2126729, 0.7151522, 0.0721750],
              [0.0193339, 0.1191920, 0.9503041]])
WHITE = np.array([0.95047, 1.0, 1.08883])  # D65


def lab(rgb_lin):
    xyz = M @ rgb_lin / WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])


def mean_linear(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=float).reshape(-1, 3)
    lin = srgb_to_linear(a)
    keep = (a.max(1) < 250) & (a.min(1) > 5)  # drop clipped and black pixels
    return lin[keep].mean(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("skin")
    ap.add_argument("--white", help="crop of a white/neutral chart patch in the same frame")
    ap.add_argument("--white-reflectance", type=float, default=0.90,
                    help="reflectance of that patch (check your chart's data sheet)")
    a = ap.parse_args()
    s = mean_linear(a.skin)
    if a.white:
        w = mean_linear(a.white)
        s = s / w * a.white_reflectance  # per-channel white balance to the patch's known reflectance
    L, A, B = lab(s)
    ita = math.degrees(math.atan2(L - 50, B)) if B != 0 else float("nan")
    print(f"L*={L:.1f} a*={A:.1f} b*={B:.1f}  ITA={ita:.1f} deg")

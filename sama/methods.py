"""The six unsupervised rPPG methods of the proposal's baseline set.

Each takes RGB traces `rgb` of shape (T, 3) (spatial means of skin pixels per frame, any
scale) and the frame rate, and returns a pulse signal of length T. POS, CHROM, GREEN, LGI and
PBV follow the rPPG-Toolbox (Liu et al. 2023) implementations step for step and are checked
against them in tests/test_toolbox_equivalence.py. ICA is a FastICA variant (the toolbox uses
JADE), so it is not claimed equivalent (DECISIONS.md D-08).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

from .signals import detrend_tarvainen

METHODS = ("GREEN", "ICA", "CHROM", "POS", "LGI", "PBV")


def green(rgb: np.ndarray, fs: float) -> np.ndarray:
    return np.asarray(rgb, float)[:, 1].copy()


def pos(rgb: np.ndarray, fs: float, win_s: float = 1.6, post: bool = True) -> np.ndarray:
    """Plane-orthogonal-to-skin (Wang et al. 2017), overlap-add over 1.6 s windows."""
    rgb = np.asarray(rgb, float)
    N = rgb.shape[0]
    L = math.ceil(win_s * fs)
    H = np.zeros(N)
    P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    for n in range(L, N):
        m = n - L
        C = rgb[m:n] / rgb[m:n].mean(axis=0)
        S = P @ C.T
        s1 = S[1].std()
        h = S[0] + (S[0].std() / s1 if s1 > 0 else 0.0) * S[1]
        H[m:n] += h - h.mean()
    if not post:
        return H
    H = detrend_tarvainen(H, 100.0)
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype="bandpass")
    return signal.filtfilt(b, a, H)


def chrom(rgb: np.ndarray, fs: float, win_s: float = 1.6) -> np.ndarray:
    """Chrominance method (de Haan & Jeanne 2013), Hann overlap-add."""
    rgb = np.asarray(rgb, float)
    FN = rgb.shape[0]
    B, A = signal.butter(3, [0.7 / (fs / 2), 2.5 / (fs / 2)], "bandpass")
    WinL = math.ceil(win_s * fs)
    if WinL % 2:
        WinL += 1
    NWin = math.floor((FN - WinL // 2) / (WinL // 2))
    S = np.zeros((WinL // 2) * (NWin + 1))
    WinS, WinM, WinE = 0, WinL // 2, WinL
    hann = signal.windows.hann(WinL)
    for _ in range(NWin):
        norm = rgb[WinS:WinE] / rgb[WinS:WinE].mean(axis=0)
        Xs = 3 * norm[:, 0] - 2 * norm[:, 1]
        Ys = 1.5 * norm[:, 0] + norm[:, 1] - 1.5 * norm[:, 2]
        Xf = signal.filtfilt(B, A, Xs)
        Yf = signal.filtfilt(B, A, Ys)
        sw = (Xf - (Xf.std() / Yf.std() if Yf.std() > 0 else 0.0) * Yf) * hann
        S[WinS:WinM] += sw[: WinL // 2]
        S[WinM:WinE] = sw[WinL // 2:]
        WinS, WinM, WinE = WinM, WinM + WinL // 2, WinM + WinL
    out = np.zeros(FN)
    out[: min(FN, S.size)] = S[:FN]
    return out


def lgi(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Local group invariance (Pilz et al. 2018): project out the dominant RGB direction."""
    X = np.asarray(rgb, float).T  # 3 x T
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    s = U[:, :1]
    P = np.eye(3) - s @ s.T
    return (P @ X)[1]


def pbv(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Blood-volume-pulse signature (de Haan & van Leest 2014), data-derived signature as in
    rPPG-Toolbox."""
    X = np.asarray(rgb, float).T
    Xn = X / X.mean(axis=1, keepdims=True)
    pbv_n = Xn.std(axis=1)
    sig = pbv_n / np.sqrt((Xn.var(axis=1)).sum())
    Q = Xn @ Xn.T
    W = np.linalg.solve(Q, sig)
    return (Xn.T @ W) / (sig @ W)


def ica(rgb: np.ndarray, fs: float, seed: int = 0) -> np.ndarray:
    """FastICA on normalised, detrended RGB; returns the component with the largest
    in-band spectral peak share (Poh et al. 2010 selection rule)."""
    from sklearn.decomposition import FastICA

    X = np.asarray(rgb, float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    X = np.column_stack([detrend_tarvainen(X[:, k], 100.0) for k in range(3)])
    try:
        S = FastICA(n_components=3, random_state=seed, whiten="unit-variance", max_iter=400).fit_transform(X)
    except Exception:  # degenerate input
        return X[:, 1]
    best, score = S[:, 1], -1.0
    for k in range(3):
        f, p = signal.periodogram(S[:, k], fs=fs, nfft=2048)
        m = (f >= 0.7) & (f <= 3.0)
        sc = p[m].max() / (p.sum() + 1e-30)
        if sc > score:
            best, score = S[:, k], sc
    return best


_FUN = {"GREEN": green, "ICA": ica, "CHROM": chrom, "POS": pos, "LGI": lgi, "PBV": pbv}


def run(method: str, rgb: np.ndarray, fs: float) -> np.ndarray:
    """Run a named method; NaN-safe (returns zeros if input is not finite)."""
    rgb = np.asarray(rgb, float)
    if not np.isfinite(rgb).all() or (rgb <= 0).any():
        rgb = np.where(np.isfinite(rgb) & (rgb > 0), rgb, np.nanmean(rgb[rgb > 0]) if (rgb > 0).any() else 1.0)
    return _FUN[method.upper()](rgb, fs)

"""Signal utilities: filtering, windowing, spectral heart rate, beat timing and HRV.

All functions are pure numpy/scipy and deterministic.
"""
from __future__ import annotations

import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve

BAND_HZ = (0.7, 3.0)  # spec.yaml windows.band_hz (proposal s.4 WP3 step 5)


def bandpass(x: np.ndarray, fs: float, lo: float = BAND_HZ[0], hi: float = BAND_HZ[1], order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis."""
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    padlen = min(x.shape[-1] - 1, 3 * (2 * order + 1) * 4)
    return signal.sosfiltfilt(sos, x, axis=-1, padlen=max(padlen, 0))


def detrend_tarvainen(x: np.ndarray, lam: float = 100.0) -> np.ndarray:
    """Smoothness-priors detrending (Tarvainen et al. 2002), sparse solve; same result as
    rPPG-Toolbox utils.detrend but O(n) memory."""
    n = x.shape[0]
    if n < 3:
        return x - x.mean()
    e = np.ones(n)
    D = sparse.spdiags([e, -2 * e, e], [0, 1, 2], n - 2, n)
    A = sparse.identity(n) + lam ** 2 * (D.T @ D)
    trend = spsolve(A.tocsc(), x)
    return x - trend


def window_starts(n_samples: int, fs: float, length_s: float, step_s: float) -> list[tuple[int, int]]:
    """Index ranges [a, b) of full windows of length_s seconds every step_s seconds."""
    L, S = int(round(length_s * fs)), int(round(step_s * fs))
    return [(a, a + L) for a in range(0, n_samples - L + 1, S)]


def spectrum(x: np.ndarray, fs: float, nfft_min: int = 2048):
    """Hann-windowed periodogram, zero-padded to at least nfft_min."""
    n = x.shape[-1]
    nfft = max(nfft_min, 1 << int(np.ceil(np.log2(n))))
    f, p = signal.periodogram(x, fs=fs, window="hann", nfft=nfft, detrend="constant", axis=-1)
    return f, p


def hr_from_signal(x: np.ndarray, fs: float, band=BAND_HZ) -> tuple[float, float, float]:
    """Heart rate (bpm) from the spectral peak in band, with parabolic refinement.

    Returns (hr_bpm, snr_db, peak_sharpness). SNR follows de Haan & Jeanne (2013): power
    within ±0.1 Hz of the peak and of its first harmonic over the rest of the band.
    """
    f, p = spectrum(x, fs)
    m = (f >= band[0]) & (f <= band[1])
    fb, pb = f[m], p[m]
    if pb.size < 3 or not np.isfinite(pb).all() or pb.max() <= 0:
        return float("nan"), float("-inf"), 0.0
    k = int(np.argmax(pb))
    f0 = fb[k]
    if 0 < k < pb.size - 1:  # parabolic interpolation of the peak
        a, b, c = np.log(pb[k - 1] + 1e-30), np.log(pb[k] + 1e-30), np.log(pb[k + 1] + 1e-30)
        den = a - 2 * b + c
        if den < 0:
            f0 = fb[k] + 0.5 * (a - c) / den * (fb[1] - fb[0])
    sig = (np.abs(f - f0) <= 0.1) | (np.abs(f - 2 * f0) <= 0.2)
    wide = (f >= band[0]) & (f <= min(2 * band[1], fs / 2))
    ps = p[sig & wide].sum()
    pn = p[wide & ~sig].sum()
    snr = 10 * np.log10(ps / pn) if pn > 0 and ps > 0 else float("-inf")
    sharp = float(pb[k] / (pb.sum() + 1e-30))
    return float(f0 * 60.0), float(snr), sharp


def peaks_parabolic(x: np.ndarray, fs: float, hr_hint_bpm: float | None = None) -> np.ndarray:
    """Pulse peak times (s) with parabolic sub-sample interpolation (step 5).

    Minimum peak distance is 60 % of the expected beat interval (from hr_hint, else 180 bpm).
    """
    hr = hr_hint_bpm if hr_hint_bpm and np.isfinite(hr_hint_bpm) else 180.0
    dist = max(1, int(0.6 * fs * 60.0 / hr))
    idx, _ = signal.find_peaks(x, distance=dist, prominence=0.15 * np.std(x))
    idx = idx[(idx > 0) & (idx < len(x) - 1)]
    a, b, c = x[idx - 1], x[idx], x[idx + 1]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    off = np.clip(off, -0.5, 0.5)
    t = (idx + off) / fs
    return np.round(t, 3)  # 1 ms resolution (spec step5 resolution_ms)


def clean_ibi(ibi_s: np.ndarray, lo: float = 0.3, hi: float = 2.0, rel: float = 0.2) -> np.ndarray:
    """Artefact rules applied identically to ECG and camera IBIs: physiological range, then
    removal of intervals deviating more than `rel` from the local median of 5 (Malik-type rule)."""
    ibi = np.asarray(ibi_s, float)
    ok = (ibi >= lo) & (ibi <= hi)
    ibi = ibi[ok]
    if ibi.size < 5:
        return ibi
    med = signal.medfilt(ibi, 5)
    return ibi[np.abs(ibi - med) <= rel * med]


def rmssd_ms(ibi_s: np.ndarray) -> float:
    ibi = np.asarray(ibi_s, float)
    if ibi.size < 3:
        return float("nan")
    return float(1000.0 * np.sqrt(np.mean(np.diff(ibi) ** 2)))


def sdnn_ms(ibi_s: np.ndarray) -> float:
    ibi = np.asarray(ibi_s, float)
    if ibi.size < 3:
        return float("nan")
    return float(1000.0 * np.std(ibi, ddof=1))


def hr_from_beats(beat_t: np.ndarray, a: float, b: float) -> float:
    """Reference heart rate in [a, b) seconds from beat times: 60 / mean IBI of beats inside."""
    t = beat_t[(beat_t >= a) & (beat_t < b)]
    if t.size < 3:
        return float("nan")
    ibi = clean_ibi(np.diff(t))
    return float(60.0 / ibi.mean()) if ibi.size else float("nan")

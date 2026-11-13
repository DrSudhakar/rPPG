"""Offline pipeline: RGB traces -> pulse signal (any method) -> windowed heart rate against the
reference beats. Used for every tier (REAL-PUBLIC loaders produce the same inputs)."""
from __future__ import annotations

import numpy as np

from . import methods as M
from .signals import hr_from_beats, hr_from_signal, window_starts

HR_WIN = (30.0, 15.0)   # spec.yaml windows.hr
HRV_WIN = 60.0


def ref_hr_windows(beats: np.ndarray, beats_valid: np.ndarray | None, n_samples: int, fps: float,
                   length_s: float = HR_WIN[0], step_s: float = HR_WIN[1]):
    """Reference HR per window from ECG beats; a window is unusable if >10 % of its beats are
    flagged (pre-set ECG quality rule, proposal s.6 'Missing data')."""
    out = []
    for a, b in window_starts(n_samples, fps, length_s, step_s):
        ta, tb = a / fps, b / fps
        inside = (beats >= ta) & (beats < tb)
        ok = True
        if beats_valid is not None and inside.any():
            ok = (~beats_valid[inside]).mean() <= 0.10
        out.append((a, b, hr_from_beats(beats[inside & (beats_valid if beats_valid is not None else True)], ta, tb) if ok else np.nan))
    return out


def windowed_hr(pulse: np.ndarray, fps: float, ref: list) -> list[dict]:
    rows = []
    for a, b, h_ref in ref:
        h, snr, sharp = hr_from_signal(pulse[a:b], fps)
        rows.append({"t0": a / fps, "hr_cam": h, "hr_ref": h_ref, "snr_db": snr, "sharp": sharp})
    return rows


def run_methods(rgb: np.ndarray, fps: float, beats: np.ndarray, beats_valid=None, methods=M.METHODS) -> dict:
    """All baselines on one recording: {method: [window rows]}."""
    ref = ref_hr_windows(beats, beats_valid, rgb.shape[0], fps)
    return {m: windowed_hr(M.run(m, rgb, fps), fps, ref) for m in methods}

"""Aligning a camera stream with the reference.

Step 1 (wall clock): the recorder writes the PC time of its first frame; oximeter software on the
same PC writes PC time stamps. Their difference is the starting alignment.
Step 2 (cross-correlation, sync.method = xcorr): within each 30 s window the camera pulse (POS,
band-passed) is cross-correlated with the reference waveform over +/- max_lag_s around step 1. The
alignment used is the median of the per-window lags whose correlation peak is clear.

The spread of the per-window lags (robust SD) is reported as the synchronisation error, the
"[x] ms" of proposal section 5. A constant offset between face and finger pulse arrival (pulse
transit time, ~0.1-0.2 s) is absorbed into the alignment and does not change heart rate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import methods as M
from .signals import bandpass


@dataclass
class SyncResult:
    lag_s: float                 # add to camera time to get reference time
    spread_ms: float             # robust SD of per-window lags (nan if not estimable)
    n_windows: int
    method: str
    wall_offset_s: float
    per_window: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def orient(x: np.ndarray) -> np.ndarray:
    """Flip a pulse so that systolic peaks point up: the upstroke is steeper than the decline, so
    the derivative of a correctly oriented pulse is positively skewed. Camera rPPG and oximeter
    exports differ in polarity; orienting both avoids a half-period alignment error."""
    from scipy.stats import skew
    d = np.diff(x)
    return x if (not d.size or skew(d) >= 0) else -x


def camera_pulse(t: np.ndarray, patches: np.ndarray, fs: float = 30.0):
    rgb = patches.mean(1)
    g = np.arange(t[0], t[-1], 1.0 / fs)
    u = np.column_stack([np.interp(g, t, rgb[:, c]) for c in range(3)])
    return g, orient(bandpass(M.pos(u, fs), fs))


def align(cam_t: np.ndarray, cam_patches: np.ndarray, cam_t0_wall: float, ref, method: str = "xcorr",
          max_lag_s: float = 3.0, window_s: float = 30.0, fs: float = 60.0, min_peak: float = 0.3) -> SyncResult:
    """ref: sama.reference.Reference (its own clock starts at ref.t0_wall)."""
    notes = []
    if np.isfinite(cam_t0_wall) and np.isfinite(ref.t0_wall):
        wall = float(cam_t0_wall - ref.t0_wall)     # camera t=0 is at reference time `wall`
    else:
        wall = 0.0
        notes.append("no wall-clock times: initial alignment assumes both started together")
    if method == "wallclock" or ref.wave is None:
        if method == "xcorr":
            notes.append("reference has no waveform: wall-clock alignment only")
        return SyncResult(wall, float("nan"), 0, "wallclock", wall, notes=notes)
    gc, pc = camera_pulse(cam_t, cam_patches, fs)
    rw = orient(bandpass(np.interp(np.arange(ref.wave_t[0], ref.wave_t[-1], 1.0 / fs), ref.wave_t, ref.wave), fs))
    rt = np.arange(ref.wave_t[0], ref.wave_t[-1], 1.0 / fs)[: rw.size]
    def per_window(center, half):
        out = []
        for a in np.arange(gc[0], gc[-1] - window_s, window_s / 2):
            sel = (gc >= a) & (gc < a + window_s)
            c = pc[sel]
            if c.size < window_s * fs * 0.9:
                continue
            ra, rb = a + center - half, a + window_s + center + half
            rs = (rt >= ra) & (rt < rb)
            r = rw[rs]
            if r.size < c.size + 2:
                continue
            c = (c - c.mean()) / (c.std() + 1e-12)
            cc = np.array([np.dot(c, (r[k:k + c.size] - r[k:k + c.size].mean()) / (r[k:k + c.size].std() + 1e-12)) / c.size
                           for k in range(0, r.size - c.size + 1)])
            bk = int(np.argmax(cc))
            off = 0.0
            if 0 < bk < cc.size - 1:          # parabolic interpolation of the peak: sub-sample lag
                y0, y1, y2 = cc[bk - 1], cc[bk], cc[bk + 1]
                den = y0 - 2 * y1 + y2
                off = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den < 0 else 0.0
            edge = bk in (0, cc.size - 1)
            out.append({"t0": float(a), "lag_s": float(rt[rs][0] + (bk + off) / fs - a), "peak": float(cc[bk]), "edge": bool(edge)})
        return out

    # pass 1: wide search around the wall clock; the median is robust to windows that lock one beat off
    per1 = [p for p in per_window(wall, max_lag_s) if p["peak"] >= min_peak]
    if len(per1) < 2:
        notes.append("cross-correlation unclear: wall-clock alignment kept")
        return SyncResult(wall, float("nan"), len(per1), "wallclock", wall, per1, notes)
    med1 = float(np.median([p["lag_s"] for p in per1]))
    # pass 2: narrow search (+/- 0.3 s, less than one beat) around that median
    per = per_window(med1, 0.3)
    good = [p for p in per if p["peak"] >= min_peak and not p["edge"]]
    if len(good) < 2:
        return SyncResult(med1, float("nan"), len(good), "xcorr", wall, per, notes)
    lags = np.array([p["lag_s"] for p in good])
    med = float(np.median(lags))
    spread = float(1.4826 * np.median(np.abs(lags - med)) * 1000.0)
    slips = sum(abs(p["lag_s"] - med1) > 0.3 for p in per1)
    if slips:
        notes.append(f"{slips} of {len(per1)} windows locked a beat away in the wide search (handled by the median)")
    if abs(med - wall) > max_lag_s * 0.95:
        notes.append("xcorr lag at the edge of the search range: check the clocks")
    return SyncResult(med, spread, len(good), "xcorr", wall, per, notes)

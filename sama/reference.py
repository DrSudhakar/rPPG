"""Reference heart-rate signals: fingertip pulse oximeter, chest strap or ECG.

One CSV per session (pilot_data/sessions/<id>/reference*.csv). Column names are set in
configs/sama.yaml `reference.columns`, so most oximeter export files can be used as they are:

  oximeter_pleth  time + pleth waveform (e.g. CMS50D+/CMS50E via their PC software at 60 Hz)
                  -> beats from the waveform -> HR per window and pulse-rate variability (PRV)
  oximeter_hr     time + displayed pulse rate (bpm), about 1 Hz. HR only. The displayed value is
                  averaged over several beats inside the device, so it lags the true rate by a few
                  seconds; 30 s window means are still usable. No HRV, no cross-correlation sync.
  polar_rr        time + RR interval (ms), one row per beat
  ecg             time + ECG waveform -> R peaks

Timestamps: `reference.time_format` unix_s / unix_ms (PC clock: same clock as the recorder, which
is what makes wall-clock alignment possible) or relative_s (then sync must be by cross-correlation).

A fingertip oximeter measures the pulse at the finger, not the heart's electrical beats: its HRV is
pulse-rate variability, which tracks ECG HRV at rest but not exactly (pulse transit time varies).
The pilot report labels it so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .signals import bandpass, clean_ibi, peaks_parabolic, rmssd_ms, sdnn_ms


@dataclass
class Reference:
    kind: str
    t0_wall: float                   # wall-clock time (unix s) of t = 0, nan if unknown
    beats: np.ndarray | None = None  # beat times (s, reference clock relative to t0_wall)
    wave_t: np.ndarray | None = None
    wave: np.ndarray | None = None
    fs: float = float("nan")
    hr_t: np.ndarray | None = None
    hr: np.ndarray | None = None
    spo2: np.ndarray | None = None
    notes: list = field(default_factory=list)

    @property
    def has_beats(self) -> bool:
        return self.beats is not None and self.beats.size > 5

    @property
    def duration(self) -> float:
        for arr in (self.wave_t, self.hr_t, self.beats):
            if arr is not None and arr.size:
                return float(arr[-1])
        return 0.0

    def shifted(self, lag: float) -> "Reference":
        """Same reference with its clock moved by `lag` seconds (reference t -> t + lag)."""
        f = lambda a: None if a is None else a + lag  # noqa: E731
        return Reference(self.kind, self.t0_wall, f(self.beats), f(self.wave_t), self.wave, self.fs, f(self.hr_t), self.hr,
                         self.spo2, list(self.notes))

    def window_hr(self, a: float, b: float, min_valid: float = 0.9) -> float:
        """Reference HR in [a, b): 60 / mean clean IBI (beats), or the mean displayed rate."""
        if self.has_beats:
            t = self.beats[(self.beats >= a) & (self.beats < b)]
            if t.size < 3:
                return float("nan")
            ibi = np.diff(t)
            ok = clean_ibi(ibi)
            # coverage rule: beats must span the window and few may be edited out
            if (t[-1] - t[0]) < min_valid * (b - a) - 2.0 or ok.size < 0.8 * ibi.size:
                return float("nan")
            return float(60.0 / ok.mean())
        if self.hr is not None:
            m = (self.hr_t >= a) & (self.hr_t < b) & np.isfinite(self.hr) & (self.hr > 25) & (self.hr < 230)
            if m.sum() < min_valid * (b - a) * 0.8:   # ~1 Hz readings expected
                return float("nan")
            return float(np.mean(self.hr[m]))
        return float("nan")

    def window_hrv(self, a: float, b: float, max_edited: float = 0.05, min_beats: int = 40) -> dict:
        if not self.has_beats:
            return {"rmssd": np.nan, "sdnn": np.nan, "pnn50": np.nan, "n_beats": 0, "edited": np.nan}
        t = self.beats[(self.beats >= a) & (self.beats < b)]
        ibi = np.diff(t)
        ok = clean_ibi(ibi)
        edited = 1 - ok.size / max(ibi.size, 1)
        if t.size < min_beats or edited > max_edited:
            return {"rmssd": np.nan, "sdnn": np.nan, "pnn50": np.nan, "n_beats": int(t.size), "edited": float(edited)}
        return {"rmssd": rmssd_ms(ok), "sdnn": sdnn_ms(ok), "pnn50": pnn50(ok), "n_beats": int(t.size), "edited": float(edited)}


def pnn50(ibi_s: np.ndarray) -> float:
    ibi = np.asarray(ibi_s, float)
    if ibi.size < 3:
        return float("nan")
    return float(100.0 * np.mean(np.abs(np.diff(ibi)) > 0.05))


def ecg_beats(x: np.ndarray, fs: float) -> np.ndarray:
    """R peaks: 5-20 Hz band-pass, squared slope, peaks at least 0.3 s apart, refined to the raw maximum."""
    y = bandpass(np.asarray(x, float), fs, 5.0, min(20.0, fs / 2 - 1))
    e = np.gradient(y) ** 2
    win = max(1, int(0.1 * fs))
    e = np.convolve(e, np.ones(win) / win, "same")
    idx, _ = signal.find_peaks(e, distance=int(0.3 * fs), height=np.percentile(e, 90) * 0.3)
    r = int(0.05 * fs)
    ref = [i - r + int(np.argmax(np.abs(y[max(0, i - r):i + r]))) if i - r >= 0 else i for i in idx]
    return np.unique(np.asarray(ref)) / fs


def _times(df, col, fmt, fs, n):
    if col and col in df.columns:
        t = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        if fmt == "unix_ms":
            t = t / 1000.0
        return t
    return np.arange(n) / fs


def load_reference(path, kind: str, columns: dict, time_format: str = "unix_s", fs: float = 60.0) -> Reference:
    df = pd.read_csv(path)
    tcol = columns.get("time")
    t = _times(df, tcol, time_format, fs, len(df))
    good = np.isfinite(t)
    df, t = df[good].reset_index(drop=True), t[good]
    if time_format in ("unix_s", "unix_ms") and tcol in df.columns:
        t0 = float(t[0])
        t = t - t0
    else:
        t0 = float("nan")
    ref = Reference(kind, t0)
    if kind in ("oximeter_pleth", "ecg"):
        col = columns.get("pleth" if kind == "oximeter_pleth" else "ecg")
        if col not in df.columns:
            raise ValueError(f"{path}: column {col!r} not found; set reference.columns in configs/sama.yaml")
        w = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        ok = np.isfinite(w)
        ft = np.median(np.diff(t[ok])) if ok.sum() > 2 else 1 / fs
        fs_est = 1.0 / ft if ft > 0 else fs
        grid = np.arange(t[ok][0], t[ok][-1], 1.0 / fs_est)
        wave = np.interp(grid, t[ok], w[ok])
        ref.wave_t, ref.wave, ref.fs = grid, wave, float(fs_est)
        ref.beats = ppg_peaks(wave, fs_est) + grid[0] if kind == "oximeter_pleth" else ecg_beats(wave, fs_est) + grid[0]
        if (~ok).mean() > 0.02:
            ref.notes.append(f"{(~ok).mean():.1%} of waveform samples missing (interpolated)")
        hcol = columns.get("hr")
        if hcol in df.columns:
            ref.hr_t, ref.hr = t, pd.to_numeric(df[hcol], errors="coerce").to_numpy(float)
    elif kind == "oximeter_hr":
        col = columns.get("hr")
        if col not in df.columns:
            raise ValueError(f"{path}: column {col!r} not found; set reference.columns in configs/sama.yaml")
        ref.hr_t, ref.hr = t, pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        ref.notes.append("displayed pulse rate only: no beats, no HRV reference")
    elif kind == "polar_rr":
        col = columns.get("rr")
        rr = pd.to_numeric(df[col], errors="coerce").to_numpy(float) / 1000.0
        ok = np.isfinite(rr)
        # beat k occurs at the row's timestamp; if timestamps repeat, rebuild from cumulative RR
        tt = t[ok]
        ref.beats = tt if np.all(np.diff(tt) > 0.2) else tt[0] + np.concatenate([[0], np.cumsum(rr[ok][1:])])
    else:
        raise ValueError(kind)
    sp = columns.get("spo2")
    if sp in df.columns:
        ref.spo2 = pd.to_numeric(df[sp], errors="coerce").to_numpy(float)
    return ref


def ppg_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Systolic peaks of a contact PPG waveform with parabolic timing (same rule as step 5)."""
    y = bandpass(np.asarray(x, float), fs)
    from .signals import hr_from_signal
    hint, _, _ = hr_from_signal(y[: int(min(len(y), 60 * fs))], fs)
    # hint * 1.4 keeps beats when the rate rises later in the recording (e.g. after the step test)
    return peaks_parabolic(y, fs, min(180.0, 1.4 * hint) if np.isfinite(hint) else None)


def find_reference(session_dir: Path, pattern: str) -> Path | None:
    hits = sorted(Path(session_dir).glob(pattern))
    return hits[0] if hits else None

"""Frame-by-frame streaming reference implementation (deployment spec item 3).

Input per frame: timestamp (s), the 30 skin-patch mean RGB values (or None when no face), and
optionally the patch pixel counts. Frames may arrive at a variable rate; the ring buffer keeps
the last 30 s and is resampled to a uniform 30 fps grid before analysis. A reading is produced
every `update_s` seconds once 30 s are buffered, or a coded abstention:

  WARMING_UP          fewer than 30 s buffered
  FACE_LOST           > 10 % of the window had no face
  FRAME_RATE          effective frame rate < 12 fps or gaps > 0.5 s
  LOW_LIGHT           mean green below 25 DN (8-bit scale)
  MOTION              luminance motion proxy above the limit
  IMPLAUSIBLE         estimate outside 40-180 bpm
  INTERVAL_TOO_WIDE   conformal half-width > 5 bpm (step 6)

No frame is stored or transmitted: only patch means enter the buffer.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import methods as M
from . import net as N
from .conformal import MondrianGate
from .ita import stratum as ita_stratum
from .signals import bandpass, hr_from_signal

REASONS = ("OK", "WARMING_UP", "FACE_LOST", "FRAME_RATE", "LOW_LIGHT", "MOTION", "IMPLAUSIBLE", "INTERVAL_TOO_WIDE")
NO_READING = "No reliable reading"


@dataclass
class Reading:
    t: float
    shown: bool
    reason: str
    hr: float | None = None
    lo: float | None = None
    hi: float | None = None
    halfwidth: float | None = None
    snr_db: float | None = None
    latency_ms: float = 0.0

    def display(self) -> str:
        if not self.shown:
            return NO_READING
        return f"{self.hr:.0f} bpm (90% interval {self.lo:.0f}-{self.hi:.0f}) - research use only"

    def to_dict(self):
        d = dict(self.__dict__)
        d["display"] = self.display()
        if not self.shown:  # never leak a number when abstaining
            d.update(hr=None, lo=None, hi=None)
        return d


@dataclass
class StreamConfig:
    window_s: float = 30.0
    update_s: float = 1.0
    fps: float = 30.0
    min_fps: float = 12.0
    max_gap_s: float = 0.5
    face_lost_max: float = 0.10
    low_light_dn: float = 25.0
    motion_max: float = 25.0  # x1e-3 relative luminance step SD
    hr_range: tuple = (40.0, 180.0)


@dataclass
class StreamingEstimator:
    gate: MondrianGate
    sigma_model: object                       # SigmaModel
    params: dict | None = None                # network weights (numpy); None -> candidate only
    onnx_path: str | None = None
    ita: float = float("nan")
    cfg: StreamConfig = field(default_factory=StreamConfig)
    tone: object = None        # step 2 chart calibration (ToneCurve); None = uncalibrated (web demo, D-25)
    noise: object = None       # step 2 photon-transfer model (NoiseModel)

    def __post_init__(self):
        self.buf = deque()
        self.last_emit = -1e9
        self._sess = None
        if self.onnx_path:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            self._sess = ort.InferenceSession(self.onnx_path, so, providers=["CPUExecutionProvider"])

    def reset(self):
        self.buf.clear()
        self.last_emit = -1e9

    def push(self, t: float, patches: np.ndarray | None, pix: np.ndarray | None = None) -> Reading | None:
        self.buf.append((t, None if patches is None else np.asarray(patches, float), pix))
        while self.buf and self.buf[0][0] < t - self.cfg.window_s - 1.0:
            self.buf.popleft()
        if t - self.last_emit < self.cfg.update_s:
            return None
        self.last_emit = t
        t0 = time.perf_counter()
        r = self._estimate(t)
        r.latency_ms = (time.perf_counter() - t0) * 1000
        return r

    def _estimate(self, t_now: float) -> Reading:
        cfg = self.cfg
        items = [b for b in self.buf if b[0] >= t_now - cfg.window_s]
        if not items or items[-1][0] - items[0][0] < cfg.window_s - 2.0 / cfg.fps or (t_now - self.buf[0][0]) < cfg.window_s - 1e-6:
            return Reading(t_now, False, "WARMING_UP")
        ts = np.array([b[0] for b in items])
        face = np.array([b[1] is not None for b in items])
        if (~face).mean() > cfg.face_lost_max:
            return Reading(t_now, False, "FACE_LOST")
        span = ts[-1] - ts[0]
        if len(ts) / max(span, 1e-6) < cfg.min_fps or np.diff(ts).max(initial=0) > cfg.max_gap_s:
            return Reading(t_now, False, "FRAME_RATE")
        tt = ts[face]
        P = np.stack([b[1] for b in items if b[1] is not None])            # (n, P, 3)
        grid = np.arange(t_now - cfg.window_s, t_now, 1.0 / cfg.fps)
        flat = P.reshape(len(tt), -1)
        U = np.stack([np.interp(grid, tt, flat[:, j]) for j in range(flat.shape[1])], 1).reshape(len(grid), *P.shape[1:])
        pix = items[-1][2] if items[-1][2] is not None else np.ones(U.shape[1])
        w = np.asarray(pix, float) / np.sum(pix)
        face_rgb = (U * w[None, :, None]).sum(1)
        bright = float(face_rgb[:, 1].mean())
        if bright < cfg.low_light_dn:
            return Reading(t_now, False, "LOW_LIGHT")
        lum = face_rgb.mean(1)
        motion = float(np.std(np.diff(lum)) / max(lum.mean(), 1e-6) * 1e3)
        if motion > cfg.motion_max:
            return Reading(t_now, False, "MOTION")
        from .sama_method import window_features
        ref = [(0, len(grid), np.nan)]
        wf = window_features(U, cfg.fps, np.asarray(pix, float), self.ita if np.isfinite(self.ita) else 15.0,
                             self.tone, self.noise, ref, ppg=None)
        if self.params is not None or self._sess is not None:
            cond = wf.cond if np.isfinite(self.ita) else np.column_stack([np.full(1, 0.3), wf.cond[:, 1]])
            y = (self._sess.run(["y"], {"x": wf.X.astype(np.float32), "cond": cond.astype(np.float32)})[0]
                 if self._sess is not None else N.forward(self.params, wf.X, cond))
            h, snr, sh = hr_from_signal(y[0], N.FS_NET)
        else:
            h, snr, sh = wf.hr_cand[0], wf.q[0, 0], wf.q[0, 1]
        q = np.column_stack([wf.q, [snr], [sh], [abs(h - wf.hr_cand[0])]])
        sig = float(np.exp(self.sigma_model.predict(q))[0])
        g = ita_stratum(self.ita) if np.isfinite(self.ita) else "darker"   # unknown tone -> most conservative quantile
        lo, hi, hw, shown = self.gate.decide(np.array([h]), np.array([sig]), np.array([g]))
        if not np.isfinite(h) or not (cfg.hr_range[0] <= h <= cfg.hr_range[1]):
            return Reading(t_now, False, "IMPLAUSIBLE", snr_db=float(snr))
        if not shown[0]:
            return Reading(t_now, False, "INTERVAL_TOO_WIDE", halfwidth=float(hw[0]), snr_db=float(snr))
        return Reading(t_now, True, "OK", float(h), float(lo[0]), float(hi[0]), float(hw[0]), float(snr))


def estimate_offline(face_or_patches: np.ndarray, fps: float) -> dict:
    """Quick POS heart rate for an uploaded clip (no gate), used by the web app's file mode as the
    baseline comparison."""
    x = np.asarray(face_or_patches, float)
    rgb = x if x.ndim == 2 else x.mean(1)
    pulse = M.pos(rgb, fps)
    h, snr, _ = hr_from_signal(pulse, fps)
    return {"hr_pos": h, "snr_db": snr, "pulse": bandpass(pulse, fps)}

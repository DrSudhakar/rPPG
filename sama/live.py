"""Live analysis for the interface: heart rate, HRV, beat times, pulse trace and quality warnings.

Input per frame (from the browser or the lab recorder): time (s), 30 patch colours, skin pixels per
patch, motion index, skin fraction, face brightness. Every `live.update_s` seconds `state()` returns
everything the screen shows:

  hr, interval, shown/reason   heart rate from the last live.hr_window_s seconds (POS, CHROM, or the
                               SAMA bundle with its reliability gate when live.method = SAMA)
  hrv                          RMSSD, SDNN, pNN50 from camera beats over the last hrv.window_s, only
                               when the whole period was still enough (live.require_still_for_hrv)
  trace, beats                 the last live.display_s of band-passed pulse and the beat times on it
                               (the dots), oriented so that systolic peaks point up
  quality, warnings            motion, face presence, light, frame rate; each ok / warn / bad, with the
                               message shown to the participant ("Please hold still", ...)

Abstention codes (same as sama.streaming): WARMING_UP, FACE_LOST, FRAME_RATE, LOW_LIGHT, MOTION,
LOW_SIGNAL, IMPLAUSIBLE, INTERVAL_TOO_WIDE. A reading is never shown with a number when abstaining.
Research use only; not a medical device.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from . import methods as M
from .reference import pnn50
from .signals import bandpass, clean_ibi, hr_from_signal, peaks_parabolic, rmssd_ms, sdnn_ms
from .sync import orient

MESSAGES = {
    "WARMING_UP": "Collecting signal",
    "FACE_LOST": "Face not in the box",
    "FRAME_RATE": "Camera frame rate too low",
    "LOW_LIGHT": "Too dark: add light in front of the face",
    "MOTION": "Moving too much: please hold still",
    "LOW_SIGNAL": "Pulse signal too weak",
    "IMPLAUSIBLE": "Implausible value withheld",
    "INTERVAL_TOO_WIDE": "No reliable reading (interval too wide)",
}


class LiveAnalyzer:
    def __init__(self, cfg, model=None, model_meta=None, ita: float = float("nan"), stratum: str = "all"):
        self.cfg = cfg
        self.model, self.meta = model, model_meta or {}
        self.ita, self.stratum = ita, stratum
        self.buf = deque()
        self.knobs = {"method": cfg.live.method, "gate": bool(cfg.live.gate), "show_hrv": bool(cfg.live.show_hrv),
                      "show_beats": bool(cfg.live.show_beats), "motion_warn": float(cfg.quality.motion_warn),
                      "motion_abstain": float(cfg.quality.motion_abstain), "hr_window_s": float(cfg.live.hr_window_s),
                      "update_s": float(cfg.live.update_s), "require_still_for_hrv": bool(cfg.live.require_still_for_hrv)}
        self.last_state = None
        self.last_emit = -1e9
        self.n_frames = 0

    # ------------------------------------------------------------------ input
    def push(self, t, patches, pix=None, motion=0.0, skin=1.0, bright=None):
        p = None if patches is None else np.asarray(patches, float).reshape(30, 3)
        b = float(np.mean(p[:, 1])) if (bright is None and p is not None) else (float(bright) if bright is not None else 0.0)
        self.buf.append((float(t), p, None if pix is None else np.asarray(pix, float), float(motion), float(skin), b))
        keep = max(float(self.cfg.hrv.window_s), self.knobs["hr_window_s"], 30.0) + 5.0
        while self.buf and self.buf[0][0] < t - keep:
            self.buf.popleft()
        self.n_frames += 1

    def due(self, t) -> bool:
        return t - self.last_emit >= self.knobs["update_s"]

    def set_knobs(self, **kw):
        for k, v in kw.items():
            if k in self.knobs:
                self.knobs[k] = type(self.knobs[k])(v) if not isinstance(self.knobs[k], bool) else (v in (True, "true", "1", 1))

    # ------------------------------------------------------------------ helpers
    def _since(self, seconds):
        t_now = self.buf[-1][0]
        return [b for b in self.buf if b[0] >= t_now - seconds]

    @staticmethod
    def _uniform(items, fps=30.0):
        ts = np.array([b[0] for b in items if b[1] is not None])
        P = np.stack([b[1] for b in items if b[1] is not None])
        grid = np.arange(ts[0], ts[-1], 1.0 / fps)
        flat = P.reshape(len(ts), -1)
        U = np.stack([np.interp(grid, ts, flat[:, j]) for j in range(flat.shape[1])], 1).reshape(len(grid), 30, 3)
        return grid, U

    def _level(self, value, warn, bad, higher_is_worse=True):
        if not np.isfinite(value):
            return "bad"
        if higher_is_worse:
            return "bad" if value > bad else ("warn" if value > warn else "ok")
        return "bad" if value < bad else ("warn" if value < warn else "ok")

    # ------------------------------------------------------------------ main
    def state(self, t_now=None) -> dict:
        t0 = time.perf_counter()
        cfg, k = self.cfg, self.knobs
        if not self.buf:
            return {"reason": "WARMING_UP", "message": MESSAGES["WARMING_UP"], "shown": False}
        t_now = t_now if t_now is not None else self.buf[-1][0]
        self.last_emit = t_now
        recent = self._since(1.0)
        last5 = self._since(5.0)
        span = self.buf[-1][0] - self.buf[0][0]
        motion_now = float(np.median([b[3] for b in recent]))
        motion_p90 = float(np.percentile([b[3] for b in last5], 90))
        skin = float(np.mean([b[4] for b in recent]))
        bright = float(np.mean([b[5] for b in recent]))
        fps_now = (len(last5) - 1) / max(last5[-1][0] - last5[0][0], 1e-6) if len(last5) > 1 else 0.0
        q = {"motion": motion_now, "motion_p90_5s": motion_p90,
             "motion_level": self._level(motion_p90, k["motion_warn"], k["motion_abstain"]),
             "face": skin, "face_level": self._level(skin, cfg.face.min_skin_fraction + 0.1, cfg.face.min_skin_fraction, False),
             "light": bright, "light_level": self._level(bright, cfg.quality.low_light_dn + 15, cfg.quality.low_light_dn, False),
             "fps": fps_now, "fps_level": self._level(fps_now, cfg.quality.min_fps + 6, cfg.quality.min_fps, False)}
        warnings = []
        if q["motion_level"] == "bad":
            warnings.append({"level": "bad", "code": "MOTION", "text": "Moving too much: please hold still. Readings paused."})
        elif q["motion_level"] == "warn":
            warnings.append({"level": "warn", "code": "MOTION", "text": "Some movement: heart rate may be affected; HRV paused."})
        if q["face_level"] != "ok":
            warnings.append({"level": q["face_level"], "code": "FACE_LOST", "text": "Keep your face inside the box."})
        if q["light_level"] != "ok":
            warnings.append({"level": q["light_level"], "code": "LOW_LIGHT", "text": "Too dark: add light in front of the face."})
        if q["fps_level"] != "ok":
            warnings.append({"level": q["fps_level"], "code": "FRAME_RATE", "text": f"Camera delivers {fps_now:.0f} frames/s."})
        out = {"t": t_now, "method": k["method"], "quality": q, "warnings": warnings, "knobs": dict(k),
               "buffer_s": span, "ita": self.ita, "stratum": self.stratum}
        W = k["hr_window_s"]
        items = self._since(max(W, cfg.live.display_s))
        have = [b for b in items if b[1] is not None]
        if span < min(W, 10.0) or len(have) < 30:
            out.update(shown=False, reason="WARMING_UP", message=f"{MESSAGES['WARMING_UP']} ({span:.0f} of {W:.0f} s)")
            return self._finish(out, t0)
        grid, U = self._uniform(have)
        pix = next((b[2] for b in reversed(have) if b[2] is not None), np.ones(30))
        w = pix / pix.sum()
        face = (U * w[None, :, None]).sum(1)
        fps = 30.0
        meth = "CHROM" if k["method"] == "CHROM" else "POS"
        pulse = orient(bandpass(M.run(meth, face, fps), fps))
        hr_win = grid >= grid[-1] - W
        h, snr, _ = hr_from_signal(pulse[hr_win], fps)
        lo = hi = hw = None
        reason = "OK"
        if k["method"] == "SAMA" and self.model is not None and span >= 29.5:
            have30 = [b for b in self._since(30.0) if b[1] is not None]
            g30, U30 = self._uniform(have30)
            h, lo, hi, hw, gate_shown = self._sama(U30[-int(30 * fps):], pix, fps, h)
            if k["gate"] and not gate_shown:
                reason = "INTERVAL_TOO_WIDE"
        elif k["method"] == "SAMA":
            out["method_note"] = "no SAMA model loaded: showing POS" if self.model is None else f"SAMA needs 30 s ({span:.0f} s so far): showing POS"
        # display trace and beats
        disp = grid >= grid[-1] - cfg.live.display_s
        y = pulse[disp]
        ys = y / (np.std(y) + 1e-9)
        hint = h if np.isfinite(h) else None
        bt = peaks_parabolic(pulse, fps, min(180.0, 1.3 * hint) if hint else None) + grid[0]
        out["trace"] = {"t0": float(grid[disp][0]), "fs": 15.0, "y": np.round(ys[::2], 3).tolist()}
        out["beats"] = [float(b) for b in bt if b >= grid[disp][0]]
        out["last_beat"] = out["beats"][-1] if out["beats"] else None
        # gate on quality
        if q["face_level"] == "bad":
            reason = "FACE_LOST"
        elif q["fps_level"] == "bad":
            reason = "FRAME_RATE"
        elif q["light_level"] == "bad":
            reason = "LOW_LIGHT"
        elif q["motion_level"] == "bad":
            reason = "MOTION"
        elif not np.isfinite(h) or not (cfg.signal.hr_range_bpm[0] <= h <= cfg.signal.hr_range_bpm[1]):
            reason = "IMPLAUSIBLE"
        elif snr < cfg.quality.min_snr_db and k["method"] != "SAMA":
            reason = "LOW_SIGNAL"
        shown = reason == "OK"
        out.update(shown=shown, reason=reason, message="" if shown else MESSAGES.get(reason, reason),
                   hr=float(h) if shown else None, lo=lo if shown else None, hi=hi if shown else None,
                   halfwidth=hw, snr_db=float(snr) if np.isfinite(snr) else None)
        out["hrv"] = self._hrv(q) if k["show_hrv"] and cfg.hrv.enabled else {"valid": False, "reason": "HRV off"}
        return self._finish(out, t0)

    def _sama(self, U, pix, fps, h_pos):
        from .realtrain import gate_group_for
        from .sama_method import window_features
        ita_c = self.ita if (self.model.film and np.isfinite(self.ita)) else 0.0
        wf = window_features(U, fps, pix, ita_c, None, None, [(0, U.shape[0], np.nan)], ppg=None,
                             use_step2=False, use_step3=bool(self.meta.get("noise_weighting", True)))
        if not self.model.film:
            wf.cond[:, 0] = 0.0
        pr = self.model.predict(wf, np.array([gate_group_for(self.meta, self.stratum)]))
        h = float(pr["hr"][0])
        return h, float(pr["lo"][0]), float(pr["hi"][0]), float(pr["halfwidth"][0]), bool(pr["shown"][0])

    def _hrv(self, q) -> dict:
        cfg, k = self.cfg, self.knobs
        HW = float(cfg.hrv.window_s)
        items = self._since(HW)
        have = [b for b in items if b[1] is not None]
        span = (have[-1][0] - have[0][0]) if len(have) > 1 else 0.0
        if span < HW - 1.0:
            return {"valid": False, "reason": f"Collecting {span:.0f} of {HW:.0f} s"}
        if k["require_still_for_hrv"] and max(b[3] for b in items) > k["motion_warn"] * 1.5:
            return {"valid": False, "reason": "Paused: movement in the last minute"}
        grid, U = self._uniform(have)
        pix = next((b[2] for b in reversed(have) if b[2] is not None), np.ones(30))
        face = (U * (pix / pix.sum())[None, :, None]).sum(1)
        x = orient(bandpass(M.pos(face, 30.0), 30.0))
        h, _, _ = hr_from_signal(x, 30.0)
        t = peaks_parabolic(x, 30.0, min(180.0, 1.3 * h) if np.isfinite(h) else None)
        ibi = np.diff(t)
        ok = clean_ibi(ibi)
        edited = 1 - ok.size / max(ibi.size, 1)
        if ok.size < 0.5 * HW * (h / 60 if np.isfinite(h) else 1) or edited > 0.2:
            return {"valid": False, "reason": f"Too many unclear beats ({edited:.0%} edited)", "edited": edited}
        return {"valid": True, "rmssd": rmssd_ms(ok), "sdnn": sdnn_ms(ok), "pnn50": pnn50(ok), "mean_ibi_ms": 1000 * float(ok.mean()),
                "n_beats": int(t.size), "edited": float(edited), "window_s": HW,
                "note": "camera pulse-rate variability; research use only"}

    def _finish(self, out, t0):
        out["compute_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self.last_state = out
        return out

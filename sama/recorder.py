"""Lab recorder: one webcam, two outputs from one ffmpeg process.

  1. the full-resolution lossless video (FFV1 in Matroska) -> <session>/webcam.mkv, the source of
     the compression ladder and of every offline analysis;
  2. a reduced copy (recorder.analysis_size) piped to Python for the live display: heart rate, HRV,
     beat dots, motion and light warnings (sama.live.LiveAnalyzer).

The operator draws the face box and the grey/white card box on the preview; they are saved in
session.json with the PC time of the first frame (video_t0_wall). Block markers (start/end of rest,
consultation, arithmetic ... under light 1 and 2) go to events.csv with PC time stamps. Live readings
go to live_readings.csv. The fingertip-oximeter software runs on the same PC, so both share one clock.

Nothing is uploaded anywhere. Video stays on the recording PC (privacy.save_video must be true for the
pilot, because the compression ladder needs the lossless original).
"""
from __future__ import annotations

import csv
import json
import shlex
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .live import LiveAnalyzer
from .session import SESSION_SCHEMA
from .video import FaceTracker, FrameMeasurer, crop


def ffmpeg_cmd(cfg, out_file: Path | None, input_override: str | None = None) -> list[str]:
    r = cfg.recorder
    w, h = (int(v) for v in str(r.analysis_size).split("x"))
    if r.input_format == "file" or input_override:
        src = ["-re", "-i", str(input_override or r.device)]
    else:
        src = ["-f", r.input_format, "-framerate", str(r.fps), "-video_size", str(r.size)]
        if r.pixel_format:
            src += ["-input_format" if r.input_format == "v4l2" else "-pixel_format", str(r.pixel_format)]
        dev = str(r.device)
        if r.input_format == "dshow" and not dev.startswith("video="):
            dev = f"video={dev}"
        src += ["-i", dev]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-y", *src]
    if out_file is not None:
        enc = (["-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1"] if r.codec == "ffv1"
               else ["-c:v", "libx264rgb", "-preset", "ultrafast", "-qp", "0"])
        cmd += ["-map", "0:v", *enc, "-fps_mode", "passthrough", str(out_file)]
    cmd += ["-map", "0:v", "-vf", f"scale={w}:{h}:flags=area", "-pix_fmt", "rgb24", "-f", "rawvideo", "-fps_mode", "passthrough", "pipe:1"]
    return cmd


class LabRecorder:
    def __init__(self, cfg, model=None, model_meta=None, log=print):
        self.cfg, self.model, self.meta, self.log = cfg, model, model_meta or {}, log
        self.proc = None
        self.thread = None
        self.lock = threading.Lock()
        self.session_dir: Path | None = None
        self.analyzer: LiveAnalyzer | None = None
        self.latest_frame = None
        self.latest_state: dict = {"recording": False}
        self.boxes = {"face_box": list(cfg.face.fixed_box), "white_box": None, "chart_corners": None}
        self.white_reflectance = float(cfg.ita.camera.white_reflectance)
        self.t0_wall = None
        self.n_frames = 0
        self.white_check = {}
        self._readings_fh = None
        self.error = None
        # preview-only capture: opens the camera and fills latest_frame so the operator can see the
        # feed and draw the face/card boxes *before* pressing "Start recording" (which reopens the
        # device with the lossless-file output added). Independent of self.proc/self.thread above.
        self.preview_proc = None
        self.preview_thread = None
        self.preview_error = None

    # -------------------------------------------------------------- session files
    def _session_json(self):
        p = self.session_dir / "session.json"
        meta = json.loads(p.read_text()) if p.exists() else json.loads(json.dumps(SESSION_SCHEMA))
        return p, meta

    def _write_meta(self, **kw):
        p, meta = self._session_json()
        meta.update(kw)
        meta["boxes"] = self.boxes
        meta["white_reflectance"] = self.white_reflectance
        p.write_text(json.dumps(meta, indent=1))

    def event(self, event: str, block: str = "", light: str = "", note: str = "") -> dict:
        if self.session_dir is None:
            raise RuntimeError("no session")
        p = self.session_dir / "events.csv"
        new = not p.exists()
        row = {"wall_time": f"{time.time():.3f}", "event": event, "block": block, "light": light, "note": note}
        with open(p, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        return row

    def set_boxes(self, face_box=None, white_box=None, chart_corners=None, white_reflectance=None):
        with self.lock:
            if face_box is not None:
                self.boxes["face_box"] = [float(v) for v in face_box]
                if self.tracker is not None:
                    self.tracker._box = np.array(self.boxes["face_box"], float)
                    self.tracker.manual_box = tuple(self.boxes["face_box"])
            if white_box is not None:
                self.boxes["white_box"] = [float(v) for v in white_box] if white_box else None
            if chart_corners is not None:
                self.boxes["chart_corners"] = chart_corners or None
            if white_reflectance:
                self.white_reflectance = float(white_reflectance)
        if self.session_dir is not None:
            self._write_meta()

    # -------------------------------------------------------------- preview (camera open, no recording)
    def start_preview(self, input_override: str | None = None) -> dict:
        """Open the camera and start filling `latest_frame`, without writing any session file. A
        no-op if recording is already running (its stream already fills `latest_frame`) or a preview
        is already open. Call `stop_preview()` first if you need to change the input."""
        if self.proc is not None:
            return {"preview": True, "recording": True}
        if self.preview_proc is not None:
            return {"preview": True, "recording": False}
        cmd = ffmpeg_cmd(self.cfg, None, input_override)
        self.log("[recorder] preview: " + " ".join(shlex.quote(c) for c in cmd))
        self.preview_error = None
        try:
            self.preview_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:  # noqa: BLE001
            self.preview_error = str(e)
            return {"preview": False, "error": self.preview_error}
        self.preview_thread = threading.Thread(target=self._preview_loop, args=(self.preview_proc,), daemon=True)
        self.preview_thread.start()
        return {"preview": True, "recording": False}

    def stop_preview(self) -> dict:
        p = self.preview_proc
        if p is None:
            return {"stopped": False}
        self.preview_proc = None  # let the loop thread see this and exit cleanly
        try:
            p.stdin.write(b"q")
            p.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.terminate()
        if self.preview_thread is not None:
            self.preview_thread.join(timeout=5)
            self.preview_thread = None
        for fh in (p.stdin, p.stdout, p.stderr):
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass
        return {"stopped": True}

    def _preview_loop(self, proc):
        w, h = (int(v) for v in str(self.cfg.recorder.analysis_size).split("x"))
        n = w * h * 3
        out = proc.stdout
        while self.preview_proc is proc:
            buf = out.read(n)
            if len(buf) < n:
                break
            fr = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            with self.lock:
                self.latest_frame = fr
        try:
            err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        except Exception:  # noqa: BLE001
            err = ""
        if err.strip() and self.preview_proc is proc:
            self.preview_error = err.strip()[-2000:]
            self.log("[recorder] preview ffmpeg: " + self.preview_error)

    # -------------------------------------------------------------- start / stop
    tracker = None

    def start(self, participant_id: str, operator: str = "", notes: str = "", input_override: str | None = None) -> dict:
        if self.proc is not None:
            raise RuntimeError("already recording")
        self.stop_preview()  # the camera device can only be opened by one ffmpeg process at a time
        pid = "".join(ch for ch in participant_id if ch.isalnum() or ch in "-_") or "P00"
        self.session_dir = Path(self.cfg.paths.pilot_root) / "sessions" / pid
        if (self.session_dir / "webcam.mkv").exists():
            raise RuntimeError(f"{self.session_dir}/webcam.mkv exists: use a new participant id or move the old session")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        out_file = self.session_dir / "webcam.mkv" if self.cfg.privacy.save_video else None
        self.cmd = ffmpeg_cmd(self.cfg, out_file, input_override)
        self.log("[recorder] " + " ".join(shlex.quote(c) for c in self.cmd))
        self.tracker = FaceTracker(mode="manual_box" if self.cfg.face.tracker in ("auto", "manual_box") else self.cfg.face.tracker,
                                   fixed_box=tuple(self.cfg.face.fixed_box), manual_box=tuple(self.boxes["face_box"]))
        self.measure = FrameMeasurer()
        self.analyzer = LiveAnalyzer(self.cfg, self.model, self.meta)
        self.t0_wall, self.n_frames, self.error = None, 0, None
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        r = self.cfg.recorder
        self._write_meta(participant_id=pid, date=time.strftime("%Y-%m-%d"), operator=operator, notes=notes,
                         webcam={"file": "webcam.mkv" if out_file else None, "device": str(r.device), "size": str(r.size),
                                 "fps": r.fps, "codec": r.codec, "input_format": r.input_format},
                         provenance=self.cfg.project.provenance_label)
        self._readings_fh = open(self.session_dir / "live_readings.csv", "a", newline="")
        self._readings = csv.writer(self._readings_fh)
        self._readings.writerow(["wall_time", "t", "shown", "reason", "hr", "lo", "hi", "rmssd", "sdnn", "motion", "face", "light", "fps"])
        self.event("recording_start")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return {"session_dir": str(self.session_dir), "participant_id": pid}

    def stop(self) -> dict:
        if self.proc is None:
            return {"stopped": False}
        self.event("recording_stop")
        try:
            self.proc.stdin.write(b"q")
            self.proc.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
        if self.thread is not None:
            self.thread.join(timeout=5)
        for fh in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass
        self.proc = None
        if self._readings_fh:
            self._readings_fh.close()
            self._readings_fh = None
        self._write_meta(video_t0_wall=self.t0_wall, n_frames_live=self.n_frames)
        self.latest_state = {**(self.latest_state or {}), "recording": False}
        self.start_preview()  # reopen the camera so the operator sees a live feed for the next participant
        return {"stopped": True, "frames": self.n_frames, "session_dir": str(self.session_dir)}

    # -------------------------------------------------------------- frame loop
    def _loop(self):
        w, h = (int(v) for v in str(self.cfg.recorder.analysis_size).split("x"))
        n = w * h * 3
        out = self.proc.stdout
        while True:
            buf = out.read(n)
            if len(buf) < n:
                break
            now = time.time()
            if self.t0_wall is None:
                self.t0_wall = now
                self._write_meta(video_t0_wall=now)
            fr = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            with self.lock:
                box = self.tracker.update(fr)
                m = self.measure(fr, box)
                self.latest_frame = fr
                wb = self.boxes.get("white_box")
            t = now - self.t0_wall
            self.analyzer.push(t, m.patches, m.pix, m.motion, m.skin_fraction, m.brightness)
            self.n_frames += 1
            if wb and self.n_frames % 15 == 0:
                c = crop(fr, wb).reshape(-1, 3)
                self.white_check = {"clipped_frac": float((c.max(1) >= 250).mean()), "mean": float(c.mean()),
                                    "ok": bool((c.max(1) >= 250).mean() < 0.02 and c.mean() > 30)}
            if self.analyzer.due(t):
                st = self.analyzer.state(t)
                st.update(recording=True, frames=self.n_frames, participant_id=self.session_dir.name, box=list(map(float, box)),
                          white_check=self.white_check)
                self.latest_state = st
                hrv = st.get("hrv") or {}
                q = st.get("quality") or {}
                self._readings.writerow([f"{now:.3f}", f"{t:.3f}", st.get("shown"), st.get("reason"), st.get("hr"), st.get("lo"), st.get("hi"),
                                         hrv.get("rmssd"), hrv.get("sdnn"), q.get("motion"), q.get("face"), q.get("light"), q.get("fps")])
        try:
            err = self.proc.stderr.read().decode(errors="replace") if self.proc and self.proc.stderr else ""
        except Exception:  # noqa: BLE001
            err = ""
        if err.strip():
            self.error = err.strip()[-2000:]
            self.log("[recorder] ffmpeg: " + self.error)
        self.latest_state = {**(self.latest_state or {}), "recording": False, "error": self.error}

    def preview_jpeg(self, quality: int = 70) -> bytes | None:
        from io import BytesIO
        from PIL import Image
        with self.lock:
            fr = None if self.latest_frame is None else self.latest_frame.copy()
        if fr is None:
            return None
        b = BytesIO()
        Image.fromarray(fr).save(b, "JPEG", quality=quality)
        return b.getvalue()

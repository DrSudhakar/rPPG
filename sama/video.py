"""Video input/output and per-frame measurements, shared by offline analysis and the recorder.

Decoding goes through the ffmpeg / ffprobe command-line tools, never through OpenCV's bundled
FFmpeg (that build crashed with heap corruption on UBFC-rPPG files; ISSUES I-36). OpenCV is
used only, if present, for the Haar face detector.

Per frame we compute, inside the face box:
  patches        30 skin-patch mean RGB values (sama.model_io.LAYOUT), non-skin pixels masked
  skin_fraction  share of box pixels passing a YCbCr skin rule (face present?)
  motion         mean absolute luma change from the previous frame (8-bit units, on a 48x48
                 thumbnail of the box): the motion index used for warnings and abstention
  brightness     mean green value of the face box (8-bit DN)
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model_io import LAYOUT

FOREHEAD = list(range(12))
CHEEKS = list(range(12, 30))


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe(path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate,codec_name,nb_frames,bit_rate:format=duration,start_time",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s, f = j["streams"][0], j.get("format", {})

    def rate(r):
        try:
            n, d = r.split("/")
            return float(n) / float(d) if float(d) else float("nan")
        except Exception:  # noqa: BLE001
            return float("nan")
    fps = rate(s.get("avg_frame_rate", "0/0"))
    if not np.isfinite(fps) or fps <= 0:
        fps = rate(s.get("r_frame_rate", "30/1"))
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps, "codec": s.get("codec_name"),
            "duration": float(f.get("duration", "nan")), "start_time": float(f.get("start_time", 0) or 0),
            "bit_rate": float(s.get("bit_rate") or f.get("bit_rate") or "nan") if (s.get("bit_rate") or f.get("bit_rate")) else float("nan")}


def frame_times(path) -> np.ndarray:
    """Presentation times (s) of every video frame, from the container (no decoding)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    t = np.array([float(x) for x in out.split() if x and x != "N/A"])
    return np.sort(t)


def iter_frames(path, size: tuple[int, int] | None = None, max_s: float | None = None, start_s: float = 0.0):
    """Yield RGB uint8 frames (h, w, 3) decoded by the ffmpeg CLI, one at a time."""
    info = probe(path)
    w, h = size or (info["width"], info["height"])
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", f"{start_s:.3f}"] if start_s > 0 else []) + ["-i", str(path)]
    if max_s:
        cmd += ["-t", str(max_s)]
    if size:
        cmd += ["-vf", f"scale={w}:{h}:flags=area"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "passthrough", "-"]
    n = w * h * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)


def write_video(path, frames, fps: float, codec: str = "ffv1"):
    """Write RGB uint8 frames with ffmpeg (used by tests and the synthetic pilot)."""
    frames = iter(frames)
    first = next(frames)
    h, w = first.shape[:2]
    args = {"ffv1": ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp"],
            "h264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]}[codec]
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           *args, str(path)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as p:
        p.stdin.write(np.ascontiguousarray(first).tobytes())
        for fr in frames:
            p.stdin.write(np.ascontiguousarray(fr).tobytes())
        p.stdin.close()
        p.wait()
    if p.returncode:
        raise RuntimeError(f"ffmpeg failed writing {path}")


# ------------------------------------------------------------------ face box ------------
def haar_detector():
    """OpenCV's frontal-face Haar cascade if the model file is installed, else None."""
    try:
        import cv2
        xml = Path(getattr(cv2, "data", None).haarcascades) / "haarcascade_frontalface_default.xml"
        if not xml.exists():
            return None
        det = cv2.CascadeClassifier(str(xml))
        return None if det.empty() else det
    except Exception:  # noqa: BLE001
        return None


@dataclass
class FaceTracker:
    """Face box per frame as fractions (x, y, w, h).

    mode: haar (detect every redetect_every_s, smooth), manual_box (fixed box drawn by the
    operator), fixed_box (configured guide box). auto = haar if available, else manual box if
    given, else fixed box. `source` records what was actually used."""
    mode: str = "auto"
    fixed_box: tuple = (0.30, 0.15, 0.40, 0.60)
    manual_box: tuple | None = None
    redetect_every: int = 30
    smoothing: float = 0.8
    source: str = ""
    _det: object = None
    _box: np.ndarray | None = None
    _n: int = 0
    _misses: int = 0
    log: list = field(default_factory=list)

    def __post_init__(self):
        if self.mode in ("auto", "haar"):
            self._det = haar_detector()
        if self.mode == "haar" and self._det is None:
            raise RuntimeError("face.tracker=haar but OpenCV's Haar model file is not installed")
        if self._det is not None:
            self.source = "haar"
        elif self.mode in ("auto", "manual_box") and self.manual_box is not None:
            self.source = "manual_box"
        else:
            self.source = "fixed_box"
        start = self.manual_box if (self.manual_box is not None and self.source != "fixed_box") else self.fixed_box
        self._box = np.array(start, float)

    def update(self, frame: np.ndarray) -> np.ndarray:
        if self.source == "haar" and self._n % max(1, self.redetect_every) == 0:
            import cv2
            g = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2GRAY)
            H, W = g.shape
            faces = self._det.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5, minSize=(W // 8, W // 8))
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                # Haar boxes run from the eyebrows to the chin: extend upwards to include the forehead
                new = np.array([x / W, max(0.0, (y - 0.25 * h) / H), w / W, min(1.0, 1.25 * h / H)])
                self._box = self.smoothing * self._box + (1 - self.smoothing) * new if self._n else new
                self._misses = 0
            else:
                self._misses += 1
        self._n += 1
        return self._box.copy()


def crop(frame: np.ndarray, box) -> np.ndarray:
    H, W = frame.shape[:2]
    x, y, w, h = box
    x0, y0 = int(max(0, x) * W), int(max(0, y) * H)
    x1, y1 = int(min(1, x + w) * W), int(min(1, y + h) * H)
    return frame[y0:max(y1, y0 + 2), x0:max(x1, x0 + 2)]


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """YCbCr skin rule (Chai & Ngan 1999 bounds), deliberately wide to include dark skin."""
    r, g, b = (rgb[..., i].astype(float) for i in range(3))
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return (cb >= 73) & (cb <= 132) & (cr >= 130) & (cr <= 180) & (y > 15)


def _thumb(gray: np.ndarray, n: int = 48) -> np.ndarray:
    H, W = gray.shape
    ys = (np.arange(n) * H / n).astype(int)
    xs = (np.arange(n) * W / n).astype(int)
    return gray[np.ix_(ys, xs)]


@dataclass
class FrameMeasure:
    patches: np.ndarray       # (30, 3)
    pix: np.ndarray           # (30,) skin pixels used per patch
    skin_fraction: float
    motion: float
    brightness: float
    box: np.ndarray


class FrameMeasurer:
    """Stateful per-frame measurements (motion needs the previous frame)."""

    def __init__(self, clip_dn: float = 250.0, mask_skin: bool = True):
        self.prev = None
        self.clip = clip_dn
        self.mask_skin = mask_skin

    def __call__(self, frame: np.ndarray, box) -> FrameMeasure:
        roi = crop(frame, box).astype(float)
        H, W, _ = roi.shape
        sk = skin_mask(roi) if self.mask_skin else np.ones((H, W), bool)
        gray = roi @ np.array([0.299, 0.587, 0.114])
        th = _thumb(gray)
        motion = float(np.mean(np.abs(th - self.prev))) if self.prev is not None and self.prev.shape == th.shape else 0.0
        self.prev = th
        out = np.zeros((30, 3))
        pix = np.zeros(30)
        for i, (x, y, w, h) in enumerate(LAYOUT):
            ya, yb = int(y * H), max(int((y + h) * H), int(y * H) + 1)
            xa, xb = int(x * W), max(int((x + w) * W), int(x * W) + 1)
            p = roi[ya:yb, xa:xb].reshape(-1, 3)
            m = sk[ya:yb, xa:xb].reshape(-1)
            keep = (p.max(1) < self.clip) & (p.min(1) > 2)
            k2 = keep & m
            use = k2 if k2.sum() >= max(4, 0.3 * len(p)) else keep
            if not use.any():
                use = np.ones(len(p), bool)
            out[i] = p[use].mean(0)
            pix[i] = use.sum()
        return FrameMeasure(out, pix, float(sk.mean()), motion, float(roi[..., 1].mean()), np.asarray(box, float))


@dataclass
class VideoSignals:
    """Everything the analysis needs from one video, without keeping any frame."""
    t: np.ndarray              # (T,) frame times, s (container PTS, rebased to the file start)
    patches: np.ndarray        # (T, 30, 3)
    pix: np.ndarray            # (T, 30)
    skin_fraction: np.ndarray
    motion: np.ndarray
    brightness: np.ndarray
    boxes: np.ndarray          # (T, 4)
    fps: float
    tracker: str
    info: dict

    def save(self, path):
        np.savez_compressed(path, t=self.t, patches=self.patches.astype(np.float32), pix=self.pix.astype(np.float32),
                            skin_fraction=self.skin_fraction, motion=self.motion, brightness=self.brightness,
                            boxes=self.boxes, fps=self.fps, tracker=self.tracker, info=json.dumps(self.info))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(d["t"], d["patches"].astype(float), d["pix"].astype(float), d["skin_fraction"], d["motion"],
                   d["brightness"], d["boxes"], float(d["fps"]), str(d["tracker"]), json.loads(str(d["info"])))


def extract_signals(path, tracker: FaceTracker | None = None, boxes: np.ndarray | None = None,
                    max_s: float | None = None, size=None) -> VideoSignals:
    """Decode a video once and reduce it to per-frame measurements. If `boxes` is given (one per
    frame, e.g. tracked on the lossless video), they are reused, so every compression level is
    measured on exactly the same face region."""
    info = probe(path)
    t_all = frame_times(path)
    meas = FrameMeasurer()
    tr = tracker or FaceTracker(mode="fixed_box")
    rows = []
    for i, fr in enumerate(iter_frames(path, size=size, max_s=max_s)):
        box = boxes[min(i, len(boxes) - 1)] if boxes is not None else tr.update(fr)
        rows.append(meas(fr, box))
    n = len(rows)
    if n == 0:
        raise RuntimeError(f"no frames decoded from {path}")
    t = t_all[:n] if t_all.size >= n else np.arange(n) / info["fps"]
    t = t - t[0]
    fps = (n - 1) / max(t[-1], 1e-6) if n > 1 else info["fps"]
    return VideoSignals(t, np.stack([r.patches for r in rows]), np.stack([r.pix for r in rows]),
                        np.array([r.skin_fraction for r in rows]), np.array([r.motion for r in rows]),
                        np.array([r.brightness for r in rows]), np.stack([r.box for r in rows]), float(fps),
                        "given_boxes" if boxes is not None else tr.source, info)


def resample_uniform(t: np.ndarray, x: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of (T, ...) samples onto a uniform grid (variable frame rate input)."""
    grid = np.arange(t[0], t[-1], 1.0 / fps)
    flat = x.reshape(len(t), -1)
    out = np.stack([np.interp(grid, t, flat[:, j]) for j in range(flat.shape[1])], 1)
    return grid, out.reshape(len(grid), *x.shape[1:])

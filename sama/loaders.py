"""The common recording type (`RealRecording`) and the UBFC-rPPG / PURE loaders.

Every other public dataset -- VIPL-HR, iBVP, SUMS, LADH, M3PD, RLAP, MMPD -- lives in
sama.datasets, which also owns the registry behind `discover` and `load` below. Anything that
consumes recordings (training, the pilot analysis) goes through `RealRecording` and never needs to
know which dataset produced it.


UBFC-rPPG (dataset 2): <root>/subject*/vid.avi and ground_truth.txt (line 1: PPG trace, line 2: HR,
line 3: time stamps in s).
PURE: <root>/NN-NN/ (PNG frames named Image<timestamp_ns>.png) and <root>/NN-NN.json with
"/FullPackage" entries (Timestamp, Value.waveform) and "/Image" timestamps.

Video is decoded with the ffmpeg CLI (sama.video), never OpenCV's bundled FFmpeg, which crashed
with heap corruption on UBFC vid.avi files (ISSUES I-36). PURE PNGs are read with Pillow.

Face region: sama.video.FaceTracker (Haar if installed, else the fixed guide box). Neither dataset
has a white reference or colour chart, so any ITA° estimated from them is approximate (I-03).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .signals import bandpass, peaks_parabolic
from .video import FaceTracker, FrameMeasurer, VideoSignals, extract_signals


@dataclass
class RealRecording:
    name: str
    patches: np.ndarray       # (T, 30, 3)
    fps: float
    beats: np.ndarray         # reference beat times (s, video clock) from the contact PPG
    source: str
    ppg: np.ndarray | None = None        # reference waveform resampled to the video frames (T,)
    pix: np.ndarray | None = None        # (T, 30) skin pixels per patch
    motion: np.ndarray | None = None
    skin_fraction: np.ndarray | None = None
    participant: str = ""
    meta: dict = field(default_factory=dict)


def ppg_beats(ppg: np.ndarray, fs: float) -> np.ndarray:
    return peaks_parabolic(bandpass(np.asarray(ppg, float), fs), fs)


def _from_signals(vs: VideoSignals, name, source, beats, ppg_t, ppg, participant, meta) -> RealRecording:
    t = vs.t
    ppg_v = np.interp(t, ppg_t, ppg) if ppg is not None else None
    return RealRecording(name, vs.patches, vs.fps, beats, source, ppg_v, vs.pix, vs.motion, vs.skin_fraction,
                         participant, dict(meta, tracker=vs.tracker))


def load_ubfc_rppg(subject_dir, tracker_mode: str = "auto") -> RealRecording:
    d = Path(subject_dir)
    vs = extract_signals(d / "vid.avi", FaceTracker(mode=tracker_mode))
    lines = (d / "ground_truth.txt").read_text().strip().splitlines()
    ppg = np.array(lines[0].split(), float)
    t = np.array(lines[2].split(), float) if len(lines) > 2 else np.arange(ppg.size) / vs.fps
    t = t - t[0]
    fs_ppg = 1.0 / np.median(np.diff(t))
    beats = ppg_beats(ppg, fs_ppg)
    return _from_signals(vs, d.name, "UBFC-rPPG", beats, t, ppg, f"UBFC:{d.name}", {"fs_ref": fs_ppg})


def load_pure(seq_dir, tracker_mode: str = "auto") -> RealRecording:
    from PIL import Image
    d = Path(seq_dir)
    meta = json.loads(Path(str(d) + ".json").read_text())
    frames = sorted(d.glob("Image*.png"), key=lambda p: int(p.stem[5:]))
    ts = np.array([int(p.stem[5:]) for p in frames]) * 1e-9
    tr, meas = FaceTracker(mode=tracker_mode), FrameMeasurer()
    rows = []
    for p in frames:
        fr = np.asarray(Image.open(p).convert("RGB"))
        rows.append(meas(fr, tr.update(fr)))
    pk = meta["/FullPackage"]
    tp = np.array([e["Timestamp"] for e in pk]) * 1e-9
    wave = np.array([e["Value"]["waveform"] for e in pk], float)
    fs_ppg = 1.0 / np.median(np.diff(tp))
    t0 = ts[0]
    vs = VideoSignals(ts - t0, np.stack([r.patches for r in rows]), np.stack([r.pix for r in rows]),
                      np.array([r.skin_fraction for r in rows]), np.array([r.motion for r in rows]),
                      np.array([r.brightness for r in rows]), np.stack([r.box for r in rows]),
                      float(1.0 / np.median(np.diff(ts))), tr.source, {})
    beats = ppg_beats(wave, fs_ppg) + tp[0] - t0
    subj = d.name.split("-")[0]
    return _from_signals(vs, d.name, "PURE", beats, tp - t0, wave, f"PURE:{subj}", {"fs_ref": fs_ppg, "task": d.name.split("-")[-1]})


def discover(root, only: list[str] | None = None) -> list:
    """Find every recording under data/raw/, in any supported dataset's layout.

    Delegates to the dataset registry (sama.datasets), which covers UBFC-rPPG, PURE, VIPL-HR,
    iBVP, SUMS, LADH, M3PD, RLAP and MMPD. Returns [(dataset name, path)].
    """
    from .datasets import discover as _discover
    return _discover(root, only)


def load(kind, path, tracker_mode: str = "auto") -> RealRecording:
    """Load one recording of any supported dataset (see sama.datasets.REGISTRY)."""
    from .datasets import load as _load
    return _load(kind, path, tracker_mode)

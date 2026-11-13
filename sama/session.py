"""Pilot session folders: layout, metadata and protocol blocks.

pilot_data/
  participants.csv          one row per participant (case report form; see pilot/forms/)
  monk_ratings.csv          one row per rater per participant
  ita_measured.csv          optional, colorimeter readings once an instrument is available
  sessions/<participant_id>/
    session.json            written by the recorder (or by hand): start time, camera, boxes, lux
    events.csv              wall_time, event, block, light   (block_start / block_end / note)
    webcam.mkv              lossless FFV1 video from the recorder
    phone.mp4               optional, the phone's own recording
    reference.csv           oximeter (or other reference) export
    live_readings.csv       optional, what the live interface showed
    ladder/                 compression ladder (made by `sama ladder`)
    derived/                per-frame signals, camera ITA, sync and QC (made by the pipeline)

Times: every wall-clock value is unix seconds from the recording PC. The video's frame times start
at 0 at its first frame, which session.json records as `video_t0_wall`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_SCHEMA = {
    "participant_id": "P01",
    "date": "2026-11-10",
    "operator": "",
    "video_t0_wall": None,          # unix time of the first webcam frame (recorder fills this)
    "webcam": {"file": "webcam.mkv", "device": "", "size": "640x480", "fps": 30, "codec": "ffv1"},
    "phone": {"file": "phone.mp4", "model": "", "t0_wall_approx": None},
    "boxes": {"face_box": None, "white_box": None, "chart_corners": None},
    "white_reference": "card",
    "lux": {"light1_500lux": None, "light2_100lux": None},
    "reference": {"device": "", "file": "reference.csv"},
    "notes": "",
}


@dataclass
class Block:
    name: str
    light: str
    start: float            # video time (s)
    end: float


@dataclass
class Session:
    root: Path
    meta: dict
    events: pd.DataFrame
    blocks: list = field(default_factory=list)

    @property
    def pid(self) -> str:
        return self.meta.get("participant_id") or self.root.name

    @property
    def derived(self) -> Path:
        d = self.root / "derived"
        d.mkdir(exist_ok=True)
        return d

    def video(self, stream: str = "webcam") -> Path | None:
        m = self.meta.get(stream) or {}
        p = self.root / m.get("file", f"{stream}.mkv")
        if p.exists():
            return p
        hits = sorted(self.root.glob(f"{stream}.*"))
        return hits[0] if hits else None

    def ladder_dir(self) -> Path:
        return self.root / "ladder"

    def box(self, name: str):
        b = (self.meta.get("boxes") or {}).get(name)
        return None if b is None else (np.asarray(b, float) if name != "chart_corners" else b)

    def block_of(self, t: float) -> Block | None:
        for b in self.blocks:
            if b.start <= t < b.end:
                return b
        return None


def _blocks_from_events(ev: pd.DataFrame, t0: float, duration: float) -> list[Block]:
    out = []
    if ev is None or ev.empty or not np.isfinite(t0):
        return [Block("whole_recording", "unknown", 0.0, duration)]
    ev = ev.sort_values("wall_time")
    open_ = {}
    for _, r in ev.iterrows():
        if r["event"] == "block_start":
            open_[r["block"]] = (float(r["wall_time"]) - t0, str(r.get("light", "") or "unknown"))
        elif r["event"] == "block_end" and r["block"] in open_:
            s, light = open_.pop(r["block"])
            out.append(Block(str(r["block"]), light, max(0.0, s), min(duration, float(r["wall_time"]) - t0)))
    for name, (s, light) in open_.items():   # block never closed: runs to the end
        out.append(Block(str(name), light, max(0.0, s), duration))
    out = [b for b in out if b.end - b.start > 1.0]
    return sorted(out, key=lambda b: b.start) or [Block("whole_recording", "unknown", 0.0, duration)]


def load_session(root, duration: float | None = None) -> Session:
    root = Path(root)
    meta = json.loads((root / "session.json").read_text()) if (root / "session.json").exists() else {"participant_id": root.name}
    ev = pd.read_csv(root / "events.csv") if (root / "events.csv").exists() else pd.DataFrame(columns=["wall_time", "event", "block", "light"])
    s = Session(root, meta, ev)
    t0 = meta.get("video_t0_wall")
    if duration is None:
        v = s.video("webcam")
        if v is not None:
            from .video import probe
            duration = probe(v)["duration"]
        else:
            duration = 0.0
    s.blocks = _blocks_from_events(ev, float(t0) if t0 is not None else float("nan"), float(duration or 0.0))
    return s


def list_sessions(pilot_root) -> list[Path]:
    d = Path(pilot_root) / "sessions"
    return sorted(p for p in d.glob("*") if p.is_dir()) if d.exists() else []


def session_minutes(s: Session) -> float:
    ev = s.events
    if ev is not None and len(ev) >= 2:
        return float((ev.wall_time.max() - ev.wall_time.min()) / 60.0)
    return float("nan")


def load_participants(pilot_root) -> pd.DataFrame:
    p = Path(pilot_root) / "participants.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["participant_id"])

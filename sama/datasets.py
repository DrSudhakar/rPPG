"""Dataset registry: every public rPPG dataset SAMA can train and test on, behind one interface.

    python -m sama datasets                 # what is actually on disk under paths.public_root
    python -m sama datasets --root <dir>    # probe another folder
    python -m sama train --set training.datasets="[PURE, MMPD, RLAP]"

Each dataset is an adapter with two jobs: find its recordings under a root folder (`discover`) and
turn one recording into a `RealRecording` (`load`, see sama.loaders). Everything downstream --
training, cross-validation, the gate, the pilot analysis -- consumes `RealRecording` and does not
know or care which dataset it came from, so adding a dataset needs no changes anywhere else.

Supported (`NAMES`): UBFC-rPPG, PURE, VIPL-HR, iBVP, SUMS, LADH, M3PD, RLAP, MMPD, PILOT.

Layout on disk. Put each dataset in its own folder under `paths.public_root` and name that folder
after the dataset, e.g. `data/raw/MMPD/`, `data/raw/RLAP/`. The folder name is what tells two
otherwise identical layouts apart (SUMS, LADH and RLAP all come from the same PhysRecorder tool),
so it matters; `python -m sama datasets` prints what each folder was recognised as, which is the
quickest way to confirm a download is in the right shape before starting a long training run.

    data/raw/
      UBFC-rPPG/subject*/{vid.avi, ground_truth.txt}
      PURE/{NN-NN/Image*.png, NN-NN.json}
      VIPL-HR/data/pNN/vN/sourceN/{video.avi, wave.csv, gt_HR.csv, time.txt}
      iBVP/pNN_x/{pNN_x_rgb/*.bmp, pNN_x_t/, pNN_x_bvp.csv}
      SUMS/pNNN/vNN/{video_*.avi, BVP.csv, HR.csv, SpO2.csv, frames_timestamp.csv}
      LADH/<MM_DD>/p_<MM_DD>_<name>/vNN/{video_RGB_H264.avi, BVP.csv, frames_timestamp_RGB.csv}
      M3PD/<subject>/{dual_camera_session_*/front_camera_*.mp4, vNN/BVP.csv}
      RLAP/pNNN/<task>/{video_ZIP_H264.mp4 | pictures_ZIP_RAW_RGB/, BVP.csv, frames_timestamp.csv}
      MMPD/subject*/pN_*.mat

PILOT is not here: the pilot's own sessions carry a synchronised reference signal, protocol blocks
and a grey card, and are loaded by sama.pilotitems (`training.datasets: [PILOT]`).

Reference signals. The waveform file of each dataset (`wave.csv`, `BVP.csv`, `GT_ppg`, ...) is the
network's training target and the source of reference beats. Column layouts are not documented by
most of these datasets, so `read_signal_csv` infers them (header or not, with or without a UNIX
timestamp column, seconds/ms/ns) rather than hard-coding a schema that would silently misread a
variant; what it inferred is reported by `probe`. When a file carries timestamps they are used to
place the waveform on the video clock; when it does not, the waveform is assumed to span the video
uniformly, which is what every one of these datasets intends.

What these datasets cannot give you (I-03): none has a white reference or colour chart in view, so
any ITA° estimated from them is uncalibrated (`training.public_ita`), and none is recorded on the
pilot's compression ladder, so they cannot be used for the proposal's bitrate analysis.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NAMES = ["UBFC-rPPG", "PURE", "VIPL-HR", "iBVP", "SUMS", "LADH", "M3PD", "RLAP", "MMPD", "PILOT"]
PUBLIC_NAMES = [n for n in NAMES if n != "PILOT"]
IMAGE_EXT = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


# ------------------------------------------------------------------ tolerant readers ----
def _numbers(row: list[str]) -> list[float] | None:
    try:
        return [float(x) for x in row if x.strip() != ""]
    except ValueError:
        return None


def _as_unix_seconds(v: np.ndarray) -> np.ndarray | None:
    """Recognise a UNIX timestamp column and return it in seconds, else None.

    PhysRecorder-family files store UNIX time in seconds, milliseconds or nanoseconds depending on
    the dataset; VIPL-HR's time.txt stores milliseconds since the recording started. Anything that
    is monotonic and spans a plausible number of seconds is accepted as a clock.
    """
    if v.size < 2 or not np.all(np.isfinite(v)):
        return None
    d = np.diff(v)
    # Must be non-decreasing and actually advance. Note: comparing against v[0] with np.allclose
    # would be wrong here -- a 10 s span is "close" to 1.7e9 at the default rtol.
    if not np.all(d >= 0) or float(v[-1] - v[0]) <= 0:
        return None
    m = float(np.median(np.abs(v)))
    for scale in (1.0, 1e-3, 1e-6, 1e-9):
        s = v * scale
        span, step = float(s[-1] - s[0]), float(np.median(np.diff(s)))
        if 0.2 < span < 24 * 3600 and 1e-4 < step < 5.0:
            # absolute UNIX seconds (>1e9) or a relative clock starting near zero
            if m * scale > 1e8 or s[0] < 3600:
                return s
    return None


def read_signal_csv(path, log=None) -> tuple[np.ndarray | None, np.ndarray, dict]:
    """Read a reference waveform (BVP/PPG/HR) from a CSV or TXT file of unknown layout.

    Returns (t_seconds | None, values, info). `t` is None when the file carries no clock, in which
    case the caller spreads `values` uniformly over the video. `info` records what was inferred so
    `probe` can show it.
    """
    path = Path(path)
    rows, header = [], None
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if len(row) == 1 and (" " in row[0] or "\t" in row[0]):
                row = row[0].replace("\t", " ").split()
            nums = _numbers(row)
            if nums is None:
                if i == 0:
                    header = [c.strip().lower() for c in row]
                continue
            rows.append(nums)
    if not rows:
        raise ValueError(f"{path}: no numeric rows found")
    width = max(len(r) for r in rows)
    a = np.array([r + [np.nan] * (width - len(r)) for r in rows], float)
    info = {"file": path.name, "rows": int(a.shape[0]), "columns": int(width), "header": header}

    if width == 1:
        info["layout"] = "value only (no clock)"
        return None, a[:, 0], info

    named = {}
    if header and len(header) >= width:
        for j, h in enumerate(header[:width]):
            if any(k in h for k in ("time", "timestamp", "unix", "stamp")):
                named["t"] = j
            elif any(k in h for k in ("bvp", "ppg", "wave", "pleth", "value", "hr", "spo2", "signal")):
                named.setdefault("v", j)
    t_idx = named.get("t")
    if t_idx is None:
        for j in range(width):
            if _as_unix_seconds(a[:, j]) is not None:
                t_idx = j
                break
    if t_idx is None:
        # no clock: take the named column, else the one that varies most (the waveform, not an
        # index or a constant flag)
        v_idx = named.get("v", int(np.argmax([np.nanstd(a[:, j]) for j in range(width)])))
        info["layout"] = f"no clock column; values from column {v_idx}"
        return None, a[:, v_idx], info

    t = _as_unix_seconds(a[:, t_idx])
    if t is None:                                   # named a time column that is not usable
        v_idx = named.get("v", 1 if width > 1 else 0)
        info["layout"] = f"unusable clock in column {t_idx}; values from column {v_idx}"
        return None, a[:, v_idx], info
    v_idx = named.get("v")
    if v_idx is None or v_idx == t_idx:
        others = [j for j in range(width) if j != t_idx]
        # prefer the column that varies most (the waveform, not a constant or a quality flag)
        v_idx = max(others, key=lambda j: float(np.nanstd(a[:, j])))
    info["layout"] = f"clock from column {t_idx}, values from column {v_idx}"
    info["fs_hz"] = round(1.0 / float(np.median(np.diff(t))), 3) if t.size > 1 else None
    if log:
        log(f"[{path.parent.name}/{path.name}] {info['layout']} ({info['rows']} rows)")
    return t, a[:, v_idx], info


def read_timestamps_csv(path) -> np.ndarray | None:
    """Per-frame UNIX timestamps (frames_timestamp.csv / time.txt), in seconds, or None."""
    try:
        t, v, _ = read_signal_csv(path)
    except (OSError, ValueError):
        return None
    if t is not None and t.size:
        return t
    s = _as_unix_seconds(np.asarray(v, float))
    return s if s is not None else None


# ------------------------------------------------------------------ frame sequences -----
def signals_from_frames(frame_paths: list[Path], tracker, ts: np.ndarray | None = None):
    """Reduce an image sequence to the same VideoSignals a video file would give."""
    from PIL import Image

    from .video import FrameMeasurer, VideoSignals
    if not frame_paths:
        raise ValueError("no frames to read")
    meas = FrameMeasurer()
    rows = []
    for p in frame_paths:
        fr = np.asarray(Image.open(p).convert("RGB"))
        rows.append(meas(fr, tracker.update(fr)))
    n = len(rows)
    if ts is not None and ts.size >= n:
        t = np.asarray(ts[:n], float) - float(ts[0])
    else:
        t = np.arange(n) / 30.0
    fps = (n - 1) / max(float(t[-1]), 1e-6) if n > 1 else 30.0
    return VideoSignals(t, np.stack([r.patches for r in rows]), np.stack([r.pix for r in rows]),
                        np.array([r.skin_fraction for r in rows]), np.array([r.motion for r in rows]),
                        np.array([r.brightness for r in rows]), np.stack([r.box for r in rows]),
                        float(fps), tracker.source, {"frames": n, "source": "image_sequence"})


def sorted_frames(d: Path) -> list[Path]:
    def key(p: Path):
        digits = re.findall(r"\d+", p.stem)
        return (int(digits[-1]) if digits else 0, p.name)
    return sorted((p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXT), key=key)


def _recording(vs, name, dataset, participant, t_ref, wave, meta, fallback_fs=None):
    """Attach a reference waveform to measured video signals -> RealRecording."""
    from .loaders import RealRecording, ppg_beats
    wave = np.asarray(wave, float)
    ok = np.isfinite(wave)
    if ok.sum() < 8:
        raise ValueError(f"{name}: reference waveform has too few usable samples ({int(ok.sum())})")
    wave = np.interp(np.arange(wave.size), np.flatnonzero(ok), wave[ok]) if not ok.all() else wave
    if t_ref is None:
        # no clock in the reference file: spread it uniformly over the measured video
        span = float(vs.t[-1]) if vs.t.size > 1 else wave.size / (fallback_fs or 30.0)
        t_ref = np.linspace(0.0, span, wave.size)
    else:
        t_ref = np.asarray(t_ref, float) - float(t_ref[0])
    fs_ref = 1.0 / float(np.median(np.diff(t_ref))) if t_ref.size > 1 else (fallback_fs or 30.0)
    beats = ppg_beats(wave, fs_ref) + float(t_ref[0])
    ppg_on_frames = np.interp(vs.t, t_ref, wave)
    return RealRecording(name, vs.patches, vs.fps, beats, dataset, ppg_on_frames, vs.pix, vs.motion,
                         vs.skin_fraction, participant, dict(meta, fs_ref=round(float(fs_ref), 3),
                                                             tracker=vs.tracker))


def _recording_from_hr(vs, name, dataset, participant, t_hr, hr, meta):
    """Build a recording from an HR series instead of a waveform.

    Some releases (VIPL-HR without wave.csv) ship only beats-per-minute over time. Peak-picking
    such a series would be meaningless, so instead the HR is integrated into a beat train: window
    HR comes out right, and `ppg` stays None so training falls back to the spectral loss
    (`training.loss: spectral`) rather than fitting a waveform that was never measured.
    """
    from .loaders import RealRecording
    hr = np.asarray(hr, float)
    ok = np.isfinite(hr) & (hr > 20) & (hr < 240)
    if ok.sum() < 2:
        raise ValueError(f"{name}: HR ground truth has too few usable samples ({int(ok.sum())})")
    hr = hr[ok]
    t_hr = (np.asarray(t_hr, float)[ok] - float(np.asarray(t_hr, float)[ok][0])) if t_hr is not None \
        else np.linspace(0.0, float(vs.t[-1]) if vs.t.size > 1 else hr.size, hr.size)
    grid = np.arange(0.0, float(t_hr[-1]), 1.0 / 256)
    phase = np.cumsum(np.interp(grid, t_hr, hr) / 60.0) / 256.0        # beats elapsed
    n_beats = int(np.floor(phase[-1]))
    beats = np.interp(np.arange(1, n_beats + 1), phase, grid) if n_beats >= 1 else np.array([])
    return RealRecording(name, vs.patches, vs.fps, beats, dataset, None, vs.pix, vs.motion,
                         vs.skin_fraction, participant,
                         dict(meta, reference_kind="hr_series", tracker=vs.tracker))


def _pick(paths: list[Path], prefer: tuple[str, ...], avoid: tuple[str, ...] = ()) -> Path | None:
    """Choose one file by name preference (e.g. the RGB face video, not the IR or finger one)."""
    cands = [p for p in paths if not any(a in p.name.lower() for a in avoid)] or list(paths)
    if not cands:
        return None
    def score(p: Path):
        n = p.name.lower()
        return (min((i for i, k in enumerate(prefer) if k in n), default=len(prefer)), len(p.name))
    return sorted(cands, key=score)[0]


# ------------------------------------------------------------------ the adapters --------
@dataclass
class Dataset:
    name: str
    markers: str                                  # one-line description of what discovery looks for
    discover: object                              # (root: Path) -> list[Path]
    load: object                                  # (path: Path, tracker_mode: str) -> RealRecording
    notes: str = ""
    extras: dict = field(default_factory=dict)


def _tracker(mode):
    from .video import FaceTracker
    return FaceTracker(mode=mode if mode != "manual_box" else "auto")


# -- UBFC-rPPG, PURE: already in sama.loaders --------------------------------------------
def _d_ubfc(root: Path):
    return [p.parent for p in sorted(root.glob("**/ground_truth.txt"))]


def _l_ubfc(path: Path, mode: str):
    from .loaders import load_ubfc_rppg
    return load_ubfc_rppg(path, mode)


def _d_pure(root: Path):
    return [Path(str(j)[:-5]) for j in sorted(root.glob("**/[0-9][0-9]-[0-9][0-9].json"))
            if Path(str(j)[:-5]).is_dir()]


def _l_pure(path: Path, mode: str):
    from .loaders import load_pure
    return load_pure(path, mode)


# -- VIPL-HR -----------------------------------------------------------------------------
def _d_vipl(root: Path):
    out = []
    for v in sorted(root.glob("**/*.avi")):
        d = v.parent
        if (d / "wave.csv").exists() or (d / "gt_HR.csv").exists():
            out.append(d)
    return sorted(set(out))


def _l_vipl(path: Path, mode: str):
    from .video import extract_signals
    d = Path(path)
    vid = _pick(sorted(d.glob("*.avi")) + sorted(d.glob("*.mp4")), ("video", ""), avoid=("nir", "ir_"))
    if vid is None:
        raise ValueError(f"{d}: no video file")
    vs = extract_signals(vid, _tracker(mode))
    # .../pNN/vN/sourceM -> participant pNN
    parts = [q for q in d.parts if re.fullmatch(r"p\d+", q, re.I)]
    participant = f"VIPL:{parts[-1]}" if parts else f"VIPL:{d.parent.parent.name}"
    name = "_".join(d.parts[-3:])
    # VIPL-HR frame rates vary (25-30 fps) and frames are dropped, so time.txt is the real clock.
    ts = read_timestamps_csv(d / "time.txt") if (d / "time.txt").exists() else None
    span = float(ts[-1] - ts[0]) if ts is not None and ts.size > 1 else \
        (float(vs.t[-1]) if vs.t.size > 1 else 0.0)
    wave_file = d / "wave.csv"
    if wave_file.exists():
        t, wave, info = read_signal_csv(wave_file)
        if t is None and span > 0:
            t = np.linspace(0.0, span, np.asarray(wave).size)
        return _recording(vs, name, "VIPL-HR", participant, t, wave,
                          {"ref": info, "video": vid.name, "clock": "time.txt" if ts is not None else "video"})
    t, hr, info = read_signal_csv(d / "gt_HR.csv")           # HR-only release: no BVP waveform
    info["note"] = "HR-only ground truth (no BVP waveform in this release); " \
                   "beats are integrated from HR and training falls back to the spectral loss"
    if t is None and span > 0:
        t = np.linspace(0.0, span, np.asarray(hr).size)
    return _recording_from_hr(vs, name, "VIPL-HR", participant, t, hr, {"ref": info, "video": vid.name})


# -- iBVP --------------------------------------------------------------------------------
def _d_ibvp(root: Path):
    return [p.parent for p in sorted(root.glob("**/*_bvp.csv"))]


def _l_ibvp(path: Path, mode: str):
    d = Path(path)
    bvp = sorted(d.glob("*_bvp.csv"))[0]
    stem = bvp.name[:-8]                                        # p01_a_bvp.csv -> p01_a
    rgb = next((c for c in (d / f"{stem}_rgb", d / "rgb") if c.is_dir()),
               next((c for c in sorted(d.iterdir()) if c.is_dir() and "rgb" in c.name.lower()), None))
    t, wave, info = read_signal_csv(bvp)
    tr = _tracker(mode)
    if rgb is not None and sorted_frames(rgb):
        vs = signals_from_frames(sorted_frames(rgb), tr)
    else:
        vid = _pick(sorted(d.glob("*.avi")) + sorted(d.glob("*.mp4")), ("rgb", "video"), avoid=("_t", "thermal"))
        if vid is None:
            raise ValueError(f"{d}: no RGB frame folder (*_rgb/) or RGB video found")
        from .video import extract_signals
        vs = extract_signals(vid, tr)
    return _recording(vs, stem, "iBVP", f"iBVP:{stem.split('_')[0]}", t, wave,
                      {"ref": info, "frames": vs.patches.shape[0]}, fallback_fs=30.0)


# -- the PhysRecorder family: SUMS, LADH, M3PD, RLAP -------------------------------------
PHYSREC_TS = ("frames_timestamp_RGB.csv", "frames_timestamp.csv", "frames_timestamp_IR.csv")


def _physrec_sessions(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.glob("**/BVP.csv")})


def _physrec_video(d: Path):
    vids = [p for p in d.iterdir() if p.suffix.lower() in (".avi", ".mp4", ".mkv", ".mov")]
    return _pick(vids, ("rgb", "zip_h264", "h264", "front", "face", "mjpg", "video"),
                 avoid=("_ir", "ir_", "infrared", "thermal", "finger", "back_camera", "back_"))


def _physrec_frames(d: Path):
    for c in sorted(d.iterdir()):
        if c.is_dir() and ("picture" in c.name.lower() or "rgb" in c.name.lower()) and sorted_frames(c):
            return sorted_frames(c)
    return []


def _load_physrec(path: Path, mode: str, dataset: str, participant: str, label_dir: Path | None = None):
    """One PhysRecorder-style session: a video (or PNG sequence) plus BVP.csv on a UNIX clock."""
    d, ld = Path(path), Path(label_dir or path)
    bvp = ld / "BVP.csv"
    if not bvp.exists():
        raise ValueError(f"{ld}: no BVP.csv")
    t, wave, info = read_signal_csv(bvp)
    ts = next((read_timestamps_csv(ld / n) for n in PHYSREC_TS if (ld / n).exists()), None)
    tr = _tracker(mode)
    vid, frames = _physrec_video(d), None
    if vid is None:
        frames = _physrec_frames(d)
        if not frames:
            raise ValueError(f"{d}: no RGB video or frame folder found")
        vs = signals_from_frames(frames, tr, ts)
    else:
        from .video import extract_signals
        vs = extract_signals(vid, tr)
    # Both clocks are UNIX time when timestamps are present: rebase the reference onto frame 0.
    if t is not None and ts is not None and ts.size:
        t = t - float(ts[0])
        vs.t[:] = vs.t - vs.t[0]
        t_ref = t
    else:
        t_ref = None if t is None else t - float(t[0])
    extra = {"ref": info, "video": vid.name if vid else f"{len(frames or [])} frames",
             "clock": "frame timestamps" if ts is not None else "uniform"}
    for side in ("HR.csv", "SpO2.csv", "RR.csv"):
        if (ld / side).exists():
            extra.setdefault("also", []).append(side)
    return _recording(vs, "_".join(d.parts[-2:]), dataset, participant, t_ref, wave, extra)


def _subject_of(d: Path, pattern: str, fallback_depth: int = 2) -> str:
    for q in reversed(d.parts):
        if re.fullmatch(pattern, q, re.I):
            return q
    return d.parts[-fallback_depth] if len(d.parts) >= fallback_depth else d.name


def _d_sums(root: Path):
    return _physrec_sessions(root)


def _l_sums(path: Path, mode: str):
    return _load_physrec(path, mode, "SUMS", f"SUMS:{_subject_of(Path(path), r'p\d+')}")


def _d_ladh(root: Path):
    return _physrec_sessions(root)


def _l_ladh(path: Path, mode: str):
    # .../<MM_DD>/p_<MM_DD>_<name>/vNN  -> participant is the person, across days
    d = Path(path)
    who = next((q for q in reversed(d.parts) if q.lower().startswith("p_")), d.parts[-2])
    person = who.split("_")[-1] if "_" in who else who
    return _load_physrec(d, mode, "LADH", f"LADH:{person}")


def _d_m3pd(root: Path):
    out = set()
    for b in sorted(root.glob("**/BVP.csv")):
        subj = b.parent.parent                       # <subject>/vNN/BVP.csv
        sess = sorted(subj.glob("dual_camera_session_*")) or \
            [p.parent for p in sorted(subj.glob("**/front_camera_*.mp4"))]
        out.add(sess[0] if sess else b.parent)
    return sorted(out)


def _l_m3pd(path: Path, mode: str):
    d = Path(path)
    label_dir = d if (d / "BVP.csv").exists() else next(
        (c for c in sorted(d.parent.iterdir()) if c.is_dir() and (c / "BVP.csv").exists()), d)
    subj = _subject_of(d, r"\d{3}", fallback_depth=2)
    return _load_physrec(d, mode, "M3PD", f"M3PD:{subj}", label_dir=label_dir)


def _d_rlap(root: Path):
    return _physrec_sessions(root)


def _l_rlap(path: Path, mode: str):
    return _load_physrec(path, mode, "RLAP", f"RLAP:{_subject_of(Path(path), r'p\d+')}")


# -- MMPD -------------------------------------------------------------------------------
def _d_mmpd(root: Path):
    return sorted(p for p in root.glob("**/*.mat") if re.fullmatch(r"p\d+_\d+", p.stem, re.I))


def _read_mat(path: Path) -> dict:
    from scipy.io import loadmat
    try:
        return {k: v for k, v in loadmat(str(path)).items() if not k.startswith("__")}
    except NotImplementedError:                      # MATLAB v7.3 == HDF5
        try:
            import h5py
        except ImportError as e:
            raise RuntimeError(f"{path.name} is a MATLAB v7.3 file; `pip install h5py` to read it") from e
        with h5py.File(path, "r") as f:
            return {k: np.array(f[k]).T for k in f}


def _l_mmpd(path: Path, mode: str):
    from .video import FrameMeasurer, VideoSignals
    p = Path(path)
    m = _read_mat(p)
    vkey = next((k for k in ("video", "frames", "Video") if k in m), None)
    bkey = next((k for k in ("GT_ppg", "gt_ppg", "ppg", "BVP", "bvp") if k in m), None)
    if vkey is None or bkey is None:
        raise ValueError(f"{p.name}: expected 'video' and 'GT_ppg' in the .mat, found {sorted(m)}")
    vid = np.asarray(m[vkey])
    if vid.ndim != 4:
        raise ValueError(f"{p.name}: 'video' should be (T, H, W, 3), got {vid.shape}")
    if vid.shape[-1] != 3 and vid.shape[1] == 3:                 # (T, 3, H, W)
        vid = np.transpose(vid, (0, 2, 3, 1))
    if vid.dtype != np.uint8:                                    # MMPD stores float in [0, 1]
        mx = float(np.nanmax(vid)) if vid.size else 1.0
        vid = np.clip(vid * (255.0 if mx <= 1.5 else 1.0), 0, 255).astype(np.uint8)
    wave = np.asarray(m[bkey], float).ravel()
    tr, meas = _tracker(mode), FrameMeasurer()
    rows = [meas(fr, tr.update(fr)) for fr in vid]
    n, fps = len(rows), 30.0
    t = np.arange(n) / fps
    vs = VideoSignals(t, np.stack([r.patches for r in rows]), np.stack([r.pix for r in rows]),
                      np.array([r.skin_fraction for r in rows]), np.array([r.motion for r in rows]),
                      np.array([r.brightness for r in rows]), np.stack([r.box for r in rows]),
                      fps, tr.source, {"frames": n, "source": "mat"})
    labels = {k: (v.ravel()[0] if isinstance(v, np.ndarray) and v.size == 1 else v)
              for k, v in m.items() if k not in (vkey, bkey) and np.asarray(v).size <= 4}
    labels = {k: (str(v[0]) if isinstance(v, np.ndarray) and v.dtype.kind in "OU" and v.size else
                  (v.item() if isinstance(v, np.ndarray) and v.size == 1 else str(v))) for k, v in labels.items()}
    return _recording(vs, p.stem, "MMPD", f"MMPD:{p.parent.name}", None, wave,
                      {"labels": labels, "mat": p.name}, fallback_fs=fps)


# ---------------------------------------------------------------------- the registry ----
REGISTRY: dict[str, Dataset] = {
    "UBFC-rPPG": Dataset("UBFC-rPPG", "subject*/ground_truth.txt + vid.avi", _d_ubfc, _l_ubfc,
                         "Contact PPG in ground_truth.txt (line 1 wave, line 3 time)."),
    "PURE": Dataset("PURE", "NN-NN.json + NN-NN/Image*.png", _d_pure, _l_pure,
                    "PNG sequence with nanosecond names; waveform in /FullPackage."),
    "VIPL-HR": Dataset("VIPL-HR", "folder with video.avi + wave.csv (or gt_HR.csv)", _d_vipl, _l_vipl,
                       "9 scenarios x up to 4 sources per subject; NIR sources are skipped. "
                       "Variable frame rate, so frame timestamps (time.txt) are used when present."),
    "iBVP": Dataset("iBVP", "pNN_x/pNN_x_bvp.csv + pNN_x_rgb/", _d_ibvp, _l_ibvp,
                    "RGB frames only; the paired thermal stream (*_t/) is not used."),
    "SUMS": Dataset("SUMS", "pNNN/vNN/BVP.csv + video (PhysRecorder)", _d_sums, _l_sums,
                    "Face camera only; the fingertip camera is skipped. Recorded at altitude, "
                    "so SpO2 varies more than in the other datasets."),
    "LADH": Dataset("LADH", "<MM_DD>/p_<MM_DD>_<name>/vNN/BVP.csv + video_RGB_H264.avi", _d_ladh, _l_ladh,
                    "RGB only (the IR stream is skipped). The same person recurs across days and "
                    "is kept as one participant so cross-validation folds stay honest."),
    "M3PD": Dataset("M3PD", "<subject>/dual_camera_session_*/front_camera_*.mp4 + vNN/BVP.csv", _d_m3pd, _l_m3pd,
                    "Front (face) camera only; the rear fingertip view is skipped."),
    "RLAP": Dataset("RLAP", "pNNN/<task>/BVP.csv + video_ZIP_H264.mp4 or pictures_ZIP_RAW_RGB/", _d_rlap, _l_rlap,
                    "Prefers the lossless PNG sequence when the compressed videos are absent."),
    "MMPD": Dataset("MMPD", "subject*/pN_M.mat", _d_mmpd, _l_mmpd,
                    "Frames and waveform live inside the .mat. Its per-video labels (light, motion, "
                    "skin_color, ...) are carried into the recording's meta."),
    "PILOT": Dataset("PILOT", "pilot_data/sessions/<id>/ (loaded by sama.pilotitems)",
                     lambda root: [], lambda path, mode: (_ for _ in ()).throw(
                         RuntimeError("PILOT sessions are loaded by sama.pilotitems, not here")),
                     "The pilot's own sessions: synchronised reference, protocol blocks, grey card."),
}


def _named_in_path(path: Path, root: Path) -> str | None:
    """Which dataset does this path's folder name claim to be?"""
    try:
        parts = [q.lower() for q in path.relative_to(root).parts]
    except ValueError:
        parts = [q.lower() for q in path.parts]
    for n in PUBLIC_NAMES:
        key = n.lower()
        if any(q == key or q.replace("_", "-") == key or q.startswith(key + "_") for q in parts):
            return n
    return None


def discover(root, only: list[str] | None = None) -> list[tuple[str, Path]]:
    """Find every recording under `root`, as [(dataset name, path)].

    A dataset's own folder name (data/raw/RLAP/...) decides which adapter claims a recording; this
    matters for the PhysRecorder-family datasets (SUMS, LADH, M3PD, RLAP), whose layouts are hard
    to tell apart. Unnamed folders fall back to structure, which can only distinguish M3PD.
    """
    root = Path(root)
    if not root.exists():
        return []
    wanted = [n for n in (only or PUBLIC_NAMES) if n in REGISTRY and n != "PILOT"]
    hits: dict[Path, str] = {}
    for name in wanted:
        for p in REGISTRY[name].discover(root):
            claimed = _named_in_path(p, root)
            if claimed is not None and claimed != name:
                continue                              # the folder says it belongs to another dataset
            if claimed is None and name in ("SUMS", "LADH", "RLAP") and \
                    _d_m3pd(root) and p in set(_d_m3pd(root)):
                continue
            hits.setdefault(p, name)
    out = [(n, p) for p, n in hits.items()]
    return sorted(out, key=lambda kp: (NAMES.index(kp[0]), str(kp[1])))


def load(kind: str, path, tracker_mode: str = "auto"):
    if kind not in REGISTRY:
        raise ValueError(f"unknown dataset {kind!r}; known: {', '.join(NAMES)}")
    return REGISTRY[kind].load(Path(path), tracker_mode)


def probe(root, only: list[str] | None = None, pilot_root=None) -> dict:
    """What is on disk, per dataset: counts, participants, and one example path each."""
    found = discover(root, only)
    per: dict[str, dict] = {}
    for name, path in found:
        e = per.setdefault(name, {"recordings": 0, "example": str(path), "folders": set()})
        e["recordings"] += 1
        e["folders"].add(str(Path(path).parent))
    for e in per.values():
        e["folders"] = len(e["folders"])
    out = {"root": str(root), "datasets": per,
           "missing": [n for n in (only or PUBLIC_NAMES) if n != "PILOT" and n not in per]}
    if pilot_root is not None:
        sess = sorted(Path(pilot_root).glob("sessions/*/session.json"))
        out["datasets"].setdefault("PILOT", {"recordings": len(sess), "folders": len(sess),
                                             "example": str(sess[0].parent) if sess else ""})
        if not sess:
            out["datasets"].pop("PILOT")
            out["missing"].append("PILOT")
    return out


def probe_text(root, only: list[str] | None = None, pilot_root=None) -> str:
    p = probe(root, only, pilot_root)
    lines = [f"Datasets under {p['root']}" + (f" (and pilot under {pilot_root})" if pilot_root else "")]
    if not p["datasets"]:
        lines.append("  nothing recognised -- see the layout in sama/datasets.py or data/DATA_INVENTORY.md")
    for name, e in p["datasets"].items():
        lines.append(f"  {name:<10} {e['recordings']:>5} recording(s)   e.g. {e['example']}")
        if REGISTRY[name].notes:
            lines.append(f"             {REGISTRY[name].notes}")
    if p["missing"]:
        lines.append("  not found: " + ", ".join(p["missing"]))
        lines.append("  (each dataset goes in its own folder named after it, e.g. "
                     f"{Path(p['root']) / 'MMPD'})")
    return "\n".join(lines)

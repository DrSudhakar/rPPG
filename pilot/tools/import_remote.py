#!/usr/bin/env python3
"""Turn downloaded webexp/ (Firebase) sessions into the same session-folder layout the rest of the
pipeline reads, so `python -m sama prepare/qc/pilot` can run on them exactly as on lab sessions.

Expected input layout (after you download files from the Firebase console -- see
webexp/REMOTE_PILOT_GUIDE.md "Collecting what came in"):

    <downloads>/<pid>/webcam.webm        (or .mp4)
    <downloads>/<pid>/metadata.json      (the sidecar app.js uploads next to the video)

Usage:
    python pilot/tools/import_remote.py <downloads_dir> --out pilot_data_REMOTE

Then analyse the remote arm SEPARATELY from the lab pilot (it has no lossless source, so the
compression ladder cannot be built from it, and there is no external Monk rater -- see the guide):

    python -m sama prepare --set paths.pilot_root=pilot_data_REMOTE
    python -m sama qc      --set paths.pilot_root=pilot_data_REMOTE
    python -m sama pilot   --set paths.pilot_root=pilot_data_REMOTE --set paths.outputs=results/pilot_remote \\
                           --set pilot.ladder.enabled=false --set ita.source=monk --set raters.n=1
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

FITZ_MAP = {"always_burns": 1, "usually_burns": 2, "sometimes_burns": 3, "rarely_burns": 4,
            "never_burns_tans_a_lot": 5, "never_burns_always_dark": 6}


def import_one(src: Path, out_root: Path, log=print) -> str | None:
    meta_path = src / "metadata.json"
    if not meta_path.exists():
        log(f"[skip] {src}: no metadata.json")
        return None
    video = next((src / n for n in ("webcam.webm", "webcam.mp4") if (src / n).exists()), None)
    if video is None:
        log(f"[skip] {src}: no webcam.webm/webcam.mp4")
        return None
    meta = json.loads(meta_path.read_text())
    pid = meta.get("participant_id") or src.name
    sdir = out_root / "sessions" / pid
    sdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(video, sdir / video.name)

    events = meta.get("events", [])
    t0_wall = next((e["wall"] for e in events if e.get("type") == "recording_start"), meta.get("recorded_at"))
    dev = meta.get("device") or {}
    settings = dev.get("settings") or {}
    session_json = {
        "participant_id": pid, "date": (meta.get("recorded_at") or "")[:10], "operator": "",
        "source": "browser_webm", "app_version": meta.get("app_version"),
        "video_t0_wall": _to_epoch(t0_wall),
        "webcam": {"file": video.name, "device": dev.get("label"),
                   "size": f"{settings.get('width', '?')}x{settings.get('height', '?')}",
                   "fps": settings.get("frameRate"), "codec": meta.get("recording", {}).get("mime_type")},
        "boxes": {"face_box": None, "white_box": None, "chart_corners": None},
        "capture_device": (meta.get("ratings") or {}).get("capture_device"),   # webcam_builtin | webcam_external | mobile_stable | mobile_handheld
        "white_reference": "none",   # no card at home: camera ITA is skipped; strata come from the Monk self-rating
        "reference": {"device": "none (no reference signal collected remotely)", "file": None},
        "notes": (f"remote unsupervised session; capture device={meta.get('ratings', {}).get('capture_device')}; "
                  f"lighting self-report={meta.get('ratings', {}).get('lighting_self_report')}; "
                  f"browser={dev.get('user_agent')}; timezone={dev.get('timezone')}; consent_time={meta.get('consent_time')}"),
    }
    (sdir / "session.json").write_text(json.dumps(session_json, indent=1))

    with open(sdir / "events.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["wall_time", "event", "block", "light", "note"])
        w.writeheader()
        for e in events:
            w.writerow({"wall_time": _to_epoch(e.get("wall")), "event": e.get("type"), "block": e.get("block", ""),
                       "light": "", "note": ""})

    ratings = meta.get("ratings") or {}
    _append_row(out_root / "participants.csv",
                ["participant_id", "age", "sex", "monk_self", "fitzpatrick_self", "beard", "spectacles", "bindi",
                 "makeup", "head_covering", "hr_lowering_medication", "caffeine_2h", "consent_tier", "notes"],
                {"participant_id": pid, "monk_self": ratings.get("monk"),
                 "fitzpatrick_self": FITZ_MAP.get(ratings.get("fitzpatrick")),
                 "consent_tier": "research_only",
                 "notes": f"remote/unsupervised; device={ratings.get('capture_device')}; self-report only, no external rater"})
    return pid


def _to_epoch(iso) -> float | None:
    if not iso:
        return None
    from datetime import datetime
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _append_row(path: Path, fieldnames: list, row: dict):
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("downloads_dir", help="folder with one subfolder per participant (webcam.* + metadata.json)")
    ap.add_argument("--out", default="pilot_data_REMOTE", help="where to write sessions/, participants.csv (default: %(default)s)")
    a = ap.parse_args()
    src_root, out_root = Path(a.downloads_dir), Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    done = [p for d in sorted(src_root.iterdir()) if d.is_dir() and (p := import_one(d, out_root))]
    print(f"Imported {len(done)} session(s) into {out_root}: {', '.join(done) if done else '(none)'}")


if __name__ == "__main__":
    main()

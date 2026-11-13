"""Regression test for pilot/tools/import_remote.py -- run with: python3 pilot/tools/test_import_remote.py"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import import_remote as ir  # noqa: E402

META = {
    "participant_id": "P99", "app_version": "1.0",
    "recorded_at": "2026-11-05T14:32:10.000Z", "consent_time": "2026-11-05T14:30:00.000Z",
    "device": {"label": "FaceTime HD Camera", "user_agent": "Mozilla/5.0 Chrome/120",
               "timezone": "America/New_York", "settings": {"width": 1280, "height": 720, "frameRate": 30}},
    "recording": {"mime_type": "video/webm;codecs=vp9"},
    "ratings": {"monk": 4, "fitzpatrick": "sometimes_burns", "lighting_self_report": "room_light", "capture_device": "mobile_stable"},
    "events": [
        {"type": "consent_given", "wall": "2026-11-05T14:30:00.000Z"},
        {"type": "recording_start", "wall": "2026-11-05T14:32:10.000Z", "block": "motion_check"},
        {"type": "recording_stop", "wall": "2026-11-05T14:40:50.000Z"},
    ],
}


def main():
    tmp = Path(tempfile.mkdtemp(prefix="import_remote_test_"))
    try:
        src, out = tmp / "downloads" / "P99", tmp / "pilot_data_REMOTE"
        src.mkdir(parents=True)
        (src / "metadata.json").write_text(json.dumps(META))
        (src / "webcam.webm").write_bytes(b"not a real video, just a placeholder")

        pid = ir.import_one(src, out, log=lambda *a: None)
        assert pid == "P99", pid

        sdir = out / "sessions" / "P99"
        assert (sdir / "webcam.webm").exists()
        sess = json.loads((sdir / "session.json").read_text())
        assert sess["webcam"]["device"] == "FaceTime HD Camera"
        assert sess["white_reference"] == "none"
        assert sess["capture_device"] == "mobile_stable"
        assert sess["video_t0_wall"] is not None
        assert sess["boxes"]["white_box"] is None

        events_text = (sdir / "events.csv").read_text()
        assert "recording_start" in events_text and "recording_stop" in events_text

        rows = (out / "participants.csv").read_text().splitlines()
        assert rows[0].startswith("participant_id,age,sex,monk_self,fitzpatrick_self")
        assert rows[1].startswith("P99,,,4,3,")
        assert "no external rater" in rows[1] and "device=mobile_stable" in rows[1]

        assert not (out / "monk_ratings.csv").exists(), "must not fabricate an external-rater row"

        # missing metadata.json / video -> skipped cleanly, no exception
        empty_src = tmp / "downloads" / "P_empty"
        empty_src.mkdir()
        assert ir.import_one(empty_src, out, log=lambda *a: None) is None

        # second real participant appends, doesn't clobber the first row
        src2 = tmp / "downloads" / "P100"
        src2.mkdir()
        meta2 = dict(META, participant_id="P100")
        (src2 / "metadata.json").write_text(json.dumps(meta2))
        (src2 / "webcam.webm").write_bytes(b"x")
        ir.import_one(src2, out, log=lambda *a: None)
        rows2 = (out / "participants.csv").read_text().splitlines()
        assert len(rows2) == 3, rows2  # header + P99 + P100

        print("ALL IMPORT_REMOTE CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

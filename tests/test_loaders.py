"""Loaders tested on synthetic files written in the UBFC-rPPG layout (no real data available)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sama import loaders, methods, signals  # noqa: E402


class UbfcLayout(unittest.TestCase):
    def test_roundtrip(self):
        from sama.video import write_video   # ffmpeg CLI, never OpenCV (I-36)
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "raw/UBFC/subject1"
            sub.mkdir(parents=True)
            fps, n, bpm = 30.0, 900, 72.0
            t = np.arange(n) / fps
            rng = np.random.default_rng(0)

            def frames():
                for k in range(n):
                    v = 1 + 0.01 * np.sin(2 * np.pi * bpm / 60 * t[k])
                    rgb = np.array([170, 120, 90]) * np.array([0.33 * v + 0.67, 0.77 * v + 0.23, 0.53 * v + 0.47])
                    fr = np.zeros((120, 160, 3)) + rgb + rng.normal(0, 2, (120, 160, 3))
                    yield np.clip(np.round(fr), 0, 255).astype(np.uint8)
            write_video(sub / "vid.avi", frames(), fps, "ffv1")
            tp = np.arange(int(30 * 60)) / 60.0
            ppg = np.sin(2 * np.pi * bpm / 60 * tp)
            (sub / "ground_truth.txt").write_text(" ".join(map(str, ppg)) + "\n" + " ".join(["72"] * tp.size) + "\n" + " ".join(map(str, tp)))
            found = loaders.discover(Path(d) / "raw")
            self.assertEqual(found[0][0], "UBFC-rPPG")
            r = loaders.load(*found[0], tracker_mode="fixed_box")
            self.assertEqual(r.patches.shape[1:], (30, 3))
            h, _, _ = signals.hr_from_signal(methods.pos(r.patches.mean(1), r.fps), r.fps)
            self.assertAlmostEqual(h, bpm, delta=1.5)
            self.assertAlmostEqual(60 / np.median(np.diff(r.beats)), bpm, delta=1.0)


if __name__ == "__main__":
    unittest.main()

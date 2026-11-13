"""Every dataset adapter in sama.datasets, exercised on synthetic files written in that dataset's
documented layout (no real data is available here, and none may be redistributed anyway).

Each case writes a tiny pulsatile recording in the real folder/file shape, then checks that
`discover` recognises it as the right dataset and `load` returns a RealRecording whose recovered
heart rate matches the synthetic one. That is the property that matters: if a layout or column
order is misread, the HR comes out wrong and the test fails, rather than training silently on
mislabelled data.

Run: python -m unittest tests.test_datasets   (or: python -m unittest discover -s tests)
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sama import datasets as D  # noqa: E402
from sama import methods, signals  # noqa: E402

FPS, N, BPM, AMP = 30.0, 450, 72.0, 0.04          # 15 s is enough for a stable HR estimate
H, W = 96, 128


def _frames(n=N, fps=FPS, bpm=BPM, seed=0):
    """A synthetic face: a skin-coloured patch whose brightness pulses at `bpm`."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    for k in range(n):
        v = 1 + AMP * np.sin(2 * np.pi * bpm / 60 * t[k])
        rgb = np.array([170, 120, 90]) * np.array([0.33 * v + 0.67, 0.77 * v + 0.23, 0.53 * v + 0.47])
        fr = np.zeros((H, W, 3)) + rgb + rng.normal(0, 1.5, (H, W, 3))
        yield np.clip(np.round(fr), 0, 255).astype(np.uint8)


def _wave(n_samples, fs, bpm=BPM, t0=0.0):
    t = t0 + np.arange(n_samples) / fs
    return t, np.sin(2 * np.pi * bpm / 60 * (t - t0))


def _write_video(path: Path, **kw):
    """FFV1 in .avi (lossless, like RLAP's MJPG/PNG copies); H.264 in .mp4, as those datasets ship."""
    from sama.video import write_video
    path.parent.mkdir(parents=True, exist_ok=True)
    write_video(path, _frames(**kw), FPS, "h264" if path.suffix.lower() == ".mp4" else "ffv1")


def _write_png_sequence(d: Path, n=N):
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(_frames(n=n)):
        Image.fromarray(fr).save(d / f"{i:08d}.png")


def _write_csv(path: Path, t, v, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ([",".join(header)] if header else []) + \
            ([f"{a},{b}" for a, b in zip(t, v)] if t is not None else [str(b) for b in v])
    path.write_text("\n".join(lines) + "\n")


class ReaderTests(unittest.TestCase):
    """read_signal_csv has to cope with layouts these datasets do not document."""

    def test_layout_variants(self):
        t, v = _wave(200, 20.0, t0=1.7e9)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cases = {
                "unix_header.csv": (t, v, ["timestamp", "bvp"]),
                "unix_nohdr.csv": (t, v, None),
                "ms.csv": (t * 1e3, v, ["time", "value"]),
                "ns.csv": (t * 1e9, v, None),
                "value_only.csv": (None, v, None),
            }
            for name, (tt, vv, hdr) in cases.items():
                _write_csv(d / name, tt, vv, hdr)
                got_t, got_v, info = D.read_signal_csv(d / name)
                self.assertEqual(got_v.size, v.size, name)
                np.testing.assert_allclose(got_v, v, atol=1e-6, err_msg=name)
                if tt is None:
                    self.assertIsNone(got_t, name)
                else:
                    self.assertIsNotNone(got_t, f"{name}: clock column not recognised ({info})")
                    self.assertAlmostEqual(float(np.median(np.diff(got_t))), 0.05, places=4, msg=name)

    def test_quality_column_is_not_mistaken_for_the_waveform(self):
        """iBVP ships a near-constant quality flag beside the waveform; pick the waveform."""
        t, v = _wave(200, 30.0, t0=1.7e9)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p01_a_bvp.csv"
            p.write_text("\n".join(f"{a},{b},{1}" for a, b in zip(t, v)) + "\n")
            _, got, _ = D.read_signal_csv(p)
            np.testing.assert_allclose(got, v, atol=1e-6)


class DatasetLayouts(unittest.TestCase):
    def _check(self, root: Path, expect: str, min_hits: int = 1):
        found = D.discover(root)
        kinds = {k for k, _ in found}
        self.assertIn(expect, kinds, f"{expect} not discovered under {root} (found {found})")
        hits = [(k, p) for k, p in found if k == expect]
        self.assertGreaterEqual(len(hits), min_hits)
        rec = D.load(expect, hits[0][1], tracker_mode="fixed_box")
        self.assertEqual(rec.source, expect)
        self.assertEqual(rec.patches.shape[1:], (30, 3))
        self.assertTrue(rec.participant, "participant id must not be empty (CV folds need it)")
        hr, _, _ = signals.hr_from_signal(methods.pos(rec.patches.mean(1), rec.fps), rec.fps)
        self.assertAlmostEqual(hr, BPM, delta=3.0, msg=f"{expect}: rPPG HR off ({hr:.1f})")
        self.assertIsNotNone(rec.ppg, f"{expect}: reference waveform not attached")
        self.assertAlmostEqual(60 / np.median(np.diff(rec.beats)), BPM, delta=2.0,
                               msg=f"{expect}: reference beats off")
        return rec

    # -- VIPL-HR -------------------------------------------------------------------------
    def test_vipl_hr(self):
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "VIPL-HR/data/p12/v3/source2"
            _write_video(s / "video.avi")
            t, v = _wave(300, 20.0)
            _write_csv(s / "wave.csv", None, v)
            _write_csv(s / "gt_HR.csv", None, [BPM] * 15)
            (s / "time.txt").write_text("\n".join(str(int(x * 1000)) for x in np.arange(N) / FPS))
            rec = self._check(Path(d), "VIPL-HR")
            self.assertEqual(rec.participant, "VIPL:p12")

    def test_vipl_hr_without_waveform_release(self):
        """Some VIPL-HR copies ship gt_HR.csv but no wave.csv: load, don't crash."""
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "VIPL-HR/data/p01/v1/source1"
            _write_video(s / "video.avi")
            _write_csv(s / "gt_HR.csv", None, list(np.full(15, BPM)))
            found = [(k, p) for k, p in D.discover(Path(d)) if k == "VIPL-HR"]
            self.assertTrue(found)
            rec = D.load("VIPL-HR", found[0][1], tracker_mode="fixed_box")
            self.assertIn("HR-only", rec.meta["ref"].get("note", ""))
            self.assertIsNone(rec.ppg, "an HR-only release must not fabricate a waveform target")
            self.assertEqual(rec.meta["reference_kind"], "hr_series")
            self.assertAlmostEqual(60 / np.median(np.diff(rec.beats)), BPM, delta=1.0)

    # -- iBVP ----------------------------------------------------------------------------
    def test_ibvp(self):
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "iBVP/p01_a"
            _write_png_sequence(s / "p01_a_rgb", n=240)
            (s / "p01_a_t").mkdir(parents=True, exist_ok=True)     # thermal, must be ignored
            t, v = _wave(240, 30.0)
            (s / "p01_a_bvp.csv").write_text("\n".join(f"{b},1,1" for b in v) + "\n")
            rec = self._check(Path(d), "iBVP")
            self.assertEqual(rec.participant, "iBVP:p01")

    # -- the PhysRecorder family ---------------------------------------------------------
    def _physrec(self, session: Path, video_name="video_ZIP_H264.avi", ts_name="frames_timestamp.csv",
                 extras=("HR.csv", "SpO2.csv"), t0=1.7e9):
        _write_video(session / video_name)
        ft = t0 + np.arange(N) / FPS
        _write_csv(session / ts_name, None, ft)
        t, v = _wave(300, 20.0, t0=t0)
        _write_csv(session / "BVP.csv", t, v, ["timestamp", "bvp"])
        for e in extras:
            _write_csv(session / e, t[::20], np.full(t[::20].size, BPM), ["timestamp", "value"])

    def test_rlap(self):
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "RLAP/p003/R1"
            self._physrec(s, video_name="video_ZIP_H264.mp4", extras=("HR.csv",))
            rec = self._check(Path(d), "RLAP")
            self.assertEqual(rec.participant, "RLAP:p003")

    def test_rlap_png_sequence_when_no_video(self):
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "RLAP/p004/E2"
            _write_png_sequence(s / "pictures_ZIP_RAW_RGB", n=240)
            t0 = 1.7e9
            _write_csv(s / "frames_timestamp.csv", None, t0 + np.arange(240) / FPS)
            t, v = _wave(240, 20.0, t0=t0)
            _write_csv(s / "BVP.csv", t, v, ["timestamp", "bvp"])
            self._check(Path(d), "RLAP")

    def test_sums(self):
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "SUMS/p001/v02"
            self._physrec(s, video_name="video_RGB_ZIP_H264.avi")
            _write_video(s / "video_finger_ZIP_H264.avi")      # fingertip camera, must be skipped
            rec = self._check(Path(d), "SUMS")
            self.assertEqual(rec.participant, "SUMS:p001")
            self.assertIn("rgb", rec.meta["video"].lower())

    def test_ladh(self):
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "LADH/12_05/p_12_05_caip/v01"
            self._physrec(s, video_name="video_RGB_H264.avi", ts_name="frames_timestamp_RGB.csv",
                          extras=("HR.csv", "RR.csv", "SpO2.csv"))
            _write_video(s / "video_IR_H264.avi")              # IR stream, must be skipped
            rec = self._check(Path(d), "LADH")
            self.assertEqual(rec.participant, "LADH:caip")     # same person across days
            self.assertNotIn("IR", rec.meta["video"])

    def test_m3pd(self):
        with tempfile.TemporaryDirectory() as d:
            subj = Path(d) / "M3PD/001"
            sess = subj / "dual_camera_session_20250115_103012"
            _write_video(sess / "front_camera_20250115_103012.mp4")
            _write_video(sess / "back_camera_20250115_103012.mp4")   # fingertip view, must be skipped
            t0 = 1.7e9
            _write_csv(subj / "v01/frames_timestamp.csv", None, t0 + np.arange(N) / FPS)
            t, v = _wave(300, 20.0, t0=t0)
            _write_csv(subj / "v01/BVP.csv", t, v, ["timestamp", "bvp"])
            rec = self._check(Path(d), "M3PD")
            self.assertIn("front", rec.meta["video"])

    # -- MMPD ----------------------------------------------------------------------------
    def test_mmpd(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "MMPD/subject3"
            s.mkdir(parents=True)
            vid = np.stack(list(_frames(n=240)))
            _, v = _wave(240, 30.0)
            savemat(s / "p3_0.mat", {"video": vid, "GT_ppg": v, "light": "LED-low",
                                     "motion": "Stationary", "skin_color": 3})
            rec = self._check(Path(d), "MMPD")
            self.assertEqual(rec.participant, "MMPD:subject3")
            self.assertEqual(rec.meta["labels"].get("light"), "LED-low")

    def test_mmpd_float_video_in_unit_range(self):
        """MMPD .mat files store frames as float in [0, 1] as well as uint8."""
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "MMPD/subject9"
            s.mkdir(parents=True)
            vid = np.stack(list(_frames(n=240))).astype(np.float32) / 255.0
            _, v = _wave(240, 30.0)
            savemat(s / "p9_1.mat", {"video": vid, "GT_ppg": v})
            self._check(Path(d), "MMPD")


class RegistryBehaviour(unittest.TestCase):
    def test_folder_name_disambiguates_the_physrecorder_family(self):
        """SUMS, LADH and RLAP share a layout, so the dataset folder name has to decide."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name, sess in (("SUMS", "p001/v01"), ("RLAP", "p002/R1")):
                s = root / name / sess
                _write_video(s / "video_ZIP_H264.avi", n=120)
                t0 = 1.7e9
                _write_csv(s / "frames_timestamp.csv", None, t0 + np.arange(120) / FPS)
                t, v = _wave(80, 20.0, t0=t0)
                _write_csv(s / "BVP.csv", t, v, ["timestamp", "bvp"])
            found = D.discover(root)
            by_name = {k: [p for kk, p in found if kk == k] for k in ("SUMS", "RLAP")}
            self.assertEqual(len(by_name["SUMS"]), 1, found)
            self.assertEqual(len(by_name["RLAP"]), 1, found)
            self.assertIn("SUMS", str(by_name["SUMS"][0]))
            self.assertIn("RLAP", str(by_name["RLAP"][0]))

    def test_only_filter_and_probe(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sub = root / "UBFC-rPPG/subject1"
            _write_video(sub / "vid.avi", n=120)
            tp = np.arange(80) / 20.0
            ppg = np.sin(2 * np.pi * BPM / 60 * tp)
            (sub / "ground_truth.txt").write_text(
                " ".join(map(str, ppg)) + "\n" + " ".join(["72"] * tp.size) + "\n" + " ".join(map(str, tp)))
            self.assertEqual([k for k, _ in D.discover(root)], ["UBFC-rPPG"])
            self.assertEqual(D.discover(root, only=["PURE"]), [])
            p = D.probe(root)
            self.assertEqual(p["datasets"]["UBFC-rPPG"]["recordings"], 1)
            self.assertIn("MMPD", p["missing"])
            self.assertIn("UBFC-rPPG", D.probe_text(root))

    def test_empty_and_missing_roots(self):
        self.assertEqual(D.discover(Path("/nonexistent-xyz")), [])
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(D.discover(Path(d)), [])
            self.assertIn("nothing recognised", D.probe_text(Path(d)))

    def test_every_name_is_registered_and_documented(self):
        for n in D.NAMES:
            self.assertIn(n, D.REGISTRY, n)
            self.assertTrue(D.REGISTRY[n].markers, n)
        self.assertNotIn("PILOT", D.PUBLIC_NAMES)
        with self.assertRaises(ValueError):
            D.load("NOT-A-DATASET", "/tmp")

    def test_config_accepts_every_dataset_name(self):
        from sama import config as C
        cfg = C.load(None, [f"training.datasets=[{','.join(D.NAMES)}]"])
        self.assertEqual(C.validate(cfg), [])
        with self.assertRaises(ValueError) as e:                   # load validates and refuses
            C.load(None, ["training.datasets=[NOPE]"])
        self.assertIn("unknown datasets", str(e.exception))
        with self.assertRaises(ValueError):
            C.load(None, ["training.datasets=[]"])


class TrainAndTestAcrossDatasets(unittest.TestCase):
    """Train on one dataset and test on another: the loaders have to work end to end, and a
    held-out dataset must be reported as held out (not silently as in-sample)."""

    def _build(self, root: Path, name: str, bpm: float, sub: str):
        t0 = 1.7e9
        s = root / name / sub
        _write_video(s / "video_ZIP_H264.avi", n=N, bpm=bpm)
        _write_csv(s / "frames_timestamp.csv", None, t0 + np.arange(N) / FPS)
        t, v = _wave(300, 20.0, bpm, t0=t0)
        _write_csv(s / "BVP.csv", t, v, ["timestamp", "bvp"])

    def test_cross_dataset(self):
        from sama import config as C
        from sama.realtrain import evaluate, public_items, run
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "raw"
            for i, bpm in enumerate((66.0, 84.0)):
                self._build(root, "RLAP", bpm, f"p00{i + 1}/R1")
                self._build(root, "SUMS", bpm, f"p10{i + 1}/v01")
            over = [f"paths.public_root={root}", "signal.window_s=8", "signal.step_s=4",
                    "training.window_step_s=2", "ita.source=none", "training.folds=2",
                    "training.epochs=3", "training.datasets=[RLAP]"]
            cfg = C.load(None, over)
            items = public_items(cfg, log=lambda *a: None)
            self.assertEqual({it.dataset for it in items}, {"RLAP"})
            self.assertEqual(len({it.participant for it in items}), 2)
            bundle = Path(d) / "bundle"
            meta = run(items, cfg, bundle, log=lambda *a: None)
            self.assertEqual(meta["trained_on"], ["RLAP"])
            self.assertIn("dataset:RLAP", meta["cv"])

            m = evaluate(cfg, bundle, ["SUMS"], out_dir=Path(d) / "eval", log=lambda *a: None)
            self.assertEqual(m["_eval"]["held_out"], ["SUMS"])
            self.assertEqual(m["_eval"]["also_in_training"], [])
            self.assertIn("dataset:SUMS", m)
            self.assertTrue((Path(d) / "eval/eval_metrics.json").exists())
            self.assertIn("true generalisation", (Path(d) / "eval/eval_report.md").read_text())

            m2 = evaluate(cfg, bundle, ["RLAP"], log=lambda *a: None)
            self.assertEqual(m2["_eval"]["also_in_training"], ["RLAP"])
            self.assertIn("Not a generalisation result", __import__("sama.realtrain", fromlist=["x"]).eval_report(m2))

    def test_max_recordings_caps_each_dataset_separately(self):
        from sama import config as C
        from sama.realtrain import public_items
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "raw"
            for i, bpm in enumerate((66.0, 84.0)):
                self._build(root, "RLAP", bpm, f"p00{i + 1}/R1")
                self._build(root, "SUMS", bpm, f"p10{i + 1}/v01")
            cfg = C.load(None, [f"paths.public_root={root}", "ita.source=none",
                                "training.datasets=[RLAP,SUMS]", "training.max_recordings=1"])
            got = {it.dataset for it in public_items(cfg, log=lambda *a: None)}
            self.assertEqual(got, {"RLAP", "SUMS"}, "a smoke run must still cover every dataset")


if __name__ == "__main__":
    unittest.main()

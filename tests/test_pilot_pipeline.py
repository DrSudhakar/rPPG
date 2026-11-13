"""Tests for the real-data / pilot layer: config knobs, camera ITA, Monk ratings, reference and sync,
live analysis, video->skin-tone model, statistics helpers, web interface and lab recorder.
All inputs are SYNTHETIC (generated here); nothing is real data."""
import json
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sama import config as C  # noqa: E402
from sama import ita_camera as IC  # noqa: E402
from sama import monk as MK  # noqa: E402
from sama.live import LiveAnalyzer  # noqa: E402
from sama.pilot import ba_clustered  # noqa: E402
from sama.power_check import h1_power, sd_ci  # noqa: E402
from sama.reference import Reference, load_reference  # noqa: E402
from sama.session import _blocks_from_events  # noqa: E402
from sama.sync import align, orient  # noqa: E402
from sim import samasim as S  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


class Config(unittest.TestCase):
    def test_defaults_valid(self):
        c = C.load()
        self.assertEqual(c.ita.source, "camera")
        self.assertEqual(c.get_path("gate.type"), "pooled")

    def test_override_and_reconcile(self):
        c = C.load(None, ["ita.source=none", "gate.type=mondrian", "training.dro_groups=stratum"])
        self.assertFalse(c.ita.conditioning)
        self.assertEqual(c.strata.by, "none")
        self.assertEqual(c.gate.type, "pooled")
        self.assertEqual(c.training.dro_groups, "dataset")
        self.assertTrue(any("mondrian" in n for n in c._notes))

    def test_rejects_unknown_key_and_value(self):
        with self.assertRaises(KeyError):
            C.load(None, ["ita.sourse=none"])
        with self.assertRaises(ValueError):
            C.load(None, ["gate.type=magic"])
        with self.assertRaises(ValueError):
            C.load(None, ["raters.n=3"])

    def test_oximeter_hr_disables_xcorr(self):
        c = C.load(None, ["reference.type=oximeter_hr"])
        self.assertEqual(c.sync.method, "wallclock")

    def test_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            p = C.snapshot(C.load(None, ["ita.source=monk"]), d)
            self.assertIn("reconciled", p.read_text())


def _frame_with(skin_lin, card_lin, refl=0.18, W=160, H=120):
    """Frame whose face box holds a uniform skin colour and whose card box holds the card."""
    to8 = lambda x: np.clip(np.round(255 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.clip(x, 0, 1) ** (1 / 2.4) - 0.055)), 0, 255)  # noqa: E731
    fr = np.zeros((H, W, 3))
    fr[:] = [60, 70, 80]
    fr[int(0.15 * H):int(0.75 * H), int(0.30 * W):int(0.70 * W)] = to8(skin_lin)
    fr[int(0.42 * H):int(0.58 * H), int(0.76 * W):int(0.88 * W)] = to8(card_lin)
    rng = np.random.default_rng(0)
    return np.clip(fr + rng.normal(0, 1.0, fr.shape), 0, 255).astype(np.uint8)


class CameraIta(unittest.TestCase):
    def test_card_recovers_ita_under_coloured_light(self):
        for ita in (40.0, 15.0, -10.0):
            R = S.skin_reflectance(ita, 17.0, 12.0)
            light = np.array([1.0, 0.85, 0.6]) * 0.9          # warm light the camera does not correct
            fr = _frame_with(R * light, 0.18 * light)
            est = IC.estimate([fr] * 3, np.array([0.30, 0.15, 0.40, 0.60]), white_box=(0.76, 0.42, 0.12, 0.16), method="card",
                              reflectance=0.18, shadow_pct=0, specular=False)
            self.assertLess(abs(est.ita - ita), 4.0, (ita, est))

    def test_no_reference_is_flagged_and_biased(self):
        R = S.skin_reflectance(15.0, 17.0, 12.0)
        fr = _frame_with(R * np.array([1.0, 0.85, 0.6]), 0.18 * np.ones(3))
        est = IC.estimate([fr], np.array([0.30, 0.15, 0.40, 0.60]), method="none", shadow_pct=0, specular=False)
        self.assertIn("NO_WHITE_REF", est.flags)
        self.assertGreater(abs(est.ita - 15.0), 4.0)

    def test_clipped_card_rejected(self):
        R = S.skin_reflectance(10.0, 17.0, 12.0)
        fr = _frame_with(R * 0.9, np.ones(3) * 3.0)
        est = IC.estimate([fr], np.array([0.30, 0.15, 0.40, 0.60]), white_box=(0.76, 0.42, 0.12, 0.16), method="card")
        self.assertIn("WHITE_CLIPPED", est.flags)

    def test_ccm_identity_on_nominal_chart(self):
        M = IC.fit_ccm(IC.NOMINAL_CHART_SRGB)
        self.assertTrue(np.allclose(M, np.eye(3), atol=1e-6))


class Monk(unittest.TestCase):
    def test_one_and_two_raters(self):
        df = pd.DataFrame({"participant_id": ["A", "A", "B", "B", "C", "C", "D", "D"], "rater_id": ["R1", "R2"] * 4,
                           "monk": [3, 4, 6, 6, 8, 9, 2, 2]})
        c1 = MK.consensus(df, 1)
        self.assertEqual(c1.set_index("participant_id").monk["A"], 3)
        self.assertIn("cannot be estimated", MK.agreement(c1, 1)["note"])
        c2 = MK.consensus(df, 2)
        self.assertEqual(c2.set_index("participant_id").monk["A"], 3.5)
        a = MK.agreement(c2, 2)
        self.assertEqual(a["exact"], 0.5)
        self.assertEqual(a["within_one"], 1.0)
        self.assertGreater(a["kappa_quadratic"], 0.8)

    def test_strata_and_validation(self):
        self.assertEqual(MK.stratum_from_monk(3), "lighter")
        self.assertEqual(MK.stratum_from_monk(6), "middle")
        self.assertEqual(MK.stratum_from_monk(9), "darker")
        r = MK.ita_vs_monk(np.array([40, 30, 20, 10, 0, -10.0]), np.array([2, 3, 5, 6, 8, 9.0]), n_boot=200)
        self.assertLess(r["spearman_rho"], -0.9)

    def test_rejects_bad_ratings(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("participant_id,rater_id,monk\nA,R1,11\n")
        with self.assertRaises(ValueError):
            MK.load_ratings(f.name)


def _pleth_csv(path, beats, dur, t_wall0=1.7e9, offset=0.0, fs=60.0, invert=False):
    tt = np.arange(0, dur, 1 / fs)
    w = S.pulse_wave(beats, tt, 0.30)
    w = (-w if invert else w) * 300 + 2000
    pd.DataFrame({"time_s": t_wall0 + tt + offset, "pleth": w, "hr_bpm": 70}).to_csv(path, index=False)


class ReferenceAndSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.part = S.make_participant("x", "middle", np.random.default_rng(3), ita=18.0)
        cls.rec = S.simulate_block(cls.part, "rest", "standard", "reference30", S.SimParams(), dur_s=120.0)

    def test_pleth_beats_and_window_hr(self):
        with tempfile.TemporaryDirectory() as d:
            _pleth_csv(Path(d) / "r.csv", self.rec.beats, 120.0)
            ref = load_reference(Path(d) / "r.csv", "oximeter_pleth", {"time": "time_s", "pleth": "pleth", "hr": "hr_bpm"})
            true = 60 / np.mean(np.diff(self.rec.beats[(self.rec.beats >= 10) & (self.rec.beats < 40)]))
            self.assertLess(abs(ref.window_hr(10, 40) - true), 1.0)
            self.assertTrue(np.isfinite(ref.window_hrv(0, 60, 0.2, 30)["rmssd"]))

    def test_sync_recovers_clock_offset_and_polarity(self):
        for invert in (False, True):
            with tempfile.TemporaryDirectory() as d:
                _pleth_csv(Path(d) / "r.csv", self.rec.beats, 120.0, offset=0.8, invert=invert)
                ref = load_reference(Path(d) / "r.csv", "oximeter_pleth", {"time": "time_s", "pleth": "pleth"})
                t = np.arange(self.rec.rgb_patches.shape[0]) / self.rec.fps
                r = align(t, self.rec.rgb_patches, 1.7e9, ref, "xcorr", 3.0)
                # reference-relative time = true time, so the lag is the finger-minus-face transit time
                self.assertEqual(r.method, "xcorr")
                self.assertLess(abs(r.lag_s - (0.30 - self.part.ptt)), 0.12, (invert, r))

    def test_orient(self):
        x = S.pulse_wave(np.arange(1, 30, 0.8), np.arange(0, 30, 1 / 30), 0.2)
        self.assertTrue(np.allclose(orient(x), x))
        self.assertTrue(np.allclose(orient(-x), x))

    def test_oximeter_hr_only(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({"time_s": 1.7e9 + np.arange(60.0), "hr_bpm": 72.0}).to_csv(Path(d) / "r.csv", index=False)
            ref = load_reference(Path(d) / "r.csv", "oximeter_hr", {"time": "time_s", "hr": "hr_bpm"})
            self.assertAlmostEqual(ref.window_hr(0, 30), 72.0)
            self.assertFalse(ref.has_beats)


class Blocks(unittest.TestCase):
    def test_blocks_from_events(self):
        ev = pd.DataFrame({"wall_time": [100, 130, 131, 190, 195], "event": ["block_start", "block_end", "block_start", "block_end", "block_start"],
                           "block": ["rest", "rest", "arithmetic", "arithmetic", "recovery"], "light": ["L1", "L1", "L1", "L1", "L2"]})
        b = _blocks_from_events(ev, 90.0, 200.0)
        self.assertEqual([x.name for x in b], ["rest", "arithmetic", "recovery"])
        self.assertEqual((b[0].start, b[0].end), (10.0, 40.0))
        self.assertEqual(b[2].end, 200.0)


class Live(unittest.TestCase):
    def test_hr_beats_and_motion_warning(self):
        cfg = C.load(None, ["live.method=POS"])
        part = S.make_participant("l", "lighter", np.random.default_rng(1), ita=35.0)
        rec = S.simulate_block(part, "rest", "standard", "webcam", S.SimParams(), dur_s=75.0)
        an = LiveAnalyzer(cfg)
        states = []
        for i in range(rec.rgb_patches.shape[0]):
            t = i / rec.fps
            moving = 50 < t < 60
            an.push(t, rec.rgb_patches[i], rec.meta["pix"], 8.0 if moving else 0.5, 0.9)
            if an.due(t):
                states.append((t, an.state(t)))
        good = [s for t, s in states if t > 25 and t < 50 and s["shown"]]
        self.assertTrue(good)
        beats = rec.beats[(rec.beats >= 30) & (rec.beats < 50)]
        self.assertLess(abs(np.median([s["hr"] for s in good]) - 60 / np.mean(np.diff(beats))), 4.0)
        self.assertTrue(good[-1]["beats"])
        mv = [s for t, s in states if 52 < t < 60]
        self.assertTrue(all(s["reason"] == "MOTION" and s["hr"] is None for s in mv))
        self.assertTrue(any(w["code"] == "MOTION" and w["level"] == "bad" for w in mv[-1]["warnings"]))
        last = states[-1][1]
        self.assertIn("valid", last["hrv"])


class ItaModel(unittest.TestCase):
    def test_cv_on_synthetic_features(self):
        from sama.ita_model import ItaModel as IM, cross_validate
        rng = np.random.default_rng(0)
        feats, y = [], []
        for _ in range(20):
            m = rng.uniform(1, 10)
            feats.append({"cheekwb_L": 80 - 6 * m + rng.normal(0, 2), "cheekwb_bb": 15 + rng.normal(0, 2), "white_g": rng.uniform(0.3, 0.9)})
            y.append(m)
        res = cross_validate(feats, np.array(y), "monk", folds=5, cuts=(4, 7))
        self.assertLess(res["mae"], 1.0)
        m = IM("monk").fit(feats, np.array(y))
        with tempfile.TemporaryDirectory() as d:
            m.save(Path(d) / "m.json")
            m2 = IM.load(Path(d) / "m.json")
            self.assertTrue(np.allclose(m.predict(feats[:3]), m2.predict(feats[:3])))


class Stats(unittest.TestCase):
    def test_ba_clustered_wider_than_naive_when_clustered(self):
        rng = np.random.default_rng(0)
        rows = []
        for p in range(10):
            u = rng.normal(0, 3)
            rows += [{"participant_id": p, "hr_cam_bpm": 70 + u + rng.normal(0, 1), "hr_ref_bpm": 70} for _ in range(8)]
        r = ba_clustered(pd.DataFrame(rows))
        self.assertGreaterEqual(r["sd"], r["sd_naive"] * 0.95)

    def test_power_and_sd_ci(self):
        lo, hi = sd_ci(2.0, 10)
        self.assertLess(lo, 2.0)
        self.assertGreater(hi, 2.0)
        self.assertGreater(h1_power(36, 1.0, sims=300), h1_power(36, 4.0, sims=300))

    def test_h1_power_rises_with_n_and_h2_power(self):
        from sama.power_check import h2_power
        self.assertGreater(h1_power(45, 0.55, sims=400, seed=1), h1_power(20, 0.55, sims=400, seed=1))
        self.assertGreater(h2_power(45, 1.0), h2_power(45, 3.0))

    def test_failures_scored_not_dropped(self):
        from sama.pilot import score_failures
        df = pd.DataFrame({"participant_id": ["a"] * 4, "method": ["POS"] * 4, "level": ["lossless"] * 4,
                           "hr_cam_bpm": [70, 72, np.nan, 71], "hr_ref_bpm": [70, 70, 90, 70]})
        out = score_failures(df)
        self.assertEqual(int(out.failed.sum()), 1)
        self.assertAlmostEqual(float(out.abs_err.iloc[2]), 19.0)   # median estimate 71 vs reference 90

    def test_participant_without_ita_placed_by_monk(self):
        from sama.pilot import _participant_tone, strata_fallback
        cfg = C.load(None, ["ita.source=camera", "strata.by=ita"])
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "ita_camera.json").write_text(json.dumps({"ita": float("nan"), "flags": ["WHITE_CLIPPED", "NO_SKIN_PIXELS"], "usable": False}))
            t = _participant_tone("P1", cfg, d, {}, {"P1": 9}, None)
        self.assertEqual(t["stratum"], "darker")
        self.assertTrue(t["stratum_source"].startswith("monk"))
        tones = {"P1": t, "P2": {"stratum": "lighter", "monk": 2, "stratum_source": "ita"},
                 "P3": {"stratum": "lighter", "monk": 3, "stratum_source": "ita"},
                 "P4": {"stratum": "darker", "monk": 8, "stratum_source": "ita"}}
        note = strata_fallback(tones, cfg, ["P1", "P2", "P3", "P4"], log=lambda *a: None)
        self.assertIn("1 participant(s) without a usable ITA", note)

    def test_build_flags_to_fill_fields(self):
        import re
        src = (ROOT / "proposal/build_proposal.py").read_text()
        self.assertIn("TO FILL", src)
        md = "Approval: SVNIT IEC [TO FILL: reference and date]."
        self.assertTrue(re.findall(r"\[(?:number|Status[^\]]*|List any[^\]]*|TO FILL[^\]]*)\]|_{4,}", md))


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class WebInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import uvicorn
        import app.server as srv
        cls.tmp = tempfile.TemporaryDirectory()
        cfg = C.load(None, [f"paths.pilot_root={cls.tmp.name}/pilot", f"training.bundle={cls.tmp.name}/nomodel", "live.method=POS"])
        srv.STATE.update(cfg=cfg, mode="serve", loaded=False, recorder=None)
        cls.port = _free_port()
        cls.server = uvicorn.Server(uvicorn.Config(srv.app, host="127.0.0.1", port=cls.port, log_level="error"))
        threading.Thread(target=cls.server.run, daemon=True).start()
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{cls.port}/healthz", timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.tmp.cleanup()

    def call(self, path, body=None, method=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method or ("POST" if data else "GET"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def test_info_and_page(self):
        j = self.call("/api/v1/info")
        self.assertTrue(j["research_only"])
        self.assertFalse(j["medical_device"])
        self.assertFalse(j["model"]["loaded"])
        html = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=5).read().decode()
        self.assertIn("Not a medical device", html)

    def test_stream_knobs_and_saving(self):
        sid = self.call("/api/v1/session", {"participant_id": "T01"})["session"]
        part = S.make_participant("w", "darker", np.random.default_rng(2), ita=-5.0)
        rec = S.simulate_block(part, "rest", "standard", "webcam", S.SimParams(), dur_s=40.0)
        frames = [{"t": i / rec.fps, "patches": rec.rgb_patches[i].round(3).tolist(), "motion": 0.4, "skin": 0.9}
                  for i in range(rec.rgb_patches.shape[0])]
        states = []
        for k in range(0, len(frames), 200):
            st = self.call(f"/api/v1/session/{sid}/frames", {"frames": frames[k:k + 200]})["state"]
            if st:
                states.append(st)
        self.assertTrue(any(s["shown"] for s in states))
        for s in states:
            self.assertTrue(s["shown"] or s.get("hr") is None)
        self.assertEqual(self.call(f"/api/v1/session/{sid}/knobs", {"method": "CHROM"})["knobs"]["method"], "CHROM")
        self.call(f"/api/v1/session/{sid}", method="DELETE")
        saved = list(Path(self.tmp.name, "pilot/browser_sessions/T01").glob("live_readings_*.csv"))
        self.assertTrue(saved and len(pd.read_csv(saved[0])) > 0)

    def test_simulate_demo(self):
        j = self.call("/api/v1/simulate", {"ita": -10, "light": "dim", "seconds": 35})
        self.assertEqual(j["provenance"], "SYNTHETIC (SAMA-Sim)")
        self.assertTrue(j["readings"])

    def test_front_end_sends_no_pixels(self):
        js = (ROOT / "app/static/app.js").read_text()
        for bad in ("toDataURL", "toBlob", "MediaRecorder"):
            self.assertNotIn(bad, js)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class Recorder(unittest.TestCase):
    def test_record_from_file(self):
        from sama.recorder import LabRecorder
        from sama.video import probe, write_video
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(0)
            frames = (np.clip(rng.normal(120, 3, (90, 120, 160, 3)), 0, 255)).astype(np.uint8)
            write_video(Path(d) / "in.mkv", iter(frames), 30.0, "ffv1")
            cfg = C.load(None, [f"paths.pilot_root={d}/pilot", "recorder.input_format=file", f"recorder.device={d}/in.mkv",
                                "recorder.analysis_size=160x120", "live.method=POS"])
            r = LabRecorder(cfg, log=lambda *a: None)
            r.start("T02")
            r.set_boxes(white_box=[0.8, 0.4, 0.1, 0.1])
            r.event("block_start", "rest", "light1_500lux")
            for _ in range(80):
                if r.proc is None or r.proc.poll() is not None:
                    break
                time.sleep(0.1)
            time.sleep(0.5)
            r.stop()
            sd = Path(d) / "pilot/sessions/T02"
            meta = json.loads((sd / "session.json").read_text())
            self.assertIsNotNone(meta["video_t0_wall"])
            self.assertEqual(meta["boxes"]["white_box"], [0.8, 0.4, 0.1, 0.1])
            self.assertGreaterEqual(probe(sd / "webcam.mkv")["width"], 160)
            ev = pd.read_csv(sd / "events.csv")
            self.assertIn("block_start", set(ev.event))
            self.assertGreater(r.n_frames, 60)
            r.stop_preview()  # stop() reopens the preview for the next participant; clean it up for the test

    def test_preview_before_recording(self):
        """A frame (and so something to draw boxes on) must be available before 'Start recording' is
        pressed, and start() must be able to take over the camera from a running preview."""
        from sama.recorder import LabRecorder
        from sama.video import write_video
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(1)
            frames = (np.clip(rng.normal(120, 3, (90, 120, 160, 3)), 0, 255)).astype(np.uint8)
            write_video(Path(d) / "in.mkv", iter(frames), 30.0, "ffv1")
            cfg = C.load(None, [f"paths.pilot_root={d}/pilot", "recorder.input_format=file", f"recorder.device={d}/in.mkv",
                                "recorder.analysis_size=160x120", "live.method=POS"])
            r = LabRecorder(cfg, log=lambda *a: None)
            self.assertIsNone(r.latest_frame)
            r.start_preview()
            for _ in range(50):
                if r.latest_frame is not None:
                    break
                time.sleep(0.1)
            self.assertIsNotNone(r.latest_frame, "no frame arrived from start_preview() before recording began")
            self.assertIsNotNone(r.preview_jpeg())
            self.assertIsNone(r.proc)  # preview must not count as "recording"
            r.start("T03")  # must take over the camera cleanly (stop_preview() happens inside start())
            self.assertIsNone(r.preview_proc)
            for _ in range(80):
                if r.proc is None or r.proc.poll() is not None:
                    break
                time.sleep(0.1)
            r.stop()
            r.stop_preview()  # stop() reopens the preview for the next participant; clean it up for the test


if __name__ == "__main__":
    unittest.main()

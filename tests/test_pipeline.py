"""End-to-end tests: SAMA steps 2-6 on SAMA-Sim recordings, model bundle I/O, and the web API
(served by uvicorn in a thread; queried with urllib because httpx is unavailable)."""
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sama import calib, conformal, model_io, net, pipeline, sama_method as SM, streaming  # noqa: E402
from sim import samasim as S  # noqa: E402


def _features(ita, light="standard", device="reference30", dur=75.0, seed=1, step2=True, step3=True):
    part = S.make_participant(f"p{seed}", S.stratum_of(ita), np.random.default_rng(seed), ita=ita)
    p = S.SimParams()
    rec = S.simulate_block(part, "rest", light, device, p, dur_s=dur)
    dn, static = S.chart_for(part, device, light, p)
    tone, noise = calib.fit_tone_curve(dn, S.CHART_REFLECTANCE), calib.photon_transfer(static, 400.0)
    ref = pipeline.ref_hr_windows(rec.beats, rec.beats_valid, rec.rgb_face.shape[0], rec.fps)
    wf = SM.window_features(rec.rgb_patches, rec.fps, rec.meta["pix"], ita, tone, noise, ref, ppg=rec.meta["ppg_ecg"],
                            light=light, meta={"pid": part.pid, "stratum": part.stratum}, use_step2=step2, use_step3=step3)
    return wf, rec


class SamaSteps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wfs = [_features(ita, seed=i)[0] for i, ita in enumerate((45, 40, 20, 15, 0, -15))]

    def test_window_features_shapes_and_accuracy(self):
        wf = self.wfs[0]
        self.assertEqual(wf.X.shape[1:], (4, net.T_NET))
        ok = np.isfinite(wf.hr_ref)
        self.assertLess(np.nanmedian(np.abs(wf.hr_cand[ok] - wf.hr_ref[ok])), 3.0)
        wf2, _ = _features(45, step2=False, step3=False, seed=0)
        self.assertEqual(wf2.X.shape, wf.X.shape)

    def test_fit_predict_and_gate(self):
        cat = lambda ws: SM.WindowFeatures(*(np.concatenate([getattr(w, k) for w in ws]) for k in ("X", "cond", "ppg", "hr_ref", "hr_pos", "hr_cand", "q")))  # noqa: E731
        tr, ca = cat(self.wfs[:4]), cat(self.wfs[2:])
        g = np.zeros(len(tr.hr_ref), int)
        strata = np.array(["middle"] * len(ca.hr_ref))
        pids = np.repeat(np.arange(len(self.wfs[2:])), [len(w.hr_ref) for w in self.wfs[2:]])
        for use_net in (True, False):
            m = SM.SamaModel(use_net=use_net).fit(tr, g, ca, strata, pids, cfg=net.TrainConfig(epochs=1))
            out = m.predict(ca, strata)
            self.assertEqual(out["hr"].shape, ca.hr_ref.shape)
            self.assertTrue(np.all(out["sigma"] > 0))
        rep = conformal.coverage_report(ca.hr_ref, out["lo"], out["hi"], out["shown"], strata, pids)
        self.assertIn("middle", rep)
        with tempfile.TemporaryDirectory() as d:
            model_io.save_bundle(d, m.params or net.init_params(), m.sigma_model, m.gate, {"version": "test"})
            b = model_io.load_bundle(d)
            self.assertEqual(b["gate"].q, m.gate.q)
            q2 = np.column_stack([ca.q[:2], np.zeros((2, 3))])
            self.assertTrue(np.allclose(b["sigma"].predict(q2), m.sigma_model.predict(q2)))
            self.assertIsNone(model_io.load_bundle(Path(d) / "missing"))

    def test_hrv_from_pulse(self):
        _, rec = _features(30, dur=70.0, seed=9)
        lin = rec.rgb_patches.astype(float)
        wrel = (lin / lin.mean(0, keepdims=True)).mean(1)
        from sama.methods import pos
        r = SM.rmssd_from_pulse(pos(wrel, rec.fps)[: int(60 * rec.fps)], rec.fps)
        self.assertTrue(np.isfinite(r))
        self.assertTrue(30 < SM.hr_autocorr(pos(wrel, rec.fps), rec.fps) < 180)

    def test_patch_means_layout(self):
        box = np.full((100, 80, 3), 120.0)
        box[:5] = 255.0
        pm = model_io.patch_means(box)
        self.assertEqual(pm.shape, (30, 3))
        self.assertTrue(np.allclose(pm, 120.0))

    def test_noise_model_patch_var(self):
        nm = calib.NoiseModel(0.05, 4.0)
        self.assertAlmostEqual(float(nm.patch_var(100.0, 100)), (0.05 * 100 + 4) / 100)

    def test_streaming_with_network(self):
        sm = SM.SigmaModel()
        sm.mu, sm.sd, sm.coef = np.zeros(14), np.ones(14), np.zeros(15)
        g = conformal.MondrianGate()
        g.q = {"lighter": 1.5, "middle": 1.5, "darker": 1.5}
        P = net.init_params(seed=0)
        est = streaming.StreamingEstimator(gate=g, sigma_model=sm, params=P, ita=float("nan"))
        _, rec = _features(20, dur=40.0, seed=3)
        out = [r for i in range(rec.rgb_patches.shape[0]) if (r := est.push(i / rec.fps, rec.rgb_patches[i], rec.meta["pix"]))]
        self.assertTrue(any(r.reason != "WARMING_UP" for r in out))
        est.reset()
        self.assertEqual(len(est.buf), 0)
        self.assertIn("pulse", streaming.estimate_offline(rec.rgb_patches, rec.fps))


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# The web interface is tested in tests/test_pilot_pipeline.py (WebInterface, Recorder).


if __name__ == "__main__":
    unittest.main()

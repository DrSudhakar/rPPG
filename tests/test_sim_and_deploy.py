"""Unit tests for SAMA-Sim (C5 module checks), the network, ONNX export, streaming (fault
injection, privacy) and the web application."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sama import conformal, export, net, sama_method, streaming  # noqa: E402
from sim import samasim as S  # noqa: E402


def _part(ita, seed=1):
    return S.make_participant("t", S.stratum_of(ita), np.random.default_rng(seed), ita=ita)


class Simulator(unittest.TestCase):
    def test_relative_pulse_amplitude_independent_of_melanin(self):
        p = S.with_params(motion=(("rest", 0.0),), roi_noise_rel=0.0, chan_noise_rel=0.0, specular=0.0, read_noise_e=0.0, adc_noise_dn=0.0)
        amps = []
        for ita in (45, 0, -25):
            rec = S.simulate_block(_part(ita), "rest", "standard", "reference30", p, dur_s=40)
            g = rec.rgb_face[:, 1]
            amps.append(np.std(g - np.convolve(g, np.ones(61) / 61, "same")) / g.mean())
        self.assertLess(max(amps) / min(amps), 1.25)  # within 25 % (pixel noise residual)

    def test_counter_world_lowers_relative_amplitude(self):
        p0 = S.with_params(motion=(("rest", 0.0),), roi_noise_rel=0.0, chan_noise_rel=0.0, read_noise_e=0.0)
        p1 = S.with_params(motion=(("rest", 0.0),), roi_noise_rel=0.0, chan_noise_rel=0.0, read_noise_e=0.0, world="melanin", melanin_rel_amp_slope=0.5)
        r0 = S.simulate_block(_part(-25), "rest", "standard", "reference30", p0, dur_s=40)
        r1 = S.simulate_block(_part(-25), "rest", "standard", "reference30", p1, dur_s=40)
        self.assertLess(np.std(np.diff(r1.rgb_face[:, 1])), np.std(np.diff(r0.rgb_face[:, 1])) * 1.01)

    def test_exposure_cap_and_clipping(self):
        p = S.SimParams()
        for ita in (50, -40):
            for lux in (100, 500):
                for dev in ("phoneA", "phoneB", "webcam"):
                    e, g = S.choose_exposure(p, S.DEVICES[dev], lux, float(S.skin_reflectance(ita)[1]))
                    self.assertLessEqual(e, p.max_exposure_s + 1e-12)
                    self.assertGreaterEqual(g, 1.0)
        rec = S.simulate_block(_part(50), "rest", "standard", "phoneB", p, dur_s=31)
        self.assertLess(rec.meta["clip_frac"], 0.01)

    def test_hrv_generator_hits_target(self):
        rng = np.random.default_rng(5)
        for target in (20, 45, 90):
            t = S.ibi_series(300, 70, np.log(target), 0.25, rng, ectopic_rate=0.0)
            r = 1000 * np.sqrt(np.mean(np.diff(np.diff(t)) ** 2))
            self.assertAlmostEqual(r, target, delta=0.25 * target)

    def test_codec_surrogate_worlds(self):
        T = 900
        t = np.arange(T) / 30
        ref = np.sin(2 * np.pi * 1.2 * t)

        def kept(world, amp):
            x = np.repeat((100 + amp * ref)[:, None, None], 3, 2).repeat(2, 1)
            y = S.codec_surrogate(x, "500", S.SimParams(world=world), 30, np.random.default_rng(6))[:, 0, 1]
            return np.dot(y - y.mean(), ref) / np.dot(ref, ref) / amp

        self.assertLess(kept("A", 0.2), 0.3)        # World A: small pulses removed first
        self.assertGreater(kept("A", 6.0), 0.7)     # large ones kept
        self.assertGreater(kept("B", 0.2), 0.8)     # World B: no amplitude-dependent loss

    def test_deterministic(self):
        r1 = S.simulate_block(_part(10), "rest", "dim", "phoneA", S.SimParams(), dur_s=31)
        r2 = S.simulate_block(_part(10), "rest", "dim", "phoneA", S.SimParams(), dur_s=31)
        self.assertTrue(np.array_equal(r1.rgb_patches, r2.rgb_patches))

    def test_devices_share_beats_not_noise(self):
        a = S.simulate_block(_part(10), "rest", "standard", "phoneA", S.SimParams(), dur_s=31)
        b = S.simulate_block(_part(10), "rest", "standard", "phoneB", S.SimParams(), dur_s=31)
        self.assertTrue(np.array_equal(a.beats, b.beats))
        ra, rb = a.rgb_face[:, 1] / a.rgb_face[:, 1].mean(), b.rgb_face[:, 1] / b.rgb_face[:, 1].mean()
        self.assertFalse(np.allclose(ra, rb))


class Network(unittest.TestCase):
    def test_param_budget(self):
        self.assertLess(net.n_params(net.init_params()), 500_000)

    def test_gradients(self):
        rng = np.random.default_rng(0)
        P = net.init_params(C=4, seed=1)
        for k in ("Wg1", "Wb1", "Wg2", "Wb2"):
            P[k] = rng.normal(0, 0.3, P[k].shape)
        X, c = rng.normal(size=(2, 4, net.T_NET)), rng.normal(size=(2, 2))
        ppg, hr = rng.normal(size=(2, net.T_NET)), np.array([70.0, 120.0])
        y, cache = net.forward(P, X, c, cache=True)
        _, dy = net.loss_and_grad_y(y, ppg, hr)
        G = net.backward(P, dy, cache)
        for name in ("W1", "W3", "Wf", "Wg2"):
            idx = tuple(0 for _ in P[name].shape)
            old = P[name][idx]
            P[name][idx] = old + 1e-5
            lp = net.loss_and_grad_y(net.forward(P, X, c), ppg, hr)[0].sum()
            P[name][idx] = old - 1e-5
            lm = net.loss_and_grad_y(net.forward(P, X, c), ppg, hr)[0].sum()
            P[name][idx] = old
            self.assertAlmostEqual((lp - lm) / 2e-5, G[name][idx], delta=1e-4 * max(1, abs(G[name][idx])))

    def test_training_reduces_loss(self):
        rng = np.random.default_rng(1)
        n = 64
        t = np.arange(net.T_NET) / net.FS_NET
        hr = rng.uniform(55, 120, n)
        ppg = np.sin(2 * np.pi * hr[:, None] / 60 * t)
        X = np.stack([ppg + rng.normal(0, 1.5, ppg.shape)] + [rng.normal(size=ppg.shape) for _ in range(3)], 1)
        _, hist = net.train(X, rng.normal(size=(n, 2)), ppg, hr, rng.integers(0, 6, n), net.TrainConfig(epochs=4, batch=16))
        self.assertLess(hist[-1]["loss"], hist[0]["loss"])


class Export(unittest.TestCase):
    def test_onnx_parity(self):
        rng = np.random.default_rng(2)
        P = net.init_params(seed=3)
        for k in ("Wg1", "Wb1", "Wg2", "Wb2"):
            P[k] = rng.normal(0, 0.3, P[k].shape)
        X, c = rng.normal(size=(3, 4, 450)), rng.normal(size=(3, 2))
        ref = net.forward(P, X, c)
        with tempfile.TemporaryDirectory() as d:
            paths = export.save_all(P, d)
            self.assertLess(np.abs(export.run_onnx(paths["fp32"], X, c) - ref).max(), 1e-4)
            self.assertLess(np.abs(export.run_onnx(paths["fp16"], X, c) - ref).max(), 1e-2)
            self.assertLess(np.abs(export.run_onnx(paths["int8"], X, c) - ref).max(), 0.1)


def _toy_estimator():
    from sama.sama_method import SigmaModel
    sm = SigmaModel()
    sm.mu, sm.sd = np.zeros(14), np.ones(14)
    sm.coef = np.zeros(15)               # sigma_hat = 1
    g = conformal.MondrianGate()
    g.q = {"lighter": 2.0, "middle": 2.0, "darker": 2.0}
    return streaming.StreamingEstimator(gate=g, sigma_model=sm, params=None, ita=20.0)


def _frames(seconds, fps=30.0, bpm=72, light=120.0, rng=None, motion=0.0):
    rng = rng or np.random.default_rng(0)
    t = np.arange(int(seconds * fps)) / fps
    pulse = np.sin(2 * np.pi * bpm / 60 * t)
    base = np.array([1.2, 1.0, 0.8]) * light
    out = base[None, None, :] * (1 + 0.004 * pulse[:, None, None] * np.array([0.33, 0.77, 0.53]) + motion * rng.standard_normal((len(t), 1, 1))) \
        * (1 + 0.0005 * rng.standard_normal((len(t), 30, 3)))
    return t, out


class Streaming(unittest.TestCase):
    def run_stream(self, t, fr, drop=None):
        est = _toy_estimator()
        out = []
        for i, (ti, f) in enumerate(zip(t, fr)):
            r = est.push(float(ti), None if (drop is not None and drop[i]) else f)
            if r is not None:
                out.append(r)
        return out

    def test_clean_signal_gives_correct_reading(self):
        t, fr = _frames(40)
        rs = [r for r in self.run_stream(t, fr) if r.shown]
        self.assertTrue(rs)
        self.assertTrue(all(abs(r.hr - 72) < 2 for r in rs))
        self.assertTrue(all(r.reason == "WARMING_UP" for r in self.run_stream(*_frames(25))))

    def test_faults_never_give_confident_wrong_number(self):
        rng = np.random.default_rng(1)
        t, fr = _frames(45)
        cases = {"face lost": (t, fr, rng.random(len(t)) < 0.3), "low light": (*_frames(45, light=10.0), None),
                 "heavy motion": (*_frames(45, motion=0.05), None), "frame drop": (t[::3], fr[::3], None)}
        for name, (tt, ff, drop) in cases.items():
            for r in self.run_stream(tt, ff, drop):
                self.assertTrue((not r.shown) or abs(r.hr - 72) <= 5, f"{name}: {r}")

    def test_abstention_hides_number(self):
        r = streaming.Reading(1.0, False, "INTERVAL_TOO_WIDE", 99.0, 80.0, 118.0)
        d = r.to_dict()
        self.assertIsNone(d["hr"])
        self.assertEqual(d["display"], streaming.NO_READING)

    def test_no_files_written(self):
        before = {p for p in Path(tempfile.gettempdir()).iterdir()}
        cwd_before = set(os.listdir(ROOT))
        self.run_stream(*_frames(35))
        self.assertEqual(cwd_before, set(os.listdir(ROOT)))
        self.assertEqual(before, {p for p in Path(tempfile.gettempdir()).iterdir()})


class Privacy(unittest.TestCase):
    def test_android_manifest_has_no_network_permission(self):
        m = (ROOT / "deploy/android/app/src/main/AndroidManifest.xml").read_text()
        self.assertNotIn("android.permission.INTERNET", m)
        self.assertIn("android.permission.CAMERA", m)

    def test_web_front_end_sends_no_pixels(self):
        js = (ROOT / "app/static/app.js").read_text()
        self.assertNotIn("toDataURL", js)
        self.assertNotIn("toBlob", js)
        self.assertNotIn("MediaRecorder", js)


if __name__ == "__main__":
    unittest.main()

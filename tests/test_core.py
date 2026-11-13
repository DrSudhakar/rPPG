"""Unit tests for sama/ core modules (stdlib unittest; run: python -m unittest discover tests)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sama import calib, conformal, ita, methods, signals, stats  # noqa: E402


class Signals(unittest.TestCase):
    def test_hr_from_signal_recovers_frequency(self):
        fs = 30.0
        t = np.arange(900) / fs
        for bpm in (48, 72, 110, 165):
            h, snr, _ = signals.hr_from_signal(np.sin(2 * np.pi * bpm / 60 * t), fs)
            self.assertAlmostEqual(h, bpm, delta=0.3)
            self.assertGreater(snr, 10)

    def test_windows(self):
        w = signals.window_starts(900 * 3, 30, 30, 15)
        self.assertEqual(w[0], (0, 900))
        self.assertEqual(w[1], (450, 1350))
        self.assertEqual(len(w), 5)

    def test_parabolic_peaks_subframe(self):
        fs = 30.0
        true = np.cumsum(np.full(40, 0.8137)) + 0.01
        t = np.arange(int(35 * fs)) / fs
        x = sum(np.exp(-((t - b) / 0.08) ** 2) for b in true)
        p = signals.peaks_parabolic(x, fs, 74)
        n = min(len(p), len(true))
        self.assertLess(np.median(np.abs(p[:n] - true[:n])), 0.005)  # < 5 ms at 33 ms frame interval

    def test_rmssd_sdnn(self):
        ibi = np.array([0.8, 0.82, 0.78, 0.81, 0.79])
        self.assertAlmostEqual(signals.rmssd_ms(ibi), 1000 * np.sqrt(np.mean(np.diff(ibi) ** 2)))
        self.assertAlmostEqual(signals.sdnn_ms(ibi), 1000 * np.std(ibi, ddof=1))
        self.assertTrue(np.isnan(signals.rmssd_ms(np.array([0.8]))))

    def test_clean_ibi_removes_ectopic(self):
        ibi = np.array([0.8] * 10 + [0.45, 1.15] + [0.8] * 10)
        self.assertLess(len(signals.clean_ibi(ibi)), len(ibi))

    def test_detrend(self):
        x = np.linspace(0, 10, 500) + np.sin(np.arange(500) / 5)
        self.assertLess(abs(np.polyfit(np.arange(500), signals.detrend_tarvainen(x, 100), 1)[0]), 1e-3)


class Methods(unittest.TestCase):
    def test_all_methods_find_pulse(self):
        fs = 30.0
        t = np.arange(900) / fs
        rng = np.random.default_rng(0)
        rgb = np.array([170, 130, 100.0]) * (1 + 0.004 * np.outer(np.sin(2 * np.pi * 1.3 * t), [0.33, 0.77, 0.53])
                                              + 0.0004 * rng.standard_normal((900, 3)))
        for m in methods.METHODS:
            h, _, _ = signals.hr_from_signal(methods.run(m, rgb, fs), fs)
            self.assertAlmostEqual(h, 78, delta=1.0, msg=m)

    def test_run_handles_nonfinite(self):
        rgb = np.ones((300, 3)) * 100
        rgb[5] = np.nan
        self.assertEqual(methods.run("POS", rgb, 30).shape, (300,))


class Stats(unittest.TestCase):
    def test_icc_shrout_fleiss_example(self):
        # Shrout & Fleiss (1979) Table 2 data: ICC(2,1) = 0.29
        x = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
        r = stats.icc21(x)
        self.assertAlmostEqual(r["icc"], 0.29, delta=0.01)
        self.assertLess(r["lo95"], r["icc"])
        self.assertGreater(r["hi95"], r["icc"])

    def test_slope_and_paired(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(-25, 50, 200)
        y = 4 - 0.03 * x + rng.normal(0, 0.5, 200)
        self.assertLess(stats.slope_test(x, y, "less")["p"], 1e-6)
        self.assertGreater(stats.paired_test(rng.normal(1, 1, 50))["lo95"], 0)

    def test_holm(self):
        h = stats.holm({"a": 0.01, "b": 0.02, "c": 0.04})
        self.assertAlmostEqual(h["a"], 0.03)
        self.assertAlmostEqual(h["b"], 0.04)
        self.assertAlmostEqual(h["c"], 0.04)

    def test_lmm_recovers_planted_effects(self):
        rng = np.random.default_rng(2)
        n_p, n_w = 60, 30
        pid = np.repeat(np.arange(n_p), n_w)
        blk = np.tile(np.arange(n_w) % 6, n_p)
        x = rng.normal(size=pid.size)
        y = 1.0 + 0.5 * x + rng.normal(0, 0.7, n_p)[pid] + rng.normal(0, 0.3, 6)[blk] + rng.normal(0, 1, pid.size)
        f = stats.lmm_crossed(y, np.column_stack([np.ones_like(x), x]), [pid, blk])
        self.assertAlmostEqual(f.beta[1], 0.5, delta=3 * f.se[1])
        self.assertAlmostEqual(f.var_components["group0"], 0.49, delta=0.25)

    def test_metrics_and_fairness(self):
        m = stats.hr_metrics(np.array([70, 80, 100.0]), np.array([72, 80, 120.0]))
        self.assertAlmostEqual(m["mae"], 22 / 3)
        self.assertAlmostEqual(m["within_tol"], 2 / 3)
        self.assertAlmostEqual(stats.fairness({"a": 2.0, "b": 4.0})["ratio"], 2.0)

    def test_bootstrap(self):
        est, lo, hi = stats.bootstrap_ci(list(range(30)), lambda g: float(np.mean(g)), n_boot=300)
        self.assertLess(lo, est)
        self.assertGreater(hi, est)


class Conformal(unittest.TestCase):
    def test_marginal_coverage_exchangeable(self):
        rng = np.random.default_rng(3)
        covs = []
        for _ in range(200):
            g = conformal.MondrianGate(draws=20)
            n = 40
            h_ref = rng.normal(75, 10, n)
            h_hat = h_ref + rng.normal(0, 2, n)
            g.fit(h_hat, h_ref, np.ones(n), np.array(["a"] * n), np.arange(n))
            t_ref = rng.normal(75, 10, 200)
            t_hat = t_ref + rng.normal(0, 2, 200)
            lo, hi, _ = g.interval(t_hat, np.ones(200), np.array(["a"] * 200))
            covs.append(np.mean((t_ref >= lo) & (t_ref <= hi)))
        self.assertGreater(np.mean(covs), 0.89)

    def test_too_few_calibration_gives_infinite_and_abstains(self):
        g = conformal.MondrianGate(draws=5)
        g.fit(np.zeros(7), np.ones(7), np.ones(7), np.array(["a"] * 7), np.arange(7))
        self.assertTrue(np.isinf(g.q["a"]))
        _, _, _, shown = g.decide(np.array([70.0]), np.array([1.0]), np.array(["a"]))
        self.assertFalse(shown[0])

    def test_quantile_index(self):
        self.assertEqual(conformal.conformal_quantile(np.arange(1, 10), 0.1), 9)
        self.assertTrue(np.isinf(conformal.conformal_quantile(np.arange(1, 9), 0.1)))

    def test_serialisation(self):
        g = conformal.MondrianGate()
        g.q = {"darker": 2.0}
        self.assertEqual(conformal.MondrianGate.from_dict(g.to_dict()).q, {"darker": 2.0})


class CalibAndIta(unittest.TestCase):
    def test_tone_curve_recovers_gamma(self):
        r = np.linspace(0.03, 0.9, 24)
        dn = 255 * r ** (1 / 2.2)
        tc = calib.fit_tone_curve(dn, r)
        self.assertAlmostEqual(tc.gamma, 2.2, delta=0.01)
        self.assertTrue(np.allclose(tc.linearise(dn) / 255, r, atol=1e-3))

    def test_photon_transfer_recovers_planted_curve(self):
        rng = np.random.default_rng(4)
        means = np.linspace(10, 200, 24)
        a, b, npix = 0.05, 4.0, 400
        tr = means + rng.normal(0, np.sqrt((a * means + b) / npix), (2000, 24))
        nm = calib.photon_transfer(tr, npix)
        self.assertAlmostEqual(nm.a, a, delta=0.01)
        self.assertAlmostEqual(nm.b, b, delta=1.5)

    def test_ita(self):
        self.assertAlmostEqual(ita.ita_from_lab(50 + 17 * np.tan(np.radians(30)), 17), 30, places=6)
        self.assertEqual(ita.stratum(28.0), "middle")
        self.assertEqual(ita.stratum(28.1), "lighter")
        self.assertEqual(ita.stratum(10.0), "darker")

    def test_ita_from_camera_white_balance(self):
        from sim.samasim import skin_reflectance
        for t in (-20, 15, 45):
            rgb = skin_reflectance(t)
            self.assertAlmostEqual(ita.ita_from_camera(rgb * np.array([1.2, 1.0, 0.7]), np.array([1.2, 1.0, 0.7]) * 0.9), t, delta=1.0)


if __name__ == "__main__":
    unittest.main()

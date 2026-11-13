"""Extra branch coverage: simulator worlds and devices, streaming abstention codes, ITA helpers."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sama import ita  # noqa: E402
from sim import samasim as S  # noqa: E402
from tests.test_sim_and_deploy import _frames, _toy_estimator  # noqa: E402


class SimBranches(unittest.TestCase):
    def test_worlds_devices_cohort(self):
        part = S.make_participant("x", "darker", np.random.default_rng(2), ita=-5.0)
        for w in ("null", "melanin", "B"):
            r = S.simulate_block(part, "post_step", "dim", "webcam", S.with_params(world=w), codec="250", dur_s=31)
            self.assertEqual(r.rgb_patches.shape[1:], (30, 3))
        r = S.simulate_block(part, "mental_arithmetic", "standard", "reference", S.SimParams(), dur_s=31)
        self.assertEqual(r.fps, 60.0)
        dn, static = S.chart_for(part, "phoneB", "dim", S.SimParams())
        self.assertEqual(dn.shape, (24,))
        c = S.make_cohort(2, np.random.default_rng(0))
        self.assertEqual(len(c), 6)
        self.assertEqual(sum(p.sex == "F" for p in c), 3)
        self.assertGreater(S.melanin_index(-30), S.melanin_index(40))
        m, v = S.expected_quantised(np.array([10.3, 200.0]), np.array([0.2, 3.0]))
        self.assertAlmostEqual(m[1], 200.0)
        self.assertTrue(0 < v[0] < 1)


class StreamReasons(unittest.TestCase):
    def reasons(self, t, fr, drop=None):
        est = _toy_estimator()
        out = set()
        for i, (ti, f) in enumerate(zip(t, fr)):
            r = est.push(float(ti), None if (drop is not None and drop[i]) else f)
            if r is not None:
                out.add(r.reason)
                r.display()
        return out

    def test_codes(self):
        t, fr = _frames(40)
        self.assertIn("FACE_LOST", self.reasons(t, fr, np.random.default_rng(0).random(len(t)) < 0.5))
        self.assertIn("FRAME_RATE", self.reasons(t[::4], fr[::4]))
        self.assertIn("LOW_LIGHT", self.reasons(*_frames(40, light=10.0)))
        self.assertIn("MOTION", self.reasons(*_frames(40, motion=0.2)))
        self.assertIn("OK", self.reasons(t, fr))


class ItaHelpers(unittest.TestCase):
    def test_lab_roundtrip(self):
        L, a, b = ita.lab_from_linear_rgb(S.lab_to_linear_rgb(60.0, 12.0, 17.0))
        self.assertAlmostEqual(L, 60.0, delta=0.5)
        self.assertEqual(ita.stratum(float("nan")), "unknown")
        self.assertTrue(np.isnan(ita.ita_from_lab(50, 0)))


if __name__ == "__main__":
    unittest.main()

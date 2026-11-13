"""Implementation check (C2 replication substitute): our GREEN, POS, CHROM, LGI and PBV must
reproduce the rPPG-Toolbox code (pinned commit, cloned by `make third_party`, not vendored:
Responsible AI License) on identical synthetic RGB traces. Real-data replication of the
toolbox's published MAE needs datasets that could not be downloaded (ISSUES I-22).
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TB = Path(os.environ.get("RPPG_TOOLBOX", ROOT / "third_party/rPPG-Toolbox"))

from sama import methods as M  # noqa: E402


def _frames_from_rgb(rgb):
    # toolbox methods take frames (T, H, W, 3) and spatially average; a 1x1 frame is exact
    return rgb[:, None, None, :].astype(np.float64)


@unittest.skipUnless((TB / "unsupervised_methods").exists(), "rPPG-Toolbox not cloned (make third_party)")
class ToolboxEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(np, "mat"):  # toolbox uses np.mat, removed in NumPy 2
            np.mat = np.asmatrix
        sys.path.insert(0, str(TB))
        try:
            import skimage  # noqa: F401  (imported by the toolbox's utils)
        except ImportError:
            raise unittest.SkipTest("scikit-image not installed (pip install scikit-image); toolbox equivalence not checked")
        if not TB.exists():
            raise unittest.SkipTest("third_party/rPPG-Toolbox not cloned (make third_party)")
        from unsupervised_methods.methods import CHROME_DEHAAN, GREEN, LGI, PBV, POS_WANG
        _solve = np.linalg.solve

        def solve_np1(a, b):  # NumPy 1 broadcasting of solve((k,m,m),(k,m)) used by the toolbox's PBV
            b = np.asarray(b)
            return _solve(a, b[..., None])[..., 0] if b.ndim == np.ndim(a) - 1 else _solve(a, b)

        def pbv_tb(f, fs):
            np.linalg.solve = solve_np1
            try:
                return PBV.PBV(f)
            finally:
                np.linalg.solve = _solve

        cls.tb = {"GREEN": lambda f, fs: GREEN.GREEN(f), "POS": POS_WANG.POS_WANG, "CHROM": CHROME_DEHAAN.CHROME_DEHAAN,
                  "LGI": lambda f, fs: LGI.LGI(f), "PBV": pbv_tb}
        rng = np.random.default_rng(7)
        fs, T = 30.0, 900
        t = np.arange(T) / fs
        pulse = np.sin(2 * np.pi * 1.25 * t) + 0.3 * np.sin(2 * np.pi * 2.5 * t)
        motion = 0.01 * np.sin(2 * np.pi * 0.4 * t) + 0.004 * rng.standard_normal(T)
        cls.rgb = np.array([180.0, 140.0, 110.0]) * (1 + 0.003 * np.outer(pulse, [0.33, 0.77, 0.53]) + motion[:, None]
                                                    + 0.0005 * rng.standard_normal((T, 3)))
        cls.fs = fs

    def check(self, name, tol=1e-6):
        ours = M.run(name, self.rgb, self.fs)
        theirs = np.asarray(self.tb[name](_frames_from_rgb(self.rgb), self.fs)).ravel()
        n = min(len(ours), len(theirs))
        r = np.corrcoef(ours[:n], theirs[:n])[0, 1]
        self.assertGreater(r, 1 - tol, f"{name}: r={r}")

    def test_green(self):
        self.check("GREEN")

    def test_pos(self):
        self.check("POS")

    def test_chrom(self):
        self.check("CHROM")

    def test_lgi(self):
        self.check("LGI")

    def test_pbv(self):
        self.check("PBV")


if __name__ == "__main__":
    unittest.main()

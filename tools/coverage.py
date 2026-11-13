"""Line coverage of core modules with the stdlib `trace` module (coverage.py is not installable).
Usage: python tools/coverage.py  -> results/tables/coverage.csv; exit 1 if core coverage < 80 %."""
import ast
import csv
import sys
import trace
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CORE = ["sama/signals.py", "sama/methods.py", "sama/stats.py", "sama/conformal.py", "sama/calib.py", "sama/ita.py",
        "sama/net.py", "sama/export.py", "sama/streaming.py", "sama/sama_method.py", "sama/model_io.py", "sim/samasim.py"]


def executable_lines(path):
    tree = ast.parse(Path(path).read_text())
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
                continue  # docstrings
            lines.add(node.lineno)
    return lines


def main():
    import sysconfig
    ign = sorted({sysconfig.get_paths()[k] for k in ("stdlib", "platstdlib", "purelib", "platlib")} | {p for p in sys.path if "packages" in p})
    # trace caches ignore decisions by bare module name, so a site-packages "signals.py" would hide
    # ours; trace everything and filter afterwards
    del ign
    tracer = trace.Trace(count=True, trace=False)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    res = tracer.runfunc(unittest.TextTestRunner(verbosity=0).run, suite)
    counts = tracer.results().counts
    rows, tot_e, tot_c = [], 0, 0
    for rel in CORE:
        p = str((ROOT / rel).resolve())
        ex = executable_lines(p)
        hit = {ln for (f, ln), c in counts.items() if f == p and c > 0} & ex
        rows.append({"module": rel, "statements": len(ex), "covered": len(hit), "pct": round(100 * len(hit) / max(len(ex), 1), 1)})
        tot_e += len(ex)
        tot_c += len(hit)
    rows.append({"module": "TOTAL (core)", "statements": tot_e, "covered": tot_c, "pct": round(100 * tot_c / tot_e, 1)})
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results/tables/coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['module']:<22} {r['covered']:>5}/{r['statements']:<5} {r['pct']:>5}%")
    sys.exit(0 if res.wasSuccessful() and tot_c / tot_e >= 0.80 else 1)


if __name__ == "__main__":
    main()

"""Render a template with {{name:fmt}} numbers resolved through results/manifest.json."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from report.build_report import resolve  # noqa: E402

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    Path(dst).write_text(resolve(Path(src).read_text()))
    print("rendered", dst)

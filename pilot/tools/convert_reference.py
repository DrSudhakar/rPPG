"""Convert an oximeter (or chest-strap) export to the canonical reference CSV.

    python pilot/tools/convert_reference.py export.csv out.csv --time-col Time \
        --time-format "%Y-%m-%d %H:%M:%S.%f" --pleth-col Pleth --hr-col PR --spo2-col SpO2

--time-format: a strftime pattern for clock times (local time of the PC), or one of
  unix_s / unix_ms (already numeric) / relative_s (seconds from the start; then also give
  --start "2026-11-12 10:31:05" if you know the PC time of the first sample).
--fs: sampling rate when the file has no time column at all (times are then relative).
Writes columns: time_s (unix seconds, or relative seconds), pleth, hr_bpm, spo2, rr_ms (if given).
"""
import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--time-col")
    ap.add_argument("--time-format", default="unix_s")
    ap.add_argument("--start", help="PC time of the first sample for relative times")
    ap.add_argument("--fs", type=float, default=60.0)
    ap.add_argument("--pleth-col")
    ap.add_argument("--hr-col")
    ap.add_argument("--spo2-col")
    ap.add_argument("--rr-col")
    ap.add_argument("--sep", default=None, help="column separator (default: detect)")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.src, sep=a.sep, engine="python")
    df.columns = [c.strip() for c in df.columns]
    n = len(df)
    if a.time_col:
        col = df[a.time_col]
        if a.time_format == "unix_s":
            t = pd.to_numeric(col, errors="coerce").to_numpy(float)
        elif a.time_format == "unix_ms":
            t = pd.to_numeric(col, errors="coerce").to_numpy(float) / 1000.0
        elif a.time_format == "relative_s":
            t = pd.to_numeric(col, errors="coerce").to_numpy(float)
            if a.start:
                t = t + datetime.fromisoformat(a.start).timestamp()
        else:
            t = np.array([datetime.strptime(str(x).strip(), a.time_format).timestamp() for x in col])
    else:
        t = np.arange(n) / a.fs + (datetime.fromisoformat(a.start).timestamp() if a.start else 0.0)
    out = pd.DataFrame({"time_s": np.round(t, 4)})
    for name, c in (("pleth", a.pleth_col), ("hr_bpm", a.hr_col), ("spo2", a.spo2_col), ("rr_ms", a.rr_col)):
        if c:
            if c not in df.columns:
                sys.exit(f"column {c!r} not in {list(df.columns)}")
            out[name] = pd.to_numeric(df[c], errors="coerce")
    out.to_csv(a.dst, index=False)
    dt = np.diff(t[np.isfinite(t)])
    fs = 1 / np.median(dt) if dt.size else float("nan")
    miss = {k: f"{out[k].isna().mean():.1%}" for k in out.columns if k != "time_s"}
    print(f"wrote {a.dst}: {n} rows, {t[-1] - t[0]:.1f} s, about {fs:.1f} Hz, missing {miss}")
    if dt.size and (dt <= 0).any():
        print("WARNING: time stamps are not increasing; check --time-format")


if __name__ == "__main__":
    main()

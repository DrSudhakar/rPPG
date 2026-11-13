"""C9 acceptance tests for the streaming pipeline (thresholds fixed in DECISIONS D-24 before measuring).

real-time factor <= 0.5 on one CPU core (CPU proxy, labelled), peak memory < 200 MB (tracemalloc,
Python allocations), latency (time to first reading, update interval, per-update compute),
fault injection (never a confident wrong number: shown and |error| > 5 bpm counts as a failure),
privacy (no file written during streaming), abstention per stratum and condition.
SYNTHETIC inputs from SAMA-Sim.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # before numpy is imported (R7-07)
    os.environ[_v] = "1"
import json  # noqa: E402
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sama import model_io  # noqa: E402
from sama.streaming import StreamingEstimator  # noqa: E402
from sim import samasim as S  # noqa: E402

from sama import calib  # noqa: E402


def estimator(b, ita, use_onnx=True, part=None, device="phoneA", light="standard", p=None):
    """Step 2 on, as in training (R7-01): chart calibration for this device and light."""
    tone = noise = None
    if part is not None:
        dn, static = S.chart_for(part, device, light, p or S.SimParams())
        tone, noise = calib.fit_tone_curve(dn, S.CHART_REFLECTANCE), calib.photon_transfer(static, 400.0)
    return StreamingEstimator(gate=b["gate"], sigma_model=b["sigma"], params=None if (use_onnx and b["onnx"]) else b["params"],
                              onnx_path=b["onnx"] if use_onnx else None, ita=ita, tone=tone, noise=noise)


def stream(est, rec, faults=None, rng=None):
    rng = rng or np.random.default_rng(0)
    fps = rec.fps
    out = []
    t_proc = 0.0
    n = rec.rgb_patches.shape[0]
    for i in range(n):
        t = i / fps
        pt = rec.rgb_patches[i].astype(float)
        if faults == "dropped_frames" and rng.random() < 0.25:
            continue
        if faults == "frame_rate_change" and i > n // 2 and i % 2:
            continue
        if faults == "face_lost" and (n // 2 < i < n // 2 + 6 * fps):
            pt = None
        if faults == "exposure_flips" and pt is not None and (i // int(2 * fps)) % 2:
            pt = pt * 1.6
        if faults == "lighting_change" and pt is not None and i > n // 2:
            pt = pt * (0.35 + 0.05 * np.sin(2 * np.pi * 1.1 * t))   # dimming with in-band flicker
        t0 = time.perf_counter()
        r = est.push(t, pt, rec.meta["pix"])
        t_proc += time.perf_counter() - t0
        if r is not None:
            beats = rec.beats[(rec.beats >= t - 30) & (rec.beats < t)]
            out.append({**r.to_dict(), "hr_true": float(60 / np.mean(np.diff(beats))) if beats.size > 2 else np.nan,
                        "hr_internal": r.hr})
    return out, t_proc


def main():
    b = model_io.load_bundle(ROOT / "deploy/model")
    p = S.SimParams()
    rows, faults_rows, abst_rows = [], [], []
    # real-time factor, memory, latency on 120 s of 30 fps video
    part = S.make_participant("acc", "middle", np.random.default_rng(11), ita=18.0)
    rec = S.simulate_block(part, "rest", "standard", "phoneA", p, dur_s=120.0)
    for use_onnx in (True, False):
        est = estimator(b, 18.0, use_onnx, part=part)
        tracemalloc.start()
        out, t_proc = stream(est, rec)
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        first = next((o["t"] for o in out if o["reason"] != "WARMING_UP"), np.nan)
        steady = [o["latency_ms"] for o in out if o["reason"] != "WARMING_UP"]
        rtf = (sum(steady) / 1000.0) / max(len(steady), 1)   # compute seconds per second of video, steady state (R10-03)
        rows.append({"test": f"real-time factor, steady state ({'onnxruntime' if use_onnx else 'numpy'})", "value": rtf,
                     "threshold": "<= 0.5", "pass": rtf <= 0.5, "note": f"CPU proxy: {platform.processor() or 'x86_64'}, 1 thread; per-1 s update after warm-up"})
        rows.append({"test": f"peak Python memory MB ({'onnx' if use_onnx else 'numpy'})", "value": peak, "threshold": "< 200", "pass": peak < 200,
                     "note": "tracemalloc (Python allocations; excludes onnxruntime arena)"})
        lat = [o["latency_ms"] for o in out if o["reason"] != "WARMING_UP"]
        rows.append({"test": f"time to first reading s ({'onnx' if use_onnx else 'numpy'})", "value": first, "threshold": "report",
                     "pass": True, "note": "30 s window implies >= 30 s"})
        rows.append({"test": f"per-update compute ms, median ({'onnx' if use_onnx else 'numpy'})", "value": float(np.median(lat)),
                     "threshold": "report", "pass": True, "note": f"update interval 1 s; p95 {np.percentile(lat, 95):.0f} ms"})
    # fault injection across strata
    rng = np.random.default_rng(3)
    for ita in (40.0, 18.0, -15.0):
        part = S.make_participant(f"f{ita}", S.stratum_of(ita), np.random.default_rng(int(ita) + 100), ita=ita)
        base = S.simulate_block(part, "rest", "standard", "phoneA", p, dur_s=90.0)
        recs = {"none": base, "dropped_frames": base, "frame_rate_change": base, "face_lost": base, "exposure_flips": base,
                "lighting_change": base,
                "heavy_motion": S.simulate_block(part, "rest", "standard", "phoneA", S.with_params(motion=(("rest", 0.03),)), dur_s=90.0),
                "codec_artefacts_250k": S.simulate_block(part, "rest", "standard", "phoneA", p, codec="250", dur_s=90.0),
                "dim_100lux": S.simulate_block(part, "rest", "dim", "phoneA", p, dur_s=90.0)}
        for name, r in recs.items():
            out, _ = stream(estimator(b, ita, part=part, light="dim" if name == "dim_100lux" else "standard"), r, faults=name if name in ("dropped_frames", "frame_rate_change", "face_lost", "exposure_flips", "lighting_change") else None, rng=rng)
            o = [x for x in out if x["reason"] != "WARMING_UP"]
            shown = [x for x in o if x["shown"]]
            wrong = [x for x in shown if np.isfinite(x["hr_true"]) and abs(x["hr_internal"] - x["hr_true"]) > 5]
            faults_rows.append({"fault": name, "stratum": S.stratum_of(ita), "updates": len(o), "shown": len(shown),
                                "confident_wrong": len(wrong), "reasons": json.dumps(pd.Series([x["reason"] for x in o]).value_counts().to_dict())})
    fdf = pd.DataFrame(faults_rows)
    fdf.to_csv(ROOT / "results/tables/acceptance_faults.csv", index=False)
    n_wrong = int(fdf.confident_wrong.sum())
    rows.append({"test": "fault injection: confident wrong numbers", "value": n_wrong, "threshold": "0", "pass": n_wrong == 0,
                 "note": f"{len(fdf)} fault x stratum runs; shown and |error| > 5 bpm counts as wrong"})
    # privacy: no file written while streaming
    before = {str(q) for q in ROOT.rglob("*") if ".git" not in q.parts and "cache" not in q.parts}
    stream(estimator(b, 18.0, part=part), rec)
    after = {str(q) for q in ROOT.rglob("*") if ".git" not in q.parts and "cache" not in q.parts}
    manifest = (ROOT / "deploy/android/app/src/main/AndroidManifest.xml").read_text()
    rows.append({"test": "privacy: files written during streaming", "value": len(after - before), "threshold": "0", "pass": after == before, "note": ""})
    rows.append({"test": "privacy: Android manifest network permission", "value": int("android.permission.INTERNET" in manifest), "threshold": "0",
                 "pass": "android.permission.INTERNET" not in manifest, "note": "source NOT COMPILED"})
    # abstention per stratum and condition (from the C7 shift table if present)
    cs = ROOT / "results/tables/c7_coverage_shift.csv"
    if cs.exists():
        rows.append({"test": "abstention per stratum x condition reported", "value": len(pd.read_csv(cs)), "threshold": "report", "pass": True,
                     "note": "results/tables/c7_coverage_shift.csv and abstention_tradeoff.csv (I-08)"})
    pd.DataFrame(rows).to_csv(ROOT / "results/tables/acceptance.csv", index=False)
    print(pd.DataFrame(rows).to_string())
    print(fdf.groupby("fault")[["updates", "shown", "confident_wrong"]].sum())


if __name__ == "__main__":
    main()

"""Generate SYNTHETIC data for the rehearsal.

  python sim/generate.py pool   --world A --per-stratum 90 --seed 1    # C6: POS windows, lossless + 500 kbps
  python sim/generate.py cohort --world A --per-stratum 40  --seed 7   # C7/C8: full SAMA-120 features

Outputs go to results/cache/ (large, git-ignored) and are summarised by the analysis scripts.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sama import calib as C  # noqa: E402
from sama import methods as M  # noqa: E402
from sama import pipeline as P  # noqa: E402
from sama import sama_method as SM  # noqa: E402
from sama.signals import clean_ibi, hr_from_signal, rmssd_ms  # noqa: E402
from sim import samasim as S  # noqa: E402

LIGHTS = ("standard", "dim")
WORLDS = {"A": {}, "B": {"world": "B"}, "null": {"world": "null"}, "melanin": {"world": "melanin", "melanin_rel_amp_slope": 0.5}}


def params_for(world: str) -> S.SimParams:
    return S.with_params(**WORLDS[world])


# ------------------------------------------------------------------ pool (C6)


def pool_participant(args):
    part, world = args
    p = params_for(world)
    rows = []
    for light in LIGHTS:
        for blk in S.SEATED_BLOCKS:
            for codec in ("lossless", "500"):
                rec = S.simulate_block(part, blk, light, "reference30", p, codec=codec)
                ref = P.ref_hr_windows(rec.beats, rec.beats_valid, rec.rgb_face.shape[0], rec.fps)
                for r in P.windowed_hr(M.pos(rec.rgb_face, rec.fps), rec.fps, ref):
                    rows.append((part.pid, part.ita, part.stratum, part.sex, blk, light, codec, r["hr_ref"], r["hr_cam"], r["snr_db"]))
    return rows


def run_pool(world: str, per_stratum: int, seed: int, procs: int = 2):
    rng = np.random.default_rng(seed)
    cohort = S.make_cohort(per_stratum, rng)
    t = time.time()
    with Pool(procs) as pool:
        out = pool.map(pool_participant, [(pp, world) for pp in cohort], chunksize=2)
    cols = ("pid", "ita", "stratum", "sex", "block", "light", "codec", "hr_ref", "hr_cam", "snr_db")
    path = ROOT / f"results/cache/pool_{world}_s{seed}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write(",".join(cols) + "\n")
        for rows in out:
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")
    print(f"pool {world}: {len(cohort)} participants in {time.time() - t:.0f}s -> {path}")


# ------------------------------------------------------------------ cohort (C7/C8)


def _features(part, blk, light, device, p, codec="lossless", ita_obs=None):
    rec = S.simulate_block(part, blk, light, device, p, codec=codec)
    dn, static = S.chart_for(part, device, light, p)
    tone = C.fit_tone_curve(dn, S.CHART_REFLECTANCE)
    noise = C.photon_transfer(static, 400.0)
    ref = P.ref_hr_windows(rec.beats, rec.beats_valid, rec.rgb_face.shape[0], rec.fps)
    meta = {"pid": part.pid, "stratum": part.stratum, "ita": part.ita, "block": blk, "device": device, "codec": codec,
            "sex": part.sex, "exposure_s": rec.meta["exposure_s"], "gain": rec.meta["gain"],
            "electrons_g": rec.meta["electrons_g"], "face_dn_g": rec.meta["face_dn_g"]}
    ita_in = part.ita if ita_obs is None else ita_obs
    wf = SM.window_features(rec.rgb_patches, rec.fps, rec.meta["pix"], ita_in, tone, noise, ref,
                            ppg=rec.meta["ppg_ecg"], light=light, meta=meta)
    base = P.windowed_hr  # baselines on the face mean
    bl = {m: [r["hr_cam"] for r in base(M.run(m, rec.rgb_face, rec.fps), rec.fps, ref)] for m in M.METHODS}
    return wf, bl, rec


def _hrv(part, p, device, codec="lossless"):
    """RMSSD (ECG vs camera candidate) for rest and arithmetic, standard light, 60 s windows."""
    out = {}
    for blk in ("rest", "mental_arithmetic"):
        rec = S.simulate_block(part, blk, "standard", device, p, codec=codec)
        fs = rec.fps
        beats = rec.beats[rec.beats_valid]
        lin = rec.rgb_patches.astype(float)
        rel = lin / lin.mean(0, keepdims=True)
        pix = rec.meta["pix"]
        wrel = (rel * (pix / pix.sum())[None, :, None]).sum(1)
        cand = M.pos(wrel, fs)
        rm_ecg, rm_cam = [], []
        for a in np.arange(0, rec.rgb_face.shape[0] / fs - 60 + 1e-9, 60.0):
            inside = (rec.beats >= a) & (rec.beats < a + 60)
            if inside.any() and (~rec.beats_valid[inside]).mean() > 0.10:   # SAP s.1 ECG quality rule
                continue
            e = beats[(beats >= a) & (beats < a + 60)]
            rm_ecg.append(rmssd_ms(clean_ibi(np.diff(e))))
            seg = cand[int(a * fs):int((a + 60) * fs)]
            h_cam = hr_from_signal(seg, fs)[0]                 # camera's own HR: no ECG information (R3-02)
            rm_cam.append(SM.rmssd_from_pulse(seg, fs, hr_hint=h_cam))
        out[blk] = (float(np.nanmean(rm_ecg)), float(np.nanmean(rm_cam)))
    return out


def cohort_participant(args):
    part, world, out_dir = args
    p = params_for(world)
    rec_out = {"part": part.__dict__, "lossless": {}, "shift": {}, "hrv": {}}
    for light in LIGHTS:
        for blk in S.SEATED_BLOCKS:
            wf, bl, rec = _features(part, blk, light, "reference30", p)
            rec_out["lossless"][(light, blk)] = (wf, bl)
    # shift / H3 / codec conditions on two seated blocks
    for blk in ("rest", "mock_teleconsultation"):
        for light in LIGHTS:
            for dev in ("phoneA", "phoneB"):
                rec_out["shift"][(dev, light, blk, "lossless")] = _features(part, blk, light, dev, p)[:2]
        rec_out["shift"][("webcam", "standard", blk, "lossless")] = _features(part, blk, "standard", "webcam", p)[:2]
        for codec in ("500", "250"):
            rec_out["shift"][("reference30", "standard", blk, codec)] = _features(part, blk, "standard", "reference30", p, codec=codec)[:2]
    for dev in ("reference", "reference30hq", "reference30"):
        rec_out["hrv"][dev] = _hrv(part, p, dev)
    rec_out["hrv"]["reference30_500"] = _hrv(part, p, "reference30", "500")
    with open(out_dir / f"{part.pid}.pkl", "wb") as f:
        pickle.dump(rec_out, f, protocol=4)
    return part.pid


def run_cohort(world: str, per_stratum: int, seed: int, procs: int = 2):
    rng = np.random.default_rng(seed)
    cohort = S.make_cohort(per_stratum, rng)
    out_dir = ROOT / f"results/cache/cohort_{world}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [pp for pp in cohort if not (out_dir / f"{pp.pid}.pkl").exists()]
    t = time.time()
    with Pool(procs) as pool:
        for i, pid in enumerate(pool.imap_unordered(cohort_participant, [(pp, world, out_dir) for pp in todo])):
            if i % 10 == 0:
                print(f"{i + 1}/{len(todo)} {time.time() - t:.0f}s", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps({"world": world, "per_stratum": per_stratum, "seed": seed,
                                                       "n": len(cohort), "seconds": time.time() - t}))
    print(f"cohort {world}: {len(cohort)} participants in {time.time() - t:.0f}s -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pool", "cohort"])
    ap.add_argument("--world", default="A")
    ap.add_argument("--per-stratum", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    (run_pool if a.what == "pool" else run_cohort)(a.world, a.per_stratum, a.seed, a.procs)

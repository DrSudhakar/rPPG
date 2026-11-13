"""Pilot pipeline: from recorded sessions to the results promised in proposal section 5.

Per session (`prepare`), each step cached in <session>/derived/ (re-run with --force):
  1. webcam (lossless) -> per-frame patch colours, face boxes, motion, skin fraction
  2. compression ladder (H.264 2000/1000/500/250 kbps, VP8 500/250) from the lossless video,
     each level measured on exactly the same face boxes
  3. phone video (optional), its own face tracking
  4. camera ITA° from the ITA capture block (white card or chart), and colour features for the
     video->skin-tone model
  5. reference (oximeter) loaded and synchronised to each stream (wall clock + cross-correlation)
  6. quality control: decodable, frame rate and gaps, face present, light, reference coverage

Per session (`analyse`): 30 s windows (15 s step) inside protocol blocks; reference HR per window;
camera HR per method (POS, CHROM, GREEN, ICA, LGI, PBV and SAMA) on every stream and level; SAMA's
interval and abstention; HRV (RMSSD, SDNN, pNN50) in 60 s windows of the rest blocks.

Across sessions (`results`): the four section-5 results (feasibility, skin tone, compression,
agreement), Figure 6 (the proposal's own pilot_figure.py), SAMA's per-stratum error, coverage and
abstention, HRV agreement, camera-ITA vs Monk, rater agreement, and the power re-check with the
pilot's spread. Nothing is a hypothesis test: the pilot is too small, and the report says so.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import methods as M
from .ita_camera import estimate as estimate_ita, stratum_from
from .monk import agreement as rater_agreement, consensus, ita_vs_monk, load_ratings, stratum_from_monk
from .reference import find_reference, load_reference, pnn50
from .sama_method import weighted_pulse, window_features
from .session import list_sessions, load_participants, load_session, session_minutes
from .signals import bandpass, clean_ibi, hr_from_signal, peaks_parabolic, rmssd_ms, sdnn_ms, window_starts
from .sync import align
from .video import FaceTracker, VideoSignals, extract_signals, have_ffmpeg, iter_frames, probe, resample_uniform

ROOT = Path(__file__).resolve().parents[1]
LOSSLESS = "lossless"


# ---------------------------------------------------------------- per-session preparation --
def ladder(src: Path, out: Path, cfg, force=False, log=print) -> dict:
    """Compression ladder from the lossless video, as scripts/transcode_ladder.sh (same settings)."""
    out.mkdir(exist_ok=True)
    made = {}
    vf = f"scale='min({int(cfg.pilot.ladder.max_width)},iw)':-2,fps=30"
    for kb in cfg.pilot.ladder.h264_kbps:
        dst = out / f"h264_{kb}k.mp4"
        if force or not dst.exists():
            pl = out / f"x264pass_{kb}"
            for p in (1, 2):
                cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
                       "-b:v", f"{kb}k", "-maxrate", f"{kb}k", "-bufsize", f"{2 * kb}k", "-pix_fmt", "yuv420p", "-pass", str(p),
                       "-passlogfile", str(pl), "-an"] + (["-f", "mp4", "/dev/null"] if p == 1 else [str(dst)])
                subprocess.run(cmd, check=True)
            for f in out.glob(f"x264pass_{kb}*"):
                f.unlink()
        made[f"h264_{kb}"] = dst
    for kb in cfg.pilot.ladder.vp8_kbps:
        dst = out / f"vp8_{kb}k.webm"
        if force or not dst.exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf, "-c:v", "libvpx", "-b:v", f"{kb}k",
                            "-minrate", f"{kb}k", "-maxrate", f"{kb}k", "-deadline", "realtime", "-cpu-used", "8",
                            "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)
        made[f"vp8_{kb}"] = dst
    log(f"[ladder] {len(made)} levels in {out}")
    return made


def _analysis_size(path, cfg):
    """Frames are reduced (area averaging) to signal.analysis_width before measurement: patch means
    are unchanged and decoding is several times faster. 0 keeps the native size."""
    w = int(cfg.signal.get("analysis_width", 0) or 0)
    info = probe(path)
    if not w or info["width"] <= w:
        return None
    h = int(round(info["height"] * w / info["width"] / 2) * 2)
    return (w, h)


def _tracker(cfg, s) -> FaceTracker:
    return FaceTracker(mode=cfg.face.tracker, fixed_box=tuple(cfg.face.fixed_box),
                       manual_box=None if s.box("face_box") is None else tuple(s.box("face_box")),
                       redetect_every=max(1, int(cfg.face.redetect_every_s * 30)), smoothing=float(cfg.face.smoothing))


def _ita_segments(s, cfg):
    """Every block named ita.camera.segment (the protocol has one per light); first 30 s of each.
    Without such a block, the first 20 s of the recording."""
    segs = [(b.start, min(b.end, b.start + 30.0), b.light) for b in s.blocks if b.name == cfg.ita.camera.segment]
    if not segs:
        b = s.blocks[0]
        segs = [(b.start, min(b.end, b.start + 20.0), b.light)]
    return segs


BLOCKING_ITA_FLAGS = {"NO_SKIN_PIXELS", "L_OUT_OF_SKIN_RANGE", "ITA_OUT_OF_HUMAN_RANGE", "CARD_REFLECTANCE_MISSING",
                      "FELL_BACK_TO_NO_REFERENCE", "NO_WHITE_REF"}


def prepare(sdir, cfg, force=False, log=print) -> dict:
    s = load_session(sdir)
    d = s.derived
    status = {"participant_id": s.pid, "steps": {}}
    web = s.video("webcam")
    if web is None:
        status["error"] = "no webcam video"
        (d / "qc.json").write_text(json.dumps({"participant_id": s.pid, "usable": False, "reasons": ["no webcam video"]}, indent=1))
        return status
    sig_path = d / "signals_webcam_lossless.npz"
    if force or not sig_path.exists():
        t = time.time()
        vs = extract_signals(web, _tracker(cfg, s), size=_analysis_size(web, cfg))
        vs.save(sig_path)
        log(f"[{s.pid}] webcam: {len(vs.t)} frames at {vs.fps:.2f} fps, face box from {vs.tracker} ({time.time() - t:.0f} s)")
    vs = VideoSignals.load(sig_path)
    status["steps"]["webcam"] = "ok"
    if cfg.pilot.ladder.enabled and have_ffmpeg():
        made = ladder(web, s.ladder_dir(), cfg, force, log)
        for lvl, path in made.items():
            p = d / f"signals_webcam_{lvl}.npz"
            if force or not p.exists():
                extract_signals(path, boxes=vs.boxes, size=_analysis_size(path, cfg)).save(p)
        status["steps"]["ladder"] = sorted(made)
    ph = s.video("phone")
    if ph is not None and "phone" in cfg.pilot.streams:
        p = d / "signals_phone_native.npz"
        if force or not p.exists():
            extract_signals(ph, FaceTracker(mode="auto" if cfg.face.tracker != "fixed_box" else "fixed_box",
                                            fixed_box=tuple(cfg.face.fixed_box))).save(p)
        status["steps"]["phone"] = "ok"
    # camera ITA and colour features (one estimate per ITA block, i.e. per light; their mean is used)
    ita_path = d / "ita_camera.json"
    if force or not ita_path.exists():
        method = cfg.ita.camera.white_reference
        wb, cc = s.box("white_box"), s.box("chart_corners")
        extra, note = [], ""
        if method == "card" and wb is None:
            method, note = "none", "white_reference=card but no white_box in session.json: fell back to none"
            extra.append("FELL_BACK_TO_NO_REFERENCE")
        elif method == "chart" and cc is None:
            method, note = "none", "white_reference=chart but no chart_corners in session.json: fell back to none"
            extra.append("FELL_BACK_TO_NO_REFERENCE")
        refl = s.meta.get("white_reflectance")
        if method == "card" and refl is None:
            extra.append("CARD_REFLECTANCE_MISSING")     # never guess the card: a wrong value shifts every ITA°
            refl = float(cfg.ita.camera.white_reflectance)
        refl = float(refl if refl is not None else cfg.ita.camera.white_reflectance)
        per, feats = [], {}
        for a, b, light in _ita_segments(s, cfg):
            n_want = int(cfg.ita.camera.frames)
            fr = list(iter_frames(web, max_s=b - a, start_s=a))
            step = max(1, len(fr) // max(n_want, 1))
            fr = fr[::step][:n_want]
            i0 = int(np.searchsorted(vs.t, a))
            boxes = vs.boxes[i0:i0 + len(fr) * step:step] if len(vs.boxes) > i0 else vs.boxes[-1:]
            est = estimate_ita(fr, boxes, wb, cc, method, refl, tuple(cfg.ita.camera.regions),
                               float(cfg.ita.camera.trim_pct), True, float(cfg.ita.camera.max_iqr_deg), tuple(cfg.strata.ita_cuts))
            e = est.to_dict()
            e.update(segment_s=[a, b], light=light)
            per.append(e)
            if not feats:
                from .ita_model import video_features
                feats = video_features(fr, boxes, wb, refl)
        ok = [e for e in per if np.isfinite(e["ita"])]
        ita = float(np.mean([e["ita"] for e in ok])) if ok else float("nan")
        flags = sorted(set(sum((e["flags"] for e in per), []) + extra))
        res = {"ita": ita, "per_light": per, "flags": flags, "method": method, "white_reflectance": refl,
               "light_difference_deg": float(ok[0]["ita"] - ok[1]["ita"]) if len(ok) > 1 else None,
               "iqr": float(np.max([e["iqr"] for e in ok])) if ok else float("nan"),
               "usable": bool(ok) and not (set(flags) & BLOCKING_ITA_FLAGS)}
        if note:
            res["note"] = note
        ita_path.write_text(json.dumps(res, indent=1, default=float))
        (d / "ita_features.json").write_text(json.dumps(feats, indent=1))
        log(f"[{s.pid}] camera ITA {ita:.1f}° from {len(ok)} block(s) ({method}, card reflectance {refl}) flags={flags}")
    status["steps"]["ita"] = "ok"
    # reference and synchronisation
    rp = find_reference(s.root, cfg.reference.file_glob)
    sync = {}
    if rp is not None:
        ref = load_reference(rp, cfg.reference.type, dict(cfg.reference.columns), cfg.reference.time_format, float(cfg.reference.fs_hz))
        t0 = s.meta.get("video_t0_wall")
        r = align(vs.t, vs.patches, float(t0) if t0 is not None else float("nan"), ref, cfg.sync.method, float(cfg.sync.max_lag_s))
        sync["webcam"] = r.to_dict()
        if "phone" in status["steps"]:
            pv = VideoSignals.load(d / "signals_phone_native.npz")
            pt0 = (s.meta.get("phone") or {}).get("t0_wall_approx")
            rr = align(pv.t, pv.patches, float(pt0) if pt0 is not None else float("nan"), ref, cfg.sync.method,
                       float(cfg.sync.phone_max_lag_s))
            sync["phone"] = rr.to_dict()
        sync["reference_notes"] = ref.notes
        sync["reference_kind"] = ref.kind
        log(f"[{s.pid}] sync webcam: lag {r.lag_s:+.3f} s ({r.method}), spread {r.spread_ms:.0f} ms over {r.n_windows} windows")
    else:
        sync["error"] = f"no reference file matching {cfg.reference.file_glob}"
    (d / "sync.json").write_text(json.dumps(sync, indent=1, default=float))
    qc = quality_control(s, vs, cfg, sync)
    (d / "qc.json").write_text(json.dumps(qc, indent=1, default=float))
    status["qc"] = qc
    return status


def quality_control(s, vs: VideoSignals, cfg, sync: dict) -> dict:
    q = cfg.quality
    reasons = []
    dt = np.diff(vs.t)
    gaps = int((dt > q.max_gap_s).sum()) if dt.size else 0
    face = float((vs.skin_fraction >= cfg.face.min_skin_fraction).mean())
    bright = float(np.median(vs.brightness))
    if vs.fps < q.min_fps:
        reasons.append(f"frame rate {vs.fps:.1f} < {q.min_fps}")
    if face < q.qc_min_face_frac:
        reasons.append(f"face present in {face:.0%} of frames (< {q.qc_min_face_frac:.0%})")
    if bright < q.low_light_dn:
        reasons.append(f"face too dark (median green {bright:.0f} DN)")
    ref_cov = float("nan")
    if "webcam" in sync:
        rp = find_reference(s.root, cfg.reference.file_glob)
        ref = load_reference(rp, cfg.reference.type, dict(cfg.reference.columns), cfg.reference.time_format, float(cfg.reference.fs_hz))
        ref = ref.shifted(-sync["webcam"]["lag_s"])
        hs = [ref.window_hr(a, a + 30) for a in np.arange(0, max(vs.t[-1] - 30, 0), 15)]
        ref_cov = float(np.mean(np.isfinite(hs))) if hs else 0.0
        if ref_cov < q.qc_min_reference_valid:
            reasons.append(f"reference usable in {ref_cov:.0%} of windows (< {q.qc_min_reference_valid:.0%})")
    else:
        reasons.append("no reference")
    mc = {}
    for b in s.blocks:
        if b.name == "motion_check":
            still = vs.motion[(vs.t >= b.start + 2) & (vs.t < b.start + 18)]
            move = vs.motion[(vs.t >= b.start + 20) & (vs.t < b.end)]
            if still.size and move.size:
                mc = {"still_median": float(np.median(still)), "still_p90": float(np.percentile(still, 90)),
                      "move_median": float(np.median(move)),
                      "suggested_motion_warn": float(np.percentile(still, 99) * 1.5),
                      "suggested_motion_abstain": float(max(np.percentile(still, 99) * 3, np.median(move) * 0.7))}
    return {"participant_id": s.pid, "usable": not reasons, "reasons": reasons, "duration_s": float(vs.t[-1]), "motion_check": mc,
            "fps": vs.fps, "gaps_over_max": gaps, "face_fraction": face, "median_brightness_dn": bright,
            "median_motion": float(np.median(vs.motion)), "reference_window_coverage": ref_cov,
            "session_minutes": session_minutes(s), "blocks": [b.__dict__ for b in s.blocks], "tracker": vs.tracker}


# ---------------------------------------------------------------- per-session analysis -----
def _participant_tone(pid, cfg, derived: Path, measured: dict, monk_cons: dict, ita_model) -> dict:
    cam = json.loads((derived / "ita_camera.json").read_text()) if (derived / "ita_camera.json").exists() else {}
    feats = json.loads((derived / "ita_features.json").read_text()) if (derived / "ita_features.json").exists() else {}
    cam_ok = cam.get("usable", not (set(cam.get("flags", [])) & BLOCKING_ITA_FLAGS))
    out = {"ita_camera": cam.get("ita", np.nan) if cam_ok else np.nan, "ita_camera_flags": ";".join(cam.get("flags", [])),
           "ita_camera_raw": cam.get("ita", np.nan),
           "ita_measured": measured.get(pid, np.nan), "monk": monk_cons.get(pid, np.nan), "ita_predicted": np.nan}
    if ita_model is not None and feats and ita_model.target == "ita":
        out["ita_predicted"] = float(ita_model.predict([feats])[0])
    src = cfg.ita.source
    out["ita_source"] = src
    out["ita_deg"] = {"measured": out["ita_measured"], "camera": out["ita_camera"], "predicted": out["ita_predicted"]}.get(src, np.nan)
    if cfg.strata.by == "ita":
        out["stratum"] = stratum_from(out["ita_deg"], tuple(cfg.strata.ita_cuts))
        out["stratum_source"] = "ita"
        if out["stratum"] == "unknown" and np.isfinite(out["monk"]):
            # pre-specified: a participant without a usable ITA° (e.g. card never in its box) is placed by Monk rating
            out["stratum"] = stratum_from_monk(out["monk"], tuple(cfg.strata.monk_cuts))
            out["stratum_source"] = "monk (no usable ITA° for this participant)"
    elif cfg.strata.by == "monk":
        out["stratum"] = stratum_from_monk(out["monk"], tuple(cfg.strata.monk_cuts))
    else:
        out["stratum"] = "all"
    return out


def _segment(P, fps, a, b):
    return P[a:b]


def analyse(sdir, cfg, model=None, model_meta=None, tone: dict | None = None, log=print) -> tuple[list, list]:
    s = load_session(sdir)
    d = s.derived
    sync = json.loads((d / "sync.json").read_text()) if (d / "sync.json").exists() else {}
    rp = find_reference(s.root, cfg.reference.file_glob)
    if rp is None or "webcam" not in sync:
        return [], []
    ref0 = load_reference(rp, cfg.reference.type, dict(cfg.reference.columns), cfg.reference.time_format, float(cfg.reference.fs_hz))
    tone = tone or {}
    W, S = float(cfg.signal.window_s), float(cfg.signal.step_s)
    rows, hrv_rows = [], []
    streams = [("webcam", LOSSLESS, d / "signals_webcam_lossless.npz")]
    streams += [("webcam", p.stem.replace("signals_webcam_", ""), p) for p in sorted(d.glob("signals_webcam_*.npz")) if "lossless" not in p.stem]
    if (d / "signals_phone_native.npz").exists() and "phone" in sync:
        streams.append(("phone", "native", d / "signals_phone_native.npz"))
    methods = [m for m in cfg.signal.methods if m != "SAMA"]
    use_sama = "SAMA" in cfg.signal.methods and model is not None
    for stream, level, path in streams:
        vs = VideoSignals.load(path)
        lag = sync["phone" if stream == "phone" else "webcam"]["lag_s"]
        ref = ref0.shifted(-lag)                                       # reference on the video clock
        fps = 30.0
        g, P = resample_uniform(vs.t, vs.patches, fps)
        _, mot = resample_uniform(vs.t, vs.motion[:, None], fps)
        _, skf = resample_uniform(vs.t, vs.skin_fraction[:, None], fps)
        pix = np.nanmean(vs.pix, 0)
        face = (P * (pix / pix.sum())[None, :, None]).sum(1)
        codec, kbps = ("lossless", 0) if level == LOSSLESS else (("native", -1) if level == "native" else (level.split("_")[0], int(level.split("_")[1])))
        blocks = s.blocks if (cfg.signal.windows_within_blocks and stream == "webcam") else [type(s.blocks[0])("whole_recording", "unknown", 0.0, g[-1])]
        skip = set(cfg.pilot.exclude_blocks_from_accuracy)
        for blk in blocks:
            if blk.name in skip:
                continue
            a0, b0 = int(round(blk.start * fps)), min(int(round(blk.end * fps)), len(g))
            for a, b in window_starts(b0 - a0, fps, W, S):
                a, b = a + a0, b + a0
                t0 = a / fps
                h_ref = ref.window_hr(t0, t0 + W)
                seg = face[a:b]
                base = {"participant_id": s.pid, "stream": stream, "codec": codec, "bitrate_kbps": kbps, "level": level,
                        "block": blk.name, "light": blk.light, "t0": t0, "hr_ref_bpm": h_ref,
                        "motion": float(np.median(mot[a:b])), "motion_p90": float(np.percentile(mot[a:b], 90)),
                        "face_fraction": float(np.mean(skf[a:b] >= cfg.face.min_skin_fraction)),
                        "brightness": float(seg[:, 1].mean()), **{k: tone.get(k) for k in ("ita_deg", "ita_source", "ita_camera", "monk", "stratum")}}
                for m in methods:
                    try:
                        h, snr, _ = hr_from_signal(M.run(m, seg, fps), fps)
                    except Exception:  # noqa: BLE001
                        h, snr = np.nan, np.nan
                    rows.append({**base, "method": m, "hr_cam_bpm": h, "snr_db": snr, "shown": True, "lo": np.nan, "hi": np.nan})
                if use_sama:
                    ita_c = tone.get("ita_deg", np.nan)
                    wf = window_features(P[a:b], fps, pix, ita_c if (model.film and np.isfinite(ita_c)) else 0.0, None, None,
                                         [(0, b - a, h_ref)], ppg=None, light=blk.light, use_step2=False,
                                         use_step3=bool(model_meta.get("noise_weighting", True)))
                    if not model.film:
                        wf.cond[:, 0] = 0.0
                    from .realtrain import gate_group_for
                    grp = gate_group_for(model_meta, tone.get("stratum", "all"))
                    pr = model.predict(wf, np.array([grp]))
                    shown = bool(pr["shown"][0]) and base["motion_p90"] <= cfg.quality.motion_abstain
                    rows.append({**base, "method": "SAMA", "hr_cam_bpm": float(pr["hr"][0]), "snr_db": float(wf.q[0, 0]),
                                 "shown": shown, "lo": float(pr["lo"][0]), "hi": float(pr["hi"][0]),
                                 "sigma": float(pr["sigma"][0]), "gate_group": grp})
        # HRV on the lossless webcam (and phone), 60 s windows in rest-like blocks
        if cfg.hrv.enabled and (level in (LOSSLESS, "native")):
            HW = float(cfg.hrv.window_s)
            for blk in blocks:
                if blk.name in skip:
                    continue
                a0, b0 = int(round(blk.start * fps)), min(int(round(blk.end * fps)), len(g))
                for a, b in window_starts(b0 - a0, fps, HW, HW):
                    a, b = a + a0, b + a0
                    t0 = a / fps
                    rh = ref.window_hrv(t0, t0 + HW, float(cfg.hrv.max_edited_frac), int(cfg.hrv.min_beats))
                    for name, pulse in (("POS", M.pos(face[a:b], fps)), ("SAMA-weighted", weighted_pulse(P[a:b], fps)[0])):
                        cam = camera_hrv(pulse, fps, float(cfg.hrv.max_edited_frac))
                        hrv_rows.append({"participant_id": s.pid, "stream": stream, "block": blk.name, "light": blk.light, "t0": t0,
                                         "method": name, "motion_p90": float(np.percentile(mot[a:b], 90)),
                                         "ref_rmssd": rh["rmssd"], "ref_sdnn": rh["sdnn"], "ref_pnn50": rh["pnn50"],
                                         "cam_rmssd": cam["rmssd"], "cam_sdnn": cam["sdnn"], "cam_pnn50": cam["pnn50"],
                                         "cam_edited": cam["edited"], "stratum": tone.get("stratum"), "ita_deg": tone.get("ita_deg")})
    return rows, hrv_rows


def camera_hrv(pulse: np.ndarray, fps: float, max_edited: float = 0.05) -> dict:
    """Step 5: band-pass, parabolic peak timing, the same artefact rules as the reference."""
    from .sama_method import hr_autocorr
    x = bandpass(pulse, fps)
    hint = hr_autocorr(x, fps)
    t = peaks_parabolic(x, fps, hint)
    ibi = np.diff(t)
    ok = clean_ibi(ibi)
    edited = 1 - ok.size / max(ibi.size, 1)
    if ok.size < 20 or edited > max_edited:          # the same rule as the reference (proposal WP1)
        return {"rmssd": np.nan, "sdnn": np.nan, "pnn50": np.nan, "edited": float(edited)}
    return {"rmssd": rmssd_ms(ok), "sdnn": sdnn_ms(ok), "pnn50": pnn50(ok), "edited": float(edited)}


# ---------------------------------------------------------------- across sessions ----------
def load_measured(path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "ita_deg" in df.columns:
        return df.groupby("participant_id").ita_deg.mean().to_dict()
    if {"L", "b"} <= set(df.columns):        # several colorimeter readings: ITA of the mean L*, b*
        g = df.groupby("participant_id")[["L", "b"]].mean()
        return {k: float(np.degrees(np.arctan2(r.L - 50, r.b))) for k, r in g.iterrows()}
    raise ValueError(f"{p}: need ita_deg, or L and b columns")


def ba_clustered(d: pd.DataFrame, a="hr_cam_bpm", b="hr_ref_bpm", pid="participant_id") -> dict:
    """Bland-Altman with repeated windows per participant (one-way random effects; Bland & Altman 2007)."""
    x = d[[pid]].assign(diff=d[a] - d[b]).dropna()
    if x.empty:
        return {}
    bias = float(x["diff"].mean())
    naive = float(x["diff"].std(ddof=1))
    g = x.groupby(pid)["diff"]
    k, N = g.ngroups, len(x)
    if k < 2:
        return {"bias": bias, "loa_lo": bias - 1.96 * naive, "loa_hi": bias + 1.96 * naive, "n_windows": N, "n_participants": k, "sd": naive}
    ni = g.size().to_numpy()
    msb = float((ni * (g.mean() - x["diff"].mean()) ** 2).sum() / (k - 1))
    msw = float(((x["diff"] - g.transform("mean")) ** 2).sum() / max(N - k, 1))
    n0 = (N - (ni ** 2).sum() / N) / (k - 1)
    s2b = max((msb - msw) / max(n0, 1e-9), 0.0)
    sd = float(np.sqrt(s2b + msw))
    return {"bias": bias, "loa_lo": bias - 1.96 * sd, "loa_hi": bias + 1.96 * sd, "sd": sd, "sd_naive": naive,
            "n_windows": int(N), "n_participants": int(k)}


def boot(values_by_pid: dict, fn, n=2000, seed=0):
    keys = list(values_by_pid)
    rng = np.random.default_rng(seed)
    est = fn(keys)
    vals = []
    for _ in range(n):
        try:
            v = fn([keys[i] for i in rng.integers(0, len(keys), len(keys))])
        except Exception:  # noqa: BLE001
            v = np.nan
        vals.append(v)
    vals = np.array(vals, float)
    vals = vals[np.isfinite(vals)]
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals.size > 10 else (np.nan, np.nan))
    return float(est), float(lo), float(hi)


def strata_fallback(tones: dict, cfg, used: list, log=print) -> str:
    """Pre-specified fallback: if strata come from ITA° and the lighter or darker group of the analysed
    participants has fewer than 2 people while Monk ratings exist, strata are taken from Monk ratings
    (strata.monk_cuts) for every participant, and the report says so."""
    if cfg.strata.by != "ita":
        return f"strata from {cfg.strata.by}"
    st = [tones[p]["stratum"] for p in used if p in tones]
    n_ind = sum(1 for p in used if p in tones and str(tones[p].get("stratum_source", "")).startswith("monk"))
    extra = f"; {n_ind} participant(s) without a usable ITA° placed by Monk rating" if n_ind else ""
    if min(st.count("lighter"), st.count("darker")) >= 2:
        return "strata from ITA°" + extra
    if all(np.isfinite(tones[p].get("monk", np.nan)) for p in used if p in tones) and used:
        for p in used:
            tones[p]["stratum"] = stratum_from_monk(tones[p]["monk"], tuple(cfg.strata.monk_cuts))
            tones[p]["stratum_source"] = "monk (pre-specified fallback)"
        log("[pilot] fewer than 2 participants in the lighter or darker ITA° group: strata from Monk ratings (pre-specified fallback)")
        return "strata from Monk ratings (pre-specified fallback: an ITA° group had fewer than 2 participants)"
    return "strata from ITA° (a group has fewer than 2 participants; no Monk ratings for the fallback)" + extra


def score_failures(df: pd.DataFrame) -> pd.DataFrame:
    """A window where a method returned no estimate is a failure, not a missing value: dropping it
    would favour methods (and skin tones) that fail most. It is scored against the participant's median
    estimate from the same method and level (or 75 bpm if there is none) and flagged."""
    df = df.copy()
    df["failed"] = df.hr_cam_bpm.isna() & df.hr_ref_bpm.notna()
    med = df.groupby(["participant_id", "method", "level"]).hr_cam_bpm.transform("median").fillna(75.0)
    est = df.hr_cam_bpm.where(~df.failed, med)
    df["abs_err"] = (est - df.hr_ref_bpm).abs()
    return df


def recalibrate_gate_loo(df: pd.DataFrame, cfg, meta: dict) -> pd.DataFrame:
    """pilot.role = calibrate: each participant's SAMA intervals use a pooled gate calibrated on the
    other pilot participants' lossless-webcam windows (one window per participant per draw)."""
    from .conformal import MondrianGate
    df = df.copy()
    s = df[(df.method == "SAMA")]
    cal = s[(s.level == LOSSLESS) & (s.stream == "webcam") & s.hr_ref_bpm.notna() & s.hr_cam_bpm.notna()]
    for pid in s.participant_id.unique():
        c = cal[cal.participant_id != pid]
        g = MondrianGate(alpha=cfg.gate.alpha, halfwidth_max=cfg.gate.halfwidth_max_bpm, draws=int(cfg.gate.draws))
        g.fit(c.hr_cam_bpm, c.hr_ref_bpm, c.sigma, np.array(["all"] * len(c)), c.participant_id.to_numpy(), seed=0)
        idx = s.index[s.participant_id == pid]
        lo, hi, hw, shown = g.decide(df.loc[idx, "hr_cam_bpm"].to_numpy(), df.loc[idx, "sigma"].to_numpy(), np.array(["all"] * len(idx)))
        df.loc[idx, "lo"], df.loc[idx, "hi"] = lo, hi
        df.loc[idx, "shown"] = shown & (df.loc[idx, "motion_p90"] <= cfg.quality.motion_abstain).to_numpy()
        df.loc[idx, "gate_group"] = "pilot_loo"
    return df


def results(cfg, out_dir=None, log=print) -> dict:
    """Aggregate every prepared session into the section-5 results, figures and tables."""
    out = Path(out_dir or cfg.paths.outputs)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg.paths.pilot_root)
    measured = load_measured(cfg.ita.measured_file) if cfg.ita.source == "measured" else {}
    ratings_path = root / "monk_ratings.csv"
    monk_cons, agree = {}, {"note": "no monk_ratings.csv"}
    cons_df = pd.DataFrame()
    if ratings_path.exists():
        cons_df = consensus(load_ratings(ratings_path), int(cfg.raters.n), cfg.raters.consensus)
        monk_cons = dict(zip(cons_df.participant_id, cons_df.monk))
        agree = rater_agreement(cons_df, int(cfg.raters.n))
    ita_model = None
    if cfg.ita.source == "predicted":
        from .ita_model import ItaModel
        ita_model = ItaModel.load(cfg.ita.model_file)
    model, meta = (None, None)
    if "SAMA" in cfg.signal.methods:
        from .realtrain import load_model
        model, meta = load_model(cfg.training.bundle)
        if model is None:
            log(f"[pilot] no SAMA bundle at {cfg.training.bundle}: SAMA rows skipped (run `python -m sama train` first)")
    if meta and "PILOT" in [str(x).upper() for x in meta.get("trained_on", [])]:
        raise SystemExit(f"the SAMA bundle {cfg.training.bundle} was trained on PILOT data; it cannot be tested on the same pilot. "
                         "Train on public data only (training.datasets) or drop SAMA from signal.methods.")
    qcs, syncs, tones, todo = [], [], {}, []
    for sd in list_sessions(root):
        qc_p = sd / "derived/qc.json"
        if not qc_p.exists():
            log(f"[pilot] {sd.name}: not prepared, skipped (run `python -m sama prepare`)")
            continue
        qc = json.loads(qc_p.read_text())
        qcs.append(qc)
        sy = json.loads((sd / "derived/sync.json").read_text()) if (sd / "derived/sync.json").exists() else {}
        syncs.append({"participant_id": qc["participant_id"], **(sy.get("webcam") or {})})
        tones[qc["participant_id"]] = _participant_tone(qc["participant_id"], cfg, sd / "derived", measured, monk_cons, ita_model)
        if not qc["usable"]:
            log(f"[pilot] {qc['participant_id']}: excluded by QC ({'; '.join(qc['reasons'])})")
            continue
        todo.append(sd)
    strata_note = strata_fallback(tones, cfg, [load_session(sd).pid for sd in todo], log)
    rows, hrv_rows = [], []
    for sd in todo:
        r, h = analyse(sd, cfg, model, meta or {}, tones[load_session(sd).pid], log)
        rows += r
        hrv_rows += h
    if not rows:
        raise SystemExit("no usable pilot windows: record and prepare sessions first (see pilot/PROTOCOL.md)")
    df = pd.DataFrame(rows)
    df = score_failures(df)
    if (df.method == "SAMA").any() and cfg.pilot.role == "calibrate":
        df = recalibrate_gate_loo(df, cfg, meta or {})
    df["provenance"] = cfg.project.provenance_label
    df.to_csv(out / "pilot_windows.csv", index=False)
    hv = pd.DataFrame(hrv_rows)
    hv.to_csv(out / "pilot_hrv_windows.csv", index=False)
    tone_df = pd.DataFrame([{"participant_id": k, **v} for k, v in tones.items()])
    tone_df.to_csv(out / "pilot_skin_tone.csv", index=False)
    summary = summarise_pilot(df, hv, tone_df, qcs, syncs, cons_df, agree, cfg, meta)
    summary["provenance"] = cfg.project.provenance_label
    summary["config_notes"] = cfg.get("_notes", [])
    summary["participants"]["strata_note"] = strata_note
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
    figures(df, hv, tone_df, summary, cfg, out, meta)
    write_section5(summary, cfg, out)
    from .config import snapshot
    snapshot(cfg, out)
    log(f"[pilot] wrote {out}")
    return summary


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def summarise_pilot(df, hv, tone_df, qcs, syncs, cons_df, agree, cfg, meta) -> dict:
    prim = cfg.signal.primary_method
    nb = int(cfg.pilot.bootstrap)
    web = df[df.stream == "webcam"]
    loss = web[web.codec == "lossless"]
    S = {}
    # 1. feasibility
    sp = [s.get("spread_ms") for s in syncs if s.get("spread_ms") is not None and np.isfinite(s.get("spread_ms", np.nan))]
    S["feasibility"] = {
        "n_recorded": len(qcs), "n_usable": int(sum(q["usable"] for q in qcs)),
        "excluded": {q["participant_id"]: q["reasons"] for q in qcs if not q["usable"]},
        "sync_spread_ms_median": float(np.median(sp)) if sp else None, "sync_spread_ms_max": float(np.max(sp)) if sp else None,
        "sync_method": sorted({s.get("method") for s in syncs if s.get("method")}),
        "session_minutes_median": float(np.nanmedian([q.get("session_minutes", np.nan) for q in qcs])) if qcs else None,
        "recording_minutes_median": float(np.median([q["duration_s"] / 60 for q in qcs])) if qcs else None,
        "face_fraction_median": float(np.median([q["face_fraction"] for q in qcs])) if qcs else None,
        "n_with_phone": int(df[df.stream == "phone"].participant_id.nunique())}
    # participants and skin tone
    used = sorted(loss.participant_id.unique())
    tdf = tone_df[tone_df.participant_id.isin(used)]
    ita = tdf.ita_deg.astype(float)
    S["participants"] = {"n": len(used), "ita_source": cfg.ita.source, "ita_min": float(ita.min()) if ita.notna().any() else None,
                         "ita_max": float(ita.max()) if ita.notna().any() else None,
                         "strata_by": cfg.strata.by, "per_stratum": tdf.stratum.value_counts().to_dict(),
                         "monk_min": float(tdf.monk.min()) if tdf.monk.notna().any() else None,
                         "monk_max": float(tdf.monk.max()) if tdf.monk.notna().any() else None,
                         "ita_flags": {r.participant_id: r.ita_camera_flags for r in tdf.itertuples() if r.ita_camera_flags}}
    # 2. skin tone: slope of per-participant MAE (primary method, lossless webcam) on ITA
    per = loss.groupby(["method", "participant_id"]).agg(mae=("abs_err", "mean"), ita=("ita_deg", "first"),
                                                         monk=("monk", "first"), stratum=("stratum", "first")).reset_index()
    per.to_csv(Path(cfg.paths.outputs) / "pilot_participant_mae.csv", index=False)
    S["skin_tone"] = {}
    for m in per.method.unique():
        d = per[(per.method == m) & per.ita.notna() & per.mae.notna()].set_index("participant_id")
        res = {"n": int(len(d)), "mae_mean": float(d.mae.mean()) if len(d) else None,
               "between_person_sd": float(d.mae.std(ddof=1)) if len(d) > 1 else None}
        if len(d) >= 4 and d.ita.nunique() > 2:
            x, lm = d.ita.astype(float), np.log(d.mae.clip(lower=0.05))
            fn = lambda ks, d=d: -10 * np.polyfit(d.loc[ks].ita.astype(float), d.loc[ks].mae, 1)[0] if len(set(ks)) > 2 else np.nan  # noqa: E731
            est, lo, hi = boot({k: None for k in d.index}, fn, nb, 1)
            # H1's estimand: ratio of errors per 10° fall in ITA° from the log-MAE slope
            fr = lambda ks: float(np.exp(-10 * np.polyfit(x.loc[ks], lm.loc[ks], 1)[0])) if len(set(ks)) > 2 else np.nan  # noqa: E731
            r_est, r_lo, r_hi = boot({k: None for k in d.index}, fr, nb, 3)
            k1, k0 = np.polyfit(x, lm, 1)
            tau = float(np.std(lm - (k1 * x + k0), ddof=2))
            res.update({"rise_per_10deg_fall": est, "lo95": lo, "hi95": hi, "ratio_per_10deg_fall": r_est,
                        "ratio_lo95": r_lo, "ratio_hi95": r_hi, "log_mae_residual_sd": tau})
        dm = per[(per.method == m) & per.monk.notna() & per.mae.notna()]
        if len(dm) >= 4 and dm.monk.nunique() > 2:
            from scipy import stats as sst
            res["spearman_mae_monk"] = float(sst.spearmanr(dm.monk, dm.mae).statistic)
        S["skin_tone"][m] = res
    # 3. compression: penalty at the quoted bitrate, per stratum (participant means)
    kb = int(cfg.pilot.compression_kbps)
    S["compression"] = {}
    for m in web.method.unique():
        d = web[web.method == m]
        lv = d.groupby(["participant_id", "codec", "bitrate_kbps"]).abs_err.mean().unstack(["codec", "bitrate_kbps"])
        if ("lossless", 0) not in lv.columns:
            continue
        res = {}
        for codec in ("h264", "vp8"):
            if (codec, kb) not in lv.columns:
                continue
            pen = (lv[(codec, kb)] - lv[("lossless", 0)]).dropna()
            st = tdf.set_index("participant_id").stratum.reindex(pen.index)
            byst = {}
            for sname in ("lighter", "middle", "darker", "all"):
                p = pen[st == sname] if sname != "all" else pen
                if len(p) >= 2:
                    est, lo, hi = boot({k: None for k in p.index}, lambda ks, p=p: float(p.loc[ks].mean()), nb, 2)
                    byst[sname] = {"n": int(len(p)), "penalty": est, "lo95": lo, "hi95": hi}
                elif len(p) == 1:
                    byst[sname] = {"n": 1, "penalty": float(p.iloc[0]), "lo95": None, "hi95": None}
            itas = tdf.set_index("participant_id").ita_deg.reindex(pen.index).astype(float)
            ok = itas.notna()
            res[codec] = {"by_stratum": byst, "within_person_sd": float(pen.std(ddof=1)) if len(pen) > 1 else None,
                          "residual_sd_on_ita": float(np.std(pen[ok] - np.polyval(np.polyfit(itas[ok], pen[ok], 1), itas[ok]), ddof=2))
                          if ok.sum() > 3 else None,
                          "mae_by_bitrate": {f"{c}_{b}": float(lv[(c, b)].mean()) for c, b in lv.columns}}
        S["compression"][m] = res
    # 4. agreement (lossless webcam)
    S["agreement"] = {m: ba_clustered(loss[loss.method == m]) for m in loss.method.unique()}
    S["accuracy"] = {}
    for m in web.method.unique():
        for lvl, d0 in web[web.method == m].groupby("level"):
            d0 = d0.dropna(subset=["hr_ref_bpm"])
            d = d0.dropna(subset=["hr_cam_bpm"])
            if d.empty:
                continue
            e = d.hr_cam_bpm - d.hr_ref_bpm
            tol = np.maximum(5, 0.1 * d.hr_ref_bpm)
            S["accuracy"].setdefault(m, {})[lvl] = {"mae": float(d0.abs_err.mean()), "failure_rate": float(d0.failed.mean()),
                                                   "rmse": float(np.sqrt((e ** 2).mean())),
                                                   "r": float(np.corrcoef(d.hr_cam_bpm, d.hr_ref_bpm)[0, 1]) if len(d) > 2 else None,
                                                   "within_aami": float((e.abs() <= tol).mean()), "n_windows": int(len(d)),
                                                   "n_participants": int(d.participant_id.nunique())}
    fl = web.dropna(subset=["hr_ref_bpm"]).groupby(["method", "level", "stratum"]).failed.mean()
    S["failures"] = {f"{m}|{lv}|{st}": float(v) for (m, lv, st), v in fl.items() if v > 0}
    # SAMA: gate behaviour per stratum on the lossless webcam and at the quoted bitrate
    if (df.method == "SAMA").any():
        S["sama"] = {"bundle": (meta or {}).get("version"), "trained_on": (meta or {}).get("trained_on"),
                     "pilot_role": cfg.pilot.role, "gate_type": (meta or {}).get("gate_type"), "film": (meta or {}).get("film"),
                     "cv_on_training_data": (meta or {}).get("cv", {}).get("all")}
        S["sama"]["trained_on_text"] = ", ".join((meta or {}).get("trained_on") or [])
        key = ["participant_id", "stream", "level", "block", "light", "t0"]
        for lvl in ("lossless", f"h264_{kb}"):
            d = web[(web.method == "SAMA") & (web.level == lvl)].dropna(subset=["hr_ref_bpm"])
            pos = web[(web.method == prim) & (web.level == lvl)].dropna(subset=["hr_ref_bpm"])
            # the comparison uses exactly the same windows for both methods
            d = d.merge(pos[key + ["abs_err"]].rename(columns={"abs_err": "abs_err_primary"}), on=key, how="inner")
            out = {}
            for sname, g in list(d.groupby("stratum")) + [("all", d)]:
                cov = ((g.hr_ref_bpm >= g.lo) & (g.hr_ref_bpm <= g.hi))
                sh = g[g.shown]
                out[sname] = {"n_participants": int(g.participant_id.nunique()), "n_windows": int(len(g)),
                              "mae_sama_all": float(g.abs_err.mean()), "mae_primary_all": float(g.abs_err_primary.mean()),
                              "abstention": float(1 - g.shown.mean()), "coverage_all": float(cov.mean()),
                              "coverage_shown": float(cov[g.shown].mean()) if g.shown.any() else None,
                              "mae_shown": float(sh.abs_err.mean()) if len(sh) else None,
                              "useful_yield": float((g.shown & (g.abs_err <= 5)).mean())}
            S["sama"][lvl] = out
    # HRV agreement (exploratory): per participant mean ln RMSSD in rest blocks, still windows
    S["hrv"] = {}
    if not hv.empty:
        h = hv[(hv.stream == "webcam") & hv.ref_rmssd.notna() & hv.cam_rmssd.notna()]
        h = h[h.motion_p90 <= cfg.quality.motion_warn] if cfg.live.require_still_for_hrv else h
        for m, g in h.groupby("method"):
            rest = g[g.block.str.contains("rest", case=False)]
            use = rest if rest.participant_id.nunique() >= 3 else g
            pp = use.groupby("participant_id")[["ref_rmssd", "cam_rmssd"]].mean()
            res = {"n_windows": int(len(use)), "n_participants": int(len(pp)), "blocks": "rest" if use is rest else "all",
                   "ba_rmssd_ms": ba_clustered(use, "cam_rmssd", "ref_rmssd")}
            if len(pp) >= 3:
                from .stats import icc21
                res["icc21_ln_rmssd"] = icc21(np.log(pp[["cam_rmssd", "ref_rmssd"]].to_numpy()))
            S["hrv"][m] = res
        S["hrv"]["reference_note"] = "fingertip oximeter: pulse-rate variability (PRV), not ECG HRV" if cfg.reference.type.startswith("oximeter") else cfg.reference.type
    # skin-tone measurement checks
    S["ita_vs_monk"] = ita_vs_monk(tdf.ita_camera.astype(float), tdf.monk.astype(float), nb) if tdf.monk.notna().any() else {"note": "no Monk ratings"}
    S["raters"] = dict(agree)
    nr = int(S["raters"].get("n_raters", 0))
    S["raters"]["rater_text"] = "nobody (no Monk ratings were found)" if nr == 0 else "one trained rater" if nr < 2 else (
        "two trained raters (consensus: " + ("their mean" if cfg.raters.consensus == "mean" else "rater 1") + ")")
    # power re-check with the pilot's spread
    S["power_recheck"] = power_recheck(S, cfg)
    return S


def power_recheck(S: dict, cfg) -> dict:
    """H1 with the proposal's own model (sama/power_check.py: log MAE, multiplicative spread) using the
    pilot's residual SD of log MAE on ITA° and its upper 95 % bound; H2 with the pilot's residual SD of
    the penalty on ITA°."""
    from .power_check import h1_power, h2_power, sd_ci
    prim = cfg.signal.primary_method
    out = {"assumed": {"H1_log_mae_sd": 0.55, "H2_penalty_sd": 1.0}}
    st = S["skin_tone"].get(prim, {})
    tau, n1 = st.get("log_mae_residual_sd"), st.get("n", 0)
    c = S["compression"].get(prim, {}).get(cfg.pilot.figure_codec, {})
    sd2 = c.get("residual_sd_on_ita") or c.get("within_person_sd")
    n2 = c.get("by_stratum", {}).get("all", {}).get("n", 0) if c else 0
    if tau:
        lo, hi = sd_ci(tau, n1 - 1)
        out["H1"] = {"pilot_log_sd": tau, "sd_ci95": [lo, hi], "n": n1,
                     "power_36_at_pilot_sd": h1_power(36, tau), "power_45_at_pilot_sd": h1_power(45, tau),
                     "power_45_at_upper_sd": h1_power(45, hi) if np.isfinite(hi) else None,
                     "power_55_at_upper_sd": h1_power(55, hi) if np.isfinite(hi) else None}
        h = out["H1"]
        need = next((n for n in (45, 55, 70, 90, 120, 150) if h1_power(n, tau, sims=1000) >= 0.80), None)
        h["n_for_080_at_pilot_sd"] = need
        if h["power_45_at_pilot_sd"] >= 0.80:
            h["verdict"] = "This is consistent with the planned sample."
        else:
            h["verdict"] = ("This is below 0.80, but the pilot's spread is imprecise (see its interval), pools both lights and all blocks, "
                            "and rests on a finger reference that SAMA-150 replaces with ECG. The month-10 audit re-estimates it on "
                            "ECG-referenced SAMA-150 data; if power at 45 is still below 0.80, up to 55 per stratum are recruited, and "
                            "H1 is in any case reported as an estimated ratio with its confidence interval"
                            + (f" (at the pilot's spread, 0.80 would need about {need} per stratum)." if need else "."))
    if sd2:
        lo, hi = sd_ci(sd2, n2 - 1)
        out["H2"] = {"pilot_sd": sd2, "sd_ci95": [lo, hi], "n": n2,
                     "power_36_at_pilot_sd": h2_power(36, sd2), "power_45_at_pilot_sd": h2_power(45, sd2),
                     "power_45_at_upper_sd": h2_power(45, hi) if np.isfinite(hi) else None,
                     "power_55_at_upper_sd": h2_power(55, hi) if np.isfinite(hi) else None}
    out["note"] = ("Spreads from a pilot of this size are imprecise, so the upper 95 % bound of each SD is also used. "
                   "The pilot's spread pools both lights and all blocks and uses a finger-PPG reference, so it is likely "
                   "larger than SAMA-150's. If power at 45 falls below 0.80, the safeguard (up to 55 recruited per stratum) applies.")
    return out


# ---------------------------------------------------------------- figures and text ---------
def figures(df, hv, tone_df, S, cfg, out: Path, meta):
    sys.path.insert(0, str(ROOT / "scripts"))
    import pilot_figure as PF
    codec = cfg.pilot.figure_codec
    fig_df = df[(df.stream == "webcam") & df.codec.isin(["lossless", codec])][
        ["participant_id", "ita_deg", "method", "bitrate_kbps", "hr_ref_bpm", "hr_cam_bpm"]].dropna(subset=["hr_ref_bpm", "hr_cam_bpm"])
    if fig_df.ita_deg.isna().all():
        fig_df = fig_df.assign(ita_deg=fig_df.participant_id.map(dict(zip(tone_df.participant_id, tone_df.ita_camera))))
    fig_df.to_csv(out / "pilot_hr.csv", index=False)
    show = [m for m in ("POS", "CHROM", "SAMA") if m in set(fig_df.method)]
    fd = fig_df[fig_df.method.isin(show)]
    fd.to_csv(out / "pilot_hr_figure.csv", index=False)
    ref = cfg.reference.label
    for m in show:
        if fd.ita_deg.notna().any():
            PF.make(str(out / "pilot_hr_figure.csv"), m, str(out / f"figure6_{m}.png"), ref)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # SAMA gate per stratum
    if "sama" in S:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        st = [k for k in ("lighter", "middle", "darker", "all") if k in S["sama"].get("lossless", {})]
        x = np.arange(len(st))
        for i, (lvl, lab) in enumerate((("lossless", "lossless"), (f"h264_{cfg.pilot.compression_kbps}", f"H.264 {cfg.pilot.compression_kbps} kbps"))):
            v = S["sama"].get(lvl, {})
            ax[0].bar(x + i * 0.38, [v.get(k, {}).get("abstention", np.nan) * 100 for k in st], 0.36, label=lab)
            ax[1].bar(x + i * 0.38, [(v.get(k, {}).get("coverage_shown") or np.nan) * 100 for k in st], 0.36, label=lab)
        for a, t in zip(ax, ("Abstention (% of windows)", "Coverage of shown 90 % intervals (%)")):
            a.set_xticks(x + 0.19)
            a.set_xticklabels(st)
            a.set_title(t, loc="left", fontsize=11)
            a.legend(frameon=False, fontsize=8)
        ax[1].axhline(90, color="k", lw=0.8, ls="--")
        fig.suptitle(f"SAMA reliability gate on the pilot ({cfg.project.provenance_label}; model trained on {', '.join((meta or {}).get('trained_on', []))})",
                     x=0.01, ha="left", fontsize=10)
        fig.savefig(out / "figure_sama_gate.png", dpi=180)
        plt.close(fig)
    # camera ITA vs Monk
    t = tone_df.dropna(subset=["ita_camera"])
    if len(t) and t.monk.notna().any():
        fig, ax = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
        ax.scatter(t.monk, t.ita_camera, color="#2F4BA0")
        for r in t.itertuples():
            ax.annotate(r.participant_id, (r.monk, r.ita_camera), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel("Monk Skin Tone (rater consensus)")
        ax.set_ylabel("Camera ITA° (white-referenced)")
        rv = S.get("ita_vs_monk", {})
        ax.set_title(f"Camera ITA° vs Monk: Spearman {rv.get('spearman_rho', float('nan')):.2f}", loc="left", fontsize=10)
        fig.savefig(out / "figure_ita_vs_monk.png", dpi=180)
        plt.close(fig)
    # HRV Bland-Altman (POS)
    if not hv.empty and hv.ref_rmssd.notna().any():
        h = hv[(hv.method == "POS") & hv.ref_rmssd.notna() & hv.cam_rmssd.notna()]
        if len(h):
            fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
            m, dd = (h.cam_rmssd + h.ref_rmssd) / 2, h.cam_rmssd - h.ref_rmssd
            ax.scatter(m, dd, s=14, color="#0B6E6E")
            b = S["hrv"].get("POS", {}).get("ba_rmssd_ms", {})
            for y in (b.get("bias"), b.get("loa_lo"), b.get("loa_hi")):
                if y is not None:
                    ax.axhline(y, color="#46535F", lw=1, ls="-" if y == b.get("bias") else "--")
            ax.set_xlabel("Mean RMSSD (ms)")
            ax.set_ylabel("Camera − oximeter RMSSD (ms)")
            ax.set_title("HRV agreement, 60 s windows (exploratory)", loc="left", fontsize=10)
            fig.savefig(out / "figure_hrv_ba.png", dpi=180)
            plt.close(fig)


def _f(x, fmt=".1f", missing="[not measured]"):
    try:
        if x is None or not np.isfinite(float(x)):
            return missing
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return missing


def write_section5(S: dict, cfg, out: Path):
    """A plain-text version of the section 5 pilot paragraph for reading and pasting. The proposal
    itself is filled by proposal/build_proposal.py from pilot_summary.json."""
    prim = cfg.signal.primary_method
    F, P = S["feasibility"], S["participants"]
    st = S["skin_tone"].get(prim, {})
    c = S["compression"].get(prim, {}).get(cfg.pilot.figure_codec, {}).get("by_stratum", {})
    ag = S["agreement"].get(prim, {})
    kb = cfg.pilot.compression_kbps
    iv = S.get("ita_vs_monk", {})
    ita_label = {"camera": "camera-estimated ITA° (grey-card referenced; approximate)", "measured": "colorimeter ITA°",
                 "predicted": "model-predicted ITA°", "monk": "Monk Skin Tone", "none": "skin tone (not measured)"}[cfg.ita.source]
    phone = " and a phone" if F.get("n_with_phone") else ""

    def grp(name):
        g = c.get(name, {})
        return (f"{_f(g.get('penalty'), '+.2f')} bpm in the {name} group (n = {g.get('n', 0)}; 95 % CI "
                f"{_f(g.get('lo95'), '+.2f')} to {_f(g.get('hi95'), '+.2f')})")
    lines = [
        "## Pilot results (auto-generated from pilot_summary.json; the proposal is filled by proposal/build_proposal.py)", "",
        f"{P['n']} volunteers spanning {ita_label} {_f(P.get('ita_min'))}° to {_f(P.get('ita_max'))}° "
        f"(Monk {_f(P.get('monk_min'), 'g')}–{_f(P.get('monk_max'), 'g')}) were recorded with a webcam saved losslessly{phone} against a "
        f"{cfg.reference.label}, under two light levels; {P.get('strata_note', '')}.", "",
        f"1. **Feasibility.** The camera-to-reference alignment varied by {_f(F.get('sync_spread_ms_median'), '.0f')} ms between 30-second windows "
        f"(robust SD, median over sessions; worst session {_f(F.get('sync_spread_ms_max'), '.0f')} ms); a session took "
        f"{_f(F.get('session_minutes_median'), '.0f')} minutes; {F['n_usable']} of {F['n_recorded']} recordings passed the automatic quality check.",
        f"2. **Skin tone.** Per 10° fall in ITA°, {prim} error changed by a factor of {_f(st.get('ratio_per_10deg_fall'), '.2f')} "
        f"(95 % CI {_f(st.get('ratio_lo95'), '.2f')} to {_f(st.get('ratio_hi95'), '.2f')}), i.e. {_f(st.get('rise_per_10deg_fall'), '+.2f')} bpm "
        f"(n = {st.get('n', 0)}). Camera ITA° vs Monk: Spearman ρ = {_f(iv.get('spearman_rho'), '.2f')} "
        f"(95 % CI {_f(iv.get('lo95'), '.2f')} to {_f(iv.get('hi95'), '.2f')}).",
        f"3. **Compression.** At {kb} kbps, {prim} error changed by {grp('darker')} and by {grp('lighter')}.",
        f"4. **Agreement.** Camera minus reference heart rate: bias {_f(ag.get('bias'), '+.2f')}, 95 % limits {_f(ag.get('loa_lo'), '+.1f')} "
        f"to {_f(ag.get('loa_hi'), '+.1f')} beats a minute ({ag.get('n_windows', 0)} windows, {ag.get('n_participants', 0)} participants; "
        f"limits widened for repeated windows).", ""]
    if "sama" in S:
        a = S["sama"].get("lossless", {}).get("all", {})
        lines += [f"5. **SAMA (trained on {S['sama'].get('trained_on_text', '')} only; the pilot is an untouched test set).** "
                  f"Error {_f(a.get('mae_sama_all'), '.2f')} vs {_f(a.get('mae_primary_all'), '.2f')} bpm for {prim} on the same windows; "
                  f"the gate withheld {_f((a.get('abstention') if a.get('abstention') is not None else np.nan) * 100, '.0f')} % of windows and "
                  f"its shown 90 % intervals covered the reference in {_f((a.get('coverage_shown') if a.get('coverage_shown') is not None else np.nan) * 100, '.0f')} %.", ""]
    pr = S["power_recheck"]
    if "H1" in pr:
        lines += [f"Power re-check: residual SD of log {prim} error on ITA° {_f(pr['H1']['pilot_log_sd'], '.2f')} (assumed 0.55; 95 % CI "
                  f"{_f(pr['H1']['sd_ci95'][0], '.2f')}–{_f(pr['H1']['sd_ci95'][1], '.2f')}); H1 power {_f(pr['H1']['power_45_at_pilot_sd'], '.2f')} "
                  f"at 45 per stratum ({_f(pr['H1']['power_45_at_upper_sd'], '.2f')} at the upper bound). {pr['H1'].get('verdict', '')}", ""]
    lines += ["These are feasibility numbers from a small sample, not tests of the hypotheses. Skin tone was "
              + ("measured with a colorimeter." if cfg.ita.source == "measured" else
                 "estimated from video against a grey card, not measured with the colorimeter the main study will use; "
                 "its agreement with Monk ratings is reported alongside.")]
    (out / "section5_filled.md").write_text("\n".join(lines) + "\n")

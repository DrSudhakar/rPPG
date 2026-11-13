"""SAMA research interface (Starlette + uvicorn).

    python -m sama serve                 browser camera: live BPM, HRV, beat dots, warnings
    python -m sama record                lab recorder: lossless webcam video + the same live display,
                                         block markers and box drawing for the pilot protocol

Browser mode: the page computes 30 skin-patch colour means, a motion index, the skin fraction and the
brightness of the face box for each frame and sends only those numbers. No frame, image or video
leaves the browser, and nothing is stored unless a participant id is given (then only the readings).
Recorder mode: video comes from the webcam via ffmpeg on this PC; the preview image is served only to
the local page (bind address 127.0.0.1 by default).
Research use only. Not a medical device. No diagnostic output.
"""
from __future__ import annotations

import csv
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sama import config as C  # noqa: E402
from sama.live import LiveAnalyzer  # noqa: E402

STATIC = Path(__file__).parent / "static"
STATE = {"cfg": None, "mode": "serve", "participant": None, "model": None, "meta": None, "recorder": None, "loaded": False}
SESSIONS: dict[str, dict] = {}
MAX_SESSIONS = 16


def cfg():
    if STATE["cfg"] is None:
        STATE["cfg"] = C.load(os.environ.get("SAMA_CONFIG"))
    return STATE["cfg"]


def model():
    if not STATE["loaded"]:
        from sama.realtrain import load_model
        bundle = os.environ.get("SAMA_MODEL_DIR") or cfg().training.bundle
        try:
            STATE["model"], STATE["meta"] = load_model(ROOT / bundle if not Path(bundle).is_absolute() else bundle)
        except Exception as e:  # noqa: BLE001
            STATE["model"], STATE["meta"] = None, {"error": str(e)}
        STATE["loaded"] = True
    return STATE["model"], STATE["meta"]


def _json(x):
    if isinstance(x, dict):
        return {k: _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


async def index(request: Request):
    return FileResponse(STATIC / "index.html")


async def healthz(request: Request):
    m, _ = model()
    return JSONResponse({"ok": True, "model_loaded": m is not None, "mode": STATE["mode"]})


async def info(request: Request):
    c = cfg()
    m, meta = model()
    meta = meta or {}
    return JSONResponse(_json({
        "name": "SAMA research interface", "research_only": True, "medical_device": False, "mode": STATE["mode"],
        "participant": STATE["participant"],
        "model": {"loaded": m is not None, "version": meta.get("version"), "trained_on": meta.get("trained_on"),
                  "gate_type": meta.get("gate_type"), "film": meta.get("film"), "provenance": meta.get("provenance"),
                  "cv": (meta.get("cv") or {}).get("all"), "error": meta.get("error")},
        "knobs": {"method": c.live.method, "gate": c.live.gate, "show_hrv": c.live.show_hrv, "show_beats": c.live.show_beats,
                  "motion_warn": c.quality.motion_warn, "motion_abstain": c.quality.motion_abstain,
                  "hr_window_s": c.live.hr_window_s, "update_s": c.live.update_s, "require_still_for_hrv": c.live.require_still_for_hrv},
        "face_box": list(c.face.fixed_box), "skin_rule": "YCbCr: 73<=Cb<=132, 130<=Cr<=180, Y>15",
        "blocks": list(c.pilot.blocks), "lights": list(c.pilot.lights), "display_s": c.live.display_s,
        "white_reflectance": c.ita.camera.white_reflectance, "save_signals": c.privacy.save_signals,
        "provenance_label": c.project.provenance_label}))


# ------------------------------------------------------------------ browser camera mode ----
async def session_new(request: Request):
    body = await request.json() if (await request.body()) else {}
    now = time.time()
    for k in [k for k, v in SESSIONS.items() if now - v["seen"] > 900]:
        _close(k)
    if len(SESSIONS) >= MAX_SESSIONS:
        return JSONResponse({"error": "too many sessions"}, status_code=429)
    c = cfg()
    m, meta = model()
    ita = float(body["ita"]) if body.get("ita") not in (None, "") else float("nan")
    from sama.ita_camera import stratum_from
    an = LiveAnalyzer(c, m, meta, ita, stratum_from(ita, tuple(c.strata.ita_cuts)) if np.isfinite(ita) else "all")
    an.set_knobs(**{k: v for k, v in (body.get("knobs") or {}).items()})
    sid = uuid.uuid4().hex
    rec = {"an": an, "seen": now, "fh": None, "writer": None}
    pid = "".join(ch for ch in str(body.get("participant_id") or "") if ch.isalnum() or ch in "-_")
    if pid and c.privacy.save_signals:
        d = Path(c.paths.pilot_root) / "browser_sessions" / pid
        d.mkdir(parents=True, exist_ok=True)
        rec["fh"] = open(d / f"live_readings_{time.strftime('%Y%m%d-%H%M%S')}.csv", "w", newline="")
        rec["writer"] = csv.writer(rec["fh"])
        rec["writer"].writerow(["wall_time", "t", "shown", "reason", "hr", "lo", "hi", "rmssd", "sdnn", "motion", "face", "light", "fps"])
    SESSIONS[sid] = rec
    return JSONResponse({"session": sid, "saving": bool(rec["fh"])})


def _close(sid):
    rec = SESSIONS.pop(sid, None)
    if rec and rec.get("fh"):
        rec["fh"].close()


async def session_frames(request: Request):
    sid = request.path_params["sid"]
    if sid not in SESSIONS:
        return JSONResponse({"error": "unknown session"}, status_code=404)
    rec = SESSIONS[sid]
    rec["seen"] = time.time()
    an: LiveAnalyzer = rec["an"]
    body = await request.json()
    state = None
    for fr in body.get("frames", [])[:900]:
        p = fr.get("patches")
        an.push(float(fr["t"]), None if p is None else np.asarray(p, float).reshape(30, 3), fr.get("pix"),
                float(fr.get("motion", 0.0)), float(fr.get("skin", 1.0)), fr.get("bright"))
        if an.due(float(fr["t"])):
            state = an.state(float(fr["t"]))
            if rec["writer"]:
                h, q = state.get("hrv") or {}, state.get("quality") or {}
                rec["writer"].writerow([f"{time.time():.3f}", f"{state['t']:.3f}", state.get("shown"), state.get("reason"), state.get("hr"),
                                        state.get("lo"), state.get("hi"), h.get("rmssd"), h.get("sdnn"), q.get("motion"), q.get("face"),
                                        q.get("light"), q.get("fps")])
    return JSONResponse(_json({"state": state}))


async def session_knobs(request: Request):
    sid = request.path_params["sid"]
    if sid not in SESSIONS:
        return JSONResponse({"error": "unknown session"}, status_code=404)
    SESSIONS[sid]["an"].set_knobs(**(await request.json()))
    return JSONResponse({"knobs": SESSIONS[sid]["an"].knobs})


async def session_end(request: Request):
    _close(request.path_params["sid"])
    return JSONResponse({"ended": True})


# ------------------------------------------------------------------ lab recorder mode ------
def recorder():
    if STATE["recorder"] is None:
        from sama.recorder import LabRecorder
        m, meta = model()
        STATE["recorder"] = LabRecorder(cfg(), m, meta)
    return STATE["recorder"]


async def lab_start(request: Request):
    if STATE["mode"] != "record":
        return JSONResponse({"error": "start the server with `python -m sama record` for the lab recorder"}, status_code=409)
    b = await request.json()
    try:
        return JSONResponse(recorder().start(b.get("participant_id") or STATE["participant"] or "P00", b.get("operator", ""),
                                             b.get("notes", ""), b.get("input_override")))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


async def lab_stop(request: Request):
    return JSONResponse(recorder().stop())


async def lab_preview_start(request: Request):
    if STATE["mode"] != "record":
        return JSONResponse({"error": "start the server with `python -m sama record` for the lab recorder"}, status_code=409)
    b = {}
    try:
        b = await request.json()
    except Exception:  # noqa: BLE001
        pass
    return JSONResponse(recorder().start_preview(b.get("input_override")))


async def lab_event(request: Request):
    b = await request.json()
    try:
        return JSONResponse(recorder().event(b.get("event", "note"), b.get("block", ""), b.get("light", ""), b.get("note", "")))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


async def lab_boxes(request: Request):
    b = await request.json()
    r = recorder()
    r.set_boxes(b.get("face_box"), b.get("white_box"), b.get("chart_corners"), b.get("white_reflectance"))
    return JSONResponse({"boxes": r.boxes, "white_reflectance": r.white_reflectance})


async def lab_knobs(request: Request):
    r = recorder()
    if r.analyzer:
        r.analyzer.set_knobs(**(await request.json()))
        return JSONResponse({"knobs": r.analyzer.knobs})
    return JSONResponse({"error": "not recording"}, status_code=409)


async def lab_state(request: Request):
    r = recorder()
    return JSONResponse(_json({**(r.latest_state or {}), "boxes": r.boxes, "session_dir": str(r.session_dir) if r.session_dir else None,
                               "preview_error": r.preview_error}))


async def lab_preview(request: Request):
    jpg = recorder().preview_jpeg()
    if jpg is None:
        return Response(status_code=204)
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------ demo without a camera --
async def simulate(request: Request):
    """Educational: one SYNTHETIC participant from SAMA-Sim streamed through the live analyser."""
    body = await request.json()
    from sim import samasim as S
    ita = float(body.get("ita", 15.0))
    part = S.make_participant("demo", S.stratum_of(ita), np.random.default_rng(int(body.get("seed", 1))), ita=ita)
    rec = S.simulate_block(part, body.get("block", "rest"), body.get("light", "standard"), body.get("device", "webcam"),
                           S.SimParams(), codec=body.get("codec", "lossless"), dur_s=float(body.get("seconds", 60)))
    m, meta = model()
    an = LiveAnalyzer(cfg(), m, meta, ita)
    states = []
    for i in range(rec.rgb_patches.shape[0]):
        t = i / rec.fps
        an.push(t, rec.rgb_patches[i], rec.meta["pix"], 0.5, 0.9)
        if an.due(t):
            st = an.state(t)
            beats = rec.beats[(rec.beats >= t - an.knobs["hr_window_s"]) & (rec.beats < t)]
            st["hr_true"] = float(60 / np.mean(np.diff(beats))) if beats.size > 2 else None
            states.append({k: st.get(k) for k in ("t", "shown", "reason", "hr", "lo", "hi", "hr_true", "hrv")})
    return JSONResponse(_json({"provenance": "SYNTHETIC (SAMA-Sim)", "readings": states, "final": an.last_state}))


routes = [
    Route("/", index), Route("/healthz", healthz), Route("/api/v1/info", info),
    Route("/api/v1/session", session_new, methods=["POST"]),
    Route("/api/v1/session/{sid}/frames", session_frames, methods=["POST"]),
    Route("/api/v1/session/{sid}/knobs", session_knobs, methods=["POST"]),
    Route("/api/v1/session/{sid}", session_end, methods=["DELETE"]),
    Route("/api/v1/lab/start", lab_start, methods=["POST"]), Route("/api/v1/lab/stop", lab_stop, methods=["POST"]),
    Route("/api/v1/lab/preview_start", lab_preview_start, methods=["POST"]),
    Route("/api/v1/lab/event", lab_event, methods=["POST"]), Route("/api/v1/lab/boxes", lab_boxes, methods=["POST"]),
    Route("/api/v1/lab/knobs", lab_knobs, methods=["POST"]), Route("/api/v1/lab/state", lab_state),
    Route("/api/v1/lab/preview.jpg", lab_preview),
    Route("/api/v1/simulate", simulate, methods=["POST"]),
    Mount("/static", StaticFiles(directory=STATIC), name="static"),
]
app = Starlette(routes=routes)


def run(c=None, mode="serve", participant=None, host="127.0.0.1", port=8000):
    import uvicorn
    STATE.update(cfg=c or cfg(), mode=mode, participant=participant, loaded=False, recorder=None)
    m, meta = model()
    print(f"SAMA {mode} on http://{host}:{port}  (model: {'loaded ' + str((meta or {}).get('version')) if m else 'none, POS only'})")
    if mode == "record":
        print("Open the page, draw the face and card boxes, enter the participant id and press Start recording.")
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        if STATE["recorder"] is not None and STATE["recorder"].proc is not None:
            STATE["recorder"].stop()


if __name__ == "__main__":
    run()

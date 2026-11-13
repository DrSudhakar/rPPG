"""SYNTHETIC pilot sessions for testing the whole pilot pipeline end to end.

    python -m sama synth-pilot --n 6 --out pilot_data_SYNTHETIC

Each virtual participant comes from SAMA-Sim (sim/samasim.py): beats, pulse, skin optics, light and
camera noise. The patch colours are painted into real video frames (face box + a grey card),
written as lossless FFV1, with an oximeter CSV (60 Hz pleth, PC time stamps with a small clock
offset), session.json, events.csv, participants.csv and Monk ratings, all in the exact layout the
recorder and the forms produce. Every file is labelled SYNTHETIC; nothing here is evidence about
people. Its only purpose is to prove that the pipeline runs and that its numbers can be traced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACE_BOX = (0.30, 0.15, 0.40, 0.60)
CARD_BOX = (0.76, 0.42, 0.12, 0.16)
CARD_REFL = 0.18          # mid-grey card (avoids clipping under auto-exposure on darker faces)


def _srgb_inv(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def _frames(patch_dn: np.ndarray, card_dn: np.ndarray, rng, size=(160, 120), face_box=FACE_BOX, card_box=CARD_BOX, jitter=None):
    """Paint (T, 30, 3) patch colours into frames; per-pixel noise dithers sub-DN pulse changes."""
    from .model_io import LAYOUT
    W, H = size
    T = patch_dn.shape[0]
    bg = np.array([70, 80, 95], float)
    for i in range(T):
        fr = np.empty((H, W, 3), float)
        fr[:] = bg
        dx = 0 if jitter is None else jitter[i]
        x, y, w, h = face_box
        x0, y0, x1, y1 = int((x + dx) * W), int(y * H), int((x + dx + w) * W), int((y + h) * H)
        fr[y0:y1, x0:x1] = patch_dn[i].mean(0)
        bw, bh = x1 - x0, y1 - y0
        for k, (px, py, pw, ph) in enumerate(LAYOUT):
            fr[y0 + int(py * bh):y0 + int((py + ph) * bh), x0 + int(px * bw):x0 + int((px + pw) * bw)] = patch_dn[i, k]
        cx, cy, cw, ch = card_box
        fr[int(cy * H):int((cy + ch) * H), int(cx * W):int((cx + cw) * W)] = card_dn[i]
        fr += rng.normal(0, 2.0, fr.shape)
        yield np.clip(np.round(fr), 0, 255).astype(np.uint8)


def generate(n: int = 6, out_root="pilot_data_SYNTHETIC", seed: int = 7, block_s: float = 45.0, size=(160, 120),
             ita_values=None, log=print) -> Path:
    sys.path.insert(0, str(ROOT))
    from sim import samasim as S
    from .video import write_video
    out = Path(out_root)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    p = S.SimParams()
    itas = ita_values if ita_values is not None else np.linspace(45, -20, n)
    blocks = [("ita_capture", "light1_500lux", "rest", "standard", 20.0)]
    for light, sl in (("light1_500lux", "standard"), ("light2_100lux", "dim")):
        blocks += [("rest", light, "rest", sl, block_s), ("consultation", light, "mock_teleconsultation", sl, block_s),
                   ("arithmetic", light, "mental_arithmetic", sl, block_s)]
    parts, ratings = [], []
    for j, ita in enumerate(itas):
        pid = f"S{j + 1:02d}"
        part = S.make_participant(pid, S.stratum_of(float(ita)), rng, ita=float(ita))
        sd = out / "sessions" / pid
        sd.mkdir(parents=True, exist_ok=True)
        t_wall0 = 1_794_000_000.0 + j * 3600.0
        patches, cards, beats, ev, t = [], [], [], [], 0.0
        R = S.skin_reflectance(part.ita, part.b_star, p.a_star)
        for name, light, kind, sl, dur in blocks:
            rec = S.simulate_block(part, kind, sl, "webcam", p, dur_s=dur, seed_offset=len(ev))
            dn = rec.rgb_patches.astype(float)
            lin = (np.clip(dn, 0.5, 255) / 255.0) ** 2.2                    # camera tone curve of the "webcam" device
            card_lin = lin.mean(1) * (CARD_REFL / R)[None, :]                 # same light and exposure, grey card
            cards.append(255.0 * np.clip(card_lin, 0, 1) ** (1 / 2.2))
            patches.append(dn)
            beats.append(rec.beats[rec.beats < dur] + t)
            ev += [{"wall_time": t_wall0 + t, "event": "block_start", "block": name, "light": light},
                   {"wall_time": t_wall0 + t + dur, "event": "block_end", "block": name, "light": light}]
            t += dur
        P = np.concatenate(patches)
        C = np.concatenate(cards)
        Bt = np.concatenate(beats)
        fps = 30.0
        frames = _frames(P, C, rng, size)
        write_video(sd / "webcam.mkv", frames, fps, "ffv1")
        # oximeter: finger pulse (later than the face), 60 Hz, PC clock offset +0.35 s
        fs = 60.0
        tt = np.arange(0, t, 1 / fs)
        wave = S.pulse_wave(Bt, tt, 0.30)
        wave = 2000 + 400 * (wave - wave.mean()) / (wave.std() + 1e-9) + rng.normal(0, 15, tt.size)
        hr_disp = np.interp(tt, Bt[1:], 60 / np.diff(Bt))
        pd.DataFrame({"time_s": np.round(t_wall0 + tt + 0.35, 4), "pleth": np.round(wave, 1), "hr_bpm": np.round(hr_disp),
                      "spo2": 98}).to_csv(sd / "reference.csv", index=False)
        pd.DataFrame(ev).to_csv(sd / "events.csv", index=False)
        meta = {"participant_id": pid, "date": "2026-11-10", "operator": "SYNTHETIC", "provenance": "SYNTHETIC",
                "video_t0_wall": t_wall0,
                "webcam": {"file": "webcam.mkv", "device": "synthetic", "size": f"{size[0]}x{size[1]}", "fps": fps, "codec": "ffv1"},
                "phone": {"file": "phone.mp4", "model": "", "t0_wall_approx": None},
                "boxes": {"face_box": list(FACE_BOX), "white_box": list(CARD_BOX), "chart_corners": None},
                "white_reference": "card", "white_reflectance": CARD_REFL,
                "lux": {"light1_500lux": 500, "light2_100lux": 100}, "reference": {"device": "synthetic oximeter", "file": "reference.csv"},
                "true_ita_SYNTHETIC": part.ita, "true_lag_SYNTHETIC": 0.30 - part.ptt}
        (sd / "session.json").write_text(json.dumps(meta, indent=1))
        monk_true = float(np.clip(2 + 7 * (50 - part.ita) / 75, 1, 10))
        parts.append({"participant_id": pid, "age": part.age, "sex": part.sex, "monk_self": int(np.clip(round(monk_true + rng.normal(0, 1)), 1, 10)),
                      "fitzpatrick_self": int(np.clip(round(1 + 5 * (50 - part.ita) / 75), 1, 6)), "beard": int(part.beard),
                      "spectacles": int(part.glasses), "bindi": 0, "makeup": 0, "head_covering": 0, "hr_lowering_medication": 0,
                      "parq_any_yes": 0, "consent_tier": "research_only", "notes": "SYNTHETIC"})
        for r in (1, 2):
            ratings.append({"participant_id": pid, "rater_id": f"R{r}", "monk": int(np.clip(round(monk_true + rng.normal(0, 0.6)), 1, 10)),
                            "light": "light1_500lux", "notes": "SYNTHETIC"})
        log(f"[synth] {pid}: ITA {part.ita:.1f}°, {P.shape[0]} frames, {Bt.size} beats")
    pd.DataFrame(parts).to_csv(out / "participants.csv", index=False)
    pd.DataFrame(ratings).to_csv(out / "monk_ratings.csv", index=False)
    (out / "README_SYNTHETIC.txt").write_text("Synthetic pilot sessions generated by `python -m sama synth-pilot`. NOT REAL DATA.\n")
    return out


def generate_public_like(n: int = 12, out_root="data_SYNTHETIC/raw", seed: int = 11, dur_s: float = 60.0,
                         size=(160, 120), log=print) -> Path:
    """SYNTHETIC recordings in the UBFC-rPPG folder layout (vid.avi + ground_truth.txt), skewed to
    lighter skin like the public datasets, to exercise `python -m sama train` without real data."""
    sys.path.insert(0, str(ROOT))
    from sim import samasim as S
    from .video import write_video
    out = Path(out_root) / "UBFC_SYNTHETIC"
    rng = np.random.default_rng(seed)
    p = S.SimParams()
    for j in range(n):
        ita = float(rng.choice([rng.uniform(28, 50), rng.uniform(10, 28), rng.uniform(-20, 10)], p=[0.6, 0.3, 0.1]))
        part = S.make_participant(f"u{j}", S.stratum_of(ita), rng, ita=ita)
        rec = S.simulate_block(part, "rest" if j % 2 else "mock_teleconsultation", "standard", "webcam", p, dur_s=dur_s)
        sd = out / f"subject{j + 1}"
        sd.mkdir(parents=True, exist_ok=True)
        dn = rec.rgb_patches.astype(float)
        write_video(sd / "vid.avi", _frames(dn, dn.mean(1), rng, size), rec.fps, "ffv1")
        tt = np.arange(0, dur_s, 1 / 30.0)
        ppg = S.pulse_wave(rec.beats, tt, 0.30)
        hr = np.interp(tt, rec.beats[1:], 60 / np.diff(rec.beats))
        (sd / "ground_truth.txt").write_text(" ".join(f"{v:.5f}" for v in ppg) + "\n" + " ".join(f"{v:.2f}" for v in hr) + "\n"
                                             + " ".join(f"{v:.5f}" for v in tt) + "\n")
        log(f"[synth-public] subject{j + 1}: ITA {ita:.1f}°")
    (Path(out_root) / "README_SYNTHETIC.txt").write_text("Synthetic UBFC-layout recordings. NOT REAL DATA.\n")
    return Path(out_root)

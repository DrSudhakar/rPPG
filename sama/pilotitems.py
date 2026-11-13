"""Pilot sessions as training data (training.datasets includes PILOT) and the skin-tone model.

pilot_items: each prepared, QC-passed session -> a realtrain.TrainItem on the video clock, with the
oximeter waveform (shifted by the synchronisation lag) as the training target and the protocol
blocks as windows. Use this to train a model for later data (e.g. SAMA-150 pre-training). A model trained
on the pilot must not then be evaluated on the same pilot: `python -m sama pilot` stops if it is.

train_ita: fits the video -> skin-tone model (sama.ita_model) on every participant that has the
requested label (Monk consensus now; colorimeter ITA° once the instrument exists), reports
participant-level cross-validation and saves the model to ita.model_file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ita_model import ItaModel, cross_validate
from .monk import consensus, load_ratings
from .reference import find_reference, load_reference
from .realtrain import TrainItem
from .session import load_session
from .video import VideoSignals, resample_uniform


def pilot_items(cfg, sessions) -> list[TrainItem]:
    from .pilot import _participant_tone, load_measured
    root = Path(cfg.paths.pilot_root)
    measured = load_measured(cfg.ita.measured_file) if cfg.ita.source == "measured" else {}
    monk = {}
    if (root / "monk_ratings.csv").exists():
        c = consensus(load_ratings(root / "monk_ratings.csv"), int(cfg.raters.n), cfg.raters.consensus)
        monk = dict(zip(c.participant_id, c.monk))
    items = []
    for sd in sessions:
        d = Path(sd) / "derived"
        if not (d / "qc.json").exists() or not json.loads((d / "qc.json").read_text())["usable"]:
            continue
        s = load_session(sd)
        vs = VideoSignals.load(d / "signals_webcam_lossless.npz")
        sync = json.loads((d / "sync.json").read_text())
        ref = load_reference(find_reference(s.root, cfg.reference.file_glob), cfg.reference.type, dict(cfg.reference.columns),
                             cfg.reference.time_format, float(cfg.reference.fs_hz)).shifted(-sync["webcam"]["lag_s"])
        fps = 30.0
        g, P = resample_uniform(vs.t, vs.patches, fps)
        ppg = np.interp(g, ref.wave_t, ref.wave) if ref.wave is not None else None
        tone = _participant_tone(s.pid, cfg, d, measured, monk, None)
        items.append(TrainItem(f"PILOT:{s.pid}", "PILOT", P, np.nanmean(vs.pix, 0), fps, ref, ppg,
                               float(tone["ita_deg"]) if tone["ita_deg"] is not None else float("nan"), tone["stratum"],
                               [(b.start, b.end, b.light, b.name) for b in s.blocks if b.name != "ita_capture"], s.pid))
    return items


def train_ita(cfg, sessions, target="monk", learner="ridge", out: Path | None = None) -> dict:
    from .pilot import load_measured
    root = Path(cfg.paths.pilot_root)
    if target == "monk":
        if not (root / "monk_ratings.csv").exists():
            raise SystemExit(f"no {root / 'monk_ratings.csv'}")
        c = consensus(load_ratings(root / "monk_ratings.csv"), int(cfg.raters.n), cfg.raters.consensus)
        labels = dict(zip(c.participant_id, c.monk))
        cuts = tuple(cfg.strata.monk_cuts)
    else:
        labels = load_measured(cfg.ita.measured_file)
        if not labels:
            raise SystemExit(f"target=ita needs colorimeter values in {cfg.ita.measured_file}")
        cuts = tuple(cfg.strata.ita_cuts)
    feats, y, ids = [], [], []
    for sd in sessions:
        f = Path(sd) / "derived/ita_features.json"
        pid = load_session(sd).pid
        if f.exists() and pid in labels:
            fe = json.loads(f.read_text())
            if fe:
                feats.append(fe)
                y.append(labels[pid])
                ids.append(pid)
    if len(y) < 5:
        raise SystemExit(f"only {len(y)} participants have both video features and a {target} label: need at least 5")
    res = cross_validate(feats, np.array(y), target, learner, int(cfg.training.folds), int(cfg.project.seed) % 2 ** 31, cuts)
    res["participants"] = ids
    model = ItaModel(target, learner).fit(feats, np.array(y))
    model.meta = {"cv": {k: v for k, v in res.items() if k != "pred"}, "n": len(y), "target": target,
                  "provenance": cfg.project.provenance_label,
                  "warning": "small-sample model: use the cross-validated error, not the training fit"}
    path = Path(out) / "ita_model.json" if out else Path(cfg.ita.model_file)
    if learner == "ridge":
        model.save(path)
        res["saved"] = str(path)
    else:
        res["saved"] = "not saved (gbr is for comparison only; use ridge for a portable model)"
    return res

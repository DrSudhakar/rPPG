"""Training the SAMA method (steps 3-6) on real recordings: public datasets, the pilot, or both.

    python -m sama train                                   # defaults from configs/sama.yaml
    python -m sama train --set training.group_dro=false --set gate.type=pooled

What happens:
 1. Each recording is reduced to per-frame patch colours (sama.video) and windows (30 s; denser
    steps for training only). The reference pulse (contact PPG of UBFC-rPPG/PURE, or the pilot's
    oximeter waveform after synchronisation) is the network's training target.
 2. Participant-level K-fold cross-validation: the network (step 4) and sigma_hat (step 6) are
    trained on K-1 folds and predict the held-out fold. Out-of-fold heart rates give the
    cross-validated error for SAMA and for POS on the same windows.
 3. The reliability gate is calibrated with cross-conformal scores (each participant scored by a
    model that never saw them; Barber et al. 2021), pooled or per stratum. Coverage is checked
    fold by fold with quantiles from the other folds only.
 4. A final model is trained on every participant and saved as a bundle with the gate, the
    configuration snapshot, the CV metrics and the provenance of every training recording.

Knobs: training.* (datasets, folds, epochs, group_dro, dro_groups, loss, noise_weighting,
use_network, public_ita), ita.conditioning, gate.* (type, alpha, halfwidth_max_bpm).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import net as N
from .conformal import MondrianGate
from .sama_method import SamaModel, SigmaModel, WindowFeatures, window_features
from .signals import hr_from_beats, window_starts
from .video import resample_uniform


@dataclass
class TrainItem:
    participant: str
    dataset: str
    patches: np.ndarray          # (T, 30, 3) on a uniform grid at fps
    pix: np.ndarray              # (30,)
    fps: float
    ref: object                  # sama.reference.Reference on the same clock as the patches
    ppg: np.ndarray | None       # (T,) reference waveform on the frame grid, or None
    ita: float = float("nan")
    stratum: str = "all"
    blocks: list = field(default_factory=list)   # [(start_s, end_s, light, name)]
    name: str = ""


def item_from_recording(rec, ita=float("nan"), stratum="all") -> TrainItem:
    """sama.loaders.RealRecording -> TrainItem (uniform 30 fps grid)."""
    from .reference import Reference
    t = np.arange(rec.patches.shape[0]) / rec.fps
    fps = 30.0
    g, P = resample_uniform(t, rec.patches, fps)
    ppg = np.interp(g, t, rec.ppg) if rec.ppg is not None else None
    pix = np.nanmean(rec.pix, 0) if rec.pix is not None else np.ones(30)
    ref = Reference("contact_ppg", float("nan"), beats=np.asarray(rec.beats) - g[0])
    return TrainItem(rec.participant or rec.name, rec.source, P, pix, fps, ref, ppg, ita, stratum,
                     [(0.0, P.shape[0] / fps, "standard", "whole")], rec.name)


def windows(item: TrainItem, length_s=30.0, step_s=15.0) -> list:
    rows = []
    for start, end, light, bname in item.blocks:
        a0, b0 = int(round(start * item.fps)), int(round(end * item.fps))
        for a, b in window_starts(b0 - a0, item.fps, length_s, step_s):
            a, b = a + a0, b + a0
            rows.append((a, b, item.ref.window_hr(a / item.fps, b / item.fps), light, bname))
    return rows


def features(item: TrainItem, length_s=30.0, step_s=15.0, conditioning=True, noise_weighting=True) -> WindowFeatures | None:
    rows = windows(item, length_s, step_s)
    if not rows:
        return None
    ita_c = item.ita if (conditioning and np.isfinite(item.ita)) else 0.0
    out = []
    for light in sorted({r[3] for r in rows}):
        rr = [(a, b, h) for a, b, h, li, _ in rows if li == light]
        wf = window_features(item.patches, item.fps, item.pix, ita_c, None, None, rr, ppg=item.ppg, light=light,
                             use_step2=False, use_step3=noise_weighting,
                             meta={"participant": item.participant, "dataset": item.dataset, "stratum": item.stratum,
                                   "recording": item.name, "ita": item.ita})
        if not conditioning:
            wf.cond[:, 0] = 0.0
        blocks = [bn for a, b, h, li, bn in rows if li == light]
        for m, bn in zip(wf.meta, blocks):
            m["block"] = bn
        out.append(wf)
    return concat(out)


def concat(parts: list) -> WindowFeatures:
    parts = [p for p in parts if p is not None and len(p.hr_ref)]
    wf = WindowFeatures(*(np.concatenate([getattr(p, k) for p in parts]) for k in ("X", "cond", "ppg", "hr_ref", "hr_pos", "hr_cand", "q")))
    wf.meta = [m for p in parts for m in p.meta]
    return wf


def subset(wf: WindowFeatures, mask) -> WindowFeatures:
    mask = np.asarray(mask, bool)
    out = WindowFeatures(*(getattr(wf, k)[mask] for k in ("X", "cond", "ppg", "hr_ref", "hr_pos", "hr_cand", "q")))
    out.meta = [m for m, k in zip(wf.meta, mask) if k]
    return out


def group_ids(wf: WindowFeatures, how: str) -> np.ndarray:
    if how == "none":
        return np.zeros(len(wf.hr_ref), int)
    keys = {"dataset": lambda m: m["dataset"], "stratum": lambda m: m["stratum"],
            "stratum_x_dataset": lambda m: f'{m["stratum"]}|{m["dataset"]}'}[how]
    labels = [keys(m) for m in wf.meta]
    uniq = {u: i for i, u in enumerate(sorted(set(labels)))}
    return np.array([uniq[x] for x in labels], int)


def train_config(cfg, film: bool) -> N.TrainConfig:
    t = cfg.training
    return N.TrainConfig(epochs=int(t.epochs), lr=float(t.lr), group_dro=bool(t.group_dro), film=film,
                         seed=int(cfg.project.seed) % (2 ** 31), channels=int(t.channels),
                         lam=0.5 if t.loss == "pearson_spectral" else 1.0)


def _fit(train: WindowFeatures, cfg, film: bool, log=None) -> SamaModel:
    """Network + sigma_hat on `train` (no gate yet)."""
    model = SamaModel(use_net=bool(cfg.training.use_network))
    tc = train_config(cfg, film)
    ok = np.isfinite(train.hr_ref)
    if model.use_net:
        ppg = train.ppg[ok].copy()
        if cfg.training.loss == "spectral":
            ppg[:] = 0.0
        model.film = film
        model.params, model.hist = N.train(train.X[ok], train.cond[ok], ppg, train.hr_ref[ok],
                                           group_ids(subset(train, ok), cfg.training.dro_groups if cfg.training.group_dro else "none"),
                                           tc, log=log)
    h, q = model.predict_point(train)
    m = np.isfinite(train.hr_ref) & np.isfinite(h)
    model.sigma_model = SigmaModel().fit(q[m], np.log(np.abs(h[m] - train.hr_ref[m]) + 0.5))
    return model


def gate_groups(wf: WindowFeatures, gate_type: str) -> np.ndarray:
    return np.array([m["stratum"] if gate_type == "mondrian" else "all" for m in wf.meta])


def run(items: list[TrainItem], cfg, out_dir, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    film = bool(cfg.ita.conditioning) and all(np.isfinite(it.ita) for it in items)
    notes = []
    if cfg.ita.conditioning and not film:
        notes.append("ita.conditioning requested but some training recordings have no ITA°: network trained without FiLM conditioning")
    gate_type = cfg.gate.type
    if gate_type == "mondrian" and any(it.stratum in ("all", "unknown") for it in items):
        notes.append("gate.type=mondrian needs a stratum for every training participant: pooled gate used")
        gate_type = "pooled"
    step = float(cfg.training.window_step_s)
    feats_train = {it.name: features(it, cfg.signal.window_s, step, film, cfg.training.noise_weighting) for it in items}
    feats_eval = {it.name: features(it, cfg.signal.window_s, cfg.signal.step_s, film, cfg.training.noise_weighting) for it in items}
    items = [it for it in items if feats_train[it.name] is not None and feats_eval[it.name] is not None]
    parts = sorted({it.participant for it in items})
    k = int(min(cfg.training.folds, len(parts)))
    if k < 2:
        raise ValueError("need at least 2 participants for cross-validation")
    rng = np.random.default_rng(int(cfg.project.seed) % (2 ** 31))
    order = rng.permutation(len(parts))
    fold_of = {parts[i]: j % k for j, i in enumerate(order)}
    log(f"[train] {len(items)} recordings, {len(parts)} participants, {k} folds, FiLM={film}, "
        f"group-DRO={cfg.training.group_dro}/{cfg.training.dro_groups}, gate={gate_type}")
    oof = []
    fold_models = {}
    for f in range(k):
        tr = concat([feats_train[it.name] for it in items if fold_of[it.participant] != f])
        te = concat([feats_eval[it.name] for it in items if fold_of[it.participant] == f])
        model = _fit(tr, cfg, film)
        h, q = model.predict_point(te)
        sig = model.sigma(q)
        for i, m in enumerate(te.meta):
            oof.append({**{k2: m[k2] for k2 in ("participant", "dataset", "stratum", "recording", "block", "t0", "light")},
                        "fold": f, "hr_ref": te.hr_ref[i], "hr_sama": h[i], "hr_pos": te.hr_pos[i], "hr_cand": te.hr_cand[i],
                        "sigma": sig[i]})
        fold_models[f] = model
        log(f"[train] fold {f + 1}/{k}: {len(tr.hr_ref)} train windows, {len(te.hr_ref)} test windows")
    df = pd.DataFrame(oof)
    df["gate_group"] = df.stratum if gate_type == "mondrian" else "all"
    df["score"] = (df.hr_sama - df.hr_ref).abs() / df.sigma.clip(lower=1e-6)
    # fold-wise coverage with quantiles from the other folds only (honest check of the gate)
    cov_rows = []
    for f in range(k):
        cal, te = df[(df.fold != f) & np.isfinite(df.score)], df[df.fold == f]
        g = MondrianGate(alpha=cfg.gate.alpha, halfwidth_max=cfg.gate.halfwidth_max_bpm, draws=int(cfg.gate.draws))
        g.fit(cal.hr_sama, cal.hr_ref, cal.sigma, cal.gate_group, cal.participant, seed=f)
        lo, hi, hw, shown = g.decide(te.hr_sama.to_numpy(), te.sigma.to_numpy(), te.gate_group.to_numpy())
        df.loc[te.index, ["lo", "hi", "halfwidth", "shown"]] = np.column_stack([lo, hi, hw, shown])
        cov_rows.append(f)
    df["shown"] = df.shown.astype(bool)
    df["covered"] = (df.hr_ref >= df.lo) & (df.hr_ref <= df.hi)
    metrics = summarise(df)
    # final model on everyone; gate from all out-of-fold scores (cross-conformal)
    allf = concat([feats_train[it.name] for it in items])
    final = _fit(allf, cfg, film, log=None)
    ok = np.isfinite(df.score)
    gate = MondrianGate(alpha=cfg.gate.alpha, halfwidth_max=cfg.gate.halfwidth_max_bpm, draws=int(cfg.gate.draws))
    gate.fit(df.hr_sama[ok], df.hr_ref[ok], df.sigma[ok], df.gate_group[ok], df.participant[ok], seed=0)
    final.gate = gate
    meta = {"version": f"sama-real-{time.strftime('%Y%m%d-%H%M')}", "trained_on": sorted({it.dataset for it in items}),
            "n_recordings": len(items), "n_participants": len(parts), "film": film, "gate_type": gate_type,
            "group_dro": bool(cfg.training.group_dro), "dro_groups": cfg.training.dro_groups, "loss": cfg.training.loss,
            "noise_weighting": bool(cfg.training.noise_weighting), "use_network": bool(cfg.training.use_network),
            "public_ita": cfg.training.public_ita, "n_params": N.n_params(final.params) if final.params else 0,
            "provenance": "REAL-PUBLIC" if all(it.dataset != "PILOT" for it in items) else "REAL-PUBLIC + REAL-PILOT",
            "cv": metrics, "notes": notes, "minutes": round((time.time() - t_start) / 60, 1),
            "recordings": [{"name": it.name, "dataset": it.dataset, "participant": it.participant, "ita": it.ita,
                            "stratum": it.stratum} for it in items]}
    save_bundle(out, final, meta)
    df.to_csv(out / "cv_predictions.csv", index=False)
    (out / "cv_metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    log(f"[train] CV MAE: SAMA {metrics['all']['mae_sama']:.2f} vs POS {metrics['all']['mae_pos']:.2f} bpm; "
        f"gate coverage {metrics['all']['coverage_shown']:.2f}, abstention {metrics['all']['abstention']:.2f}")
    return meta


def evaluate(cfg, bundle_dir, datasets=None, out_dir=None, sessions=None, log=print) -> dict:
    """Test a trained bundle on whole datasets it may never have seen (cross-dataset protocol).

        python -m sama test --bundle models/sama_real --datasets MMPD,iBVP

    `sama train` already reports per-dataset accuracy, but only within participant-level
    cross-validation of the datasets it trained on. This evaluates a finished bundle on any subset
    of datasets, which is the generalisation question: train on RLAP + PURE, test on MMPD. Any
    dataset that was also in training is flagged as such in the output instead of being quietly
    reported as a generalisation result.
    """
    model, meta = load_model(bundle_dir)
    if model is None:
        raise FileNotFoundError(f"no SAMA bundle in {bundle_dir} (run `python -m sama train` first)")
    names = list(datasets) if datasets else list(cfg.training.datasets)
    from .datasets import PUBLIC_NAMES
    import copy
    cfg_eval = copy.deepcopy(cfg)      # Cfg is a dict subclass: a shallow copy would alias training
    cfg_eval.training.datasets = names
    items = public_items(cfg_eval, log=log) if set(names) & set(PUBLIC_NAMES) else []
    if "PILOT" in names:
        from .pilotitems import pilot_items
        items += pilot_items(cfg_eval, sessions or [])
    if not items:
        raise ValueError(f"no recordings found for {names}; run `python -m sama datasets` to check the layout")

    film = bool(meta.get("film"))
    feats = {it.name: features(it, cfg.signal.window_s, cfg.signal.step_s, film, cfg.training.noise_weighting)
             for it in items}
    items = [it for it in items if feats[it.name] is not None]
    if not items:
        raise ValueError(f"no analysis windows: recordings are shorter than signal.window_s "
                         f"({cfg.signal.window_s} s)")
    te = concat([feats[it.name] for it in items])
    h, q = model.predict_point(te)
    sig = model.sigma(q)
    rows = [{**{k: m[k] for k in ("participant", "dataset", "stratum", "recording", "block", "t0", "light")},
             "hr_ref": te.hr_ref[i], "hr_sama": h[i], "hr_pos": te.hr_pos[i], "hr_cand": te.hr_cand[i],
             "sigma": sig[i]} for i, m in enumerate(te.meta)]
    df = pd.DataFrame(rows)
    df["gate_group"] = df.stratum if meta.get("gate_type") == "mondrian" else "all"
    groups = df.gate_group.to_numpy()
    lo, hi, hw, shown = model.gate.decide(df.hr_sama.to_numpy(), df.sigma.to_numpy(), groups)
    df["lo"], df["hi"], df["halfwidth"], df["shown"] = lo, hi, hw, shown.astype(bool)
    df["covered"] = (df.hr_ref >= df.lo) & (df.hr_ref <= df.hi)
    metrics = summarise(df)
    trained_on = set(meta.get("trained_on", []))
    metrics["_eval"] = {"bundle": str(bundle_dir), "bundle_version": meta.get("version"),
                        "trained_on": sorted(trained_on), "tested_on": sorted({it.dataset for it in items}),
                        "held_out": sorted({it.dataset for it in items} - trained_on),
                        "also_in_training": sorted({it.dataset for it in items} & trained_on),
                        "n_recordings": len(items), "window_s": cfg.signal.window_s}
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "eval_predictions.csv", index=False)
        (out / "eval_metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
        (out / "eval_report.md").write_text(eval_report(metrics))
    log(eval_report(metrics))
    return metrics


def eval_report(metrics: dict) -> str:
    e = metrics["_eval"]
    lines = [f"# SAMA cross-dataset test ({e['bundle_version']})", "",
             f"Bundle trained on: {', '.join(e['trained_on']) or 'unknown'}",
             f"Tested on: {', '.join(e['tested_on'])}  ({e['n_recordings']} recordings, "
             f"{metrics['all']['n_windows']} windows of {e['window_s']} s)", ""]
    if e["also_in_training"]:
        lines += [f"**Not a generalisation result for {', '.join(e['also_in_training'])}**: that data was also "
                  "used in training, so its numbers below are in-sample.", ""]
    if e["held_out"]:
        lines += [f"Held out of training (true generalisation): {', '.join(e['held_out'])}", ""]
    lines += ["| set | n windows | participants | MAE SAMA | MAE POS | within 5 bpm (SAMA) | abstention | coverage (shown) |",
              "|---|---|---|---|---|---|---|---|"]
    for key in ["all"] + sorted(k for k in metrics if k.startswith("dataset:")):
        b = metrics[key]
        tag = key.replace("dataset:", "")
        if key != "all" and tag in e["also_in_training"]:
            tag += " (in training)"
        lines.append(f"| {tag} | {b['n_windows']} | {b['n_participants']} | {b['mae_sama']:.2f} | "
                     f"{b['mae_pos']:.2f} | {b['within5_sama']:.2f} | {b['abstention']:.2f} | "
                     f"{b['coverage_shown']:.2f} |")
    return "\n".join(lines) + "\n"


def summarise(df: pd.DataFrame) -> dict:
    def block(d):
        d = d[np.isfinite(d.hr_ref)]
        per = d.groupby("participant").apply(lambda g: pd.Series({"s": (g.hr_sama - g.hr_ref).abs().mean(),
                                                                  "p": (g.hr_pos - g.hr_ref).abs().mean()}), include_groups=False)
        sh = d[d.shown]
        return {"n_windows": int(len(d)), "n_participants": int(d.participant.nunique()),
                "mae_sama": float((d.hr_sama - d.hr_ref).abs().mean()), "mae_pos": float((d.hr_pos - d.hr_ref).abs().mean()),
                "mae_cand": float((d.hr_cand - d.hr_ref).abs().mean()),
                "participant_mae_sama": float(per.s.mean()), "participant_mae_pos": float(per.p.mean()),
                "within5_sama": float(((d.hr_sama - d.hr_ref).abs() <= 5).mean()),
                "within5_pos": float(((d.hr_pos - d.hr_ref).abs() <= 5).mean()),
                "abstention": float(1 - d.shown.mean()), "coverage_all": float(d.covered.mean()),
                "coverage_shown": float(sh.covered.mean()) if len(sh) else float("nan"),
                "mae_shown": float((sh.hr_sama - sh.hr_ref).abs().mean()) if len(sh) else float("nan"),
                "useful_yield": float((d.shown & ((d.hr_sama - d.hr_ref).abs() <= 5)).mean())}
    out = {"all": block(df)}
    for k, g in df.groupby("dataset"):
        out[f"dataset:{k}"] = block(g)
    for k, g in df.groupby("stratum"):
        out[f"stratum:{k}"] = block(g)
    return out


def save_bundle(out: Path, model: SamaModel, meta: dict):
    if model.params is not None:
        N.save(model.params, out / "sama_net.npz", meta)
    (out / "sama_model.json").write_text(json.dumps({"gate": model.gate.to_dict(), "sigma": model.sigma_model.to_dict(),
                                                     "meta": meta}, indent=1, default=float))


def load_model(bundle_dir) -> tuple[SamaModel, dict] | tuple[None, None]:
    d = Path(bundle_dir)
    if not (d / "sama_model.json").exists():
        return None, None
    j = json.loads((d / "sama_model.json").read_text())
    meta = j["meta"]
    m = SamaModel(use_net=meta.get("use_network", True) and (d / "sama_net.npz").exists())
    m.params = N.load(d / "sama_net.npz")[0] if m.use_net else None
    m.film = bool(meta.get("film", True))
    m.sigma_model = SigmaModel.from_dict(j["sigma"])
    m.gate = MondrianGate.from_dict(j["gate"])
    return m, meta


def gate_group_for(meta: dict, stratum: str) -> str:
    return stratum if meta.get("gate_type") == "mondrian" else "all"


def public_items(cfg, log=print) -> list[TrainItem]:
    """Load every configured public dataset into TrainItems (see sama.datasets for the layouts).

    `training.max_recordings` caps each dataset separately, so a smoke run still covers all of
    them instead of spending its whole budget on whichever dataset sorts first.
    """
    from . import loaders
    from .datasets import PUBLIC_NAMES
    from .ita_camera import stratum_from
    wanted = [d for d in cfg.training.datasets if d in PUBLIC_NAMES]
    found = loaders.discover(cfg.paths.public_root, only=wanted)
    if cfg.training.max_recordings:
        cap, seen, kept = int(cfg.training.max_recordings), {}, []
        for kind, path in found:
            if seen.get(kind, 0) < cap:
                seen[kind] = seen.get(kind, 0) + 1
                kept.append((kind, path))
        found = kept
    counts = {k: sum(1 for kk, _ in found if kk == k) for k in wanted}
    log(f"[train] datasets under {cfg.paths.public_root}: " +
        ", ".join(f"{k}={counts[k]}" for k in wanted) if wanted else "[train] no public datasets configured")
    for k in [k for k in wanted if not counts[k]]:
        log(f"[train] {k} is configured but nothing was found on disk "
            f"(expected {cfg.paths.public_root}/{k}/...; run `python -m sama datasets` to check)")
    items = []
    for kind, path in found:
        try:
            rec = loaders.load(kind, path, cfg.face.tracker if cfg.face.tracker != "manual_box" else "auto")
        except Exception as e:  # noqa: BLE001
            log(f"[train] skipped {path}: {e}")
            continue
        ita = float("nan")
        if cfg.training.public_ita == "camera_uncalibrated":
            from .ita_camera import ita_from_linear, srgb_to_linear
            ita = ita_from_linear(srgb_to_linear(rec.patches[:, 12:].mean((0, 1))))[0]
        st = stratum_from(ita, cfg.strata.ita_cuts) if np.isfinite(ita) else "unknown"
        items.append(item_from_recording(rec, ita, st))
        log(f"[train] loaded {kind} {rec.name}: {rec.patches.shape[0]} frames at {rec.fps:.1f} fps, {len(rec.beats)} reference beats")
    return items

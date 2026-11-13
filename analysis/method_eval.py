"""C7-C8 and WP2 error budget on the virtual SAMA-120 cohort (SYNTHETIC).

  python analysis/method_eval.py [--epochs 5] [--folds 5]

Implements spec/SAP_SIM.md s.2-3 with amendment 1. Writes results/tables/c7_*.csv, c8_*.csv,
budget_*.csv, figures, deploy/model (final model: trained on half of the analysable participants, calibrated on the other half) and
results/c7_summary.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sama import model_io  # noqa: E402
from sama import net as N  # noqa: E402
from sama.conformal import MondrianGate, coverage_report  # noqa: E402
from sama.methods import METHODS  # noqa: E402
from sama.sama_method import SamaModel, WindowFeatures  # noqa: E402
from sama.stats import bland_altman, icc21, lmm_crossed, paired_test  # noqa: E402

T, F = ROOT / "results/tables", ROOT / "results/figures"
STRATA = ("lighter", "middle", "darker")
SID = {s: i for i, s in enumerate(STRATA)}
RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ data

def cat_wf(items):
    wfs = [w for w, _ in items]
    out = WindowFeatures(*(np.concatenate([getattr(w, k) for w in wfs]) for k in ("X", "cond", "ppg", "hr_ref", "hr_pos", "hr_cand", "q")))
    out.meta = sum([w.meta for w in wfs], [])
    bl = {m: np.concatenate([np.asarray(b[m], float) for _, b in items]) for m in METHODS}
    return out, bl


def load_cohort(path):
    files = sorted(glob.glob(str(Path(path) / "*.pkl")))
    parts, lossless, shift, hrv = [], [], {}, {}
    for f in files:
        d = pickle.load(open(f, "rb"))
        parts.append(d["part"])
        lossless += [d["lossless"][k] for k in sorted(d["lossless"])]
        for k, v in d["shift"].items():
            shift.setdefault(k[0] + "|" + k[1] + "|" + k[3], []).append(v)
        hrv[d["part"]["pid"]] = d["hrv"]
    wf, bl = cat_wf(lossless)
    sh = {k: cat_wf(v) for k, v in shift.items()}
    return pd.DataFrame(parts), wf, bl, sh, hrv


def meta_df(wf):
    m = pd.DataFrame(wf.meta)
    m["stratum_id"] = m.stratum.map(SID)
    m["group"] = m.stratum_id * 2 + (m.light == "dim").astype(int)
    return m


def subset(wf, bl, idx):
    s = WindowFeatures(*(getattr(wf, k)[idx] for k in ("X", "cond", "ppg", "hr_ref", "hr_pos", "hr_cand", "q")))
    s.meta = [wf.meta[i] for i in idx]
    return s, {m: v[idx] for m, v in bl.items()}


# ------------------------------------------------------------------ helpers

def part_mae(err, pid):
    d = pd.DataFrame({"e": np.abs(err), "pid": pid}).dropna()
    return d.groupby("pid").e.mean()


def stratum_mae(err, pid, stratum_of):
    pm = part_mae(err, pid)
    return {s: float(pm[[p for p in pm.index if stratum_of[p] == s]].mean()) for s in STRATA}


def criteria(m_sama, m_best, cov, abst):
    gap = lambda m: max(m.values()) - min(m.values())  # noqa: E731
    return {"darker_reduction": 1 - m_sama["darker"] / m_best["darker"],
            "gap_reduction": 1 - gap(m_sama) / gap(m_best) if gap(m_best) > 0 else np.nan,
            "lighter_increase": m_sama["lighter"] / m_best["lighter"] - 1,
            "min_coverage": min(cov.values()), "abstention_spread_pp": 100 * (max(abst.values()) - min(abst.values()))}


def passes(c):
    return {"c1": c["darker_reduction"] >= 0.30, "c2": c["gap_reduction"] >= 0.50, "c3": c["lighter_increase"] <= 0.10,
            "c4": c["min_coverage"] >= 0.88, "c5": c["abstention_spread_pp"] <= 10}


# ------------------------------------------------------------------ main evaluation

def run(cohort_dir, folds=5, epochs=5, log=print):
    parts, wf, bl, shift, hrv = load_cohort(cohort_dir)
    m = meta_df(wf)
    stratum_of = dict(zip(parts.pid, parts.stratum))
    ita_of = dict(zip(parts.pid, parts.ita))
    # 10 % session loss per stratum
    keep = []
    for s in STRATA:
        ids = parts.pid[parts.stratum == s].to_numpy()
        keep += list(RNG.choice(ids, size=int(round(len(ids) * 0.9)), replace=False))
    keep = set(keep)
    # stratified participant folds
    fold_of = {}
    for s in STRATA:
        ids = RNG.permutation(sorted(p for p in keep if stratum_of[p] == s))
        for i, p in enumerate(ids):
            fold_of[p] = i % folds
    variants = {"SAMA": dict(use_net=True, film=True, dro=True), "-FiLM": dict(use_net=True, film=False, dro=True),
                "-groupDRO": dict(use_net=True, film=True, dro=False), "-CNN": dict(use_net=False, film=True, dro=True)}
    pred = {v: {} for v in list(variants) + ["POS+gate"]}
    best_by_fold, frozen_q, hist_all = {}, [], []
    test_rows = []
    shift_preds = []
    for k in range(folds):
        t0 = time.time()
        te = [p for p in keep if fold_of[p] == k]
        tr = [p for p in keep if fold_of[p] != k]
        # A1: 50/50 train-proper / calibration, stratified
        tp, cal, cal_frozen = [], [], []
        for s in STRATA:
            ids = RNG.permutation(sorted(p for p in tr if stratum_of[p] == s))
            h = len(ids) // 2
            tp += list(ids[:h]); cal += list(ids[h:])
            cal_frozen += list(ids[: int(round(len(ids) * 0.25))])
        idx = lambda ps: np.where(m.pid.isin(ps).to_numpy())[0]  # noqa: E731
        i_tp, i_cal, i_te = idx(tp), idx(cal), idx(te)
        i_tp_train = i_tp[(m.t0.to_numpy()[i_tp] % 30) == 0]          # non-overlapping windows for training
        wf_tp, bl_tp = subset(wf, bl, i_tp_train)
        wf_tpall, _ = subset(wf, bl, i_tp)
        wf_cal, _ = subset(wf, bl, i_cal)
        wf_te, bl_te = subset(wf, bl, i_te)
        g_tp = m.group.to_numpy()[i_tp_train]
        s_cal, p_cal = m.stratum.to_numpy()[i_cal], m.pid.to_numpy()[i_cal]
        s_te = m.stratum.to_numpy()[i_te]
        # best baseline on outer-training participants (D-11)
        i_trall = idx(tr)
        darker_tr = [p for p in tr if stratum_of[p] == "darker"]
        mae_tr = {meth: part_mae(bl[meth][i_trall] - wf.hr_ref[i_trall], m.pid.to_numpy()[i_trall])[darker_tr].mean() for meth in METHODS}
        best = min(mae_tr, key=mae_tr.get)
        best_by_fold[k] = best
        models = {}
        for v, o in variants.items():
            cfg = N.TrainConfig(epochs=epochs, film=o["film"], group_dro=o["dro"], seed=k)
            sm = SamaModel(use_net=o["use_net"])
            sm.fit(wf_tp, g_tp, wf_cal, s_cal, p_cal, cfg=cfg, sigma_train=wf_tpall,
                   log=(lambda h, v=v: hist_all.append({"fold": k, "variant": v, **h})) if o["use_net"] else None)
            models[v] = sm
            out = sm.predict(wf_te, s_te)
            pred[v][k] = out
        # baseline with the same sigma/gate machinery: POS on the face mean as the point estimate
        def as_pos(w):
            w2 = deepcopy(w)
            w2.hr_cand = w.hr_pos.copy()
            w2.q = w.q.copy()
            w2.q[:, 0] = w.q[:, 2]
            return w2
        pg = SamaModel(use_net=False).fit(as_pos(wf_tp), g_tp, as_pos(wf_cal), s_cal, p_cal, sigma_train=as_pos(wf_tpall))
        pred["POS+gate"][k] = pg.predict(as_pos(wf_te), s_te)
        # frozen 75/25 calibration: quantiles with the same SAMA model
        i_cf = idx(cal_frozen)
        wf_cf, _ = subset(wf, bl, i_cf)
        hc, qc = models["SAMA"].predict_point(wf_cf)
        g = MondrianGate().fit(hc, wf_cf.hr_ref, models["SAMA"].sigma(qc), m.stratum.to_numpy()[i_cf], m.pid.to_numpy()[i_cf])
        frozen_q.append({"fold": k, **{f"n_cal_{s}": sum(stratum_of[p] == s for p in cal_frozen) for s in STRATA},
                         **{f"q_{s}": g.q.get(s) for s in STRATA}})
        for j, ii in enumerate(i_te):
            test_rows.append({"fold": k, "pid": m.pid.iat[ii], "stratum": m.stratum.iat[ii], "light": m.light.iat[ii],
                              "block": m.block.iat[ii], "hr_ref": wf.hr_ref[ii], "best": best,
                              **{f"hr_{meth}": bl[meth][ii] for meth in METHODS}})
        # shift conditions for this fold's test participants
        for key, (wfs, bls) in shift.items():
            ms = pd.DataFrame(wfs.meta)
            ii = np.where(ms.pid.isin(te).to_numpy())[0]
            if ii.size == 0:
                continue
            w_s, b_s = subset(wfs, bls, ii)
            st = ms.stratum.to_numpy()[ii]
            o = models["SAMA"].predict(w_s, st)
            shift_preds.append(pd.DataFrame({"condition": key, "pid": ms.pid.to_numpy()[ii], "stratum": st,
                                             "device": ms.device.to_numpy()[ii], "light": [mm["light"] for mm in w_s.meta],
                                             "hr_ref": w_s.hr_ref, "hr": o["hr"], "lo": o["lo"], "hi": o["hi"], "shown": o["shown"],
                                             "hr_pos": w_s.hr_pos, "hr_cand": w_s.hr_cand,
                                             "electrons_g": ms.electrons_g.to_numpy()[ii], "gain": ms.gain.to_numpy()[ii],
                                             "exposure_s": ms.exposure_s.to_numpy()[ii]}))
        # ITA estimated with error (SD 6°): stratum and FiLM input from the estimate (I-10)
        ita_obs = {p: ita_of[p] + RNG.normal(0, 6.0) for p in te}
        st_obs = np.array(["lighter" if ita_obs[pp] > 28 else ("middle" if ita_obs[pp] > 10 else "darker") for pp in m.pid.to_numpy()[i_te]])
        w2 = deepcopy(wf_te)
        w2.cond[:, 0] = np.array([ita_obs[pp] for pp in m.pid.to_numpy()[i_te]]) / 50.0
        o = models["SAMA"].predict(w2, st_obs)
        shift_preds.append(pd.DataFrame({"condition": "ITA estimated (SD 6°)", "pid": m.pid.to_numpy()[i_te], "stratum": s_te,
                                         "device": "reference30", "light": m.light.to_numpy()[i_te], "hr_ref": wf_te.hr_ref,
                                         "hr": o["hr"], "lo": o["lo"], "hi": o["hi"], "shown": o["shown"], "hr_pos": wf_te.hr_pos,
                                         "hr_cand": wf_te.hr_cand, "electrons_g": np.nan, "gain": np.nan, "exposure_s": np.nan}))
        log(f"fold {k}: best baseline {best}; {time.time() - t0:.0f}s")
    tr_df = pd.DataFrame(test_rows)
    for v in pred:
        for col in ("hr", "lo", "hi", "shown", "sigma"):
            tr_df[f"{v}_{col}"] = np.concatenate([pred[v][k][col] for k in range(folds)])
    return parts, keep, stratum_of, tr_df, pd.concat(shift_preds), pd.DataFrame(frozen_q), pd.DataFrame(hist_all), hrv, wf, m, bl, shift


def evaluate(parts, stratum_of, d, shift_df, frozen_q, hist, hrv, keep):
    out = {"provenance": "SYNTHETIC (virtual SAMA-120, World A); neural baselines absent (I-24)"}
    d = d[np.isfinite(d.hr_ref)].copy()
    d["hr_best"] = [r[f"hr_{r.best}"] for _, r in d.iterrows()]
    pid = d.pid.to_numpy()
    rows, crit = [], {}
    best_m = stratum_mae(d.hr_best - d.hr_ref, pid, stratum_of)
    variants = ["SAMA", "-FiLM", "-groupDRO", "-CNN", "POS+gate"]
    for v in variants + ["best baseline"] + list(METHODS):
        hr = d[f"{v}_hr"] if v in variants else (d.hr_best if v == "best baseline" else d[f"hr_{v}"])
        e = hr - d.hr_ref
        sm = stratum_mae(e, pid, stratum_of)
        r = {"variant": v, **{f"mae_{s}": sm[s] for s in STRATA}, "gap": max(sm.values()) - min(sm.values())}
        if v in variants:
            cr = coverage_report(d.hr_ref, d[f"{v}_lo"], d[f"{v}_hi"], d[f"{v}_shown"], d.stratum, pid)
            for s in STRATA:
                r[f"cov_all_{s}"] = cr[s]["coverage_all"]
                r[f"cov_shown_{s}"] = cr[s]["coverage_shown"]
                r[f"abst_{s}"] = cr[s]["abstention"]
                r[f"yield_{s}"] = cr[s]["useful_yield"]
                shown = d[f"{v}_shown"].to_numpy(bool)
            rm = stratum_mae(np.where(shown, e, np.nan), pid, stratum_of)
            for s in STRATA:
                r[f"mae_retained_{s}"] = rm[s]
            c = criteria(sm, best_m, {s: cr[s]["coverage_all"] for s in STRATA}, {s: cr[s]["abstention"] for s in STRATA})
            r.update(c)
            r.update(passes(c))
            crit[v] = c
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(T / "c7_ablation.csv", index=False)
    out["best_baseline_by_fold"] = sorted(set(d.best))
    # participant bootstrap (stratified) of the SAMA criteria -> CI and observed power at 36/stratum
    pm = {}
    for col in ("SAMA_hr", "hr_best"):
        pm[col] = d.assign(e=(d[col] - d.hr_ref).abs()).groupby("pid").e.mean()
    covp = d.assign(c=((d.hr_ref >= d.SAMA_lo) & (d.hr_ref <= d.SAMA_hi)).astype(float), a=(~d.SAMA_shown.astype(bool)).astype(float)).groupby("pid")[["c", "a"]].mean()
    by_s = {s: [p for p in pm["SAMA_hr"].index if stratum_of[p] == s] for s in STRATA}
    boots = []
    for _ in range(2000):
        ms, mb, cv, ab = {}, {}, {}, {}
        for s in STRATA:
            pick = RNG.choice(by_s[s], size=36, replace=True)
            ms[s], mb[s] = pm["SAMA_hr"][pick].mean(), pm["hr_best"][pick].mean()
            cv[s], ab[s] = covp.c[pick].mean(), covp.a[pick].mean()
        c = criteria(ms, mb, cv, ab)
        boots.append({**c, **passes(c)})
    b = pd.DataFrame(boots)
    h4 = {"point": crit["SAMA"], "ci95": {k: [float(b[k].quantile(0.025)), float(b[k].quantile(0.975))] for k in crit["SAMA"]},
          "p_pass": {k: float(b[k].mean()) for k in ("c1", "c2", "c3", "c4", "c5")},
          "p_all_five": float(b[["c1", "c2", "c3", "c4", "c5"]].all(1).mean()),
          "note": "observed (bootstrap) power of one cohort; network, sigma and gate not refitted (R3-16)"}
    pd.DataFrame([{"criterion": k, "point": crit["SAMA"][k], "lo95": h4["ci95"][k][0], "hi95": h4["ci95"][k][1]} for k in crit["SAMA"]]).to_csv(T / "c7_h4_criteria.csv", index=False)
    out["h4"] = h4
    # paired test on darker participants: best baseline MAE - SAMA MAE
    dk = by_s["darker"]
    out["h4_paired_darker"] = paired_test((pm["hr_best"][dk] - pm["SAMA_hr"][dk]).to_numpy(), "greater")
    # matched abstention: best baseline restricted to the same share per stratum by its own sigma
    mt = []
    for s in STRATA:
        q = d[d.stratum == s]
        a = 1 - q.SAMA_shown.mean()
        thr = np.quantile(q["POS+gate_sigma"], 1 - a) if a > 0 else np.inf
        keep_b = q["POS+gate_sigma"] <= thr
        mt.append({"stratum": s, "abstention": a, "SAMA_retained_mae": float((q.SAMA_hr - q.hr_ref).abs()[q.SAMA_shown.astype(bool)].mean()),
                   "POS_retained_mae_matched": float((q.hr_POS - q.hr_ref).abs()[keep_b].mean())})
    pd.DataFrame(mt).to_csv(T / "c7_matched_abstention.csv", index=False)
    out["matched_abstention"] = mt
    # abstention threshold trade-off (I-08)
    tt = []
    for hwmax in (3, 4, 5, 6, 8, 10):
        shown = (d.SAMA_hi - d.SAMA_lo) / 2 <= hwmax
        for s in STRATA:
            q = d.stratum == s
            tt.append({"halfwidth_max": hwmax, "stratum": s, "abstention": float(1 - shown[q].mean()),
                       "useful_yield": float((shown & ((d.SAMA_hr - d.hr_ref).abs() <= 5))[q].mean())})
    pd.DataFrame(tt).to_csv(T / "abstention_tradeoff.csv", index=False)
    # coverage under shift
    cs = []
    for (cond, s), g in shift_df.groupby(["condition", "stratum"]):
        g = g[np.isfinite(g.hr_ref)]
        cs.append({"condition": cond, "stratum": s, "n": len(g), "coverage_all": float(((g.hr_ref >= g.lo) & (g.hr_ref <= g.hi)).mean()),
                   "abstention": float(1 - g.shown.mean()), "mae": float((g.hr - g.hr_ref).abs().mean()),
                   "mae_pos": float((g.hr_pos - g.hr_ref).abs().mean())})
    for (light, s), g in d.groupby(["light", "stratum"]):
        cs.append({"condition": f"reference30|{light}|lossless (test)", "stratum": s, "n": len(g),
                   "coverage_all": float(((g.hr_ref >= g.SAMA_lo) & (g.hr_ref <= g.SAMA_hi)).mean()),
                   "abstention": float(1 - g.SAMA_shown.mean()), "mae": float((g.SAMA_hr - g.hr_ref).abs().mean()),
                   "mae_pos": float((g.hr_POS - g.hr_ref).abs().mean())})
    pd.DataFrame(cs).to_csv(T / "c7_coverage_shift.csv", index=False)
    frozen_q.to_csv(T / "c7_frozen_split_quantiles.csv", index=False)
    out["frozen_split"] = frozen_q.to_dict("records")
    hist.to_csv(T / "c7_training_history.csv", index=False)
    # H3s: phone B (adaptive) vs phone A (auto), POS on face mean, darker stratum; pooled lights primary (A4)
    ph = shift_df[shift_df.device.isin(["phoneA", "phoneB"])].copy()
    ph["e"] = (ph.hr_pos - ph.hr_ref).abs()
    pp = ph.groupby(["pid", "stratum", "device", "light"]).e.mean().unstack("device").reset_index()
    pp["diff"] = pp.phoneA - pp.phoneB
    h3 = {}
    for s in STRATA:
        q = pp[pp.stratum == s]
        pooled = q.groupby("pid")["diff"].mean()
        h3[s] = {"pooled": paired_test(pooled.to_numpy(), "greater"),
                 **{lt: paired_test(q[q.light == lt]["diff"].to_numpy(), "greater") for lt in ("standard", "dim")}}
    out["h3"] = h3
    budget = ph.groupby(["stratum", "device", "light"]).agg(electrons=("electrons_g", "mean"), gain=("gain", "mean"),
                                                            exposure_ms=("exposure_s", lambda x: 1000 * x.mean()),
                                                            mae_pos=("e", "mean")).reset_index()
    budget.to_csv(T / "h3_photon_budget.csv", index=False)
    # H5s
    out["h5"] = h5(parts, stratum_of, hrv, keep)
    out["e1"] = e1(parts, stratum_of, hrv, keep, d)
    out["table"] = tab.to_dict("records")
    return out


def h5(parts, stratum_of, hrv, keep):
    res, rows = {}, []
    for dev in ("reference", "reference30hq", "reference30", "reference30_500"):
        x, s = [], []
        for p in sorted(keep):
            h = hrv[p][dev]
            (e_r, c_r), (e_a, c_a) = h["rest"], h["mental_arithmetic"]
            if all(np.isfinite(v) and v > 0 for v in (e_r, c_r, e_a, c_a)):
                x.append([np.log(e_a) - np.log(e_r), np.log(c_a) - np.log(c_r), np.log(e_r), np.log(c_r)])
                s.append(stratum_of[p])
        x, s = np.array(x), np.array(s)
        r = {"overall": icc21(x[:, :2]), "level_rest": icc21(x[:, 2:]), "ba": bland_altman(x[:, 1], x[:, 0])}
        for st in STRATA:
            r[st] = icc21(x[s == st][:, :2])
        # bootstrap pass probability at 36 per stratum (observed power)
        hits = 0
        for _ in range(500):
            ok = True
            allx = []
            for st in STRATA:
                xs = x[s == st]
                pick = xs[RNG.integers(0, len(xs), 36)]
                allx.append(pick)
                ok &= icc21(pick[:, :2])["lo95"] >= 0.60
            ok &= icc21(np.vstack(allx)[:, :2])["icc"] >= 0.75
            hits += ok
        r["p_pass_36"] = hits / 500
        res[dev] = r
        rows.append({"device": dev, "icc_overall": r["overall"]["icc"], "lo95": r["overall"]["lo95"], "hi95": r["overall"]["hi95"],
                     **{f"icc_{st}": r[st]["icc"] for st in STRATA}, **{f"lo95_{st}": r[st]["lo95"] for st in STRATA},
                     "icc_rest_level": r["level_rest"]["icc"], "ba_bias": r["ba"]["bias"], "ba_lo": r["ba"]["loa_lo"],
                     "ba_hi": r["ba"]["loa_hi"], "p_pass_36": r["p_pass_36"], "n": len(x)})
    pd.DataFrame(rows).to_csv(T / "c8_h5_icc.csv", index=False)
    return res


def e1(parts, stratum_of, hrv, keep, d):
    """LOSO logistic regression, arithmetic vs rest, features HR and ln RMSSD (exploratory)."""
    from sklearn.linear_model import LogisticRegression
    hr = d[d.light == "standard"].groupby(["pid", "block"]).agg(ecg=("hr_ref", "mean"), cam=("SAMA_hr", "mean")).reset_index()
    X = {"ecg": [], "cam": []}
    y, pids = [], []
    for p in sorted(keep):
        for blk, lab in (("rest", 0), ("mental_arithmetic", 1)):
            q = hr[(hr.pid == p) & (hr.block == blk)]
            e, c = hrv[p]["reference"][blk]
            if q.empty or not (np.isfinite(e) and np.isfinite(c) and e > 0 and c > 0):
                continue
            X["ecg"].append([q.ecg.iat[0], np.log(e)])
            X["cam"].append([q.cam.iat[0], np.log(c)])
            y.append(lab)
            pids.append(p)
    y, pids = np.array(y), np.array(pids)
    out = {}
    for src in ("ecg", "cam"):
        Xs = np.array(X[src])
        correct = np.zeros(len(y), bool)
        for p in np.unique(pids):
            tr, te = pids != p, pids == p
            # within-person centring is not allowed (needs both blocks); features standardised on training set
            mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-9
            clf = LogisticRegression().fit((Xs[tr] - mu) / sd, y[tr])
            correct[te] = clf.predict((Xs[te] - mu) / sd) == y[te]
        out[src] = {"accuracy": float(correct.mean()), **{s: float(correct[[stratum_of[p] == s for p in pids]].mean()) for s in STRATA}}
    pd.DataFrame([{"features": k, **v} for k, v in out.items()]).to_csv(T / "c8_e1_loso.csv", index=False)
    return out


def error_budget(parts, keep, stratum_of, wf, m, shift, n_boot=40):
    """WP2 mixed model on POS window errors (lossless reference both lights + shift conditions)."""
    frames = []
    mm = m.copy()
    mm["err"] = np.abs(wf.hr_pos - wf.hr_ref)
    mm["condition"] = "reference30|" + mm.light + "|lossless"
    frames.append(mm[["pid", "stratum", "ita", "block", "light", "condition", "err"]])
    for key, (w, _) in shift.items():
        q = pd.DataFrame(w.meta)
        q["err"] = np.abs(w.hr_pos - w.hr_ref)
        q["condition"] = key
        frames.append(q[["pid", "stratum", "ita", "block", "light", "condition", "err"]])
    d = pd.concat(frames)
    d = d[d.pid.isin(keep) & np.isfinite(d.err)]
    dev = d.condition.str.split("|").str[0]
    codec = d.condition.str.split("|").str[2]
    comps = {"dim": (d.light == "dim").to_numpy(float), "webcam": (dev == "webcam").to_numpy(float),
             "phoneA": (dev == "phoneA").to_numpy(float), "phoneB": (dev == "phoneB").to_numpy(float),
             "codec500": (codec == "500").to_numpy(float), "codec250": (codec == "250").to_numpy(float)}
    ita10 = ((d.ita - 15) / 10).to_numpy()
    cols = [np.ones(len(d)), ita10] + list(comps.values()) + [ita10 * v for v in comps.values()]
    names = ["intercept", "ita10"] + [f"g_{k}" for k in comps] + [f"d_ita10x{k}" for k in comps]
    X = np.column_stack(cols)
    y = np.log1p(d.err.to_numpy())
    fit = lmm_crossed(y, X, [d.pid.to_numpy(), d.block.to_numpy()], names)
    tab = pd.DataFrame(fit.table())
    tab["message"] = fit.message
    tab.to_csv(T / "budget_mixed_model.csv", index=False)
    # stratum budgets: predicted mean log1p error -> bpm-equivalent increments, participant bootstrap
    ita_mid = {s: float(np.mean([parts.set_index("pid").ita[p] for p in keep if stratum_of[p] == s])) for s in STRATA}

    def budget_from(beta):
        rows = []
        for s in STRATA:
            z = (ita_mid[s] - 15) / 10
            base = beta[0] + beta[1] * z
            r = {"stratum": s, "base": float(np.expm1(base))}
            for j, k in enumerate(comps):
                r[k] = float(np.expm1(base + beta[2 + j] + beta[2 + len(comps) + j] * z) - np.expm1(base))
            rows.append(r)
        return rows
    est = budget_from(fit.beta)
    pids = np.array(sorted(keep))
    boots = []
    for _ in range(n_boot):
        pick = RNG.choice(pids, len(pids), replace=True)
        cnt = pd.Series(pick).value_counts()
        dd = d[d.pid.isin(cnt.index)]
        rep = np.repeat(np.arange(len(dd)), dd.pid.map(cnt).to_numpy())
        Xb = X[d.pid.isin(cnt.index).to_numpy()][rep]
        yb = y[d.pid.isin(cnt.index).to_numpy()][rep]
        pid_b = (dd.pid.to_numpy()[rep]).astype(str)
        fb = lmm_crossed(yb, Xb, [pid_b, dd.block.to_numpy()[rep]], names)
        boots.append(budget_from(fb.beta))
    rows = []
    for i, s in enumerate(STRATA):
        r = dict(est[i])
        for k in ["base"] + list(comps):
            v = [b[i][k] for b in boots]
            r[f"{k}_lo"], r[f"{k}_hi"] = float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))
        rows.append(r)
    pd.DataFrame(rows).to_csv(T / "budget_by_stratum.csv", index=False)
    # planted-parameter recovery on the same design
    rec = []
    beta_true = np.array([0.6, -0.08] + [0.3, 0.2, 0.15, 0.05, 0.25, 0.5] + [-0.05, 0.0, 0.0, 0.0, -0.06, -0.1])
    up = {p: RNG.normal(0, 0.4) for p in pids}
    vb = {b: RNG.normal(0, 0.15) for b in d.block.unique()}
    for _ in range(100):
        yy = X @ beta_true + d.pid.map(up).to_numpy() + d.block.map(vb).to_numpy() + RNG.normal(0, 0.5, len(d))
        f2 = lmm_crossed(yy, X, [d.pid.to_numpy(), d.block.to_numpy()], names)
        rec.append(((f2.beta - 1.96 * f2.se <= beta_true) & (beta_true <= f2.beta + 1.96 * f2.se)).tolist() + [f2.converged])
    rec = np.array(rec)
    cov = pd.DataFrame({"term": names, "coverage_95": rec[:, :-1].mean(0)})
    cov.to_csv(T / "budget_planted_recovery.csv", index=False)
    return {"fit_message": fit.message, "converged": fit.converged, "var_components": fit.var_components,
            "budget": rows, "planted_coverage_min": float(cov.coverage_95.min()), "planted_converged": float(rec[:, -1].mean()),
            "n_windows": int(len(d))}


def final_model(wf, m, keep, stratum_of, epochs, out_dir):
    """Deployment model: train on half of the analysable participants, calibrate on the other half."""
    tp, cal = [], []
    for s in STRATA:
        ids = RNG.permutation(sorted(p for p in keep if stratum_of[p] == s))
        tp += list(ids[: len(ids) // 2]); cal += list(ids[len(ids) // 2:])
    i_tp = np.where(m.pid.isin(tp).to_numpy())[0]
    i_tr = i_tp[(m.t0.to_numpy()[i_tp] % 30) == 0]
    i_cal = np.where(m.pid.isin(cal).to_numpy())[0]
    sub = lambda ii: subset(wf, {k: np.zeros(len(wf.hr_ref)) for k in METHODS}, ii)[0]  # noqa: E731
    sm = SamaModel().fit(sub(i_tr), m.group.to_numpy()[i_tr], sub(i_cal), m.stratum.to_numpy()[i_cal], m.pid.to_numpy()[i_cal],
                         cfg=N.TrainConfig(epochs=epochs, seed=99), sigma_train=sub(i_tp))
    meta = {"version": "0.1.0-insilico", "trained_on": "SAMA-Sim virtual SAMA-120 (SYNTHETIC), World A, seed 7",
            "n_params": N.n_params(sm.params), "fs": N.FS_NET, "window_s": 30, "research_only": True}
    model_io.save_bundle(out_dir, sm.params, sm.sigma_model, sm.gate, meta)
    return sm, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=str(ROOT / "results/cache/cohort_A_s7"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    T.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    parts, keep, stratum_of, d, shift_df, fq, hist, hrv, wf, m, bl, shift = run(a.cohort, a.folds, a.epochs)
    d.to_csv(ROOT / "results/cache/c7_test_predictions.csv.gz", index=False)
    shift_df.to_csv(ROOT / "results/cache/c7_shift_predictions.csv.gz", index=False)
    summ = evaluate(parts, stratum_of, d, shift_df, fq, hist, hrv, keep)
    summ["budget"] = error_budget(parts, keep, stratum_of, wf, m, shift)
    sm, meta = final_model(wf, m, keep, stratum_of, a.epochs, ROOT / "deploy/model")
    summ["final_model"] = meta
    summ["seconds"] = time.time() - t0
    (ROOT / "results/c7_summary.json").write_text(json.dumps(summ, indent=1, default=lambda o: o if not isinstance(o, (np.floating, np.integer, np.bool_)) else o.item()))
    print(json.dumps({k: summ[k] for k in ("h4", "h3", "best_baseline_by_fold")}, indent=1, default=float))

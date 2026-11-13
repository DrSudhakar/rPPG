"""A learned video -> skin-tone model, trained whenever reference labels exist.

Targets (knob `--target`):
  ita   colorimeter ITA° (ita.measured_file): the proposal's reference; regression
  monk  Monk consensus rating (1-10): available now from the pilot raters; regression on the
        ordinal scale, reported with exact/within-one accuracy

Inputs are per-participant colour features from the ITA capture segment (no image is kept):
raw and white-balanced cheek and forehead colour in linear RGB, chromaticities, CIELAB, luma
percentiles, face brightness, and the white reference itself (so the model can learn the camera's
colour cast). The default learner is ridge regression (closed form, a coefficient vector that can
be ported to the phone); `learner=gbr` uses scikit-learn gradient boosting for comparison.

Every model is evaluated by participant-level cross-validation before it is saved, and the saved
file carries those metrics, the number of participants, the target and the provenance label. With
the pilot's 10-25 participants this is a feasibility check of the pipeline, not a validated
instrument: the pilot report quotes the cross-validated error, never the training error.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ita_camera import ita_from_linear, skin_pixels, srgb_to_linear
from .video import crop

FEATURES = None  # filled by feature_names()


def _stats(px_lin: np.ndarray, tag: str) -> dict:
    med = np.median(px_lin, 0)
    s = med.sum() + 1e-9
    ita, L, a, b = ita_from_linear(med)
    luma = px_lin @ np.array([0.2126, 0.7152, 0.0722])
    p10, p50, p90 = np.percentile(luma, [10, 50, 90])
    return {f"{tag}_r": med[0], f"{tag}_g": med[1], f"{tag}_b": med[2], f"{tag}_rc": med[0] / s, f"{tag}_gc": med[1] / s,
            f"{tag}_L": L, f"{tag}_a": a, f"{tag}_bb": b, f"{tag}_ita_raw": ita, f"{tag}_l10": p10, f"{tag}_l50": p50,
            f"{tag}_l90": p90}


def frame_features(frame, box, white_box=None, reflectance: float = 0.9) -> dict | None:
    ch = skin_pixels(frame, box, ("cheeks",))
    fh = skin_pixels(frame, box, ("forehead",))
    if len(ch) < 50:
        return None
    feats = {}
    chl = srgb_to_linear(ch)
    fhl = srgb_to_linear(fh) if len(fh) >= 50 else chl
    feats.update(_stats(chl, "cheek"))
    feats.update(_stats(fhl, "fore"))
    if white_box is not None:
        w = np.median(srgb_to_linear(crop(frame, white_box).reshape(-1, 3)), 0)
        feats.update({"white_r": w[0], "white_g": w[1], "white_b": w[2]})
        feats.update({k.replace("cheek", "cheekwb"): v for k, v in _stats(chl / np.maximum(w, 1e-3) * reflectance, "cheek").items()})
    else:
        feats.update({"white_r": np.nan, "white_g": np.nan, "white_b": np.nan})
        feats.update({k.replace("cheek", "cheekwb"): np.nan for k in _stats(chl, "cheek")})
    return feats


def video_features(frames, boxes, white_box=None, reflectance: float = 0.9) -> dict:
    """Median of frame features over the sampled frames."""
    rows = []
    boxes = np.asarray(boxes, float)
    for i, fr in enumerate(frames):
        f = frame_features(fr, boxes if boxes.ndim == 1 else boxes[min(i, len(boxes) - 1)], white_box, reflectance)
        if f:
            rows.append(f)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.nanmedian([r[k] for r in rows])) if not all(np.isnan(r[k]) for r in rows) else float("nan") for k in keys}


class ItaModel:
    def __init__(self, target: str = "monk", learner: str = "ridge", ridge: float = 3.0):
        self.target, self.learner, self.ridge = target, learner, ridge
        self.names: list[str] = []
        self.mu = self.sd = self.coef = None
        self.gbr = None
        self.meta: dict = {}

    def _X(self, feats: list[dict]) -> np.ndarray:
        X = np.array([[f.get(k, np.nan) for k in self.names] for f in feats], float)
        return X

    def fit(self, feats: list[dict], y: np.ndarray):
        if not self.names:
            allk = sorted(set().union(*[f.keys() for f in feats]))
            # keep features present (finite) for every training participant
            self.names = [k for k in allk if all(np.isfinite(f.get(k, np.nan)) for f in feats)]
        X = self._X(feats)
        y = np.asarray(y, float)
        if self.learner == "gbr":
            from sklearn.ensemble import GradientBoostingRegressor
            self.gbr = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                                                 random_state=0).fit(X, y)
            return self
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        A = np.column_stack([np.ones(len(Z)), Z])
        R = self.ridge * np.eye(A.shape[1])
        R[0, 0] = 0
        self.coef = np.linalg.solve(A.T @ A + R, A.T @ y)
        return self

    def predict(self, feats: list[dict]) -> np.ndarray:
        X = self._X(feats)
        X = np.where(np.isfinite(X), X, (self.mu if self.mu is not None else np.nanmean(X, 0)))
        if self.learner == "gbr":
            p = self.gbr.predict(X)
        else:
            p = self.coef[0] + ((X - self.mu) / self.sd) @ self.coef[1:]
        return np.clip(p, 1, 10) if self.target == "monk" else p

    def to_dict(self):
        if self.learner == "gbr":
            raise ValueError("only the ridge model is saved as a portable file; use learner=ridge for deployment")
        return {"target": self.target, "learner": self.learner, "ridge": self.ridge, "features": self.names,
                "mu": self.mu.tolist(), "sd": self.sd.tolist(), "coef": self.coef.tolist(), "meta": self.meta}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["target"], d["learner"], d["ridge"])
        m.names, m.mu, m.sd, m.coef, m.meta = d["features"], np.array(d["mu"]), np.array(d["sd"]), np.array(d["coef"]), d.get("meta", {})
        return m

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def cross_validate(feats: list[dict], y: np.ndarray, target: str, learner: str = "ridge", folds: int = 5,
                   seed: int = 0, cuts=None) -> dict:
    """Participant-level K-fold (leave-one-out when n <= folds). Returns metrics on held-out
    participants and the out-of-fold predictions."""
    y = np.asarray(y, float)
    n = len(y)
    k = min(folds, n)
    order = np.random.default_rng(seed).permutation(n)
    fold = np.empty(n, int)
    fold[order] = np.arange(n) % k
    pred = np.full(n, np.nan)
    names = None
    for f in range(k):
        tr, te = fold != f, fold == f
        m = ItaModel(target, learner)
        if names:
            m.names = names
        m.fit([feats[i] for i in np.where(tr)[0]], y[tr])
        names = names or m.names
        pred[te] = m.predict([feats[i] for i in np.where(te)[0]])
    err = pred - y
    out = {"n": int(n), "folds": int(k), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
           "r2": float(1 - np.sum(err ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)), "pred": pred.tolist()}
    from scipy import stats as sst
    out["spearman"] = float(sst.spearmanr(pred, y).statistic) if n >= 4 else float("nan")
    if target == "monk":
        out["exact"] = float(np.mean(np.round(pred) == np.round(y)))
        out["within_one"] = float(np.mean(np.abs(np.round(pred) - np.round(y)) <= 1))
    if cuts is not None:
        lo, hi = cuts

        def strat(v):
            return np.where(v > hi, 2, np.where(v > lo, 1, 0)) if target == "ita" else np.where(np.round(v) <= lo, 2, np.where(np.round(v) <= hi, 1, 0))
        out["stratum_accuracy"] = float(np.mean(strat(pred) == strat(y)))
    return out

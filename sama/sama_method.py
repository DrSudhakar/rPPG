"""The SAMA method, steps 2-6, as a fit/predict pipeline over 30 s windows.

Step 1 (adaptive exposure) acts at capture (sim/samasim.choose_exposure, Android
CameraController); this module receives what the camera delivered.

Per window:
  step 2  linearise patch DN with the chart's tone curve; expected patch noise from the
          photon-transfer model;
  step 3  weight each patch by pulse-band power / expected noise variance;
  step 4  POS on the weighted relative RGB (candidate) -> 1D CNN with FiLM(ITA°, brightness),
          trained with group-DRO over 3 strata x 2 lights;
  step 5  sub-frame beat timing on the refined pulse -> IBIs (HRV);
  step 6  sigma_hat(x) from quality features -> Mondrian conformal gate per stratum.
Ablation flags switch each step off (used in C7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from . import methods as M
from . import net as N
from .calib import NoiseModel, ToneCurve
from .conformal import MondrianGate
from .signals import bandpass, clean_ibi, hr_from_signal, peaks_parabolic, rmssd_ms, window_starts

STRATUM_ID = {"lighter": 0, "middle": 1, "darker": 2}
LIGHT_ID = {"standard": 0, "dim": 1}


def pos_batch(rel: np.ndarray, fs: float, win_s: float = 1.6) -> np.ndarray:
    """Vectorised POS without post-processing. rel: (T, K, 3) positive traces -> (T, K)."""
    T, K, _ = rel.shape
    L = int(np.ceil(win_s * fs))
    if T <= L:
        return np.zeros((T, K))
    W = np.lib.stride_tricks.sliding_window_view(rel, L, axis=0)      # (T-L+1, K, 3, L)
    W = W[:-1] if W.shape[0] > T - L else W                            # windows m = 0..T-L-1
    C = W / W.mean(-1, keepdims=True)
    S0 = C[:, :, 1] - C[:, :, 2]
    S1 = -2 * C[:, :, 0] + C[:, :, 1] + C[:, :, 2]
    a = S0.std(-1) / np.maximum(S1.std(-1), 1e-12)
    h = S0 + a[..., None] * S1
    h = h - h.mean(-1, keepdims=True)
    H = np.zeros((T, K))
    for j in range(L):  # overlap-add: window m covers frames m..m+L-1
        H[j:j + h.shape[0]] += h[:, :, j]
    return H


@dataclass
class WindowFeatures:
    X: np.ndarray            # (N, 4, T_NET) network input
    cond: np.ndarray         # (N, 2) FiLM input: ITA/50, brightness/128
    ppg: np.ndarray          # (N, T_NET) reference pulse (training target)
    hr_ref: np.ndarray
    hr_pos: np.ndarray       # POS on face mean (the baseline)
    hr_cand: np.ndarray      # POS on weighted patches (steps 2-3)
    q: np.ndarray            # (N, k) quality features for sigma_hat
    meta: list = field(default_factory=list)


def _decimate(x: np.ndarray, fs: float) -> np.ndarray:
    f = int(round(fs / N.FS_NET))
    y = signal.resample_poly(x, 1, f, axis=-1) if f > 1 else x
    return y[..., :N.T_NET] if y.shape[-1] >= N.T_NET else np.pad(y, [(0, 0)] * (y.ndim - 1) + [(0, N.T_NET - y.shape[-1])])


def _z(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-9)


def window_features(rgb_patches: np.ndarray, fps: float, pix: np.ndarray, ita: float, tone: ToneCurve | None,
                    noise: NoiseModel | None, ref_rows: list, ppg: np.ndarray | None = None, light: str = "standard",
                    use_step2: bool = True, use_step3: bool = True, meta: dict | None = None) -> WindowFeatures:
    """Compute per-window inputs. ref_rows: [(a, b, hr_ref)] from pipeline.ref_hr_windows."""
    x = np.asarray(rgb_patches, float)
    lin = tone.linearise(x) if (use_step2 and tone is not None) else x
    face_raw = (x * (pix / pix.sum())[None, :, None]).sum(1)
    rows = {k: [] for k in ("X", "cond", "ppg", "hr_ref", "hr_pos", "hr_cand", "q")}
    pos_face = M.pos(face_raw, fps)
    for a, b, h_ref in ref_rows:
        seg = lin[a:b]
        mu = seg.mean(0)                                            # (P,3)
        rel = seg / np.maximum(mu[None], 1e-6)
        if use_step3:
            hp = pos_batch(seg, fps)                                # (T,P)
            hp = bandpass(hp.T, fps).T
            f, pw = signal.periodogram(hp, fs=fps, axis=0, nfft=2048)
            band = (f >= 0.7) & (f <= 3.0)
            # peak-concentrated band power: max of smoothed spectrum (robust to broadband noise)
            pband = pw[band].max(0)
            if noise is not None and use_step2:
                var_rel = noise.patch_var(mu[:, 1], pix) / np.maximum(mu[:, 1], 1e-6) ** 2
            else:
                var_rel = np.var(np.diff(rel[:, :, 1], axis=0), axis=0) / 2
            w = pband / np.maximum(var_rel, 1e-12)
            w = np.clip(w / (np.median(w) + 1e-12), 0, 20)
        else:
            w = pix.astype(float).copy()
        w = w / w.sum()
        wrel = (rel * w[None, :, None]).sum(1)                       # (T,3) weighted relative RGB
        cand = M.pos(wrel, fps)
        h_c, snr_c, sh_c = hr_from_signal(cand, fps)
        h_p, snr_p, sh_p = hr_from_signal(pos_face[a:b], fps)
        h_ch, _, _ = hr_from_signal(M.chrom(wrel, fps), fps)
        chans = np.vstack([bandpass(cand, fps), *(bandpass(wrel[:, c] - wrel[:, c].mean(), fps) for c in range(3))])
        rows["X"].append(_z(_decimate(chans, fps)))
        bright = float(face_raw[a:b, 1].mean())
        rows["cond"].append([ita / 50.0, bright / 128.0])
        rows["ppg"].append(_z(_decimate(ppg[a:b], fps)) if ppg is not None else np.zeros(N.T_NET))
        rows["hr_ref"].append(h_ref)
        rows["hr_pos"].append(h_p)
        rows["hr_cand"].append(h_c)
        lum = face_raw[a:b].mean(1)
        motion = float(np.std(np.diff(lum)) / max(lum.mean(), 1e-6))
        neff = float(1.0 / (w ** 2).sum())
        rows["q"].append([snr_c, sh_c, snr_p, motion * 1e3, bright, ita, neff, abs(h_c - h_ch), abs(h_c - h_p)])
    out = WindowFeatures(*(np.array(rows[k], float) for k in ("X", "cond", "ppg", "hr_ref", "hr_pos", "hr_cand", "q")))
    out.meta = [dict(meta or {}, t0=a / fps, light=light) for a, _, _ in ref_rows]
    return out


Q_NAMES = ["snr_cand", "sharp_cand", "snr_pos", "motion", "brightness", "ita", "n_eff", "d_cand_chrom", "d_cand_pos", "snr_out", "sharp_out", "d_out_cand"]


def hr_from_net_output(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(y) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    hr, snr, sh = zip(*(hr_from_signal(v, N.FS_NET) for v in y))
    return np.array(hr), np.array(snr), np.array(sh)


@dataclass
class SigmaModel:
    """sigma_hat(x): ridge regression of log(|error| + 0.5) on standardised quality features
    (portable to the phone as a coefficient vector; DECISIONS D-17)."""
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    coef: np.ndarray | None = None
    ridge: float = 1.0

    @staticmethod
    def _tf(q):
        q = np.where(np.isfinite(q), q, -50.0)
        q = q.copy()
        q[:, [0, 2, 9]] = np.clip(q[:, [0, 2, 9]], -20, 20)          # SNRs (dB)
        q[:, [7, 8, 11]] = np.log1p(np.clip(q[:, [7, 8, 11]], 0, 100))  # disagreements (bpm)
        return np.column_stack([q, q[:, 0] ** 2 / 20.0, q[:, 9] ** 2 / 20.0])

    def fit(self, q, y):
        Z = self._tf(q)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        Z = (Z - self.mu) / self.sd
        A = np.column_stack([np.ones(len(Z)), Z])
        R = self.ridge * np.eye(A.shape[1])
        R[0, 0] = 0
        self.coef = np.linalg.solve(A.T @ A + R, A.T @ y)
        return self

    def predict(self, q):
        Z = (self._tf(q) - self.mu) / self.sd
        return self.coef[0] + Z @ self.coef[1:]

    def to_dict(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist(), "coef": self.coef.tolist(), "features": Q_NAMES}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["mu"]), np.array(d["sd"]), np.array(d["coef"]))


@dataclass
class SamaModel:
    params: dict | None = None
    sigma_model: object = None
    gate: MondrianGate | None = None
    use_net: bool = True
    film: bool = True
    hist: list = field(default_factory=list)

    def fit(self, train: WindowFeatures, groups_train: np.ndarray, calib: WindowFeatures, strata_calib, pid_calib,
            cfg: N.TrainConfig = N.TrainConfig(), sigma_train: WindowFeatures | None = None, log=None):
        ok = np.isfinite(train.hr_ref)
        self.film = cfg.film  # one flag for training and prediction (R3-22)
        if self.use_net:
            self.params, self.hist = N.train(train.X[ok], train.cond[ok], train.ppg[ok], train.hr_ref[ok],
                                             groups_train[ok], cfg, log=log)
        st = sigma_train or train
        h, q = self.predict_point(st)
        m = np.isfinite(st.hr_ref) & np.isfinite(h)
        y = np.log(np.abs(h[m] - st.hr_ref[m]) + 0.5)
        self.sigma_model = SigmaModel().fit(q[m], y)
        hc, qc = self.predict_point(calib)
        sig = self.sigma(qc)
        self.gate = MondrianGate().fit(hc, calib.hr_ref, sig, strata_calib, pid_calib, seed=cfg.seed)
        return self

    def predict_point(self, wf: WindowFeatures):
        if self.use_net and self.params is not None:
            y = N.predict(self.params, wf.X, wf.cond, film=self.film)
            h, snr, sh = hr_from_net_output(y)
        else:
            h, snr, sh = wf.hr_cand, wf.q[:, 0], wf.q[:, 1]
        q = np.column_stack([wf.q, snr, sh, np.abs(h - wf.hr_cand)])
        q = np.where(np.isfinite(q), q, -50.0)
        return h, q

    def sigma(self, q):
        return np.exp(self.sigma_model.predict(q))

    def predict(self, wf: WindowFeatures, strata):
        h, q = self.predict_point(wf)
        sig = self.sigma(q)
        lo, hi, hw, shown = self.gate.decide(h, sig, strata)
        return {"hr": h, "sigma": sig, "lo": lo, "hi": hi, "halfwidth": hw, "shown": shown}


def beats_from_pulse(pulse: np.ndarray, fs: float, hr_hint: float | None = None) -> np.ndarray:
    """Step 5: band-pass 0.7-3.0 Hz, parabolic peak timing to 1 ms."""
    return peaks_parabolic(bandpass(pulse, fs), fs, hr_hint)


def hr_autocorr(pulse: np.ndarray, fs: float) -> float:
    """Beat rate from the first autocorrelation peak between 0.33 and 1.5 s (robust to the
    spectral-harmonic errors that would halve the minimum peak distance)."""
    x = bandpass(pulse, fs)
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[x.size - 1:]
    lo, hi = int(0.33 * fs), int(1.5 * fs)
    if hi >= ac.size:
        return float("nan")
    seg = ac[lo:hi]
    pk = signal.find_peaks(seg)[0]
    k = (pk[np.argmax(seg[pk])] if pk.size else np.argmax(seg)) + lo
    return float(60.0 * fs / k)


def rmssd_from_pulse(pulse: np.ndarray, fs: float, hr_hint=None) -> float:
    """Step 5 + HRV. The peak-distance hint comes from the camera signal itself (autocorrelation),
    never from the reference; hr_hint is accepted for API compatibility and ignored if it disagrees
    with the autocorrelation estimate by more than 25 %."""
    h_ac = hr_autocorr(pulse, fs)
    if hr_hint is None or not np.isfinite(hr_hint) or (np.isfinite(h_ac) and abs(hr_hint - h_ac) > 0.25 * h_ac):
        hr_hint = h_ac
    t = beats_from_pulse(pulse, fs, hr_hint)
    return rmssd_ms(clean_ibi(np.diff(t)))


def weighted_pulse(seg: np.ndarray, fps: float, use_step3: bool = True, pix: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Steps 3-4a on one segment (T, P, 3): noise-weighted patch combination then POS, at the full
    frame rate (for beat timing and HRV). Same weighting rule as window_features without a
    measured noise model. Returns (pulse, weights)."""
    mu = seg.mean(0)
    rel = seg / np.maximum(mu[None], 1e-6)
    if use_step3:
        hp = bandpass(pos_batch(seg, fps).T, fps).T
        f, pw = signal.periodogram(hp, fs=fps, axis=0, nfft=2048)
        band = (f >= 0.7) & (f <= 3.0)
        pband = pw[band].max(0)
        var_rel = np.var(np.diff(rel[:, :, 1], axis=0), axis=0) / 2
        w = pband / np.maximum(var_rel, 1e-12)
        w = np.clip(w / (np.median(w) + 1e-12), 0, 20)
    else:
        w = np.ones(seg.shape[1]) if pix is None else np.asarray(pix, float).copy()
    w = w / max(w.sum(), 1e-12)
    wrel = (rel * w[None, :, None]).sum(1)
    return M.pos(wrel, fps), w

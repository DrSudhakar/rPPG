"""SAMA step 4: a small 1D CNN with FiLM conditioning on (ITA°, brightness), trained with
group distributionally robust optimisation (Sagawa et al. 2020).

PyTorch is not installable in this environment (ISSUES I-23), so the network, its gradients and
Adam are written in numpy. Gradients are checked against finite differences in tests/test_net.py.

Architecture (input 4 x 450, i.e. 30 s at 15 Hz):
  conv7(4->C) -> FiLM1 -> ReLU -> conv7 dil2 (C->C) -> FiLM2 -> ReLU -> conv7 dil4 (C->C) -> ReLU
  -> conv1 (C->1) ; output = POS channel + conv output (residual refinement).
  FiLM: h = tanh(Wf c + bf); gamma_l = 1 + Wg_l h; beta_l = Wb_l h.
Loss per window: (1 - Pearson(y, ppg_ref)) + lam * CE(normalised band power of y, Gaussian label at
reference HR, sigma 2 bpm). Group-DRO weights q_g <- q_g exp(eta * L_g), renormalised each step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

FS_NET = 15.0
T_NET = 450
F_BPM = np.arange(42.0, 181.0, 1.0)


def dft_mats(T: int = T_NET, fs: float = FS_NET):
    t = np.arange(T) / fs
    f = F_BPM / 60.0
    return np.cos(2 * np.pi * np.outer(t, f)), np.sin(2 * np.pi * np.outer(t, f))


COS, SIN = dft_mats()


def n_params(params: dict) -> int:
    return int(sum(v.size for v in params.values()))


def init_params(C: int = 24, cin: int = 4, hf: int = 8, seed: int = 0) -> dict:
    r = np.random.default_rng(seed)

    def he(shape, fan):
        return (r.standard_normal(shape) * np.sqrt(2.0 / fan)).astype(np.float64)

    P = {"W1": he((C, cin, 7), cin * 7), "b1": np.zeros(C),
         "W2": he((C, C, 7), C * 7), "b2": np.zeros(C),
         "W3": he((C, C, 7), C * 7), "b3": np.zeros(C),
         "W4": he((1, C, 1), C) * 0.1, "b4": np.zeros(1),
         "Wf": he((hf, 2), 2), "bf": np.zeros(hf),
         "Wg1": np.zeros((C, hf)), "Wb1": np.zeros((C, hf)),
         "Wg2": np.zeros((C, hf)), "Wb2": np.zeros((C, hf))}
    return P


def conv_fwd(x, W, b, d):
    """x (B,Ci,T), W (Co,Ci,K), 'same' padding with dilation d."""
    B, Ci, T = x.shape
    Co, _, K = W.shape
    pad = d * (K - 1) // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    out = np.zeros((B, Co, T))
    for k in range(K):
        out += np.einsum("oi,bit->bot", W[:, :, k], xp[:, :, k * d:k * d + T], optimize=True)
    return out + b[None, :, None], xp


def conv_bwd(dout, xp, W, d):
    B, Co, T = dout.shape
    _, Ci, K = W.shape
    pad = d * (K - 1) // 2
    dW = np.zeros_like(W)
    dxp = np.zeros_like(xp)
    for k in range(K):
        sl = xp[:, :, k * d:k * d + T]
        dW[:, :, k] = np.einsum("bot,bit->oi", dout, sl, optimize=True)
        dxp[:, :, k * d:k * d + T] += np.einsum("oi,bot->bit", W[:, :, k], dout, optimize=True)
    db = dout.sum((0, 2))
    dx = dxp[:, :, pad:pad + T] if pad else dxp
    return dx, dW, db


def forward(P, x, c, cache=False):
    h = np.tanh(c @ P["Wf"].T + P["bf"])                      # (B,hf)
    g1, be1 = 1 + h @ P["Wg1"].T, h @ P["Wb1"].T              # (B,C)
    g2, be2 = 1 + h @ P["Wg2"].T, h @ P["Wb2"].T
    z1, xp1 = conv_fwd(x, P["W1"], P["b1"], 1)
    f1 = z1 * g1[:, :, None] + be1[:, :, None]
    a1 = np.maximum(f1, 0)
    z2, xp2 = conv_fwd(a1, P["W2"], P["b2"], 2)
    f2 = z2 * g2[:, :, None] + be2[:, :, None]
    a2 = np.maximum(f2, 0)
    z3, xp3 = conv_fwd(a2, P["W3"], P["b3"], 4)
    a3 = np.maximum(z3, 0)
    z4, xp4 = conv_fwd(a3, P["W4"], P["b4"], 1)
    y = x[:, 0, :] + z4[:, 0, :]
    if not cache:
        return y
    return y, dict(h=h, g1=g1, g2=g2, z1=z1, f1=f1, xp1=xp1, z2=z2, f2=f2, xp2=xp2, z3=z3, xp3=xp3, xp4=xp4, c=c)


def backward(P, dy, k):
    G = {}
    dz4 = dy[:, None, :]
    da3, G["W4"], G["b4"] = conv_bwd(dz4, k["xp4"], P["W4"], 1)
    dz3 = da3 * (k["z3"] > 0)
    da2, G["W3"], G["b3"] = conv_bwd(dz3, k["xp3"], P["W3"], 4)
    df2 = da2 * (k["f2"] > 0)
    dz2 = df2 * k["g2"][:, :, None]
    dg2 = (df2 * k["z2"]).sum(2)
    dbe2 = df2.sum(2)
    da1, G["W2"], G["b2"] = conv_bwd(dz2, k["xp2"], P["W2"], 2)
    df1 = da1 * (k["f1"] > 0)
    dz1 = df1 * k["g1"][:, :, None]
    dg1 = (df1 * k["z1"]).sum(2)
    dbe1 = df1.sum(2)
    _, G["W1"], G["b1"] = conv_bwd(dz1, k["xp1"], P["W1"], 1)
    h = k["h"]
    G["Wg1"], G["Wb1"] = dg1.T @ h, dbe1.T @ h
    G["Wg2"], G["Wb2"] = dg2.T @ h, dbe2.T @ h
    dh = dg1 @ P["Wg1"] + dbe1 @ P["Wb1"] + dg2 @ P["Wg2"] + dbe2 @ P["Wb2"]
    dpre = dh * (1 - h ** 2)
    G["Wf"], G["bf"] = dpre.T @ k["c"], dpre.sum(0)
    return G


def loss_and_grad_y(y, ppg, hr_ref, lam=0.5, sig_bpm=2.0):
    """Per-sample loss (B,) and dL/dy (B,T) for 1 - Pearson + lam * spectral CE."""
    yc = y - y.mean(1, keepdims=True)
    tc = ppg - ppg.mean(1, keepdims=True)
    ny = np.linalg.norm(yc, axis=1, keepdims=True) + 1e-9
    nt = np.linalg.norm(tc, axis=1, keepdims=True) + 1e-9
    r = (yc * tc).sum(1, keepdims=True) / (ny * nt)
    dr = tc / (ny * nt) - r * yc / ny ** 2
    dr = dr - dr.mean(1, keepdims=True)
    Fc, Fs = y @ COS, y @ SIN
    Pw = Fc ** 2 + Fs ** 2 + 1e-12
    S = Pw.sum(1, keepdims=True)
    q = np.exp(-0.5 * ((F_BPM[None, :] - hr_ref[:, None]) / sig_bpm) ** 2)
    q /= q.sum(1, keepdims=True)
    ce = -(q * np.log(Pw / S)).sum(1)
    dP = -q / Pw + 1.0 / S
    dy_spec = 2 * (dP * Fc) @ COS.T + 2 * (dP * Fs) @ SIN.T
    loss = (1 - r[:, 0]) + lam * ce
    return loss, -dr + lam * dy_spec


@dataclass
class TrainConfig:
    epochs: int = 8
    batch: int = 64
    lr: float = 2e-3
    lam: float = 0.5
    dro_eta: float = 0.02
    group_dro: bool = True
    film: bool = True
    seed: int = 0
    channels: int = 24


def train(X, cond, ppg, hr, groups, cfg: TrainConfig = TrainConfig(), log=None):
    """X (N,4,T), cond (N,2), ppg (N,T), hr (N,), groups (N,) ints 0..5."""
    rng = np.random.default_rng(cfg.seed)
    P = init_params(C=cfg.channels, cin=X.shape[1], seed=cfg.seed)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    ng = int(groups.max()) + 1
    q = np.ones(ng) / ng
    step = 0
    c_in = cond if cfg.film else np.zeros_like(cond)
    hist = []
    for ep in range(cfg.epochs):
        order = rng.permutation(len(X))
        ep_loss = []
        for s in range(0, len(order), cfg.batch):
            idx = order[s:s + cfg.batch]
            y, k = forward(P, X[idx], c_in[idx], cache=True)
            loss, dy = loss_and_grad_y(y, ppg[idx], hr[idx], cfg.lam)
            g = groups[idx]
            w = np.ones(len(idx)) / len(idx)
            if cfg.group_dro:
                Lg = np.array([loss[g == j].mean() if (g == j).any() else 0.0 for j in range(ng)])
                present = np.array([(g == j).any() for j in range(ng)])
                q = q * np.exp(cfg.dro_eta * Lg * present)
                q /= q.sum()
                cnt = np.bincount(g, minlength=ng)
                w = q[g] / np.maximum(cnt[g], 1)
                w = w / w.sum()
            G = backward(P, dy * w[:, None], k)
            step += 1
            for name in P:
                m[name] = 0.9 * m[name] + 0.1 * G[name]
                v[name] = 0.999 * v[name] + 0.001 * G[name] ** 2
                mh, vh = m[name] / (1 - 0.9 ** step), v[name] / (1 - 0.999 ** step)
                P[name] -= cfg.lr * mh / (np.sqrt(vh) + 1e-8)
            ep_loss.append(float((loss * w).sum()))
        hist.append({"epoch": ep, "loss": float(np.mean(ep_loss)), "q": q.round(3).tolist()})
        if log:
            log(hist[-1])
    return P, hist


def predict(P, X, cond, film=True, batch=256):
    c = cond if film else np.zeros_like(cond)
    if len(X) == 0:
        return np.zeros((0, X.shape[-1]))
    return np.concatenate([forward(P, X[i:i + batch], c[i:i + batch]) for i in range(0, len(X), batch)])


def save(P, path, meta=None):
    np.savez(path, **P, __meta__=json.dumps(meta or {}))


def load(path):
    d = np.load(path, allow_pickle=False)
    P = {k: d[k] for k in d.files if k != "__meta__"}
    meta = json.loads(str(d["__meta__"])) if "__meta__" in d.files else {}
    return P, meta

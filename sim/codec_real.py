"""Real-codec check of the codec surrogate (C5): render SAMA-Sim-style tiles to 720p30 video,
encode with real ffmpeg (FFV1 lossless, H.264 at 2000/1000/500/250 kbps, VP8 500/250), decode,
and measure how much of each tile's pulse amplitude survives, as a function of the amplitude in
DN. A gain that falls for small amplitudes is World A behaviour; a flat gain is World B.

SYNTHETIC: rendered tiles have a static texture and Gaussian pixel noise, not real skin, so this
says how the codec treats small uniform-texture modulations, not what it does to real faces.
Output: results/tables/codec_real_gain.csv, results/figures/fig_codec_gain.png
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
W, H, FPS, DUR = 1280, 720, 30, 40
GRID = (6, 5)
TILE = (48, 56)  # w, h px
AMPS = np.array([0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.6, 2.5, 4.0, 6.0])  # pulse amplitude (DN, SD of the sinusoid)
LEVELS = [("lossless", ["-c:v", "ffv1"], "mkv"), ("h264_2000", ["-c:v", "libx264", "-preset", "veryfast", "-b:v", "2000k", "-maxrate", "2000k", "-bufsize", "4000k"], "mp4"),
          ("h264_1000", ["-c:v", "libx264", "-preset", "veryfast", "-b:v", "1000k", "-maxrate", "1000k", "-bufsize", "2000k"], "mp4"),
          ("h264_500", ["-c:v", "libx264", "-preset", "veryfast", "-b:v", "500k", "-maxrate", "500k", "-bufsize", "1000k"], "mp4"),
          ("h264_250", ["-c:v", "libx264", "-preset", "veryfast", "-b:v", "250k", "-maxrate", "250k", "-bufsize", "500k"], "mp4"),
          ("vp8_500", ["-c:v", "libvpx", "-b:v", "500k", "-minrate", "500k", "-maxrate", "500k", "-deadline", "realtime", "-cpu-used", "8"], "webm"),
          ("vp8_250", ["-c:v", "libvpx", "-b:v", "250k", "-minrate", "250k", "-maxrate", "250k", "-deadline", "realtime", "-cpu-used", "8"], "webm")]


def tiles_xy():
    x0 = W // 2 - GRID[0] * TILE[0] // 2
    y0 = H // 2 - GRID[1] * TILE[1] // 2
    return [(x0 + i * TILE[0], y0 + j * TILE[1]) for j in range(GRID[1]) for i in range(GRID[0])]


def render(path, base_dn=(150.0, 112.0, 92.0), pix_sd=2.0, seed=0):
    rng = np.random.default_rng(seed)
    n = FPS * DUR
    t = np.arange(n) / FPS
    xy = tiles_xy()
    amp = np.resize(AMPS, len(xy)) * rng.uniform(0.9, 1.1, len(xy))
    phase = rng.uniform(0, 2 * np.pi, len(xy))
    tex_bg = rng.normal(0, 6, (H, W, 1))
    tex = rng.normal(0, 0.03, (GRID[1] * TILE[1], GRID[0] * TILE[0], 1))
    sig = np.array([0.33, 0.77, 0.53]) / 0.77
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-c:v", "ffv1", "-pix_fmt", "yuv444p", str(path)], stdin=subprocess.PIPE)
    x0, y0 = xy[0]
    for k in range(n):
        fr = np.clip(110 + tex_bg + rng.normal(0, pix_sd, (H, W, 1)), 0, 255)
        face = np.zeros((GRID[1] * TILE[1], GRID[0] * TILE[0], 3))
        for i, (x, y) in enumerate(xy):
            v = np.array(base_dn) + amp[i] * np.sqrt(2) * np.sin(2 * np.pi * 1.2 * t[k] + phase[i]) * sig
            face[y - y0:y - y0 + TILE[1], x - x0:x - x0 + TILE[0]] = v
        face = face * (1 + tex) + rng.normal(0, pix_sd, face.shape)
        fr = np.repeat(fr, 3, axis=2)
        fr[y0:y0 + face.shape[0], x0:x0 + face.shape[1]] = face
        p.stdin.write(np.clip(np.round(fr), 0, 255).astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    return amp, phase


def decode_tiles(path):
    """Stream-decode frame by frame (memory: one frame) and return tile means (T, tiles, 3)."""
    xy = tiles_xy()
    m = 6  # ignore tile borders (block edges)
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    out, nb = [], W * H * 3
    while True:
        buf = p.stdout.read(nb)
        if len(buf) < nb:
            break
        v = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        out.append([v[y + m:y + TILE[1] - m, x + m:x + TILE[0] - m].reshape(-1, 3).mean(0) for x, y in xy])
    p.wait()
    return np.array(out, float)


def amp_of(tr):
    from scipy import signal as S
    f, pw = S.periodogram(tr - tr.mean(0), fs=FPS, axis=0)
    k = np.argmin(np.abs(f - 1.2))
    return np.sqrt(2 * pw[k - 1:k + 2].sum(0) * (f[1] - f[0]))


def main():
    rows = []
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src.mkv"
        amp, _ = render(src)
        ref = decode_tiles(src)
        a_ref = amp_of(ref[:, :, 1])
        for name, args, ext in LEVELS:
            out = Path(td) / f"{name}.{ext}"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *args, "-pix_fmt", "yuv420p", "-an", str(out)], check=True)
            dec = decode_tiles(out)
            a = amp_of(dec[:, :, 1])
            resid = dec[:, :, 1] - ref[:, :, 1]
            for i in range(len(a)):
                rows.append({"level": name, "amp_dn": a_ref[i], "gain": a[i] / a_ref[i], "added_noise_sd_dn": float(resid[:, i].std())})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results/tables/codec_real_gain.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.figstyle import setup
    setup()
    fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
    for name, g in d.groupby("level"):
        g = g.sort_values("amp_dn")
        ax.plot(g.amp_dn, g.gain, marker="o", ms=3, lw=1.5, label=name)
    ax.set_xscale("log")
    ax.axhline(1, color="#888", lw=1, ls="--")
    ax.set_xlabel("Pulse amplitude in the tile (DN, green)")
    ax.set_ylabel("Amplitude kept after encoding")
    ax.set_title("Real ffmpeg on rendered tiles: how much pulse survives", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("SYNTHETIC rendering, real codec", x=0.01, ha="left", fontsize=9, color="#555")
    fig.savefig(ROOT / "results/figures/fig_codec_gain.png", dpi=180)
    print(d.groupby("level").apply(lambda g: pd.Series({"gain_small": g.gain[g.amp_dn < 0.3].mean(), "gain_large": g.gain[g.amp_dn > 1.5].mean(),
                                                         "noise_sd": g.added_noise_sd_dn.mean()})).round(3))


if __name__ == "__main__":
    main()

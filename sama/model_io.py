"""Model bundle I/O and the shared 30-patch face layout (same as app/static/app.js and the
Android PatchExtractor)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import net as N
from .conformal import MondrianGate

# patch layout in face-box coordinates (fractions): forehead 2 x 6, each cheek 3 x 3 = 30 patches
LAYOUT = ([(0.10 + 0.8 * c / 6, 0.06 + 0.12 * r, 0.8 / 6, 0.12) for r in range(2) for c in range(6)]
          + [(0.06 + 0.11 * c, 0.46 + 0.10 * r, 0.11, 0.10) for r in range(3) for c in range(3)]
          + [(0.61 + 0.11 * c, 0.46 + 0.10 * r, 0.11, 0.10) for r in range(3) for c in range(3)])
assert len(LAYOUT) == 30


def patch_means(box_rgb: np.ndarray, clip_dn: float = 250.0) -> np.ndarray:
    """(H, W, 3) RGB face box -> (30, 3) patch means, ignoring clipped/specular pixels (any
    channel >= clip_dn) and black pixels."""
    H, W, _ = box_rgb.shape
    out = np.zeros((30, 3))
    for i, (x, y, w, h) in enumerate(LAYOUT):
        p = box_rgb[int(y * H):max(int((y + h) * H), int(y * H) + 1), int(x * W):max(int((x + w) * W), int(x * W) + 1)].reshape(-1, 3)
        keep = (p.max(1) < clip_dn) & (p.min(1) > 2)
        out[i] = p[keep].mean(0) if keep.any() else p.mean(0)
    return out


def save_bundle(out_dir, params, sigma_model, gate: MondrianGate, meta: dict):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    N.save(params, out / "sama_net.npz", meta)
    (out / "sama_model.json").write_text(json.dumps({"gate": gate.to_dict(), "sigma": sigma_model.to_dict(),
                                                     "meta": meta, "layout": LAYOUT}, indent=1))


def load_bundle(model_dir) -> dict | None:
    from .sama_method import SigmaModel
    d = Path(model_dir)
    if not (d / "sama_model.json").exists():
        return None
    j = json.loads((d / "sama_model.json").read_text())
    params = N.load(d / "sama_net.npz")[0] if (d / "sama_net.npz").exists() else None
    onnx = next((str(p) for p in (d.parent / "onnx").glob("sama_net_fp32.onnx")), None)
    return {"gate": MondrianGate.from_dict(j["gate"]), "sigma": SigmaModel.from_dict(j["sigma"]), "params": params,
            "onnx": onnx, "meta": j["meta"]}

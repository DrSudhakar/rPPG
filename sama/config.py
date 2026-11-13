"""Configuration knobs for the whole pipeline (configs/sama.yaml).

    cfg = load()                                   # defaults
    cfg = load("configs/x.yaml", ["ita.source=none", "gate.type=pooled"])
    cfg.ita.source, cfg["ita"]["source"]           # both work
    snapshot(cfg, out_dir)                         # writes out_dir/config_used.yaml

Overrides use dotted keys; values are parsed as YAML (so `true`, `5`, `[1, 2]` work).
Unknown keys are rejected so a typo cannot silently do nothing. Enumerated values are checked
against CHOICES. Dependent switches are reconciled in `reconcile` and every change it makes is
recorded in cfg._notes, which is written into the snapshot.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs/sama.yaml"

CHOICES = {
    "project.provenance_label": {"REAL-PILOT", "REAL-PUBLIC", "SYNTHETIC"},
    "face.tracker": {"auto", "haar", "manual_box", "fixed_box"},
    "ita.source": {"measured", "camera", "predicted", "monk", "none"},
    "ita.camera.white_reference": {"card", "chart", "none"},
    "strata.by": {"ita", "monk", "none"},
    "raters.n": {1, 2},
    "raters.consensus": {"mean", "rater1"},
    "reference.type": {"oximeter_pleth", "oximeter_hr", "polar_rr", "ecg"},
    "reference.time_format": {"unix_s", "unix_ms", "relative_s"},
    "sync.method": {"wallclock", "xcorr"},
    "training.dro_groups": {"stratum", "dataset", "stratum_x_dataset", "none"},
    "training.loss": {"pearson_spectral", "spectral"},
    "training.public_ita": {"none", "camera_uncalibrated"},
    "gate.type": {"pooled", "mondrian", "off"},
    "gate.calibration": {"cross", "split"},
    "pilot.role": {"test_only", "calibrate"},
    "pilot.figure_codec": {"h264", "vp8"},
    "live.method": {"POS", "CHROM", "SAMA"},
    "recorder.input_format": {"v4l2", "dshow", "avfoundation", "file"},
    "recorder.codec": {"ffv1", "h264_lossless"},
}
METHOD_NAMES = {"POS", "CHROM", "GREEN", "ICA", "LGI", "PBV", "SAMA"}


class Cfg(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return v

    def __setattr__(self, k, v):
        self[k] = v

    def get_path(self, dotted: str):
        v = self
        for k in dotted.split("."):
            v = v[k]
        return v

    def set_path(self, dotted: str, value):
        keys = dotted.split(".")
        d = self
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                raise KeyError(f"unknown config key: {dotted}")
            d = d[k]
        if keys[-1] not in d:
            raise KeyError(f"unknown config key: {dotted}")
        d[keys[-1]] = _wrap(value)


def _wrap(x):
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    return x


def _merge(base: dict, new: dict, prefix=""):
    for k, v in new.items():
        key = f"{prefix}{k}"
        if k not in base:
            raise KeyError(f"unknown config key: {key}")
        if isinstance(v, dict) and isinstance(base[k], dict) and not key.endswith("reference.columns"):
            _merge(base[k], v, key + ".")
        else:
            base[k] = _wrap(v)


def validate(cfg: Cfg) -> list[str]:
    errs = []
    for key, allowed in CHOICES.items():
        try:
            v = cfg.get_path(key)
        except KeyError:
            errs.append(f"missing key {key}")
            continue
        if v not in allowed:
            errs.append(f"{key} = {v!r}; allowed: {sorted(map(str, allowed))}")
    bad = set(cfg.signal.methods) - METHOD_NAMES
    if bad:
        errs.append(f"signal.methods has unknown methods {sorted(bad)}")
    for k in ("ita.camera.regions",):
        if not set(cfg.get_path(k)) <= {"cheeks", "forehead"}:
            errs.append(f"{k} must be a subset of [cheeks, forehead]")
    from .datasets import NAMES as DATASET_NAMES
    unknown = [d for d in cfg.training.datasets if d not in DATASET_NAMES]
    if unknown:
        errs.append(f"training.datasets has unknown datasets {unknown}; allowed: {DATASET_NAMES}")
    if not cfg.training.datasets:
        errs.append("training.datasets is empty; list at least one of " + str(DATASET_NAMES))
    if cfg.quality.motion_warn > cfg.quality.motion_abstain:
        errs.append("quality.motion_warn must not exceed quality.motion_abstain")
    lo, hi = cfg.strata.ita_cuts
    if lo >= hi:
        errs.append("strata.ita_cuts must be increasing")
    return errs


def reconcile(cfg: Cfg) -> Cfg:
    """Make dependent switches consistent and record each change."""
    notes = []
    if cfg.ita.source == "none":
        if cfg.ita.conditioning:
            cfg.ita.conditioning = False
            notes.append("ita.source=none -> ita.conditioning forced to false")
        if cfg.strata.by == "ita":
            cfg.strata.by = "none"
            notes.append("ita.source=none -> strata.by forced to none")
    if cfg.ita.source == "monk":
        if cfg.ita.conditioning:
            cfg.ita.conditioning = False
            notes.append("ita.source=monk gives no ITA value -> ita.conditioning forced to false")
        if cfg.strata.by == "ita":
            cfg.strata.by = "monk"
            notes.append("ita.source=monk -> strata.by set to monk")
    if cfg.strata.by == "none":
        if cfg.gate.type == "mondrian":
            cfg.gate.type = "pooled"
            notes.append("strata.by=none -> gate.type mondrian replaced by pooled")
        if cfg.training.dro_groups in ("stratum", "stratum_x_dataset"):
            cfg.training.dro_groups = "dataset"
            notes.append("strata.by=none -> training.dro_groups set to dataset")
    if not cfg.training.group_dro and cfg.training.dro_groups != "none":
        notes.append("training.group_dro=false: dro_groups is ignored")
    if cfg.reference.type in ("oximeter_hr",):
        if cfg.sync.method == "xcorr":
            cfg.sync.method = "wallclock"
            notes.append("reference.type=oximeter_hr has no waveform -> sync.method set to wallclock")
        if cfg.hrv.enabled:
            notes.append("reference.type=oximeter_hr has no beats -> reference HRV unavailable (camera HRV still computed)")
        if cfg.training.loss == "pearson_spectral":
            notes.append("reference.type=oximeter_hr: pilot windows train with the spectral loss only")
    cfg["_notes"] = notes
    return cfg


def load(path: str | Path | None = None, overrides: list[str] | None = None) -> Cfg:
    base = _wrap(yaml.safe_load(DEFAULT.read_text()))
    if path and Path(path).resolve() != DEFAULT.resolve():
        _merge(base, yaml.safe_load(Path(path).read_text()) or {})
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        base.set_path(k.strip(), yaml.safe_load(v))
    cfg = reconcile(base)
    errs = validate(cfg)
    if errs:
        raise ValueError("invalid configuration:\n  " + "\n  ".join(errs))
    return cfg


def snapshot(cfg: Cfg, out_dir: str | Path, name: str = "config_used.yaml") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = _plain(copy.deepcopy(cfg))
    text = yaml.safe_dump(d, sort_keys=False)
    if cfg.get("_notes"):
        text = "# reconciled: " + "\n# reconciled: ".join(cfg["_notes"]) + "\n" + text
    (out / name).write_text(text)
    return out / name


def as_text(cfg: Cfg) -> str:
    text = yaml.safe_dump(_plain(cfg), sort_keys=False)
    if cfg.get("_notes"):
        text = "# reconciled: " + "\n# reconciled: ".join(cfg["_notes"]) + "\n" + text
    return text

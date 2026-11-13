"""Build results/manifest.json: every result file with its SHA-256, generating script, provenance
label and the commit; plus `numbers`, the named values the report quotes, each pointing to a file
and a key path. The report builder reads numbers only through this manifest."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = {"design_checks": "analysis/design_checks.py", "power_rerun": "analysis/design_checks.py", "h1_power_sensitivity": "analysis/design_checks.py",
       "h4_power_by_rule": "analysis/design_checks.py", "coverage_criterion": "analysis/design_checks.py", "h5_reactivity_icc": "analysis/design_checks.py",
       "recruitment_sim": "analysis/design_checks.py", "sim_calibration": "sim/calibrate.py", "codec_real_gain": "sim/codec_real.py",
       "c6_": "analysis/rehearsal.py", "c7_": "analysis/method_eval.py", "c8_": "analysis/method_eval.py", "budget_": "analysis/method_eval.py",
       "abstention_tradeoff": "analysis/method_eval.py", "h3_photon_budget": "analysis/method_eval.py", "export_parity": "analysis/export_models.py",
       "acceptance": "analysis/acceptance.py", "coverage": "tools/coverage.py", "fig_": "analysis/figures.py or analysis/rehearsal.py"}
LABEL = {"design_checks": "ASSUMPTION-DRIVEN (proposal assumptions) + LITERATURE", "sim_calibration": "SYNTHETIC (calibrated to LITERATURE anchor)",
         "codec_real_gain": "SYNTHETIC rendering, real codec", "export_parity": "SYNTHETIC inputs", "acceptance": "SYNTHETIC inputs; CPU proxy",
         "coverage": "software test", "recruitment_sim": "LITERATURE (Pawar et al. 2024)"}


def label_for(name):
    for k, v in LABEL.items():
        if name.startswith(k):
            return v
    return "SYNTHETIC; ASSUMPTION-DRIVEN where it depends on simulator premises"


def gen_for(name):
    for k, v in GEN.items():
        if name.startswith(k):
            return v
    return "see Makefile"


def get(path, keys):
    v = json.loads((ROOT / path).read_text())
    for k in keys:
        v = v[k]
    return v


def main():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    files = []
    for p in sorted(list((ROOT / "results/tables").glob("*")) + list((ROOT / "results/figures").glob("*")) + list((ROOT / "results").glob("*.json"))):
        if p.name == "manifest.json" or p.is_dir():
            continue
        files.append({"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                      "generated_by": gen_for(p.stem), "provenance": label_for(p.stem)})
    numbers = {}
    spec = json.loads((ROOT / "tools/numbers.json").read_text()) if (ROOT / "tools/numbers.json").exists() else {}
    for name, (path, keys) in spec.items():
        try:
            numbers[name] = {"value": get(path, keys), "file": path, "keys": keys}
        except Exception as e:  # noqa: BLE001
            numbers[name] = {"value": None, "file": path, "keys": keys, "error": str(e)}
    out = {"commit": commit, "files": files, "numbers": numbers}
    (ROOT / "results/manifest.json").write_text(json.dumps(out, indent=1, default=str))
    missing = [k for k, v in numbers.items() if v["value"] is None]
    print(f"{len(files)} files, {len(numbers)} named numbers, {len(missing)} missing: {missing}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()

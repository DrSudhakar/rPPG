"""Command-line entry point:  python -m sama <command> [--config FILE] [--set key=value ...]

Commands (in the order a pilot uses them):
  config        print the effective configuration (after overrides) and check it
  record        lab recorder: lossless webcam video + live BPM/HRV display + block markers
  serve         live interface only (browser camera; nothing recorded unless switched on)
  prepare       per session: signals, compression ladder, camera ITA, sync with the oximeter, QC
  qc            print the QC table of every prepared session
  datasets      list the public datasets found on disk and how each was recognised
  train         train the SAMA method on real data (public datasets and/or the pilot)
  test          test a trained bundle on whole datasets (cross-dataset generalisation)
  train-ita     train the video -> skin-tone model on the participants that have labels
  pilot         the section-5 pilot results, Figure 6 and the pilot report
  synth-pilot   make SYNTHETIC pilot sessions to test the pipeline (never real data)
  all           prepare + pilot (train first if the SAMA bundle is missing)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config as C


def _cfg(a):
    cfg = C.load(a.config, a.set)
    for n in cfg.get("_notes", []):
        print(f"[config] {n}")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sama", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["config", "record", "serve", "prepare", "qc", "datasets", "train",
                                        "test", "train-ita", "pilot", "synth-pilot", "all"])
    ap.add_argument("--config", default=None, help="YAML file merged over configs/sama.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one knob (repeatable)")
    ap.add_argument("--session", action="append", default=[], help="limit to these session folders or participant ids")
    ap.add_argument("--participant", help="record: participant id (e.g. P01)")
    ap.add_argument("--force", action="store_true", help="recompute cached steps")
    ap.add_argument("--out", help="output folder (pilot, train, synth-pilot)")
    ap.add_argument("--n", type=int, default=6, help="synth-pilot: number of participants")
    ap.add_argument("--block-s", type=float, default=45.0, help="synth-pilot: seconds per block")
    ap.add_argument("--target", choices=["monk", "ita"], default="monk", help="train-ita: label to learn")
    ap.add_argument("--learner", choices=["ridge", "gbr"], default="ridge", help="train-ita: model")
    ap.add_argument("--bundle", help="test: trained bundle to evaluate (default: training.bundle)")
    ap.add_argument("--datasets", help="test: comma-separated datasets to test on (default: training.datasets)")
    ap.add_argument("--root", help="datasets: folder to probe (default: paths.public_root)")
    ap.add_argument("--json", action="store_true", help="datasets: machine-readable output")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    cfg = _cfg(a)

    if a.command == "config":
        print(C.as_text(cfg))
        return 0

    if a.command == "synth-pilot":
        from .synthpilot import generate
        out = generate(a.n, a.out or "pilot_data_SYNTHETIC", block_s=a.block_s)
        if a.target == "ita":   # --target ita also writes a synthetic public-layout set for `train`
            from .synthpilot import generate_public_like
            generate_public_like(max(a.n * 2, 10), str(Path(out).parent / "data_SYNTHETIC/raw"))
        print(f"SYNTHETIC sessions in {out}. Analyse with:\n  python -m sama all --set paths.pilot_root={out} "
              f"--set project.provenance_label=SYNTHETIC --set paths.outputs=results/pilot_SYNTHETIC")
        return 0

    if a.command in ("record", "serve"):
        from app.server import run as run_server
        run_server(cfg, mode=a.command, participant=a.participant, host=a.host, port=a.port)
        return 0

    from .session import list_sessions
    sessions = list_sessions(cfg.paths.pilot_root)
    if a.session:
        keep = set(a.session)
        sessions = [s for s in sessions if s.name in keep or str(s) in keep]

    if a.command in ("prepare", "all"):
        from .pilot import prepare
        if not sessions:
            print(f"no sessions under {cfg.paths.pilot_root}/sessions")
        for s in sessions:
            st = prepare(s, cfg, force=a.force)
            q = st.get("qc", {})
            print(f"[prepare] {st['participant_id']}: usable={q.get('usable')} {'; '.join(q.get('reasons', []))}")
        if a.command == "prepare":
            return 0

    if a.command == "qc":
        rows = []
        for s in sessions:
            p = s / "derived/qc.json"
            if p.exists():
                q = json.loads(p.read_text())
                rows.append(f"{q['participant_id']:>8}  usable={str(q['usable']):5}  fps={q['fps']:.1f}  face={q['face_fraction']:.0%}  "
                            f"ref={q['reference_window_coverage']:.0%}  motion={q['median_motion']:.2f}  {'; '.join(q['reasons'])}")
            else:
                rows.append(f"{s.name:>8}  not prepared")
        print("\n".join(rows) or "no sessions")
        return 0

    if a.command == "datasets":
        from .datasets import probe, probe_text
        root = a.root or cfg.paths.public_root
        if a.json:
            print(json.dumps(probe(root, cfg.training.datasets or None, cfg.paths.pilot_root), indent=1))
        else:
            print(probe_text(root, cfg.training.datasets or None, cfg.paths.pilot_root))
        return 0

    if a.command in ("train", "all"):
        from .datasets import PUBLIC_NAMES
        bundle = Path(a.out or cfg.training.bundle)
        if a.command == "all" and (bundle / "sama_model.json").exists() and not a.force:
            print(f"[all] using existing SAMA bundle {bundle}")
        elif "SAMA" in cfg.signal.methods or a.command == "train":
            from .realtrain import public_items, run
            items = public_items(cfg) if set(cfg.training.datasets) & set(PUBLIC_NAMES) else []
            if "PILOT" in cfg.training.datasets:
                from .pilotitems import pilot_items
                items += pilot_items(cfg, sessions)
            if not items:
                print(f"[train] no training recordings found (public data under {cfg.paths.public_root}; see data/DATA_INVENTORY.md)")
                if a.command == "train":
                    return 1
            else:
                C.snapshot(cfg, bundle)
                run(items, cfg, bundle)
        if a.command == "train":
            return 0

    if a.command == "test":
        from .realtrain import evaluate
        names = [x.strip() for x in a.datasets.split(",") if x.strip()] if a.datasets else None
        evaluate(cfg, Path(a.bundle or cfg.training.bundle), names,
                 Path(a.out) if a.out else Path(cfg.paths.outputs) / "eval", sessions)
        return 0

    if a.command == "train-ita":
        from .pilotitems import train_ita
        res = train_ita(cfg, sessions, a.target, a.learner, Path(a.out) if a.out else None)
        print(json.dumps({k: v for k, v in res.items() if k != "pred"}, indent=1))
        return 0

    if a.command in ("pilot", "all"):
        from .pilot import results
        if a.out:
            cfg.paths.outputs = a.out
        S = results(cfg, cfg.paths.outputs)
        print((Path(cfg.paths.outputs) / "section5_filled.md").read_text())
        from .pilotreport import write_report
        write_report(S, cfg, Path(cfg.paths.outputs))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

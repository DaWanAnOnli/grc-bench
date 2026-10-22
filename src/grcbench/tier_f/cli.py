"""`./gb tier-f <subcommand>`

    list                         list Tier-F task directories and their gate status
    gates <task_dir|all> [--reps N] [--workers N]   run the quality gates (writes gates.json)
    check <task_dir>             run the ladder once on the task's reference
    sheet                        write the team review sheet handoff/review_sheets/tier_f_first_set.csv
    report                       write the first-set report reports/tier_f_first_set.md
    reserved                     write handoff/training_team/flowgraph_reserved_signatures.txt
    seed <example> [--out f]     export a usable GNU Radio example as a headless compact task flowgraph
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..common.io import read_json
from ..common.paths import HANDOFF
from . import gates, taskio
from .ladder import check


def _dirs(target: str) -> list[Path]:
    return taskio.task_dirs() if target == "all" else [Path(target).resolve()]


def main(argv: list[str]) -> int:
    if argv and argv[0] == "seed":
        from .seedtools import main as seed_main
        return seed_main(argv[1:])
    ap = argparse.ArgumentParser(prog="gb tier-f")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    g = sub.add_parser("gates")
    g.add_argument("target")
    g.add_argument("--reps", type=int, default=gates.REPS)
    g.add_argument("--workers", type=int, default=gates.WORKERS)
    c = sub.add_parser("check")
    c.add_argument("task_dir")
    sub.add_parser("sheet")
    sub.add_parser("report")
    sub.add_parser("reserved")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for d in taskio.task_dirs():
            g_path = d / "gates.json"
            status = ("PASS" if read_json(g_path)["passed"] else "FAIL") if g_path.exists() else "not run"
            t = taskio.load(d)
            print(f"{t['split']:5s} {t['id']:28s} {t['family']:15s} {t['level']}  gates: {status}")
        return 0
    if args.cmd == "gates":
        bad = 0
        for d in _dirs(args.target):
            res = gates.run(d, reps=args.reps, workers=args.workers)
            print(json.dumps({"task": res["task"], "passed": res["passed"],
                              **{k: v["passed"] for k, v in res["gates"].items()}}))
            bad += not res["passed"]
        return 1 if bad else 0
    if args.cmd == "check":
        d = Path(args.task_dir).resolve()
        res = check(taskio.read_grc(d, "reference.grc"), taskio.load(d))
        print(json.dumps(res, indent=1, default=str))
        return 0 if res["first_failed"] is None else 1
    if args.cmd == "sheet":
        path = HANDOFF / "review_sheets" / "tier_f_first_set.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["task_id", "split", "family", "level", "source", "instruction", "observation",
                        "oracle_checks", "gates", "known_limits", "reviewer_1_decision", "reviewer_1_edits",
                        "reviewer_2_decision", "reviewer_2_edits", "final_decision"])
            for d in taskio.task_dirs():
                t = taskio.load(d)
                gp = d / "gates.json"
                gsum = "not run"
                if gp.exists():
                    gj = read_json(gp)
                    gsum = ("PASS" if gj["passed"] else "FAIL") + " " + ", ".join(
                        f"{k}={'ok' if v['passed'] else 'fail'}" for k, v in gj["gates"].items())
                oracle = t.get("oracle") or {}
                checks = "; ".join([f"{c.get('group', 'main')}:{c['feature']}@{c['point']}"
                                    for c in oracle.get("checks", [])]
                                   + [f"{g.get('group', 'main')}:graph:{g['kind']}"
                                      for g in oracle.get("graph_checks", [])])
                w.writerow([t["id"], t["split"], t["family"], t["level"], t["source"].get("ref", t["source"]["kind"]),
                            t["instruction"], (t.get("observation") or {}).get("contract", ""), checks, gsum,
                            t["review"].get("notes", ""), "", "", "", "", ""])
        print(f"wrote {path}")
        return 0
    if args.cmd == "reserved":
        from ..common.paths import REPO
        from . import compose
        rows = compose.audit_items(REPO)["reserved"]
        path = HANDOFF / "training_team" / "flowgraph_reserved_signatures.txt"
        head = [f"# GRC-Bench reserved flowgraph signatures, signature v{compose.SIGNATURE_VERSION} (seed "
                f"{compose.SIGNATURE_SEED}, class map {compose.CLASS_MAP_DATE}); {len(rows)} entries.",
                "# One opaque digest per line: grcbench.tier_f.compose.signature_digest(compose.signature(graph)).",
                "# Every reference and variant cell of a GRC-Bench test or dev task in Tier F, and every start and",
                "# golden end-state cell of a Tier-A task (written by `./gb tier-f reserved`). A training-side",
                "# flowgraph whose digest is listed here is a functional",
                "# twin of such a task: do not emit it, whatever its signature_split. See flowgraph_composed_chains.md."]
        path.write_text("\n".join(head + sorted(r["digest"] for r in rows)) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {path} ({len(rows)} entries)")
        return 0
    if args.cmd == "report":
        from .set_report import write
        print(f"wrote {write()}")
        return 0
    return 2

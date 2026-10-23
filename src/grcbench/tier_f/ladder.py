"""The Tier-F validity ladder V1-V5 (proposal Table 5; eval plan §4.3).

Each rung is evaluated only if the previous rung passed, so a result shows exactly where a
flowgraph fails:
  V1 Parse     YAML parses and conforms to the .grc schema            (in-process, grcfile.v1_parse)
  V2 Load      GRC loads it; every block id exists in the pinned catalog (sandboxed worker)
  V3 Compile   GRC validation passes and GRC generates the Python top block (sandboxed worker)
  V4 Run       headless bounded run with liveness                     (sandboxed worker)
  V5 Behave    the task's oracle: signal checks on the captured signals (oracles.evaluate) and
               structural checks on the final graph (graph_checks.evaluate), in-process

    ./gb ladder <file.grc> [--task task.json] [--keep]

--task sets the task's directory to the folder holding task.json (as taskio.load does), so the
start graph (task["start_grc"]) is found for same_blocks_as_start / same_connections_as_start.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..common import sandbox
from ..common.paths import WORK, ensure
from . import graph_checks, grcfile, oracles
from ..common.paths import GNURADIO_CATALOG_SHA256
from .grc import catalog_hash, install_warm_cache

RUNGS = ["V1", "V2", "V3", "V4", "V5"]
DEFAULT_RUN = {"n_items": 8192, "min_items": 8192, "timeout_s": 10.0, "grace_s": 5.0}
_CATALOG_HASH: str | None = None


def pinned_catalog_hash() -> str:
    """Hash of the installed block catalog, computed once per process and compared with the pin."""
    global _CATALOG_HASH
    if _CATALOG_HASH is None:
        _CATALOG_HASH = catalog_hash()
    return _CATALOG_HASH


def catalog_matches_pin() -> bool:
    return pinned_catalog_hash() == GNURADIO_CATALOG_SHA256


def check(grc_text: str, task: dict[str, Any] | None = None, *, max_rung: int = 5,
          keep: bool = False, run_root: Path | None = None, timeout: float = 120.0) -> dict[str, Any]:
    """Run the ladder on one flowgraph (as .grc YAML text). Returns per-rung results."""
    task = task or {}
    result: dict[str, Any] = {"rungs": {}, "highest_passed": None, "first_failed": None,
                              "hallucinated_blocks": []}
    ok, errs, data = grcfile.v1_parse(grc_text)
    result["rungs"]["V1"] = {"passed": ok, "errors": errs}
    if ok and max_rung >= 2:
        run_dir = Path(tempfile.mkdtemp(prefix="ladder-", dir=ensure(run_root or WORK / "ladder")))
        try:
            (run_dir / "model.grc").write_text(grc_text, encoding="utf-8")
            obs_spec = task.get("observation", {})
            install_warm_cache(run_dir / "home")
            spec = {"grc_path": str(run_dir / "model.grc"), "run_dir": str(run_dir), "max_rung": max_rung,
                    "catalog_ok": catalog_matches_pin(), "points": obs_spec.get("points", []),
                    "run": {**DEFAULT_RUN, **obs_spec.get("run", {})}}
            (run_dir / "spec.json").write_text(json.dumps(spec))
            res = sandbox.run([sys.executable, "-m", "grcbench.tier_f._worker", str(run_dir / "spec.json")],
                              run_dir, timeout=timeout, cpu_seconds=int(timeout * 4))
            wres = _read_json(run_dir / "result.json")
            if wres is None:
                reason = "timed out" if res.timed_out else f"worker exited with code {res.returncode}"
                result["rungs"]["V2"] = {"passed": False, "errors": [f"harness: {reason}"],
                                        "stderr": res.stderr[-2000:]}
            else:
                result["rungs"].update(wres.get("rungs", {}))
                result["hallucinated_blocks"] = wres.get("hallucinated_blocks", [])
                v4 = result["rungs"].get("V4")
                if v4 is not None:
                    errs4 = _stderr_errors(res.stderr)
                    if errs4:
                        v4["passed"] = False
                        v4.setdefault("errors", []).extend(errs4)
                    if res.timed_out:
                        v4["passed"] = False
                        v4.setdefault("errors", []).append("harness timeout during the run")
                if max_rung >= 5 and v4 and v4.get("passed") and task.get("oracle"):
                    oracle = task["oracle"]
                    obs = {p.stem: np.load(p) for p in (run_dir / "obs").glob("*.npy")}
                    obs_errs = v4.get("observation_errors") or []
                    if oracle.get("checks"):
                        ev = oracles.evaluate(oracle, obs)
                    else:
                        ev = {"passed": True, "checks": [], "groups": {}}
                    g_res = graph_checks.evaluate(oracle.get("graph_checks", []), data, _start_data(task))
                    groups = dict(ev.get("groups", {}))
                    for g in g_res:
                        groups[g["group"]] = groups.get(g["group"], True) and g["passed"]
                    has_checks = bool(oracle.get("checks") or oracle.get("graph_checks"))
                    passed = has_checks and ev["passed"] and all(g["passed"] for g in g_res) and not obs_errs
                    result["rungs"]["V5"] = {**ev, "passed": passed, "errors": obs_errs,
                                             "graph_checks": g_res, "groups": groups}
                elif max_rung >= 5 and v4 and v4.get("passed"):
                    result["rungs"]["V5"] = {"passed": None, "errors": [], "note": "task has no V5 oracle"}
            if keep:
                result["run_dir"] = str(run_dir)
        finally:
            if not keep:
                shutil.rmtree(run_dir, ignore_errors=True)
    failed = [r for r in RUNGS if result["rungs"].get(r, {}).get("passed") is False]
    result["highest_passed"] = _highest_consecutive(result["rungs"])
    result["first_failed"] = failed[0] if failed else None
    return result


def _start_data(task: dict) -> dict | None:
    """The task's starting flowgraph (for graph checks such as same_blocks_as_start)."""
    if task.get("_start_data") is not None:
        return task["_start_data"]
    if task.get("_dir") and task.get("start_grc"):
        path = Path(task["_dir"]) / task["start_grc"]
        if path.exists():
            import yaml
            return yaml.safe_load(path.read_text(encoding="utf-8"))
    return None


def _highest_consecutive(rungs: dict) -> str | None:
    best = None
    for r in RUNGS:
        if rungs.get(r, {}).get("passed") is True:
            best = r
        else:
            break
    return best


BENIGN = ("vmcircbuf", "Using Volk machine", "fftw", "QStandardPaths", "propagate_size_hints",
          "This plugin does not support", "XDG_RUNTIME_DIR")


def _stderr_errors(stderr: str) -> list[str]:
    out = []
    for line in stderr.splitlines():
        if any(b in line for b in BENIGN):
            continue
        if ("Traceback" in line or "terminate called" in line or " ERROR" in line
                or line.startswith("Exception") or "Segmentation fault" in line):
            out.append(line.strip()[:300])
    return out[:10]


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_task_file(path: str | Path) -> dict[str, Any]:
    """Read a task.json for the CLI. Like taskio.load, it sets "_dir" to the file's folder, so
    start_grc resolves next to task.json and the *_as_start graph checks work from the CLI."""
    task_path = Path(path).resolve()
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task.setdefault("_dir", str(task_path.parent))
    return task


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb ladder")
    ap.add_argument("grc")
    ap.add_argument("--task", help="task.json with observation and oracle")
    ap.add_argument("--max-rung", type=int, default=5)
    ap.add_argument("--keep", action="store_true", help="keep the sandbox run directory")
    args = ap.parse_args(argv)
    task = load_task_file(args.task) if args.task else None
    res = check(Path(args.grc).read_text(encoding="utf-8"), task, max_rung=args.max_rung, keep=args.keep)
    print(json.dumps(res, indent=1, default=str))
    return 0 if res["first_failed"] is None else 1


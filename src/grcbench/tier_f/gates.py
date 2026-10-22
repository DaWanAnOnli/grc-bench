"""Quality gates for a Tier-F task (eval plan §5.4; approved plan Phase 3).

A task is accepted into GRC-Bench only if all gates pass:
  solvability   the reference passes V1-V6 in 10 of 10 runs, run 8 at a time (evaluation-level
                concurrency, so timing flakiness shows up);
  size          the reference, in compact form, is at most 3,000 cl100k tokens (fits the 8K answer cap);
  variants      at least two further valid solutions pass V1-V6 (the oracle accepts equivalent answers);
  start fails   for modify / repair / parameter-tune tasks the starting flowgraph fails (V1-V6), so a
                model that changes nothing cannot pass; repair tasks must fail at the rung the fault targets;
  mutation      at least five spec-breaking mutants of the reference that still reach V5; the V6 oracle
                must reject at least 90% of them.
Results go to <task_dir>/gates.json.
"""

from __future__ import annotations

import fcntl
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from ..common import tokens
from ..common.io import write_json
from ..common.paths import WORK
from . import edit, grcfile, taskio
from .ladder import check

REPS = 10
WORKERS = 8
MAX_REF_TOKENS = 3000
MIN_VARIANTS = 2
MIN_MUTANTS = 5
MIN_KILL = 0.90


def _slots() -> int:
    """How many gate runs may run at once: one per 8 CPUs (each run uses 8 sandboxed workers),
    at most 4. GRCBENCH_GATE_SLOTS overrides."""
    env = os.environ.get("GRCBENCH_GATE_SLOTS")
    if env:
        return max(1, int(env))
    return max(1, min(4, (os.cpu_count() or WORKERS) // WORKERS))


@contextmanager
def _slot():
    """Hold one of the gate-run slots, shared across all processes through lock files."""
    WORK.mkdir(parents=True, exist_ok=True)
    n = _slots()
    while True:
        for i in range(n):
            fh = open(WORK / f"tier_f_gates.{i}.lock", "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
            return
        time.sleep(2)


def _ok(res: dict) -> bool:
    return res["first_failed"] is None and res["highest_passed"] == "V6"


def run(task_dir: Path, reps: int = REPS, workers: int = WORKERS) -> dict[str, Any]:
    with _slot():
        return _run(task_dir, reps, workers)


def _run(task_dir: Path, reps: int, workers: int) -> dict[str, Any]:
    task = taskio.load(task_dir)
    ref_text = taskio.read_grc(task_dir, "reference.grc")
    out: dict[str, Any] = {"task": task["id"], "gates": {}}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        ref_runs = list(ex.map(lambda _: check(ref_text, task), range(reps)))
    n_ok = sum(_ok(r) for r in ref_runs)
    out["gates"]["solvability"] = {"passed": n_ok == reps, "passes": n_ok, "runs": reps,
                                   "failures": [r["rungs"].get(r["first_failed"] or "V6") for r in ref_runs if not _ok(r)][:3]}

    n_tok = tokens.count(ref_text)
    out["gates"]["size"] = {"passed": n_tok <= MAX_REF_TOKENS, "tokens": n_tok, "max": MAX_REF_TOKENS}

    var_paths = taskio.variants(task_dir)
    var_res = {p.name: check(p.read_text(encoding="utf-8"), task) for p in var_paths}
    out["gates"]["variants"] = {"passed": len(var_paths) >= MIN_VARIANTS and all(_ok(r) for r in var_res.values()),
                                "n": len(var_paths),
                                "results": {k: (v["highest_passed"], v["first_failed"]) for k, v in var_res.items()}}

    start = taskio.read_grc(task_dir, "start.grc")
    if task["family"] in ("modify", "repair", "parameter-tune"):
        if start is None:
            out["gates"]["start_fails"] = {"passed": False, "reason": "missing start.grc"}
        else:
            sres = check(start, task)
            expect = (task.get("fault") or {}).get("expected_rung")
            ok = sres["first_failed"] is not None and (expect is None or sres["first_failed"] == expect)
            out["gates"]["start_fails"] = {"passed": ok, "first_failed": sres["first_failed"], "expected": expect,
                                           "v6_groups": (sres["rungs"].get("V6") or {}).get("groups")}
    else:
        out["gates"]["start_fails"] = {"passed": True, "note": "build task: no starting flowgraph"}

    ref_data = yaml.safe_load(ref_text)
    mutants = []
    for i, op in enumerate(task.get("mutations", [])):
        try:
            text = grcfile.dump(edit.apply(ref_data, op))
        except Exception as exc:
            mutants.append({"i": i, "op": op, "error": f"could not apply: {exc}"})
            continue
        res = check(text, task)
        reached_v5 = (res["rungs"].get("V5") or {}).get("passed") is True
        killed = reached_v5 and (res["rungs"].get("V6") or {}).get("passed") is False
        mutants.append({"i": i, "op": op, "reached_v5": reached_v5, "killed": killed,
                        "first_failed": res["first_failed"]})
    valid = [m for m in mutants if m.get("reached_v5")]
    kill_rate = (sum(m["killed"] for m in valid) / len(valid)) if valid else 0.0
    out["gates"]["mutation"] = {"passed": len(valid) >= MIN_MUTANTS and kill_rate >= MIN_KILL,
                                "valid_mutants": len(valid), "kill_rate": round(kill_rate, 3),
                                "survivors": [m["op"] for m in valid if not m["killed"]],
                                "invalid_mutants": [m for m in mutants if not m.get("reached_v5")]}
    out["passed"] = all(g["passed"] for g in out["gates"].values())
    write_json(task_dir / "gates.json", out)
    return out

"""Golden replays and the Tier-A task gates (eval plan §4.4, §5.4; build plan Phase 4 "Gates").

A Tier-A task is accepted only if all four gates pass on it (`gates`):

  golden replay     the task's golden tool-call sequence (golden_calls.json) is replayed by a
                    pydantic-ai FunctionModel (models.scripted) through the REAL runner: GRC-Agent's
                    real executor and tools, in a fresh isolated run directory each time; it must
                    pass the grader in 10 of 10 replays (REPS, run WORKERS at a time), every
                    replayed call must have run, and its number of tool calls must lie in the
                    task's expected range (task.json "expected_tool_calls", default and bounds 8-20:
                    the build plan's 8-20 call range for a Tier-A task);
  do nothing        a model that answers "done" at once fails the task (eval plan §4.4: "an empty
                    or missing answer never counts as a success");
  wrong end state   the task's deliberately wrong end state (wrong_end_state.grc, optionally
                    wrong_end_state.json) is rejected by the grader (eval plan §4.4: "every task's
                    grader is tested against a deliberately wrong end state");
  grader reruns     the grader judged the golden replay's own final graph file (its SHA-256 equals
                    the episode's saved graph) by re-running it through the ladder or GRC itself,
                    never by trusting the agent's run.

golden_calls.json (task directory; never shown to the agent):
    {"calls": [{"tool": name, "args": {...} | "raw JSON text"} | {"text": "an intermediate reply"}, ...],
     "final_text": "the closing answer"}
Each call is one model request holding that one tool call; a {"text"} entry ends the agent's turn
with that reply (the scripted user answers it); "final_text" is the last reply.

gates.json and golden_end_state.grc (written by `gates(write=True)`): the gate results, and the final
graph of a passing golden replay, whose functional signature is reserved from training
(tier_f.compose.audit_items).

wrong_end_state.json (optional): {"final_text": str (default: the golden final text),
"files": {"<path in project>": "content"}, "library": {"<file name>": "content"}} added to the
synthetic end state built from the start files plus wrong_end_state.grc as the final graph.

The Tier-F setting's gate (`tier_f_gates`, build plan Decision 16 Oct "Solvability becomes golden
replay 10/10 through the real runner"): the Tier-F task's golden_calls.json (tier_f.golden: one
query_knowledge + one change_graph) is replayed through run_tier_f_sample by a FunctionModel (the
lookup, then the edit) and must pass V1-V6 in 10 of 10 samples; a do-nothing model must fail.

Gate replays use scripted models only, never a model under test, so they skip the run-log guard
(their run-log entry says so); nothing here ever sends a benchmark item to a model.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from ..common.io import read_json, sha256_file, write_json
from ..common.paths import WORK
from . import isolation, models
from .grader import EndState, acceptance_errors, grade
from .task import Task, load_task

GOLDEN_FILE = "golden_calls.json"
WRONG_GRC = "wrong_end_state.grc"
WRONG_JSON = "wrong_end_state.json"
GATES_FILE = "gates.json"
GOLDEN_END_GRC = "golden_end_state.grc"   # written by the gates: a passing golden replay's final graph
REPS = 10
WORKERS = 5
CALL_RANGE = (8, 20)
DO_NOTHING_TEXT = "done"
GATE_CONFIG = "gate"
GATES_ROOT = WORK / "tier_a" / "gates"


class GoldenError(ValueError):
    """A golden_calls.json file is unusable."""


def scripted_run_log_entry(spec: models.ModelSpec, what: str) -> dict[str, Any]:
    """The run-log entry of a gate replay: scripted models only (never a model under test)."""
    if spec.kind not in ("scripted", "wire_scripted"):
        raise ValueError("gate replays run scripted models only")
    return {"logged": False, "gate": what, "model": spec.describe(),
            "reason": "task gate: scripted FunctionModel, no model under test (eval plan §5.4)"}


# --------------------------------------------------------------------------------------------
# Golden sequences
# --------------------------------------------------------------------------------------------

def load_golden(task_dir: Path | str) -> dict[str, Any]:
    """A Tier-A task's golden_calls.json, checked (module docstring)."""
    path = Path(task_dir) / GOLDEN_FILE
    if not path.is_file():
        raise GoldenError(f"{Path(task_dir).name}: no {GOLDEN_FILE}")
    data = read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("calls"), list) or not data["calls"]:
        raise GoldenError(f"{path}: needs a non-empty 'calls' list")
    for i, entry in enumerate(data["calls"]):
        if not isinstance(entry, dict) or not (set(entry) <= {"tool", "args"} and "tool" in entry
                                               or set(entry) == {"text"}):
            raise GoldenError(f"{path}: calls[{i}] must be {{tool, args}} or {{text}}")
        if "tool" in entry and not isinstance(entry.get("args", {}), (dict, str)):
            raise GoldenError(f"{path}: calls[{i}].args must be an object or JSON text")
    if not str(data.get("final_text") or "").strip():
        raise GoldenError(f"{path}: needs a non-empty final_text")
    return data


def tool_call_count(golden: dict[str, Any]) -> int:
    return sum(1 for e in golden["calls"] if "tool" in e)


def golden_turns(golden: dict[str, Any]) -> list[dict[str, Any]]:
    """The FunctionModel turns replaying a golden sequence (one model request per entry)."""
    turns: list[dict[str, Any]] = []
    for i, entry in enumerate(golden["calls"]):
        if "tool" in entry:
            turns.append({"tool_calls": [{"name": entry["tool"], "args": entry.get("args", {}),
                                          "id": f"golden_{i:03d}"}]})
        else:
            turns.append({"text": entry["text"]})
    turns.append({"text": golden["final_text"]})
    return turns


def golden_model(task_dir: Path | str) -> models.ModelSpec:
    return models.scripted(golden_turns(load_golden(task_dir)))


def do_nothing_model() -> models.ModelSpec:
    """A model that answers "done" at once (and to every later turn)."""
    return models.scripted([{"text": DO_NOTHING_TEXT}], on_exhausted="repeat_last")


def expected_range(task: Task) -> tuple[int, int]:
    lo, hi = task.record.get("expected_tool_calls") or CALL_RANGE
    return int(lo), int(hi)


# --------------------------------------------------------------------------------------------
# Replays and synthetic end states
# --------------------------------------------------------------------------------------------

def replay(task: Task | Path | str, spec: models.ModelSpec, run_root: Path, *, sample: int = 0,
           what: str = "golden replay", ladder_check: Callable[..., dict] | None = None) -> tuple[Any, dict]:
    """Run one scripted episode of `task` through the real runner and grade it."""
    from .runner import run_episode
    task = task if isinstance(task, Task) else load_task(task)
    res = run_episode(task, spec, "benchmark", run_root, config_id=GATE_CONFIG, sample=sample,
                      run_log_entry=scripted_run_log_entry(spec, what))
    return res, grade(task, res, ladder_check=ladder_check)


def synthetic_end_state(task: Task, run_dir: Path, graph: Path | None, *, final_text: str,
                        files: dict[str, str] | None = None, library: dict[str, str] | None = None) -> EndState:
    """An end state built on disk without an episode: the task's start files in project/, `graph`
    as the final graph (replacing the active start graph, or project/flowgraph.grc for an untitled
    start), extra project files and library files."""
    run_dir = isolation.check_run_dir(run_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    lay = isolation.RunLayout(run_dir)
    lay.project.mkdir(parents=True)
    lay.hier_dir.mkdir(parents=True)
    for rel in task.start_files:
        dest = lay.project / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(task.start_dir / rel, dest)
    final = None
    if graph is not None:
        rel = (task.acceptance.get("graph") or {}).get("path") or (str(task.active_graph) if task.active_graph
                                                                   else "flowgraph.grc")
        final = lay.project / rel
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(graph, final)
    for rel, text in sorted((files or {}).items()):
        dest = lay.project / rel
        if not isolation.is_inside(dest, lay.project):
            raise ValueError(f"synthetic file {rel!r} is outside the project folder")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    for name, text in sorted((library or {}).items()):
        (lay.hier_dir / Path(name).name).write_text(text, encoding="utf-8")
    return EndState(project=lay.project, status="completed", end_reason="synthetic",
                    final_graph_path=str(final) if final else None, final_text=final_text,
                    library_dir=lay.hier_dir, run_dir=run_dir, synthetic=True)


def wrong_end_states(task: Task, run_root: Path, golden: dict[str, Any] | None = None,
                     ladder_check: Callable[..., dict] | None = None) -> list[dict[str, Any]]:
    """Grade the task's deliberately wrong end state(s); each must be rejected."""
    golden = golden or load_golden(task.task_dir)
    extra = read_json(task.task_dir / WRONG_JSON) if (task.task_dir / WRONG_JSON).is_file() else {}
    unknown = set(extra) - {"final_text", "files", "library", "graph"}
    if unknown:
        raise GoldenError(f"{WRONG_JSON}: unknown keys {sorted(unknown)}")
    wrong_grc = task.task_dir / WRONG_GRC
    if not wrong_grc.is_file():
        return []
    end = synthetic_end_state(task, run_root / "wrong_end_state", wrong_grc,
                              final_text=extra.get("final_text", golden["final_text"]),
                              files=extra.get("files"), library=extra.get("library"))
    g = grade(task, end, ladder_check=ladder_check)
    return [{"source": WRONG_GRC + (f" + {WRONG_JSON}" if extra else ""), "rejected": not g["passed"],
             "reasons": g["reasons"]}]


# --------------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------------

def _replay_record(res: Any, g: dict[str, Any], expected_calls: int) -> dict[str, Any]:
    ran = [c for c in res.tool_calls if not c.get("prefilled")]
    executed_ok = sum(1 for c in ran if c.get("execution") == "ok")
    graph_sha = res.final_graph_sha256
    graded_sha = (g.get("graph") or {}).get("sha256")
    ok = (res.status == "completed" and g["passed"] and len(ran) == expected_calls
          and executed_ok == expected_calls)
    reasons = list(g["reasons"])
    if res.status != "completed":
        reasons.insert(0, f"episode {res.status}: {res.end_reason}")
    if len(ran) != expected_calls or executed_ok != expected_calls:
        bad = [f"{c['tool']}:{c.get('execution') or c.get('validation')}" for c in ran if c.get("execution") != "ok"]
        reasons.append(f"{executed_ok}/{expected_calls} golden calls ran ok {bad}")
    return {"sample": res.sample, "passed": ok, "status": res.status, "tool_calls": len(ran),
            "requests": res.usage.get("requests"), "graph_sha256": graph_sha, "graded_sha256": graded_sha,
            "ladder_rerun": any(c["name"] == "ladder" for c in g["checks"]), "reasons": reasons[:10]}


def gates(task_dir: Path | str, *, reps: int = REPS, workers: int = WORKERS, run_root: Path | None = None,
          write: bool = False, keep: bool = False, ladder_check: Callable[..., dict] | None = None) -> dict[str, Any]:
    """Run every Tier-A gate on one task (module docstring); `write` stores <task>/gates.json."""
    task = load_task(task_dir)
    out: dict[str, Any] = {"task": task.id, "passed": False, "gates": {},
                           "acceptance_errors": acceptance_errors(task.acceptance)}
    try:
        golden = load_golden(task.task_dir)
    except GoldenError as exc:
        out["error"] = str(exc)
        return out
    if out["acceptance_errors"]:
        return out
    root = isolation.check_run_dir(Path(run_root) if run_root is not None else GATES_ROOT / task.id)
    n_calls = tool_call_count(golden)
    lo, hi = expected_range(task)
    spec = models.scripted(golden_turns(golden))
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(replay, task, spec, root, sample=k, ladder_check=ladder_check)
                       for k in range(reps)]
            nothing = pool.submit(replay, task, do_nothing_model(), root, sample=reps, what="do-nothing model",
                                  ladder_check=ladder_check)
            wrong = pool.submit(wrong_end_states, task, root, golden, ladder_check)
            replays = [f.result() for f in futures]
            nothing_res, nothing_grade = nothing.result()
            wrong_cases = wrong.result()
        records = [_replay_record(res, g, n_calls) for res, g in replays]
        passes = sum(r["passed"] for r in records)
        in_range = lo <= n_calls <= hi and CALL_RANGE[0] <= lo and hi <= CALL_RANGE[1]
        out["gates"]["golden_replay"] = {
            "passed": passes == reps and in_range, "runs": reps, "passes": passes, "tool_calls": n_calls,
            "expected_tool_calls": [lo, hi], "in_range": in_range,
            "failures": [r for r in records if not r["passed"]][:5]}
        out["gates"]["do_nothing"] = {"passed": not nothing_grade["passed"], "status": nothing_res.status,
                                      "end_reason": nothing_res.end_reason, "reasons": nothing_grade["reasons"][:10]}
        out["gates"]["wrong_end_state"] = {"passed": bool(wrong_cases) and all(c["rejected"] for c in wrong_cases),
                                           "cases": wrong_cases or [{"source": WRONG_GRC, "rejected": False,
                                                                     "reasons": [f"{WRONG_GRC} is missing"]}]}
        reruns = [r for r in records if r["graph_sha256"] and r["graph_sha256"] == r["graded_sha256"]]
        graph_task = any(k in task.acceptance for k in ("ladder", "blocks_present", "blocks_absent",
                                                        "connections", "graph_unchanged"))
        out["gates"]["grader_reruns"] = {
            "passed": graph_task and len(reruns) == len(records) and (
                "ladder" not in task.acceptance or all(r["ladder_rerun"] for r in records)),
            "detail": "the grader judged each replay's saved final graph file itself"
                      + (" and re-ran it through the ladder" if "ladder" in task.acceptance else "")}
        out["passed"] = all(g["passed"] for g in out["gates"].values())
        golden_graphs = [Path(res.final_graph_path) for (res, _), r in zip(replays, records)
                         if r["passed"] and res.final_graph_path and Path(res.final_graph_path).is_file()]
        if write and golden_graphs:     # the golden end state, for the reserved flowgraph signatures
            shutil.copyfile(golden_graphs[0], task.task_dir / GOLDEN_END_GRC)
            out["golden_end_state_sha256"] = sha256_file(task.task_dir / GOLDEN_END_GRC)
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
    if write:
        write_json(task.task_dir / GATES_FILE, out)
    return out


def tier_f_gates(task_dir: Path | str, *, reps: int = REPS, workers: int = WORKERS, run_root: Path | None = None,
                 keep: bool = False) -> dict[str, Any]:
    """The Tier-F setting's golden-replay gate (module docstring) on one Tier-F task directory."""
    from ..tier_f import golden as tf_golden
    from .tier_f_mode import load_tier_f_task, run_tier_f_sample
    task = load_tier_f_task(task_dir)
    calls = tf_golden.load_golden_calls(task_dir)
    root = isolation.check_run_dir(Path(run_root) if run_root is not None else GATES_ROOT / f"tier_f-{task['id']}")
    spec = models.scripted(tf_golden.golden_turns(calls))
    nothing_spec = models.scripted([{"text": DO_NOTHING_TEXT}])

    def one(k: int, s: models.ModelSpec, what: str) -> Any:
        return run_tier_f_sample(task_dir, s, root, config_id=GATE_CONFIG, sample=k, run_log_entry=scripted_run_log_entry(s, what))
    out: dict[str, Any] = {"task": task["id"], "passed": False, "gates": {}}
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(one, k, spec, "tier_f golden replay") for k in range(reps)]
            nothing_future = pool.submit(one, reps, nothing_spec, "tier_f do-nothing model")
            results = [f.result() for f in futures]
            nothing = nothing_future.result()
        recs = []
        for r in results:
            rec = r.tier_f or {}
            g = rec.get("grade") or {}
            recs.append({"sample": r.sample, "passed": bool(g.get("passed")), "status": r.status,
                         "end_reason": r.end_reason, "first_failed": g.get("first_failed"),
                         "lookup_recall": (rec.get("lookup") or {}).get("reference_recall"),
                         "new_block_recall": (rec.get("lookup") or {}).get("new_block_recall"),
                         "edit_accepted": (rec.get("edit") or {}).get("accepted"),
                         "well_formed": (rec.get("edit") or {}).get("well_formed")})
        passes = sum(r["passed"] for r in recs)
        out["gates"]["tier_f_replay"] = {"passed": passes == reps, "runs": reps, "passes": passes,
                                         "lookup_recall": sorted({r["lookup_recall"] for r in recs}, key=str),
                                         "failures": [r for r in recs if not r["passed"]][:5]}
        ng = (nothing.tier_f or {}).get("grade") or {}
        out["gates"]["do_nothing"] = {"passed": not ng.get("passed"), "failure": ng.get("failure"),
                                      "end_reason": nothing.end_reason}
        out["passed"] = all(g["passed"] for g in out["gates"].values())
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
    return out


def summary_line(result: dict[str, Any]) -> str:
    g = result.get("gates", {})
    parts = []
    for name, rec in g.items():
        extra = f" {rec['passes']}/{rec['runs']}" if "runs" in rec else ""
        parts.append(f"{name}={'pass' if rec['passed'] else 'FAIL'}{extra}")
    if result.get("error"):
        parts.append(f"error: {result['error']}")
    if result.get("acceptance_errors"):
        parts.append(f"acceptance errors: {result['acceptance_errors']}")
    return f"{result['task']:28s} {'PASS' if result.get('passed') else 'FAIL'}  " + " ".join(parts)


def dumps(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=1, sort_keys=True, default=str)


__all__ = ["CALL_RANGE", "GOLDEN_FILE", "GoldenError", "REPS", "WRONG_GRC", "WRONG_JSON", "do_nothing_model",
           "expected_range", "gates", "golden_model", "golden_turns", "load_golden", "replay",
           "summary_line", "synthetic_end_state", "tier_f_gates", "tool_call_count",
           "wrong_end_states"]

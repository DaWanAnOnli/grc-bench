"""`./gb tier-a <subcommand>`

    conformance [--network host] [--no-report]
                       build GRC-Agent's executor in an isolated episode process, compare its tools
                       and system prompt with Qoherent's SFT rows, exercise the offline web tools and
                       the sandboxed shell tools; writes reports/tier_a_conformance.md (exit 1 on FAIL).
                       The episode has no network by default; --network host shares the host's.
    vectors            make the read-only copies of GRC-Agent's lexical knowledge databases
    run-env <run_dir> [--base-url URL] [--model NAME]
                       prepare an episode run directory and print its environment (debugging)
    run <task_dir>... [model] [--setting S] [--config-id ID] [--reason TEXT] [--samples N]
                       [--mode evaluation|collection] [--run-root DIR] [--no-grade]
                       Tier-A episodes (runner.run_tasks), each graded by grader.grade: one run-log
                       guard for the batch (a test-split run needs a config id on
                       docs/preregistration/run_list.yaml and is logged before it starts; the
                       default config id "adhoc" is never on it); one line per episode
    gates <task_dir|all> [--reps N] [--workers N] [--no-write] [--keep] [--json]
                       the Tier-A task gates (replay.gates: golden replay 10/10 within 8-20 calls,
                       do-nothing fails, wrong end state rejected, grader re-runs the final graph);
                       writes <task_dir>/gates.json; "all" = every items/tier_a task
    smoke [--base-url URL --model NAME | --start-server [--threads N]] [--reps N] [--only a,b]
          [--no-golden] [--no-probe] [--tier-f-tasks a,b] [--tier-f-samples N] [--note TEXT]...
          [--report | --no-report] | --report-from <summary.json>
                       the smoke test on FIXTURE tasks only (tests/fixtures, never benchmark items), a
                       plumbing test, never a model result (smoke.py): golden FunctionModel replays of
                       every fixture task that has a golden_calls.json (Tier A and the Tier-F
                       setting), graded; with --start-server the grammar probe (does --jinja
                       grammar-constrain tool calls: B0 vs. B2) and the pinned llama-server with the
                       smoke model; with a server, one graded real-model episode per Tier-A fixture
                       and N Tier-F samples per --tier-f-tasks fixture; writes
                       reports/tier_a_smoke.md when a real model ran
    tier-f-golden <task_dir|all> [--no-write] [--no-gate] [--reps N] [--workers N]
                       derive the Tier-F task's golden_calls.json (tier_f.golden), verify it through
                       GRC-Agent's change_graph, write it, then run the "tier_f replay" gate
                       (replay.tier_f_gates); "all" = every items/tier_f task
    tier-f-sample <task_dir> [model] [--config-id ID] [--sample K] [--setting S]
                       one Tier-F sample ("one lookup + one edit"), graded; prints its record
    tier-f <task_dir>... --config-id ID --reason TEXT [model] [--setting S] [--samples N]
                       Tier-F samples in the "one lookup + one edit" setting (tier_f_mode), graded
                       by V1 (well-formed edit) + the ladder V2-V5
    trace <run_dir> [--no-thinking]
                       print the episode's SFT-format row (trace.export_run_dir)

    [model] = --base-url URL --model NAME (llama-server, the default endpoint) | --uds PATH (a
              llama-server listening on a UNIX socket; the episode then has no network at all) |
              --scripted turns.json (models.scripted; plumbing tests only)

Subcommands are looked up lazily in SUBCOMMANDS ("module:function", each taking argv), so every
Tier-A workstream adds its own entry without importing the others.
"""

from __future__ import annotations

import importlib
import json
import sys
from typing import Any

SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "conformance": ("grcbench.tier_a.cli:_conformance", "tools/prompt conformance + isolation report"),
    "vectors": ("grcbench.tier_a.cli:_vectors", "read-only copies of the lexical knowledge DBs"),
    "run-env": ("grcbench.tier_a.cli:_run_env", "prepare a run directory and print its environment"),
    "run": ("grcbench.tier_a.cli:_run", "run Tier-A episodes (guarded, logged)"),
    "tier-f": ("grcbench.tier_a.cli:_tier_f", "run Tier-F samples in GRC-Agent's harness (one lookup + one edit)"),
    "trace": ("grcbench.tier_a.cli:_trace", "print an episode's SFT-format trace row"),
    "gates": ("grcbench.tier_a.cli:_gates", "Tier-A task gates (golden replay, do-nothing, wrong end state)"),
    "first-set": ("grcbench.tier_a.first_set:main", "write the Tier-A first set under items/tier_a"),
    "sheet": ("grcbench.tier_a.spec_drafts:main", "test spec drafts, the Tier-A review sheet and first-set report"),
    "smoke": ("grcbench.tier_a.cli:_smoke", "smoke test on fixture tasks (golden replays; optional real model)"),
    "tier-f-golden": ("grcbench.tier_a.cli:_tier_f_golden", "derive + verify Tier-F golden_calls.json, replay gate"),
    "tier-f-sample": ("grcbench.tier_a.cli:_tier_f_sample", "one graded Tier-F sample in GRC-Agent's harness"),
}


def _conformance(argv: list[str]) -> int:
    from .conformance import REPORT, run_conformance
    network = "host" if "--network=host" in argv or ("--network" in argv and "host" in argv) else "none"
    result = run_conformance(report="--no-report" not in argv, network=network)
    checks = result["checks"]
    for name, ok in checks["ok"].items():
        print(f"{'pass' if ok else 'FAIL'}  {name}")
    if checks["tool_diffs"]:
        print(f"{len(checks['tool_diffs'])} tool-definition difference(s) from the SFT rows "
              f"({len(checks['tool_diffs_unexplained'])} unexplained)")
    print(f"run directory: {result['run_dir']}")
    print(f"report: {REPORT}")
    return 0 if checks["passed"] else 1


def _vectors(argv: list[str]) -> int:
    from .isolation import ensure_vectors_cache
    print(ensure_vectors_cache())
    return 0


def _run_env(argv: list[str]) -> int:
    import argparse
    from pathlib import Path

    from .isolation import DEFAULT_BASE_URL, DEFAULT_MODEL, IsolationError, make_run_env
    p = argparse.ArgumentParser(prog="gb tier-a run-env")
    p.add_argument("run_dir", help="episode run directory under ~/grc-bench-work")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="llama-server OpenAI-compatible endpoint")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model name written to the run .env")
    a = p.parse_args(argv)
    try:
        env = make_run_env(Path(a.run_dir), base_url=a.base_url, model_name=a.model)
    except (IsolationError, ValueError) as exc:
        print(f"refused: {exc}")
        return 2
    print(json.dumps(env, indent=1, sort_keys=True))
    return 0


def _batch_args(prog: str, argv: list[str]) -> tuple[Any, Any]:
    import argparse
    from pathlib import Path

    from . import models
    from .isolation import DEFAULT_BASE_URL, DEFAULT_MODEL
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument("tasks", nargs="+", help="task directories")
    p.add_argument("--config-id", default="adhoc", help="configuration id (test split: on the run list)")
    p.add_argument("--reason", default=f"manual {prog}", help="reason recorded in the test-run log")
    p.add_argument("--no-grade", action="store_true", help="do not grade the end states (run only)")
    p.add_argument("--setting", default="benchmark", choices=["benchmark", "grc_agent_defaults"])
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--mode", default="evaluation", choices=["evaluation", "collection"])
    p.add_argument("--run-root", default=None)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--uds", default=None)
    p.add_argument("--api-base", default=None, help="hosted OpenAI-compatible API (through relay.py; needs --model)")
    p.add_argument("--api-key-env", default=None, help="NAME of the environment variable holding the API key")
    p.add_argument("--scripted", default=None, help="JSON file with scripted turns (plumbing only)")
    p.add_argument("--no-stream", action="store_true", help="do not stream the model requests (models.py)")
    a = p.parse_args(argv)
    if a.scripted:
        spec = models.scripted(json.loads(Path(a.scripted).read_text(encoding="utf-8")))
    elif a.api_base or a.api_key_env:
        if not (a.api_base and a.api_key_env and a.model != DEFAULT_MODEL):
            p.error("a hosted model needs --api-base, --api-key-env and --model")
        spec = models.hosted(a.api_base, a.model, a.api_key_env, stream=False if a.no_stream else None)
    else:
        spec = models.openai_compatible(a.base_url, a.model, uds_path=a.uds, stream=False if a.no_stream else None)
    return a, spec


def _run(argv: list[str]) -> int:
    from pathlib import Path

    from ..runlog.log import RunRefused
    from .relay import RelayError
    from .runner import run_tasks
    from .task import TaskError
    a, spec = _batch_args("gb tier-a run", argv)

    def show(res: Any) -> None:
        verdict = "" if res.grade is None else ("PASS" if res.grade.get("passed") else "fail")
        print(f"{res.task_id:28s} s{res.sample:02d}  {verdict:4s} {res.status:20s} {res.end_reason:28s} "
              f"requests={res.usage.get('requests')}  {res.run_dir}")
        if res.grade is not None and not res.grade.get("passed"):
            for reason in res.grade.get("reasons", [])[:8]:
                print(f"    - {reason}")
    try:
        results = run_tasks([Path(t) for t in a.tasks], spec, config_id=a.config_id, reason=a.reason,
                            setting=a.setting, samples=a.samples, mode=a.mode,
                            run_root=Path(a.run_root) if a.run_root else None, on_result=show,
                            grade_episodes=not a.no_grade)
    except (RunRefused, TaskError, RelayError) as exc:
        print(f"refused: {exc}")
        return 2
    return 0 if all(r.valid for r in results) else 1


def _tier_f(argv: list[str]) -> int:
    from pathlib import Path

    from ..runlog.log import RunRefused
    from .relay import RelayError
    from .tier_f_mode import TierFTaskError, run_tier_f_samples
    a, spec = _batch_args("gb tier-a tier-f", argv)

    def show(res: Any) -> None:
        grade = (res.tier_f or {}).get("grade") or {}
        print(f"{res.task_id:28s} s{res.sample:02d}  {'PASS' if grade.get('passed') else 'fail':4s} "
              f"{grade.get('failure') or '':22s} {res.end_reason:26s} {res.run_dir}")
    try:
        results = run_tier_f_samples([Path(t) for t in a.tasks], spec, config_id=a.config_id, reason=a.reason,
                                     samples=a.samples, setting=a.setting, mode=a.mode,
                                     run_root=Path(a.run_root) if a.run_root else None, on_result=show)
    except (RunRefused, TierFTaskError, RelayError) as exc:
        print(f"refused: {exc}")
        return 2
    return 0 if all(r.valid for r in results) else 1


def _trace(argv: list[str]) -> int:
    from pathlib import Path

    from .trace import export_run_dir
    if not argv:
        print("usage: gb tier-a trace <run_dir> [--no-thinking]")
        return 2
    row = export_run_dir(Path(argv[0]), include_thinking="--no-thinking" not in argv)
    print(json.dumps(row))
    return 0


def _task_dirs(arg: str, tier: str) -> list[Any]:
    from pathlib import Path
    if arg != "all":
        return [Path(arg)]
    if tier == "A":
        from .task import task_dirs
        return task_dirs()
    from ..tier_f.taskio import task_dirs as f_dirs
    return f_dirs()


def _gates(argv: list[str]) -> int:
    import argparse

    from . import replay
    p = argparse.ArgumentParser(prog="gb tier-a gates")
    p.add_argument("task", help="a Tier-A task directory, or 'all' (items/tier_a)")
    p.add_argument("--reps", type=int, default=replay.REPS)
    p.add_argument("--workers", type=int, default=replay.WORKERS)
    p.add_argument("--no-write", action="store_true", help="do not write <task_dir>/gates.json")
    p.add_argument("--keep", action="store_true", help="keep the gate run directories")
    p.add_argument("--json", action="store_true", help="print each full result")
    a = p.parse_args(argv)
    dirs = _task_dirs(a.task, "A")
    if not dirs:
        print("no Tier-A tasks found")
        return 1
    ok = True
    for d in dirs:
        res = replay.gates(d, reps=a.reps, workers=a.workers, write=not a.no_write, keep=a.keep)
        ok &= bool(res["passed"])
        print(replay.summary_line(res))
        if a.json or not res["passed"]:
            print(replay.dumps(res))
    return 0 if ok else 1


def _model_args(p: Any) -> None:
    from .isolation import DEFAULT_BASE_URL, DEFAULT_MODEL
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="llama-server OpenAI-compatible endpoint")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model name sent to the server")
    p.add_argument("--uds", default=None, help="llama-server UNIX socket (the episode then has no network)")
    p.add_argument("--no-stream", action="store_true", help="do not stream the model requests (models.py)")


def _tier_f_sample(argv: list[str]) -> int:
    import argparse
    from pathlib import Path

    from ..runlog.log import RunRefused
    from . import models
    from .tier_f_mode import TierFTaskError, run_tier_f_sample
    p = argparse.ArgumentParser(prog="gb tier-a tier-f-sample")
    p.add_argument("task", help="a Tier-F task directory")
    _model_args(p)
    p.add_argument("--config-id", default="adhoc")
    p.add_argument("--reason", default="manual gb tier-a tier-f-sample")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--setting", default="benchmark", choices=["benchmark", "grc_agent_defaults"])
    p.add_argument("--run-root", default=None)
    a = p.parse_args(argv)
    spec = models.openai_compatible(a.base_url, a.model, uds_path=a.uds, stream=False if a.no_stream else None)
    try:
        res = run_tier_f_sample(Path(a.task), spec, Path(a.run_root) if a.run_root else None,
                                config_id=a.config_id, sample=a.sample, setting=a.setting, reason=a.reason)
    except (RunRefused, TierFTaskError) as exc:
        print(f"refused: {exc}")
        return 2
    print(json.dumps(tier_f_summary(res), indent=1))
    return 0 if res.valid else 1


def tier_f_summary(res: Any) -> dict[str, Any]:
    """The printed digest of a Tier-F sample (EpisodeResult with tier_f)."""
    rec = res.tier_f or {}
    grade = rec.get("grade") or {}
    return {"task": res.task_id, "sample": res.sample, "status": res.status, "end_reason": res.end_reason,
            "passed": grade.get("passed"), "failure": grade.get("failure"), "first_failed": grade.get("first_failed"),
            "highest_passed": grade.get("highest_passed"),
            "lookup": {k: (rec.get("lookup") or {}).get(k) for k in ("used", "query", "reference_recall")},
            "edit": {k: (rec.get("edit") or {}).get(k) for k in ("called", "well_formed", "accepted", "error_codes")},
            "blocked_calls": [b["tool"] for b in rec.get("blocked_calls") or []], "run_dir": res.run_dir}


def _tier_f_golden(argv: list[str]) -> int:
    import argparse

    from ..tier_f import golden
    from . import replay
    p = argparse.ArgumentParser(prog="gb tier-a tier-f-golden")
    p.add_argument("task", help="a Tier-F task directory, or 'all' (items/tier_f)")
    p.add_argument("--no-write", action="store_true", help="do not write golden_calls.json")
    p.add_argument("--no-gate", action="store_true", help="skip the tier_f replay gate")
    p.add_argument("--reps", type=int, default=replay.REPS)
    p.add_argument("--workers", type=int, default=replay.WORKERS)
    a = p.parse_args(argv)
    dirs = _task_dirs(a.task, "F")
    ok = bool(dirs)
    for d in dirs:
        res = golden.derive_and_verify(d, write=not a.no_write)
        print(f"{res['task']:28s} golden {'equivalent' if res['equivalent'] else 'NOT EQUIVALENT'}"
              + (f" (wrote {golden.GOLDEN_FILE})" if res["written"] else ""))
        for diff in res["diffs"][:20]:
            print(f"    - {diff}")
        good = res["equivalent"]
        if good and not a.no_gate:
            if not res["written"]:
                print("    (gate skipped: golden_calls.json was not written)")
            else:
                gate = replay.tier_f_gates(d, reps=a.reps, workers=a.workers)
                good = gate["passed"]
                print(replay.summary_line(gate))
                if not gate["passed"]:
                    print(replay.dumps(gate))
        ok &= good
    return 0 if ok else 1


def _smoke(argv: list[str]) -> int:
    import argparse
    from pathlib import Path

    from ..common.io import read_json, write_text
    from . import smoke
    p = argparse.ArgumentParser(prog="gb tier-a smoke")
    p.add_argument("--base-url", default=None, help="also run a real model on the fixture tasks (local llama-server)")
    p.add_argument("--start-server", action="store_true",
                   help="start the pinned llama-server with the smoke model (Qwen3-4B): grammar probe + step 3")
    p.add_argument("--model", default=smoke.SERVER_ALIAS, help="model name sent to the server")
    p.add_argument("--threads", type=int, default=None, help="llama-server threads (default: min(16, CPUs - 4))")
    p.add_argument("--backend", choices=smoke.BACKENDS, default=None,
                   help="llama.cpp build: cpu, cuda (b10333 built for the GPU) or auto (cuda when available)")
    p.add_argument("--reps", type=int, default=1, help="golden replays per fixture task")
    p.add_argument("--only", default=None, help="comma-separated fixture directory names")
    p.add_argument("--no-golden", action="store_true", help="skip the golden replays (step 1)")
    p.add_argument("--no-probe", action="store_true", help="skip the grammar probe (step 2)")
    p.add_argument("--tier-f-tasks", default=",".join(smoke.DEFAULT_TIER_F_TASKS),
                   help="Tier-F fixture directories for the real-model samples")
    p.add_argument("--tier-f-samples", type=int, default=smoke.DEFAULT_TIER_F_SAMPLES)
    p.add_argument("--episode-seconds", type=float, default=None,
                   help="Tier-A episode wall-clock cap for the real-model episodes (default: the benchmark's)")
    p.add_argument("--note", action="append", default=[], help="a note for the report (repeatable)")
    p.add_argument("--report", action="store_true", help=f"write {smoke.REPORT.name} (default with a real model)")
    p.add_argument("--no-report", action="store_true", help="never write the report")
    p.add_argument("--report-from", default=None, help="only re-render the report from a summary.json")
    a = p.parse_args(argv)
    if a.backend:
        import os
        os.environ["GRC_BENCH_LLAMA_BACKEND"] = a.backend
    if a.report_from:
        summary = read_json(Path(a.report_from))
        if a.note:
            summary["notes"] = list(summary.get("notes") or []) + a.note
        write_text(smoke.REPORT, smoke.render_report(summary))
        print(f"report: {smoke.REPORT}")
        return 0
    real = bool(a.base_url or a.start_server)
    report = smoke.REPORT if (a.report or real) and not a.no_report else None
    if a.start_server:
        import signal

        def _stop(signum: int, frame: Any) -> None:     # noqa: ARG001 - signal handler API
            raise SystemExit(128 + signum)               # leaves run_smoke's finally: the server stops
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, _stop)
    summary = smoke.run_smoke(base_url=a.base_url, model_name=a.model, reps=a.reps, start_server=a.start_server,
                              only=a.only.split(",") if a.only else None, golden=not a.no_golden,
                              probe=not a.no_probe, tier_f_tasks=[t for t in a.tier_f_tasks.split(",") if t],
                              tier_f_samples=a.tier_f_samples, threads=a.threads, notes=a.note, report=report,
                              episode_seconds=a.episode_seconds)
    for line in summary["lines"]:
        print(line)
    verdict = ((summary.get("probe") or {}).get("verdict") or {}).get("grammar_constrains_tool_calls")
    if summary.get("probe"):
        print(f"grammar probe: --jinja grammar-constrains tool calls: {verdict}")
    print(f"summary: {summary['path']}")
    if report is not None:
        print(f"report: {report}")
    return 0 if summary["golden_passed"] else 1


def _help() -> None:
    print("usage: ./gb tier-a <subcommand> [args]\n\nsubcommands:")
    for name in sorted(SUBCOMMANDS):
        print(f"  {name:14s} {SUBCOMMANDS[name][1]}")


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"help", "-h", "--help"}:
        _help()
        return 0
    name, rest = argv[0], argv[1:]
    if name not in SUBCOMMANDS:
        print(f"unknown subcommand: {name}\n")
        _help()
        return 2
    module, func = SUBCOMMANDS[name][0].split(":")
    return int(getattr(importlib.import_module(module), func)(rest) or 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

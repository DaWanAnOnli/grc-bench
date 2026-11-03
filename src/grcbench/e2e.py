"""`./gb e2e`: benchmark v0.9 end to end on the dev split (gate G2 evidence), with one local model.

    ./gb e2e --out DIR [--n-k N] [--k-a K] [--limit-k M] [--limit-f M] [--limit-a M] [--only K,F,A,E]
             [--rubrics FILE] [--report PATH] [--e-name NAME --machine-label TEXT]
             [--base-url URL --model NAME [--gguf FILE]]

One llama-server (tier_a.smoke.LocalServer: the CUDA build of the pinned llama.cpp tag, the smoke model
Qwen3-4B-Q4_K_M, --jinja, every layer on the GPU) serves, in turn:
  Tier K  the dev questions, n answers each (tier_k.run.run, then tier_k.run.score);
  Tier F  the 10 dev tasks, one sample each in the "one lookup + one edit" setting, graded V1-V5;
  Tier A  the 12 dev tasks, k episodes each, each graded by the task's acceptance check;
then the server stops and Tier E runs the CUDA build's llama-bench on the same file at 4K context
(5 warm runs, purpose "plumbing"; hardware label H3, the nearest one, with the record's mismatch list).
Only dev-split items reach the model (runlog.guard allows the dev split; no test-split item is sent).
Everything goes to DIR; the report is reports/e2e_dev.md. With --only, the tiers not named keep their
results from DIR/summary.json, so one tier can be run again without losing the others.

What it measures: that every tier runs, records and grades end to end, the wall time (= RTX 5090 GPU
time: one GPU, one job) and per tier K: the routes, the reasons and the share of answers with a
final-answer line; F: the V1-V5 funnel; A: task success; E: speeds. The dev rubrics are empty
templates (numeric null) and the dev split has no keys, so Tier-K answers are counted, not scored,
and the rule cascade's deferral rate on dev answers is not measured (round-2 finding F5 stays open).
It is a plumbing run of a smoke model, not a result about GRC-Bench models.

On another machine (a Colab GPU): --limit-f/--limit-a run only the first M dev tasks, --rubrics FILE
scores the Tier-K answers with a dev rubric file (placeholders allowed: tier_k/placeholder_rubrics.py),
--report writes the report elsewhere than reports/e2e_dev.md, and --e-name/--machine-label name the
Tier-E record and the machine it ran on.

--base-url URL --model NAME: a llama-server that is already running (another notebook's, e.g. the team's
evaluation notebook) serves Tiers K, F and A instead of the smoke model's own server. Its /props must
report the pinned llama.cpp build (else the run is refused, exit 2); its context and its default sampling
are recorded, and Tier-K requests send that same sampling, so Tier K matches Tiers F and A (whose requests
leave sampling to the server). --gguf FILE is the model file Tier E measures (default: the smoke model).
Tier E measures with no other llama.cpp process running, so with --base-url it runs alone, after that server
has stopped: `--only K,F,A`, stop the server, then `--only E`, which asks the server nothing and keeps its record.
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import urllib.request
import sys
import time
from pathlib import Path
from typing import Any

from grcbench.common.io import read_json, write_json, write_text
from grcbench.common.paths import LLAMA_CPP_BUILD, REPO, REPORTS

REPORT = REPORTS / "e2e_dev.md"
CONFIG_ID = "e2e-dev"
REASON = "G2 end-to-end run of benchmark v0.9 on the dev split (sign-off plan Task 10)"
RUNGS = ("V1", "V2", "V3", "V4", "V5")
E_NAME = "e2e_dev_qwen3_4b_rtx5090"
E_HARDWARE = "H3"            # the nearest label: a desktop GPU (H3 is Qoherent's RTX 3090; this is an RTX 5090)
E_MACHINE = "the build workstation (RTX 5090)"
E_NOTE = ("G2 end-to-end dev run: plumbing of the Tier-E protocol; label H3 is the nearest setup (a desktop GPU), "
          "not Qoherent's RTX 3090")
SAMPLING_KEYS = ("temperature", "top_k", "top_p", "min_p")


def props_summary(props: dict[str, Any]) -> dict[str, Any]:
    """A running llama-server's build, context, slots, model file and default sampling, from its /props."""
    info = props.get("build_info") or ""
    settings = props.get("default_generation_settings") or {}
    params = settings.get("params") or {}
    return {"llama_cpp_build": info.split("-")[0] or None, "build_info": info or None,
            "context": settings.get("n_ctx"), "slots": props.get("total_slots"), "model_path": props.get("model_path"),
            "sampling": {k: round(params[k], 6) if isinstance(params[k], float) else params[k]   # float32 on the server
                         for k in SAMPLING_KEYS if k in params}}


def server_props(base_url: str) -> dict[str, Any]:
    """GET /props of the llama-server behind an OpenAI-compatible base URL (".../v1")."""
    root = base_url.rstrip("/").removesuffix("/v1")
    with urllib.request.urlopen(f"{root}/props", timeout=30) as resp:
        return json.loads(resp.read())


def _funnel(grades: list[dict[str, Any]]) -> dict[str, int]:
    """Samples that reach each rung, every earlier rung passed (the ladder runs a rung only after the
    previous one passed): from each grade's highest_passed."""
    out = dict.fromkeys(RUNGS, 0)
    for g in grades:
        top = g.get("highest_passed")
        if top in RUNGS:
            for rung in RUNGS[:RUNGS.index(top) + 1]:
                out[rung] += 1
    return out


def _parse_e(rec: dict[str, Any]) -> dict[str, Any]:
    """The speeds at 4K and the hardware note of a Tier-E record (tier_e.bench)."""
    ctx = next((c for c in rec.get("contexts", []) if c.get("context") == 4096), {})
    hw = rec.get("hardware") or {}
    return {"generation_tps": (ctx.get("generation") or {}).get("median_ts"),
            "prompt_tps": (ctx.get("prompt") or {}).get("median_ts"),
            "gpu_used": hw.get("gpu_used"), "mismatches": hw.get("mismatches") or []}


def _tier_k(base_url: str, model: str, out: Path, n: int, limit: int | None, sampling: dict[str, Any], *,
            rubrics: Path | None = None) -> dict[str, Any]:
    from grcbench.tier_k import run
    run.run("dev", base_url=base_url, model=model, config_id=CONFIG_ID, reason=REASON, n=n, sampling=sampling,
            out_dir=out, limit=limit)
    s = run.score(out, rubrics=rubrics)
    return {"items": len(run.load_split("dev")[:limit]), "answers": s["n_answers"],
            **{k: s.get(k) for k in ("n_rules", "n_judge", "n_no_reference", "n_request_failed",
                                     "n_final_answer_line", "by_reason", "rubrics")}, "out": str(out)}


def _tier_f(base_url: str, model: str, out: Path, *, limit: int | None = None) -> dict[str, Any]:
    from grcbench.tier_a import models
    from grcbench.tier_a.tier_f_mode import run_tier_f_samples
    from grcbench.tier_f.taskio import task_dirs as f_task_dirs
    tasks = f_task_dirs("dev")[:limit]
    results = run_tier_f_samples(tasks, models.openai_compatible(base_url, model), config_id=CONFIG_ID,
                                 reason=REASON, samples=1, run_root=out)
    grades = [(r.tier_f or {}).get("grade") or {} for r in results]
    return {"tasks": len(tasks), "samples": len(results), "passed": sum(bool(g.get("passed")) for g in grades),
            "funnel": _funnel(grades), "out": str(out)}


def _tier_a(base_url: str, model: str, out: Path, k: int, *, limit: int | None = None) -> dict[str, Any]:
    from grcbench.tier_a import models
    from grcbench.tier_a.runner import run_tasks
    from grcbench.tier_a.task import task_dirs as a_task_dirs
    tasks = a_task_dirs("dev")[:limit]
    results = run_tasks(tasks, models.openai_compatible(base_url, model), config_id=CONFIG_ID, reason=REASON,
                        samples=k, run_root=out, grade_episodes=True)
    return {"tasks": len(tasks), "episodes": len(results),
            "passed": sum(bool(r.grade and r.grade.get("passed")) for r in results),
            "valid": sum(bool(r.valid) for r in results), "out": str(out)}


def _tier_e(model_file: Path, llama_dir: Path, out: Path, *, name: str = E_NAME,
            machine_label: str = E_MACHINE) -> dict[str, Any]:
    """llama-bench at 4K, 5 warm runs, through `./gb tier-e run` (its own process and output files)."""
    note = (E_NOTE if machine_label == E_MACHINE else
            f"end-to-end dev run on {machine_label}: plumbing of the Tier-E protocol; H3 is a placeholder label")
    proc = subprocess.run([str(REPO / "gb"), "tier-e", "run", "--model", str(model_file), "--hardware", E_HARDWARE,
                           "--llama-dir", str(llama_dir), "--contexts", "4096", "--purpose", "plumbing",
                           "--name", name, "--machine-label", machine_label, "--note", note],
                          cwd=REPO, capture_output=True, text=True)
    write_text(out / "tier_e.log", proc.stdout + proc.stderr)
    res: dict[str, Any] = {"contexts": [4096], "runs": 5, "exit_code": proc.returncode, "out": str(out),
                           "name": name, "hardware_label": E_HARDWARE, "machine_label": machine_label}
    record = REPORTS / "tier_e" / f"{name}.json"
    if proc.returncode != 0 or not record.exists():
        return {**res, "error": f"./gb tier-e run exited {proc.returncode} (log: {out / 'tier_e.log'})"}
    return {**res, **_parse_e(read_json(record)), "record": str(record)}


def _hours(seconds: float | None) -> str:
    return "n/a" if seconds is None else f"{seconds / 3600:.2f}"


def _cell(text: Any) -> str:
    return " ".join(str(text).split()).replace("|", "/")


def render_report(summary: dict[str, Any]) -> str:
    """reports/e2e_dev.md from the run's summary."""
    t = summary.get("tiers", {})
    server = summary.get("server", {})
    external = bool(server.get("base_url"))
    machine = summary.get("machine_label") or "the RTX 5090"
    gpu = summary.get("machine_label") or "RTX 5090"
    out = ["# End-to-end run of GRC-Bench v0.9 on the dev split", "",
           ("> **Plumbing run with a smoke model, not a result about GRC-Bench models.**" if not external else
            "> **Plumbing run on a few dev items, not a result about the model.**") + " It shows that every "
           "tier runs, records and grades end to end on the dev split; no test-split item is sent to a model.",
           "", (f"Generated by `./gb e2e` (`src/grcbench/e2e.py`). Model: {summary.get('model')} (the smoke model), "
                f"served by llama.cpp {server.get('llama_cpp_build', 'b10333')} (CUDA build) with `--jinja` on the RTX "
                f"5090, context {server.get('context', 'n/a')} tokens (the model's native maximum, below Tier A's 64K "
                f"cap: the server refuses a longer request, and the harness then records the episode as an "
                f"infrastructure error, not valid). Server load "
                f"{server.get('load_seconds', 'n/a')} s. Run: {summary.get('started')} to {summary.get('finished')} "
                "(UTC)." if not external else
                f"Generated by `./gb e2e` (`src/grcbench/e2e.py`). Model: {summary.get('model')}, served by a "
                f"llama-server already running on {machine} (`{server.get('base_url')}`, llama.cpp "
                f"{server.get('build_info')}), context {server.get('context', 'n/a')} tokens, default sampling "
                f"`{_cell(json.dumps(server.get('sampling') or {}))}`; model file `{server.get('model_file')}`. Run: "
                f"{summary.get('started')} to {summary.get('finished')} (UTC)."),
           "", f"| Tier | Items run | Wall time (h) = GPU-hours ({gpu}) | Outcome |", "|---|---|--:|---|"]

    def row(name: str, key: str, items: str, outcome: str) -> None:
        d = t.get(key) or {}
        if "error" in d:
            out.append(f"| {name} | - | {_hours(d.get('seconds'))} | did not complete: {_cell(d['error'])} |")
        elif d:
            out.append(f"| {name} | {items} | {_hours(d.get('seconds'))} | {_cell(outcome)} |")
        else:
            out.append(f"| {name} | - | - | not run |")

    k, f, a, e = (t.get(x) or {} for x in "KFAE")
    row("Tier K", "K", f"{k.get('items')} questions, {k.get('answers')} answers",
        f"{k.get('n_final_answer_line')} of {k.get('answers')} answers end with a final-answer line; routes: rules "
        f"{k.get('n_rules')}, judge {k.get('n_judge')}, no reference {k.get('n_no_reference')}, failed requests "
        f"{k.get('n_request_failed', 'n/a')}")
    row("Tier F", "F", f"{f.get('tasks')} tasks, {f.get('samples')} samples",
        f"{f.get('passed')} of {f.get('samples')} samples reach V5; samples reaching each rung "
        + ", ".join(f"{r} {(f.get('funnel') or {}).get(r)}" for r in RUNGS))
    row("Tier A", "A", f"{a.get('tasks')} tasks, {a.get('episodes')} episodes",
        f"{a.get('passed')} of {a.get('episodes')} episodes pass the acceptance check; {a.get('valid')} valid")
    row("Tier E", "E", "llama-bench at 4K context, 5 warm runs",
        f"generation {e.get('generation_tps')} tok/s, prompt {e.get('prompt_tps')} tok/s (median); GPU used: "
        f"{e.get('gpu_used')}")
    total = sum((t.get(x) or {}).get("seconds") or 0 for x in "KFAE")
    out += ["", f"Total: {_hours(total)} GPU-hours on {machine}.", "",
            f"Tier E is recorded under hardware label {e.get('hardware_label', E_HARDWARE)}, the nearest setup (a "
            "desktop GPU); it is not Qoherent's RTX 3090, and the Tier-E record lists the differences"
            + (f": {_cell('; '.join(e['mismatches']))}." if e.get("mismatches") else "."), "",
            "## What this run does not measure", "",
            ("- **Tier-K scores.** The dev rubrics are empty templates (`numeric` null, no points) and the dev "
             "split has no block-question or calculation keys, so every Tier-K answer that is not truncated or empty "
             "is counted (route `no_reference`), not scored. The rule cascade's deferral rate on dev answers "
             "(round-2 finding F5) is therefore not measured here; F5 stays open until dev rubrics carry `numeric` "
             "fields." if not k.get("rubrics") else
             f"- **Tier-K scores.** The answers are routed with the rubric file `{k['rubrics'].get('file')}` "
             f"({k['rubrics'].get('n_placeholder')} of {k['rubrics'].get('n_records')} placeholder rubrics: the "
             "training release's reference answer as the only point), which tests the scoring route, not the model."),
            *(["- **The judge.** No answer reaches the judge (no approved rubric), so the judge is not called."]
              if k.get("n_judge") == 0 else
              [f"- **The judge.** {k.get('n_judge')} answers are queued for the judge; it is not called here."]),
            ("- **Model quality.** Qwen3-4B is the smoke model; its pass rates say nothing about the GRC-Bench "
             "models." if not external else
             "- **Model quality.** A few dev items, scored with the rubric file the run was given; the pass "
             "rates say nothing about the model."), ""]
    if k.get("by_reason"):
        out += ["## Tier K: reasons", "", "| Reason | Answers |", "|---|--:|"]
        out += [f"| `{r}` | {n} |" for r, n in sorted(k["by_reason"].items())] + [""]
    if summary.get("notes"):
        out += ["## Notes", ""] + [f"- {_cell(n)}" for n in summary["notes"]] + [""]
    out += ["## Where the outputs are", ""] + [f"- Tier {x}: `{(t.get(x) or {}).get('out')}`" for x in "KFAE"] + [""]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb e2e", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-k", type=int, default=1, help="answers per Tier-K question")
    ap.add_argument("--k-a", type=int, default=1, help="episodes per Tier-A task")
    ap.add_argument("--limit-k", type=int, default=None, help="only the first M Tier-K questions")
    ap.add_argument("--limit-f", type=int, default=None, help="only the first M Tier-F dev tasks")
    ap.add_argument("--limit-a", type=int, default=None, help="only the first M Tier-A dev tasks")
    ap.add_argument("--rubrics", type=Path, default=None, help="score Tier K with this dev rubric file")
    ap.add_argument("--report", type=Path, default=REPORT, help=f"where the report goes (default {REPORT})")
    ap.add_argument("--e-name", default=E_NAME, help="the Tier-E record's name")
    ap.add_argument("--machine-label", default=E_MACHINE, help="the machine the run is on (Tier-E record)")
    ap.add_argument("--report-only", action="store_true", help="re-render the report from DIR/summary.json")
    ap.add_argument("--only", default="K,F,A,E",
                    help="tiers to run, comma-separated; the others keep their results from DIR/summary.json")
    ap.add_argument("--base-url", default=None, help="a running llama-server serves Tiers K, F and A (module docstring)")
    ap.add_argument("--model", default=None, help="the running server's model name (with --base-url)")
    ap.add_argument("--gguf", type=Path, default=None, help="the model file Tier E measures (default: the smoke model)")
    a = ap.parse_args(argv)
    if a.base_url and not a.model:
        ap.error("--base-url needs --model")
    only_arg = {x.strip().upper() for x in a.only.split(",") if x.strip()}
    if a.base_url and "E" in only_arg and only_arg & set("KFA"):
        ap.error("with --base-url, Tier E runs alone (--only E) after that server has stopped: Tier E measures with "
                 "no other llama.cpp process running, and would wait up to an hour for the server to stop")
    report = a.report
    if a.report_only:
        write_text(report, render_report(read_json(a.out / "summary.json")))
        print(f"report: {report}")
        return 0
    from grcbench.runlog.log import utc_now
    from grcbench.tier_a import smoke
    only = only_arg
    served = [x for x in "KFA" if x in only]
    props = None
    if a.base_url and served:                         # Tier E alone needs no server (it may be stopped)
        props = props_summary(server_props(a.base_url))
        if props["llama_cpp_build"] != LLAMA_CPP_BUILD:
            print(f"refused: the server at {a.base_url} runs llama.cpp {props['build_info']}; GRC-Bench pins "
                  f"{LLAMA_CPP_BUILD}", file=sys.stderr)
            return 2
    a.out.mkdir(parents=True, exist_ok=True)
    previous = read_json(a.out / "summary.json") if (a.out / "summary.json").exists() else {}
    kept = {k: v for k, v in (previous.get("tiers") or {}).items() if k not in only}
    # the run window covers the kept tiers too
    summary: dict[str, Any] = {"model": a.model or smoke.SERVER_ALIAS,
                               "started": previous.get("started") if kept and previous.get("started") else utc_now(),
                               "tiers": kept}
    if a.machine_label != E_MACHINE:
        summary["machine_label"] = a.machine_label
    model_file = a.gguf or smoke.SMOKE_MODEL
    if a.base_url:
        sampling = (props or {}).get("sampling") or {}                  # the running server's own defaults
        summary["server"] = {**(previous.get("server") or {}), **(props or {}), "base_url": a.base_url,
                             "model_file": str(model_file)}
    else:
        sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}   # the model's recommended settings
        ctx = smoke.server_context(smoke.native_context(smoke.SMOKE_MODEL))
        summary["server"] = {**(previous.get("server") or {}), "context": ctx, "llama_cpp_build": LLAMA_CPP_BUILD,
                             "model_file": str(model_file)}

    def save() -> None:
        write_json(a.out / "summary.json", summary)

    def tier(key: str, fn: Any, *args: Any) -> None:
        t0 = time.monotonic()
        try:
            summary["tiers"][key] = fn(*args)
        except Exception as exc:                          # recorded in the report; the run goes on
            summary["tiers"][key] = {"error": f"{type(exc).__name__}: {exc}"}
        summary["tiers"][key]["seconds"] = round(time.monotonic() - t0, 1)
        save()

    def served_tiers(base_url: str, model: str) -> None:
        if "K" in only:
            tier("K", functools.partial(_tier_k, rubrics=a.rubrics), base_url, model, a.out / "tier_k", a.n_k,
                 a.limit_k, sampling)
        if "F" in only:
            tier("F", functools.partial(_tier_f, limit=a.limit_f), base_url, model, a.out / "tier_f")
        if "A" in only:
            tier("A", functools.partial(_tier_a, limit=a.limit_a), base_url, model, a.out / "tier_a", a.k_a)

    if served and a.base_url:
        served_tiers(a.base_url, a.model)
    elif served:
        server = smoke.LocalServer(a.out / "llama-server.log", ctx=ctx)
        summary["server"]["load_seconds"] = None
        try:
            with server as base_url:
                summary["server"]["load_seconds"] = server.load_seconds
                served_tiers(base_url, smoke.SERVER_ALIAS)
        except Exception as exc:                          # the server did not start (or stop): recorded
            for x in served:
                summary["tiers"].setdefault(x, {"error": f"server: {type(exc).__name__}: {exc}"})
            save()
    if "E" in only:
        tier("E", functools.partial(_tier_e, name=a.e_name, machine_label=a.machine_label), model_file,
             smoke.LLAMA_DIR_CUDA, a.out)
    summary["finished"] = utc_now()
    save()
    write_text(report, render_report(read_json(a.out / "summary.json")))
    print(f"report: {report}")
    return 0 if all("error" not in v for v in summary["tiers"].values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

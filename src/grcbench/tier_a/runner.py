"""The Tier-A episode runner (eval plan §4.4, §7.4; build plan Phase 4).

`run_episode(task, model_factory, setting, run_root)` runs one task once with GRC-Agent's real
executor and returns an EpisodeResult. On the host it:
  1. loads and validates the task (task.py) and runs the run-log guard (grcbench.runlog.log): a
     test-split run needs a declared configuration id and is logged BEFORE anything starts;
     collection mode refuses benchmark items;
  2. makes a fresh run directory <run_root>/<config>/<task>/<setting>-sNN under ~/grc-bench-work
     (isolation.make_run_env: .env, HOME/XDG, GRC_HIER_PATH, read-only knowledge DBs) and copies the
     task's start/ files into its project folder (the originals are never edited);
  3. starts ONE episode process inside bubblewrap (isolation.run_episode_process; one process per
     episode is required, see isolation.py) and waits for it with a process timeout;
  4. reads the episode's result and writes its trace (trace.py, Qoherent's SFT row format).

Inside the episode process (`python -m grcbench.tier_a.runner episode --spec <run>/episode.json`):
  * `import gi` is blocked; the start graph is opened in a HeadlessFlowgraphProxy (GTK-free port of
    GRC-Agent's NativeFlowgraphProxy; File > New when the task has no start graph);
  * the agent is GRC-Agent's real executor (agent_build.build_benchmark_agent), the model is built
    from the ModelSpec (models.py);
  * the conversation follows GRC-Agent's own turn driver (chat/turn_driver.py _run_agent_turn):
    agent.iter(prompt, message_history=...) driven node by node, each model request streamed
    (node.stream(run.ctx), as chat/stream_view.py _stream_request does) when the model spec
    streams (models.ModelSpec.streams(): the llama-server path does); a run that ends with
    DeferredToolRequests is resumed with prompt=None and DeferredToolResults; the runner approves
    every approval request (change_graph, run_flowgraph start, run_command, start_command) with
    ToolApproved(), as GRC-Agent's sidebar does in "yolo" mode (chat/approvals.py L107-110), which
    Qoherent's collection used (GRC_AGENT_APPROVE_CHANGES=yolo). After each finished turn the
    scripted user (user.py) either replies or ends the episode;
  * caps (caps.py): one RunUsage shared by every agent run of the episode with UsageLimits(40
    requests, 65,536 input tokens per request), the outermost HarnessCapability (4K tool-output cap,
    raw arguments before JSON repair, StopGracefully detection, flowgraph-run time cap), and an
    episode wall-clock timeout. Any cap hit ends the episode as a cap failure;
  * everything is recorded: every message (pydantic-ai's own JSON, messages.json), every tool call
    (raw arguments, validation, approval, execution, result size, truncation), every model request
    (tokens, seconds), the scripted-user turns, the flowgraph runs, the final graph and project
    files, the final answer, and the caps hit.

Episode status: "completed" (the agent finished, or the scripted user had no turns left), "cap_failure"
(a cap was hit: the task fails whatever its end state, eval plan §4.4), "agent_error" (the agent run
raised: retries exhausted, a tool crashed; the end state is still graded) or "infrastructure_error"
(the harness or the model server failed: the episode is invalid and is re-run, not scored).

The same episode process runs the Tier-F setting (spec "kind": "tier_f", tier_f_mode.py): the task's
start graph is materialised as project/flowgraph.grc, the history is prefilled with one full
inspect_graph call that the first step executes for real, the HarnessCapability carries the Tier-F
budget, the turn stops once the edit has been processed, and the final graph is exported for the
host to grade. Everything else (agent, deps, caps, recording, trace) is shared with Tier A.

Differences from GRC-Agent's desktop turn driver, all without effect on what the model is sent:
  * scripted test models (FunctionModel) are not streamed; the llama-server path is (models.py,
    "Streaming": with GRC-Agent's 120 s HTTP read timeout a non-streamed request must be complete
    within 120 s, a streamed one only needs a byte every 120 s, as in the GUI);
  * the GUI also streams the tool-call nodes, to render tool events; the runner just runs them
    (run.next), which executes the same calls through the same capability chain;
  * after an agent run raises, the episode ends; the GUI would salvage the history and let the user
    type again (turn_driver._recover_history_after_failure).

Run directory (on the WSL filesystem):
    episode.json   the episode spec        result.json    EpisodeResult (episode + host fields)
    messages.json  full message history     tools.json     the tool snapshot (freeze_tools)
    trace.jsonl    the SFT-format row        episode.log    the episode process's stderr
    project/       the agent's project folder (end state); plus .env, .grc_agent/, home/, runtime/
    Tier F adds    tier_f/start_compact.grc, tier_f/start_full.grc (the start graph before and after
                   GRC's writer) and final.grc (the graded end state); wire.jsonl holds the request
                   bodies when the model is models.wire_scripted.

Options used outside the benchmark (adhoc.py, `./gb agent`); a benchmark run sets none of them:
    export_state   at the end, the active page is exported: state/live.grc (the graph in GRC's
                   memory, written by GRC's own writer, saved or not) and state/page.json (its file
                   path relative to project/ or null for an untitled graph, its saved flag and the
                   conversation id)
    resume_from    a finished run directory that was run with export_state: its project/ and
                   .grc_agent/ are copied in, its message history becomes the start of this one
                   (resume/messages.json) and its exported page is opened as it was (same file path,
                   same saved flag: unsaved edits carry over); the conversation id is kept
    a relay        models.hosted(...): relay.Relay runs on the host for the episode and the model
                   goes over its UNIX socket (the API key never enters the sandbox)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..common.io import read_json, sha256_file, write_json, write_text
from ..common.llama_server import server_info
from ..common.paths import ITEMS
from ..runlog import log as runlog
from . import isolation
from .caps import CAP_CONTEXT, ENGINE_REFUSAL, Caps, CapHit, EpisodeMeter, HarnessCapability, classify_exception
from .models import ModelSpec, as_spec, build_model
from .task import Task, load_task, task_from_dict
from .user import reply_text

HARNESS_VERSION = "grcbench.tier_a.runner/1"
SPEC_FILE, RESULT_FILE, MESSAGES_FILE, TOOLS_FILE = "episode.json", "result.json", "messages.json", "tools.json"
TRACE_FILE, STDERR_FILE = "trace.jsonl", "episode.log"
STATE_DIR, LIVE_GRC, PAGE_FILE, RESUME_DIR = "state", "live.grc", "page.json", "resume"
PROCESS_MARGIN = 300.0          # seconds the episode process gets beyond the episode cap (setup, teardown)
STATUSES = ("completed", "cap_failure", "agent_error", "infrastructure_error")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class UnresolvableDeferred(RuntimeError):
    """The agent ended a run with deferred external calls the harness cannot resolve."""


class HarnessStopped:
    """Output of an agent turn the harness ended between two graph nodes (the Tier-F setting stops
    right after its one edit has been processed, before the next model request)."""

    def __repr__(self) -> str:
        return "HarnessStopped()"


@dataclass
class EpisodeResult:
    """Everything recorded about one episode (written to <run>/result.json)."""

    run_id: str
    task_id: str
    task_version: int
    split: str
    category: str
    config_id: str
    setting: str
    sample: int
    mode: str
    status: str
    end_reason: str
    run_dir: str
    caps: dict[str, Any]
    caps_hit: list[dict[str, Any]] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)
    harness: dict[str, Any] = field(default_factory=dict)
    agent_info: dict[str, Any] = field(default_factory=dict)
    final_graph_path: str | None = None
    final_graph_saved: bool | None = None
    final_graph_sha256: str | None = None
    project_files: dict[str, str] = field(default_factory=dict)
    final_text: str = ""
    final_output: Any = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    unknown_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model_requests: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    flowgraph_runs: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    process: dict[str, Any] = field(default_factory=dict)
    run_log_entry: dict[str, Any] | None = None
    tier: str = "A"
    tier_f: dict[str, Any] | None = None          # the Tier-F sample record (tier_f_mode.py)
    grade: dict[str, Any] | None = None           # the Tier-A grade (grader.grade), when graded

    @property
    def cap_failure(self) -> bool:
        """A cap was hit: the task counts as failed whatever its end state (eval plan §4.4)."""
        return self.status == "cap_failure"

    @property
    def valid(self) -> bool:
        """False for infrastructure errors: such an episode is re-run, not scored."""
        return self.status != "infrastructure_error"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeResult":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def _safe(text: str) -> str:
    out = _SAFE.sub("_", str(text)).strip("._")
    if not out:
        raise ValueError(f"unusable name {text!r}")
    return out


def run_id_for(task_id: str, config_id: str, setting: str, sample: int) -> str:
    return f"{_safe(config_id)}__{_safe(task_id)}__{setting}-s{sample:02d}"


def _settings(setting: str, caps: Caps, invalid_gate: str, streaming: bool = False) -> dict[str, Any]:
    return {"setting": setting, "caps": caps.to_dict(), "invalid_gate": invalid_gate,
            "approvals": "all approved (GRC-Agent yolo)", "streaming": streaming, "harness": HARNESS_VERSION}


# --------------------------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------------------------

def prepare_run_dir(task: Task, run_dir: Path, spec: ModelSpec) -> tuple[Path, dict[str, str]]:
    """A fresh run directory with the task's start state copied into project/; returns
    (active graph path or None-path, process environment)."""
    run_dir = isolation.check_run_dir(run_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    env = isolation.make_run_env(run_dir, base_url=spec.base_url, model_name=spec.model_name)
    project = isolation.RunLayout(run_dir).project
    for rel in task.start_files:
        dest = project / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(task.start_dir / rel, dest)
    return run_dir, env


def _carry_over(parent: Path, run_dir: Path) -> dict[str, Any]:
    """Copy a finished run's end state into the fresh `run_dir` (module docstring, resume_from)."""
    parent = isolation.check_run_dir(parent)
    page_file, live = parent / STATE_DIR / PAGE_FILE, parent / STATE_DIR / LIVE_GRC
    if not (page_file.is_file() and live.is_file() and (parent / MESSAGES_FILE).is_file()):
        raise ValueError(f"{parent} has no exported end state (it must be run with export_state)")
    page = read_json(page_file)
    if page.get("error"):
        raise ValueError(f"{parent}: its end state could not be exported ({page['error']})")
    lay, old = isolation.RunLayout(run_dir), isolation.RunLayout(parent)
    shutil.rmtree(lay.project)
    shutil.copytree(old.project, lay.project, symlinks=True)
    if old.state_dir.is_dir():
        shutil.copytree(old.state_dir, lay.state_dir, symlinks=True, dirs_exist_ok=True)
    (run_dir / RESUME_DIR).mkdir()
    shutil.copyfile(parent / MESSAGES_FILE, run_dir / RESUME_DIR / MESSAGES_FILE)
    shutil.copyfile(live, run_dir / RESUME_DIR / LIVE_GRC)
    return {"parent": str(parent), "page": page}


def model_server_context(base_url: str | None, uds_path: str | None = None,
                         timeout: float = 5.0) -> dict[str, int | None]:
    """A llama-server's context per slot and the model's own (common.llama_server, over TCP or the
    socket); None where the server reports neither (an API)."""
    info = server_info(base_url, uds_path=uds_path, timeout=timeout)
    return {"n_ctx": info["n_ctx"], "n_ctx_train": info["n_ctx_train"]}


def server_below_model(server: dict[str, Any] | None, caps: Caps) -> bool:
    """The model server's context per slot is below min(the context cap, the model's own context); with
    the model's context unknown, below the cap. False when the server reports no context (an API)."""
    n_ctx = (server or {}).get("n_ctx")
    bounds = [x for x in (caps.context_tokens, (server or {}).get("n_ctx_train")) if isinstance(x, int)]
    return isinstance(n_ctx, int) and bool(bounds) and n_ctx < min(bounds)


def refusal_status(status: str, caps_hit: list[dict[str, Any]], server: dict[str, Any] | None, caps: Caps) -> str:
    """A model server's context refusal (caps.classify_exception) is the model's failure only if the
    server is not set below the model's own context: n_ctx >= min(the context cap, n_ctx_train), or the
    server reports no context (an API, whose context is the model's). Otherwise the episode is an
    infrastructure error (pre-registration §5.3)."""
    if status != "cap_failure" or not caps_hit:
        return status
    hit = caps_hit[0]
    if hit.get("cap") != CAP_CONTEXT or not str(hit.get("detail", "")).startswith(ENGINE_REFUSAL):
        return status
    return "infrastructure_error" if server_below_model(server, caps) else status


def run_episode(task: Task | Path | str, model_factory: ModelSpec | str | Callable[..., Any],
                setting: str = "benchmark", run_root: Path | None = None, *, config_id: str = "adhoc",
                sample: int = 0, caps: Caps | None = None, mode: str = "evaluation", reason: str = "",
                invalid_gate: str = "pinned", run_log_entry: dict[str, Any] | None = None,
                log_path: Path = runlog.LOG, run_list: Path = runlog.RUN_LIST, items_root: Path = ITEMS,
                include_thinking: bool = True, grade_episode: bool = False, export_state: bool = False,
                resume_from: Path | None = None, run_dir: Path | None = None) -> EpisodeResult:
    """Run one episode of `task` with the model `model_factory` (module docstring).

    task           a Task or a task directory
    model_factory  a ModelSpec (models.openai_compatible / scripted / factory), a factory reference
                   "module:function" / "file.py:function", or a top-level function returning a model
    setting        "benchmark" or "grc_agent_defaults" (agent_build.SETTINGS)
    run_root       parent of the run directories (default ~/grc-bench-work/tier_a)
    caps           override of Caps.for_setting(setting) (tests); recorded with the result
    mode           "evaluation" or "collection" (the run-log guard's mode)
    run_log_entry  set by run_tasks, which guards and logs a whole batch once; when None this
                   episode is guarded (and, on the test split, logged) on its own
    grade_episode  grade the end state (grader.grade) into result.grade; the trace verdict then
                   says whether the task's acceptance passed
    export_state, resume_from   module docstring (used by adhoc.py, never by the benchmark);
                   with resume_from the task must have no start files and no start graph
    run_dir        an explicit run directory under ~/grc-bench-work (adhoc.py's session layout)
                   instead of <run_root>/<config>/<task>/<setting>-sNN
    Raises runlog.RunRefused (nothing is started) when the guard refuses the run.
    """
    task = task if isinstance(task, Task) else load_task(task)
    spec = as_spec(model_factory)
    caps = caps or Caps.for_setting(setting)
    if invalid_gate not in ("pinned", "intended"):
        raise ValueError("invalid_gate must be 'pinned' or 'intended'")
    if resume_from is not None and (task.start_files or task.active_graph is not None):
        raise ValueError("a resumed episode starts from its parent's state: the task must have no start files")
    # a local model server reports its context per slot and the model's own (server_below_model)
    server = (model_server_context(spec.base_url, spec.uds_path)
              if spec.kind == "openai_compatible" and spec.relay is None else None)
    if server_below_model(server, caps):
        raise runlog.RunRefused(f"the model server's context per slot ({server['n_ctx']} tokens) is below the "
                                f"model's own ({server['n_ctx_train'] or 'not reported'}) and the "
                                f"{caps.context_tokens}-token cap: start it with a larger -c (or fewer slots)")
    if run_log_entry is None:
        run_log_entry = runlog.start_run(
            split=task.split, config_id=config_id, model=spec.describe(),
            settings=_settings(setting, caps, invalid_gate, spec.streams()), items=[task.id], item_splits=[task.split],
            reason=reason or f"Tier-A episode {task.id}", mode=mode, log_path=log_path,
            run_list=run_list, items_root=items_root)
    run_id = run_id_for(task.id, config_id, setting, sample)
    root = Path(run_root) if run_root is not None else isolation.RUNS_ROOT
    with contextlib.ExitStack() as stack:
        binds: list[Path] = []
        if spec.relay is not None:
            from .relay import LOCAL_BASE_URL, Relay
            relay = stack.enter_context(Relay(spec.relay["upstream"], spec.relay["api_key_env"],
                                              allow_http_loopback=bool(spec.relay.get("allow_http_loopback"))))
            spec = dataclasses.replace(spec, base_url=LOCAL_BASE_URL, uds_path=str(relay.socket_path))
            binds.append(relay.dir)
        target = Path(run_dir) if run_dir is not None else root / _safe(config_id) / _safe(task.id) / f"{setting}-s{sample:02d}"
        run_dir, env = prepare_run_dir(task, target, spec)
        episode = {"run_id": run_id, "task": task.to_dict(), "model": spec.to_dict(), "setting": setting,
                   "caps": caps.to_dict(), "config_id": config_id, "sample": sample, "mode": mode,
                   "invalid_gate": invalid_gate, "run_log_entry": run_log_entry}
        if export_state:
            episode["export_state"] = True
        if resume_from is not None:
            episode["resume"] = _carry_over(Path(resume_from), run_dir)
        write_json(run_dir / SPEC_FILE, episode)
        t0 = time.monotonic()
        proc = isolation.run_episode_process(run_dir, "grcbench.tier_a.runner",
                                             ["episode", "--spec", str(run_dir / SPEC_FILE)], env,
                                             timeout=caps.episode_seconds + PROCESS_MARGIN,
                                             network=spec.network(), extra_ro_binds=binds)
    elapsed = round(time.monotonic() - t0, 3)
    write_text(run_dir / STDERR_FILE, proc.stderr or "")
    process = {"returncode": proc.returncode, "timed_out": proc.timed_out, "seconds": elapsed,
               "network": spec.network(), "stderr_tail": (proc.stderr or "")[-4000:]}
    if (run_dir / RESULT_FILE).is_file():
        result = EpisodeResult.from_dict(read_json(run_dir / RESULT_FILE))
    else:
        result = _failed_result(episode, run_dir, task, spec, caps, proc.timed_out, process)
    result.process = process
    result.run_log_entry = run_log_entry
    if server is not None:
        result.harness = {**(result.harness or {}), "model_server": server}
        status = refusal_status(result.status, result.caps_hit, server, caps)
        if status != result.status:
            result.status = status
            result.end_reason = (f"model server set below the model's context (n_ctx {server['n_ctx']}, "
                                 f"model {server['n_ctx_train'] or 'not reported'}, cap {caps.context_tokens})")
    if grade_episode:
        from .grader import grade
        try:
            result.grade = grade(task, result)
        except Exception as exc:            # noqa: BLE001 - recorded; the episode itself stands
            result.grade = {"passed": False, "valid": False, "reasons": [f"grader failed: {exc}"],
                            "error": _error(exc)}
    write_json(run_dir / RESULT_FILE, result.to_dict())
    from .trace import export_run_dir, write_trace
    passed = (result.grade or {}).get("passed") if grade_episode and (result.grade or {}).get("valid", True) else None
    write_trace(run_dir / TRACE_FILE, [export_run_dir(run_dir, include_thinking=include_thinking,
                                                      acceptance_passed=passed)])
    return result


def _failed_result(episode: dict[str, Any], run_dir: Path, task: Task, spec: ModelSpec, caps: Caps,
                   timed_out: bool, process: dict[str, Any]) -> EpisodeResult:
    """The result of an episode process that ended without writing one."""
    if timed_out:
        hit = CapHit("episode_time", "episode process killed at the process timeout",
                     caps.episode_seconds + PROCESS_MARGIN)
        status, reason, hits = "cap_failure", "cap:episode_time", [hit.to_dict()]
    else:
        status, reason, hits = "infrastructure_error", "episode process failed", []
    return EpisodeResult(
        run_id=episode["run_id"], task_id=task.id, task_version=task.version, split=task.split,
        category=task.category, config_id=episode["config_id"], setting=episode["setting"],
        sample=episode["sample"], mode=episode["mode"], status=status, end_reason=reason,
        run_dir=str(run_dir), caps=caps.to_dict(), caps_hit=hits, model=spec.describe(),
        harness={"version": HARNESS_VERSION, "invalid_gate": episode["invalid_gate"], "streaming": spec.streams()},
        error={"type": "EpisodeProcess", "message": f"returncode={process['returncode']}",
               "traceback": process["stderr_tail"]})


def run_tasks(tasks: Iterable[Task | Path | str], model_factory: ModelSpec | str | Callable[..., Any], *,
              config_id: str, reason: str, setting: str = "benchmark", samples: int = 1,
              run_root: Path | None = None, mode: str = "evaluation", caps: Caps | None = None,
              invalid_gate: str = "pinned", log_path: Path = runlog.LOG, run_list: Path = runlog.RUN_LIST,
              items_root: Path = ITEMS, on_result: Callable[[EpisodeResult], None] | None = None,
              grade_episodes: bool = False) -> list[EpisodeResult]:
    """Run `samples` episodes of every task: one guard and one run-log entry for the whole batch,
    written before the first episode starts (eval plan §4.1)."""
    loaded = [t if isinstance(t, Task) else load_task(t) for t in tasks]
    if not loaded:
        return []
    spec = as_spec(model_factory)
    caps = caps or Caps.for_setting(setting)
    splits = sorted({t.split for t in loaded})
    settings = dict(_settings(setting, caps, invalid_gate, spec.streams()), samples=samples)
    entry = runlog.start_run(split=splits[0] if len(splits) == 1 else "+".join(splits), config_id=config_id,
                             model=spec.describe(), settings=settings, items=[t.id for t in loaded],
                             item_splits=splits, reason=reason, mode=mode, log_path=log_path,
                             run_list=run_list, items_root=items_root)
    results = []
    for t in loaded:
        for k in range(samples):
            res = run_episode(t, spec, setting, run_root, config_id=config_id, sample=k, caps=caps, mode=mode,
                              reason=reason, invalid_gate=invalid_gate, run_log_entry=entry or {"logged": False},
                              log_path=log_path, run_list=run_list, items_root=items_root,
                              grade_episode=grade_episodes)
            results.append(res)
            if on_result is not None:
                on_result(res)
    return results


# --------------------------------------------------------------------------------------------
# Episode process
# --------------------------------------------------------------------------------------------

@dataclass
class _State:
    history: list[Any] = field(default_factory=list)
    output: Any = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    stream: bool = False            # stream model requests (models.ModelSpec.streams())


async def _agent_turn(agent: Any, deps: Any, prompt: str | None, state: _State, meter: EpisodeMeter,
                      capability: HarnessCapability, conversation_id: str, turn: dict[str, Any],
                      stop: Callable[[], bool] | None = None) -> Any:
    """One user turn, resumed through every approval round (turn_driver._run_agent_turn).

    prompt=None continues `state.history` as it stands (pydantic-ai: a history that ends with a
    response holding unanswered tool calls has those calls executed first). `stop`, checked after
    every graph node, lets the harness end the turn early: the pending request (tool results) is
    appended to the history and HarnessStopped() is returned. With `state.stream` every model
    request node is streamed first (GRC-Agent's stream_view._stream_request), then advanced.
    """
    from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolApproved
    from pydantic_graph import End
    deferred: DeferredToolResults | None = None
    approvals: list[dict[str, Any]] = turn.setdefault("approvals", [])
    while True:
        active = None
        stopped_at = None
        try:
            async with agent.iter(prompt if deferred is None else None, message_history=state.history or None,
                                  deferred_tool_results=deferred, deps=deps, usage=meter.usage,
                                  usage_limits=meter.caps.usage_limits(), capabilities=[capability],
                                  conversation_id=conversation_id) as run:
                active = run
                node = run.next_node
                while not isinstance(node, End):
                    if run.result is not None:
                        break
                    if state.stream and Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as events:
                            async for _event in events:
                                pass
                    node = await run.next(node)
                    if stop is not None and not isinstance(node, End) and stop():
                        stopped_at = node
                        break
            result = run.result
        except BaseException:
            if active is not None:
                try:
                    state.history = list(active.all_messages())
                except Exception:           # noqa: BLE001 - keep the last known history
                    pass
            raise
        if stopped_at is not None:
            history = list(run.all_messages())
            request = getattr(stopped_at, "request", None)
            if Agent.is_model_request_node(stopped_at) and request is not None and request.parts:
                history.append(request)          # the tool results the model was about to be sent
            state.history = history
            state.output = HarnessStopped()
            return state.output
        state.history = list(result.all_messages())
        output = result.output
        state.output = output
        if not isinstance(output, DeferredToolRequests):
            return output
        if output.calls:
            raise UnresolvableDeferred(f"deferred external calls: {[c.tool_name for c in output.calls]}")
        approvals += [{"tool_call_id": c.tool_call_id, "tool": c.tool_name, "request": meter.usage.requests}
                      for c in output.approvals]
        deferred = DeferredToolResults(approvals={c.tool_call_id: ToolApproved() for c in output.approvals})


async def _conversation(agent: Any, deps: Any, task: Task, state: _State, meter: EpisodeMeter,
                        capability: HarnessCapability, conversation_id: str, t0: float) -> str:
    """The whole episode: instruction, approvals, scripted-user turns. Returns the end reason."""
    user = task.user()
    prompt, source = task.instruction, "instruction"
    while True:
        index = len(state.turns)
        meter.turn_index = index
        turn: dict[str, Any] = {"index": index, "source": source, "user_message": prompt,
                                "requests_before": meter.usage.requests,
                                "started_s": round(time.monotonic() - t0, 3)}
        state.turns.append(turn)
        output = await _agent_turn(agent, deps, prompt, state, meter, capability, conversation_id, turn)
        reply = reply_text(state.history, output)
        turn.update(requests_after=meter.usage.requests, agent_reply=reply,
                    output_kind=type(output).__name__, seconds=round(time.monotonic() - t0 - turn["started_s"], 3))
        if meter.hits:
            return f"cap:{meter.hits[0].cap}"
        nxt = user.respond(reply)
        if nxt is None:
            if user.waiting:
                return "agent_asked"            # no scripted reply (adhoc.py: answer by resuming)
            return "user_turns_exhausted" if user.exhausted else "agent_finished"
        prompt, source = nxt.message, nxt.source


def _project_files(project: Path) -> dict[str, str]:
    return {str(p.relative_to(project)): sha256_file(p)
            for p in sorted(project.rglob("*")) if p.is_file() and not p.is_symlink()}


def _final_output(output: Any) -> Any:
    if output is None or isinstance(output, (str, int, float, bool)):
        return output
    dump = getattr(output, "model_dump", None)
    if callable(dump):
        return {"type": type(output).__name__, **dump(mode="json")}
    return {"type": type(output).__name__, "repr": repr(output)[:2000]}


def _unknown_calls(history: list[Any], known: set[str]) -> list[dict[str, Any]]:
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    out = []
    if not known:
        return out
    for msg in history:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_name not in known:
                    out.append({"tool_call_id": part.tool_call_id, "tool": part.tool_name,
                                "raw_args": part.args if isinstance(part.args, str) else json.dumps(part.args)})
    return out


def _export_state(lay: isolation.RunLayout, page: Any, conversation_id: str) -> None:
    """state/live.grc + state/page.json (module docstring, export_state). A failure is written into
    page.json as "error"; the episode's result stands either way."""
    out = lay.run_dir / STATE_DIR
    out.mkdir(exist_ok=True)
    record: dict[str, Any] = {"file": None, "saved": False, "conversation_id": conversation_id}
    try:
        if page is None:
            raise RuntimeError("no active page")
        fg = page.flow_graph
        fg.parent_platform.save_flow_graph(str(out / LIVE_GRC), fg)
        if page.file_path and isolation.is_inside(Path(page.file_path), lay.project):
            record["file"] = Path(page.file_path).resolve().relative_to(lay.project.resolve()).as_posix()
            record["saved"] = bool(page.saved)
    except Exception as exc:                # noqa: BLE001 - recorded in page.json
        record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    write_json(out / PAGE_FILE, record)


def _error(exc: BaseException) -> dict[str, Any]:
    return {"type": type(exc).__name__, "message": str(exc)[:4000],
            "traceback": "".join(traceback.format_exception(exc))[-8000:]}


async def _episode(spec: dict[str, Any]) -> dict[str, Any]:
    from pydantic_ai.exceptions import ModelAPIError
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    from . import agent_build
    from .headless_proxy import HeadlessFlowgraphProxy

    t0 = time.monotonic()
    lay = isolation.check_process_isolation()
    tier_f = spec.get("kind") == "tier_f"
    if tier_f:
        from . import tier_f_mode
        task = tier_f_mode.task_from_spec(spec)
    else:
        task = task_from_dict(spec["task"])
    caps = Caps.from_dict(spec["caps"])
    model_spec = ModelSpec.from_dict(spec["model"])
    setting = spec["setting"]
    meter = EpisodeMeter(caps)
    capability = HarnessCapability(meter=meter)
    state = _State(stream=model_spec.streams())
    sample_record: dict[str, Any] | None = None
    status, end_reason, error = "completed", "agent_finished", None
    info: dict[str, Any] = {}
    proxy = None
    timing: dict[str, Any] = {}
    try:
        model = build_model(model_spec)
        active = task.active_graph
        kwargs = {"project_dir": lay.project, "max_run_seconds": caps.flowgraph_run_seconds,
                  "invalid_gate": spec.get("invalid_gate", "pinned")}
        if tier_f:
            tier_f_mode.materialise_start_graph(spec, lay)
            capability.policy = tier_f_mode.TierFBudget(prefill_id=spec["tier_f"]["prefill_id"], meter=meter)
            meter.prefilled_ids.add(spec["tier_f"]["prefill_id"])
        resume = spec.get("resume")
        if resume is not None:
            page = resume["page"]
            proxy = HeadlessFlowgraphProxy.restore(
                lay.run_dir / RESUME_DIR / LIVE_GRC, saved=bool(page.get("saved")),
                file_path=str(lay.project / page["file"]) if page.get("file") else "", **kwargs)
            state.history = list(ModelMessagesTypeAdapter.validate_json(
                (lay.run_dir / RESUME_DIR / MESSAGES_FILE).read_bytes()))
        else:
            proxy = (HeadlessFlowgraphProxy.open(lay.project / active, **kwargs) if active is not None
                     else HeadlessFlowgraphProxy.new_untitled(**kwargs))
        meter.flowgraph_cap_probe = lambda: next((r for r in proxy.runs if r.get("hard_cap_hit")), None)
        agent, info = agent_build.build_benchmark_agent(model, proxy, lay.project, setting)
        # key order kept (name, description, parameters; schemas as GRC-Agent builds them)
        write_text(lay.run_dir / TOOLS_FILE, json.dumps(await agent_build.freeze_tools(agent, proxy),
                                                        indent=1, ensure_ascii=False))
        timing["setup_seconds"] = round(time.monotonic() - t0, 3)
    except Exception as exc:                 # noqa: BLE001 - the harness could not start the episode
        status, end_reason, error = "infrastructure_error", "setup failed", _error(exc)
    conversation_id = (spec.get("resume") or {}).get("page", {}).get("conversation_id") or f"grcbench-{spec['run_id']}"
    if status == "completed":
        t1 = time.monotonic()
        try:
            async with asyncio.timeout(caps.episode_seconds):
                converse = tier_f_mode.conversation if tier_f else _conversation
                end_reason = await converse(agent, proxy, task, state, meter, capability,
                                            conversation_id, t0)
        except Exception as exc:             # noqa: BLE001 - classified below
            hit = classify_exception(exc, caps, meter.usage.requests)
            if hit is not None:
                meter.hit(hit)
                end_reason = f"cap:{hit.cap}"
            elif isinstance(exc, ModelAPIError):
                status, end_reason, error = "infrastructure_error", "model server error", _error(exc)
            else:
                status, end_reason, error = "agent_error", f"agent run raised {type(exc).__name__}", _error(exc)
        timing["agent_seconds"] = round(time.monotonic() - t1, 3)
        if meter.hits:
            status = "cap_failure"
            end_reason = f"cap:{meter.hits[0].cap}"
    final: dict[str, Any] = {}
    if proxy is not None:
        late = meter.flowgraph_cap()                     # a run killed during the last step
        if late is not None and status in ("completed", "agent_error") and not meter.hits:
            meter.hit(late)
            status, end_reason = "cap_failure", f"cap:{late.cap}"
        page = proxy.page
        if tier_f:
            try:
                sample_record = tier_f_mode.episode_record(spec, state, meter, lay, proxy, capability.policy)
            except Exception as exc:        # noqa: BLE001 - recorded; the host grades what exists
                sample_record = {"record_error": _error(exc)}
        if spec.get("export_state"):
            _export_state(lay, page, conversation_id)
        path = (page.file_path or None) if page is not None else None
        final = {"final_graph_path": path, "final_graph_saved": bool(page.saved) if page is not None else None,
                 "final_graph_sha256": sha256_file(Path(path)) if path and Path(path).is_file() else None,
                 "flowgraph_left_running": bool(page is not None and page.process is not None)}
        try:
            await proxy.aclose()
        except Exception as exc:            # noqa: BLE001 - teardown must not hide the result
            final["teardown_error"] = _error(exc)
        final["flowgraph_runs"] = proxy.runs
    timing["episode_seconds"] = round(time.monotonic() - t0, 3)
    (lay.run_dir / MESSAGES_FILE).write_bytes(ModelMessagesTypeAdapter.dump_json(state.history, indent=1))
    known = meter.tool_names | meter.output_tool_names
    result = EpisodeResult(
        run_id=spec["run_id"], task_id=task.id, task_version=task.version, split=task.split,
        category=task.category, config_id=spec["config_id"], setting=setting, sample=spec["sample"],
        mode=spec["mode"], status=status, end_reason=end_reason, run_dir=str(lay.run_dir),
        caps=caps.to_dict(), caps_hit=[h.to_dict() for h in meter.hits], model=model_spec.describe(),
        harness={"version": HARNESS_VERSION, "invalid_gate": spec.get("invalid_gate", "pinned"),
                 "streaming": state.stream, "approvals": "all approved (GRC-Agent yolo)",
                 "flowgraph_left_running": final.pop("flowgraph_left_running", None)},
        agent_info=info, final_graph_path=final.get("final_graph_path"),
        final_graph_saved=final.get("final_graph_saved"), final_graph_sha256=final.get("final_graph_sha256"),
        project_files=_project_files(lay.project), final_text=(state.turns[-1].get("agent_reply", "")
                                                               if state.turns else ""),
        final_output=_final_output(state.output), turns=state.turns, tool_calls=list(meter.calls.values()),
        unknown_tool_calls=_unknown_calls(state.history, known), model_requests=meter.requests,
        usage=meter.summary(), flowgraph_runs=final.get("flowgraph_runs", []), timing=timing,
        error=error if error is not None else final.get("teardown_error"),
        tier="F" if tier_f else "A", tier_f=sample_record)
    return result.to_dict()


def episode_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="grcbench.tier_a.runner episode")
    p.add_argument("--spec", required=True)
    a = p.parse_args(argv)
    spec = read_json(Path(a.spec))
    sys.modules["gi"] = None              # any GTK / PyGObject import now raises ImportError
    result = asyncio.run(_episode(spec))
    write_json(Path(result["run_dir"]) / RESULT_FILE, result)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "episode":
        return episode_main(argv[1:])
    print("usage: python -m grcbench.tier_a.runner episode --spec <run_dir>/episode.json")
    return 2


if __name__ == "__main__":
    sys.exit(main())

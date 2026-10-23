"""`./gb agent`: GRC-Agent, headless, on any request, with every output written as files.

The Tier-A runner (runner.py) runs GRC-Agent's real executor without GTK on a task folder. This
module turns a plain request into such a folder and runs it outside the benchmark:
  * the task is written with split "train" and run in COLLECTION mode, so the run-log guard refuses
    every benchmark item and nothing is logged as a benchmark run (grcbench.runlog.log);
  * the episode exports its end state (runner `export_state`), so a later call can continue the
    same conversation on the same graph (`continue_run`, runner `resume_from`): the next message
    can depend on what the agent said;
  * the model is a local OpenAI-compatible server (llama-server, the default) or a hosted
    OpenAI-compatible API through the relay (models.hosted, relay.py: the key stays on the host);
  * the limits are GRC-Agent's own ("gui" preset: the grc_agent_defaults setting, with GRC-Agent's
    compaction) or the benchmark's ("benchmark" preset), with per-cap overrides.

Follow-up messages inside one call:
  then        messages sent in order after each agent turn, whatever the agent said (a rule that
              always matches, as in Qoherent's two-turn sessions)
  replies     (regex, text) answers, sent when the agent asks something that matches
  auto_reply  the answer to any other question; None (the default) ends the call instead, with
              end reason "agent_asked", so the caller decides the answer and continues

Session folder  ~/grc-bench-work/agent/<name>/
    session.json        every call in order: run folder, parent, message, status, end reason, usage
                        and the model and limits (reused by `continue_run` unless overridden)
    tasks/call-NN/      the task folder written for call NN (task.json + start/)
    call-NN/            the run folder of call NN (runner.py "Run directory"), plus:
        summary.json    status, end reason, final reply, turns, every tool call of this call with its
                        arguments, outcome and result text, flowgraph runs, files added/changed/
                        removed (GRC-Agent's own .grc_agent/ lock and backup files left out), the
                        graph (its file in the project folder, GRC's saved flag, whether that file
                        holds the same graph as GRC's memory, validity), usage (and the session's
                        totals), model, limits. GRC-Agent's change_graph writes each edit through
                        to the graph's file when it has one, and GRC then marks the page modified,
                        so "page_saved_flag" can be false while "file_matches_graph" is true.
        transcript.md   the whole conversation so far as plain text (from trace.jsonl)
        graph.grc       the current flowgraph, saved or not (GRC's own file format)
        graph_view.json what GNU Radio makes of it: blocks, parameters with their evaluated values,
                        connections, validation errors (_grc_view.views)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from ..common.io import read_json, sha256_file, write_json, write_text
from ..common.paths import WORK
from . import isolation, models
from .caps import Caps
from .models import ModelSpec

AGENT_ROOT = WORK / "agent"
CONFIG_ID = "agent"
PRESETS = {"gui": "grc_agent_defaults", "benchmark": "benchmark"}
CAP_FIELDS = ("requests", "context_tokens", "tool_output_tokens", "episode_seconds", "flowgraph_run_seconds")
SESSION_FILE, SUMMARY_FILE, TRANSCRIPT_FILE = "session.json", "summary.json", "transcript.md"
GRAPH_FILE, VIEW_FILE = "graph.grc", "graph_view.json"
ALWAYS = r"[\s\S]*"                     # a `then` rule matches any agent reply
DEFAULT_AUTO_TURNS = 3                  # scripted replies allowed on top of `then` and `replies`


class AgentRunError(ValueError):
    """The request cannot be run (bad option, missing file, not a finished run folder)."""


# --------------------------------------------------------------------------------------------
# Requests -> task folders
# --------------------------------------------------------------------------------------------

def _scripted_user(then: list[str], replies: list[tuple[str, str]], auto_reply: str | None,
                   max_user_turns: int | None) -> dict[str, Any]:
    rules = [{"pattern": p, "reply": r, "require_question": True} for p, r in replies]
    rules += [{"pattern": ALWAYS, "reply": m, "require_question": False} for m in then]
    turns = max_user_turns if max_user_turns is not None else \
        len(then) + len(replies) + (DEFAULT_AUTO_TURNS if auto_reply else 0)
    return {"rules": rules, "default_reply": auto_reply, "max_turns": turns}


def write_task(task_dir: Path, task_id: str, message: str, *, graph: Path | None = None,
               files: Path | None = None, then: list[str] = (), replies: list[tuple[str, str]] = (),
               auto_reply: str | None = None, max_user_turns: int | None = None) -> Path:
    """A "train"-split task folder for one request (task.py format; acceptance empty: never graded).

    `files` (a folder) is copied into start/ and so into the agent's project folder; `graph` is
    copied there too (or found inside `files`) and opened as the active flowgraph."""
    if not message or not message.strip():
        raise AgentRunError("the message is empty")
    if task_dir.exists():
        shutil.rmtree(task_dir)
    start = task_dir / "start"
    start.mkdir(parents=True)
    if files is not None:
        files = Path(files)
        if not files.is_dir():
            raise AgentRunError(f"--files must be a folder: {files}")
        for src in sorted(files.rglob("*")):
            if src.is_symlink():
                raise AgentRunError(f"--files must not contain symbolic links: {src}")
            if src.is_file():
                dest = start / src.relative_to(files)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
    active = None
    if graph is not None:
        graph = Path(graph)
        if not graph.is_file():
            raise AgentRunError(f"--graph is not a file: {graph}")
        if files is not None and isolation.is_inside(graph, files):
            active = graph.resolve().relative_to(Path(files).resolve()).as_posix()
        else:
            shutil.copyfile(graph, start / graph.name)
            active = graph.name
    record = {                              # no canary: this is a training-side record (task.py)
        "id": task_id, "tier": "A", "split": "train", "version": 1,
        "license": "not for release (ad-hoc request)", "source": {"kind": "adhoc_request"},
        "cluster_id": task_id, "provenance": {"drafted_by": "the requester", "generator": "grcbench.tier_a.adhoc"},
        "review": {"status": "not-required"}, "category": "modify" if active else "build",
        "instruction": message, "start_state": {"active_graph": active},
        "scripted_user": _scripted_user(list(then), list(replies), auto_reply, max_user_turns),
        "acceptance": {},
    }
    write_json(task_dir / "task.json", record)
    return task_dir


# --------------------------------------------------------------------------------------------
# Limits and models
# --------------------------------------------------------------------------------------------

def make_caps(preset: str, overrides: dict[str, Any] | None = None) -> tuple[str, Caps]:
    """(setting, caps) for a preset ("gui" or "benchmark") with per-cap overrides (None = no cap
    for requests, context_tokens and tool_output_tokens)."""
    if preset not in PRESETS:
        raise AgentRunError(f"unknown preset {preset!r}; expected one of {sorted(PRESETS)}")
    setting = PRESETS[preset]
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(CAP_FIELDS)
    if unknown:
        raise AgentRunError(f"unknown limits {sorted(unknown)}; expected {list(CAP_FIELDS)}")
    for key in ("episode_seconds", "flowgraph_run_seconds"):
        if key in overrides and not (isinstance(overrides[key], (int, float)) and overrides[key] > 0):
            raise AgentRunError(f"{key} must be a positive number of seconds")
    return setting, Caps.for_setting(setting, **overrides)


def model_spec(*, base_url: str | None = None, model: str | None = None, uds: str | None = None,
               api_base: str | None = None, api_key_env: str | None = None, stream: bool | None = None) -> ModelSpec:
    """A local llama-server (base_url / uds) or a hosted OpenAI-compatible API (api_base +
    api_key_env, the NAME of the variable holding the key)."""
    if api_base or api_key_env:
        if not (api_base and api_key_env and model):
            raise AgentRunError("a hosted model needs --api-base, --api-key-env and --model")
        if base_url or uds:
            raise AgentRunError("use either a local server (--base-url/--uds) or a hosted API (--api-base)")
        return models.hosted(api_base, model, api_key_env, stream=stream)
    return models.openai_compatible(base_url or isolation.DEFAULT_BASE_URL, model or isolation.DEFAULT_MODEL,
                                    uds_path=uds, stream=stream)


# --------------------------------------------------------------------------------------------
# Running calls
# --------------------------------------------------------------------------------------------

def _new_name() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def _session(root: Path, name: str) -> Path:
    from .runner import _safe
    session = isolation.check_run_dir(Path(root) / _safe(name))
    session.mkdir(parents=True, exist_ok=True)
    return session


def _calls(session: Path) -> list[dict[str, Any]]:
    path = session / SESSION_FILE
    return read_json(path)["calls"] if path.is_file() else []


def run(message: str, *, graph: Path | str | None = None, files: Path | str | None = None,
        then: list[str] = (), replies: list[tuple[str, str]] = (), auto_reply: str | None = None,
        max_user_turns: int | None = None, name: str | None = None, model: ModelSpec | None = None,
        preset: str = "gui", caps: dict[str, Any] | None = None, root: Path = AGENT_ROOT) -> dict[str, Any]:
    """Start a session with one request; returns the call's summary (also in summary.json)."""
    session = _session(root, name or _new_name())
    if _calls(session):
        raise AgentRunError(f"session {session.name} already exists: continue it, or pick another --name")
    return _call(session, message, graph=Path(graph).expanduser() if graph else None,
                 files=Path(files).expanduser() if files else None,
                 then=then, replies=replies, auto_reply=auto_reply, max_user_turns=max_user_turns,
                 model=model or model_spec(), preset=preset, caps=caps, parent=None)


def continue_run(run_dir: Path | str, message: str, *, then: list[str] = (),
                 replies: list[tuple[str, str]] = (), auto_reply: str | None = None,
                 max_user_turns: int | None = None, model: ModelSpec | None = None,
                 preset: str | None = None, caps: dict[str, Any] | None = None) -> dict[str, Any]:
    """The next message of a session, from the end state of its call in `run_dir` (any earlier
    call works too: the session then branches). Model and limits default to that call's."""
    parent = isolation.check_run_dir(Path(run_dir).expanduser())
    session = parent.parent
    previous = next((c for c in _calls(session) if Path(c["run_dir"]).resolve() == parent), None)
    if previous is None:
        raise AgentRunError(f"{parent} is not a call of an `./gb agent` session")
    if model is None:
        model = ModelSpec.from_dict(previous["model_spec"])
    if preset is None:
        preset = previous["preset"]
    if caps is None:
        caps = previous["caps_overrides"]
    return _call(session, message, graph=None, files=None, then=then, replies=replies, auto_reply=auto_reply,
                 max_user_turns=max_user_turns, model=model, preset=preset, caps=caps, parent=parent)


def _call(session: Path, message: str, *, graph: Path | None, files: Path | None, then: list[str],
          replies: list[tuple[str, str]], auto_reply: str | None, max_user_turns: int | None,
          model: ModelSpec, preset: str, caps: dict[str, Any] | None, parent: Path | None) -> dict[str, Any]:
    from .runner import _safe, run_episode
    from .task import load_task
    setting, cap_values = make_caps(preset, caps)
    calls = _calls(session)
    n = len(calls) + 1
    task_id = f"agent-{_safe(session.name)}-c{n:02d}"
    task_dir = write_task(session / "tasks" / f"call-{n:02d}", task_id, message, graph=graph, files=files,
                          then=then, replies=replies, auto_reply=auto_reply, max_user_turns=max_user_turns)
    task = load_task(task_dir)
    start = ({str(p.as_posix()): sha256_file(task.start_dir / p) for p in task.start_files} if parent is None
             else read_json(parent / "result.json").get("project_files", {}))
    run_dir = session / f"call-{n:02d}"
    run_episode(task, model, setting, config_id=CONFIG_ID, caps=cap_values, mode="collection",
                reason=f"./gb agent session {session.name} call {n}", export_state=True,
                resume_from=parent, run_dir=run_dir)
    entry = {"call": n, "run_dir": str(run_dir), "parent": str(parent) if parent else None, "message": message,
             "model_spec": model.to_dict(), "preset": preset, "caps_overrides": dict(caps or {})}
    summary = write_outputs(run_dir, session=session, call=n, parent=parent, message=message, start=start,
                            previous_calls=calls)
    entry.update(status=summary["status"], end_reason=summary["end_reason"], usage=summary["usage"])
    write_json(session / SESSION_FILE, {"session": session.name, "calls": [*calls, entry]})
    return summary


# --------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------

GRC_AGENT_STATE = ".grc_agent/"        # GRC-Agent's own lock and backup files in the project folder


def _file_changes(start: dict[str, str], end: dict[str, str]) -> dict[str, list[str]]:
    start = {k: v for k, v in start.items() if not k.startswith(GRC_AGENT_STATE)}
    end = {k: v for k, v in end.items() if not k.startswith(GRC_AGENT_STATE)}
    return {"added": sorted(set(end) - set(start)), "removed": sorted(set(start) - set(end)),
            "changed": sorted(k for k in set(start) & set(end) if start[k] != end[k])}


def _graph_views(run_dir: Path, file_in_project: str | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """GNU Radio's view of the graph in memory (state/live.grc) and of the page's file on disk."""
    from ._grc_view import views
    live = run_dir / "state" / "live.grc"
    if not live.is_file():
        return {"ok": False, "error": "no exported graph (see state/page.json and episode.log)"}, None
    graphs: dict[str, Path | None] = {"graph": live}
    if file_in_project:
        graphs["file"] = run_dir / "project" / file_in_project
    try:
        out = views(graphs, library_dir=isolation.RunLayout(run_dir).hier_dir)["graphs"]
        return out["graph"], out.get("file")
    except Exception as exc:                # noqa: BLE001 - recorded in the view file
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}, None


def _same_graph(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool | None:
    if not a or not b or not a.get("ok") or not b.get("ok"):
        return None
    return a.get("blocks") == b.get("blocks") and a.get("connections") == b.get("connections")


def transcript(row: dict[str, Any], title: str) -> str:
    """The conversation of a trace row (trace.episode_row) as Markdown text."""
    out = [f"# {title}", ""]
    for m in row.get("messages", []):
        role = m.get("role")
        if role == "user":
            out += ["## User", "", m["content"], ""]
        elif role == "assistant":
            out += ["## Agent", ""]
            if m.get("thinking"):
                out += ["<details><summary>thinking</summary>", "", m["thinking"], "", "</details>", ""]
            if m.get("content"):
                out += [m["content"], ""]
            for c in m.get("tool_calls", []):
                out += [f"**Tool call** `{c['name']}` ({c['id']})", "", "```json", c["arguments"], "```", ""]
        elif role == "tool":
            out += [f"**Result** of `{m['name']}` ({m['tool_call_id']})", "", "```", m["content"], "```", ""]
    return "\n".join(out)


def write_outputs(run_dir: Path, *, session: Path, call: int, parent: Path | None, message: str,
                  start: dict[str, str], previous_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """summary.json, transcript.md, graph.grc and graph_view.json for a finished call."""
    result = read_json(run_dir / "result.json")
    trace = run_dir / "trace.jsonl"
    row = json.loads(trace.read_text(encoding="utf-8").splitlines()[0]) if trace.is_file() else {}
    write_text(run_dir / TRANSCRIPT_FILE, transcript(row, f"GRC-Agent session {session.name}, call {call}"))
    results = {m["tool_call_id"]: m["content"] for m in row.get("messages", []) if m.get("role") == "tool"}
    live = run_dir / "state" / "live.grc"
    if live.is_file():
        shutil.copyfile(live, run_dir / GRAPH_FILE)
    page = read_json(run_dir / "state" / "page.json") if (run_dir / "state" / "page.json").is_file() else {}
    view, file_view = _graph_views(run_dir, page.get("file"))
    write_json(run_dir / VIEW_FILE, view)
    usage = result.get("usage", {})
    totals = {k: sum(int((c.get("usage") or {}).get(k) or 0) for c in previous_calls) + int(usage.get(k) or 0)
              for k in ("requests", "input_tokens", "output_tokens", "tool_calls")}
    error = result.get("error") or None
    summary = {
        "session": session.name, "call": call, "run_dir": str(run_dir),
        "parent": str(parent) if parent else None, "message": message,
        "status": result["status"], "end_reason": result["end_reason"],
        "waiting_for_reply": result["end_reason"] == "agent_asked",
        "final_reply": result.get("final_text", ""),
        "turns": [{"user": t.get("user_message"), "source": t.get("source"), "agent": t.get("agent_reply")}
                  for t in result.get("turns", [])],
        "tool_calls": [{"tool": c["tool"], "tool_call_id": c["tool_call_id"], "turn": c.get("turn"),
                        "arguments": c.get("raw_args"), "validation": c.get("validation"),
                        "execution": c.get("execution"), "result": results.get(c["tool_call_id"])}
                       for c in result.get("tool_calls", [])],
        "flowgraph_runs": [{k: r.get(k) for k in ("command", "return_code", "program_return_code", "wall_seconds",
                                                    "hard_cap_hit", "spawn_error")}
                           for r in result.get("flowgraph_runs", [])],
        "graph": {"file_in_project": page.get("file"), "page_saved_flag": page.get("saved"),
                  "file_matches_graph": _same_graph(view, file_view),
                  "exported": live.is_file(), "valid": view.get("valid"), "errors": view.get("errors", []),
                  "export_error": page.get("error")},
        "files": _file_changes(start, result.get("project_files", {})),
        "usage": usage, "session_totals": totals,
        "setting": result.get("setting"), "caps": result.get("caps"), "caps_hit": result.get("caps_hit", []),
        "model": result.get("model"),
        "error": {"type": error.get("type"), "message": error.get("message")} if error else None,
        "outputs": {"summary": SUMMARY_FILE, "transcript": TRANSCRIPT_FILE, "graph": GRAPH_FILE,
                    "graph_view": VIEW_FILE, "trace": "trace.jsonl", "messages": "messages.json",
                    "result": "result.json", "project": "project/", "log": "episode.log"},
    }
    write_json(run_dir / SUMMARY_FILE, summary)
    return summary


# --------------------------------------------------------------------------------------------
# Self-test (no model server needed)
# --------------------------------------------------------------------------------------------

SELFTEST_BUILD = {"name": "change_graph", "args": {
    "reason": "Build a 1 kHz tone into a file.",
    "add_blocks": [
        {"block_id": "analog_sig_source_x", "instance_name": "src",
         "params": {"type": "float", "samp_rate": "samp_rate", "waveform": "analog.GR_COS_WAVE",
                    "freq": "1000", "amp": "1"}},
        {"block_id": "blocks_throttle2", "instance_name": "thr",
         "params": {"type": "float", "samples_per_second": "samp_rate"}},
        {"block_id": "blocks_file_sink", "instance_name": "out", "params": {"type": "float", "file": "selftest.dat"}}],
    "add_connections": ["src:0->thr:0", "thr:0->out:0"]}}


def _fake_provider() -> tuple[Any, list[dict[str, Any]]]:
    """A local OpenAI-compatible stand-in on 127.0.0.1 (one streamed reply per request), for the
    relay check; returns (server, requests seen: path and whether a bearer token arrived)."""
    import http.server
    import threading
    seen: list[dict[str, Any]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("content-length") or 0))
            seen.append({"path": self.path, "bearer": (self.headers.get("authorization") or "").startswith("Bearer ")})
            wire = models.WireScript([{"text": "Relay works."}], "text", "fake", Path("/dev/null"))
            body = ("".join(f"data: {json.dumps(c)}\n\n" for c in wire.stream_chunks(wire.completion(0)))
                    + "data: [DONE]\n\n").encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def selftest(*, keep: bool = False, root: Path = AGENT_ROOT) -> int:
    """Check the whole setup with scripted models: GRC-Agent's executor in the sandbox, GNU Radio,
    a real flowgraph run, the output files, `continue`, and the hosted-model relay (against a local
    stand-in, with a throwaway key). Prints one line per check; the session folders are removed
    unless `keep` or a check failed."""
    import os
    results: list[tuple[bool, str]] = []

    def check(ok: Any, what: str) -> None:
        results.append((bool(ok), what))
        print(f"{'ok  ' if ok else 'FAIL'}  {what}", flush=True)

    name = "selftest-" + _new_name()
    sessions = [root / name, root / (name + "-relay")]
    try:
        first = run("Build a 1 kHz tone into a file, save it and run it.", name=name, root=root, model=models.scripted([
            {"tool_calls": [SELFTEST_BUILD]}, {"tool_calls": [{"name": "save_graph", "args": {}}]},
            {"tool_calls": [{"name": "run_flowgraph", "args": {"action": "start", "stop_after_seconds": 1}}]},
            {"text": "Done."}]))
        one = Path(first["run_dir"])
        check(first["status"] == "completed", f"GRC-Agent ran headless in the sandbox (status {first['status']})")
        check([c["execution"] for c in first["tool_calls"]] == ["ok", "ok", "ok"],
              "change_graph, save_graph and run_flowgraph executed")
        runs = first["flowgraph_runs"]
        check(runs and not runs[0].get("spawn_error") and not runs[0].get("hard_cap_hit"),
              f"the flowgraph ran in GNU Radio, sandboxed (return code {runs[0].get('return_code') if runs else None})")
        data = one / "project" / "selftest.dat"
        check(data.is_file() and data.stat().st_size > 0, "the flowgraph wrote project/selftest.dat")
        check(first["graph"]["valid"] and first["graph"]["file_matches_graph"],
              "graph.grc / graph_view.json: valid, the saved file matches GRC's memory")
        check(all((one / f).is_file() for f in (SUMMARY_FILE, TRANSCRIPT_FILE, "trace.jsonl", GRAPH_FILE, VIEW_FILE)),
              "summary.json, transcript.md, trace.jsonl, graph.grc and graph_view.json written")
        second = continue_run(one, "Thanks.", model=models.scripted([{"text": "You're welcome."}]))
        row = json.loads((Path(second["run_dir"]) / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0])
        users = [m["content"] for m in row["messages"] if m["role"] == "user"]
        check(second["status"] == "completed" and len(users) == 2 and users[1] == "Thanks.",
              "continue: the next call has the whole conversation")
        key_env = "GRC_AGENT_HEADLESS_SELFTEST_KEY"
        os.environ[key_env] = "selftest-" + uuid.uuid4().hex
        server, seen = _fake_provider()
        try:
            hosted = ModelSpec("openai_compatible", model_name="fake", relay={
                "upstream": f"http://127.0.0.1:{server.server_port}/v1", "api_key_env": key_env,
                "allow_http_loopback": True})
            third = run("Say something.", name=name + "-relay", root=root, model=hosted)
        finally:
            server.shutdown()
        check(third["final_reply"] == "Relay works." and seen and all(s["bearer"] for s in seen),
              "hosted-model relay: the request went out with the key and the reply came back")
        leaked = [p for p in (root / (name + "-relay")).rglob("*")
                  if p.is_file() and os.environ[key_env].encode() in p.read_bytes()]
        check(not leaked, "hosted-model relay: the key is in no output file")
        os.environ.pop(key_env, None)
    except Exception as exc:                # noqa: BLE001 - reported as a failed check
        check(False, f"self-test stopped: {type(exc).__name__}: {exc}")
    failed = [w for ok, w in results if not ok]
    if failed or keep:
        print(f"session folders kept: {', '.join(str(s) for s in sessions if s.exists())}")
    else:
        for s in sessions:
            shutil.rmtree(s, ignore_errors=True)
    print(f"self-test: {len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def _limit(text: str) -> int | None:
    return None if text.lower() == "none" else int(text)


def _reply(text: str) -> tuple[str, str]:
    pattern, sep, reply = text.partition("=")
    if not sep or not pattern or not reply:
        raise argparse.ArgumentTypeError("--reply takes REGEX=TEXT")
    return pattern, reply


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--then", action="append", default=None, metavar="MESSAGE",
                   help="a follow-up message sent after the agent's turn (repeatable, in order)")
    p.add_argument("--reply", action="append", type=_reply, default=None, metavar="REGEX=TEXT",
                   help="answer TEXT when the agent asks something matching REGEX (repeatable)")
    p.add_argument("--auto-reply", default=None, metavar="TEXT",
                   help="answer to any other question (default: stop and wait, end reason agent_asked)")
    p.add_argument("--max-user-turns", type=int, default=None)
    m = p.add_argument_group("model (default: the local llama-server)")
    m.add_argument("--base-url", default=None, help=f"OpenAI-compatible server (default {isolation.DEFAULT_BASE_URL})")
    m.add_argument("--uds", default=None, help="the local server's UNIX socket")
    m.add_argument("--model", default=None, help="model name (required with --api-base)")
    m.add_argument("--api-base", default=None, help="hosted OpenAI-compatible API, e.g. https://openrouter.ai/api/v1")
    m.add_argument("--api-key-env", default=None, help="NAME of the environment variable holding the API key")
    m.add_argument("--no-stream", action="store_true", help="do not stream model requests")
    c = p.add_argument_group("limits")
    c.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="gui (default): GRC-Agent's own settings; benchmark: the benchmark's caps")
    c.add_argument("--max-requests", type=_limit, default=argparse.SUPPRESS, metavar="N|none")
    c.add_argument("--context-tokens", type=_limit, default=argparse.SUPPRESS, metavar="N|none")
    c.add_argument("--tool-output-tokens", type=_limit, default=argparse.SUPPRESS, metavar="N|none")
    c.add_argument("--episode-seconds", type=float, default=argparse.SUPPRESS)
    c.add_argument("--flowgraph-seconds", type=float, default=argparse.SUPPRESS)
    p.add_argument("--request", default=None, help="a JSON file with the same fields (CLI options override it)")
    p.add_argument("--json", action="store_true", help="print summary.json instead of the short report")


_CAP_OPTIONS = {"max_requests": "requests", "context_tokens": "context_tokens",
                "tool_output_tokens": "tool_output_tokens", "episode_seconds": "episode_seconds",
                "flowgraph_seconds": "flowgraph_run_seconds"}


def _request(a: argparse.Namespace) -> dict[str, Any]:
    """The request: the --request JSON file, then every option given on the command line."""
    req: dict[str, Any] = read_json(Path(a.request).expanduser()) if a.request else {}
    for key in ("message", "graph", "files", "name", "run_dir", "then", "reply", "auto_reply", "max_user_turns",
                "base_url", "uds", "model", "api_base", "api_key_env", "preset"):
        value = getattr(a, key, None)
        if value is not None:
            req[key] = value
    if a.no_stream:
        req["stream"] = False
    caps = dict(req.get("caps") or {})
    for opt, field in _CAP_OPTIONS.items():
        if hasattr(a, opt):
            caps[field] = getattr(a, opt)
    if caps:
        req["caps"] = caps
    return req


def _model_from(req: dict[str, Any]) -> ModelSpec | None:
    keys = ("base_url", "uds", "model", "api_base", "api_key_env", "stream")
    if not any(k in req for k in keys):
        return None
    return model_spec(base_url=req.get("base_url"), model=req.get("model"), uds=req.get("uds"),
                      api_base=req.get("api_base"), api_key_env=req.get("api_key_env"), stream=req.get("stream"))


def _replies(req: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for r in req.get("reply") or req.get("replies") or []:
        out.append((r["pattern"], r["reply"]) if isinstance(r, dict) else (r[0], r[1]))
    return out


def _report(summary: dict[str, Any]) -> str:
    lines = [f"session {summary['session']}, call {summary['call']}: {summary['status']} ({summary['end_reason']})",
             f"run folder: {summary['run_dir']}"]
    if summary["waiting_for_reply"]:
        lines.append("the agent asked a question; answer with: continue <run folder> \"<answer>\"")
    g = summary["graph"]
    where = (f"project/{g['file_in_project']}" + ("" if g["file_matches_graph"] else " (differs from the graph in memory)")
             if g["file_in_project"] else "untitled, never saved")
    lines.append(f"graph: {'valid' if g['valid'] else 'NOT valid'}, {where} (graph.grc, graph_view.json)")
    f = summary["files"]
    if any(f.values()):
        lines.append("files: " + "; ".join(f"{k} {', '.join(v)}" for k, v in f.items() if v))
    u = summary["usage"]
    lines.append(f"requests {u.get('requests')}, tokens in {u.get('input_tokens')} / out {u.get('output_tokens')}, "
                 f"tool calls {len(summary['tool_calls'])}")
    if summary["error"]:
        lines.append(f"error: {summary['error']['type']}: {summary['error']['message'][:300]}")
    lines += ["", "final reply:", summary["final_reply"] or "(none)"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="gb agent", description="Run GRC-Agent headless on a request; every "
                                "output is written to the call's run folder (TUTORIAL.md of grc-agent-headless; "
                                "in GRC-Bench: handoff/agent_package/TUTORIAL.md).")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="start a session with a message")
    r.add_argument("message", nargs="?", default=None)
    r.add_argument("--graph", default=None, help="a .grc file opened as the active flowgraph")
    r.add_argument("--files", default=None, help="a folder copied into the agent's project folder")
    r.add_argument("--name", default=None, help="session name (default: date-time)")
    _common(r)
    c = sub.add_parser("continue", help="send the next message of a session")
    c.add_argument("run_dir", nargs="?", default=None, help="a call's run folder (normally the last one)")
    c.add_argument("message", nargs="?", default=None)
    _common(c)
    s = sub.add_parser("show", help="print a call's summary")
    s.add_argument("run_dir")
    s.add_argument("--transcript", action="store_true", help="print transcript.md instead")
    t = sub.add_parser("selftest", help="check the setup with scripted models (no model server needed)")
    t.add_argument("--keep", action="store_true", help="keep the self-test's session folders")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest(keep=a.keep)
    from ..runlog.log import RunRefused
    from .relay import RelayError
    try:
        if a.cmd == "show":
            run_dir = Path(a.run_dir).expanduser()
            print((run_dir / TRANSCRIPT_FILE).read_text(encoding="utf-8") if a.transcript
                  else json.dumps(read_json(run_dir / SUMMARY_FILE), indent=1, ensure_ascii=False))
            return 0
        req = _request(a)
        if not req.get("message"):
            raise AgentRunError("a message is required (argument or the request file's \"message\")")
        common = {"then": list(req.get("then") or []), "replies": _replies(req), "auto_reply": req.get("auto_reply"),
                  "max_user_turns": req.get("max_user_turns"), "model": _model_from(req), "caps": req.get("caps")}
        if a.cmd == "run":
            summary = run(req["message"], graph=req.get("graph"), files=req.get("files"), name=req.get("name"),
                          preset=req.get("preset") or "gui", **common)
        else:
            if not req.get("run_dir"):
                raise AgentRunError("continue needs the run folder of a call")
            summary = continue_run(req["run_dir"], req["message"], preset=req.get("preset"), **common)
    except (AgentRunError, RunRefused, RelayError, ValueError, OSError) as exc:
        print(f"refused: {exc}")
        return 2
    print(json.dumps(summary, indent=1, ensure_ascii=False) if a.json else _report(summary))
    return 0 if summary["status"] != "infrastructure_error" else 1


if __name__ == "__main__":
    sys.exit(main())

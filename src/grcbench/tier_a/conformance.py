"""Conformance of the benchmark agent with GRC-Agent's real tool definitions and prompt.

Eval plan §4.4: "One pinned GRC-Agent version and its real tool definitions". Qoherent's SFT
release (grc-agent-sft-v0.1.0) records, per session, the `system` prompt the model saw and the
`tools` snapshot of the executor (collect.py freeze_tools). This module checks that the agent
`agent_build.build_benchmark_agent` makes exposes the same:

  (i)  tools: the tool definitions the model is offered, captured as pydantic-ai ToolDefinitions
       (FunctionModel AgentInfo in both settings and with HeadlessFlowgraphProxy deps, and Qoherent's
       own toolset snapshot), are compared
       field by field with the SFT `tools`; the only differences allowed are KNOWN_TOOL_DIFFS, each
       traced to its cause; the pinned commit is authoritative for the benchmark. On the wire, the
       JSON body llama-server receives (GRC-Agent's OpenAIChatModel path) must carry exactly the
       tool JSON the collection teacher received (pydantic-ai's OllamaModel path, same agent).
  (ii) system prompt: the SFT prompts are build_system_prompt("pai-desktop-chat") followed by
       "\\n\\nActive flowgraph file path: <path>"; only the path varies per session. The fixed part
       must match exactly; the varying part must name the episode's own graph.
  (iii) isolation: the build runs in an episode process (isolation.run_episode_process) where any
       `import gi` fails, every GRC-Agent path points into the run directory, and a file-tree
       manifest of ~/grc-agent (including git-ignored files) is identical before and after. The
       selftest also drives the offline web tools and the sandboxed shell tools through the agent.

    python -m grcbench.tier_a.conformance selftest --graph G --out F   # inside an episode process
    ./gb tier-a conformance                                           # host side: run + report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

from ..common import manifest
from ..common.io import read_json, sha256_file, write_json, write_text
from ..common.paths import FIXTURES, GRC_AGENT, GRC_AGENT_COMMIT, HOME, REPORTS, SFT_RELEASE
from . import isolation

SFT_FILE = SFT_RELEASE / "sft_train.jsonl"
ACTIVE_PATH = re.compile(r"\n\nActive flowgraph file path: (?P<path>[^\n]+)\Z")
UNSAVED_NOTE = ("\n\nActive flowgraph: a live in-memory graph that has never been written to disk "
                "(no file path). A .grc file in the project directory is a different graph, "
                "even when its name matches.")
PROBE_GRAPH = FIXTURES / "tier_a" / "probe_tone.grc"     # full .grc format (GRC's own writer)
REPORT = REPORTS / "tier_a_conformance.md"
WATCHED_ROOTS = {"~/grc-agent": GRC_AGENT, "~/.config/grc_agent": HOME / ".config" / "grc_agent",
                 "~/.cache/gnuradio": HOME / ".cache" / "gnuradio"}

# Every difference between the SFT `tools` and the pinned commit 6c98128, with its cause. Found by
# the first conformance run on 14 Oct 2026; GRC-Agent history read with `git log --all -S` (read
# only). The SFT rows were collected on 12 Sep 2026 with a local GRC-Agent checkout
# (/opt/grc_agent, Qoherent's container); 6c98128 is dated 11 Sep 2026; main has no later code
# commit before 18 Sep (7d0de8c, README only).
KNOWN_TOOL_DIFFS: list[dict[str, Any]] = [
    {"tool": "run_command", "field": "parameters", "path": "/properties/timeout_seconds/description",
     "sft": "Maximum seconds to wait (default: GRC_SHELL_TIMEOUT, 120s).",
     "pinned": "Maximum seconds to wait (default: GRC_SHELL_TIMEOUT, 600s).",
     "cause": "collection setting (the shell timeout), not a schema change",
     "evidence": "The text is generated, not fixed: GrcShellToolset._apply_exec_approval (shell_tools.py "
                 "L229-244 at 6c98128) writes f\"{default_timeout():g}s\" into it, and default_timeout() "
                 "(L104-113) returns GRC_SHELL_TIMEOUT, else 600. 120 s is the value Qoherent used: "
                 "GRC-Agent commit e437987 (28 Aug) records \"GRC_SHELL_TIMEOUT set to 120s\", and the "
                 "collection harness sets it (experiments/dataset_collection/collect.py L483, "
                 "`os.environ.setdefault(\"GRC_SHELL_TIMEOUT\", \"120\")`). Every published revision "
                 "from a6379e2 to 6c98128 reads the value from the .env file only "
                 "(settings.get_env_value), and the recovered collect.py writes the campaign .env with "
                 "GRC_PROVIDER and OLLAMA_CLOUD_MODEL only (L472-476). So the collection's .env or its "
                 "GRC-Agent build (/opt/grc_agent in the SFT rows) differed from the published files in "
                 "this one setting.",
     "decision": "pinned default (600 s): the benchmark's .env sets no GRC_SHELL_TIMEOUT. Adding "
                 "GRC_SHELL_TIMEOUT=120 to the run .env would reproduce the SFT text exactly and cap "
                 "shell commands at 120 s; a team decision, recorded here."},
    {"tool": "run_flowgraph", "field": "parameters", "path": "/properties/action/description",
     "sft": "'start' to run the active flowgraph, or 'stop' to terminate it (SIGTERM, escalating to "
            "SIGKILL when shutdown stalls).",
     "pinned": "'start' to run the active flowgraph, or 'stop' to terminate it (SIGTERM).",
     "cause": "unpublished GRC-Agent revision used for the collection",
     "evidence": "The text is run_flowgraph's docstring (agent.py L598 at 6c98128). `git log --all -S "
                 "'escalating to SIGKILL'` finds the phrase only in an audit note, "
                 "docs/investigation/verify-c-new-tools.md (ba6658b, 25 Aug, an ancestor of 6c98128); no "
                 "published commit up to 7d0de8c (18 Sep) has it under src/. At 6c98128 the stop path "
                 "sends SIGTERM only (native_canvas.py L178, L406-440; exec_monitor.py has no SIGKILL), "
                 "so the pinned text describes the pinned behaviour.",
     "decision": "pinned text: it matches what the benchmark's stop actually does."},
]


# --------------------------------------------------------------------------------------------
# Reference (SFT rows) and comparison
# --------------------------------------------------------------------------------------------

def load_sft_reference(path: Path = SFT_FILE) -> dict[str, Any]:
    """The SFT rows' common tool snapshot and per-session system prompts."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tools = rows[0]["tools"]
    canon = json.dumps(tools, sort_keys=True)
    if any(json.dumps(r["tools"], sort_keys=True) != canon for r in rows):
        raise ValueError("SFT rows do not share one tool snapshot")
    return {"path": str(path), "sha256": sha256_file(path), "n_rows": len(rows), "tools": tools,
            "systems": [(r["session_id"], r["task_id"], r["system"]) for r in rows]}


def split_system(text: str) -> tuple[str, str | None]:
    """(fixed part, active-flowgraph path or None for the unsaved-graph note)."""
    m = ACTIVE_PATH.search(text)
    if m:
        return text[:m.start()], m.group("path")
    if text.endswith(UNSAVED_NOTE):
        return text[:-len(UNSAVED_NOTE)], None
    raise ValueError("system prompt lacks the per-turn active-flowgraph note")


def _json_diff(a: Any, b: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            out += _json_diff(a.get(key, "<absent>"), b.get(key, "<absent>"), f"{path}/{key}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += _json_diff(x, y, f"{path}/{i}")
        return out
    return [] if a == b else [(path or "/", a, b)]


def diff_tools(observed: list[dict[str, Any]], reference: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every difference between two tool lists ({name, description, parameters}), as records
    {tool, field, path, sft, pinned}; empty when they are identical (order ignored)."""
    obs = {t["name"]: t for t in observed}
    ref = {t["name"]: t for t in reference}
    out: list[dict[str, Any]] = []
    for name in sorted(set(ref) - set(obs)):
        out.append({"tool": name, "field": "tool", "path": "", "sft": "present", "pinned": "<absent>"})
    for name in sorted(set(obs) - set(ref)):
        out.append({"tool": name, "field": "tool", "path": "", "sft": "<absent>", "pinned": "present"})
    for name in sorted(set(obs) & set(ref)):
        for field in ("description", "parameters"):
            for path, a, b in _json_diff(ref[name].get(field), obs[name].get(field)):
                out.append({"tool": name, "field": field, "path": path, "sft": a, "pinned": b})
    return out


def unexplained(diffs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Diffs not covered by KNOWN_TOOL_DIFFS (matched on tool, field, path, sft and pinned values)."""
    known = {(k["tool"], k["field"], k["path"], json.dumps(k["sft"]), json.dumps(k["pinned"]))
             for k in KNOWN_TOOL_DIFFS}
    return [d for d in diffs
            if (d["tool"], d["field"], d["path"], json.dumps(d["sft"]), json.dumps(d["pinned"])) not in known]


def wire_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    """{name, description, parameters} of the function tools in an OpenAI chat-completions body."""
    out = []
    for t in body.get("tools") or []:
        fn = t.get("function", {})
        out.append({"name": fn.get("name"), "description": fn.get("description", ""),
                    "parameters": fn.get("parameters")})
    return out


# --------------------------------------------------------------------------------------------
# Selftest: runs inside an episode process
# --------------------------------------------------------------------------------------------

def _bg_id(results: dict[str, dict[str, Any]]) -> str:
    content = results.get("sh-bg-start", {}).get("content", "")
    return content.split("ID: ", 1)[1].split()[0] if "ID: " in content else "missing"


def _sh_quote(text: str) -> str:
    import shlex
    return shlex.quote(text)


def _tool_defs(defs: list[Any]) -> list[dict[str, Any]]:
    return [{"name": d.name, "description": d.description or "", "parameters": d.parameters_json_schema,
             "kind": d.kind, "strict": d.strict} for d in defs]


class _Script:
    """Scripted FunctionModel: records what the model is shown, then drives the substituted tools.

    Each turn is a list of (tool, args, call id), or a function of the results so far that returns
    one (to use a value a tool returned, such as a background command id).
    """

    def __init__(self, turns: list[Any]) -> None:
        self.turns = turns
        self.first: dict[str, Any] | None = None
        self.results: dict[str, dict[str, Any]] = {}
        self.n = 0

    def __call__(self, messages: list[Any], info: Any) -> Any:
        from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
        if self.first is None:
            self.first = {"function_tools": _tool_defs(info.function_tools),
                          "output_tools": _tool_defs(info.output_tools),
                          "instructions": info.instructions,
                          "allow_text_output": info.allow_text_output}
        for msg in messages:
            for part in getattr(msg, "parts", ()):
                if isinstance(part, (ToolReturnPart, RetryPromptPart)) and part.tool_call_id:
                    content = part.content if isinstance(part.content, str) else json.dumps(
                        part.content, default=str)
                    self.results[part.tool_call_id] = {
                        "tool": part.tool_name, "kind": part.part_kind, "content": content}
        if self.n < len(self.turns):
            calls = self.turns[self.n]
            calls = calls(self.results) if callable(calls) else calls
            self.n += 1
            return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=cid) for name, args, cid in calls])
        return ModelResponse(parts=[TextPart("selftest finished")])


async def _drive(agent: Any, deps: Any, prompt: str) -> dict[str, Any]:
    """Run to the end, approving every deferred call (what the runner does in the benchmark)."""
    from pydantic_ai import DeferredToolRequests, DeferredToolResults
    result = await agent.run(prompt, deps=deps)
    rounds = 0
    while isinstance(result.output, DeferredToolRequests) and rounds < 5:
        rounds += 1
        approvals = {c.tool_call_id: True for c in result.output.approvals}
        result = await agent.run(message_history=result.all_messages(), deps=deps,
                                 deferred_tool_results=DeferredToolResults(approvals=approvals))
    return {"output": str(result.output), "approval_rounds": rounds}


_COMPLETION = {"id": "chatcmpl-grcbench", "object": "chat.completion", "created": 0, "model": "probe",
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "uds probe finished"}}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


async def _uds_round_trip(sock: Path, deps: Any, project: Path, model_name: str) -> dict[str, Any]:
    """Serve one canned chat completion on a UNIX socket and run the agent against it."""
    import http.server
    import socketserver
    import threading

    from . import agent_build
    received: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server API
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append(self.path)
            payload = json.dumps(_COMPLETION).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def address_string(self) -> str:
            return "uds"

        def log_message(self, *_args: Any) -> None:
            pass

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    if sock.exists():
        sock.unlink()
    server = Server(str(sock), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        model = agent_build.make_openai_compatible_model("http://llama-server/v1", model_name, uds_path=str(sock))
        agent, _ = agent_build.build_benchmark_agent(model, deps, project, "benchmark")
        run = await _drive(agent, deps, "GRC-Bench UNIX-socket probe.")
    finally:
        server.shutdown()
        server.server_close()
        sock.unlink(missing_ok=True)
    return {"requests": received, "output": run["output"]}


async def _selftest(graph: Path) -> dict[str, Any]:
    import httpx2
    from pydantic_ai.models.function import FunctionModel

    from . import agent_build

    lay = isolation.check_process_isolation()
    from grc_agent.adapter import load_flow_graph
    fg = load_flow_graph(str(graph))
    out: dict[str, Any] = {"graph": str(graph), "run_dir": str(lay.run_dir)}

    # A listener in this (episode) process, on the host's loopback: reachable from here, so a
    # sandboxed command that cannot connect to it proves the shell has no path to host services
    # such as llama-server. The external address checks that there is no route out at all.
    import socket
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    probe = ("import socket\n"
             "for name, addr in (('host_loopback', ('127.0.0.1', %d)), ('external', ('1.1.1.1', 53))):\n"
             "    try:\n"
             "        socket.create_connection(addr, 2).close(); print(name, 'CONNECTED')\n"
             "    except OSError as e:\n"
             "        print(name, type(e).__name__, e.strerror)\n") % port
    net_cmd = "python3 -c " + _sh_quote(probe)

    # A. benchmark setting, scripted: offline web tools and sandboxed shell tools via the agent.
    turns = [
        [("duckduckgo_search", {"query": "GNU Radio throttle block"}, "web-search"),
         ("web_fetch", {"url": "https://wiki.gnuradio.org/"}, "web-fetch"),
         ("duckduckgo_search", {"q": "the required argument is missing"}, "web-search-malformed")],
        [("run_command", {"command": "echo inside > inside.txt && cat inside.txt"}, "sh-project-write"),
         ("run_command", {"command": f"echo x > {lay.run_dir}/outside.txt"}, "sh-run-dir-write"),
         ("run_command", {"command": net_cmd}, "sh-network"),
         ("run_command", {"command": "rm -f inside.txt"}, "sh-denylisted"),
         ("list_directory", {"path": "."}, "fs-list")],
        # Background commands spawn through the same launcher. Check and stop go in separate turns:
        # pydantic-ai runs one turn's tool calls concurrently.
        [("start_command", {"command": "echo bg > bg.txt; sleep 30"}, "sh-bg-start")],
        lambda res: [("check_command", {"command_id": _bg_id(res)}, "sh-bg-check")],
        lambda res: [("stop_command", {"command_id": _bg_id(res)}, "sh-bg-stop")],
    ]
    script = _Script(turns)
    agent, info = agent_build.build_benchmark_agent(FunctionModel(script), fg, lay.project, "benchmark")
    # Self-check of the listener before the sandboxed attempt: this process can reach it.
    socket.create_connection(("127.0.0.1", port), 2).close()
    run = await _drive(agent, fg, "GRC-Bench conformance selftest.")
    listener.close()
    out["benchmark"] = {"info": info, "first_request": script.first, "results": script.results,
                        "run": run, "outside_written": (lay.run_dir / "outside.txt").exists(),
                        "inside_written": (lay.project / "inside.txt").exists(),
                        "background_written": (lay.project / "bg.txt").exists()}
    out["freeze_tools"] = await agent_build.freeze_tools(agent)

    # B. GRC-Agent defaults setting: same tools and prompt, GRC-Agent's compaction kept.
    script_b = _Script([])
    agent_b, info_b = agent_build.build_benchmark_agent(FunctionModel(script_b), fg, lay.project,
                                                        "grc_agent_defaults")
    await _drive(agent_b, fg, "GRC-Bench conformance selftest (defaults).")
    out["grc_agent_defaults"] = {"info": info_b, "first_request": script_b.first}

    # C. On the wire. (1) The benchmark path: GRC-Agent's own OpenAIChatModel construction for
    # provider openai_compatible (llama-server). (2) The collection path: pydantic-ai's OllamaModel
    # for the teacher of the SFT rows (provider ollama_cloud, deepseek-v4.1-flash:cloud), so the JSON
    # the model under test receives can be compared with the JSON the teacher received.
    from grc_agent.settings import load_settings
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.providers.ollama import OllamaProvider
    output_names = {t["name"] for t in script.first["output_tools"]}
    cfg = load_settings()

    async def wire_probe(make_model: Any) -> dict[str, Any]:
        bodies: list[dict[str, Any]] = []

        def handler(request: Any) -> Any:
            bodies.append(json.loads(request.content.decode("utf-8")))
            return httpx2.Response(200, json={
                "id": "chatcmpl-grcbench", "object": "chat.completion", "created": 0, "model": "probe",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "wire probe finished"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

        agent_w, info_w = agent_build.build_benchmark_agent(make_model(httpx2.MockTransport(handler)), fg,
                                                            lay.project, "benchmark")
        await _drive(agent_w, fg, "GRC-Bench wire probe.")
        body = bodies[0] if bodies else {}
        all_wire = wire_tools(body)
        systems = [m.get("content") for m in body.get("messages", []) if m.get("role") in ("system", "developer")]
        return {"info": info_w, "n_requests": len(bodies),
                "tools": [t for t in all_wire if t["name"] not in output_names],
                "output_tools": [t for t in all_wire if t["name"] in output_names],
                "tool_keys": sorted({k for t in body.get("tools") or [] for k in t.get("function", {})}),
                "strict_tools": sorted(t["function"]["name"] for t in body.get("tools") or []
                                       if t.get("function", {}).get("strict")),
                "system_messages": systems, "system": "\n\n".join(s for s in systems if s),
                "message_roles": [m.get("role") for m in body.get("messages", [])],
                "request_keys": sorted(body), "tool_choice": body.get("tool_choice"), "body": body}

    out["wire"] = await wire_probe(lambda transport: agent_build.make_openai_compatible_model(
        cfg["openai_compatible_base_url"], cfg["model"], transport=transport))
    out["wire_collection_teacher"] = await wire_probe(lambda transport: OllamaModel(
        "deepseek-v4.1-flash:cloud",
        provider=OllamaProvider(base_url="https://ollama.com/v1", api_key="not-used",
                                http_client=httpx2.AsyncClient(transport=transport))))

    # E. model=None: GRC-Agent builds the model itself from the run's .env (the real-run default).
    _, info_e = agent_build.build_benchmark_agent(None, fg, lay.project, "benchmark")
    out["factory_model"] = {"info": info_e, "env_base_url": cfg["openai_compatible_base_url"],
                            "env_model": cfg["model"]}

    # F. A real HTTP round trip over a UNIX socket (llama-server --host <path>.sock), the route
    # that lets an episode process run with no network namespace at all (network="none").
    out["uds"] = await _uds_round_trip(lay.run_dir / "llama.sock", fg, lay.project, cfg["model"])

    # G. The deps class real episodes use: HeadlessFlowgraphProxy (headless_proxy.py, the GTK-free
    # port of NativeFlowgraphProxy). The model must be offered the same tools and prompt, and the
    # domain and file tools must reach the live graph through it.
    from .headless_proxy import HeadlessFlowgraphProxy
    proxy = HeadlessFlowgraphProxy.open(graph, project_dir=lay.project)
    try:
        script_g = _Script([[("inspect_graph", {}, "px-inspect"),
                             ("read_file", {"path": graph.name}, "px-read"),
                             ("get_run_log", {}, "px-log"),
                             # The knowledge tool on the read-only database copies (the first
                             # query_knowledge call of the SFT rows, and a docs query).
                             ("query_knowledge", {"query": "QPSK modulator constellation modulator",
                                                  "domain": "catalog"}, "px-kq-catalog"),
                             ("query_knowledge", {"query": "throttle block sample rate",
                                                  "domain": "docs"}, "px-kq-docs")]])
        agent_g, info_g = agent_build.build_benchmark_agent(FunctionModel(script_g), proxy, lay.project,
                                                            "benchmark")
        await _drive(agent_g, proxy, "GRC-Bench headless-proxy probe.")
    finally:
        await proxy.aclose()
    out["headless_proxy"] = {"info": info_g, "first_request": script_g.first, "results": script_g.results}

    # D. Where GRC-Agent's state went, from inside the process.
    from gnuradio.grc.core.Config import Config

    from grc_agent import _paths, db, embed_runtime, fs_tools, settings, shell_tools
    store = db.get_db_path()
    with sqlite3.connect(f"file:{store}?mode=ro", uri=True) as conn:
        runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    out["isolation"] = {
        "gi_import_blocked": sys.modules.get("gi", "missing") is None,
        "gtk_modules_loaded": sorted(m for m in sys.modules if m.startswith("gi.") and sys.modules[m]),
        "native_canvas_in_sys_modules": "grc_agent.native_canvas" in sys.modules,
        "env_path": str(settings.env_path()),
        "step_store": str(store), "step_store_runs": runs,
        "vectors_dir": str(_paths.vectors_dir()),
        "embed_runtime_dir": str(embed_runtime.data_dir()),
        "hier_block_lib_dir": str(Config.hier_block_lib_dir),
        "home": str(Path.home()),
        "fs_root": str(fs_tools.active_project_dir()),
        "shell_cwd": str(shell_tools.resolve_shell_cwd()),
        "shell_sandbox_installed": isolation.shell_sandbox_installed(),
    }
    return out


def selftest_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="grcbench.tier_a.conformance selftest")
    p.add_argument("--graph", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    sys.modules["gi"] = None          # any GTK/PyGObject import now raises ImportError
    result = asyncio.run(_selftest(Path(a.graph)))
    write_json(Path(a.out), result)
    return 0


# --------------------------------------------------------------------------------------------
# Host side: run the selftest in an episode process, compare, report
# --------------------------------------------------------------------------------------------

def _snapshot_watched() -> dict[str, dict[str, Any]]:
    return {label: (manifest.build(root) if root.exists() else {"files": {}})
            for label, root in WATCHED_ROOTS.items()}


def evaluate(selftest: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """All conformance checks on one selftest result; `ok` is True only if every check passes."""
    checks: dict[str, Any] = {}
    sft_tools = ref["tools"]
    bench = selftest["benchmark"]

    def defs(first_request: dict[str, Any]) -> list[dict[str, Any]]:
        return [{k: t[k] for k in ("name", "description", "parameters")} for t in first_request["function_tools"]]

    # Tool definitions as pydantic-ai holds them (what the SFT `tools` field records) ...
    px = selftest["headless_proxy"]
    views = {"function_model": defs(bench["first_request"]), "freeze_tools": selftest["freeze_tools"],
             "grc_agent_defaults": defs(selftest["grc_agent_defaults"]["first_request"]),
             "headless_proxy": defs(px["first_request"])}
    # ... and as JSON on the wire, after the provider's schema transformer.
    wire, teacher = selftest["wire"], selftest["wire_collection_teacher"]
    checks["tool_count"] = {**{n: len(t) for n, t in views.items()},
                            "wire_llama_server": len(wire["tools"]), "wire_collection_teacher": len(teacher["tools"])}
    diffs = {name: diff_tools(tools, sft_tools) for name, tools in views.items()}
    checks["tool_diffs"] = diffs["function_model"]
    checks["tool_views_agree"] = all(diff_tools(t, views["function_model"]) == [] for t in views.values())
    checks["tool_diffs_unexplained"] = unexplained(diffs["function_model"])
    checks["wire_vs_collection_teacher"] = diff_tools(wire["tools"], teacher["tools"])
    checks["wire_vs_sft"] = diff_tools(wire["tools"], sft_tools)
    # Wire-vs-SFT differences beyond KNOWN_TOOL_DIFFS come from pydantic-ai's JSON-schema
    # transformer for OpenAI-style chat APIs; the teacher's wire shows the same rewrite.
    checks["wire_schema_rewrites"] = unexplained(checks["wire_vs_sft"])
    checks["wire_strict"] = {"llama_server": wire["strict_tools"], "collection_teacher": teacher["strict_tools"]}
    checks["wire_shape"] = {k: {"roles": w["message_roles"], "tool_keys": w["tool_keys"],
                                "tool_choice": w["tool_choice"], "output_tools": [t["name"] for t in w["output_tools"]]}
                            for k, w in (("llama_server", wire), ("collection_teacher", teacher))}

    sft_fixed = {split_system(s)[0] for _, _, s in ref["systems"]}
    sft_paths = [split_system(s)[1] for _, _, s in ref["systems"]]
    graph = selftest["graph"]
    systems = {"function_model": bench["first_request"]["instructions"],
               "grc_agent_defaults": selftest["grc_agent_defaults"]["first_request"]["instructions"],
               "headless_proxy": px["first_request"]["instructions"],
               "wire_llama_server": wire["system"], "wire_collection_teacher": teacher["system"]}
    prompt_checks = {}
    for name, text in systems.items():
        fixed, path = split_system(text or "")
        prompt_checks[name] = {"fixed_part_equal": len(sft_fixed) == 1 and fixed == next(iter(sft_fixed)),
                               "active_path": path, "path_is_episode_graph": path == graph}
    checks["system_prompt"] = {"sft_fixed_parts": len(sft_fixed), "sft_paths": sft_paths,
                               "fixed_len": len(next(iter(sft_fixed))), "views": prompt_checks}

    res = bench["results"]
    checks["web_offline"] = {
        "search": res.get("web-search", {}).get("content"),
        "fetch": res.get("web-fetch", {}).get("content"),
        "malformed_is_retry": res.get("web-search-malformed", {}).get("kind") == "retry-prompt",
    }
    checks["shell"] = {cid: res.get(cid, {}) for cid in
                       ("sh-project-write", "sh-run-dir-write", "sh-network", "sh-denylisted")}
    checks["shell"]["inside_written"] = bench["inside_written"]
    checks["shell"]["outside_written"] = bench["outside_written"]
    iso = selftest["isolation"]
    run_dir = selftest["run_dir"]
    checks["isolation"] = iso
    checks["capabilities"] = {"benchmark": bench["info"]["capabilities"],
                              "grc_agent_defaults": selftest["grc_agent_defaults"]["info"]["capabilities"],
                              "headless_proxy": px["info"]["capabilities"]}
    pxr = px["results"]
    checks["headless_proxy"] = {
        "inspect_graph": pxr.get("px-inspect", {}).get("kind"),
        "inspect_graph_sees_blocks": "analog_sig_source_x" in pxr.get("px-inspect", {}).get("content", ""),
        "read_file_header": pxr.get("px-read", {}).get("content", "").split("\n", 1)[0],
        "get_run_log": pxr.get("px-log", {}).get("content"),
        "get_run_log_kind": pxr.get("px-log", {}).get("kind"),
        "query_knowledge": {d: _knowledge_summary(pxr.get(f"px-kq-{d}", {})) for d in ("catalog", "docs")},
    }
    ok = {
        "tools_match_sft_or_known_diffs": not checks["tool_diffs_unexplained"],
        "tool_views_agree": checks["tool_views_agree"],
        "22_function_tools": all(n == 22 for n in checks["tool_count"].values()),
        "wire_tools_equal_collection_teacher_wire": not checks["wire_vs_collection_teacher"],
        "system_prompt_fixed_part_equal": all(v["fixed_part_equal"] for v in prompt_checks.values()),
        "system_prompt_names_episode_graph": all(v["path_is_episode_graph"] for v in prompt_checks.values()),
        "web_tools_offline": checks["web_offline"]["search"] == checks["web_offline"]["fetch"]
        == _offline_message(),
        "web_tool_argument_validation_kept": checks["web_offline"]["malformed_is_retry"],
        "shell_project_writable": bench["inside_written"],
        "shell_run_dir_read_only": not bench["outside_written"]
        and "Read-only file system" in res.get("sh-run-dir-write", {}).get("content", ""),
        "shell_no_network": _no_network(res.get("sh-network", {}).get("content", "")),
        "shell_denylist_kept": res.get("sh-denylisted", {}).get("kind") == "retry-prompt"
        and "denied" in res.get("sh-denylisted", {}).get("content", ""),
        "shell_background_commands_sandboxed": bench["background_written"]
        and "ID: " in res.get("sh-bg-start", {}).get("content", "")
        and "[status: running]" in res.get("sh-bg-check", {}).get("content", "")
        and "[stopped]" in res.get("sh-bg-stop", {}).get("content", ""),
        "gtk_never_imported": iso["gi_import_blocked"] and not iso["gtk_modules_loaded"]
        and not iso["native_canvas_in_sys_modules"],
        "state_inside_run_dir": all(str(iso[k]).startswith(run_dir) for k in
                                    ("env_path", "step_store", "embed_runtime_dir", "hier_block_lib_dir",
                                     "home", "fs_root", "shell_cwd")),
        "step_persistence_recorded": iso["step_store_runs"] > 0,
        "vectors_read_only_copy": iso["vectors_dir"] == str(isolation.VECTORS_CACHE),
        "compaction_off_in_benchmark": "CompactionOff" in checks["capabilities"]["benchmark"]
        and "TranscriptPreservingTieredCompaction" not in checks["capabilities"]["benchmark"],
        "compaction_on_in_grc_agent_defaults":
            "TranscriptPreservingTieredCompaction" in checks["capabilities"]["grc_agent_defaults"],
        "factory_builds_model_from_run_env": _factory_model_ok(selftest["factory_model"]),
        "uds_round_trip": selftest["uds"]["requests"] == ["/v1/chat/completions"]
        and selftest["uds"]["output"] == "uds probe finished",
        "headless_proxy_same_capabilities": checks["capabilities"]["headless_proxy"]
        == checks["capabilities"]["benchmark"],
        "headless_proxy_tools_reach_live_graph": _proxy_tools_ok(checks["headless_proxy"]),
        "knowledge_tool_works_on_read_only_copy": all(
            q["kind"] == "tool-return" and q["ok"] and q["n_results"] > 0 and q["first"]
            for q in checks["headless_proxy"]["query_knowledge"].values()),
    }
    checks["factory_model"] = selftest["factory_model"]["info"]["model"]
    checks["ok"] = ok
    checks["passed"] = all(ok.values())
    return checks


def _factory_model_ok(fm: dict[str, Any]) -> bool:
    """With model=None GRC-Agent built OpenAIChatModel for the run .env's endpoint and model."""
    model = fm["info"]["model"]
    return (model["class"] == "OpenAIChatModel" and model["model_name"] == fm["env_model"]
            and model["base_url"].rstrip("/") == fm["env_base_url"].rstrip("/"))


def _knowledge_summary(result: dict[str, Any]) -> dict[str, Any]:
    """kind, ok flag, search mode, result count and first hit (catalog: block id; docs: the page
    named on the hit's leading "path: ..." line) of one query_knowledge tool result."""
    try:
        data = json.loads(result.get("content") or "")
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    hits = data.get("results") or []
    first = hits[0] if hits and isinstance(hits[0], dict) else {}
    page = str(first.get("text", "")).split("\n", 1)[0]
    return {"kind": result.get("kind"), "ok": bool(data.get("ok")), "search_mode": data.get("search_mode"),
            "n_results": len(hits),
            "first": first.get("block_id") or (page[len("path: "):] if page.startswith("path: ") else None)}


def _proxy_tools_ok(px: dict[str, Any]) -> bool:
    """Through HeadlessFlowgraphProxy deps: inspect_graph reads the graph, read_file of the active
    graph takes GRC-Agent's live in-memory branch (fs_tools: active path == the file), and
    get_run_log finds the run-log protocol (with a bare FlowGraph it raises "not wired up")."""
    try:
        log = json.loads(px["get_run_log"] or "")
    except ValueError:
        return False
    return (px["inspect_graph"] == "tool-return" and px["inspect_graph_sees_blocks"]
            and "live in-memory flowgraph" in px["read_file_header"]
            and px["get_run_log_kind"] == "tool-return" and log.get("log_text") == ""
            and "No flowgraph has been run yet" in log.get("message", ""))


def _no_network(output: str) -> bool:
    """The sandboxed network probe printed a failure for both the host-loopback listener (which
    the episode process itself could reach) and the external address, and no success."""
    lines = {ln.split(" ", 1)[0]: ln for ln in output.splitlines() if ln.startswith(("host_loopback", "external"))}
    return set(lines) == {"host_loopback", "external"} and not any("CONNECTED" in ln for ln in lines.values())


def _offline_message() -> str:
    from .agent_build import OFFLINE_MESSAGE
    return OFFLINE_MESSAGE


def run_conformance(*, report: bool = True, network: str = "none") -> dict[str, Any]:
    """Run the selftest in a fresh episode process, compare with the SFT rows, write the report.

    network="none" (default) runs the episode process without any network namespace access, the
    configuration recommended for real runs with llama-server on a UNIX socket; "host" shares the
    host network, as a llama-server on a TCP port requires.
    """
    if network not in ("none", "host"):
        raise ValueError("network must be 'none' or 'host'")
    run_dir = isolation.new_run_dir("conformance")
    env = isolation.make_run_env(run_dir)
    graph = isolation.copy_start_graph(run_dir, PROBE_GRAPH)
    before = _snapshot_watched()
    out_file = run_dir / "selftest.json"
    res = isolation.run_episode_process(run_dir, "grcbench.tier_a.conformance",
                                        ["selftest", "--graph", str(graph), "--out", str(out_file)],
                                        env, timeout=900, network=network)  # type: ignore[arg-type]
    after = _snapshot_watched()
    if res.returncode != 0 or not out_file.exists():
        raise RuntimeError(f"selftest failed (rc={res.returncode}, timed_out={res.timed_out}):\n"
                           f"{res.stderr[-6000:]}")
    selftest = read_json(out_file)
    ref = load_sft_reference()
    checks = evaluate(selftest, ref)
    watched = {label: manifest.compare(before[label], after[label]) for label in WATCHED_ROOTS}
    checks["watched_roots_unchanged"] = {label: not any(d.values()) for label, d in watched.items()}
    checks["watched_roots_files"] = {label: len(before[label]["files"]) for label in WATCHED_ROOTS}
    checks["ok"]["grc_agent_tree_unchanged"] = all(checks["watched_roots_unchanged"].values())
    checks["passed"] = all(checks["ok"].values())
    checks["episode_network"] = network
    write_json(run_dir / "conformance.json", {"checks": checks, "watched_diff": watched})
    if report:
        write_text(REPORT, render_report(selftest, ref, checks))
    return {"run_dir": str(run_dir), "selftest": selftest, "checks": checks, "watched_diff": watched,
            "stderr_tail": res.stderr[-4000:]}


def render_report(selftest: dict[str, Any], ref: dict[str, Any], checks: dict[str, Any]) -> str:
    """reports/tier_a_conformance.md (deterministic: no timestamps; run paths shown as <run>)."""
    def esc(v: Any) -> str:
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return s.replace("|", "\\|").replace("\n", "\\n")

    def yes(flag: bool, good: str = "yes", bad: str = "NO") -> str:
        return good if flag else bad

    run_dir = selftest["run_dir"]
    bench = selftest["benchmark"]
    ok = checks["ok"]
    lines = [
        "# Tier A: conformance of the benchmark agent with GRC-Agent",
        "",
        "Generated by `./gb tier-a conformance` (`grcbench.tier_a.conformance`); re-run it after any change",
        "to the agent build. Eval plan §4.4: \"One pinned GRC-Agent version and its real tool definitions\".",
        "",
        f"- Pinned GRC-Agent commit `{GRC_AGENT_COMMIT}`. The benchmark agent is GRC-Agent's own executor, built",
        "  by its `agent_factory.build_agents_from_cfg` (`grcbench.tier_a.agent_build.build_benchmark_agent`).",
        f"- Reference: Qoherent's SFT release `grc-agent-sft-v0.1.0/sft_train.jsonl`: {ref['n_rows']} rows, sha256",
        f"  `{ref['sha256'][:16]}…`. All rows share one `tools` snapshot of {len(ref['tools'])} tools.",
        f"- Overall: **{'PASS' if checks['passed'] else 'FAIL'}** ({sum(ok.values())} of {len(ok)} checks pass).",
        "",
        "## Checks",
        "",
        "| check | result |",
        "|---|---|",
    ]
    lines += [f"| {name} | {'pass' if value else 'FAIL'} |" for name, value in ok.items()]
    lines += [
        "",
        "## (i) Tool definitions",
        "",
        "Captured inside the episode process, in six views:",
        "",
        "| view | what it is | function tools |",
        "|---|---|---|",
        "| function_model | tool definitions in the first model request (pydantic-ai FunctionModel AgentInfo), "
        f"benchmark setting, a bare GRC FlowGraph as deps | {checks['tool_count']['function_model']} |",
        f"| grc_agent_defaults | the same, GRC-Agent-defaults setting | {checks['tool_count']['grc_agent_defaults']} |",
        "| headless_proxy | the same, benchmark setting, with the deps real episodes use "
        f"(`HeadlessFlowgraphProxy`) | {checks['tool_count']['headless_proxy']} |",
        "| freeze_tools | Qoherent's snapshot method (`collect.py` `freeze_tools`), which wrote the SFT `tools` "
        f"field | {checks['tool_count']['freeze_tools']} |",
        "| wire_llama_server | `tools` of the JSON body an OpenAI-compatible server (llama-server) receives from "
        f"GRC-Agent's OpenAIChatModel | {checks['tool_count']['wire_llama_server']} |",
        "| wire_collection_teacher | the same agent through pydantic-ai's OllamaModel for the SFT teacher "
        f"(`ollama_cloud`, `deepseek-v4.1-flash:cloud`) | {checks['tool_count']['wire_collection_teacher']} |",
        "",
        f"- The four pydantic-ai views are identical: {yes(checks['tool_views_agree'])}.",
        f"- Against the SFT `tools` they differ in exactly {len(checks['tool_diffs'])} place(s), all explained below "
        f"(unexplained: {len(checks['tool_diffs_unexplained'])}).",
        "- The JSON llama-server receives equals the JSON the collection teacher received, tool by tool "
        f"(name, description, parameters): {yes(not checks['wire_vs_collection_teacher'])}.",
        "",
    ]
    if checks["tool_diffs"]:
        lines += ["### Differences between the SFT rows and the pinned commit", "",
                  "| tool | field | path | SFT rows | pinned 6c98128 |", "|---|---|---|---|---|"]
        for d in checks["tool_diffs"]:
            lines.append(f"| {d['tool']} | {d['field']} | `{d['path']}` | {esc(d['sft'])} | {esc(d['pinned'])} |")
        lines += ["", "Cause, evidence and decision (the pinned commit is authoritative for the benchmark):", ""]
        for k in KNOWN_TOOL_DIFFS:
            lines += [f"- **{k['tool']}** `{k['path']}`: {k['cause']}.",
                      f"  - Evidence: {k['evidence']}",
                      f"  - Decision: {k['decision']}"]
        lines.append("")
    else:
        lines += ["The tool definitions are identical to the SFT `tools`.", ""]
    rewrites = checks["wire_schema_rewrites"]
    shape = checks["wire_shape"]
    lines += [
        "### On the wire",
        "",
        "- Schema rewrite by pydantic-ai: the SFT stores pydantic-ai's tool definitions; before sending, "
        f"pydantic-ai's JSON-schema transformer for OpenAI-style chat APIs rewrites {len(rewrites)} place(s) "
        f"({', '.join(sorted({d['tool'] for d in rewrites})) or 'none'}): nested `$defs` get "
        "`additionalProperties: false` and lose their `title`, and free-form objects get `properties: {}`. "
        "OllamaModel (the teacher) applies the same transformer, so the teacher saw the same JSON.",
        f"- `strict`: GRC-Agent's OpenAIChatModel path marks {len(checks['wire_strict']['llama_server'])} tools "
        f"`\"strict\": true` ({', '.join(checks['wire_strict']['llama_server']) or 'none'}); pydantic-ai sets it "
        "when a schema is strict-compatible and the tool leaves `strict` unset (pydantic_ai/models/__init__.py "
        "L1816). The Ollama path never sends `strict` (its profile sets openai_supports_strict_tool_definition=False), so "
        f"the teacher's requests had none ({len(checks['wire_strict']['collection_teacher'])}). This is GRC-Agent's "
        "real behaviour against an OpenAI-compatible server and is kept; whether llama-server enforces it is "
        "recorded by the smoke test (it bears on the B0/B2 distinction).",
        f"- Message layout: roles {shape['llama_server']['roles']}: the fixed instructions and the "
        "active-flowgraph note travel as two system messages (the teacher's request: "
        f"{shape['collection_teacher']['roles']}); `tool_choice` {esc(shape['llama_server']['tool_choice'])}; "
        f"per-tool keys {', '.join(shape['llama_server']['tool_keys'])}.",
        "- Output tool: besides the 22 function tools the model is offered "
        f"{', '.join('`' + n + '`' for n in shape['llama_server']['output_tools'])}, GRC-Agent's structured final "
        "answer (output type `[GrcAgentResponse, str, DeferredToolRequests]`). The SFT `tools` field leaves it out "
        "by design (release README: \"The `final_result` terminator tool isn't in the frozen tool schema\").",
        "",
    ]
    sp = checks["system_prompt"]
    lines += [
        "## (ii) System prompt",
        "",
        "Each SFT `system` is `build_system_prompt(\"pai-desktop-chat\")` (the line \"Session ID: pai-desktop-chat\"",
        "and the executor instructions) followed by GRC-Agent's per-turn note `\\n\\nActive flowgraph file path: <path>`.",
        f"Only the path varies: {len(set(sp['sft_paths']))} distinct paths in {len(sp['sft_paths'])} rows, all "
        "Qoherent collection fixtures. In the benchmark the path is the episode's copy of the task's start graph.",
        "",
        f"- Distinct fixed parts across the SFT rows: {sp['sft_fixed_parts']} ({sp['fixed_len']} characters).",
    ]
    for view, v in sp["views"].items():
        rel = (v["active_path"] or "").replace(run_dir, "<run>")
        lines.append(f"- {view}: fixed part identical to the SFT rows: {yes(v['fixed_part_equal'])}; "
                     f"note names `{rel}`, the episode graph: {yes(v['path_is_episode_graph'])}.")
    iso = checks["isolation"]
    rel = {k: str(v).replace(run_dir, "<run>") for k, v in iso.items()}
    web = checks["web_offline"]
    net = checks.get("episode_network", "none")
    fm = checks["factory_model"]
    lines += [
        "",
        "## (iii) Headless build and isolation",
        "",
        "The selftest ran as one episode process (`isolation.run_episode_process`: bubblewrap, whole filesystem",
        "read-only except the run directory) with `sys.modules['gi'] = None`, so any GTK/PyGObject import would",
        "have failed. Episode network: "
        + ("`none` (own empty network namespace; the model is reached over a UNIX socket)." if net == "none"
           else "`host` (shared with the host, as a llama-server on a TCP port requires)."),
        "",
        f"- With `model=None` GRC-Agent builds the model itself from the run `.env`: `{fm['class']}`, model "
        f"`{fm['model_name']}`, endpoint `{fm['base_url']}` "
        f"({yes(ok['factory_builds_model_from_run_env'], 'as configured', 'NOT as configured')}).",
        "- A full request/response over a UNIX socket (llama-server `--host <path>.sock`) through GRC-Agent's "
        f"retrying HTTP client: {yes(ok['uds_round_trip'], 'works', 'FAILED')}.",
        f"- GTK never imported: {yes(ok['gtk_never_imported'])}. `grc_agent.native_canvas` (GTK) is replaced by a",
        "  GTK-free stand-in for the duration of the factory call only.",
        f"- GRC-Agent settings `{rel['env_path']}`; step store `{rel['step_store']}` ({iso['step_store_runs']} runs",
        f"  recorded); embedding-runtime dir `{rel['embed_runtime_dir']}`; GRC hier-block library",
        f"  `{rel['hier_block_lib_dir']}`; HOME `{rel['home']}`.",
        "- Knowledge databases: read-only copies in `~/grc-bench-cache/grc-agent-vectors` (checked by SHA-256",
        "  against `~/grc-agent/src/grc_agent/vectors/*_lexical.db`).",
        f"- File-tool root and shell working directory: `{rel['fs_root']}`, `{rel['shell_cwd']}`.",
    ]
    for label, unchanged in checks["watched_roots_unchanged"].items():
        lines.append(f"- File-tree manifest of `{label}` ({checks['watched_roots_files'][label]} files, "
                     f"git-ignored files included) before vs after: {yes(unchanged, 'unchanged', 'CHANGED')}.")
    lines += [
        "",
        "Substituted behaviour, driven through the agent by a scripted FunctionModel (deferred calls approved as",
        "the runner approves them):",
        "",
        (f"- `duckduckgo_search` and `web_fetch` both return \"{web['search']}\"."
         if web["search"] == web["fetch"] else
         f"- `duckduckgo_search` returns \"{web['search']}\"; `web_fetch` returns \"{web['fetch']}\"."),
        "  A call without the required argument is still rejected by the original argument validator: "
        f"{yes(web['malformed_is_retry'], 'retry prompt', 'NOT rejected')}.",
        f"- `run_command`, writing inside the project folder: {yes(ok['shell_project_writable'], 'works', 'FAILED')};",
        "  writing in the run directory outside it: "
        f"{yes(ok['shell_run_dir_read_only'], 'blocked (read-only file system)', 'NOT BLOCKED')};",
        "  connecting to a listener on the host loopback (reachable from the episode process) and to an external",
        f"  address: {yes(ok['shell_no_network'], 'both blocked (connection refused, network unreachable)', 'NOT BLOCKED')};",
        f"  `rm` (GRC-Agent's own denylist): {yes(ok['shell_denylist_kept'], 'denied', 'NOT DENIED')}.",
        "- `start_command` / `check_command` / `stop_command` (a background command writing in the project "
        f"folder, checked while running, then stopped): "
        f"{yes(ok['shell_background_commands_sandboxed'], 'work through the same sandbox', 'FAILED')}.",
        "- With `HeadlessFlowgraphProxy` as deps (the class real episodes use), the build has the same "
        f"capabilities ({yes(ok['headless_proxy_same_capabilities'])}), and the tools reach the live graph: "
        f"`inspect_graph` returns its blocks; `read_file` on the active graph reports "
        f"\"{esc(checks['headless_proxy']['read_file_header'])}\"; `get_run_log` before any run returns "
        "GRC-Agent's \"No flowgraph has been run yet\" result (with a bare FlowGraph it fails as \"not wired "
        f"up\"): {yes(ok['headless_proxy_tools_reach_live_graph'], 'all as expected', 'NOT as expected')}.",
        "- `query_knowledge` on the read-only database copies, inside the read-only episode sandbox: "
        + "; ".join(f"{d} domain {q['n_results']} results, search mode `{q['search_mode']}` (first `{q['first']}`)"
                    for d, q in checks["headless_proxy"]["query_knowledge"].items())
        + f": {yes(ok['knowledge_tool_works_on_read_only_copy'], 'works', 'FAILED')}.",
        "",
        "## Substitutions in the benchmark build",
        "",
    ]
    lines += [f"- {s}" for s in bench["info"]["substitutions"]]
    lines += [
        "",
        "Capabilities of the built executor, benchmark setting: "
        + ", ".join(f"`{c}`" for c in checks["capabilities"]["benchmark"]) + ".",
        "",
        "GRC-Agent-defaults setting: "
        + ", ".join(f"`{c}`" for c in checks["capabilities"]["grc_agent_defaults"]) + ".",
        "",
        "`ToolSearch` and `PendingMessageDrainCapability` are added by pydantic-ai itself to every agent.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "selftest":
        return selftest_main(argv[1:])
    print("usage: python -m grcbench.tier_a.conformance selftest --graph G --out F")
    return 2


if __name__ == "__main__":
    sys.exit(main())

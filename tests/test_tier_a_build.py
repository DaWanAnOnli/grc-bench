"""Tier-A agent build, episode isolation and conformance with GRC-Agent (eval plan §4.4).

Fast tests check the isolation contract and the comparison logic in this process. The slow test
builds GRC-Agent's real executor in an isolated episode process (bubblewrap, `import gi` blocked)
and checks it against Qoherent's SFT rows: tools, system prompt, offline web tools, sandboxed
shell tools, and an unchanged ~/grc-agent file tree.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import sys
import uuid
from pathlib import Path

import pytest

from grcbench.common.io import sha256_file
from grcbench.common.paths import GRC_AGENT, HOME, PROJECT, WORK
from grcbench.tier_a import conformance, isolation


@pytest.fixture
def run_dir():
    path = isolation.RUNS_ROOT / "pytest" / uuid.uuid4().hex[:12]
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------- isolation contract

def test_dotenv_selects_llama_server_and_holds_no_secret():
    text = isolation.dotenv_text("http://127.0.0.1:8080/v1", "qwen3-4b")
    keys = dict(line.split("=", 1) for line in text.splitlines() if line and not line.startswith("#"))
    assert keys == {"GRC_PROVIDER": "openai_compatible", "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:8080/v1",
                    "OPENAI_COMPATIBLE_MODEL": "qwen3-4b", "GRC_EMBED_BACKEND": "lexical"}
    assert not any("KEY" in k or "TOKEN" in k for k in keys)
    # GRC-Agent prints GRC_SHELL_TIMEOUT into run_command's schema: the benchmark leaves its default.
    assert "GRC_SHELL_TIMEOUT" not in text and "GRC_AGENT_APPROVE_CHANGES" not in text
    with pytest.raises(ValueError):
        isolation.dotenv_text("http://x/v1\nOPENAI_API_KEY=leak", "m")


def test_run_dirs_only_under_work_and_never_in_read_only_inputs():
    for bad in (GRC_AGENT / "runs" / "x", HOME / ".config" / "grc_agent" / "x", PROJECT / "data" / "x",
                Path("/tmp") / "grcbench-x", WORK):
        with pytest.raises(isolation.IsolationError):
            isolation.check_run_dir(bad)
    assert isolation.check_run_dir(WORK / "tier_a" / "x") == (WORK / "tier_a" / "x").resolve()


def test_make_run_env_points_every_grc_agent_path_into_the_run(run_dir):
    env = isolation.make_run_env(run_dir, base_url="http://127.0.0.1:9999/v1", model_name="m")
    lay = isolation.RunLayout(run_dir.resolve())
    assert env["GRC_AGENT_ENV"] == str(lay.env_file) and lay.env_file.is_file()
    assert env["GRC_AGENT_RUNTIME_DIR"] == str(lay.runtime_dir)
    assert env["GRC_HIER_PATH"] == str(lay.hier_dir) and lay.hier_dir.is_dir()
    assert env["HOME"] == str(lay.home)
    for key in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        assert env[key].startswith(str(lay.home))
    assert env["PYTHONDONTWRITEBYTECODE"] == "1" and env["QT_QPA_PLATFORM"] == "offscreen"
    assert env["GRC_AGENT_VECTORS_DIR"] == str(isolation.VECTORS_CACHE)
    assert not [k for k in env if k.endswith("_API_KEY")]
    assert lay.project.is_dir()
    assert (lay.home / ".cache" / "gnuradio" / "grc" / "cache_v2.json").is_file()   # warm GRC cache
    # The knowledge databases are byte-identical, read-only copies; the originals are untouched.
    for name in isolation.LEXICAL_DBS:
        copy = isolation.VECTORS_CACHE / name
        assert sha256_file(copy) == sha256_file(isolation.VECTORS_SRC / name)
        assert not os.stat(copy).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    # The env passes the in-process check exactly when it is the process environment.
    saved = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(env)
        assert isolation.check_process_isolation().run_dir == lay.run_dir
        os.environ["GRC_AGENT_ENV"] = str(GRC_AGENT / ".env")
        with pytest.raises(isolation.IsolationError):
            isolation.check_process_isolation()
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_build_refuses_a_process_that_was_not_prepared(monkeypatch):
    from grcbench.tier_a import agent_build
    monkeypatch.delenv("GRC_BENCH_RUN_DIR", raising=False)
    with pytest.raises(isolation.IsolationError):
        agent_build.build_benchmark_agent(None, object(), WORK / "tier_a" / "nowhere")
    with pytest.raises(ValueError):
        agent_build.build_benchmark_agent(None, object(), WORK, setting="no-such-setting")


def test_copy_start_graph_never_touches_the_original(run_dir):
    src = conformance.PROBE_GRAPH
    before = sha256_file(src)
    dest = isolation.copy_start_graph(run_dir, src)
    dest.write_text("edited\n")
    assert dest.parent == isolation.RunLayout(run_dir.resolve()).project
    assert sha256_file(src) == before


def test_shell_sandbox_argv_writable_project_only_and_no_network(run_dir):
    project = run_dir / "project"
    project.mkdir()
    with pytest.raises(PermissionError):           # not registered: fail closed
        isolation.shell_sandbox_argv("ls", project)
    isolation._SHELL_PROJECTS.add(project.resolve())
    try:
        argv = isolation.shell_sandbox_argv("ls -la", project)
    finally:
        isolation._SHELL_PROJECTS.discard(project.resolve())
    joined = " ".join(argv)
    assert argv[-3:] == ["/bin/sh", "-c", "ls -la"]
    assert "--unshare-net" in argv and "--ro-bind / /" in joined
    assert f"--bind {project.resolve()} {project.resolve()}" in joined
    assert joined.count("--bind ") == 1                 # the project folder is the only writable bind
    with pytest.raises(PermissionError):
        isolation.shell_sandbox_argv(["ls"], project)   # only command strings, as GrcShell spawns them


# ---------------------------------------------------------------- substitutions (in process)

def test_offline_web_tools_keep_name_description_schema_and_validator():
    from pydantic_core import ValidationError

    from grc_agent.agent import web_fetch_cap, web_search_cap
    from grcbench.tier_a.agent_build import OFFLINE_MESSAGE, offline_web_capabilities
    search, fetch = offline_web_capabilities()
    for new, old, args in ((search, web_search_cap, {"query": "gnuradio"}),
                           (fetch, web_fetch_cap, {"url": "https://wiki.gnuradio.org"})):
        a, b = new.local.tool_def, old.local.tool_def
        assert (a.name, a.description, a.parameters_json_schema) == (b.name, b.description, b.parameters_json_schema)
        assert new.native is False                       # always offered, for every model class
        assert asyncio.run(new.local.function(**args)) == OFFLINE_MESSAGE
        with pytest.raises(ValidationError):
            new.local.function_schema.validator.validate_python({"wrong": 1})


def test_gtk_free_native_canvas_is_scoped_and_minimal():
    from grcbench.tier_a.agent_build import NATIVE_CANVAS, gtk_free_native_canvas

    class Proxy:
        pass

    had = NATIVE_CANVAS in sys.modules
    with gtk_free_native_canvas(Proxy) as stub:
        from grc_agent.native_canvas import NativeFlowgraphProxy
        assert NativeFlowgraphProxy is Proxy and sys.modules[NATIVE_CANVAS] is stub
        with pytest.raises(AttributeError):
            _ = stub.NativeCanvasManager            # anything GTK-backed is absent
    assert (NATIVE_CANVAS in sys.modules) == had


def test_fs_providers_mirror_the_desktop_canvas():
    """canvas.path / canvas.current_flow_graph: None when no graph is open, never an exception."""
    from grcbench.tier_a.agent_build import _graph_path, _live_flow_graph

    class NoPage:                                   # NativeFlowgraphProxy with no page open
        def _get_target(self):
            raise RuntimeError("No flowgraph is open.")

        def __getattr__(self, name):
            return getattr(self._get_target(), name)

    class Graph:                                    # a bare GRC FlowGraph, as in GRC-Agent's tests
        grc_file_path = "/run/project/a.grc"

    assert _live_flow_graph(NoPage()) is None and _graph_path(NoPage()) is None
    g = Graph()
    assert _live_flow_graph(g) is g and _graph_path(g) == "/run/project/a.grc"
    g.grc_file_path = ""                            # never saved
    assert _graph_path(g) is None


# ---------------------------------------------------------------- comparison logic

def test_sft_system_prompts_are_grc_agent_prompt_plus_active_path():
    from grc_agent.prompts import build_system_prompt
    ref = conformance.load_sft_reference()
    fixed = {conformance.split_system(s)[0] for _, _, s in ref["systems"]}
    paths = [conformance.split_system(s)[1] for _, _, s in ref["systems"]]
    # pydantic-ai strips each instruction (pydantic_ai/_instructions.py) and joins them with "\n\n".
    assert fixed == {build_system_prompt("pai-desktop-chat").strip()}
    assert all(p and p.endswith(".grc") for p in paths)
    with pytest.raises(ValueError):
        conformance.split_system("no active-flowgraph note")


def test_known_tool_diffs_are_real_sft_values():
    ref = {t["name"]: t for t in conformance.load_sft_reference()["tools"]}
    assert len(ref) == 22
    for k in conformance.KNOWN_TOOL_DIFFS:
        node = ref[k["tool"]][k["field"]]
        for part in k["path"].strip("/").split("/"):
            node = node[part]
        assert node == k["sft"] and k["pinned"] != k["sft"] and k["evidence"] and k["decision"]


def test_diff_tools_reports_every_field_and_ignores_order():
    a = [{"name": "t1", "description": "d", "parameters": {"properties": {"x": {"type": "string"}}}},
         {"name": "t2", "description": "e", "parameters": {}}]
    assert conformance.diff_tools(list(reversed(a)), a) == []
    b = json.loads(json.dumps(a))
    b[0]["parameters"]["properties"]["x"]["type"] = "integer"
    b[1]["description"] = "changed"
    b.append({"name": "t3", "description": "", "parameters": {}})
    d = conformance.diff_tools(b, a)
    assert {(x["tool"], x["field"], x["path"]) for x in d} == {
        ("t1", "parameters", "/properties/x/type"), ("t2", "description", "/"), ("t3", "tool", "")}
    assert conformance.unexplained(d) == d


def test_wire_tools_reads_openai_chat_bodies():
    body = {"tools": [{"type": "function", "function": {"name": "n", "description": "d",
                                                         "parameters": {"type": "object"}, "strict": True}}]}
    assert conformance.wire_tools(body) == [{"name": "n", "description": "d", "parameters": {"type": "object"}}]


# ---------------------------------------------------------------- CLI

def test_cli_subcommands(run_dir, capsys):
    from grcbench.tier_a import cli
    assert cli.main([]) == 0 and "conformance" in capsys.readouterr().out
    assert cli.main(["no-such-subcommand"]) == 2
    assert cli.main(["run-env", "/tmp/grcbench-not-under-work"]) == 2
    assert "refused" in capsys.readouterr().out
    assert cli.main(["run-env", str(run_dir), "--base-url", "http://127.0.0.1:9/v1", "--model", "m"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["GRC_AGENT_ENV"] == str(run_dir.resolve() / ".env")
    assert "OPENAI_COMPATIBLE_BASE_URL=http://127.0.0.1:9/v1" in (run_dir / ".env").read_text()


# ---------------------------------------------------------------- the real build, isolated

@pytest.mark.slow
@pytest.mark.gnuradio
def test_real_executor_conforms_in_an_isolated_episode_process():
    result = conformance.run_conformance(report=False)
    checks = result["checks"]
    failed = [name for name, ok in checks["ok"].items() if not ok]
    assert not failed, (failed, result["stderr_tail"])
    # (i) 22 tools; the only differences from the SFT rows are the two traced ones.
    assert set(checks["tool_count"].values()) == {22}
    assert [(d["tool"], d["path"]) for d in checks["tool_diffs"]] == [
        (k["tool"], k["path"]) for k in conformance.KNOWN_TOOL_DIFFS]
    assert checks["wire_vs_collection_teacher"] == []
    # (ii) GRC-Agent's own prompt, naming the episode's copy of the start graph.
    assert all(v["fixed_part_equal"] and v["path_is_episode_graph"]
               for v in checks["system_prompt"]["views"].values())
    # (iii) no GTK, nothing written to ~/grc-agent, its config or GNU Radio's user cache.
    assert checks["isolation"]["gi_import_blocked"] and not checks["isolation"]["gtk_modules_loaded"]
    assert all(checks["watched_roots_unchanged"].values()), result["watched_diff"]

"""HeadlessFlowgraphProxy (Tier A, eval plan §4.4): run, save, log and save-block parity with
GRC-Agent's NativeFlowgraphProxy (6c98128), checked against real tool results in Qoherent's SFT rows.

All proxy scenarios run in ONE child process with its own HOME, XDG directories and GRC_HIER_PATH:
GNU Radio fixes the hier-block library path when it is first imported and keeps one block registry
and one console-messenger list per process, and save_block writes into the hier-block library, so
the test process (and the real ~/.grc_gnuradio) is never touched. Every scenario drives GRC-Agent's
real tool functions (grc_agent.agent.run_flowgraph_func, get_run_log_func, save_graph_func,
save_block_func, change_graph_func), so the strings checked are exactly what a model would read.

Stops are timed against readiness markers, not sleeps, wherever the return code is asserted: a
SIGTERM that reaches a generated program before its handler is installed ends it with -15 (GRC's
own race, more likely on a loaded machine). Qt programs announce readiness from a zero-delay QTimer
(it fires inside ``qapp.exec_()``, after ``signal.signal(SIGTERM, ...)``); no_gui programs install
their handler before anything they print.

    ./gb test tests/test_tier_a_proxy.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

from grcbench.common.io import iter_jsonl
from grcbench.common.paths import REPO, SFT_RELEASE, WORK, ensure

CHILD_TIMEOUT = 900
TOOLS = ("run_flowgraph", "get_run_log", "save_graph", "save_block")


# =============================================================================== flowgraph fixtures
def _states(i: int) -> dict:
    return {"bus_sink": False, "bus_source": False, "bus_structure": None,
            "coordinate": [8 + 192 * (i % 6), 8 + 160 * (i // 6)], "rotation": 0, "state": "enabled"}


def _block(name: str, block_id: str, **params) -> dict:
    return {"name": name, "id": block_id, "parameters": {k: str(v) for k, v in params.items()}}


def _snippet(name: str, code: str, section: str = "main_after_init") -> dict:
    return _block(name, "snippet", section=section, priority="0", code=code)


def _write_grc(path: Path, options: dict, blocks: list[dict], connections: list[list[str]]) -> Path:
    """A complete .grc as GRC saves it (states included; output_language before the workflow params)."""
    data = {"options": {"parameters": {"output_language": "python", **options}, "states": _states(0)},
            "blocks": [dict(b, states=_states(i + 1)) for i, b in enumerate(blocks)],
            "connections": connections, "metadata": {"file_format": 1}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


SAMP = _block("samp_rate", "variable", value="32000")
QT_READY_MARK = "GRCBENCH_QT_READY"
QT_READY = _snippet("qt_ready", "from PyQt5 import Qt as _qt\n"
                                f"_qt.QTimer.singleShot(0, lambda: print('{QT_READY_MARK}', flush=True))",
                    "main_after_start")


def _tone(kind: str = "float") -> dict:
    return _block("src", "analog_sig_source_x", type=kind, samp_rate="samp_rate",
                  waveform="analog.GR_COS_WAVE", freq="1000", amp="0.5")


def _nogui_run(path: Path, extra: list[dict] = (), items: int = 2000, **options) -> Path:
    """no_gui, Run to Completion: tone -> head -> null sink."""
    blocks = [SAMP, _tone(), _block("head", "blocks_head", type="float", num_items=items),
              _block("sink", "blocks_null_sink", type="float"), *extra]
    return _write_grc(path, {"id": path.stem, "generate_options": "no_gui", "run_options": "run", **options},
                      blocks, [["src", "0", "head", "0"], ["head", "0", "sink", "0"]])


def _nogui_prompt(path: Path, extra: list[dict] = ()) -> Path:
    """no_gui, Prompt for Exit: tone -> throttle -> null sink (runs until stopped)."""
    blocks = [SAMP, _tone(), _block("thr", "blocks_throttle2", type="float", samples_per_second="samp_rate"),
              _block("sink", "blocks_null_sink", type="float"), *extra]
    return _write_grc(path, {"id": path.stem, "generate_options": "no_gui", "run_options": "prompt"},
                      blocks, [["src", "0", "thr", "0"], ["thr", "0", "sink", "0"]])


def _qt_scope(path: Path, extra: list[dict] = ()) -> Path:
    """QT GUI: tone -> throttle -> time sink (runs until stopped)."""
    blocks = [SAMP, _tone(), _block("thr", "blocks_throttle2", type="float", samples_per_second="samp_rate"),
              _block("time_sink", "qtgui_time_sink_x", type="float", srate="samp_rate", size="1024",
                     nconnections="1"), *extra]
    return _write_grc(path, {"id": path.stem, "generate_options": "qt_gui"}, blocks,
                      [["src", "0", "thr", "0"], ["thr", "0", "time_sink", "0"]])


def _epy_chain(path: Path, middle: dict) -> Path:
    """no_gui, Run to Completion, complex: tone -> <middle> -> head -> null sink."""
    blocks = [SAMP, _tone("complex"), middle, _block("head", "blocks_head", type="complex", num_items=2000),
              _block("sink", "blocks_null_sink", type="complex")]
    name = middle["name"]
    return _write_grc(path, {"id": path.stem, "generate_options": "no_gui", "run_options": "run"}, blocks,
                      [["src", "0", name, "0"], [name, "0", "head", "0"], ["head", "0", "sink", "0"]])


# =============================================================================== child process side
class _Console:
    """GRC's console as the user sees it: one more messenger on gnuradio.grc.core.Messages."""

    def __init__(self) -> None:
        from gnuradio.grc.core import Messages
        self.parts: list[str] = []
        Messages.register_messenger(self.append)

    def append(self, text: str) -> None:
        self.parts.append(text)

    def mark(self) -> int:
        return len("".join(self.parts))

    async def wait_for(self, marker: str, since: int = 0, timeout: float = 60.0) -> None:
        import asyncio
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while marker not in "".join(self.parts)[since:]:
            if loop.time() > deadline:
                raise TimeoutError(f"{marker!r} never reached GRC's console")
            await asyncio.sleep(0.05)

    def close(self) -> None:
        from gnuradio.grc.core import Messages
        if self.append in Messages.MESSENGERS_LIST:
            Messages.MESSENGERS_LIST.remove(self.append)


async def _wait_for_window(proxy, marker: str, timeout: float = 60.0) -> None:
    """Wait until the current run's terminal window (no_gui, xterm mode) shows ``marker``."""
    import asyncio
    path = Path(proxy.runs[-1]["terminal_transcript"])
    for _ in range(int(timeout / 0.05)):
        if marker in path.read_text(encoding="utf-8", errors="replace"):
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"{marker!r} never reached the terminal window")


class _Session:
    """One scenario: its own project directory, the tool calls it made, the proxies it opened."""

    def __init__(self, work: Path, name: str) -> None:
        self.dir = work / name
        self.project = self.dir / "project"
        self.project.mkdir(parents=True, exist_ok=True)
        self.calls: list[dict] = []
        self.proxies: list = []
        self.console = _Console()

    def open(self, path: Path, **kwargs):
        from grcbench.tier_a.headless_proxy import HeadlessFlowgraphProxy
        proxy = HeadlessFlowgraphProxy.open(path, **kwargs)
        self.proxies.append(proxy)
        return proxy

    def untitled(self, project_dir, **kwargs):
        from grcbench.tier_a.headless_proxy import HeadlessFlowgraphProxy
        proxy = HeadlessFlowgraphProxy.new_untitled(project_dir=project_dir, **kwargs)
        self.proxies.append(proxy)
        return proxy

    async def tool(self, name: str, proxy, **kwargs) -> dict:
        """Call GRC-Agent's real tool function, as pydantic-ai would, and record what the model sees."""
        from types import SimpleNamespace
        from grc_agent import agent
        fn = {"run_flowgraph": agent.run_flowgraph_func, "get_run_log": agent.get_run_log_func,
              "save_graph": agent.save_graph_func, "save_block": agent.save_block_func,
              "change_graph": agent.change_graph_func}[name]
        rec: dict = {"tool": name, "args": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))},
                     "mode": getattr(proxy, "_no_gui_terminal", None)}
        try:
            rec["result"] = json.loads(await fn(SimpleNamespace(deps=proxy), **kwargs))
        except Exception as exc:
            rec["exception"] = type(exc).__name__
            rec["message"] = str(exc)
        self.calls.append(rec)
        return rec

    async def close(self) -> None:
        try:
            for proxy in self.proxies:
                await proxy.aclose()
        finally:
            self.console.close()


def _grc_terminal_wrap(args: list[str]) -> list[str]:
    """What GRC's own add_xterm_to_run_command makes of ``args`` with the GUI Config's terminal."""
    from gnuradio import gr
    from gnuradio.grc.workflows.common import add_xterm_to_run_command
    return add_xterm_to_run_command(list(args), gr.prefs().get_string("grc", "xterm_executable", "xterm"))


async def _sc_nogui_terminal(s: _Session) -> dict:
    code = "print('GRCBENCH_HELLO µ')\nprint('GRCBENCH_CR', end='\\r\\n')"
    path = _nogui_run(s.project / "tone.grc", [_snippet("hello", code)])
    p = s.open(path)
    await s.tool("run_flowgraph", p, action="start", wait=True, timeout_seconds=60)
    await s.tool("get_run_log", p)
    from grcbench.tier_a.headless_proxy import terminal_command
    program = [sys.executable, "-u", str(path)[:-4] + ".py"]
    return {"path": str(path), "py_exists": (s.project / "tone.py").exists(), "saved": p.page.saved,
            "runs": p.runs, "ours": terminal_command(program), "grc": _grc_terminal_wrap(program)}


async def _sc_nogui_capture(s: _Session) -> dict:
    code = "print('GRCBENCH_HELLO µ')\nprint('GRCBENCH_CR', end='\\r\\n')"
    path = _nogui_run(s.project / "tone.grc", [_snippet("hello", code)])
    p = s.open(path, no_gui_terminal="capture")
    await s.tool("run_flowgraph", p, action="start", wait=True, timeout_seconds=60)
    await s.tool("get_run_log", p)
    return {"path": str(path), "py_exists": (s.project / "tone.py").exists(), "saved": p.page.saved,
            "runs": p.runs}


async def _sc_qt_bounded_background(s: _Session) -> dict:
    p = s.open(_qt_scope(s.project / "scope.grc", [QT_READY]))
    await s.tool("run_flowgraph", p, action="start", wait=True, stop_after_seconds=3)
    await s.tool("get_run_log", p)
    mark = s.console.mark()
    await s.tool("run_flowgraph", p, action="start", wait=False)
    await s.console.wait_for(QT_READY_MARK, mark)
    await s.tool("get_run_log", p)                  # the previous run's log, run_in_progress
    await s.tool("run_flowgraph", p, action="start")
    await s.tool("run_flowgraph", p, action="stop")
    await s.tool("get_run_log", p)
    await s.tool("run_flowgraph", p, action="stop")
    return {"runs": p.runs}


async def _sc_qt_timeout(s: _Session) -> dict:
    p = s.open(_qt_scope(s.project / "slow.grc", [QT_READY]))
    mark = s.console.mark()
    await s.tool("run_flowgraph", p, action="start", wait=True, timeout_seconds=4)
    await s.console.wait_for(QT_READY_MARK, mark)
    await s.tool("run_flowgraph", p, action="stop")
    await s.tool("get_run_log", p)
    return {"runs": p.runs}


async def _sc_nogui_prompt(s: _Session) -> dict:
    path = _nogui_prompt(s.project / "prompt.grc")
    p = s.open(path)                                # xterm: the terminal is stopped
    await s.tool("run_flowgraph", p, action="start", wait=True, stop_after_seconds=2)
    await s.tool("get_run_log", p)
    await s.tool("run_flowgraph", p, action="start", wait=True, timeout_seconds=2)
    await s.tool("get_run_log", p)
    await _wait_for_window(p, "Press Enter to quit: ")
    await s.tool("run_flowgraph", p, action="stop")
    await s.tool("get_run_log", p)
    runs = p.runs
    await p.aclose()
    q = s.open(path, no_gui_terminal="capture")     # capture: the program itself is stopped
    mark = s.console.mark()
    await s.tool("run_flowgraph", q, action="start", wait=False)
    await s.console.wait_for("Press Enter to quit: ", mark)
    await s.tool("run_flowgraph", q, action="stop")
    await s.tool("get_run_log", q)
    return {"runs": runs, "capture_runs": q.runs}


async def _sc_stop_requested(s: _Session) -> dict:
    code = ("import signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "print('GRCBENCH_IGNORING_SIGTERM', flush=True)")
    path = _nogui_prompt(s.project / "stubborn.grc", [_snippet("deaf", code, "main_after_start")])
    p = s.open(path, no_gui_terminal="capture")
    mark = s.console.mark()
    await s.tool("run_flowgraph", p, action="start", wait=False)
    await s.console.wait_for("GRCBENCH_IGNORING_SIGTERM", mark)
    await s.tool("run_flowgraph", p, action="stop")
    still_running = p.page.process is not None
    await p.aclose()                                # the session's cleanup kills the sandbox
    facts = {"still_running_after_stop": still_running, "runs": p.runs,
             "log_after_close": p.exec_monitor.get_last_run_log()}
    q = s.open(path)                                # xterm: closing the window ends it anyway
    await s.tool("run_flowgraph", q, action="start", wait=False)
    await _wait_for_window(q, "GRCBENCH_IGNORING_SIGTERM")
    await s.tool("run_flowgraph", q, action="stop")
    await s.tool("get_run_log", q)
    facts["xterm_runs"] = q.runs
    return facts


async def _sc_invalid(s: _Session) -> dict:
    blocks = [SAMP, _tone(), _block("sink", "blocks_null_sink", type="float")]     # sink input unconnected
    path = _write_grc(s.project / "broken.grc", {"id": "broken", "generate_options": "no_gui", "run_options": "run"},
                      blocks, [])
    p = s.open(path)
    await s.tool("run_flowgraph", p, action="start")         # 6c98128: TypeError (headless_proxy D13)
    await s.tool("get_run_log", p)
    await s.tool("run_flowgraph", p, action="start", wait=False, stop_after_seconds=3)
    await s.tool("run_flowgraph", p, action="start", wait=True, stop_after_seconds=0)
    try:
        await p.run_flowgraph(action="pause")
        bad_action = None
    except ValueError as exc:
        bad_action = str(exc)
    runs = p.runs
    await p.aclose()
    q = s.open(path, invalid_gate="intended")
    await s.tool("run_flowgraph", q, action="start")
    return {"bad_action": bad_action, "py_exists": (s.project / "broken.py").exists(), "runs": runs + q.runs}


async def _sc_untitled_save(s: _Session) -> dict:
    p = s.untitled(s.project)
    facts: dict = {"id_before": p.get_option("id"), "saved_before": p.page.saved}
    await s.tool("run_flowgraph", p, action="start")
    await s.tool("save_graph", p)
    target = s.project / "untitled.grc"
    reference = s.dir / "reference.grc"
    p.parent_platform.save_flow_graph(str(reference), p.page.flow_graph)
    facts.update(id_after=p.get_option("id"), grc_file_path=p.grc_file_path, saved_after=p.page.saved,
                 lock_exists=(s.project / ".grc_agent" / "untitled.grc.lock").exists(),
                 same_bytes_as_grc=target.read_bytes() == reference.read_bytes(),
                 mode=oct(target.stat().st_mode & 0o777))
    await s.tool("save_graph", p)
    await p.aclose()
    p2 = s.untitled(s.project)
    await s.tool("save_graph", p2)
    facts["second_id"] = p2.get_option("id")
    await p2.aclose()
    p3 = s.untitled(None)
    await s.tool("save_graph", p3)
    await p3.aclose()
    from grc_agent.adapter.graph import _NO_ACTIVE_GRAPH_MSG
    facts["no_project_msg"] = _NO_ACTIVE_GRAPH_MSG
    return facts


async def _sc_crash(s: _Session) -> dict:
    boom = _snippet("boom", "raise RuntimeError('GRCBENCH_BOOM')")
    path = _nogui_run(s.project / "crash.grc", [boom])
    runs = []
    for mode in ("xterm", "capture"):
        p = s.open(path, no_gui_terminal=mode)
        await s.tool("run_flowgraph", p, action="start")
        await s.tool("get_run_log", p)
        await p.aclose()
        runs += p.runs
    q = s.open(_qt_scope(s.project / "qtcrash.grc", [boom]))
    await s.tool("run_flowgraph", q, action="start")
    await s.tool("get_run_log", q)
    return {"runs": runs + q.runs}


async def _sc_spawn_failure(s: _Session) -> dict:
    path = _nogui_run(s.project / "spawn.grc", run_command="/nonexistent/grcbench-python -u {filename}")
    p = s.open(path, no_gui_terminal="capture")
    await s.tool("run_flowgraph", p, action="start")
    await s.tool("get_run_log", p)
    await p.aclose()
    q = s.open(path)
    await s.tool("run_flowgraph", q, action="start")
    await s.tool("get_run_log", q)
    return {"path": str(path), "runs": p.runs, "run_failures": p.run_failures, "xterm_runs": q.runs}


async def _sc_hier_not_started(s: _Session) -> dict:
    blocks = [_block("pad_in", "pad_source", label="in", type="float"),
              _block("gain", "blocks_multiply_const_vxx", type="float", const="2"),
              _block("pad_out", "pad_sink", label="out", type="float")]
    path = _write_grc(s.project / "grcbench_hier.grc", {"id": "grcbench_hier", "generate_options": "hb"},
                      blocks, [["pad_in", "0", "gain", "0"], ["gain", "0", "pad_out", "0"]])
    p = s.open(path)
    await s.tool("run_flowgraph", p, action="start")
    hier = p._hier_dir()
    return {"hier_files": sorted(f.name for f in hier.glob("grcbench_hier*")), "runs": p.runs}


async def _sc_notify_edit(s: _Session) -> dict:
    from grc_agent.agent import ParamUpdate
    path = _qt_scope(s.project / "edit.grc")
    p = s.open(path)
    await s.tool("run_flowgraph", p, action="start", wait=True, stop_after_seconds=2)
    await s.tool("change_graph", p, reason="raise the tone frequency", add_blocks=[], remove_blocks=[],
                 update_params=[ParamUpdate(instance_name="src", params={"freq": "2000"})],
                 update_states=[], add_connections=[], remove_connections=[], force=False)
    facts = {"saved_after_edit": p.page.saved}
    await s.tool("get_run_log", p)
    await s.tool("run_flowgraph", p, action="start", wait=True, stop_after_seconds=2)
    await s.tool("get_run_log", p)
    disk = yaml.safe_load(path.read_text())
    facts["disk_freq"] = next(b for b in disk["blocks"] if b["name"] == "src")["parameters"]["freq"]
    facts["saved_after_run"] = p.page.saved
    return facts


async def _sc_save_block(s: _Session) -> dict:
    from gnuradio.grc.core.blocks.embedded_python import DEFAULT_CODE
    epy = _block("epy_block_0", "epy_block", _source_code=DEFAULT_CODE, example_param="1.0")
    p = s.open(_epy_chain(s.project / "with_epy.grc", epy))
    await s.tool("save_block", p, instance_name="epy_block_0", block_id="grcbench_mult")
    await s.tool("save_block", p, instance_name="epy_block_0", block_id="grcbench_mult")
    await s.tool("save_block", p, instance_name="epy_block_0", block_id="grcbench_mult", overwrite=True)
    await s.tool("save_block", p, instance_name="src", block_id="grcbench_not_epy")
    hier = p._hier_dir()
    facts = {"hier": str(hier), "files": sorted(f.name for f in hier.glob("grcbench_mult*")),
             "in_catalog": "grcbench_mult" in p.parent_platform.blocks}
    await p.aclose()
    user = _block("mult", "grcbench_mult", example_param="2.0")
    q = s.open(_epy_chain(s.project / "uses_saved.grc", user), no_gui_terminal="capture")
    q.validate()
    facts["user_valid"] = q.is_valid()
    facts["user_errors"] = [str(e) for e in q.iter_error_messages()]
    await s.tool("run_flowgraph", q, action="start")
    await s.tool("get_run_log", q)
    return facts


async def _sc_sandbox(s: _Session) -> dict:
    import pwd
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    probes = [REPO / ".grcbench_sandbox_probe", real_home / ".grcbench_sandbox_probe",
              Path(os.environ["HOME"]) / ".grcbench_sandbox_probe"]
    targets = ", ".join(repr(str(t)) for t in probes) + ", 'cwd_probe.txt'"
    code = ("import socket\n"
            f"for target in [{targets}]:\n"
            "    try:\n"
            "        with open(target, 'w') as fh:\n"
            "            fh.write('x')\n"
            "        print('WROTE', target)\n"
            "    except OSError as exc:\n"
            "        print('BLOCKED', target, type(exc).__name__)\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
            "    print('NET_OPEN')\n"
            "except OSError as exc:\n"
            "    print('NET_BLOCKED', type(exc).__name__)\n")
    p = s.open(_nogui_run(s.project / "probe.grc", [_snippet("probe_snippet", code)]))
    await s.tool("run_flowgraph", p, action="start")
    leaked = [str(t) for t in probes if t.exists()]
    for t in leaked:                                # never leave a probe behind, even on failure
        Path(t).unlink()
    return {"probes": [str(t) for t in probes], "leaked": leaked, "runs": p.runs,
            "cwd_probe": (s.project / "cwd_probe.txt").exists()}


async def _sc_hard_cap(s: _Session) -> dict:
    p = s.open(_qt_scope(s.project / "capped.grc"), max_run_seconds=3)
    await s.tool("run_flowgraph", p, action="start", wait=True, timeout_seconds=30)
    await s.tool("get_run_log", p)
    return {"runs": p.runs}


async def _sc_delegation(s: _Session) -> dict:
    from gnuradio.grc.core import Messages
    from grc_agent import deps, fs_tools
    from grcbench.tier_a.headless_proxy import HeadlessFlowgraphProxy, native_canvas_module
    path = _nogui_run(s.project / "deleg.grc")
    p = s.open(path)
    facts = {
        "grc_file_path": p.grc_file_path, "option_id": p.get_option("id"),
        "blocks": sorted(b.name for b in p.blocks),
        "protocols": {name: isinstance(p, getattr(deps, name)) for name in (
            "SupportsRunFlowgraph", "SupportsGetRunLog", "SupportsSaveGraph", "SupportsSaveBlock",
            "SupportsNotifyEdit")},
        "active_project_dir": str(fs_tools.active_project_dir()),
        "active_grc_path": str(fs_tools.active_grc_path()),
        "monitor_registered": p.exec_monitor.handle_message in Messages.MESSENGERS_LIST,
        "echo_present": any(getattr(m, "__name__", "") == "write" for m in Messages.MESSENGERS_LIST),
        "stub_class_ok": native_canvas_module().NativeFlowgraphProxy is HeadlessFlowgraphProxy,
    }
    p.grcbench_marker = 7
    facts["setattr_delegated"] = getattr(p.page.flow_graph, "grcbench_marker", None) == 7
    try:
        HeadlessFlowgraphProxy.open(path)
        facts["second_open"] = None
    except RuntimeError as exc:
        facts["second_open"] = str(exc)
    await p.aclose()
    facts.update(
        monitor_registered_after=p.exec_monitor.handle_message in Messages.MESSENGERS_LIST,
        echo_restored=any(getattr(m, "__name__", "") == "write" for m in Messages.MESSENGERS_LIST),
        active_project_dir_after=fs_tools.active_project_dir())
    try:
        HeadlessFlowgraphProxy.open(path, no_gui_terminal="window")
        facts["bad_mode"] = None
    except ValueError as exc:
        facts["bad_mode"] = str(exc)
    return facts


SCENARIOS = {
    "nogui_terminal": _sc_nogui_terminal, "nogui_capture": _sc_nogui_capture,
    "qt_bounded_background": _sc_qt_bounded_background, "qt_timeout": _sc_qt_timeout,
    "nogui_prompt": _sc_nogui_prompt, "stop_requested": _sc_stop_requested, "invalid": _sc_invalid,
    "untitled_save": _sc_untitled_save, "crash": _sc_crash, "spawn_failure": _sc_spawn_failure,
    "hier_not_started": _sc_hier_not_started, "notify_edit": _sc_notify_edit, "save_block": _sc_save_block,
    "sandbox": _sc_sandbox, "hard_cap": _sc_hard_cap, "delegation": _sc_delegation,
}


def _child(work: Path, out: Path, only: list[str]) -> int:
    import asyncio

    async def run_all() -> dict:
        from gnuradio.grc.core.Config import Config
        result = {"python": sys.executable, "hier": Config.hier_block_lib_dir, "scenarios": {}}
        for name, fn in SCENARIOS.items():
            if only and name not in only:
                continue
            s = _Session(work, name)
            entry: dict = {}
            try:
                entry = {"ok": True, "facts": await fn(s)}
            except BaseException:
                entry = {"ok": False, "error": traceback.format_exc()}
            finally:
                try:
                    await s.close()
                except BaseException:
                    entry = {"ok": False, "error": traceback.format_exc()}
            entry["calls"] = s.calls
            result["scenarios"][name] = entry
        return result

    data = asyncio.run(run_all())
    out.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    return 0


# =============================================================================== test process side
@pytest.fixture(scope="module")
def child():
    from grcbench.tier_f.grc import install_warm_cache
    work = Path(tempfile.mkdtemp(prefix="tier-a-proxy-", dir=ensure(WORK / "tests")))
    home = work / "agent-home"
    for sub in (".cache", ".config", ".local/share", ".local/state", ".grc_gnuradio"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    install_warm_cache(home)
    src = str(REPO / "src")
    env = dict(os.environ, HOME=str(home), XDG_CACHE_HOME=str(home / ".cache"),
               XDG_CONFIG_HOME=str(home / ".config"), XDG_DATA_HOME=str(home / ".local/share"),
               XDG_STATE_HOME=str(home / ".local/state"), GRC_HIER_PATH=str(home / ".grc_gnuradio"),
               PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH", "")])))
    out = work / "results.json"
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(work), str(out)],
                          env=env, cwd=str(work), capture_output=True, text=True, timeout=CHILD_TIMEOUT)
    if not out.exists():
        pytest.fail(f"proxy scenario process failed (code {proc.returncode})\n"
                    f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
    data = json.loads(out.read_text(encoding="utf-8"))
    data["work"] = str(work)
    yield data
    shutil.rmtree(work, ignore_errors=True)


def _scenario(child: dict, name: str) -> dict:
    sc = child["scenarios"][name]
    assert sc["ok"], sc.get("error")
    return sc


def _calls(sc: dict, tool: str, mode: str | None = None) -> list[dict]:
    return [c for c in sc["calls"] if c["tool"] == tool and (mode is None or c["mode"] == mode)]


def _ok(call: dict) -> dict:
    assert "result" in call, f"{call['tool']} raised {call.get('exception')}: {call.get('message')}"
    return call["result"]


def _retry(call: dict) -> str:
    assert call.get("exception") == "ModelRetry", call
    return call["message"]


DONE = re.compile(r"\n>>> Done( \(return code -?\d+\))?\n$")
NOGUI_RUN_NOTE = ("This flowgraph was generated with generate_options='no_gui'. GNU Radio runs "
                  "no_gui flowgraphs in an external terminal wrapper; read get_run_log for output details.")


# ------------------------------------------------------------------------------- behaviour
@pytest.mark.gnuradio
@pytest.mark.slow
def test_nogui_program_runs_in_the_emulated_terminal_window(child):
    sc = _scenario(child, "nogui_terminal")
    facts = sc["facts"]
    res = _ok(_calls(sc, "run_flowgraph")[0])
    assert res == {"status": "completed", "return_code": 0, "ran_successfully": True, "note": NOGUI_RUN_NOTE,
                   "generate_options": "no_gui"}
    if facts["grc"] != facts["ours"][2:]:            # a terminal is installed: GRC's own wrapping, exactly
        assert facts["ours"] == facts["grc"]
    log = _ok(_calls(sc, "get_run_log")[0])
    # The console shows the terminal's command and nothing of the program's output (it went to the
    # window); the terminal exits 0 when its program ends.
    assert log["log_text"] == f"\nExecuting: {' '.join(facts['ours'])}\n\n\n>>> Done\n"
    assert log["return_code"] == 0 and log["ran_successfully"] is True and log["run_in_progress"] is False
    assert log["generate_options"] == "no_gui" and "external_terminal_note" in log
    run, = facts["runs"]
    assert run["terminal"] is True and run["program_return_code"] == 0 and run["return_code"] == 0
    assert run["command"] == " ".join(facts["ours"]) and run["program_command"] == " ".join(facts["ours"][2:])
    assert run["terminal_output_tail"] == "GRCBENCH_HELLO µ\nGRCBENCH_CR\r\n"
    assert facts["py_exists"] and facts["saved"] and run["hard_cap_hit"] is False


@pytest.mark.gnuradio
@pytest.mark.slow
def test_capture_mode_log_is_grc_console_output(child):
    sc = _scenario(child, "nogui_capture")
    res = _ok(_calls(sc, "run_flowgraph")[0])
    assert res["status"] == "completed" and res["return_code"] == 0 and res["ran_successfully"] is True
    assert res["generate_options"] == "no_gui"
    log = _ok(_calls(sc, "get_run_log")[0])
    path = sc["facts"]["path"]
    # Executor: "\nExecuting: <cmd>\n", the read loop's leading "\n", the output (universal newlines,
    # UTF-8), then "\n>>> Done\n" for return code 0.
    assert log["log_text"] == (f"\nExecuting: {child['python']} -u {path[:-4]}.py\n\n"
                               "GRCBENCH_HELLO µ\nGRCBENCH_CR\n\n>>> Done\n")
    assert log["return_code"] == 0 and log["ran_successfully"] is True and log["run_in_progress"] is False
    assert log["generate_options"] == "no_gui" and "external_terminal_note" in log
    run, = sc["facts"]["runs"]
    assert run["terminal"] is False and run["terminal_transcript"] is None and run["program_return_code"] == 0


@pytest.mark.gnuradio
@pytest.mark.slow
def test_qt_bounded_run_background_run_and_stop(child):
    sc = _scenario(child, "qt_bounded_background")
    runs = _calls(sc, "run_flowgraph")
    logs = _calls(sc, "get_run_log")
    bounded = _ok(runs[0])
    # A cold Qt program may still be starting after 3s: then the SIGTERM meets the default action
    # (-15) instead of the generated handler (0), exactly as under GRC.
    assert bounded["status"] == "stopped_after_timeout" and bounded["return_code"] in (0, -15)
    assert bounded["note"].startswith("Auto-stopped after 3s of runtime (SIGTERM")
    first_log = _ok(logs[0])
    assert first_log["return_code"] == bounded["return_code"] and first_log["run_in_progress"] is False
    assert "generate_options" not in first_log and DONE.search(first_log["log_text"])
    assert _ok(runs[1])["status"] == "started"
    during = _ok(logs[1])
    assert during["run_in_progress"] is True and "in_progress_note" in during
    assert during["log_text"] == first_log["log_text"]          # still the previous run's log
    assert "already in progress" in _retry(runs[2])
    assert _ok(runs[3])["status"] == "stopped"
    after = _ok(logs[2])                                         # stopped once ready: the handler ran
    assert after["run_in_progress"] is False and after["return_code"] == 0 and after["ran_successfully"]
    assert QT_READY_MARK in after["log_text"] and after["log_text"].endswith("\n\n>>> Done\n")
    assert _ok(runs[4]) == {"status": "not_running", "note": "No flowgraph execution is in progress."}


@pytest.mark.gnuradio
@pytest.mark.slow
def test_timeout_reports_still_running_then_stop(child):
    sc = _scenario(child, "qt_timeout")
    runs = _calls(sc, "run_flowgraph")
    res = _ok(runs[0])
    assert res == {"status": "still_running", "note": (
        "The run did not finish within 4s. GUI flowgraphs run until stopped — call "
        "run_flowgraph(action='stop') when done, then read get_run_log.")}
    assert _ok(runs[1])["status"] == "stopped"
    log = _ok(_calls(sc, "get_run_log")[0])
    assert log["run_in_progress"] is False and log["return_code"] == 0 and log["ran_successfully"] is True


@pytest.mark.gnuradio
@pytest.mark.slow
def test_prompt_program_runs_until_stopped_like_in_its_terminal(child):
    sc = _scenario(child, "nogui_prompt")
    runs = _calls(sc, "run_flowgraph", "xterm")
    logs = _calls(sc, "get_run_log", "xterm")
    # xterm: Stop terminates the terminal, which exits with status 15 (Qoherent's rows show the same).
    assert _ok(runs[0])["status"] == "stopped_after_timeout" and _ok(runs[0])["return_code"] == 15
    first = _ok(logs[0])
    assert first["log_text"].endswith("\n\n\n>>> Done (return code 15)\n") and "Press Enter" not in first["log_text"]
    assert first["return_code"] == 15 and first["ran_successfully"] is False
    assert first["generate_options"] == "no_gui" and "external_terminal_note" in first
    still = _ok(runs[1])
    assert still["status"] == "still_running" and still["generate_options"] == "no_gui"
    during = _ok(logs[1])
    assert during["run_in_progress"] is True and {"in_progress_note", "external_terminal_note"} <= set(during)
    assert _ok(runs[2])["status"] == "stopped" and _ok(logs[2])["return_code"] == 15
    bounded_run, stopped_run = sc["facts"]["runs"]
    for run in (bounded_run, stopped_run):
        assert run["return_code"] == 15 and run["program_return_code"] == -1   # the window's hang-up
    assert bounded_run["terminal_output_tail"] in ("", "Press Enter to quit: ")   # may still be starting
    assert stopped_run["terminal_output_tail"] == "Press Enter to quit: "
    # capture: the program's own SIGTERM handler ends it with 0, and its prompt is in the console.
    cap_run, = _calls(sc, "run_flowgraph", "capture")[1:]
    assert _ok(cap_run)["status"] == "stopped"
    cap_log = _ok(_calls(sc, "get_run_log", "capture")[0])
    assert "Press Enter to quit: " in cap_log["log_text"] and cap_log["return_code"] == 0
    assert cap_log["ran_successfully"] is True


@pytest.mark.gnuradio
@pytest.mark.slow
def test_stop_requested_when_the_program_ignores_sigterm(child):
    sc = _scenario(child, "stop_requested")
    runs = _calls(sc, "run_flowgraph", "capture")
    assert _ok(runs[0])["status"] == "started"
    assert _ok(runs[1]) == {"status": "stop_requested", "note": (
        "SIGTERM sent; the process is still shutting down. Check get_run_log "
        "shortly (its run_in_progress field) to confirm it ended.")}
    facts = sc["facts"]
    assert facts["still_running_after_stop"] is True
    run, = facts["runs"]
    assert run["return_code"] == -9                       # aclose() tore the sandbox down
    assert "GRCBENCH_IGNORING_SIGTERM" in facts["log_after_close"]["log_text"]
    # In its terminal the same program is stopped at once: the terminal exits, the window hangs up.
    xruns = _calls(sc, "run_flowgraph", "xterm")
    assert _ok(xruns[0])["status"] == "started" and _ok(xruns[1])["status"] == "stopped"
    assert _ok(_calls(sc, "get_run_log", "xterm")[0])["return_code"] == 15
    xrun, = facts["xterm_runs"]
    assert xrun["return_code"] == 15 and xrun["program_return_code"] == -1 and xrun["wall_seconds"] < 30


@pytest.mark.gnuradio
@pytest.mark.slow
def test_invalid_graph_and_bad_arguments_are_refused_before_execute(child):
    sc = _scenario(child, "invalid")
    runs = _calls(sc, "run_flowgraph")
    # Pinned 6c98128: the gate joins GRC's (element, message) tuples as strings (headless_proxy D13).
    assert runs[0]["exception"] == "TypeError"
    assert runs[0]["message"] == "sequence item 0: expected str instance, tuple found"
    intended = _retry(runs[3])
    assert intended == ("The flowgraph is invalid, so GRC will refuse to execute it: "
                        "src: Source - out(0): Port is not connected.; sink: Sink - in(0): Port is not "
                        "connected.. Fix the graph with change_graph first.")
    assert _ok(_calls(sc, "get_run_log")[0]) == {"log_text": "", "message": (
        "No flowgraph has been run yet. Use GRC's Execute button to run the flowgraph first.")}
    assert _retry(runs[1]).startswith("stop_after_seconds requires wait=True")
    assert _retry(runs[2]).startswith("Invalid stop_after_seconds 0")
    assert sc["facts"]["bad_action"] == "Invalid action 'pause': must be 'start' or 'stop'"
    assert not sc["facts"]["py_exists"] and sc["facts"]["runs"] == []


@pytest.mark.gnuradio
@pytest.mark.slow
def test_untitled_graph_must_be_saved_and_saves_like_grc(child):
    sc = _scenario(child, "untitled_save")
    facts = sc["facts"]
    assert "has never been saved" in _retry(_calls(sc, "run_flowgraph")[0])
    saves = _calls(sc, "save_graph")
    first = _ok(saves[0])
    assert first["page"] == "untitled" and first["path"].endswith("/project/untitled.grc")
    assert facts["id_before"] == "default" and facts["id_after"] == "untitled"   # SAVE_AS id rename
    assert facts["grc_file_path"] == first["path"] and facts["saved_after"] and not facts["saved_before"]
    assert facts["lock_exists"] and facts["same_bytes_as_grc"]
    assert _ok(saves[1]) == first                                   # titled now: saves in place
    second = _ok(saves[2])
    assert second["page"] == "untitled(1)" and facts["second_id"] == "untitled_1"
    assert _retry(saves[3]) == facts["no_project_msg"]


@pytest.mark.gnuradio
@pytest.mark.slow
def test_runtime_crash_reports_the_return_code(child):
    sc = _scenario(child, "crash")
    xterm_run, capture_run, qt = (_ok(c) for c in _calls(sc, "run_flowgraph"))
    xterm_log, capture_log, qt_log = (_ok(c) for c in _calls(sc, "get_run_log"))
    # In its terminal the crash is only visible in the window; the terminal still exits 0 (GRC's
    # no_gui note warns the model about exactly this).
    assert xterm_run == {"status": "completed", "return_code": 0, "ran_successfully": True,
                         "note": NOGUI_RUN_NOTE, "generate_options": "no_gui"}
    assert "Traceback" not in xterm_log["log_text"] and xterm_log["log_text"].endswith("\n\n\n>>> Done\n")
    xrun, crun, qrun = sc["facts"]["runs"]
    assert xrun["program_return_code"] == 1 and "RuntimeError: GRCBENCH_BOOM" in xrun["terminal_output_tail"]
    assert capture_run["status"] == "completed" and capture_run["return_code"] == 1
    assert not capture_run["ran_successfully"] and capture_run["generate_options"] == "no_gui"
    assert qt == {"status": "completed", "return_code": 1, "ran_successfully": False, "note": (
        "Read the full console output with the get_run_log tool. An empty log "
        "with an immediate completion can mean the graph ran in an external "
        "terminal (no_gui graphs) or failed to spawn — check the log and ask "
        "the user if in doubt.")}
    for log in (capture_log, qt_log):
        assert "Traceback" in log["log_text"] and "RuntimeError: GRCBENCH_BOOM" in log["log_text"]
        assert log["log_text"].endswith("\n>>> Done (return code 1)\n")
    assert crun["program_return_code"] == 1 == crun["return_code"] and qrun["return_code"] == 1


@pytest.mark.gnuradio
@pytest.mark.slow
def test_spawn_failure_is_reported_like_grc_executor(child):
    sc = _scenario(child, "spawn_failure")
    res = _ok(_calls(sc, "run_flowgraph", "capture")[0])
    assert res["status"] == "completed" and res["return_code"] == 0
    assert res["ran_successfully"] is False and res["spawn_failed"] is True
    assert res["note"].startswith("The flowgraph process failed to spawn")
    log = _ok(_calls(sc, "get_run_log", "capture")[0])
    path = sc["facts"]["path"]
    err = "[Errno 2] No such file or directory: '/nonexistent/grcbench-python'"
    # Executor.__init__'s except branch: send_verbose_exec(str(e)); send_end_exec() (no code).
    assert log["log_text"] == f"\nExecuting: /nonexistent/grcbench-python -u {path[:-4]}.py\n{err}\n>>> Done\n"
    assert log["spawn_failed"] is True and "spawn_note" in log and log["ran_successfully"] is False
    assert sc["facts"]["runs"][0]["spawn_error"] == err
    assert sc["facts"]["run_failures"] == []          # agent-initiated: no follow-up failure turn
    # In a terminal, the terminal itself starts; the program's exec error is shown in its window.
    xres = _ok(_calls(sc, "run_flowgraph", "xterm")[0])
    assert xres["status"] == "completed" and xres["return_code"] == 0 and "spawn_failed" not in xres
    xlog = _ok(_calls(sc, "get_run_log", "xterm")[0])
    assert xlog["log_text"].endswith(f" -e /nonexistent/grcbench-python -u {path[:-4]}.py\n\n\n>>> Done\n")
    xrun, = sc["facts"]["xterm_runs"]
    assert xrun["program_spawn_error"] == err and xrun["terminal_output_tail"] == err + "\n"


@pytest.mark.gnuradio
@pytest.mark.slow
def test_hier_block_graph_generates_but_is_not_started(child):
    sc = _scenario(child, "hier_not_started")
    res = _ok(_calls(sc, "run_flowgraph")[0])
    assert res["status"] == "not_started" and set(res) == {"status", "note"}
    assert "grcbench_hier.block.yml" in sc["facts"]["hier_files"]
    assert sc["facts"]["runs"] == []


@pytest.mark.gnuradio
@pytest.mark.slow
def test_notify_edit_marks_the_log_stale_and_execute_saves_first(child):
    sc = _scenario(child, "notify_edit")
    assert _ok(_calls(sc, "change_graph")[0])["ok"] is True
    facts = sc["facts"]
    assert facts["saved_after_edit"] is False and facts["saved_after_run"] is True
    stale, fresh = (_ok(c) for c in _calls(sc, "get_run_log"))
    assert stale["note"].startswith("The flowgraph has been modified in memory since this run completed")
    assert "note" not in fresh
    assert [_ok(c)["status"] for c in _calls(sc, "run_flowgraph")] == ["stopped_after_timeout"] * 2
    assert facts["disk_freq"] == "2000"


@pytest.mark.gnuradio
@pytest.mark.slow
def test_save_block_round_trip(child):
    sc = _scenario(child, "save_block")
    saves = _calls(sc, "save_block")
    first = _ok(saves[0])
    hier = child["hier"]
    assert first == {"ok": True, "block_id": "grcbench_mult", "label": "Grcbench Mult", "category": "[Custom]",
                     "saved_to": {"block_yml": f"{hier}/grcbench_mult.block.yml", "py": f"{hier}/grcbench_mult.py"},
                     "params": ["example_param"], "inputs": ["0"], "outputs": ["0"]}
    assert "already exists as a previously agent-saved block" in _retry(saves[1])
    assert _ok(saves[2])["ok"] is True
    assert "not an epy_block" in _retry(saves[3])
    facts = sc["facts"]
    assert facts["files"] == ["grcbench_mult.block.yml", "grcbench_mult.py"] and facts["in_catalog"]
    assert facts["user_valid"], facts["user_errors"]
    run = _ok(_calls(sc, "run_flowgraph")[0])                   # capture mode: the program's own code
    assert run["status"] == "completed" and run["return_code"] == 0 and run["ran_successfully"]
    assert "Traceback" not in _ok(_calls(sc, "get_run_log")[0])["log_text"]


@pytest.mark.gnuradio
@pytest.mark.slow
def test_flowgraph_runs_in_the_sandbox(child):
    sc = _scenario(child, "sandbox")
    run, = sc["facts"]["runs"]
    window = run["terminal_output_tail"]
    for probe in sc["facts"]["probes"]:
        assert f"BLOCKED {probe} OSError" in window, window
    assert sc["facts"]["leaked"] == []
    assert "WROTE cwd_probe.txt" in window and sc["facts"]["cwd_probe"]       # the project dir is writable
    assert "NET_BLOCKED OSError" in window and "NET_OPEN" not in window


@pytest.mark.gnuradio
@pytest.mark.slow
def test_hard_cap_kills_the_run(child):
    sc = _scenario(child, "hard_cap")
    res = _ok(_calls(sc, "run_flowgraph")[0])
    assert res["status"] == "completed" and res["return_code"] == -9 and res["ran_successfully"] is False
    assert _ok(_calls(sc, "get_run_log")[0])["log_text"].endswith("\n>>> Done (return code -9)\n")
    run, = sc["facts"]["runs"]
    assert run["hard_cap_hit"] is True and 2.5 < run["wall_seconds"] < 10


@pytest.mark.gnuradio
@pytest.mark.slow
def test_delegation_protocols_and_process_global_wiring(child):
    facts = _scenario(child, "delegation")["facts"]
    assert facts["grc_file_path"].endswith("/project/deleg.grc") and facts["option_id"] == "deleg"
    assert facts["blocks"] == ["deleg", "head", "samp_rate", "sink", "src"]      # options block included
    assert all(facts["protocols"].values()), facts["protocols"]
    assert facts["active_project_dir"].endswith("/project")
    assert facts["active_grc_path"] == facts["grc_file_path"]
    assert facts["monitor_registered"] and not facts["echo_present"]
    assert facts["setattr_delegated"] and facts["stub_class_ok"]
    assert facts["second_open"] and "Another HeadlessFlowgraphProxy is still open" in facts["second_open"]
    assert not facts["monitor_registered_after"] and facts["echo_restored"]
    assert facts["active_project_dir_after"] is None
    assert facts["bad_mode"] == "no_gui_terminal must be 'xterm' or 'capture', not 'window'"


# ------------------------------------------------------------------------------- SFT conformance
@lru_cache(maxsize=1)
def _sft_results() -> dict[str, list[dict]]:
    """Every run/log/save tool result in Qoherent's release (training rows + quarantined rows: all
    are real outputs of the collection build)."""
    files = [SFT_RELEASE / "sft_train.jsonl", SFT_RELEASE / "quarantine.jsonl"]
    if not files[0].exists():
        pytest.skip("Qoherent's SFT release is not present")
    out: dict[str, list[dict]] = defaultdict(list)
    for path in files:
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            names = {tc["id"]: tc["name"] for m in row["messages"] for tc in (m.get("tool_calls") or [])
                     if "id" in tc}
            for m in row["messages"]:
                if m.get("role") != "tool":
                    continue
                name = m.get("name") or names.get(m.get("tool_call_id"))
                if name not in TOOLS:
                    continue
                try:
                    obj = json.loads(m["content"])
                except (TypeError, ValueError):
                    continue
                if isinstance(obj, dict):
                    out[name].append(obj)
    return out


def _ours(child: dict, tool: str) -> list[dict]:
    return [c["result"] for sc in child["scenarios"].values() for c in sc["calls"]
            if c["tool"] == tool and "result" in c]


def _norm(text: str) -> str:
    return re.sub(r"\b\d+(\.\d+)?s\b", "<N>s", text)


# run_flowgraph shapes the pinned 6c98128 can return that Qoherent's traces never show.
EXTRA_RUN_SHAPES = {
    ("completed", frozenset({"status", "return_code", "ran_successfully", "note", "generate_options", "spawn_failed"})),
    ("completed", frozenset({"status", "return_code", "ran_successfully", "note", "spawn_failed"})),
    ("still_running", frozenset({"status", "note"})),           # a GUI graph (SFT: only no_gui)
    ("stop_requested", frozenset({"status", "note"})),
}
SFT_BUILD_STOP_NOTE = ("Auto-stopped after <N>s of runtime (stop escalation: SIGTERM, then SIGKILL if the "
                       "shutdown stalls — a deliberate stop, not a failure). Read the full output with get_run_log.")
PINNED_STOP_NOTE = ("Auto-stopped after <N>s of runtime (SIGTERM — the same Stop the toolbar button sends; "
                    "a deliberate stop, not a failure). Read the full output with get_run_log.")
# The terminal's own stderr in Qoherent's collection image (missing X fonts), not GRC behaviour.
SFT_XTERM_NOISE = re.compile(r'^xterm: cannot load font "[^"\n]*"\n', re.M)


def _log_shape(text: str) -> str:
    return re.sub(r"(?<=\s)/\S+", "<PATH>", SFT_XTERM_NOISE.sub("", text))


@pytest.mark.gnuradio
@pytest.mark.slow
def test_run_flowgraph_results_match_the_sft_rows(child):
    from grcbench.tier_a.headless_proxy import RUN_STATUSES
    sft = _sft_results()["run_flowgraph"]
    ours = _ours(child, "run_flowgraph")
    assert len(sft) > 50
    sft_statuses = {r["status"] for r in sft}
    assert sft_statuses <= set(RUN_STATUSES)
    assert sft_statuses <= {r["status"] for r in ours}, "every status in the traces is reproduced"
    sft_shapes = {(r["status"], frozenset(r)) for r in sft}
    our_shapes = {(r["status"], frozenset(r)) for r in ours}
    assert sft_shapes - our_shapes == set(), "every result shape in the traces is reproduced"
    assert our_shapes - sft_shapes <= EXTRA_RUN_SHAPES
    sft_notes = {(r["status"], _norm(r["note"])) for r in sft}
    our_notes = {(r["status"], _norm(r["note"])) for r in ours if "note" in r}
    differing = sft_notes - our_notes
    # The only text that differs is the collection build's stop note (headless_proxy D11).
    assert differing == {("stopped_after_timeout", SFT_BUILD_STOP_NOTE)}
    assert ("stopped_after_timeout", PINNED_STOP_NOTE) in our_notes
    # Return codes of stops: 0 (a program's handler) and 15 (a terminal) in both.
    sft_stop_codes = {r["return_code"] for r in sft if r["status"] == "stopped_after_timeout"}
    our_stop_codes = {r["return_code"] for r in ours if r["status"] == "stopped_after_timeout"}
    assert sft_stop_codes == {0, 15} and 15 in our_stop_codes and our_stop_codes <= {0, 15, -15}


@pytest.mark.gnuradio
@pytest.mark.slow
def test_get_run_log_results_match_the_sft_rows(child):
    sft = _sft_results()["get_run_log"]
    ours = _ours(child, "get_run_log")
    assert len(sft) > 30
    with_log = [r for r in sft if "return_code" in r]
    # The collection build adds run_id to every run log; 6c98128 never does (headless_proxy D11).
    assert with_log and all("run_id" in r for r in with_log)
    assert not any("run_id" in r for r in ours)
    sft_shapes = {frozenset(r) - {"run_id"} for r in sft}
    our_shapes = {frozenset(r) for r in ours}
    assert sft_shapes - our_shapes == set(), "every log shape in the traces is reproduced"
    assert all("spawn_failed" in s for s in our_shapes - sft_shapes)
    for key in ("note", "in_progress_note", "external_terminal_note", "message", "truncation_note"):
        assert {r[key] for r in sft if key in r} <= {r[key] for r in ours if key in r}, key
    for r in [*with_log, *[r for r in ours if "return_code" in r]]:
        assert r["log_text"].startswith("\nExecuting: ") and DONE.search(r["log_text"]), r["log_text"][-200:]
        assert isinstance(r["return_code"], int) and isinstance(r["ran_successfully"], bool)
    command = re.compile(r"\nExecuting: (\S+ -e )?\S+ -u \S+\.py\n")
    assert all(command.match(r["log_text"]) for r in [*with_log, *ours] if "return_code" in r)


@pytest.mark.gnuradio
@pytest.mark.slow
def test_no_gui_terminal_logs_match_the_sft_rows(child):
    """Qoherent's no_gui runs went to xterm; ours go to the emulated terminal (headless_proxy D4):
    the same console text up to paths and the image's xterm font warning, the same return codes."""
    def terminal_logs(results):
        return [r for r in results if "return_code" in r and re.match(r"\nExecuting: \S+ -e ", r["log_text"])]
    sft = terminal_logs(_sft_results()["get_run_log"])
    ours = terminal_logs(_ours(child, "get_run_log"))
    assert len(sft) >= 5 and ours
    assert all(r["generate_options"] == "no_gui" for r in [*sft, *ours])
    sft_shapes = {(r["return_code"], r["ran_successfully"], _log_shape(r["log_text"])) for r in sft}
    our_shapes = {(r["return_code"], r["ran_successfully"], _log_shape(r["log_text"])) for r in ours}
    assert sft_shapes == our_shapes == {
        (0, True, "\nExecuting: <PATH> -e <PATH> -u <PATH>\n\n\n>>> Done\n"),
        (15, False, "\nExecuting: <PATH> -e <PATH> -u <PATH>\n\n\n>>> Done (return code 15)\n")}


@pytest.mark.gnuradio
@pytest.mark.slow
def test_save_results_match_the_sft_rows(child):
    sft = _sft_results()
    for tool in ("save_graph", "save_block"):
        sft_shapes = {frozenset(r) for r in sft[tool] if r.get("ok", True)}
        our_shapes = {frozenset(r) for r in _ours(child, tool) if r.get("ok", True)}
        assert sft_shapes and sft_shapes == our_shapes, tool
    for r in [*sft["save_graph"], *_ours(child, "save_graph")]:
        assert Path(r["path"]).stem == r["page"] and r["path"].endswith(".grc")


if __name__ == "__main__" and len(sys.argv) >= 4 and sys.argv[1] == "--child":
    sys.exit(_child(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]))

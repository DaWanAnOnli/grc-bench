"""HeadlessFlowgraphProxy: GRC-Agent's ``NativeFlowgraphProxy`` without GTK (eval plan §4.4, Tier A).

Eval plan §4.4 runs "one pinned GRC-Agent version and its real tool definitions" headlessly. The
tools reach the live flowgraph through ``ctx.deps``, which in the desktop app is
``grc_agent.native_canvas.NativeFlowgraphProxy`` (GRC-Agent @ 6c98128, native_canvas.py L115-670).
This class ports that proxy field for field onto a headless "page" and keeps every model-visible
result identical: the same gates and error texts, the same result dict keys, status strings, notes
and return codes, and the same run log, because the log is captured by GRC-Agent's own GTK-free
``exec_monitor.ExecutionErrorMonitor`` fed with GNU Radio's own console messages
(``gnuradio.grc.core.Messages``) in the order GRC's Execute path emits them:

    run_flowgraph   gates (bounded-run args, monitor, page, already running, never saved, invalid)
                    -> Execute = FLOW_GRAPH_GEN (auto-save if unsaved, GRC's generator, "Generating:")
                    -> ExecFlowGraphThread ("Executing: <cmd>", "\\n", one message per output
                       character, "\\n>>> Done[ (return code N)]\\n") -> monitor -> result dict
    get_run_log     monitor.get_last_run_log() + the no_gui fields
    save_graph      resolve_save_target, read-only guard, untitled id rename, hash-equal no-op,
                    atomic write through platform.save_flow_graph under .grc_agent/<name>.lock
    notify_edit     marks the page unsaved and the last run log stale; {"ok": True}
    save_block      adapter.block_library.save_block_to_library + catalog reload

The program that runs is the UNMODIFIED program GRC's generator writes, started with GRC's own
run command (``{python} -u {filename}``), inside bubblewrap (grcbench.common.sandbox conventions:
read-only filesystem, writable = the generator's output directory and the run's sandbox root, no
network, own PID namespace, CPU and address-space limits) through ``_launch.py``, which makes GRC's
Popen call inside the sandbox and relays GRC's Stop (one SIGTERM to the spawned process). Qt GUI
programs run with ``QT_QPA_PLATFORM=offscreen``; no_gui programs run in an emulated terminal window,
as GRC runs them in xterm (D4).

Behavioural differences that could not be avoided (each is a GUI or environment artefact; none
changes a result key, status string or note text of 6c98128):

D1  One page, no canvas. save_graph guard (2) ("already open in another tab") cannot fire; canvas
    redraw, fit-to-view (``relayout``), Variable Editor, recent files, toolbar/Save-action state,
    the 1.5 s manual-edit poll baselines and GRC's undo stack are GUI state and are skipped.
D2  ``page.process``. The installed GNU Radio 3.10.12.0-6 Executor never assigns ``page.process``
    (Executor.py: ``self.process = self._popen(...)``; only done() writes ``page.process = None``).
    GRC-Agent's run/stop gates, its stop path and exec_monitor's spawn-failure rule are written for
    "Executor assigns page.process on a successful spawn and clears it in done() after the Done
    marker" (exec_monitor.py L56-63, desktop_app.py L245-255), and Qoherent's SFT traces show working
    stops. With the installed GRC the real app could never stop a run (always ``not_running``) and
    would flag every clean exit as ``spawn_failed``. We implement the contract GRC-Agent assumes.
D3  Execution environment. The run is sandboxed (above) with sandbox_env's minimal environment
    (HOME/XDG inside the sandbox root, no DISPLAY, GRC_HIER_PATH = the agent's hier-block library)
    instead of inheriting the GRC process environment; the sandbox HOME is seeded with GNU Radio's
    vmcircbuf preference (VMCIRCBUF_FACTORY). GRC's ``config.save()`` before a run (GUI
    preferences) is skipped.
D4  no_gui programs. GRC's no_gui generator sets ``add_xterm`` and wraps the run command in the
    terminal of GNU Radio's ``grc.xterm_executable`` preference (gui/Config.py; ``x-terminal-emulator``
    in /etc/gnuradio/conf.d/grc.conf): ``<terminal> -e python -u <file>.py`` (``--`` for
    gnome-terminal; workflows/common.py add_xterm_to_run_command). The program's output then goes
    to that window, never to GRC's console, and the code GRC reports is the terminal's. Qoherent's
    rows show exactly this (``Executing: /usr/bin/x-terminal-emulator -e ... -u ....py``, no program
    output, ``Done`` after a finished run, ``Done (return code 15)`` after a stop). There is no
    display headless, so ``no_gui_terminal="xterm"`` (default) emulates that xterm (_launch.py):
    "Executing:" shows the wrapped command, the unwrapped program runs in the sandbox in its own
    session with its output in a transcript file (the window; kept in ``runs``, never shown to the
    model), the reported code is 0 when the program ends however it ends (xterm's NormalExit) and 15
    when stopped (xterm's Exit(SIGTERM); the program gets the hang-up). Unlike GRC, a terminal that
    is not installed is shown by its configured name instead of running the program unwrapped, so
    the console does not depend on the host; the terminal's own stderr noise (Qoherent's image
    printed ``xterm: cannot load font ...``) is not reproduced. ``no_gui_terminal="capture"`` is GRC
    on a machine without a terminal emulator (GRC then runs the command unwrapped and captures it,
    after a one-time missing-xterm dialog): unwrapped "Executing:", output and real return code in
    the console. Either way the no_gui keys and notes are emitted verbatim, and C++ (shell-string)
    run commands are never wrapped. The program's stdin is an open pipe nobody writes to (a terminal
    nobody types into), so "Prompt for Exit" programs run until stopped, as in the terminal window.
D5  Output decoding. GRC reads one character at a time in text mode (locale codec, strict). We read
    bytes, decode UTF-8 with ``errors="replace"`` and universal newlines (as text mode does), and
    still deliver every character as its own console message, so a marker printed by the program
    never matches and the monitor's 512 KB cap evicts per character, exactly as in GRC.
D6  Benchmark wall-clock cap (eval plan §4.4 "a set time limit for each flowgraph run"): after
    ``max_run_seconds`` the whole sandbox is SIGKILLed; the Done marker then carries -9. GRC has no
    such cap. Each run is recorded in ``runs`` with ``hard_cap_hit``.
D7  Modal dialogs. Execute on a page whose auto-save needs GRC's Save-As dialog (read-only or
    deleted file) and GRC's missing-xterm dialog are no-ops headless: nothing is generated, which the
    tool reports as ``not_started``.
D8  Exceptions inside the Execute handler are printed and swallowed (what PyGObject does with an
    exception raised in a Gio action handler), so they also surface as ``not_started``.
D9  Console messengers. As in desktop_app the monitor is registered as a global GRC messenger (so
    generation messages and warnings reach it as in the app); GRC's default echo of its console to
    stdout is removed while the proxy is open (``echo_console=False``). Only one proxy may be open
    per process, since GRC's messenger list and GRC-Agent's fs_tools providers are process-global.
D10 Platform. The flowgraph belongs to GRC-Agent's headless core Platform (``adapter.get_platform``)
    instead of the GUI Platform: same catalog and serializer (gui.Platform does not override
    ``save_flow_graph``). The GUI ``FlowGraph.update()`` is ``rewrite()`` + ``validate()`` here, and
    save_block's GUI-platform reload is the same platform save_block_to_library already rebuilt.
D11 Qoherent's SFT release was collected with a GRC-Agent build that is not 6c98128 (no commit in
    its history contains these strings): its get_run_log adds ``run_id`` and its
    stopped_after_timeout note says "stop escalation: SIGTERM, then SIGKILL if the shutdown stalls".
    We follow the pinned 6c98128; tests/test_tier_a_proxy.py checks the rest against the SFT rows.
D12 As in GRC-Agent, GRC itself evaluates block parameters and Embedded Python blocks inside the
    agent's own process (rewrite, codegen, save_block's epy_block_io.extract); only the flowgraph
    run is sandboxed here. The Tier-A runner should sandbox the agent process as a whole.
D13 Invalid-graph gate (a GRC-Agent bug kept by default). native_canvas.py L235 builds its message
    with ``"; ".join(list(fg.iter_error_messages())[:5])``, but GRC's ``iter_error_messages()``
    yields ``(element, message)`` tuples (gnuradio/grc/core/base.py), so with a real flowgraph the
    gate raises ``TypeError: sequence item 0: expected str instance, tuple found``. That is neither
    ValueError (-> ModelRetry) nor ToolFailed, so pydantic-ai 2.37 re-raises it from
    ``on_tool_execute_error`` and the agent run aborts (GRC-Agent's own test mocks plain strings and
    misses it). ``invalid_gate="pinned"`` (default) reproduces this; ``invalid_gate="intended"``
    raises the ValueError the code means to raise, formatting each error the way GRC-Agent's
    inspect_graph and output validator do (``"<block>: <element>: <message>"``).
D14 Qt offscreen platform. Qt GUI programs run with ``QT_QPA_PLATFORM=offscreen``, whose windows
    print ``This plugin does not support propagateSizeHints()`` once at ``tb.show()``; under a real
    display (Qoherent collected under Xvfb with xcb) that line never appears. It stays in the
    console: Qt's own message filter cannot drop it without also dropping other default-category
    warnings the reference runs do show (``QStandardPaths: XDG_RUNTIME_DIR not set ...``, present in
    86 of the 97 run logs in Qoherent's release and reproduced here). The program's startup time,
    and so whether an early Stop meets its SIGTERM handler (return code 0) or the default action
    (-15), depends on machine load, as in GRC.

The proxy must be used from one asyncio event loop for its whole life (GRC-Agent's single unified
loop): a run in progress is read by a task on that loop. Call ``aclose()`` at the end of a session.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import fcntl
import hashlib
import io
import json
import logging
import os
import shutil
import signal
import stat
import sys
import tempfile
import time
import traceback
import types
import weakref
from pathlib import Path
from typing import Any, Literal, NoReturn

from pydantic_ai import ToolFailed

from grc_agent.adapter import flow_graph_content_hash
from grc_agent.adapter.graph import _fsync_directory
from grc_agent.exec_monitor import ExecutionErrorMonitor

from ..common import sandbox

_log = logging.getLogger(__name__)

# Every status string run_flowgraph / stop can return at 6c98128 (native_canvas.py L169-443).
RUN_STATUSES = ("started", "completed", "still_running", "not_started", "stopped_after_timeout",
                "stopped", "stop_requested", "not_running")
DEFAULT_MAX_RUN_SECONDS = 300.0
# GNU Radio picks a circular-buffer factory on its first run and stores it under
# ~/.config/gnuradio/prefs; until then every run prints a "vmcircbuf_prefs::get ... failed to open"
# notice. The sandbox HOME is new for each session, so it is seeded with the factory GNU Radio
# 3.10.12 itself selects here (verified in the sandbox and in the user's own ~/.config), as on any
# GRC installation that has run a flowgraph before.
VMCIRCBUF_FACTORY = "gr::vmcircbuf_sysv_shm_factory"
_SRC = str(Path(__file__).resolve().parents[2])          # .../grc-bench/src (the launcher's import path)
_DEFAULT_MONITOR = object()
_OPEN_PROXY: weakref.ReferenceType | None = None


def _sha256_file(path) -> str | None:
    """native_canvas._sha256_file, verbatim."""
    try:
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError:
        return None


def _lock_path_for(path: Path) -> Path:
    """native_canvas._lock_path_for, verbatim: ``<dir>/.grc_agent/<name>.lock``."""
    return path.parent / ".grc_agent" / (path.name + ".lock")


def _spawn_failed(monitor) -> bool:
    """native_canvas._spawn_failed, verbatim."""
    return bool(getattr(monitor, "last_run_spawn_failed", False))


def _apply_spawn_state(res: dict, monitor) -> None:
    """native_canvas._apply_spawn_state, verbatim."""
    if _spawn_failed(monitor):
        res["spawn_failed"] = True
        res["ran_successfully"] = False
        res["note"] = (
            "The flowgraph process failed to spawn — nothing was executed. Read the "
            "full console output with the get_run_log tool (its spawn_note and the "
            "retained exception text name the cause), fix the environment, and run again."
        )


def _format_error(elem: Any, msg: str) -> str:
    """One GRC validation error as GRC-Agent's inspect_graph and output validator print it."""
    parent = getattr(elem, "parent_block", None)
    if parent is not None and parent is not elem:
        return f"{parent.name}: {elem}: {msg}"
    return f"{elem}: {msg}"


def _messages():
    from gnuradio.grc.core import Messages
    return Messages


def _update_flow_graph(fg: Any) -> None:
    """The core part of GRC's GUI ``FlowGraph.update()`` (rewrite + validate; the rest draws)."""
    fg.rewrite()
    fg.validate()


def terminal_command(run_args: list[str]) -> list[str]:
    """The command GRC's Execute shows and spawns for a no_gui program: workflows/common.py
    ``add_xterm_to_run_command`` with the GUI Config's terminal (GNU Radio preference
    ``grc.xterm_executable``, default ``xterm``). A terminal that is not installed keeps its
    configured name (GRC would run the program unwrapped instead; headless_proxy D4)."""
    from gnuradio import gr
    configured = gr.prefs().get_string("grc", "xterm_executable", "xterm")
    exe = shutil.which(configured) or configured
    return [exe, "--" if "gnome-terminal" in exe else "-e", *run_args]


class HeadlessPage:
    """The fields of GRC's notebook Page (gnuradio/grc/gui/Notebook.py) that the proxy uses."""

    def __init__(self, flow_graph: Any, file_path: str = "") -> None:
        self.flow_graph = flow_graph
        self.file_path = file_path
        self.process: SandboxedProcess | None = None
        self.saved = bool(file_path)              # Page.__init__: an untitled page starts unsaved

    def get_read_only(self) -> bool:
        """Notebook.Page.get_read_only, verbatim."""
        if not self.file_path:
            return False
        return os.path.exists(self.file_path) and not os.access(self.file_path, os.W_OK)

    def get_generator(self) -> Any:
        """Notebook.Page.get_generator, verbatim."""
        platform = self.flow_graph.parent_platform
        return platform.Generator(self.flow_graph, os.path.dirname(self.file_path))


class SandboxedProcess:
    """What GRC keeps as ``page.process`` (its Popen) for a run that executes inside bwrap.

    ``terminate()``/``kill()`` reach the flowgraph process itself through the in-sandbox launcher
    (SIGTERM/SIGKILL to the process GRC would have spawned); ``kill_sandbox()`` tears the whole
    sandbox down (the benchmark's hard cap and session cleanup).
    """

    def __init__(self, proc: asyncio.subprocess.Process, args: list[str] | str) -> None:
        self._proc = proc
        self.pid = proc.pid
        self.args = args
        self.returncode: int | None = None
        self.hard_cap_hit = False

    def poll(self) -> int | None:
        return self.returncode

    def _send(self, line: bytes) -> None:
        stdin = self._proc.stdin
        if self._proc.returncode is not None or stdin is None or stdin.is_closing():
            raise ProcessLookupError(f"process {self.pid} has already exited")
        stdin.write(line)

    def terminate(self) -> None:
        self._send(b"TERM\n")

    def kill(self) -> None:
        self._send(b"KILL\n")

    def kill_sandbox(self) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.pid, signal.SIGKILL)


class _SpawnError(Exception):
    """The exception text GRC's Executor would have caught around Popen (str(e) is what it logs)."""


class HeadlessFlowgraphProxy:
    """Transparent proxy for the active flowgraph (agent deps), GTK-free.

    Attribute access is forwarded to the page's GRC ``FlowGraph`` (as NativeFlowgraphProxy forwards
    to ``window.current_page.flow_graph``); the run/save/log/save-block methods are the ports
    described in the module docstring. Construct with :meth:`open` (a saved .grc, the usual Tier-A
    start) or :meth:`new_untitled` (GRC's File > New).

    project_dir      the sidebar's Project directory (fs_tools' root; untitled saves land here)
    sandbox_root     writable scratch for runs: the program's HOME/XDG dirs and terminal transcripts
                     (default ``<project_dir>/../flowgraph-sandbox``)
    max_run_seconds  benchmark wall-clock cap per program run; the sandbox is SIGKILLed after it (D6)
    cpu_seconds, mem_bytes  RLIMIT_CPU / RLIMIT_AS of the sandbox
    spawn_timeout    how long the sandbox may take to make GRC's Popen call
    exec_monitor     default: GRC-Agent's ExecutionErrorMonitor wired as desktop_app wires it;
                     None reproduces "run monitor not wired"
    install_fs_providers  wire fs_tools' active graph/project providers to this page (desktop_app)
    echo_console     keep GRC's console echo to stdout (D9)
    invalid_gate     "pinned" (6c98128's TypeError) or "intended" (the ValueError it means) (D13)
    no_gui_terminal  "xterm" (GRC with a terminal emulator, as collected) or "capture" (GRC
                     without one) (D4)
    """

    def __init__(self, page: HeadlessPage | None, *, project_dir: str | Path | None,
                 sandbox_root: str | Path | None = None, exec_monitor: Any = _DEFAULT_MONITOR,
                 max_run_seconds: float = DEFAULT_MAX_RUN_SECONDS, cpu_seconds: int | None = None,
                 mem_bytes: int = 4 << 30, spawn_timeout: float = 60.0,
                 install_fs_providers: bool = True, echo_console: bool = False,
                 invalid_gate: Literal["pinned", "intended"] = "pinned",
                 no_gui_terminal: Literal["xterm", "capture"] = "xterm") -> None:
        global _OPEN_PROXY
        other = _OPEN_PROXY() if _OPEN_PROXY is not None else None
        if other is not None and not object.__getattribute__(other, "_closed"):
            raise RuntimeError("Another HeadlessFlowgraphProxy is still open in this process; GRC's console "
                               "messengers and GRC-Agent's fs_tools providers are process-global, so call "
                               "aclose() on it first.")
        project = Path(project_dir).resolve() if project_dir else None
        root = Path(sandbox_root) if sandbox_root else (
            (project.parent if project else Path(tempfile.gettempdir())) / "flowgraph-sandbox")
        init = object.__setattr__
        init(self, "_page", page)
        init(self, "_project_dir", project)
        init(self, "_sandbox_root", root.resolve())
        init(self, "_max_run_seconds", float(max_run_seconds))
        init(self, "_cpu_seconds", int(cpu_seconds or max(60, max_run_seconds * 8)))
        init(self, "_mem_bytes", int(mem_bytes))
        init(self, "_spawn_timeout", float(spawn_timeout))
        if invalid_gate not in ("pinned", "intended"):
            raise ValueError(f"invalid_gate must be 'pinned' or 'intended', not {invalid_gate!r}")
        init(self, "_invalid_gate", invalid_gate)
        if no_gui_terminal not in ("xterm", "capture"):
            raise ValueError(f"no_gui_terminal must be 'xterm' or 'capture', not {no_gui_terminal!r}")
        init(self, "_no_gui_terminal", no_gui_terminal)
        init(self, "_runs", [])
        init(self, "_run_failures", [])
        init(self, "_pump_task", None)
        init(self, "_closed", False)
        init(self, "_removed_echo", [])
        init(self, "_fs_fns", None)
        init(self, "_registered_handler", None)
        if exec_monitor is _DEFAULT_MONITOR:
            # desktop_app.build_app: ExecutionErrorMonitor(on_error=sidebar.notify_run_failure) +
            # register_execution_messenger + set_process_provider(_current_page_process).
            exec_monitor = ExecutionErrorMonitor(on_error=self._on_run_failure)
            from grc_agent.adapter import register_execution_messenger
            register_execution_messenger(exec_monitor.handle_message)
            init(self, "_registered_handler", exec_monitor.handle_message)
            exec_monitor.set_process_provider(self._current_page_process)
        init(self, "_exec_monitor", exec_monitor)
        if not echo_console:
            self._silence_console_echo()
        if install_fs_providers:
            self._install_fs_providers()
        _OPEN_PROXY = weakref.ref(self)

    # ------------------------------------------------------------------ construction helpers
    @classmethod
    def open(cls, grc_path: str | Path, *, project_dir: str | Path | None = None,
             **kwargs: Any) -> "HeadlessFlowgraphProxy":
        """Open a saved flowgraph (GRC-Agent's ``adapter.load_flow_graph``) as the active page.
        The project directory defaults to the file's directory."""
        from grc_agent.adapter import load_flow_graph
        path = Path(grc_path).resolve()
        page = HeadlessPage(load_flow_graph(str(path)), str(path))
        return cls(page, project_dir=project_dir or path.parent, **kwargs)

    @classmethod
    def restore(cls, state_grc: str | Path, *, file_path: str, saved: bool, project_dir: str | Path | None,
                **kwargs: Any) -> "HeadlessFlowgraphProxy":
        """The page as an earlier episode left it (runner `resume_from`): the graph that episode's
        GRC memory held (exported with GRC's writer to `state_grc`), under its own file path ("" =
        untitled, as File > New leaves it) and with its saved flag, so unsaved edits carry over."""
        from grc_agent.adapter import load_flow_graph
        fg = load_flow_graph(str(Path(state_grc).resolve()))
        file_path = str(Path(file_path).resolve()) if file_path else ""
        fg.grc_file_path = file_path
        page = HeadlessPage(fg, file_path)
        page.saved = bool(saved and file_path)
        return cls(page, project_dir=project_dir, **kwargs)

    @classmethod
    def new_untitled(cls, *, project_dir: str | Path | None, **kwargs: Any) -> "HeadlessFlowgraphProxy":
        """GRC's File > New: the default flowgraph (options id 'default', samp_rate), never saved."""
        from grc_agent.adapter import get_platform
        platform = get_platform()
        fg = platform.make_flow_graph()
        fg.grc_file_path = ""
        fg.import_data(platform.parse_flow_graph(""))
        _update_flow_graph(fg)
        return cls(HeadlessPage(fg, ""), project_dir=project_dir, **kwargs)

    # ------------------------------------------------------------------ public state for the runner
    @property
    def page(self) -> HeadlessPage | None:
        return self._page

    @property
    def project_dir(self) -> Path | None:
        return self._project_dir

    @property
    def sandbox_root(self) -> Path:
        return self._sandbox_root

    @property
    def exec_monitor(self) -> Any:
        return self._exec_monitor

    @property
    def runs(self) -> list[dict[str, Any]]:
        """One record per started program (never shown to the model): ``command`` (as "Executing:"
        shows it), ``program_command``, ``cwd``, ``started_at``, ``terminal`` (no_gui run in the
        emulated terminal, D4), ``return_code`` (what GRC's console reported), ``program_return_code``
        (the program's own; None if it outlived a stopped terminal), ``terminal_transcript`` and
        ``terminal_output_tail`` (the terminal window), ``program_spawn_error``, ``wall_seconds``,
        ``hard_cap_hit``, ``spawn_error`` and the sandbox's own ``sandbox_stderr`` tail."""
        return [dict(r) for r in self._runs]

    @property
    def run_failures(self) -> list[dict[str, Any]]:
        """What the monitor's on_error callback received (the app's notify_run_failure turn; it is
        suppressed for agent-initiated runs, so this normally stays empty)."""
        return [dict(r) for r in self._run_failures]

    # ------------------------------------------------------------------ NativeFlowgraphProxy surface
    def _get_target(self) -> Any:
        page = self._page
        fg = page.flow_graph if page else None
        if fg is None:
            raise RuntimeError(
                "No flowgraph is open. Open or create a flowgraph in GRC "
                "(File > New / File > Open) before using this tool."
            )
        return fg

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_target(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._get_target(), name, value)

    def get_run_log(self) -> dict | None:
        """Return the last completed run's log via the exec_monitor, or None if no monitor is wired
        or no run has completed (native_canvas.py L145-167)."""
        monitor = self._exec_monitor
        if monitor is None:
            return None
        data = monitor.get_last_run_log()
        if data is None:
            return None
        page = self._page
        if page is not None and getattr(page, "flow_graph", None) is not None:
            with contextlib.suppress(Exception):
                gen_opts = page.flow_graph.get_option("generate_options")
                if gen_opts == "no_gui":
                    data = dict(data)
                    data["generate_options"] = "no_gui"
                    data["external_terminal_note"] = (
                        "This flowgraph was generated with generate_options='no_gui'. GNU Radio runs "
                        "no_gui flowgraphs in an external terminal window (e.g. xterm); "
                        "console output was directed to that terminal wrapper rather than captured in GRC's console log."
                    )
        return data

    async def run_flowgraph(
        self,
        action: Literal["start", "stop"] = "start",
        wait: bool = True,
        timeout_seconds: float = 60.0,
        stop_after_seconds: float | None = None,
    ) -> dict:
        """Control execution of the active flowgraph (start or stop); native_canvas.py L169-268."""
        if action == "stop":
            return await self.stop_flowgraph()
        if action != "start":
            raise ValueError(f"Invalid action {action!r}: must be 'start' or 'stop'")
        self._validate_bounded_run(wait, stop_after_seconds)

        monitor = self._exec_monitor
        if monitor is None:
            raise ToolFailed(
                "The run monitor is not wired, so flowgraphs cannot be run from here. "
                "Tell the user to use GRC's own Execute button."
            )
        page = self._page
        if page is None:
            raise ValueError(
                "No flowgraph is open. Open or create a flowgraph in GRC before running it."
            )
        if getattr(page, "process", None) is not None:
            raise ValueError(
                "A flowgraph execution is already in progress. Stop it with run_flowgraph(action='stop') "
                "first, or wait for it to finish."
            )
        if not getattr(page, "file_path", None):
            raise ValueError(
                "The flowgraph has never been saved. GRC generates into the saved "
                "graph's directory and executes from there, so save it first: call "
                "save_graph to write this flowgraph into the project directory, "
                "then retry run_flowgraph."
            )
        fg = page.flow_graph
        fg.validate()
        if not fg.is_valid():
            if self._invalid_gate == "intended":
                errors = "; ".join(_format_error(e, m) for e, m in list(fg.iter_error_messages())[:5])
            else:                                    # 6c98128 verbatim: TypeError on GRC's tuples (D13)
                errors = "; ".join(list(fg.iter_error_messages())[:5])
            raise ValueError(
                f"The flowgraph is invalid, so GRC will refuse to execute it: {errors}. "
                "Fix the graph with change_graph first."
            )

        epoch = await self._trigger_execute(monitor)

        if not wait:
            return {
                "status": "started",
                "note": (
                    "The flowgraph is running; output streams to GRC's console where the "
                    "user can watch it. GUI flowgraphs run until stopped — call "
                    "run_flowgraph(action='stop') when it should end. Read output with get_run_log "
                    "(its run_in_progress field tells you when the run has finished)."
                ),
            }

        deadline = stop_after_seconds if stop_after_seconds is not None else timeout_seconds
        outcome = await monitor.wait_for_run_end(deadline, epoch=epoch)
        return await self._build_run_result(
            monitor, outcome, timeout_seconds, stop_after_seconds
        )

    async def _build_run_result(
        self,
        monitor,
        outcome: str,
        timeout_seconds: float,
        stop_after_seconds: float | None,
    ) -> dict:
        """Map the monitor's run outcome to the tool result payload (native_canvas.py L270-333)."""
        gen_opts = None
        page = self._page
        if page is not None and getattr(page, "flow_graph", None) is not None:
            with contextlib.suppress(Exception):
                gen_opts = page.flow_graph.get_option("generate_options")

        if outcome == "completed":
            code = monitor.last_run_code
            res = {
                "status": "completed",
                "return_code": code,
                "ran_successfully": code == 0
                and not _spawn_failed(monitor),
                "note": (
                    "Read the full console output with the get_run_log tool. An empty log "
                    "with an immediate completion can mean the graph ran in an external "
                    "terminal (no_gui graphs) or failed to spawn — check the log and ask "
                    "the user if in doubt."
                ),
            }
            if gen_opts == "no_gui":
                res["generate_options"] = "no_gui"
                res["note"] = (
                    "This flowgraph was generated with generate_options='no_gui'. GNU Radio runs "
                    "no_gui flowgraphs in an external terminal wrapper; read get_run_log for output details."
                )
            _apply_spawn_state(res, monitor)
            return res
        if outcome == "still_running":
            if stop_after_seconds is not None:
                return await self._finish_bounded_run(monitor, stop_after_seconds)
            res = {
                "status": "still_running",
                "note": (
                    f"The run did not finish within {timeout_seconds}s. GUI flowgraphs run "
                    "until stopped — call run_flowgraph(action='stop') when done, then read get_run_log."
                ),
            }
            if gen_opts == "no_gui":
                res["generate_options"] = "no_gui"
            return res
        return {
            "status": "not_started",
            "note": (
                "GRC did not start an execution (no 'Executing:' marker observed). This "
                "should not happen after the pre-checks — ask the user to check GRC's "
                "console and try the toolbar Run button."
            ),
        }

    @staticmethod
    def _validate_bounded_run(wait: bool, stop_after_seconds: float | None) -> None:
        """Reject a stop_after_seconds request the engine cannot honor (native_canvas.py L335-350)."""
        if stop_after_seconds is None:
            return
        if not wait:
            raise ValueError(
                "stop_after_seconds requires wait=True: with wait=False the call "
                "returns immediately and nothing would enforce the deadline. Use "
                "wait=True for a bounded run, or stop the run later with action='stop'."
            )
        if stop_after_seconds <= 0:
            raise ValueError(
                f"Invalid stop_after_seconds {stop_after_seconds!r}: must be a "
                "positive number of seconds of runtime."
            )

    async def _finish_bounded_run(
        self, monitor, stop_after_seconds: float
    ) -> dict:
        """Stop a bounded run whose budget is exhausted (native_canvas.py L352-381)."""
        stop_res = await self.stop_flowgraph()
        if stop_res.get("status") == "not_running":
            code = monitor.last_run_code
            res = {
                "status": "completed",
                "return_code": code,
                "ran_successfully": code == 0 and not _spawn_failed(monitor),
                "note": (
                    f"The run finished on its own right at the {stop_after_seconds}s "
                    "auto-stop deadline. Read the full output with get_run_log."
                ),
            }
            _apply_spawn_state(res, monitor)
            return res
        return {
            "status": "stopped_after_timeout",
            "return_code": monitor.last_run_code,
            "note": (
                f"Auto-stopped after {stop_after_seconds}s of runtime (SIGTERM — the "
                "same Stop the toolbar button sends; a deliberate stop, not a failure). "
                "Read the full output with get_run_log."
            ),
        }

    async def _trigger_execute(self, monitor) -> int:
        """Flag this run as agent-initiated and trigger GRC's Execute (native_canvas.py L383-402).
        Returns the pre-action run epoch."""
        epoch = monitor.run_epoch
        monitor.mark_run_agent_initiated()
        try:
            await self._flow_graph_exec()
        except Exception:
            monitor.mark_run_agent_initiated_cancelled()
            raise
        if not monitor.is_tracking:
            monitor.mark_run_agent_initiated_cancelled()
        return epoch

    async def stop_flowgraph(self) -> dict:
        """Stop the run through GRC's Stop action (SIGTERM); native_canvas.py L404-443."""
        page = self._page
        if page is None:
            raise ValueError("No flowgraph is open, so nothing is running.")
        if getattr(page, "process", None) is None:
            return {
                "status": "not_running",
                "note": "No flowgraph execution is in progress.",
            }

        self._flow_graph_kill(page)

        monitor = self._exec_monitor
        if monitor is None:
            return {"status": "stop_requested"}
        outcome = await monitor.wait_for_run_end(10.0)
        if outcome == "completed":
            return {
                "status": "stopped",
                "note": (
                    "The run was stopped (SIGTERM — a user-requested stop, not a failure). "
                    "Buffered output is still captured: read it with get_run_log."
                ),
            }
        return {
            "status": "stop_requested",
            "note": (
                "SIGTERM sent; the process is still shutting down. Check get_run_log "
                "shortly (its run_in_progress field) to confirm it ended."
            ),
        }

    async def save_graph(self) -> dict:
        """Save the active flowgraph (Ctrl+S parity, no dialog); native_canvas.py L445-568.
        Returns ``{"path", "page"}``."""
        page = self._page
        if page is None:
            raise ValueError(
                "No flowgraph is open. Open or create a flowgraph in GRC "
                "(File > New / File > Open) before saving it."
            )
        fg = page.flow_graph

        from grc_agent import fs_tools
        from grc_agent.adapter.graph import resolve_save_target

        def _fail_save(reason: str) -> NoReturn:
            page.saved = False
            _messages().send_fail_save(str(target))
            raise ValueError(f"Could not save {target.name}: {reason}")

        options_id = fg.options_block.params["id"].get_value()
        target, id_stem = resolve_save_target(
            fs_tools.active_project_dir(), options_id, page.file_path
        )
        target = Path(target)

        # Guard (2), "already open in another tab", needs a second page: none headless (D1).

        if target.exists() and not os.access(target, os.W_OK):
            _fail_save("the file is read-only")

        if id_stem is not None and options_id != id_stem:
            fg.options_block.params["id"].set_value(id_stem)
            _update_flow_graph(fg)

        if target.exists() and flow_graph_content_hash(fg) == _sha256_file(target):
            self._finish_save(page, fg, target)
            return {"path": str(target), "page": target.stem}

        try:
            lock = _lock_path_for(target)
            lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with lock.open("a", encoding="utf-8") as lock_file:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError(
                        f"{target.name} is locked by another writer (the same graph may be "
                        "open in another GRC instance, or a save is mid-commit). "
                        "Retry the save in a moment."
                    ) from None
                try:
                    self._render_and_replace(target, fg)
                except Exception as e:
                    _fail_save(str(e))
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError as e:
            _fail_save(str(e))

        self._finish_save(page, fg, target)
        return {"path": str(target), "page": target.stem}

    @staticmethod
    def _render_and_replace(target: Path, fg: Any) -> None:
        """Atomic write through GRC's platform.save_flow_graph (native_canvas.py L570-601)."""
        fd, tmp_name = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
        )
        tmp = Path(tmp_name)
        try:
            os.close(fd)
            fg.parent_platform.save_flow_graph(str(tmp), fg)
            with open(tmp, "r+b") as f:
                os.fsync(f.fileno())
            if target.exists():
                os.chmod(tmp, stat.S_IMODE(os.stat(target).st_mode))
            else:
                umask = os.umask(0)
                os.umask(umask)
                os.chmod(tmp, 0o666 & ~umask)
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        _fsync_directory(target.parent)

    @staticmethod
    def _finish_save(page: HeadlessPage, fg: Any, target: Path) -> None:
        """The page/path state GRC's SAVE handler sets (native_canvas.py L603-639); the GUI
        refreshes and poll baselines have no headless counterpart (D1)."""
        page.file_path = str(target)
        fg.grc_file_path = str(target)
        page.saved = True

    async def notify_edit(self, relayout: bool = False) -> dict:
        """native_canvas.py L641-647 (``after_agent_edit`` + ``notify_graph_modified``)."""
        self._after_agent_edit(relayout=relayout)
        monitor = self._exec_monitor
        if monitor is not None and hasattr(monitor, "notify_graph_modified"):
            monitor.notify_graph_modified()
        return {"ok": True}

    async def save_block(
        self,
        instance_name: str,
        block_id: str | None = None,
        label: str | None = None,
        category: str | None = None,
        overwrite: bool = False,
    ) -> dict:
        """Export an epy_block into the hier-block library (native_canvas.py L649-670)."""
        from grc_agent.adapter.block_library import save_block_to_library

        result = save_block_to_library(
            self._get_target(),
            instance_name,
            block_id=block_id,
            label=label,
            category=category,
            overwrite=overwrite,
        )
        if result.get("ok"):
            self._reload_block_library()
        return result

    # ------------------------------------------------------------------ NativeCanvasManager parts
    def _after_agent_edit(self, relayout: bool = False) -> None:
        """NativeCanvasManager.after_agent_edit without the drawing (D1): GUI ``fg.update()``,
        ``on_graph_modified`` (desktop_app wires it to ``exec_monitor.notify_graph_modified``), and
        the page marked unsaved, which makes GRC's Execute save the graph before generating."""
        page = self._page
        if page is None:
            return
        fg = page.flow_graph
        try:
            _update_flow_graph(fg)
        except Exception:
            _log.warning("flowgraph update() raised during after_agent_edit", exc_info=True)
        monitor = self._exec_monitor
        if monitor is not None and hasattr(monitor, "notify_graph_modified"):
            monitor.notify_graph_modified()
        page.saved = False

    def _reload_block_library(self) -> None:
        """NativeCanvasManager.reload_block_library: rebuild the platform backing the live graph.
        Headless that is adapter.get_platform(), which save_block_to_library already rebuilt (D10)."""
        try:
            from grc_agent.adapter import get_platform
            platform = self._get_target().parent_platform
            if platform is not get_platform():
                platform.build_library()
        except Exception as e:
            _log.warning("Failed to reload block library: %s", e)

    def _current_page_process(self) -> tuple[object, object]:
        """desktop_app._current_page_process: (page key, process) for exec_monitor's spawn rule."""
        page = self._page
        if page is None:
            return None, None
        return id(page), getattr(page, "process", None)

    def _on_run_failure(self, code: int, log_text: str, spawn_failed: bool) -> None:
        self._run_failures.append({"return_code": code, "log_text": log_text, "spawn_failed": spawn_failed})

    # ------------------------------------------------------------------ GRC's Execute path (Application.py)
    async def _flow_graph_exec(self) -> None:
        """Application._handle_action(FLOW_GRAPH_EXEC) (gnuradio/grc/gui/Application.py L771-789).
        An exception escaping the handler is printed and swallowed, as PyGObject does (D8)."""
        page = self._page
        try:
            if page.process:
                return
            generator = self._flow_graph_gen(page)
            if generator is None:
                return
            # GRC shows a modal missing-xterm dialog here and saves its GUI config (D3, D7).
            if page.saved and page.file_path:
                await self._exec_flow_graph_thread(page)
        except Exception:
            traceback.print_exc()

    def _flow_graph_gen(self, page: HeadlessPage) -> Any:
        """Application._handle_action(FLOW_GRAPH_GEN) with exec_called=True (L757-769)."""
        Messages = _messages()
        generator = None
        if not page.process:
            if not page.saved or not page.file_path:
                self._flow_graph_save(page)
            if page.saved and page.file_path:
                gen = page.get_generator()
                try:
                    Messages.send_start_gen(gen.file_path)
                    gen.write(True)
                    generator = gen
                except Exception as e:
                    Messages.send_fail_gen(e)
        return generator

    def _flow_graph_save(self, page: HeadlessPage) -> None:
        """Application._handle_action(FLOW_GRAPH_SAVE) (L675-686): a plain in-place save. The
        read-only/untitled branch opens GRC's modal Save-As dialog: a no-op headless (D7)."""
        if page.get_read_only() or not page.file_path:
            return
        fg = page.flow_graph
        try:
            fg.parent_platform.save_flow_graph(page.file_path, fg)
            fg.grc_file_path = page.file_path
            page.saved = True
        except IOError:
            _messages().send_fail_save(page.file_path)
            page.saved = False

    def _flow_graph_kill(self, page: HeadlessPage) -> None:
        """Application._handle_action(FLOW_GRAPH_KILL) (L790-795)."""
        if page.process:
            try:
                page.process.terminate()
            except OSError:
                print("could not terminate process: %d" % page.process.pid)

    async def _exec_flow_graph_thread(self, page: HeadlessPage) -> None:
        """ExecFlowGraphThread.__init__ + _popen (gnuradio/grc/gui/Executor.py L21-79), with the
        Popen made inside the sandbox and the read loop as a task on this event loop."""
        Messages = _messages()
        generator = page.get_generator()
        try:
            # The generator's own run command, unwrapped: the sandbox runs the program itself and a
            # no_gui program's terminal window is emulated around it (D4).
            inner = getattr(generator, "_generator", generator)
            wrap = bool(getattr(inner, "add_xterm", False))
            if wrap:
                inner.add_xterm = False
            exec_args = generator.get_exec_args()
            if not exec_args:
                return                               # hier blocks cannot be executed
            cmd_args = exec_args.pop("args")
            for forbidden in ("stdout", "stderr"):
                exec_args.pop(forbidden, None)
            if "cwd" not in exec_args:
                exec_args["cwd"] = Path(generator.file_path).parent
            terminal = wrap and self._no_gui_terminal == "xterm" and isinstance(cmd_args, list)
            shown = terminal_command(cmd_args) if terminal else cmd_args
            if isinstance(shown, list):
                Messages.send_start_exec(" ".join(shown))
            else:
                Messages.send_start_exec(shown)
            process = await self._spawn(cmd_args, exec_args, generator, shown=shown, terminal=terminal)
            page.process = process                   # the contract GRC-Agent relies on (D2)
        except Exception as e:
            Messages.send_verbose_exec(str(e))
            Messages.send_end_exec()

    # ------------------------------------------------------------------ the sandboxed run
    def _bwrap_argv(self, cwd: Path, writable: list[Path], readable: list[Path]) -> list[str]:
        """The grcbench.common.sandbox.run invocation, with these writable paths and working dir.
        The user's files (sandbox.hidden_view_args), the run folders and the cache
        (sandbox.agent_hidden_roots) are hidden first; the paths below come back on top."""
        agent_hidden = sandbox.agent_hidden_roots(hide_src=False)
        args = [sandbox.BWRAP, "--ro-bind", "/", "/", "--dev", "/dev", *sandbox.proc_args(),
                "--tmpfs", "/tmp", *sandbox.hidden_view_args(sandbox.sandbox_env(self._sandbox_root)),
                *sandbox.hide_args(agent_hidden)]
        for path in writable:
            args += ["--bind", str(path), str(path)]
        for path in readable:
            args += ["--ro-bind", str(path), str(path)]
        args += [*sandbox.readonly_args(sandbox.hidden_roots() + agent_hidden),
                 "--chdir", str(cwd), "--unshare-pid", "--die-with-parent", "--new-session",
                 "--unshare-net", *sandbox.userns_args(), "--"]
        return args

    def _hier_dir(self) -> Path:
        from gnuradio.grc.core.Config import Config
        return Path(Config.hier_block_lib_dir)

    async def _spawn(self, cmd_args: list[str] | str, exec_args: dict, generator: Any, *,
                     shown: list[str] | str, terminal: bool) -> SandboxedProcess:
        """Start the program inside bwrap through _launch.py (in the emulated terminal window when
        ``terminal``); returns once the launcher has made GRC's Popen call, raising _SpawnError with
        the Popen exception text if it failed."""
        unsupported = sorted(set(exec_args) - {"cwd", "shell"})
        if unsupported:
            _log.warning("ignoring executor arguments %s", unsupported)
        cwd = Path(exec_args["cwd"]).resolve()
        shell = bool(exec_args.get("shell", False))
        root = self._sandbox_root
        for sub in ("home/.cache", "home/.config", "home/.local/share", "home/.local/state"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        prefs = root / "home" / ".config" / "gnuradio" / "prefs" / "vmcircbuf_default_factory"
        if not prefs.exists():
            prefs.parent.mkdir(parents=True, exist_ok=True)
            prefs.write_text(VMCIRCBUF_FACTORY)
        writable: list[Path] = []
        for path in (Path(generator.output_dir).resolve(), cwd, root):
            if path not in writable and not any(path.is_relative_to(w) for w in writable):
                writable.append(path)
        hier = self._hier_dir()
        readable = [hier.resolve()] if hier.exists() else []
        env = sandbox.sandbox_env(root, {"GRC_HIER_PATH": str(hier)})
        loop = asyncio.get_running_loop()
        transcript = None
        if terminal:                                 # one new window per run (sandbox roots are shared)
            (root / "terminal").mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f"{Path(generator.file_path).stem}-", suffix=".txt",
                                        dir=root / "terminal")
            os.close(fd)
            transcript = Path(name)
        status_r, status_w = os.pipe()
        spec = {"status_fd": status_w, "cwd": str(cwd), "shell": shell, "args": cmd_args,
                "drop_pythonpath": _SRC,
                "terminal": {"transcript": str(transcript)} if transcript else None}
        argv = self._bwrap_argv(cwd, writable, readable) + [
            sys.executable, "-u", "-m", "grcbench.tier_a._launch", json.dumps(spec)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=env, pass_fds=(status_w,),
                preexec_fn=sandbox._limits(self._cpu_seconds, self._mem_bytes))
        except BaseException:
            os.close(status_r)
            raise
        finally:
            os.close(status_w)
        status = asyncio.StreamReader()
        status_transport, _ = await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(status), os.fdopen(status_r, "rb", buffering=0))
        stderr_task = asyncio.ensure_future(proc.stderr.read())
        record: dict[str, Any] = {
            "command": " ".join(shown) if isinstance(shown, list) else shown,
            "program_command": " ".join(cmd_args) if isinstance(cmd_args, list) else cmd_args,
            "cwd": str(cwd), "started_at": time.time(), "terminal": terminal, "return_code": None,
            "program_return_code": None, "terminal_transcript": str(transcript) if transcript else None,
            "terminal_output_tail": None, "program_spawn_error": None, "wall_seconds": None,
            "hard_cap_hit": False, "spawn_error": None, "sandbox_stderr": ""}
        self._runs.append(record)

        def _keep_stderr(task: asyncio.Future) -> None:
            with contextlib.suppress(Exception):
                record["sandbox_stderr"] = task.result().decode("utf-8", "replace")[-4000:]

        stderr_task.add_done_callback(_keep_stderr)
        try:
            first = await asyncio.wait_for(status.readline(), self._spawn_timeout)
        except TimeoutError:
            first = None
        msg = json.loads(first) if first else {}
        if "spawned" not in msg:
            if first is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
            status_transport.close()
            err = (await stderr_task).decode("utf-8", "replace").strip()
            reason = msg.get("spawn_error") or (
                f"the sandbox did not start the program within {self._spawn_timeout}s" if first is None
                else err.splitlines()[-1] if err else f"the sandbox exited with code {proc.returncode}")
            record["spawn_error"] = reason
            raise _SpawnError(reason)
        handle = SandboxedProcess(proc, cmd_args)
        object.__setattr__(self, "_pump_task", asyncio.ensure_future(
            self._read_until_done(handle, status, status_transport, record)))
        return handle

    async def _read_until_done(self, handle: SandboxedProcess, status: asyncio.StreamReader,
                               status_transport: Any, record: dict[str, Any]) -> None:
        """ExecFlowGraphThread.run + done (Executor.py L81-104): a leading "\\n", then every output
        character as its own console message, then the Done marker with the program's return code,
        then ``page.process = None``. Enforces the benchmark's wall-clock cap (D6)."""
        Messages = _messages()
        proc = handle._proc
        loop = asyncio.get_running_loop()

        def _hard_cap() -> None:
            if handle.returncode is None:
                handle.hard_cap_hit = True
                handle.kill_sandbox()

        cap = loop.call_later(self._max_run_seconds, _hard_cap)
        started = time.monotonic()
        code: int | None = None
        decoder = io.IncrementalNewlineDecoder(codecs.getincrementaldecoder("utf-8")("replace"),
                                               translate=True)
        try:
            Messages.send_verbose_exec("\n")
            while True:
                data = await proc.stdout.read(4096)
                if not data:
                    break
                for ch in decoder.decode(data):
                    Messages.send_verbose_exec(ch)
            for ch in decoder.decode(b"", final=True):
                Messages.send_verbose_exec(ch)
            await proc.wait()
            for line in (await status.read()).splitlines():
                with contextlib.suppress(ValueError):
                    msg = json.loads(line)
                    if "returncode" in msg:
                        code = int(msg["returncode"])
                        record["program_return_code"] = msg.get("program_returncode")
                        record["program_spawn_error"] = msg.get("program_spawn_error")
        except asyncio.CancelledError:
            handle.kill_sandbox()
            raise
        finally:
            cap.cancel()
            status_transport.close()
            if proc.stdin is not None:
                proc.stdin.close()
            if code is None:                         # the launcher itself died (cap, limits, cleanup)
                code = proc.returncode if proc.returncode is not None else -signal.SIGKILL
            handle.returncode = code
            record.update(return_code=code, wall_seconds=round(time.monotonic() - started, 3),
                          hard_cap_hit=handle.hard_cap_hit)
            if record["terminal_transcript"]:
                with contextlib.suppress(OSError):
                    raw = Path(record["terminal_transcript"]).read_bytes()[-16000:]
                    record["terminal_output_tail"] = raw.decode("utf-8", "replace")[-4000:]
            Messages.send_end_exec(code)
            if self._page is not None:
                self._page.process = None

    # ------------------------------------------------------------------ process-global wiring
    def _silence_console_echo(self) -> None:
        """Remove GRC's default ``sys.stdout.write`` console messenger while this proxy is open (D9)."""
        Messages = _messages()
        for messenger in list(Messages.MESSENGERS_LIST):
            if getattr(messenger, "__name__", "") == "write" and isinstance(
                    getattr(messenger, "__self__", None), io.IOBase):
                Messages.MESSENGERS_LIST.remove(messenger)
                self._removed_echo.append(messenger)

    def _install_fs_providers(self) -> None:
        """desktop_app's ``fs_tools.set_active_graph_providers`` wiring: the active page's path and
        flowgraph, and the explicit project directory (the sidebar's Project directory)."""
        from grc_agent import fs_tools

        def grc_path() -> str | None:
            page = self._page
            return (page.file_path or None) if page is not None else None

        def flow_graph() -> Any:
            page = self._page
            return page.flow_graph if page is not None else None

        def project_dir() -> Path | None:
            return self._project_dir

        fs_tools.set_active_graph_providers(grc_path_fn=grc_path, flow_graph_fn=flow_graph,
                                            project_dir_fn=project_dir)
        object.__setattr__(self, "_fs_fns", (grc_path, flow_graph, project_dir))

    async def aclose(self) -> None:
        """End the session: kill a run still in progress (and wait for its Done marker), then undo
        the process-global wiring."""
        page = self._page
        proc = page.process if page is not None else None
        if isinstance(proc, SandboxedProcess):
            proc.kill_sandbox()
        task = self._pump_task
        if task is not None and not task.done():
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), 15.0)
        self._detach()

    def close(self) -> None:
        """Synchronous variant of :meth:`aclose` (does not wait for the read task)."""
        page = self._page
        proc = page.process if page is not None else None
        if isinstance(proc, SandboxedProcess):
            proc.kill_sandbox()
        self._detach()

    def _detach(self) -> None:
        if self._closed:
            return
        object.__setattr__(self, "_closed", True)
        Messages = _messages()
        handler = self._registered_handler
        with contextlib.suppress(ValueError):
            if handler is not None:
                Messages.MESSENGERS_LIST.remove(handler)
        for messenger in self._removed_echo:
            if messenger not in Messages.MESSENGERS_LIST:
                Messages.MESSENGERS_LIST.insert(0, messenger)
        self._removed_echo.clear()
        if self._fs_fns is not None:
            from grc_agent import fs_tools
            if fs_tools._active_grc_path_fn is self._fs_fns[0]:
                fs_tools.set_active_graph_providers(fs_tools._no_grc, fs_tools._no_flow_graph,
                                                    fs_tools._no_grc)


def native_canvas_module() -> types.ModuleType:
    """A GTK-free stand-in for ``grc_agent.native_canvas`` (what agent_factory imports at L934:
    ``NativeFlowgraphProxy``), bound to :class:`HeadlessFlowgraphProxy`. The agent build installs it
    in ``sys.modules`` before calling ``build_agents_from_cfg``; nothing here installs it."""
    module = types.ModuleType("grc_agent.native_canvas", "GRC-Bench headless stand-in (no GTK).")
    module.NativeFlowgraphProxy = HeadlessFlowgraphProxy
    module._lock_path_for = _lock_path_for
    module._sha256_file = _sha256_file
    module.__file__ = __file__
    return module


__all__ = ["HeadlessFlowgraphProxy", "HeadlessPage", "SandboxedProcess", "RUN_STATUSES",
           "DEFAULT_MAX_RUN_SECONDS", "native_canvas_module", "terminal_command"]


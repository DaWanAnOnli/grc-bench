"""In-sandbox launcher for HeadlessFlowgraphProxy.run_flowgraph (eval plan §4.4, Tier A).

GRC's Execute action spawns the generated program with
``subprocess.Popen(args, stdout=PIPE, stderr=STDOUT, universal_newlines=True, cwd=...)`` and its
Stop action calls ``page.process.terminate()``: one SIGTERM to the process GRC spawned
(gnuradio/grc/gui/Executor.py, Application.py FLOW_GRAPH_KILL). Inside bubblewrap the program is
not our direct child (bwrap's PID-1 reaper sits in between, in its own PID namespace), so a signal
sent from outside would reach bwrap, not the flowgraph. This launcher runs as the sandboxed command
and makes the same Popen call GRC makes, so that:

  * the flowgraph's stdout and stderr go, merged, straight to the launcher's fd 1 (the pipe the
    proxy reads; the launcher itself never writes to stdout);
  * a line ``TERM`` on the launcher's stdin becomes ``child.terminate()`` (SIGTERM to the direct
    child, exactly GRC's Stop) and ``KILL`` becomes ``child.kill()``;
  * spawn success, a spawn exception (GRC's ``except Exception`` path in Executor.__init__) and the
    child's real ``Popen.returncode`` (negative for a signal, as GRC reports it) are reported as
    JSON lines on a separate status pipe, never mixed into the console output.

Terminal mode (``spec["terminal"]``) stands in for the terminal window GRC wraps no_gui programs in
(generator ``add_xterm``: ``x-terminal-emulator -e python -u <file>.py``, as in Qoherent's
collection, where x-terminal-emulator is xterm). The process GRC spawned is then the terminal, so
GRC's console and return code are the terminal's, and the program's output goes to the window:

  * the program runs in its own session (xterm's child calls setsid) with stdout and stderr going
    to the transcript file (the window), never to fd 1, so GRC's console shows nothing of it;
  * when the program ends, however it ends, the terminal closes and exits with status 0 (xterm's
    NormalExit -> Cleanup(0));
  * ``TERM`` is SIGTERM to the terminal: xterm exits at once with status 15 (``Exit(SIGTERM)``; the
    "Done (return code 15)" in Qoherent's rows) and its program gets the pty hang-up (SIGHUP to its
    session). The launcher then exits, and bubblewrap's PID namespace takes any program that
    ignored the hang-up with it, as the closed window would;
  * a program that cannot be started writes the error into the window and the terminal exits 0.

The program's real return code is reported separately (``program_returncode``) for the run record.

The child's stdin is an open pipe nobody writes to, the headless stand-in for the terminal window
GRC runs no_gui programs in: a "Prompt for Exit" program waits at ``input()`` until it is stopped,
instead of reading EOF at once.

    python -u -m grcbench.tier_a._launch '<json spec>'
    spec = {"status_fd": int, "cwd": str, "shell": bool, "args": [str] | str, "drop_pythonpath": str,
            "terminal": null | {"transcript": str}}

Standard library only, so it starts quickly inside the sandbox.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time

# xterm's exit status when it is itself terminated by GRC's Stop (SIGTERM): Exit(SIGTERM).
TERMINAL_STOP_STATUS = int(signal.SIGTERM)
# How long a stopped terminal-mode program may take to die of the hang-up before the launcher
# reports the terminal's exit anyway (only the run record's program_returncode depends on it).
_HANGUP_GRACE = 1.0


def _child_env(drop: str | None) -> dict[str, str]:
    """The sandbox environment minus GRC-Bench's own source entry on PYTHONPATH (the launcher needs
    it; the flowgraph does not, and in GRC it would not be there)."""
    env = dict(os.environ)
    if drop and "PYTHONPATH" in env:
        kept = [p for p in env["PYTHONPATH"].split(os.pathsep) if p and p != drop]
        if kept:
            env["PYTHONPATH"] = os.pathsep.join(kept)
        else:
            del env["PYTHONPATH"]
    return env


class _Status:
    """The status pipe. ``finish`` reports the final return code exactly once (the main thread and
    the stop relay can both reach it in terminal mode)."""

    def __init__(self, fd: int) -> None:
        self._out = os.fdopen(fd, "w", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()
        self._finished = False

    def report(self, **fields) -> None:
        with self._lock:
            if not self._finished:
                self._out.write(json.dumps(fields, sort_keys=True) + "\n")
                self._out.flush()

    def finish(self, **fields) -> bool:
        with self._lock:
            if self._finished:
                return False
            self._finished = True
            self._out.write(json.dumps(fields, sort_keys=True) + "\n")
            self._out.close()
            return True


def _hang_up(child: subprocess.Popen) -> int | None:
    """The terminal window closes: SIGHUP to the program's session, then a short wait for it."""
    try:
        os.killpg(child.pid, signal.SIGHUP)
    except OSError:
        pass
    deadline = time.monotonic() + _HANGUP_GRACE
    while child.returncode is None and time.monotonic() < deadline:
        child.poll()
        time.sleep(0.02)
    return child.returncode


def _relay_stop_requests(child: subprocess.Popen, status: _Status, terminal: bool,
                         stopping: threading.Event) -> None:
    """Read stop requests from fd 0 unbuffered (a buffered sys.stdin read in a daemon thread aborts
    the interpreter at shutdown)."""
    pending = b""
    while True:
        try:
            data = os.read(0, 64)
        except OSError:
            return
        if not data:
            return
        pending += data
        while b"\n" in pending:
            cmd, pending = pending.split(b"\n", 1)
            cmd = cmd.strip()
            try:
                if cmd == b"TERM" and terminal:
                    stopping.set()               # from here on the terminal's status is 15
                    code = _hang_up(child)
                    if status.finish(returncode=TERMINAL_STOP_STATUS, program_returncode=code):
                        os._exit(0)          # the terminal has exited; the namespace goes with it
                    return
                if cmd == b"TERM":
                    child.terminate()
                elif cmd == b"KILL":
                    child.kill()
            except OSError:
                pass


def main() -> int:
    spec = json.loads(sys.argv[1])
    status = _Status(spec["status_fd"])
    terminal = spec.get("terminal")
    stdin_r, stdin_w = os.pipe()
    window = open(terminal["transcript"], "ab", buffering=0) if terminal else None
    try:
        child = subprocess.Popen(spec["args"], cwd=spec["cwd"], shell=bool(spec.get("shell")),
                                 stdin=stdin_r, stdout=window, stderr=subprocess.STDOUT,
                                 start_new_session=bool(terminal),
                                 env=_child_env(spec.get("drop_pythonpath")))
    except Exception as exc:  # the exception GRC's Executor would have caught around Popen
        if window is None:
            status.report(spawn_error=str(exc))
            return 0
        window.write(f"{exc}\n".encode("utf-8", "replace"))       # shown in the terminal window
        status.report(spawned=0)
        status.finish(returncode=0, program_returncode=None, program_spawn_error=str(exc))
        return 0
    finally:
        os.close(stdin_r)
        if window is not None:
            window.close()
    status.report(spawned=child.pid)
    stopping = threading.Event()
    threading.Thread(target=_relay_stop_requests, args=(child, status, bool(terminal), stopping),
                     daemon=True).start()
    code = child.wait()
    if terminal:
        # A program that died of the stop's hang-up must not race the relay into reporting the
        # "program ended" status: the terminal was stopped, so it exits with 15.
        status.finish(returncode=TERMINAL_STOP_STATUS if stopping.is_set() else 0,
                      program_returncode=code)
    else:
        status.finish(returncode=code, program_returncode=code)
    os._exit(0)          # skip interpreter finalisation while the relay thread sits in os.read


if __name__ == "__main__":
    sys.exit(main())

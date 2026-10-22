"""Run untrusted code (model-written flowgraphs, agent shell commands) in a bubblewrap sandbox.

GNU Radio evaluates block parameters as Python, Embedded-Python blocks run on load, and V5 runs
generated code, so every rung from V2 up and every agent tool that executes code runs here:
  * the whole filesystem is read-only (the repo, data/ and ~/grc-agent cannot be touched);
  * only the run directory is writable; HOME and the XDG directories point inside it;
  * no network (--unshare-net), its own PID namespace, killed with the parent;
  * CPU-time and address-space limits plus a wall-clock timeout.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

BWRAP = shutil.which("bwrap") or "/usr/bin/bwrap"


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def sandbox_env(run_dir: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    home = run_dir / "home"
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "XDG_STATE_HOME": str(home / ".local" / "state"),
        "GRC_HIER_PATH": str(home / ".grc_gnuradio"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "QT_QPA_PLATFORM": "offscreen",
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
    }
    for key in ("PYTHONPATH", "TIKTOKEN_CACHE_DIR", "LD_LIBRARY_PATH"):
        if key in os.environ:
            # the sandbox runs in run_dir, so relative entries must be made absolute
            env[key] = os.pathsep.join(str(Path(p).resolve()) for p in os.environ[key].split(os.pathsep) if p)
    if "TIKTOKEN_CACHE_DIR" not in env:
        # the cl100k file cached by `./gb fetch` (tokens.py counts tool outputs with it); without it
        # tiktoken would try to download the file, which the sandbox's missing network refuses. Set
        # here, where HOME is still the real one, whatever started the process (./gb or not).
        from .paths import CACHE
        env["TIKTOKEN_CACHE_DIR"] = str(CACHE / "tiktoken")
    src = str(Path(__file__).resolve().parents[2])           # .../grc-bench/src
    if src not in env.get("PYTHONPATH", "").split(os.pathsep):
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH", "")]))
    if extra:
        env.update(extra)
    return env


def _limits(cpu_seconds: int, mem_bytes: int):
    def apply():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
        if mem_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    return apply


def run(cmd: list[str], run_dir: Path, *, timeout: float = 60.0, cpu_seconds: int = 120,
        mem_bytes: int = 4 << 30, env: dict[str, str] | None = None, network: bool = False,
        stdin: str | None = None, extra_ro_binds: list[Path] | None = None) -> Result:
    """Run `cmd` inside the sandbox with `run_dir` as the only writable path and working dir."""
    run_dir = Path(run_dir).resolve()
    for sub in ("home/.cache", "home/.config", "home/.local/share", "home/.local/state",
                "home/.grc_gnuradio"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    args = [BWRAP, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
            "--bind", str(run_dir), str(run_dir), "--chdir", str(run_dir),
            "--unshare-pid", "--die-with-parent", "--new-session"]
    if not network:
        args.append("--unshare-net")
    for path in extra_ro_binds or []:
        args += ["--ro-bind", str(path), str(path)]
    args.append("--")
    full_env = sandbox_env(run_dir, env)
    proc = subprocess.Popen(args + cmd, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=full_env,
                            preexec_fn=_limits(cpu_seconds, mem_bytes))
    try:
        out, err = proc.communicate(input=stdin, timeout=timeout)
        return Result(proc.returncode, out, err, False)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
        return Result(-9, out, err, True)

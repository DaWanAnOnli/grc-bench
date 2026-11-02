"""Run untrusted code (model-written flowgraphs, agent shell commands) in a bubblewrap sandbox.

GNU Radio evaluates block parameters as Python, Embedded-Python blocks run on load, and V4 runs
generated code, so every rung from V2 up and every agent tool that executes code runs here:
  * the whole filesystem is read-only (the repo, data/ and ~/grc-agent cannot be touched);
  * the user's files are hidden (`hidden_view_args`): the real home directory and /mnt (the
    Windows drives under WSL) are empty, apart from GRC-Agent, the cache and the Python import path,
    which come back read-only. Secrets (~/.config/grc_agent/.env, credentials on the Windows side),
    the benchmark's items and answer keys and the other runs' directories cannot be read;
  * only the run directory is writable; HOME and the XDG directories point inside it;
  * no network (--unshare-net), its own PID namespace, killed with the parent;
  * CPU-time and address-space limits plus a wall-clock timeout.

GRC_BENCH_BWRAP_USERNS=1 adds --unshare-user: a root shell in a container without CAP_SYS_ADMIN or
CAP_NET_ADMIN (a Colab runtime) can create the other namespaces, and bring up the loopback, only inside a
user namespace of its own.

GRC_BENCH_BWRAP_PROC=host binds the machine's /proc instead of a fresh one, which a Colab runtime refuses
to mount (writable, as a fresh one is: the kernel checks each write, and bubblewrap inside writes its id
maps there). It is accepted only together with GRC_BENCH_BWRAP_USERNS=1: the sandbox then
sees the numbers and command lines of the processes outside, but from its own user namespace it cannot
inspect them (/proc/<pid>/root, environ, fd: refused). All three bubblewrap commands (run, the Tier-A
shell sandbox and flowgraph runs) take these settings from proc_args() and userns_args(), and a sandbox
passes them on to the sandboxes started inside it. In that mode bubblewrap must find its own processes
under the machine's /proc, so the one sandbox that starts sandboxes itself (the Tier-A episode process,
run(starts_sandboxes=True), which runs GRC-Agent's harness) stays in the machine's PID namespace; the
sandboxes it starts for model-written code (shell commands, flowgraph runs) have their own.
"""

from __future__ import annotations

import os
import pwd
import resource
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BWRAP = shutil.which("bwrap") or "/usr/bin/bwrap"
SRC = Path(__file__).resolve().parents[2]                    # .../grc-bench/src


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
    for key in ("GRC_BENCH_BWRAP_USERNS", "GRC_BENCH_BWRAP_PROC"):
        if key in os.environ:            # a sandbox started inside (Tier A) runs with the same settings
            env[key] = os.environ[key]
    if "TIKTOKEN_CACHE_DIR" not in env:
        # the cl100k file cached by `./gb fetch` (tokens.py counts tool outputs with it); without it
        # tiktoken would try to download the file, which the sandbox's missing network refuses. Set
        # here, where HOME is still the real one, whatever started the process (./gb or not).
        from .paths import CACHE
        env["TIKTOKEN_CACHE_DIR"] = str(CACHE / "tiktoken")
    src = str(SRC)
    if src not in env.get("PYTHONPATH", "").split(os.pathsep):
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH", "")]))
    if extra:
        env.update(extra)
    return env


def real_home() -> Path | None:
    """The user's home directory from the password database (HOME may already point into a run)."""
    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    except (KeyError, OSError):
        return None


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def hidden_roots() -> list[Path]:
    """The user's files: the real home directory and /mnt (the Windows drives under WSL)."""
    return [p for p in (real_home(), Path("/mnt")) if p is not None and p.is_dir()]


def agent_hidden_roots(*, hide_src: bool) -> list[Path]:
    """What a sandbox for code the AGENT starts (its shell commands, its flowgraph runs) also hides
    inside an episode: the run directories (the episode's own run folder holds the task record,
    episode.json with the acceptance checks, and the model relay's socket; the other folders hold
    other runs), the GRC-Bench cache and, with `hide_src`, the grcbench source (a flowgraph run
    needs it for its launcher; a shell command does not)."""
    from .paths import CACHE, WORK
    roots = [Path(p).resolve() for p in (WORK, CACHE, *([SRC] if hide_src else []))]
    return [p for p in roots if p.is_dir()]


def hide_args(roots: list[Path], keep: list[Path] = ()) -> list[str]:
    """bwrap arguments (after `--ro-bind / /`) that empty each of `roots` (a tmpfs) and bring the
    `keep` paths inside them back read-only. The caller binds its own working paths after these
    and ends with `readonly_args(roots)`, so the emptied areas stay read-only, as before."""
    args: list[str] = []
    for path in roots:
        args += ["--tmpfs", str(path)]
    seen: set[Path] = set()
    for path in keep:
        path = Path(path).resolve()
        if path in seen or not path.exists() or not any(_is_inside(path, r) for r in roots):
            continue
        seen.add(path)
        args += ["--ro-bind", str(path), str(path)]
    return args


def readonly_args(roots: list[Path]) -> list[str]:
    return [a for path in roots for a in ("--remount-ro", str(path))]


def hidden_view_args(env: dict[str, str]) -> list[str]:
    """`hide_args` for the user's files: the real home and /mnt are emptied, then what a sandboxed
    Python run needs comes back read-only: GRC-Agent (its code and virtualenv), the GRC-Bench cache
    (knowledge databases, tiktoken file, tools), this interpreter's prefix and import path
    (sys.prefix, sys.path: the virtualenv, editable installs) and every PYTHONPATH /
    LD_LIBRARY_PATH entry (the grcbench src/). A path that contains the repository, the project's
    data folders, ~/.config or the run directories is never brought back."""
    from .paths import CACHE, GRC_AGENT, PROJECT, REPO, WORK
    keep = [GRC_AGENT, CACHE, Path(sys.prefix), *(Path(p) for p in sys.path if p)]
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        keep += [Path(p) for p in env.get(key, "").split(os.pathsep) if p]
    home = real_home()
    sensitive = [REPO, PROJECT / "data", PROJECT / "new_data", WORK, *([home / ".config"] if home else [])]
    sensitive = [Path(s).resolve() for s in sensitive]
    keep = [p for p in keep if p.is_absolute() and not any(_is_inside(s, p.resolve()) for s in sensitive)]
    return hide_args(hidden_roots(), keep)


def _limits(cpu_seconds: int, mem_bytes: int):
    def apply():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
        if mem_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    return apply


def userns_args() -> list[str]:
    """--unshare-user when GRC_BENCH_BWRAP_USERNS=1 (see the module text)."""
    return ["--unshare-user"] if os.environ.get("GRC_BENCH_BWRAP_USERNS") == "1" else []


def proc_args() -> list[str]:
    """A fresh /proc for the sandbox's PID namespace, or a bind of the machine's when
    GRC_BENCH_BWRAP_PROC=host, which needs GRC_BENCH_BWRAP_USERNS=1 (see the module text)."""
    mode = os.environ.get("GRC_BENCH_BWRAP_PROC", "new")
    if mode == "new":
        return ["--proc", "/proc"]
    if mode != "host":
        raise ValueError(f"GRC_BENCH_BWRAP_PROC={mode!r}: use 'new' (the default) or 'host'")
    if not userns_args():
        raise ValueError("GRC_BENCH_BWRAP_PROC=host needs GRC_BENCH_BWRAP_USERNS=1: without a user namespace "
                         "of its own the sandbox could inspect the processes outside through /proc")
    return ["--bind", "/proc", "/proc"]


def run(cmd: list[str], run_dir: Path, *, timeout: float = 60.0, cpu_seconds: int = 120,
        mem_bytes: int = 4 << 30, env: dict[str, str] | None = None, network: bool = False,
        stdin: str | None = None, extra_ro_binds: list[Path] | None = None,
        starts_sandboxes: bool = False) -> Result:
    """Run `cmd` inside the sandbox with `run_dir` as the only writable path and working dir.
    starts_sandboxes: `cmd` starts sandboxes itself (see the module text on the PID namespace)."""
    run_dir = Path(run_dir).resolve()
    for sub in ("home/.cache", "home/.config", "home/.local/share", "home/.local/state",
                "home/.grc_gnuradio"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    full_env = sandbox_env(run_dir, env)
    proc = proc_args()
    pidns = ["--unshare-pid"] if proc[0] == "--proc" or not starts_sandboxes else []
    args = [BWRAP, "--ro-bind", "/", "/", "--dev", "/dev", *proc, "--tmpfs", "/tmp",
            *hidden_view_args(full_env),
            "--bind", str(run_dir), str(run_dir), "--chdir", str(run_dir),
            *pidns, "--die-with-parent", "--new-session", *userns_args()]
    if not network:
        args.append("--unshare-net")
    for path in extra_ro_binds or []:
        args += ["--ro-bind", str(path), str(path)]
    args += [*readonly_args(hidden_roots()), "--"]
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

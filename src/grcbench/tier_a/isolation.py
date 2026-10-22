"""Per-episode isolation for Tier-A agent runs (eval plan §4.4; build plan "Isolation").

Every Tier-A episode runs GRC-Agent in its own OS process, inside bubblewrap, with its own run
directory under ~/grc-bench-work. This module writes that directory's settings and builds the
process environment; `run_episode_process` starts the process.

Why one process per episode is required, not just recommended. GRC-Agent fixes part of its state
for the life of a process:
  * GRC_HIER_PATH is read once, when gnuradio.grc.core.Config is imported
    (Config.hier_block_lib_dir is a class attribute; grc_agent.adapter.block_library says so);
  * the GRC block library is built once per process (grc_agent.adapter.graph._PLATFORM), so a block
    saved by save_block in one episode would still be in the catalog of the next;
  * the fs/shell tools resolve their root through module-level providers
    (grc_agent.fs_tools.set_active_graph_providers);
  * the step store, .env reader and context-window probe cache per path or per model in module
    globals (grc_agent.db._step_stores, settings._dotenv_cache, agent_factory._context_length_cache).
A fresh process per episode resets all of them; the env vars below are set before Python starts.

Run directory layout (all of it on the WSL filesystem, under ~/grc-bench-work):
    <run>/.env                       GRC-Agent settings: provider, endpoint, model (no secrets)
    <run>/.grc_agent/chat_sessions.db  GRC-Agent's step store (it lives next to the .env)
    <run>/home/                      HOME and XDG_*; GRC library cache; .grc_gnuradio = GRC_HIER_PATH
    <run>/runtime/                   GRC_AGENT_RUNTIME_DIR (embedding runtime; unused with lexical)
    <run>/project/                   the episode's project folder: a copy of the start graph
The lexical knowledge databases are copied once into ~/grc-bench-cache/grc-agent-vectors and used
read-only by every episode (GRC_AGENT_VECTORS_DIR); ~/grc-agent itself is never written.

Inside an episode, the shell tools' commands run in a second, nested sandbox (`shell_sandbox_argv`):
the episode project folder is the only writable path, HOME and /tmp are empty scratch tmpfs, the
rest of the run folder (episode.json holds the task record), the other runs, the cache and the
grcbench source are hidden, and there is no network at all.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..common import sandbox
from ..common.io import sha256_file, write_json, write_text
from ..common.paths import CACHE, GRC_AGENT, HOME, PROJECT, WORK

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"      # llama-server's OpenAI-compatible endpoint
DEFAULT_MODEL = "model-under-test"                  # llama-server serves one model; name is a label
VECTORS_SRC = GRC_AGENT / "src" / "grc_agent" / "vectors"
VECTORS_CACHE = CACHE / "grc-agent-vectors"
LEXICAL_DBS = ("catalog_lexical.db", "docs_lexical.db")
RUNS_ROOT = WORK / "tier_a"
PRLIMIT = shutil.which("prlimit") or "/usr/bin/prlimit"
_VECTORS_LOCK = threading.Lock()

# Never written by a Tier-A run (baseline manifests in `./gb baseline` cover the same roots).
FORBIDDEN_ROOTS = (GRC_AGENT, HOME / ".config" / "grc_agent", HOME / ".cache" / "gnuradio",
                   HOME / ".local" / "state" / "gnuradio", PROJECT / "data")


class IsolationError(RuntimeError):
    """The process or run directory does not meet the Tier-A isolation contract."""


@dataclass(frozen=True)
class RunLayout:
    """Paths of one episode's run directory (see the module docstring)."""

    run_dir: Path

    @property
    def env_file(self) -> Path:
        return self.run_dir / ".env"

    @property
    def state_dir(self) -> Path:
        return self.run_dir / ".grc_agent"

    @property
    def step_db(self) -> Path:
        return self.state_dir / "chat_sessions.db"

    @property
    def home(self) -> Path:
        return self.run_dir / "home"

    @property
    def hier_dir(self) -> Path:
        return self.home / ".grc_gnuradio"

    @property
    def runtime_dir(self) -> Path:
        return self.run_dir / "runtime"

    @property
    def project(self) -> Path:
        return self.run_dir / "project"


def is_inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def check_run_dir(run_dir: Path) -> Path:
    """Resolve `run_dir` and refuse anything outside ~/grc-bench-work or inside a read-only input."""
    run_dir = Path(run_dir).resolve()
    if not is_inside(run_dir, WORK) or run_dir == WORK.resolve():
        raise IsolationError(f"Tier-A run directories live under {WORK}; got {run_dir}")
    for root in FORBIDDEN_ROOTS:
        if is_inside(run_dir, root):
            raise IsolationError(f"run directory {run_dir} is inside read-only input {root}")
    return run_dir


def new_run_dir(*parts: str) -> Path:
    """A fresh, empty run directory RUNS_ROOT/<parts...> (an existing one is removed first)."""
    run_dir = check_run_dir(RUNS_ROOT.joinpath(*parts))
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def ensure_vectors_cache() -> Path:
    """Copy GRC-Agent's lexical knowledge databases once into ~/grc-bench-cache (read-only copies).

    The copies are compared by SHA-256 with the pinned originals on every call, so a stale or
    damaged cache is replaced. The originals are only read. Both databases use SQLite's rollback
    journal (journal_mode=delete), so readers never need to create files next to them.

    Safe to call from concurrent threads and processes (gate replays prepare many run directories
    at once): one thread lock plus an flock on CACHE/.grc-agent-vectors.lock, and files that are
    already right are not rewritten.
    """
    import fcntl
    VECTORS_CACHE.mkdir(parents=True, exist_ok=True)
    with _VECTORS_LOCK, open(CACHE / ".grc-agent-vectors.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _ensure_vectors_cache()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _ensure_vectors_cache() -> Path:
    manifest = {}
    for name in LEXICAL_DBS:
        src, dst = VECTORS_SRC / name, VECTORS_CACHE / name
        if not src.is_file():
            raise IsolationError(f"GRC-Agent knowledge database missing: {src}")
        want = sha256_file(src)
        if not (dst.is_file() and sha256_file(dst) == want):
            tmp = dst.with_name(dst.name + ".tmp")
            if tmp.exists():
                tmp.unlink()
            shutil.copyfile(src, tmp)
            if dst.exists():
                dst.chmod(0o644)
            os.replace(tmp, dst)
        dst.chmod(0o444)
        manifest[name] = want
    manifest_path = VECTORS_CACHE / "MANIFEST.json"
    want_manifest = {"source": str(VECTORS_SRC), "sha256": manifest}
    try:
        current = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    except (OSError, ValueError):
        current = None
    if current != want_manifest:
        if manifest_path.exists():
            manifest_path.chmod(0o644)
        write_json(manifest_path, want_manifest)
    manifest_path.chmod(0o444)
    return VECTORS_CACHE


def dotenv_text(base_url: str, model_name: str) -> str:
    """The episode's GRC-Agent .env. Only what selects the model and the knowledge backend.

    Deliberately absent: API keys (llama-server needs none); GRC_AGENT_APPROVE_CHANGES (read only
    by the GTK sidebar; the runner approves deferred calls itself); GRC_SHELL_TIMEOUT and
    GRC_SHELL_DENIED_COMMANDS (GRC-Agent's defaults apply; the timeout default is also printed
    into run_command's schema); GRC_COMPACTION_TARGET_TOKENS (GRC-Agent's default rule applies in
    the grc_agent_defaults setting).
    """
    for value in (base_url, model_name):
        if any(c in value for c in "\n\r\"' #"):
            raise ValueError(f"unsafe .env value: {value!r}")
    return ("# GRC-Agent settings for one GRC-Bench Tier-A episode (grcbench.tier_a.isolation).\n"
            "# No secrets: the model under test is served by a local llama-server.\n"
            "GRC_PROVIDER=openai_compatible\n"
            f"OPENAI_COMPATIBLE_BASE_URL={base_url}\n"
            f"OPENAI_COMPATIBLE_MODEL={model_name}\n"
            "GRC_EMBED_BACKEND=lexical\n")


def make_run_env(run_dir: Path, *, base_url: str = DEFAULT_BASE_URL, model_name: str = DEFAULT_MODEL,
                 warm_grc_cache: bool = True) -> dict[str, str]:
    """Prepare `run_dir` for one episode and return the complete process environment for it.

    Writes <run>/.env, creates home/, runtime/ and project/, installs the warm GRC library cache
    into home/ (so loading the block catalog takes about a second), and makes sure the read-only
    knowledge-database copies exist. The environment starts from `common.sandbox.sandbox_env`
    (HOME and XDG_* inside the run, GRC_HIER_PATH, PYTHONDONTWRITEBYTECODE=1, QT offscreen, no
    inherited secrets) and adds GRC-Agent's own variables.
    """
    run_dir = check_run_dir(run_dir)
    lay = RunLayout(run_dir)
    for d in (lay.home, lay.hier_dir, lay.runtime_dir, lay.project):
        d.mkdir(parents=True, exist_ok=True)
    write_text(lay.env_file, dotenv_text(base_url, model_name))
    vectors = ensure_vectors_cache()
    if warm_grc_cache:
        from ..tier_f.grc import install_warm_cache
        install_warm_cache(lay.home)
    env = sandbox.sandbox_env(run_dir)
    env.update({
        "GRC_AGENT_ENV": str(lay.env_file),
        "GRC_AGENT_VECTORS_DIR": str(vectors),
        "GRC_AGENT_RUNTIME_DIR": str(lay.runtime_dir),
        "GRC_HIER_PATH": str(lay.hier_dir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "QT_QPA_PLATFORM": "offscreen",
        # grcbench.common.paths resolves these from HOME by default; HOME is now inside the run.
        "GRC_AGENT_REPO": str(GRC_AGENT),
        "GRC_BENCH_CACHE": str(CACHE),
        "GRC_BENCH_WORK": str(WORK),
        "GRC_BENCH_RUN_DIR": str(run_dir),
    })
    return env


def check_process_isolation(run_dir: Path | None = None) -> RunLayout:
    """Verify, inside an episode process, that every GRC-Agent path points into the run directory.

    Raises IsolationError naming the first variable that is missing or points elsewhere.
    `run_dir` defaults to GRC_BENCH_RUN_DIR (set by make_run_env).
    """
    env = os.environ
    raw = run_dir or env.get("GRC_BENCH_RUN_DIR")
    if not raw:
        raise IsolationError("GRC_BENCH_RUN_DIR is not set: start episodes with make_run_env()")
    lay = RunLayout(check_run_dir(Path(raw)))
    expect = {
        "GRC_AGENT_ENV": lay.env_file,
        "GRC_AGENT_RUNTIME_DIR": lay.runtime_dir,
        "GRC_HIER_PATH": lay.hier_dir,
        "HOME": lay.home,
    }
    for key, path in expect.items():
        if Path(env.get(key, "")).resolve() != path.resolve():
            raise IsolationError(f"{key}={env.get(key)!r}, expected {path}")
    for key in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        if not env.get(key) or not is_inside(Path(env[key]), lay.home):
            raise IsolationError(f"{key}={env.get(key)!r} is not inside {lay.home}")
    if Path(env.get("GRC_AGENT_VECTORS_DIR", "")).resolve() != VECTORS_CACHE.resolve():
        raise IsolationError(f"GRC_AGENT_VECTORS_DIR must be the read-only copy {VECTORS_CACHE}")
    if env.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise IsolationError("PYTHONDONTWRITEBYTECODE must be 1 (no __pycache__ in ~/grc-agent)")
    if not lay.env_file.is_file():
        raise IsolationError(f"missing {lay.env_file}")
    return lay


def copy_start_graph(run_dir: Path, start_graph: Path, name: str | None = None) -> Path:
    """Copy a task's start graph into the episode project folder; the original is never edited."""
    lay = RunLayout(check_run_dir(run_dir))
    lay.project.mkdir(parents=True, exist_ok=True)
    dest = lay.project / (name or Path(start_graph).name)
    shutil.copyfile(start_graph, dest)
    return dest


def run_episode_process(run_dir: Path, module: str, args: list[str], env: dict[str, str], *,
                        timeout: float = 1800.0, network: Literal["host", "none"] = "host",
                        cpu_seconds: int = 3600, mem_bytes: int = 8 << 30,
                        extra_ro_binds: list[Path] | None = None) -> sandbox.Result:
    """Run `python -m <module> <args>` as one episode process inside bubblewrap.

    The whole filesystem is read-only except `run_dir`; HOME/XDG point into it (env from
    make_run_env); the user's files are hidden (sandbox.hidden_view_args). network="host" shares
    the host network namespace, which a llama-server on a TCP port needs; network="none" unshares it
    (for a model server on a UNIX socket: a llama-server, or the relay of a hosted API, whose socket
    directory comes in through `extra_ro_binds`). Commands the agent runs through its shell tools
    additionally get a no-network sandbox of their own (shell_sandbox_argv).
    """
    run_dir = check_run_dir(run_dir)
    cmd = [sys.executable, "-m", module, *args]
    return sandbox.run(cmd, run_dir, timeout=timeout, cpu_seconds=cpu_seconds, mem_bytes=mem_bytes,
                       env=env, network=(network == "host"), extra_ro_binds=extra_ro_binds)


# --------------------------------------------------------------------------------------------
# Shell-tool sandbox (nested bubblewrap)
# --------------------------------------------------------------------------------------------

SHELL_CPU_SECONDS = 600            # same order as GrcShell's default 600 s wall timeout
SHELL_MEM_BYTES = 4 << 30
_SHELL_PROJECTS: set[Path] = set()


def shell_sandbox_argv(command: str, cwd: Path) -> list[str]:
    """argv that runs `command` with /bin/sh inside a no-network bubblewrap sandbox.

    Writable: only the episode project folder `cwd` (registered by install_shell_sandbox). HOME
    and /tmp are fresh tmpfs scratch that disappears with the command. The rest of the run folder,
    the other runs, the cache and the grcbench source are hidden (sandbox.agent_hidden_roots), on
    top of the episode sandbox's hidden view, and stay read-only (a write there fails with
    "Read-only file system", as before). CPU-time and address-space limits come from prlimit;
    GrcShell's own wall-clock timeout and process-group kill still apply.
    """
    if not isinstance(command, str):
        raise PermissionError("GRC-Bench shell sandbox: only shell command strings can be run")
    project = Path(cwd).resolve()
    if project not in _SHELL_PROJECTS:
        raise PermissionError("GRC-Bench shell sandbox: commands may only run in the episode "
                              "project directory")
    args = [PRLIMIT, f"--cpu={SHELL_CPU_SECONDS}", f"--as={SHELL_MEM_BYTES}", "--",
            sandbox.BWRAP, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/tmp", *sandbox.hidden_view_args(dict(os.environ))]
    agent_hidden = sandbox.agent_hidden_roots(hide_src=True)
    args += sandbox.hide_args(agent_hidden)
    home = os.environ.get("HOME")
    if home and Path(home).is_dir() and not is_inside(project, Path(home)) \
            and not is_inside(Path(home), project):
        args += ["--tmpfs", str(Path(home).resolve())]
    args += ["--bind", str(project), str(project), *sandbox.readonly_args(sandbox.hidden_roots() + agent_hidden),
             "--chdir", str(project), "--unshare-pid", "--unshare-net", "--die-with-parent", "--new-session",
             "--", "/bin/sh", "-c", command]
    return args


class _SandboxedAnyio:
    """Stands in for the name `anyio` inside pydantic_ai_harness.shell._toolset.

    Identical to anyio except `open_process`, which the harness ShellToolset (and so GrcShell's
    run_command and start_command) uses for every spawn: the command string is rewritten into
    shell_sandbox_argv(...). Everything before the spawn is GRC-Agent's own code: the approval
    gate, GrcShell's denylist and project-directory gate, env scrubbing, timeouts, output caps.
    """

    def __init__(self, real: Any) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    async def open_process(self, command: Any, **kwargs: Any) -> Any:
        cwd = kwargs.get("cwd")
        if cwd is None:
            raise PermissionError("GRC-Bench shell sandbox: no working directory")
        return await self._real.open_process(shell_sandbox_argv(command, Path(cwd)), **kwargs)


def install_shell_sandbox(project_dir: Path) -> None:
    """Route every GrcShell command of this process through the nested sandbox (idempotent).

    Process-wide by design: an episode is one process (module docstring). A command whose working
    directory is not a registered episode project directory is refused (fail closed).
    """
    _SHELL_PROJECTS.add(Path(project_dir).resolve())
    mod = importlib.import_module("pydantic_ai_harness.shell._toolset")
    if not isinstance(mod.anyio, _SandboxedAnyio):
        mod.anyio = _SandboxedAnyio(mod.anyio)


def shell_sandbox_installed() -> bool:
    mod = sys.modules.get("pydantic_ai_harness.shell._toolset")
    return mod is not None and isinstance(getattr(mod, "anyio", None), _SandboxedAnyio)

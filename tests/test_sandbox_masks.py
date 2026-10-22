"""What sandboxed code can read (common.sandbox, tier_a.isolation, tier_a.headless_proxy).

Each test runs a real probe inside the sandbox and lists which paths are readable:
  * every sandbox hides the user's files: the real home (apart from GRC-Agent and the cache) and
    /mnt (the Windows drives: credentials, the repo's items and answer keys);
  * the sandboxes for code the agent starts (shell commands, flowgraph runs) also hide the rest of
    the episode's run folder (episode.json holds the task record), the other runs and the cache.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from grcbench.common import sandbox
from grcbench.common.paths import CACHE, GRC_AGENT, REPO, WORK
from grcbench.tier_a import isolation
from grcbench.tier_a.headless_proxy import HeadlessFlowgraphProxy

pytestmark = pytest.mark.skipif(not Path(sandbox.BWRAP).exists(), reason="bubblewrap is not installed")

NESTED_SHELL = """
import subprocess, sys
from pathlib import Path
from grcbench.tier_a import isolation
project = Path(sys.argv[1]).resolve()
isolation._SHELL_PROJECTS.add(project)
argv = isolation.shell_sandbox_argv(sys.argv[2], project)
res = subprocess.run(argv, capture_output=True, text=True, cwd=project)
print(res.stdout)
print(res.stderr, file=sys.stderr)
"""


@pytest.fixture
def runs():
    root = WORK / "pytest-masks" / uuid.uuid4().hex[:10]
    run, other = root / "run", root / "other"
    (run / "project").mkdir(parents=True)
    other.mkdir(parents=True)
    (run / "episode.json").write_text("{}\n")
    (run / "project" / "flowgraph.grc").write_text("options: {}\n")
    (other / "golden_end_state.grc").write_text("options: {}\n")
    isolation.ensure_vectors_cache()
    yield run, other
    shutil.rmtree(root, ignore_errors=True)


def _probe(paths: list[Path]) -> str:
    quoted = " ".join(shlex.quote(str(p)) for p in paths)
    return f'for p in {quoted}; do if [ -r "$p" ]; then echo "R $p"; fi; done'


def _readable(stdout: str) -> set[str]:
    return {line[2:] for line in stdout.splitlines() if line.startswith("R ")}


def _secrets() -> list[Path]:
    """Paths that exist on this machine and must never be readable from a sandbox. (Files, or
    folders that are absent from the sandbox's view: an emptied folder itself stays listable.)"""
    home = sandbox.real_home()
    candidates = [home / ".config" / "grc_agent" / ".env", REPO / "README.md", REPO / "items",
                  REPO.parent / "data", *Path("/mnt/c/Users").glob("*/.claude/.credentials.json")]
    return [p for p in candidates if p.exists()]


def _env() -> dict[str, str]:
    return {"GRC_BENCH_WORK": str(WORK), "GRC_BENCH_CACHE": str(CACHE), "GRC_AGENT_REPO": str(GRC_AGENT)}


CACHE_FILE = isolation.VECTORS_CACHE / "MANIFEST.json"         # made by the `runs` fixture


def test_the_outer_sandbox_hides_the_users_files(runs):
    run, other = runs
    secrets = _secrets()
    needed = [GRC_AGENT / "src", CACHE_FILE, sandbox.SRC / "grcbench", run / "episode.json"]
    res = sandbox.run(["/bin/sh", "-c", _probe([*secrets, other / "golden_end_state.grc", *needed])],
                      run, timeout=60, env=_env())
    seen = _readable(res.stdout)
    assert {str(p) for p in needed if p.exists()} <= seen, res.stderr
    assert not seen & {str(p) for p in secrets}
    assert str(other / "golden_end_state.grc") not in seen


def test_agent_shell_commands_see_only_the_project_folder(runs):
    run, other = runs
    project = run / "project"
    hidden = [*_secrets(), run / "episode.json", other / "golden_end_state.grc", CACHE_FILE,
              sandbox.SRC / "grcbench"]
    needed = [project / "flowgraph.grc", GRC_AGENT / "src"]
    cmd = [sys.executable, "-c", NESTED_SHELL, str(project), _probe([*hidden, *needed])]
    # the CPU limit of an episode process (isolation.run_episode_process), above the shell's own
    res = sandbox.run(cmd, run, timeout=60, cpu_seconds=3600, env=_env())
    seen = _readable(res.stdout)
    assert {str(p) for p in needed} <= seen, res.stderr
    assert not seen & {str(p) for p in hidden}


def test_flowgraph_runs_see_only_their_own_paths(runs):
    run, other = runs
    project = run / "project"
    hidden = [*_secrets(), run / "episode.json", other / "golden_end_state.grc", CACHE_FILE]
    needed = [project / "flowgraph.grc", sandbox.SRC / "grcbench" / "tier_a" / "_launch.py"]
    me = SimpleNamespace(_sandbox_root=(run / "fg").resolve())
    me._sandbox_root.mkdir()
    argv = HeadlessFlowgraphProxy._bwrap_argv(me, project.resolve(), [project.resolve(), me._sandbox_root], [])
    res = subprocess.run([*argv, "/bin/sh", "-c", _probe([*hidden, *needed])], capture_output=True, text=True,
                         timeout=60, check=False)
    seen = _readable(res.stdout)
    assert {str(p) for p in needed} <= seen, res.stderr
    assert not seen & {str(p) for p in hidden}

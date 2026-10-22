"""`./gb build-all [--tier-f] [--keep]`: regenerate the generated artefacts and check that they are
deterministic and current (build plan, verification step 2: "build-all run twice gives identical
hashes").

Each generator that takes `--out` is run twice, each time in a fresh process, into two scratch roots
under <WORK>/build_all/. A file is
  deterministic  when both runs write the same bytes;
  current        when the first run's bytes equal the repository's copy.
The block catalog and the calculation answers also run their own `--check` against the repository.
With --tier-f every Tier-F task's build.py runs in a mirror of the repository and every file it
writes is compared with the task folder (NOT_BUILT excluded). Nothing in the repository
is written. Exit status 0 only if every file is deterministic and current.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .paths import REPO, WORK

# command, extra arguments; each takes --out <root> and writes the paths it owns below that root
GENERATORS = [
    ("split-dev", []),
    ("labels", ["build"]),
    ("tier-k-items", []),
    ("calc", []),
    ("blockq", []),
]
CHECKS = [("blocks", ["--check"]), ("calc", ["--check"])]
# files in a Tier-F task folder that build.py does not write: gate results, the script itself, and the
# golden edit derived by `./gb tier-a tier-f-golden`
NOT_BUILT = ("gates.json", "build.py", "golden_calls.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def _gb(args: list[str], cwd: Path = REPO) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-m", "grcbench", *args], cwd=cwd, env=env,
                          capture_output=True, text=True)


def _generators(work: Path) -> list[str]:
    problems = []
    for cmd, extra in GENERATORS:
        trees = []
        for k in (1, 2):
            root = work / f"run{k}" / cmd
            root.mkdir(parents=True)
            r = _gb([cmd, *extra, "--out", str(root)])
            if r.returncode:
                problems.append(f"{cmd}: run {k} failed: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
                break
            trees.append(_tree(root))
        if len(trees) < 2:
            continue
        a, b = trees
        nondet = sorted(f for f in set(a) | set(b) if a.get(f) != b.get(f))
        stale = sorted(f for f in a if not (REPO / f).is_file() or _sha(REPO / f) != a[f])
        problems += [f"{cmd}: not deterministic: {f}" for f in nondet]
        problems += [f"{cmd}: differs from the repository: {f}" for f in stale]
        print(f"{cmd:14s} {len(a):3d} files  deterministic: {'yes' if not nondet else 'NO'}  "
              f"current: {'yes' if not stale else 'NO'}")
    for cmd, extra in CHECKS:
        r = _gb([cmd, *extra])
        print(f"{cmd + ' ' + ' '.join(extra):14s} {'ok' if r.returncode == 0 else 'FAILED'}")
        if r.returncode:
            problems.append(f"{cmd} {' '.join(extra)}: {(r.stdout + r.stderr).strip().splitlines()[-1:]}")
    return problems


def _tier_f(work: Path) -> list[str]:
    from ..tier_f import taskio
    mirror = work / "tier_f_mirror" / "grc-bench"
    (mirror / "items").mkdir(parents=True)
    for name in ("src", "splits", "reports"):
        shutil.copytree(REPO / name, mirror / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "items" / "tier_f", mirror / "items" / "tier_f",
                    ignore=shutil.ignore_patterns("__pycache__"))
    problems = []
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(mirror / "src")}
    for d in taskio.task_dirs():
        rel = d.relative_to(REPO)
        r = subprocess.run([sys.executable, str(mirror / rel / "build.py")], cwd=mirror, env=env,
                           capture_output=True, text=True)
        if r.returncode:
            problems.append(f"tier-f {d.name}: build.py failed: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
            continue
        built = {f: h for f, h in _tree(mirror / rel).items() if f not in NOT_BUILT}
        held = {f: h for f, h in _tree(d).items() if f not in NOT_BUILT}
        diff = sorted(f for f in set(built) | set(held) if built.get(f) != held.get(f))
        problems += [f"tier-f {d.name}: differs from build.py: {f}" for f in diff]
    print(f"tier-f         {len(taskio.task_dirs()):3d} tasks rebuilt in a mirror: "
          f"{'all identical' if not problems else f'{len(problems)} problem(s)'}")
    return problems


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb build-all", description=__doc__.splitlines()[0])
    ap.add_argument("--tier-f", action="store_true", help="also rebuild every Tier-F task in a mirror (minutes)")
    ap.add_argument("--keep", action="store_true", help=f"keep the scratch roots under {WORK / 'build_all'}")
    args = ap.parse_args(argv)
    work = WORK / "build_all"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        problems = _generators(work)
        if args.tier_f:
            problems += _tier_f(work)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    for p in problems:
        print("  " + p)
    print("build-all: " + ("OK" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0

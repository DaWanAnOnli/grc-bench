"""`./gb agent-package`: the shareable zip of the headless GRC-Agent runner (`./gb agent`), for a machine
without GRC-Bench (handoff/agent_runner.md).

Contents (top folder grc-agent-headless/):
  src/grcbench/...   the modules the runner reaches from its entry points (ENTRIES): every import is
                     followed, imports inside functions included (`closure`)
  agent, setup.sh, README.md, TUTORIAL.md, examples/
                     from handoff/agent_package/
  VERSION.txt        the GRC-Bench commit (and whether the tree had uncommitted changes), the pins, the
                     build time and the SHA-256 of every packaged file

The package never contains benchmark items, answer keys, golden solutions, reports, review sheets or
splits. Two guards make the build fail instead:
  * a module under CONTENT_MODULES (the generators and audits that hold benchmark content) is refused;
  * no packaged file may name a benchmark item (runlog.log.benchmark_item_ids: every Tier-K, F and A
    item id of items/).
`agent` and `setup.sh` keep their executable bit in the zip.

`--verify` unzips the package into a fresh folder in the home directory (the sandbox hides /tmp and
/mnt), runs its setup.sh there with this machine's GRC-Agent (setup.sh ends with the self-test, which
uses scripted models) and removes the folder.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path

from ..common.paths import GNURADIO_VERSION, GRC_AGENT_COMMIT, HANDOFF, ITEMS, PROJECT, REPO

NAME = "grc-agent-headless"
SRC = REPO / "src"
STATIC = HANDOFF / "agent_package"
STATIC_FILES = ("README.md", "TUTORIAL.md", "agent", "setup.sh", "examples/request.json", "examples/continue.json")
EXECUTABLE = {"agent", "setup.sh"}
OUT_DIR = PROJECT / "share"
ENTRIES = ("grcbench.__main__", "grcbench.tier_a.adhoc", "grcbench.tier_a.runner", "grcbench.tier_a._launch",
           "grcbench.tier_a._grc_view", "grcbench.common.environment")
CONTENT_MODULES = ("grcbench/tier_a/first_set.py", "grcbench/tier_a/second_set.py", "grcbench/tier_a/spec_drafts.py",
                   "grcbench/tier_a/agent_package.py",
                   "grcbench/tier_k/", "grcbench/tier_f/compose.py", "grcbench/tier_f/seeds.py",
                   "grcbench/splits/", "grcbench/labels/", "grcbench/contamination/", "grcbench/stats/")
_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]*[A-Za-z0-9_]")


class PackageError(RuntimeError):
    """The package would contain something it must not, or a file is missing."""


def _module_file(mod: str, src: Path) -> Path | None:
    base = src / Path(*mod.split("."))
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return base.with_suffix(".py") if base.with_suffix(".py").is_file() else None


def _imports(mod: str, path: Path) -> set[str]:
    pkg = mod if path.name == "__init__.py" else mod.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = pkg.split(".")
                target = ".".join(parts[: len(parts) - (node.level - 1)] + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            out.add(target)
            out |= {f"{target}.{a.name}" for a in node.names}       # `from package import module`
    return {m for m in out if m == "grcbench" or m.startswith("grcbench.")}


def closure(entries: tuple[str, ...] = ENTRIES, src: Path = SRC) -> list[str]:
    """The source files (relative to src/, posix) of every grcbench module reachable from `entries`."""
    seen: dict[str, Path] = {}
    todo = list(entries)
    while todo:
        mod = todo.pop()
        path = _module_file(mod, src)
        if mod in seen or path is None:
            continue
        seen[mod] = path
        parts = mod.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]       # parent packages
        todo += sorted(_imports(mod, path))
    return sorted(p.relative_to(src).as_posix() for p in seen.values())


def check_content(files: dict[str, bytes], item_ids: set[str]) -> None:
    """Raise PackageError if a packaged path is a content module or a file names a benchmark item."""
    bad = [p for p in files if any(p.startswith("src/" + c) for c in CONTENT_MODULES)]
    if bad:
        raise PackageError(f"benchmark-content modules would be packaged: {bad}")
    for path, data in files.items():
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        named = sorted(set(_TOKEN.findall(text)) & item_ids)
        if named:
            raise PackageError(f"{path} names benchmark items: {named[:5]}")


def _git(*args: str) -> str:
    out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False)
    return out.stdout.strip()


def build(out_dir: Path = OUT_DIR) -> Path:
    """Write the zip (module docstring) and return its path."""
    from ..runlog.log import benchmark_item_ids
    files: dict[str, bytes] = {}
    for rel in closure():
        files[f"src/{rel}"] = (SRC / rel).read_bytes()
    for rel in STATIC_FILES:
        path = STATIC / rel
        if not path.is_file():
            raise PackageError(f"missing package file {path}")
        files[rel] = path.read_bytes()
    check_content(files, set(benchmark_item_ids(ITEMS)))
    commit = _git("rev-parse", "--short=7", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain", "--", "src", "handoff/agent_package"))
    lines = [NAME, f"built {time.strftime('%Y-%m-%dT%H:%M:%S%z')} from GRC-Bench commit {commit}"
             f"{' with uncommitted changes' if dirty else ''}",
             f"pins: GRC-Agent {GRC_AGENT_COMMIT}, GNU Radio {GNURADIO_VERSION}", "", "sha256 of every file:"]
    lines += [f"{hashlib.sha256(data).hexdigest()}  {rel}" for rel, data in sorted(files.items())]
    files["VERSION.txt"] = ("\n".join(lines) + "\n").encode("utf-8")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{NAME}-{commit}{'-dirty' if dirty else ''}.zip"
    stamp = time.localtime()[:6]
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel, data in sorted(files.items()):
            info = zipfile.ZipInfo(f"{NAME}/{rel}", date_time=stamp)
            info.external_attr = (0o100755 if rel in EXECUTABLE else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return target


def verify(zip_path: Path) -> int:
    """Unzip into a fresh home-directory folder, run its setup.sh, remove the folder; setup's exit code."""
    dest = Path.home() / f"{NAME}-verify-{uuid.uuid4().hex[:8]}"
    dest.mkdir()
    try:
        if shutil.which("unzip"):
            subprocess.run(["unzip", "-q", str(zip_path), "-d", str(dest)], check=True)
        else:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
                for info in zf.infolist():
                    os.chmod(dest / info.filename, (info.external_attr >> 16) & 0o777)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}   # the package's own src only
        proc = subprocess.run(["bash", str(dest / NAME / "setup.sh")], env=env, capture_output=True, text=True,
                              check=False)
        print(proc.stdout[-6000:] + proc.stderr[-2000:])
        return proc.returncode
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gb agent-package", description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(OUT_DIR), help=f"folder for the zip (default {OUT_DIR})")
    p.add_argument("--verify", action="store_true", help="unzip into a fresh folder and run its setup.sh")
    a = p.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        target = build(Path(a.out))
    except PackageError as exc:
        print(f"refused: {exc}")
        return 2
    with zipfile.ZipFile(target) as zf:
        n = len(zf.infolist())
    print(f"wrote {target} ({target.stat().st_size // 1024} KB, {n} files)")
    if a.verify:
        code = verify(target)
        print(f"verify: setup.sh in a fresh copy {'passed' if code == 0 else f'FAILED (exit {code})'}")
        return code
    return 0


if __name__ == "__main__":
    sys.exit(main())

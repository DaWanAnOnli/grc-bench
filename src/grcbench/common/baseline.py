"""Take or check the baseline manifests of everything GRC-Bench must never modify.

    gb baseline take          # once, before the first run
    gb baseline take --add    # add the manifest of a root that has none yet; the others stay as they are
    gb baseline check         # at any time; non-zero exit if anything changed

Full manifests live in ~/grc-bench-work/baseline (machine-specific); the summary goes into
reports/environment.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import manifest
from .io import read_json, sha256_file, write_json
from .paths import GRC_AGENT, HOME, PROJECT, WORK, ensure

ROOTS = {
    "grc_agent_repo": GRC_AGENT,
    "grc_agent_config": HOME / ".config" / "grc_agent",
    "gnuradio_user_cache": HOME / ".cache" / "gnuradio",
    "gnuradio_user_state": HOME / ".local" / "state" / "gnuradio",
    "project_data": PROJECT / "data",
    "documents_benchmark_and_evaluation": PROJECT / "benchmark and evaluation",
    "project_new_data": PROJECT / "new_data",        # Qoherent's agent traces of 15-18 Oct
}
ROOT_FILES = sorted(p.name for p in PROJECT.glob("*.docx"))   # proposals and data summaries
BASE_DIR = WORK / "baseline"


def _snapshot() -> dict[str, dict]:
    snaps = {}
    for name, root in ROOTS.items():
        snaps[name] = manifest.build(root) if root.exists() else {"root": str(root), "n_files": 0,
                                                                  "aggregate_sha256": None, "files": {}}
    docs = {"root": str(PROJECT), "files": {}}
    for fname in ROOT_FILES:
        st = (PROJECT / fname).stat()
        docs["files"][fname] = [st.st_size, st.st_mtime_ns, sha256_file(PROJECT / fname)]
    docs["n_files"] = len(docs["files"])
    snaps["project_root_documents"] = docs
    return snaps


def take() -> None:
    ensure(BASE_DIR)
    snaps = _snapshot()
    for name, snap in snaps.items():
        write_json(BASE_DIR / f"{name}.json", snap)
    summary = {name: {"root": s["root"], "n_files": s["n_files"],
                      "aggregate_sha256": s.get("aggregate_sha256")} for name, s in snaps.items()}
    print(json.dumps(summary, indent=1))


def add_missing() -> list[str]:
    """Write the manifest of every root that has none in the baseline yet; never touch existing ones."""
    ensure(BASE_DIR)
    added = []
    for name, snap in _snapshot().items():
        if not (BASE_DIR / f"{name}.json").exists():
            write_json(BASE_DIR / f"{name}.json", snap)
            added.append(f"{name} ({snap['n_files']} files)")
    return added


def check() -> int:
    snaps = _snapshot()
    bad = 0
    for name, snap in snaps.items():
        if not (BASE_DIR / f"{name}.json").exists():
            print(f"{name:40s} NOT IN THE BASELINE YET (gb baseline take --add)")
            bad += 1
            continue
        before = read_json(BASE_DIR / f"{name}.json")
        diff = manifest.compare(before, snap)
        n = sum(len(v) for v in diff.values())
        status = "unchanged" if n == 0 else f"CHANGED ({n} files)"
        print(f"{name:40s} {status}")
        for kind, paths in diff.items():
            for p in paths[:20]:
                print(f"    {kind}: {p}")
        bad += n
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "check"
    if cmd == "take" and "--add" in argv:
        added = add_missing()
        print("added: " + (", ".join(added) if added else "nothing (every root is in the baseline)"))
        return 0
    if cmd == "take":
        if (BASE_DIR / "grc_agent_repo.json").exists() and "--force" not in argv:
            print(f"baseline already taken in {BASE_DIR}; use --force to replace it")
            return 1
        take()
        return 0
    return check()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""File-tree manifests that prove read-only inputs were not modified.

A manifest records every file under a root (including git-ignored and hidden files): relative path,
size, modification time and, outside virtual environments, a SHA-256 of the content. `compare`
reports any added, removed or changed file. `git status` is not enough: it cannot see ignored files.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .io import sha256_file

HASH_SKIP_PARTS = {".venv", "__pycache__", ".git"}   # recorded by size + mtime only
MAX_HASH_BYTES = 200 * 1024 * 1024


def build(root: Path) -> dict[str, Any]:
    root = Path(root)
    files: dict[str, list[Any]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            try:
                st = p.lstat()
            except OSError:
                continue
            digest = None
            if p.is_file() and not p.is_symlink() and st.st_size <= MAX_HASH_BYTES \
                    and not (set(Path(rel).parts) & HASH_SKIP_PARTS):
                digest = sha256_file(p)
            files[rel] = [st.st_size, st.st_mtime_ns, digest]
    agg = hashlib.sha256()
    for rel in sorted(files):
        agg.update(f"{rel}\0{files[rel][0]}\0{files[rel][1]}\0{files[rel][2]}\n".encode())
    return {"root": str(root), "n_files": len(files), "aggregate_sha256": agg.hexdigest(), "files": files}


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[str]]:
    b, a = before["files"], after["files"]
    return {
        "added": sorted(set(a) - set(b)),
        "removed": sorted(set(b) - set(a)),
        "changed": sorted(k for k in set(a) & set(b) if a[k] != b[k]),
    }

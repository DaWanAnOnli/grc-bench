"""Deterministic reading and writing, so that re-running a generator gives byte-identical files.

A file written under reports/ is published: its private device paths are redacted as it is written
(common/privacy.py)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def dumps(obj: Any) -> str:
    """Canonical JSON: sorted keys, no trailing spaces, UTF-8 kept readable."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ": "))


def _published(path: Path) -> bool:
    from . import paths
    try:
        Path(path).resolve().relative_to(Path(paths.REPORTS).resolve())
    except ValueError:
        return False
    return True


def _clean(path: Path, obj: Any) -> Any:
    if not _published(path):
        return obj
    from .privacy import redact
    return redact(obj)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(_clean(path, row), sort_keys=True, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(_clean(path, obj), sort_keys=True, ensure_ascii=False, indent=1) + "\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _clean(path, text)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

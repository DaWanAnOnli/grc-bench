"""The declared review runs (eval plan §6: test items reach a model only in a declared, logged run).

One entry per agent run in reviews/review_log.jsonl: agent number, role, sheets, row count, model,
date, the SHA-256 of the brief, and once the run ends the SHA-256 of its outputs. The review uses at
most 30 agent runs in total; a 31st declaration is refused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from grcbench.common.paths import REPO

LOG = REPO / "reviews" / "review_log.jsonl"
MODEL = "claude-opus-5-5"
MAX_AGENTS = 30


def _entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in entries),
                    encoding="utf-8")


def declare(path: Path, *, agent: int, role: str, sheets: list[str], rows: int, brief_text: str,
            date: str, model: str = MODEL) -> dict:
    entries = _entries(path)
    if any(e["agent"] == agent for e in entries):
        raise ValueError(f"agent {agent} is already declared")
    if len(entries) >= MAX_AGENTS:
        raise ValueError(f"the review is capped at {MAX_AGENTS} agent runs; {len(entries)} are declared")
    e = {"agent": agent, "role": role, "sheets": list(sheets), "rows": rows, "model": model, "date": date,
         "brief_sha256": hashlib.sha256(brief_text.encode("utf-8")).hexdigest(), "output_sha256": None,
         "rows_written": None}
    _write(path, entries + [e])
    return e


def finish(path: Path, *, agent: int, outputs: Iterable[Path]) -> dict:
    entries = _entries(path)
    h = hashlib.sha256()
    n = 0
    for out in sorted(Path(p) for p in outputs):
        data = out.read_bytes()
        h.update(data)
        n += sum(1 for line in data.decode("utf-8").splitlines() if line.strip())
    for e in entries:
        if e["agent"] == agent:
            e["output_sha256"], e["rows_written"] = h.hexdigest(), n
            _write(path, entries)
            return e
    raise ValueError(f"agent {agent} was never declared")

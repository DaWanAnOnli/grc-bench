"""Tier-F task directories: items/tier_f/<split>/<task_id>/

    task.json          the item record (schema grcbench.common.schemas.TIER_F) incl. instruction,
                       observation contract and V5 oracle
    start.grc          starting flowgraph for modify / repair / parameter-tune tasks (absent for build)
    reference.grc      the golden solution, in compact form (eval plan §7.1: answers fit the 8K cap)
    variants/*.grc     at least two further valid solutions (equivalent-variant acceptance tests)
    gates.json         the latest gate results (written by `./gb tier-f gates`)

The instruction and every file a model sees never contain the canary string.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common.canary import CANARY, CANARY_GUID
from ..common.io import read_json, write_json, write_text
from ..common.paths import ITEMS
from ..common.schemas import validate

TIER_F_DIR = ITEMS / "tier_f"


def task_dirs(split: str | None = None) -> list[Path]:
    splits = [split] if split else ["dev", "test"]
    out = []
    for s in splits:
        root = TIER_F_DIR / s
        if root.exists():
            out += sorted(p for p in root.iterdir() if (p / "task.json").exists())
    return out


def load(task_dir: Path) -> dict[str, Any]:
    task = read_json(task_dir / "task.json")
    task["_dir"] = str(task_dir)
    return task


def read_grc(task_dir: Path, name: str) -> str | None:
    path = task_dir / name
    return path.read_text(encoding="utf-8") if path.exists() else None


def variants(task_dir: Path) -> list[Path]:
    return sorted((task_dir / "variants").glob("*.grc")) if (task_dir / "variants").exists() else []


def save(task_dir: Path, record: dict[str, Any], files: dict[str, str]) -> list[str]:
    """Write a task; returns schema errors (the task is written only if there are none)."""
    record = {k: v for k, v in record.items() if not k.startswith("_")}
    record.setdefault("canary", CANARY)
    errors = validate(record)
    for name, text in files.items():
        if CANARY_GUID in text:
            errors.append(f"{name} contains the canary (files shown to models must not)")
    if CANARY_GUID in record.get("instruction", ""):
        errors.append("instruction contains the canary")
    if errors:
        return errors
    task_dir.mkdir(parents=True, exist_ok=True)
    write_json(task_dir / "task.json", record)
    for name, text in files.items():
        write_text(task_dir / name, text)
    return []

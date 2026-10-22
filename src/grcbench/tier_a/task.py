"""Tier-A task directories: items/tier_a/<split>/<task_id>/ (eval plan §4.4, §5.4).

    task.json     the item record (schema grcbench.common.schemas.TIER_A)
    start/        the start state: every file in it is copied into the episode's project folder
                  (the agent's working directory) before the episode starts; nothing outside
                  start/ (references, grader files) is ever shown to the agent

Fields of task.json the runner reads (the rest is for the grader and the release):
    instruction     the first user message of the episode
    start_state     {"active_graph": "<path inside start/>" | null}
                    the graph GRC has open when the episode starts (opened like File > Open);
                    null = GRC's File > New (an untitled graph that was never saved)
    scripted_user   the scripted user (grcbench.tier_a.user; rules, default reply, max_turns)
    acceptance      end-state checks, read by the grader (eval plan §4.4 "Grading"); opaque here

Checks (task_errors): the record matches the TIER_A schema; start/ exists and every start file is a
regular file inside it; the active graph is one of them and parses as a .grc (V1); the scripted user
is well-formed; and the canary string appears in no text the agent can see (instruction, start
files, scripted replies), since the canary stays out of prompts (eval plan §6).

Training-session collection (proposal §4.4) uses the same directory format with "split": "train";
those specs live outside items/ and are validated against the same schema with the split relaxed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from ..common.canary import CANARY_GUID
from ..common.io import read_json
from ..common.paths import ITEMS
from ..common.schemas import TIER_A
from .user import ScriptedUser

TASK_FILE = "task.json"
START_DIR = "start"
TIER_A_DIR = ITEMS / "tier_a"
SPLITS = ("dev", "test")


class TaskError(ValueError):
    """A Tier-A task directory is not usable."""


def _schema_for(split: Any) -> dict[str, Any]:
    if split != "train":
        return TIER_A
    relaxed = copy.deepcopy(TIER_A)
    relaxed["allOf"][0]["properties"]["split"] = {"enum": ["train"]}
    # a training-side request may be short ("add a throttle"); benchmark instructions keep minLength 20
    relaxed["allOf"][1]["properties"]["instruction"] = {"type": "string", "minLength": 1}
    return relaxed


@dataclass(frozen=True)
class Task:
    id: str
    split: str
    version: int
    category: str
    instruction: str
    start_state: dict[str, Any]
    scripted_user: dict[str, Any]
    acceptance: dict[str, Any]
    task_dir: Path
    record: dict[str, Any] = field(repr=False, compare=False)

    @property
    def start_dir(self) -> Path:
        return self.task_dir / START_DIR

    @property
    def start_files(self) -> list[Path]:
        """Files copied into the project folder, as paths relative to start/ (sorted)."""
        root = self.start_dir
        return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file()) if root.is_dir() else []

    @property
    def active_graph(self) -> Path | None:
        """The start graph opened as GRC's active page, relative to start/; None = untitled."""
        name = self.start_state.get("active_graph")
        return Path(name) if name else None

    def user(self) -> ScriptedUser:
        return ScriptedUser.from_spec(self.scripted_user)

    def to_dict(self) -> dict[str, Any]:
        """The record plus its directory (what the episode process receives)."""
        return {"record": self.record, "task_dir": str(self.task_dir)}


def task_errors(record: dict[str, Any], task_dir: Path) -> list[str]:
    """Human-readable problems with a task record and its directory (empty if usable)."""
    errors: list[str] = []
    validator = jsonschema.Draft202012Validator(_schema_for(record.get("split")))
    for e in sorted(validator.iter_errors(record), key=lambda e: list(e.absolute_path)):
        errors.append(f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}")
    if errors:
        return errors
    start = task_dir / START_DIR
    active = (record.get("start_state") or {}).get("active_graph")
    if not start.is_dir():
        if active:
            errors.append(f"start_state.active_graph is {active!r} but {START_DIR}/ does not exist")
    else:
        root = start.resolve()
        for p in sorted(start.rglob("*")):
            if p.is_symlink() or not (p.is_file() or p.is_dir()):
                errors.append(f"{p.relative_to(task_dir)}: start files must be regular files")
            elif not p.resolve().is_relative_to(root):
                errors.append(f"{p.relative_to(task_dir)}: points outside {START_DIR}/")
            elif p.is_file():
                try:
                    if CANARY_GUID in p.read_text(encoding="utf-8", errors="replace"):
                        errors.append(f"{p.relative_to(task_dir)} contains the canary (files shown to "
                                      "the agent must not)")
                except OSError as exc:
                    errors.append(f"{p.relative_to(task_dir)}: {exc}")
    if active:
        path = start / active
        if Path(active).is_absolute() or ".." in Path(active).parts:
            errors.append("start_state.active_graph must be a relative path inside start/")
        elif not path.is_file():
            errors.append(f"start_state.active_graph {active!r} is not a file in {START_DIR}/")
        elif path.suffix != ".grc":
            errors.append(f"start_state.active_graph {active!r} is not a .grc file")
        else:
            from ..tier_f import grcfile
            ok, errs, _ = grcfile.v1_parse(path.read_text(encoding="utf-8"))
            if not ok:
                errors.append(f"start_state.active_graph does not parse: {errs[:3]}")
    if CANARY_GUID in record.get("instruction", ""):
        errors.append("instruction contains the canary")
    try:
        user = ScriptedUser.from_spec(record.get("scripted_user"))
        if any(CANARY_GUID in r for r in user.replies()):
            errors.append("a scripted-user reply contains the canary")
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(f"scripted_user: {exc}")
    return errors


def load_task(task_dir: Path | str, *, validate: bool = True) -> Task:
    """Load (and by default validate) one task directory; raises TaskError if it is unusable."""
    task_dir = Path(task_dir).resolve()
    if not (task_dir / TASK_FILE).is_file():
        raise TaskError(f"{task_dir} has no {TASK_FILE}")
    record = read_json(task_dir / TASK_FILE)
    if validate:
        errors = task_errors(record, task_dir)
        if errors:
            raise TaskError(f"{task_dir.name}: " + "; ".join(errors))
    return Task(id=record["id"], split=record["split"], version=int(record["version"]),
                category=record["category"], instruction=record["instruction"],
                start_state=dict(record.get("start_state") or {}),
                scripted_user=dict(record.get("scripted_user") or {}),
                acceptance=dict(record.get("acceptance") or {}), task_dir=task_dir, record=record)


def task_from_dict(data: dict[str, Any]) -> Task:
    """Inverse of Task.to_dict (inside the episode process; the host has validated it)."""
    record, task_dir = data["record"], Path(data["task_dir"])
    return Task(id=record["id"], split=record["split"], version=int(record["version"]),
                category=record["category"], instruction=record["instruction"],
                start_state=dict(record.get("start_state") or {}),
                scripted_user=dict(record.get("scripted_user") or {}),
                acceptance=dict(record.get("acceptance") or {}), task_dir=task_dir, record=record)


def task_dirs(split: str | None = None, root: Path = TIER_A_DIR) -> list[Path]:
    """Task directories of the benchmark (both splits by default), sorted by id."""
    out: list[Path] = []
    for s in ([split] if split else list(SPLITS)):
        base = root / s
        if base.is_dir():
            out += sorted(p for p in base.iterdir() if (p / TASK_FILE).is_file())
    return out


def iter_tasks(split: str | None = None, root: Path = TIER_A_DIR) -> list[Task]:
    return [load_task(d) for d in task_dirs(split, root)]

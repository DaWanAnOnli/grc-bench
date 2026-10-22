"""The append-only test-run log and the guard in front of every run (eval plan §4.1, §7.2).

Eval plan §4.1: "The test split is run only for configurations on the declared run list (§7.2)" and
"Every test run is written to the test-run log, with its model, settings, code version, date and
reason." Tier-A runs (and any other harness) call `start_run` before anything executes:

  * guard (`guard`)
      evaluation mode, test split  the configuration id must be declared in
                                   docs/preregistration/run_list.yaml; while that file does not
                                   exist, or its `status` is not "declared" (a draft awaiting the
                                   team's sign-off at G2), every test run is refused.
                                   Runs outside the list are refused, not run as "exploratory".
      evaluation mode, dev split   allowed (the dev split is for every choice; it is not logged here)
      collection mode              (training-session collection, proposal §4.4) refuses every test
                                   or dev item: by its split, and by its id if a benchmark item of
                                   any tier carries the same id, whatever split the spec claims.
  * entry (`append`): one JSON line in runs/test_run_log.jsonl, written and fsynced BEFORE the run
    starts, so a run that crashes or is aborted is still on record. The file is only ever opened
    for appending; nothing here rewrites or removes a line.

Entry fields: time (UTC, ISO 8601), config_id, model, settings, git_commit (`git rev-parse HEAD` of
this repository, or "uncommitted"), git_dirty (tracked files changed since that commit; null if git
cannot tell), split, items, reason, plus tier and mode.

run_list.yaml shape read here (WS7 owns the file; any of these forms is accepted):
    status: declared           # optional; any other value (e.g. "draft") declares nothing
    configurations:            # or "runs:"; or a top-level list
      - id: nmc-q4-b0          # or config_id:
        ...
or a top-level mapping keyed by configuration id.

    ./gb runlog show [--log PATH]        print the log
    ./gb runlog check <config_id>        is the configuration on the declared run list?
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from ..common.paths import DOCS, ITEMS, REPO, RUNS

LOG = RUNS / "test_run_log.jsonl"
RUN_LIST = DOCS / "preregistration" / "run_list.yaml"
MODES = ("evaluation", "collection")
REQUIRED = ("time", "config_id", "model", "settings", "git_commit", "split", "items", "reason")


class RunRefused(RuntimeError):
    """The guard refused a run; nothing was started and nothing was logged."""


# --------------------------------------------------------------------------------------------
# Code version
# --------------------------------------------------------------------------------------------

def _git(*args: str, repo: Path = REPO) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None


def git_commit(repo: Path = REPO) -> str:
    """`git -C <repo> rev-parse HEAD`, or "uncommitted" when there is no commit to name."""
    res = _git("rev-parse", "HEAD", repo=repo)
    head = res.stdout.strip() if res is not None and res.returncode == 0 else ""
    return head if len(head) == 40 else "uncommitted"


def git_dirty(repo: Path = REPO) -> bool | None:
    """Whether tracked files differ from HEAD (None if git cannot tell)."""
    res = _git("status", "--porcelain", "--untracked-files=no", repo=repo)
    if res is None or res.returncode != 0:
        return None
    return bool(res.stdout.strip())


# --------------------------------------------------------------------------------------------
# The declared run list and the benchmark item ids
# --------------------------------------------------------------------------------------------

def declared_config_ids(run_list: Path = RUN_LIST) -> set[str] | None:
    """Configuration ids on the declared run list; None if the list does not exist yet or is not
    declared (its `status` is set to anything but "declared")."""
    import yaml
    if not Path(run_list).is_file():
        return None
    data = yaml.safe_load(Path(run_list).read_text(encoding="utf-8"))
    if isinstance(data, dict) and str(data.get("status", "declared")) != "declared":
        return None
    entries: Iterable[Any]
    if isinstance(data, dict) and isinstance(data.get("configurations", data.get("runs")), list):
        entries = data.get("configurations", data.get("runs"))
    elif isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        return {str(k) for k in data if k not in ("configurations", "runs", "status")}
    else:
        return set()
    ids: set[str] = set()
    for entry in entries:
        if isinstance(entry, dict):
            value = entry.get("id", entry.get("config_id"))
            if value is not None:
                ids.add(str(value))
        elif isinstance(entry, str):
            ids.add(entry)
    return ids


def benchmark_item_ids(items_root: Path = ITEMS) -> dict[str, str]:
    """{item id: split} for every Tier-A and Tier-F item directory and Tier-K item file record."""
    ids: dict[str, str] = {}
    for tier in ("tier_a", "tier_f"):
        for split in ("dev", "test"):
            base = Path(items_root) / tier / split
            if not base.is_dir():
                continue
            for d in sorted(base.iterdir()):
                if not d.is_dir():
                    continue
                ids[d.name] = split
                task = d / "task.json"
                if task.is_file():
                    try:
                        ids[str(json.loads(task.read_text(encoding="utf-8")).get("id", d.name))] = split
                    except (OSError, json.JSONDecodeError):
                        pass
    for split in ("dev", "test"):
        base = Path(items_root) / "tier_k" / split
        for f in sorted(base.glob("*.jsonl")) if base.is_dir() else []:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            ids[str(json.loads(line)["id"])] = split
                        except (json.JSONDecodeError, KeyError):
                            pass
    return ids


# --------------------------------------------------------------------------------------------
# Guard and log
# --------------------------------------------------------------------------------------------

def guard(*, split: str, config_id: str, items: Iterable[str] = (), item_splits: Iterable[str] = (),
          mode: str = "evaluation", run_list: Path = RUN_LIST, items_root: Path = ITEMS) -> None:
    """Raise RunRefused unless this run may start (module docstring)."""
    if mode not in MODES:
        raise RunRefused(f"unknown mode {mode!r}; expected one of {MODES}")
    splits = set(item_splits) | {split}
    if mode == "collection":
        bad = sorted(splits & {"test", "dev"})
        if bad:
            raise RunRefused(f"collection mode refuses benchmark items: split {', '.join(bad)}")
        known = benchmark_item_ids(items_root)
        clash = sorted(i for i in items if i in known)
        if clash:
            raise RunRefused(f"collection mode refuses benchmark items: {', '.join(clash[:10])}")
        return
    if len(splits) != 1:
        raise RunRefused(f"one run covers one split; got {sorted(splits)}")
    if split == "test":
        declared = declared_config_ids(run_list)
        if declared is None:
            raise RunRefused(f"test-split runs are refused: the declared run list {run_list} does not exist "
                             "yet or is not declared (status: declared is set when the pre-registration is "
                             "signed off, eval plan §4.1)")
        if config_id not in declared:
            raise RunRefused(f"test-split runs are refused: configuration {config_id!r} is not on the "
                             f"declared run list {run_list}")
    elif split != "dev":
        raise RunRefused(f"evaluation runs use the dev or test split, not {split!r}")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_entry(*, config_id: str, model: Any, settings: dict[str, Any], split: str, items: list[str],
               reason: str, tier: str = "A", mode: str = "evaluation", repo: Path = REPO) -> dict[str, Any]:
    if not reason or not reason.strip():
        raise RunRefused("every logged run needs a reason")
    return {"time": utc_now(), "config_id": config_id, "model": model, "settings": settings,
            "git_commit": git_commit(repo), "git_dirty": git_dirty(repo), "split": split,
            "items": list(items), "reason": reason.strip(), "tier": tier, "mode": mode}


def append(entry: dict[str, Any], log_path: Path = LOG) -> dict[str, Any]:
    """Append one entry (one JSON line) and fsync it. Append-only: the file is never rewritten."""
    missing = [k for k in REQUIRED if k not in entry]
    if missing:
        raise ValueError(f"run-log entry lacks {missing}")
    line = json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        if os.fstat(fd).st_size > 0:
            with open(log_path, "rb") as fh:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    os.write(fd, b"\n")
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return entry


def start_run(*, split: str, config_id: str, model: Any, settings: dict[str, Any], items: list[str],
              reason: str, item_splits: Iterable[str] = (), tier: str = "A", mode: str = "evaluation",
              log_path: Path = LOG, run_list: Path = RUN_LIST, items_root: Path = ITEMS) -> dict[str, Any] | None:
    """Guard, then (for the test split) log the run BEFORE it starts. Returns the entry written, or
    None when the run is allowed but not logged (dev split, collection)."""
    guard(split=split, config_id=config_id, items=items, item_splits=item_splits, mode=mode,
          run_list=run_list, items_root=items_root)
    if mode == "evaluation" and split == "test":
        return append(make_entry(config_id=config_id, model=model, settings=settings, split=split,
                                 items=items, reason=reason, tier=tier, mode=mode), log_path)
    return None


def read_log(log_path: Path = LOG) -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="gb runlog")
    sub = p.add_subparsers(dest="cmd")
    show = sub.add_parser("show", help="print the test-run log")
    show.add_argument("--log", default=str(LOG))
    chk = sub.add_parser("check", help="is a configuration on the declared run list?")
    chk.add_argument("config_id")
    chk.add_argument("--run-list", default=str(RUN_LIST))
    a = p.parse_args(argv or ["show"])
    if a.cmd == "check":
        declared = declared_config_ids(Path(a.run_list))
        if declared is None:
            print(f"no declared run list at {a.run_list} (missing, or its status is not 'declared'): "
                  "every test-split run is refused")
            return 1
        ok = a.config_id in declared
        print(f"{a.config_id}: {'declared' if ok else 'NOT on the declared run list'}")
        return 0 if ok else 1
    entries = read_log(Path(a.log))
    for e in entries:
        print(f"{e['time']}  {e['config_id']:24s} {e['split']:5s} {len(e['items']):4d} items  "
              f"{e['git_commit'][:12]}  {e['reason']}")
    print(f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'} in {a.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

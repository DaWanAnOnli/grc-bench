"""Decided edits as patches: the coordinator writes each edit the review decided on a task as one or more
machine-applied patches (reviews/<tier>_patches.json), and the task's generator applies them when it
writes the task, so a re-run keeps every decided edit.

A patch: {"item": <task id>, "file": "task.json" (default), "path": [key or index, ...], "op": ..., "edit":
<the decided edit it implements>}, with
  set           "value": replace (or create the last key of) the value at `path`
  delete        remove the value at `path`
  append        "value": append to the list at `path`
  replace_text  "old", "new": replace one occurrence of `old` in the string at `path` (refused if absent)
  append_text   "value": append to the string at `path`
  insert_between  on a GRC graph dict at `path`: "block" is appended and the connection "src" -> "dst"
                ("name:port") becomes src -> block:0 and block:0 -> dst (refused if that connection is absent)
  rewire        on a GRC graph dict at `path`: the connection "old" ("a:0->b:0") becomes "new"
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

OPS = ("set", "delete", "append", "replace_text", "append_text", "insert_between", "rewire")


def _end(text: str) -> list[str]:
    name, port = text.strip().rsplit(":", 1)
    return [name.strip(), port.strip()]


def _conn_index(graph: Mapping[str, Any], src: str, dst: str, item: str) -> int:
    want = _end(src) + _end(dst)
    for i, c in enumerate(graph.get("connections") or []):
        if [str(x) for x in c] == want:
            return i
    raise ValueError(f"{item}: connection {src} -> {dst} not found")


def _graph_op(graph: dict[str, Any], p: Mapping[str, Any], item: str) -> None:
    if p["op"] == "insert_between":
        i = _conn_index(graph, p["src"], p["dst"], item)
        name = p["block"]["name"]
        graph["blocks"].append(copy.deepcopy(p["block"]))
        graph["connections"][i:i + 1] = [_end(p["src"]) + [name, "0"], [name, "0"] + _end(p["dst"])]
    else:
        old_src, old_dst = p["old"].split("->")
        new_src, new_dst = p["new"].split("->")
        i = _conn_index(graph, old_src, old_dst, item)
        graph["connections"][i] = _end(new_src) + _end(new_dst)


def _walk(node: Any, path: Sequence[Any], item: str) -> Any:
    for step in path:
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"{item}: path {list(path)} does not exist (at {step!r})") from None
    return node


def apply(record: Mapping[str, Any], patches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = copy.deepcopy(dict(record))
    for p in patches:
        op, path, item = p.get("op"), list(p["path"]), p.get("item", "?")
        if op not in OPS:
            raise ValueError(f"{item}: unknown op {op!r}")
        if op in ("insert_between", "rewire"):
            try:
                _graph_op(_walk(out, path, item), p, item)
            except ValueError:
                if not p.get("optional"):        # "optional": the graph may lack the connection (skip it)
                    raise
            continue
        parent = _walk(out, path[:-1], item)
        last = path[-1]
        if op == "set":
            if isinstance(parent, list):
                _walk(parent, [last], item)
            parent[last] = copy.deepcopy(p["value"])
        elif op == "delete":
            _walk(parent, [last], item)
            del parent[last]
        elif op == "append":
            target = _walk(parent, [last], item)
            if not isinstance(target, list):
                raise ValueError(f"{item}: path {path} is not a list")
            target.append(copy.deepcopy(p["value"]))
        elif op == "append_text":
            text = _walk(parent, [last], item)
            if not isinstance(text, str):
                raise ValueError(f"{item}: path {path} is not a string")
            parent[last] = text + p["value"]
        else:
            text = _walk(parent, [last], item)
            if not isinstance(text, str) or p["old"] not in text:
                raise ValueError(f"{item}: text to replace not found at {path}: {p['old'][:60]!r}")
            parent[last] = text.replace(p["old"], p["new"], 1)
    return out


def for_item(path: Path, item: str, file: str = "task.json") -> list[dict[str, Any]]:
    """The patches of one item and file, in the order written ([] when there is no patch file)."""
    if not Path(path).exists():
        return []
    return [p for p in json.loads(Path(path).read_text(encoding="utf-8"))
            if p["item"] == item and p.get("file", "task.json") == file]

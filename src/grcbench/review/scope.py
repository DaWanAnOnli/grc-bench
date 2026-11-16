"""The items of the expert review (eval plan §5.3): every GNU Radio question and every Tier-F and
Tier-A task, dev items included (docs/checklists/expert_review.md, "What is in scope").

A frozen or dev question is about GNU Radio when its text names GNU Radio, GRC, GNU Radio Companion, a
flowgraph, a GNU Radio module (gr-..., gr....), VOLK or pybind, or its task type is gnuradio_usage.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from grcbench.common.io import read_jsonl
from grcbench.common.paths import ITEMS

GNURADIO = re.compile(r"\bgnu\s*-?\s*radio\b|\bgrc\b|flow\s*-?graph|\bgr-[a-z]|\bgr\.[a-z]|\bgnuradio\b|"
                      r"\bcompanion\b|\bvolk\b|\bpybind", re.IGNORECASE)


def about_gnuradio(q: Mapping[str, Any]) -> bool:
    return q.get("task_type") == "gnuradio_usage" or bool(GNURADIO.search(q.get("question", "")))


def expert_items() -> list[dict[str, str]]:
    """One row per item: item_id, item_kind, split, packet_ref (the files the expert reads)."""
    rows: list[dict[str, str]] = []
    for q in read_jsonl(ITEMS / "tier_k" / "test" / "block_questions.jsonl"):
        rows.append({"item_id": q["id"], "item_kind": "gnuradio_block_question", "split": "test",
                     "packet_ref": "items/tier_k/test/block_questions.jsonl; items/tier_k/test/block_answers.jsonl"})
    for split, path, kind in (("test", "frozen_questions.jsonl", "gnuradio_frozen_question"),
                              ("dev", "questions.jsonl", "gnuradio_dev_question")):
        for q in read_jsonl(ITEMS / "tier_k" / split / path):
            if about_gnuradio(q):
                rows.append({"item_id": q["id"], "item_kind": kind, "split": split,
                             "packet_ref": f"items/tier_k/{split}/{path}; items/tier_k/{split}/rubrics.jsonl"})
    for tier, kind in (("tier_f", "tier_f_task"), ("tier_a", "tier_a_task")):
        for split in ("test", "dev"):
            for task in sorted((ITEMS / tier / split).glob("*/task.json")):
                t = json.loads(task.read_text(encoding="utf-8"))
                rows.append({"item_id": t.get("id") or task.parent.name, "item_kind": kind, "split": split,
                             "packet_ref": f"items/{tier}/{split}/{task.parent.name}/"})
    return rows


def removed_ids() -> set[str]:
    """The items in scope that the two-reviewer or rubric review removed: a task or block question whose
    own review status is `rejected`, a frozen or dev question whose rubric is `rejected`."""
    out = {q["id"] for q in read_jsonl(ITEMS / "tier_k" / "test" / "block_questions.jsonl")
           if (q.get("review") or {}).get("status") == "rejected"}
    for split in ("test", "dev"):
        path = ITEMS / "tier_k" / split / "rubrics.jsonl"
        if path.exists():
            out |= {r["question_id"] for r in read_jsonl(path) if (r.get("review") or {}).get("status") == "rejected"}
    for tier in ("tier_f", "tier_a"):
        for task in (ITEMS / tier).glob("*/*/task.json"):
            t = json.loads(task.read_text(encoding="utf-8"))
            if (t.get("review") or {}).get("status") == "rejected":
                out.add(t.get("id") or task.parent.name)
    return out

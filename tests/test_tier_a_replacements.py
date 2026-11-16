"""The replacements of the rejected Tier-A tasks (grcbench.tier_a.replacements; pre-registration §10)."""

from __future__ import annotations

import json
from collections import Counter

from grcbench.common.paths import ITEMS
from grcbench.tier_a import grader
from grcbench.tier_a import replacements as R

SPECS = {s["id"]: s for s in (make() for make in R.TEST_TASKS + R.DEV_TASKS)}


def test_one_replacement_per_rejected_task_plus_a_spare_per_category():
    assert set(SPECS) == R.REPLACEMENT_IDS
    for category, splits in R.SLOTS.items():
        for split, (ids, slots) in splits.items():
            assert all(SPECS[i]["category"] == category and f"-{split}-" in i for i in ids)
            rejected = [t for t in _tasks(split) if t["category"] == category
                        and t["review"]["status"] == "rejected" and t["id"] not in R.REPLACEMENT_IDS]
            assert slots == len(rejected)
    assert Counter(s["category"] for s in SPECS.values()) == Counter(knowledge=5, save_block=4)


def test_acceptance_is_well_formed_and_every_knowledge_fact_has_a_label():
    for s in SPECS.values():
        assert grader.acceptance_errors(s["acceptance"]) == [], s["id"]
        assert 8 <= len(s["golden"]["calls"]) <= 20, s["id"]
        if s["category"] == "knowledge":
            facts = s["acceptance"]["answer"]["facts"]
            assert all(f.get("label") for f in facts) and s["acceptance"]["graph_unchanged"]
            kinds = Counter(next(iter(k for k in ("graph", "block_yaml", "file") if k in f["source"])) for f in facts)
            assert kinds["block_yaml"] >= 1 and kinds["file"] >= 2 and kinds["graph"] >= 2, s["id"]
        else:
            assert s["acceptance"]["hier_block"]["block_id"] and "graph_unchanged" not in s["acceptance"]
            assert not any(b["id"] == "epy_block" for b in s["start"][1]["blocks"])     # nothing of the block yet


def test_every_replacement_starts_from_a_tier_f_graph_no_other_tier_a_task_uses():
    clusters = Counter(s["cluster_id"] for s in SPECS.values())
    assert max(clusters.values()) == 1
    others = {t["cluster_id"] for split in ("test", "dev") for t in _tasks(split) if t["id"] not in R.REPLACEMENT_IDS}
    assert not set(clusters) & others


def _tasks(split: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ITEMS / "tier_a" / split).glob("*/task.json"))]

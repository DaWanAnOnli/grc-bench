"""Replacements for the Tier-F tasks the review rejected (pre-registration §10; generate.replacement_plan)."""

from __future__ import annotations

import json
from collections import Counter

from grcbench.tier_f import generate as G
from grcbench.tier_f.taskio import task_dirs


def _rejected(split: str) -> list[dict]:
    out = []
    for d in task_dirs(split):
        t = json.loads((d / "task.json").read_text(encoding="utf-8"))
        if (t.get("review") or {}).get("status") == "rejected" and not G.is_replacement(t["id"]):
            out.append(t)
    return out


def test_one_replacement_per_rejected_task_in_its_family_plus_spares():
    for split, start, spares in (("test", 151, 3), ("dev", 46, 1)):
        rp = G.replacement_plan(split)
        rejected = _rejected(split)
        assert list(rp) == [f"gbf-{split}-{start + i:02d}" for i in range(len(rejected) + spares)]
        assert G.REPLACEMENTS[split]["slots"] == len(rejected)
        fams = Counter(f for f, _ in rp.values())
        assert fams - Counter(t["family"] for t in rejected) == Counter(G.REPLACEMENTS[split]["spare_families"])
        assert fams - Counter(G.REPLACEMENTS[split]["spare_families"]) == Counter(t["family"] for t in rejected)
        assert {lv for _, lv in rp.values()} == {"L1"}
        assert not set(rp) & set(G.plan(split))                     # new ids, after the planned ones
        assert all(G.is_replacement(t) for t in rp) and not any(G.is_replacement(t) for t in G.plan(split))


def test_the_replacement_plan_is_deterministic_and_the_original_plan_is_unchanged():
    assert G.replacement_plan("test") == G.replacement_plan("test")
    assert len(G.plan("test")) == 130 and len(G.plan("dev")) == 35
    assert G.all_plans("test") == {**G.plan("test"), **G.replacement_plan("test")}


def test_rejected_tasks_do_not_hold_cells():
    rejected = G.rejected_ids()
    assert {t["id"] for s in ("test", "dev") for t in _rejected(s)} <= rejected
    assert "gbf-test-101" in rejected and "gbf-test-21" not in rejected


def test_a_fallback_task_rebuilds_at_its_own_level():
    text = G.build_py("gbf-test-151", 2345, "test", level="L2")
    assert 'build_task("gbf-test-151", 2345, HERE, level="L2")' in text
    assert 'build_task("gbf-test-21", 7, HERE)' in G.build_py("gbf-test-21", 7, "test")

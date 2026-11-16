"""Tests for the 20 Tier-A test tasks built from their specs (`./gb tier-a second-set`): each built task is its
spec, word for word where the spec is written, and passed every gate (no GNU Radio needed: the task folders)."""

from __future__ import annotations

import json

from grcbench.common.paths import ITEMS
from grcbench.tier_a import second_set, spec_drafts

TEST = ITEMS / "tier_a" / "test"
FROM_SPEC = ("instruction", "category", "source", "cluster_id", "license", "disjointness")


def _task(task_id: str) -> dict:
    return json.loads((TEST / task_id / "task.json").read_text(encoding="utf-8"))


def test_every_spec_is_built_with_the_spec_s_own_wording():
    specs = spec_drafts.draft_records()
    assert len(specs) == 20 and {s["id"] for s in specs} == {f"gba-test-{i}" for i in range(11, 31)}
    from grcbench.review import patches
    from grcbench.tier_a import first_set
    for spec in specs:
        task = _task(spec["id"])
        # the spec's wording, with the review's decided edits to those fields (reviews/tier_a_patches.json)
        decided = [p for p in patches.for_item(first_set.REVIEWS / "tier_a_patches.json", spec["id"])
                   if p["path"][0] in FROM_SPEC]
        want = patches.apply({key: spec[key] for key in FROM_SPEC}, decided)
        for key in FROM_SPEC:
            assert task[key] == want[key], (spec["id"], key)
        assert task["provenance"]["generator"] == second_set.GENERATOR[0]
        assert task["review"] == first_set.review_of(spec["id"])


def test_every_built_task_passed_its_gates():
    for spec in spec_drafts.draft_records():
        gates = json.loads((TEST / spec["id"] / "gates.json").read_text(encoding="utf-8"))
        assert gates["passed"], spec["id"]
        assert gates["gates"]["golden_replay"]["passes"] == 10, spec["id"]


def test_the_review_sheet_lists_built_tasks_only_and_their_notes():
    rows = spec_drafts.built_rows()
    assert not spec_drafts.draft_rows()                        # every spec is built
    by_id = {r["task_id"]: r for r in rows}
    assert by_id["gba-test-20"]["build_notes"].startswith(spec_drafts.SPECS[next(
        i for i, s in enumerate(spec_drafts.SPECS) if s["id"] == "gba-test-20")]["build_notes"])
    for task_id, note in second_set.BUILT_NOTES.items():
        assert f"Built: {note}" in by_id[task_id]["build_notes"], task_id

"""The Tier-A first set (items/tier_a), the 20 test specs drafted as specs only, and the review sheet."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from grcbench.common.canary import CANARY
from grcbench.tier_a import first_set, spec_drafts
from grcbench.tier_a.task import task_dirs

REPO = Path(__file__).resolve().parents[1]
CHECKLIST = REPO / "docs" / "checklists" / "tier_a_task_review.md"
DECISIONS = ["reviewer_1_decision", "reviewer_1_edits", "reviewer_2_decision", "reviewer_2_edits", "final_decision"]


def _built() -> list[dict]:
    return [json.loads((d / "task.json").read_text(encoding="utf-8")) for d in task_dirs()]


def test_first_set_is_10_test_and_12_dev_tasks_covering_every_category():
    first = [t for t in _built() if t["provenance"]["generator"] == "grcbench.tier_a.first_set"]
    by_split = Counter(t["split"] for t in first)
    assert by_split == {"test": 10, "dev": 12}
    for split in ("test", "dev"):
        assert {t["category"] for t in first if t["split"] == split} == set(spec_drafts.CATEGORIES)


def test_the_test_split_has_three_tasks_per_category():
    cats = Counter(t["category"] for t in _built() if t["split"] == "test")
    assert sum(cats.values()) == 30
    assert set(cats.values()) == {3} and set(cats) == set(spec_drafts.CATEGORIES)


def test_ids_are_unique_and_follow_the_split():
    ids = [t["id"] for t in _built()]
    assert len(ids) == len(set(ids))
    assert [s["id"] for s in spec_drafts.SPECS] == [f"gba-test-{i}" for i in range(11, 31)]
    assert {s["id"] for s in spec_drafts.SPECS} <= set(ids)              # every spec is built
    for t in _built():
        assert t["id"].startswith(f"gba-{t['split']}-")


@pytest.mark.parametrize("task_dir", task_dirs(), ids=lambda d: d.name)
def test_built_task_passed_its_gates_and_is_well_formed(task_dir: Path):
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    gates = json.loads((task_dir / "gates.json").read_text(encoding="utf-8"))
    assert gates["passed"], gates
    replay = gates["gates"]["golden_replay"]
    assert replay["passes"] == replay["runs"] == 10
    golden = json.loads((task_dir / "golden_calls.json").read_text(encoding="utf-8"))
    assert 8 <= len(golden["calls"]) <= 20 and golden["final_text"]
    assert task["provenance"]["drafted_by"] == "Claude (Anthropic)"
    assert task["review"]["status"] == "pending-team-review"
    assert task["canary"] == CANARY
    seen = [task["instruction"], json.dumps(task["scripted_user"])]
    seen += [p.read_text(encoding="utf-8") for p in (task_dir / "start").rglob("*") if p.is_file()]
    assert not any(CANARY in s for s in seen)
    ref = task["source"].get("ref", "")
    if ref.startswith("gbf-"):                      # start graphs come from the same split's Tier-F graphs
        assert ref.startswith(f"gbf-{task['split']}-")


def test_wrong_end_states_are_rejected_by_a_check_not_by_a_crash():
    for d in task_dirs():
        gates = json.loads((d / "gates.json").read_text(encoding="utf-8"))
        reasons = [r for c in gates["gates"]["wrong_end_state"]["cases"] for r in c["reasons"]]
        assert reasons, d.name
        assert not any("Error:" in r and "fails at V4" in r for r in reasons), (d.name, reasons)


def test_spec_records_are_test_side_and_pending_review():
    for rec in spec_drafts.draft_records():
        assert rec["split"] == "test" and rec["status"] == "spec-only" and rec["canary"] == CANARY
        assert rec["provenance"]["drafted_by"] == "Claude (Anthropic)"
        assert rec["review"]["status"] == "pending-team-review"
        assert CANARY not in rec["instruction"]
        lo, hi = rec["expected_calls"]
        assert 8 <= lo <= hi <= 20
        assert rec["graded"] and rec["golden_outline"] and rec["disjointness"]
        start = rec["start"]["tier_f_task"]
        assert start is None or start.startswith("gbf-test-")
        if start and rec["source"]["kind"] == "tier_f_graph":
            assert rec["license"] == first_set.tf_task(start)["license"]


def test_sheet_columns_match_the_checklist():
    doc = CHECKLIST.read_text(encoding="utf-8")
    section = doc.split("## Record your decision", 1)[1].split("\n## ", 1)[0]
    documented = [name for line in section.splitlines() if line.startswith("| `")
                  for name in re.findall(r"`([a-z_0-9]+)`", line.split("|")[1])]
    assert set(documented) <= set(spec_drafts.SHEET_COLUMNS)
    assert spec_drafts.SHEET_COLUMNS[-len(DECISIONS):] == DECISIONS
    assert "./gb tier-a sheet" in doc and "tier_a_first_set.csv" in doc


def test_sheet_rows_cover_built_tasks_and_specs():
    rows = spec_drafts.built_rows() + spec_drafts.draft_rows()
    assert len(rows) == 42
    for r in rows:
        assert set(r) <= set(spec_drafts.SHEET_COLUMNS)
        assert r["instruction"] and r["acceptance"] and r["start_state"]

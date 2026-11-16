"""The Tier-F set report (`./gb tier-f report`) states what the review decided, from the task files."""

from __future__ import annotations

from collections import Counter

from grcbench.tier_f import set_report, taskio


def test_the_report_counts_the_review_outcome_and_the_levels_without_the_rejected_tasks():
    tasks = [taskio.load(d) for d in taskio.task_dirs()]
    text = set_report.build(today="2026-11-15")
    status = Counter(t["review"]["status"] for t in tasks)
    assert text.startswith("# Tier F first set: status after the review")
    assert (f"{status['approved']} approved, {status['approved-with-edits']} approved with edits "
            f"and {status['rejected']} rejected") in text
    from grcbench.review.replacements import all_ids   # an undecided replacement is not in the benchmark yet
    active = [t for t in tasks if t["review"]["status"] not in ("rejected", "reserve")
              and not (t["review"]["status"] == "pending-team-review" and t["id"] in all_ids())]
    test = Counter(t["level"] for t in active if t["split"] == "test")
    assert (f"the test split has {sum(test.values())} tasks: L1 {test['L1']}, L2 {test['L2']}, "
            f"L3 {test['L3']}") in text
    assert "two-person team review" not in text and "waits for" not in text

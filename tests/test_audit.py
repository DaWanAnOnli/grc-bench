"""`./gb audit`: the checks pass on the repository and catch the faults they are for."""

from __future__ import annotations

import copy

from grcbench.common import audit
from grcbench.common.canary import CANARY, CANARY_GUID
from grcbench.common.io import read_json
from grcbench.common.paths import SPLITS


def _records():
    return list(audit._records())


def test_audit_passes_on_the_repository():
    results = audit.run()
    assert [r["check"] for r in results if not r["passed"]] == []


def test_canary_check_catches_a_leak_and_a_missing_canary():
    recs = _records()
    path, rec, kind = next(r for r in recs if r[1].get("tier") == "A" and r[2] == "item" and "scripted_user" in r[1])
    leaked = copy.deepcopy(rec)
    leaked["instruction"] += f" {CANARY_GUID}"
    missing = copy.deepcopy(rec)
    missing["canary"] = CANARY[:-1]
    res = audit.check_canary([(path, leaked, kind), (path, missing, kind)])
    assert not res["passed"] and res["n_failures"] == 2


def test_licence_check_catches_a_tier_a_task_that_drops_the_gpl():
    recs = _records()
    gpl = [r for r in recs if r[1].get("tier") == "A" and r[1].get("license", "").startswith("GPL")
           and r[1].get("source", {}).get("kind") == "tier_f_graph"]
    assert gpl
    path, rec, kind = gpl[0]
    wrong = copy.deepcopy(rec)
    wrong["license"] = "CC-BY-4.0 (original GRC-Bench task)"
    f_tasks = [r for r in recs if r[1].get("tier") == "F" and r[0].endswith("task.json")]
    assert not audit.check_licence(f_tasks + [(path, wrong, kind)])["passed"]


def test_split_checks_catch_cross_split_reuse_and_duplicate_ids():
    recs = _records()
    path, rec, kind = next(r for r in recs if r[1].get("tier") == "A" and r[1].get("split") == "dev"
                           and r[1].get("source", {}).get("kind") == "tier_f_graph")
    moved = copy.deepcopy(rec)
    moved["source"]["ref"] = "gbf-test-01/reference.grc"
    assert not audit.check_tier_a_splits([(path, moved, kind)])["passed"]
    assert not audit.check_unique_ids([(path, rec, kind), (path, rec, kind)])["passed"]


def test_training_data_check_catches_dev_and_test_questions_and_the_canary(monkeypatch):
    recs = _records()
    assert audit.check_training_release(recs)["passed"]
    dev_ids = read_json(SPLITS / "tier_k_dev.json")["dev_ids"]
    dev_id = dev_ids[0]
    test_id = next(r["id"] for _, r, _ in recs if r.get("item_kind") == "frozen_question")
    monkeypatch.setattr(audit.release, "training_question_ids", lambda: {dev_id, test_id})
    monkeypatch.setattr(audit.release, "training_files_containing", lambda text: ["train.jsonl"])
    monkeypatch.setattr(audit.release, "validation_question_ids", lambda: dev_ids[1:])
    res = audit.check_training_release(recs)
    assert not res["passed"] and res["n_failures"] == 4

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


# ------------------------------------------------------------------ the decided edits of the second-round review
def test_a_fact_label_matches_only_an_item_value_line_so_a_lead_in_sentence_does_not_trip_it():
    for tid in ("gba-test-31", "gba-test-32", "gba-test-33", "gba-test-36", "gba-dev-13"):
        assert all(f["label"].endswith(R.LABEL_END) for f in SPECS[tid]["acceptance"]["answer"]["facts"]), tid
    fact = next(f for f in SPECS["gba-test-31"]["acceptance"]["answer"]["facts"] if f["name"] == "received RMS")
    answer = ("I ran it for 2 s and measured the received RMS and mean power after skipping the first 0.1 s.\n"
              "| received RMS | 0.5 |\n")
    assert grader._labelled(0.5, answer, fact)[0]
    assert grader._labelled(0.5, "**received RMS** (received.dat): 0.5", fact)[0]
    assert not grader._labelled(0.5, "received RMS: 1.0", fact)[0]


def test_the_knowledge_tasks_name_the_closest_training_specs_and_ask_for_single_numbers_where_needed():
    for tid in ("gba-test-31", "gba-test-32", "gba-test-33", "gba-dev-13"):
        assert "t08_run_verify_probe" in SPECS[tid]["disjointness"] and "t19_rssi_meter_build" in SPECS[tid]["disjointness"]
    assert "t08_run_verify_probe" in SPECS["gba-test-36"]["disjointness"]
    assert "t19_rssi_meter_build" not in SPECS["gba-test-36"]["disjointness"]          # no power meter in its graph
    single = "in the form 'item: value', with each value as a single number, using the item names below"
    assert single in SPECS["gba-test-33"]["instruction"] and single in SPECS["gba-test-36"]["instruction"]
    assert single not in SPECS["gba-test-31"]["instruction"]
    assert "6. received RMS: the RMS level of received.dat, to three significant figures." in SPECS["gba-test-36"]["instruction"]
    fm = {f["name"]: f for f in SPECS["gba-test-33"]["acceptance"]["answer"]["facts"]}
    assert fm["modulator sensitivity"]["rel_tol"] == fm["demodulator gain"]["rel_tol"] == 0.01


def test_the_save_block_tasks_answer_the_category_question_and_accept_a_file_name_set_through_a_variable():
    import re
    for tid in ("gba-test-34", "gba-test-35", "gba-test-37", "gba-dev-14"):
        s = SPECS[tid]
        rules = s["scripted_user"]["rules"]
        first = lambda q: next(r["reply"] for r in rules if re.search(r["pattern"], q, re.IGNORECASE))
        assert first("Which category should I put the block in?") == "Use the default."
        assert first("Should I overwrite the existing block?") == "Yes, go ahead."
        assert first("Do you want me to save it?") == "Yes, please save it."
        assert s["acceptance"]["blocks_present"] == [{"id": "epy_block"}], tid
    clip = SPECS["gba-test-37"]
    assert clip["acceptance"]["ladder"]["oracle"]["graph_checks"][0]["from_ids"] == ["analog_sig_source_x"]
    assert "Keep mirrored.dat and second_quantized.dat as they are." in clip["instruction"]
    assert "Keep quantized.dat as it is." in SPECS["gba-test-34"]["instruction"]
    sign = [c for c in SPECS["gba-dev-14"]["acceptance"]["ladder"]["oracle"]["checks"] if c.get("point") == "added"]
    assert {"feature": "mean", "min": -0.1, "max": 0.1, "point": "added", "skip": 4800, "group": "f2p"} in sign
    assert {"feature": "peak_freq", "expect": 9500, "tol": 95.0, "fs": 96000.0, "n": 16384, "point": "added",
            "skip": 4800, "group": "f2p"} in sign

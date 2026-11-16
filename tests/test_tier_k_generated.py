"""The generated Tier-K material: block questions (`./gb blockq`) and rubric templates (`./gb rubrics`).

Fast tests on the files in the repository (no GNU Radio): the block questions are on eligible catalog
blocks with a behaviour model, schema-valid, scorable by the rule cascade and kept apart from their
answer key; the generator's seeds reach wide instance spaces; the rubric
templates cover every test and dev question, hold no prose, never reach the judge unfilled, and the
generator never overwrites what the review has filled in.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from grcbench.common.canary import CANARY_GUID
from grcbench.common.paths import ITEMS, REPO
from grcbench.common.schemas import validate
from grcbench.tier_k import blockgen, judge_prompt, rubrics, rules


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


QUESTIONS = _jsonl(REPO / blockgen.OUT_QUESTIONS)
ANSWERS = _jsonl(REPO / blockgen.OUT_ANSWERS)


# ---------------------------------------------------------------- block questions
def test_block_questions_are_valid_and_about_100():
    catalog = {b["id"]: b for b in json.loads(blockgen.CATALOG_JSON.read_text(encoding="utf-8"))["blocks"]}
    modelled = {s.block for s in blockgen.SETTINGS}
    assert len(QUESTIONS) == blockgen.TARGET == len(ANSWERS)
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)
    for q in QUESTIONS:
        assert validate(q) == [], q["id"]
        block = q["source"]["block_id"]
        assert catalog[block]["eligible"] and block in modelled            # no test/train block split
        assert q["cluster_id"] == f"block_family:{catalog[block]['family']}"
        assert f"({blockgen.REFERENCE_SEED})" in q["provenance"]["generator_version"]
        assert q["item_kind"] == "block_question"
        assert q["review"] == blockgen.review_of(q["id"], q)               # the review protocol's decision
        assert q["block_question_kind"] in ("output_value", "output_count")   # behaviour, never catalog facts
        assert q["source"]["kind"] == "gnuradio_run"
        assert CANARY_GUID not in q["question"]
    per_block = Counter(q["source"]["block_id"] for q in QUESTIONS)
    assert max(per_block.values()) <= blockgen.MAX_PER_BLOCK
    assert len({q["question"] for q in QUESTIONS}) == len(QUESTIONS)        # no repeated question


def test_the_decided_wording_edits_are_in_the_questions():
    """reviews/block_questions.jsonl: the port base is stated on every multi-port question; the interleaver
    mapping and the rounding rule are stated; the patterned interleaver's answers are exact."""
    by_id = {a["question_id"]: a for a in ANSWERS}
    for q in QUESTIONS:
        block, text = q["source"]["block_id"], q["question"]
        if " on its output port " in text:
            assert "(ports counted from 0)" in text, q["id"]
        for b, sentence in blockgen.BLOCK_NOTES.items():
            assert (sentence in text) == (block == b), (q["id"], b)
        if block in blockgen.EXACT_COPY:
            assert by_id[q["id"]]["reference"]["tolerance"] == blockgen.INT_TOL["tolerance"]
    assert set(blockgen.BLOCK_NOTES) == {"blocks_blockinterleaver_xx", "blocks_float_to_int"}
    assert blockgen.EXACT_COPY == {"blocks_patterned_interleaver"}


def test_review_status_follows_the_decided_records():
    decided = {"approve": "approved", "approve with edits": "approved-with-edits", "reject": "rejected"}
    rec = {"final": "approve with edits", "decided_by": "EX", "route": "disagreement"}
    assert blockgen.review_status(rec) == {"status": "approved-with-edits", "reviewers": ["R1", "R2", "EX"],
                                           "notes": "decided by EX (disagreement); edits applied in the template"}
    assert blockgen.review_status(None)["status"] == "pending-team-review"
    agreed = {"final": "approve", "decided_by": "R1;R2", "route": "agreement"}
    assert blockgen.review_status(agreed)["status"] == "approved"
    templ = blockgen.review_status(agreed, template_edited=True)
    assert templ["status"] == "approved-with-edits" and "template edit" in templ["notes"]
    assert set(decided) >= {"approve"}
    for q in QUESTIONS:                                     # every question a template edit reached says so
        edited = blockgen.template_edited(q["source"]["block_id"], q["question"])
        assert (q["review"]["status"] == "approved-with-edits") == (
            edited or blockgen.review_of(q["id"])["status"] == "approved-with-edits")


def test_block_question_records_hold_no_answers_and_the_key_scores_them():
    by_id = {a["question_id"]: a for a in ANSWERS}
    assert set(by_id) == {q["id"] for q in QUESTIONS}
    for q in QUESTIONS:
        assert "answer" not in q and "reference" not in q
        key = by_id[q["id"]]
        integer = key["reference"]["tolerance"] < 1e-6
        problem = blockgen._rules_check({"answer": key["answer"], "integer": integer,
                                         "tolerance": {k: key["reference"][k] for k in ("tolerance", "abs_tolerance")}})
        assert problem is None, (q["id"], problem)
        assert "GNU Radio 3.10.12 run" in key["evidence"]


def test_no_question_on_a_block_a_training_trace_ran():
    executed = blockgen._executed_blocks()
    assert executed, "splits/agent_traces.json (./gb traces) lists the blocks built in trace code"
    assert not {q["source"]["block_id"] for q in QUESTIONS} & executed


def test_reference_models_match_known_values():
    assert blockgen.linear2ulaw(0) == 0xFF and blockgen.linear2alaw(0) == 0xD5      # ITU-T G.711 silence codes
    assert blockgen._c_div(-7, 2) == -3 and blockgen._c_div(7, -2) == -3
    assert blockgen._iir([1.0], [1.0, 0.5], [1.0, 0.0, 0.0], True) == [1.0, 0.5, 0.25]
    assert blockgen._iir([1.0], [1.0, 0.5], [1.0, 0.0, 0.0], False) == [1.0, -0.5, 0.25]


# ---------------------------------------------------------------- rubric templates
@pytest.mark.parametrize("split,n", [("test", 185), ("dev", 107)])
def test_one_empty_template_per_question(split, n):
    temps = [json.loads(line) for line in rubrics.outputs()[rubrics.OUT[split]].splitlines()]
    qs = _jsonl(rubrics.SOURCES[split])
    assert len(temps) == len(qs) == n
    assert [t["question_id"] for t in temps] == [q["id"] for q in qs]
    for t, q in zip(temps, qs):
        assert t["required_points"] == [] and t["disqualifying_errors"] == [] and t["sources"] == []
        assert t["numeric"] is None and t["numeric_tolerance"] is None
        assert t["review"]["status"] == "pending-team-review"
        assert (t["calc_status"] == rubrics.PENDING_CALC) == (q["task_type"] == "calculation")
        with pytest.raises(judge_prompt.RubricError):
            judge_prompt.normalize_rubric(t)                                 # an unfilled template never reaches the judge
    rows = list(csv.DictReader((REPO / rubrics.SHEET[split]).open(encoding="utf-8")))
    assert [r["question_id"] for r in rows] == [q["id"] for q in qs]


@pytest.mark.parametrize("split", ["test", "dev"])
def test_the_repository_rubrics_are_templates_or_the_decided_rubrics(split):
    """Before the rubric review is decided the repository holds the templates; after it, every question's
    decided rubric (./gb review rubrics): approved ones the judge reads, rejected ones removed from runs."""
    from grcbench.review import records
    from grcbench.tier_k import run as krun
    recs = _jsonl(REPO / rubrics.OUT[split])
    qs = _jsonl(rubrics.SOURCES[split])
    assert [r["question_id"] for r in recs] == [q["id"] for q in qs]
    decided = records.decided(f"rubric_{split}")
    if not decided:
        assert all(r["review"]["status"] == "pending-team-review" for r in recs)
        return
    assert {k[0] for k in decided} == {q["id"] for q in qs}
    gone = {r["question_id"] for r in krun.removed(split)}
    for r in recs:
        status = r["review"]["status"]
        assert status in ("approved", "approved-with-edits", "rejected"), r["question_id"]
        assert r["review"]["reviewers"][:3] == ["W", "C1", "C2"]
        if status == "rejected":
            assert r["question_id"] in gone and r["required_points"] == [] and r["review"]["notes"]
        else:
            judge_prompt.check_reviewed(r)
            judge_prompt.normalize_rubric(r)
            assert r["question_id"] not in gone
        if r["task_type"] == "calculation":
            assert r["calc_status"] and r["calc_status"] != rubrics.PENDING_CALC


def test_the_generator_never_overwrites_review_content(tmp_path, capsys):
    assert rubrics.main(["--out", str(tmp_path)]) == 0
    path = tmp_path / rubrics.OUT["dev"]
    rows = _jsonl(path)
    rows[3]["required_points"] = [{"id": "P1", "text": "a point the team wrote"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    assert rubrics.main(["--out", str(tmp_path)]) == 1
    assert path.read_text(encoding="utf-8") == before and "holds review content" in capsys.readouterr().out
    sheet = tmp_path / rubrics.SHEET["test"]
    text = sheet.read_text(encoding="utf-8").splitlines()
    text[1] = text[1] + "JR"                                                  # a filled cell at the row's end
    sheet.write_text("\n".join(text) + "\n", encoding="utf-8")
    assert rubrics._review_content(rubrics.SHEET["test"], tmp_path)

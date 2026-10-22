"""The generated Tier-K material: block questions (`./gb blockq`) and rubric templates (`./gb rubrics`).

Fast tests on the files in the repository (no GNU Radio): the block questions are test-side,
schema-valid, scorable by the rule cascade and kept apart from their answer key; the rubric
templates cover every test and dev question, hold no prose, never reach the judge unfilled, and the
generator never overwrites what the team has filled in.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from grcbench.common.canary import CANARY_GUID
from grcbench.common.paths import ITEMS, REPO, SPLITS
from grcbench.common.schemas import validate
from grcbench.tier_k import blockgen, judge_prompt, rubrics, rules


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


QUESTIONS = _jsonl(REPO / blockgen.OUT_QUESTIONS)
ANSWERS = _jsonl(REPO / blockgen.OUT_ANSWERS)


# ---------------------------------------------------------------- block questions
def test_block_questions_are_test_side_valid_and_about_100():
    part = {b["id"]: b for b in json.loads((SPLITS / "block_partition.json").read_text())["blocks"]}
    assert len(QUESTIONS) == blockgen.TARGET == len(ANSWERS)
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)
    for q in QUESTIONS:
        assert validate(q) == [], q["id"]
        assert part[q["source"]["block_id"]]["split"] == "test"          # training uses other blocks
        assert q["cluster_id"] == f"block_family:{part[q['source']['block_id']]['family']}"
        assert q["item_kind"] == "block_question" and q["review"]["status"] == "pending-team-review"
        assert CANARY_GUID not in q["question"]
    pairs = Counter((q["source"]["block_id"], q["block_question_kind"]) for q in QUESTIONS)
    assert max(pairs.values()) == 1                                          # one per block and kind


def test_block_question_records_hold_no_answers_and_the_key_scores_them():
    by_id = {a["question_id"]: a for a in ANSWERS}
    assert set(by_id) == {q["id"] for q in QUESTIONS}
    for q in QUESTIONS:
        assert "answer" not in q and "reference" not in q
        key = by_id[q["id"]]
        right, wrong = blockgen._answer_text({"kind": key["kind"], "answer": key["answer"]})
        assert rules.score(right, key["reference"])["credit"] == 1.0, q["id"]
        r0 = rules.score(wrong, key["reference"])
        assert r0["method"] == "judge" or r0["credit"] < 1.0, q["id"]


def test_type_questions_never_give_the_answer_away():
    by_id = {a["question_id"]: a for a in ANSWERS}
    for q in QUESTIONS:
        if q["block_question_kind"] in blockgen.TYPE_KINDS:
            assert not blockgen._gives_away(by_id[q["id"]]["answer"], q["question"].split("(`")[0]), q["question"]


def test_output_rates_were_confirmed_by_a_run():
    rates = [a for a in ANSWERS if a["kind"] == "output_rate"]
    assert rates and all("confirmed by running the flowgraph" in a["evidence"] for a in rates)


# ---------------------------------------------------------------- rubric templates
@pytest.mark.parametrize("split,n", [("test", 185), ("dev", 200)])
def test_one_empty_template_per_question(split, n):
    temps = _jsonl(REPO / rubrics.OUT[split])
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


def test_the_generator_never_overwrites_team_content(tmp_path, capsys):
    assert rubrics.main(["--out", str(tmp_path)]) == 0
    path = tmp_path / rubrics.OUT["dev"]
    rows = _jsonl(path)
    rows[3]["required_points"] = [{"id": "P1", "text": "a point the team wrote"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    assert rubrics.main(["--out", str(tmp_path)]) == 1
    assert path.read_text(encoding="utf-8") == before and "holds team content" in capsys.readouterr().out
    sheet = tmp_path / rubrics.SHEET["test"]
    text = sheet.read_text(encoding="utf-8").splitlines()
    text[1] = text[1] + "JR"                                                  # a filled cell at the row's end
    sheet.write_text("\n".join(text) + "\n", encoding="utf-8")
    assert rubrics._team_content(rubrics.SHEET["test"], tmp_path)

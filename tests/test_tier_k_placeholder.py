"""Tests for rubric files and placeholder rubrics (`./gb placeholder-rubrics`, `./gb tier-k score DIR --rubrics
FILE`): toy items and toy rubrics, except the check of the generated file against the real dev split and
release (no question reaches a model here)."""

from __future__ import annotations

import hashlib
import json

import pytest

from grcbench.common.io import read_jsonl, write_json, write_jsonl
from grcbench.tier_k import judge_prompt as jp
from grcbench.tier_k import judge_run, placeholder_rubrics
from grcbench.tier_k import run as krun

QUESTION = "Toy question A: name a primary colour of light."
ITEM = {"id": "q_toy_a", "split": "dev", "question": QUESTION, "rubric": None, "key": None}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


PLACEHOLDER = {"question_id": "q_toy_a", "split": "dev", "question_sha256": _sha(QUESTION),
               "required_points": [{"id": "P1", "text": "Red, green or blue.", "weight": 1.0}],
               "disqualifying_errors": [], "numeric": None, "numeric_tolerance": None,
               "review": {"status": jp.PLACEHOLDER_REVIEW, "reviewers": [], "notes": ""}}


def _file(tmp_path, rows):
    path = tmp_path / "rubrics.jsonl"
    write_jsonl(path, rows)
    return path


def test_a_placeholder_rubric_is_refused_unless_allowed():
    with pytest.raises(jp.RubricError, match="placeholder"):
        jp.check_reviewed(PLACEHOLDER)
    jp.check_reviewed(PLACEHOLDER, allow_placeholder=True)


def test_a_rubric_file_gives_a_dev_item_its_placeholder_rubric(tmp_path):
    assert krun.route_answer(dict(ITEM), "Red.")["route"] == "no_reference"
    items = krun.with_rubric_file([dict(ITEM)], _file(tmp_path, [PLACEHOLDER]), "dev")
    routed = krun.route_answer(items[0], "Red.")
    assert routed["route"] == "judge" and "Red, green or blue." in routed["judge_messages"][-1]["content"]


def test_a_placeholder_never_counts_without_the_file():
    item = {**ITEM, "rubric": PLACEHOLDER}                       # e.g. pasted into the repo's rubrics.jsonl
    assert krun.rubric_for(item) is None


def test_a_rubric_file_is_refused_on_the_test_split(tmp_path):
    with pytest.raises(ValueError, match="dev split only"):
        krun.with_rubric_file([{**ITEM, "split": "test"}], _file(tmp_path, [PLACEHOLDER]), "test")


@pytest.mark.parametrize("change, match", [({"question_sha256": "0" * 64}, "question text"),
                                           ({"question_id": "q_other"}, "not in the dev split")])
def test_a_rubric_file_that_does_not_fit_the_split_is_refused(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        krun.with_rubric_file([dict(ITEM)], _file(tmp_path, [{**PLACEHOLDER, **change}]), "dev")


@pytest.fixture
def toy_run(tmp_path, monkeypatch):
    monkeypatch.setattr(krun, "load_split", lambda split: [dict(ITEM)])
    out = tmp_path / "run"
    out.mkdir()
    write_json(out / "manifest.json", {"split": "dev", "model": "toy", "config_id": "dev-smoke", "n_items": 1, "n": 1})
    write_jsonl(out / "answers.jsonl", [{"item_id": "q_toy_a", "sample": 0, "text": "Red.", "reasoning": None,
                                         "finish_reason": "stop", "truncated": False}])
    return out


def test_score_with_a_rubric_file_keeps_a_copy_and_records_it(toy_run, tmp_path):
    path = _file(tmp_path, [PLACEHOLDER])
    s = krun.score(toy_run, rubrics=path)
    assert s["n_judge"] == 1 and s["rubrics"]["n_placeholder"] == 1
    assert s["rubrics"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (toy_run / krun.RUBRICS_COPY).read_bytes() == path.read_bytes()
    s = krun.score(toy_run)                                      # scored again with the repository's rubrics
    assert s["n_no_reference"] == 1 and s["rubrics"] is None and not (toy_run / krun.RUBRICS_COPY).exists()


def test_the_judge_uses_the_runs_copy_of_the_rubric_file(toy_run, tmp_path, monkeypatch):
    krun.score(toy_run, rubrics=_file(tmp_path, [PLACEHOLDER]))
    sent = []
    verdict = json.dumps({"points": [{"id": "P1", "verdict": "met", "reason": "Red is named."}], "numbers": [],
                          "disqualifying_errors": [], "hallucinated_entities": []})

    def fake(ep, messages, seed, timeout):
        sent.append(messages)
        return {"text": verdict, "finish_reason": "stop", "usage": {}, "seconds": 0.0}

    monkeypatch.setattr(judge_run, "_request", fake)
    ep = judge_run.Endpoint(base_url="http://127.0.0.1:9/v1", model="stand-in")
    summary = judge_run.judge(toy_run, primary=ep, fallback=ep, retry_wait=0)
    assert summary["judge"] == "complete" and len(sent) == 1
    assert read_jsonl(toy_run / "judge_scores.jsonl")[-1]["score"] == 1.0


def test_the_judge_refuses_a_changed_copy(toy_run, tmp_path):
    krun.score(toy_run, rubrics=_file(tmp_path, [PLACEHOLDER]))
    copy = toy_run / krun.RUBRICS_COPY
    copy.write_text(copy.read_text(encoding="utf-8").replace("Red, green", "Cyan"), encoding="utf-8")
    ep = judge_run.Endpoint(base_url="http://127.0.0.1:9/v1", model="stand-in")
    with pytest.raises(judge_run.JudgeRefused, match="changed"):
        judge_run.judge(toy_run, primary=ep, fallback=ep, retry_wait=0)


def test_placeholders_copy_the_reference_answer_word_for_word():
    release = {"q_toy_a": {"question": QUESTION, "reference_answer": " Red, green or blue. ",
                           "reference_model": "toy-30b"}}
    rows, skipped = placeholder_rubrics.records([dict(ITEM), {**ITEM, "id": "q_toy_b", "question": "B?"}], release)
    assert skipped == ["q_toy_b"]
    (r,) = rows
    assert r["required_points"] == [{"id": "P1", "text": "Red, green or blue.", "weight": 1.0}]
    assert r["review"]["status"] == jp.PLACEHOLDER_REVIEW and r["question_sha256"] == _sha(QUESTION)
    assert r["numeric"] is None and r["disqualifying_errors"] == []


def test_the_placeholder_file_covers_the_dev_split(tmp_path):
    out = tmp_path / "dev_placeholder.jsonl"
    assert placeholder_rubrics.main(["--out", str(out)]) == 0
    items = krun.load_split("dev")
    with_file = krun.with_rubric_file(items, out, "dev")
    assert all(krun.rubric_for(i) is not None for i in with_file)
    assert len(read_jsonl(out)) == len(items)

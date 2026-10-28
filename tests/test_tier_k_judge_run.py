"""Tests for `./gb tier-k judge` (grcbench.tier_k.judge_run) against stand-in judge servers: toy items
and toy rubrics only (no benchmark question reaches a judge here)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from grcbench.common.io import read_jsonl, write_json, write_jsonl
from grcbench.tier_k import judge_prompt as jp
from grcbench.tier_k import judge_run, rules
from grcbench.tier_k import run as krun

APPROVED = {"status": "approved", "reviewers": ["a", "b"]}
POINT_RUBRIC = {"review": APPROVED, "required_points": ["Names red."], "disqualifying_errors": [],
                "numeric": None, "numeric_tolerance": None}
GAIN_RUBRIC = {"review": APPROVED, "required_points": [], "disqualifying_errors": [],
               "numeric": {"value": -3, "unit": "dB"}, "numeric_tolerance": "toy: as in numeric"}
ITEMS = [{"id": "q_toy_a", "question": "Toy question A: name a primary colour of light.", "rubric": POINT_RUBRIC,
          "key": None},
         {"id": "q_toy_b", "question": "Toy question B: what is the gain?", "rubric": GAIN_RUBRIC, "key": None}]
ANSWERS = {"q_toy_a": "Red is one.", "q_toy_b": "Final answer: 3 dB of margin loss"}   # B: the rules' tie-break
VALID = json.dumps({"points": [{"id": "P1", "verdict": "met", "reason": "Red is named."}], "numbers": [],
                    "disqualifying_errors": [], "hallucinated_entities": []})
SECRET_KEY = "sk-test-judge-key-0123456789"


class _Judge:
    """A stand-in judge: replies[i] answers request i (the last one repeats): ("ok", content, finish) or
    ("http", status)."""

    def __init__(self, replies: list[tuple]) -> None:
        self.replies, self.requests, self.headers = replies, [], []
        judge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:                                   # noqa: N802 - http.server API
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                judge.requests.append(body)
                judge.headers.append(dict(self.headers))
                reply = judge.replies[min(len(judge.requests) - 1, len(judge.replies) - 1)]
                if reply[0] == "http":
                    self.send_error(reply[1])
                    return
                data = json.dumps({"choices": [{"message": {"content": reply[1]}, "finish_reason": reply[2]}],
                                   "usage": {"prompt_tokens": 100, "completion_tokens": 20}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def endpoint(self, **kw) -> judge_run.Endpoint:
        return judge_run.Endpoint(base_url=self.url, model="stand-in", **kw)


@pytest.fixture
def judges():
    made: list[_Judge] = []

    def make(replies):
        made.append(_Judge(replies))
        return made[-1]

    yield make
    for j in made:
        j.server.shutdown()


@pytest.fixture
def scored_run(tmp_path, monkeypatch):
    """A toy dev run scored by `./gb tier-k score`: answer A goes to the judge, B is decided by the rules."""
    monkeypatch.setattr(krun, "load_split", lambda split: [dict(i) for i in ITEMS])
    out = tmp_path / "run"
    out.mkdir()
    write_json(out / "manifest.json", {"split": "dev", "model": "toy", "config_id": "dev-smoke", "n_items": 2, "n": 1})
    write_jsonl(out / "answers.jsonl", [{"item_id": i, "sample": 0, "text": t, "reasoning": None,
                                         "finish_reason": "stop", "truncated": False} for i, t in ANSWERS.items()])
    krun.score(out)
    assert [q["item_id"] for q in read_jsonl(out / "judge_queue.jsonl")] == ["q_toy_a"]
    return out


def _run(out, primary, fallback, **kw):
    return judge_run.judge(out, primary=primary, fallback=fallback, retry_wait=0, **kw)


def test_a_valid_verdict_is_scored(scored_run, judges):
    p, f = judges([("ok", VALID, "stop")]), judges([("ok", VALID, "stop")])
    summary = _run(scored_run, p.endpoint(), f.endpoint())
    row = read_jsonl(scored_run / "judge_scores.jsonl")[0]
    assert (row["item_id"], row["judge"], row["score"], row["judge_failed"]) == ("q_toy_a", "primary", 1.0, False)
    assert len(p.requests) == 1 and f.requests == []
    body = p.requests[0]
    assert body["seed"] == jp.JUDGE_REQUEST["seed"] and body["max_tokens"] == jp.JUDGE_REQUEST["max_tokens"]
    assert body["temperature"] == 0.0 and body["messages"] == read_jsonl(scored_run / "judge_queue.jsonl")[0]["messages"]
    req = read_jsonl(scored_run / "judge_requests.jsonl")[0]
    assert req["valid"] and req["judge_prompt_version"] == jp.PROMPT_VERSION and req["usage"]["completion_tokens"] == 20
    assert "reasoning_setting" in req and req["seed"] == jp.JUDGE_REQUEST["seed"]
    assert summary["judge"] == "complete" and summary["n_judge_failed"] == 0


def test_a_malformed_reply_is_asked_again_with_the_next_seed(scored_run, judges):
    p, f = judges([("ok", "not json", "stop"), ("ok", VALID, "stop")]), judges([("ok", VALID, "stop")])
    _run(scored_run, p.endpoint(), f.endpoint())
    assert [r["seed"] for r in p.requests] == [jp.JUDGE_REQUEST["seed"], jp.JUDGE_REQUEST["seed"] + 1]
    assert read_jsonl(scored_run / "judge_scores.jsonl")[0]["score"] == 1.0


def test_a_reply_cut_at_the_limit_is_malformed(scored_run, judges):
    p, f = judges([("ok", VALID, "length"), ("ok", VALID, "stop")]), judges([("ok", VALID, "stop")])
    _run(scored_run, p.endpoint(), f.endpoint())
    reqs = read_jsonl(scored_run / "judge_requests.jsonl")
    assert len(p.requests) == 2 and reqs[0]["valid"] is False and "cut" in reqs[0]["errors"][0]


def test_the_fallback_is_asked_once_with_the_base_seed(scored_run, judges):
    p, f = judges([("ok", "{}", "stop")]), judges([("ok", VALID, "stop")])
    _run(scored_run, p.endpoint(), f.endpoint())
    assert len(p.requests) == jp.MAX_ATTEMPTS and [r["seed"] for r in f.requests] == [jp.JUDGE_REQUEST["seed"]]
    row = read_jsonl(scored_run / "judge_scores.jsonl")[0]
    assert (row["judge"], row["score"]) == ("fallback", 1.0)


def test_both_judges_failing_scores_0(scored_run, judges):
    p, f = judges([("ok", "nope", "stop")]), judges([("ok", "nope", "stop")])
    summary = _run(scored_run, p.endpoint(), f.endpoint())
    row = read_jsonl(scored_run / "judge_scores.jsonl")[0]
    assert (row["judge_failed"], row["score"], row["reason"]) == (True, 0.0, "judge_failed")
    assert len(f.requests) == 1 and summary["n_judge_failed"] == 1


def test_a_request_failure_is_not_scored_and_is_sent_again_on_resume(scored_run, judges):
    down, f = judges([("http", 503)]), judges([("ok", VALID, "stop")])
    summary = _run(scored_run, down.endpoint(), f.endpoint())
    row = read_jsonl(scored_run / "judge_scores.jsonl")[0]
    assert row["request_failed"] and row["score"] is None and f.requests == []     # never a malformed verdict
    assert summary["judge"] == "incomplete" and summary["n_judge_request_failed"] == 1
    up = judges([("ok", VALID, "stop")])
    summary = _run(scored_run, up.endpoint(), f.endpoint())
    assert len(up.requests) == 1 and summary["judge"] == "complete"
    assert read_jsonl(scored_run / "judge_scores.jsonl")[0]["score"] == 1.0


def test_a_queue_built_under_another_judge_version_is_refused(scored_run, judges):
    queue = read_jsonl(scored_run / "judge_queue.jsonl")
    queue[0]["messages"][0]["content"] += " (v3)"
    write_jsonl(scored_run / "judge_queue.jsonl", queue)
    p = judges([("ok", VALID, "stop")])
    with pytest.raises(judge_run.JudgeRefused, match="score the run again"):
        _run(scored_run, p.endpoint(), p.endpoint())
    assert p.requests == []


def test_a_run_scored_with_other_rules_is_refused(scored_run, judges):
    summary = json.loads((scored_run / "summary.json").read_text())
    summary["rules_version"] = "tier-k-rules 0.5"
    write_json(scored_run / "summary.json", summary)
    p = judges([("ok", VALID, "stop")])
    with pytest.raises(judge_run.JudgeRefused, match="rules"):
        _run(scored_run, p.endpoint(), p.endpoint())


def test_the_test_split_needs_allow_test_and_a_passed_calibration(scored_run, judges, tmp_path):
    manifest = json.loads((scored_run / "manifest.json").read_text())
    write_json(scored_run / "manifest.json", {**manifest, "split": "test"})
    p = judges([("ok", VALID, "stop")])
    with pytest.raises(judge_run.JudgeRefused, match="allow"):
        _run(scored_run, p.endpoint(), p.endpoint())
    with pytest.raises(judge_run.JudgeRefused, match="calibration"):
        _run(scored_run, p.endpoint(), p.endpoint(), allow_test=True)
    failed = tmp_path / "calibration_failed.json"
    write_json(failed, {"passed": False, "prompt_version": jp.PROMPT_VERSION})
    with pytest.raises(judge_run.JudgeRefused, match="calibration"):
        _run(scored_run, p.endpoint(), p.endpoint(), allow_test=True, calibration=failed)
    passed = tmp_path / "calibration.json"
    write_json(passed, {"passed": True, "prompt_version": jp.PROMPT_VERSION, "prompt_sha256": jp.prompt_sha256()})
    _run(scored_run, p.endpoint(), p.endpoint(), allow_test=True, calibration=passed)
    assert read_jsonl(scored_run / "judge_requests.jsonl")[0]["calibration_sha256"]
    assert p.requests


def test_the_counts_cover_every_answer(scored_run, judges):
    p = judges([("ok", VALID, "stop")])
    summary = _run(scored_run, p.endpoint(), p.endpoint())
    assert summary["report_counts"]["tie_break"] == 1              # answer B, decided by the rules
    assert set(summary["report_counts"]) == set(jp.report_counts([]))
    assert summary["n_unknown_unit"] == 0 and summary["rules_version"] == rules.RULES_VERSION


def test_the_api_key_is_sent_but_never_written(scored_run, judges, monkeypatch):
    monkeypatch.setenv("GRCB_TEST_JUDGE_KEY", SECRET_KEY)
    p = judges([("ok", VALID, "stop")])
    _run(scored_run, p.endpoint(api_key_env="GRCB_TEST_JUDGE_KEY"), p.endpoint())
    assert p.headers[0]["Authorization"] == f"Bearer {SECRET_KEY}"
    for path in scored_run.rglob("*"):
        if path.is_file():
            assert SECRET_KEY not in path.read_text(encoding="utf-8", errors="replace"), path
    monkeypatch.delenv("GRCB_TEST_JUDGE_KEY")
    with pytest.raises(judge_run.JudgeRefused, match="GRCB_TEST_JUDGE_KEY"):
        _run(scored_run, p.endpoint(api_key_env="GRCB_TEST_JUDGE_KEY"), p.endpoint())


def test_gb_tier_k_judge_runs_the_judge(scored_run, judges, capsys):
    p = judges([("ok", VALID, "stop")])
    assert krun.main(["judge", str(scored_run), "--base-url", p.url, "--model", "stand-in",
                      "--fallback-base-url", p.url]) == 0
    assert '"judge": "complete"' in capsys.readouterr().out


ITEM_C = {"id": "q_toy_c", "question": "Toy question C: name another primary colour of light.",
          "rubric": POINT_RUBRIC, "key": None}
ITEM_D = {"id": "q_toy_d", "question": "Toy question D: name a third primary colour of light.",
          "rubric": POINT_RUBRIC, "key": None}


@pytest.fixture
def run3(tmp_path, monkeypatch):
    """A toy run with two judged answers (A, C) and one answer that is only reasoning (D)."""
    items = [dict(i) for i in ITEMS] + [dict(ITEM_C), dict(ITEM_D)]
    monkeypatch.setattr(krun, "load_split", lambda split: [dict(i) for i in items])
    out = tmp_path / "run3"
    out.mkdir()
    texts = {**ANSWERS, "q_toy_c": "Green is one.", "q_toy_d": "<think>Red, green or blue.</think>"}
    write_json(out / "manifest.json", {"split": "dev", "model": "toy", "config_id": "dev-smoke", "n_items": 4, "n": 1})
    write_jsonl(out / "answers.jsonl", [{"item_id": i, "sample": 0, "text": t, "reasoning": None,
                                         "finish_reason": "stop", "truncated": False} for i, t in texts.items()])
    krun.score(out)
    return out, items, monkeypatch


def test_every_verdict_is_kept_when_the_run_stops(run3, judges):
    out, _, monkeypatch = run3
    p = judges([("ok", VALID, "stop")])
    calls = []
    real = judge_run._judge_answer

    def interrupted(*args, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args, **kw)

    monkeypatch.setattr(judge_run, "_judge_answer", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _run(out, p.endpoint(), p.endpoint())
    assert [r["item_id"] for r in read_jsonl(out / "judge_scores.jsonl")] == ["q_toy_a"]     # kept
    monkeypatch.setattr(judge_run, "_judge_answer", real)
    _run(out, p.endpoint(), p.endpoint())
    assert len(p.requests) == 2                                   # A once, C once: A is not sent again


def test_a_stored_score_is_reused_only_for_the_same_request(run3, judges):
    out, items, monkeypatch = run3
    p = judges([("ok", VALID, "stop")])
    _run(out, p.endpoint(), p.endpoint())
    assert len(p.requests) == 2
    changed = {**POINT_RUBRIC, "required_points": ["Names a primary colour."]}
    monkeypatch.setattr(krun, "load_split", lambda split: [{**i, "rubric": changed} if i["id"] == "q_toy_a" else dict(i)
                                                           for i in items])
    krun.score(out)                                               # the rubric changed: the queue too
    _run(out, p.endpoint(), p.endpoint())
    assert len(p.requests) == 3                                   # A sent again, C reused


def test_an_answer_the_queue_misses_is_refused(run3, judges):
    out, items, monkeypatch = run3
    monkeypatch.setattr(krun, "load_split", lambda split: [{**i, "rubric": POINT_RUBRIC} if i["id"] == "q_toy_b"
                                                           else dict(i) for i in items])
    p = judges([("ok", VALID, "stop")])
    with pytest.raises(judge_run.JudgeRefused, match="score the run again"):
        _run(out, p.endpoint(), p.endpoint())                     # B routes to the judge but is not queued
    assert p.requests == []


def test_an_answer_that_is_only_reasoning_counts_without_a_crash(run3, judges):
    out, _, _ = run3
    p = judges([("ok", VALID, "stop")])
    summary = _run(out, p.endpoint(), p.endpoint())
    assert summary["judge"] == "complete" and summary["n_judged"] == 2


def test_three_failed_answers_in_a_row_stop_the_judge(run3, judges, tmp_path, monkeypatch):
    out, items, _ = run3
    more = items + [{**ITEM_C, "id": f"q_toy_e{i}"} for i in range(3)]
    monkeypatch.setattr(krun, "load_split", lambda split: [dict(i) for i in more])
    rows = read_jsonl(out / "answers.jsonl") + [{"item_id": f"q_toy_e{i}", "sample": 0, "text": "Blue.",
                                                 "reasoning": None, "finish_reason": "stop", "truncated": False}
                                                for i in range(3)]
    write_jsonl(out / "answers.jsonl", rows)
    write_json(out / "manifest.json", {"split": "dev", "model": "toy", "config_id": "dev-smoke", "n_items": 7, "n": 1})
    krun.score(out)
    down = judges([("http", 503)])
    summary = _run(out, down.endpoint(), down.endpoint())
    assert len(down.requests) == 3 * judge_run.MAX_TRIES and summary["judge"] == "incomplete"


def test_a_verdict_with_the_wrong_ids_is_asked_again(scored_run, judges):
    wrong = VALID.replace('"P1"', '"P9"')
    p = judges([("ok", wrong, "stop"), ("ok", VALID, "stop")])
    _run(scored_run, p.endpoint(), p.endpoint())
    assert len(p.requests) == 2 and read_jsonl(scored_run / "judge_scores.jsonl")[0]["score"] == 1.0


def test_the_calibration_record_names_this_prompt(scored_run, judges, tmp_path):
    manifest = json.loads((scored_run / "manifest.json").read_text())
    write_json(scored_run / "manifest.json", {**manifest, "split": "test"})
    stale = tmp_path / "calibration_stale.json"
    write_json(stale, {"passed": True, "prompt_version": jp.PROMPT_VERSION, "prompt_sha256": "0" * 64})
    p = judges([("ok", VALID, "stop")])
    with pytest.raises(judge_run.JudgeRefused, match="calibration"):
        _run(scored_run, p.endpoint(), p.endpoint(), allow_test=True, calibration=stale)


def test_a_reused_verdict_is_scored_with_the_current_rubric(tmp_path, monkeypatch, judges):
    both = {"review": APPROVED, "required_points": ["Names red."], "disqualifying_errors": [],
            "numeric": {"value": 5, "unit": "V"}, "numeric_tolerance": "toy: as in numeric"}
    item = {"id": "q_toy_v", "question": "Toy question V: name a colour and give the voltage.", "rubric": both,
            "key": None}
    monkeypatch.setattr(krun, "load_split", lambda split: [dict(item)])
    out = tmp_path / "runv"
    out.mkdir()
    write_json(out / "manifest.json", {"split": "dev", "model": "toy", "config_id": "dev-smoke", "n_items": 1, "n": 1})
    write_jsonl(out / "answers.jsonl", [{"item_id": "q_toy_v", "sample": 0, "text": "Red. Final answer: 5 V",
                                         "reasoning": None, "finish_reason": "stop", "truncated": False}])
    krun.score(out)
    verdict = json.dumps({"points": [{"id": "P1", "verdict": "met", "reason": "Red."}],
                          "numbers": [{"id": "N1", "values": [{"value": 5, "unit": "V", "unit_written": True}],
                                       "reason": "5 V."}], "disqualifying_errors": [], "hallucinated_entities": []})
    p = judges([("ok", verdict, "stop")])
    _run(out, p.endpoint(), p.endpoint())
    assert read_jsonl(out / "judge_scores.jsonl")[0]["score"] == 1.0
    changed = {**both, "numeric": {"value": 10, "unit": "V"}}      # the judge never sees the reference value
    monkeypatch.setattr(krun, "load_split", lambda split: [{**item, "rubric": changed}])
    krun.score(out)
    _run(out, p.endpoint(), p.endpoint())
    assert len(p.requests) == 1                                     # the same request: the verdict is reused
    assert read_jsonl(out / "judge_scores.jsonl")[0]["score"] == 0.5  # and scored with the current rubric


def test_duplicate_queue_rows_are_refused(scored_run, judges):
    queue = read_jsonl(scored_run / "judge_queue.jsonl")
    write_jsonl(scored_run / "judge_queue.jsonl", queue + queue)
    p = judges([("ok", VALID, "stop")])
    with pytest.raises(judge_run.JudgeRefused, match="score the run again"):
        _run(scored_run, p.endpoint(), p.endpoint())


def test_a_stored_verdict_that_no_longer_fits_is_judged_again(tmp_path, monkeypatch, judges):
    both = {"review": APPROVED, "required_points": ["Names red."], "disqualifying_errors": [],
            "numeric": {"value": 7, "unit": "V"}, "numeric_tolerance": "toy: as in numeric"}
    item = {"id": "q_toy_w", "question": "Toy question W: name a colour and give the voltage.", "rubric": both,
            "key": None}
    monkeypatch.setattr(krun, "load_split", lambda split: [dict(item)])
    out = tmp_path / "runw"
    out.mkdir()
    write_json(out / "manifest.json", {"split": "dev", "model": "toy", "config_id": "dev-smoke", "n_items": 1, "n": 1})
    write_jsonl(out / "answers.jsonl", [{"item_id": "q_toy_w", "sample": 0, "text": "Red. Final answer: 5 V or 10 V",
                                         "reasoning": None, "finish_reason": "stop", "truncated": False}])
    krun.score(out)

    def verdict(unit):
        return json.dumps({"points": [{"id": "P1", "verdict": "met", "reason": "Red."}],
                           "numbers": [{"id": "N1", "values": [{"value": 5, "unit": unit, "unit_written": True}],
                                        "reason": "5."}], "disqualifying_errors": [], "hallucinated_entities": []})

    # against 7 V the rules decide the number, so the blank unit the judge reports is never graded
    p = judges([("ok", verdict(""), "stop"), ("ok", verdict("V"), "stop")])
    _run(out, p.endpoint(), p.endpoint())
    monkeypatch.setattr(krun, "load_split", lambda split: [{**item, "rubric": {**both, "numeric": {"value": 5, "unit": "V"}}}])
    krun.score(out)                                     # against 5 V the rules leave the number to the judge
    _run(out, p.endpoint(), p.endpoint())
    assert len(p.requests) == 2                         # the stored verdict does not fit: asked again
    assert read_jsonl(out / "judge_scores.jsonl")[0]["score"] == 1.0

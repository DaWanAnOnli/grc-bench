"""Tests for `./gb tier-k run | score` against a stub OpenAI-compatible server (no model; the dev split
only: a test-split run is refused before any request)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from grcbench.common.canary import CANARY_GUID
from grcbench.common.io import read_jsonl
from grcbench.common.paths import ITEMS
from grcbench.tier_k import run

N_DEV = len(read_jsonl(ITEMS / "tier_k" / "dev" / "questions.jsonl"))


class _Stub:
    def __init__(self, modes: dict[int, str] | None = None) -> None:
        self.requests: list[dict] = []
        self.modes = modes or {}
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:       # quiet
                pass

            def do_POST(self) -> None:                  # noqa: N802 - http.server API
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append(body)
                mode = stub.modes.get(len(stub.requests) - 1, stub.modes.get("default",
                                      "length" if len(stub.requests) == 1 else "ok"))
                message = {"role": "assistant", "content": "Reasoning.\nFinal answer: 30 kHz"}
                if mode == "canary_reasoning":
                    message["reasoning_content"] = f"thinking {CANARY_GUID}"
                reply = {"choices": [{"message": message, "finish_reason": "length" if mode == "length" else "stop"}],
                         "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
                if mode == "no_choices":
                    reply = {"error": {"message": "context exceeded"}}
                data = json.dumps(reply).encode()
                self.send_response(503 if mode == "http_error" else 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()


@pytest.fixture
def stub():
    s = _Stub()
    yield s
    s.server.shutdown()


@pytest.fixture
def failing_stub():
    s = _Stub({1: "http_error", 2: "http_error", 3: "http_error", 4: "http_error", 5: "no_choices", 6: "no_choices",
               7: "no_choices", 8: "no_choices", 9: "canary_reasoning"})
    yield s
    s.server.shutdown()


def test_run_writes_one_row_per_item_and_sample(stub, tmp_path):
    out = run.run("dev", base_url=stub.url, model="stub", config_id="dev-smoke", reason="test", n=2, out_dir=tmp_path)
    rows = read_jsonl(out / "answers.jsonl")
    assert len(rows) == 2 * N_DEV and {r["sample"] for r in rows} == {0, 1}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "dev" and manifest["answer_prompt_version"] == "tier-k-answer-v2"


def test_length_cut_answers_are_truncated(stub, tmp_path):
    out = run.run("dev", base_url=stub.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path)
    first = read_jsonl(out / "answers.jsonl")[0]
    assert first["truncated"] is True
    run.score(out)
    row = read_jsonl(out / "scores.jsonl")[0]
    assert row["credit"] == 0.0 and row["reason"] == "truncated"


def test_prompts_carry_no_canary(stub, tmp_path):
    run.run("dev", base_url=stub.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path)
    assert len(stub.requests) == N_DEV and CANARY_GUID not in json.dumps(stub.requests)


def test_test_split_is_refused(stub, tmp_path):
    for cid in ("dev-smoke", "nb-bf16-b0"):          # undeclared, and declared without allow_test
        with pytest.raises(run.RunRefused):
            run.run("test", base_url=stub.url, model="stub", config_id=cid, reason="x", out_dir=tmp_path)
    assert stub.requests == []


def test_score_routes_every_answer(stub, tmp_path):
    out = run.run("dev", base_url=stub.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path)
    s = run.score(out)
    assert s["n_answers"] == N_DEV and s["n_rules"] + s["n_judge"] + s["n_no_reference"] == N_DEV
    assert {"n_tie_break", "n_wrong_form", "n_redacted", "n_judge_failed"} <= set(s)
    assert len(read_jsonl(out / "scores.jsonl")) == N_DEV
    assert s["n_final_answer_line"] == N_DEV - 1                          # the length-cut answer has none


BLOCK_ITEM = {"id": "bq_toy", "item_kind": "block_question", "question": "Toy block question.",
              "rubric": None, "key": {"value": 34.0, "unit": "", "abs_tolerance": 0.01}}


def test_a_keyed_answer_the_rules_cannot_decide_goes_to_the_judge_with_a_key_rubric():
    assert run.route_answer(BLOCK_ITEM, "Final answer: 34")["credit"] == 1.0
    r = run.route_answer(BLOCK_ITEM, "Final answer: 34, or maybe 35")
    assert r["route"] == "judge" and r["judge_messages"]
    assert run.route_answer({**BLOCK_ITEM, "key": None}, "Final answer: 34")["route"] == "no_reference"
    assert run.route_answer(BLOCK_ITEM, "x", truncated=True)["reason"] == "truncated"
    assert run.route_answer(BLOCK_ITEM, "<think>only reasoning</think>")["reason"] == "empty_answer"
    r = run.route_answer(BLOCK_ITEM, f"Final answer: 34 {CANARY_GUID}")
    assert (r["route"], r["credit"], r["reason"]) == ("rules", 0.0, "canary_in_answer")


def test_a_calculation_key_is_never_a_reference():
    calc = {"id": "q_toy", "item_kind": "frozen_question", "question": "Toy?", "rubric": None,
            "key": {"value": 30, "unit": "kHz"}}
    assert run.reference_for(calc) is None and run.rubric_for(calc) is None


def test_the_request_body_keeps_its_fixed_keys(stub, tmp_path):
    run.run("dev", base_url=stub.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path,
            sampling={"temperature": 0.6, "top_k": 20}, limit=2)
    body = stub.requests[0]
    assert body["max_tokens"] == 4096 and body["top_k"] == 20 and body["model"] == "stub"
    with pytest.raises(ValueError):
        run.run("dev", base_url=stub.url, model="stub", config_id="dev-smoke", reason="test",
                out_dir=tmp_path / "b", sampling={"max_tokens": 32768}, limit=1)
    with pytest.raises(ValueError):
        run.run("dev", base_url=stub.url, model="stub", config_id="dev-smoke", reason="test",
                out_dir=tmp_path / "c", sampling={"max_completion_tokens": 32768}, limit=1)


def test_failed_requests_are_retried_then_recorded_not_scored(failing_stub, tmp_path):
    out = run.run("dev", base_url=failing_stub.url, model="stub", config_id="dev-smoke", reason="test",
                  out_dir=tmp_path, limit=4, retry_wait=0.0)
    rows = read_jsonl(out / "answers.jsonl")
    assert len(rows) == 4 and rows[1]["error"] and rows[2]["error"] and "error" not in rows[3]
    s = run.score(out)
    assert s["n_request_failed"] == 2 and s["by_reason"]["request_failed"] == 2
    assert s["n_canary"] == 1                                        # the canary sat in the reasoning field
    with pytest.raises(ValueError):
        run.run("dev", base_url=failing_stub.url, model="stub", config_id="dev-smoke", reason="test",
                out_dir=tmp_path, limit=1)                          # out_dir already holds a run


def test_an_incomplete_run_is_not_scored(stub, tmp_path):
    out = run.run("dev", base_url=stub.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path, limit=3)
    lines = (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    (out / "answers.jsonl").write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        run.score(out)


def test_a_test_split_run_with_allow_test_still_needs_the_declared_list(stub, tmp_path):
    log = tmp_path / "log.jsonl"
    with pytest.raises(run.RunRefused):
        run.run("test", base_url=stub.url, model="stub", config_id="dev-smoke", reason="x", out_dir=tmp_path / "o",
                allow_test=True, log_path=log)
    assert stub.requests == [] and not log.exists()


def test_early_zeros_on_keyed_items_count_in_the_rule_only_summary():
    r = run.route_answer(BLOCK_ITEM, "x", truncated=True)
    assert r["rules_result"]["details"]["reason"] == "truncated"
    r = run.route_answer(BLOCK_ITEM, "Final answer: 34", reasoning=f"I recall {CANARY_GUID}")
    assert r["reason"] == "canary_in_answer" and r["rules_result"]["credit"] == 0.0


def test_an_approved_but_malformed_rubric_is_an_error():
    bad = {**BLOCK_ITEM, "rubric": {"numeric": {"value": 34, "unit": ""}, "numeric_tolerance": None,
                                    "review": {"status": "approved"}}}
    with pytest.raises(ValueError, match="bq_toy"):
        run.rubric_for(bad)


def test_a_server_that_stops_answering_stops_the_run(tmp_path):
    dead = _Stub({"default": "http_error"})
    try:
        with pytest.raises(run.ChatError):
            run.run("dev", base_url=dead.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path,
                    limit=10, retry_wait=0.0)
        assert len(dead.requests) == run.MAX_TRIES * run.MAX_FAILED_IN_A_ROW
    finally:
        dead.server.shutdown()


def test_score_checks_every_item_and_sample_once(stub, tmp_path):
    out = run.run("dev", base_url=stub.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path, limit=3)
    lines = (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    (out / "answers.jsonl").write_text("\n".join([lines[0], lines[0], lines[2]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        run.score(out)

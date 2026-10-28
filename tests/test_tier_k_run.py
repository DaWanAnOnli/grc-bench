"""Tests for `./gb tier-k run | score` against a stub OpenAI-compatible server (no model; the dev split
only: a test-split run is refused before any request)."""

from __future__ import annotations

import inspect
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from grcbench.common.canary import CANARY_GUID
from grcbench.common.io import read_jsonl
from grcbench.common.paths import ITEMS
from grcbench.tier_k import answer_prompt, run

N_DEV = len(read_jsonl(ITEMS / "tier_k" / "dev" / "questions.jsonl"))


class _Stub:
    def __init__(self, modes: dict[int, str] | None = None, n_ctx: int | None = None, delay: float = 0.0,
                 slots: int | None = None, delays: dict[int, float] | None = None) -> None:
        self.requests: list[dict] = []
        self.modes = modes or {}
        self.in_flight = self.max_in_flight = 0
        self.times: dict[int, tuple[float, float]] = {}       # request index -> (start, end)
        self.delays = delays or {}                             # request index -> seconds (else `delay`)
        self.lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:       # quiet
                pass

            def do_GET(self) -> None:                   # noqa: N802 - llama-server /props
                if self.path != "/props" or n_ctx is None:
                    self.send_error(404)
                    return
                props = {"default_generation_settings": {"n_ctx": n_ctx}}
                if slots is not None:
                    props["total_slots"] = slots
                data = json.dumps(props).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:                  # noqa: N802 - http.server API
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with stub.lock:
                    stub.requests.append(body)
                    index = len(stub.requests) - 1
                    stub.in_flight += 1
                    stub.max_in_flight = max(stub.max_in_flight, stub.in_flight)
                start = time.monotonic()
                time.sleep(stub.delays.get(index, delay))
                stub.times[index] = (start, time.monotonic())
                with stub.lock:
                    stub.in_flight -= 1
                mode = stub.modes.get(index, stub.modes.get("default", "length" if index == 0 else "ok"))
                message = {"role": "assistant", "content": "Reasoning.\nFinal answer: 30 kHz"}
                if mode == "canary_reasoning":
                    message["reasoning_content"] = f"thinking {CANARY_GUID}"
                # "length": cut at the request's limit; "short_length": cut earlier (the server's context is full)
                used = body.get("max_tokens", 5) if mode == "length" else 5
                reply = {"choices": [{"message": message,
                                      "finish_reason": "length" if mode in ("length", "short_length") else "stop"}],
                         "usage": {"prompt_tokens": 10, "completion_tokens": used}}
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
    assert manifest["split"] == "dev" and manifest["answer_prompt_version"] == "tier-k-answer-v3"


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


def test_a_server_context_too_small_for_the_answer_limit_is_refused(tmp_path):
    small, large = _Stub(n_ctx=8192), _Stub(n_ctx=32768)
    try:
        with pytest.raises(run.RunRefused, match="context"):
            run.run("dev", base_url=small.url, model="stub", config_id="dev-smoke", reason="test",
                    out_dir=tmp_path / "small", limit=1)
        assert small.requests == []                                   # refused before any request
        run.run("dev", base_url=large.url, model="stub", config_id="dev-smoke", reason="test",
                out_dir=tmp_path / "large", limit=1)
        assert json.loads((tmp_path / "large" / "manifest.json").read_text())["server_context"] == 32768
    finally:
        small.server.shutdown()
        large.server.shutdown()


def test_a_b1_run_needs_room_for_the_retrieved_documentation(tmp_path):
    s = _Stub(n_ctx=answer_prompt.MAX_ANSWER_TOKENS + 4096)
    try:
        run.run("dev", base_url=s.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path / "b0",
                limit=1)                                                   # enough for B0
        with pytest.raises(run.RunRefused, match="context"):
            run.run("dev", base_url=s.url, model="stub", config_id="dev-smoke", reason="test",
                    out_dir=tmp_path / "b1", limit=1, b1=True)
    finally:
        s.server.shutdown()


def test_a_reply_cut_by_a_full_server_context_is_an_error_not_a_truncation(tmp_path):
    s = _Stub({0: "short_length", "default": "ok"})
    try:
        run.run("dev", base_url=s.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path,
                limit=1, retry_wait=0)
    finally:
        s.server.shutdown()
    row = read_jsonl(tmp_path / "answers.jsonl")[0]
    assert "context" in row["error"] and row["truncated"] is False
    assert len(s.requests) == 1                                 # the same prompt cannot fit: not sent again


def test_the_request_timeout_fits_a_long_answer(stub, tmp_path):
    assert inspect.signature(run.run).parameters["timeout"].default == 1800.0
    run.run("dev", base_url=stub.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path, limit=1)
    assert json.loads((tmp_path / "manifest.json").read_text())["settings"]["timeout_s"] == 1800.0


def test_parallel_requests_keep_the_row_order(tmp_path):
    seq, par = _Stub({"default": "ok"}), _Stub({"default": "ok"}, delay=0.05)
    try:
        run.run("dev", base_url=seq.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path / "s",
                limit=6, n=2)
        run.run("dev", base_url=par.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path / "p",
                limit=6, n=2, parallel=4)
    finally:
        seq.server.shutdown()
        par.server.shutdown()

    def key(r):
        return r["item_id"], r["sample"], r["text"], r["finish_reason"], r["truncated"]

    assert [key(r) for r in read_jsonl(tmp_path / "s" / "answers.jsonl")] == \
        [key(r) for r in read_jsonl(tmp_path / "p" / "answers.jsonl")]
    assert 1 < par.max_in_flight <= 4
    assert json.loads((tmp_path / "p" / "manifest.json").read_text())["settings"]["parallel"] == 4


def test_a_long_answer_does_not_hold_the_other_slots(tmp_path):
    # the first request takes 1 s, the others 0.05 s: with 4 slots, all 8 others run while it runs
    s = _Stub({"default": "ok"}, delay=0.05, delays={0: 1.0})
    try:
        run.run("dev", base_url=s.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path,
                limit=9, parallel=4)
    finally:
        s.server.shutdown()
    first_end = s.times[0][1]
    assert sum(1 for i, (start, _) in s.times.items() if i and start < first_end) == 8
    rows = read_jsonl(tmp_path / "answers.jsonl")
    assert [r["item_id"] for r in rows] == [i["id"] for i in run.load_split("dev")[:9]]      # still in order


def test_failures_in_a_row_stop_a_parallel_run(tmp_path):
    s = _Stub({"default": "http_error"})
    try:
        with pytest.raises(run.ChatError, match="in a row"):
            run.run("dev", base_url=s.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path,
                    limit=8, retry_wait=0, parallel=2)
    finally:
        s.server.shutdown()
    assert len(read_jsonl(tmp_path / "answers.jsonl")) == run.MAX_FAILED_IN_A_ROW


def test_parallel_requests_need_as_many_server_slots(tmp_path):
    s = _Stub(n_ctx=32768, slots=1)
    try:
        with pytest.raises(run.RunRefused, match="slots"):
            run.run("dev", base_url=s.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path / "p",
                    limit=1, parallel=4)
        run.run("dev", base_url=s.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path / "s",
                limit=1)
    finally:
        s.server.shutdown()
    assert json.loads((tmp_path / "s" / "manifest.json").read_text())["server_slots"] == 1


def test_a_server_without_props_is_not_checked(stub, tmp_path):
    run.run("dev", base_url=stub.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path, limit=1)
    assert json.loads((tmp_path / "manifest.json").read_text())["server_context"] is None


def test_a_calculation_key_is_never_a_reference():
    calc = {"id": "q_toy", "item_kind": "frozen_question", "question": "Toy?", "rubric": None,
            "key": {"value": 30, "unit": "kHz"}}
    assert run.reference_for(calc) is None and run.rubric_for(calc) is None


def test_the_request_body_keeps_its_fixed_keys(stub, tmp_path):
    run.run("dev", base_url=stub.url, model="stub", config_id="dev-smoke", reason="test", out_dir=tmp_path,
            sampling={"temperature": 0.6, "top_k": 20}, limit=2)
    body = stub.requests[0]
    assert body["max_tokens"] == 16384 and body["top_k"] == 20 and body["model"] == "stub"
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

"""Tier-A smoke test (grcbench.tier_a.smoke) and the streamed request path (models.py "Streaming").

The fast tests need no llama-server: the GGUF reader, the server command, the grammar-probe reading
on canned replies and the report. The slow test runs one fixture episode twice through the real
runner with GRC-Agent's own OpenAIChatModel over a scripted transport, streamed and not, and checks
that streaming changes nothing the model is sent except the stream flags.
"""

from __future__ import annotations

import json
import shutil
import struct
import uuid
from pathlib import Path

import pytest

from grcbench.common.paths import FIXTURES, WORK
from grcbench.tier_a import models, runner, smoke, trace

TASKS = FIXTURES / "tier_a" / "tasks"
MODIFY = TASKS / "fx-a-modify-tone"
SET_2K = {"reason": "set the tone to 2 kHz", "update_params": [{"instance_name": "src", "params": {"freq": "2000"}}]}


# ================================================================ GGUF header and server command

def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path: Path, kvs: list[tuple[str, int, bytes]]) -> Path:
    """A GGUF v3 header with the given (key, value type, packed value) pairs and no tensors."""
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    for key, vtype, value in kvs:
        out += _gguf_string(key) + struct.pack("<I", vtype) + value
    path.write_bytes(out)
    return path


def test_gguf_reader_finds_the_native_context(tmp_path):
    tokens = struct.pack("<I", 8) + struct.pack("<Q", 3) + b"".join(_gguf_string(t) for t in ("a", "bb", "<tool_call>"))
    nested = struct.pack("<I", 9) + struct.pack("<Q", 1) + struct.pack("<I", 4) + struct.pack("<Q", 2) + struct.pack("<II", 1, 2)
    path = write_gguf(tmp_path / "m.gguf", [
        ("general.architecture", 8, _gguf_string("qwen3")),
        ("tokenizer.ggml.tokens", 9, tokens),
        ("weird.nested", 9, nested),
        ("qwen3.block_count", 4, struct.pack("<I", 36)),
        ("qwen3.context_length", 4, struct.pack("<I", 40960)),
        ("general.flag", 7, struct.pack("<?", True)),
    ])
    meta = smoke.gguf_metadata(path)
    assert meta == {"general.architecture": "qwen3", "qwen3.block_count": 36, "qwen3.context_length": 40960,
                    "general.flag": True}
    assert smoke.native_context(path) == 40960
    assert smoke.server_context(40960) == 40960
    assert smoke.server_context(131072) == 65536 and smoke.server_context(None) == 65536
    (tmp_path / "x.gguf").write_bytes(b"GGML" + b"\0" * 20)
    with pytest.raises(ValueError, match="not a GGUF"):
        smoke.gguf_metadata(tmp_path / "x.gguf")


def test_server_command_has_the_smoke_engine_settings(monkeypatch):
    monkeypatch.setenv("GRC_BENCH_LLAMA_BACKEND", "cpu")
    cmd = smoke.server_command(8123, ctx=40960, threads=6)
    joined = " ".join(cmd)
    assert cmd[0].endswith("b10333/llama-server")
    for flags in (["--host", "127.0.0.1"], ["--port", "8123"], ["-np", "1"], ["-c", "40960"], ["-ctk", "q8_0"],
                  ["-ctv", "q8_0"], ["-fa", "on"], ["--reasoning-format", "deepseek"], ["--sse-ping-interval", "30"],
                  ["-t", "6"], ["-tb", "6"], ["--temp", "0.6"], ["--top-k", "20"], ["--top-p", "0.95"], ["--min-p", "0"]):
        i = cmd.index(flags[0])
        assert cmd[i:i + 2] == flags, flags
    assert "--jinja" in cmd and "--no-context-shift" in cmd and "--skip-chat-parsing" not in joined
    assert smoke.server_command(1, ctx=4096, extra=("--skip-chat-parsing",))[-1] == "--skip-chat-parsing"
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "4096")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/x")
    env = smoke.server_env()
    assert "LLAMA_ARG_CTX_SIZE" not in env and env["LD_LIBRARY_PATH"] == f"{smoke.LLAMA_DIR}:/x"
    assert "-ngl" not in cmd


def test_cuda_backend_uses_the_gpu_build_and_offloads_every_layer(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/x")
    cmd = smoke.server_command(8123, ctx=4096, backend="cuda")
    assert cmd[0] == str(smoke.LLAMA_DIR_CUDA / "llama-server") and cmd[cmd.index("-ngl") + 1] == smoke.GPU_LAYERS
    env = smoke.server_env(backend="cuda")
    assert env["LD_LIBRARY_PATH"] == f"{smoke.LLAMA_DIR_CUDA}:{smoke.WSL_GPU_LIBS}:/x"
    with pytest.raises(ValueError, match="llama backend"):
        smoke.llama_backend("rocm")


def test_smoke_refuses_non_local_servers_and_non_fixtures(tmp_path):
    with pytest.raises(ValueError, match="local"):
        smoke.server_props("http://10.0.0.1:8080/v1")
    with pytest.raises(smoke.NotAFixture):
        smoke._tier_f_dirs(["no_such_fixture"])
    assert [d.name for d in smoke._tier_f_dirs(list(smoke.DEFAULT_TIER_F_TASKS))] == ["fx_modify_tone"]


# ================================================================ grammar probe reading

def _reply(content: str = "", calls: list[tuple[str, str]] = (), reasoning: str = "") -> dict:
    msg = {"role": "assistant", "content": content or None}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": a}}
                             for i, (n, a) in enumerate(calls)]
    return {"choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def _cases(reply_for: dict, refused: bool = True) -> list[dict]:
    out = []
    for case in smoke.PROBE_CASES:
        if case["name"] == "own_grammar_with_tools" and refused:
            out.append(smoke.probe_record(case, 400, {"error": {"message": "Cannot use custom grammar constraints with tools."}}, 0.1))
        else:
            out.append(smoke.probe_record(case, 200, reply_for, 1.0))
    return out


def test_probe_reading_constrained_default_and_unconstrained_control():
    valid = _reply(calls=[("set_mode", '{"mode": "alpha", "level": 1}')], reasoning="hm")
    violating_text = _reply(content='<tool_call>\n{"name": "set_mode", "arguments": {"mode": "gamma", "level": "high"}}\n</tool_call>')
    servers = {"default": {"cases": _cases(valid), "log_lines": ["srv params_from_: Grammar (lazy): root ::= ..."]},
               "control_skip_chat_parsing": {"cases": _cases(violating_text), "log_lines": []}}
    v = smoke.probe_verdict(servers)
    assert v["default_parsed_tool_calls"] == 3 and v["default_calls_schema_valid"] == 3
    assert v["control_calls_violating"] == 3 and v["own_grammar_with_tools_refused"]
    assert v["grammar_constrains_tool_calls"] is True
    rec = servers["control_skip_chat_parsing"]["cases"][0]
    assert rec["text_tool_calls"][0]["name"] == "set_mode" and rec["text_tool_calls"][0]["schema_errors"]
    assert servers["default"]["cases"][0]["reasoning_chars"] == 2


def test_probe_log_evidence_finds_schema_grammar_rules():
    log = "\n".join(["srv  params_from_: Chat format: peg-native",
                     "srv  params_from_: Grammar (lazy): root ::= tool-call",
                     'set-mode-mode ::= ("\\"alpha\\"" | "\\"beta\\"") space',
                     "prompt: ... alpha ... beta ...",
                     "srv  params_from_: Grammar trigger (pattern): <tool_call>"])
    ev = smoke.probe_log_evidence(log)
    assert len(ev["grammar_lines"]) == 2 and len(ev["enum_rule_lines"]) == 1 and len(ev["trigger_lines"]) == 2
    assert ev["chat_format_lines"] == ["srv  params_from_: Chat format: peg-native"]
    valid = _reply(calls=[("set_mode", '{"mode": "beta", "level": 2}')])
    violating_text = _reply(content='<tool_call>{"name": "set_mode", "arguments": {"mode": "gamma", "level": 3}}</tool_call>')
    v = smoke.probe_verdict({"default": {"cases": _cases(valid), "log_evidence": ev},
                             "control_skip_chat_parsing": {"cases": _cases(violating_text), "log_evidence": smoke.probe_log_evidence("")}})
    assert v["grammar_constrains_tool_calls"] and v["default_enum_rule_lines"] == 1 and v["control_grammar_log_lines"] == 0


def test_probe_reading_unconstrained_default_is_not_called_constrained():
    violating = _reply(calls=[("set_mode", '{"mode": "gamma", "level": "high"}')])
    servers = {"default": {"cases": _cases(violating, refused=False), "log_lines": ["Grammar (lazy): x"]},
               "control_skip_chat_parsing": {"cases": _cases(violating), "log_lines": []}}
    v = smoke.probe_verdict(servers)
    assert v["default_calls_schema_valid"] == 0 and v["grammar_constrains_tool_calls"] is False
    assert not v["own_grammar_with_tools_refused"]
    assert smoke.calls_in_text("x <tool_call>{bad json</tool_call>")[0]["parsed"] is False
    assert smoke.probe_body({"name": "required", "tool_choice": "required"})["tool_choice"] == "required"
    assert smoke.probe_body({"name": "t", "thinking": True})["chat_template_kwargs"] == {"enable_thinking": True}


def test_tool_call_settings_b0_b2_from_a_constrained_default():
    valid = _reply(calls=[("set_mode", '{"mode": "alpha", "level": 1}')])
    violating_text = _reply(content='<tool_call>{"name": "set_mode", "arguments": {"mode": "gamma", "level": "high"}}</tool_call>')
    no_jinja = [smoke.probe_record(c, 500, {"error": {"message": "tools param requires --jinja flag"}}, 0.1)
                for c in smoke.PROBE_CASES]
    servers = {
        "default": {"cases": _cases(valid), "log_evidence": smoke.probe_log_evidence(
            "0.22.2 I srv    operator(): chat format: peg-native\n0.22.2 D srv    operator(): Grammar (tool_calls): root ::= x")},
        "control_skip_chat_parsing": {"cases": _cases(violating_text), "log_evidence": smoke.probe_log_evidence(
            "0.1 I srv    operator(): chat format: Content-only")},
        "no_jinja": {"cases": no_jinja, "log_evidence": smoke.probe_log_evidence("")}}
    v = smoke.probe_verdict(servers)
    assert v["chat_formats"] == {"default": "peg-native", "control_skip_chat_parsing": "Content-only", "no_jinja": None}
    assert v["control_parsed_tool_calls"] == 0 and v["no_jinja_tool_requests_refused"] == 3 and v["no_jinja_cases"] == 3
    s = smoke.tool_call_settings(v)
    assert s["default_is_grammar_constrained"] and not s["flag_only_unconstrained_setting_with_parsed_calls"]
    assert s["B2"]["chat_format"] == "peg-native" and s["B2"]["tool_calls_schema_constrained"]
    assert s["B2"]["usable_by_grc_agent_as_is"] and s["B2"]["grammar_in_log"]
    b0 = s["B0_candidate"]
    assert b0["server_flags"] == "--jinja --skip-chat-parsing" and not b0["tool_calls_schema_constrained"]
    assert not b0["server_parses_tool_calls"] and not b0["usable_by_grc_agent_as_is"] and not b0["grammar_in_log"]
    assert s["client_grammar_with_tools_refused"] and s["no_jinja"]["error"] == "tools param requires --jinja flag"
    text = "\n".join(smoke._settings_section(s))
    assert "the server's default IS the B2 setting" in text and "(HTTP 400)" in text
    assert "| B0 candidate: calls as the model writes them | `--jinja --skip-chat-parsing` | Content-only | no | no | no | no |" in text
    summary = {"probe": {"servers": servers, "verdict": v, "tool_call_settings": s}, "golden": [], "model": {}}
    report = smoke.render_report(summary)
    assert "| no_jinja | auto | 500 | refused: tools param requires --jinja flag |" in report
    assert "grammar-constrains tool calls for this model in this build: yes" in report and text in report
    # a default that does NOT hold calls to the schema is never read as B2
    loose = smoke.tool_call_settings(smoke.probe_verdict({**servers, "default": {"cases": _cases(violating_text)}}))
    assert not loose["default_is_grammar_constrained"]
    assert "did not show the default server" in "\n".join(smoke._settings_section(loose))


def test_request_log_evidence_and_a_working_request_level_b0():
    log = ("0.22.211.649 D srv    operator(): Grammar (tool_calls): integer ::= (\"-\"? integral-part)\n"
           "root ::= tool-call | tool-calls\n0.22.211.651 I srv    operator(): chat format: peg-native\n"
           "0.35.514.210 D Grammar still awaiting trigger after token 151657 (`<tool_call>`)\n"
           "0.41.369.600 D Grammar triggered on regex: '<tool_call>")
    assert smoke.request_log_evidence(log) == {
        "grammar_kinds": ["tool_calls"], "root_rules": ["root ::= tool-call | tool-calls"], "triggered": True,
        "awaiting_trigger": True, "chat_format": "peg-native"}
    assert smoke.request_log_evidence("x")["grammar_kinds"] == [] and smoke.request_log_evidence("x")["chat_format"] is None
    valid = _reply(calls=[("set_mode", '{"mode": "alpha", "level": 1}')])
    violating = _reply(calls=[("set_mode", '{"mode": "gamma", "level": "high"}')])
    violating_text = _reply(content='<tool_call>{"name": "set_mode", "arguments": {"mode": "gamma", "level": "x"}}</tool_call>')
    default = []
    for case in smoke.PROBE_CASES:
        if case["role"] == "own_grammar":
            rec = smoke.probe_record(case, 400, {"error": {"message": "Cannot use custom grammar constraints with tools."}}, 0.1)
        else:
            rec = smoke.probe_record(case, 200, violating if case["role"] == "b0_request" else valid, 1.0)
        rec["log"] = smoke.request_log_evidence("" if case["role"] == "b0_request" else log)
        default.append(rec)
    servers = {"default": {"cases": default, "log_evidence": smoke.probe_log_evidence(log)},
               "control_skip_chat_parsing": {"cases": _cases(violating_text)}}
    v = smoke.probe_verdict(servers)
    assert v["default_parsed_tool_calls"] == 3 and v["default_schema_constrained"]      # b0_request not counted
    assert v["default_b0_request"]["parsed_calls_violating"] == 1 and v["default_b0_request"]["grammar_kinds"] == []
    assert v["default_case_grammars"]["auto"]["triggered"] is True
    s = smoke.tool_call_settings(v)
    assert s["request_level_unconstrained_with_parsed_calls"] and s["default_is_grammar_constrained"]
    assert s["B0_request_candidate"]["server_parses_tool_calls"] and not s["B0_request_candidate"]["grammar_in_log"]
    text = "\n".join(smoke._settings_section(s))
    assert "possible on the same server without a flag, at the request level" in text
    assert "| B0 request-level candidate: tool_choice \"none\" + \"parse_tool_calls\": true" in text
    report = smoke.render_report({"probe": {"servers": servers, "verdict": v, "tool_call_settings": s}})
    assert "| 1.0 | tool_calls (triggered); format peg-native |" in report
    assert smoke.probe_body(smoke.PROBE_CASES[-1])["parse_tool_calls"] is True
    assert smoke.probe_body(smoke.PROBE_CASES[-1])["tool_choice"] == "none"


def test_server_timing_from_print_timing_lines():
    log = "\n".join([
        "2.58.689.688 I slot print_timing: id  0 | task 0 | prompt eval time =   13302.08 ms /   237 tokens "
        "(   56.13 ms per token,    17.82 tokens per second)",
        "2.58.689.689 I slot print_timing: id  0 | task 0 |        eval time =  143175.43 ms /    26 tokens "
        "( 5506.75 ms per token,     0.18 tokens per second)",
        "3.00.000.000 I slot print_timing: id  0 | task 9 | prompt eval time =    6697.92 ms /   263 tokens",
        "3.00.000.001 I slot print_timing: id  0 | task 9 |        eval time =    6824.57 ms /    74 tokens",
        "3.00.000.002 I slot      release: id  0 | task 9 | stop processing: n_tokens = 600, truncated = 0"])
    t = smoke.server_timing(log)
    assert t["prompt"] == {"requests": 2, "tokens": 500, "seconds": 20.0, "tokens_per_second": 25.0}
    assert t["generation"] == {"requests": 2, "tokens": 100, "seconds": 150.0, "tokens_per_second": 0.67}
    assert smoke.server_timing("")["prompt"]["tokens_per_second"] is None
    assert "generation 100 tokens in 150.0 s (0.67 tokens/s)" in smoke._timing_lines(t)[0]
    assert smoke._timing_lines({}) == []


def test_machine_state_records_load_and_llama_processes():
    m = smoke.machine_state()
    assert m["cpus"] >= 1 and len(m["load_1_5_15"]) == 3 and isinstance(m["other_llama_processes"], list)
    lines = smoke._machine_lines({"start": m, "end": m})
    assert len(lines) == 2 and lines[0].startswith("| CPU load (start, ")


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


@pytest.mark.slow
def test_server_child_dies_with_its_parent(tmp_path):
    """LocalServer's preexec_fn (PR_SET_PDEATHSIG): a child started with it, in its own session as
    the server is, gets SIGTERM when the process that started it dies, even by os._exit."""
    import os
    import subprocess
    import sys
    import time
    pid_file = tmp_path / "child.pid"
    code = ("import os, subprocess; from grcbench.tier_a import smoke; "
            "p = subprocess.Popen(['sleep', '120'], start_new_session=True, preexec_fn=smoke._die_with_parent()); "
            f"open({str(pid_file)!r}, 'w').write(str(p.pid)); os._exit(0)")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _alive(pid):
        os.kill(pid, 9)
        pytest.fail("the child outlived its parent")


# ================================================================ report

def test_report_says_plumbing_and_lists_every_episode():
    summary = {
        "started": "2026-10-18T00:00:00Z", "finished": "2026-10-18T01:00:00Z", "root": "/w/smoke/x",
        "path": "/w/smoke/x/summary.json", "notes": ["a note"], "golden_passed": True,
        "golden": [{"tier": "A", "task": "fx-a-modify-tone", "sample": 0, "passed": True, "status": "completed",
                    "tool_calls": 9}],
        "server": {"llama_cpp_build": "b10333", "model_file": "/c/Qwen3-4B-Q4_K_M.gguf", "model_sha256": "ab" * 32,
                   "native_context": 40960, "context": 40960, "context_cap": 65536, "kv_cache": "q8_0",
                   "reasoning_format": "deepseek", "sse_ping_seconds": 30, "threads": 16, "context_shift": False,
                   "slots": 1, "command": ["llama-server", "-np", "1"]},
        "probe": None,
        "model": {"base_url": "http://127.0.0.1:1/v1", "model_name": "m", "props": {"n_ctx": 40960},
                  "episodes": [{"task": "fx-a-modify-tone", "completed": True, "status": "completed",
                                "end_reason": "agent_finished", "requests": 7, "tool_calls": 6, "malformed_calls": 1,
                                "invalid_json_calls": 0, "unknown_tool_calls": 0, "unparsed_tool_call_texts": 0,
                                "caps_hit": [], "passed": False, "wall_seconds": 600, "tools_used": ["inspect_graph"],
                                "reasons": ["blocks_present[0]: freq expects 2000"],
                                "malformed_examples": [{"tool": "change_graph", "errors": ["x: bad"]}]}],
                  "tier_f": [{"task": "fx-modify-tone", "sample": 0, "lookup_used": True, "lookup_query": "signal source",
                              "lookup_domain": "catalog", "lookup_k": 3, "edit_called": True, "edit_well_formed": True,
                              "edit_accepted": True, "passed": True, "model_requests": 2,
                              "end_reason": "tier_f_edit_applied", "wall_seconds": 120, "edit_raw_args": "{}"}]},
        "traces": {"path": "/w/smoke/x/traces.jsonl", "rows": 2, "rows_off_format": 0},
    }
    text = smoke.render_report(summary)
    assert smoke.BANNER in text and "not a model result" in text
    assert "| fx-a-modify-tone | yes (completed) | agent_finished | 7 | 6 | 1 (0) | 0 | 0 | none | fail | 10.0 min |" in text
    assert "| fx-modify-tone | 0 | yes (signal source, catalog, k=3) | yes | yes | yes | PASS (V1-V5) | 2 |" in text
    assert "grader: blocks_present[0]: freq expects 2000" in text and "- a note" in text
    assert "The grammar probe did not run" in text
    assert smoke.render_report(summary) == text


# ================================================================ streaming

def test_model_spec_streaming_defaults():
    assert models.openai_compatible().streams() and models.openai_compatible().describe()["stream"] is True
    assert not models.openai_compatible(stream=False).streams()
    assert not models.scripted([{"text": "x"}]).streams() and not models.wire_scripted([]).streams()
    assert models.wire_scripted([], stream=True).streams()
    with pytest.raises(ValueError, match="do not stream"):
        models.ModelSpec("scripted", stream=True)
    old = models.openai_compatible().to_dict()
    old.pop("stream")
    assert models.ModelSpec.from_dict(old).streams()            # records written before the field existed


def test_wire_script_stream_chunks_rebuild_the_completion():
    ws = models.WireScript([{"text": "hi", "reasoning_content": "think",
                             "tool_calls": [{"id": "a1", "name": "inspect_graph", "arguments": {}}]}],
                           "text", "m", Path("/nonexistent"))
    completion = ws.completion(0)
    chunks = ws.stream_chunks(completion)
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    assert deltas[0]["role"] == "assistant" and {"reasoning_content": "think"} in deltas and {"content": "hi"} in deltas
    calls = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
    assert calls[0]["id"] == "a1" and calls[0]["function"]["name"] == "inspect_graph"
    assert "".join(c["function"]["arguments"] for c in calls) == "{}"
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls" and chunks[-1]["usage"] == completion["usage"]


@pytest.fixture(scope="module")
def root():
    path = WORK / "tier_a" / "pytest" / f"smoke-{uuid.uuid4().hex[:10]}"
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.gnuradio
def test_streamed_episode_sends_the_same_requests_as_unstreamed(root, tmp_path):
    turns = [{"reasoning_content": "First look at the graph.",
              "tool_calls": [{"id": "call_inspect_0000000000000000000", "name": "inspect_graph", "arguments": "{}"}],
              "usage": {"prompt_tokens": 1200, "completion_tokens": 30}},
             {"tool_calls": [{"id": "call_change_00000000000000000000", "name": "change_graph",
                              "arguments": json.dumps(SET_2K)}],
              "usage": {"prompt_tokens": 1500, "completion_tokens": 40}},
             {"text": "The tone is now 2 kHz.", "usage": {"prompt_tokens": 1600, "completion_tokens": 9}}]
    guard = {"log_path": tmp_path / "log.jsonl", "run_list": tmp_path / "run_list.yaml"}
    res = {}
    for stream in (False, True):
        res[stream] = runner.run_episode(MODIFY, models.wire_scripted(turns, stream=stream), "benchmark", root,
                                         config_id=f"pytest-stream-{int(stream)}", **guard)
        assert res[stream].status == "completed", res[stream].error
        assert res[stream].harness["streaming"] is stream
    wires = {s: [json.loads(line) for line in (Path(r.run_dir) / models.WIRE_FILE).read_text().splitlines()]
             for s, r in res.items()}
    assert len(wires[False]) == len(wires[True]) == 3
    for plain, streamed in zip(wires[False], wires[True]):
        a = json.loads(plain["body"].replace(res[False].run_dir, "<RUN>"))
        b = json.loads(streamed["body"].replace(res[True].run_dir, "<RUN>"))
        assert b.pop("stream") is True and a.pop("stream") is False
        assert b.pop("stream_options") == {"include_usage": True} and "stream_options" not in a
        assert a == b
    for r in res.values():
        assert [c["tool"] for c in r.tool_calls] == ["inspect_graph", "change_graph"]
        assert all(c["execution"] == "ok" and c["schema_valid"] for c in r.tool_calls)
        assert [q["input_tokens"] for q in r.model_requests] == [1200, 1500, 1600]
        row = trace.export_run_dir(Path(r.run_dir))
        assert row["messages"][1]["thinking"] == "First look at the graph." and not trace.row_format_errors(row)
    msgs = {s: json.dumps(trace.export_run_dir(Path(r.run_dir))["messages"]).replace(r.run_dir, "<RUN>")
            for s, r in res.items()}
    assert msgs[True] == msgs[False]


# ================================================================ run_smoke against a fake llama-server

class FakeLlamaServer:
    """A local OpenAI-compatible server shaped like llama-server: /health, /props, /v1/models and a
    streaming /v1/chat/completions that sends an SSE ping comment first. It answers a conversation
    whose last message is the user's with one inspect_graph call, and anything else with a short
    text reply (so a Tier-F sample ends without an edit)."""

    def __init__(self) -> None:
        import http.server
        import threading
        fake = self
        self.bodies: list[dict] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):          # noqa: ANN002 - silence the test output
                pass

            def _json(self, obj: dict, status: int = 200) -> None:
                raw = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:              # noqa: N802 - http.server API
                if self.path == "/health":
                    self._json({"status": "ok"})
                elif self.path == "/props":
                    self._json({"default_generation_settings": {"n_ctx": 8192}, "build_info": "fake-b0",
                                "chat_template": "{{ tools }}", "total_slots": 1})
                elif self.path == "/v1/models":
                    self._json({"data": [{"id": "fake", "meta": {"n_ctx_train": 8192}}]})
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self) -> None:             # noqa: N802 - http.server API
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.bodies.append(body)
                n = len(fake.bodies)
                last = body["messages"][-1]["role"]
                turn = ({"reasoning_content": "Inspect first.",
                         "tool_calls": [{"id": f"call{n:029d}", "name": "inspect_graph", "arguments": "{}"}]}
                        if last == "user" else {"text": "Done, nothing else to change."})
                turn["usage"] = {"prompt_tokens": 900 + n, "completion_tokens": 12}
                completion = models.WireScript([turn], "text", "fake", Path("/nonexistent")).completion(0)
                if not body.get("stream"):
                    self._json(completion)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b": ping\n\n")
                for chunk in models.WireScript.stream_chunks(completion):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def __enter__(self) -> "FakeLlamaServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.mark.slow
@pytest.mark.gnuradio
def test_run_smoke_end_to_end_over_http_with_a_fake_server(root, tmp_path):
    report = tmp_path / "tier_a_smoke.md"
    with FakeLlamaServer() as fake:
        summary = smoke.run_smoke(base_url=fake.base_url, golden=False, only=["fx-a-knowledge-tone", "fx_modify_tone"],
                                  tier_f_samples=1, run_root=root / "smoke-fake", report=report, notes=["fake server"])
    assert all(b.get("stream") is True for b in fake.bodies) and len(fake.bodies) == 3
    model = summary["model"]
    assert model["props"]["n_ctx"] == 8192 and model["props"]["model_meta"] == {"n_ctx_train": 8192}
    (f,), (a,) = model["tier_f"], model["episodes"]
    assert f["task"] == "fx-modify-tone" and f["status"] == "completed" and f["edit_called"] is False
    assert f["failure"] == "no_edit" and f["passed"] is False and f["lookup_used"] is False
    assert a["task"] == "fx-a-knowledge-tone" and a["completed"] and a["tool_calls"] == 1 and a["malformed_calls"] == 0
    assert a["passed"] is False and a["requests"] == 2 and a["caps_hit"] == []
    assert summary["traces"]["rows"] == 2 and summary["traces"]["rows_off_format"] == 0
    rows = [json.loads(line) for line in Path(summary["traces"]["path"]).read_text().splitlines()]
    assert [r["task_id"] for r in rows] == ["fx-a-knowledge-tone", "fx-modify-tone"]
    assert rows[0]["messages"][1]["thinking"] == "Inspect first." and rows[0]["grcbench"]["harness"]["streaming"] is True
    assert rows[1]["messages"][1]["tool_calls"][0]["name"] == "inspect_graph"        # the prefilled call
    text = report.read_text()
    assert smoke.BANNER in text and "| fx-a-knowledge-tone | yes (completed) |" in text and "- fake server" in text

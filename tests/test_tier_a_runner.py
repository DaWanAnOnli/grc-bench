"""The Tier-A episode runner and its Tier-F setting (eval plan §4.4, §4.1, §7.4; build plan Phase 4).

Fast tests check task loading, the scripted user, the caps helpers, the run-log guard, the trace
format (against Qoherent's released rows) and the pure parts of the Tier-F setting. Slow tests run
real episodes: GRC-Agent's real executor in an isolated bubblewrap episode process, driven by
scripted pydantic-ai FunctionModels (or GRC-Agent's own OpenAIChatModel over a scripted transport)
on fixture tasks only (tests/fixtures/tier_a/tasks, tests/fixtures/tier_f_task); no benchmark item
is ever run through a model.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest
import yaml

from grcbench.common.canary import CANARY, CANARY_GUID
from grcbench.common.io import read_json, sha256_file
from grcbench.common.paths import FIXTURES, ITEMS, SFT_RELEASE, WORK
from grcbench.runlog import log as runlog
from grcbench.tier_a import caps as capsmod
from grcbench.tier_a import models, runner, tier_f_mode, trace
from grcbench.tier_a.task import TaskError, load_task, task_errors
from grcbench.tier_a.user import DEFAULT_REPLY, ScriptedUser, is_asking

TASKS = FIXTURES / "tier_a" / "tasks"
MODIFY = TASKS / "fx-a-modify-tone"
BUILD = TASKS / "fx-a-build-new"
F_MODIFY = FIXTURES / "tier_f_task" / "fx_modify_tone"
F_BUILD = FIXTURES / "tier_f_task" / "fx_build_tone"

SET_2K = {"reason": "set the tone to 2 kHz", "update_params": [{"instance_name": "src", "params": {"freq": "2000"}}]}
BUILD_TONE = {"reason": "build the tone generator",
              "add_blocks": [{"block_id": "analog_sig_source_x", "instance_name": "src",
                              "params": {"type": "float", "freq": "2000", "amp": "1", "samp_rate": "samp_rate"}},
                             {"block_id": "blocks_file_sink", "instance_name": "out",
                              "params": {"type": "float", "file": "output.dat"}}],
              "add_connections": ["src:0->out:0"]}
LOOKUP = {"query": "signal source frequency", "domain": "catalog", "k": 3}


def call(name: str, args, cid: str | None = None) -> dict:
    out = {"name": name, "args": args}
    if cid:
        out["id"] = cid
    return out


@pytest.fixture(scope="module")
def root():
    path = WORK / "tier_a" / "pytest" / f"runner-{uuid.uuid4().hex[:10]}"
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def guard(tmp_path):
    """Run-log guard files private to the test (never the repository's runs/ log)."""
    return {"log_path": tmp_path / "test_run_log.jsonl", "run_list": tmp_path / "run_list.yaml"}


def copy_task(src: Path, dest_root: Path, **changes) -> Path:
    dest = dest_root / src.name
    shutil.copytree(src, dest)
    rec = read_json(dest / "task.json")
    rec.update(changes)
    (dest / "task.json").write_text(json.dumps(rec, indent=1) + "\n")
    return dest


def episode(task, turns, root, guard, *, caps=None, sample=0, on_exhausted="text", config_id="pytest"):
    return runner.run_episode(task, models.scripted(turns, on_exhausted=on_exhausted), "benchmark", root,
                              config_id=config_id, sample=sample, caps=caps, **guard)


def messages_of(res) -> list:
    return trace.load_messages(Path(res.run_dir) / runner.MESSAGES_FILE)


def tool_returns(res) -> dict:
    from pydantic_ai.messages import ModelRequest, RetryPromptPart, ToolReturnPart
    out = {}
    for msg in messages_of(res):
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                    out[part.tool_call_id] = part
    return out


# ================================================================ tasks

def test_fixture_tasks_load_and_validate():
    t = load_task(MODIFY)
    assert (t.id, t.split, t.category) == ("fx-a-modify-tone", "dev", "modify")
    assert t.start_files == [Path("tone.grc")] and t.active_graph == Path("tone.grc")
    assert t.user().max_turns == 2
    b = load_task(BUILD)
    assert b.active_graph is None and b.start_files == []
    assert runner.run_id_for(t.id, "cfg/1", "benchmark", 3) == "cfg_1__fx-a-modify-tone__benchmark-s03"


def test_task_errors_catch_unusable_tasks(tmp_path):
    good = copy_task(MODIFY, tmp_path / "a")
    rec = read_json(good / "task.json")
    assert task_errors(rec, good) == []
    cases = {
        "category": ({**rec, "category": "nonsense"}, "category"),
        "canary": ({**rec, "instruction": rec["instruction"] + " " + CANARY_GUID}, "canary"),
        "active": ({**rec, "start_state": {"active_graph": "missing.grc"}}, "not a file"),
        "escape": ({**rec, "start_state": {"active_graph": "../task.json"}}, "inside start"),
        "regex": ({**rec, "scripted_user": {"rules": [{"pattern": "(", "reply": "x"}]}}, "pattern"),
    }
    for name, (bad, needle) in cases.items():
        errs = task_errors(bad, good)
        assert errs and any(needle in e for e in errs), (name, errs)
    (good / "start" / "notes.txt").write_text(f"secret {CANARY_GUID}\n")
    assert any("canary" in e for e in task_errors(rec, good))
    with pytest.raises(TaskError):
        load_task(good)


# ================================================================ scripted user

def test_scripted_user_rules_default_reply_and_end():
    user = ScriptedUser.from_spec({"rules": [{"pattern": "which frequency", "reply": "2 kHz"}], "max_turns": 3})
    assert user.respond("Which frequency do you want?").message == "2 kHz"
    # the rule is used up (max_uses 1): the same question now gets the default reply
    nxt = user.respond("Which frequency do you want?")
    assert (nxt.message, nxt.source) == (DEFAULT_REPLY, "default")
    assert user.respond("I changed the frequency to 2 kHz and saved the graph.") is None   # not asking
    assert user.turns_sent == 2 and not user.exhausted


def test_scripted_user_max_turns_and_asking_detection():
    user = ScriptedUser.from_spec({"max_turns": 1})
    assert user.respond("Should I also add a throttle?").source == "default"
    assert user.respond("Shall I continue?") is None and user.exhausted
    assert is_asking("Let me know if you want a different rate.")
    assert not is_asking("Done.\n```python\nx = a if b else c?\n```")      # code is not prose
    assert not is_asking("The flowgraph now runs at `samp_rate?` as before.")
    silent = ScriptedUser.from_spec({"max_turns": 0})
    assert silent.respond("Which file name should I use?") is None
    with pytest.raises(ValueError):
        ScriptedUser.from_spec({"default_reply": "  "})


# ================================================================ caps helpers

def test_caps_per_setting_and_usage_limits():
    bench = capsmod.Caps.for_setting("benchmark")
    assert (bench.requests, bench.context_tokens, bench.tool_output_tokens) == (40, 65536, 4096)
    limits = bench.usage_limits()
    assert limits.request_limit == 40 and limits.per_request_input_tokens_limit == 65536
    defaults = capsmod.Caps.for_setting("grc_agent_defaults")
    assert defaults.requests == 40 and defaults.context_tokens is None and defaults.tool_output_tokens is None
    assert capsmod.Caps.from_dict(bench.to_dict()) == bench
    assert capsmod.Caps.for_setting("benchmark", requests=3).requests == 3
    with pytest.raises(ValueError):
        capsmod.Caps.for_setting("nope")
    f = tier_f_mode.tier_f_caps()
    assert f.requests == 3 and f.context_tokens == 65536 and f.tool_output_tokens == 4096


def test_tool_output_cap_truncates_without_failing():
    small = {"ok": True, "x": 1}
    same, rec = capsmod.cap_tool_result(small, 4096)
    assert same is small and rec["truncated"] is False
    big = "\n".join(f"line {i}: value {i * i}" for i in range(4000)) + "\nTraceback: boom error\n"
    cut, rec = capsmod.cap_tool_result(big, 200)
    assert rec["truncated"] and rec["tokens"] > 200 and rec["tokens_after"] <= 200
    assert cut.startswith("line 0") and "error" in cut and "4K tool-output cap" in cut
    unchanged, rec = capsmod.cap_tool_result(big, None)
    assert unchanged is big and not rec["truncated"]


def test_stop_gracefully_texts_and_exception_classes():
    assert capsmod.stop_gracefully_reason(capsmod.STOP_TEXT_MAX_REQUESTS) == "max_requests"
    text = ("`run_command` is not available in this environment and kept failing, so I'm stopping "
            "rather than retrying it further. The rest of the session is unaffected.")
    assert capsmod.stop_gracefully_reason(text) == "repeated_failure:run_command"
    assert capsmod.stop_gracefully_reason("All done.") is None
    # the canned texts are GRC-Agent's own (agent.py at the pinned commit)
    from grcbench.common.paths import GRC_AGENT
    src = " ".join((GRC_AGENT / "src" / "grc_agent" / "agent.py").read_text().split())
    for fragment in ("I've made a lot of tool calls without finishing, so I'm stopping here",
                     "is not available in this environment and kept"):
        assert fragment in src
    from pydantic_ai.exceptions import UsageLimitExceeded
    caps = capsmod.Caps()
    hit = capsmod.classify_exception(UsageLimitExceeded("The next request would exceed the request_limit of 40"), caps, 40)
    assert hit.cap == capsmod.CAP_REQUESTS
    hit = capsmod.classify_exception(UsageLimitExceeded(
        "Exceeded the per_request_input_tokens_limit of 65536 (request_input_tokens=70000)"), caps, 1)
    assert hit.cap == capsmod.CAP_CONTEXT
    assert capsmod.classify_exception(TimeoutError(), caps).cap == capsmod.CAP_EPISODE_TIME
    assert capsmod.classify_exception(RuntimeError("x"), caps) is None


def test_schema_check_is_the_well_formed_call_rule():
    schema = {"type": "object", "properties": {"items": {"type": "array"}, "reason": {"type": "string"}},
              "required": ["reason"]}
    assert capsmod.schema_check('{"reason": "r", "items": [1]}', schema) == (True, [])
    ok, errs = capsmod.schema_check('{"reason": "r", "items": "[1]"}', schema)   # JSON-stringified array
    assert not ok and "items" in errs[0]
    assert capsmod.schema_check("{not json", schema)[0] is False
    assert capsmod.schema_check("[1, 2]", schema)[0] is False
    assert capsmod.schema_check("", {"type": "object", "properties": {}})[0] is True
    assert capsmod.schema_check({"reason": "r"}, schema)[0] is True


# ================================================================ run-log guard

def test_test_split_refused_without_declared_run_list(tmp_path, root, guard):
    test_task = copy_task(MODIFY, tmp_path, split="test")
    with pytest.raises(runlog.RunRefused, match="does not exist"):
        runner.run_episode(test_task, models.scripted([]), "benchmark", root / "refused", config_id="cfg-x", **guard)
    guard["run_list"].write_text("configurations:\n  - id: cfg-declared\n")
    with pytest.raises(runlog.RunRefused, match="not on the declared run list"):
        runner.run_episode(test_task, models.scripted([]), "benchmark", root / "refused", config_id="cfg-x", **guard)
    assert not (root / "refused").exists() and not guard["log_path"].exists()   # nothing started or logged
    # until the team signs the pre-registration off, the repository's run list is a draft: test runs
    # are refused, even for the configurations it lists
    if runlog.declared_config_ids() is None:
        with pytest.raises(runlog.RunRefused):
            runlog.guard(split="test", config_id="anything")


def test_a_draft_run_list_declares_nothing(tmp_path):
    run_list = tmp_path / "run_list.yaml"
    run_list.write_text("status: draft\nconfigurations:\n  - id: cfg-a\n")
    assert runlog.declared_config_ids(run_list) is None
    with pytest.raises(runlog.RunRefused, match="not declared"):
        runlog.guard(split="test", config_id="cfg-a", run_list=run_list)
    run_list.write_text("status: declared\nconfigurations:\n  - id: cfg-a\n")
    assert runlog.declared_config_ids(run_list) == {"cfg-a"}
    runlog.guard(split="test", config_id="cfg-a", run_list=run_list)
    run_list.write_text("configurations:\n  - id: cfg-a\n")          # no status: declared, as before
    assert runlog.declared_config_ids(run_list) == {"cfg-a"}


def test_test_split_batch_is_logged_before_it_starts(tmp_path, guard, monkeypatch):
    test_task = copy_task(MODIFY, tmp_path, split="test")
    guard["run_list"].write_text("configurations:\n  - id: cfg-declared\n")
    seen = []

    def fake_run_episode(task, spec, setting, run_root, **kw):
        seen.append(runlog.read_log(guard["log_path"]))      # what the log holds when the episode starts
        return kw["run_log_entry"]
    monkeypatch.setattr(runner, "run_episode", fake_run_episode)
    out = runner.run_tasks([test_task], models.scripted([]), config_id="cfg-declared", reason="unit test",
                           samples=2, **guard)
    assert len(seen) == 2 and all(len(s) == 1 for s in seen)
    entry = seen[0][0]
    for key in runlog.REQUIRED + ("tier", "mode"):
        assert key in entry
    assert entry["split"] == "test" and entry["items"] == ["fx-a-modify-tone"] and entry["config_id"] == "cfg-declared"
    assert entry["git_commit"] == runlog.git_commit() and (entry["git_commit"] == "uncommitted" or len(entry["git_commit"]) == 40)
    assert entry["settings"]["samples"] == 2 and entry["settings"]["caps"]["requests"] == 40
    assert out[0] == entry
    # append-only: a second batch adds a line, the first is untouched
    runner.run_tasks([test_task], models.scripted([]), config_id="cfg-declared", reason="again", **guard)
    lines = guard["log_path"].read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0]) == entry


def test_dev_runs_allowed_and_not_logged(guard):
    assert runlog.start_run(split="dev", config_id="x", model={}, settings={}, items=["a"], reason="r", **guard) is None
    assert not guard["log_path"].exists()
    with pytest.raises(runlog.RunRefused):
        runlog.start_run(split="dev", config_id="x", model={}, settings={}, items=["a"], reason="r",
                         item_splits=["test"], **guard)


def test_collection_mode_refuses_benchmark_items(guard):
    for split in ("dev", "test"):
        with pytest.raises(runlog.RunRefused, match="collection mode"):
            runlog.guard(split=split, config_id="c", items=["x"], mode="collection", run_list=guard["run_list"])
    bench_id = next(iter(runlog.benchmark_item_ids(ITEMS)))
    with pytest.raises(runlog.RunRefused, match="collection mode"):
        runlog.guard(split="train", config_id="c", items=[bench_id], mode="collection", run_list=guard["run_list"])
    runlog.guard(split="train", config_id="c", items=["t99-new-training-task"], mode="collection",
                 run_list=guard["run_list"])
    with pytest.raises(runlog.RunRefused):
        runlog.guard(split="dev", config_id="c", mode="sampling")


# ================================================================ trace format

def _release_rows() -> list[dict]:
    rows = []
    for name in ("sft_train.jsonl", "quarantine.jsonl"):
        with open(SFT_RELEASE / name, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def test_release_rows_have_the_format_the_exporter_writes():
    rows = _release_rows()
    assert len(rows) == 32
    for row in rows:
        assert trace.row_format_errors(row) == [], row["session_id"]


def test_map_messages_follows_the_release_conventions():
    from pydantic_ai.messages import (
        ModelRequest, ModelResponse, RetryPromptPart, TextPart, ThinkingPart, ToolCallPart, ToolReturnPart,
        UserPromptPart,
    )
    history = [
        ModelRequest(parts=[UserPromptPart("Build it.")], instructions="SYSTEM"),
        ModelResponse(parts=[ThinkingPart("hmm "), ToolCallPart("inspect_graph", '{"targets": "[\\"a\\"]"}', tool_call_id="c1")]),
        ModelRequest(parts=[ToolReturnPart("inspect_graph", {"ok": True}, tool_call_id="c1")]),
        ModelResponse(parts=[ToolCallPart("change_graph", {"reason": "r"}, tool_call_id="c2")]),
        ModelRequest(parts=[RetryPromptPart("Graph modification failed.", tool_name="change_graph", tool_call_id="c2")]),
        ModelResponse(parts=[TextPart("Done.")]),
        ModelRequest(parts=[RetryPromptPart("Please return text.", tool_call_id="c3")]),
        ModelResponse(parts=[ThinkingPart("only thinking")]),
    ]
    out = trace.map_messages(history)
    assert out[0] == {"role": "user", "content": "Build it."}
    assert out[1] == {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "name": "inspect_graph", "arguments": '{"targets": "[\\"a\\"]"}'}], "thinking": "hmm"}
    assert out[2] == {"role": "tool", "tool_call_id": "c1", "name": "inspect_graph", "content": '{"ok": true}'}
    assert out[3]["tool_calls"][0]["arguments"] == '{"reason": "r"}'
    assert out[4]["name"] == "change_graph" and out[6]["name"] == "unknown"
    assert out[7] == {"role": "assistant", "content": "", "thinking": "only thinking"}
    assert all("thinking" not in m for m in trace.map_messages(history, include_thinking=False))
    assert len(trace.map_messages(history, include_thinking=False)) == 7
    assert trace.system_prompt(history) == "SYSTEM"
    row = trace.episode_row({"run_id": "r", "task_id": "t", "category": "build", "status": "completed",
                             "split": "dev", "config_id": "c", "setting": "benchmark"}, history, [])
    assert trace.row_format_errors(row) == []
    assert list(row)[:9] == list(trace.ROW_KEYS) and row["verdict"] == "ungraded"


# ================================================================ Tier-F setting: pure parts

def test_stated_sample_rate_rule():
    rate = tier_f_mode.stated_sample_rate
    assert rate("Build a tone generator at a sample rate of 44.1 kHz (44100 samples per second).") == "44100"
    assert rate("Everything runs at a sample rate of 50 kHz (no decimation).") == "50000"
    assert rate("Use a 48 kHz sample rate and decimate to 8 kHz.") == "48000"
    assert rate("The sampling rate is 2 MHz.") == "2000000"
    assert rate("Generate 32000 samples per second of noise.") == "32000"
    # the first stated rate wins (the input rate of a decimating graph)
    assert rate("Input at a sample rate of 40 kHz; output at a sample rate of 5 kHz.") == "40000"
    assert rate("Generate a 1 kHz tone.") is None


def test_build_start_graph_and_prefill(tmp_path):
    task = {"instruction": "Build it at a sample rate of 48 kHz.", "start_grc": None, "_dir": str(tmp_path)}
    text, rule = tier_f_mode.start_graph_text(task)
    data = yaml.safe_load(text)
    assert rule == {"kind": "empty_build_graph", "samp_rate": "48000", "samp_rate_source": "instruction"}
    assert data["blocks"] == [{"name": "samp_rate", "id": "variable", "parameters": {"value": "48000"}}]
    assert data["connections"] == [] and data["options"]["parameters"]["generate_options"] == "no_gui"
    _, rule = tier_f_mode.start_graph_text({**task, "instruction": "Build a tone."})
    assert rule["samp_rate"] == "32000" and rule["samp_rate_source"] == "GRC default"
    _, rule = tier_f_mode.start_graph_text({**task, "tier_a_setting": {"start_samp_rate": 8000}})
    assert rule["samp_rate"] == "8000" and rule["samp_rate_source"] == "task override"
    text, rule = tier_f_mode.start_graph_text(tier_f_mode.load_tier_f_task(F_MODIFY))
    assert rule["kind"] == "task_start_graph" and text == (F_MODIFY / "start.grc").read_text()
    pid = tier_f_mode.prefill_call_id("fx-modify-tone")
    assert pid == tier_f_mode.prefill_call_id("fx-modify-tone") != tier_f_mode.prefill_call_id("other")
    assert len(pid) == 32 and pid.isalnum()
    msgs = tier_f_mode.prefill_messages("Do it.", pid)
    assert msgs[1].parts[0].tool_name == "inspect_graph" and msgs[1].parts[0].args == "{}"


def test_budget_allows_first_lookup_and_first_edit_only():
    from types import SimpleNamespace

    from pydantic_ai.messages import ModelResponse, ToolCallPart
    meter = capsmod.EpisodeMeter(capsmod.Caps())
    budget = tier_f_mode.TierFBudget(prefill_id="P", meter=meter)
    history = [ModelResponse(parts=[ToolCallPart("inspect_graph", "{}", tool_call_id="P")]),
               ModelResponse(parts=[ToolCallPart("query_knowledge", "{}", tool_call_id="q1"),
                                    ToolCallPart("change_graph", "{}", tool_call_id="e1"),
                                    ToolCallPart("query_knowledge", "{}", tool_call_id="q2"),
                                    ToolCallPart("run_flowgraph", "{}", tool_call_id="r1")])]
    ctx = SimpleNamespace(messages=history)
    blocked = {cid: budget.blocked(ctx, SimpleNamespace(tool_call_id=cid, tool_name=name))
               for cid, name in (("P", "inspect_graph"), ("q2", "query_knowledge"), ("q1", "query_knowledge"),
                                 ("e1", "change_graph"), ("r1", "run_flowgraph"))}
    assert blocked == {"P": False, "q1": False, "e1": False, "q2": True, "r1": True}
    assert not budget.edit_processed()
    meter.calls["e1"] = {"validation": "ok"}
    assert not budget.edit_processed()
    meter.calls["e1"]["execution"] = "retry"
    assert budget.edit_processed()


def test_rejection_rungs():
    assert tier_f_mode.rejection_rung(["block_type_not_found"]) == "V2"
    assert tier_f_mode.rejection_rung(["gnu_validation"]) == "V3"
    assert tier_f_mode.rejection_rung(["gnu_validation", "connection_not_found"]) == "V2"
    assert tier_f_mode.rejection_rung(["args_invalid"]) == "V1"
    assert tier_f_mode.rejection_rung(["something_new"]) == "V3"
    assert tier_f_mode.rejection_rung(["save_failed"]) is None


# ================================================================ Tier-A episodes (slow)

@pytest.fixture(scope="module")
def modify_episode(root):
    before = sha256_file(MODIFY / "start" / "tone.grc")
    turns = [
        {"tool_calls": [call("inspect_graph", {})]},
        # JSON-stringified update_params: GRC-Agent's JSON repair accepts it; the raw call is logged first
        {"tool_calls": [call("change_graph", json.dumps({**SET_2K, "update_params": json.dumps(SET_2K["update_params"])}))]},
        {"text": "Done: the tone is now 2 kHz."},
    ]
    guard = {"log_path": root / "never.jsonl", "run_list": root / "no_run_list.yaml"}
    res = episode(MODIFY, turns, root, guard)
    return res, before


@pytest.mark.slow
@pytest.mark.gnuradio
def test_episode_runs_grc_agent_and_records_everything(modify_episode):
    res, before = modify_episode
    run = Path(res.run_dir)
    assert res.status == "completed" and res.end_reason == "agent_finished", res.error
    assert res.valid and not res.cap_failure and res.caps_hit == []
    assert run.is_relative_to(WORK) and (run / "project" / "tone.grc").is_file()
    # the start state was copied, never edited in place; the copy now holds the edit
    assert sha256_file(MODIFY / "start" / "tone.grc") == before
    assert res.final_graph_path == str(run / "project" / "tone.grc")
    final = yaml.safe_load(Path(res.final_graph_path).read_text())
    src = next(b for b in final["blocks"] if b["name"] == "src")
    assert src["parameters"]["freq"] == "2000" and res.final_graph_sha256 == sha256_file(Path(res.final_graph_path))
    assert res.final_text == "Done: the tone is now 2 kHz."
    # usage, requests, approvals
    assert res.usage["requests"] == 3 and len(res.model_requests) == 3
    assert [a["tool"] for a in res.turns[0]["approvals"]] == ["change_graph"]
    # every tool call: raw arguments before JSON repair, validation, execution, result size
    inspect, change = res.tool_calls
    assert inspect["tool"] == "inspect_graph" and inspect["execution"] == "ok" and inspect["truncated"] is False
    assert change["tool"] == "change_graph" and change["raw_args_kind"] == "str"
    assert json.loads(change["raw_args"])["update_params"] == json.dumps(SET_2K["update_params"])
    assert change["repair_changed_args"] is True and change["schema_valid"] is False and change["execution"] == "ok"
    assert any(e.get("approved") for e in change["events"] if e["stage"] == "execute")
    assert res.usage["model_tool_calls"] == 2 and res.usage["model_tool_calls_schema_valid"] == 1
    # recorded files: messages, tools, trace, result
    assert len(read_json(run / runner.TOOLS_FILE)) == 22
    assert len(messages_of(res)) == 6
    assert res.agent_info["grc_agent_commit"] == "6c98128" and res.agent_info["setting"] == "benchmark"
    assert "tone.grc" in res.project_files
    assert read_json(run / runner.RESULT_FILE)["run_id"] == res.run_id


@pytest.mark.slow
@pytest.mark.gnuradio
def test_trace_row_is_in_the_sft_release_format(modify_episode):
    res, _ = modify_episode
    lines = (Path(res.run_dir) / runner.TRACE_FILE).read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert trace.row_format_errors(row) == []
    assert list(row) == list(trace.ROW_KEYS) + ["grcbench"]
    assert row["task_id"] == "fx-a-modify-tone" and row["category"] == "modify" and row["verdict"] == "ungraded"
    assert len(row["tools"]) == 22 and row["system"].endswith(f"Active flowgraph file path: {res.final_graph_path}")
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    # the raw call as the model emitted it, the tool result as the model saw it
    assert row["messages"][3]["tool_calls"][0]["arguments"] == res.tool_calls[1]["raw_args"]
    assert row["usage"]["requests"] == 3 and row["grcbench"]["caps_hit"] == []
    assert CANARY_GUID not in lines[0]
    release = _release_rows()[0]
    assert [t["name"] for t in row["tools"]] == sorted(t["name"] for t in release["tools"])


@pytest.mark.slow
@pytest.mark.gnuradio
def test_scripted_user_turns_and_one_request_counter_for_the_episode(root, guard):
    """Requests are counted across scripted-user turns and approval resumes (one RunUsage)."""
    turns = [
        {"text": "Which frequency should I use?"},                 # -> rule reply
        {"tool_calls": [call("change_graph", SET_2K)]},             # approval resume, same turn
        {"text": "Done. Should I also save it?"},                    # -> default reply
        {"tool_calls": [call("inspect_graph", {})]},
        {"text": "never sent"},                                      # request 5 > cap of 4
    ]
    res = episode(MODIFY, turns, root, guard, caps=capsmod.Caps.for_setting("benchmark", requests=4), sample=1)
    assert res.status == "cap_failure" and res.end_reason == "cap:requests"
    assert res.caps_hit[0]["cap"] == "requests" and res.usage["requests"] == 4
    assert [t["source"] for t in res.turns] == ["instruction", "rule:0", "default"]
    assert res.turns[1]["user_message"] == "Use 2 kHz, as I said." and res.turns[2]["user_message"] == DEFAULT_REPLY
    assert [t["requests_before"] for t in res.turns] == [0, 1, 3]
    assert res.turns[1]["approvals"][0]["tool"] == "change_graph"


@pytest.mark.slow
@pytest.mark.gnuradio
def test_context_cap_is_a_failure(root, guard):
    turns = [{"tool_calls": [call("inspect_graph", {})], "usage": {"input_tokens": 70000, "output_tokens": 5}},
             {"text": "never reached"}]
    res = episode(MODIFY, turns, root, guard, sample=2)
    assert res.status == "cap_failure" and res.caps_hit[0]["cap"] == "context"
    assert res.caps_hit[0]["limit"] == 65536 and res.usage["max_request_input_tokens"] == 70000


# llama-server b10333's reply to a prompt longer than the slot's context (e2e dev run, 28 Oct 2026)
LLAMA_CONTEXT_BODY = (b'{"error":{"code":400,"message":"request (41041 tokens) exceeds the available context size '
                      b'(40960 tokens), try increasing it","type":"exceed_context_size_error"}}')


def _refusal_chain(body: bytes, read: bool) -> BaseException:
    """ModelAPIError("Connection error.") <- openai.APIConnectionError <- httpx2.HTTPStatusError(400), the
    chain GRC-Agent's retrying client produces for a refused streamed request."""
    import httpx2
    import openai
    from pydantic_ai.exceptions import ModelAPIError
    req = httpx2.Request("POST", "http://127.0.0.1:9/v1/chat/completions")
    resp = httpx2.Response(400, request=req, stream=httpx2.ByteStream(body))
    if read:
        resp.read()
    try:
        try:
            raise httpx2.HTTPStatusError("Client error '400 Bad Request'", request=req, response=resp)
        except httpx2.HTTPStatusError as status:
            raise openai.APIConnectionError(request=req) from status
    except openai.APIConnectionError as conn:
        api = ModelAPIError(model_name="m", message="Connection error.")
        api.__cause__ = conn
        return api


def test_a_context_refusal_in_the_exception_chain_is_the_context_cap():
    caps = capsmod.Caps.for_setting("benchmark")
    hit = capsmod.classify_exception(_refusal_chain(LLAMA_CONTEXT_BODY, read=True), caps, 7)
    assert hit is not None and hit.cap == capsmod.CAP_CONTEXT and "exceed_context_size_error" in hit.detail
    assert hit.limit == 65536 and hit.at_request == 7
    assert capsmod.classify_exception(_refusal_chain(LLAMA_CONTEXT_BODY, read=False), caps) is None  # unread
    assert capsmod.classify_exception(_refusal_chain(b'{"error": "invalid tool schema"}', read=True), caps) is None
    a, b = RuntimeError("a"), RuntimeError("b")
    a.__context__, b.__context__ = b, a
    assert capsmod.classify_exception(a, caps) is None                                 # a cycle ends


def test_error_bodies_are_read_and_success_streams_are_not():
    import asyncio

    import httpx2

    from grcbench.tier_a import agent_build

    def handler(request):
        if request.url.path.endswith("/bad"):
            return httpx2.Response(400, stream=httpx2.ByteStream(LLAMA_CONTEXT_BODY))
        return httpx2.Response(200, stream=httpx2.ByteStream(b"data: {}\n\n"))

    transport = agent_build.error_body_transport(httpx2.MockTransport(handler))

    async def go():
        return (await transport.handle_async_request(httpx2.Request("POST", "http://x/bad")),
                await transport.handle_async_request(httpx2.Request("POST", "http://x/ok")))

    bad, ok = asyncio.run(go())
    assert b"exceed_context_size_error" in bad.content
    with pytest.raises(httpx2.ResponseNotRead):
        ok.content                                                                     # noqa: B018


def test_the_benchmark_client_keeps_grc_agents_retries_and_reads_error_bodies():
    import asyncio

    import httpx2
    from grc_agent import agent_factory as af

    from grcbench.tier_a import agent_build
    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(400, stream=httpx2.ByteStream(LLAMA_CONTEXT_BODY))

    client = agent_build.retrying_client(httpx2.MockTransport(handler))
    assert client.timeout == httpx2.Timeout(**af._HTTP_TIMEOUT)

    async def go():
        try:
            await client.post("http://x/v1/chat/completions", json={})
        except httpx2.HTTPStatusError as exc:
            return exc
        return None

    exc = asyncio.run(go())
    assert exc is not None and len(calls) == 3                       # GRC-Agent's stop_after_attempt(3)
    hit = capsmod.classify_exception(exc, capsmod.Caps.for_setting("benchmark"))
    assert hit is not None and hit.cap == capsmod.CAP_CONTEXT


@pytest.mark.parametrize("n_ctx, n_ctx_train, status", [
    (None, None, "cap_failure"),                # an API, or a server that reports no context
    (40960, 40960, "cap_failure"),              # the model's own context is full
    (16384, 40960, "infrastructure_error"),     # the server was started below the model's context
])
def test_a_refusal_is_the_models_failure_only_if_the_server_is_not_set_below_it(n_ctx, n_ctx_train, status):
    server = {"n_ctx": n_ctx, "n_ctx_train": n_ctx_train}
    hit = {"cap": capsmod.CAP_CONTEXT, "detail": "engine refused the prompt: HTTP 400 ...", "limit": 65536}
    assert runner.refusal_status("cap_failure", [hit], server, capsmod.Caps.for_setting("benchmark")) == status
    other = {"cap": capsmod.CAP_CONTEXT, "detail": "per_request_input_tokens_limit", "limit": 65536}
    assert runner.refusal_status("cap_failure", [other], server, capsmod.Caps.for_setting("benchmark")) == "cap_failure"


def _refusing_server(n_ctx):
    """A stand-in llama-server that refuses every prompt as llama.cpp does for an over-long one; it reports
    a context of n_ctx per slot (None: it reports none) and a model context of 40,960."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Refuse(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_GET(self) -> None:                                    # noqa: N802 - llama-server /props, /v1/models
            if n_ctx is None:
                self.send_error(404)
                return
            body = ({"default_generation_settings": {"n_ctx": n_ctx}} if self.path == "/props"
                    else {"data": [{"id": "stub", "meta": {"n_ctx_train": 40960}}]})
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:                                   # noqa: N802 - http.server API
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(LLAMA_CONTEXT_BODY)))
            self.end_headers()
            self.wfile.write(LLAMA_CONTEXT_BODY)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Refuse)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_a_server_set_below_the_models_context_is_refused_before_the_episode(root, guard):
    server = _refusing_server(16384)
    try:
        spec = models.openai_compatible(f"http://127.0.0.1:{server.server_address[1]}/v1", "stub")
        with pytest.raises(runlog.RunRefused, match="context"):
            runner.run_episode(MODIFY, spec, "benchmark", root, config_id="pytest", sample=6, **guard)
    finally:
        server.shutdown()


@pytest.mark.slow
@pytest.mark.gnuradio
@pytest.mark.parametrize("n_ctx", [None, 40960])
def test_a_context_refusal_by_the_model_server_is_a_context_cap_failure(root, guard, n_ctx):
    server = _refusing_server(n_ctx)
    try:
        spec = models.openai_compatible(f"http://127.0.0.1:{server.server_address[1]}/v1", "stub")
        res = runner.run_episode(MODIFY, spec, "benchmark", root, config_id="pytest",
                                 sample=4 if n_ctx is None else 5, **guard)
    finally:
        server.shutdown()
    assert res.status == "cap_failure" and res.caps_hit[0]["cap"] == "context" and res.valid
    assert res.harness["model_server"] == {"n_ctx": n_ctx, "n_ctx_train": None if n_ctx is None else 40960}


@pytest.mark.slow
@pytest.mark.gnuradio
def test_grc_agent_stop_gracefully_counts_as_a_cap_failure(root, guard):
    """A looping agent: GRC-Agent's own StopGracefully ends the run at its 41st request."""
    res = episode(BUILD, [{"tool_calls": [call("inspect_graph", {})]}], root, guard,
                  caps=capsmod.Caps.for_setting("benchmark", requests=100), on_exhausted="repeat_last")
    assert res.status == "cap_failure" and res.end_reason == "cap:grc_agent_stop_gracefully"
    assert res.caps_hit[0]["detail"] == "max_requests" and res.usage["requests"] == 40
    assert res.final_output == capsmod.STOP_TEXT_MAX_REQUESTS


@pytest.mark.slow
@pytest.mark.gnuradio
def test_build_from_untitled_graph_with_truncated_tool_output(root, guard):
    turns = [{"tool_calls": [call("inspect_graph", {})]},
             {"tool_calls": [call("change_graph", BUILD_TONE)]},
             {"tool_calls": [call("save_graph", {})]},
             {"text": "Built and saved."}]
    res = episode(BUILD, turns, root, guard, caps=capsmod.Caps.for_setting("benchmark", tool_output_tokens=60))
    assert res.status == "completed", res.error
    # the 4K cap (here 60 tokens) shortens what the model sees and is not a failure
    inspect = res.tool_calls[0]
    assert inspect["truncated"] is True and res.usage["tool_results_truncated"] >= 1
    seen = tool_returns(res)[inspect["tool_call_id"]].content
    assert "truncated by the GRC-Bench 4K tool-output cap" in seen
    # GRC's File > New graph was saved into the project folder by save_graph
    final = Path(res.final_graph_path)
    assert final.parent == Path(res.run_dir) / "project" and final.suffix == ".grc" and res.final_graph_saved
    ids = {b["id"] for b in yaml.safe_load(final.read_text())["blocks"]}
    assert {"analog_sig_source_x", "blocks_file_sink", "variable"} <= ids


@pytest.mark.slow
@pytest.mark.gnuradio
def test_flowgraph_run_time_cap(root, guard):
    turns = [{"tool_calls": [call("run_flowgraph", {"action": "start", "wait": True, "timeout_seconds": 60})]},
             {"text": "never reached"}]
    res = episode(MODIFY, turns, root, guard, caps=capsmod.Caps.for_setting("benchmark", flowgraph_run_seconds=4),
                  sample=3)
    assert res.status == "cap_failure" and res.caps_hit[0]["cap"] == "flowgraph_run_time"
    assert res.flowgraph_runs and res.flowgraph_runs[0]["hard_cap_hit"] is True
    assert res.usage["requests"] == 1


# ================================================================ Tier-F setting (slow)

def tier_f(task_dir, turns, root, guard, *, sample=0, wire=False, grade_sample=True):
    spec = models.wire_scripted(turns) if wire else models.scripted(turns)
    return tier_f_mode.run_tier_f_sample(task_dir, spec, root, config_id="pytest-f", sample=sample,
                                         grade_sample=grade_sample, **guard)


@pytest.mark.slow
@pytest.mark.gnuradio
def test_tier_f_lookup_then_edit_passes_the_ladder(root, guard):
    turns = [{"tool_calls": [call("query_knowledge", LOOKUP)]},
             {"tool_calls": [call("run_flowgraph", {}), call("change_graph", SET_2K)]},
             {"text": "never requested: the episode ends once the edit has been applied"}]
    res = tier_f(F_MODIFY, turns, root, guard)
    rec = res.tier_f
    assert res.tier == "F" and res.status == "completed" and res.end_reason == "tier_f_edit_applied", res.error
    assert rec["model_requests"] == 2 and rec["request_cap"] == 3
    assert rec["prefill"]["tool_call_id"] == tier_f_mode.prefill_call_id("fx-modify-tone")
    assert rec["lookup"]["used"] and rec["lookup"]["query"] == LOOKUP["query"] and rec["lookup"]["k"] == 3
    assert len(rec["lookup"]["retrieved_block_ids"]) == 3 and 0.0 <= rec["lookup"]["reference_recall"] <= 1.0
    edit = rec["edit"]
    assert edit["called"] and edit["accepted"] and edit["well_formed"] and not edit["rejected"]
    assert edit["result"]["ok"] is True and not rec["edit_same_reply_as_lookup"]
    assert [b["tool"] for b in rec["blocked_calls"]] == ["run_flowgraph"]
    returns = tool_returns(res)
    assert returns[rec["blocked_calls"][0]["tool_call_id"]].content == tier_f_mode.BLOCKED_TEXT
    grade = rec["grade"]
    assert grade["passed"] and grade["highest_passed"] == "V5" and grade["first_failed"] is None
    assert all(grade["rungs"][r]["passed"] for r in tier_f_mode.RUNGS)
    assert yaml.safe_load(Path(rec["final_graph"]).read_text())["blocks"]
    row = json.loads((Path(res.run_dir) / runner.TRACE_FILE).read_text())
    assert row["verdict"] == "completed" and trace.row_format_errors(row) == []
    assert row["messages"][1]["tool_calls"][0]["name"] == "inspect_graph"
    # the start graph was copied and converted, never edited in place
    assert (F_MODIFY / "start.grc").read_text() == tier_f_mode.start_graph_text(tier_f_mode.load_tier_f_task(F_MODIFY))[0]


@pytest.mark.slow
@pytest.mark.gnuradio
def test_tier_f_rejected_edit_rolls_back_and_fails_at_its_rung(root, guard):
    bad = {"reason": "add a source", "add_blocks": [{"block_id": "analog_tone_maker_x", "instance_name": "src2"}]}
    turns = [{"tool_calls": [call("query_knowledge", LOOKUP), call("change_graph", bad)]}, {"text": "never"}]
    res = tier_f(F_MODIFY, turns, root, guard, sample=1)
    rec = res.tier_f
    assert res.end_reason == "tier_f_edit_rejected" and rec["model_requests"] == 1
    assert rec["edit"]["rejected"] and not rec["edit"]["accepted"] and rec["edit"]["well_formed"]
    assert "block_type_not_found" in rec["edit"]["error_codes"]
    assert rec["edit_same_reply_as_lookup"] is True and rec["lookup"]["used"]
    grade = rec["grade"]
    assert not grade["passed"] and grade["rejection_rung"] == "V2" and grade["first_failed"] == "V2"
    assert grade["rungs"]["V1"]["passed"] is True
    # rolled back: the final graph is the start graph (no src2, tone still 1 kHz)
    final = yaml.safe_load(Path(rec["final_graph"]).read_text())
    assert "src2" not in {b["name"] for b in final["blocks"]}
    assert next(b for b in final["blocks"] if b["name"] == "src")["parameters"]["freq"] == "1000"


@pytest.mark.slow
@pytest.mark.gnuradio
def test_tier_f_no_edit_fails(root, guard):
    res = tier_f(F_MODIFY, [{"text": "The tone is now 2 kHz."}], root, guard, sample=2)
    assert res.end_reason == "agent_finished" and res.tier_f["edit"] == {"called": False}
    assert res.tier_f["grade"]["passed"] is False and res.tier_f["grade"]["failure"] == "no_edit"
    assert res.tier_f["grade"]["first_failed"] == "V1"


@pytest.mark.slow
@pytest.mark.gnuradio
def test_tier_f_budget_and_three_request_cap(root, guard):
    turns = [{"tool_calls": [call("query_knowledge", LOOKUP)]},
             {"tool_calls": [call("query_knowledge", {"query": "file sink", "domain": "catalog"}),
                             call("inspect_graph", {})]},
             {"tool_calls": [call("generate_python", {})]},
             {"tool_calls": [call("change_graph", SET_2K)]}]           # would be request 4: never sent
    res = tier_f(F_MODIFY, turns, root, guard, sample=3)
    rec = res.tier_f
    assert res.status == "cap_failure" and res.caps_hit[0]["cap"] == "requests" and rec["model_requests"] == 3
    assert sorted(b["tool"] for b in rec["blocked_calls"]) == ["generate_python", "inspect_graph", "query_knowledge"]
    returns = tool_returns(res)
    assert all(returns[b["tool_call_id"]].content == tier_f_mode.BLOCKED_TEXT for b in rec["blocked_calls"])
    assert rec["lookup"]["used"] and rec["edit"] == {"called": False}
    assert rec["grade"]["passed"] is False and rec["grade"]["failure"] == "cap:requests"


@pytest.mark.slow
@pytest.mark.gnuradio
def test_tier_f_repaired_but_malformed_edit_fails_v1(root, guard):
    raw = json.dumps({**SET_2K, "update_params": json.dumps(SET_2K["update_params"])})
    res = tier_f(F_MODIFY, [{"tool_calls": [call("change_graph", raw)]}], root, guard, sample=4)
    edit = res.tier_f["edit"]
    assert edit["accepted"] and edit["raw_args"] == raw and edit["repair_changed_args"] is True
    assert edit["well_formed"] is False and edit["schema_errors"]
    grade = res.tier_f["grade"]
    assert grade["passed"] is False and grade["first_failed"] == "V1" and grade["ladder"]["highest_passed"] == "V5"


@pytest.mark.slow
@pytest.mark.gnuradio
def test_tier_f_build_task_starts_from_samp_rate_only_graph(root, guard):
    res = tier_f(F_BUILD, [{"tool_calls": [call("change_graph", BUILD_TONE)]}], root, guard)
    rec = res.tier_f
    assert rec["start"] == {"kind": "empty_build_graph", "samp_rate": "48000", "samp_rate_source": "instruction"}
    start = yaml.safe_load((Path(res.run_dir) / "tier_f" / "start_full.grc").read_text())
    assert [(b["name"], b["id"], b["parameters"]["value"]) for b in start["blocks"]] == [("samp_rate", "variable", "48000")]
    seen = tool_returns(res)[rec["prefill"]["tool_call_id"]].content
    assert '"samp_rate"' in seen and "48000" in seen
    assert rec["edit"]["accepted"] and rec["grade"]["passed"], rec["grade"]


@pytest.mark.slow
@pytest.mark.gnuradio
def test_tier_f_prefill_is_byte_identical_to_a_real_inspect_graph_call(root, guard, tmp_path):
    """The request the model gets after the prefilled inspect_graph equals, byte for byte on the
    wire (the JSON body an OpenAI-compatible server receives), the request a Tier-A episode sends
    after the model itself called inspect_graph with the same id. Only each run's own directory
    path (in the active-flowgraph note and the inspect_graph result) differs."""
    lookup = {"id": "call0lookup000000000000000000000", "name": "query_knowledge",
              "arguments": json.dumps(LOOKUP)}
    fres = tier_f(F_MODIFY, [{"tool_calls": [lookup]}, {"text": "ok"}], root, guard, sample=5, wire=True,
                  grade_sample=False)
    frun = Path(fres.run_dir)
    record = read_json(F_MODIFY / "task.json")
    adir = tmp_path / "fx-a-same-start"
    (adir / "start").mkdir(parents=True)
    shutil.copyfile(frun / "tier_f" / "start_full.grc", adir / "start" / tier_f_mode.START_GRAPH_NAME)
    (adir / "task.json").write_text(json.dumps({
        "id": "fx-a-same-start", "tier": "A", "split": "dev", "version": 1, "canary": CANARY,
        "license": "CC-BY-4.0 (test fixture)", "source": {"kind": "test_fixture"}, "cluster_id": "fixture",
        "provenance": {"drafted_by": "Claude (Anthropic)", "generator": "tests"},
        "review": {"status": "not-required"}, "category": "modify", "instruction": record["instruction"],
        "start_state": {"active_graph": tier_f_mode.START_GRAPH_NAME}, "scripted_user": {"max_turns": 0},
        "acceptance": {}}))
    inspect = {"id": tier_f_mode.prefill_call_id(record["id"]), "name": "inspect_graph", "arguments": "{}"}
    ares = runner.run_episode(adir, models.wire_scripted([{"tool_calls": [inspect]}, {"tool_calls": [lookup]},
                                                          {"text": "ok"}]),
                              "benchmark", root, config_id="pytest-conformance", **guard)
    arun = Path(ares.run_dir)
    assert ares.status == "completed" and fres.status == "completed"
    fw = [json.loads(line) for line in (frun / models.WIRE_FILE).read_text().splitlines()]
    aw = [json.loads(line) for line in (arun / models.WIRE_FILE).read_text().splitlines()]
    assert len(fw) == 2 and len(aw) == 3

    def norm(body: str, run: Path) -> str:
        return body.replace(str(run), "<RUN>")
    # request 1 of Tier F == request 2 of Tier A; and the next request (after the lookup) too
    for f_i, a_i in ((0, 1), (1, 2)):
        assert fw[f_i]["path"] == aw[a_i]["path"]
        assert norm(fw[f_i]["body"], frun) == norm(aw[a_i]["body"], arun)
    body = json.loads(fw[0]["body"])
    assert len([t for t in body["tools"] if t["function"]["name"] != "final_result"]) == 22
    # GRC-Agent's system prompt, then its active-flowgraph note, then the prefilled turn
    assert [m["role"] for m in body["messages"]] == ["system", "system", "user", "assistant", "tool"]
    assert body["messages"][0]["content"].startswith("Session ID: pai-desktop-chat")
    assert body["messages"][1]["content"] == f"Active flowgraph file path: {frun}/project/{tier_f_mode.START_GRAPH_NAME}"
    assert body["messages"][3]["tool_calls"][0]["function"] == {"name": "inspect_graph", "arguments": "{}"}


@pytest.mark.slow
@pytest.mark.gnuradio
def test_agent_shell_cannot_write_outside_the_project_or_reach_the_network(root, guard, tmp_path):
    """GRC-Agent's own run_command, called by the model in a real episode, runs inside the nested
    no-network sandbox: it can write in the episode's project folder, but neither outside it nor to
    the network (eval plan §7.4 isolation)."""
    outside = tmp_path / "escape_probe.txt"
    net = ("python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), 2)\" 2>/dev/null "
           "&& echo NET_OPEN || echo NET_BLOCKED")
    turns = [
        {"tool_calls": [call("run_command", {"command": f"echo in > inside_probe.txt && echo WROTE_INSIDE; {net}"},
                             "call_shell_inside")]},
        {"tool_calls": [call("run_command", {"command": f"echo out > {outside} 2>/dev/null && echo OUT_WROTE "
                                                        "|| echo OUT_BLOCKED"}, "call_shell_outside")]},
        {"text": "Done."},
    ]
    res = episode(MODIFY, turns, root, guard, sample=7)
    assert res.status == "completed", res.end_reason
    returns = tool_returns(res)
    inside = str(returns["call_shell_inside"].content)
    assert "WROTE_INSIDE" in inside and "NET_BLOCKED" in inside and "NET_OPEN" not in inside
    from grcbench.tier_a import isolation
    assert (isolation.RunLayout(Path(res.run_dir)).project / "inside_probe.txt").read_text().strip() == "in"
    assert "OUT_WROTE" not in str(returns["call_shell_outside"].content)    # refused by GrcShell or the sandbox
    assert not outside.exists()


@pytest.mark.slow
@pytest.mark.gnuradio
def test_episode_wall_clock_cap(root, guard):
    turns = [{"tool_calls": [call("run_flowgraph", {"action": "start", "wait": True, "timeout_seconds": 60})]},
             {"text": "never reached"}]
    res = episode(MODIFY, turns, root, guard, caps=capsmod.Caps.for_setting("benchmark", episode_seconds=5), sample=4)
    assert res.status == "cap_failure" and res.caps_hit[0]["cap"] == "episode_time", res.error
    assert res.timing["agent_seconds"] < 30 and res.process["timed_out"] is False
    assert res.harness["flowgraph_left_running"] in (True, False)      # recorded; the proxy stops it on close


# ================================================================ CLI

def test_cli_refuses_unguarded_test_runs_and_prints_traces(tmp_path, capsys, monkeypatch):
    from grcbench.tier_a import cli
    test_task = copy_task(MODIFY, tmp_path, split="test")
    turns = tmp_path / "turns.json"
    turns.write_text(json.dumps([{"text": "done"}]))
    monkeypatch.setattr(runlog, "RUN_LIST", tmp_path / "absent.yaml")
    import grcbench.tier_a.runner as r
    monkeypatch.setattr(r.run_tasks, "__kwdefaults__", {**r.run_tasks.__kwdefaults__,
                                                        "run_list": tmp_path / "absent.yaml",
                                                        "log_path": tmp_path / "log.jsonl"})
    assert cli.main(["run", str(test_task), "--config-id", "c", "--reason", "r", "--scripted", str(turns)]) == 2
    assert "refused" in capsys.readouterr().out and not (tmp_path / "log.jsonl").exists()
    assert cli.main(["help"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in ("run", "tier-f", "trace", "conformance"))
    assert cli.main(["trace"]) == 2

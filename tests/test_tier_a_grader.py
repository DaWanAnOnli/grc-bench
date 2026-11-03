"""The Tier-A grader, the task gates and the Tier-F golden calls (eval plan §4.4 "Grading", §5.4;
build plan Phase 4 "Gates", Decision 16 Oct).

Fast tests check the pure parts: answer-fact matching, parameter constraints, acceptance-spec
checks, graph equivalence up to renaming, the golden-call derivation and the golden file formats.
Slow tests grade synthetic end states with GNU Radio (sandboxed), run every gate on every fixture
task through the REAL runner (GRC-Agent's executor in isolated episode processes, scripted
pydantic-ai FunctionModels) and verify the Tier-F golden calls through GRC-Agent's change_graph.
Fixture tasks only (tests/fixtures); no benchmark item is run through a model.
"""

from __future__ import annotations

import copy
import json
import shutil
import uuid
from pathlib import Path

import pytest
import yaml

from grcbench.common.io import read_json
from grcbench.common.paths import FIXTURES, WORK
from grcbench.tier_a import grader, models, replay, runner
from grcbench.tier_a.task import load_task
from grcbench.tier_f import golden

TASKS = FIXTURES / "tier_a" / "tasks"
GATED = ("fx-a-build-file", "fx-a-knowledge-tone", "fx-a-modify-tone", "fx-a-repair-type", "fx-a-save-block")
CATEGORIES = {"build", "repair", "knowledge", "save_block", "modify"}
F_TASKS = FIXTURES / "tier_f_task"
F_GATED = ("fx_build_tone", "fx_modify_tone", "fx_rewire_tone")


@pytest.fixture(scope="module")
def root():
    path = WORK / "tier_a" / "pytest" / f"grader-{uuid.uuid4().hex[:10]}"
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def block(key, params, evaluated=None, state="enabled"):
    return {"key": key, "state": state, "dummy": False, "params": dict(params),
            "evaluated": dict(evaluated if evaluated is not None else params)}


def view(blocks, connections):
    return {"ok": True, "error": None, "valid": True, "errors": [], "blocks": blocks,
            "connections": sorted(list(c) for c in connections)}


TONE_VIEW = view({"opts": block("options", {"id": "opts", "generate_options": "no_gui"}),
                  "samp_rate": block("variable", {"id": "samp_rate", "value": "32000"}, {"value": 32000}),
                  "src": block("analog_sig_source_x", {"id": "src", "freq": "2e3", "type": "float"},
                               {"freq": 2000.0, "type": "float"}),
                  "out": block("blocks_file_sink", {"id": "out", "file": "'tone.dat'"}, {"file": "tone.dat"})},
                 [("src", "0", "out", "0")])


# ================================================================ answer facts and constraints

def test_numbers_in_answers_with_si_prefixes():
    nums = grader.numbers_in("It runs at 32 kHz (32,000 samples/s), the tone is 2e3 Hz, gain -1.5 and 10 ms.")
    for want in (32, 32000, 2000, -1.5, 10, 0.01):
        assert any(abs(n - want) < 1e-9 for n in nums), (want, nums)
    assert grader.numbers_in("no numbers here") == []
    assert 1e6 in grader.numbers_in("a 1 MHz carrier")


def test_match_fact_rules():
    assert grader.match_fact(32000, "The sample rate is 32 kHz.", "number")[0]
    assert grader.match_fact("32000", "The sample rate is 32000.", "number", rel_tol=1e-6)[0]
    assert not grader.match_fact(32000, "The sample rate is 48 kHz.", "number")[0]
    assert grader.match_fact(1000.0, "about 1001 Hz", "number", tol=2)[0]
    assert grader.match_fact("float", "It is a Float block.", "text")[0]
    assert not grader.match_fact("int", "It is an integer block.", "text")[0]       # whole words only
    ok, detail = grader.match_fact(["complex", "float", "int", "short"], "complex, float or int", "all_of")
    assert not ok and "short" in detail
    assert grader.match_fact(["complex", "float"], "only float", "any_of")[0]
    assert not grader.match_fact("x", "", "text")[0]


def test_parameter_constraints_numeric_tolerance_and_text():
    src = TONE_VIEW["blocks"]["src"]
    out = TONE_VIEW["blocks"]["out"]
    assert grader.param_ok(src, "freq", 2000)[0]                      # '2e3' evaluates to 2000
    assert grader.param_ok(src, "freq", "2000")[0]
    assert not grader.param_ok(src, "freq", 2001)[0]
    assert grader.param_ok(src, "freq", {"value": 2001, "tol": 1.5})[0]
    assert grader.param_ok(src, "freq", {"value": 2100, "rel_tol": 0.06})[0]
    assert grader.param_ok(src, "type", "float")[0] and not grader.param_ok(src, "type", "complex")[0]
    assert grader.param_ok(out, "file", "tone.dat")[0]                # quotes ignored
    assert grader.param_ok(out, "file", {"regex": r"tone\.dat"})[0]
    assert grader.param_ok(src, "type", {"in": ["float", "complex"]})[0]
    assert not grader.param_ok(src, "nope", 1)[0]


def test_block_and_connection_checks_are_name_independent():
    renamed = copy.deepcopy(TONE_VIEW)
    renamed["blocks"] = {"tone_gen" if n == "src" else n: b for n, b in renamed["blocks"].items()}
    renamed["connections"] = [["tone_gen", "0", "out", "0"]]
    for v in (TONE_VIEW, renamed):
        assert grader._blocks_present(v, {"id": "analog_sig_source_x", "params": {"freq": 2000}}, 0)["passed"]
        assert grader._connections(v, {"src": {"id": "analog_sig_source_x", "port": "0"},
                                       "dst": {"ids": ["blocks_file_sink"], "params": {"file": "tone.dat"}}}, 0)["passed"]
        assert grader._blocks_absent(v, {"ids": ["blocks_throttle2"]}, 0)["passed"]
    disabled = copy.deepcopy(TONE_VIEW)
    disabled["blocks"]["src"]["state"] = "disabled"
    assert not grader._blocks_present(disabled, {"id": "analog_sig_source_x"}, 0)["passed"]
    assert grader._blocks_present(disabled, {"id": "analog_sig_source_x", "state": "any"}, 0)["passed"]
    assert grader._blocks_absent(disabled, {"id": "analog_sig_source_x"}, 0)["passed"]
    assert not grader._connections(disabled, {"src": {"id": "analog_sig_source_x"}, "dst": {"id": "blocks_file_sink"}},
                                   0)["passed"]
    assert not grader._blocks_present(TONE_VIEW, {"id": "analog_sig_source_x", "min": 2}, 0)["passed"]


def test_block_yaml_paths():
    data = grader.block_yaml("blocks_multiply_const_vxx")
    assert grader.yaml_path(data, "parameters[id=type].options") == ["complex", "float", "int", "short"]
    assert grader.yaml_path(data, "inputs.0.domain") == "stream"
    with pytest.raises(KeyError):
        grader.yaml_path(data, "parameters[id=nope].options")
    with pytest.raises(KeyError):
        grader.block_yaml("no_such_block")


def test_a_fact_can_take_a_number_out_of_a_parameter_expression():
    """A message strobe's message is one expression, pmt.to_pmt({"freq": 4200, "ampl": 1.2}): a regex on the
    raw text takes one value out of it, still derived from the graph at grading time."""
    view = {"ok": True, "blocks": {"strobe": {"key": "blocks_message_strobe", "evaluated": {},
                                              "params": {"msg": 'pmt.to_pmt({"freq": 4200, "ampl": 1.2})'}}}}
    src = {"graph": "start", "block": {"name": "strobe"}, "param": "msg", "use": "raw"}
    freq = {"source": {**src, "regex": r'"freq"\s*:\s*([-+0-9.eE]+)'}, "match": "number"}
    ampl = {"source": {**src, "regex": r'"ampl"\s*:\s*([-+0-9.eE]+)'}, "match": "number"}
    assert grader._fact_value(freq, {"start": view}) == "4200"
    assert grader._fact_value(ampl, {"start": view}) == "1.2"
    with pytest.raises(KeyError):
        grader._fact_value({"source": {**src, "regex": r'"phase"\s*:\s*([0-9.]+)'}, "match": "number"},
                           {"start": view})
    facts = {"answer": {"facts": [{"name": "f", **freq}]}}
    assert grader.acceptance_errors(facts) == []
    bad = {"answer": {"facts": [{"name": "f", "source": {**src, "regex": "no group"}, "match": "number"}]}}
    assert any("regex" in e for e in grader.acceptance_errors(bad))


def test_a_csv_manifest_is_checked_against_the_sizes_of_the_files_it_lists(tmp_path):
    """outputs/manifest.csv: a header, then one row per copied file whose size it states; the sizes depend on
    how long the agent ran, so they are compared with the files themselves."""
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "peaks.dat").write_bytes(b"x" * 40)
    (out / "average.dat").write_bytes(b"y" * 24)
    spec = {"path": "outputs/manifest.csv", "csv_sizes": {"dir": "outputs", "header": ["file", "bytes"],
                                                          "names": ["peaks.dat", "average.dat"]}}
    (out / "manifest.csv").write_text("file,bytes\npeaks.dat,40\naverage.dat,24\n")
    assert grader._files(tmp_path, spec, 0)["passed"]
    (out / "manifest.csv").write_text("file,bytes\npeaks.dat,40\naverage.dat,25\n")
    assert "size" in grader._files(tmp_path, spec, 0)["detail"]
    (out / "manifest.csv").write_text("file,bytes\npeaks.dat,40\n")
    assert "average.dat" in grader._files(tmp_path, spec, 0)["detail"]
    (out / "manifest.csv").write_text("name,size\npeaks.dat,40\naverage.dat,24\n")
    assert "header" in grader._files(tmp_path, spec, 0)["detail"]
    assert grader.acceptance_errors({"files": [spec]}) == []
    assert grader.acceptance_errors({"files": [{**spec, "csv_sizes": {"dir": "../x", "names": []}}]})


def test_a_fact_can_be_measured_from_a_file_the_run_wrote(tmp_path):
    """A run-and-verify answer (the RMS of recovered.dat) is computed at grading time from the run's own file
    with an oracle feature, never written by hand."""
    import numpy as np
    x = np.concatenate([np.zeros(800), 2.0 * np.sin(np.linspace(0, 200 * np.pi, 8000))]).astype(np.float32)
    x.tofile(tmp_path / "recovered.dat")
    fact = {"source": {"file": "recovered.dat", "dtype": "float32", "skip": 800, "feature": "rms"},
            "match": "number", "rel_tol": 0.03}
    assert abs(float(grader._fact_value(fact, {}, tmp_path)) - 2 ** 0.5) < 1e-3
    with pytest.raises(KeyError):
        grader._fact_value({**fact, "source": {**fact["source"], "file": "missing.dat"}}, {}, tmp_path)
    assert grader.acceptance_errors({"answer": {"facts": [{"name": "rms", **fact}]}}) == []
    bad = {"answer": {"facts": [{"name": "rms", **fact, "source": {**fact["source"], "feature": "equal_prefix"}}]}}
    assert grader.acceptance_errors(bad)


def test_acceptance_errors():
    assert grader.acceptance_errors({"blocks_absent": [{"id": "x"}]}) == []
    cases = {
        "no assertion": {"graph": {"path": "a.grc"}, "note": "x"},
        "unknown acceptance keys": {"blocks_absent": [{"id": "x"}], "rubric": "looks good"},
        "needs an oracle": {"ladder": {"rung": "V5"}},
        "ladder.rung": {"ladder": {"rung": "V7"}},
        "names no block id": {"connections": [{"src": {"id": "a"}, "dst": {}}]},
        "relative path": {"files": [{"path": "../escape.txt"}]},
        "regex": {"files": [{"path": "a.txt", "regex": ["("]}]},
        "Python identifier": {"hier_block": {"block_id": "not-an-id"}},
        "graph_unchanged must be true": {"graph_unchanged": False},
        "match must be": {"answer": {"facts": [{"source": {"block_yaml": "x", "path": "id"}, "match": "vibes"}]}},
        "graph or block_yaml": {"answer": {"facts": [{"source": {"value": 3}, "match": "number"}]}},
    }
    for needle, acc in cases.items():
        errs = grader.acceptance_errors(acc)
        assert errs and any(needle in e for e in errs), (needle, errs)
    for d in GATED:
        assert grader.acceptance_errors(load_task(TASKS / d).acceptance) == [], d
    assert {load_task(TASKS / d).category for d in GATED} == CATEGORIES


def test_graph_diff_and_equivalence_up_to_renaming():
    a = copy.deepcopy(TONE_VIEW)
    assert grader.graph_diff(a, TONE_VIEW) == [] and golden.equivalence_diffs(a, TONE_VIEW) == []
    b = copy.deepcopy(TONE_VIEW)
    b["blocks"]["src"]["params"]["freq"] = "1000"
    assert grader.graph_diff(TONE_VIEW, b) == ["src.freq: '2e3' != '1000'"]
    assert golden.equivalence_diffs(b, TONE_VIEW)
    # renaming a block (and the options id) is equivalent, but not unchanged
    r = copy.deepcopy(TONE_VIEW)
    r["blocks"] = {("tone_gen" if n == "src" else "other" if n == "opts" else n): blk for n, blk in r["blocks"].items()}
    r["blocks"]["other"]["params"]["id"] = "other"
    r["connections"] = [["tone_gen", "0", "out", "0"]]
    assert golden.equivalence_diffs(r, TONE_VIEW) == []
    assert grader.graph_diff(r, TONE_VIEW)
    # same blocks, different wiring: not equivalent
    w = copy.deepcopy(TONE_VIEW)
    w["connections"] = []
    assert any("connection" in d for d in golden.equivalence_diffs(w, TONE_VIEW))


# ================================================================ golden files

def test_tier_a_golden_files_are_well_formed_and_in_range():
    for d in GATED:
        g = replay.load_golden(TASKS / d)
        n = replay.tool_call_count(g)
        lo, hi = replay.expected_range(load_task(TASKS / d))
        assert replay.CALL_RANGE[0] <= lo <= n <= hi <= replay.CALL_RANGE[1], (d, n)
        turns = replay.golden_turns(g)
        assert len(turns) == len(g["calls"]) + 1 and turns[-1] == {"text": g["final_text"]}
        models.scripted(turns)                               # survives the trip to the episode process
        assert (TASKS / d / replay.WRONG_GRC).is_file()
    with pytest.raises(replay.GoldenError):
        replay.load_golden(TASKS / "fx-a-build-new")


def test_golden_file_errors(tmp_path):
    def bad(obj):
        (tmp_path / replay.GOLDEN_FILE).write_text(json.dumps(obj))
        with pytest.raises(replay.GoldenError):
            replay.load_golden(tmp_path)
    bad({"calls": [], "final_text": "x"})
    bad({"calls": [{"tool": "inspect_graph"}], "final_text": " "})
    bad({"calls": [{"tool": "inspect_graph", "extra": 1}], "final_text": "x"})
    bad({"calls": [{"tool": "inspect_graph", "args": [1]}], "final_text": "x"})
    (tmp_path / replay.GOLDEN_FILE).write_text(json.dumps({"calls": [{"text": "Which file?"}, {"tool": "save_graph"}],
                                                           "final_text": "ok"}))
    g = replay.load_golden(tmp_path)
    assert replay.tool_call_count(g) == 1
    assert replay.golden_turns(g)[0] == {"text": "Which file?"}
    with pytest.raises(ValueError):
        replay.scripted_run_log_entry(models.openai_compatible(), "gate")


def test_tier_f_golden_derivation_is_deterministic_and_stored():
    for name in F_GATED:
        derived = golden.derive_golden_calls(F_TASKS / name)
        assert derived == golden.load_golden_calls(F_TASKS / name) == golden.derive_golden_calls(F_TASKS / name)
        assert set(derived) == {"query_knowledge", "change_graph"}
        assert derived["query_knowledge"]["domain"] == "catalog" and 5 <= derived["query_knowledge"]["k"] <= 20
        edit = derived["change_graph"]
        assert edit["reason"] and set(edit) - {"reason"} <= set(golden.EDIT_OPS)
    rewire = golden.derive_golden_calls(F_TASKS / "fx_rewire_tone")["change_graph"]
    assert set(rewire) == {"reason", *golden.EDIT_OPS}                     # every operation is exercised
    assert rewire["remove_blocks"] == ["thr", "gain"]
    assert rewire["update_params"] == [{"instance_name": "out", "params": {"unbuffered": "False"}}]
    assert rewire["update_states"] == [{"instance_name": "spare", "state": "enabled"}]
    assert "gain:0->spare:0" in rewire["add_connections"]
    q = golden.derive_golden_calls(F_TASKS / "fx_modify_tone")["query_knowledge"]
    assert q["query"] == "analog_sig_source_x blocks_throttle2 blocks_multiply_const_vxx blocks_file_sink"
    build = golden.derive_golden_calls(F_TASKS / "fx_build_tone")["change_graph"]
    assert [b["instance_name"] for b in build["add_blocks"]] == ["src", "out"] and "remove_blocks" not in build
    assert golden.catalog_default("blocks_file_sink", "unbuffered") == "False"
    assert golden.catalog_default("blocks_multiply_const_vxx", "type") == "complex"    # enum: first option
    turns = golden.golden_turns(golden.load_golden_calls(F_TASKS / "fx_modify_tone"))
    assert [t["tool_calls"][0]["name"] for t in turns[:2]] == ["query_knowledge", "change_graph"]


def test_tier_f_golden_file_errors(tmp_path):
    with pytest.raises(golden.GoldenCallsError):
        golden.load_golden_calls(tmp_path)
    (tmp_path / golden.GOLDEN_FILE).write_text(json.dumps({"change_graph": {"reason": "x"}}))
    with pytest.raises(golden.GoldenCallsError):
        golden.load_golden_calls(tmp_path)


# ================================================================ grading synthetic end states (slow)

def synthetic(task, root, name, graph, **kw):
    return replay.synthetic_end_state(task, root / "synthetic" / name, graph, **kw)


@pytest.mark.slow
@pytest.mark.gnuradio
def test_knowledge_answers_are_rule_checked(root):
    task = load_task(TASKS / "fx-a-knowledge-tone")
    start = task.start_dir / "tone.grc"
    good = replay.load_golden(task.task_dir)["final_text"]
    g = grader.grade(task, synthetic(task, root, "k-good", start, final_text=good))
    assert g["passed"] and g["valid"] and [c["name"] for c in g["checks"]] == [
        "graph_unchanged", "answer[0]", "answer[1]", "answer[2]"], g
    assert g["checks"][1]["expected"] == 32000 and g["checks"][2]["expected"] == 1000
    wrong = grader.grade(task, synthetic(task, root, "k-wrong", start, final_text=good.replace("32000", "48000")
                                         .replace("32 kHz", "48 kHz")))
    assert not wrong["passed"] and wrong["reasons"] == ["answer[0]: sample rate: expects 32000"]
    empty = grader.grade(task, synthetic(task, root, "k-empty", start, final_text="   "))
    assert not empty["passed"] and all("empty" in r for r in empty["reasons"])
    gone = grader.grade(task, synthetic(task, root, "k-nograph", None, final_text=good))
    assert not gone["passed"] and any("graph_unchanged: the final graph is missing" in r for r in gone["reasons"])


@pytest.mark.slow
@pytest.mark.gnuradio
def test_caps_infrastructure_and_missing_graphs(root):
    task = load_task(TASKS / "fx-a-knowledge-tone")
    good = replay.load_golden(task.task_dir)["final_text"]
    end = synthetic(task, root, "caps", task.start_dir / "tone.grc", final_text=good)
    end.status, end.caps_hit = "cap_failure", [{"cap": "requests"}]
    g = grader.grade(task, end)
    assert not g["passed"] and all(c["passed"] for c in g["checks"]) and "cap failure (requests)" in g["reasons"][0]
    end.status = "infrastructure_error"
    g = grader.grade(task, end)
    assert not g["valid"] and not g["passed"] and g["checks"] == []
    build = load_task(TASKS / "fx-a-build-file")
    g = grader.grade(build, synthetic(build, root, "nothing", None, final_text="done"))
    assert not g["passed"] and g["graph"]["exists"] is False
    assert all(not c["passed"] for c in g["checks"])
    assert any(r.startswith("files[0]: run.json does not exist") for r in g["reasons"])
    with pytest.raises(grader.AcceptanceError):
        grader.grade(load_task(TASKS / "fx-a-build-new"), end)


@pytest.mark.slow
@pytest.mark.gnuradio
def test_build_grading_reruns_the_graph_and_ignores_instance_names(root):
    """A correct end state with every instance name changed still passes (name-independent
    assertions); the ladder is re-run on the saved file; a bad run.json fails."""
    task = load_task(TASKS / "fx-a-build-file")
    data = yaml.safe_load((TASKS / "fx-a-build-file" / replay.WRONG_GRC).read_text())
    for b in data["blocks"]:
        if b["name"] == "src":
            b["parameters"]["freq"] = "1e3"
    names = {"src": "tone_gen", "thr": "realtime", "gain": "scale", "out": "writer"}
    for b in data["blocks"]:
        b["name"] = names.get(b["name"], b["name"])
    data["connections"] = [[names.get(c[0], c[0]), c[1], names.get(c[2], c[2]), c[3]] for c in data["connections"]]
    graph = root / "renamed_build.grc"
    graph.write_text(yaml.safe_dump(data, sort_keys=False))
    files = {"run.json": "{\"tone_hz\": 1000, \"samp_rate\": 32000}\n"}
    calls = []

    def spy(text, task_like, max_rung=5, **kw):
        from grcbench.tier_f import ladder
        calls.append((max_rung, "writer" in text))
        return ladder.check(text, task_like, max_rung=max_rung, **kw)
    g = grader.grade(task, synthetic(task, root, "renamed", graph, final_text="built", files=files), ladder_check=spy)
    assert g["passed"], g["reasons"]
    assert calls == [(5, True)]                           # the saved file itself went through V1-V5
    bad = grader.grade(task, synthetic(task, root, "badjson", graph, final_text="built",
                                       files={"run.json": "{\"tone_hz\": 1000}"}))
    assert not bad["passed"] and len(bad["reasons"]) == 1
    assert bad["reasons"][0].startswith("files[0]: run.json: regex") and "missing key 'samp_rate'" in bad["reasons"][0]
    # the source must feed a throttle directly (connections), not only reach the sink
    direct = copy.deepcopy(data)
    direct["connections"] = [c for c in direct["connections"] if c[0] != "tone_gen"] + [["tone_gen", "0", "scale", "0"]]
    direct["connections"] = [c for c in direct["connections"] if c[0] != "realtime"]
    direct["blocks"] = [b for b in direct["blocks"] if b["name"] != "realtime"]
    graph2 = root / "unthrottled_build.grc"
    graph2.write_text(yaml.safe_dump(direct, sort_keys=False))
    g2 = grader.grade(task, synthetic(task, root, "unthrottled", graph2, final_text="built", files=files))
    assert not g2["passed"] and any(r.startswith("connections[0]: 0 matching") for r in g2["reasons"])
    assert any(r.startswith("ladder: fails at V5") for r in g2["reasons"])          # the path check


@pytest.mark.slow
@pytest.mark.gnuradio
def test_wrong_end_states_are_rejected_with_reasons(root):
    expect = {"fx-a-modify-tone": "ladder: fails at V5", "fx-a-repair-type": "ladder: fails at V5",
              "fx-a-build-file": "ladder: fails at V5", "fx-a-knowledge-tone": "graph_unchanged: src.freq",
              "fx-a-save-block": "hier_block: triple_gain.block.yml is not in the run's block library"}
    for d, needle in expect.items():
        task = load_task(TASKS / d)
        cases = replay.wrong_end_states(task, root / "wrong" / d)
        assert len(cases) == 1 and cases[0]["rejected"], (d, cases)
        assert any(r.startswith(needle) for r in cases[0]["reasons"]), (d, cases[0]["reasons"])


# ================================================================ gates through the real runner (slow)

@pytest.mark.slow
@pytest.mark.gnuradio
@pytest.mark.parametrize("name", GATED)
def test_tier_a_gates_pass_on_fixture_tasks(name, root):
    """Golden replay 10/10 (fresh run dir each), 8-20 calls, do-nothing fails, wrong end state
    rejected, the grader re-ran the saved graph; nothing is written into the fixture directory."""
    before = sorted(p.name for p in (TASKS / name).iterdir())
    res = replay.gates(TASKS / name, run_root=root / "gates" / name)
    assert res["passed"], replay.dumps(res)
    gr = res["gates"]["golden_replay"]
    assert gr["runs"] == gr["passes"] == 10 and gr["in_range"] and 8 <= gr["tool_calls"] <= 20
    assert res["gates"]["do_nothing"]["passed"] and res["gates"]["do_nothing"]["reasons"]
    assert res["gates"]["wrong_end_state"]["cases"][0]["rejected"]
    assert sorted(p.name for p in (TASKS / name).iterdir()) == before
    assert not (root / "gates" / name).exists()                 # gate run directories are removed


@pytest.mark.slow
@pytest.mark.gnuradio
def test_golden_replay_records_and_trace_verdict(root):
    """One golden replay through runner.run_episode(grade_episode=True): every call ran, the
    result carries the grade and the trace row's verdict says the acceptance passed."""
    task = load_task(TASKS / "fx-a-save-block")
    spec = replay.golden_model(task.task_dir)
    res = runner.run_episode(task, spec, "benchmark", root / "graded", config_id="pytest-grader",
                             run_log_entry=replay.scripted_run_log_entry(spec, "pytest"), grade_episode=True)
    assert res.status == "completed" and res.grade["passed"], res.grade
    assert [c["tool"] for c in res.tool_calls] == [c["tool"] for c in replay.load_golden(task.task_dir)["calls"]]
    assert all(c["execution"] == "ok" for c in res.tool_calls)
    lib = Path(res.run_dir) / "home" / ".grc_gnuradio"
    assert sorted(p.name for p in lib.iterdir()) == ["triple_gain.block.yml", "triple_gain.py"]
    row = json.loads((Path(res.run_dir) / runner.TRACE_FILE).read_text())
    assert row["verdict"] == "completed" and row["grcbench"]["acceptance_passed"] is True
    assert read_json(Path(res.run_dir) / runner.RESULT_FILE)["grade"]["passed"] is True


@pytest.mark.slow
@pytest.mark.gnuradio
@pytest.mark.parametrize("name", F_GATED)
def test_tier_f_replay_gate_on_fixtures(name, root):
    res = replay.tier_f_gates(F_TASKS / name, run_root=root / "tier_f" / name)
    assert res["passed"], replay.dumps(res)
    assert res["gates"]["tier_f_replay"]["passes"] == 10
    assert res["gates"]["do_nothing"]["failure"] == "no_edit"


@pytest.mark.slow
@pytest.mark.gnuradio
def test_tier_f_golden_verification_through_grc_agent_change_graph():
    for name in F_GATED:
        res = golden.verify_golden_calls(F_TASKS / name)
        assert res["accepted"] and res["equivalent"] and res["diffs"] == [], (name, res)
    calls = golden.load_golden_calls(F_TASKS / "fx_modify_tone")
    wrong = copy.deepcopy(calls)
    wrong["change_graph"]["update_params"][0]["params"]["freq"] = "1500"
    res = golden.verify_golden_calls(F_TASKS / "fx_modify_tone", wrong)
    assert res["accepted"] and not res["equivalent"] and res["diffs"] == ["src.freq: '1500' != '2000'"]
    bad = copy.deepcopy(calls)
    bad["change_graph"] = {"reason": "x", "add_blocks": [{"block_id": "no_such_block_x", "instance_name": "n"}]}
    res = golden.verify_golden_calls(F_TASKS / "fx_modify_tone", bad)
    assert not res["accepted"] and "block_type_not_found" in res["diffs"][0]


# ================================================================ CLI and smoke

def test_cli_lists_the_new_subcommands(capsys):
    from grcbench.tier_a import cli
    assert cli.main(["help"]) == 0
    out = capsys.readouterr().out
    for name in ("gates", "run", "smoke", "tier-f-golden", "tier-f-sample"):
        assert name in out


def test_cli_tier_f_sample_refuses_unguarded_test_runs(tmp_path, capsys, monkeypatch):
    from grcbench.tier_a import cli
    task = tmp_path / "fx_modify_tone"
    shutil.copytree(F_TASKS / "fx_modify_tone", task)
    rec = read_json(task / "task.json")
    rec["split"] = "test"
    (task / "task.json").write_text(json.dumps(rec))
    import grcbench.tier_a.tier_f_mode as tfm
    monkeypatch.setattr(tfm.run_tier_f_sample, "__kwdefaults__", {**tfm.run_tier_f_sample.__kwdefaults__,
                                                                   "run_list": tmp_path / "absent.yaml",
                                                                   "log_path": tmp_path / "log.jsonl"})
    assert cli.main(["tier-f-sample", str(task), "--base-url", "http://127.0.0.1:9/v1"]) == 2
    assert "refused" in capsys.readouterr().out and not (tmp_path / "log.jsonl").exists()


def test_smoke_talks_to_local_servers_only():
    from grcbench.tier_a import smoke
    with pytest.raises(ValueError, match="local"):
        smoke.server_props("http://example.com:8080/v1")
    a, f = smoke.fixture_tasks()
    assert [d.name for d in a] == sorted(GATED) and [d.name for d in f] == sorted(F_GATED)


@pytest.mark.slow
@pytest.mark.gnuradio
def test_cli_run_grades_and_tier_f_golden_checks(tmp_path, capsys, root):
    from grcbench.tier_a import cli
    turns = tmp_path / "turns.json"
    turns.write_text(json.dumps(replay.golden_turns(replay.load_golden(TASKS / "fx-a-knowledge-tone"))))
    assert cli.main(["run", str(TASKS / "fx-a-knowledge-tone"), "--scripted", str(turns),
                     "--run-root", str(root / "cli")]) == 0
    out = capsys.readouterr().out
    assert "fx-a-knowledge-tone" in out and " PASS " in out
    nothing = tmp_path / "nothing.json"
    nothing.write_text(json.dumps([{"text": "done"}]))
    assert cli.main(["run", str(TASKS / "fx-a-knowledge-tone"), "--scripted", str(nothing),
                     "--run-root", str(root / "cli")]) == 0
    out = capsys.readouterr().out
    assert " fail " in out and "answer[0]: sample rate: expects 32000" in out
    assert cli.main(["tier-f-golden", str(F_TASKS / "fx_rewire_tone"), "--no-write", "--no-gate"]) == 0
    assert "golden equivalent" in capsys.readouterr().out


@pytest.mark.slow
@pytest.mark.gnuradio
def test_smoke_golden_replays_on_fixtures(root):
    from grcbench.tier_a import smoke
    summary = smoke.run_smoke(run_root=root / "smoke")
    assert summary["golden_passed"], summary["lines"]
    assert len(summary["golden"]) == len(GATED) + len(F_GATED) and summary["model"] is None
    assert Path(summary["path"]).is_file()

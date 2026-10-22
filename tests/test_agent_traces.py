"""`./gb traces`: the audit of Qoherent's agent traces, its manifest and the contamination corpus."""

from __future__ import annotations

import json

import pytest

from grcbench.common.paths import AGENT_TRACES, REPORTS, SPLITS
from grcbench.contamination import agent_traces as at
from grcbench.contamination import corpora

HAVE_TRACES = all(p.is_file() for p in AGENT_TRACES.values())
needs_traces = pytest.mark.skipif(not HAVE_TRACES, reason="new_data/ (Qoherent's agent traces) is not on this machine")


def test_helpers():
    assert at.family("t1_addwire_audio_sink_in0") == "t1_addwire"
    assert at.family("t2_build_a_to_b") == "t2_build"
    assert at.functional(["variable", "blocks_null_sink", "analog_agc2_xx", "blocks_throttle2"]) == ("analog_agc2_xx",)
    row = {"system": "Session ID: x\nActive flowgraph file path: /tmp/p/project/t1_analog_agc2_xx.grc\nmore",
           "messages": [{"role": "assistant", "tool_calls": [
               {"id": "c1", "function": {"name": "change_graph",
                                         "arguments": json.dumps({"add_blocks": [{"block_id": "blocks_null_source"}]})}}]}]}
    assert at.start_graph(row, set()) == "analog_agc2_xx"
    assert at.final_blocks(row, "analog_agc2_xx") == ["analog_agc2_xx", "blocks_null_source"]
    empty = {"system": "Active flowgraph file path: /tmp/p/project/t1_empty.grc", "messages": []}
    assert at.start_graph(empty, set()) is None


def test_the_committed_report_and_manifest_agree():
    doc = json.loads((REPORTS / "agent_traces.json").read_text(encoding="utf-8"))
    man = json.loads((SPLITS / "agent_traces.json").read_text(encoding="utf-8"))
    assert man == at.manifest(doc)
    assert doc["tool_use"]["tier1"]["rows"] == 1253 and doc["tool_use"]["tier2"]["rows"] == 600
    assert doc["task_level"]["same_start_graph"] == [] and doc["task_level"]["same_final_blocks"] == []
    assert doc["harness"]["system_prompt_equal"] and doc["harness"]["tools_only_in_traces"] == ["final_result"]
    replies = {**doc["tool_use"]["tier1"]["run_flowgraph_replies"], **doc["tool_use"]["tier2"]["run_flowgraph_replies"]}
    assert all("not wired up" in r for r in replies)                        # no trace ran a flowgraph


@needs_traces
def test_the_audit_reproduces_the_committed_report():
    assert json.loads(json.dumps(at.build(), default=str)) == json.loads((REPORTS / "agent_traces.json").read_text(encoding="utf-8"))


@needs_traces
def test_the_traces_corpus_loads_every_trace_once():
    corpus = corpora.training_corpus("agent_traces", None)
    ids = [d.id for d in corpus.docs]
    assert len(ids) == 1853 + 1 and len(set(ids)) == len(ids)              # traces + the shared system prompt
    assert ids[-1] == "system_prompt" and "Session ID" not in corpus.docs[-1].text
    assert all(d.text for d in corpus.docs)

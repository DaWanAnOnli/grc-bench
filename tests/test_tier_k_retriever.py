"""B1 retriever (grcbench.tier_k.retriever): GRC-Agent's own retrieval, added to the prompt.

Fast tests check the pure parts (interleaving, the token cap, the block format). The live tests run
GRC-Agent's query_catalog / query_docs in isolated episode processes on the read-only database
copies and check: a known block's catalog entry comes back for its name; the hits, ids and scores
are deterministic; every passage is byte-identical to what GRC-Agent's real query_knowledge tool
returns; the B1 block respects its token cap and shortens an oversized top hit instead of
withholding it; a database rebuild request fails loudly; and nothing under ~/grc-agent (nor the
database copies) is written. Queries and questions here are written for the tests; no benchmark
item is used.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import uuid

import pytest

from grcbench.common import manifest, sandbox, tokens
from grcbench.common.io import read_json, sha256_file
from grcbench.common.paths import GRC_AGENT
from grcbench.tier_k import retriever as rt
from grcbench.tier_k.retriever import B1Context, Passage, SearchResult

REQUESTS = [
    ("analog_sig_source_x", "catalog", 5),
    ("Signal Source", "catalog", 5),
    ("throttle block sample rate", "docs", 5),
    ("low pass filter cutoff frequency transition width", "catalog", 3),
    ("???", "docs", 5),                 # no word token: GRC-Agent answers ok with no results
    ("   ", "catalog", 5),              # empty after normalisation: GRC-Agent answers ok=False
]
QUESTIONS = [
    "What does the Throttle block do, and when should it be used in a flowgraph?",
    "How is the sample rate of a Signal Source related to the frequency of the waveform it produces, "
    "what happens when that frequency is above half the sample rate, and which block would you add "
    "to keep a simulation without hardware from using all of the CPU?",
    # GRC-Agent's docs rank 1 here is a chunk (gnu_native_helpers_reference#588, ~6.8K tokens)
    # longer than the whole B1 cap: it must be shortened into the block, not withheld.
    "firdes filter design helpers window taps",
]
OVERSIZED_TOP_HIT = "docs:gnu_native_helpers_reference#588"
# A block that is not in the catalog the pinned database was built from: GRC-Agent's freshness
# check then wants to rebuild the (read-only) catalog database.
EXTRA_BLOCK_YML = """id: grcbench_test_extra_block
label: GRC-Bench Test Extra Block
category: '[GRC Hier Blocks]'
parameters: []
inputs: []
outputs: []
templates:
  imports: 'import math'
  make: 'math.sqrt(1)'
documentation: ''
file_format: 1
"""


# ---------------------------------------------------------------- pure parts

def _passage(domain: str, rank: int, n_words: int, tag: str) -> Passage:
    text = f"{tag} " + " ".join(["word"] * n_words)
    pid = f"catalog:{tag}" if domain == "catalog" else f"docs:{tag}#{rank}"
    return Passage(pid, domain, rank, float(100 - rank), tag, text, {"tag": tag})


def _result(domain: str, sizes: list[int], k: int = 5, ok: bool = True) -> SearchResult:
    ps = tuple(_passage(domain, i + 1, n, f"{domain[0]}{i + 1}") for i, n in enumerate(sizes))
    return SearchResult("q", domain, k, ok, ps, "lexical", (), False, "{}")


def test_k_and_domain_are_checked_like_grc_agents_schema():
    for bad in (0, 21, -1, True, 2.0, "5"):
        with pytest.raises(ValueError):
            rt.check_k(bad)
    assert rt.check_k(1) == 1 and rt.check_k(20) == 20
    with pytest.raises(ValueError):
        rt.check_domain("web")
    with pytest.raises(ValueError):
        rt.search_many([("q", "catalog", 0)])          # rejected before any process starts


def test_interleave_takes_rank_one_of_each_domain_first():
    order = [p.id for p in rt.interleave([_result("catalog", [5, 5, 5]), _result("docs", [5])])]
    assert order == ["catalog:c1", "docs:d1#1", "catalog:c2", "catalog:c3"]
    assert rt.interleave([]) == []


def test_b1_block_respects_the_cap_and_cuts_no_passage_but_a_top_hit():
    results = [_result("catalog", [40, 900, 30]), _result("docs", [60, 20, 2000])]
    for cap in (200, 400, 1500, rt.B1_TOKEN_CAP):
        ctx = rt.build_b1_context(results, token_cap=cap)
        assert ctx.tokens == tokens.count(ctx.text) <= cap
        assert sorted(ctx.included + ctx.skipped) == sorted(p.id for p in rt.interleave(results))
        assert ctx.shortened == ()                               # every top hit here fits whole
        by_id = {p.id: p for p in rt.interleave(results)}
        for pid in ctx.included:
            assert by_id[pid].text in ctx.text                   # whole
        for pid in ctx.skipped:
            assert by_id[pid].text.split()[0] + " " not in ctx.text   # absent, not cut
    ctx = rt.build_b1_context(results, token_cap=400)
    # 900- and 2000-word passages (not top hits) are skipped; later, shorter ones are still added.
    assert ctx.included == ("catalog:c1", "docs:d1#1", "docs:d2#2", "catalog:c3")
    assert ctx.skipped == ("catalog:c2", "docs:d3#3")
    assert ctx.text.startswith(rt.OPEN_TAG + "\n" + rt.B1_HEADER + "\n\n[1] Block catalog entry: c1\n")
    assert "\n\n[2] Documentation page: d1\n" in ctx.text and ctx.text.endswith("\n" + rt.CLOSE_TAG)
    assert ctx.k == 5


def _big_top_hit(n_words: int = 6000) -> Passage:
    text = "path: big\nheading: Helpers\nBEGINNING " + " ".join(["filter"] * n_words) + " ENDING"
    return Passage("docs:big#7", "docs", 1, 50.0, "big", text, {"text": text})


def test_an_oversized_top_hit_is_shortened_like_a_tier_a_tool_output_not_withheld():
    big = _big_top_hit()
    rest = tuple(dataclasses.replace(p, rank=p.rank + 1) for p in _result("docs", [60, 20]).passages)
    docs = SearchResult("q", "docs", 5, True, (big, *rest), "lexical", (), False, "{}")
    results = [_result("catalog", [40, 30, 50]), docs]
    empty = tokens.count(rt._wrap(rt.B1_HEADER, [], rt.NO_PASSAGES))
    for cap in (700, 1500, rt.B1_TOKEN_CAP):
        ctx = rt.build_b1_context(results, token_cap=cap)
        assert ctx.tokens == tokens.count(ctx.text) <= cap
        assert ctx.included[:2] == ("catalog:c1", "docs:big#7") and ctx.shortened == ("docs:big#7",)
        start = ctx.text.index("[2] Documentation page: big\n")
        end = ctx.text.index("\n\n[3] ") if "\n\n[3] " in ctx.text else ctx.text.index(rt.CLOSE_TAG)
        shown = ctx.text[start:end].rstrip("\n")
        assert shown.startswith("[2] Documentation page: big\npath: big\nheading: Helpers\n"
                                "BEGINNING filter")
        assert shown.endswith("filter ENDING")
        assert "truncated by the GRC-Bench 4K tool-output cap" in shown
        assert tokens.count(shown) <= (cap - empty) // 2 + 16       # at most half the passage budget
    # The rest of the ranking still gets the other half.
    full = rt.build_b1_context(results)
    assert set(full.included) == {p.id for p in rt.interleave(results)} and full.skipped == ()
    for p in (*results[0].passages, *rest):
        assert p.text in full.text
    # An oversized passage below rank 1 is skipped, as before.
    later = SearchResult("q", "docs", 5, True, (dataclasses.replace(rest[0], rank=1),
                                                dataclasses.replace(big, rank=2)),
                         "lexical", (), False, "{}")
    ctx = rt.build_b1_context([later])
    assert ctx.included == ("docs:d1#1",) and ctx.skipped == ("docs:big#7",) and ctx.shortened == ()
    assert "BEGINNING" not in ctx.text


def test_b1_block_without_passages_says_why_and_a_failed_search_is_an_error():
    ctx = rt.build_b1_context([_result("catalog", []), _result("docs", [])])
    assert ctx.text == f"{rt.OPEN_TAG}\n{rt.B1_HEADER}\n\n{rt.NO_PASSAGES}\n{rt.CLOSE_TAG}"
    assert ctx.included == ctx.skipped == ctx.notices == ()
    floor = max(tokens.count(rt._wrap(rt.B1_HEADER, [], e)) for e in (rt.NO_PASSAGES, rt.NO_FIT))
    assert rt.check_cap(floor) == floor
    tiny = rt.build_b1_context([_result("catalog", [40]), _result("docs", [60])], token_cap=floor + 5)
    assert tiny.included == () and tiny.skipped == ("catalog:c1", "docs:d1#1")
    # passages matched but none fitted: said so, not "no passages matched"
    assert tiny.text == f"{rt.OPEN_TAG}\n{rt.B1_HEADER}\n\n{rt.NO_FIT}\n{rt.CLOSE_TAG}"
    with pytest.raises(rt.RetrieverError, match="failed search"):
        rt.build_b1_context([_result("catalog", [], ok=False), _result("docs", [5])])
    for bad_cap in (10, floor - 1, True, 4096.0):
        with pytest.raises(ValueError):
            rt.build_b1_context([], token_cap=bad_cap)
    with pytest.raises(ValueError):                      # rejected before any process starts
        rt.b1_contexts(["What is a Throttle?", " \n\t"])
    with pytest.raises(ValueError):
        rt.b1_contexts(["What is a Throttle?"], token_cap=10)
    assert rt.add_to_prompt("Question: why?", "BLOCK") == "BLOCK\n\nQuestion: why?"


def test_reference_block_ids_are_every_enabled_block_of_the_graph():
    graph = {"options": {"parameters": {"id": "x"}},
             "blocks": [{"name": "samp_rate", "id": "variable"},
                        {"name": "src", "id": "analog_sig_source_x"},
                        {"name": "src2", "id": "analog_sig_source_x"},
                        {"name": "old", "id": "blocks_throttle2", "states": {"state": "disabled"}},
                        {"name": "snk", "id": "blocks_null_sink", "states": {"state": "enabled"}}]}
    assert rt.reference_block_ids(graph) == ["analog_sig_source_x", "blocks_null_sink", "variable"]
    assert rt.reference_block_ids({}) == []


def test_search_result_round_trips_through_its_dict():
    r = _result("docs", [3, 4])
    assert SearchResult.from_dict(json.loads(json.dumps(r.to_dict()))) == r


# ---------------------------------------------------------------- live: GRC-Agent in episode processes

def _db_state() -> dict[str, list]:
    from grcbench.tier_a import isolation
    out = {}
    for name in isolation.LEXICAL_DBS:
        for root in (isolation.VECTORS_SRC, isolation.VECTORS_CACHE):
            p = root / name
            st = p.stat()
            out[str(p)] = [st.st_size, st.st_mtime_ns, sha256_file(p)]
    return out


def _real_tool_results(requests) -> list[dict]:
    """GRC-Agent's real query_knowledge tool function, called in its own episode process."""
    from grcbench.tier_a import isolation
    run_dir = isolation.check_run_dir(rt.RUNS_ROOT / f"pytest-tool-{uuid.uuid4().hex[:8]}")
    run_dir.mkdir(parents=True)
    try:
        env = isolation.make_run_env(run_dir)
        (run_dir / "requests.json").write_text(json.dumps(requests), encoding="utf-8")
        code = (
            "import asyncio, json, sys\n"
            "sys.modules['gi'] = None\n"
            "from pydantic_ai import ModelRetry\n"
            "from grc_agent.agent import query_knowledge_func\n"
            "out = []\n"
            "for q, d, k in json.load(open('requests.json')):\n"
            "    try:\n"
            "        out.append({'tool': asyncio.run(query_knowledge_func(None, q, d, k))})\n"
            "    except ModelRetry as exc:\n"
            "        out.append({'retry': str(exc)})\n"
            "json.dump(out, open('tool.json', 'w'))\n")
        res = sandbox.run([sys.executable, "-c", code], run_dir, timeout=600, cpu_seconds=600,
                          mem_bytes=8 << 30, env=env)
        assert res.returncode == 0, res.stderr[-3000:]
        return read_json(run_dir / "tool.json")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def live():
    if not shutil.which("bwrap") or not (GRC_AGENT / "src" / "grc_agent" / "vectors").is_dir():
        pytest.skip("needs WSL with bubblewrap and ~/grc-agent")
    before = {"grc_agent": manifest.build(GRC_AGENT), "dbs": _db_state()}
    data = {
        "batch1": rt.search_many(REQUESTS),
        "batch2": rt.search_many(REQUESTS),
        "ctx1": rt.b1_contexts(QUESTIONS, 5),
        "ctx2": rt.b1_contexts(QUESTIONS, 5),
        "ctx_small": rt.b1_contexts(QUESTIONS, 5, token_cap=700),
        "single": rt.b1_context(QUESTIONS[0]),
        "retrieve": rt.retrieve("analog_sig_source_x", "catalog", 5),
        "grounding": rt.oracle_grounding(["blocks_throttle2", "analog_sig_source_x", "analog_sig_source_x"]),
        "tool": _real_tool_results(REQUESTS),
    }
    after = {"grc_agent": manifest.build(GRC_AGENT), "dbs": _db_state()}
    data["before"], data["after"] = before, after
    return data


@pytest.mark.gnuradio
@pytest.mark.slow
def test_known_block_entry_is_returned_for_its_name(live):
    by_id, by_label = live["batch1"][0], live["batch1"][1]
    assert by_id.ok and by_id.search_mode == "lexical"
    top = by_id.passages[0]
    assert top.id == "catalog:analog_sig_source_x" and top.rank == 1 and top.title == "analog_sig_source_x"
    assert top.result["block_id"] == "analog_sig_source_x" and top.result["label"] == "Signal Source"
    assert {"params", "inputs", "outputs", "category", "doc"} <= set(top.result)
    assert by_label.passages[0].id == "catalog:analog_sig_source_x"      # its GRC label finds it too
    assert [p.id for p in live["retrieve"]] == [p.id for p in by_id.passages]


@pytest.mark.gnuradio
@pytest.mark.slow
def test_ranked_passages_have_unique_ids_and_bm25_scores(live):
    for (query, domain, k), r in zip(REQUESTS, live["batch1"]):
        assert (r.query, r.domain, r.k) == (query, domain, k)
        assert len(r.passages) <= k
        assert [p.rank for p in r.passages] == list(range(1, len(r.passages) + 1))
        assert len({p.id for p in r.passages}) == len(r.passages)
        scores = [p.score for p in r.passages]
        assert all(s > 0 for s in scores) and scores == sorted(scores, reverse=True)
        prefix = "catalog:" if domain == "catalog" else "docs:"
        assert all(p.id.startswith(prefix) and p.domain == domain for p in r.passages)
    docs = live["batch1"][2]
    assert len(docs.passages) == 5 and all("#" in p.id for p in docs.passages)
    assert all(p.text.startswith(f"path: {p.title}\nheading: ") for p in docs.passages)
    nothing, empty = live["batch1"][4], live["batch1"][5]
    assert nothing.ok and nothing.passages == ()
    assert not empty.ok and empty.passages == () and "non-empty" in empty.tool_result


@pytest.mark.gnuradio
@pytest.mark.slow
def test_same_output_as_grc_agents_query_knowledge_tool(live):
    for r, tool in zip(live["batch1"], live["tool"]):
        if r.ok:
            assert tool == {"tool": r.tool_result}
            for p in r.passages:   # each passage is the tool's own bytes for that hit
                assert (p.text if p.domain == "catalog" else json.dumps(p.text)) in r.tool_result
                assert json.dumps(p.result) in r.tool_result
        else:
            assert "retry" in tool and json.loads(r.tool_result)["message"] in tool["retry"]


@pytest.mark.gnuradio
@pytest.mark.slow
def test_retrieval_is_deterministic(live):
    assert [r.to_dict() for r in live["batch1"]] == [r.to_dict() for r in live["batch2"]]
    assert [c.to_dict() for c in live["ctx1"]] == [c.to_dict() for c in live["ctx2"]]
    assert live["single"] == live["ctx1"][0].text


@pytest.mark.gnuradio
@pytest.mark.slow
def test_b1_context_token_cap_is_respected(live):
    for ctxs, cap in ((live["ctx1"], rt.B1_TOKEN_CAP), (live["ctx_small"], 700)):
        for c in ctxs:
            assert isinstance(c, B1Context) and c.token_cap == cap and c.k == 5
            assert c.tokens == tokens.count(c.text) <= cap
            assert c.included, "at least one passage fits"
            assert c.text.startswith(rt.OPEN_TAG) and c.text.endswith(rt.CLOSE_TAG)
    small, full = live["ctx_small"][0], live["ctx1"][0]
    assert small.skipped and len(small.included) < len(full.included)
    assert set(small.included) | set(small.skipped) == set(full.included) | set(full.skipped)
    assert "catalog:blocks_throttle2" in full.included or "catalog:blocks_throttle" in full.included
    long_q = live["ctx1"][1]   # 35 distinct word tokens: GRC-Agent's 32-token cap is disclosed
    assert any("truncated the query" in n for n in long_q.notices)


@pytest.mark.gnuradio
@pytest.mark.slow
def test_an_oversized_docs_top_hit_is_shortened_into_the_block(live):
    ctx = live["ctx1"][2]
    assert ctx.shortened == (OVERSIZED_TOP_HIT,) and ctx.included[1] == OVERSIZED_TOP_HIT
    # the other half of the budget still holds the rest of the ranking (k=5 in both domains)
    assert ctx.skipped == () and len(ctx.included) == 10 and ctx.tokens <= rt.B1_TOKEN_CAP
    top = rt.retrieve(QUESTIONS[2], "docs", 1)[0]
    assert top.id == OVERSIZED_TOP_HIT and tokens.count(top.text) > rt.B1_TOKEN_CAP
    assert top.text[:200] in ctx.text and top.text[-200:] in ctx.text   # beginning and end kept
    assert top.text not in ctx.text


@pytest.mark.gnuradio
@pytest.mark.slow
def test_a_database_rebuild_request_fails_loudly(monkeypatch, capsys):
    if not shutil.which("bwrap") or not (GRC_AGENT / "src" / "grc_agent" / "vectors").is_dir():
        pytest.skip("needs WSL with bubblewrap and ~/grc-agent")
    from grcbench.tier_a import isolation
    real = isolation.make_run_env

    def with_extra_block(run_dir, **kw):
        env = real(run_dir, **kw)
        (isolation.RunLayout(run_dir).hier_dir / "grcbench_test_extra_block.block.yml").write_text(
            EXTRA_BLOCK_YML, encoding="utf-8")
        return env

    before = _db_state()
    monkeypatch.setattr(isolation, "make_run_env", with_extra_block)
    with pytest.raises(rt.RetrieverError,
                       match="catalog: GRC-Agent's search failed: Catalog DB build failed"):
        rt.b1_contexts([QUESTIONS[0]])
    assert rt.main(["--b1", "throttle"]) == 1
    assert "Catalog DB build failed" in capsys.readouterr().err
    assert rt.retrieve("throttle", "docs", 3)      # the docs database is still current
    assert _db_state() == before                  # nothing was rebuilt or written


@pytest.mark.gnuradio
@pytest.mark.slow
def test_oracle_grounding_returns_grc_agents_catalog_entries(live):
    entries = live["grounding"]
    assert [e["block_id"] for e in entries] == ["analog_sig_source_x", "blocks_throttle2"]
    top = live["batch1"][0].passages[0]
    assert entries[0] == top.result and json.dumps(entries[0]) == top.text   # same entry, same bytes
    text = rt.grounding_context_text(entries)
    assert text.count("Block catalog entry: ") == 2 and json.dumps(entries[1]) in text
    with pytest.raises(rt.RetrieverError, match="no_such_block_xyz"):
        rt.oracle_grounding(["analog_sig_source_x", "no_such_block_xyz"])
    assert rt.oracle_grounding([]) == []


@pytest.mark.gnuradio
@pytest.mark.slow
def test_nothing_under_grc_agent_or_the_database_copies_is_written(live):
    changes = manifest.compare(live["before"]["grc_agent"], live["after"]["grc_agent"])
    assert changes == {"added": [], "removed": [], "changed": []}, changes
    assert live["before"]["dbs"] == live["after"]["dbs"]          # size, mtime and SHA-256
    leftovers = [p.name for p in rt.RUNS_ROOT.glob(f"run-{os.getpid()}-*")]
    assert leftovers == []                                         # run directories are removed


@pytest.mark.gnuradio
@pytest.mark.slow
def test_cli_prints_ranked_results(capsys, live):
    assert rt.main(["analog_sig_source_x", "--domain", "catalog"]) == 0
    out = capsys.readouterr().out
    assert "catalog (search mode lexical, k=5)" in out and "catalog:analog_sig_source_x" in out
    assert rt.main(["--grounding", "analog_sig_source_x", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0] == live["grounding"][0]

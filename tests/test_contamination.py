"""Tests for the contamination audit tooling (eval plan §6): tokenizer, the 26 near-twins, TF-IDF,
passages, the embedding client against a mocked llama-server, and the audit on toy corpora."""

from __future__ import annotations

import csv
import json
import math
import os
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from grcbench.common.io import read_jsonl
from grcbench.common.paths import FROZEN
from grcbench.contamination import audit, corpora, embed, ngram, tfidf
from grcbench.contamination.corpora import Corpus, Doc


# --------------------------------------------------------------------------- tokenizer, 8-grams --

def test_tokenizer_is_the_documented_one():
    assert ngram.tokenize("Hello, World! FM-radio @ 2.4GHz_band\tSeñal") == \
        ["hello", "world", "fm", "radio", "2", "4ghz_band", "señal"]
    assert ngram.tokenize("  ...  ") == []
    toks = ngram.tokenize("one two three four five six seven eight nine")
    assert ngram.ngrams(toks) == ["one two three four five six seven eight", "two three four five six seven eight nine"]
    assert ngram.ngrams(toks[:7]) == []


def test_reproduces_26_test_questions_with_an_8gram_training_twin():
    test = [r["question"] for r in read_jsonl(FROZEN / "eval_questions.jsonl")]
    train = [r["question"] for r in read_jsonl(FROZEN / "train_questions.jsonl")]
    assert (len(test), len(train)) == (185, 1668)
    assert len(ngram.items_with_twin(test, train)) == 26


def test_ngram_index_counts_shared_grams_and_documents():
    q = ["a b c d e f g h i j", "nothing here at all to match with anything"]
    docs = ["x a b c d e f g h y", "a b c d e f g h i j k", "unrelated"]
    out = ngram.NgramIndex(q).scan(docs)
    assert out[0].n_grams == 3 and len(out[0].shared) == 3 and out[0].has_twin
    assert out[0].docs == {0: 1, 1: 3}
    assert out[0].top_docs(1) == [(1, 3)]
    assert out[0].coverage == 1.0
    assert not out[1].has_twin and out[1].coverage == 0.0
    text = "Intro. Then A b, C d e f g h and more."
    a, b = ngram.locate(text, ["zz zz zz zz zz zz zz zz", "a b c d e f g h"], context=0)
    assert text[a:b] == "A b, C d e f g h"
    a, b = ngram.locate(text, ["a b c d e f g h"], context=1)
    assert text[a:b] == "Then A b, C d e f g h and"
    assert ngram.locate(text, ["q r s t u v w x"]) is None


# --------------------------------------------------------------------------- TF-IDF -------------

def test_tfidf_weights_and_cosine_on_a_toy_corpus():
    docs = [ngram.tokenize(t) for t in ["the cat sat on the mat", "dogs chase cats", "quantum radio waves"]]
    m = tfidf.Tfidf.fit(docs)
    assert m.n_docs == 3
    # w = (1 + ln tf) * (ln((1 + N) / (1 + df)) + 1); "the" has tf 2, df 1
    row = m.transform([ngram.tokenize("the the")]).toarray()[0]
    assert row[m.vocab["the"]] == pytest.approx(1.0)
    q = m.transform_texts(["the cat sat on the mat", "the cat sat on a mat", "zebra"])
    d = m.transform(docs)
    sims = tfidf.cosine(q, d)
    assert sims[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert 0.5 < sims[1, 0] < 1.0                       # "a" is unknown to the corpus but keeps its norm weight
    assert sims[1, 1] == 0.0 and sims[1, 2] == 0.0
    assert np.all(sims[2] == 0.0)                       # only out-of-vocabulary words
    # by hand: query "the cat sat on a mat" (tf 1 each; "a" unseen, df 0) vs doc 0 (tf("the") = 2)
    idf = {w: m.idf[m.vocab[w]] for w in ["the", "cat", "sat", "on", "mat"]}
    idf_a = math.log((1 + 3) / (1 + 0)) + 1.0
    q_w = np.array([idf["the"], idf["cat"], idf["sat"], idf["on"], idf["mat"], idf_a])
    d_w = np.array([(1 + math.log(2)) * idf["the"], idf["cat"], idf["sat"], idf["on"], idf["mat"]])
    manual = float(q_w[:5] @ d_w) / (np.linalg.norm(q_w) * np.linalg.norm(d_w))
    assert sims[1, 0] == pytest.approx(manual, abs=1e-5)


def test_tfidf_nearest_is_chunk_invariant_and_deterministic():
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(30)]
    corpus = [list(rng.choice(words, 8)) for _ in range(50)] + [["w1", "w2"], ["w1", "w2"]]
    m = tfidf.Tfidf.fit(corpus)
    q = m.transform([["w1", "w2"], ["w3", "w4", "w5"]])
    whole = tfidf.nearest(q, [m.transform(corpus)], 5)
    chunked = tfidf.nearest(q, [m.transform(corpus[a:a + 7]) for a in range(0, len(corpus), 7)], 5)
    assert np.array_equal(whole[1], chunked[1]) and np.allclose(whole[0], chunked[0])
    assert whole[1][0, :2].tolist() == [50, 51]          # tie at cosine 1.0 broken by lower index
    assert np.all(np.diff(whole[0], axis=1) <= 0)


def test_passages_cover_long_documents():
    short = " ".join(f"w{i}" for i in range(corpora.PASSAGE_SINGLE_MAX))
    assert corpora.passages(short) == [(0, len(short))]
    long = " ".join(f"w{i}" for i in range(200))
    spans = corpora.passages(long)
    assert len(spans) == 6
    assert long[spans[0][0]:spans[0][1]].split()[0] == "w0"
    assert long[spans[-1][0]:spans[-1][1]].split()[-1] == "w199"
    assert all(len(long[a:b].split()) == corpora.PASSAGE_WORDS for a, b in spans)
    assert corpora.passages("  !!  ") == []


def test_benchmark_tiers_read_items_and_merge_frozen_duplicates(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(corpora, "ITEMS", tmp_path)
    frozen = read_jsonl(FROZEN / "eval_questions.jsonl")
    (tmp_path / "tier_k" / "test").mkdir(parents=True)
    (tmp_path / "tier_k" / "test" / "a.jsonl").write_text(
        json.dumps({"id": "K-test-0001", "question": frozen[0]["question"] + "  "}) + "\n"
        + json.dumps({"id": "K-test-block-0001", "question": "Which block converts complex to magnitude squared?"}) + "\n")
    task = tmp_path / "tier_f" / "test" / "F-001"
    task.mkdir(parents=True)
    (task / "task.json").write_text(json.dumps({"id": "F-001", "instruction": "Build a 1 kHz tone flowgraph."}))
    k = corpora.benchmark_tier("tier_k_test")
    assert len(k.docs) == 186 and k.docs[0].aliases == ("K-test-0001",) and k.docs[-1].id == "K-test-block-0001"
    f = corpora.benchmark_tier("tier_f_test")
    assert [d.id for d in f.docs] == ["F-001"]
    assert corpora.benchmark_tier("tier_a_dev") is None
    assert "items/tier_a/dev/" in corpora.absent_reason("tier_a_dev", set())
    assert corpora.benchmark_tier("tier_k_dev", None) is None
    assert "tier_k_dev.json" in corpora.absent_reason("tier_k_dev", None)


def test_dev_ids_accepts_the_split_file_formats(tmp_path: Path):
    p = tmp_path / "dev.json"
    assert corpora.dev_ids(p) is None
    p.write_text(json.dumps({"dev_ids": ["q_1", "q_2"]}))
    assert corpora.dev_ids(p) == {"q_1", "q_2"}
    p.write_text(json.dumps(["q_3"]))
    assert corpora.dev_ids(p) == {"q_3"}


# --------------------------------------------------------------------------- embedding client ---

def _vec(text: str) -> list[float]:
    return [float(text.count(c)) for c in "aeiourstn"] + [1.0]


def _respond(handler: BaseHTTPRequestHandler, code: int, obj: dict) -> None:
    body = json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class _MockLlama(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        _respond(self, 200 if self.path == "/health" else 404, {"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append({"input": body["input"], "auth": self.headers.get("Authorization")})
        data = [{"index": i, "embedding": _vec(t)} for i, t in enumerate(body["input"])][::-1]   # out of order
        _respond(self, 200, {"data": data, "usage": {"prompt_tokens": 3 * len(body["input"])}})


@pytest.fixture()
def mock_server():
    _MockLlama.calls = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockLlama)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_embed_client_batches_normalises_orders_and_caches(mock_server: str, tmp_path: Path):
    texts = ["alpha  signal", "beta\nsource", "gamma sink", "alpha  signal", "delta"]
    client = embed.EmbedClient(mock_server, cache_dir=tmp_path / "cache", batch=2, api_key="k3y")
    v = client.embed(texts)
    assert v.shape == (5, 10)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose(v[0], v[3])                                    # duplicate text, same vector
    expected = embed.normalise(np.array([_vec(embed.PROMPT + corpora.clean(t)) for t in texts]))
    assert np.allclose(v, expected, atol=1e-6)                        # order restored from "index"
    sent = [s for c in _MockLlama.calls for s in c["input"]]
    assert len(_MockLlama.calls) == 2 and len(sent) == 4              # 4 unique texts, batches of 2
    assert all(s.startswith(embed.PROMPT) for s in sent) and embed.PROMPT + "alpha signal" in sent
    assert all(c["auth"] == "Bearer k3y" for c in _MockLlama.calls)
    assert client.computed == 4 and client.tokens == 12
    # a second client with the same cache sends nothing
    again = embed.EmbedClient(mock_server, cache_dir=tmp_path / "cache", batch=2)
    assert np.allclose(again.embed(texts), v)
    assert len(_MockLlama.calls) == 2 and again.cached == 4
    assert len(list((tmp_path / "cache").rglob("*.f32"))) == 4
    assert again.embed([]).shape[0] == 0


def test_embed_cosine_of_normalised_rows():
    a = embed.normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(a[0], [0.6, 0.8]) and np.all(a[1] == 0)
    assert embed.cosine(a, a)[0, 0] == pytest.approx(1.0)


FAKE_SERVER = """#!{python}
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
with open(os.environ["FAKE_ENV_OUT"], "w") as fh:
    json.dump({{"ld": os.environ.get("LD_LIBRARY_PATH", ""), "argv": sys.argv[1:]}}, fh)
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        self._send({{"status": "ok"}})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send({{"data": [{{"index": i, "embedding": [1.0, float(len(t))]}} for i, t in enumerate(body["input"])]}})
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def _fake_binary(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "llama-server"
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.mark.skipif(os.name != "posix", reason="starts a subprocess server")
def test_server_context_starts_and_always_stops(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FAKE_ENV_OUT", str(tmp_path / "env.json"))
    binary = _fake_binary(tmp_path, FAKE_SERVER.format(python=sys.executable))
    server = embed.LlamaEmbedServer(model=tmp_path / "m.gguf", llama_dir=tmp_path, binary=binary,
                                    cache_dir=None, log_path=tmp_path / "server.log", ready_timeout=30)
    with server as client:
        proc = server.proc
        assert proc is not None and proc.poll() is None
        v = client.embed(["abc", "abcdef"])
        assert v.shape == (2, 2) and np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert proc.poll() is not None and server.proc is None                 # stopped on exit
    seen = json.loads((tmp_path / "env.json").read_text())
    assert seen["ld"].split(":")[0] == str(tmp_path)
    for flag in ("--embeddings", "--pooling", "mean", "--host", "127.0.0.1", "-ub", "2048"):
        assert flag in seen["argv"]
    log = (tmp_path / "server.log").read_text()
    assert "--embeddings" in log and server._api_key not in log             # the key is never logged
    # stopped also when the body raises
    with pytest.raises(RuntimeError):
        with server:
            proc = server.proc
            raise RuntimeError("boom")
    assert proc.poll() is not None


@pytest.mark.skipif(os.name != "posix", reason="starts a subprocess server")
def test_server_that_dies_raises(tmp_path: Path):
    binary = _fake_binary(tmp_path, f"#!{sys.executable}\nimport sys\nsys.exit(3)\n")
    server = embed.LlamaEmbedServer(model=tmp_path / "m.gguf", llama_dir=tmp_path, binary=binary,
                                    cache_dir=None, log_path=tmp_path / "server.log", ready_timeout=30)
    with pytest.raises(embed.EmbedError, match="exited with code 3"):
        with server:
            pass
    assert server.proc is None


# --------------------------------------------------------------------------- audit on toy data --

class _FakeClient:
    def embed(self, texts, progress=None):
        return embed.normalise(np.array([_vec(corpora.clean(t)) for t in texts])) if texts else np.zeros((0, 10))


def _toy():
    tiers = {"tier_k_test": Corpus("tier_k_test", "toy", [
        Doc("q1", "How do I set the sample rate of the signal source block in GNU Radio Companion?"),
        Doc("q2", "What is a polyphase clock sync block used for?")])}
    questions = Corpus("train_questions", "toy", [
        Doc("t1", "Tell me how do I set the sample rate of the signal source block in GNU Radio."),
        Doc("t2", "Explain the Nyquist sampling theorem."),
        Doc("t3", "Unrelated words about cooking pasta.")], embed_all=True)
    filler = " ".join(f"filler{i}" for i in range(150))
    answers = Corpus("teacher_answers", "toy", [
        Doc("m/t1", filler + " The polyphase clock sync block recovers symbol timing. " + filler, ref="t1"),
        Doc("m/t2", "Sampling at twice the bandwidth avoids aliasing.", ref="t2")])
    return tiers, questions, answers


def test_audit_scans_flags_and_reports_toy_corpora():
    tiers, questions, answers = _toy()
    index = ngram.NgramIndex([d.text for d in tiers["tier_k_test"].docs])
    runs = [audit.scan_corpus(questions, tiers, index, embed_topk=2),
            audit.scan_corpus(answers, tiers, index, embed_topk=1)]
    q_pairs, a_pairs = runs[0].pairs["tier_k_test"], runs[1].pairs["tier_k_test"]
    assert q_pairs[0].ngram_shared > 0 and q_pairs[1].ngram_shared == 0
    assert q_pairs[0].tfidf_top[0][0] == 0 and q_pairs[0].tfidf_max >= audit.TFIDF_FLAG
    assert runs[0].candidates["tier_k_test"] == [0, 1, 2]                 # short corpus: every passage
    assert runs[1].n_passages > 2 and len(runs[1].candidates["tier_k_test"]) <= 2
    assert a_pairs[1].tfidf_top[0][0] == 0                                # polyphase passage inside the long answer

    report, items = audit.tier_report("tier_k_test", tiers["tier_k_test"], runs, "skipped (--skip-embed)")
    assert report["corpora"]["train_questions"]["ngram"]["items_with_twin"] == 1
    assert report["corpora"]["train_questions"]["embedding"]["rate"] is None
    assert items[0]["flags"][:1] == ["train_questions:ngram"]

    audit.embed_runs(_FakeClient(), tiers, runs)
    assert all(p.embed_max is not None for r in runs for p in r.pairs["tier_k_test"])
    report, items = audit.tier_report("tier_k_test", tiers["tier_k_test"], runs, "done")
    e = report["corpora"]["train_questions"]["embedding"]
    assert e["status"] == "done" and e["passages_embedded"] == 3 and e["max_cosine_quantiles"]["max"] <= 1.0
    rows = audit.review_rows("tier_k_test", tiers["tier_k_test"], runs)
    q1 = [r for r in rows if r["item_id"] == "q1" and r["corpus"] == "train_questions"]
    assert q1 and q1[0]["doc_id"] == "t1" and "ngram" in q1[0]["signals"] and q1[0]["ngram_shared"] > 0
    assert report["flagged_items"]["q1"][0] == "train_questions:ngram"
    assert json.loads(json.dumps(report)) == report                       # JSON-serialisable


def test_quantiles_and_rates():
    assert audit.quantiles([]) is None
    q = audit.quantiles([0.0, 0.5, 1.0])
    assert q["min"] == 0.0 and q["p50"] == 0.5 and q["max"] == 1.0
    assert audit.rate(1, 3) == 0.3333 and audit.rate(0, 0) == 0.0


def test_sft_session_text_includes_system_prompt_and_tool_calls():
    row = {"system": "You are the execution agent.",
           "messages": [{"role": "user", "content": "Build a GMSK chain."},
                        {"role": "assistant", "content": "", "thinking": "plan it",
                         "tool_calls": [{"function": {"name": "add_block", "arguments": "{\"id\": \"gmsk_mod\"}"}}]},
                        {"role": "tool", "content": "ok"}]}
    text = corpora.session_text(row)
    assert text.split("\n\n") == ["You are the execution agent.", "Build a GMSK chain.", "plan it",
                                  "add_block {\"id\": \"gmsk_mod\"}", "ok"]
    assert corpora.session_text({"messages": [{"role": "user", "content": "hi"}]}) == "hi"


def test_empty_corpus_is_reported_without_errors():
    tiers, _, _ = _toy()
    index = ngram.NgramIndex([d.text for d in tiers["tier_k_test"].docs])
    run = audit.scan_corpus(Corpus("task_specs", "empty", [], embed_all=True), tiers, index, embed_topk=3)
    report, items = audit.tier_report("tier_k_test", tiers["tier_k_test"], [run], "skipped (--skip-embed)")
    e = report["corpora"]["task_specs"]
    assert e["n_docs"] == 0 and e["top_pairs"] == [] and e["any_signal"]["items_flagged"] == 0
    assert audit.review_rows("tier_k_test", tiers["tier_k_test"], [run]) == []


def test_review_sheet_keeps_team_decisions_across_reruns(tmp_path: Path):
    path = tmp_path / "contamination_tier_k_test.csv"
    row = {"tier": "tier_k_test", "item_id": "q1", "corpus": "train_questions", "doc_id": "t1", "doc_ref": "",
           "signals": "ngram", "ngram_shared": 3, "tfidf": 0.7, "embed": None, "item_text": "a", "training_text": "b"}
    other = {**row, "item_id": "q2", "doc_id": "t9"}
    assert audit.write_sheet("tier_k_test", [row, other], path) == 0
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["decision"] for r in rows] == ["", ""]
    rows[0].update(decision="template twin", reviewer="JR", notes="different signal")   # the team fills in
    rows[1].update(decision="same question", reviewer="JP")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=audit.CSV_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    # re-run: q1 still flagged (new scores), q2 no longer flagged
    assert audit.write_sheet("tier_k_test", [{**row, "tfidf": 0.71}], path) == 2
    with open(path, encoding="utf-8", newline="") as fh:
        again = list(csv.DictReader(fh))
    assert [(r["item_id"], r["tfidf"], r["decision"], r["reviewer"], r["notes"]) for r in again] == [
        ("q1", "0.71", "template twin", "JR", "different signal"), ("q2", "0.7", "same question", "JP", "")]
    assert again[1]["signals"] == audit.NOT_FLAGGED


def test_cli_handles_absent_tiers_and_rejects_unknown_names(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(corpora, "ITEMS", tmp_path / "items")                 # no Tier-F/A items
    for name, value in (("OUT", tmp_path / "out"), ("SUMMARY", tmp_path / "contamination.md"),
                        ("CALIBRATION", tmp_path / "out" / "calibration.json"), ("SHEETS", tmp_path / "sheets")):
        monkeypatch.setattr(audit, name, value)
    assert audit.main(["--only", "tier_a_test,tier_f_dev", "--skip-embed"]) == 0
    out = capsys.readouterr().out
    assert "tier_a_test: absent (no items under items/tier_a/test/ yet)" in out
    md = (tmp_path / "contamination.md").read_text(encoding="utf-8")
    assert "| tier_a_test | – | not audited: no items under items/tier_a/test/ yet |" in md
    assert "nothing is removed" in md.lower() or "Nothing is removed" in md
    assert not (tmp_path / "out").exists() and not (tmp_path / "sheets").exists()
    with pytest.raises(SystemExit):
        audit.main(["--only", "tier_z_test", "--skip-embed"])

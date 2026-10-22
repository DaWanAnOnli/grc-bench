"""B1 baseline "base + documentation search": GRC-Agent's own retrieval, added to the prompt.

Implements B1 as the evaluation plan defines it (Table 1, "Baselines B1 and B2": "B1 (documentation
search) runs on Tiers K and F, with the retrieved pages added to the prompt"; Table 6, "B1 +
documentation search") and the proposal does (§6, "Models, Baselines, and Post-Training": "B1,
base + GRC-Agent retrieval (query_knowledge)"), with the build plan's Phase 5 rule ("GRC-Agent's
lexical BM25 database (a copy), with retrieved pages added to the prompt") and the 16 Oct Tier-F
decision ("B1 for Tier F is redefined as oracle grounding: catalog entries for every reference
block").

One retriever for B1 and Tier A
  Retrieval is GRC-Agent's own: `grc_agent.adapter.rag.query_catalog` / `query_docs` at the pinned
  commit, the two engines its `query_knowledge` tool dispatches to (`engine(query, k)`), called
  unchanged. Nothing is re-implemented here: query normalisation (whitespace collapsed; the first 32
  distinct word tokens, quoted and OR-joined), the BM25 ranking, the over-fetch and the rendering of
  catalog hits all come from GRC-Agent. `tool_result` on a SearchResult is the exact string the
  tool would return to a model (`json.dumps` of the engine's answer).
  The calls run in the environment of a Tier-A episode: a fresh run directory under
  ~/grc-bench-work/b1 prepared by `tier_a.isolation.make_run_env` (GRC_EMBED_BACKEND=lexical,
  GRC_AGENT_VECTORS_DIR = the read-only copies of the lexical databases in
  ~/grc-bench-cache/grc-agent-vectors, checked by SHA-256 against the pinned originals; HOME with the
  warm GRC cache; an empty GRC_HIER_PATH, so the catalog is GNU Radio's own), and in one process
  inside bubblewrap where everything except that run directory is read-only and there is no network.
  GRC-Agent therefore cannot write to ~/grc-agent or to the database copies. If its freshness check
  asks for a rebuild (for example because the machine's GNU Radio block set differs from the one
  the pinned database was built from), the rebuild hits the read-only file system, GRC-Agent answers
  ok=False ("Catalog DB build failed: ..."), and the search raises RetrieverError: a failed search
  never becomes a B1 block with fewer passages, and the CLI exits 1. The one soft failure is
  GRC-Agent's own "query must be non-empty" answer to a query with no words, which `search` returns
  as ok=False (the tool raises ModelRetry there); `b1_contexts` refuses such a question outright.
  A caller that is itself an isolated episode process (GRC_BENCH_RUN_DIR set and
  `check_process_isolation` passing) runs the calls in-process.

Ids and scores
  catalog passage id  "catalog:<block_id>";
  docs passage id     "docs:<page>#<rowid>" (rowid of the chunk in the pinned docs database; two
                      (page, heading) pairs occur twice, so page and heading alone are not unique).
  GRC-Agent's lexical mode returns a rank, not a score. `score` is SQLite FTS5's bm25() for the same
  MATCH expression (GRC-Agent's `_fts_query_string`) on the same database, negated so that higher
  means more relevant. Scores must not increase along GRC-Agent's ranking; this is checked, which
  also confirms that the ranking is BM25's.

B1 context for Tier K (`b1_context`)
  The question is the query, in both domains, with the same k (default 5, GRC-Agent's default;
  1-20 as in its schema). The block is placed before the Tier-K prompt (`add_to_prompt`):

      <retrieved_documentation>
      HEADER (B1_HEADER)

      [1] Block catalog entry: <block_id>
      <the entry exactly as query_knowledge returns it, as JSON>

      [2] Documentation page: <page>
      <the chunk exactly as query_knowledge returns it: "path: ...", "heading: ...", body>
      ...
      </retrieved_documentation>

  Passages are taken in interleaved rank order (catalog 1, docs 1, catalog 2, docs 2, ...). The
  whole block is capped at B1_TOKEN_CAP = 4096 cl100k tokens (`common.tokens.count`), the cap on one
  tool output in every Tier-A run (eval plan §4.4 and Table 1, "Tool outputs during evaluation"),
  so B1 never adds more retrieved text than one capped `query_knowledge` result can.
  * A passage is added whole if it fits in what is left of the budget.
  * A domain's top-ranked passage (rank 1) that does not fit whole is shortened instead of skipped,
    with the Tier-A cap policy (`common.tokens.truncate_tool_output`: beginning and end kept, error
    lines kept, its truncation marker), to what is left but to at most half of the passage budget
    (cap minus the empty block), so one oversized page can neither be withheld (a Tier-A agent
    would see the start of it in its capped tool output) nor crowd out the rest of the ranking. It
    is skipped if fewer than MIN_EXCERPT_TOKENS would remain. Its id is listed in `shortened`.
    At the default cap this only affects docs pages: every catalog entry (at most 1,754 tokens)
    fits whole as the first passage.
  * Any other passage that does not fit is skipped; later, shorter passages are still tried.
  With no passage at all, the body is NO_PASSAGES when nothing matched and NO_FIT when passages
  matched but none fitted the cap.

Oracle grounding for Tier F (`oracle_grounding`, `oracle_grounding_context`)
  The catalog entry of each given block id, rendered by GRC-Agent's `render_catalog_block`, the same
  renderer as a catalog hit, so an entry is identical to what `query_knowledge` would return for that
  block. Entries are de-duplicated and sorted by block id (the order never reveals a reference
  chain). An id GRC's catalog does not know is an error. The context block is not capped: every
  reference block must be present. `reference_block_ids` reads "every reference block" off a
  reference .grc: every enabled entry of its `blocks` list, variables included.

Template status: the wording of B1_HEADER, GROUNDING_HEADER, NO_PASSAGES, NO_FIT and the entry
labels is a prompt template drafted by Claude (Anthropic), pending team review (14 Oct authorship
decision); the content of every passage is GRC-Agent's.

    ./gb retriever <query ...> [--domain catalog|docs|both] [-k 5] [--json]
    ./gb retriever --b1 <question ...> [-k 5] [--cap 4096]
    ./gb retriever --grounding <block_id | reference.grc> [...]
    ./gb retriever --items <items.jsonl> [--out <contexts.jsonl>] [-k 5] [--cap 4096]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..common import tokens
from ..common.io import iter_jsonl, read_json, write_json, write_jsonl
from ..common.paths import WORK

DOMAINS = ("catalog", "docs")
K_DEFAULT = 5                    # query_knowledge's default k
K_MIN, K_MAX = 1, 20             # GRC-Agent's ResultCount: Field(ge=1, le=20)
B1_TOKEN_CAP = tokens.TOOL_OUTPUT_CAP
RUNS_ROOT = WORK / "b1"
REQUEST_FILE = "b1_request.json"
RESULT_FILE = "b1_result.json"
SCORE_DECIMALS = 6

OPEN_TAG, CLOSE_TAG = "<retrieved_documentation>", "</retrieved_documentation>"
B1_HEADER = ("The passages below come from a keyword search of GRC-Agent's GNU Radio knowledge base "
             "(block catalog and documentation pages), with the question as the query. They may be "
             "incomplete or irrelevant.")
GROUNDING_HEADER = "Block catalog entries from GRC-Agent's GNU Radio knowledge base."
NO_PASSAGES = "No passages matched the query."
NO_FIT = "The retrieved passages did not fit in the token budget."
MIN_EXCERPT_TOKENS = 256         # a shortened top hit keeps at least this many tokens, or is skipped
LABELS = {"catalog": "Block catalog entry", "docs": "Documentation page"}


class RetrieverError(RuntimeError):
    """The retrieval could not be run, or GRC-Agent's answer broke an invariant checked here."""


# --------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Passage:
    """One ranked hit. `result` is the object GRC-Agent returned for it (catalog: the rendered
    entry; docs: {"text": chunk}); `text` is what B1 puts into a prompt."""

    id: str
    domain: str
    rank: int
    score: float
    title: str
    text: str
    result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "domain": self.domain, "rank": self.rank, "score": self.score,
                "title": self.title, "text": self.text, "result": self.result}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Passage":
        return cls(d["id"], d["domain"], int(d["rank"]), float(d["score"]), d["title"], d["text"],
                   d["result"])


@dataclass(frozen=True)
class SearchResult:
    """One query_knowledge-equivalent search. ok=False mirrors the tool's ModelRetry case."""

    query: str
    domain: str
    k: int
    ok: bool
    passages: tuple[Passage, ...]
    search_mode: str | None
    notices: tuple[str, ...]
    output_truncated: bool
    tool_result: str

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "domain": self.domain, "k": self.k, "ok": self.ok,
                "passages": [p.to_dict() for p in self.passages], "search_mode": self.search_mode,
                "notices": list(self.notices), "output_truncated": self.output_truncated,
                "tool_result": self.tool_result}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SearchResult":
        return cls(d["query"], d["domain"], int(d["k"]), bool(d["ok"]),
                   tuple(Passage.from_dict(p) for p in d["passages"]), d.get("search_mode"),
                   tuple(d.get("notices") or ()), bool(d.get("output_truncated")), d["tool_result"])


@dataclass(frozen=True)
class B1Context:
    """The B1 block for one question and what went into it (`shortened` is a subset of `included`:
    the top hits that are in the block only in shortened form)."""

    text: str
    tokens: int
    token_cap: int
    k: int
    included: tuple[str, ...]
    shortened: tuple[str, ...]
    skipped: tuple[str, ...]
    notices: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "tokens": self.tokens, "token_cap": self.token_cap, "k": self.k,
                "included": list(self.included), "shortened": list(self.shortened),
                "skipped": list(self.skipped), "notices": list(self.notices)}


# --------------------------------------------------------------------------------------------
# Pure helpers (no GNU Radio, no GRC-Agent)
# --------------------------------------------------------------------------------------------

def check_domain(domain: str) -> str:
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")
    return domain


def check_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or not K_MIN <= k <= K_MAX:
        raise ValueError(f"k must be an integer in [{K_MIN}, {K_MAX}] (GRC-Agent's ResultCount), got {k!r}")
    return k


def check_cap(token_cap: int) -> int:
    floor = max(tokens.count(_wrap(B1_HEADER, [], empty)) for empty in (NO_PASSAGES, NO_FIT))
    if isinstance(token_cap, bool) or not isinstance(token_cap, int) or token_cap < floor:
        raise ValueError(f"token_cap must be an integer of at least {floor} (the empty B1 block), "
                         f"got {token_cap!r}")
    return token_cap


def check_question(question: str) -> str:
    if not " ".join(str(question).split()):      # GRC-Agent's normalisation; it rejects an empty query
        raise ValueError("a B1 question must contain at least one non-blank character")
    return question


def interleave(results: Sequence[SearchResult]) -> list[Passage]:
    """Passages in interleaved rank order: rank 1 of each result (in the given order), then rank 2..."""
    out: list[Passage] = []
    depth = max((len(r.passages) for r in results), default=0)
    for i in range(depth):
        out.extend(r.passages[i] for r in results if i < len(r.passages))
    return out


def _entry_text(label: str, title: str, text: str, number: int) -> str:
    return f"[{number}] {label}: {title}\n{text}"


def _wrap(header: str, entries: Sequence[str], empty: str) -> str:
    body = "\n\n".join(entries) if entries else empty
    return f"{OPEN_TAG}\n{header}\n\n{body}\n{CLOSE_TAG}"


def _fits(entries: Sequence[str], token_cap: int) -> bool:
    return tokens.count(_wrap(B1_HEADER, entries, NO_PASSAGES)) <= token_cap


def _shortened_entry(entries: Sequence[str], passage: Passage, number: int, token_cap: int,
                     limit: int) -> str | None:
    """The entry of `passage` with its text shortened by the Tier-A cap policy
    (`tokens.truncate_tool_output`) to at most `limit` tokens and to what is left of `token_cap`;
    None if fewer than MIN_EXCERPT_TOKENS would remain."""
    label = LABELS[passage.domain]
    bare = _entry_text(label, passage.title, "", number)
    n = min(limit, token_cap - tokens.count(_wrap(B1_HEADER, [*entries, bare], NO_PASSAGES)))
    while n >= MIN_EXCERPT_TOKENS:
        entry = _entry_text(label, passage.title, tokens.truncate_tool_output(passage.text, n), number)
        if _fits([*entries, entry], token_cap):
            return entry
        n -= 8                            # the joins can re-tokenise by a token or two
    return None


def build_b1_context(results: Sequence[SearchResult], *, token_cap: int = B1_TOKEN_CAP) -> B1Context:
    """The B1 block from search results (normally [catalog, docs] for the same question).

    Passages are taken in interleaved rank order; each is added whole if the block (header, tags
    and all) still fits in `token_cap` cl100k tokens. A domain's top hit that does not fit is
    shortened (see the module docstring); any other passage that does not fit is skipped. A failed
    search is an error, never a block with fewer passages.
    """
    check_cap(token_cap)
    failed = [r for r in results if not r.ok]
    if failed:
        raise RetrieverError(f"{failed[0].domain}: a B1 block is never built from a failed search: "
                             f"{failed[0].tool_result}")
    ks = {r.k for r in results}           # one k for the whole block (0 if the results differ)
    top_limit = (token_cap - tokens.count(_wrap(B1_HEADER, [], NO_PASSAGES))) // 2
    entries: list[str] = []
    included: list[str] = []
    shortened: list[str] = []
    skipped: list[str] = []
    for passage in interleave(results):
        number = len(entries) + 1
        entry: str | None = _entry_text(LABELS[passage.domain], passage.title, passage.text, number)
        if not _fits([*entries, entry], token_cap):
            entry = (_shortened_entry(entries, passage, number, token_cap, top_limit)
                     if passage.rank == 1 else None)
            if entry is None:
                skipped.append(passage.id)
                continue
            shortened.append(passage.id)
        entries.append(entry)
        included.append(passage.id)
    text = _wrap(B1_HEADER, entries, NO_FIT if skipped else NO_PASSAGES)
    notices = tuple(f"{r.domain}: {n}" for r in results for n in r.notices)
    return B1Context(text=text, tokens=tokens.count(text), token_cap=token_cap,
                     k=ks.pop() if len(ks) == 1 else 0, included=tuple(included),
                     shortened=tuple(shortened), skipped=tuple(skipped), notices=notices)


def reference_block_ids(graph: dict[str, Any]) -> list[str]:
    """Distinct block ids of a flowgraph (a .grc file loaded as YAML), sorted.

    "Every reference block" (16 Oct decision) is every entry of the reference graph's `blocks` list,
    variables included; the `options` section is not a block a model adds. A block whose state is
    "disabled" takes no part in the solution and is left out.
    """
    ids = set()
    for blk in graph.get("blocks") or []:
        if str((blk.get("states") or {}).get("state", "enabled")) != "disabled":
            ids.add(str(blk["id"]))
    return sorted(ids)


def grounding_context_text(entries: Sequence[dict[str, Any]]) -> str:
    """The Tier-F oracle-grounding block for already rendered entries (see the module docstring)."""
    parts = [_entry_text(LABELS["catalog"], e["block_id"], json.dumps(e), i)
             for i, e in enumerate(entries, start=1)]
    return _wrap(GROUNDING_HEADER, parts, "No blocks.")


def add_to_prompt(prompt: str, block: str) -> str:
    """The B1 prompt: the retrieved block, a blank line, then the unchanged B0 prompt."""
    return f"{block}\n\n{prompt}"


# --------------------------------------------------------------------------------------------
# Public API (each call is one batch; a batch runs in one isolated process)
# --------------------------------------------------------------------------------------------

def search_many(requests: Iterable[tuple[str, str, int]]) -> list[SearchResult]:
    """GRC-Agent's query_knowledge search for each (query, domain, k), in one isolated process."""
    ops = [{"op": "search", "query": str(q), "domain": check_domain(d), "k": check_k(k)}
           for q, d, k in requests]
    return [SearchResult.from_dict(r) for r in _run_ops(ops)]


def search(query: str, domain: str, k: int = K_DEFAULT) -> SearchResult:
    return search_many([(query, domain, k)])[0]


def retrieve(query: str, domain: str, k: int = K_DEFAULT) -> list[Passage]:
    """Ranked passages (ids, BM25 scores, text) for `query` in `domain` ("catalog" or "docs")."""
    return list(search(query, domain, k).passages)


def b1_contexts(questions: Sequence[str], k: int = K_DEFAULT, *,
                token_cap: int = B1_TOKEN_CAP) -> list[B1Context]:
    """B1 blocks for many questions, all searched in one isolated process. Raises RetrieverError
    if any search fails (nothing is returned for the other questions either)."""
    check_k(k)
    check_cap(token_cap)
    for q in questions:
        check_question(q)
    results = search_many((q, d, k) for q in questions for d in DOMAINS)
    n = len(DOMAINS)
    return [build_b1_context(results[i * n:(i + 1) * n], token_cap=token_cap)
            for i in range(len(questions))]


def b1_context(question: str, k: int = K_DEFAULT, *, token_cap: int = B1_TOKEN_CAP) -> str:
    """The text block B1 adds to a Tier-K prompt for `question` (format in the module docstring)."""
    return b1_contexts([question], k, token_cap=token_cap)[0].text


def oracle_grounding(block_ids: Iterable[str]) -> list[dict[str, Any]]:
    """GRC-Agent's catalog entries for the given block ids, de-duplicated and sorted by id."""
    ids = sorted({str(b) for b in block_ids})
    if not ids:
        return []
    rows = _run_ops([{"op": "entry", "block_id": b} for b in ids])
    unknown = [r["block_id"] for r in rows if r["entry"] is None]
    if unknown:
        raise RetrieverError(f"not in GRC's block catalog: {', '.join(unknown)}")
    return [r["entry"] for r in rows]


def oracle_grounding_context(block_ids: Iterable[str]) -> str:
    """The Tier-F B1 block: the catalog entry of every given (reference) block, uncapped."""
    return grounding_context_text(oracle_grounding(block_ids))


# --------------------------------------------------------------------------------------------
# Running GRC-Agent: isolated worker process, or in-process inside an episode
# --------------------------------------------------------------------------------------------

def _inside_episode() -> bool:
    if not os.environ.get("GRC_BENCH_RUN_DIR"):
        return False
    from ..tier_a import isolation
    try:
        isolation.check_process_isolation()
    except isolation.IsolationError:
        return False
    return True


def _run_ops(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ops:
        return []
    if _inside_episode():
        return _execute_ops(ops)
    return _run_worker(ops)


def _make_env(run_dir: Path) -> dict[str, str]:
    from ..tier_a import isolation
    for attempt in range(5):
        try:
            return isolation.make_run_env(run_dir)
        except PermissionError:
            # ensure_vectors_cache rewrites MANIFEST.json (0444) on every call; a concurrent
            # Tier-A run can hold it writable for an instant. Retry, then give up loudly.
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))
    raise AssertionError("unreachable")


def _run_worker(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from ..tier_a import isolation
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = isolation.check_run_dir(RUNS_ROOT / f"run-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    run_dir.mkdir(parents=True)
    try:
        env = _make_env(run_dir)
        write_json(run_dir / REQUEST_FILE, {"ops": ops})
        res = isolation.run_episode_process(
            run_dir, "grcbench.tier_k.retriever", ["--worker", str(run_dir)], env,
            timeout=600.0 + 5.0 * len(ops), network="none", cpu_seconds=3600 + 5 * len(ops))
        out = run_dir / RESULT_FILE
        if res.returncode != 0 or not out.is_file():
            tail = "\n".join((res.stderr or "").strip().splitlines()[-25:])
            raise RetrieverError(f"retrieval worker failed (exit {res.returncode}, timed out "
                                 f"{res.timed_out}):\n{tail}")
        payload = read_json(out)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    if payload.get("error"):
        raise RetrieverError(payload["error"])
    results = payload["results"]
    if len(results) != len(ops):
        raise RetrieverError(f"worker returned {len(results)} results for {len(ops)} requests")
    return results


def _execute_ops(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the requests with GRC-Agent in this (isolated) process."""
    from grc_agent.adapter import rag
    out = []
    for op in ops:
        if op["op"] == "search":
            out.append(_search_in_process(rag, op["query"], op["domain"], op["k"]))
        elif op["op"] == "entry":
            out.append({"block_id": op["block_id"], "entry": rag.render_catalog_block(op["block_id"])})
        else:
            raise RetrieverError(f"unknown op {op['op']!r}")
    return out


def _search_in_process(rag: Any, query: str, domain: str, k: int) -> dict[str, Any]:
    check_domain(domain)
    check_k(k)
    engine = rag.query_catalog if domain == "catalog" else rag.query_docs   # as query_knowledge_func
    res = engine(query, k)
    tool_result = json.dumps(res)            # the exact string query_knowledge returns
    base = {"query": query, "domain": domain, "k": k, "tool_result": tool_result,
            "search_mode": res.get("search_mode"), "notices": list(res.get("notices") or []),
            "output_truncated": bool(res.get("output_truncated"))}
    if not res.get("ok", True):
        if " ".join(str(query).split()):
            # A database GRC-Agent could not open or wanted to rebuild ("... DB build failed: ..."
            # on the read-only copy): fail loudly, never a B1 block with fewer passages.
            raise RetrieverError(f"{domain}: GRC-Agent's search failed: {res.get('message')}")
        # GRC-Agent's "query must be non-empty" (query_knowledge raises ModelRetry): kept as ok=False.
        return {**base, "ok": False, "passages": []}
    db_path, model = rag.get_db_and_model(domain)
    if model is not None or res.get("search_mode") != "lexical":
        raise RetrieverError(f"B1 uses GRC-Agent's lexical (BM25) databases; got backend model "
                             f"{model!r}, search mode {res.get('search_mode')!r}")
    hits = list(res.get("results") or [])
    if len(hits) > k:
        raise RetrieverError(f"{domain}: GRC-Agent returned {len(hits)} results for k={k}")
    q = " ".join(str(query).split())
    fts = rag._fts_query_string(q)
    passages = _score_hits(db_path, domain, fts[0] if fts else None, hits)
    return {**base, "ok": True, "passages": [p.to_dict() for p in passages]}


def _score_hits(db_path: str, domain: str, match: str | None,
                hits: list[dict[str, Any]]) -> list[Passage]:
    """Attach ids and BM25 scores to GRC-Agent's ranked hits (read-only database connection)."""
    if not hits:
        return []
    if match is None:
        raise RetrieverError(f"{domain}: hits without a MATCH expression")
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        if domain == "catalog":
            keys = [h["block_id"] for h in hits]
            rows = conn.execute(
                f"SELECT rowid, block_id FROM catalog_chunks WHERE block_id IN ({_marks(keys)})",
                keys).fetchall()
            rowid_of = _unique({r[1]: r[0] for r in rows}, keys, domain)
            meta = {key: {} for key in keys}
        else:
            keys = [h["text"] for h in hits]
            rows = conn.execute(
                f"SELECT rowid, path, heading, payload FROM docs_chunks WHERE payload IN ({_marks(keys)})",
                keys).fetchall()
            rowid_of = _unique({r[3]: r[0] for r in rows}, keys, domain)
            meta = {r[3]: {"page": r[1], "heading": r[2]} for r in rows}
        rowids = [rowid_of[key] for key in keys]
        table = f"{domain}_fts"
        bm25 = dict(conn.execute(
            f"SELECT rowid, bm25({table}) FROM {table} WHERE {table} MATCH ? AND rowid IN ({_marks(rowids)})",
            [match, *rowids]).fetchall())
    finally:
        conn.close()
    passages = []
    for rank, (hit, key, rowid) in enumerate(zip(hits, keys, rowids), start=1):
        if rowid not in bm25:
            raise RetrieverError(f"{domain}: hit {rank} (rowid {rowid}) does not match the query")
        score = round(-float(bm25[rowid]), SCORE_DECIMALS)
        if domain == "catalog":
            pid, title, text = f"catalog:{key}", key, json.dumps(hit)
        else:
            m = meta[key]
            pid, title, text = f"docs:{m['page']}#{rowid}", m["page"], key
        passages.append(Passage(pid, domain, rank, score, title, text, hit))
    for a, b in zip(passages, passages[1:]):
        if b.score > a.score + 1e-9:
            raise RetrieverError(f"{domain}: GRC-Agent's ranking is not BM25 order "
                                 f"({a.id} {a.score} before {b.id} {b.score})")
    return passages


def _marks(values: Sequence[Any]) -> str:
    return ",".join("?" for _ in values)


def _unique(found: dict[str, int], keys: list[str], domain: str) -> dict[str, int]:
    missing = [k[:60] for k in keys if k not in found]
    if missing:
        raise RetrieverError(f"{domain}: hits not found in the database: {missing}")
    if len(set(keys)) != len(keys):
        raise RetrieverError(f"{domain}: duplicate hits in one result")
    return found


def worker_main(run_dir: str) -> int:
    """Entry point inside the sandbox: read the request, run it, write the result file."""
    sys.modules["gi"] = None          # headless, as every Tier-A episode: GTK imports now fail
    from ..tier_a import isolation
    lay = isolation.check_process_isolation(Path(run_dir))
    request = read_json(lay.run_dir / REQUEST_FILE)
    try:
        payload: dict[str, Any] = {"results": _execute_ops(request["ops"])}
    except RetrieverError as exc:
        payload = {"error": str(exc)}
    # Key order is kept (no sort_keys): a catalog entry's key and parameter order is GRC-Agent's,
    # and json.dumps of the entry must stay byte-identical to the tool's output.
    (lay.run_dir / RESULT_FILE).write_text(json.dumps(payload, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    return 0


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def _print_search(r: SearchResult) -> None:
    status = "" if r.ok else "  FAILED"
    print(f"{r.domain} (search mode {r.search_mode}, k={r.k}): {len(r.passages)} results{status}")
    for p in r.passages:
        extra = ""
        if p.domain == "catalog":
            extra = str(p.result.get("label", ""))
        else:
            heading = p.text.split("\n", 2)[1] if p.text.count("\n") >= 1 else ""
            extra = heading[len("heading: "):] if heading.startswith("heading: ") else ""
        print(f"  {p.rank:2d}  {p.score:10.4f}  {p.id:45s} {extra}")
    for n in r.notices:
        print(f"  notice: {n}")
    if not r.ok:
        print(f"  {r.tool_result}")


def _items_main(items: Path, out: Path | None, k: int, cap: int) -> int:
    from ..tier_a.isolation import FORBIDDEN_ROOTS, is_inside
    records = [r for r in iter_jsonl(items) if r.get("question")]
    out = out or RUNS_ROOT / "contexts" / f"{items.parent.name}_{items.stem}_k{k}_cap{cap}.jsonl"
    if any(is_inside(out, root) for root in FORBIDDEN_ROOTS):
        print(f"refusing to write into a read-only input: {out}")
        return 2
    t0 = time.monotonic()
    ctxs = b1_contexts([r["question"] for r in records], k, token_cap=cap)
    rows = [{"id": r["id"], **c.to_dict()} for r, c in zip(records, ctxs)]
    write_jsonl(out, rows)
    toks = sorted(c.tokens for c in ctxs)
    n = len(ctxs)
    if n:
        print(f"{n} B1 contexts (k={k}, cap {cap}) in {time.monotonic() - t0:.1f} s -> {out}")
        print(f"tokens: min {toks[0]}, median {toks[n // 2]}, max {toks[-1]}; "
              f"passages included per question: mean {sum(len(c.included) for c in ctxs) / n:.2f}; "
              f"questions with a shortened top hit: {sum(1 for c in ctxs if c.shortened)}; "
              f"with a skipped passage: {sum(1 for c in ctxs if c.skipped)}; "
              f"with a notice: {sum(1 for c in ctxs if c.notices)}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb retriever", description=__doc__.splitlines()[0])
    ap.add_argument("words", nargs="*", help="the query (or, with --grounding, block ids)")
    ap.add_argument("--domain", choices=[*DOMAINS, "both"], default="both")
    ap.add_argument("-k", type=int, default=K_DEFAULT, help=f"results per domain ({K_MIN}-{K_MAX})")
    ap.add_argument("--cap", type=int, default=B1_TOKEN_CAP, help="token cap of a B1 block")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--b1", action="store_true", help="print the B1 block for the words as question")
    mode.add_argument("--grounding", action="store_true", help="print the catalog entries of block ids")
    mode.add_argument("--items", type=Path, help="write B1 blocks for every question of a JSONL file")
    mode.add_argument("--worker", metavar="RUN_DIR", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, help="output of --items")
    a = ap.parse_args(argv)
    if a.worker:
        return worker_main(a.worker)
    try:
        check_k(a.k)
        check_cap(a.cap)
    except ValueError as exc:
        ap.error(str(exc))
    try:
        return _cli(ap, a)
    except (RetrieverError, ValueError) as exc:
        print(f"gb retriever: error: {exc}", file=sys.stderr)
        return 1


def _cli(ap: argparse.ArgumentParser, a: argparse.Namespace) -> int:
    if a.items:
        return _items_main(a.items, a.out, a.k, a.cap)
    if not a.words:
        ap.error("give a query (or block ids with --grounding)")
    if a.grounding:
        ids: list[str] = []
        for word in a.words:          # a .grc path stands for every block of that flowgraph
            if word.endswith(".grc") and Path(word).is_file():
                import yaml
                ids += reference_block_ids(yaml.safe_load(Path(word).read_text(encoding="utf-8")))
            else:
                ids.append(word)
        entries = oracle_grounding(ids)
        text = grounding_context_text(entries)
        print(json.dumps(entries, indent=1) if a.json else text)
        if not a.json:
            print(f"\n[{len(entries)} entries, {tokens.count(text)} tokens]")
        return 0
    query = " ".join(a.words)
    if a.b1:
        ctx = b1_contexts([query], a.k, token_cap=a.cap)[0]
        if a.json:
            print(json.dumps(ctx.to_dict(), indent=1, ensure_ascii=False))
        else:
            print(ctx.text)
            print(f"\n[{ctx.tokens} tokens (cap {ctx.token_cap}); included {len(ctx.included)}: "
                  f"{', '.join(ctx.included)}; shortened {len(ctx.shortened)}"
                  + (f": {', '.join(ctx.shortened)}" if ctx.shortened else "")
                  + f"; skipped {len(ctx.skipped)}"
                  + (f": {', '.join(ctx.skipped)}" if ctx.skipped else "") + "]")
            for n in ctx.notices:
                print(f"[notice] {n}")
        return 0
    domains = DOMAINS if a.domain == "both" else (a.domain,)
    results = search_many((query, d, a.k) for d in domains)
    if a.json:
        print(json.dumps([r.to_dict() for r in results], indent=1, ensure_ascii=False))
    else:
        for r in results:
            _print_search(r)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

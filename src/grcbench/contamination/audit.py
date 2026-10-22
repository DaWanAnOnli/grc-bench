"""`./gb contamination` — 8-gram, TF-IDF and embedding overlap audit (eval plan §6; proposal §4.5).

Every benchmark tier (corpora.TIERS) is compared with every training corpus (corpora.CORPORA) by
three signals. Each is computed per item as its strongest match anywhere in the corpus:
  ngram  the item shares >= 1 word 8-gram with a corpus document (eval plan §6 "8-word overlap",
         the definition behind its count of 26 near-twins). Coverage >= 50% of the item's 8-grams
         is reported as well.
  tfidf  max TF-IDF cosine to any corpus passage >= TFIDF_FLAG. LEXICAL: it is not paraphrase
         detection.
  embed  max EmbeddingGemma cosine >= EMBED_FLAG (semantic). Short corpora (training questions,
         task specs) are embedded completely. For the long corpora (answers, repairs, sessions; ~15M
         words, far beyond what a CPU embeds in a session) only the best passage of each item's
         EMBED_TOPK nearest TF-IDF documents is embedded, so there the maximum is a lower bound and
         a paraphrase with little word overlap can be missed.
The contamination rate of a tier (eval plan §6) is reported as the share of its items flagged by
each signal, per corpus and over all corpora. Flags are candidates for human review, never
removals (dataset README: similarity scores never remove questions); the team judges them in
handoff/review_sheets/contamination_<tier>.csv. The LLM rephrase detector of eval plan §6 [54] waits
for API access and is not part of this audit yet.

Thresholds (Claude's reading, 14 Oct 2026, pending team review):
  TFIDF_FLAG 0.55  the lexical threshold of grcbench.splits.twins: in its reading of training pairs,
                   the lowest TF-IDF band in which at least half of the pairs were twins.
  EMBED_FLAG 0.88  from reports/contamination/calibration.json (training-side pairs only, never test
                   items; run of 14 Oct 2026): above every one of 2,000 random training-question pairs
                   (max 0.828, p99 0.758) and below every pair the dataset team adjudicated in its
                   dedupe step (min 0.937; DUPLICATE and KEEP_BOTH alike). Training 8-gram template
                   twins straddle it (p50 0.777, p90 0.906).

Outputs (deterministic: sorted keys, stable order, cosines rounded to 4 decimals, no timestamps):
  reports/contamination/<tier>.json          per corpus: counts, rates, quantiles, top pairs
  reports/contamination/<tier>_items.jsonl   per item: scores against every corpus, and flags
  reports/contamination/calibration.json     the training-side calibration behind EMBED_FLAG
  reports/contamination.md                   method, thresholds and per-tier rate table
  handoff/review_sheets/contamination_<tier>.csv   flagged pairs for the team to judge; the team's
                                             decision/reviewer/notes are carried over on re-runs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy import sparse

from grcbench.common.io import iter_jsonl, read_json, write_json, write_jsonl, write_text
from grcbench.common.paths import DATASET, HANDOFF, LLAMA_CPP_BUILD, REPORTS

from . import corpora, embed, ngram, tfidf
from .corpora import Corpus, snippet

OUT = REPORTS / "contamination"
SUMMARY = REPORTS / "contamination.md"
SHEETS = HANDOFF / "review_sheets"
CALIBRATION = OUT / "calibration.json"

NGRAM_HIGH = 0.5            # share of an item's 8-grams found in the corpus, reported separately
TFIDF_FLAG = 0.55
EMBED_FLAG = 0.88
EMBED_TOPK = 3              # long corpora: TF-IDF nearest documents per item whose passage is embedded
TOP_DOCS = 3                # nearest documents kept per item and corpus
REVIEW_K = 10               # top pairs per tier and corpus in the JSON report
K_PASSAGES = 64             # nearest passages kept before collapsing to documents
CHUNK = 16384               # passages per TF-IDF chunk (memory)
CALIB_RANDOM = 2000         # random training pairs in the calibration
QUANTILES = (0, 10, 25, 50, 75, 90, 95, 99, 100)
SIGNALS = ("ngram", "tfidf", "embed")


def r4(x: float) -> float:
    return round(float(x), 4)


def rate(k: int, n: int) -> float:
    return round(k / n, 4) if n else 0.0


def quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not len(values):
        return None
    q = np.percentile(np.asarray(values, dtype=np.float64), QUANTILES)
    return {("min" if p == 0 else "max" if p == 100 else f"p{p}"): r4(v) for p, v in zip(QUANTILES, q)}


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- per-item results ---

@dataclass
class Pair:
    """One benchmark item against one training corpus."""
    ngram_total: int
    ngram_shared: int = 0
    ngram_docs: int = 0
    ngram_by_doc: dict[int, int] = field(default_factory=dict)
    ngram_top: tuple[int, str] | None = None                       # (doc, located snippet)
    tfidf_top: list[tuple[int, int, float]] = field(default_factory=list)   # (doc, passage, cosine)
    tfidf_row: np.ndarray | None = None                            # all passages (short corpora)
    embed_max: float | None = None
    embed_best: tuple[int, int] | None = None                      # (doc, passage)

    @property
    def tfidf_max(self) -> float:
        return self.tfidf_top[0][2] if self.tfidf_top else 0.0

    @property
    def coverage(self) -> float:
        return self.ngram_shared / self.ngram_total if self.ngram_total else 0.0

    def flags(self) -> list[str]:
        out = []
        if self.ngram_shared:
            out.append("ngram")
        if self.tfidf_max >= TFIDF_FLAG:
            out.append("tfidf")
        if self.embed_max is not None and self.embed_max >= EMBED_FLAG:
            out.append("embed")
        return out

    def tfidf_of(self, doc: int, passage: int | None) -> float | None:
        if self.tfidf_row is not None and passage is not None:
            return r4(self.tfidf_row[passage])
        return next((s for d, _, s in self.tfidf_top if d == doc), None)


@dataclass
class CorpusRun:
    corpus: str
    description: str
    sources: list[dict[str, str]]
    notes: list[str]
    embed_all: bool
    n_docs: int
    n_passages: int
    doc_ids: list[str]
    doc_refs: list[str]
    embed_topk: int = EMBED_TOPK
    texts: dict[int, str] = field(default_factory=dict)       # kept passages: index -> text
    pdoc: dict[int, int] = field(default_factory=dict)        # kept passages: index -> document
    pairs: dict[str, list[Pair]] = field(default_factory=dict)
    candidates: dict[str, list[int]] = field(default_factory=dict)
    embedded: bool = False

    @property
    def embed_scope(self) -> str:
        return ("every passage" if self.embed_all else
                f"best passage of each item's {self.embed_topk} nearest TF-IDF documents (max is a lower bound)")


def scan_corpus(corpus: Corpus, tiers: dict[str, Corpus], index: ngram.NgramIndex, embed_topk: int) -> CorpusRun:
    """8-gram and TF-IDF pass of one training corpus against the items of every tier."""
    items = [(t, k) for t in tiers for k in range(len(tiers[t].docs))]
    psg = corpora.corpus_passages(corpus)
    run = CorpusRun(corpus.name, corpus.description, corpus.sources, corpus.notes, corpus.embed_all,
                    len(corpus.docs), len(psg), [d.id for d in corpus.docs], [d.ref for d in corpus.docs],
                    embed_topk)
    pairs: list[Pair] = []

    def keep(pi: int) -> None:
        if pi not in run.texts:
            run.texts[pi] = corpora.passage_text(corpus, psg[pi])
            run.pdoc[pi] = psg[pi].doc

    # 8-gram: whole documents
    for o in index.scan(d.text for d in corpus.docs):
        p = Pair(o.n_grams, len(o.shared), len(o.docs), dict(o.docs))
        if o.docs:
            d = o.top_docs(1)[0][0]
            span = ngram.locate(corpus.docs[d].text, sorted(o.shared))
            text = corpus.docs[d].text
            p.ngram_top = (d, text[span[0]:span[1]] if span else text[:200])
        pairs.append(p)

    # TF-IDF: passages; df over this corpus only
    model = tfidf.Tfidf.fit(ngram.tokenize(corpora.passage_text(corpus, p)) for p in psg)
    queries = model.transform_texts([tiers[t].docs[k].text for t, k in items])

    def chunks():
        for a in range(0, len(psg), CHUNK):
            yield model.transform(ngram.tokenize(corpora.passage_text(corpus, p)) for p in psg[a:a + CHUNK])

    dense = None
    if corpus.embed_all and psg:
        mats = list(chunks())
        dense = tfidf.cosine(queries, mats[0] if len(mats) == 1 else sparse.vstack(mats).tocsr())
        scores, idx = _top_dense(dense)
    else:
        scores, idx = tfidf.nearest(queries, chunks(), K_PASSAGES)
    n_keep = max(TOP_DOCS, embed_topk)
    for row, p in enumerate(pairs):
        seen: set[int] = set()
        for s, pi in zip(scores[row].tolist(), idx[row].tolist()):
            d = psg[pi].doc
            if d in seen:
                continue
            seen.add(d)
            p.tfidf_top.append((d, pi, r4(s)))
            keep(pi)
            if len(p.tfidf_top) == n_keep:
                break
        if dense is not None:
            p.tfidf_row = dense[row]

    # split back into tiers; embedding candidates per tier
    for t in tiers:
        run.pairs[t] = [pairs[i] for i, (tt, _) in enumerate(items) if tt == t]
        if corpus.embed_all:
            cand = list(range(len(psg)))
        else:
            cand = sorted({pi for p in run.pairs[t] for _, pi, _ in p.tfidf_top[:embed_topk]})
        for pi in cand:
            keep(pi)
        run.candidates[t] = cand
    return run


def _top_dense(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = min(K_PASSAGES, dense.shape[1])
    idx = np.broadcast_to(np.arange(dense.shape[1], dtype=np.int64), dense.shape)
    order = np.lexsort((idx, -dense), axis=1)[:, :k]
    return np.take_along_axis(dense, order, 1), order


# --------------------------------------------------------------------------- embeddings ---------

def _progress(label: str) -> Callable[[int, int], None]:
    state = {"next": 0.0}

    def report(done: int, total: int) -> None:
        if total and (done / total >= state["next"] or done == total):
            _log(f"    embedding {label}: {done}/{total}")
            state["next"] = done / total + 0.1
    return report


def embed_runs(client: embed.EmbedClient, tiers: dict[str, Corpus], runs: list[CorpusRun]) -> None:
    for t, tc in tiers.items():
        iv = client.embed([d.text for d in tc.docs], _progress(f"{t} items"))
        for run in runs:
            cand = run.candidates.get(t) or []
            if not cand:
                continue
            cv = client.embed([run.texts[pi] for pi in cand], _progress(f"{t} x {run.corpus}"))
            sims = embed.cosine(iv, cv)
            best = sims.argmax(axis=1)              # first maximum: lowest passage index on ties
            for k, p in enumerate(run.pairs[t]):
                j = int(best[k])
                p.embed_max = r4(sims[k, j])
                p.embed_best = (run.pdoc[cand[j]], cand[j])
            run.embedded = True


# --------------------------------------------------------------------------- calibration --------

def adjudicated_pairs() -> list[dict[str, Any]]:
    """Pairs the dataset team adjudicated as DUPLICATE or KEEP_BOTH in the dedupe step."""
    root = DATASET / "03_questions_merged"
    out = []
    for prefix in ("", "cross_bank_"):
        decisions = {r["cluster_id"]: r for r in iter_jsonl(root / f"{prefix}semantic_dedupe_decisions.jsonl")}
        for c in iter_jsonl(root / f"{prefix}semantic_duplicate_clusters.jsonl"):
            qs = c.get("questions") or (c.get("source_questions", []) + c.get("blind_questions", []))
            dec = decisions.get(c.get("cluster_id"))
            if not dec:
                continue
            for a, b in itertools.combinations(sorted(qs, key=lambda q: q["id"]), 2):
                out.append({"cluster_id": c["cluster_id"], "decision": dec["decision"],
                            "id_a": a["id"], "id_b": b["id"], "text_a": a["question"], "text_b": b["question"]})
    return sorted(out, key=lambda r: (r["cluster_id"], r["id_a"], r["id_b"]))


def _random_pairs(n: int, k: int) -> list[tuple[int, int]]:
    out, i = set(), 0
    while len(out) < min(k, n * (n - 1) // 2):
        h = hashlib.sha256(f"grc-bench-contamination-calibration\0{i}".encode()).digest()
        a, b = int.from_bytes(h[:8], "big") % n, int.from_bytes(h[8:16], "big") % n
        if a != b:
            out.add((min(a, b), max(a, b)))
        i += 1
    return sorted(out)


def _twin_pairs(texts: Sequence[str]) -> list[tuple[int, int]]:
    index: dict[str, set[int]] = defaultdict(set)
    for i, t in enumerate(texts):
        for g in set(ngram.ngrams(ngram.tokenize(t))):
            index[g].add(i)
    return sorted({(a, b) for ids in index.values() for a, b in itertools.combinations(sorted(ids), 2)})


def calibration(client: embed.EmbedClient | None) -> dict[str, Any]:
    """Training-side calibration of the cosine thresholds (no test item is used)."""
    train = list(iter_jsonl(corpora.FROZEN / "train_questions.jsonl"))
    texts = [r["question"] for r in train]
    adj = adjudicated_pairs()
    model = tfidf.Tfidf.fit(ngram.tokenize(t) for t in texts)
    tv = model.transform_texts(texts)
    ev = client.embed(texts, _progress("calibration (training questions)")) if client else None
    twins = _twin_pairs(texts)
    rnd = _random_pairs(len(texts), CALIB_RANDOM)

    def block(pairs: list[tuple[int, int]], label: str) -> dict[str, Any]:
        a = np.array([p[0] for p in pairs], dtype=np.int64)
        b = np.array([p[1] for p in pairs], dtype=np.int64)
        tf = np.asarray(tv[a].multiply(tv[b]).sum(axis=1)).ravel() if len(pairs) else np.zeros(0)
        out: dict[str, Any] = {"pairs": len(pairs), "description": label,
                               "tfidf": {"quantiles": quantiles(tf), "share_ge_threshold": rate(int((np.round(tf, 4) >= TFIDF_FLAG).sum()), len(pairs))}}
        if ev is not None:
            em = np.einsum("ij,ij->i", ev[a], ev[b]) if len(pairs) else np.zeros(0)
            out["embedding"] = {"quantiles": quantiles(em),
                                "share_ge_threshold": rate(int((np.round(em, 4) >= EMBED_FLAG).sum()), len(pairs))}
        return out

    rows = []
    if adj:
        at = model.transform_texts([r["text_a"] for r in adj])
        bt = model.transform_texts([r["text_b"] for r in adj])
        tf = np.asarray(at.multiply(bt).sum(axis=1)).ravel()
        if client:
            ea = client.embed([r["text_a"] for r in adj])
            eb = client.embed([r["text_b"] for r in adj])
            em = np.einsum("ij,ij->i", ea, eb)
        for i, r in enumerate(adj):
            rows.append({"cluster_id": r["cluster_id"], "decision": r["decision"], "id_a": r["id_a"],
                         "id_b": r["id_b"], "tfidf": r4(tf[i]), "embed": r4(em[i]) if client else None})
    return {
        "purpose": ("Training-side calibration of the TF-IDF and embedding thresholds. Uses only the 1,668 "
                    "training questions and the dataset's adjudicated dedupe pairs; no test item."),
        "thresholds": {"tfidf": TFIDF_FLAG, "embed": EMBED_FLAG},
        "embedding_status": "done" if client else "skipped (--skip-embed)",
        "adjudicated_pairs": rows,
        "train_8gram_twin_pairs": block(twins, "training-question pairs sharing a word 8-gram (template twins)"),
        "random_train_pairs": block(rnd, f"{len(rnd)} sha256-sampled training-question pairs (background)"),
    }


# --------------------------------------------------------------------------- tier reports -------

def _pair_row(tier: str, item, run: CorpusRun, p: Pair, doc: int, passage: int | None,
              signals: list[str], text: str) -> dict[str, Any]:
    return {"tier": tier, "item_id": item.id, "corpus": run.corpus, "doc_id": run.doc_ids[doc],
            "doc_ref": run.doc_refs[doc], "signals": "+".join(signals),
            "ngram_shared": p.ngram_by_doc.get(doc, 0), "tfidf": p.tfidf_of(doc, passage),
            "embed": p.embed_max if p.embed_best and p.embed_best[0] == doc else None,
            "item_text": snippet(item.text, 300), "training_text": snippet(text, 300)}


def review_rows(tier: str, tc: Corpus, runs: list[CorpusRun]) -> list[dict[str, Any]]:
    rows = []
    for k, item in enumerate(tc.docs):
        for run in runs:
            p = run.pairs[tier][k]
            flags = p.flags()
            if not flags:
                continue
            by_doc: dict[int, dict[str, Any]] = {}

            def add(doc: int, passage: int | None, signal: str, text: str) -> None:
                e = by_doc.setdefault(doc, {"signals": [], "passage": None, "text": text})
                e["signals"].append(signal)
                if passage is not None and e["passage"] is None:
                    e["passage"], e["text"] = passage, text

            if "ngram" in flags and p.ngram_top:
                add(p.ngram_top[0], None, "ngram", p.ngram_top[1])
            if "tfidf" in flags:
                d, pi, _ = p.tfidf_top[0]
                add(d, pi, "tfidf", run.texts[pi])
            if "embed" in flags and p.embed_best:
                d, pi = p.embed_best
                add(d, pi, "embed", run.texts[pi])
            for doc in sorted(by_doc):
                e = by_doc[doc]
                rows.append(_pair_row(tier, item, run, p, doc, e["passage"], e["signals"], e["text"]))
    return rows


def _top_pairs(tier: str, tc: Corpus, run: CorpusRun) -> list[dict[str, Any]]:
    pairs = run.pairs[tier]
    key = (lambda k: (-(pairs[k].embed_max or 0.0), -pairs[k].tfidf_max, tc.docs[k].id)) if run.embedded else \
          (lambda k: (-pairs[k].tfidf_max, -pairs[k].ngram_shared, tc.docs[k].id))
    out = []
    if not run.n_docs:
        return out
    for k in sorted(range(len(pairs)), key=key)[:REVIEW_K]:
        p = pairs[k]
        doc, pi = p.embed_best if run.embedded and p.embed_best else p.tfidf_top[0][:2] if p.tfidf_top else (0, None)
        out.append({"item_id": tc.docs[k].id, "doc_id": run.doc_ids[doc], "doc_ref": run.doc_refs[doc],
                    "ngram_shared": p.ngram_by_doc.get(doc, 0), "tfidf": p.tfidf_of(doc, pi),
                    "embed": p.embed_max if run.embedded else None,
                    "item": snippet(tc.docs[k].text, 160), "training": snippet(run.texts.get(pi, ""), 240)})
    return out


def tier_report(tier: str, tc: Corpus, runs: list[CorpusRun], embed_status: str) -> tuple[dict, list[dict]]:
    n = len(tc.docs)
    report: dict[str, Any] = {
        "tier": tier, "description": tc.description, "n_items": n, "sources": tc.sources,
        "implements": ["eval plan §6: contamination audit, a contamination rate for each tier",
                       "proposal §4.5: 8-gram decontamination; embedding near-duplicates"],
        "status": "flags are candidates for team review; nothing is removed",
        "thresholds": {"ngram_n": ngram.N, "ngram_high_coverage": NGRAM_HIGH, "tfidf": TFIDF_FLAG,
                       "embed": EMBED_FLAG},
        "embedding_status": embed_status, "corpora": {},
    }
    items: list[dict[str, Any]] = [{"tier": tier, "id": d.id, "aliases": list(d.aliases), "flags": [], "scores": {}}
                                   for d in tc.docs]
    any_signal = {s: set() for s in SIGNALS}
    for run in runs:
        pairs = run.pairs[tier]
        flags = [set(p.flags()) for p in pairs]
        count = {s: sum(s in f for f in flags) for s in SIGNALS}
        hi = sum(1 for p in pairs if p.ngram_shared and p.coverage >= NGRAM_HIGH)
        flagged_any = sum(1 for f in flags if f)
        entry = {
            "description": run.description, "n_docs": run.n_docs, "n_passages": run.n_passages,
            "notes": run.notes, "sources": run.sources,
            "ngram": {"items_with_twin": count["ngram"], "rate": rate(count["ngram"], n),
                      "items_coverage_ge_50pct": hi, "rate_coverage_ge_50pct": rate(hi, n)},
            "tfidf": {"signal": "lexical (not paraphrase detection)", "threshold": TFIDF_FLAG,
                      "max_cosine_quantiles": quantiles([p.tfidf_max for p in pairs]),
                      "items_flagged": count["tfidf"], "rate": rate(count["tfidf"], n)},
            "embedding": {"signal": "semantic", "status": "done" if run.embedded else embed_status,
                          "scope": run.embed_scope, "passages_embedded": len(run.candidates.get(tier, [])) if run.embedded else 0,
                          "threshold": EMBED_FLAG,
                          "max_cosine_quantiles": quantiles([p.embed_max for p in pairs]) if run.embedded else None,
                          "items_flagged": count["embed"] if run.embedded else None,
                          "rate": rate(count["embed"], n) if run.embedded else None},
            "any_signal": {"items_flagged": flagged_any, "rate": rate(flagged_any, n)},
            "signal_overlap": {"+".join(c): sum(1 for f in flags if set(c) <= f)
                               for r in (2, 3) for c in itertools.combinations(SIGNALS, r)},
            "top_pairs": _top_pairs(tier, tc, run),
        }
        report["corpora"][run.corpus] = entry
        for k, (p, f) in enumerate(zip(pairs, flags)):
            for s in f:
                any_signal[s].add(k)
            items[k]["flags"] += [f"{run.corpus}:{s}" for s in SIGNALS if s in f]
            top = p.tfidf_top[0] if p.tfidf_top else None
            items[k]["scores"][run.corpus] = {
                "ngram_shared": p.ngram_shared, "ngram_coverage": r4(p.coverage), "ngram_docs": p.ngram_docs,
                "tfidf_max": p.tfidf_max, "tfidf_doc": run.doc_ids[top[0]] if top else None,
                "embed_max": p.embed_max, "embed_doc": run.doc_ids[p.embed_best[0]] if p.embed_best else None}
    flagged = {it["id"]: it["flags"] for it in items if it["flags"]}
    report["all_corpora"] = {s: {"items_flagged": len(any_signal[s]), "rate": rate(len(any_signal[s]), n)}
                             for s in SIGNALS}
    report["all_corpora"]["any"] = {"items_flagged": len(flagged), "rate": rate(len(flagged), n)}
    if not any(run.embedded for run in runs):
        report["all_corpora"]["embed"] = {"items_flagged": None, "rate": None}
    report["flagged_items"] = flagged
    return report, items


CSV_FIELDS = ["tier", "item_id", "corpus", "doc_id", "doc_ref", "signals", "ngram_shared", "tfidf", "embed",
              "item_text", "training_text", "decision", "reviewer", "notes"]
REVIEW_FIELDS = ("decision", "reviewer", "notes")       # filled in by the team; kept across re-runs
NOT_FLAGGED = "not flagged in this run"


def _sheet_key(r: dict[str, Any]) -> tuple[str, str, str, str]:
    return (str(r.get("tier", "")), str(r.get("item_id", "")), str(r.get("corpus", "")), str(r.get("doc_id", "")))


def read_sheet(path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    """Rows of an existing review sheet in which the team wrote something, keyed by pair."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        return {_sheet_key(r): r for r in csv.DictReader(fh)
                if any((r.get(f) or "").strip() for f in REVIEW_FIELDS)}


def write_sheet(tier: str, rows: list[dict[str, Any]], path=None) -> int:
    """Write the review sheet of one tier. The team's decision, reviewer and notes are carried over
    from the sheet on disk by (tier, item, corpus, training document); a reviewed pair that is no
    longer flagged is kept at the end with signals 'not flagged in this run', so no review is lost.
    Returns the number of reviewed rows carried over."""
    path = path or SHEETS / f"contamination_{tier}.csv"
    old = read_sheet(path)
    carried = len(old)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        prev = old.pop(_sheet_key(r), {})
        w.writerow({**r, **{f: prev.get(f, "") for f in REVIEW_FIELDS}})
    for key in sorted(old):
        w.writerow({**old[key], "signals": NOT_FLAGGED})
    write_text(path, buf.getvalue())
    return carried


# --------------------------------------------------------------------------- summary ------------

def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def _q(qs: dict | None, key: str) -> str:
    return "–" if not qs else f"{qs[key]:.3f}"


def _ordered(entries: dict[str, Any]) -> list[tuple[str, Any]]:
    """Corpora in corpora.CORPORA order (JSON files store them with sorted keys)."""
    rank = {c: i for i, c in enumerate(corpora.CORPORA)}
    return sorted(entries.items(), key=lambda kv: (rank.get(kv[0], len(rank)), kv[0]))


def _not_audited(tier: str, dev: set[str] | None) -> str:
    if corpora.benchmark_tier(tier, dev) is None:
        return corpora.absent_reason(tier, dev)
    return "has items but has not been audited yet (run `./gb contamination` without --only)"


def summary_md() -> str:
    """reports/contamination.md, rebuilt from every tier report on disk (so --only runs keep the rest)."""
    reports = {t: read_json(OUT / f"{t}.json") for t in corpora.TIERS if (OUT / f"{t}.json").exists()}
    calib = read_json(CALIBRATION) if CALIBRATION.exists() else None
    dev = corpora.dev_ids()
    L = ["# Contamination audit (eval plan §6)", "",
         "Status: **draft tooling run, flags only.** Every number below counts *candidates* for human review. "
         "Nothing is removed from any dataset (dataset README: similarity scores never remove questions). "
         "The team judges each flagged pair in `handoff/review_sheets/contamination_<tier>.csv`; the confirmed "
         "contamination rate per tier is reported after that review. The LLM rephrase detector of eval plan §6 "
         "[54] waits for API access. Generated by `./gb contamination` (grcbench.contamination.audit).", "",
         "## What is compared", "",
         "Every benchmark tier is compared with every training-side corpus:", "",
         "| Benchmark tier | Items | Status |", "|---|---:|---|"]
    for t in corpora.TIERS:
        if t in reports:
            L.append(f"| {t} | {reports[t]['n_items']} | audited; embeddings {reports[t]['embedding_status']} |")
        else:
            L.append(f"| {t} | – | not audited: {_not_audited(t, dev)} |")
    first = next(iter(reports.values()), None)
    if first:
        L += ["", "| Training corpus | Documents | Passages | Notes |", "|---|---:|---:|---|"]
        for c, e in _ordered(first["corpora"]):
            L.append(f"| {c} | {e['n_docs']:,} | {e['n_passages']:,} | {'; '.join(e['notes'])} |")
    L += ["", "## Method and thresholds", "",
          "| Signal | Unit | Flag when | Kind |", "|---|---|---|---|",
          f"| 8-gram | whole documents | the item shares ≥ 1 word {ngram.N}-gram with a document "
          f"(coverage ≥ {int(NGRAM_HIGH * 100)}% of the item's 8-grams also reported) | lexical, exact |",
          f"| TF-IDF cosine | passages | max cosine ≥ {TFIDF_FLAG} | **lexical** (not paraphrase detection) |",
          f"| Embedding cosine | passages | max cosine ≥ {EMBED_FLAG} | semantic |", "",
          "- Tokenizer (all lexical signals): lowercase, every non-word character (`\\W`) to a space, split on "
          "whitespace. With it, 26 of the 185 frozen test questions share an 8-gram with the 1,668 training "
          "questions (eval plan §6), reproduced by `tests/test_contamination.py`.",
          f"- Passages: a document of at most {corpora.PASSAGE_SINGLE_MAX} words is one passage; longer documents "
          f"are cut into {corpora.PASSAGE_WORDS}-word windows every {corpora.PASSAGE_STRIDE} words, so a question "
          "repeated inside a long answer or session is still visible to the cosine signals.",
          "- TF-IDF: w = (1 + ln tf)·(ln((1 + N)/(1 + df)) + 1), df over the corpus passages, rows L2-normalised "
          "(the weighting of `grcbench.splits.twins`). The threshold is that module's lexical threshold: in its "
          "reading of training pairs, the lowest band in which at least half of the pairs were twins.",
          f"- Embeddings: EmbeddingGemma 300M (QAT Q8_0 GGUF, SHA-256 `{embed.MODEL_SHA256[:16]}…`, the file "
          f"GRC-Agent pins) on llama.cpp {LLAMA_CPP_BUILD} `llama-server --embeddings --pooling mean`, context "
          f"{embed.CONTEXT}, prompt `{embed.PROMPT}` on both sides, vectors L2-normalised. Training questions and "
          f"task specs are embedded completely. For the long corpora only the best passage of each item's "
          f"{EMBED_TOPK} (default) nearest TF-IDF documents is embedded (≈15M words do not fit a CPU session), so there the "
          "embedding maximum is a lower bound and a paraphrase with little word overlap can be missed.",
          f"- The embedding threshold {EMBED_FLAG} comes from the training-side calibration below (no test item was "
          "used to set it)" + (f": it lies above every random training-question pair (max "
          f"{calib['random_train_pairs']['embedding']['quantiles']['max']:.3f}) and below every pair adjudicated in "
          f"the dataset's dedupe step (min {min(r['embed'] for r in calib['adjudicated_pairs']):.3f})"
          if calib and calib.get("embedding_status") == "done" and calib["adjudicated_pairs"] else "")
          + ". Both thresholds are Claude's reading of 14 Oct 2026, pending team review.",
          "- Dev questions are excluded from all training use, so the training questions corpus drops them and the "
          "answer and repair corpora drop the rows of dev questions.", ""]
    for t, rep in reports.items():
        L += [f"## {t}: contamination rate ({rep['n_items']} items)", "",
              "| Training corpus | 8-gram twin | 8-gram ≥ 50% | TF-IDF ≥ thr (lexical) | Embedding ≥ thr | Any signal |",
              "|---|---:|---:|---:|---:|---:|"]
        for c, e in _ordered(rep["corpora"]):
            L.append(f"| {c} | {e['ngram']['items_with_twin']} ({_pct(e['ngram']['rate'])}) | "
                     f"{e['ngram']['items_coverage_ge_50pct']} ({_pct(e['ngram']['rate_coverage_ge_50pct'])}) | "
                     f"{e['tfidf']['items_flagged']} ({_pct(e['tfidf']['rate'])}) | "
                     + (f"{e['embedding']['items_flagged']} ({_pct(e['embedding']['rate'])})" if e['embedding']['rate'] is not None else "not run")
                     + f" | {e['any_signal']['items_flagged']} ({_pct(e['any_signal']['rate'])}) |")
        a = rep["all_corpora"]
        L.append(f"| **all corpora** | {a['ngram']['items_flagged']} ({_pct(a['ngram']['rate'])}) | | "
                 f"{a['tfidf']['items_flagged']} ({_pct(a['tfidf']['rate'])}) | "
                 + (f"{a['embed']['items_flagged']} ({_pct(a['embed']['rate'])})" if a['embed']['rate'] is not None else "not run")
                 + " | "
                 f"**{a['any']['items_flagged']} ({_pct(a['any']['rate'])})** |")
        L += ["", "Distribution of each item's maximum cosine (quantiles over items):", "",
              "| Training corpus | TF-IDF p50 | p90 | p99 | max | Embedding p50 | p90 | p99 | max | Embedding scope |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for c, e in _ordered(rep["corpora"]):
            tq, eq = e["tfidf"]["max_cosine_quantiles"], e["embedding"]["max_cosine_quantiles"]
            L.append(f"| {c} | {_q(tq, 'p50')} | {_q(tq, 'p90')} | {_q(tq, 'p99')} | {_q(tq, 'max')} | "
                     f"{_q(eq, 'p50')} | {_q(eq, 'p90')} | {_q(eq, 'p99')} | {_q(eq, 'max')} | "
                     f"{'all passages' if e['embedding']['scope'].startswith('every') else 'TF-IDF candidates'} |")
        ov = {c: e["signal_overlap"] for c, e in _ordered(rep["corpora"])}
        L += ["", "Agreement between signals (items flagged by both):", "",
              "| Training corpus | 8-gram & TF-IDF | 8-gram & embedding | TF-IDF & embedding | all three |",
              "|---|---:|---:|---:|---:|"]
        for c, o in ov.items():
            L.append(f"| {c} | {o['ngram+tfidf']} | {o['ngram+embed']} | {o['tfidf+embed']} | {o['ngram+tfidf+embed']} |")
        L += ["", f"Per-item scores: `reports/contamination/{t}_items.jsonl`; top pairs per corpus: "
              f"`reports/contamination/{t}.json`; flagged pairs for review: "
              f"`handoff/review_sheets/contamination_{t}.csv`.", ""]
    if calib:
        L += ["## Calibration (training side only)", "",
              "| Pair set | Pairs | TF-IDF p10 | p50 | ≥ thr | Embedding p10 | p50 | ≥ thr |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for key in ("train_8gram_twin_pairs", "random_train_pairs"):
            b = calib[key]
            em = b.get("embedding")
            L.append(f"| {b['description']} | {b['pairs']:,} | {_q(b['tfidf']['quantiles'], 'p10')} | "
                     f"{_q(b['tfidf']['quantiles'], 'p50')} | {_pct(b['tfidf']['share_ge_threshold'])} | "
                     f"{_q(em['quantiles'], 'p10') if em else '–'} | {_q(em['quantiles'], 'p50') if em else '–'} | "
                     f"{_pct(em['share_ge_threshold']) if em else '–'} |")
        if calib["adjudicated_pairs"]:
            L += ["", "Pairs the dataset team adjudicated in its dedupe step (both decisions score high, which is why "
                  "similarity alone never removes a question):", "",
                  "| Cluster | Decision | TF-IDF | Embedding |", "|---|---|---:|---:|"]
            for r in calib["adjudicated_pairs"]:
                L.append(f"| {r['cluster_id']} | {r['decision']} | {r['tfidf']:.3f} | "
                         f"{'–' if r['embed'] is None else format(r['embed'], '.3f')} |")
        L.append("")
    L += ["## Limits", "",
          "- Flags are candidates. Template twins (the same flowgraph question for a different signal) share "
          "8-grams and score high on both cosines without being the same question; the review sheet separates them.",
          "- The embedding maximum over the long corpora is a lower bound (TF-IDF candidates only); "
          "`--embed-topk` widens it.",
          "- Task-level overlap for agent tasks (same goal or same starting flowgraph, eval plan §6) needs the task "
          "metadata of Tier-A items and is checked when those items exist.",
          "- Re-run after the dev split is final, after Phases 3–4 (Tier F and A items) and again in W6: "
          "`./gb contamination` (all tiers), `--only tier_k_test`, `--skip-embed` for the lexical signals alone. "
          "A re-run keeps the decision, reviewer and notes the team wrote in the review sheets; a reviewed pair "
          f"that is no longer flagged stays in the sheet marked '{NOT_FLAGGED}'.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- CLI ----------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="./gb contamination", description=__doc__.split("\n")[0])
    ap.add_argument("--only", action="append", default=[], metavar="TIER",
                    help=f"audit only these tiers (repeat or comma-separate): {', '.join(corpora.TIERS)}")
    ap.add_argument("--corpora", default=",".join(corpora.CORPORA),
                    help="training corpora to compare with (comma-separated; default all; the tier reports "
                         "then cover only these)")
    ap.add_argument("--skip-embed", action="store_true", help="lexical signals only (no llama-server)")
    ap.add_argument("--embed-topk", type=int, default=EMBED_TOPK,
                    help="long corpora: TF-IDF nearest documents per item whose best passage is embedded")
    ap.add_argument("--threads", type=int, default=None, help="llama-server threads (default 4)")
    ap.add_argument("--summary-only", action="store_true",
                    help="only rebuild reports/contamination.md from the tier reports on disk")
    args = ap.parse_args(argv)
    if args.summary_only:
        write_text(SUMMARY, summary_md())
        _log(f"rewrote {SUMMARY.relative_to(REPORTS.parent)}")
        return 0

    only = [t for arg in args.only for t in arg.split(",") if t] or list(corpora.TIERS)
    bad = [t for t in only if t not in corpora.TIERS]
    names = [c for c in args.corpora.split(",") if c]
    bad += [c for c in names if c not in corpora.CORPORA]
    if bad:
        ap.error(f"unknown tier or corpus: {', '.join(bad)}")
    t0 = time.monotonic()
    dev = corpora.dev_ids()
    _log(f"dev split: {'%d ids' % len(dev) if dev is not None else 'not drawn yet (splits/tier_k_dev.json absent)'}")
    tiers: dict[str, Corpus] = {}
    for t in only:
        c = corpora.benchmark_tier(t, dev)
        if c is None:
            _log(f"{t}: absent ({corpora.absent_reason(t, dev)})")
        else:
            tiers[t] = c
            _log(f"{t}: {len(c.docs)} items")
    if not tiers:
        write_text(SUMMARY, summary_md())
        _log("no benchmark tier has items; summary rewritten")
        return 0

    index = ngram.NgramIndex([d.text for t in tiers for d in tiers[t].docs])
    runs: list[CorpusRun] = []
    for name in names:
        t1 = time.monotonic()
        corpus = corpora.training_corpus(name, dev)
        runs.append(scan_corpus(corpus, tiers, index, args.embed_topk))
        del corpus
        r = runs[-1]
        _log(f"  {name}: {r.n_docs:,} docs, {r.n_passages:,} passages, lexical pass {time.monotonic() - t1:.0f}s")

    embed_status = "skipped (--skip-embed)"
    if not args.skip_embed:
        embed.verify_model()
        with embed.LlamaEmbedServer(threads=args.threads) as client:
            t1 = time.monotonic()
            embed_runs(client, tiers, runs)
            calib = calibration(client)
            _log(f"  embeddings: {client.computed} computed, {client.cached} from cache, "
                 f"{client.tokens:,} tokens in {client.seconds:.0f}s of requests ({time.monotonic() - t1:.0f}s)")
        embed_status = "done"
        write_json(CALIBRATION, calib)
    elif CALIBRATION.exists() and read_json(CALIBRATION).get("embedding_status") == "done":
        _log("  calibration: kept the existing one with embeddings (--skip-embed)")
    else:
        write_json(CALIBRATION, calibration(None))

    for t, tc in tiers.items():
        report, items = tier_report(t, tc, runs, embed_status)
        write_json(OUT / f"{t}.json", report)
        write_jsonl(OUT / f"{t}_items.jsonl", items)
        kept = write_sheet(t, review_rows(t, tc, runs))
        if kept:
            _log(f"  {t}: carried {kept} team review decision(s) over into the new review sheet")
        a = report["all_corpora"]
        _log(f"{t}: {a['any']['items_flagged']}/{len(tc.docs)} items flagged by any signal "
             f"(8-gram {a['ngram']['items_flagged']}, TF-IDF {a['tfidf']['items_flagged']}, "
             f"embedding {a['embed']['items_flagged']})")
    write_text(SUMMARY, summary_md())
    _log(f"wrote {SUMMARY.relative_to(REPORTS.parent)} and reports/contamination/ in {time.monotonic() - t0:.0f}s")
    return 0

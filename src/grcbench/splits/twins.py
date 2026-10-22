"""Twin groups over the Tier-K training questions, and the training twins of test questions.

Eval plan §4.1 (Table 4) holds 200 of the 1,668 training questions back as the dev split. §6 counts
"near-twins", meaning test questions that share an 8-word sequence with a training question. A dev
question must have no sibling left in training: a sibling's teacher answer would teach the dev
answer, and checkpoint selection would then reward memorization. Twin groups are therefore the
connected components (union-find) of the training questions under four kinds of evidence:

  8gram      the two questions share a word 8-gram. Tokenizer: lowercase, replace every non-word
             character (regex \\W) with a space, split on whitespace. This is the tokenizer behind the
             eval plan's count of 26 test questions with a training twin.
  semantic   a pair in 03_questions_merged/semantic_duplicate_candidates.jsonl with similarity >=
             SEMANTIC_THRESHOLD. That file scores only the source lane (its ids are
             questions_extracted ids, mapped here to canonical ids via (source_file, source_index)
             provenance). Rows whose endpoint was later removed as an adjudicated DUPLICATE are dropped.
  lexical    TF-IDF cosine >= LEXICAL_THRESHOLD, applied ONLY to pairs that the semantic file never
             scored (at least one endpoint from the blind lane). Weights: (1 + ln tf) *
             (ln((1+N)/(1+df)) + 1), with df over the N training questions, L2-normalized, and cosine
             rounded to 4 decimals. This closes the blind-lane gap; it is not in the WS1 brief and is
             recorded as a deviation for team review.
  keep_both  a KEEP_BOTH decision in any *semantic_dedupe_decisions.jsonl. The dataset team judged
             these "near repeats ... useful training variation", which is exactly a dev-sibling risk.

Thresholds (Claude's reading on 14 Oct 2026, pending team review; see BAND_READING): for each scored
source, the threshold is the lower edge of the lowest band in which at least half of the pairs read
were twins, meaning an answer to one would substantially answer the other. Below 0.92 the semantic
scores are dominated by template neighbours ("Design a GNU Radio flowgraph to ..."), and union-find
starts chaining unrelated families: the largest group grows from 28 to 54 at 0.91. Similarity scores
never remove a question from the dataset (dataset README). They only keep a question out of dev, and
every such exclusion is reported.

Test twins: training questions sharing an 8-gram with any of the 185 frozen test questions (eval plan
§6). Test neighbours: training questions linked by semantic, lexical or keep_both evidence to a test
question they share no 8-gram with. They are flagged separately and are also kept out of dev.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scipy import sparse

from grcbench.common.io import read_jsonl
from grcbench.common.paths import DATASET

NGRAM_N = 8
SEMANTIC_THRESHOLD = 0.92
LEXICAL_THRESHOLD = 0.55
SEMANTIC_SWEEP = (0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.88, 0.86, 0.84, 0.82)
LEXICAL_SWEEP = (0.80, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40)
LEXICAL_FLOOR = min(LEXICAL_SWEEP)

SEMANTIC_CANDIDATES = DATASET / "03_questions_merged" / "semantic_duplicate_candidates.jsonl"

# Claude's reading of every training pair (both ends training questions, no shared 8-gram) in each
# band, 14 Oct 2026. "twins" = an answer to one would substantially answer the other. The counts of
# pairs are re-computed and checked by the tests; the twin judgments are pending team review
# (handoff/review_sheets/tier_k_twin_threshold_review.csv).
BAND_READING: dict[str, list[dict[str, Any]]] = {
    "semantic": [
        {"band": [0.94, 0.95], "pairs": 9, "twins": 8},
        {"band": [0.93, 0.94], "pairs": 13, "twins": 9},
        {"band": [0.92, 0.93], "pairs": 26, "twins": 14},
        {"band": [0.91, 0.92], "pairs": 40, "twins": 12},
    ],
    "lexical": [
        {"band": [0.70, 1.01], "pairs": 11, "twins": 11},
        {"band": [0.60, 0.70], "pairs": 32, "twins": 25},
        {"band": [0.55, 0.60], "pairs": 34, "twins": 24},
        {"band": [0.50, 0.55], "pairs": 70, "twins": 30},
    ],
}

Pair = tuple[str, str]


def tokens(text: str) -> list[str]:
    """Lowercase, every non-word character to a space, split on whitespace."""
    return re.sub(r"\W", " ", text.lower()).split()


def ngrams(text: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    w = tokens(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def _pair(a: str, b: str) -> Pair:
    return (a, b) if a < b else (b, a)


# ---------------------------------------------------------------- evidence -----------------------

def ngram_index(records: Iterable[Mapping[str, Any]], n: int = NGRAM_N) -> dict[tuple[str, ...], set[str]]:
    index: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for r in records:
        for g in ngrams(r["question"], n):
            index[g].add(r["id"])
    return index


def ngram_pairs(records: Sequence[Mapping[str, Any]], n: int = NGRAM_N) -> set[Pair]:
    """Pairs of records that share at least one word n-gram."""
    out: set[Pair] = set()
    for ids in ngram_index(records, n).values():
        s = sorted(ids)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                out.add((s[i], s[j]))
    return out


def ngram_hits(queries: Sequence[Mapping[str, Any]], pool: Sequence[Mapping[str, Any]],
               n: int = NGRAM_N) -> dict[str, set[str]]:
    """For each pool id, the query ids it shares an n-gram with (only ids with a hit)."""
    index = ngram_index(pool, n)
    hits: dict[str, set[str]] = defaultdict(set)
    for q in queries:
        for g in ngrams(q["question"], n):
            for pid in index.get(g, ()):
                hits[pid].add(q["id"])
    return dict(hits)


def tfidf(fit: Sequence[str], texts: Sequence[str]) -> sparse.csr_matrix:
    """L2-normalized sparse TF-IDF rows for `texts`, vocabulary and df from `fit` (see module doc)."""
    fit_toks = [tokens(t) for t in fit]
    df = Counter(w for d in fit_toks for w in set(d))
    vocab = {w: i for i, w in enumerate(sorted(df))}
    n = len(fit_toks)
    rows, cols, vals = [], [], []
    for row, text in enumerate(texts):
        weights = {vocab[w]: (1.0 + math.log(c)) * (math.log((1 + n) / (1 + df[w])) + 1.0)
                   for w, c in Counter(tokens(text)).items() if w in vocab}
        norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
        for col in sorted(weights):
            rows.append(row)
            cols.append(col)
            vals.append(weights[col] / norm)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(vocab)))


def lexical_scores(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]],
                   fit: Sequence[Mapping[str, Any]], floor: float = LEXICAL_FLOOR,
                   same: bool = False) -> dict[Pair, float]:
    """TF-IDF cosine (4 d.p.) >= floor between left and right records, only for pairs that the
    semantic file did not score (not both from the source lane). same=True: left is right."""
    xl = tfidf([r["question"] for r in fit], [r["question"] for r in left])
    xr = xl if same else tfidf([r["question"] for r in fit], [r["question"] for r in right])
    sims = (xl @ xr.T).tocoo()
    keep = sims.data >= floor - 1e-4                     # cheap pre-filter; exact test after rounding
    out: dict[Pair, float] = {}
    for i, j, s in sorted(zip(sims.row[keep].tolist(), sims.col[keep].tolist(), sims.data[keep].tolist())):
        s = round(s, 4)
        if s < floor:
            continue
        a, b = left[i], right[j]
        if (same and i >= j) or a["id"] == b["id"]:
            continue
        if a["origin"] == "source" and b["origin"] == "source":
            continue
        out[_pair(a["id"], b["id"])] = s
    return out


def semantic_scores(records: Sequence[Mapping[str, Any]],
                    path: Path = SEMANTIC_CANDIDATES) -> tuple[dict[Pair, float], dict[str, int]]:
    """Map candidate rows to canonical id pairs (max similarity per pair) plus mapping statistics."""
    prov: dict[tuple[str, int], str] = {}
    for r in records:
        for p in r.get("provenance", []):
            prov[(p["source_file"], int(p["source_index"]))] = r["id"]
    scores: dict[Pair, float] = {}
    stats = Counter()
    sims: list[float] = []
    for row in read_jsonl(path):
        stats["rows"] += 1
        sims.append(float(row["similarity"]))
        ends = []
        for side in ("a", "b"):
            src = row[side]["source"]
            ends.append(prov.get((src["source_file"], int(src["source_index"]))))
        if None in ends:
            stats["dropped_endpoint_removed_by_adjudicated_dedupe"] += 1
            continue
        if ends[0] == ends[1]:
            stats["dropped_same_canonical_question"] += 1
            continue
        stats["mapped"] += 1
        key = _pair(ends[0], ends[1])
        scores[key] = max(scores.get(key, 0.0), float(row["similarity"]))
    stats["unique_canonical_pairs"] = len(scores)
    out_stats: dict[str, Any] = dict(sorted(stats.items()))
    out_stats["min_similarity_in_file"] = min(sims) if sims else None
    out_stats["max_similarity_in_file"] = max(sims) if sims else None
    return scores, out_stats


DECISION_DIRS = ("02_questions_blind/normalized", "03_questions_merged")


def decision_files(root: Path = DATASET) -> list[Path]:
    """The adjudicated semantic-dedupe decisions of the blind lane, the source lane and the cross-bank
    merge (the only folders holding them; a full rglob over the dataset is slow on /mnt/c)."""
    return sorted(p for d in DECISION_DIRS for p in (root / d).glob("*semantic_dedupe_decisions.jsonl"))


def keep_both_pairs(root: Path = DATASET) -> tuple[list[tuple[Pair, str]], list[dict[str, Any]]]:
    """(pair, cluster_id) for every KEEP_BOTH decision, and a per-file summary."""
    pairs: list[tuple[Pair, str]] = []
    summary = []
    for path in decision_files(root):
        clusters_path = path.with_name(path.name.replace("dedupe_decisions", "duplicate_clusters"))
        members: dict[str, list[str]] = {}
        for c in read_jsonl(clusters_path):
            qs = c.get("questions", []) + c.get("source_questions", []) + c.get("blind_questions", [])
            members[c["cluster_id"]] = sorted(q["id"] for q in qs)
        counts = Counter()
        for d in read_jsonl(path):
            counts[d["decision"]] += 1
            if d["decision"] != "KEEP_BOTH":
                continue
            ids = members[d["cluster_id"]]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pairs.append((_pair(ids[i], ids[j]), d["cluster_id"]))
        summary.append({"file": str(path.relative_to(root.parent)), "decisions": dict(sorted(counts.items()))})
    return sorted(pairs), summary


# ---------------------------------------------------------------- graph --------------------------

class UnionFind:
    def __init__(self, items: Iterable[str] = ()):
        self.parent: dict[str, str] = {i: i for i in items}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self, min_size: int = 2) -> list[list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return sorted((sorted(v) for v in out.values() if len(v) >= min_size), key=lambda g: g[0])


def group_id(members: Sequence[str]) -> str:
    """Stable, readable group id: 'tg_' + the hex part of the smallest member id."""
    first = min(members)
    return "tg_" + (first[2:] if first.startswith("q_") else first)


@dataclass
class Evidence:
    """Every scored relation, computed once; thresholds are applied by `graph`."""
    train_ids: list[str]
    eval_ids: list[str]
    ngram_train: set[Pair]
    ngram_test: dict[str, set[str]]           # train id -> eval ids sharing an 8-gram
    semantic: dict[Pair, float]               # canonical pairs (train/eval) with candidate scores
    semantic_stats: dict[str, Any]
    lexical: dict[Pair, float]                # pairs not scored by `semantic`, cosine >= LEXICAL_FLOOR
    keep_both: list[tuple[Pair, str]]
    keep_both_summary: list[dict[str, Any]]


def collect(train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]) -> Evidence:
    semantic, sem_stats = semantic_scores(list(train) + list(test))
    lexical = lexical_scores(train, train, fit=train, same=True)
    lexical.update(lexical_scores(train, test, fit=train))
    kb, kb_summary = keep_both_pairs()
    return Evidence(train_ids=[r["id"] for r in train], eval_ids=[r["id"] for r in test],
                    ngram_train=ngram_pairs(train), ngram_test=ngram_hits(test, train),
                    semantic=semantic, semantic_stats=sem_stats, lexical=lexical, keep_both=kb,
                    keep_both_summary=kb_summary)


def graph(ev: Evidence, semantic_threshold: float = SEMANTIC_THRESHOLD,
          lexical_threshold: float | None = LEXICAL_THRESHOLD) -> dict[str, Any]:
    """Twin groups and test neighbours at the given thresholds (None disables a source)."""
    train, test = set(ev.train_ids), set(ev.eval_ids)
    edges: dict[Pair, list[str]] = defaultdict(list)
    neighbours: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    def add(pair: Pair, via: str) -> None:
        a, b = pair
        if a in train and b in train:
            edges[pair].append(via)
            return
        if a in test:
            a, b = b, a
        if a in train and b in test and b not in ev.ngram_test.get(a, ()):
            neighbours[a][b].append(via)   # pairs sharing an 8-gram are test twins already

    for p in ev.ngram_train:
        add(p, "8gram")
    if semantic_threshold is not None:
        for p, s in ev.semantic.items():
            if s >= semantic_threshold:
                add(p, f"semantic:{s:.4f}")
    if lexical_threshold is not None:
        for p, s in ev.lexical.items():
            if s >= lexical_threshold:
                add(p, f"lexical:{s:.4f}")
    for p, cid in ev.keep_both:
        add(p, f"keep_both:{cid}")
    uf = UnionFind()
    for a, b in edges:
        uf.union(a, b)
    groups = {group_id(g): g for g in uf.groups()}
    return {
        "edges": {p: sorted(v) for p, v in sorted(edges.items())},
        "groups": dict(sorted(groups.items())),
        "members": sorted({m for g in groups.values() for m in g}),
        "test_neighbours": {t: {e: sorted(v) for e, v in sorted(d.items())} for t, d in sorted(neighbours.items())},
    }


def edge_kind_counts(edges: Mapping[Pair, Sequence[str]]) -> dict[str, int]:
    counts = Counter()
    for via in edges.values():
        for kind in sorted({v.split(":")[0] for v in via}):
            counts[kind] += 1
    return dict(sorted(counts.items()))


def band_counts(ev: Evidence, kind: str) -> list[dict[str, Any]]:
    """Number of train-train pairs without a shared 8-gram in each BAND_READING band of `kind`."""
    scores = ev.semantic if kind == "semantic" else ev.lexical
    train = set(ev.train_ids)
    out = []
    for row in BAND_READING[kind]:
        lo, hi = row["band"]
        n = sum(1 for (a, b), s in scores.items()
                if a in train and b in train and lo <= s < hi and (a, b) not in ev.ngram_train)
        out.append({**row, "pairs_now": n, "twin_share": round(row["twins"] / row["pairs"], 3)})
    return out

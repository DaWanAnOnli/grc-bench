"""Word TF-IDF cosine nearest neighbours: a LEXICAL overlap signal (eval plan §6 audit, plan WS2).

This is not paraphrase detection. Two texts that ask the same thing in different words can score
low; the embedding signal (embed.py) and, later, the LLM rephrase detector [54] cover paraphrases.

Weights are the ones grcbench.splits.twins uses, so a lexical score means the same thing across
GRC-Bench: w(t, d) = (1 + ln tf) * (ln((1 + N) / (1 + df)) + 1), rows L2-normalised, with df counted
over the N passages of the training corpus. A benchmark term that never occurs in the corpus keeps
its weight in the item's norm (df = 0) and adds nothing to any dot product, so an item's score does
not depend on which other items are audited with it. Matrices are scipy.sparse float32, and the
corpus is scored in chunks so that memory stays small.
"""

from __future__ import annotations

import math
from array import array
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse

from .ngram import tokenize


class Tfidf:
    def __init__(self, df: Counter, n_docs: int):
        self.n_docs = n_docs
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        self.idf = np.array([self._idf(df[t]) for t in sorted(df)], dtype=np.float64)

    def _idf(self, df: int) -> float:
        return math.log((1 + self.n_docs) / (1 + df)) + 1.0

    @classmethod
    def fit(cls, token_lists: Iterable[Sequence[str]]) -> "Tfidf":
        df: Counter = Counter()
        n = 0
        for toks in token_lists:
            df.update(set(toks))
            n += 1
        return cls(df, n)

    def transform(self, token_lists: Iterable[Sequence[str]]) -> sparse.csr_matrix:
        indptr, indices, data = array("q", [0]), array("i"), array("f")
        oov_idf = self._idf(0)
        idf, vocab = self.idf.tolist(), self.vocab
        for toks in token_lists:
            row, norm2 = [], 0.0
            for term, tf in Counter(toks).items():
                col = vocab.get(term)
                w = (1.0 + math.log(tf)) * (oov_idf if col is None else idf[col])
                norm2 += w * w
                if col is not None:
                    row.append((col, w))
            norm = math.sqrt(norm2) or 1.0
            row.sort()
            indices.extend(c for c, _ in row)
            data.extend(w / norm for _, w in row)
            indptr.append(len(indices))
        return sparse.csr_matrix((np.frombuffer(data, dtype=np.float32), np.frombuffer(indices, dtype=np.int32),
                                  np.frombuffer(indptr, dtype=np.int64)), shape=(len(indptr) - 1, len(self.vocab)))

    def transform_texts(self, texts: Iterable[str]) -> sparse.csr_matrix:
        return self.transform(tokenize(t) for t in texts)


def cosine(queries: sparse.csr_matrix, docs: sparse.csr_matrix) -> np.ndarray:
    """Dense (n_queries, n_docs) cosine matrix of L2-normalised TF-IDF rows."""
    return np.asarray((docs @ queries.T).T.todense(), dtype=np.float32)


def _top(scores: np.ndarray, idx: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the k best (score desc, index asc)."""
    if scores.shape[1] > k:
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        scores = np.take_along_axis(scores, part, 1)
        idx = np.take_along_axis(idx, part, 1)
    order = np.lexsort((idx, -scores), axis=1) if scores.size else np.zeros_like(idx)
    return np.take_along_axis(scores, order, 1), np.take_along_axis(idx, order, 1)


def nearest(queries: sparse.csr_matrix, chunks: Iterable[sparse.csr_matrix], k: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """Top-k corpus rows by cosine for every query, streaming the corpus in row chunks.

    Returns (scores, indices), each (n_queries, k'), k' = min(k, rows seen), best first; indices
    count rows across all chunks."""
    n = queries.shape[0]
    best_s = np.zeros((n, 0), dtype=np.float32)
    best_i = np.zeros((n, 0), dtype=np.int64)
    offset = 0
    for chunk in chunks:
        m = chunk.shape[0]
        if m == 0:
            continue
        s = cosine(queries, chunk)
        i = np.broadcast_to(np.arange(offset, offset + m, dtype=np.int64), (n, m))
        best_s, best_i = _top(np.concatenate([best_s, s], 1), np.concatenate([best_i, i], 1), k)
        offset += m
    return best_s, best_i

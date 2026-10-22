"""Word 8-gram overlap between benchmark items and training text (eval plan §6 "8-word overlap").

Tokenizer, exactly as documented for the eval plan's count of near-twins: lowercase, replace every
non-word character (regex \\W) with a space, split on whitespace. A benchmark item has an 8-gram
twin in a corpus when at least one of its word 8-grams occurs in a corpus document. With this
tokenizer 26 of the 185 frozen test questions have a twin among the 1,668 training questions (eval
plan §6; checked by tests/test_contamination.py).

Only the benchmark side is indexed (a few thousand 8-grams), and corpus documents are streamed past
it, so even the ~10M words of teacher answers are scanned in constant memory. The methodology's
8-gram prompt decontamination (proposal §4.5) uses the same tokenizer.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

N = 8
_NONWORD = re.compile(r"\W")
_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase, every non-word character to a space, split on whitespace."""
    return _NONWORD.sub(" ", text.lower()).split()


def ngrams(tokens: Sequence[str], n: int = N) -> list[str]:
    """The word n-grams of a token list, each joined by single spaces, in order."""
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


@dataclass
class Overlap:
    """What one benchmark item shares with one corpus."""
    n_grams: int                                        # distinct n-grams of the item
    shared: set[str] = field(default_factory=set)       # those found in at least one document
    docs: Counter = field(default_factory=Counter)      # document index -> distinct shared n-grams

    @property
    def has_twin(self) -> bool:
        return bool(self.shared)

    @property
    def coverage(self) -> float:
        return len(self.shared) / self.n_grams if self.n_grams else 0.0

    def top_docs(self, k: int) -> list[tuple[int, int]]:
        """(document index, shared n-grams), most shared first, ties by document index."""
        return sorted(self.docs.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


class NgramIndex:
    """Index of the benchmark items' n-grams; `scan` streams a corpus past it."""

    def __init__(self, texts: Iterable[str], n: int = N):
        self.n = n
        self.item_grams = [set(ngrams(tokenize(t), n)) for t in texts]
        self.index: dict[str, list[int]] = defaultdict(list)
        for i, grams in enumerate(self.item_grams):
            for g in grams:
                self.index[g].append(i)
        self.first = {g.split(" ", 1)[0] for g in self.index}

    def scan(self, docs: Iterable[str]) -> list[Overlap]:
        out = [Overlap(len(g)) for g in self.item_grams]
        n, index, first = self.n, self.index, self.first
        for d, text in enumerate(docs):
            toks = tokenize(text)
            seen: set[str] = set()
            for i in range(len(toks) - n + 1):
                if toks[i] not in first:
                    continue
                g = " ".join(toks[i:i + n])
                if g in index and g not in seen:
                    seen.add(g)
                    for item in index[g]:
                        out[item].shared.add(g)
                        out[item].docs[d] += 1
        return out


def items_with_twin(queries: Sequence[str], pool: Iterable[str], n: int = N) -> list[int]:
    """Indices of the queries that share at least one word n-gram with any pool text."""
    return [i for i, o in enumerate(NgramIndex(queries, n).scan(pool)) if o.has_twin]


def locate(text: str, grams: Iterable[str], context: int = 10) -> tuple[int, int] | None:
    """Character span (with `context` words either side) of the first of `grams`, tried in order,
    that occurs in `text`; None if none can be located. Used for review snippets."""
    spans = [m.span() for m in _WORD.finditer(text)]
    words = [text[a:b].lower() for a, b in spans]
    for gram in grams:
        want = gram.split(" ")
        first = want[0]
        for i in range(len(words) - len(want) + 1):
            if words[i] == first and words[i:i + len(want)] == want:
                return spans[max(i - context, 0)][0], spans[min(i + len(want) + context, len(spans)) - 1][1]
    return None

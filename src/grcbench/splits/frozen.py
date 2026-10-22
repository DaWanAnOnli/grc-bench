"""The frozen-split algorithm `stratified-largest-remainder-sha256-v1` (eval plan §4.1, §4.2).

This algorithm froze the 185 Tier-K test questions (dataset/04_eval_frozen, manifest seed
"grc-agent-eval-v1-2026-08-13"). GRC-Bench reuses it for the Tier-K dev split and for the anchor and
CI subsets, so every split is drawn in the same checkable way.

Drawing n items from a pool:
  1. Strata are the exact raw values of `fields` (default origin, task_type, difficulty). Labels are
     not normalized.
  2. Quotas: a stratum s with N_s members in a quota population of size N gets floor(n*N_s/N). The
     n - sum(floor) leftover slots go one each to the strata with the largest remainders
     n*N_s/N - floor(n*N_s/N). Ties are broken by ascending stratum tuple, and remainders are
     compared as exact fractions.
  3. Within a stratum, members are ranked by ascending sha256(seed + "\\0" + id).hexdigest(), and the
     first `quota` members are taken.
  4. The chosen ids are returned in pool order (the order of `records`), which is how the frozen
     manifest lists them.

Capacity extension, used only when `eligible` excludes records: if a stratum's quota is larger than
its number of eligible members, the stratum takes all of its eligible members and is fixed. The
remaining slots are re-apportioned by step 2 over the strata not yet fixed, still using the
quota-population sizes N_s. This repeats until no stratum's quota exceeds its eligible count. With
no exclusions it reduces exactly to steps 1-4.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Collection, Iterable, Mapping, Sequence

from grcbench.common.io import read_json, read_jsonl, sha256_file
from grcbench.common.paths import DATASET, FROZEN

ALGORITHM = "stratified-largest-remainder-sha256-v1"
FIELDS = ("origin", "task_type", "difficulty")
CANONICAL = DATASET / "03_questions_merged" / "questions_canonical.jsonl"
MANIFEST = FROZEN / "eval_manifest.json"

Stratum = tuple[str, ...]


def stratum_of(record: Mapping[str, Any], fields: Sequence[str] = FIELDS) -> Stratum:
    """The exact raw stratum tuple of a record (a missing field raises KeyError)."""
    return tuple(str(record[f]) for f in fields)


def rank_key(seed: str, item_id: str) -> str:
    """Within-stratum rank: ascending SHA-256 hex digest of seed, NUL, id."""
    return hashlib.sha256(f"{seed}\0{item_id}".encode("utf-8")).hexdigest()


def largest_remainder(sizes: Mapping[Stratum, int], n: int) -> dict[Stratum, int]:
    """Step 2: proportional floors plus leftover slots by largest exact remainder, ties by stratum."""
    total = sum(sizes.values())
    if n < 0 or (n > 0 and total <= 0):
        raise ValueError(f"cannot apportion {n} slots over a population of {total}")
    if n == 0:
        return {s: 0 for s in sizes}
    quotas = {s: Fraction(n * k, total) for s, k in sizes.items()}
    alloc = {s: q.numerator // q.denominator for s, q in quotas.items()}
    leftover = n - sum(alloc.values())
    order = sorted(sizes, key=lambda s: (-(quotas[s] - alloc[s]), s))
    for s in order[:leftover]:
        alloc[s] += 1
    return alloc


def apportion(sizes: Mapping[Stratum, int], n: int,
              capacity: Mapping[Stratum, int] | None = None) -> tuple[dict[Stratum, int], list[list[Stratum]]]:
    """Step 2 with the capacity extension. Returns (allocation, strata fixed at capacity per round)."""
    cap = dict(sizes) if capacity is None else {s: int(capacity.get(s, 0)) for s in sizes}
    if sum(cap.values()) < n:
        raise ValueError(f"only {sum(cap.values())} eligible items for {n} slots")
    fixed: dict[Stratum, int] = {}
    rounds: list[list[Stratum]] = []
    while True:
        free = {s: k for s, k in sizes.items() if s not in fixed}
        alloc = largest_remainder(free, n - sum(fixed.values()))
        over = sorted(s for s, a in alloc.items() if a > cap[s])
        if not over:
            alloc.update(fixed)
            return alloc, rounds
        rounds.append(over)
        for s in over:
            fixed[s] = cap[s]


@dataclass(frozen=True)
class Allocation:
    """Result of `allocate`: chosen ids in pool order plus the per-stratum bookkeeping."""
    ids: list[str]
    seed: str
    n: int
    fields: tuple[str, ...]
    strata: list[dict[str, Any]] = field(default_factory=list)
    capped_rounds: list[list[Stratum]] = field(default_factory=list)

    def by_stratum(self) -> dict[Stratum, dict[str, Any]]:
        return {tuple(row["stratum"]): row for row in self.strata}


def allocate(records: Sequence[Mapping[str, Any]], n: int, seed: str,
             fields: Iterable[str] = FIELDS,
             quota_population: Sequence[Mapping[str, Any]] | None = None,
             eligible: Collection[str] | None = None) -> Allocation:
    """Draw n ids from `records` (see the module docstring).

    quota_population  records whose stratum sizes set the quotas (default: `records`)
    eligible          ids that may be chosen (default: every record); others only count toward quotas
    """
    fields = tuple(fields)
    ids_in_order = [str(r["id"]) for r in records]
    if len(set(ids_in_order)) != len(ids_in_order):
        raise ValueError("duplicate ids in the pool")
    population = records if quota_population is None else quota_population
    sizes = Counter(stratum_of(r, fields) for r in population)
    allowed = None if eligible is None else set(eligible)
    pool_count: Counter[Stratum] = Counter()
    members: dict[Stratum, list[str]] = defaultdict(list)
    for r in records:
        s = stratum_of(r, fields)
        pool_count[s] += 1
        if allowed is None or str(r["id"]) in allowed:
            members[s].append(str(r["id"]))
    capacity = {s: len(members.get(s, [])) for s in sizes}
    alloc, rounds = apportion(sizes, n, capacity)
    chosen: set[str] = set()
    for s, k in alloc.items():
        chosen.update(sorted(members.get(s, []), key=lambda i: rank_key(seed, i))[:k])
    total = sum(sizes.values())
    rows = [{"stratum": list(s), "quota_population": sizes[s], "pool": pool_count.get(s, 0),
             "eligible": capacity[s], "proportional_quota": round(n * sizes[s] / total, 6),
             "allocated": alloc[s]} for s in sorted(sizes)]
    return Allocation(ids=[i for i in ids_in_order if i in chosen], seed=seed, n=n, fields=fields,
                      strata=rows, capped_rounds=rounds)


def reproduce_frozen() -> dict[str, Any]:
    """Regenerate the frozen eval ids from questions_canonical.jsonl and compare with the manifest."""
    manifest = read_json(MANIFEST)
    canonical = read_jsonl(CANONICAL)
    alloc = allocate(canonical, int(manifest["eval_count"]), manifest["random_seed"],
                     tuple(manifest["stratification_fields"]))
    eval_file = [r["id"] for r in read_jsonl(FROZEN / "eval_questions.jsonl")]
    train_file = [r["id"] for r in read_jsonl(FROZEN / "train_questions.jsonl")]
    chosen = set(alloc.ids)
    return {
        "algorithm": manifest["split_algorithm"],
        "algorithm_matches": manifest["split_algorithm"] == ALGORITHM,
        "seed": manifest["random_seed"],
        "source_sha256": manifest["source_canonical_sha256"],
        "source_sha256_matches": sha256_file(CANONICAL) == manifest["source_canonical_sha256"],
        "reproduced_ids": len(alloc.ids),
        "manifest_ids": len(manifest["eval_ids"]),
        "ids_match_in_order": alloc.ids == manifest["eval_ids"],
        "eval_file_matches": eval_file == manifest["eval_ids"],
        "train_file_is_complement": train_file == [r["id"] for r in canonical if r["id"] not in chosen],
    }


def frozen_ok(report: Mapping[str, Any]) -> bool:
    return all(report[k] for k in ("algorithm_matches", "source_sha256_matches", "ids_match_in_order",
                                   "eval_file_matches", "train_file_is_complement"))

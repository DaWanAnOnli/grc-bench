"""The replacement items: the Tier-F and Tier-A tasks that replace the ones the review rejected
(pre-registration §10), and which of them are scored.

Each group of replacements has a number of slots, one per rejected task it replaces. The replacements of a
group are numbered before the review; the first ones the review accepts, in id order, fill the slots. An
accepted replacement beyond them is a reserve: it keeps its record and is not scored, like a rejected task. A
group is settled only when every replacement in it is decided; until then an undecided replacement is not part
of the benchmark.

The groups are plain data here, so that a benchmark run's code holds no item content; a test pins them to
grcbench.tier_f.generate.replacement_plan and grcbench.tier_a.replacements.SLOTS.

    groups(sheet) -> [(ids in order, slots)]       the groups of tier_f or tier_a
    pick(groups, finals) -> {reserve ids}          pure: the accepted ids beyond each settled group's slots
    reserve(sheet, reviews_dir=None) -> {ids}      pick over the decided records
    undecided(sheet, reviews_dir=None) -> {ids}    replacements with no decided record yet
    all_ids() -> {ids}                             every replacement id (H1b counts the original pool only)
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

ACCEPTED = ("approve", "approve with edits")

_GROUPS: dict[str, list[tuple[list[str], int]]] = {
    "tier_f": [([f"gbf-test-{i}" for i in range(151, 175)], 21),
               ([f"gbf-dev-{i}" for i in range(46, 49)], 2)],
    "tier_a": [([f"gba-test-{i}" for i in (31, 32, 33, 36)], 3),        # knowledge, test
               ([f"gba-dev-{i}" for i in (13,)], 1),                     # knowledge, dev
               ([f"gba-test-{i}" for i in (34, 35, 37)], 2),            # save_block, test
               ([f"gba-dev-{i}" for i in (14,)], 1)],                    # save_block, dev
}


def groups(sheet: str) -> list[tuple[list[str], int]]:
    return [(list(ids), slots) for ids, slots in _GROUPS.get(sheet, [])]


def pick(groups_: Sequence[tuple[Sequence[str], int]], finals: Mapping[str, str]) -> set[str]:
    out: set[str] = set()
    for ids, slots in groups_:
        if not all(i in finals for i in ids):
            continue
        accepted = [i for i in ids if finals[i] in ACCEPTED]
        out |= set(accepted[slots:])
    return out


def _finals(sheet: str, reviews_dir: Path | None) -> dict[str, str]:
    from grcbench.review.records import decided
    return {k[0]: r["final"] for k, r in decided(sheet, reviews_dir=reviews_dir).items()}


def reserve(sheet: str, reviews_dir: Path | None = None) -> set[str]:
    return pick(groups(sheet), _finals(sheet, reviews_dir))


def undecided(sheet: str, reviews_dir: Path | None = None) -> set[str]:
    finals = _finals(sheet, reviews_dir)
    return {i for ids, _ in groups(sheet) for i in ids if i not in finals}


def all_ids() -> set[str]:
    return {i for sheet in _GROUPS for ids, _ in groups(sheet) for i in ids}

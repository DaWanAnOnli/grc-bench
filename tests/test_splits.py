"""Tests for WS1: the frozen-split algorithm, twin groups and the Tier-K dev split (eval plan §4.1, §6)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from grcbench.common.io import read_json, read_jsonl
from grcbench.common.paths import FROZEN, REPO
from grcbench.splits import dev, frozen, release, twins


# ---------------------------------------------------------------- frozen algorithm ---------------

def test_frozen_split_reproduces_manifest_exactly():
    report = frozen.reproduce_frozen()
    assert report["source_sha256_matches"], "questions_canonical.jsonl changed since the freeze"
    assert report["algorithm_matches"]
    assert report["reproduced_ids"] == report["manifest_ids"] == 185
    assert report["ids_match_in_order"]
    assert report["eval_file_matches"] and report["train_file_is_complement"]
    assert frozen.frozen_ok(report)


def test_largest_remainder_floors_then_remainders_with_stratum_ties():
    alloc = frozen.largest_remainder({("c",): 1, ("a",): 1, ("b",): 1}, 2)
    assert alloc == {("a",): 1, ("b",): 1, ("c",): 0}            # equal remainders: stratum order
    alloc = frozen.largest_remainder({("x",): 7, ("y",): 2, ("z",): 1}, 5)
    assert alloc == {("x",): 4, ("y",): 1, ("z",): 0}            # 3.5/1.0/0.5 -> floors 3,1,0 + x
    assert sum(frozen.largest_remainder({("a",): 13, ("b",): 29, ("c",): 58}, 17).values()) == 17


def _pool():
    recs = []
    for s, k in (("A", 50), ("B", 30), ("C", 20)):
        recs += [{"id": f"{s}{i:03d}", "origin": s, "task_type": "t", "difficulty": "d"} for i in range(k)]
    return recs


def test_allocate_redistributes_capped_stratum_by_largest_remainder():
    recs = _pool()
    eligible = [r["id"] for r in recs if r["origin"] != "C"] + ["C000"]
    alloc = frozen.allocate(recs, 10, "seed", eligible=eligible)
    by = {row["stratum"][0]: row for row in alloc.strata}
    # quotas 5/3/2; C has 1 eligible -> fixed at 1; 9 slots over A, B (50:30) = 5.625/3.375 -> 6/3
    assert (by["A"]["allocated"], by["B"]["allocated"], by["C"]["allocated"]) == (6, 3, 1)
    assert alloc.capped_rounds == [[("C", "t", "d")]]
    assert len(alloc.ids) == 10 and set(alloc.ids) <= set(eligible)
    expect_a = sorted((i for i in eligible if i.startswith("A")), key=lambda i: frozen.rank_key("seed", i))[:6]
    assert [i for i in alloc.ids if i.startswith("A")] == sorted(expect_a)   # pool order == id order here


def test_allocate_quota_population_and_eligible_defaults():
    recs = _pool()
    plain = frozen.allocate(recs, 10, "s")
    assert plain.ids == frozen.allocate(recs, 10, "s", eligible=[r["id"] for r in recs]).ids
    assert plain.capped_rounds == []
    assert [i for i in plain.ids] == [r["id"] for r in recs if r["id"] in set(plain.ids)]   # pool order
    with pytest.raises(ValueError):
        frozen.allocate(recs, 10, "s", eligible=["A000", "B000"])
    with pytest.raises(ValueError):
        frozen.allocate(recs + recs[:1], 10, "s")                                       # duplicate id


# ---------------------------------------------------------------- twin helpers -------------------

def test_tokenizer_and_ngrams():
    assert twins.tokens("Hello, GNU-Radio's  World_1!") == ["hello", "gnu", "radio", "s", "world_1"]
    assert twins.ngrams("one two three", n=8) == set()
    assert len(twins.ngrams("a b c d e f g h i", n=8)) == 2


def test_union_find_groups_and_ids():
    uf = twins.UnionFind()
    for a, b in (("q_b", "q_c"), ("q_c", "q_d"), ("q_x", "q_y")):
        uf.union(a, b)
    uf.find("q_lonely")
    assert uf.groups() == [["q_b", "q_c", "q_d"], ["q_x", "q_y"]]
    assert twins.group_id(["q_d", "q_b"]) == "tg_b"


def test_lexical_scores_skip_pairs_the_semantic_file_scored():
    recs = [{"id": "q1", "origin": "source", "question": "What is the cyclic prefix in OFDM?"},
            {"id": "q2", "origin": "source", "question": "What is the cyclic prefix in OFDM?"},
            {"id": "q3", "origin": "blind", "question": "What is the cyclic prefix in OFDM?"},
            {"id": "q4", "origin": "blind", "question": "How does a Costas loop recover the carrier?"}]
    scores = twins.lexical_scores(recs, recs, fit=recs, floor=0.5, same=True)
    assert scores == {("q1", "q3"): 1.0, ("q2", "q3"): 1.0}


# ---------------------------------------------------------------- twin evidence on the data ------

@pytest.fixture(scope="module")
def evidence():
    train, test = read_jsonl(FROZEN / "train_questions.jsonl"), read_jsonl(FROZEN / "eval_questions.jsonl")
    return twins.collect(train, test)


def test_test_twins_match_eval_plan_count(evidence):
    # eval plan §6: 26 of the 185 test questions share an 8-gram with a training question
    assert len({e for v in evidence.ngram_test.values() for e in v}) == 26
    assert len(evidence.ngram_test) == 59


def test_semantic_candidates_map_to_canonical_ids(evidence):
    st = evidence.semantic_stats
    assert st["rows"] == 5720
    assert st["mapped"] + st["dropped_same_canonical_question"] + \
        st["dropped_endpoint_removed_by_adjudicated_dedupe"] == st["rows"]
    ids = set(evidence.train_ids) | set(evidence.eval_ids)
    assert all(a in ids and b in ids for a, b in evidence.semantic)


def test_band_reading_matches_current_data(evidence):
    """The recorded threshold reading must refer to exactly the pairs the data now contains."""
    for kind in ("semantic", "lexical"):
        for row in twins.band_counts(evidence, kind):
            assert row["pairs_now"] == row["pairs"], (kind, row)


def test_keep_both_decisions_found(evidence):
    files = {s["file"].split("/")[-1] for s in evidence.keep_both_summary}
    assert files == {"blind_semantic_dedupe_decisions.jsonl", "cross_bank_semantic_dedupe_decisions.jsonl",
                     "semantic_dedupe_decisions.jsonl"}
    assert len(evidence.keep_both) == 10


# ---------------------------------------------------------------- the dev split ------------------

@pytest.fixture(scope="module")
def built():
    files = dev.build()
    return files, json.loads(files[dev.OUT_DEV]), json.loads(files[dev.OUT_GROUPS])


def test_dev_split_shape_and_disjointness(built):
    _, d, _ = built
    train = {r["id"] for r in read_jsonl(FROZEN / "train_questions.jsonl")}
    test = set(read_json(FROZEN / "eval_manifest.json")["eval_ids"])
    ids = d["dev_ids"]
    assert len(ids) == len(set(ids)) == 107
    assert set(ids) <= train and not set(ids) & test
    assert len(d["anchor_ids"]) == 60 and set(d["anchor_ids"]) <= set(ids)
    assert len(d["ci_ids"]) == 20 and set(d["ci_ids"]) <= set(ids)
    assert d["version"] == 2
    assert d["seeds"] == {"anchor": "grc-bench-anchor-v2-2026-11-02", "ci": "grc-bench-ci-v2-2026-11-02",
                          "frozen_test": "grc-agent-eval-v1-2026-08-13"}


def test_dev_is_the_eligible_part_of_the_release_validation(built):
    _, d, _ = built
    train = [r["id"] for r in read_jsonl(FROZEN / "train_questions.jsonl")]
    val = set(release.validation_question_ids())
    src = d["dev_source"]
    excluded = set(src["excluded_ids"])
    assert d["dev_ids"] == [i for i in train if i in val and i not in excluded]
    assert excluded == val - set(d["dev_ids"]) and len(val) == src["validation_questions"] == 148
    assert all(r in dev.REASONS for rs in src["excluded_ids"].values() for r in rs)
    assert sum(src["excluded_by_reason"].values()) >= len(excluded)
    assert src["release"] == release.describe()
    assert not set(d["dev_ids"]) & release.training_question_ids()


def test_dev_has_no_twin_test_twin_neighbour_or_all_reject(built, evidence):
    _, d, groups = built
    dev_ids = set(d["dev_ids"])
    edge_ids = {x for a, b, _ in groups["edges"] for x in (a, b)}
    assert not dev_ids & edge_ids, "a dev question has a twin edge to another training question"
    assert not dev_ids & {m for g in groups["groups"].values() for m in g}
    assert not dev_ids & set(evidence.ngram_test)
    assert not dev_ids & set(d["test_neighbours"]["by_training_id"])
    assert len(d["all_reject_ids"]) == 8 and not dev_ids & set(d["all_reject_ids"])
    # independent check of the all-REJECT rule
    status = dev.judgment_status(set(d["all_reject_ids"]))
    assert all(set(c) == {"REJECT"} for c in status.values())


def test_dev_strata_count_the_dev_questions(built):
    _, d, _ = built
    train = read_jsonl(FROZEN / "train_questions.jsonl")
    by_id = {r["id"]: r for r in train}
    got = Counter(frozen.stratum_of(by_id[i]) for i in d["dev_ids"])
    rows = {tuple(s["stratum"]): s for s in d["strata"]}
    assert {s: r["dev"] for s, r in rows.items() if r["dev"]} == dict(got)
    assert all(r["dev"] <= r["validation"] <= r["training_pool"] and r["anchor"] <= r["dev"]
               and r["ci"] <= r["dev"] for r in rows.values())
    assert sum(r["training_pool"] for r in rows.values()) == len(train)
    assert d["composition"]["pools"]["dev"] == len(d["dev_ids"])


def test_handoff_files(built):
    files, d, _ = built
    lines = files[dev.OUT_EXCLUDE].splitlines()
    assert all(x.startswith("#") for x in lines[:3])
    assert [x for x in lines if not x.startswith("#")] == sorted(d["dev_ids"])
    readme = files[dev.OUT_README]
    for word in ("SFT", "DPO", "RAFT", "SliCK", "where possible", "validation", release.RELEASE_SHA256):
        assert word in readme
    tsv_ids = [x.split("\t")[0] for x in files[dev.OUT_TWINS_TSV].splitlines()[5:]]
    assert len(tsv_ids) == len(set(tsv_ids))
    assert set(d["test_twins"]["by_training_id"]) <= set(tsv_ids)


def test_no_test_question_text_in_any_output(built):
    """Frozen README: test questions are never handed to training-side people or models."""
    files, _, _ = built
    test_texts = [r["question"] for r in read_jsonl(FROZEN / "eval_questions.jsonl")]
    for rel, text in files.items():
        leaked = [q for q in test_texts if q in text]
        assert not leaked, f"{rel} contains test question text"


def test_review_sheet_rows_match_band_reading(built):
    files, d, _ = built
    rows = files[dev.OUT_SHEET].splitlines()
    expected = sum(b["pairs"] for k in ("semantic", "lexical") for b in twins.BAND_READING[k])
    assert len(rows) - 1 == expected


@pytest.mark.slow
def test_split_dev_twice_is_byte_identical(built, tmp_path: Path):
    files, _, _ = built
    for name in ("a", "b"):
        assert dev.main(["--out", str(tmp_path / name)]) == 0
    for rel in files:
        a, b = (tmp_path / "a" / rel).read_bytes(), (tmp_path / "b" / rel).read_bytes()
        assert a == b, f"{rel} differs between runs"
        assert a == files[rel].encode("utf-8"), f"{rel} differs from the in-memory build"


def test_repository_outputs_are_current(built):
    """The committed outputs equal a fresh build (skipped before the first `./gb split-dev`)."""
    files, _, _ = built
    target = REPO / dev.OUT_DEV
    if not target.exists():
        pytest.skip("run ./gb split-dev first")
    for rel, text in files.items():
        assert (REPO / rel).read_text(encoding="utf-8") == text, f"{rel} is stale: re-run ./gb split-dev"

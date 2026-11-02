"""Tests for `./gb tier-k-items`: the Tier-K test and dev item records (eval plan §4.1, §4.2, §5.1, §6, §9)."""

from __future__ import annotations

import copy
import csv
import io
from collections import Counter
from pathlib import Path

import pytest

from grcbench.common.canary import CANARY
from grcbench.common.io import read_json, read_jsonl, sha256_file
from grcbench.common.paths import DATA, FROZEN, REPO, SPLITS
from grcbench.common.schemas import validate
from grcbench.splits import twins
from grcbench.tier_k import items

OUTPUTS = (items.OUT_TEST, items.OUT_DEV, items.OUT_SHEET, items.OUT_REPORT)


@pytest.fixture(scope="module")
def built() -> items.Build:
    return items.build()


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> tuple[Path, Path]:
    """Two independent `./gb tier-k-items --out` runs."""
    roots = tmp_path_factory.mktemp("a"), tmp_path_factory.mktemp("b")
    for root in roots:
        assert items.main(["--out", str(root)]) == 0
    return roots


@pytest.fixture(scope="module")
def frozen_records() -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(FROZEN / "eval_questions.jsonl") + read_jsonl(FROZEN / "train_questions.jsonl")}


# ---------------------------------------------------------------- counts, ids, schema ------------

def test_counts_and_ids_match_the_manifests(built):
    manifest = read_json(FROZEN / "eval_manifest.json")
    dev = read_json(SPLITS / "tier_k_dev.json")
    assert len(built.test) == 185 and [r["id"] for r in built.test] == manifest["eval_ids"]
    assert len(built.dev) == 107 and [r["id"] for r in built.dev] == dev["dev_ids"]
    assert {r["split"] for r in built.test} == {"test"} and {r["split"] for r in built.dev} == {"dev"}
    assert not {r["id"] for r in built.test} & {r["id"] for r in built.dev}


def test_every_record_validates_and_carries_the_canary(built):
    for rec in built.test + built.dev:
        assert validate(rec) == [], rec["id"]
        assert rec["canary"] == CANARY and rec["tier"] == "K" and rec["version"] == 1
        assert rec["review"] == {"status": "not-required", "notes": items.REVIEW_NOTES[rec["split"]]}
        assert rec["item_kind"] == rec["source"]["kind"] == items.KIND[rec["split"]]
        assert rec["license"].startswith("licence: as in GRC_dataset_20261007 (Qoherent); to be confirmed before release.")
        assert ("source-lane" in rec["license"]) == (rec["origin"] == "source")
        assert rec["provenance"]["generator"] == "GRC_dataset_20261007"


def test_source_points_at_the_hashed_dataset_file(built):
    expected = {"test": "GRC_dataset_20261007/dataset/04_eval_frozen/eval_questions.jsonl",
                "dev": "GRC_dataset_20261007/dataset/04_eval_frozen/train_questions.jsonl"}
    for split, recs in (("test", built.test), ("dev", built.dev)):
        refs = {(r["source"]["ref"], r["source"]["sha256"]) for r in recs}
        assert refs == {(expected[split], sha256_file(DATA / expected[split]))}


def test_question_fields_are_copied_unchanged(built, frozen_records):
    labels = {r["id"]: r for r in read_jsonl(SPLITS / "task_type_labels.jsonl")}
    for rec in built.test + built.dev:
        src = frozen_records[rec["id"]]
        for f in ("question", "difficulty", "origin", "primary_topic"):
            assert rec[f] == src[f], (rec["id"], f)
        assert rec["task_type_raw"] == src["task_type"] == labels[rec["id"]]["raw"]
        assert rec["task_type"] == labels[rec["id"]]["clean"]
        assert rec["task_type_labeller"] == labels[rec["id"]]["labeller"]


def test_generator_version_is_the_canonical_dataset_hash(built):
    canonical = sha256_file(items.CANONICAL)
    assert canonical == read_json(FROZEN / "eval_manifest.json")["source_canonical_sha256"]
    versions = {r["provenance"]["generator_version"] for r in built.test + built.dev}
    assert versions == {f"questions_canonical.jsonl sha256:{canonical[:16]}"}


# ---------------------------------------------------------------- writers (§5.1) -----------------

def test_writer_families_match_eval_plan_table_5(built, frozen_records):
    # eval plan §5.1: Mistral wrote 145 of the 185 test questions; gemma4 11, gpt-oss 11, qwen 10, nemotron 8
    fam = Counter(r["provenance"]["writer_family"] for r in built.test)
    assert fam == {"Mistral": 145, "Gemma": 11, "gpt-oss": 11, "Qwen": 10, "Nemotron": 8}
    for rec in built.test + built.dev:
        src = frozen_records[rec["id"]]
        p = rec["provenance"]
        if rec["origin"] == "source":
            assert p["drafted_by"].endswith(f"({src['provenance'][0]['source_file']})")
        else:
            assert p["drafted_by"].endswith(f"(generator {src['generator']}, model {src['model']})")
        assert p["drafted_by"].split(" (")[0] in {w.split(" (")[0] for w in p["writers"]}
        assert len(p["writers"]) == len({x["source_file"] for x in src["provenance"]})
    drafted = {r["provenance"]["drafted_by"] for r in built.test + built.dev}
    assert "Mistral Large (source2_mistral_large.json)" in drafted
    assert "Gemma 4 (generator gemma4, model gemma)" in drafted


def test_unknown_writers_stop_the_build():
    with pytest.raises(SystemExit):
        items.writer_info({"id": "q_x", "origin": "source", "provenance": [{"source_file": "source9_llama.json"}]})
    with pytest.raises(SystemExit):
        items.writer_info({"id": "q_x", "origin": "blind", "generator": "llama", "model": "llama",
                           "provenance": [{"generator": "llama", "model": "llama", "source_file": "l.json"}]})
    merged = items.writer_info({"id": "q_x", "origin": "source", "provenance": [
        {"source_file": "source4_mistral_vibe.json"}, {"source_file": "source4_mistral_large2.json"}]})
    assert merged == {"drafted_by": "Mistral Vibe (source4_mistral_vibe.json)", "writer_family": "Mistral",
                      "writers": ["Mistral Large 2 (source4_mistral_large2.json)", "Mistral Vibe (source4_mistral_vibe.json)"]}


# ---------------------------------------------------------------- near twins (§6) ----------------

def test_near_twin_split_matches_eval_plan_and_dev_split(built):
    near = [r for r in built.test if r["near_twin_in_train"]]
    assert len(near) == 26                                                  # eval plan §6
    d = read_json(SPLITS / "tier_k_dev.json")
    expected = items.invert(d["test_twins"]["by_training_id"])
    assert {r["id"]: r["near_twin_train_ids"] for r in near} == expected
    assert all(r["near_twin_train_ids"] == [] for r in built.test if not r["near_twin_in_train"])
    # independent recount with the 8-gram tokenizer
    train = read_jsonl(FROZEN / "train_questions.jsonl")
    grams = [(t["id"], twins.ngrams(t["question"])) for t in train]
    for r in near[:5]:
        mine = twins.ngrams(r["question"])
        assert sorted(i for i, g in grams if g & mine) == r["near_twin_train_ids"]
    # flagged neighbours are listed, never counted as near-twins
    assert {r["id"]: r["train_neighbour_ids"] for r in built.test if r["train_neighbour_ids"]} == \
        items.invert(d["test_neighbours"]["by_training_id"])
    assert not {x for r in built.test for x in r["near_twin_train_ids"] + r["train_neighbour_ids"]} & set(d["dev_ids"])


def test_dev_records_have_no_twin_left_in_training(built):
    assert not any(r["near_twin_in_train"] or r["near_twin_train_ids"] or r["train_neighbour_ids"] for r in built.dev)
    assert built.dev_ngram == {} and built.dev_other == {}
    d = read_json(SPLITS / "tier_k_dev.json")
    for r in built.dev:
        assert r["subsets"] == [s for s, k in (("anchor", "anchor_ids"), ("ci", "ci_ids")) if r["id"] in d[k]]
    assert sum("anchor" in r["subsets"] for r in built.dev) == 60
    assert sum("ci" in r["subsets"] for r in built.dev) == 20
    assert all("subsets" not in r for r in built.test)


# ---------------------------------------------------------------- clusters (§9) ------------------

def test_dev_cluster_ids_follow_the_twin_group_rule(built):
    groups = read_json(SPLITS / "tier_k_twin_groups.json")["groups"]
    member_of = {m: gid for gid, ms in groups.items() for m in ms}
    for r in built.dev:
        assert r["cluster_id"] == member_of.get(r["id"], r["id"])
    assert all(r["cluster_id"] == r["id"] for r in built.dev)                 # dev excludes group members


def test_test_clusters_are_components_of_the_twin_evidence(built):
    by_cluster: dict[str, list[str]] = {}
    for r in built.test:
        by_cluster.setdefault(r["cluster_id"], []).append(r["id"])
    for cid, members in by_cluster.items():
        if len(members) == 1:
            assert cid == members[0]
        else:
            assert cid == items.TEST_CLUSTER_PREFIX + min(members).removeprefix("q_")
    cluster = {r["id"]: r["cluster_id"] for r in built.test}
    for (a, b), via in built.links.items():
        assert cluster[a] == cluster[b] and via
    # every test<->test 8-gram pair (template family) is in one cluster
    for a, b in twins.ngram_pairs(read_jsonl(FROZEN / "eval_questions.jsonl")):
        assert cluster[a] == cluster[b] == built.ngram_only[a] == built.ngram_only[b]
    assert sum(len(m) > 1 for m in by_cluster.values()) == 5
    assert len({c for c in built.ngram_only.values() if c.startswith(items.TEST_CLUSTER_PREFIX)}) == 3


def test_components_on_toy_pairs():
    got = items.components(["q_c", "q_a", "q_b", "q_d"], [("q_b", "q_c")])
    assert got == {"q_a": "q_a", "q_b": "tc_b", "q_c": "tc_b", "q_d": "q_d"}


def test_review_sheet_lists_every_multi_member_cluster(built):
    rows = list(csv.DictReader(io.StringIO(items.sheet(built))))
    multi = sorted(r["id"] for r in built.test if r["cluster_id"].startswith(items.TEST_CLUSTER_PREFIX))
    assert sorted(r["id"] for r in rows) == multi and len(multi) == 10
    assert all(r["linked_to (evidence)"] for r in rows)


# ---------------------------------------------------------------- guards ---------------------------

def test_inconsistent_inputs_are_refused():
    inp = items.load()
    items.check_inputs(inp)                                                   # current inputs are fine
    stale = copy.deepcopy(inp)
    stale.dev_split["sources"] = [dict(s, sha256="0" * 64) for s in stale.dev_split["sources"]]
    with pytest.raises(SystemExit):
        items.check_inputs(stale)
    reordered = copy.deepcopy(inp)
    reordered.test = reordered.test[::-1]
    with pytest.raises(SystemExit):
        items.check_inputs(reordered)


def test_contamination_loader_sees_the_same_185_test_documents(runs, monkeypatch):
    from grcbench.contamination import corpora
    monkeypatch.setattr(corpora, "ITEMS", runs[0] / "items")
    corpus = corpora.benchmark_tier("tier_k_test")
    assert corpus is not None and len(corpus.docs) == 185                     # no double counting


# ---------------------------------------------------------------- determinism ----------------------

def test_two_runs_are_byte_identical(runs):
    a, b = runs
    for rel in OUTPUTS:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), f"{rel} differs between runs"
    lines = (a / items.OUT_TEST).read_bytes().split(b"\n")
    assert lines[-1] == b"" and b"\r" not in (a / items.OUT_TEST).read_bytes()
    assert len(read_jsonl(a / items.OUT_TEST)) == 185 and len(read_jsonl(a / items.OUT_DEV)) == 107


def test_repository_outputs_are_current(runs):
    """The repository's files equal a fresh build (skipped before the first `./gb tier-k-items`)."""
    if not (REPO / items.OUT_TEST).exists():
        pytest.skip("run ./gb tier-k-items first")
    for rel in OUTPUTS:
        assert (REPO / rel).read_bytes() == (runs[0] / rel).read_bytes(), f"{rel} is stale: re-run ./gb tier-k-items"

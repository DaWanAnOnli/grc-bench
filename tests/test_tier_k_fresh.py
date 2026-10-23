"""Fresh Tier-K GNU Radio question sets (tier_k/blockgen.py): seeds, composition, pairing, no shared
instance with the reference set, wide instance spaces, nothing written into the repository, and the
private seed whose value is never printed or stored in the repository."""

from __future__ import annotations

import json
import os
import stat
from collections import Counter
from pathlib import Path

import pytest

from grcbench.common.paths import REPO
from grcbench.common.schemas import validate
from grcbench.stats import functions as fn
from grcbench.tier_k import blockgen as bg


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


REF_KEYS = _jsonl(REPO / bg.OUT_ANSWERS)
REF_BLOCKS = {k["block_id"] for k in REF_KEYS}


def _effective(spec: dict) -> str:
    """What decides the answer: for a value question on one input, the setting and the asked item."""
    if "inputs" in spec and len(spec["inputs"]) == 1 and "k" in spec:
        return json.dumps([spec["params"], spec["inputs"][0][spec["k"] - 1]], sort_keys=True, default=str)
    return json.dumps([spec["params"], spec.get("inputs"), spec.get("count_input"), spec.get("k"),
                       spec.get("port")], sort_keys=True, default=str)


def test_seeds_are_deterministic_and_distinct():
    a, b = bg._rng("s1", "blocks_add_xx", 0), bg._rng("s1", "blocks_add_xx", 0)
    assert [a.random() for _ in range(3)] == [b.random() for _ in range(3)]
    assert bg._qid("s1", "blocks_add_xx", 0) != bg._qid("s2", "blocks_add_xx", 0)
    assert bg._qid(bg.REFERENCE_SEED, "blocks_add_xx", 0) == bg._qid(bg.REFERENCE_SEED, "blocks_add_xx", 0)
    assert bg.seed_hash("s1") != bg.seed_hash("s2") and len(bg.seed_hash("s1")) == 64


@pytest.mark.parametrize("setting", [s for s in bg.SETTINGS if s.block in REF_BLOCKS], ids=lambda s: s.block)
def test_every_question_block_has_a_wide_instance_space(setting):
    seen = {_effective(setting.make(bg._rng(f"space-{i}", setting.block, 0))) for i in range(200)}
    assert len(seen) >= 30, (setting.block, len(seen))


def test_fresh_candidates_repeat_no_reference_instance():
    exclude = bg.reference_keys()
    draws = {b: 6 for b in REF_BLOCKS}
    fresh, _ = bg._candidates(set(), "a-fresh-seed", draws=draws, exclude=exclude)
    assert {c["block"] for c in fresh} == REF_BLOCKS
    for c in fresh:
        assert not bg._variant_keys(c["spec"]) & exclude[c["block"]], c["block"]


def _q(block, variant, kind="output_value", difficulty="basic"):
    return {"block": block, "variant": variant, "kind": kind, "family": block, "yaml_file": f"{block}.block.yml",
            "yaml_sha256": "0" * 64, "difficulty": difficulty, "answer": 2.0, "integer": False,
            "tolerance": {"tolerance": 0.01, "abs_tolerance": 0.005}, "setting": {"x": "1"}, "stated_items": 4,
            "question": f"In GNU Radio 3.10, take block {block} variant {variant}. What is output item 1?",
            "evidence": "GNU Radio 3.10.12 run (twice, identical): 2"}


def test_select_like_copies_the_composition_and_pairs_every_slot():
    slots = {("blk_a", "output_value"): ["r2", "r1"], ("blk_b", "output_count"): ["r3"]}
    qs = [_q("blk_a", v) for v in range(5)] + [_q("blk_b", v, "output_count") for v in range(3)]
    chosen = bg.select_like(qs, "seed-x", slots)
    assert Counter((q["block"], q["kind"]) for q in chosen) == {k: len(v) for k, v in slots.items()}
    assert sorted(q["pairs_with"] for q in chosen) == ["r1", "r2", "r3"]
    assert chosen == bg.select_like(list(reversed(qs)), "seed-x", slots)          # order-independent
    with pytest.raises(RuntimeError, match="slots not filled"):
        bg.select_like(qs[:1], "seed-x", slots)


def test_fresh_records_and_manifest_hold_the_seed_hash_only():
    seed = "my-private-seed-value"
    doc = {"seed": seed, "fresh": True, "candidates": 3, "dropped": [],
           "questions": [dict(_q("blk_a", 0), pairs_with="bq_ref1"), dict(_q("blk_b", 1), pairs_with="bq_ref2")]}
    files = bg.fresh_outputs(doc)
    assert set(files) == {"questions.jsonl", "answers.jsonl", "manifest.json"}
    assert all(seed not in text for text in files.values())
    items = [json.loads(x) for x in files["questions.jsonl"].splitlines()]
    keys = [json.loads(x) for x in files["answers.jsonl"].splitlines()]
    assert all(validate(i) == [] for i in items)
    assert not {i["id"] for i in items} & {k["question_id"] for k in REF_KEYS}
    assert [k["pairs_with"] for k in keys] == ["bq_ref1", "bq_ref2"]
    assert all(bg.seed_hash(seed)[:16] in i["provenance"]["generator_version"] for i in items)
    manifest = json.loads(files["manifest.json"])
    assert manifest["seed_sha256"] == bg.seed_hash(seed) and manifest["like_reference"]
    assert manifest["composition"] == {"blk_a/output_value": 1, "blk_b/output_value": 1}


def test_score_rows_join_answers_to_the_key():
    keys = [{"question_id": "q1", "reference": {"value": 2.0}}, {"question_id": "q2", "reference": {"value": 3}}]
    rows = bg.score_rows(keys, {"q1": "Final answer: 2"})
    assert rows == [{"id": "q1", "answer": "Final answer: 2", "reference": {"value": 2.0}}]


def test_nothing_fresh_or_private_is_written_into_the_repository(tmp_path):
    for where in (REPO, REPO / "items" / "tier_k" / "fresh"):
        with pytest.raises(SystemExit, match="never written inside the repository"):
            bg._outside_repo(where, "a fresh set")
        with pytest.raises(SystemExit, match="never written inside the repository"):
            bg.private_seed_init(where / "seed.json")
    with pytest.raises(SystemExit, match="reference seed"):
        bg.fresh_set(bg.REFERENCE_SEED, tmp_path / "x")


def test_private_seed_is_owner_only_never_overwritten_and_readable(tmp_path):
    path = tmp_path / "private" / "seed.json"
    digest = bg.private_seed_init(path)
    seed = bg.read_seed_file(path)
    assert digest == bg.seed_hash(seed) and len(seed) == 32
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(SystemExit, match="never overwritten"):
        bg.private_seed_init(path)
    plain = tmp_path / "plain.txt"
    plain.write_text("  a-plain-seed\n", encoding="utf-8")
    assert bg.read_seed_file(plain) == "a-plain-seed"


def test_fresh_gap_flags_a_drop_on_the_fresh_set():
    ref = {f"r{i}": 1.0 for i in range(10)}
    pairs = {f"f{i}": f"r{i}" for i in range(10)}
    blocks = {f"r{i}": f"b{i % 5}" for i in range(10)}
    flat = fn.fresh_gap(ref, {f"f{i}": 1.0 for i in range(10)}, pairs, blocks, B=300)
    drop = fn.fresh_gap(ref, {f"f{i}": 0.0 if i < 6 else 1.0 for i in range(10)}, pairs, blocks, B=300)
    assert flat["diff"] == 0 and not flat["memorisation"]
    assert drop["diff"] == pytest.approx(0.6) and drop["memorisation"] and drop["n_clusters"] == 5


def test_the_fresh_set_report_shows_the_same_composition():
    text = (REPO / bg.OUT_FRESH_REPORT).read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| check ") or line.startswith("| reference ")]
    assert len(rows) >= 4 and all("| yes |" in r for r in rows)


@pytest.mark.gnuradio
def test_a_small_fresh_build_runs_gnu_radio_and_matches_its_slots(tmp_path):
    pytest.importorskip("gnuradio")
    blocks = ("blocks_add_const_vxx", "blocks_keep_one_in_n")
    slots = {k: v for k, v in bg.reference_slots().items() if k[0] in blocks}
    doc = bg.build(workers=4, seed="a-test-seed", slots=slots)
    assert Counter((q["block"], q["kind"]) for q in doc["questions"]) == {k: len(v) for k, v in slots.items()}
    ref_ids = {i for ids in slots.values() for i in ids}
    for q in doc["questions"]:
        assert q["pairs_with"] in ref_ids and "GNU Radio 3.10.12 run" in q["evidence"]

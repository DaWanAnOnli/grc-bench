"""Tests for task-type label normalization and agreement (WS2a; eval plan §4.2, proposal §4.2)."""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import pytest

from grcbench.common.io import read_json, read_jsonl, sha256_file, write_json
from grcbench.common.paths import REPO
from grcbench.common.schemas import TIER_K
from grcbench.labels import agreement
from grcbench.labels import task_types as tt

OUTPUTS = [tt.LABELS_REL, tt.KEY_REL, tt.REPORT_REL, tt.TEST_SHEET_REL, tt.TRAIN_SHEET_REL,
           tt.INSTRUCTIONS_REL]


@pytest.fixture(scope="module")
def questions():
    return tt.load_questions()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("labels_run1")
    return out, tt.build(out)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


# ---- the mapping --------------------------------------------------------------------------------

def test_canonical_types_match_spec_and_item_schema():
    spec = tt.load_spec()
    assert tuple(sorted(spec["task_types"])) == tt.CANONICAL
    schema_enum = TIER_K["allOf"][1]["properties"]["task_type"]["enum"]
    assert sorted(schema_enum) == list(tt.CANONICAL)


def test_rule_map_and_ambiguous_spellings_cover_all_23(questions):
    eval_rows, train_rows = questions
    raw = {r["task_type"] for r in eval_rows + train_rows}
    assert len(raw) == 23
    assert raw == set(tt.RULES) | set(tt.AMBIGUOUS)
    assert not set(tt.RULES) & set(tt.AMBIGUOUS)
    assert len(tt.RULES) == 16 and len(tt.AMBIGUOUS) == 7
    assert set(tt.RULE_MAP.values()) == set(tt.CANONICAL)
    for name in tt.CANONICAL:
        assert tt.RULE_MAP[name] == name
    expected = {"debugging": "diagnosis", "diagnosis from symptoms": "diagnosis",
                "diagnosis_from_symptoms": "diagnosis", "system design": "design",
                "system_design": "design", "calculations": "calculation",
                "factual knowledge": "factual", "factual_knowledge": "factual",
                "gnu_radio_usage": "gnuradio_usage", "GNU Radio usage": "gnuradio_usage"}
    for raw_spelling, clean in expected.items():
        assert tt.RULE_MAP[raw_spelling] == clean


def test_decisions_cover_exactly_the_ambiguous_rows(questions):
    eval_rows, train_rows = questions
    doc = tt.load_decisions()
    assert doc["labeller"] == "Claude (Anthropic)"
    amb = {r["id"]: r["task_type"] for r in eval_rows + train_rows if r["task_type"] in tt.AMBIGUOUS}
    assert len(amb) == 134
    assert set(doc["decisions"]) == set(amb)
    assert sum(1 for r in eval_rows if r["task_type"] in tt.AMBIGUOUS) == 11
    rule_type = {"diagnosis": "diagnosis", "gnuradio": "gnuradio_usage", "quantities": "calculation",
                 "selection": "design", "comparison": "reasoning", "concept": "factual"}
    assert set(rule_type) == set(doc["rules"])
    for qid, d in doc["decisions"].items():
        assert d["raw"] == amb[qid]
        assert d["clean"] in tt.CANONICAL
        assert rule_type[d["rule"]] == d["clean"], qid
        assert 10 <= len(d["reason"]) <= 200 and "\n" not in d["reason"]


def test_missing_information_questions_never_get_a_separate_type():
    """task_types.json: insufficient-information questions are reasoning, design or diagnosis."""
    doc = tt.load_decisions()
    info = {"ask_for_info", "asking for missing information", "information_gathering"}
    for d in doc["decisions"].values():
        if d["raw"] in info and d["rule"] != "gnuradio":
            assert d["clean"] in {"reasoning", "design", "diagnosis"}


def test_label_rows_rejects_bad_inputs():
    good = [{"id": "q1", "task_type": "debugging"}, {"id": "q2", "task_type": "conceptual"}]
    dec = {"q2": {"raw": "conceptual", "clean": "factual", "reason": "r", "rule": "concept"}}
    rows = tt.label_rows(good, [], dec)
    assert [(r["id"], r["clean"], r["method"]) for r in rows] == [("q1", "diagnosis", "rule"),
                                                                 ("q2", "factual", "classified")]
    with pytest.raises(ValueError, match="unknown raw"):
        tt.label_rows([{"id": "q3", "task_type": "trivia"}], [], {})
    with pytest.raises(ValueError, match="no recorded decision"):
        tt.label_rows(good, [], {})
    with pytest.raises(ValueError, match="not ambiguous"):
        tt.label_rows(good, [], dec | {"q9": {"raw": "compare", "clean": "reasoning", "reason": "r"}})
    with pytest.raises(ValueError, match="decision was made"):
        tt.label_rows(good, [], {"q2": {**dec["q2"], "raw": "compare"}})


# ---- outputs ------------------------------------------------------------------------------------

def test_label_file(built, questions):
    out, _ = built
    eval_rows, train_rows = questions
    rows = read_jsonl(out / tt.LABELS_REL)
    assert len(rows) == 1853 and len({r["id"] for r in rows}) == 1853
    assert Counter(r["split"] for r in rows) == {"eval": 185, "train": 1668}
    source = {r["id"]: r["task_type"] for r in eval_rows + train_rows}
    split_of = {r["id"]: "eval" for r in eval_rows} | {r["id"]: "train" for r in train_rows}
    for r in rows:
        assert set(r) == {"id", "split", "raw", "clean", "method", "reason", "labeller"}
        assert r["raw"] == source[r["id"]], "raw labels are kept unchanged"
        assert r["split"] == split_of[r["id"]]
        assert r["clean"] in tt.CANONICAL and r["reason"]
        if r["method"] == "rule":
            assert r["clean"] == tt.RULE_MAP[r["raw"]] and r["labeller"] == tt.LABELLER_RULE
        else:
            assert r["method"] == "classified" and r["raw"] in tt.AMBIGUOUS
            assert r["labeller"] == "Claude (Anthropic)"


def test_test_counts_and_small_groups(built):
    _, res = built
    assert sum(res["test_counts"].values()) == 185
    assert res["small_groups"] == [t for t in tt.CANONICAL if res["test_counts"][t] < tt.MIN_GROUP]
    assert "gnuradio_usage" in res["small_groups"]   # eval plan §4.2: "only about 10 test items"


def test_blind_test_sheet(built):
    out, _ = built
    columns, rows = _read_csv(out / tt.TEST_SHEET_REL)
    assert columns == ["id", "question", "raw_label", "annotator_1", "annotator_2", "notes"]
    labels = {r["id"]: r for r in read_jsonl(out / tt.LABELS_REL)}
    expected = {i for i, r in labels.items() if r["split"] == "eval" and r["method"] == "classified"}
    assert {r["id"] for r in rows} == expected and len(rows) == 11
    for r in rows:
        assert r["annotator_1"] == r["annotator_2"] == r["notes"] == ""
        assert r["raw_label"] == labels[r["id"]]["raw"] and r["raw_label"] in tt.AMBIGUOUS
        assert not ({r["raw_label"], r["notes"]} & set(tt.CANONICAL)), "Claude's label must not show"
    key = read_json(out / tt.KEY_REL)["ambiguous_test"]
    assert [k["id"] for k in key["rows"]] == [r["id"] for r in rows]
    assert [k["row"] for k in key["rows"]] == list(range(1, 12))
    for k in key["rows"]:
        assert k["claude_label"] == labels[k["id"]]["clean"]
    # rows are shuffled by the seeded hash, not grouped by raw label or id
    assert [r["id"] for r in rows] == sorted(expected, key=lambda i: tt.rank_key(tt.ORDER_SEED, i))


def test_spotcheck_sheet(built):
    out, _ = built
    columns, rows = _read_csv(out / tt.TRAIN_SHEET_REL)
    assert columns == tt.TRAIN_SHEET_COLUMNS
    assert "claude_or_rule_label" in columns and "method" in columns
    labels = {r["id"]: r for r in read_jsonl(out / tt.LABELS_REL)}
    train = [r for r in labels.values() if r["split"] == "train"]
    key = read_json(out / tt.KEY_REL)["spotcheck_train"]
    assert key["n"] == len(rows) == len(key["rows"])
    assert len(rows) >= round(0.10 * len(train))
    sizes = Counter(r["raw"] for r in train)
    picked = Counter(r["raw_label"] for r in rows)
    assert set(picked) == set(sizes), "every raw spelling is spot-checked"
    n_total = round(0.10 * len(train))
    for raw, n in sizes.items():
        quota = n_total * n / len(train)
        assert max(1, math.floor(quota)) <= picked[raw] <= max(1, math.ceil(quota))
    for r in rows:
        lab = labels[r["id"]]
        assert lab["split"] == "train"
        assert r["method"] == lab["method"] and r["claude_or_rule_label"] == lab["clean"]
        assert r["reviewer_label"] == ""
    assert {r["method"] for r in rows} == {"rule", "classified"}
    again, _ = tt.spotcheck_sample(train)
    assert [r["id"] for r in again] == [r["id"] for r in rows]


def test_largest_remainder():
    assert tt.largest_remainder({"a": 5, "b": 3, "c": 2}, 5) == {"a": 3, "b": 1, "c": 1}
    assert tt.largest_remainder({"b": 1, "a": 1}, 1) == {"a": 1, "b": 0}   # tie broken by key


def test_report_content(built):
    out, res = built
    text = (out / tt.REPORT_REL).read_text(encoding="utf-8")
    assert "for team decision" not in text and "Proposal: merge into" not in text
    assert "## Raw to clean counts" in text and "## Clean type counts" in text
    for g in res["small_groups"]:
        assert f"### `{g}`" in text and "Merged into" in text
    assert "factual recall and reasoning" in text.lower()
    assert "return your labels before reading this report" in text
    for line in ("design + gnuradio_usage: 50 test items", "recall: 30 test items", "reasoning: 155 test items",
                 "GNU Radio behaviour: 100 test items"):
        assert line in text, line


def test_gnuradio_usage_reports_with_design():
    assert tt.report_group("gnuradio_usage", "frozen_question") == "design + gnuradio_usage"
    assert tt.report_group("design", "bank_question") == "design + gnuradio_usage"
    assert tt.report_group("factual", "frozen_question") == "factual"


def test_families_cover_the_frozen_questions_only():
    assert tt.family("factual", "frozen_question") == "recall"
    assert tt.family("gnuradio_usage", "frozen_question") == "reasoning"
    assert tt.family("calculation", "bank_question") == "reasoning"          # dev: the same mapping
    assert tt.family("reasoning", "block_question") is None
    assert tt.report_group("reasoning", "block_question") == "gnuradio_behaviour"
    with pytest.raises(ValueError):
        tt.family("factual", "frozen")
    with pytest.raises(ValueError):
        tt.report_group("trivia", "frozen_question")


def test_build_is_deterministic(built, tmp_path):
    out1, _ = built
    tt.build(tmp_path)
    for rel in OUTPUTS:
        assert sha256_file(out1 / rel) == sha256_file(tmp_path / rel), rel


def test_repository_outputs_are_current(built):
    """If the outputs are in the repository, they must equal a fresh build (no hand edits)."""
    if not (REPO / tt.LABELS_REL).exists():
        pytest.skip("./gb labels has not been run in the repository yet")
    out, _ = built
    for rel in OUTPUTS:
        assert sha256_file(REPO / rel) == sha256_file(out / rel), f"{rel} is stale: run ./gb labels"


# ---- agreement ----------------------------------------------------------------------------------

def test_cohen_kappa_known_values():
    # Two-category textbook examples (κ = 0.4 and κ = 0.06 / 0.46).
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert agreement.cohen_kappa(a, b) == pytest.approx(0.4)
    a = ["y"] * 60 + ["n"] * 40
    b = ["y"] * 45 + ["n"] * 15 + ["y"] * 25 + ["n"] * 15
    assert agreement.cohen_kappa(a, b) == pytest.approx(0.06 / 0.46)
    # Three categories, worked by hand: p_o = 23/30, p_e = 312/900, κ = 378/588 = 9/14.
    rows = {("x", "x"): 10, ("x", "y"): 2, ("y", "x"): 1, ("y", "y"): 8, ("y", "z"): 3,
            ("z", "y"): 1, ("z", "z"): 5}
    a = [p[0] for p, n in rows.items() for _ in range(n)]
    b = [p[1] for p, n in rows.items() for _ in range(n)]
    assert agreement.cohen_kappa(a, b) == pytest.approx(9 / 14)
    assert agreement.cohen_kappa(b, a) == pytest.approx(9 / 14)
    assert agreement.cohen_kappa(a, a) == 1.0
    assert agreement.cohen_kappa(["x", "y"] * 5, ["y", "x"] * 5) == pytest.approx(-1.0)
    assert math.isnan(agreement.cohen_kappa(["x"] * 4, ["x"] * 4))
    with pytest.raises(ValueError):
        agreement.cohen_kappa(["x"], ["x", "y"])


def test_normalise_label():
    assert agreement.normalise_label("  Factual ") == "factual"
    assert agreement.normalise_label("GNU Radio usage") == "gnuradio_usage"
    assert agreement.normalise_label("gnuradio usage") == "gnuradio_usage"
    assert agreement.normalise_label("debugging") == "diagnosis"
    assert agreement.normalise_label("") == ""
    with pytest.raises(ValueError):
        agreement.normalise_label("conceptual")


def _write_sheet(path: Path, columns: list[str], rows: list[dict[str, str]], bom: bool) -> None:
    with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def test_agreement_from_returned_sheets(tmp_path):
    # 50 items laid out as the κ = 0.4 table, plus 2 items only annotator 1 labelled.
    a1 = ["factual"] * 25 + ["reasoning"] * 25 + ["design"] * 2
    a2 = ["Factual"] * 20 + ["reasoning"] * 5 + ["factual"] * 10 + ["reasoning"] * 15 + ["", ""]
    ids = [f"s{i:02d}" for i in range(len(a1))]
    key = {"ambiguous_test": {"rows": [{"row": i + 1, "id": q, "claude_label": a1[i]}
                                       for i, q in enumerate(ids)]},
           "spotcheck_train": {"rows": []}}
    given = ["factual", "reasoning", "design", "factual", "reasoning", "design",
             "factual", "factual", "reasoning", "reasoning"]
    review = given[:6] + ["factual", "reasoning", "reasoning", "factual"]
    methods = ["rule"] * 6 + ["classified"] * 4
    key["spotcheck_train"]["rows"] = [{"row": i + 1, "id": f"t{i}", "method": m, "label": g}
                                      for i, (m, g) in enumerate(zip(methods, given))]
    write_json(tmp_path / "key.json", key)
    cols = ["id", "question", "raw_label", "annotator_1", "annotator_2", "notes"]
    _write_sheet(tmp_path / "copy1.csv", cols,
                 [{"id": q, "question": "?", "annotator_1": x} for q, x in zip(ids, a1)], bom=True)
    _write_sheet(tmp_path / "copy2.csv", cols,
                 [{"id": q, "question": "?", "annotator_2": y} for q, y in zip(ids, a2)], bom=False)
    _write_sheet(tmp_path / "spot.csv", tt.TRAIN_SHEET_COLUMNS,
                 [{"id": f"t{i}", "reviewer_label": r} for i, r in enumerate(review)], bom=True)

    rc = tt.main(["agreement", "--test", str(tmp_path / "copy1.csv"), str(tmp_path / "copy2.csv"),
                  "--train", str(tmp_path / "spot.csv"), "--key", str(tmp_path / "key.json"),
                  "--out", str(tmp_path / "out")])
    assert rc == 0
    res = read_json(tmp_path / "out" / agreement.AGREEMENT_JSON_REL)
    pairs = res["ambiguous_test"]["pairs"]
    assert pairs["annotator_1 vs annotator_2"] == {"n": 50, "agreement": 0.7, "kappa": 0.4}
    assert pairs["annotator_1 vs claude"] == {"n": 52, "agreement": 1.0, "kappa": 1.0}
    assert pairs["annotator_2 vs claude"]["kappa"] == 0.4
    assert res["ambiguous_test"]["problems"] == []
    spot = res["spotcheck_train"]["by_method"]
    assert spot["rule"] == {"n": 6, "agreement": 1.0, "kappa": 1.0}
    assert spot["classified"] == {"n": 4, "agreement": 0.5, "kappa": 0.0}
    assert spot["all"]["n"] == 10
    assert len(res["spotcheck_train"]["disagreements"]) == 2
    md = (tmp_path / "out" / agreement.AGREEMENT_MD_REL).read_text(encoding="utf-8")
    assert "annotator_1 vs annotator_2 | 50 | 70.0% | 0.400" in md


def test_agreement_flags_bad_entries(tmp_path):
    key = {"ambiguous_test": {"rows": [{"row": 1, "id": "a", "claude_label": "factual"}]}}
    cols = ["id", "question", "raw_label", "annotator_1", "annotator_2", "notes"]
    _write_sheet(tmp_path / "s.csv", cols, [{"id": "a", "annotator_1": "trivia", "annotator_2": "factual"},
                                            {"id": "zz", "annotator_1": "factual"}], bom=False)
    res = agreement.ambiguous_test_agreement([tmp_path / "s.csv"], key)
    assert any("trivia" in p for p in res["problems"])
    assert any("zz" in p for p in res["problems"])
    assert res["pairs"]["annotator_1 vs annotator_2"]["n"] == 0


def test_blank_sheets_give_empty_stats(built, tmp_path):
    out, _ = built
    key = read_json(out / tt.KEY_REL)
    res = agreement.ambiguous_test_agreement([out / tt.TEST_SHEET_REL], key)
    assert res["labelled"] == {"annotator_1": 0, "annotator_2": 0}
    assert all(s["n"] == 0 for s in res["pairs"].values())
    spot = agreement.spotcheck_agreement(out / tt.TRAIN_SHEET_REL, key)
    assert spot["labelled"] == 0 and spot["problems"] == []

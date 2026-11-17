"""The agent review protocol (`grcbench.review`, `./gb review`): sheet specs, routing rules, the expert's
audit sample, blinded inputs, decided records, filled sheets and the executed comparisons."""

from __future__ import annotations

import csv
import io
import json
import re

import pytest

from grcbench.common.paths import HANDOFF
from grcbench.review import decide, sheets
from grcbench.review import compare
from grcbench.review import records

SHEET_DIR = HANDOFF / "review_sheets"


# ------------------------------------------------------------------ sheet specs ------------------

def test_every_sheet_spec_names_an_existing_sheet_and_its_key_columns():
    for spec in sheets.SHEETS.values():
        path = SHEET_DIR / spec.file
        assert path.exists(), spec.file
        header = next(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
        assert set(spec.key) <= set(header), (spec.name, spec.key, header)
        for col in spec.hidden:
            assert col in header, (spec.name, col)


def test_every_sheet_has_a_guide_that_spells_its_values():
    from grcbench.common.paths import REPO
    for spec in sheets.SHEETS.values():
        guide = REPO / spec.guide
        assert guide.exists(), (spec.name, spec.guide)
        text = guide.read_text(encoding="utf-8")
        if spec.kind != "calc":
            for v in spec.values:
                assert f"`{v}`" in text, (spec.guide, v)


def test_allowed_values_are_the_guides_values():
    assert sheets.SHEETS["tier_f"].values == ("approve", "approve with edits", "reject")
    assert sheets.SHEETS["tier_a"].values == ("approve", "approve with edits", "reject")
    assert sheets.SHEETS["rubric_test"].values == ("approve", "approve with edits", "reject", "underspecified")
    assert sheets.SHEETS["twins"].values == ("twin", "not twin", "unsure")
    assert sheets.SHEETS["clusters"].values == ("same family", "split out")
    assert sheets.SHEETS["contamination_tier_k_test"].values == ("contaminated", "not contaminated", "unsure")
    assert set(sheets.SHEETS["task_type_test"].values) == {
        "calculation", "design", "diagnosis", "factual", "gnuradio_usage", "reasoning"}


def test_the_expert_may_not_answer_unsure_and_every_default_is_an_expert_value():
    for spec in sheets.SHEETS.values():
        assert "unsure" not in spec.expert_values
        if spec.default is not None:
            assert spec.default in spec.expert_values, spec.name
    assert sheets.SHEETS["clusters"].default == "same family"
    assert sheets.SHEETS["contamination_tier_f_dev"].default == "contaminated"
    assert sheets.SHEETS["twins"].default == "twin"
    assert sheets.SHEETS["tier_f"].default == "reject"
    assert sheets.SHEETS["rubric_dev"].default == "underspecified"


def test_blinded_input_drops_the_answer_columns():
    assert {"answer", "evidence"} <= set(sheets.SHEETS["block_questions"].hidden)
    assert "linked_now" in sheets.SHEETS["twins"].hidden


# ------------------------------------------------------------------ routing --------------------

def _out(value, **kw):
    return {"value": value, "edits": kw.get("edits", []), "notes": kw.get("notes", "")}


def test_same_verdict_is_final():
    r = decide.route(sheets.SHEETS["tier_f"], _out("approve"), _out("approve"))
    assert (r.to, r.value) == ("final", "approve")


def test_different_verdicts_go_to_the_expert():
    r = decide.route(sheets.SHEETS["tier_f"], _out("approve"), _out("reject"))
    assert r.to == "expert" and r.reason == "disagreement"


def test_two_approve_with_edits_go_to_the_expert_to_merge():
    r = decide.route(sheets.SHEETS["tier_a"], _out("approve with edits", edits=["a"]),
                     _out("approve with edits", edits=["b"]))
    assert r.to == "expert" and r.reason == "edits to merge"


def test_two_rubric_rejects_go_to_the_expert_to_rewrite_the_rubric():
    spec = sheets.SHEETS["rubric_dev"]
    r = decide.route(spec, _out("reject"), _out("reject"))
    assert r.to == "expert" and r.reason == "rejected: the expert rewrites the rubric"
    assert decide.route(sheets.SHEETS["tier_f"], _out("reject"), _out("reject")).to == "final"


def test_unsure_goes_to_the_expert_even_when_both_say_it():
    r = decide.route(sheets.SHEETS["twins"], _out("unsure"), _out("unsure"))
    assert r.to == "expert" and r.reason == "unsure"


def test_a_value_outside_the_allowed_list_is_refused():
    with pytest.raises(ValueError):
        decide.route(sheets.SHEETS["twins"], _out("twins"), _out("twin"))


def test_expert_value_must_be_decided():
    spec = sheets.SHEETS["contamination_tier_k_dev"]
    assert decide.expert_value(spec, _out("not contaminated")) == "not contaminated"
    with pytest.raises(ValueError):
        decide.expert_value(spec, _out("unsure"))


def test_audit_sample_is_seeded_deterministic_and_at_least_five():
    keys = [(f"q{i}",) for i in range(30)]
    a = decide.audit_sample("twins", keys)
    assert a == decide.audit_sample("twins", list(reversed(keys)))
    assert len(a) == 5 and set(a) <= set(keys)
    assert len(decide.audit_sample("twins", [(f"q{i}",) for i in range(200)])) == 20
    assert len(decide.audit_sample("twins", keys[:3])) == 3
    assert decide.audit_sample("clusters", keys) != a


def test_audit_outcome_asks_for_a_full_review_above_five_percent():
    ok = decide.audit_outcome([("k1", "twin", "twin")] * 20)
    assert ok["disagreements"] == 0 and not ok["full_review"]
    bad = decide.audit_outcome([("k1", "twin", "twin")] * 18 + [("k2", "twin", "not twin")] * 2)
    assert bad["disagreements"] == 2 and bad["full_review"]


# ------------------------------------------------------------------ decided records --------------

def test_records_take_the_agreed_value_or_the_experts_and_say_who_decided():
    spec = sheets.SHEETS["twins"]
    r1 = {("lexical", "a", "b"): _out("twin"), ("lexical", "c", "d"): _out("twin")}
    r2 = {("lexical", "a", "b"): _out("twin"), ("lexical", "c", "d"): _out("not twin")}
    ex = {("lexical", "c", "d"): _out("not twin", notes="different asks")}
    recs = records.build(spec, r1, r2, ex)
    by = {r["key"]: r for r in recs}
    assert by[("lexical", "a", "b")]["final"] == "twin" and by[("lexical", "a", "b")]["decided_by"] == "R1;R2"
    assert by[("lexical", "c", "d")]["final"] == "not twin" and by[("lexical", "c", "d")]["decided_by"] == "EX"


def test_records_refuse_a_missing_expert_decision_and_a_missing_reviewer_row():
    spec = sheets.SHEETS["twins"]
    r1 = {("lexical", "c", "d"): _out("twin")}
    with pytest.raises(ValueError, match="expert"):
        records.build(spec, r1, {("lexical", "c", "d"): _out("not twin")}, {})
    with pytest.raises(ValueError, match="R2"):
        records.build(spec, r1, {}, {})


def test_an_audit_disagreement_is_decided_by_the_expert():
    spec = sheets.SHEETS["twins"]
    key = ("lexical", "a", "b")
    recs = records.build(spec, {key: _out("twin")}, {key: _out("twin")}, {},
                         audit={key: _out("not twin", notes="different asks")})
    assert recs[0]["final"] == "not twin" and recs[0]["decided_by"] == "EX (audit)"


PRIVATE_PATH = re.compile(r"/mnt/[a-z]/Users/|\b[A-Za-z]:[\\/]+Users[\\/]|/c/Users/|/home/(?!user/)[a-z]")


def test_private_device_paths_are_redacted_from_the_decided_records():
    from pathlib import Path as P
    root, home = P("/mnt/c/Users/someone/Docs/proj"), P("/home/someone")
    row = {"evidence": ["$ wsl.exe -- bash /mnt/c/Users/someone/Docs/proj/review_work/x.sh => ok",
                        "C:\\Users\\someone\\Docs\\proj\\review_work\\out\\a.md :: \"q\"",
                        "C:/Users/someone/AppData/Local/Temp/t.log", "/home/someone/grc-agent/x.py",
                        "~/grc-agent/docs/a.md :: \"keep\"",
                        "the sandbox path '/home/user/project/data/x.dat' stays"], "n": 3}
    out = records.redact(row, project_root=root, home=home)
    assert out["evidence"] == ["$ wsl.exe -- bash <project>/review_work/x.sh => ok",
                               "<project>\\review_work\\out\\a.md :: \"q\"", "<home>/AppData/Local/Temp/t.log",
                               "~/grc-agent/x.py", "~/grc-agent/docs/a.md :: \"keep\"",
                               "the sandbox path '/home/user/project/data/x.dat' stays"]
    assert out["n"] == 3 and not PRIVATE_PATH.search(json.dumps(out))


def test_no_file_of_the_review_names_a_private_device_path():
    from grcbench.common.paths import REPO
    files = sorted((REPO / "reviews").glob("*.json*")) + sorted((REPO / "handoff" / "review_sheets").glob("*.csv"))
    hits = {f.name: len(PRIVATE_PATH.findall(f.read_text(encoding="utf-8"))) for f in files}
    assert {k: v for k, v in hits.items() if v} == {}


def test_cluster_rows_must_carry_one_decision():
    spec = sheets.SHEETS["clusters"]
    rows = [{"cluster_id": "c1", "id": "q1"}, {"cluster_id": "c1", "id": "q2"}]
    with pytest.raises(ValueError, match="cluster"):
        records.check_groups(spec, rows, {("q1",): _out("same family"), ("q2",): _out("split out")})
    records.check_groups(spec, rows, {("q1",): _out("split out"), ("q2",): _out("split out")})


# ------------------------------------------------------------------ comparisons (executed) -------

def test_a_recomputed_value_agrees_within_the_keys_tolerance():
    key = {"specification": "determined", "value": 10000.0, "unit": "Hz", "tolerance": 0.01, "tolerance_db": None}
    assert compare.calc_status(key, {"value": "10 kHz", "unit": ""}) == "agrees"
    assert compare.calc_status(key, {"value": "10050", "unit": "Hz"}) == "agrees"
    assert compare.calc_status(key, {"value": "10500", "unit": "Hz"}) == "disagrees"
    assert compare.calc_status(key, {"value": "", "unit": "", "notes": "underspecified: X"}) == "underspecified"


def test_a_decibel_value_uses_the_keys_tolerance_in_db():
    key = {"specification": "determined", "value": 3.0103, "unit": "dB", "tolerance": None, "tolerance_db": 0.1}
    assert compare.calc_status(key, {"value": "3.01", "unit": "dB"}) == "agrees"
    assert compare.calc_status(key, {"value": "3.5", "unit": "dB"}) == "disagrees"


def test_several_values_are_compared_part_by_part():
    key = {"specification": "determined", "value": [1000.0, 2.0], "unit": ["Hz", "s"],
           "tolerance": [0.01, 0.01], "tolerance_db": [None, None]}
    assert compare.calc_status(key, {"value": "1000; 2", "unit": "Hz; s"}) == "agrees"
    assert compare.calc_status(key, {"value": "1000; 3", "unit": "Hz; s"}) == "disagrees"
    assert compare.calc_status(key, {"value": "1000", "unit": "Hz"}) == "disagrees"


def test_calc_route_confirms_only_when_both_recomputers_agree_with_the_key():
    assert compare.calc_route("determined", "agrees", "agrees") == "final"
    assert compare.calc_route("determined", "agrees", "disagrees") == "expert"
    assert compare.calc_route("underspecified", "underspecified", "underspecified") == "final"
    assert compare.calc_route("team reference", "no value", "no value") == "final"
    assert compare.calc_route("underspecified", "agrees", "underspecified") == "expert"


def test_a_blind_block_answer_is_scored_against_the_key_by_the_rules():
    ref = {"abs_tolerance": 0.01, "tolerance": 1e-09, "unit": "", "value": 34.0}
    assert compare.block_matches(ref, "34")
    assert not compare.block_matches(ref, "35")
    assert not compare.block_matches(ref, "")


# ------------------------------------------------------------------ coordinator ------------------

def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _twin(a, b, v):
    return {"sheet": "twins", "key": {"source": "lexical", "id_a": a, "id_b": b}, "value": v,
            "notes": "", "edits": [], "evidence": []}


def test_reviewer_files_are_assigned_their_role_by_name(tmp_path):
    from grcbench.review import coordinate
    _jsonl(tmp_path / "out" / "a01-R1-K.jsonl", [_twin("a", "b", "twin")])
    _jsonl(tmp_path / "out" / "a02-R2-K.jsonl", [_twin("a", "b", "not twin")])
    _jsonl(tmp_path / "out" / "a07-EX.jsonl", [])
    r1, r2 = coordinate.collect(tmp_path, "twins")
    assert r1[("lexical", "a", "b")]["value"] == "twin" and r2[("lexical", "a", "b")]["value"] == "not twin"


def test_route_writes_the_expert_queue_and_a_blind_audit_sample(tmp_path):
    from grcbench.review import coordinate
    pairs = [(f"a{i}", f"b{i}") for i in range(12)]
    _jsonl(tmp_path / "out" / "a01-R1-K.jsonl", [_twin(a, b, "twin") for a, b in pairs])
    _jsonl(tmp_path / "out" / "a02-R2-K.jsonl",
           [_twin(a, b, "not twin" if i == 0 else "twin") for i, (a, b) in enumerate(pairs)])
    summary = coordinate.route(tmp_path, "twins")
    assert summary == {"rows": 12, "final": 11, "expert": 1, "audit": 5}
    queue = [json.loads(l) for l in (tmp_path / "expert" / "twins.queue.jsonl").read_text().splitlines()]
    assert queue[0]["key"] == {"source": "lexical", "id_a": "a0", "id_b": "b0"}
    assert queue[0]["reason"] == "disagreement" and queue[0]["R1"]["value"] == "twin"
    audit = [json.loads(l) for l in (tmp_path / "expert" / "twins.audit.jsonl").read_text().splitlines()]
    assert len(audit) == 5 and all(set(a) == {"sheet", "key"} for a in audit)


def test_a_block_question_whose_blind_value_misses_the_key_goes_to_the_expert():
    from grcbench.review import coordinate
    keys = {"bq_1": {"reference": {"value": 34.0, "unit": "", "abs_tolerance": 0.01, "tolerance": 1e-9},
                     "answer": 34.0}}
    router = coordinate.block_router(keys)
    ok = router(("bq_1",), {"value": "approve", "blind_value": "34"}, {"value": "approve", "blind_value": "34.0"})
    assert ok.to == "final"
    miss = router(("bq_1",), {"value": "approve", "blind_value": "34"}, {"value": "approve", "blind_value": "33"})
    assert miss.to == "expert" and miss.reason == "blind value misses the key"


def test_calc_router_scores_both_recomputations_against_the_key():
    from grcbench.review import coordinate
    keys = {"q1": {"specification": "determined", "value": 10000.0, "unit": "Hz", "tolerance": 0.01,
                   "tolerance_db": None}}
    router = coordinate.calc_router(keys)
    assert router(("q1",), {"value": "10 kHz", "unit": ""}, {"value": "10000", "unit": "Hz"}).to == "final"
    r = router(("q1",), {"value": "10 kHz", "unit": ""}, {"value": "12000", "unit": "Hz"})
    assert r.to == "expert" and "disagrees" in r.reason


def test_decide_sheet_uses_the_experts_decisions_and_audit_rows(tmp_path):
    from grcbench.review import coordinate
    _jsonl(tmp_path / "out" / "a01-R1-K.jsonl", [_twin("a", "b", "twin"), _twin("c", "d", "twin")])
    _jsonl(tmp_path / "out" / "a02-R2-K.jsonl", [_twin("a", "b", "twin"), _twin("c", "d", "not twin")])
    _jsonl(tmp_path / "out" / "a07-EX.jsonl", [{**_twin("c", "d", "twin"), "notes": "same ask"},
                                               {**_twin("a", "b", "not twin"), "pass": "audit"}])
    recs = coordinate.decide_sheet(tmp_path, "twins", reviews_dir=tmp_path / "reviews",
                                   sheet_dir=tmp_path / "sheets", require_all=False)
    by = {r["key"]: r for r in recs}
    assert by[("lexical", "c", "d")]["decided_by"] == "EX" and by[("lexical", "c", "d")]["final"] == "twin"
    assert by[("lexical", "a", "b")]["decided_by"] == "EX (audit)"
    assert (tmp_path / "reviews" / "twins.jsonl").exists()
    assert (tmp_path / "sheets" / "tier_k_twin_threshold_review.csv").exists()


def _bq(qid, v):
    return {"sheet": "tier_f", "key": {"task_id": qid}, "value": v, "notes": "", "edits": [], "evidence": []}


def test_the_expert_review_serves_as_the_full_review_of_the_agreed_rows():
    from grcbench.review import coordinate
    spec = sheets.SHEETS["tier_f"]
    keys = [f"t{i}" for i in range(4)]
    r1 = {(k,): _bq(k, "reject" if k == "t3" else "approve") for k in keys}
    h1b = {("t0",): _verdict("t0"), ("t1",): _verdict("t1", verdict="reject", reason_code="broken_oracle", comment="c"),
           ("t2",): _verdict("t2", verdict="accept with edits", reason_code="other", comment="c", suggested_fix="fix X"),
           ("t3",): _verdict("t3")}                                     # an agreed duplicate reject stays rejected
    rows = coordinate.full_review_rows(spec, r1, dict(r1), h1b)
    assert rows[("t0",)]["value"] == "approve" and rows[("t1",)]["value"] == "reject"
    assert rows[("t2",)]["value"] == "approve with edits" and rows[("t2",)]["edits"] == ["fix X"]
    assert rows[("t3",)]["value"] == "reject"                    # covered by the full review, not re-opened
    assert all(r["full_review"] and r["pass"] == "audit" for r in rows.values())
    recs = {r["key"]: r for r in records.build(spec, r1, dict(r1), {}, rows)}
    assert recs[("t1",)]["final"] == "reject" and recs[("t1",)]["decided_by"] == "EX (audit)"
    assert recs[("t3",)]["final"] == "reject" and recs[("t3",)]["decided_by"] == "R1;R2"


def test_a_triggered_full_review_must_cover_every_agreed_row_and_stray_expert_rows_are_refused():
    from grcbench.review import coordinate
    spec = sheets.SHEETS["tier_f"]
    r1 = {(f"t{i}",): _bq(f"t{i}", "approve") for i in range(8)}
    sample = {("t0",): {**_bq("t0", "reject"), "pass": "audit"}}             # 1 of 1 audited: a full review
    recs = records.build(spec, r1, dict(r1), {}, sample)
    with pytest.raises(ValueError, match="full review"):
        coordinate.check_audit(spec, recs, sample_keys=[("t0",)])
    full = {(f"t{i}",): {**_bq(f"t{i}", "approve"), "pass": "audit", "full_review": True} for i in range(1, 8)}
    coordinate.check_audit(spec, records.build(spec, r1, dict(r1), {}, {**sample, **full}), sample_keys=[("t0",)])
    with pytest.raises(ValueError, match="seeded"):
        coordinate.check_audit(spec, records.build(spec, r1, dict(r1), {}, {}), sample_keys=[("t0",)])
    with pytest.raises(ValueError, match="not sent to the expert"):
        records.build(spec, r1, dict(r1), {("t5",): _bq("t5", "reject")}, {})


# ------------------------------------------------------------------ filled sheets ----------------

def _rec(sheet, key, r1, r2, final, by, ex=None):
    return {"sheet": sheet, "key": key, "r1": r1, "r2": r2, "ex": ex, "audit": None, "final": final,
            "decided_by": by, "route": "agreement" if by != "EX" else "disagreement"}


def _rows(text):
    return list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))


def test_filled_twin_sheet_has_the_final_label_and_who_decided():
    from grcbench.review import fill
    _, rows = __import__("grcbench.review.inputs", fromlist=["x"]).read_sheet("twins")
    k1 = (rows[0]["source"], rows[0]["id_a"], rows[0]["id_b"])
    k2 = (rows[1]["source"], rows[1]["id_a"], rows[1]["id_b"])
    recs = [_rec("twins", k1, _out("twin", notes="same ask"), _out("twin", notes="same"), "twin", "R1;R2"),
            _rec("twins", k2, _out("twin"), _out("not twin"), "not twin", "EX", ex=_out("not twin", notes="x"))]
    out = _rows(fill.filled("twins", recs, require_all=False))
    assert out[0]["team_label (twin / not twin)"] == "twin" and out[0]["reviewer"] == "R1;R2"
    assert out[1]["team_label (twin / not twin)"] == "not twin" and out[1]["reviewer"] == "R1;R2;EX"
    assert out[2]["team_label (twin / not twin)"] == rows[2]["team_label (twin / not twin)"]  # no record: kept


def test_filled_sheet_refuses_rows_without_a_decision_when_all_are_required():
    from grcbench.review import fill
    with pytest.raises(ValueError, match="no decision"):
        fill.filled("twins", [], require_all=True)


def test_filled_task_sheet_keeps_both_reviews_and_records_the_experts_decision():
    from grcbench.review import fill
    _, rows = __import__("grcbench.review.inputs", fromlist=["x"]).read_sheet("tier_a")
    k = (rows[0]["task_id"],)
    rec = _rec("tier_a", k, _out("approve", notes="note: fine"),
               _out("approve with edits", edits=["instruction: add X -- why"]), "approve with edits", "EX",
               ex=_out("approve with edits", edits=["instruction: add X -- why"], notes="R2 is right"))
    out = _rows(fill.filled("tier_a", [rec], require_all=False))[0]
    assert out["reviewer_1_decision"] == "approve" and out["reviewer_2_decision"] == "approve with edits"
    assert "instruction: add X -- why" in out["reviewer_2_edits"]
    assert out["final_decision"] == "approve with edits"
    assert "note: decided by EX" in out["reviewer_1_edits"]


# ------------------------------------------------------------------ applying decisions -----------

def _twin_rec(source, band, a, b, final):
    return {"sheet": "twins", "key": (source, a, b), "final": final, "band": band}


def test_band_counts_and_the_threshold_rule():
    from grcbench.review import apply
    rows = ([("semantic", "[0.92, 0.93)", f"a{i}", f"b{i}", "twin" if i < 13 else "not twin") for i in range(26)]
            + [("semantic", "[0.91, 0.92)", f"c{i}", f"d{i}", "twin" if i < 19 else "not twin") for i in range(40)])
    sheet_rows = [{"source": s, "band": band, "id_a": a, "id_b": b} for s, band, a, b, _ in rows]
    recs = [{"key": (s, a, b), "final": f} for s, _, a, b, f in rows]
    counts = apply.band_counts(sheet_rows, recs)
    assert counts["semantic"] == [{"band": [0.91, 0.92], "pairs": 40, "twins": 19},
                                  {"band": [0.92, 0.93], "pairs": 26, "twins": 13}]
    assert apply.threshold(counts["semantic"]) == 0.92          # 13/26 is half: at least half
    counts["semantic"][0]["twins"] = 20
    assert apply.threshold(counts["semantic"]) == 0.91
    assert apply.threshold([{"band": [0.5, 0.55], "pairs": 10, "twins": 2}]) is None


def test_a_generated_sheet_text_gets_the_decided_columns(tmp_path):
    from grcbench.review import fill
    text = ("cluster_id,id,linked_to (evidence),origin,task_type,question,"
            "team_decision (same family / split out),reviewer,notes\n"
            "tc_1,q1,q2: 8gram,o,t,Q1,,,\ntc_1,q2,q1: 8gram,o,t,Q2,,,\ntc_3,q3,q4: lexical,o,t,Q3,,,\n")
    spec = sheets.SHEETS["clusters"]
    recs = records.build(spec, {("q1",): _out("split out"), ("q2",): _out("split out")},
                         {("q1",): _out("split out"), ("q2",): _out("split out")}, {})
    records.write_records(tmp_path / "clusters.jsonl", recs)
    out = _rows(fill.filled_text("clusters", text, reviews_dir=tmp_path))
    assert [r["team_decision (same family / split out)"] for r in out] == ["split out", "split out", ""]
    assert fill.filled_text("clusters", text, reviews_dir=tmp_path / "none") == text


def test_decided_reads_the_final_values_of_a_sheet(tmp_path):
    spec = sheets.SHEETS["clusters"]
    recs = records.build(spec, {("q1",): _out("split out"), ("q2",): _out("split out")},
                         {("q1",): _out("split out"), ("q2",): _out("split out")}, {})
    records.write_records(tmp_path / "clusters.jsonl", recs)
    got = records.decided("clusters", reviews_dir=tmp_path)
    assert {k: r["final"] for k, r in got.items()} == {("q1",): "split out", ("q2",): "split out"}
    assert records.decided("twins", reviews_dir=tmp_path) == {}


# ------------------------------------------------------------------ rubric checks ----------------

def test_sheet_lines_become_the_judges_rubric_record():
    from grcbench.review import validate
    assert validate.parse_lines(["P1 (w=2): states X", "P2: states Y"], "P") == [
        {"id": "P1", "weight": 2.0, "text": "states X"}, {"id": "P2", "weight": 1.0, "text": "states Y"}]
    assert validate.parse_lines(["D1: says Z"], "D") == [{"id": "D1", "text": "says Z"}]
    with pytest.raises(ValueError, match="P1"):
        validate.parse_lines(["states X"], "P")


def test_a_rubric_record_is_checked_by_the_judges_and_the_rules_code():
    from grcbench.review import validate
    good = {"required_points": [{"id": "P1", "weight": 1.0, "text": "states X"}], "disqualifying_errors": [],
            "numeric": {"value": 10000, "unit": "Hz", "tolerance": 0.01},
            "numeric_tolerance": "1 % relative", "sources": ["a"]}
    assert validate.rubric_problems(good) == []
    no_tol = dict(good, numeric_tolerance=None)
    assert validate.rubric_problems(no_tol)
    bad_db = dict(good, numeric={"value": 3.0, "unit": "dB", "tolerance": 0.01})
    assert validate.rubric_problems(bad_db)


def test_quoted_sources_must_be_found_in_the_file_they_name(tmp_path):
    from grcbench.review import validate
    f = tmp_path / "blk.block.yml"
    f.write_text("parameters:\n-   id: samp_rate\n    label: Sample   Rate\n", encoding="utf-8")
    ok = [f'{f} :: "label: Sample Rate"', "PySDR, chapter 4"]
    assert validate.quote_problems(ok) == []
    bad = [f'{f} :: "label: Sampling Rate"', f'{tmp_path / "missing.yml"} :: "x"']
    probs = validate.quote_problems(bad)
    assert len(probs) == 2 and "not found" in probs[0] and "no such file" in probs[1]


def test_rubric_numbers_must_match_the_confirmed_calculation():
    from grcbench.review import validate
    numeric = {"value": 10000, "unit": "Hz", "tolerance": 0.01}
    assert validate.numeric_matches(numeric, ["10 kHz"])
    assert not validate.numeric_matches(numeric, ["11 kHz"])
    two = [{"value": 1, "unit": "s", "tolerance": 0.01, "label": "T"}, {"value": 2, "unit": "Hz", "tolerance": 0.01}]
    assert validate.numeric_matches(two, ["1 s", "2 Hz"]) and not validate.numeric_matches(two, ["1 s"])


def test_teacher_overlap_counts_shared_word_8grams():
    from grcbench.review import validate
    answer = "the sampling rate must be at least twice the highest frequency in the signal"
    assert validate.shared_8grams(["P1: the sampling rate must be at least twice the highest frequency"],
                                  [answer]) >= 1
    assert validate.shared_8grams(["P1: Nyquist says sample faster than double the bandwidth"], [answer]) == 0


# ------------------------------------------------------------------ rubric records ---------------

def _writer(qid="q1", **kw):
    row = {"sheet": "rubric_test", "key": {"question_id": qid}, "value": "written",
           "required_points": ["P1 (w=2): states X", "P2: states Y"], "disqualifying_errors": ["D1: says Z"],
           "numeric": None, "numeric_tolerance": None, "sources": ["PySDR, chapter 4"], "notes": "", "edits": [],
           "evidence": []}
    return {**row, **kw}


def test_a_rubric_that_fails_a_check_goes_to_the_expert_whatever_the_checkers_say():
    from grcbench.review import coordinate
    writer = {("q1",): _writer(), ("q2",): _writer("q2", numeric={"value": 3, "unit": "dB", "tolerance": 0.1},
                                                     numeric_tolerance="0.1 dB")}
    router = coordinate.rubric_router(writer, problems=lambda k, w: [] if k == ("q1",) else ["bad numeric"])
    assert router(("q1",), _out("approve"), _out("approve")).to == "final"
    r = router(("q2",), _out("approve"), _out("approve"))
    assert r.to == "expert" and "bad numeric" in r.reason


def test_rubric_record_takes_the_writers_fields_or_the_experts_final_fields():
    from grcbench.review import apply
    meta = {"question_id": "q1", "split": "test", "task_type": "factual", "question": "Q?"}
    rec = {"key": ("q1",), "final": "approve", "decided_by": "C1;C2", "route": "agreement", "ex": None}
    r = apply.rubric_record(meta, _writer(), rec, calc_status="")
    assert r["required_points"] == [{"id": "P1", "weight": 2.0, "text": "states X"},
                                    {"id": "P2", "weight": 1.0, "text": "states Y"}]
    assert r["review"] == {"status": "approved", "reviewers": ["W", "C1", "C2"], "notes": "decided by C1;C2"}
    ex = {"value": "approve with edits", "required_points": ["P1: states X'"], "disqualifying_errors": [],
          "numeric": None, "numeric_tolerance": None, "sources": ["PySDR, chapter 5"], "notes": "merged"}
    rec2 = {**rec, "final": "approve with edits", "decided_by": "EX", "route": "edits to merge", "ex": ex}
    r2 = apply.rubric_record(meta, _writer(), rec2, calc_status="")
    assert r2["required_points"] == [{"id": "P1", "weight": 1.0, "text": "states X'"}]
    assert r2["review"]["status"] == "approved-with-edits" and r2["sources"] == ["PySDR, chapter 5"]
    with pytest.raises(ValueError, match="final"):
        apply.rubric_record(meta, _writer(), {**rec2, "ex": {"value": "approve with edits"}}, calc_status="")
    r3 = apply.rubric_record(meta, _writer(), {**rec, "final": "underspecified"}, calc_status="")
    assert r3["review"]["status"] == "rejected" and r3["review"]["notes"].startswith("underspecified")


def _checker(qid, value, **kw):
    return {"sheet": "rubric_test", "key": {"question_id": qid}, "value": value, "source": f"source of {qid}",
            "edits": [], "notes": "", "evidence": [], **kw}


def test_rubric_decisions_keep_the_writer_row_and_fill_the_rubric_sheet(tmp_path):
    from grcbench.review import coordinate, inputs
    _, rows = inputs.read_sheet("rubric_test")
    q1, q2 = rows[0]["question_id"], rows[1]["question_id"]
    _jsonl(tmp_path / "out" / "a16-W-test1.jsonl", [_writer(q1), _writer(q2)])
    _jsonl(tmp_path / "out" / "a21-C1-test.jsonl", [_checker(q1, "approve", notes="fine"),
                                                    _checker(q2, "approve")])
    _jsonl(tmp_path / "out" / "a23-C2-test.jsonl", [_checker(q1, "approve"),
                                                    _checker(q2, "approve with edits", edits=["P2: say W -- why"])])
    ex = _writer(q2, value="approve with edits", required_points=["P1: states X", "P2: says W"],
                 notes="C2 is right")
    _jsonl(tmp_path / "out" / "a25-EX-K.jsonl", [ex])
    recs = coordinate.decide_rubrics(tmp_path, "rubric_test", problems=lambda k, w: [],
                                     reviews_dir=tmp_path / "reviews", sheet_dir=tmp_path / "sheets",
                                     require_all=False)
    by = {r["key"][0]: r for r in recs}
    assert by[q1]["writer"]["required_points"] == ["P1 (w=2): states X", "P2: states Y"]
    assert by[q1]["final"] == "approve" and by[q2]["decided_by"] == "EX"
    saved = records.read_records(tmp_path / "reviews" / "rubric_test.jsonl")
    assert saved[0]["writer"]["value"] == "written"
    out = {r["question_id"]: r for r in _rows((tmp_path / "sheets" / "rubric_sheet_test.csv").read_text("utf-8"))}
    assert out[q1]["required_points"] == "P1 (w=2): states X\nP2: states Y" and out[q1]["writer"] == "W"
    assert out[q1]["checker_1_initials"] == "C1" and out[q1]["checker_1_verdict"] == "approve"
    assert out[q1]["checker_1_source"] == f"source of {q1}" and out[q1]["checker_1_note"] == "note: fine"
    assert out[q1]["final_decision"] == "approve" and out[q1]["decided_by"] == "checkers"
    assert out[q2]["required_points"] == "P1: states X\nP2: says W", "the expert's final fields"
    assert out[q2]["checker_2_note"] == "P2: say W -- why" and out[q2]["decided_by"] == "EX"
    assert out[q2]["final_decision"] == "approve with edits"
    assert rows[2]["question_id"] not in {k[0] for k in by}


def test_the_filled_rubric_sheet_lists_the_expert_review_edits():
    from grcbench.review import fill, inputs
    _, rows = inputs.read_sheet("rubric_dev")
    q = rows[0]["question_id"]
    rec = {"sheet": "rubric_dev", "key": (q,), "r1": _checker(q, "approve"), "r2": _checker(q, "approve"),
           "ex": None, "audit": None, "final": "approve", "decided_by": "C1;C2", "route": "agreement",
           "writer": _writer(q), "h1b_edits": ["P1: accept the DC Blocker too (a28)"]}
    out = _rows(fill.filled("rubric_dev", [rec], require_all=False))[0]
    assert "note: expert review (H1b) -- P1: accept the DC Blocker too (a28)" in out["checker_1_note"]


def test_the_final_rubric_fields_are_the_writers_or_the_experts():
    from grcbench.review import apply
    rec = {"key": ("q1",), "final": "approve", "decided_by": "C1;C2", "route": "agreement", "ex": None,
           "writer": _writer()}
    assert apply.final_fields(rec) is rec["writer"]
    ex = _writer(value="approve with edits", required_points=["P1: other"])
    assert apply.final_fields({**rec, "final": "approve with edits", "decided_by": "EX", "ex": ex}) is ex
    assert apply.final_fields({**rec, "decided_by": "EX", "ex": {"value": "approve", "notes": "fine"}}) is rec["writer"]
    assert apply.final_fields({**rec, "final": "underspecified"}) is None


def test_calc_status_names_who_confirmed_the_value():
    from grcbench.review import apply
    rubric_by_checkers = {"decided_by": "C1;C2"}
    assert apply.calc_status_text("factual", None, rubric_by_checkers, "2026-11-15") == ""
    calc = {"final": "agrees", "decided_by": "R1;R2"}
    assert apply.calc_status_text("calculation", calc, rubric_by_checkers, "2026-11-15") == \
        "recomputed by R1;R2 2026-11-15"
    assert apply.calc_status_text("calculation", {**calc, "final": "disagrees", "decided_by": "EX"},
                                  rubric_by_checkers, "2026-11-15") == "decided by EX 2026-11-15"
    assert apply.calc_status_text("calculation", {**calc, "final": "no value"}, rubric_by_checkers,
                                  "2026-11-15") == "no script value (R1;R2 2026-11-15); value checked by C1;C2 2026-11-15"
    assert apply.calc_status_text("calculation", calc, rubric_by_checkers, "2026-11-15",
                                  calc_date="2026-11-10") == "recomputed by R1;R2 2026-11-10"
    assert apply.calc_status_text("calculation", {**calc, "final": "no value"}, rubric_by_checkers, "2026-11-15",
                                  calc_date="2026-11-10") == \
        "no script value (R1;R2 2026-11-10); value checked by C1;C2 2026-11-15"
    assert apply.calc_status_text("calculation", None, rubric_by_checkers, "2026-11-15") == \
        "value checked by C1;C2 2026-11-15"
    assert apply.calc_status_text("calculation", None, {"decided_by": "EX"}, "2026-11-15") == \
        "value decided by EX 2026-11-15"


def test_rubric_file_has_one_record_per_question_and_refuses_a_missing_decision():
    from grcbench.review import apply
    qs = [{"id": "q1", "split": "dev", "task_type": "factual", "question": "Q1?"},
          {"id": "q2", "split": "dev", "task_type": "calculation", "question": "Q2?"}]
    base = {"route": "agreement", "ex": None, "audit": None, "decided_by": "C1;C2"}
    recs = [{**base, "key": ("q1",), "final": "approve", "writer": _writer("q1")},
            {**base, "key": ("q2",), "final": "underspecified", "writer": _writer("q2", value="underspecified")}]
    out = apply.rubric_file(qs, recs, calc={}, date="2026-11-15")
    assert [r["question_id"] for r in out] == ["q1", "q2"]
    assert out[0]["review"]["status"] == "approved" and out[0]["calc_status"] == ""
    assert out[1]["review"]["status"] == "rejected" and out[1]["calc_status"] == "value checked by C1;C2 2026-11-15"
    with pytest.raises(ValueError, match="q2"):
        apply.rubric_file(qs, recs[:1], calc={}, date="2026-11-15")


def test_the_final_rubric_passes_the_same_automatic_checks_as_the_writers():
    from grcbench.review import rubric
    base = {"key": ("q1",), "route": "edits to merge", "audit": None, "writer": _writer()}
    good = {**base, "final": "approve with edits", "decided_by": "EX",
            "ex": _writer(value="approve with edits", required_points=["P1: states X"])}
    assert rubric.final_problems(good, confirmed=None, teacher=[]) == []
    bad = {**good, "ex": _writer(value="approve with edits", required_points=["states X"])}
    assert any("P1" in p for p in rubric.final_problems(bad, confirmed=None, teacher=[]))
    gone = {**base, "final": "underspecified", "decided_by": "C1;C2", "ex": None}
    assert rubric.final_problems(gone, confirmed=None, teacher=[]) == []


def test_an_expert_review_edit_patches_the_decided_rubric_and_says_so():
    from grcbench.review import apply
    qs = [{"id": "q1", "split": "dev", "task_type": "factual", "question": "Q1?"}]
    recs = [{"key": ("q1",), "route": "agreement", "ex": None, "audit": None, "decided_by": "C1;C2",
             "final": "approve", "writer": _writer("q1")}]
    edit = [{"item": "q1", "path": ["required_points", 0, "text"], "op": "set", "value": "states X or X2",
             "edit": "P1: accept X2 too -- expert review (H1b)"}]
    out = apply.rubric_file(qs, recs, calc={}, date="2026-11-15", patches=edit)[0]
    assert out["required_points"][0] == {"id": "P1", "weight": 2.0, "text": "states X or X2"}
    assert out["review"]["status"] == "approved-with-edits" and out["review"]["reviewers"] == ["W", "C1", "C2", "EX"]
    assert out["review"]["notes"].endswith("; expert review (H1b): P1: accept X2 too -- expert review (H1b)")
    assert apply.rubric_file(qs, recs, calc={}, date="2026-11-15")[0]["review"]["status"] == "approved"
    note = [{"item": "q1", "edit": "kept as written: a premise-correction item"}]     # no path: a decision, no change
    kept = apply.rubric_file(qs, recs, calc={}, date="2026-11-15", patches=note)[0]
    assert kept["required_points"][0]["text"] == "states X" and kept["review"]["status"] == "approved"
    assert kept["review"]["reviewers"] == ["W", "C1", "C2", "EX"]
    assert kept["review"]["notes"].endswith("; expert review (H1b): kept as written: a premise-correction item")


def test_a_numeric_reference_is_checked_against_the_confirmed_calculation():
    from grcbench.review import rubric
    conf = {"outcome": "agrees", "confirmed": {"value": 10000.0, "unit": "Hz"}}
    assert rubric.numeric_problem({"value": 10000, "unit": "Hz", "tolerance": 0.01}, conf) is None
    assert "confirmed" in rubric.numeric_problem({"value": 12000, "unit": "Hz", "tolerance": 0.01}, conf)
    assert "confirmed" in rubric.numeric_problem(None, conf)
    assert rubric.numeric_problem(None, None) is None and rubric.numeric_problem(None, {"confirmed": None}) is None


def test_an_edit_patch_on_a_rejected_rubric_is_refused():
    from grcbench.review import apply
    qs = [{"id": "q1", "split": "dev", "task_type": "factual", "question": "Q1?"}]
    recs = [{"key": ("q1",), "route": "agreement", "ex": None, "audit": None, "decided_by": "C1;C2",
             "final": "underspecified", "writer": _writer("q1", value="underspecified")}]
    edit = [{"item": "q1", "path": ["required_points", 0, "text"], "op": "set", "value": "x", "edit": "P1"}]
    with pytest.raises(ValueError, match="rejected rubric"):
        apply.rubric_file(qs, recs, calc={}, date="2026-11-15", patches=edit)


def test_a_patch_for_a_file_the_writer_does_not_pass_is_refused(tmp_path):
    from grcbench.review import items
    (tmp_path / "tier_a_patches.json").write_text(json.dumps([
        {"item": "gba-test-1", "file": "start.grc", "path": ["text"], "op": "replace_text", "old": "a", "new": "b",
         "edit": "start: rename"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="start.grc"):
        items.apply("tier_a", "gba-test-1", {"id": "gba-test-1", "version": 1, "review": {}}, {}, reviews_dir=tmp_path)


def test_an_unsure_expert_row_needs_a_comment():
    from grcbench.review import expert
    assert "comment required" in " ".join(expert.problems(_verdict("x", correct="unsure", verdict="", comment="")))


def test_one_agent_may_not_hold_both_reviewer_roles_of_a_sheet(tmp_path):
    from grcbench.review import coordinate
    _jsonl(tmp_path / "out" / "a01-R1-K.jsonl", [_twin("a", "b", "twin")])
    _jsonl(tmp_path / "out" / "a01-R2-K.jsonl", [_twin("a", "b", "twin")])
    with pytest.raises(ValueError, match="a01"):
        coordinate.check_independence(tmp_path, "twins")


def test_a_question_whose_rubric_is_rejected_is_removed_from_the_split_and_counted(tmp_path, monkeypatch):
    from grcbench.tier_k import run as krun
    folder = tmp_path / "dev"
    _jsonl(folder / "questions.jsonl", [{"id": "q1", "question": "Q1?"}, {"id": "q2", "question": "Q2?"}])
    _jsonl(folder / "rubrics.jsonl", [{"question_id": "q1", "review": {"status": "approved"}},
                                      {"question_id": "q2", "review": {"status": "rejected",
                                                                       "notes": "underspecified: no rate"}}])
    monkeypatch.setattr(krun, "ITEMS_K", tmp_path)
    assert [i["id"] for i in krun.load_split("dev")] == ["q1"]
    assert krun.removed("dev") == [{"question_id": "q2", "notes": "underspecified: no rate"}]


def test_a_rejected_block_question_is_removed_from_the_split_too(tmp_path, monkeypatch):
    from grcbench.tier_k import run as krun
    folder = tmp_path / "test"
    _jsonl(folder / "frozen_questions.jsonl", [{"id": "q1", "question": "Q1?"}])
    _jsonl(folder / "block_questions.jsonl", [{"id": "bq1", "question": "B1?", "review": {"status": "approved"}},
                                              {"id": "bq2", "question": "B2?",
                                               "review": {"status": "rejected", "notes": "ambiguous port"}}])
    _jsonl(folder / "rubrics.jsonl", [{"question_id": "q1", "review": {"status": "approved"}}])
    monkeypatch.setattr(krun, "ITEMS_K", tmp_path)
    assert [i["id"] for i in krun.load_split("test")] == ["q1", "bq1"]
    assert krun.removed("test") == [{"question_id": "bq2", "notes": "ambiguous port"}]


# ------------------------------------------------------------------ summary report ---------------

def test_sheet_summary_counts_routes_audit_and_agreement():
    from grcbench.review import report
    spec = sheets.SHEETS["twins"]
    keys = [("lexical", f"a{i}", f"b{i}") for i in range(10)]
    r1 = {k: _out("twin") for k in keys}
    r2 = {k: _out("twin" if i < 8 else "not twin") for i, k in enumerate(keys)}
    ex = {keys[8]: _out("not twin"), keys[9]: _out("twin")}
    audit = {keys[0]: _out("twin"), keys[1]: _out("not twin")}
    recs = records.build(spec, r1, r2, ex, audit)
    s = report.sheet_summary("twins", recs)
    assert s["rows"] == 10 and s["agreed"] == 8 and s["expert"] == 2
    assert s["audit"] == {"n": 2, "disagreements": 1, "rate": 0.5, "full_review": True}
    assert s["final"] == {"twin": 8, "not twin": 2}
    assert s["observed_agreement"] == 0.8 and s["kappa"] == pytest.approx(0.0)
    text = report.render([s], runs=[])
    assert "two runs of one model, not agreement between people" in text and "| twins |" in text


def test_the_audit_sample_and_the_full_review_it_triggers_are_reported_apart():
    from grcbench.review import report
    spec = sheets.SHEETS["twins"]
    keys = [("lexical", f"a{i}", f"b{i}") for i in range(6)]
    r1 = {k: _out("twin") for k in keys}
    audit = {keys[0]: _out("not twin"), keys[1]: _out("twin"),
             keys[2]: {**_out("twin"), "full_review": True}, keys[3]: {**_out("not twin"), "full_review": True}}
    s = report.sheet_summary("twins", records.build(spec, r1, dict(r1), {}, audit))
    assert s["audit"] == {"n": 2, "disagreements": 1, "rate": 0.5, "full_review": True}
    assert s["full_review"] == {"n": 2, "changed": 1}
    assert s["audit_changed"] == 2
    text = report.render([s], runs=[])
    assert "1 of 2 (full review: 1 of 2 changed)" in text
    whole = report.sheet_summary("twins", records.build(spec, {k: r1[k] for k in keys[:2]},
                                                        {k: r1[k] for k in keys[:2]}, {}, {k: audit[k] for k in keys[:2]}))
    assert "1 of 2 (full review: every agreed row is in the sample)" in report.render([whole], runs=[])
    partial = report.sheet_summary("twins", records.build(spec, r1, dict(r1), {}, {k: audit[k] for k in keys[:2]}))
    assert "1 of 2 (full review required, not done)" in report.render([partial], runs=[])


def test_the_review_summary_is_current_with_the_decided_records(tmp_path, monkeypatch):
    from grcbench.common.paths import REPO
    from grcbench.review import cli
    before = (REPO / "reports" / "review_summary.md").read_text(encoding="utf-8")
    import grcbench.common.io as cio
    written = {}
    monkeypatch.setattr(cio, "write_text", lambda path, text: written.__setitem__(str(path), text))
    assert cli.main(["summary"]) == 0
    assert list(written.values()) == [before], "run ./gb review summary after any change to reviews/"


def test_kappa_is_undefined_when_every_review_gives_one_value():
    from grcbench.review import report
    spec = sheets.SHEETS["twins"]
    keys = [("lexical", f"a{i}", f"b{i}") for i in range(3)]
    r1 = {k: _out("twin") for k in keys}
    s = report.sheet_summary("twins", records.build(spec, r1, dict(r1), {}, {}))
    assert s["kappa"] is None and s["observed_agreement"] == 1.0
    assert "| 1.0 | – |" in report.render([s], runs=[])


def test_a_removal_gives_the_reason_of_whoever_decided_it():
    from grcbench.review import report
    spec = sheets.SHEETS["tier_f"]
    k1, k2, k3 = ("t1",), ("t2",), ("t3",)
    r1 = {k1: _out("reject", notes="duplicate of t0"), k2: _out("approve", notes="fine"),
          k3: _out("approve", notes="ok")}
    r2 = {k1: _out("reject", notes="dup"), k2: _out("approve"), k3: _out("reject", notes="broken")}
    audit = {k2: _out("reject", notes="a shortcut passes V5")}
    ex = {k3: _out("reject", notes="the oracle accepts silence")}
    s = report.sheet_summary("tier_f", records.build(spec, r1, r2, ex, audit))
    why = {tuple(r["key"]): r["why"] for r in s["removed"]}
    assert why == {k1: "duplicate of t0", k2: "a shortcut passes V5", k3: "the oracle accepts silence"}
    r1[k1] = {**_out("reject", notes="note: batch -- update tone_1 freq 3600"),
              "edits": ["duplicate of t0: same cell, same family, same edit"]}
    s = report.sheet_summary("tier_f", records.build(spec, r1, r2, ex, audit))
    assert {tuple(r["key"]): r["why"] for r in s["removed"]}[k1] == "duplicate of t0: same cell, same family, same edit"


def test_the_audit_compares_the_experts_value_with_the_rules_agreed_value():
    """For calc the agreed value is the rule's (`agrees`), not a reviewer's number."""
    from grcbench.review import coordinate, report
    keys = {"q1": {"specification": "determined", "value": 10000.0, "unit": "Hz", "tolerance": 0.01,
                   "tolerance_db": None}}
    recs = records.build(sheets.SHEETS["calc"], {("q1",): {"value": "10 kHz", "unit": ""}},
                         {("q1",): {"value": "10000", "unit": "Hz"}}, {}, audit={("q1",): _out("agrees")},
                         router=coordinate.calc_router(keys))
    assert recs[0]["agreed"] == "agrees" and recs[0]["final"] == "agrees"
    s = report.sheet_summary("calc", recs, calc_keys=keys)
    assert s["audit"]["disagreements"] == 0
    assert s["observed_agreement"] == 1.0           # both recomputations agree with the key: same outcome
    text = report.render([s], runs=[], notes=["the block-question audit's full review is the H1b review"])
    assert "## Coordinator notes" in text and "H1b review" in text


# ------------------------------------------------------------------ expert review (H1b) ----------

def _verdict(item_id, correct="yes", relevant="yes", verdict="accept", **kw):
    return {"sheet": "expert_review", "key": {"item_id": item_id}, "correct": correct, "relevant": relevant,
            "verdict": verdict, "reason_code": kw.get("reason_code", ""), "comment": kw.get("comment", ""),
            "suggested_fix": kw.get("suggested_fix", ""), "evidence": [], "notes": "", "edits": []}


def test_expert_review_rows_are_checked_and_h1b_counted_per_tier():
    from grcbench.review import expert
    items = [{"item_id": "bq1", "item_kind": "gnuradio_block_question", "split": "test", "packet_ref": "p"},
             {"item_id": "bq2", "item_kind": "gnuradio_block_question", "split": "test", "packet_ref": "p"},
             {"item_id": "gbf-test-1", "item_kind": "tier_f_task", "split": "test", "packet_ref": "p"},
             {"item_id": "gbf-test-2", "item_kind": "tier_f_task", "split": "test", "packet_ref": "p"}]
    rows = {("bq1",): _verdict("bq1"),
            ("bq2",): _verdict("bq2", verdict="accept with edits", reason_code="ambiguous_wording"),
            ("gbf-test-1",): _verdict("gbf-test-1", correct="no", verdict="reject", reason_code="broken_oracle"),
            ("gbf-test-2",): _verdict("gbf-test-2", correct="unsure", verdict="")}
    assert expert.problems(rows[("bq1",)]) == []
    assert expert.problems(_verdict("x", verdict="reject", comment="c")) == ["reason_code required"]
    assert expert.problems(_verdict("x", verdict="reject", reason_code="other")) == ["comment required"]
    assert expert.problems(_verdict("x", correct="unsure", comment="no source")) == ["verdict must be empty while unsure"]
    h = expert.h1b(items, rows)
    assert h["K"] == {"items": 2, "accepted": 2, "accepted_no_edit": 1, "unsure": 0, "share": 1.0, "holds": True,
                      "kept": 2, "accepted_kept": 2, "share_kept": 1.0}
    assert h["F"]["accepted"] == 0 and h["F"]["unsure"] == 1 and h["F"]["holds"] is False
    text = expert.sheet_text(items, rows, date="2026-11-12")
    out = list(csv.DictReader(io.StringIO(text)))
    assert [r["item_id"] for r in out] == ["bq1", "bq2", "gbf-test-1", "gbf-test-2"]
    assert out[0]["expert"] == "EX" and out[0]["date"] == "2026-11-12" and out[2]["reason_code"] == "broken_oracle"
    with pytest.raises(ValueError, match="no expert row"):
        expert.sheet_text(items, {k: v for k, v in rows.items() if k != ("bq1",)}, date="2026-11-12")


def test_h1b_counts_the_first_pass_and_describes_the_items_the_review_kept(tmp_path):
    from grcbench.review import expert
    items = [{"item_id": f"gba-test-{i}", "item_kind": "tier_a_task", "split": "test", "packet_ref": "p"}
             for i in range(1, 5)]
    row = lambda i, **kw: {**_verdict(f"gba-test-{i}", **kw), "sheet": "expert_review", "key": {"item_id": f"gba-test-{i}"}}
    _jsonl(tmp_path / "out" / "a26-EX-H1b.jsonl",
           [row(1), row(2), row(3, correct="unsure", verdict=""),
            row(4, correct="no", verdict="reject", reason_code="other", comment="removed")])
    _jsonl(tmp_path / "out" / "a27-EX-H1b.jsonl", [row(3)])                 # a later row: not the first pass
    first, later = expert.first_pass(tmp_path)
    assert first[("gba-test-3",)]["correct"] == "unsure" and ("gba-test-3",) in later
    h = expert.h1b(items, first, removed={"gba-test-4"})["A"]
    assert (h["items"], h["accepted"], h["unsure"], h["share"], h["holds"]) == (4, 2, 1, 0.5, False)
    assert (h["kept"], h["accepted_kept"], h["share_kept"]) == (3, 2, 0.6667)


def test_the_expert_review_is_written_as_records_sheet_and_h1b_and_summarised(tmp_path):
    from grcbench.review import expert, report
    items = [{"item_id": "bq1", "item_kind": "gnuradio_block_question", "split": "test", "packet_ref": "p"},
             {"item_id": "gba-test-1", "item_kind": "tier_a_task", "split": "test", "packet_ref": "p"}]
    _jsonl(tmp_path / "out" / "a26-EX-H1b.jsonl",
           [{**_verdict("bq1"), "sheet": "expert_review", "key": {"item_id": "bq1"}},
            {**_verdict("gba-test-1", correct="no", verdict="reject", reason_code="other", comment="padded"),
             "sheet": "expert_review", "key": {"item_id": "gba-test-1"}}])
    h = expert.write(tmp_path, items, date="2026-11-15", reviews_dir=tmp_path / "reviews",
                     sheet=tmp_path / "expert_review.csv")
    assert h["K"]["holds"] is True and h["A"]["accepted"] == 0
    assert json.loads((tmp_path / "reviews" / "expert_review_h1b.json").read_text())["A"]["items"] == 1
    recs = [json.loads(l) for l in (tmp_path / "reviews" / "expert_review.jsonl").read_text().splitlines()]
    assert [r["item_id"] for r in recs] == ["bq1", "gba-test-1"] and recs[1]["verdict"] == "reject"
    assert (tmp_path / "expert_review.csv").exists()
    text = report.render([], runs=[], h1b=h)
    assert "## Expert review (H1b)" in text and "| A | 1 | 0 |" in text
    bad = {**_verdict("bq1", verdict="reject"), "sheet": "expert_review", "key": {"item_id": "bq1"}}
    _jsonl(tmp_path / "out" / "a26-EX-H1b.jsonl", [bad])
    with pytest.raises(ValueError, match="bq1"):
        expert.write(tmp_path, items[:1], date="2026-11-15", reviews_dir=tmp_path / "reviews",
                     sheet=tmp_path / "expert_review.csv")


# ------------------------------------------------------------------ rubric checks (all) ----------

def test_rubric_writer_row_checks(tmp_path):
    from grcbench.review import rubric
    f = tmp_path / "x.block.yml"
    f.write_text("label: Sample Rate\n", encoding="utf-8")
    ok = _writer(sources=[f'{f} :: "label: Sample Rate"'])
    assert rubric.problems(ok, confirmed=None, teacher=[]) == []
    assert any("quote not found" in p for p in rubric.problems(_writer(sources=[f'{f} :: "label: Rate X"']),
                                                               confirmed=None, teacher=[]))
    assert any("P1" in p for p in rubric.problems(_writer(required_points=["states X"]), confirmed=None, teacher=[]))
    num = _writer(numeric={"value": 10000, "unit": "Hz", "tolerance": 0.01}, numeric_tolerance="1 %")
    conf = {"outcome": "agrees", "confirmed": {"value": 10000.0, "unit": "Hz"}}
    assert rubric.problems(num, confirmed=conf, teacher=[]) == []
    bad = {"outcome": "disagrees", "confirmed": {"value": [11000.0], "unit": ["Hz"]}}
    assert any("confirmed" in p for p in rubric.problems(num, confirmed=bad, teacher=[]))
    under = {"outcome": "underspecified", "confirmed": None}
    assert any("underspecified" in p for p in rubric.problems(num, confirmed=under, teacher=[]))
    assert rubric.problems(_writer(value="underspecified", required_points=[]), confirmed=under, teacher=[]) == []
    copied = _writer(required_points=["P1: the sampling rate must be at least twice the highest frequency"])
    assert any("teacher" in p for p in rubric.problems(
        copied, confirmed=None, teacher=["so the sampling rate must be at least twice the highest frequency here"]))


# ------------------------------------------------------------------ patches (decided edits) ------

def test_patches_set_delete_append_and_replace_text():
    from grcbench.review import patches
    rec = {"instruction": "Save the flowgraph.", "acceptance": {"blocks_present": [{"params": {"file": "a.dat"}}],
                                                               "facts": [1, 2]}}
    ps = [{"item": "t1", "path": ["acceptance", "blocks_present", 0, "params", "file"], "op": "set",
           "value": {"regex": "(^|/)a\\.dat$"}, "edit": "acceptance: regex"},
          {"item": "t1", "path": ["acceptance", "facts"], "op": "append", "value": 3, "edit": "e2"},
          {"item": "t1", "path": ["instruction"], "op": "replace_text", "old": "Save the flowgraph.",
           "new": "Save the flowgraph in the project folder.", "edit": "e3"},
          {"item": "t1", "path": ["acceptance", "facts", 0], "op": "delete", "edit": "e4"}]
    out = patches.apply(rec, ps)
    assert out["acceptance"]["blocks_present"][0]["params"]["file"] == {"regex": "(^|/)a\\.dat$"}
    assert out["acceptance"]["facts"] == [2, 3]
    assert out["instruction"] == "Save the flowgraph in the project folder."
    assert rec["instruction"] == "Save the flowgraph.", "the input record is not changed"
    more = patches.apply(rec, [{"item": "t1", "path": ["instruction"], "op": "append_text", "value": " Then stop.",
                                "edit": "e5"}])
    assert more["instruction"] == "Save the flowgraph. Then stop."


def test_graph_patches_insert_a_block_between_two_and_rewire():
    from grcbench.review import patches
    spec = {"start": ["g.grc", {"blocks": [{"name": "tone", "id": "analog_sig_source_x"},
                                            {"name": "hpf", "id": "high_pass_filter"}],
                                 "connections": [["tone", "0", "hpf", "0"]]}]}
    thr = {"name": "blocks_throttle2_0", "id": "blocks_throttle2", "parameters": {"type": "float"}}
    out = patches.apply(spec, [{"item": "t", "path": ["start", 1], "op": "insert_between", "block": thr,
                                "src": "tone:0", "dst": "hpf:0", "edit": "start: throttle"}])
    g = out["start"][1]
    assert g["blocks"][-1] == thr
    assert g["connections"] == [["tone", "0", "blocks_throttle2_0", "0"], ["blocks_throttle2_0", "0", "hpf", "0"]]
    re = patches.apply(spec, [{"item": "t", "path": ["start", 1], "op": "rewire", "old": "tone:0->hpf:0",
                               "new": "hpf:0->tone:0", "edit": "e"}])
    assert re["start"][1]["connections"] == [["hpf", "0", "tone", "0"]]
    with pytest.raises(ValueError, match="connection"):
        patches.apply(spec, [{"item": "t", "path": ["start", 1], "op": "insert_between", "block": thr,
                              "src": "tone:0", "dst": "nowhere:0", "edit": "e"}])
    same = patches.apply(spec, [{"item": "t", "path": ["start", 1], "op": "insert_between", "block": thr,
                                 "src": "tone:0", "dst": "nowhere:0", "edit": "e", "optional": True}])
    assert same == spec, "an optional graph patch whose connection is absent changes nothing"


def test_a_patch_that_does_not_fit_is_refused():
    from grcbench.review import patches
    rec = {"instruction": "Save it.", "acceptance": {}}
    with pytest.raises(ValueError, match="not found"):
        patches.apply(rec, [{"item": "t", "path": ["instruction"], "op": "replace_text", "old": "nope", "new": "x",
                             "edit": "e"}])
    with pytest.raises(ValueError, match="path"):
        patches.apply(rec, [{"item": "t", "path": ["acceptance", "x", 0], "op": "set", "value": 1, "edit": "e"}])
    with pytest.raises(ValueError, match="op"):
        patches.apply(rec, [{"item": "t", "path": ["instruction"], "op": "frobnicate", "edit": "e"}])


def test_patches_are_read_per_item_and_file(tmp_path):
    from grcbench.review import patches
    f = tmp_path / "tier_a_patches.json"
    f.write_text(json.dumps([{"item": "t1", "file": "task.json", "path": ["a"], "op": "set", "value": 1, "edit": "e"},
                             {"item": "t2", "path": ["b"], "op": "set", "value": 2, "edit": "e"}]), encoding="utf-8")
    got = patches.for_item(f, "t1")
    assert len(got) == 1 and got[0]["path"] == ["a"]
    assert patches.for_item(tmp_path / "missing.json", "t1") == []


def test_decided_item_review_and_patches_for_record_and_files(tmp_path):
    from grcbench.review import items
    out = lambda v: {"value": v, "edits": [], "notes": ""}
    recs = records.build(sheets.SHEETS["tier_f"], {("gbf-test-1",): out("approve with edits"), ("gbf-test-2",): out("reject")},
                         {("gbf-test-1",): out("approve with edits"), ("gbf-test-2",): out("reject")},
                         {("gbf-test-1",): out("approve with edits")})
    records.write_records(tmp_path / "tier_f.jsonl", recs)
    (tmp_path / "tier_f_patches.json").write_text(json.dumps([
        {"item": "gbf-test-1", "path": ["instruction"], "op": "replace_text", "old": "Build", "new": "Make", "edit": "e"},
        {"item": "gbf-test-1", "file": "start.grc", "path": ["text"], "op": "replace_text", "old": "tone_440",
         "new": "tone_0", "edit": "start: rename"}]), encoding="utf-8")
    rec, files = items.apply("tier_f", "gbf-test-1", {"id": "gbf-test-1", "version": 1, "instruction": "Build it.",
                                                     "review": {"status": "pending-team-review"}},
                             {"start.grc": "name: tone_440\n"}, reviews_dir=tmp_path)
    assert rec["instruction"] == "Make it." and files["start.grc"] == "name: tone_0\n"
    assert rec["review"]["status"] == "approved-with-edits" and rec["version"] == 2
    rej, _ = items.apply("tier_f", "gbf-test-2", {"id": "gbf-test-2", "version": 1, "review": {}}, {},
                         reviews_dir=tmp_path)
    assert rej["review"]["status"] == "rejected" and rej["version"] == 1
    none, _ = items.apply("tier_f", "gbf-test-3", {"id": "gbf-test-3", "version": 1, "review": {"status": "x"}}, {},
                          reviews_dir=tmp_path)
    assert none["review"] == {"status": "x"}, "an undecided item keeps its record"
    assert items.active("tier_f", ["gbf-test-1", "gbf-test-2", "gbf-test-3"], reviews_dir=tmp_path) == [
        "gbf-test-1", "gbf-test-3"]


def test_a_decided_task_keeps_its_own_review_notes_after_the_decided_edits(tmp_path):
    from grcbench.review import items
    out = lambda v: {"value": v, "edits": [], "notes": ""}
    recs = records.build(sheets.SHEETS["tier_f"], {("gbf-test-11",): out("approve with edits"),
                                                   ("gbf-test-12",): out("approve")},
                         {("gbf-test-11",): out("approve with edits"), ("gbf-test-12",): out("approve")},
                         {("gbf-test-11",): out("approve with edits")})
    records.write_records(tmp_path / "tier_f.jsonl", recs)
    (tmp_path / "tier_f_patches.json").write_text(json.dumps([
        {"item": "gbf-test-11", "path": ["review", "notes"], "op": "replace_text", "old": "(2) Silence passes. ",
         "new": "", "edit": "metadata: delete sentence (2)"}]), encoding="utf-8")
    own = "Known oracle limits: (1) A tone passes. (2) Silence passes. (3) Noise fails."
    rec, _ = items.apply("tier_f", "gbf-test-11", {"id": "gbf-test-11", "version": 1,
                                                   "review": {"status": "pending-team-review", "notes": own}},
                         {}, reviews_dir=tmp_path)
    assert rec["review"]["notes"] == ("decided by EX (edits to merge); decided edits applied. "
                                      "Known oracle limits: (1) A tone passes. (3) Noise fails.")
    plain, _ = items.apply("tier_f", "gbf-test-12", {"id": "gbf-test-12", "version": 1,
                                                    "review": {"status": "pending-team-review", "notes": ""}},
                           {}, reviews_dir=tmp_path)
    assert plain["review"]["notes"] == "decided by R1;R2"


def test_a_rejected_task_is_refused_by_the_runners():
    from grcbench.review import items
    from grcbench.runlog.log import RunRefused
    items.refuse_rejected([{"id": "t1", "review": {"status": "approved"}}, {"id": "t2", "review": {}}])
    with pytest.raises(RunRefused, match="t3.*rejected"):            # the CLIs print "refused: ...", no traceback
        items.refuse_rejected([{"id": "t1", "review": {"status": "approved"}},
                               {"id": "t3", "review": {"status": "rejected"}}])
    import inspect
    from grcbench.tier_a import runner, tier_f_mode
    assert "refuse_rejected" in inspect.getsource(runner.run_tasks)
    assert "refuse_rejected" in inspect.getsource(tier_f_mode.run_tier_f_samples)


def test_a_split_wide_run_lists_only_the_tasks_the_review_kept(tmp_path):
    from grcbench import e2e
    from grcbench.review import items
    for name, status in (("t1", "approved"), ("t2", "rejected"), ("t3", "approved-with-edits")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "task.json").write_text(json.dumps({"id": name, "review": {"status": status}}))
    assert [d.name for d in items.active_dirs(sorted(tmp_path.iterdir()))] == ["t1", "t3"]
    import inspect
    assert "active_dirs" in inspect.getsource(e2e._tier_f) and "active_dirs" in inspect.getsource(e2e._tier_a)


def test_a_patched_task_that_was_approved_becomes_approved_with_edits(tmp_path):
    from grcbench.review import items
    out = lambda v: {"value": v, "edits": [], "notes": ""}
    recs = records.build(sheets.SHEETS["tier_f"], {("gbf-test-5",): out("approve")}, {("gbf-test-5",): out("approve")}, {})
    records.write_records(tmp_path / "tier_f.jsonl", recs)
    (tmp_path / "tier_f_patches.json").write_text(json.dumps([
        {"item": "gbf-test-5", "path": ["instruction"], "op": "replace_text", "old": "Build", "new": "Make",
         "edit": "instruction: say Make -- expert review (H1b)"}]), encoding="utf-8")
    rec, _ = items.apply("tier_f", "gbf-test-5", {"id": "gbf-test-5", "version": 1, "instruction": "Build it.",
                                                 "review": {"status": "pending-team-review", "notes": ""}}, {},
                         reviews_dir=tmp_path)
    assert rec["instruction"] == "Make it." and rec["version"] == 2
    assert rec["review"]["status"] == "approved-with-edits" and rec["review"]["reviewers"][-1] == "EX"
    assert "instruction: say Make -- expert review (H1b)" in rec["review"]["notes"]


def test_a_post_review_edit_is_named_as_one_in_the_tasks_notes(tmp_path):
    """An edit made after a task's review (a patch with `post_review`, pre-registration §10) changes an approved
    task to approved-with-edits and is named in its notes; it is not an expert edit."""
    from grcbench.review import items
    out = lambda v: {"value": v, "edits": [], "notes": ""}
    recs = records.build(sheets.SHEETS["tier_f"], {("gbf-test-5",): out("approve"), ("gbf-test-6",): out("approve with edits")},
                         {("gbf-test-5",): out("approve"), ("gbf-test-6",): out("approve with edits")},
                         {("gbf-test-6",): out("approve with edits")})
    records.write_records(tmp_path / "tier_f.jsonl", recs)
    (tmp_path / "tier_f_patches.json").write_text(json.dumps([
        {"item": i, "path": ["instruction"], "op": "replace_text", "old": old, "new": new,
         "edit": "oracle: add a check -- a gap", "post_review": "decision of 17 Nov"}
        for i in ("gbf-test-5", "gbf-test-6") for old, new in (("Build", "Make"), (" it", " this"))]),
        encoding="utf-8")       # one edit applied as two patches is named once
    task = lambda i: {"id": i, "version": 1, "instruction": "Build it.", "review": {"status": "pending-team-review"}}
    rec, _ = items.apply("tier_f", "gbf-test-5", task("gbf-test-5"), {}, reviews_dir=tmp_path)
    assert rec["instruction"] == "Make this." and rec["version"] == 2
    assert rec["review"]["status"] == "approved-with-edits" and rec["review"]["reviewers"] == ["R1", "R2"]
    assert rec["review"]["notes"] == "decided by R1;R2; post-review edit (decision of 17 Nov): oracle: add a check -- a gap"
    rec, _ = items.apply("tier_f", "gbf-test-6", task("gbf-test-6"), {}, reviews_dir=tmp_path)
    assert rec["review"]["status"] == "approved-with-edits" and "expert review (H1b)" not in rec["review"]["notes"]
    assert rec["version"] == 3                  # edited by the review (2), then after it (3)
    assert rec["review"]["notes"].endswith("; post-review edit (decision of 17 Nov): oracle: add a check -- a gap")


# ------------------------------------------------------------------ declared runs ----------------

def test_every_agent_run_is_declared_and_the_cap_of_each_round_holds(tmp_path):
    """30 runs for the first round (agents 1-30) and 8 for the second, the replacement tasks (31-38)."""
    from grcbench.review import log
    path = tmp_path / "review_log.jsonl"
    for i in range(1, 31):
        log.declare(path, agent=i, role="R1", sheets=["twins"], rows=10, brief_text=f"brief {i}",
                    date="2026-11-10")
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 30 and entries[0]["model"] == log.MODEL
    assert len(entries[0]["brief_sha256"]) == 64 and entries[0]["output_sha256"] is None
    assert log.ROUNDS == {1: (1, 30), 2: (31, 38)} and log.MAX_AGENTS == 38
    for i in range(31, 39):
        log.declare(path, agent=i, role="R1", sheets=["tier_f"], rows=1, brief_text=f"b{i}", date="2026-11-17")
    with pytest.raises(ValueError, match="38"):
        log.declare(path, agent=39, role="R1", sheets=["twins"], rows=1, brief_text="x", date="2026-11-10")
    with pytest.raises(ValueError, match="already"):
        log.declare(path, agent=3, role="R1", sheets=["twins"], rows=1, brief_text="x", date="2026-11-10")
    for n in (0, 50):                                # a number outside both rounds, even on a short log
        with pytest.raises(ValueError, match="round"):
            log.declare(tmp_path / "other.jsonl", agent=n, role="R1", sheets=["twins"], rows=1, brief_text="x",
                        date="2026-11-17")


def test_an_output_hash_is_recorded_when_the_run_ends(tmp_path):
    from grcbench.review import log
    path = tmp_path / "review_log.jsonl"
    out = tmp_path / "out.jsonl"
    out.write_text('{"a": 1}\n', encoding="utf-8")
    log.declare(path, agent=1, role="EX", sheets=["twins"], rows=3, brief_text="b", date="2026-11-10")
    log.finish(path, agent=1, outputs=[out])
    e = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert len(e["output_sha256"]) == 64 and e["rows_written"] == 1


# ------------------------------------------------------------------ expert review scope ----------

def test_the_gnu_radio_question_rule_is_fixed_and_matches_the_expert_guide():
    from grcbench.review import scope
    assert scope.about_gnuradio({"question": "How do I set a GRC variable?", "task_type": "design"})
    assert scope.about_gnuradio({"question": "What is a QPSK symbol?", "task_type": "gnuradio_usage"})
    assert not scope.about_gnuradio({"question": "What is a QPSK symbol?", "task_type": "factual"})
    items = scope.expert_items()
    kinds = {}
    for it in items:
        kinds[it["item_kind"]] = kinds.get(it["item_kind"], 0) + 1
    guide = (SHEET_DIR.parents[1] / "docs" / "checklists" / "expert_review.md").read_text(encoding="utf-8")
    assert f"| **Total** | **{len(items)}** |" in guide
    assert f"| The frozen test questions that are about GNU Radio | {kinds['gnuradio_frozen_question']} |" in guide
    assert f"| The dev questions that are about GNU Radio | {kinds['gnuradio_dev_question']} |" in guide
    assert len({it["item_id"] for it in items}) == len(items)


# ------------------------------------------------------------------ outputs and inputs -----------

def test_agent_output_rows_load_by_key_and_the_last_row_wins(tmp_path):
    p = tmp_path / "out.jsonl"
    rows = [{"sheet": "twins", "key": {"source": "lexical", "id_a": "a", "id_b": "b"}, "value": "twin",
             "notes": "", "evidence": []},
            {"sheet": "twins", "key": {"source": "lexical", "id_a": "a", "id_b": "b"}, "value": "not twin",
             "notes": "second look", "evidence": []}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    got = records.load_outputs([p])
    assert got["twins"][("lexical", "a", "b")]["value"] == "not twin"


def test_expert_review_rows_load_by_item_id_next_to_sheet_rows(tmp_path):
    p = tmp_path / "a28-EX-mix.jsonl"
    rows = [{"sheet": "expert_review", "key": {"item_id": "q_1"}, "correct": "yes", "relevant": "yes",
             "verdict": "accept", "notes": "", "edits": [], "evidence": []},
            {"sheet": "twins", "key": {"source": "lexical", "id_a": "a", "id_b": "b"}, "value": "twin",
             "notes": "", "edits": [], "evidence": []}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    got = records.load_outputs([p])
    assert got["expert_review"][("q_1",)]["verdict"] == "accept"
    assert got["twins"][("lexical", "a", "b")]["value"] == "twin"


def test_agent_output_rows_need_the_sheets_key_columns(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text(json.dumps({"sheet": "twins", "key": {"id_a": "a"}, "value": "twin"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="key"):
        records.load_outputs([p])


def test_blinded_input_copy_has_no_hidden_column(tmp_path):
    from grcbench.review import inputs
    out = inputs.write_blinded("block_questions", tmp_path)
    header = next(csv.reader(io.StringIO(out.read_text(encoding="utf-8"))))
    assert "answer" not in header and "evidence" not in header and "question" in header
    n = len(list(csv.DictReader(io.StringIO(out.read_text(encoding="utf-8")))))
    assert n == len(list(csv.DictReader(io.StringIO((SHEET_DIR / "tier_k_block_questions.csv")
                                                      .read_text(encoding="utf-8")))))


# ---------------------------------------------------------------- round 2: the replacement items
def test_a_second_round_is_decided_onto_the_first_rounds_records(tmp_path):
    from grcbench.review import coordinate
    round1, round2, reviews = tmp_path / "w1", tmp_path / "w2", tmp_path / "reviews"
    _jsonl(round1 / "out" / "a01-R1-K.jsonl", [_twin("a", "b", "twin")])
    _jsonl(round1 / "out" / "a02-R2-K.jsonl", [_twin("a", "b", "twin")])
    _jsonl(round1 / "out" / "a07-EX.jsonl", [{**_twin("a", "b", "twin"), "pass": "audit"}])
    coordinate.decide_sheet(round1, "twins", reviews_dir=reviews, sheet_dir=tmp_path / "s", require_all=False)
    _jsonl(round2 / "out" / "a31-R1-F.jsonl", [_twin("c", "d", "not twin")])
    _jsonl(round2 / "out" / "a32-R2-F.jsonl", [_twin("c", "d", "not twin")])
    _jsonl(round2 / "out" / "a35-EX.jsonl", [{**_twin("c", "d", "not twin"), "pass": "audit"}])
    recs = coordinate.decide_sheet(round2, "twins", reviews_dir=reviews, sheet_dir=tmp_path / "s",
                                   require_all=False, merge=True)
    assert [r["key"] for r in recs] == [("lexical", "a", "b"), ("lexical", "c", "d")]
    with pytest.raises(ValueError, match="already decided"):        # a second round never re-decides a row
        coordinate.decide_sheet(round2, "twins", reviews_dir=reviews, sheet_dir=tmp_path / "s",
                                require_all=False, merge=True)
    # a second round decided without --merge would drop the first round's records: refused before any write
    before = (reviews / "twins.jsonl").read_bytes()
    with pytest.raises(ValueError, match="--merge"):
        coordinate.decide_sheet(round2, "twins", reviews_dir=reviews, sheet_dir=tmp_path / "s2", require_all=False)
    assert (reviews / "twins.jsonl").read_bytes() == before and not (tmp_path / "s2").exists()
    # a failing fill writes nothing either
    with pytest.raises(ValueError, match="no decision"):
        coordinate.decide_sheet(round1, "twins", reviews_dir=tmp_path / "r3", sheet_dir=tmp_path / "s3")
    assert not (tmp_path / "r3" / "twins.jsonl").exists()


def test_a_second_rounds_input_holds_only_the_undecided_rows(tmp_path, monkeypatch):
    from grcbench.review import inputs, records
    from grcbench.review.sheets import SHEETS
    header = (SHEET_DIR / SHEETS["twins"].file).read_text(encoding="utf-8-sig").splitlines()[0]
    cols = header.split(",")
    rows = [dict.fromkeys(cols, "") | {"source": "lexical", "id_a": a, "id_b": b} for a, b in (("a", "b"), ("c", "d"))]
    sheet_dir = tmp_path / "sheets"
    sheet_dir.mkdir()
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    (sheet_dir / SHEETS["twins"].file).write_text(buf.getvalue(), encoding="utf-8")
    monkeypatch.setattr(inputs, "SHEET_DIR", sheet_dir)
    records.write_records(tmp_path / "reviews" / "twins.jsonl",
                          [_rec("twins", ("lexical", "a", "b"), {"value": "twin"}, {"value": "twin"}, "twin", "R1;R2")])
    out = inputs.write_blinded("twins", tmp_path / "in", undecided_only=True, reviews_dir=tmp_path / "reviews")
    assert [(r["id_a"], r["id_b"]) for r in _rows(out.read_text(encoding="utf-8"))] == [("c", "d")]
    # the route's check for rows a reviewer left out looks only at the rows still to decide
    from grcbench.review import coordinate
    _jsonl(tmp_path / "w2" / "out" / "a31-R1-K.jsonl", [_twin("c", "d", "twin")])
    _jsonl(tmp_path / "w2" / "out" / "a32-R2-K.jsonl", [_twin("c", "d", "twin")])
    missing = coordinate.unreviewed(tmp_path / "w2", "twins", reviews_dir=tmp_path / "reviews")
    assert missing == {"R1": [], "R2": [], "unknown": []}


def test_a_second_expert_review_round_merges_and_h1b_counts_the_original_pool(tmp_path):
    from grcbench.review import expert
    items = [{"item_id": f"gba-test-{i}", "item_kind": "tier_a_task", "split": "test", "packet_ref": "p"}
             for i in (1, 2, 31)]
    row = lambda i, **kw: {**_verdict(f"gba-test-{i}", **kw), "sheet": "expert_review", "key": {"item_id": f"gba-test-{i}"}}
    _jsonl(tmp_path / "w1" / "out" / "a27-EX-H1b.jsonl",
           [row(1), row(2, correct="no", verdict="reject", reason_code="other", comment="short")])
    expert.write(tmp_path / "w1", items[:2], date="2026-11-15", reviews_dir=tmp_path / "reviews",
                 sheet=tmp_path / "e.csv")
    first = tmp_path / "reviews" / "expert_review.jsonl"
    # a first-round record as it was written, before records carried "replacement"
    first.write_text("".join(json.dumps({k: v for k, v in json.loads(l).items() if k != "replacement"},
                                        ensure_ascii=False, sort_keys=True) + "\n"
                             for l in first.read_text(encoding="utf-8").splitlines()), encoding="utf-8")
    before = first.read_text(encoding="utf-8").splitlines()
    # the team's action on a first-round item, entered on the filled sheet, is kept by the merge
    reader = csv.DictReader(io.StringIO((tmp_path / "e.csv").read_text(encoding="utf-8")))
    sheet_rows = list(reader)
    sheet_rows[0].update(team_action="keep", team_action_by="JR")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=reader.fieldnames, lineterminator="\n")
    w.writeheader()
    w.writerows(sheet_rows)
    (tmp_path / "e.csv").write_text(buf.getvalue(), encoding="utf-8")
    _jsonl(tmp_path / "w2" / "out" / "a37-EX-H1b.jsonl", [row(31)])
    with pytest.raises(ValueError, match="no longer in scope"):       # an earlier record is never dropped silently
        expert.write(tmp_path / "w2", items[1:], date="2026-11-18", reviews_dir=tmp_path / "reviews",
                     sheet=tmp_path / "e.csv", merge=True, replacements={"gba-test-31"})
    h = expert.write(tmp_path / "w2", items, date="2026-11-18", reviews_dir=tmp_path / "reviews",
                     sheet=tmp_path / "e.csv", merge=True, replacements={"gba-test-31"})["A"]
    assert (h["items"], h["accepted"], h["share"]) == (2, 1, 0.5)          # the original pool only
    assert (h["replacements"], h["replacements_accepted"]) == (1, 1)
    lines = first.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == before                                             # the first round's records as written
    recs = [json.loads(l) for l in lines]
    assert [r["item_id"] for r in recs] == ["gba-test-1", "gba-test-2", "gba-test-31"]
    assert [r.get("replacement", False) for r in recs] == [False, False, True]
    filled = list(csv.DictReader(io.StringIO((tmp_path / "e.csv").read_text(encoding="utf-8"))))
    assert (filled[0]["team_action"], filled[0]["team_action_by"]) == ("keep", "JR")
    with pytest.raises(ValueError, match="already reviewed"):
        _jsonl(tmp_path / "w3" / "out" / "a38-EX-H1b.jsonl", [row(1)])
        expert.write(tmp_path / "w3", items, date="2026-11-18", reviews_dir=tmp_path / "reviews",
                     sheet=tmp_path / "e.csv", merge=True, replacements={"gba-test-31"})


def test_the_first_accepted_replacements_in_id_order_fill_the_slots():
    from grcbench.review import replacements
    finals = {"r1": "reject", "r2": "approve", "r3": "approve with edits", "r4": "approve"}
    assert replacements.pick([(["r1", "r2", "r3", "r4"], 2)], finals) == {"r4"}
    assert replacements.pick([(["r1", "r2", "r3", "r4"], 3)], finals) == set()
    assert replacements.pick([(["r1", "r2"], 1)], {"r2": "approve"}) == set()     # r1 undecided: nothing yet
    from grcbench.tier_a import replacements as A
    from grcbench.tier_f import generate as G
    assert replacements.groups("tier_f") == [(list(G.replacement_plan(s)), G.REPLACEMENTS[s]["slots"])
                                             for s in ("test", "dev")]
    assert sorted(replacements.groups("tier_a")) == sorted((list(ids), slots) for splits in A.SLOTS.values()
                                                          for ids, slots in splits.values())
    assert replacements.all_ids() == {i for s in ("test", "dev") for i in G.replacement_plan(s)} | A.REPLACEMENT_IDS


def test_a_reserve_task_is_neither_run_nor_counted_as_removed(tmp_path):
    from grcbench.review import items
    for name, status in (("t1", "approved"), ("t2", "reserve"), ("t3", "rejected"),
                         ("gba-test-01", "pending-team-review"), ("gba-test-31", "pending-team-review")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "task.json").write_text(json.dumps({"id": name, "review": {"status": status}}))
    # an undecided replacement is not part of the benchmark yet; an original pending task is
    assert [d.name for d in items.active_dirs(sorted(tmp_path.iterdir()))] == ["gba-test-01", "t1"]
    from grcbench.runlog.log import RunRefused
    with pytest.raises(RunRefused):
        items.refuse_rejected([{"id": "t2", "review": {"status": "reserve"}}])


def test_a_reserve_accepted_with_edits_carries_its_edits_version(tmp_path):
    """gba-test-36 is the knowledge group's fourth replacement (3 slots): accepted with edits after the first three,
    it is a reserve, and its record still says that its decided edits are applied (version 2)."""
    from grcbench.review import items
    out = lambda v: {"value": v, "edits": [], "notes": ""}
    keys = [(f"gba-test-{i}",) for i in (31, 32, 33, 36)]
    recs = records.build(sheets.SHEETS["tier_a"], {k: out("approve with edits") for k in keys},
                         {k: out("approve with edits") for k in keys}, {k: out("approve with edits") for k in keys})
    records.write_records(tmp_path / "tier_a.jsonl", recs)
    rec, _ = items.apply("tier_a", "gba-test-36", {"id": "gba-test-36", "version": 1,
                                                   "review": {"status": "pending-team-review"}}, {}, reviews_dir=tmp_path)
    assert rec["review"]["status"] == "reserve" and rec["version"] == 2
    kept, _ = items.apply("tier_a", "gba-test-33", {"id": "gba-test-33", "version": 1,
                                                    "review": {"status": "pending-team-review"}}, {}, reviews_dir=tmp_path)
    assert kept["review"]["status"] == "approved-with-edits" and kept["version"] == 2
    # a run uses neither the reserve nor a replacement still undecided (gba-test-34); an original task it does
    assert items.active("tier_a", ["gba-test-01", "gba-test-31", "gba-test-34", "gba-test-36"],
                        reviews_dir=tmp_path) == ["gba-test-01", "gba-test-31"]
    from grcbench.runlog.log import RunRefused
    with pytest.raises(RunRefused, match="gba-test-34"):
        items.refuse_rejected([{"id": "gba-test-34", "review": {"status": "pending-team-review"}}])
    items.refuse_rejected([{"id": "gba-test-01", "review": {"status": "pending-team-review"}}])
    # while a group is not fully decided, none of its replacements is used: its slots are not settled yet
    part = tmp_path / "part"
    records.write_records(part / "tier_a.jsonl", [r for r in recs if r["key"] in keys[:2]])
    assert items.active("tier_a", ["gba-test-01", "gba-test-31", "gba-test-32"], reviews_dir=part) == ["gba-test-01"]
    from grcbench.review import replacements
    assert replacements.unsettled("tier_a", reviews_dir=part) >= {"gba-test-31", "gba-test-32", "gba-test-33"}


def test_a_regenerated_sheet_appends_new_rows_and_keeps_the_reviewed_rows_as_they_were(tmp_path):
    from grcbench.review import fill, records
    sheet = tmp_path / "tier_f_first_set.csv"
    sheet.write_text("task_id,instruction,final_decision\ngbf-test-21,the text as reviewed,approve\n", encoding="utf-8")
    generated = ("task_id,instruction,final_decision\r\ngbf-test-21,the text after the edits,\r\n"
                 "gbf-test-151,a replacement,\r\n")
    assert fill.with_new_rows("tier_f", generated, reviews_dir=tmp_path / "none", path=sheet) == generated
    records.write_records(tmp_path / "reviews" / "tier_f.jsonl",
                          [_rec("tier_f", ("gbf-test-21",), {"value": "approve"}, {"value": "approve"}, "approve",
                                "R1;R2")])
    out = fill.with_new_rows("tier_f", generated, reviews_dir=tmp_path / "reviews", path=sheet)
    assert out == ("task_id,instruction,final_decision\ngbf-test-21,the text as reviewed,approve\n"
                   "gbf-test-151,a replacement,\n")


def test_the_tier_a_sheet_generator_keeps_the_reviewed_rows(tmp_path, monkeypatch):
    """`./gb tier-a sheet` on a decided sheet: the reviewed rows stay byte for byte, new tasks are appended."""
    import shutil

    from grcbench.tier_a import spec_drafts
    sheet = tmp_path / "tier_a_first_set.csv"
    shutil.copy(SHEET_DIR / "tier_a_first_set.csv", sheet)
    before = sheet.read_text(encoding="utf-8")
    monkeypatch.setattr(spec_drafts, "SHEET", sheet)
    monkeypatch.setattr(spec_drafts, "REPORT", tmp_path / "report.md")
    monkeypatch.setattr(spec_drafts, "DRAFTS", tmp_path / "drafts.jsonl")
    spec_drafts.main([])
    after = sheet.read_text(encoding="utf-8")
    assert after.startswith(before) and len(after) >= len(before)


def test_build_alls_tier_f_mirror_carries_the_decided_records_and_patches():
    """Every Tier-F build.py applies reviews/tier_f.jsonl and reviews/tier_f_patches.json (review.items.apply), so the
    mirror that `./gb build-all --tier-f` rebuilds in holds reviews/; without it every reviewed task differs."""
    import inspect
    from grcbench.common import build_all
    src = inspect.getsource(build_all._tier_f)
    assert '"reviews"' in src and '"src"' in src


def test_the_summary_names_the_reserves_and_gives_each_rounds_audit():
    from grcbench.review import report
    out = lambda v: {"value": v, "edits": [], "notes": ""}
    keys = [(f"gba-test-{i}",) for i in (31, 32, 33, 36)] + [("gba-test-01",), ("gba-test-02",)]
    recs = records.build(sheets.SHEETS["tier_a"], {k: out("approve") for k in keys}, {k: out("approve") for k in keys},
                         {}, audit={("gba-test-01",): {**out("reject"), "pass": "audit"},
                                    ("gba-test-31",): {**out("approve"), "pass": "audit"}})
    s = report.sheet_summary("tier_a", recs)
    assert s["reserves"] == ["gba-test-36"]
    assert s["audit_rounds"]["first"]["disagreements"] == 1 and s["audit_rounds"]["first"]["n"] == 1
    assert s["audit_rounds"]["second"]["disagreements"] == 0 and s["audit_rounds"]["second"]["n"] == 1
    text = report.render([s], [])
    assert "gba-test-36" in text and "## The second round" in text

"""Tests for `./gb calc`: script answers to the Tier-K calculation test questions (eval plan §5.2, §5.1 Table 5;
proposal §5.1). Nothing here sends a question to a model."""

from __future__ import annotations

import ast
import inspect
import math
import re
import textwrap
from collections import Counter
from pathlib import Path

import pytest

from grcbench.common.canary import CANARY
from grcbench.common.io import read_jsonl, sha256_text
from grcbench.common.paths import FROZEN, REPO, SPLITS
from grcbench.tier_k import calc

OUTPUTS = (calc.OUT_ANSWERS, calc.OUT_BLIND, calc.OUT_KEY, calc.OUT_INSTRUCTIONS)
# The specification classes, pinned so that a change is deliberate (module docstring, "Specification").
# methodology-F1/F6: a number whose answer changes, beyond its tolerance, under a convention the question
# leaves open is underspecified (eval plan §5.2); the five after the first four were "assumption-dependent"
# or "determined" in tier-k-calc 2.
UNDERSPECIFIED = {"q_d00f22f579b94d32", "q_ff18653bc3179694", "q_94a0c101dbf3ee99", "q_7acc67b2a3b45407",
                  "q_41dd72957bec640b", "q_bc3fb92ae337a426", "q_63f558e028ba13d5", "q_c955829cc73f83d0",
                  "q_eabcdb1b438a5afb"}
TEAM_REFERENCE = {"q_1fe7817ee6a8873f", "q_66b264c8786f2018", "q_f7bced7a5d660e9d", "q_a598b884fbd1d1f6",
                  "q_75980861b0e8d95d"}
WITHHELD_FOR_SCORER = {"q_487f23bc5c3635f0"}
ASSUMED = {"q_5ed3c28aa3901b62": {"exponent_bits": 8}, "q_976f44ddd3de3998": {"temperature_k": 290.0},
           "q_25f9f7406f3df97e": {"bytes_per_item": 8}}
REQUIRED = ("question_id", "value", "unit", "tolerance", "formula", "assumptions", "source", "status",
            "computed_by", "underspecified", "canary")
DB_FACTOR_2 = 10 * math.log10(2)


@pytest.fixture(scope="module")
def built() -> calc.Build:
    return calc.build()


@pytest.fixture(scope="module")
def stored() -> dict[str, dict]:
    """The calc_answers.jsonl in the repository (written by ./gb calc)."""
    return {r["question_id"]: r for r in read_jsonl(REPO / calc.OUT_ANSWERS)}


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> tuple[Path, Path]:
    """Two independent `./gb calc --out` runs."""
    roots = tmp_path_factory.mktemp("a"), tmp_path_factory.mktemp("b")
    for root in roots:
        assert calc.main(["--out", str(root)]) == 0
    return roots


def _rec(built, qid):
    return next(r for r in built.records if r["question_id"] == qid)


# ---------------------------------------------------------------- which questions ----------------

def test_the_25_calculation_questions_agree_across_items_labels_and_raw_data(built):
    items = read_jsonl(calc.ITEMS_TEST)
    labels = [r for r in read_jsonl(SPLITS / "task_type_labels.jsonl") if r["split"] == "eval"]
    raw = read_jsonl(FROZEN / "eval_questions.jsonl")
    ids_items = [r["id"] for r in items if r["task_type"] == "calculation"]
    ids_labels = {r["id"] for r in labels if r["clean"] == "calculation"}
    ids_raw = {r["id"] for r in raw if r["task_type"] in ("calculation", "calculations")}
    assert len(ids_items) == 25 and set(ids_items) == ids_labels == ids_raw == set(calc.REGISTRY)
    assert Counter(r["task_type"] for r in raw if r["id"] in ids_raw) == {"calculation": 23, "calculations": 2}
    assert [q["id"] for q in built.questions] == ids_items                  # frozen-split order
    assert [r["question_id"] for r in built.records] == ids_items


def test_functions_are_named_after_their_question_and_use_its_stated_numbers(built):
    for q in built.questions:
        c = calc.REGISTRY[q["id"]]
        assert c.fn.__name__ == q["id"] and c.stated
        assert all(s in q["question"] for s in c.stated), (q["id"], c.stated)


def test_load_refuses_a_question_without_a_function_or_a_missing_stated_number(monkeypatch):
    qid = next(iter(calc.REGISTRY))
    monkeypatch.delitem(calc.REGISTRY, qid)
    with pytest.raises(SystemExit, match="no function for"):
        calc.load_questions()
    monkeypatch.undo()
    good = calc.REGISTRY[qid]
    monkeypatch.setitem(calc.REGISTRY, qid, calc.Calc(qid, good.fn, good.stated + ("42 parsecs",)))
    with pytest.raises(SystemExit, match="lacks"):
        calc.load_questions()


# ---------------------------------------------------------------- the stored answers -------------

@pytest.mark.parametrize("qid", sorted(calc.REGISTRY))
def test_each_function_on_its_own_question_returns_the_stored_value(qid, built, stored):
    ans = calc.REGISTRY[qid].fn()                                           # the question's own numbers
    assert calc.answer_value(ans) == stored[qid]["value"]
    q = next(q for q in built.questions if q["id"] == qid)
    assert calc.record(q, calc.REGISTRY[qid]) == stored[qid]
    assert stored[qid]["question_sha256"] == sha256_text(q["question"])


def test_records_carry_the_required_fields_and_status(built):
    """methodology-F7: one status for every record (WS spec); the reference's route is its own field."""
    for r in built.records:
        assert all(k in r for k in REQUIRED), r["question_id"]
        assert r["status"] == "pending blind team recomputation"
        assert r["reference_route"] == calc.ROUTES[r["specification"]]
        assert r["computed_by"] == "script drafted by Claude (Anthropic)"
        assert r["canary"] == CANARY and r["kind"] in calc.KINDS and r["specification"] in calc.SPECIFICATIONS
        assert r["formula"] and r["source"] and r["assumptions"] and r["tolerance_note"]
        assert r["function"] == f"grcbench.tier_k.calc.{r['question_id']}"
        assert "inputs" not in r
    assert "Table-5 drafting model" in calc.ROUTES["team reference"]


def test_record_text_fields_come_from_the_function_docstring(built):
    for r in built.records:
        doc = calc.doc_sections(calc.REGISTRY[r["question_id"]].fn)
        assert (r["formula"], r["assumptions"], r["source"], r["tolerance_note"]) == \
            (doc["Formula"], doc["Assumptions"], doc["Source"], doc["Tolerance"])
        assert (r["missing_assumptions"], r["not_computed"]) == (doc["Missing"], doc["Not computed"])
        assert set(doc) == {"title", *calc.SECTIONS}                       # nothing the record leaves out


def test_inputs_are_split_into_stated_numbers_and_assumed_constants(built):
    """methodology-F7: room temperature (290 K), the gr_complex size (8 bytes) and the binary32 exponent
    width (8 bits) are not numbers the question states."""
    got = {r["question_id"]: r["assumed_constants"] for r in built.records if r["assumed_constants"]}
    assert got == ASSUMED
    for r in built.records:
        c = calc.REGISTRY[r["question_id"]]
        params = set(inspect.signature(c.fn).parameters)
        assert set(r["stated_inputs"]) | set(r["assumed_constants"]) == params
        assert not set(r["stated_inputs"]) & set(r["assumed_constants"])
        if r["assumed_constants"]:                                          # each is named under Assumptions
            assert any("assumed constant" in a for a in r["assumptions"]), r["question_id"]
    def q_x(rate: float = 1.0) -> calc.Answer:
        return calc.Answer("symbolic")
    with pytest.raises(ValueError, match="not parameters"):
        calc.calc("q_x", stated=(), assumed=("nope",))(q_x)
    assert "q_x" not in calc.REGISTRY


# ---------------------------------------------------------------- specification ------------------

def test_specification_classes(built):
    assert calc.SPECIFICATIONS == ("determined", "underspecified", "team reference")
    by = {s: {r["question_id"] for r in built.records if r["specification"] == s} for s in calc.SPECIFICATIONS}
    assert by["underspecified"] == UNDERSPECIFIED
    assert by["team reference"] == TEAM_REFERENCE
    assert len(by["determined"]) == 25 - len(UNDERSPECIFIED | TEAM_REFERENCE) == 11
    for r in built.records:
        assert r["underspecified"] == (r["specification"] == "underspecified")


def test_one_rule_decides_the_specification(built):
    """underspecified exactly when a part has no value; a part with a value lists only readings within its
    tolerance; a reading's flag agrees with the part's tolerance (eval plan §5.2, methodology-F1)."""
    for r in built.records:
        values = r["value"] if isinstance(r["value"], list) else [r["value"]]
        if r["specification"] == "underspecified":
            assert r["missing_assumptions"] and r["undetermined_parts"] and None in values
            assert {c["part"] for c in r["readings"] if c["within_tolerance"] is None} == set(r["undetermined_parts"])
            tols = r["tolerance"] if isinstance(r["tolerance"], list) else [r["tolerance"]]
            assert all(t is None for v, t in zip(values, tols) if v is None)
        else:
            assert not r["missing_assumptions"] and not r["undetermined_parts"]
        if r["specification"] == "determined":
            assert None not in values and all(c["within_tolerance"] is True for c in r["readings"])
            assert any("listed under readings" in a for a in r["assumptions"]) == bool(r["readings"]), r["question_id"]
        assert all(c["within_tolerance"] is not False for c in r["readings"])
    # the flags are recomputed from the parts, not copied
    for qid in sorted(calc.REGISTRY):
        ans = calc.evaluate(qid)
        for p in ans.parts:
            for _, v, _ in p.readings:
                if p.value is not None:
                    rel = abs(v - p.value) / abs(p.value)
                    inside = abs(v - p.value) <= p.abs_db if p.abs_db is not None else rel <= p.tolerance
                    assert p.within(v) == inside


def test_a_value_with_a_reading_outside_its_tolerance_is_refused(monkeypatch):
    """No middle class: a value is never given next to a rival reading outside its tolerance."""
    def q_fake() -> calc.Answer:
        return calc.Answer("numeric", (calc.Part("x", 1.0, "", 0.01, readings=(("other convention", 2.0, ""),)),))
    monkeypatch.setitem(calc.REGISTRY, "q_fake", calc.Calc("q_fake", q_fake, ()))
    with pytest.raises(ValueError, match="outside the tolerance leaves the part undetermined"):
        calc.evaluate("q_fake")


def test_unstated_conventions_make_the_number_underspecified(built):
    """methodology-F1/F6: each formerly assumption-dependent record, and the ICI record, has no value,
    names the unstated convention, and lists the value under each option, the options lying outside the
    tolerance the value would have."""
    cases = {  # qid: (words Missing must name, the readings' values, the would-be relative tolerance)
        "q_41dd72957bec640b": (("E_s/N_0", "E_b/N_0"), None, 0.05),
        "q_bc3fb92ae337a426": (("data rate", "read out"), [6.144, 12.288], 0.05),
        "q_63f558e028ba13d5": (("real or complex",), [20.0, 10.0], 0.01),
        "q_c955829cc73f83d0": (("normalisation",), [1.0, 0.5, 0.25] * 4, 0.01),
        "q_eabcdb1b438a5afb": (("signal power", "edge", "Moose"), None, None),
    }
    for qid, (words, values, tol) in cases.items():
        r = _rec(built, qid)
        assert r["specification"] == "underspecified" and r["numeric_proposal"] is None, qid
        missing = " ".join(r["missing_assumptions"])
        assert all(w in missing for w in words), (qid, missing)
        got = [c["value"] for c in r["readings"]]
        if values is not None:
            assert got == values, qid
        if tol is not None:                         # the options differ by more than the tolerance
            per_part = {}
            for c in r["readings"]:
                per_part.setdefault(c["part"], []).append(c["value"])
            for vals in per_part.values():
                assert max(vals) > (1 + tol) * min(vals) * 1.01, (qid, vals)
    ser = [c["value"] for c in _rec(built, "q_41dd72957bec640b")["readings"]]
    assert math.isclose(ser[0], 1.161629091e-05) and ser[1] < 1e-18
    ici = [c["value"] for c in _rec(built, "q_eabcdb1b438a5afb")["readings"]]
    assert max(ici) - min(ici) > 5.0                                        # dB: far beyond 0.1 dB
    assert all("assumption-dependent" not in str(v) for r in built.records for v in r.values())


def test_team_reference_items_carry_no_script_answer(built):
    """Eval plan §5.2 / Table 5: only values computed from the question's numbers come from the script."""
    for r in built.records:
        if r["specification"] != "team reference":
            assert r["kind"] in calc.VALUE_KINDS and r["value"] is not None or r["underspecified"]
            continue
        assert r["kind"] in calc.TEAM_KINDS and r["value"] is None and r["labels"] == []
        assert r["tolerance"] is None and r["tolerance_db"] is None and r["unit"] == ""
        assert r["numeric_proposal"] is None and not r["equivalents"] and not r["readings"]
        assert r["formula"].startswith("none: ") and "reference is written by the team" in r["formula"]
        assert r["source"].startswith("none: ")
        assert r["assumptions"] == ["none (no value is computed)"]
        assert r["not_computed"] == ["the whole question (team reference)"]
        assert r["stated_inputs"] == {} and r["assumed_constants"] == {}


def test_team_reference_functions_hold_no_answer_content():
    """methodology-F3: a team-reference function is its docstring and one `return Answer(kind)`: no
    formula, source, self-check prose or code that checks a candidate answer, in the file the team reads
    after the blind pass."""
    for qid in sorted(TEAM_REFERENCE):
        fn = calc.REGISTRY[qid].fn
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        (node,) = tree.body
        assert not node.args.args and not node.args.kwonlyargs, qid
        doc, ret = node.body                                               # exactly a docstring and a return
        assert isinstance(doc, ast.Expr) and isinstance(ret, ast.Return), qid
        assert ast.unparse(ret.value) in ('Answer("symbolic")', "Answer('symbolic')",
                                          'Answer("example")', "Answer('example')"), qid
        text = inspect.getdoc(fn)
        assert re.findall(r"^(\w[\w -]*):", text, re.M) == ["Formula", "Assumptions", "Source", "Tolerance",
                                                            "Not computed"], qid
    with pytest.raises(ValueError, match="unknown docstring section 'Self-check'"):
        def f():
            """Title.

            Formula: none: x
            Assumptions:
              - none (no value is computed)
            Source: none: y
            Tolerance: none: z
            Self-check: an answer
            """
        calc.doc_sections(f)


# ---------------------------------------------------------------- tolerances and proposals -------

def test_numeric_shapes_and_tolerances(built):
    for r in built.records:
        if r["kind"] not in calc.VALUE_KINDS:
            continue
        n = len(r["labels"])
        if n == 1:
            assert not isinstance(r["value"], list) and isinstance(r["unit"], str)
        else:
            assert len(r["value"]) == len(r["unit"]) == len(r["tolerance"]) == n
        tols = r["tolerance"] if n > 1 else [r["tolerance"]]
        dbs = r["tolerance_db"] if n > 1 else [r["tolerance_db"]]
        for t, d in zip(tols, dbs):
            assert t is None or 0 < t <= calc.MAX_WIDENED                  # widened at most to 10 %
            if d is not None:
                assert d <= calc.MAX_WIDENED_DB and math.isclose(t, 10 ** (d / 10) - 1, rel_tol=1e-9)
        if r["numeric_proposal"] is not None:
            refs = r["numeric_proposal"] if n > 1 else [r["numeric_proposal"]]
            assert [x["value"] for x in refs] == (r["value"] if n > 1 else [r["value"]])
            assert all(("label" in x) == (n > 1) for x in refs)


def test_tolerance_notes_say_what_they_admit_not_what_is_wrong(built):
    """Rubric content (errors, partial credit) is the team's: eval plan glossary and Table 5."""
    verdict = re.compile(r"\bscores?\b|\bscored\b|credit|\bzero\b|\bwrong\b|\bslip|mistake|penal|\boff\b|"
                         r"not the answer", re.I)
    for r in built.records:
        assert not verdict.search(r["tolerance_note"]), (r["question_id"], r["tolerance_note"])
        assert r["tolerance_note"].startswith(("none: ", "1 %", "5 %", "8 %", "0.1 dB", "0.2 dB", "output rate",
                                               "Carson")), r["question_id"]


def test_proposals_only_for_determined_numeric_records_and_withheld_ones_say_why(built):
    have = {r["question_id"] for r in built.records if r["numeric_proposal"] is not None}
    determined = {r["question_id"] for r in built.records
                  if r["specification"] == "determined" and r["kind"] == "numeric"}
    assert have == determined - WITHHELD_FOR_SCORER and len(have) == 10
    for r in built.records:
        assert (r["numeric_proposal"] is None) == bool(r["numeric_withheld"]), r["question_id"]
    for qid in WITHHELD_FOR_SCORER:
        assert "rules.py" in _rec(built, qid)["numeric_withheld"]


def _parts(r):
    refs = r["numeric_proposal"]
    return refs if isinstance(refs, list) else [refs]


def _text(ref, value, unit):
    lab = ref.get("label")
    head = f"{lab[0]}: " if lab else "Final answer: "
    return f"{head}{value:.6g} {unit}".rstrip()


def test_numeric_proposals_are_scored_soundly_by_the_rule_scorer(built):
    """Every proposal: the value itself gets full credit; every listed equivalent and every reading within
    tolerance gets full credit or goes to the judge (never a rule-decided loss); a value a factor 2 off
    either way gets 0 from the rules."""
    from grcbench.tier_k import rules
    for r in built.records:
        if r["numeric_proposal"] is None:
            continue
        rules.parse_reference(r["numeric_proposal"])
        for part, ref in zip(r["labels"], _parts(r), strict=True):
            got =rules.score(_text(ref, ref["value"], ref["unit"]), ref)
            assert (got["method"], got["credit"]) == ("numeric", 1.0), (r["question_id"], got)
            same = [(e["value"], e["unit"]) for e in r["equivalents"] if e["part"] == part]
            same += [(c["value"], c["unit"]) for c in r["readings"] if c["part"] == part and c["within_tolerance"]]
            for v, u in same:
                got = rules.score(_text(ref, v, u), ref)
                assert got["method"] == "judge" or got["credit"] == 1.0, (r["question_id"], v, u, got)
            is_db = rules.parse_unit(ref["unit"]).db
            wrong = ((ref["value"] + DB_FACTOR_2, ref["value"] - DB_FACTOR_2) if is_db
                     else (2 * ref["value"], ref["value"] / 2))
            for v in wrong:
                got = rules.score(_text(ref, v, ref["unit"]), ref)
                assert (got["method"], got["credit"]) == ("numeric", 0.0), (r["question_id"], v, got)
        if isinstance(r["numeric_proposal"], list):               # a typical one-line answer with every part
            line = "Final answer: " + ", ".join(f"{ref['label'][0]} {ref['value']:.4g} {ref['unit']}".rstrip()
                                                for ref in r["numeric_proposal"])
            got = rules.score(line, r["numeric_proposal"])
            assert got["method"] == "judge" or got["credit"] == 1.0, (r["question_id"], line, got)
        else:                                                     # a LaTeX boxed answer
            ref = r["numeric_proposal"]
            unit = f"\\text{{ {ref['unit']}}}" if ref["unit"] else ""
            got = rules.score(f"\\boxed{{{ref['value']:.4g}{unit}}}", ref)
            assert got["method"] == "judge" or got["credit"] == 1.0, (r["question_id"], got)


def _layouts(pairs):
    """A correct multi-part answer written several ways: (label, value text) pairs in the question's order."""
    return [
        "Final answer: " + ", ".join(f"{a} = {v}" for a, v in pairs),
        "\n".join(f"{a} = {v}" for a, v in pairs),
        "; ".join(f"{a}: {v}" for a, v in pairs),
        "\n".join(f"- {a}: {v}" for a, v in pairs),
        "\n".join(f"{i}. {a}: {v}" for i, (a, v) in enumerate(pairs, 1)),
        "\n".join(f"**{a}:** {v}" for a, v in pairs),
    ]


def test_every_multi_part_proposal_is_sound_for_symbol_and_list_labels(built):
    """methodology-F2: each anchor of each part (symbols such as "I/Q" included), in one-line, line,
    list and Markdown layouts, gets full credit or goes to the judge; a factor-2 error in one part never
    gets full credit."""
    from grcbench.tier_k import rules
    multi = [r for r in built.records if isinstance(r["numeric_proposal"], list)]
    assert multi
    for r in multi:
        refs = r["numeric_proposal"]
        for k in range(max(len(ref["label"]) for ref in refs)):
            names = [ref["label"][k % len(ref["label"])] for ref in refs]
            right = [(a, f"{ref['value']:.4g} {ref['unit']}".rstrip()) for a, ref in zip(names, refs)]
            for text in _layouts(right):
                got = rules.score(text, refs)
                assert got["method"] == "judge" or got["credit"] == 1.0, (r["question_id"], text, got)
            for i, ref in enumerate(refs):
                wrong = list(right)
                wrong[i] = (names[i], f"{2 * ref['value']:.4g} {ref['unit']}".rstrip())
                for text in _layouts(wrong):
                    got = rules.score(text, refs)
                    assert got["method"] == "judge" or got["credit"] < 1.0, (r["question_id"], text, got)


RADAR_PLAIN = ("Final answer: ΔR = 1.5 m, TB = 1000, R_ua = 150 km", "ΔR = 1.5 m\nTB = 1000\nR_max = 150 km",
               "ΔR = 1.5 m; TBP = 1000; max range = 150 km", "Final answer: δR = 1.5 m, BT = 1000, R_unamb = 150 km",
               "- ΔR = 1.5 m\n- TB = 1000\n- R_ua = 150 km", "Final answer: ΔR ≈ 1.5 m; B·τ = 1000; R_max ≈ 150 km",
               "Range resolution: 1.5 m. Time-bandwidth product: 1000. Maximum unambiguous range: 150 km.")
RADAR_LATEX = "Final answer: $\\Delta R = 1.5$ m, $TB = 1000$, $R_{max} = 150$ km"


def _radar_refs():
    parts = calc.evaluate("q_487f23bc5c3635f0").parts
    return [calc._numeric_ref(p, with_label=True) for p in parts]


def test_radar_anchors_decide_symbol_labels():
    """methodology-F2: with the symbol anchors, plain-text ΔR / TB / R_max answers are decided right, and a
    wrong range resolution is no longer read from the unambiguous range."""
    from grcbench.tier_k import rules
    refs = _radar_refs()
    for text in RADAR_PLAIN:
        got = rules.score(text, refs)
        assert (got["method"], got["credit"]) == ("numeric", 1.0), (text, got)
    for text, part in (("Final answer: ΔR = 3 m, TB = 1000, R_ua = 150 km", 0),
                       ("ΔR = 1.5 m; TBP = 2000; max range = 150 km", 1),
                       ("- ΔR = 1.5 m\n- TB = 1000\n- R_ua = 75 km", 2)):
        got = rules.score(text, refs)
        credits = [p["credit"] for p in got["details"]["parts"]]
        assert credits == [0.0 if i == part else 1.0 for i in range(3)], (text, got)


def test_withheld_for_the_scorer_still_needed(built):
    """The radar proposal is withheld because rules.py loses a LaTeX label ("$\\Delta R$"): skipped (not
    failed) once rules.py scores it soundly, so the withhold can then be removed."""
    from grcbench.tier_k import rules
    assert "LaTeX" in _rec(built, "q_487f23bc5c3635f0")["numeric_withheld"]
    got = rules.score(RADAR_LATEX, _radar_refs())
    if got["method"] == "judge" or got["credit"] == 1.0:
        pytest.skip("rules.py now scores the LaTeX-labelled radar answer soundly: remove its `withhold`")
    assert got["credit"] < 1.0


def test_smallest_binary32_proposal_separates_the_classic_neighbours(built):
    """correctness-F1: under tier-k-rules 0.2 the absolute slack let 2^-125, 2^-127 and 1e-37 pass."""
    from grcbench.tier_k import rules
    ref = _rec(built, "q_5ed3c28aa3901b62")["numeric_proposal"]
    for text in ("2^-126", "1.18e-38", "2^{-126} ≈ 1.18 × 10^{-38}", "1.17549435 × 10^-38"):
        assert rules.score(f"Final answer: {text}", ref)["credit"] == 1.0, text
    for text in ("2^-125", "2^-127", "1.0e-37", "1.4e-45", "3.4e38"):
        assert rules.score(f"Final answer: {text}", ref)["credit"] == 0.0, text


# ---------------------------------------------------------------- independent spot checks --------

def _readings(built, qid):
    return [c["value"] for c in _rec(built, qid)["readings"]]


def test_spot_values_recomputed_independently(built):
    from scipy.special import erfc
    c = 299_792_458.0
    assert _readings(built, "q_63f558e028ba13d5") == [20.0, 10.0]            # 2 x 10 MHz real, 10 MHz I/Q
    assert _rec(built, "q_673b0e732a9bccad")["value"] == [4.0, 2.0]
    assert math.isclose(_rec(built, "q_ccf5d9913ed192e4")["value"], c / 0.2 / 1e9, rel_tol=1e-9)
    assert math.isclose(_rec(built, "q_cf42443d85802560")["value"], c / 8e9, rel_tol=1e-9)
    res, tbp, rua = _rec(built, "q_487f23bc5c3635f0")["value"]
    assert math.isclose(res, 1.4989623, rel_tol=1e-7) and tbp == 1000 and math.isclose(rua, 149.896229, rel_tol=1e-8)
    assert abs(_rec(built, "q_976f44ddd3de3998")["value"] - (-174 + 60)) < 0.05      # -174 dBm/Hz + 60 dB
    assert _rec(built, "q_072c5b5c4eb8bcc9")["value"] == 10.0
    assert _rec(built, "q_a12e73b62696c2fe")["value"] == 30.0
    assert _rec(built, "q_9849c620793a77be")["value"] == 3.0
    assert _rec(built, "q_25f9f7406f3df97e")["value"] == 1.25
    assert _readings(built, "q_bc3fb92ae337a426") == [6.144, 12.288]
    assert _rec(built, "q_693ffc952e4828ed")["value"] == 5.0
    assert _rec(built, "q_5ed3c28aa3901b62")["value"] == float("1.175494351e-38")
    assert _rec(built, "q_c955829cc73f83d0")["value"] == [None] * 4
    assert _readings(built, "q_c955829cc73f83d0")[:3] == [1.0, 0.5, 0.25]
    eps = 100 / 15e3
    full, edge, moose = _readings(built, "q_eabcdb1b438a5afb")
    assert abs(full - 10 * math.log10((math.pi * eps) ** 2 / 3)) < 0.01
    assert abs(edge - 10 * math.log10((math.pi * eps) ** 2 / 6)) < 0.01
    assert abs(moose - 10 * math.log10(0.5947 * math.sin(math.pi * eps) ** 2)) < 1e-6
    assert _rec(built, "q_94a0c101dbf3ee99")["value"] == [1.4, None]
    assert _rec(built, "q_7acc67b2a3b45407")["value"] == [180.0, None]
    # 16-QAM at Es/N0 = 20 dB, with scipy's erfc instead of math.erfc
    p = 1.5 * 0.5 * erfc(math.sqrt(20) / math.sqrt(2))                          # 2 (1 - 1/4) Q(sqrt(20))
    assert math.isclose(_readings(built, "q_41dd72957bec640b")[0], 1 - (1 - p) ** 2, rel_tol=1e-6)
    # link budget under its named ideal reading: 20 + (-104) + 122 - 60 = -22 dBm
    (lb,) = _rec(built, "q_d00f22f579b94d32")["readings"]
    assert abs(lb["value"] - (-22.0)) < 0.05 and lb["unit"] == "dBm"


def test_resampler_cutoff_readings_name_the_image_limit(built):
    """correctness-F1: for a 0-400 kHz input at 1 MS/s, ideal cutoffs from 400 kHz to just below the first
    image (600 kHz) stop images and aliases; 500 kHz is the general min(f_in, f_out)/2 rule, not the
    largest such cutoff."""
    r = _rec(built, "q_94a0c101dbf3ee99")
    fs_in, f_max, fs_out = 1e6, 400e3, 1.4e6
    first_image = fs_in - f_max                                    # images of 0-400 kHz start at 1 MHz - 400 kHz
    assert first_image < fs_out / 2                                # so a cutoff below it aliases nothing either
    cut = [c for c in r["readings"] if c["part"] == "prototype filter cutoff"]
    assert [c["value"] for c in cut] == [f_max / 1e3, min(fs_in, fs_out) / 2e3, first_image / 1e3]
    assert "general" in cut[1]["reading"] and "largest" not in cut[1]["reading"]
    assert "first image" in cut[2]["reading"]
    assert "600 kHz" in r["formula"] and "largest" not in r["formula"]


def test_bandpass_statement_of_q673b_is_correct(built):
    """A real sampler at 2B aliases a band at 1-3 MHz onto itself: 2B is the bandpass minimum only when
    f_H is a whole multiple of B, which the assumption now says."""
    a = " ".join(_rec(built, "q_673b0e732a9bccad")["assumptions"])
    assert "2 f_H / floor(f_H / B)" in a and "whole multiple of B" in a


# ---------------------------------------------------------------- review sheets -----------------

def test_blind_sheet_holds_only_ids_question_text_and_empty_team_columns(built):
    text = calc.blind_sheet(built.questions)
    rows = calc.read_sheet(text)
    assert tuple(rows[0]) == calc.BLIND_COLUMNS == ("question_id", "question", "annotator", "value", "unit", "notes")
    assert [r[:2] for r in rows[1:]] == [[q["id"], q["question"]] for q in built.questions]
    assert all(r[2:] == ["", "", "", ""] for r in rows[1:]) and not calc.annotated(text)


def test_review_sheets_are_utf8_with_a_byte_order_mark(runs):
    a, _ = runs
    for rel in (calc.OUT_BLIND, calc.OUT_KEY):
        raw = (a / rel).read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf") and b"\r\n" not in raw
        assert "σ²" in raw.decode("utf-8-sig") and "μs" in raw.decode("utf-8-sig")
    assert not (a / calc.OUT_ANSWERS).read_bytes().startswith(b"\xef\xbb\xbf")


def test_key_sheet_lists_every_answer(built):
    rows = calc.read_sheet(calc.key_sheet(built.questions, built.records))
    assert tuple(rows[0]) == calc.KEY_COLUMNS
    body = [dict(zip(rows[0], r)) for r in rows[1:]]
    assert [b["question_id"] for b in body] == [r["question_id"] for r in built.records]
    for b, r in zip(body, built.records):
        assert b["value"] == calc._fmt(r["value"]) and b["specification"] == r["specification"]
        assert b["status"] == r["status"] == calc.STATUS and b["computed_by"] == calc.COMPUTED_BY
        assert b["reference_route"] == r["reference_route"]
        if r["specification"] == "team reference":
            assert b["value"] == "" and b["formula"].startswith("none: ") and b["source"].startswith("none: ")


def test_a_filled_in_blind_sheet_is_kept_and_passes_check(tmp_path, capsys):
    assert calc.main(["--out", str(tmp_path)]) == 0
    sheet = tmp_path / calc.OUT_BLIND
    rows = calc.read_sheet(sheet.read_bytes().decode("utf-8"))
    rows[1][2:] = ["AB", "42", "dB", "underspecified: test"]
    import csv
    import io
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    filled = calc.BOM + buf.getvalue()
    sheet.write_bytes(filled.encode("utf-8"))
    assert calc.main(["--check", "--out", str(tmp_path)]) == 0              # filled in, questions current
    assert calc.main(["--out", str(tmp_path)]) == 0
    assert sheet.read_bytes().decode("utf-8") == filled                      # not overwritten
    assert "kept, not overwritten" in capsys.readouterr().out
    rows[1][1] = "an outdated question text"                                 # filled in AND out of date
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    sheet.write_bytes((calc.BOM + buf.getvalue()).encode("utf-8"))
    assert calc.main(["--check", "--out", str(tmp_path)]) == 1
    with pytest.raises(SystemExit, match="out of date"):
        calc.main(["--out", str(tmp_path)])


def test_default_output_names_no_question(tmp_path, capsys):
    assert calc.main(["--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "determined 11, underspecified 9, team reference 5" in out and "numeric proposals 10" in out
    assert "assumption-dependent" not in out
    assert not re.search(r"q_[0-9a-f]{16}", out)


def test_instructions_send_every_flag_and_disagreement_to_the_qoherent_expert(built):
    text = calc.instructions(len(built.questions))
    flat = " ".join(text.split())
    assert "The Qoherent expert decides every disagreement" in flat and "who decided" in flat
    assert "An underspecified question is rewritten, or removed and counted (eval plan §5.2, §5.3)" in flat
    assert "assumption-dependent" not in text and "kept" not in text            # methodology-F1: no third route
    assert "do not guess: write `underspecified:`" in flat
    assert "./gb calc" not in text and "_<your initials>.csv" in text
    assert not re.search(r"q_[0-9a-f]{16}", text)


def test_blind_pass_keeps_every_file_with_script_answers_closed(built):
    """methodology-F4: the files holding the script's answers, this test file included, are on the
    do-not-open list, in the instructions and in the rubric guide."""
    flat = " ".join(calc.instructions(len(built.questions)).split())
    closed = flat[flat.index("do **not** open"):flat.index("they hold the script's answers")]
    for path in ("tier_k_calc_key.csv", "items/tier_k/test/calc_answers.jsonl", "src/grcbench/tier_k/", "tests/"):
        assert path in closed, path
    assert Path(__file__).relative_to(REPO).as_posix().startswith("tests/")
    assert Path(calc.__file__).relative_to(REPO).as_posix().startswith("src/grcbench/tier_k/")
    guide = " ".join((REPO / "docs" / "guidelines" / "rubric_writing.md").read_text(encoding="utf-8").split())
    rule = guide[guide.index("Do not look at the key"):]
    rule = rule[:rule.index("before you have your own answer")]
    for path in ("items/tier_k/test/calc_answers.jsonl", "src/grcbench/tier_k/", "tests/"):
        assert path in rule, path
    assert "assumption-dependent" not in guide


# ---------------------------------------------------------------- determinism --------------------

def test_two_runs_are_byte_identical_and_match_the_repository(runs):
    a, b = runs
    for rel in OUTPUTS:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel
        repo = (REPO / rel).read_bytes()
        if rel == calc.OUT_BLIND and calc.annotated(repo.decode("utf-8")):   # filled in by a team member
            assert calc.same_questions(repo.decode("utf-8"), (a / rel).read_bytes().decode("utf-8"))
            continue
        assert (a / rel).read_bytes() == repo, f"{rel} is stale: run ./gb calc"
    assert calc.main(["--check", "--out", str(a)]) == 0
    (a / calc.OUT_KEY).write_text("stale\n", encoding="utf-8")
    assert calc.main(["--check", "--out", str(a)]) == 1


def test_doc_sections_parser():
    def f():
        """Title line.

        Formula: a = b
            + c
        Assumptions:
          - one
            continued
          - two
        Source: a book
            and a standard
        Tolerance: 1 %
        """
    d = calc.doc_sections(f)
    assert d == {"title": "Title line.", "Formula": "a = b + c", "Assumptions": ["one continued", "two"],
                 "Source": "a book and a standard", "Tolerance": "1 %", "Missing": [], "Not computed": []}

    def g():
        """No sections."""
    with pytest.raises(ValueError, match="no Formula"):
        calc.doc_sections(g)

"""Checks on the team-facing review checklists, annotation guidelines and label-sheet templates
(eval plan §5.2-§5.4 two-person and expert review, §8.1-§8.2 judge calibration, §9 error analysis).

The documents are prose, so these tests check what can go stale or leak:
  * every document exists, is LF-only Markdown with a title and cites the eval plan;
  * relative links resolve; templates are header-only CSVs with the documented columns;
  * the documented columns and codes agree with the code that writes or reads them (the Tier-F
    review sheet in tier_f/cli.py, the judge's verdict labels, the §8.1 pass thresholds, GRC-Agent's
    22 tools in the SFT release) and with the GNU Radio 3.10 catalog;
  * the rubric guide's worked examples use dev questions only and contain placeholders only (Claude
    never drafts rubric prose, eval plan Table 5);
  * no document or template quotes a Tier-K or Tier-F test item or carries the canary.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import pytest

from grcbench.common.canary import CANARY_GUID
from grcbench.common.io import read_jsonl
from grcbench.common.paths import DOCS, GNURADIO_BLOCKS, HANDOFF, ITEMS, PROJECT, REPO, SFT_RELEASE

CHECKLISTS = DOCS / "checklists"
GUIDELINES = DOCS / "guidelines"
TEMPLATES = HANDOFF / "review_sheets" / "templates"
SPECS_TEXT = PROJECT / "benchmark and evaluation" / "_build_scripts" / "specs_text"
EVAL_PLAN = SPECS_TEXT / "p1_Benchmark_Evaluation_Plan_to_7Dec.txt"   # read-only text of the CAD 504 plan

DOC_FILES = [
    CHECKLISTS / "tier_f_task_review.md",
    CHECKLISTS / "tier_a_task_review.md",
    CHECKLISTS / "expert_review.md",
    GUIDELINES / "rubric_writing.md",
    GUIDELINES / "judge_calibration.md",
    GUIDELINES / "error_analysis.md",
]
ALL_MD = DOC_FILES + [TEMPLATES / "README.md"]

TEMPLATE_COLUMNS = {
    "judge_calibration.csv": [
        "answer_id", "task_type", "question_id", "question", "answer", "line_kind", "line_id",
        "line_text", "label_team_1", "note_team_1", "label_team_2", "note_team_2", "label_expert",
        "note_expert", "adjudicated_label", "adjudication_note"],
    "blinding_key.csv": [
        "blind_id", "sheet", "item_id", "split", "source_group", "model", "configuration", "run_id",
        "sample_index", "selection_seed", "note"],
    "error_analysis.csv": [
        "failure_id", "tier", "item_id", "split", "first_failed_rung", "auto_facts", "trace_ref",
        "primary_1", "secondary_1", "process_1", "lookup_1", "evidence_1",
        "primary_2", "secondary_2", "process_2", "lookup_2", "evidence_2",
        "final_primary", "final_process", "final_lookup", "resolved_by", "note"],
    "expert_review.csv": [
        "item_id", "item_kind", "split", "packet_ref", "correct", "relevant", "verdict",
        "reason_code", "comment", "suggested_fix", "minutes", "expert", "date", "team_action",
        "team_action_by"],
    "tier_a_task_review.csv": [
        "task_id", "split", "category", "instruction", "start_state", "scripted_user", "acceptance",
        "golden_calls", "gates", "disjointness", "reviewer_1_decision", "reviewer_1_edits",
        "reviewer_2_decision", "reviewer_2_edits", "final_decision"],
    "rubric_sheet.csv": [
        "question_id", "split", "task_type", "question", "required_points", "disqualifying_errors",
        "numeric", "numeric_tolerance", "calc_status", "sources", "writer", "checker_1_initials",
        "checker_1_verdict", "checker_1_source", "checker_1_note", "checker_2_initials",
        "checker_2_verdict", "checker_2_source", "checker_2_note", "final_decision", "decided_by"],
}
# The template that documents each sheet's columns.
TEMPLATE_DOC = {
    "judge_calibration.csv": GUIDELINES / "judge_calibration.md",
    "blinding_key.csv": GUIDELINES / "judge_calibration.md",
    "error_analysis.csv": GUIDELINES / "error_analysis.md",
    "expert_review.csv": CHECKLISTS / "expert_review.md",
    "tier_a_task_review.csv": CHECKLISTS / "tier_a_task_review.md",
    "rubric_sheet.csv": GUIDELINES / "rubric_writing.md",
}

DECISION_COLUMNS = ["reviewer_1_decision", "reviewer_1_edits", "reviewer_2_decision",
                    "reviewer_2_edits", "final_decision"]
DECISIONS = ["approve", "approve with edits", "reject"]

# eval plan §9 error types, the 16 Oct Tier-F protocol types, and the housekeeping codes.
OUTPUT_ERRORS = ["hallucinated_block", "wrong_port_type", "missing_connection", "rate_mismatch",
                 "parameter_error", "runtime_error", "wrong_behaviour"]
TIER_A_PROCESS = ["malformed_call", "unknown_tool", "edit_without_inspect", "giving_up", "looping"]
TIER_F_PROTOCOL = ["rejected_batch", "lookup_not_used", "wrong_lookup"]
HOUSEKEEPING = ["no_output", "item_defect_suspected", "other"]
INVENTED_EXAMPLE_IDS = {"blocks_low_pass_filter"}   # deliberately nonexistent (hallucination example)

QID = re.compile(r"\bq_[0-9a-f]{16}\b")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
JSON_BLOCK = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
PLACEHOLDER = re.compile(r"\A<[^\n]*>\Z")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The body of the Markdown section whose heading line starts with `heading`."""
    start = text.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    rest = text[start + len(heading):]
    m = re.search(rf"^#{{1,{level}}} ", rest, re.MULTILINE)
    return rest[: m.start()] if m else rest


def _row_cells(text: str, first_cell: str) -> str:
    """The text of the Markdown table row(s) whose first cell is `first_cell`."""
    return "\n".join(line for line in text.splitlines() if line.startswith(f"| {first_cell} |"))


# ------------------------------------------------------------------ files and links -------------

@pytest.mark.parametrize("path", ALL_MD, ids=lambda p: p.name)
def test_document_is_plain_markdown_with_a_title(path):
    raw = path.read_bytes()
    assert raw, f"{path} is empty"
    assert b"\r" not in raw, f"{path} must use LF line endings"
    text = raw.decode("utf-8")
    assert text.startswith("# ") and text.endswith("\n")
    assert "\t" not in text


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.name)
def test_document_cites_the_eval_plan_section_it_implements(path):
    head = _text(path).split("\n\n", 2)[1]
    assert head.startswith("Implements eval plan §"), head[:80]


@pytest.mark.parametrize("path", ALL_MD, ids=lambda p: p.name)
def test_relative_links_resolve(path):
    for target in LINK.findall(_text(path)):
        if re.match(r"[a-z]+://", target) or target.startswith("#"):
            continue
        resolved = (path.parent / target.split("#", 1)[0]).resolve()
        assert resolved.exists(), f"{path.name}: broken link {target}"
        assert REPO.resolve() in resolved.parents, f"{path.name}: link leaves the repository: {target}"


# ------------------------------------------------------------------ templates -------------------

def test_template_set_is_complete():
    assert sorted(p.name for p in TEMPLATES.glob("*.csv")) == sorted(TEMPLATE_COLUMNS)


@pytest.mark.parametrize("name", sorted(TEMPLATE_COLUMNS))
def test_template_is_a_header_only_csv_with_the_documented_columns(name):
    path = TEMPLATES / name
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    raw.decode("ascii")                     # plain ASCII header, no BOM
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows == [TEMPLATE_COLUMNS[name]], "templates hold the header row only"
    assert len(set(rows[0])) == len(rows[0])
    doc = _text(TEMPLATE_DOC[name])
    readme = _text(TEMPLATES / "README.md")
    assert f"`{name}`" in readme or f"[`{name}`]" in readme
    missing = [c for c in rows[0] if c not in doc]
    assert not missing, f"{TEMPLATE_DOC[name].name} does not describe columns {missing}"


# ------------------------------------------------------------------ agreement with the code ------

def _tier_f_sheet_header() -> list[str]:
    src = _text(REPO / "src" / "grcbench" / "tier_f" / "cli.py")
    m = re.search(r'if args\.cmd == "sheet":.*?w\.writerow\(\[(.*?)\]\)', src, re.DOTALL)
    assert m, "tier_f/cli.py no longer writes the review-sheet header the checklist documents"
    return re.findall(r'"([^"]+)"', m.group(1))


def test_tier_f_checklist_columns_exist_in_the_generated_sheet():
    header = _tier_f_sheet_header()
    doc = _text(CHECKLISTS / "tier_f_task_review.md")
    table = _section(doc, "## Record your decision")
    documented = [name for line in table.splitlines() if line.startswith("| `")
                  for name in re.findall(r"`([a-z_0-9]+)`", line.split("|")[1])]
    assert set(DECISION_COLUMNS) <= set(documented)
    assert set(documented) <= set(header), f"documented but not in the sheet: {set(documented) - set(header)}"
    assert "./gb tier-f sheet" in doc and "tier_f_first_set.csv" in doc


def test_tier_a_template_uses_the_tier_f_decision_columns():
    header = _tier_f_sheet_header()
    tier_a = TEMPLATE_COLUMNS["tier_a_task_review.csv"]
    assert [c for c in header if c in DECISION_COLUMNS] == DECISION_COLUMNS
    assert tier_a[-len(DECISION_COLUMNS):] == DECISION_COLUMNS


@pytest.mark.parametrize("path", [CHECKLISTS / "tier_f_task_review.md",
                                  CHECKLISTS / "tier_a_task_review.md", TEMPLATES / "README.md"],
                         ids=lambda p: p.name)
def test_decision_values_are_spelled_the_same_everywhere(path):
    text = _text(path)
    for value in DECISIONS:
        assert f"`{value}`" in text, f"{path.name} does not spell out `{value}`"


def test_calibration_labels_match_the_judge_verdict_schema():
    from grcbench.tier_k import judge_prompt
    doc = _text(GUIDELINES / "judge_calibration.md")
    readme = _text(TEMPLATES / "README.md")
    for verdict in judge_prompt.CREDIT:                     # met, partially_met, not_met
        assert f"`{verdict}`" in doc and f"`{verdict}`" in readme
    assert f"`{judge_prompt.UNIT_ERROR_ID}`" in doc
    assert judge_prompt.CREDIT == {"met": 1.0, "partially_met": 0.5, "not_met": 0.0}
    assert "met 1, partially met 0.5, not met 0" in doc


def test_calibration_pass_rule_matches_the_stats_code():
    from grcbench.stats import agreement
    assert (agreement.KAPPA_MIN, agreement.KAPPA_CI_LOW_MIN) == (0.7, 0.6)
    doc = _text(GUIDELINES / "judge_calibration.md")
    flat = " ".join(doc.replace("**", "").split())
    assert "Cohen's κ ≥ 0.7 and the lower end of its 95% interval ≥ 0.6" in flat
    for name in ("cohen_kappa_ci", "judge_passes", "scott_pi", "krippendorff_alpha",
                 "judge_human_gap", "by_group"):
        if f"`{name}`" in doc or name in doc:
            assert hasattr(agreement, name), f"the guide names stats.agreement.{name}, which does not exist"


def test_error_codes_are_documented_once_and_listed_in_the_readme():
    doc = _text(GUIDELINES / "error_analysis.md")
    readme = _text(TEMPLATES / "README.md")
    for code in OUTPUT_ERRORS + TIER_A_PROCESS + ["rejected_batch"]:
        assert len(re.findall(rf"^### `{code}`: ", doc, re.MULTILINE)) == 1, f"no single section for {code}"
    assert len(re.findall(r"^### `malformed_call` in Tier F: ", doc, re.MULTILINE)) == 1
    assert "### `lookup_not_used` and `wrong_lookup`" in doc
    glance = _section(doc, "## All codes at a glance")
    listed = set(re.findall(r"`([a-z_]+)`", glance)) - {"primary", "process", "lookup"}
    expected = set(OUTPUT_ERRORS + TIER_A_PROCESS + TIER_F_PROTOCOL + HOUSEKEEPING + ["none"])
    assert listed == expected
    readme_codes = set(re.findall(r"`([a-z_]+)`", _section(readme, "## Allowed values")))
    assert expected <= readme_codes


def test_primary_precedence_lists_every_primary_code_once():
    doc = _text(GUIDELINES / "error_analysis.md")
    order = re.findall(r"^\d+\. `([a-z_]+)`$", _section(doc, "### Choosing the primary code"), re.MULTILINE)
    assert order == ["no_output"] + OUTPUT_ERRORS + ["other"]     # OUTPUT_ERRORS is in precedence order
    assert set(order) | {"item_defect_suspected"} == set(OUTPUT_ERRORS + HOUSEKEEPING)


def test_unknown_tool_section_lists_grc_agents_22_tools():
    sft = SFT_RELEASE / "sft_train.jsonl"
    if not sft.exists():
        pytest.skip("SFT release not available")
    with open(sft, encoding="utf-8") as fh:
        row = json.loads(fh.readline())
    tools = {(t.get("function") or t)["name"] for t in row["tools"]}
    assert len(tools) == 22
    section = _section(_text(GUIDELINES / "error_analysis.md"), "### `unknown_tool`")
    listed = set(re.findall(r"`([a-z_]+)`", section.split("- **Decision rule.**")[0]))
    listed -= {"unknown_tool", "6c98128"}
    assert listed == tools


def _catalog_ids() -> set[str]:
    ids = set()
    for path in GNURADIO_BLOCKS.glob("*.block.yml"):
        m = re.search(r"^id:\s*(\S+)", path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
        if m:
            ids.add(m.group(1))
    return ids


@pytest.mark.gnuradio
def test_block_ids_in_examples_exist_in_the_gnuradio_catalog():
    if not GNURADIO_BLOCKS.is_dir():
        pytest.skip("GNU Radio block catalog not installed")
    catalog = _catalog_ids()
    assert len(catalog) > 500
    pattern = re.compile(r"`((?:analog|blocks|filter|digital)_[a-z0-9_]+|[a-z_]*(?:low|high|band)_pass_filter"
                         r"|rational_resampler_xxx|fir_filter_xxx)`")
    for path in DOC_FILES:
        for block_id in set(pattern.findall(_text(path))):
            if re.search(r"_\d+$", block_id):           # an instance name such as filter_0
                continue
            if block_id in INVENTED_EXAMPLE_IDS:
                assert block_id not in catalog, f"{block_id} was meant to be an invented id"
                continue
            assert block_id in catalog, f"{path.name}: `{block_id}` is not a GNU Radio 3.10 block id"


# ------------------------------------------------------------------ role table and leakage -------

def test_rubric_examples_use_dev_questions_and_placeholders_only():
    doc = _text(GUIDELINES / "rubric_writing.md")
    dev = {r["id"]: r for r in read_jsonl(ITEMS / "tier_k" / "dev" / "questions.jsonl")}
    test = {r["id"] for r in read_jsonl(ITEMS / "tier_k" / "test" / "frozen_questions.jsonl")}
    ids = set(QID.findall(doc))
    assert len(ids) == 2
    flat = " ".join(doc.split())
    for qid in ids:
        assert qid in dev and qid not in test, f"{qid} must be a dev question"
        assert f'"{dev[qid]["question"]}"' in flat, f"the guide misquotes {qid}"
    blocks = [json.loads(b) for b in JSON_BLOCK.findall(doc)]
    assert len(blocks) == 2
    for rubric in blocks:
        assert rubric["question_id"] in ids and rubric["split"] == "dev"
        prose = [p["text"] for p in rubric["required_points"]]
        prose += [e["text"] for e in rubric["disqualifying_errors"]]
        prose += list(rubric.get("sources", []))
        for field in ("numeric", "numeric_tolerance"):
            value = rubric.get(field)
            if isinstance(value, dict):
                prose += [v for v in value.values()]
        prose += [rubric["calc_status"]] if "calc_status" in rubric else []
        for text in prose:
            assert isinstance(text, str) and PLACEHOLDER.match(text), \
                f"worked examples show the format only; found real content: {text!r}"
        assert rubric["review"]["status"] == "pending-team-review"


def test_rubric_examples_follow_the_judges_rubric_contract():
    from grcbench.tier_k import judge_prompt
    blocks = [json.loads(b) for b in JSON_BLOCK.findall(_text(GUIDELINES / "rubric_writing.md"))]
    for rubric in blocks:
        r = judge_prompt.normalize_rubric(rubric)          # schema, ids, numeric needs a tolerance
        assert [p.id for p in r.points][0] == "P1"
        assert judge_prompt.render_rubric(rubric)
        with pytest.raises(judge_prompt.RubricError):      # not approved yet: never reaches the judge
            judge_prompt.check_reviewed(rubric)
    with_numeric = [r for r in blocks if "numeric" in r]
    assert with_numeric and all(r["numeric_tolerance"] for r in with_numeric)


def test_rubric_guide_states_the_role_table_rules():
    doc = _text(GUIDELINES / "rubric_writing.md")
    assert "Claude does not draft rubric prose" in doc
    assert "pending blind team recomputation" in doc
    assert "recomputes each answer blind" in doc


def _test_item_texts() -> list[str]:
    texts = [r["question"] for r in read_jsonl(ITEMS / "tier_k" / "test" / "frozen_questions.jsonl")]
    for task in sorted((ITEMS / "tier_f" / "test").glob("*/task.json")):
        texts.append(json.loads(task.read_text(encoding="utf-8"))["instruction"])
    return texts


def test_no_document_or_template_quotes_a_test_item_or_carries_the_canary():
    texts = _test_item_texts()
    test_ids = {r["id"] for r in read_jsonl(ITEMS / "tier_k" / "test" / "frozen_questions.jsonl")}
    files = ALL_MD + sorted(TEMPLATES.glob("*.csv"))
    for path in files:
        body = _text(path)
        flat = " ".join(body.split()).lower()
        assert CANARY_GUID not in body, f"{path.name} carries the canary"
        assert not test_ids & set(QID.findall(body)), f"{path.name} names a Tier-K test question"
        assert not re.search(r"\bgbf-test-\d+", body), f"{path.name} names a Tier-F test task"
        for text in texts:
            probe = " ".join(text.split()).lower()[:60]
            assert probe not in flat, f"{path.name} quotes a test item: {probe!r}"


# ------------------------------------------------------------------ budgets, counts and code facts --
# Each test below ties a statement corrected in the 18 Oct review round to the plan, the data or the
# code it describes, so the statement fails loudly when either side changes.

def _eval_plan() -> str:
    if not EVAL_PLAN.exists():
        pytest.skip("evaluation plan text not available")
    return EVAL_PLAN.read_text(encoding="utf-8")


def _hours_rows(section: str) -> list[tuple[str, float]]:
    """(other cells, hours) for every Markdown table row whose last cell is a plain number of hours."""
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and re.fullmatch(r"~?\d+(?:\.\d+)?(?: h)?", cells[-1]):
            rows.append((" | ".join(cells[:-1]), float(re.sub(r"[^\d.]", "", cells[-1]))))
    return rows


def test_error_analysis_sample_is_derived_from_the_table_7_budget():
    m = re.search(r"Error analysis labels \(failures, two annotators\) \| 2 team members \| ≈ (\d+)",
                  _eval_plan())
    assert m, "eval plan Table 7 no longer has the error-analysis line"
    budget = float(m.group(1))                                   # person-hours for two annotators
    section = _section(_text(GUIDELINES / "error_analysis.md"), "## What gets labelled")
    steps = _hours_rows(section)
    assert len(steps) == 4, steps
    per_annotator = math.fsum(h for _, h in steps)
    assert f"| **Total** | **{per_annotator:g}** |" in section
    assert 2 * per_annotator == pytest.approx(budget)
    flat = " ".join(section.split())
    for tier in ("F", "A"):
        m = re.search(rf"\*\*(\d+) Tier-{tier} failures, (\d+) per configuration", flat)
        total, per_config = int(m.group(1)), int(m.group(2))
        assert total == 6 * per_config                           # 2 models x 3 headline configurations
        label, hours = next((label, h) for label, h in steps if f"Tier-{tier} failures at about" in label)
        n, minutes = re.search(r"(\d+) Tier-[FA] failures at about ([\d.]+) minutes", label).groups()
        assert int(n) == total and int(n) * float(minutes) / 60 == pytest.approx(hours)


def test_tier_f_v1_failures_have_codes_and_are_not_no_output():
    from grcbench.tier_a.tier_f_mode import REJECTION_RUNG
    doc = _text(GUIDELINES / "error_analysis.md")
    section = _section(doc, "### `malformed_call` in Tier F")
    named = set(re.findall(r"`([a-z_]+)`", section)) & set(REJECTION_RUNG)
    assert named == {c for c, rung in REJECTION_RUNG.items() if rung == "V1"}
    outcomes = _section(doc, "## Tier-F errors under the 16 Oct format").split("\n### ", 1)[0]
    rows = [line for line in outcomes.splitlines() if line.startswith("| ") and "---" not in line][1:]
    assert len(rows) == 4
    glance = _section(doc, "## All codes at a glance")
    tier_f_process = set(re.findall(r"`([a-z_]+)`", _row_cells(glance, "`process` (Tier F)"))) - {"process"}
    for row in rows:
        process = re.findall(r"`([a-z_]+)`", row.split("|")[2])
        assert len(process) == 1 and process[0] in tier_f_process, row
    outside = " ".join(_section(doc, "## Codes outside the plan's list").split())
    assert "A rejected or refused `change_graph` call is **not** `no_output`" in outside
    assert "`no_output`, `none` and `other` are still **model failures**" in outside


def test_expert_review_plan_covers_every_task_and_states_its_total():
    plan = _eval_plan()
    dev_f = int(re.search(r"dev split \(200 questions, (\d+) flowgraph tasks", plan).group(1))
    expert = int(re.search(r"Expert review by Qoherent \| Qoherent engineer\(s\) \| ≈ (\d+)", plan).group(1))
    section = _section(_text(CHECKLISTS / "expert_review.md"), "## What is in scope")
    hours = math.fsum(h for _, h in _hours_rows(section))
    total = int(re.search(r"\| \*\*Total\*\* \| \| \| \*\*~(\d+) h\*\* \|", section).group(1))
    assert round(hours) == total
    first_set_dev = int(re.search(r"Tier-F first set \(\d+ test, (\d+) dev\)", section).group(1))
    assert f"| Remaining Tier-F dev tasks (after scaling to {dev_f}) | {dev_f - first_set_dev} |" in section
    flat = " ".join(section.split())
    assert f"about {total - expert} hours more than the ≈ {expert} hours of Table 7" in flat


def test_rubric_guide_counts_the_calculation_keys_proposed_tolerances():
    sheets = HANDOFF / "review_sheets"
    key = sheets / "tier_k_calc_key.csv"
    if not key.exists():
        pytest.skip("calculation key not generated")
    with open(key, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with open(sheets / "tier_k_calc_blind_recompute.csv", encoding="utf-8-sig", newline="") as fh:
        blind = list(csv.DictReader(fh))
    assert len(blind) == len(rows)
    # the guide's description of what Claude's script proposes: a tolerance wherever it gives a value,
    # a written reason for every question
    assert [bool(r["value"]) for r in rows] == [bool(r["tolerance"] or r["tolerance_db"]) for r in rows]
    assert all(r["tolerance_note"].strip() for r in rows)
    assert all(r["computed_by"] == "script drafted by Claude (Anthropic)" for r in rows)
    assert {r["specification"] for r in rows if not r["value"]} <= {"team reference", "underspecified"}
    assert "`specification` column says why: `team reference`" in " ".join(
        _text(GUIDELINES / "rubric_writing.md").split())
    flat = " ".join(_text(GUIDELINES / "rubric_writing.md").split())
    assert ("proposes a tolerance (`tolerance` or `tolerance_db`) for every question it gives a value for, "
            "and a written reason (`tolerance_note`) for every question") in flat
    answers = read_jsonl(ITEMS / "tier_k" / "test" / "calc_answers.jsonl")
    assert any(a.get("numeric_proposal") for a in answers) and "(`numeric_proposal`)" in flat
    assert f"the {len(blind)} questions" in flat
    for name in ("tier_k_calc_blind_recompute.csv", "tier_k_calc_instructions.md", "tier_k_calc_key.csv"):
        assert name in flat and (sheets / name).exists()


def test_rubric_guide_quotes_section_6_verbatim():
    plan = " ".join(_eval_plan().split())
    ground_rules = " ".join(_section(_text(GUIDELINES / "rubric_writing.md"), "## Ground rules").split())
    quotes = re.findall(r'"([^"]{20,})"', ground_rules)
    assert len(quotes) >= 2
    for quote in quotes:
        assert quote in plan, f"not in the eval plan: {quote!r}"
    # every family the guide rules out for the drafting model has a role in Table 5
    table_5 = plan[plan.index("Table 5. Which model family"):plan.index("5.2 Reference answers as rubrics")]
    ruled_out = ground_rules[ground_rules.index("That rules out:"):ground_rules.index("Do not start from")]
    families = re.findall(r"\b([A-Z][A-Za-z-]+)\b", ruled_out.split(":", 1)[1])
    assert len(families) >= 15
    for family in set(families):
        assert family in table_5, f"{family} is not in Table 5"


def test_rubric_guide_credit_rules_match_the_rule_cascade():
    from grcbench.tier_k import judge_prompt, rules
    flat = " ".join(_text(GUIDELINES / "rubric_writing.md").split())
    for name in ("CREDIT_TIERS", "HALF_BAND_EXTRA", "MAGNITUDE_RATIO", "RULES_VERSION"):
        assert f"`{name}`" in flat and hasattr(rules, name)
    assert "Half credit up to 5 times the tolerance, but never more than 4 percentage points beyond it" in flat
    assert "9.5 or more times too large or too small" in flat and "default of 0.1 dB" in flat
    assert rules.DB_TOLERANCE == 0.1 and rules.MAGNITUDE_RATIO == 9.5
    assert judge_prompt.WEIGHT_MAX == 10 and "above 0 and at most 10" in flat
    assert judge_prompt.NUMBER_WEIGHT == 1.0 and "counts like a point of weight 1" in flat
    # the guide's worked figures: default 1 % -> full to 1 %, half to 5 %; 10 % -> full to 10 %, half to 14 %
    for tol, cases in ((None, [(100.9, 1.0), (104.9, 0.5), (105.5, 0.0)]),
                       (0.10, [(109.0, 1.0), (113.9, 0.5), (114.5, 0.0)])):
        ref = {"value": 100, "unit": "Hz", **({"tolerance": tol} if tol else {})}
        for answer, credit in cases:
            got = rules.score(f"Final answer: {answer} Hz", ref)
            assert (got["method"], got["credit"]) == ("numeric", credit), (tol, answer, got)
    assert rules.score("Final answer: 950 Hz", {"value": 100, "unit": "Hz"})["credit"] == 0.0


def test_calibration_guide_follows_the_judges_contract():
    from grcbench.tier_k import judge_prompt
    doc = _text(GUIDELINES / "judge_calibration.md")
    flat = " ".join(doc.split())
    readme = _text(TEMPLATES / "README.md")
    for name in set(re.findall(r"`judge_prompt\.([a-z_]+)`", doc)) | {"explain_score", "check_entities", "route"}:
        assert hasattr(judge_prompt, name), f"the guide names judge_prompt.{name}, which does not exist"
    assert judge_prompt.NUMBER_ID.fullmatch("N1") and "`N1`" in doc and "`number`" in doc
    assert "`number`" in readme and "`N1`" in readme
    assert "only when the rubric has a numeric tolerance but no `numeric`" in flat
    assert "The kinds are the judge's: " + ", ".join(f"`{k}`" for k in judge_prompt.ENTITY_KINDS) in flat
    kinds = [f"`{k}`" for k in judge_prompt.CHECKED_KINDS]
    assert f"0 if a listed name of kind {', '.join(kinds[:-1])} or {kinds[-1]} is confirmed absent" in flat
    assert judge_prompt.HALLUCINATION_DISQUALIFIES
    # as the guide says: a rubric with points always goes to the judge, even when the rules read the number
    rubric = {"required_points": ["<point>"], "numeric": {"value": 1, "unit": "Hz"}, "numeric_tolerance": "<t>"}
    assert judge_prompt.route(rubric, {"method": "numeric", "credit": 1.0, "details": {}}) == "judge"


def test_calibration_guide_has_no_early_b3_run():
    flat = " ".join(_text(GUIDELINES / "judge_calibration.md").split())
    assert "So there is no early B3 run." in flat
    assert "Session teacher; B3 upper bound | DeepSeek" in _eval_plan()
    proposal = SPECS_TEXT / "Methodology_Proposal_to_7Dec.txt"
    if proposal.exists():
        assert "is run once, on the frozen benchmark in week 8" in proposal.read_text(encoding="utf-8")


def test_tier_f_checklist_start_rate_and_inspect_cap_match_the_runner():
    from grcbench.tier_a import tier_f_mode
    doc = _text(CHECKLISTS / "tier_f_task_review.md")
    flat = " ".join(doc.split())
    for phrase in ("a sample rate of 48 kHz", "48 kHz sample rate", "48000 samples per second"):
        assert f'"{phrase}"' in flat and tier_f_mode.stated_sample_rate(phrase) == "48000"
    assert '"fs = 48k" is not read' in flat and tier_f_mode.stated_sample_rate("fs = 48k") is None
    # the pitfall the checklist warns about: the first rate stated wins, even an output rate
    assert tier_f_mode.stated_sample_rate(
        "an output sample rate of 8 kHz from an input sample rate of 48 kHz") == "8000"
    assert tier_f_mode.DEFAULT_SAMP_RATE == "32000" and "the value is 32000" in flat
    assert tier_f_mode.tier_f_caps().tool_output_tokens == 4096 and "cut to 4K tokens" in flat
    assert "`start_graph_text`" in doc and hasattr(tier_f_mode, "start_graph_text")


def test_expert_review_discloses_what_claude_drafted():
    flat = " ".join(_text(CHECKLISTS / "expert_review.md").split())
    for part in ('the reference ("golden") flowgraphs', "the oracles", "`spec_properties`",
                 "the end-state checks", "the golden tool-call sequences",
                 "This differs from what the Data Summary (§11) told Qoherent."):
        assert part in flat, part
    drafted = {json.loads(p.read_text(encoding="utf-8")).get("provenance", {}).get("drafted_by")
               for p in (ITEMS / "tier_f").glob("*/*/task.json")}
    assert drafted <= {"Claude (Anthropic)"}


def test_tier_a_checklist_checks_collection_specs_and_reserved_signatures():
    flat = " ".join(_text(CHECKLISTS / "tier_a_task_review.md").split())
    assert "the team's own session-collection task specs" in flat
    assert "`handoff/training_team/flowgraph_reserved_signatures.txt`" in flat
    assert (HANDOFF / "training_team" / "flowgraph_reserved_signatures.txt").exists()
    assert '"Tier-A tasks reuse Tier-F flowgraphs"' in flat
    assert "Tier-A tasks reuse Tier-F flowgraphs" in _eval_plan()


# ------------------------------------------------------------------ 18 Oct methodology review -----
# Each test below ties a statement fixed in the methodology review (findings methodology-F1 to F10)
# to the code, the plan or the data it describes.

def test_rubric_guide_decibel_numeric_is_accepted_by_the_judge_and_the_rules():
    """F1: a dB rubric written exactly as the guide says is accepted; what it forbids is refused."""
    from grcbench.tier_k import judge_prompt, rules
    doc = _text(GUIDELINES / "rubric_writing.md")
    flat = " ".join(doc.split())
    forms = re.findall(r'\{"value": <number>, "unit": "dBm", "abs_tolerance": <[^>]+>\}', doc)
    assert len(forms) >= 2                                   # the field section and the sheet column
    for form in forms:
        numeric = json.loads(re.sub(r"<[^>]+>", "0.5", form)) | {"value": -100}
        rubric = {"required_points": [], "numeric": numeric, "numeric_tolerance": "<t>"}
        ref = judge_prompt.numeric_reference(rubric)
        assert rules.score("Final answer: -100.4 dBm", ref)["credit"] == 1.0
    for wrong in ({"tolerance": 0.5}, {"tolerance_db": 0.5}):     # the two forms the guide forbids
        with pytest.raises(judge_prompt.RubricError):
            judge_prompt.numeric_reference({"required_points": [], "numeric_tolerance": "<t>",
                                            "numeric": {"value": -100, "unit": "dBm", **wrong}})
    with pytest.raises(ValueError):
        rules.score("-100.4 dBm", {"value": -100, "unit": "dBm", "tolerance": 0.5})
    assert "tolerance_db" not in judge_prompt.NUMERIC_KEYS and "`judge_prompt.NUMERIC_KEYS`" in flat
    assert "they ignore it next to `abs_tolerance`" in flat
    assert rules.score("-100.4 dBm", {"value": -100, "unit": "dBm", "abs_tolerance": 0.5,
                                      "tolerance": 1e-9})["credit"] == 1.0
    # the guide's decibel figures
    assert f"about {10 * math.log10(1 + rules.HALF_BAND_EXTRA):.2f} dB beyond the tolerance" in flat
    assert f"about {10 * math.log10(rules.MAGNITUDE_RATIO):.1f} dB or more" in flat
    assert f"about {20 * math.log10(rules.MAGNITUDE_RATIO):.1f} dB for a voltage unit" in flat
    for extra, cases in (({}, [(-100.1, 1.0), (-100.25, 0.5), (-100.3, 0.0)]),
                         ({"abs_tolerance": 0.5}, [(-100.5, 1.0), (-100.66, 0.5), (-100.7, 0.0)])):
        for answer, credit in cases:
            got = rules.score(f"Final answer: {answer} dBm", {"value": -100, "unit": "dBm", **extra})
            assert (got["method"], got["credit"]) == ("numeric", credit), (extra, answer, got)
    assert "half credit up to about 0.27 dB" in flat and "half up to about 0.67 dB" in flat
    got = rules.score("Final answer: -109.8 dBm", {"value": -100, "unit": "dBm"})
    assert got["details"]["reason"] == "order_of_magnitude"


def test_calibration_pass_rule_matches_the_gates_the_stats_code_computes():
    """F2: one pre-registered rule, computed by the code the guide names, with condition 2 labelled
    as a deviation that needs approval."""
    from grcbench.stats import agreement, functions
    flat = " ".join(_text(GUIDELINES / "judge_calibration.md").split())
    assert "applies the rule **twice**, and the judge passes only if both hold" in flat
    assert "**Condition 2 is a deviation, and it needs approval.**" in flat
    assert "**If it is not approved,** the judge passes on condition 1 alone" in flat
    for name in ("gate_lines", "gate_bands", "passes_lines", "passes_bands", "h1a_decision",
                 "presence_agreement"):
        assert f"`{name}`" in flat, name
    assert hasattr(agreement, "presence_agreement") and hasattr(functions, "h1a_decision")
    lines = [1.0, 0.5, 0.0, 1.0] * 15
    answers = [f"a{i // 2}" for i in range(len(lines))]
    scores = {f"a{i}": [1.0, 0.25, 0.75, 0.0][i % 4] for i in range(30)}
    out = agreement.judge_calibration(lines, {"T1": lines, "T2": lines, "E": lines}, None, answers,
                                      judge_scores=scores, human_scores=scores, B=200)
    assert out["passes"] is (out["passes_lines"] and out["passes_bands"]) is True
    assert out["gate_lines"]["n_clusters"] == 30 and out["gate_bands"]["weights"] == "linear"
    one_gate = agreement.judge_calibration(lines, {"T1": lines, "T2": lines, "E": lines}, None,
                                           answers, B=200)
    assert one_gate["passes"] is None and not one_gate["complete"]   # condition 1 alone never passes
    bands = [agreement.score_band(x) for x in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert bands == [0, 1, 2, 2, 3]
    assert "band 1: above 0 and below 0.5" in flat and "band 2: 0.5 or more and below 1" in flat


def test_calibration_number_labels_are_compared_as_credits():
    """F2: N-row values that grade to the same credit agree once the coordinator converts them, as
    the guide says; compared raw, they would need adjudication."""
    from grcbench.stats import agreement
    from grcbench.tier_k import judge_prompt
    flat = " ".join(_text(GUIDELINES / "judge_calibration.md").split())
    assert "The coordinator turns every `P` and `N` label into a credit" in flat
    assert "`judge_prompt.grade_values`" in flat and hasattr(judge_prompt, "grade_values")
    rubric = {"required_points": ["<point>"], "numeric": {"value": 12500, "unit": "Hz"},
              "numeric_tolerance": "<t>"}
    number = judge_prompt.normalize_rubric(rubric).numbers[0]

    def credit(label: str) -> float:          # the conversion the guide describes
        if label in judge_prompt.CREDIT:
            return judge_prompt.CREDIT[label]
        if label == "none":
            return 0.0
        values = []
        for part in label.split(";"):
            written = "(unit not written)" not in part
            value, unit = part.replace("(unit not written)", "").split()
            values.append({"value": float(value), "unit": unit, "unit_written": written})
        return judge_prompt.grade_values(values, number)["credit"]

    labels = {"T1": ["met", "12.5 kHz"], "T2": ["met", "12500 Hz"],
              "E": ["met", "12500 Hz (unit not written)"]}
    with pytest.raises(ValueError, match="adjudicated"):
        agreement.human_reference(labels)
    hr = agreement.human_reference({k: [credit(x) for x in v] for k, v in labels.items()})
    assert hr["reference"] == [1.0, 1.0] and hr["n_adjudicated"] == 0
    assert credit("12.5 kHz; 25 kHz") == 0.0                 # a hedge earns its lowest credit
    assert "`12.5 kHz (unit not written)`" in flat


def test_rubric_writer_is_never_a_checker_or_the_blind_recomputer():
    """F3: three different people per rubric, recorded on the sheet; the hours are reconciled."""
    flat = " ".join(_text(GUIDELINES / "rubric_writing.md").split())
    assert "the writer is never checker 1 or checker 2 of the same rubric" in flat
    assert "the person who recomputes a calculation answer blind is not that rubric's writer" in flat
    assert "Neither of them wrote it." in flat
    cols = TEMPLATE_COLUMNS["rubric_sheet.csv"]
    assert {"writer", "checker_1_initials", "checker_2_initials", "calc_status"} <= set(cols)
    m = re.search(r"Check Tier-K rubrics \((\d+) test \+ dev questions\) \| 2 team members \| ≈ (\d+)",
                  _eval_plan())
    budget_n, budget_h = int(m.group(1)), float(m.group(2))
    n_test = len(read_jsonl(ITEMS / "tier_k" / "test" / "frozen_questions.jsonl"))
    n_dev = len(read_jsonl(ITEMS / "tier_k" / "dev" / "questions.jsonl"))
    n = n_test + n_dev
    per_check = budget_h * 60 / budget_n / 2                   # minutes per rubric per checker
    assert f"about {per_check:.1f} minutes per rubric" in flat
    assert f"rubrics for {n} questions ({n_test} test, {n_dev} dev)" in flat
    assert f"| Writing {n} rubrics, at about 10 minutes each | not budgeted | ≈ {round(n * 10 / 60)} h |" in flat
    assert (f"| Checking {n} rubrics, two checkers at Table 7's rate | ≈ {budget_h:g} h (for {budget_n}) "
            f"| ≈ {round(n * 2 * per_check / 60)} h |") in flat
    assert (f"about {round(n_test * 10 / 60)} hours to write and "
            f"{round(n_test * 2 * per_check / 60)} hours to check") in flat


def test_tier_f_checklist_lists_every_graph_check_kind():
    """F4: the checklist lists every structural check the oracle supports."""
    from grcbench.tier_f import graph_checks
    doc = _text(CHECKLISTS / "tier_f_task_review.md")
    section = _section(doc, "### C. Every stated requirement is checked by V5")
    listed = re.findall(r"^  - `([a-z_]+)`: ", section, re.MULTILINE)
    assert listed == list(graph_checks.KINDS)
    flat = " ".join(section.split())
    assert "needs **both** `same_blocks_as_start` and `same_connections_as_start`" in flat


def test_tier_f_reviews_record_the_edit_format_check_as_notes():
    """F5: the one-batch check has a place on the generated sheet, and owners skip it."""
    doc = _text(CHECKLISTS / "tier_f_task_review.md")
    flat = " ".join(doc.split())
    header = _tier_f_sheet_header()
    assert "reviewer_1_notes" not in header or "`reviewer_1_notes`" in flat
    assert "as a `note: batch` line in your edits cell" in flat
    assert ("A line that starts with `note:` is a record for the team, not an edit. The task owner "
            "does not apply it.") in flat
    assert "write `note: inspect result not checked` in your edits cell" in flat
    assert "`note:`" in _text(CHECKLISTS / "tier_a_task_review.md")
    assert "`note:` lines are not edits." in " ".join(_text(TEMPLATES / "README.md").split())


def test_expert_unsure_is_not_a_verdict():
    """F6: `unsure` belongs in correct/relevant; the verdict waits; H1b counts it as not accepted."""
    doc = _text(CHECKLISTS / "expert_review.md")
    flat = " ".join(doc.split())
    verdict_row = _row_cells(doc, "`verdict`")
    assert re.findall(r"`([a-z ]+)`", verdict_row.split("|")[2]) == ["accept", "accept with edits", "reject"]
    assert "**`unsure`:**" not in doc
    assert "`unsure` is not a verdict." in flat and "leave `verdict` empty" in flat
    assert "an item marked `unsure` on the first pass counts as **not accepted**" in flat
    readme = [line for line in _text(TEMPLATES / "README.md").splitlines() if "| `verdict` |" in line]
    assert len(readme) == 1 and "empty while `correct` or `relevant` is `unsure`" in readme[0]


def test_calibration_judge_run_is_declared_logged_and_not_retained():
    """F7: the judge sees test items, so its run follows the test-run guards."""
    from grcbench.runlog import log
    section = " ".join(_section(_text(GUIDELINES / "judge_calibration.md"), "## The judge's run").split())
    for path in (log.RUN_LIST, log.LOG):
        assert f"`{path.relative_to(REPO).as_posix()}`" in section
    assert "`./gb runlog check <config_id>`" in section
    assert "settings that do not keep prompts" in section
    assert "A Kimi K2.6 run, and the week-8 second check, are declared and logged" in section
    assert "settings that do not keep prompts where providers offer them" in " ".join(_eval_plan().split())


DISCUSSION_FIRST = ("still disagree", "cannot agree", "discussion cannot settle", "reviewers talk",
                    "after talking", "after discussion", "discuss every disagreement")


@pytest.mark.parametrize("path", [CHECKLISTS / "tier_f_task_review.md", CHECKLISTS / "tier_a_task_review.md",
                                  CHECKLISTS / "expert_review.md", GUIDELINES / "rubric_writing.md",
                                  GUIDELINES / "error_analysis.md", TEMPLATES / "README.md"],
                         ids=lambda p: p.name)
def test_one_disagreement_policy_the_expert_decides_every_disagreement(path):
    """F8: Table 1 applied literally in every document."""
    flat = " ".join(_text(path).split())
    rule = "A Qoherent expert decides every disagreement"
    assert rule in flat
    assert rule in _eval_plan()
    for phrase in DISCUSSION_FIRST:
        assert phrase not in flat.lower(), f"{path.name}: discussion-first wording {phrase!r}"


def test_calibration_hours_are_reconciled_with_table_7():
    """F9: the calibration's own estimate is set against Table 7 and recorded as a deviation."""
    m = re.search(r"Judge calibration \(150 answers\) and second check \(50\) \| 2 team members \+ 1 "
                  r"Qoherent expert \| ≈ (\d+)", _eval_plan())
    budget = float(m.group(1))
    section = _section(_text(GUIDELINES / "judge_calibration.md"), "## Hours")
    steps = _hours_rows(section)
    assert len(steps) == 5, steps
    total = math.fsum(h for _, h in steps)
    assert f"| **Total** | | **{total:g}** |" in section
    flat = " ".join(section.split())
    assert f"about {total - budget:g} hours more than Table 7" in flat
    checked = 0
    for label, hours in steps:
        m = re.search(r"(\d+) (?:dev )?answers.* at about (\d+) minutes", label)
        if m:
            practice = 0.5 if "30 minutes of discussion" in label else 0.0
            assert 3 * (int(m.group(1)) * int(m.group(2)) / 60 + practice) == pytest.approx(hours), label
            checked += 1
    assert checked == 3
    assert "this is a third team member" in flat


def test_error_analysis_tier_k_exclusion_is_a_proposal():
    """F9: the plan does not limit §9 to a tier, so leaving Tier K out is marked as our proposal."""
    flat = " ".join(_section(_text(GUIDELINES / "error_analysis.md"), "## What gets labelled").split())
    assert "Tier K is left out. This is our proposal for the pre-registration, not a rule of the plan." in flat
    assert "hallucinated GNU Radio entities" in flat
    proposal = SPECS_TEXT / "Methodology_Proposal_to_7Dec.txt"
    if proposal.exists():
        assert "hallucinated GNU Radio entities" in proposal.read_text(encoding="utf-8")


def _headings(path: Path) -> set[str]:
    return set(re.findall(r"^\[Heading \d\] (\d+(?:\.\d+)?)[. ]", path.read_text(encoding="utf-8"),
                          re.MULTILINE))


def test_cited_plan_and_proposal_sections_exist():
    """F10: every 'eval plan §x' and 'proposal §x' a document cites is a section of that document."""
    proposal_path = SPECS_TEXT / "Methodology_Proposal_to_7Dec.txt"
    if not EVAL_PLAN.exists() or not proposal_path.exists():
        pytest.skip("plan texts not available")
    known = {"eval plan": _headings(EVAL_PLAN), "proposal": _headings(proposal_path)}
    assert {"5.2", "8.1", "9"} <= known["eval plan"] and {"4.2", "5.1"} <= known["proposal"]
    cited = 0
    for path in ALL_MD:
        flat = " ".join(_text(path).split())
        for which, sections in known.items():
            for ref in re.findall(rf"(?i)\b{which},? §(\d+(?:\.\d+)?)", flat):
                assert ref in sections, f"{path.name}: {which} §{ref} does not exist"
                cited += 1
    assert cited > 20


def test_expert_review_covers_dev_gnu_radio_questions():
    """F10: eval plan §5.3 covers every GNU Radio question, dev ones included."""
    section = _section(_text(CHECKLISTS / "expert_review.md"), "## What is in scope")
    rows = [label for label, _ in _hours_rows(section)]
    assert any("dev questions that are about GNU Radio" in label for label in rows)
    assert "dev items included" in " ".join(section.split())

"""Rubric templates and rubric sheets for Tier K (`./gb rubrics`; build plan WS6; eval plan §5.2, Table 5).

One template per question: the 185 frozen test questions and the 200 dev questions. A template is the
rubric record of docs/guidelines/rubric_writing.md with every prose field empty:

  question_id, split, task_type, question_sha256 (of the question text the rubric is for)
  required_points []          the team writes them ("P1 (w=2): ..." lines on the sheet)
  disqualifying_errors []
  numeric null                the rule cascade's reference, filled after the blind recomputation
  numeric_tolerance null
  sources []
  calc_status                 "pending blind team recomputation" for calculation questions, else ""
  review                      {"status": "pending-team-review", "reviewers": [], "notes": ""}
  canary

Claude (Anthropic) wrote this template and the code; Claude never writes rubric prose (Table 5), so
every prose field stays empty until the team fills it. judge_prompt.normalize_rubric refuses blank
prose and judge_prompt refuses every review status but approved, so an unfilled template can never
reach the judge. The ~100 block questions have no rubric: the rule cascade scores them
(tier_k/blockgen.py).

Outputs (`--out` root, default the repository):
  items/tier_k/test/rubrics.jsonl, items/tier_k/dev/rubrics.jsonl     the templates
  handoff/review_sheets/rubric_sheet_test.csv, rubric_sheet_dev.csv   one row per question, with the
                         columns of handoff/review_sheets/templates/rubric_sheet.csv

The command refuses to overwrite a template file or sheet that already holds team content (any
prose, a numeric reference, a review status other than pending, or a filled sheet cell); --force
overrides that.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

from grcbench.common.canary import CANARY
from grcbench.common.io import write_text
from grcbench.common.paths import HANDOFF, ITEMS, REPO

TEMPLATE_VERSION = "rubric-template 1"
SOURCES = {"test": ITEMS / "tier_k" / "test" / "frozen_questions.jsonl",
           "dev": ITEMS / "tier_k" / "dev" / "questions.jsonl"}
OUT = {s: f"items/tier_k/{s}/rubrics.jsonl" for s in SOURCES}
SHEET = {s: f"handoff/review_sheets/rubric_sheet_{s}.csv" for s in SOURCES}
SHEET_HEADER_FILE = HANDOFF / "review_sheets" / "templates" / "rubric_sheet.csv"
PREPARED = ("question_id", "split", "task_type", "question", "calc_status")
PENDING_CALC = "pending blind team recomputation"


def _questions(split: str) -> list[dict]:
    return [json.loads(line) for line in SOURCES[split].read_text(encoding="utf-8").splitlines() if line.strip()]


def template(q: dict) -> dict[str, Any]:
    return {"question_id": q["id"], "split": q["split"], "task_type": q["task_type"],
            "question_sha256": hashlib.sha256(q["question"].encode("utf-8")).hexdigest(),
            "required_points": [], "disqualifying_errors": [], "numeric": None, "numeric_tolerance": None,
            "sources": [], "calc_status": PENDING_CALC if q["task_type"] == "calculation" else "",
            "review": {"status": "pending-team-review", "reviewers": [], "notes": ""},
            "template_version": TEMPLATE_VERSION, "canary": CANARY}


def sheet(split: str, questions: list[dict]) -> str:
    header = next(csv.reader(io.StringIO(SHEET_HEADER_FILE.read_text(encoding="utf-8"))))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for q in questions:
        row = {"question_id": q["id"], "split": split, "task_type": q["task_type"], "question": q["question"],
               "calc_status": PENDING_CALC if q["task_type"] == "calculation" else ""}
        w.writerow([row.get(col, "") for col in header])
    return buf.getvalue()


def _team_content(rel: str, root: Path) -> list[str]:
    """What the team already filled in at root/rel (empty list: safe to write)."""
    path = root / rel
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    found = []
    if rel.endswith(".jsonl"):
        for line in text.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("required_points") or r.get("disqualifying_errors") or r.get("numeric") is not None
                    or r.get("numeric_tolerance") is not None or r.get("sources")
                    or (r.get("review") or {}).get("status") != "pending-team-review"):
                found.append(r.get("question_id", "?"))
    else:
        for row in csv.DictReader(io.StringIO(text)):
            if any((v or "").strip() for k, v in row.items() if k not in PREPARED):
                found.append(row.get("question_id", "?"))
    return found


def outputs() -> dict[str, str]:
    files = {}
    for split in SOURCES:
        qs = _questions(split)
        files[OUT[split]] = "".join(json.dumps(template(q), ensure_ascii=False, sort_keys=True) + "\n" for q in qs)
        files[SHEET[split]] = sheet(split, qs)
    return files


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb rubrics", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO, help="root folder for the outputs (default: the repo)")
    ap.add_argument("--force", action="store_true", help="overwrite files that already hold team content")
    args = ap.parse_args(argv)
    files = outputs()
    filled = {rel: ids for rel in files if (ids := _team_content(rel, args.out))}
    if filled and not args.force:
        for rel, ids in filled.items():
            print(f"not written: {rel} holds team content ({len(ids)} rows, e.g. {ids[:3]})")
        print("nothing written; the team's rubrics are kept (use --force only to discard them)")
        return 1
    for rel, text in files.items():
        write_text(args.out / rel, text)
    counts = {s: len(_questions(s)) for s in SOURCES}
    print(f"wrote rubric templates and sheets: test {counts['test']}, dev {counts['dev']} under {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

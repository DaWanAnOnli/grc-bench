"""`./gb placeholder-rubrics`: stand-in rubrics for testing the Tier-K pipeline before the team's rubrics exist.

    ./gb placeholder-rubrics [--out FILE]
    ./gb tier-k score DIR --rubrics FILE        # then ./gb tier-k judge DIR ...

One record per dev question, in the rubric format of judge_prompt, built by copying: its one required
point P1 is the training release's reference answer to that question (splits.release,
evaluation/questions_validation.jsonl), word for word; no disqualifying errors, no numeric reference,
review status judge_prompt.PLACEHOLDER_REVIEW. No rubric text is written here (Table 5: Claude drafts no
rubric prose), and the reference answers are a model's (reference_model), not checked by the team.

So a placeholder score says only that the pipeline ran: answer, route, judge request, verdict, score.
It is not a GRC-Bench result. The judge accepts a placeholder only from a file given to
`./gb tier-k score --rubrics` on the dev split (run.with_rubric_file); a placeholder pasted into the
repository's rubrics.jsonl is refused like any unapproved rubric. When the team's rubrics are approved
in items/tier_k/dev/rubrics.jsonl, score without --rubrics; the answers stay as they are.

The file goes outside the repository (default $GRC_BENCH_WORK/rubrics/dev_placeholder.jsonl). Dev
questions whose text differs from the release's are skipped and listed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from grcbench.common.canary import CANARY
from grcbench.common.io import write_text
from grcbench.common.paths import WORK
from grcbench.splits import release
from grcbench.tier_k import judge_prompt as jp

OUT = WORK / "rubrics" / "dev_placeholder.jsonl"
TEMPLATE_VERSION = "placeholder-rubric 1"


def records(items: list[dict[str, Any]], answers: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Placeholder rubrics for the items whose question the release answers with the same text, and the
    ids skipped (no reference answer, or another question text)."""
    rows, skipped = [], []
    for item in items:
        ref = answers.get(item["id"]) or {}
        text = (ref.get("reference_answer") or "").strip()
        if not text or (ref.get("question") or "").strip() != item["question"].strip():
            skipped.append(item["id"])
            continue
        rows.append({"question_id": item["id"], "split": "dev", "task_type": item.get("task_type"),
                     "question_sha256": hashlib.sha256(item["question"].encode("utf-8")).hexdigest(),
                     "required_points": [{"id": "P1", "text": text, "weight": 1.0}],
                     "disqualifying_errors": [], "numeric": None, "numeric_tolerance": None,
                     "sources": [{"release": release.RELEASE.name, "member": release.ROOT + release.VALIDATION_QUESTIONS,
                                  "reference_model": ref.get("reference_model")}],
                     "calc_status": "",
                     "review": {"status": jp.PLACEHOLDER_REVIEW, "reviewers": [],
                                "notes": "pipeline test only: the release's reference answer as one point"},
                     "template_version": TEMPLATE_VERSION, "canary": CANARY})
    return rows, skipped


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb placeholder-rubrics", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT, help=f"the rubric file (default {OUT})")
    args = ap.parse_args(argv)
    from grcbench.tier_k import run as krun
    items = krun.load_split("dev")
    rows, skipped = records(items, release.validation_reference_answers())
    write_text(args.out, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    print(f"wrote {len(rows)} placeholder rubrics for {len(items)} dev questions to {args.out}")
    if skipped:
        print(f"skipped {len(skipped)} (no reference answer or another question text): {', '.join(skipped[:10])}")
    print("placeholders test the pipeline only; score with: ./gb tier-k score DIR --rubrics " + str(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

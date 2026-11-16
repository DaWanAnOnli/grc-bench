"""One spec per review sheet: its file, row key, allowed values, roles, blinded columns and the
conservative default the expert agent takes when the evidence cannot settle a row.

The values are the guides' values, spelled as the guides spell them:
  docs/checklists/tier_f_task_review.md, tier_a_task_review.md   approve / approve with edits / reject
  docs/guidelines/rubric_writing.md                                 the same, plus underspecified
  handoff/review_sheets/similarity_sheets_instructions.md           clusters, twins, contamination
  handoff/review_sheets/task_type_instructions.md                   the six task types
  handoff/review_sheets/tier_k_block_questions_instructions.md      the block-question wording review
  handoff/review_sheets/tier_k_calc_instructions.md                 the blind recomputation
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_TYPES = ("calculation", "design", "diagnosis", "factual", "gnuradio_usage", "reasoning")
DECISIONS = ("approve", "approve with edits", "reject")
RUBRIC_VERDICTS = ("approve", "approve with edits", "reject", "underspecified")
TWIN = ("twin", "not twin", "unsure")
CLUSTER = ("same family", "split out")
CONTAMINATION = ("contaminated", "not contaminated", "unsure")
CALC = ("agrees", "disagrees", "underspecified", "no value")
EXPERT_VERDICTS = ("accept", "accept with edits", "reject")

CONTAMINATION_KEY = ("tier", "item_id", "corpus", "doc_id")


@dataclass(frozen=True)
class Sheet:
    name: str                     # short name, used in file names and records
    file: str                     # the sheet in handoff/review_sheets/
    key: tuple[str, ...]          # the columns that identify a row
    kind: str                     # label | decision | rubric | calc | expert
    values: tuple[str, ...]       # what a reviewer may write
    roles: tuple[str, str]        # initials of the two independent reviewers
    guide: str                    # the guide the reviewers follow (repository path)
    default: str | None = None    # the expert's conservative default (None: the expert must choose)
    hidden: tuple[str, ...] = ()  # columns left out of the reviewers' input copy
    group: str | None = None      # rows sharing this column carry one decision

    @property
    def expert_values(self) -> tuple[str, ...]:
        return tuple(v for v in self.values if v != "unsure")


SIMILARITY_GUIDE = "handoff/review_sheets/similarity_sheets_instructions.md"
TASK_TYPE_GUIDE = "handoff/review_sheets/task_type_instructions.md"

_LIST = [
    Sheet("task_type_test", "task_type_ambiguous_test.csv", ("id",), "label", TASK_TYPES, ("A1", "A2"),
          TASK_TYPE_GUIDE, hidden=("annotator_1", "annotator_2")),
    Sheet("task_type_train", "task_type_spotcheck_train.csv", ("id",), "label", TASK_TYPES, ("R1", "R2"),
          TASK_TYPE_GUIDE, hidden=("reviewer_label",)),
    Sheet("calc", "tier_k_calc_blind_recompute.csv", ("question_id",), "calc", CALC, ("R1", "R2"),
          "handoff/review_sheets/tier_k_calc_instructions.md", hidden=("annotator", "value", "unit")),
    Sheet("clusters", "tier_k_test_clusters.csv", ("id",), "label", CLUSTER, ("R1", "R2"), SIMILARITY_GUIDE,
          default="same family", hidden=("team_decision (same family / split out)", "reviewer"),
          group="cluster_id"),
    Sheet("twins", "tier_k_twin_threshold_review.csv", ("source", "id_a", "id_b"), "label", TWIN, ("R1", "R2"),
          SIMILARITY_GUIDE, default="twin", hidden=("linked_now", "team_label (twin / not twin)", "reviewer")),
    Sheet("block_questions", "tier_k_block_questions.csv", ("question_id",), "decision", DECISIONS, ("R1", "R2"),
          "handoff/review_sheets/tier_k_block_questions_instructions.md", default="reject",
          hidden=("answer", "evidence", "reviewer_1_wording", "reviewer_1_note", "reviewer_2_wording",
                  "reviewer_2_note", "final_decision")),
    Sheet("tier_f", "tier_f_first_set.csv", ("task_id",), "decision", DECISIONS, ("R1", "R2"),
          "docs/checklists/tier_f_task_review.md", default="reject",
          hidden=("reviewer_1_decision", "reviewer_1_edits", "reviewer_2_decision", "reviewer_2_edits",
                  "final_decision")),
    Sheet("tier_a", "tier_a_first_set.csv", ("task_id",), "decision", DECISIONS, ("R1", "R2"),
          "docs/checklists/tier_a_task_review.md", default="reject",
          hidden=("reviewer_1_decision", "reviewer_1_edits", "reviewer_2_decision", "reviewer_2_edits",
                  "final_decision")),
    Sheet("rubric_test", "rubric_sheet_test.csv", ("question_id",), "rubric", RUBRIC_VERDICTS, ("C1", "C2"),
          "docs/guidelines/rubric_writing.md", default="underspecified"),
    Sheet("rubric_dev", "rubric_sheet_dev.csv", ("question_id",), "rubric", RUBRIC_VERDICTS, ("C1", "C2"),
          "docs/guidelines/rubric_writing.md", default="underspecified"),
] + [
    Sheet(f"contamination_{t}", f"contamination_{t}.csv", CONTAMINATION_KEY, "label", CONTAMINATION, ("R1", "R2"),
          SIMILARITY_GUIDE, default="contaminated", hidden=("decision", "reviewer", "notes"))
    for t in ("tier_k_test", "tier_k_dev", "tier_f_test", "tier_f_dev")
]

SHEETS: dict[str, Sheet] = {s.name: s for s in _LIST}


def key_of(spec: Sheet, row: dict) -> tuple[str, ...]:
    """The row key of a sheet row or of an agent output's `key` mapping."""
    missing = [c for c in spec.key if c not in row]
    if missing:
        raise ValueError(f"{spec.name}: key column(s) {missing} missing in {sorted(row)}")
    return tuple(str(row[c]) for c in spec.key)

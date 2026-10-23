# Label-sheet templates

Empty CSV sheets (a header row only) for the team's labelling and review work. Each one belongs to a
guide in `docs/`. **Never fill in a template itself.** Copy it, or let the coordinator generate a
filled-in copy, and give each person their own copy. All files are UTF-8 CSV and open in Excel or
LibreOffice. In a cell that holds several lines, start a new line with Alt+Enter (Excel) or
Ctrl+Enter (LibreOffice).

| Template | Guide | One row per |
|---|---|---|
| [`judge_calibration.csv`](judge_calibration.csv) | [judge_calibration.md](../../../docs/guidelines/judge_calibration.md) | answer and rubric line (`P1`, `N1`, `D1`, `DU`, `H`) |
| [`blinding_key.csv`](blinding_key.csv) | [judge_calibration.md](../../../docs/guidelines/judge_calibration.md), [error_analysis.md](../../../docs/guidelines/error_analysis.md) | blinded answer or failure (the coordinator only) |
| [`error_analysis.csv`](error_analysis.csv) | [error_analysis.md](../../../docs/guidelines/error_analysis.md) | failed Tier-F sample or Tier-A run |
| [`expert_review.csv`](expert_review.csv) | [expert_review.md](../../../docs/checklists/expert_review.md) | question or task |
| [`tier_a_task_review.csv`](tier_a_task_review.csv) | [tier_a_task_review.md](../../../docs/checklists/tier_a_task_review.md) | Tier-A task |
| [`rubric_sheet.csv`](rubric_sheet.csv) | [rubric_writing.md](../../../docs/guidelines/rubric_writing.md) | Tier-K question |

The Tier-F review sheet is not a template here. It is generated with the tasks' contents by
`./gb tier-f sheet` as `handoff/review_sheets/tier_f_first_set.csv`. Its decision columns are the
same as in `tier_a_task_review.csv`; see
[tier_f_task_review.md](../../../docs/checklists/tier_f_task_review.md).

## Allowed values

Write values exactly as shown (lower case, the same spelling), so the sheets can be read by code.

| Sheet | Column | Values |
|---|---|---|
| `judge_calibration` | `line_kind` | `point`, `number`, `error`, `hallucination` |
| | `label_*`, `adjudicated_label` on `point` rows | `met`, `partially_met`, `not_met` |
| | the same on `number` rows (`N1`, ...) | the value with its unit as the answer gives it, e.g. `12.5 kHz`; `(unit not written)` after it when the answer writes no unit; several values separated by `;`; `none` |
| | the same on `error` rows (`D1`, ..., `DU`) | `present`, `absent` |
| | the same on the `hallucination` row (`H`) | `none`, `present` (names in the note: `name (kind); ...`) |
| `blinding_key` | `sheet` | `judge_calibration`, `judge_calibration_finetuned`, `error_analysis` |
| | `source_group` | `base`, `b3`, `panel`, `final` |
| `error_analysis` | `tier` | `F`, `A` |
| | `primary_*`, `final_primary` | `hallucinated_block`, `wrong_port_type`, `missing_connection`, `rate_mismatch`, `parameter_error`, `runtime_error`, `wrong_behaviour`, `no_output`, `item_defect_suspected`, `other`; Tier F only: `none` |
| | `process_*`, `final_process` (Tier A) | `malformed_call`, `unknown_tool`, `edit_without_inspect`, `giving_up`, `looping`, `other`, `none` |
| | `process_*`, `final_process` (Tier F) | `malformed_call`, `rejected_batch`, `other`, `none` |
| | `lookup_*`, `final_lookup` (Tier F only) | `lookup_not_used`, `wrong_lookup`, `none` |
| | `secondary_*` | any of the codes above, separated by `;` |
| `expert_review` | `item_kind` | `gnuradio_block_question`, `gnuradio_frozen_question` (test), `gnuradio_dev_question`, `tier_f_task`, `tier_a_task` |
| | `correct`, `relevant` | `yes`, `no`, `unsure` |
| | `verdict` | `accept`, `accept with edits`, `reject`; empty while `correct` or `relevant` is `unsure` |
| | `reason_code` | `wrong_answer`, `ambiguous_wording`, `broken_oracle`, `not_relevant`, `outdated`, `other` (several separated by `;`) |
| `tier_a_task_review` (and the Tier-F sheet) | `reviewer_*_decision`, `final_decision` | `approve`, `approve with edits`, `reject` |
| `rubric_sheet` | `checker_*_verdict`, `final_decision` | `approve`, `approve with edits`, `reject`, `underspecified` |
| | `writer`, `checker_*_initials` | initials; three different people for each rubric |

## Rules for every sheet

- **No model sees these sheets.** Never paste test questions, answers or tasks from them into
  ChatGPT, Claude or any other model or online tool (eval plan §6).
- **People work alone first.** Each person fills in only their own columns, on their own copy, and
  does not discuss items until everyone's copy is in. Agreement is computed on these independent
  labels.
- **The coordinator keeps the key.** The `blinding_key` file stays with the coordinator. Annotators
  never see model names.
- **Disagreements go to the Qoherent expert.** A Qoherent expert decides every disagreement (eval
  plan Table 1). The people who disagree do not settle it between themselves first: each note gives
  its writer's reason, and the expert reads them all. The sheet records the decision:
  - `decided_by` in the rubric sheet;
  - `resolved_by` in the error analysis;
  - `adjudication_note` in the calibration;
  - a `note: decided by` line in the task sheets.

  The coordinator counts these decisions.
- **`note:` lines are not edits.** In the Tier-F and Tier-A edits cells, a line starting with
  `note:` is a record for the team (for example the `change_graph` batch you would submit). Task
  owners do not apply it.

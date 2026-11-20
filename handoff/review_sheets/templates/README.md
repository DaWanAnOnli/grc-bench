# Label-sheet templates

Empty CSV sheets (a header row only) for the labelling and review work of the review protocol
([review_protocol.md](../../../docs/guidelines/review_protocol.md)). Each one belongs to a guide in
`docs/`. **Never fill in a template itself.** The coordinator generates a copy with the rows filled in,
and gives each review agent its own input copy. All files are UTF-8 CSV. A cell that holds several
lines has one item per line.

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
| | `source_group` | `b0`, `b1`, `final` |
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
| | `writer`, `checker_*_initials` | `W`, `C1`, `C2`: three different agent runs for each rubric |

## Rules for every sheet

- **Declared runs only.** Test questions, answers and tasks go to no model outside the declared,
  logged runs of the review protocol and of the benchmark (eval plan §6). Every review agent run is
  declared in `reviews/review_log.jsonl`.
- **Each reviewer works alone.** Each review agent fills in only its own columns, from its own input
  copy, and never opens the other reviewer's output. Agreement is computed on these independent labels.
- **The coordinator keeps the key.** The `blinding_key` file stays with the coordinator. Annotators
  never see model names.
- **The expert agent decides every disagreement.** It takes the place of the Qoherent expert of eval
  plan Table 1 ("A Qoherent expert decides every disagreement"). The reviewers who disagree do not
  settle it between themselves first: each note gives its writer's reason, and the expert reads them
  all. The sheet records the decision:
  - `decided_by` in the rubric sheet;
  - `resolved_by` in the error analysis;
  - `adjudication_note` in the calibration;
  - a `note: decided by EX` line in the task sheets.

  The coordinator counts these decisions.
- **`note:` lines are not edits.** In the Tier-F and Tier-A edits cells, a line starting with
  `note:` is a record (for example the `change_graph` batch you would submit). The coordinator does
  not apply it.

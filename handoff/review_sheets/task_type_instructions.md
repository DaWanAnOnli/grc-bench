# Task-type review sheets: instructions

Two sheets, both UTF-8 CSV (open in Excel or LibreOffice). Background and counts: `reports/labels.md`,
but **annotators of the test sheet must not open that report, `splits/task_type_labels.jsonl`,
`splits/task_type_review_key.json` or `src/grcbench/labels/ambiguous_decisions.json` until their
labels are returned**: those files show Claude's labels.

The test sheet holds frozen test questions. Keep it inside the team, and never paste its questions
into ChatGPT, Claude or any other model or online tool (frozen-split rule: eval questions are never
sent to any model).

## Allowed labels (from `00_dataset_spec/task_types.json`)

- `calculation`: Compute an engineering result using explicit quantities, units, and checkable intermediate work.
- `design`: Select parameters, architecture, or methods while explaining constraints and tradeoffs.
- `diagnosis`: Reason from symptoms, distinguish plausible causes, identify missing evidence, and propose tests or recovery actions.
- `factual`: Recall or explain a useful engineering fact or concept.
- `gnuradio_usage`: Use or configure GNU Radio correctly, including blocks, ports, types, rates, tags, messages, and scheduler behavior.
- `reasoning`: Infer behavior, explain consequences, compare approaches, or apply concepts to a situation.

Classification rules from the same file:

- Assign the task type that best represents the main capability needed to answer correctly.
- Use secondary labels separately if a question genuinely spans multiple capabilities.
- Do not relabel a GNU Radio diagnosis question as gnuradio_usage when diagnosis is the primary challenge.
- Questions with insufficient information may be reasoning, design, or diagnosis; correct handling of uncertainty does not require a separate task type.

Write exactly one of the six labels above in each label cell (lower case, as spelled).

## Sheet 1: `task_type_ambiguous_test.csv` (11 test questions, blind double labelling)

1. Two team members label independently. Each works on **their own copy** of the sheet and fills in only
   their own column (`annotator_1` or `annotator_2`). Do not discuss items until both are done.
2. Read the question and pick the one type that best represents the main capability needed to answer it.
   `raw_label` is the question generator's own spelling, which is ambiguous by construction.
3. Use `notes` for doubts. Return both copies; agreement (Cohen's κ, each annotator against the other and
   against Claude) is computed with
   `./gb labels agreement --test <copy_1.csv> <copy_2.csv>`.
4. Disagreements are settled by discussion and the agreed label goes into the G2 pre-registration.

## Sheet 2: `task_type_spotcheck_train.csv` (170 training questions, spot-check)

A seeded 10% sample of training rows, stratified by raw spelling, with at least one row for every
spelling. `method` says whether the label came
from the rule map (`rule`) or from Claude's reading of the question (`classified`). This check is not
blind: read the question and the proposed label, then write the label you think is right in
`reviewer_label` (the same label if you agree). Agreement is computed with
`./gb labels agreement --train <sheet.csv>`.

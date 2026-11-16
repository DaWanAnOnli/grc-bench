# Task-type review sheets: instructions

Two sheets, both UTF-8 CSV, filled in under the review protocol
([review_protocol.md](../../docs/guidelines/review_protocol.md)): two review agents, each alone, and
the expert agent for every disagreement. Background and counts: `reports/labels.md`. The annotators of
the test sheet do not open that report, `splits/task_type_labels.jsonl`, `splits/task_type_review_key.json`
or `src/grcbench/labels/ambiguous_decisions.json` until their labels are recorded: those files show
Claude's earlier labels.

The test sheet holds frozen test questions. They go to no model outside the declared runs of the
review protocol and of the benchmark (eval plan §6).

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

1. Two annotator agents (`A1`, `A2`) label independently, each from its own input copy.
2. Read the question and pick the one type that best represents the main capability needed to answer it.
   `raw_label` is the question generator's own spelling, which is ambiguous by construction.
3. Give the reason for every label in `notes`, in a sentence.
4. Agreement (Cohen's κ, each annotator against the other and against Claude's earlier label) is computed
   with `./gb labels agreement --test <copy_1.csv> <copy_2.csv>`.
5. The expert agent decides every disagreement. The decided label goes into the pre-registration.

## Sheet 2: `task_type_spotcheck_train.csv` (170 training questions, spot-check)

A seeded 10% sample of training rows, stratified by raw spelling, with at least one row for every
spelling. `method` says whether the label came from the rule map (`rule`) or from Claude's reading of
the question (`classified`). This check is not blind:

1. Read the question and the proposed label.
2. Write the label you think is right as your value, the same label if you agree.
3. Give a one-line reason whenever you differ from the proposed label.

Two reviewer agents (`R1`, `R2`) each check every row. The expert agent decides where they differ.
The decided label goes in `reviewer_label`. Agreement is computed with
`./gb labels agreement --train <sheet.csv>`. The training data itself is not changed: the spot-check
measures how often the proposed labels are right.

## Output fields

One JSON object per row, written to your output file as soon as the row is done:

- `sheet`: `"task_type_test"` or `"task_type_train"`;
- `key`: `{"id": ...}`;
- `value`: one of the six labels;
- `notes`: your reason.

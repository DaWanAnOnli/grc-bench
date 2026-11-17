# The review protocol: review agents, an expert agent and fixed decision rules

Implements eval plan §5.2 (every rubric written, checked twice and every disagreement decided), §5.3
(the expert review of every GNU Radio question and every Tier-F and Tier-A task), §5.4 (two-person
review of every task), Table 1 ("A Qoherent expert decides every disagreement") and §6 (test items
reach a model only in a declared, logged run). Pre-registration §10 lists where this protocol departs
from the eval plan.

Every review sheet in `handoff/review_sheets/` is filled in by Claude (Anthropic) review agents
(`claude-opus-5-5`), following the sheet's guide. Each guide says what to check; this page says who
checks, how two reviews become one decision, and how each decision is checked again.

## Roles

| Role | Initials on the sheet | Does |
|---|---|---|
| Reviewer 1, reviewer 2 | `R1`, `R2` (`A1`, `A2` on the task-type test sheet) | review every row of a sheet, each alone |
| Rubric writer | `W` | writes every Tier-K rubric (`docs/guidelines/rubric_writing.md`) |
| Checker 1, checker 2 | `C1`, `C2` | check every rubric, each alone |
| Expert | `EX` | decides every disagreement, merges two lists of edits, audits a sample of the agreements, and does the expert review of eval plan §5.3 (`docs/checklists/expert_review.md`) |
| Coordinator | – | prepares the blinded input copies, applies the decision rules (`./gb review`), applies the decided edits, re-runs every automatic check, and counts |

Each role is a separate agent run, in its own fresh context. One agent may hold the same role on
several sheets, for example reviewer 1 of every Tier-K similarity sheet. The rules that keep the
reviews independent:

- reviewer 1 and reviewer 2 of a row are always different agent runs;
- a rubric's writer is never checker 1 or checker 2 of that rubric;
- the agent that recomputes a calculation answer blind is never that rubric's writer;
- the expert is never one of the reviewers whose disagreement it decides.

The expert agent takes the place of the Qoherent expert of Table 1: **the expert agent decides every
disagreement.** The reviewers never settle a disagreement between themselves; each reviewer's note
gives its reason, and the expert reads both.

## Test items and declared runs

Eval plan §6: test items reach a model only in a declared, logged run. Every review agent run is such a
run. It is declared in `reviews/review_log.jsonl` with:

- its role, its sheets and the ids of its rows;
- the model and the date;
- the SHA-256 of its brief and of its output.

No test item goes to any other model outside the declared model runs of the benchmark. Claude reads
every test item and writes the rubrics, so Anthropic models are not on the model panel.

## Blinding

- **Input copies.** Each reviewer works from its own input copy of the sheet, without the other
  reviewer's columns and without the columns its guide keeps from it. The block-question reviewers
  get no `answer` and no `evidence` column. The twin reviewers get no `linked_now` column.
- **Files a reviewer does not open.** No reviewer opens the other role's output, an earlier filled
  copy of a sheet, or the decided records in `reviews/`. Blind roles also stay away from the answer
  files until their own answer is recorded:
  - the calculation key, `items/tier_k/test/calc_answers.jsonl`, `src/grcbench/tier_k/` and `tests/`
    (the blind recomputation and the rubric writer's tolerance);
  - the block-question answers (`items/tier_k/test/block_answers.jsonl`).
- **Timing.** Reviewer 1 and reviewer 2 of a sheet run at the same time.

## Evidence

Every row a reviewer, writer or the expert fills in cites its evidence: a local file with an exact
quote (`<path> :: "<quote>"`), or a command the agent ran together with what the command printed. A
row without evidence for a factual claim is not accepted. The coordinator's scripts re-check these
claims, and do not take an agent's word for them:

| Sheet | Checked by running it |
|---|---|
| calculation answers | each recomputer writes and runs its own code; its value is scored against the key by the rule cascade (`tier_k/rules.py`) |
| GNU Radio block questions | each reviewer builds and runs its own flowgraph from the question's wording alone; its value is scored against GNU Radio's recorded answer |
| Tier F | each reviewer's `note: batch` and every alternative solution it reports go through the ladder again (`./gb ladder`); after the edits, `./gb tier-f gates` |
| Tier A | the gates, the golden replay and the wrong-end-state check, after the edits |
| rubrics | the record parses, the judge and the rule cascade accept it (`judge_prompt.normalize_rubric`, `rules.parse_reference`), its number equals the confirmed calculation within its tolerance, every quoted source line is found in the file it names, and no point shares a word 8-gram with a teacher answer to the question |
| similarity and task-type sheets | allowed values, one decision per test cluster, the band counts behind the twin thresholds |

## Decision rules

The rules are code (`src/grcbench/review/decide.py`), not judgement:

| The two reviews | Decision |
|---|---|
| the same verdict, other than `approve with edits` and `unsure` | final, decided by the two reviewers |
| both `approve with edits` | the expert merges the two lists; two edits that contradict each other are a disagreement, which the expert decides |
| both `reject` on a rubric (a checker's `reject` means "wrong in a way that needs rewriting") | the expert rewrites the rubric, or decides `underspecified` |
| different verdicts, or either `unsure` | the expert decides, after reading both reviewers' reasons |

The expert may not answer `unsure`. When the evidence cannot settle a row, the expert takes the
conservative value its guide names:

- a test cluster: `same family`;
- a twin pair: `twin`;
- a contamination pair: `contaminated`;
- a rubric: `underspecified`;
- a task: `reject`.

**Audit of the agreements.** The expert re-reviews, blind, a seeded sample of every sheet's agreed
rows: 10 %, at least 5 rows, ordered by SHA-256 of `grcbench-review-2026-11`, the sheet and the row.
Where the expert disagrees, the expert decides. If it disagrees on more than 5 % of a sheet's sample,
it reviews every agreed row of that sheet.

## Records

- **The filled sheets.** The coordinator writes the final values, the initials and the notes into the
  sheets in `handoff/review_sheets/`.
- **The decided records.** `reviews/<sheet>.jsonl` holds, per row:
  - both reviews;
  - the expert's decision and the audit, if any;
  - the final value and the rule that decided it.
- **The summary.** `reports/review_summary.md` gives, per sheet:
  - the counts;
  - the agreement of the two reviewers;
  - how many rows the expert decided;
  - the audit result;
  - every removal with its reason;
  - every failed check and how it was resolved.
- **The rubric records.** `./gb review rubrics <work folder>` decides both rubric sheets and writes
  every question's decided rubric into `items/tier_k/<split>/rubrics.jsonl`:
  - the writer's fields where the checkers approved them, the expert's where it decided edits;
  - each final rubric passes the same automatic checks as a written one (see "Evidence"), or nothing
    is written for its split;
  - a question whose rubric is `underspecified` or rejected keeps its record with status `rejected`
    and the reason. It is removed from runs (`./gb tier-k run` never asks it) and counted in the
    summary.
- **The expert review.** `./gb review expert <work folder>` checks every expert-review row against the
  checklist's rules and writes:
  - the verdicts into `handoff/review_sheets/expert_review.csv` and `reviews/expert_review.jsonl`;
  - the H1b counts per tier into `reviews/expert_review_h1b.json`, which the summary reports.
- **The raw outputs.** The agents' own outputs stay outside the repository (`<project>/review_work/`).

## A second round: the replacement tasks

Each Tier-F and Tier-A task the first round rejected is replaced by a new task (pre-registration §10;
`grcbench.tier_f.generate.REPLACEMENTS`, `grcbench.tier_a.replacements`). The replacements go through
this protocol in a second round, with the same roles and rules:

- **Agents.** Agents 31 to 38 (`reviews/review_log.jsonl`): two reviewers per tier, the expert agent
  that decides and audits, the expert review, a code review and one reserve run. No agent of the first
  round takes part.
- **Inputs.** Each sheet's generator appends the new rows and keeps the decided rows exactly as they were
  reviewed. The reviewers get only the rows with no decision:
  `./gb review inputs <work folder> <sheet> --undecided`.
- **Decisions.** `./gb review decide <work folder> <sheet> --merge` decides the second round's rows
  alone, with their own seeded audit sample, and adds them to the decided records. A row decided in the
  first round is refused, never decided again.
- **The expert review.** `./gb review expert <work folder> --merge` adds the new verdicts to the first
  round's records. H1b is counted on the original pool. The replacements were built after the first
  verdicts were known, so their acceptance is reported beside H1b and is not part of it.
- **Which replacements count.** The replacements are numbered before the review. The first ones the
  review accepts, in id order, fill the slots of the tasks they replace (`grcbench.review.replacements`):
  per split for Tier F, and per category and split for Tier A. An accepted replacement beyond its slots
  gets status `reserve`. It keeps its record and, like a rejected task, no run uses it. Until every
  replacement of a group is decided, no run uses any of them, because the group's slots are not settled.
- **Edits after the review.** A decided item is not reviewed again. When the team decides to give a
  first-round task an edit after its review (for example the one its replacement received), the edit
  is a patch marked `post_review`: the task's notes name it as a post-review edit, its version goes up
  by one, its gates are re-run, and pre-registration §10 lists it.

## What this protocol does not show

These limits are reported with the results (pre-registration §10):

- **Agreement within one model.** The two reviewers are two runs of one model. Their agreement
  measures how consistent the model is, not the reliability between two people, and it overstates the
  agreement two people would reach.
- **Blinding by instruction.** Blinding is enforced by what each brief allows the agent to open, not
  by file permissions.
- **Claude writes the references.** Claude labelled part of Qoherent's training data. Rubrics written
  by Claude may favour answers that read like Claude's, and so models trained on those labels. Every
  point therefore rests on a quoted source or an executed check, never on a model answer.
- **Expert acceptance (H1b) is the expert agent's.** It is not evidence of a GNU Radio engineer's
  acceptance.
- **The judge is calibrated against the agents' labels.** The calibration measures agreement with
  Claude's labels, not with people's.

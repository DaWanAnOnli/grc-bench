# Tier-A task review: checklist for the two reviewers

Implements eval plan §5.4 ("Two-person review"; "Wrong-end-state test"), with §4.4 (Tier A: 8–20
tool calls, scripted user, end-state grading), §6 (task-level overlap with training sessions) and
§5.3 (pruning). The approved plan, Phase 4, describes the runner and the gates.

Two team members review every Tier-A task, test and dev, before it is used. Eval plan Table 7
budgets about 70 person-hours for writing and reviewing all Tier-A tasks. There are 42: 30 test
(three in each category) and 12 dev. A review takes longer than for Tier F, because you also read
the golden tool-call sequence: plan on about 20 minutes per task per reviewer.

## Before you start

- **Never paste a test task into ChatGPT, Claude or any other model or online tool** (eval plan §6).
- **Work alone, on your own copy of the sheet.** Do not discuss a task with the other reviewer until
  you have both recorded a decision.
- **Do not edit the task files.** Write the edits you want in the sheet.
- **Who wrote the tasks.** Data Summary §11 D tells Qoherent that team members write the agent tasks.
  In this build, Claude (Anthropic) drafted the specs (`provenance.drafted_by`). The team reviews,
  edits and decides. This deviation is to be raised with Qoherent. Your review is what makes each
  task a team decision, so be as strict as you would be with a stranger's task.

## What a task is made of

Each task is a folder `items/tier_a/<split>/<task_id>/` (format: `src/grcbench/tier_a/task.py`).

| Part | What it is |
|---|---|
| `task.json` → `instruction` | the first user message of the episode |
| `task.json` → `category` | one of Qoherent's nine categories, or a refusal task (see the list below) |
| `task.json` → `start_state` | the graph open in GRC when the episode starts (`active_graph`), or `null` for File > New |
| `start/` | every file the agent finds in its project folder at the start |
| `task.json` → `scripted_user` | the scripted user: rules (pattern → reply), the default reply "please use your best judgement", and the maximum number of replies |
| `task.json` → `acceptance` | the end-state checks the grader runs on the final saved graph |
| `task.json` → `disjointness` | why the task differs from every training-side task specification |
| golden tool-call sequence | the calls a correct agent could make; the gate replays it 10 of 10 times |
| wrong-end-state fixture(s) | deliberately wrong final graphs the grader must reject |

The categories are:

| `category` | Qoherent's category |
|---|---|
| `build` | build |
| `repair` | repair |
| `param` | parameter tuning |
| `epy` | Embedded-Python blocks |
| `modify` | modify |
| `knowledge` | knowledge |
| `save_block` | save-block |
| `file_ops` | file operations |
| `run_verify` | run-and-verify |
| `refusal` | refusal or abstention (an added group of 2–3 tasks) |

These checks are automatic, so you do not need to repeat them:

- the golden sequence replays 10 of 10 times;
- an agent that does nothing fails;
- the wrong end states are rejected;
- the grader re-runs the final graph itself;
- the task folder is well-formed;
- the canary is absent from everything the agent sees.

Your job is to judge whether the task is fair, realistic and correctly graded.

## How the agent meets the task

The same settings apply to every model (eval plan §4.4, §7.1):

- **GRC-Agent:** the pinned version (commit `6c98128`), with its real system prompt and its 22 tools.
- **Caps:**
  - at most 40 model requests;
  - 64K tokens of context;
  - every tool output cut to 4K tokens (beginning and end kept, plus error lines);
  - a time limit on each flowgraph run.

  Hitting a cap is a failure.
- **Offline:** the web tools return a fixed "offline" message, and the shell runs in a sandbox with
  no network.
- **Scripted user:** a reply from `scripted_user` when the agent's message matches a rule and is a
  question. Otherwise the user says "please use your best judgement". When the agent stops without
  asking, the episode ends.
- **Grading:** by the end state only. The final saved flowgraph must pass the Tier-F checks and the
  task's own assertions. Any valid way of reaching the goal counts. An empty or missing answer never
  counts.

## The checklist

### A. Everything needed is stated, and nothing unstated is graded

- [ ] Everything the agent needs is in the instruction or in the start state (the open graph and the
      files in `start/`), as eval plan §5.4 requires.
- [ ] The task can be solved even if the agent never asks a question. Scripted replies may confirm
      ("yes, go ahead") or repeat what the instruction already says. They never carry information
      that only an agent that happens to ask the right question would get.
- [ ] Every acceptance check matches something the instruction asks for. Look for checks on:
  - a file name or folder;
  - a block type;
  - a parameter value;
  - a saved state;
  - a run length

  that the instruction never mentions. Each of these is a hidden requirement.
- [ ] If the grader needs a saved file, the instruction says to save, and where.
- [ ] If the grader runs the graph for a minimum number of samples, the instruction says the
      flowgraph keeps running until it is stopped (or states the length).

### B. 8–20 tool calls

- [ ] The golden sequence has between 8 and 20 tool calls (eval plan §4.4, Table 1). The gate counts
      them. Your question is whether a competent agent would really need about that many.
- [ ] The task is not trivially short with padding added (for example, the same `inspect_graph` call
      repeated to reach 8).
- [ ] The task is not so long that a careful agent would hit the 40-request cap or the 64K context.
      Long tool outputs (`generate_python`, large `inspect_graph` results, run logs) count towards
      the context.

### C. The scripted user

- [ ] There is a rule for every question the agent is expected to ask, with a reply a real user
      would give.
- [ ] The rules match the questions you would expect, worded in several ways. Patterns are regular
      expressions, so check them against two or three phrasings of the same question.
- [ ] No reply gives away the solution beyond what a real user would know.
- [ ] `max_turns` is enough for the expected conversation, and not much more.

### D. End-state grading

- [ ] The acceptance checks decide **what the user asked for**, not how the agent got there. No check
      looks at the trajectory (which tools, in which order).
- [ ] Any valid solution passes. Think of two or three other ways to do the task: another block, a
      variable instead of a literal, an extra harmless block. Would they pass?
- [ ] The checks reuse the Tier-F ladder where a flowgraph is involved: the saved graph loads,
      validates, generates code and runs. Behaviour is checked by an oracle where the task asks for
      a behaviour.
- [ ] **Knowledge and refusal tasks** have a checkable end state:
  - knowledge: the answer, checked against facts the grader works out itself (see the next item);
  - refusal: for example, the agent declines or abstains and makes no change it was not asked for.

  A correct refusal must not be confused with giving up on a solvable task.
- [ ] **Facts in the answer** (`acceptance.answer.facts`). The grader works out each expected value
      when it grades, never from a number written by hand. Check that each fact's `source` picks the
      value the instruction asks about:
  - a parameter of a block in the start or final graph. With a `regex`, the value is one number
    taken out of a longer parameter, for example the frequency inside a Message Strobe's
    `pmt.to_pmt({...})`: check that the pattern takes the right number;
  - an option list from a block's YAML file;
  - a number measured from a file the agent's run wrote (`file`, with an oracle feature such as
    `rms` or `peak_freq`). The expected value follows whatever the run wrote, so another check must
    make sure that the file comes from the graph the instruction asks for.
- [ ] **File checks** (`acceptance.files`). Besides existence, patterns and JSON or YAML keys, a
      file can be checked as a CSV manifest (`csv_sizes`): a header row, then one `name,size` row
      per file of a folder, each size equal to that file's real size. The instruction states the
      header and what each row holds.

### E. The wrong-end-state fixture

- [ ] At least one deliberately wrong final graph is stored with the task, and the grader rejects it
      (eval plan §5.4 [83]).
- [ ] The wrong end state is a **plausible near miss**, not something trivially broken. Examples: the
      right blocks with a wrong cutoff; the change made but not saved; the change saved to the wrong
      file. A grader that only rejects an empty graph proves little.
- [ ] The unchanged start state also fails (the "agent that does nothing" gate).

### F. Disjoint from training task specifications

- [ ] Compare the task with the training-side specifications listed in
      `reports/reference/training_task_specs.json`:
  - Qoherent's collection tasks t01–t15 and t16/22/24/25;
  - GRC-Agent's 15 `tests/scenarios`.

  A task with **the same goal or the same starting flowgraph counts as overlap, even if the text
  differs** (eval plan §6).
- [ ] Compare the task with **the team's own session-collection task specs** too (proposal §10,
      week W3, once they are written). Proposal §4.2 runs collected sessions "on our own task
      specifications (disjoint from the held-out Tier-A tasks)". The same rule applies: the same goal or the same
      starting flowgraph is overlap. If a collection spec overlaps, change the collection spec or
      replace the Tier-A task, and record which.
- [ ] The `disjointness` text in `task.json` is accurate and names the closest training spec.
- [ ] Start graphs taken from GNU Radio examples come from a test or dev seed file (matching the
      task's split) in `splits/flowgraph_seed_split.json`. They never come from a train seed.
- [ ] **Start graphs not taken from an example:** a composed chain, or a Tier-F task's graph (eval
      plan §13: "Tier-A tasks reuse Tier-F flowgraphs").
  - The graph's functional signature v2 falls in the task's split
    ([`reports/tier_f_signature_v2.md`](../../reports/tier_f_signature_v2.md)).
  - A graph reused from a Tier-F task comes from a Tier-F task of the same split.
  - The signature is reserved, so training generators cannot emit a functional twin of it. After
    adding the task, its owner regenerates
    `handoff/training_team/flowgraph_reserved_signatures.txt`.
- [ ] Training sessions will never be generated from this task. Dev and test Tier-A tasks are never
      used for training (eval plan Table 4).

### G. Category and coverage

- [ ] `category` matches what the task really exercises.
- [ ] Across the set, every category is covered by at least one test task. The 2–3 refusal tasks are
      present.
- [ ] No two test tasks are near-duplicates (the same goal with new numbers).

### H. Fits the harness

- [ ] The task does not need the web, the network, audio hardware or radio hardware. Signals are
      simulated; there are no ALSA devices in WSL.
- [ ] Any flowgraph the agent is expected to run stops on its own, or the instruction says how long
      to run it. It fits the per-run time limit.
- [ ] Run-and-verify tasks give the agent a way to verify in the headless harness, for example a
      File Sink or the run log, not a GUI window.

### I. Licence, provenance, canary

- [ ] Start files taken from GNU Radio examples keep their GPL-3.0 notice, and `license` says so.
- [ ] `provenance.drafted_by` and `review.status` are filled in. Claude-drafted tasks start as
      `pending-team-review`.
- [ ] Nothing the agent sees contains the canary string: the instruction, the files in `start/`
      and the scripted replies.

## Record your decision

Use `handoff/review_sheets/tier_a_first_set.csv`, written by `./gb tier-a sheet`. It lists all 42
tasks. Without it, make a copy of the template
[`tier_a_task_review.csv`](../../handoff/review_sheets/templates/tier_a_task_review.csv). Both use the
same decision columns as the Tier-F sheet:

| Column | Content |
|---|---|
| `task_id`, `split`, `category` | the task |
| `status` | `built and gated`: the task is built and passes every gate |
| `instruction`, `start_state`, `scripted_user`, `acceptance`, `golden_calls`, `gates`, `disjointness` | a short summary of the task (filled in when the sheet is made) |
| `wrong_end_state_rejected_by` | the grader's reasons for rejecting the wrong end state, from the gates |
| `build_notes` | for gba-test-11 to gba-test-30, which were built from written specs: the spec's notes on building the task and any decision it asks of the team, and where the built task differs from its spec, what differs and why. Check each difference: you approve the built task, not the spec |
| `reviewer_1_decision`, `reviewer_2_decision` | `approve`, `approve with edits` or `reject` |
| `reviewer_1_edits`, `reviewer_2_edits` | the edits, one per line, or the reason for rejecting |
| `final_decision` | the task owner when both decisions are the same; otherwise the Qoherent expert |

Write edits the same way as for Tier F, one per line:

```
<part>: <what to change> -- <why>
```

`<part>` is one of `instruction`, `start`, `scripted_user`, `acceptance`, `golden`, `wrong_state`,
`labels` or `metadata`. Examples of the form (not taken from a real task):

```
scripted_user: add a rule for "which file should I save to" -> "the same file" -- common question
acceptance: drop the check on the instance name -- the instruction never names it
wrong_state: replace the empty graph with the graph changed but not saved -- a real near miss
```

For a rejection, start the cell with the reason: `hidden requirement`, `not solvable`,
`overlaps training`, `trivial`, `too long`, `wrong grading`, `duplicate` or `other`. Then explain.

A line that starts with `note:` is a record for the team, not an edit, as for Tier F (see
[tier_f_task_review.md](tier_f_task_review.md), "Notes"). For example: `note: alternative with a
variable instead of a literal passes the acceptance checks`.

## After both reviews

The process is the same as for Tier F (see [tier_f_task_review.md](tier_f_task_review.md), "After
both reviews"):

1. Merge the two copies.
2. If the decisions differ, the Qoherent expert decides (eval plan Table 1: "A Qoherent expert
   decides every disagreement"). The reviewers do not settle it between themselves first; the
   expert reads both edits cells.
3. The task owner applies the edits and re-runs the Tier-A gates.
4. Record the result in `task.json` → `review`.
5. Rejected tasks are removed and counted (eval plan §5.3).

The Qoherent expert then reviews every Tier-A task for correctness and relevance (see
[expert_review.md](expert_review.md)).

## One-page summary

1. Everything needed is in the instruction or the start state. It is solvable without asking.
2. No acceptance check grades something the instruction does not ask for.
3. A competent agent needs 8–20 tool calls; the task fits 40 requests and 64K tokens.
4. The scripted user covers the expected questions and gives nothing away.
5. Grading is by end state only, and any valid solution passes.
6. The wrong-end-state fixture is a plausible near miss, and it is rejected.
7. No training spec (Qoherent t-specs, GRC-Agent scenarios, the team's collection specs) has the
   same goal or start graph. Seeds come from test/dev seed files; other start graphs have a v2
   signature in the task's split, and it is reserved.
8. The category is right; there are no near-duplicates; it needs no network or hardware.
9. Licence, provenance, review status are filled in. No canary in anything the agent sees.

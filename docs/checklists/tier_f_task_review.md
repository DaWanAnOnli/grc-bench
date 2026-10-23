# Tier-F task review: checklist for the two reviewers

Implements eval plan §5.4 ("Two-person review": everything needed to solve a task must be in the
instruction or the starting flowgraph), together with §4.3 (Tier F), §5.3 (pruning) and the 16 Oct
decision in [the approved plan](../implementation_plan.md) ("Tier F runs in GRC-Agent's harness as
'one lookup + one edit'", and its oracle policy).

Two team members review every Tier-F task, test and dev, before it is used. Eval plan Table 7
budgets about 60 person-hours for all Tier-F tasks. Once you know the format, plan on about
10 minutes per task per reviewer. The first tasks will take longer.

## Before you start

- **Never paste a test task into ChatGPT, Claude or any other model or online tool.** Test items
  are never sent to a model outside a declared run (eval plan §6). This covers the instruction, the
  `.grc` files and the oracle.
- **Work alone, on your own copy of the sheet.** Do not discuss a task with the other reviewer until
  you have both recorded a decision.
- **Do not edit the task files.** Write the edits you want in the sheet. The task owner applies them
  and re-runs the gates.
- **Claude (Anthropic) drafted the instructions, references and oracles** (`provenance.drafted_by`).
  Nothing is accepted until you approve it. Judge each task as strictly as a human-written one.
- **Do not run `./gb tier-f sheet` once people have started filling in the sheet.** It rewrites
  `handoff/review_sheets/tier_f_first_set.csv` with empty decision columns. Always fill in a copy.

## What a task is made of

Each task is a folder `items/tier_f/<split>/<task_id>/`.

| File | What it is | What you check in it |
|---|---|---|
| `task.json` → `instruction` | the text the model gets | sections A, B, C, I |
| `task.json` → `observation` | where the output is measured (a File Sink), how long the run is | B |
| `task.json` → `oracle` | the V5 checks: `checks` (signal) and `graph_checks` (structure); each has a `group`: `f2p` (the requested change), `p2p` (what already worked must still work) or `main` | C, D |
| `task.json` → `spec_properties` | the author's plain-words list of what the oracle checks and why | C, D |
| `task.json` → `mutations` | broken versions of the reference the oracle must reject | D |
| `task.json` → `fault` | repair tasks: the fault and the rung it should fail at | F |
| `task.json` → `source`, `license`, `perturbation`, `disjointness`, `cluster_id` | where the task comes from and how it is kept apart from training data | G, H |
| `start.grc` | the starting flowgraph (modify, repair and parameter-tune tasks only) | F, I |
| `reference.grc` | the golden solution | E |
| `variants/*.grc` | other valid solutions the oracle must accept | E |
| `gates.json` | the latest automatic gate results | all |
| `golden_calls.json` | (coming with the edit format) one lookup plus the one `change_graph` call that turns the start graph into the reference | I |

The automatic gates already check these things, so you do not need to repeat them:

- the reference passes V1–V5 in 10 of 10 runs (solvability);
- at least two variants pass;
- the start graph fails;
- at least 90% of the mutants are rejected;
- the reference fits the size cap.

Your job is the part a program cannot judge: **is the task fair, complete and correctly checked?**

## How a model meets the task (16 Oct format)

Read the instruction the way the model will meet it:

- The model works inside GRC-Agent, with GRC-Agent's real system prompt and tools.
- The conversation already contains one `inspect_graph` call and its real result for the start
  graph.
  - **The model never sees `start.grc` itself.** It sees GRC-Agent's `inspect_graph` result.
    Like every tool output, that result is cut to 4K tokens if it is longer (the beginning and end
    are kept, plus error lines).
  - **Build tasks** start from an empty graph with one `samp_rate` variable. The runner sets its
    value to the first sample rate the instruction states, for example "a sample rate of 48 kHz",
    "48 kHz sample rate" or "48000 samples per second". If the instruction states none, the value
    is 32000, GRC's File > New default. A task can override the rule with
    `tier_a_setting.start_samp_rate` (`src/grcbench/tier_a/tier_f_mode.py`, `start_graph_text`).
- The model may make **at most one `query_knowledge` lookup**, then **one `change_graph` call**. The
  edit is applied atomically: a rejected batch rolls back, and there is no retry.
- It has at most 3 turns. Any other tool (run, shell, files, web) gets the fixed reply "not available
  in this task; submit your edit with change_graph".
- The final graph is graded by the ladder:
  - V1: the `change_graph` call is well-formed and matches the tool's schema;
  - V2: load;
  - V3: compile (validate, then code generation);
  - V4: bounded run;
  - V5: oracle.

So the model can never run the flowgraph, look at a plot, read a log or ask a question. Everything
it needs must be in the instruction or visible in the `inspect_graph` result of the start graph.

## Useful commands

Run these in WSL from the `grc-bench` folder:

```bash
./gb tier-f list                                   # all tasks, family, level, gate status
./gb tier-f check items/tier_f/dev/gbf-dev-04      # run the ladder once on the reference
./gb ladder items/tier_f/dev/gbf-dev-04/start.grc --task items/tier_f/dev/gbf-dev-04/task.json
                                                   # where does the start graph fail?
./gb ladder my_alternative.grc --task items/tier_f/dev/gbf-dev-04/task.json
                                                   # does the oracle accept your own solution?
gnuradio-companion items/tier_f/dev/gbf-dev-04/reference.grc   # view a graph (needs a WSL display)
```

The ladder prints, for each rung, whether it passed and why. For V5 it prints every check with its
measured `value`, the expected value and the tolerance.

## The checklist

Tick every box or write a problem in your edits cell. The "typical problems" lists come from the
15–17 Oct reviews of the first set (`reports/tier_f_reviews/round1/`).

### A. Everything needed is in the instruction or the start graph

"The start graph" here means what the model sees of it: the `inspect_graph` result (section I).

- [ ] Every number the oracle uses is stated: the sample rate, frequencies, amplitudes, filter edges
      and attenuation.
- [ ] Units are unambiguous. For example, it says whether a dB gain is an amplitude or a power gain,
      and whether a value is RMS or peak.
- [ ] The output is named exactly: the File Sink's file name and its item type (float, complex, ...).
- [ ] Nothing depends on a file, variable or block that is not in the start graph or the catalog.
- [ ] The numbers agree with each other. Recompute any derived value yourself, for example a peak
      amplitude after a gain in dB.
- [ ] The instruction does not do the work for the model by accident, for example by giving away the
      computed gain factor when computing it is the point of the task. If this is intended, say so.

### B. No hidden requirement

- [ ] Every V5 check matches a sentence in the instruction. A check without a sentence is a hidden
      requirement: either add the sentence or remove the check.
- [ ] **Run length.** If the oracle needs a long run (`observation.run.min_items`), the instruction
      says the flowgraph keeps producing samples until it is stopped. Otherwise a correct answer that
      adds a Head block, or stops early, fails V4 for a reason it was never told.
- [ ] **Start-up transients.** If a correct answer may add a filter, the signal checks skip the
      filter's start-up (`skip`). Without it, correct filtered answers fail on overshoot.
- [ ] No tolerance is stricter than the instruction states.

Typical problems: unstated run length; no `skip`, so correct filtered answers fail; a `max_abs`
check stricter than the stated tolerance; a hidden ripple tolerance.

### C. Every stated requirement is checked by V5

Go through the instruction sentence by sentence. For each requirement, find the check that tests it
(`oracle.checks` or `oracle.graph_checks`; `spec_properties` says which is which).

- [ ] Every behaviour the instruction asks for has a signal check.
- [ ] **Structural wording has a structural check.** Words such as "filter the square wave", "use a
      low-pass filter", "only change parameter values" or "keep the File Sink" need a graph check in
      `tier_f/graph_checks.py`:
  - `block_present`: a block of this type is there (optionally with given parameter values);
  - `block_absent`: no block of this type is there;
  - `path`: a signal path from one block to another, optionally through a third;
  - `same_blocks_as_start`: no block added or removed;
  - `same_connections_as_start`: no connection changed, so any rewiring fails;
  - `max_functional_blocks`: at most this many blocks that do work.
- [ ] **"Only change parameter values"** (and every parameter-tune task) needs **both**
      `same_blocks_as_start` and `same_connections_as_start`. With `same_blocks_as_start` alone, an
      answer that keeps the same blocks but rewires them passes.
- [ ] If a requirement cannot be checked, the instruction is rephrased so that it only asks for
      behaviour. Then any valid structure passes (approved plan, 16 Oct oracle policy).
- [ ] **Shortcuts.** Ask: could a model pass without doing the task? Examples:
  - generating the expected output directly with a Signal Source instead of filtering;
  - a Vector Source holding hard-coded samples;
  - tapping the output before the faulty block.

  If a shortcut passes and the task is about the method, a graph check must block it. If the method
  does not matter, the shortcut is a valid answer.

Typical problems: shortcut answers pass oracles that only look at the output; stated filter
parameters (cutoff, transition width, stop-band attenuation) are not enforced; a stated sample rate
or "continuous run" is not checked.

### D. Tolerances follow the specification

- [ ] Each tolerance comes from a number in the instruction ("within 5 %"), or from an engineering
      reason written in `spec_properties`.
- [ ] Tolerances are not so loose that a real error passes. Look at the mutation results in
      `gates.json`: a `survivors` entry is a broken answer the oracle accepted.
- [ ] Tolerances are not so tight that a correct engineering design fails, for example a filter
      that is correct but uses a different window or a slightly different order.
- [ ] Modify, repair and parameter-tune tasks have both groups: `f2p` checks the requested change
      and `p2p` checks that what worked before still works (eval plan §4.3).

### E. Alternative valid solutions are accepted

- [ ] Think of two or three other ways a competent GNU Radio user would solve the task. Examples:
  - another block (a Low Pass Filter block or an FIR Filter with designed taps);
  - another window or filter order that still meets the specification;
  - harmless extra blocks such as a Throttle;
  - a variable instead of a literal;
  - for an edit, changing an existing block instead of replacing it.
- [ ] The `variants/` folder covers at least two different approaches, not two copies of the
      reference.
- [ ] If you doubt that one of your alternatives would pass, build it and run
      `./gb ladder <your.grc> --task <task>/task.json`. Write the result as a `note: ladder -- ...`
      line in your edits cell (see "Notes" below). If it fails and it should pass, add an edit.

### F. The start state fails for the right reason (modify, repair, parameter-tune)

- [ ] `gates.json` → `start_fails` passed.
- [ ] **Repair:** the start graph fails at the rung given in `fault.expected_rung`, with the error you
      would expect from the fault. Run `./gb ladder <task>/start.grc --task <task>/task.json`.
- [ ] **Modify and parameter-tune:** the start graph runs (V4) and fails only the `f2p` checks. The
      `p2p` checks pass on it.
- [ ] The fault or the starting design looks like a mistake a real user could make.

### G. Split and signature

- [ ] **Converted GNU Radio examples (`source.kind` = `gnuradio_example`):** the source file is in
      the list for this task's split in `splits/flowgraph_seed_split.json` (readable summary:
      [flowgraph_seed_split.md](../../handoff/training_team/flowgraph_seed_split.md)). It is never a
      train seed.
- [ ] **Composed chains (`source.kind` = `composed_chain`):** the task's signature falls in the
      task's split. Use the **functional signature v2**, described in
      [`reports/tier_f_signature_v2.md`](../../reports/tier_f_signature_v2.md). On 18 Oct it is still
      pending team review; the team reviews that report once, not task by task.
  - **v2:** the signature is the set of functional classes of the reference's blocks (tone, filter,
    mix, resample, ...). Plumbing such as a Throttle, a sink or a type conversion does not count.
    Neither do constant gains and offsets, and block order does not matter.
  - **Split:** a composed task's split must equal the v2 split of its reference's signature, and
    its `cluster_id` is `chain:` followed by that signature. Look the task up in the re-split table
    (section 6 of the report). A task listed as a conflict must be re-themed or replaced before you
    can approve it.
  - **Variants:** every accepted variant should have the same v2 signature as the reference. The
    report lists the few exceptions and why they are kept.
  - **Old rule:** signature v1 (the ordered block ids along one path) let a Throttle or a swapped
    block order move a task into another split. Do not use it for decisions; it is kept only for
    provenance.
- [ ] **Reserved signatures:** the v2 signatures of all test and dev references and variants are
      reserved, so training generators cannot produce functional twins of them. After a task is
      added, re-themed or re-split, its owner regenerates
      `handoff/training_team/flowgraph_reserved_signatures.txt`.
- [ ] **No near-duplicates.** No other test task has the same theme with only new numbers (for
      example several "square wave, low-pass filtered" tasks). No test task is a template twin of a
      dev task. Tasks that are not independent share a `cluster_id`.

### H. Licence and provenance

- [ ] `license` is right:
  - tasks derived from GNU Radio examples are GPL-3.0-or-later, and the examples' licence notices
    are kept;
  - original composed tasks are CC-BY-4.0, pending the team's confirmation.
- [ ] `source.ref` names the real example file or chain.
- [ ] `perturbation` records which numbers were changed in a public example (eval plan §4.3). Scores
      on original and perturbed tasks are reported separately.
- [ ] `provenance.drafted_by` says who drafted the text (`Claude (Anthropic)` for this first set).
- [ ] The `disjointness` text is true: no training-side task specification has the same goal or the
      same starting flowgraph.
- [ ] Nothing the model sees (the instruction, `start.grc`) contains the canary string.

### I. The task works as a GRC-Agent edit of the open flowgraph

This is the 16 Oct "one lookup + one edit" protocol.

- [ ] **What the model sees of the start graph.** Once the edit-format files exist, the task owner
      attaches to the sheet the prefilled `inspect_graph` result exactly as the model gets it. It
      comes from a kept golden-replay run (`./gb tier-a trace <run_dir>` prints it), together with
      the start-graph rule (`tier_f.start` in the run's record). Check:
  - the result was not cut by the 4K cap (`tier_f.prefill.truncated` is false);
  - every instance name, parameter value and connection the model needs appears in it.

  Until the result is attached, judge from `start.grc`, and write `note: inspect result not
  checked` in your edits cell.
- [ ] Note the **instance names** the model will see (for example `high_pass_filter_0`). The
      model's edit has to refer to blocks by these names.
- [ ] Write down, in a few words, **the one `change_graph` batch you would submit**, as a
      `note: batch` line in your edits cell (see "Notes" below):
  - blocks to add (with catalog ids);
  - blocks to remove;
  - parameters to update;
  - connections to add or remove.

  For example (not taken from a real task): `note: batch -- add low_pass_filter_0 (cutoff 1 kHz);
  connect analog_sig_source_x_0 -> low_pass_filter_0 -> blocks_file_sink_0; remove
  analog_sig_source_x_0 -> blocks_file_sink_0`. Write it for every task, also when you approve:
  it is the record that you checked the edit format. If the fix cannot be done in one batch, flag
  it as an edit.
- [ ] The instruction asks for nothing the model cannot do in this format. It cannot:
  - run the flowgraph or read a run log;
  - look at a plot;
  - save to a path of its choice or write `.grc` YAML;
  - ask the user a question.
- [ ] **Build tasks: the start rate.** The runner sets `samp_rate` to the first sample rate the
      instruction states, otherwise to 32000 (see "How a model meets the task"). Check that this
      gives the right rate:
  - The first rate mentioned is the flowgraph's rate, not, for example, an output rate after
    decimation.
  - It is written in a form the rule reads ("sample rate of 48 kHz", "48 kHz sample rate",
    "48000 samples per second"). "fs = 48k" is not read.
  - If the instruction states no rate, the model starts at 32000. Check that the task does not need
    another rate.

  If the rule picks the wrong rate, ask for a reworded instruction or for
  `tier_a_setting.start_samp_rate` in your edits cell.
- [ ] **One lookup is enough.** The blocks the model needs are either in the start graph or findable
      with one `query_knowledge` search (domain `catalog` or `docs`). If the task needs several
      unfamiliar blocks that no single search would surface, note it. That can be acceptable, but
      it makes the task harder for a reason that has nothing to do with GNU Radio skill.
- [ ] When the edit-format files exist: `golden_calls.json` holds a natural lookup (it does not
      paste the answer into the query) and a `change_graph` call. Its replay gate passes 10 of 10.

### J. Labels

- [ ] `family` is right:
  - build: from nothing;
  - modify: change working behaviour;
  - repair: fix an injected fault;
  - parameter-tune: reach a target by changing parameter values only.
- [ ] `level` follows proposal §5.2:
  - L1: at most 5 blocks, a single-rate chain;
  - L2: 6–15 blocks, rate changes, modulation or demodulation, filtering;
  - L3: more than 15 blocks, multi-rate, tags, PDUs, message ports, OFDM or Embedded Python.
- [ ] Typical problems: a "modify" task that only changes parameter values (it is parameter-tune);
      a level that does not match the block count.

## Record your decision

The sheet is `handoff/review_sheets/tier_f_first_set.csv`, written by `./gb tier-f sheet`
(`src/grcbench/tier_f/cli.py`). Its columns are:

| Column | Who fills it | Content |
|---|---|---|
| `task_id`, `split`, `family`, `level`, `source` | generated | do not change |
| `instruction`, `observation`, `oracle_checks`, `gates` | generated | do not change. `oracle_checks` lists each check as `group:feature@point`, and graph checks as `group:graph:kind` |
| `known_limits` | generated | the author's notes (`review.notes`): the oracle's known limits and the questions left to the team. Read them before section C |
| `reviewer_1_decision` | reviewer 1 | `approve`, `approve with edits` or `reject` |
| `reviewer_1_edits` | reviewer 1 | the edits or the reason for rejecting, and your `note:` lines (format below) |
| `reviewer_2_decision` | reviewer 2 | as above |
| `reviewer_2_edits` | reviewer 2 | as above |
| `final_decision` | the task owner when both decisions are the same; otherwise the Qoherent expert | `approve`, `approve with edits` or `reject` |

If the generated sheet has more columns than this table, they are information for you. Leave them
as they are.

**Decision values.** Write them exactly like this:

- `approve`: the task is fair, complete and correctly checked as it stands.
- `approve with edits`: the task is right once the listed edits are made. The edits are concrete
  enough for someone else to apply them without asking you.
- `reject`: the task cannot be fixed with edits, or should not be in the benchmark. Examples: a
  duplicate, a wrong idea, a split or licence problem, or a requirement that cannot be checked.

**How to write edits.** Write one edit per line inside the cell (Alt+Enter in Excel, Ctrl+Enter in
LibreOffice). Use this form:

```
<part>: <what to change> -- <why>
```

`<part>` is one of `instruction`, `start`, `reference`, `variants`, `oracle`, `observation`,
`labels` or `metadata`. Examples of the form (not taken from a real task):

```
instruction: add "keep producing samples until stopped" at the end -- V4 needs 12288 items
oracle: add skip 4096 to every check -- a correct FIR answer fails on start-up overshoot
oracle: add graph check block_present low_pass_filter -- the instruction says "low-pass filter it"
variants: add a solution using fir_filter_xxx with designed taps -- covers another valid design
labels: family modify -> parameter-tune -- only parameter values change
```

For a rejection, start the cell with the reason in a few words. Use one of: `wrong reference`,
`ambiguous instruction`, `broken oracle`, `duplicate`, `split`, `licence`, `cannot be checked` or
`other`. Then explain.

**Notes.** A line that starts with `note:` is a record for the team, not an edit. The task owner
does not apply it. Use it for:

- `note: batch -- ...`: the one `change_graph` batch you would submit (section I). Every review
  has one;
- `note: inspect result not checked`: the prefilled `inspect_graph` result was not attached yet
  (section I);
- `note: ladder -- ...`: the result of an alternative solution you ran (section E);
- anything else you checked and want on record.

So an `approve` row normally has `note:` lines only, and an `approve with edits` row has both. The
generated sheet has no separate notes column. Separate `reviewer_1_notes` and `reviewer_2_notes`
columns are proposed to the Tier-F owner; until they exist, notes go in the edits cell.

## After both reviews

1. **Merge.** The task owner collects both copies into one sheet.
2. **Same decision.** If both decisions are the same, that is the `final_decision`. If both are
   `approve with edits`, merge the two lists of edits. Two edits that contradict each other count
   as a disagreement.
3. **Different decisions.** The Qoherent expert decides (eval plan Table 1: "A Qoherent expert
   decides every disagreement"). The two reviewers do not settle it between themselves first:
   each edits cell already gives the reviewer's reasons, and the expert reads both. Write the
   expert's decision in `final_decision`, and add `note: decided by <expert's initials>` in an
   edits cell. The task owner counts how many tasks the expert decided.
4. **Apply the edits.** The task owner applies the edits and re-runs the gates
   (`./gb tier-f gates <task_dir>`). A task whose gates fail after the edits comes back to the
   reviewers. Changed tasks get a new `version`. Once the benchmark is frozen, items are never
   rewritten in place.
5. **Record the result** in `task.json` → `review`:
   - `status`: `approved`, `approved-with-edits` or `rejected`;
   - `reviewers`: your initials;
   - `notes`: one line.
6. **Count rejections.** Rejected tasks are removed and counted with their reason. We publish how
   many items were removed and why (eval plan §5.3).
7. **The expert reviews next.** After the team review, the Qoherent expert reviews the task (see
   [expert_review.md](expert_review.md)).

Scaling Tier F from the first set to 150 tasks waits for the team's decisions on the first set.

## One-page summary

1. Everything needed is in the instruction or the start graph, including the run length.
2. No hidden requirement: every check matches a sentence.
3. Every stated requirement is checked, and structural wording has a graph check. "Only change
   parameter values" has both `same_blocks_as_start` and `same_connections_as_start`.
4. Tolerances follow the specification: not looser, not stricter.
5. Other valid solutions pass, and shortcuts are blocked only if the method matters.
6. The start graph fails for the right reason, and `p2p` holds on it.
7. The split matches the seed file or the functional signature v2. No near-duplicates.
8. Licence, source, perturbation and provenance are right. No canary in model-visible files.
9. The fix is one `change_graph` batch on the open graph (written down as a `note: batch` line),
   and one lookup is enough. The model's `inspect_graph` result shows what it needs (not cut at
   4K), and a build task's start rate is the right one.
10. Family and level are right.

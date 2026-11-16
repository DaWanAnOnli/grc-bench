# Writing and checking Tier-K rubrics

Implements eval plan §5.2 ("Reference answers as rubrics"), with §4.2 (Tier-K scoring: rules first,
the judge only when rules cannot decide), §5.1 (Table 5, who may write what), §5.3 (rewrite or
remove underspecified questions, and count them) and proposal §5.1 (tiered relative-error credit; a
unit or order-of-magnitude error scores zero).

This guide is for the **writer agent** (`W`) and the **two checker agents** (`C1`, `C2`) who write and
check the rubrics for the Tier-K questions under the review protocol
([review_protocol.md](review_protocol.md)): the 185 frozen test questions first, then the 107 dev
questions.

## Who writes what

| Part | Who | Why |
|---|---|---|
| Rubric prose: required points, disqualifying errors, tolerance wording | **the writer agent** (Claude, Anthropic) | eval plan §5.2; Table 5 departs (see below) |
| Checking every rubric | **two checker agents, independently**, each against a trusted source. **Neither of them is the rubric's writer** | eval plan §5.2 |
| Settling disagreements between the two checkers | **the expert agent**, in the place of the Qoherent expert of Table 1 | eval plan §5.2 and Table 1 |
| Calculation answers (the values) | **a script** (`./gb calc`, drafted by Claude), then **two recomputer agents recompute each answer blind**. A recomputer is never the rubric's writer | eval plan §5.2; 14 Oct decision |
| The calculation key's proposed tolerances (`tolerance`, `tolerance_db`, `tolerance_note`) | the same script (Claude): **proposals only**, flagged `pending blind team recomputation` until both recomputations agree | see below |
| The rubric's tolerance | **the writer agent**, from the question, before opening the key | eval plan §5.2 |
| The rubric file format, these instructions, the scoring code | Claude (Anthropic) | Table 5 allows code and templates |

**Claude does not draft rubric prose** in the design of eval plan Table 5: the models that labelled
Qoherent's data, Claude among them, never draft references, and a model from an unrelated family
drafts them. The review protocol departs from this: the writer agent is Claude. Pre-registration §10
records the deviation, and the review protocol states what it costs ("What this protocol does not
show"). Two things limit it:

- every point rests on a trusted source the writer quotes, or on an executed check, never on a model
  answer;
- Anthropic models are not scored on Tier K.

The worked examples below show the format only, with placeholders such as
`<point the answer must make>`.

**The key's proposals.** The calculation key that Claude's script produced
(`handoff/review_sheets/tier_k_calc_key.csv`) proposes a tolerance (`tolerance` or `tolerance_db`)
for every question it gives a value for, and a written reason (`tolerance_note`) for every
question. `items/tier_k/test/calc_answers.jsonl` also holds a proposed `numeric` reference for some
questions (`numeric_proposal`). A tolerance is part of a rubric (eval plan §5.2). So the writer treats
all of these as proposals, not as rubric content:

- set the rubric's tolerance from the question, before opening the key;
- use the key's tolerance only as a cross-check afterwards;
- never copy a key tolerance, a `tolerance_note` or a `numeric_proposal` into a rubric without
  marking it `proposed by script (Claude)` in `sources`, so the checkers know where it came from.

**Three different agent runs for each rubric.** In eval plan §5.2 a model drafts the rubric, so both
checks are independent of the writer. The review protocol keeps that independence:

- the writer is never checker 1 or checker 2 of the same rubric;
- the agent that recomputes a calculation answer blind is not that rubric's writer.

The coordinator checks the initials in `writer`, `checker_1_initials`, `checker_2_initials` and
`calc_status` before a rubric is approved.

**Hours.** Eval plan Table 7 budgets ≈ 60 person-hours for checking the rubrics of "285 test + dev
questions", about 6.3 minutes per rubric for each of the two checkers. Tier K has rubrics for 292
questions (185 test, 107 dev). The review protocol does this work in declared agent runs instead
(`reviews/review_log.jsonl`); pre-registration §10 records the change.

**Calculation answers come from scripts.** The script computes each answer from the question's
numbers. Every script answer is flagged `pending blind team recomputation` until two recomputer agents
have recomputed it without seeing the script's value (see "Calculation questions" below). Where the
script gives no value, the key's `specification` column says why: `team reference` (the answer is
an expression or an example) or `underspecified`. The writer agent then writes the reference.

## Ground rules

- **Test questions reach a model only in a declared, logged run.** Look things up in documents and
  run checks yourself; never send a question to another model or online tool. Eval plan §6 says "no
  test item is sent to a teacher that creates training data". It adds that the models evaluated
  through an API (B2 and the panel) "do see the public test items; we use settings that do not keep
  prompts where providers offer them". The writer and checker agents are declared runs of the review
  protocol (`reviews/review_log.jsonl`).
  - **Table 5's drafting model** (eval plan §5.2) is a model whose family has no other role in Table 5.
    Claude has another role, so the review protocol's writer agent is a deviation, recorded in
    pre-registration §10, and Anthropic models are not scored on Tier K. That rules out:
    - the families that labelled Qoherent's data: Claude and OpenAI;
    - the teachers: gpt-oss, Nemotron, Gemma, Qwen, Muse, Devstral, Laguna and DeepSeek;
    - GLM and Mistral;
    - the judge (MiniMax) and the fallback judge (Kimi);
    - the models we train (Cohere, Nanbeige).
- **Do not start from a teacher answer.** Dev questions have teacher answers in Qoherent's data.
  Never open them, and never copy one into a rubric. A rubric built from one teacher's answer rewards
  answers that sound like that teacher (eval plan §5.2: "The rubric, not a single model answer, is what
  the judge compares against"). The coordinator checks every rubric for a shared word 8-gram with the
  question's teacher answers.
- **Write from a trusted source**, and name it. For a GNU Radio fact, quote a local file:
  - the block YAML files (`/usr/share/gnuradio/grc/blocks` in WSL);
  - GNU Radio's installed Python package (`/usr/lib/python3/dist-packages/gnuradio` in WSL);
  - GRC-Agent's documentation (`~/grc-agent/docs` in WSL).

  Write such a source line as `<path> :: "<exact quote>"`; the coordinator finds every quote in the
  file it names. For theory, name the reference (PySDR [10], a textbook, a standard, with the section)
  and show the derivation or the code you ran.

## What a rubric contains

A rubric has three parts (eval plan §5.2, "Key terms"). Its exact format is the rubric record in
`src/grcbench/tier_k/judge_prompt.py`.

1. **Required points** (`required_points`): the points a correct answer must contain.
   - Each point gets an id (`P1`, `P2`, ...) and optionally a weight: above 0 and at most 10,
     default 1.
   - The judge marks each point `met`, `partially_met` or `not_met`. The score is the weighted mean
     of the credits (1, 0.5 and 0) and of the graded numbers' credits (see 3).
2. **Disqualifying errors** (`disqualifying_errors`): errors that make the answer wrong whatever
   else it says. Each gets an id (`D1`, `D2`, ...). If any of them is present, the score is 0.
3. **Numeric tolerance** (`numeric_tolerance`): how close a number must be, in words.
   - **With `numeric`.** For questions with a numeric answer, a fourth field, `numeric`, gives the
     reference to the rule cascade (`src/grcbench/tier_k/rules.py`). The rules score the number
     before any judge sees the answer. If the answer still goes to the judge, each part of
     `numeric` becomes a graded number (`N1`, `N2`, ...). The judge is shown only the kind of unit,
     never the value. It reads the value the answer gives, and the code grades that value with the
     same rules. Each graded number counts like a point of weight 1. To make the number count for
     more, give the explanation points a weight below 1.
   - **Without `numeric`**, the tolerance applies to numbers stated in the points. The judge then
     also checks the reserved error `DU`: a unit of the wrong kind (for example dB where dBm is
     meant) or a factor of 9.5 or more scores 0.

Two things are automatic, so you do not need to write them:

- **Invented GNU Radio names.** The judge lists every block, parameter, port, API or module that the
  answer presents as real but that does not exist in GNU Radio 3.10 (eval plan §5.2: "an invented
  block"). The code checks each name against the pinned GNU Radio 3.10 index. A name confirmed
  absent scores 0. GRC-Agent tool names and other kinds are recorded but do not change the score.
  The check runs only when the rubric has points or errors. You do not need a disqualifying error
  for invented names. Use a disqualifying error for a real block or parameter used in a way that
  makes the answer wrong.
- **Length and style.** The judge ignores them. Never write a point such as "the answer is well
  structured".

## Step by step

1. **Read the question. Decide whether it can be answered as stated.** If the correct answer depends
   on an assumption the question does not state, stop. Mark it `underspecified` (see the section
   below). Do not write a rubric that quietly picks one assumption.
2. **Find the answer in a trusted source.** Write the source down: title, section or page, or the
   YAML file name.
3. **Write the required points.**
   - One idea per point. Write it so that a checker can say yes or no without knowing your intent.
   - State the fact or conclusion, not the words to look for. Write "states that <fact>", not
     "mentions <keyword>".
   - Most questions need 2–5 points. A short factual question may need one.
   - Give a larger weight (2 or 3) only to the point that carries the answer. Leave the others at 1.
   - If the question has several parts ("what is X, and why?"), give each part its own point.
4. **Write the disqualifying errors.** Only errors that make an answer wrong even if it also says
   the right things. Typical examples:
   - a wrong unit or reference level (dB where dBm is meant);
   - the opposite conclusion;
   - a wrong sign or direction ("increases" where it decreases);
   - a confusion the question is designed to test.

   Do not list small slips here. Those are handled by a point being `partially_met`. Many rubrics
   have no disqualifying error, and that is fine.
5. **Set the numeric tolerance**, if the answer has a number (see the next section). Otherwise set
   `numeric_tolerance` to empty (null).
6. **Write the row** to your output file (see "Output fields" below). The two checker agents then
   check it.

## Numeric answers and tolerances

The rule cascade (`./gb score-k`) reads a numeric reference with these fields:

| Field | Meaning |
|---|---|
| `value` | the reference value |
| `unit` | its unit, for example `Hz`, `MS/s`, `dB`, `dBm`, `V`, `bit/s` |
| `tolerance` | relative tolerance; default 0.01 (1%). Not for decibel units (see below) |
| `abs_tolerance` | an absolute tolerance in the reference unit: for values at or near zero, and the tolerance in dB of every decibel unit |
| `unit_required` | `true` if a bare number without a unit must score 0 |
| `accept_units` | other spellings to accept as the reference unit, for example `["dB"]` for a dBi gain |
| `label` | the quantity's name, for example `"Ts"`, when the question asks for several numbers |
| `ratio_kind` | `"power"` (default) or `"amplitude"`: how a decibel value and a plain ratio convert (20 dB is a power ratio of 100 or an amplitude ratio of 10). Set `"amplitude"` for voltage, current or field ratios. dBm and dBW are always power, dBV, dBmV and dBuV always amplitude |
| `form_required` | `true` if the question fixes the form (for example "as a power ratio, not in dB"): an answer in the other form then scores 0. Default `false`: the other form is converted and graded |
| `waveform` | `"sine"` when an RMS voltage may be compared with a peak or peak-to-peak value (factor √2); absent otherwise, and a different voltage convention then scores 0 |
| `quantity` | `"gain"` when the quantity asked for is a gain, so that "a loss of 3 dB" reads as -3 dB even against a positive reference. Absent: a positive reference is read as a loss quantity ("insertion loss of 3 dB" is +3 dB) |

**Decibel units (dB, dBm, dBi, ...).** The tolerance is in dB, and it goes in `abs_tolerance`:

```
{"value": <number>, "unit": "dBm", "abs_tolerance": <tolerance in dB>}
```

- Without `abs_tolerance`, the rules use the default of 0.1 dB.
- Do not give a relative `tolerance` for a decibel value. The rules refuse it when it is alone,
  because it is ambiguous, and they ignore it next to `abs_tolerance`. A rubric with it is refused
  before any answer is scored.
- Do not write `tolerance_db` in a rubric. It is a column of the calculation key, and the rubric
  format (`judge_prompt.NUMERIC_KEYS`) does not accept it. If you take the key's `tolerance_db`
  value, it goes in `abs_tolerance`, marked as a proposal (see "Who writes what").

For several numbers, give a list of such references, with a `label` for each when two share a unit.
The credit is the mean over the parts.

**How credit works.** These values are a proposal; they are fixed in the G2 pre-registration. The
exact values are the constants in `rules.py` (`CREDIT_TIERS`, `HALF_BAND_EXTRA`, `MAGNITUDE_RATIO`,
`RULES_VERSION`). If this guide and the code disagree, the code decides; tell the coordinator.

- A relative error within the tolerance gets full credit. For a decibel unit, the error in dB is
  compared instead.
- Half credit up to 5 times the tolerance, but never more than 4 percentage points beyond it. With
  the default 1%: full credit up to 1%, half credit up to 5%. With 10%: full up to 10%, half up to
  14%.
  - For a decibel unit, the cap is the same 4% in dB: about 0.17 dB beyond the tolerance. With the
    default 0.1 dB: full credit up to 0.1 dB, half credit up to about 0.27 dB. With 0.5 dB: full up
    to 0.5 dB, half up to about 0.67 dB.
  - For voltage levels (dBV, dBmV and dBµV), whose decibels are 20·log10 of a voltage, the 4% applies
    to the voltage: about 0.34 dB beyond the tolerance. With the default 0.1 dB, half credit up to
    about 0.44 dB.
- More than that gets 0.
- A wrong dimension (dB vs dBm, Hz vs s, bit/s vs Hz) scores 0.
- An answer 9.5 or more times too large or too small scores 0 (an order-of-magnitude error). So does
  a sign flip. The only exception is an error within an explicit `abs_tolerance` (for values at or
  near zero).
  - For a decibel unit, an error of about 9.8 dB or more is an order-of-magnitude error (about
    19.6 dB for a voltage unit such as dBV).
- With several numbers, a wrong dimension or an order-of-magnitude error in **any one** of them
  scores the whole answer 0.

**Choose the tolerance from the question**, and write down why:

- Use the default 1% when the question gives exact numbers and expects an exact calculation.
- Use a wider tolerance when the inputs are given to two significant figures, when a standard
  approximation is expected (for example a rule-of-thumb formula), or when reading a value from a
  chart or table is part of the task.
- Never widen a tolerance so far that a classic mistake would pass. Examples: a factor of √2 from
  confusing peak and RMS for a sine; a factor of 2 from confusing peak and peak-to-peak; the power
  versus amplitude decibel mix-up (10·log10 instead of 20·log10).

State the same tolerance in words in `numeric_tolerance`, for the two checkers; the scoring code
applies the tolerance in `numeric`, also to the values the judge reports for answers the rules hand
over. The scoring code refuses a rubric that has `numeric` but no `numeric_tolerance`.

## Calculation questions

The files are in `handoff/review_sheets/`, written by `./gb calc`:

- `tier_k_calc_blind_recompute.csv`: the 25 questions, with empty cells for your value;
- `tier_k_calc_instructions.md`: how to fill it in;
- `tier_k_calc_key.csv`: the script's values. The writer opens it only after step 3.

1. **The script computes the answer.** `./gb calc` computes each calculation answer from the
   question's numbers and flags it `pending blind team recomputation`.
2. **Two recomputer agents recompute it blind** from their input copies of
   `tier_k_calc_blind_recompute.csv`, following `tier_k_calc_instructions.md`. Neither is the rubric's
   writer.
   - Each takes the question text only.
   - Neither looks at the key, `items/tier_k/test/calc_answers.jsonl`, or any file under
     `src/grcbench/tier_k/` or `tests/` (they hold the script's answers).
   - Each computes the answer with its own code, runs it, and records its value, unit and method.
3. **The writer agent sets the tolerance, also before opening the key.** It chooses the tolerance from
   the question (see "Choose the tolerance from the question" above), and writes it and its reason in
   its row.
4. **Compare.** The coordinator scores both recomputations against the key with the rule cascade.
   - If both agree with the script's value, the answer is confirmed: `recomputed by R1;R2 <date>`.
   - If not, the expert agent decides why: an error in the script, an error in a recomputation, or an
     ambiguous question. Ambiguous means underspecified (see below). The key's `specification` column
     marks questions `underspecified`, with the missing assumption.
   - The writer's tolerance is checked against the confirmed value and both recomputations. If the
     writer's tolerance and the key's give different outcomes for a recomputed value, the case goes to
     the expert agent as a disagreement.
   - The expert agent decides every disagreement, and every question marked underspecified (eval plan
     §5.2).
5. **Then the writer writes the rubric's `numeric` and `numeric_tolerance`** from the confirmed value
   and its own tolerance. For a decibel value, the tolerance goes in `abs_tolerance`, in dB (see
   "Decibel units" above). It adds required points only if the question asks for more than the number
   ("calculate ... and explain ...").

## GNU Radio behaviour questions

The 100 script-generated GNU Radio questions ask for one number each: a block's k-th output for a
stated input, or how many items it produces. Their answers come from running GNU Radio
(`./gb blockq`). They are scored by rules (numbers), not by a rubric. Two review agents check **the
wording only**, each by solving every question blind from its wording
([tier_k_block_questions_instructions.md](../../handoff/review_sheets/tier_k_block_questions_instructions.md)):

- Is the question clear?
- Does it state everything the answer depends on?

## Checking a rubric

Two checker agents check every rubric. Neither of them wrote it. Each works alone, from its own input
copy, and never sees the other checker's verdict. Each writes its initials (`C1`, `C2`) in its
`checker_N_initials` cell.

For each rubric:

1. Look the answer up yourself in a trusted source, and quote it (`<path> :: "<exact quote>"`) or show
   the derivation you ran. Write the source in your `source` cell, even if it is the same as the
   writer's.
2. Check each point. Is it right? Is it needed for a correct answer? Is it worded so that a judge
   can decide it?
3. Check that no needed point is missing.
4. Check the disqualifying errors. Are they really disqualifying?
5. Check the tolerance against the question's precision.
6. Give your verdict:

| Verdict | Meaning |
|---|---|
| `approve` | the rubric is right as written |
| `approve with edits` | right after the edits you list in your note (one per line, for example `P2: <new wording> -- <why>`) |
| `reject` | wrong in a way that needs rewriting; say why |
| `underspecified` | the question cannot be answered without an unstated assumption; say which assumption |

Then:

- **Both checkers agree:** that is the decision. If both give `approve with edits`, the expert agent
  merges the two lists. Two edits that contradict each other count as a disagreement. If both give
  `reject`, the expert agent rewrites the rubric, or decides `underspecified`.
- **They disagree:** the expert agent decides (eval plan §5.2 and Table 1, "A Qoherent expert decides
  every disagreement", with the expert agent in the expert's place). The expert agent decides every
  disagreement. The checkers do not settle it between themselves first: each checker's note already
  gives its reason, and the expert reads both. The expert's decision goes in `final_decision`, and
  `EX` in `decided_by`. The coordinator counts how many rubrics the expert decided.
- **Checks the coordinator runs** before a rubric is approved, whatever the verdicts:
  - the record parses into the judge's rubric format and the rule cascade accepts its `numeric`;
  - its number equals the confirmed calculation within its tolerance;
  - every quoted source line is found in the file it names;
  - no point shares a word 8-gram with a teacher answer to the question.

  A rubric that fails one goes to the expert agent with the failure.
- **Record the result** in the rubric's `review` field:
  - `status`: `approved`, `approved-with-edits` or `rejected`;
  - `reviewers`: the checkers' initials;
  - `notes`: one line.
- **Only approved rubrics reach the judge.** The code refuses any other status.

## Underspecified questions

A question whose correct answer depends on assumptions it does not state is **rewritten, or removed
and counted** (eval plan §5.2, §5.3).

- **Rewrite** when one added sentence fixes it. For example, state which of two conventions is
  meant, or give a missing quantity. The rewritten question becomes a new version, and the original
  is kept (frozen items are never changed in place). Record the change.
- **Remove** when no small rewrite fixes it, or when the rewrite would change what the question
  tests. Record the reason.
- **Every rewrite and removal is counted** and published with its reason (eval plan §5.3).
- **How a removal is recorded.** The question's rubric record keeps status `rejected` and the reason
  in `review.notes`. The question stays in its frozen file, and runs leave it out
  (`tier_k/run.py`, `removed`).

## Where to write: the rubric sheet

`./gb rubrics` writes one sheet per split, with the question already filled in on every row:
`handoff/review_sheets/rubric_sheet_test.csv` (185 questions) and `rubric_sheet_dev.csv` (107). It also
writes the matching empty rubric records, `items/tier_k/<split>/rubrics.jsonl`. The columns are those
of [`rubric_sheet.csv`](../../handoff/review_sheets/templates/rubric_sheet.csv). The agents write rows
to their output files; the coordinator writes the decided rows into the sheet and the records
(`./gb review`). The command refuses to overwrite a sheet or record file that already holds review
content. A multi-line cell has one item per line.

| Column | Who | Content |
|---|---|---|
| `question_id`, `split`, `task_type`, `question` | prepared | the question (dev or test) |
| `required_points` | writer | one point per line: `P1: <text>`, or `P1 (w=2): <text>` for a weight |
| `disqualifying_errors` | writer | one per line: `D1: <text>`; empty if none |
| `numeric` | writer | the rule cascade's reference as JSON, e.g. `{"value": <number>, "unit": "<unit>", "tolerance": <number>}`; for a decibel unit `{"value": <number>, "unit": "dBm", "abs_tolerance": <dB>}`; empty if none |
| `numeric_tolerance` | writer | the tolerance in words for the judge; empty if none |
| `calc_status` | coordinator | calculation questions: `pending blind team recomputation`, then `recomputed by R1;R2 <date>` or `decided by EX <date>` |
| `sources` | writer | the trusted sources, one per line |
| `writer` | writer | `W` |
| `checker_1_initials`, `checker_1_verdict`, `checker_1_source`, `checker_1_note` | checker 1 | `C1`, verdict (table above), own source, edits or reason |
| `checker_2_initials`, `checker_2_verdict`, `checker_2_source`, `checker_2_note` | checker 2 | as above |
| `final_decision`, `decided_by` | after the checks | the agreed verdict, and who decided (`checkers` or `EX`) |

The sheet converts mechanically into the rubric record: each `P1 (w=2): ...` line becomes
`{"id": "P1", "weight": 2, "text": "..."}`.

**Output fields.** One JSON object per question, written to the agent's output file as soon as the
question is done.

- **Writer:**
  - `sheet`: `"rubric_test"` or `"rubric_dev"`;
  - `key`: `{"question_id": ...}`;
  - `value`: `"written"` or `"underspecified"`;
  - `required_points`, `disqualifying_errors`: lists of lines;
  - `numeric`: an object or a list, or `null`;
  - `numeric_tolerance`: a string, or `null`;
  - `sources`: lines;
  - `notes`, `edits` (`[]`), `evidence`.
- **Checkers:**
  - `sheet`, `key`;
  - `value`: the verdict;
  - `source`: your own source;
  - `edits`: the edit lines;
  - `notes`;
  - `evidence`.

## Worked examples (dev questions, format only)

These two examples show **the format only**. They are dev questions, never test questions, and
every piece of rubric content is a placeholder. They are not rubrics for these questions.

### Example 1: a calculation question

Dev question `q_978a8b499569ac5d` (calculation, basic): "Calculate the bandwidth of an AM signal
with a carrier frequency of 1 MHz and a maximum modulating frequency of 5 kHz."

The rubric record, once the calculation is confirmed:

```json
{
  "question_id": "q_978a8b499569ac5d",
  "split": "dev",
  "required_points": [
    {"id": "P1", "text": "<point the answer must make>", "weight": 1}
  ],
  "disqualifying_errors": [
    {"id": "D1", "text": "<error that makes the answer wrong>"}
  ],
  "numeric": {"value": "<value from the calculation script, confirmed by blind recomputation>",
              "unit": "<unit>", "tolerance": "<relative tolerance>"},
  "numeric_tolerance": {"quantity": "<the number that is graded>",
                        "tolerance": "<the tolerance, in words>"},
  "sources": ["<trusted source, with section or page>"],
  "calc_status": "<pending blind team recomputation | recomputed by <initials> <date>>",
  "review": {"status": "pending-team-review", "reviewers": [], "notes": ""}
}
```

The same rubric as a row of the sheet (only the filled cells are shown):

| Column | Cell |
|---|---|
| `required_points` | `P1: <point the answer must make>` |
| `disqualifying_errors` | `D1: <error that makes the answer wrong>` |
| `numeric` | `{"value": <confirmed value>, "unit": "<unit>", "tolerance": <relative tolerance>}` |
| `numeric_tolerance` | `<the tolerance, in words>` |
| `calc_status` | `pending blind team recomputation` |
| `sources` | `<trusted source, with section or page>` |

What the checkers do here:

- recompute the value blind;
- check that the unit is the one the question asks for;
- check that the tolerance fits the precision of the question's numbers.

### Example 2: an open-ended GNU Radio question

Dev question `q_9cb7ec1c4fb1625e` (gnuradio_usage, intermediate): "How can you use the 'Variable'
block in GRC to control the frequency of a 'Signal Source' dynamically?"

```json
{
  "question_id": "q_9cb7ec1c4fb1625e",
  "split": "dev",
  "required_points": [
    {"id": "P1", "text": "<the point that carries the answer>", "weight": 2},
    {"id": "P2", "text": "<second point the answer must make>", "weight": 1},
    {"id": "P3", "text": "<third point the answer must make>", "weight": 1}
  ],
  "disqualifying_errors": [
    {"id": "D1", "text": "<error that makes the answer wrong>"}
  ],
  "numeric_tolerance": null,
  "sources": ["<GNU Radio block YAML file or documentation page, with section>"],
  "review": {"status": "pending-team-review", "reviewers": [], "notes": ""}
}
```

As a row of the sheet:

| Column | Cell |
|---|---|
| `required_points` | `P1 (w=2): <the point that carries the answer>`<br>`P2: <second point the answer must make>`<br>`P3: <third point the answer must make>` |
| `disqualifying_errors` | `D1: <error that makes the answer wrong>` |
| `numeric`, `numeric_tolerance`, `calc_status` | empty |
| `sources` | `<GNU Radio block YAML file or documentation page, with section>` |

What the checkers do here:

- open the block's YAML file and its documentation page;
- check that every point is true for GNU Radio 3.10;
- check that no point requires one particular way of doing it when several are valid;
- check that D1 is a real error and not a matter of style.

Invented block names are already caught by the judge's hallucination check, so D1 should not
repeat that.

## Checklist for the writer

- [ ] The question is answerable as stated; otherwise it is marked `underspecified`.
- [ ] Each point is one checkable idea, stated as a fact, with a weight only where it matters.
- [ ] The disqualifying errors are real errors, not style. There are none if none apply.
- [ ] Numbers have `numeric` and `numeric_tolerance`, with a tolerance that fits the question.
- [ ] A decibel number has its tolerance in dB in `abs_tolerance`, with no `tolerance` and no
      `tolerance_db`.
- [ ] Calculation values are confirmed by two blind recomputations, by agent runs other than you.
- [ ] Neither checker is you.
- [ ] The tolerance was set from the question before the key was opened. Anything taken from the
      key is marked `proposed by script (Claude)`.
- [ ] The source is named, and a GNU Radio fact is quoted from a local file as
      `<path> :: "<exact quote>"`.
- [ ] Nothing came from a teacher answer or from another model.

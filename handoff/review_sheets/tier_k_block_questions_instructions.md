# Tier-K GNU Radio block questions: wording review

Implements eval plan Table 7 ("wording review of the generated GNU Radio questions") and §5.3 (an
underspecified question is rewritten, or removed and counted), under the review protocol
([review_protocol.md](../../docs/guidelines/review_protocol.md)): two review agents, each alone, and
the expert agent for every disagreement.

`tier_k_block_questions.csv` holds the 100 questions of the reference set (`./gb blockq`). Each asks
for one number: the k-th output item of a GNU Radio 3.10 block for a stated setting and input, or how
many items the block produces. The key is GNU Radio's own output (two identical runs), and an
independent model of the block's documented behaviour gives the same number. The rule cascade scores
the answers, so these questions have no rubric. The review checks **the wording**: whether a careful
GNU Radio user who reads only the question reaches exactly the key.

## For each question

1. **Solve it blind.** Your input copy has no `answer` and no `evidence` column. Do not open
   `items/tier_k/test/block_answers.jsonl` or `reports/tier_k_block_questions.md` until step 2 is done.
   - Build the flowgraph the question describes, from the wording alone: a Vector Source holding the
     stated input, the block with the stated parameters, and a Vector Sink or File Sink.
   - Run it with GNU Radio 3.10.12 (`/usr/bin/python3` in WSL).
   - Read the item or count the question asks for.
   - Record your value as `blind_value`, and the script and its output as evidence.
   - Where the wording allows two readings, run both and record both values.
2. **Compare.** Now open the key. The coordinator scores your `blind_value` against the key with the
   rule cascade, the same code that scores models.
   - A value that misses the key means the wording or the key is wrong. Find out which, and say so.
   - A question that only one of two readings answers correctly needs an edit.
3. **Check the wording.**
   - The question states everything the answer depends on:
     - the block;
     - every parameter that matters;
     - the input values and their order;
     - which output port and which item is meant, with the counting base (from 0 or from 1).
   - The answer does not hinge on a detail a developer could not know, such as a filter's start-up
     transient.
   - A number with no exact decimal form tells the reader the precision it wants, or the scorer's
     tolerance covers any reasonable rounding. The key's `reference` gives that tolerance.
4. **Decide.**

| Value | Meaning |
|---|---|
| `approve` | a careful reader reaches the key from the wording alone |
| `approve with edits` | right after the edits you list; one edit per line, as `question: <the change> -- <why>` |
| `reject` | the question cannot be fixed by a wording edit (a wrong key, or behaviour that depends on an unstated implementation detail); say why |

Write your notes as `note: <text>` lines: the blind value under each reading, the run you did, and
anything else worth keeping.

## Output fields

One JSON object per question, written to your output file as soon as the question is done:

- `sheet`: `"block_questions"`;
- `key`: `{"question_id": ...}`;
- `value`: one of the three values;
- `blind_value`: the number you got, or `""` with the reason in `notes`;
- `edits`: a list of edit lines;
- `notes`: your `note:` lines;
- `evidence`: the commands you ran with their printed output, and quoted lines of the block's YAML or
  source.

## After both reviews

The decision rules of the review protocol apply:

- the same verdict is final, unless both reviewers asked for edits;
- two lists of edits are merged by the expert agent;
- a disagreement is decided by the expert agent.

A wording edit changes the question's template in the generator (`tier_k/blockgen.py`), so it applies
to every question of that template. The reference set is then regenerated with the same seed, and its
answers are checked again by two identical runs. A rejected question is removed, counted and reported
with its reason (eval plan §5.3).

## Allowed values

| Column | Values |
|---|---|
| `reviewer_1_wording`, `reviewer_2_wording`, `final_decision` | `approve`, `approve with edits`, `reject` |
| `reviewer_1_note`, `reviewer_2_note` | the edit lines and `note:` lines |

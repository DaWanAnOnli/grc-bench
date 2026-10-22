# Tier-K calculation answers: blind recomputation

Eval plan §5.2: calculation answers are computed by a script from the question's numbers, and a team
member recomputes each one **without seeing the script's answer**. The script was drafted by Claude
(Anthropic), so every answer is "pending blind team recomputation" (role table, Table 5).
The script gives a value only where it computes one from numbers the question states. A question
whose answer is an expression or an example has no script value: its reference is written
separately (`reference_route` in the key).

1. Copy `tier_k_calc_blind_recompute.csv` (25 questions) to
   `tier_k_calc_blind_recompute_<your initials>.csv` and fill in your copy only: the original is a
   generated template. Until you have finished, do **not** open `tier_k_calc_key.csv`,
   `items/tier_k/test/calc_answers.jsonl`, or any file under `src/grcbench/tier_k/` or `tests/`:
   they hold the script's answers.
2. For each question, fill in `annotator` (your name), `value` and `unit`. Use only the numbers the
   question states plus standard constants. If the question asks for several numbers, give them in
   the question's order, separated by `;` (same for `unit`). If the answer is an expression or an
   example rather than a number, write it in `notes` and leave `value` empty. Leave out the
   explanations the question asks for: those go into the rubric, which is written separately.
3. In `notes`, write any assumption you had to make and the source you checked (textbook, standard,
   PySDR or the GNU Radio documentation). If the question does not state everything needed for a
   unique answer, do not guess: write `underspecified:`, the missing assumption and the answer under
   each option, and leave `value` empty. This includes a term with two definitions in use (in a
   textbook, a standard, PySDR or the GNU Radio documentation) that give different answers.
4. Then open `tier_k_calc_key.csv` and compare. A value agrees when it lies within the key's relative
   `tolerance` (decibel values: within `tolerance_db` dB). The `specification` column says how the
   script classed the question:
   - `determined`: one value;
   - `underspecified`: no value; `missing_assumptions` names what the question leaves open, and the
     values under each named option are under `readings` (they are not answers);
   - `team reference`: no script value (the answer is not a number computed from the question).
   The `tolerance_note` says only what the tolerance admits. The errors that make an answer wrong
   belong to the rubric, which the team writes.
5. The Qoherent expert decides every disagreement between the script and a recomputation, including
   one about whether a question is underspecified. An underspecified question is rewritten, or
   removed and counted (eval plan §5.2, §5.3). Record in your copy's `notes` who decided and what.
6. The team then writes the rubric's `numeric` and `numeric_tolerance` from the confirmed value
   (`docs/guidelines/rubric_writing.md`). The script's `numeric_proposal` in `calc_answers.jsonl` is
   only a starting point.

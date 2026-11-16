# Tier-K calculation answers: blind recomputation

Eval plan §5.2: calculation answers are computed by a script from the question's numbers, and each
answer is recomputed **without seeing the script's answer**. Under the review protocol
([review_protocol.md](../../docs/guidelines/review_protocol.md)), two recomputer agents (`R1`, `R2`) each
recompute every answer alone, and the expert agent decides every disagreement. The script was drafted
by Claude (Anthropic), so every answer is "pending blind team recomputation" until both recomputations
agree with it, or the expert has decided. The script gives a value only where it computes one from
numbers the question states. A question whose answer is an expression or an example has no script
value: its reference is written by the rubric writer (`reference_route` in the key).

1. Work from your input copy of `tier_k_calc_blind_recompute.csv` (25 questions). Until every one of
   your values is recorded, do **not** open `tier_k_calc_key.csv`, `items/tier_k/test/calc_answers.jsonl`,
   or any file under `src/grcbench/tier_k/` or `tests/`: they hold the script's answers.
2. For each question, record `value` and `unit`. Use only the numbers the question states plus
   standard constants. **Compute with your own code and run it**; the code and its printed output are
   your evidence.
   - If the question asks for several numbers, give them in the question's order, separated by `;`
     (same for `unit`).
   - If the answer is an expression or an example rather than a number, write it in `notes` and leave
     `value` empty.
   - Leave out the explanations the question asks for. Those go into the rubric, which is written
     separately.
3. In `notes`, write any assumption you had to make and the source you checked: a textbook, a
   standard, PySDR or the GNU Radio documentation.
   - If the question does not state everything needed for a unique answer, do not guess. Write
     `underspecified:`, the missing assumption and the answer under each option, and leave `value`
     empty.
   - This includes a term with two definitions in use (in a textbook, a standard, PySDR or the GNU
     Radio documentation) that give different answers.
4. **Comparison.** The coordinator scores each value against the key with the rule cascade
   (`./gb review`). A value agrees when it gets full credit against the key's relative `tolerance`
   (decibel values: within `tolerance_db` dB). The `specification` column says how the script classed
   the question:
   - `determined`: one value;
   - `underspecified`: no value. `missing_assumptions` names what the question leaves open, and the
     values under each named option are under `readings` (they are not answers);
   - `team reference`: no script value (the answer is not a number computed from the question).

   An answer is confirmed when both recomputations reach what the key says: the same value, or both
   `underspecified`, or both no value for a `team reference` question.
5. **The expert agent** decides every other case, including whether a question is underspecified.
   - It reads both recomputations and the key, and re-derives the answer with its own code.
   - Its decision says whether the key's value stands, which value replaces it, or that the question
     is underspecified.
   - An underspecified question is rewritten, or removed and counted (eval plan §5.2, §5.3).
6. The rubric writer then writes the rubric's `numeric` and `numeric_tolerance` from the confirmed
   value (`docs/guidelines/rubric_writing.md`). The script's `numeric_proposal` in `calc_answers.jsonl`
   is only a starting point.

## Output fields

One JSON object per question, written to your output file as soon as the question is done:

- `sheet`: `"calc"`;
- `key`: `{"question_id": ...}`;
- `value`: the number(s), `;`-separated, or `""`;
- `unit`: the unit(s), `;`-separated;
- `notes`: the assumptions, an `underspecified:` line, or the expression;
- `evidence`: your code and what it printed.

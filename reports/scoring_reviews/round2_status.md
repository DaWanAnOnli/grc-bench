# Scoring code and statistics: second review

The rule cascade (`rules.py`), the judge template (`judge_prompt.py`) and the statistics package (`stats/`) had a second review on 18 Oct, two lenses each (correctness, methodology; the judge had one combined review). The full findings are in `round2_<part>_<lens>.json` in this folder.

**All 6 high-severity findings are resolved (6 of 6).** The medium and low findings marked open were not re-checked after the 18 Oct fixes; they are listed for the scoring freeze at G2, when the rule cascade and the judge template are pre-registered. Checks after the changes: `./gb test` (Tier-K rules, judge and statistics tests), `./gb score-k --selftest` (85 of 85) and `./gb stats selfcheck` (36 of 36).

## rules (`src/grcbench/tier_k/rules.py`)

| id | severity | status | finding |
|---|---|---|---|
| F1 | high | fixed in the 18 Oct pass; checked 22 Oct | In the numeric path, a correction written after the answer marker is ignored, and the rules decide the score instead of passing it to the judge. *Resolution:* a revision after the answer marker sends the answer to the judge (reason revised_after_answer); all 7 reviewer probes re-run. |
| F2 | high | fixed in the 18 Oct pass; checked 22 Oct | The symbolic path (exact and set) accepts hedged or revised answers at full credit. *Resolution:* a hedged parenthetical or a revision after a symbolic answer sends it to the judge (hedged_parenthetical, revised_after_answer); all 11 reviewer probes re-run. |
| R1 | high | fixed 22 Oct | The several-numbers rule ('a value followed by a conclusion word before the candidate is a step') scores an answer stated first on a derived value of the same dimension. *Resolution:* the first value after an explicit marker or in a box is the stated answer, so a later derived value with another credit sends the answer to the judge; without a marker the documented 'final stated answer' rule still applies (known limit, rules.py step 2); tests test_a_consequence_after_the_stated_answer_goes_to_the_judge. |
| R2 | high | fixed 22 Oct | _MARKER treats any prose 'answer is' as a final-answer marker equal to 'Final answer:', and the last marker wins. *Resolution:* a prose marker ('the answer is') counts only where no explicit marker exists; tests test_a_prose_marker_never_replaces_an_explicit_one. |
| F3 | medium | open: not re-checked; for the G2 scoring freeze | The colon is optional in the "final answer" pattern of _MARKER, so ordinary prose containing the phrase "final answer" counts as an answer marker. |
| F4 | medium | open: not re-checked; for the G2 scoring freeze | For several-part references, score()["credit"] (also the per-row credit in --batch output) is the mean over the parts even when a part has a unit or order-of-magnitude error. |
| F5 | medium | open: not re-checked; for the G2 scoring freeze | The rule-only dev checks (proposal 5.1: "Development-set checks between training runs are scored by the rule cascade only, without the judge") will mostly measure answer style rather than accuracy. |
| R3 | medium | open: not re-checked; for the G2 scoring freeze | In multi-part references whose parts share a dimension, when the label scoping fails (both labels in one sentence, written value-first), each part falls back to the whole-answer candidate rule. |
| R4 | medium | open: not re-checked; for the G2 scoring freeze | A unit word the parser does not know, written after a number with no '/' or 'per' after it (msec, usec, nsec, microsec, Kbit, Msamp, Mcps), is silently dropped. |
| R5 | medium | open: not re-checked; for the G2 scoring freeze | normalize() does not turn the LaTeX non-breaking space '~' into a space, so in '30~\text{kHz}' the unit is lost. |
| R6 | medium | open: not re-checked; for the G2 scoring freeze | For an exact block-id target, an answer that names the right block in words ('Throttle block') and puts only parameter names in code or snake_case gets a decided 0 (wrong_identifier). |
| R7 | medium | open: not re-checked; for the G2 scoring freeze | The option-letter path reads a sentence-initial article 'A' as option A. |
| F6 | low | open: not re-checked; for the G2 scoring freeze | The rubric-writing guideline tells writers to give decibel tolerances in the `tolerance` field, but rules.py refuses that. |
| F7 | low | open: not re-checked; for the G2 scoring freeze | Credit for the same physical answer depends on which unit the rubric writer chose for the reference, which conflicts with the proposal's single "tiered relative-error credit" rule. |
| F8 | low | open: not re-checked; for the G2 scoring freeze | Comparator and direction words are not handled. |
| F9 | low | open: not re-checked; for the G2 scoring freeze | For an open-ended rubric (no numeric target), a truncated or empty answer is labelled method "symbolic". |
| R10 | low | open: not re-checked; for the G2 scoring freeze | Vpp, Vpk, Vp and Vrms all have scale 1 V, so a correct value in another voltage convention gets a decided 0 (outside_tolerance) instead of a conversion or the judge. |
| R11 | low | open: not re-checked; for the G2 scoring freeze | For multi-part references, score() and the --batch output rows report the raw mean as 'credit', even when a part has a unit or order-of-magnitude error that policy P9 says zeroes the whole answer. |
| R8 | low | open: not re-checked; for the G2 scoring freeze | An unevaluated expression given as the final answer is scored on its operand and decided (0 for a correct answer). |
| R9 | low | open: not re-checked; for the G2 scoring freeze | Relative-dB families each have their own pseudo-dimension, so a plain 'dB' answer against a dBc, dBi or dBc/Hz reference gets a decided 0 (unit_mismatch), even when the text says 'relative to the carrier'. |

## judge (`src/grcbench/tier_k/judge_prompt.py`)

| id | severity | status | finding |
|---|---|---|---|
| F1 | high | fixed 22 Oct | The judge route cannot settle the readings the rule cascade hands to it. *Resolution:* a third route, 'human': when the rules hold two readings of a number with different credit (literal_credit, alt_credit), a person scores the answer and the judge is not asked; tests test_two_readings_go_to_a_human and test_one_part_with_two_readings_sends_a_several_number_answer_to_a_human; docs/guidelines/judge_calibration.md excludes these answers from the calibration draw. |
| F2 | medium | open: not re-checked; for the G2 scoring freeze | The tolerance tiers and the DU rule given to the judge for numbers stated in points are not the rule cascade's. |
| F3 | medium | open: not re-checked; for the G2 scoring freeze | The approved-rubric gate applies only to build_messages. |
| F4 | medium | open: not re-checked; for the G2 scoring freeze | The hallucination check ("a judge's flag never zeroes a score unchecked") marks real names "absent" and so zeroes correct answers. |
| F5 | low | open: not re-checked; for the G2 scoring freeze | The pre-registration snapshot records nothing about how the judge is called: temperature, top_p, max_tokens, seed, JSON/structured-output mode, reasoning on or off. |
| F6 | low | open: not re-checked; for the G2 scoring freeze | parse_verdict breaks its contract ("Anything else raises VerdictError"): a deeply nested reply raises RecursionError. |
| F7 | low | open: not re-checked; for the G2 scoring freeze | The documentation contradicts the template on what the judge sees. |
| F8 | low | open: not re-checked; for the G2 scoring freeze | The judge can see which model wrote an answer when the answer names itself. |

## stats (`src/grcbench/stats/`)

| id | severity | status | finding |
|---|---|---|---|
| M1 | high | fixed in the 18 Oct pass; checked 22 Oct | The default for the pre-registered H1c test (the proposal's headline H1) mixes up the tier with the choice of test. *Resolution:* both tiers use the same test for H1c's significant-pairs count (PROPOSAL H1c-test, default Wilcoxon); test_size_matched_separation. |
| M2 | medium | fixed in the 18 Oct pass; checked 22 Oct | The proposed h1c_decision rule ('Tier F ahead in >= 95% of size-matched K subsamples') uses the share of without-replacement Tier-K subsamples as if it were a p-value. *Resolution:* h1c_decision decides on bootstrap intervals of F minus K; the subsample shares are descriptive only; test_h1a_and_h1c_decision_rules. |
| M3 | medium | fixed in the 18 Oct pass; checked 22 Oct | The H1c helpers (separation, size_matched_separation) refuse a panel whose configurations have different numbers of samples per item. *Resolution:* separation and size_matched_separation take model -> item -> samples with each model's own n; selfcheck 'H1c, mixed-n panel'. |
| M4 | medium | fixed in the 18 Oct pass; checked 22 Oct | agreement.judge_calibration()['passes'] and the judge part of h1a_decision apply only one kappa: the rubric-line gate. *Resolution:* judge_calibration computes both gates (rubric lines, answer score bands) and passes only when both pass; h1a_decision refuses a one-gate result; test_judge_calibration_gate_is_kappa_against_the_adjudicated_reference. |
| WS7-C1 | medium | fixed in the 18 Oct pass; checked 22 Oct | The H1c helpers (separation, size_matched_separation) cannot run on the panel the plan declares, because that panel mixes sample counts (n = 5, 3 and 1). *Resolution:* as M3. |
| WS7-C2 | medium | fixed in the 18 Oct pass; checked 22 Oct | judge_calibration()['passes'], and h1a_decision's judge part, test only one condition: the unweighted kappa on rubric lines. *Resolution:* as M4. |
| WS7-C3 | medium | open: not re-checked; for the G2 scoring freeze | In clustered mode, compare_paired's primary effect size uses an odds-ratio interval that assumes independent items: it is an item-level Clopper-Pearson interval on a_only/b_only. |
| M5 | low | fixed in the 18 Oct pass; checked 22 Oct | compare_paired prints Table 8's MDE, which assumes independent items, even when its primary test is the cluster sign-flip test that the module proposes for confirmatory comparisons. *Resolution:* compare_paired prints a design-effect MDE next to a clustered test; selfcheck 'design-effect MDE'. |
| M6 | low | fixed in the 18 Oct pass; checked 22 Oct | The proposed H1a decision rule passes the bootstrap part when the MEDIAN Kendall tau between resample pairs is at least 0.8. *Resolution:* the H1a bootstrap rule is a named PROPOSAL (H1A_BOOT_RULE, H1A_BOOT_SHARE). |
| M7 | low | fixed in the 18 Oct pass; checked 22 Oct | The module adds about eight PROPOSAL decisions that change or extend the approved pre-registered definitions, and they are recorded only in code docstrings. *Resolution:* every analysis PROPOSAL is listed once in stats/proposals.py (`./gb stats proposals`); selfcheck 'PROPOSALS registry'. |
| WS7-C4 | low | open: not re-checked; for the G2 scoring freeze | With a single cluster, the bootstrap intervals collapse to zero width at the point estimate, and judge_passes accepts such an interval. |
| WS7-C5 | low | open: not re-checked; for the G2 scoring freeze | Weighted Cohen's kappa on string labels without explicit categories silently orders the categories alphabetically, which gives a wrong weighted kappa for the rubric labels this project uses. |
| WS7-C6 | low | open: not re-checked; for the G2 scoring freeze | `./gb stats mde` ends in a raw Python traceback on out-of-range arguments instead of a clean argparse error. |

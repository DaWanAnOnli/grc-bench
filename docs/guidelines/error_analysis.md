# Error analysis: labelling failures with a fixed list of error types

Implements eval plan §9 ("Error analysis. Two annotators label failures with a set list of error
types ... We report their agreement (κ)"), with §8.2 (Table 7: about 17 person-hours, two team
members), §4.3 (the validity ladder V1–V5) and the 16 Oct decision in
[the approved plan](../implementation_plan.md) (Tier F as "one lookup + one edit" in GRC-Agent's
harness).

## What this is for

Scores say how often a model fails. Error labels say **why**. Two team members label the same
failures independently, using the fixed list of error types below. We report how often each type
occurs, per tier and per configuration, and how well the two annotators agree (Cohen's κ).

The labels describe model failures. A failure caused by a broken item is not a model error. Flag
it (`item_defect_suspected`) and it goes to the item review instead.

## What gets labelled

- **Tiers:** failed **Tier-F** samples and failed **Tier-A** runs.
  - **Tier K is left out. This is our proposal for the pre-registration, not a rule of the plan.**
    Eval plan §9 does not limit the error analysis to a tier. We propose to leave Tier K out
    because its failures are already broken down by code: per rubric point, per number (wrong
    unit, order of magnitude) and per invented GNU Radio name (`check_entities`, the proposal's
    "hallucinated GNU Radio entities"). Most of the list (port types, connections, runtime errors)
    does not apply to a text answer.
  - If the team decides otherwise at G2, Tier-K failures are labelled with the codes that fit a
    text answer (for example `hallucinated_block`, `parameter_error`, `rate_mismatch`,
    `wrong_behaviour`, `other`), and the sample and hours below change.
- **Configurations:** the headline configurations (eval plan "Key terms"):
  - each model's base model;
  - its final fine-tuned model;
  - the final model's 4-bit file.
- **Sample:** a seeded random sample of failures, sized to the budget (see "Time" below). Our
  proposal, to be fixed in the pre-registration:
  - **120 Tier-F failures, 20 per configuration.** Use sample 1 of the 5 per item, so each item
    appears at most once per configuration.
  - **42 Tier-A failures, 7 per configuration,** from the failed runs of run 1, so each task appears
    at most once per configuration.
  - If a configuration has fewer failures than its share, take all of them. Give the rest to the
    other configurations of the same tier, in equal parts. Record the final counts.
  - Both are drawn with the seed `grc-bench-error-analysis-v1`.
- **Blinding:** annotators do not see which configuration produced a failure. The sheet shows an
  opaque `failure_id`, and only the coordinator holds the key
  ([`blinding_key.csv`](../../handoff/review_sheets/templates/blinding_key.csv)).

**Time.** Eval plan Table 7 gives about 17 person-hours for two annotators, so about 8.5 hours
each. Most Tier-F failures take 1–2 minutes, because the ladder already says where they failed. A
Tier-A run takes about 5 minutes. The sample above is derived from these rates:

| Step, per annotator | Hours |
|---|---|
| Practice: 10 dev failures, then the discussion | 0.5 |
| 120 Tier-F failures at about 1.5 minutes | 3.0 |
| 42 Tier-A failures at about 5 minutes | 3.5 |
| Disagreements: if about a quarter of the 162 differ, re-reading about 40 and writing your reason for the expert, at 2 minutes | 1.5 |
| **Total** | **8.5** |

Two annotators: about 17 person-hours, as in Table 7. Two kinds of time are not in that line. The
coordinator records both, and reports them as a deviation in the 7 December report:

- the coordinator's preparation (sample, sheet, `auto_facts`). The coordinator holds the blinding
  key, so does not label;
- the Qoherent expert's decisions on the disagreements ("Procedure", step 5).

If the practice round shows that labelling is slower, the coordinator shrinks both tiers in
proportion before labelling starts, and records the new counts. A larger sample than this exceeds
Table 7. Record it as a deviation in the 7 December report (eval plan §9, "Pre-registration").

## What you see for each failure

- **The item:** the instruction; for Tier F the start graph and the reference; for Tier A the start
  state and the acceptance checks.
- **The model's output:**
  - Tier F: the `change_graph` call, any `query_knowledge` call with its results, and the final
    graph.
  - Tier A: the whole trace (every tool call with its raw arguments, every result, the scripted
    user's replies), and the final saved graph.
- **The ladder result:** the first failed rung and its error messages (`./gb ladder` output), and
  the oracle checks that failed.
- **`auto_facts`:** facts the logs already record, for example `V2: unknown block <id>`,
  `edit accepted: no`, `lookup: none`, `cap reached: requests`. They are facts, not labels. Use
  them, and label what they do not tell you.

## Three label columns

| Column | Tiers | What it answers | Values |
|---|---|---|---|
| `primary` | F and A | **what is wrong** with the model's output (the final graph) | one code from "Output errors", or `no_output`, `item_defect_suspected`, `other`. Tier F only: `none` when the only fault is a malformed call (see `malformed_call`) |
| `process` | F and A | **how the attempt went wrong** | Tier A: one code from "Tier-A process errors", or `none` / `other`. Tier F: `malformed_call`, `rejected_batch`, `none` or `other` |
| `lookup` | F only | **did the lookup help where it should have?** | `lookup_not_used`, `wrong_lookup` or `none`; leave it empty for Tier A |

Put any further codes that also apply in `secondary` (separated by `;`), and quote your evidence in
`evidence`: the error message, the call, or the line of the trace.

### Choosing the primary code when several apply

Pick the **first** code in this order that applies. Earlier codes either stop the ladder earlier or
are more specific.

1. `no_output`
2. `hallucinated_block`
3. `wrong_port_type`
4. `missing_connection`
5. `rate_mismatch`
6. `parameter_error`
7. `runtime_error`
8. `wrong_behaviour`
9. `other`

`item_defect_suspected` overrides this order. If you believe the item, not the model, is wrong,
use it whatever else applies.

### Choosing the process code

- Label a process error **only if it contributed to the failure**, that is, the run never
  recovered from it. A malformed call that the agent corrected on the next turn is not the reason
  the run failed. It is counted automatically in the tool-call metrics anyway.
- If several contributed, pick the one that happened **first** in the trace. Put the others in
  `secondary`.

## Output errors (Tier F and Tier A)

These are the error types of eval plan §9.

The examples below are made up to show each type. They are not taken from test items. Block ids are
real GNU Radio 3.10 ids unless the example says they are invented.

### `hallucinated_block`: hallucinated block

- **Definition.** The output uses a block id that does not exist in the pinned GNU Radio 3.10.12
  catalog.
- **Decision rule.**
  - Use it when V2 fails with an unknown block (the ladder lists `hallucinated_blocks`).
  - Under the Tier-F format, also use it when GRC-Agent rejects the `change_graph` batch because of
    an unknown block id.
  - In Tier A, use it when the final graph lacks something because every attempt to add it used an
    invented id.
  - Do **not** use it for an instance name that is not in the graph, for example `filter_0` when the
    graph calls it `lpf`. That is not an invented block type: label it by what the edit was meant
    to do (`parameter_error` for a parameter update, `missing_connection` for a connection). In
    Tier A, also consider `edit_without_inspect`.
  - Do not use it for an invented parameter name: that is `parameter_error`.
  - Do not use it when the invalid id is a repair task's injected fault (`fault` in `task.json`)
    that the model left in place or copied from the start graph. The model did not invent it. Use
    `other` and write `fault not repaired` in `evidence`. The hallucinated-block rate leaves it out
    too (Tier-F fix plan, T9).
- **Example.** The edit adds a block with id `blocks_low_pass_filter`. No such block exists; the
  real id is `low_pass_filter`. GRC-Agent rejects the batch.
  - Tier F: `primary` = `hallucinated_block`, `process` = `rejected_batch`.

### `wrong_port_type`: wrong port type

- **Definition.** A stream has the wrong data type (complex, float, int, short, byte) or the wrong
  vector length at a port. This also covers an output written with the wrong item type.
- **Decision rule.**
  - Use it when V3 (or GRC-Agent's validation of the edit) reports a type or size mismatch between
    connected ports.
  - Also use it when the graph runs but the observed output has the wrong item type, so the oracle
    reads garbage.
- **Example.** A Signal Source (`analog_sig_source_x`) is set to `complex` and connected to a Low
  Pass Filter (`low_pass_filter`) of type `fir_filter_fff`, which takes float. V3 reports
  `Source IO type "complex" does not match sink IO type "float"`.

### `missing_connection`: missing connection

- **Definition.** A connection the task needs is missing or wrong:
  - a required port is left unconnected;
  - the chain is broken;
  - a required block is bypassed (wired around);
  - an input is fed from two sources.
- **Decision rule.** Use it when the fault is in the edges, not in the blocks or their parameters.
  Typical evidence:
  - V3 reports a port that is not connected, or an input with more than one connection;
  - or the graph runs but the File Sink is fed from a point before the block the task needed.
- **Example.** The edit adds a Low Pass Filter and connects the source to it, but never connects
  the filter's output to the File Sink. V3 reports `Port is not connected` for the filter's output
  and for the File Sink's input.

### `rate_mismatch`: rate mismatch

- **Definition.** Any error in sample rates or rate changes:
  - a block configured for another rate than the stream it receives;
  - a decimation, interpolation or resampling ratio that gives the wrong output rate;
  - the wrong flowgraph sample rate.
- **Decision rule.** Use it whenever the wrong value is a rate-related parameter. Rate-related
  parameters are:
  - `samp_rate` and other sample-rate parameters;
  - `decim` and `interp`;
  - resampler ratios.

  This takes priority over `parameter_error`. The evidence is usually a V5 check on frequency or
  rate that is off by the rate factor.
- **Example.** A Rational Resampler (`rational_resampler_xxx`) with `interp` 1 and `decim` 3 is
  added, but the Low Pass Filter after it still uses `samp_rate` instead of `samp_rate/3`. Its taps
  are designed for three times the real rate, so its real cutoff is a third of the intended one.
  V5 finds the wanted signal attenuated.

### `parameter_error`: parameter error

- **Definition.** A block parameter that is not about rates has a wrong value, an invalid value, a
  name that does not exist, or refers to an undefined variable.
- **Decision rule.** Use it when:
  - the block types and connections are right but a value is wrong (a V5 check fails on a level, a
    frequency or a cutoff);
  - or V2/V3 reports an invalid value, an unknown parameter or an undefined name.
- **Example.** The instruction asks for 16 dB of amplitude gain. The Multiply Const
  (`blocks_multiply_const_vxx`) is set to `16` instead of `10**(16/20)` (about 6.31). V5 finds the
  amplitude 2.5 times too high.

### `runtime_error`: runtime error

- **Definition.** The graph validates but fails later: at V4 (the run), or at V3's code generation
  (its error starts with "code generation:"). Examples:
  - an exception during the run;
  - not stopping cleanly within the grace period;
  - too few samples reaching an observation sink (liveness);
  - a code-generation failure that no earlier code explains.
- **Decision rule.** Use it only when no more specific code explains the failure. A parameter that
  makes the run crash is `parameter_error`.
  - **Run length:** if the run stops early because the model limited it (for example with a Head
    block) although the instruction said the flowgraph keeps running until stopped, use
    `runtime_error`.
  - If the instruction did **not** say so, the item hid a requirement. Use
    `item_defect_suspected`.
- **Example.** The edit adds a File Source (`blocks_file_source`) that reads `input.dat`, which does
  not exist in the run folder. V4 fails with a "No such file" error.

### `wrong_behaviour`: wrong behaviour

- **Definition.** The graph runs (V1–V4 pass) but does the wrong thing (V5 fails). No more specific
  code explains it. Examples:
  - a wrong choice of block or design;
  - a filter that does not meet the specification;
  - the output observed at the wrong point.
- **Decision rule.** V5 is the first failed rung, and no specific wrong parameter, rate or
  connection explains it.
  - For Tier-A knowledge tasks, a wrong answer is `wrong_behaviour`.
  - For Tier-A refusal tasks, acting when the task called for a refusal is `wrong_behaviour`.
- **Example.** The task asks to remove high-frequency noise. The model inserts a High Pass Filter
  (`high_pass_filter`) instead of a low-pass one. The graph runs, and V5 finds the tone removed and
  the noise kept.

## Tier-A process errors

These are the extra Tier-A types of eval plan §9.

### `malformed_call`: malformed call

- **Definition.** A tool call whose raw arguments are not valid JSON or do not match the tool's
  schema. Examples: a missing required field, a wrong type, an unknown field, or a string that does
  not match the required pattern. The raw arguments are logged before GRC-Agent repairs the JSON.
- **Decision rule.** Use it when the run never got a correct call through for an action it needed.
  Tier F uses the same code with its own rule (see "`malformed_call` in Tier F" below).
- **Example.** `change_graph` with `add_connections: ["src -> sink"]`. The schema requires the form
  `'src_block:src_port->dst_block:dst_port'`. The agent repeats the same form and never makes the
  connection.

### `unknown_tool`: unknown tool

- **Definition.** A call to a tool name that is not one of GRC-Agent's 22 tools at commit
  `6c98128`. The 22 tools are:
  - graph: `inspect_graph`, `change_graph`, `generate_python`, `save_graph`, `save_block`;
  - running: `run_flowgraph`, `get_run_log`;
  - lookup: `query_knowledge`;
  - files: `read_file`, `write_file`, `edit_file`, `file_info`, `find_files`, `search_files`,
    `list_directory`, `create_directory`;
  - shell: `run_command`, `start_command`, `check_command`, `stop_command`;
  - web (offline in the benchmark): `duckduckgo_search`, `web_fetch`.
- **Decision rule.** Use it when the failure follows from calling a tool that does not exist
  instead of the real one.
- **Example.** The agent calls `add_block(...)` and later `save_flowgraph(...)`. Neither exists; the
  real tools are `change_graph` and `save_graph`. The graph is never changed.

### `edit_without_inspect`: editing without inspecting

- **Definition.** The agent changes the graph before reading its current state. Reading means
  `inspect_graph`, or `read_file` on the open `.grc`. The edit relies on block names, parameters or
  connections the agent never saw.
- **Decision rule.** Use it when the first graph-changing call (`change_graph`) comes before any
  call that reads the graph, and that edit fails or does the wrong thing because of what the agent
  assumed. It does not apply to Tier F, where the inspection is already in the conversation.
- **Example.** The agent's first call is `change_graph` updating `low_pass_filter_0`. The open graph
  names the filter `lpf`, so the call fails because no block of that name exists. The agent then
  adds a second filter instead of fixing the first.

### `giving_up`: giving up

- **Definition.** The agent ends the episode without reaching the goal, while it still has requests
  and context left. It says it cannot or will not continue, or it hands the work back to the user.
- **Decision rule.**
  - Use it when the episode ends with a final answer, no cap was reached, and the answer gives up
    or delegates.
  - Do **not** use it for a correct refusal in a refusal task.
  - Do not use it when the agent claims it succeeded but did not. Label that by the end-state error
    and write `claimed success` in `evidence`.
- **Example.** After one rejected `change_graph`, the agent replies "I could not add the filter;
  please add it manually in GRC", with 30 requests left.

### `looping`: looping

- **Definition.** The agent repeats the same action without progress. This means the same tool with
  the same or trivially different arguments, or a cycle of two calls, with the same result and no
  change to the graph in between.
- **Decision rule.** Use it when the same call or cycle repeats 3 or more times without progress.
  It usually ends at the request cap. The threshold of 3 is our proposal for the pre-registration.
- **Example.** `inspect_graph`, then the same rejected `change_graph`, then `inspect_graph`, and so
  on six times, until the 40-request cap.

## Tier-F errors under the 16 Oct format

Under this format, a Tier-F sample is one `query_knowledge` lookup at most, then one atomic
`change_graph` call, within 3 turns. A `change_graph` call ends in one of four ways, and each has
one rule:

| What happened to the `change_graph` call | `process` | `primary` |
|---|---|---|
| No call within the 3 turns (V1: `no change_graph call`) | `none` | `no_output` |
| Its raw arguments are malformed, or GRC-Agent refused them at argument validation (first failed rung V1) | `malformed_call` | the error in the call's content, or `none` if the content is right |
| GRC-Agent rejected the batch and rolled it back (rejection rung V2 or V3) | `rejected_batch` | the reason for the rejection |
| The call was well formed, the edit was applied, and the final graph fails V2–V5 | `none` | the output error |

A rejected or refused call is **never** `no_output`: the model did produce an edit, and the edit is
what gets labelled.

### `malformed_call` in Tier F: malformed change_graph call (V1)

- **Definition.** The single `change_graph` call fails V1, "a well-formed `change_graph` call
  matching the tool schema" (16 Oct decision). Either:
  - its raw arguments, as the model emitted them, are not valid JSON or do not match the tool's
    schema; GRC-Agent may still have repaired the JSON and applied the edit;
  - or GRC-Agent refused the arguments before applying anything (error codes `invalid_request`,
    `invalid_connection_format` or `args_invalid`).
- **Decision rule.**
  - Set `process` = `malformed_call` whenever the first failed rung is V1 and a `change_graph` call
    was made.
  - For `primary`, judge the **content** of the edit:
    - If GRC-Agent applied the edit after repairing it, the record has the ladder result of the
      final graph. If V2–V5 pass, the content is right: `primary` = `none`. Otherwise label the
      final graph's error as for any applied edit.
    - If GRC-Agent refused the call, read the intended edit from the raw arguments. If it contains
      an error from "Output errors" (an invented block id, an unknown parameter, ...), use that
      code. Otherwise `primary` = `none`.
- **Example.** `add_connections: ["src -> sink"]`. The schema requires
  `'src_block:src_port->dst_block:dst_port'`, so GRC-Agent refuses the call
  (`invalid_connection_format`). The blocks and the intended connection are right: `process` =
  `malformed_call`, `primary` = `none`.

### `rejected_batch`: rejected change_graph batch

- **Definition.** GRC-Agent rejected the single `change_graph` batch and rolled it back. There is no
  retry, so the final graph is the start graph.
- **Decision rule.**
  - Set `process` = `rejected_batch` whenever GRC-Agent rejected the edit at V2 or V3 (the record's
    rejection rung). A call refused at V1 is `malformed_call` instead.
  - Always pair it with the `primary` code for the **reason** of the rejection:
    - an unknown block id: `hallucinated_block`;
    - an unknown parameter or an invalid value: `parameter_error`;
    - a type mismatch: `wrong_port_type`;
    - an unconnected port: `missing_connection`.
  - A batch forced through with `force: true` is not rejected. The invalid graph is applied and
    fails at V3; label it by its cause, with `process` = `none`.
- **Example.** The single `change_graph` sets `cutoff` on a Low Pass Filter. The real parameter is
  `cutoff_freq`, so GRC-Agent rejects the whole batch: `primary` = `parameter_error`, `process` =
  `rejected_batch`.

### `lookup_not_used` and `wrong_lookup`: lookup not used, or wrong lookup

These two codes go in the `lookup` column. Use them only when the primary error is about GNU Radio
knowledge: `hallucinated_block`, `wrong_port_type`, `parameter_error` (a wrong parameter name or
an invalid value), or `rate_mismatch` when a block's rate parameters were misunderstood. For
any other primary error, set `lookup` = `none`.

- **`lookup_not_used`**
  - **Definition:** the knowledge was available, but the model did not use it.
  - **Decision rule:** any of these:
    - the model made no `query_knowledge` call;
    - it made the lookup in the same reply as the edit, so the edit could not use the result;
    - the lookup's results contained the needed block or parameter, and the edit contradicts them.
  - **Example:** the model invents `blocks_low_pass_filter` in its `change_graph` without any
    lookup. Or its lookup returned `low_pass_filter` with its parameters, and the edit still used
    the invented id.
- **`wrong_lookup`**
  - **Definition:** the model looked up, but its query could not surface what it needed.
  - **Decision rule:** a lookup was made in an earlier reply than the edit, and its results do not
    contain the needed block or parameter (the recorded retrieval recall of the reference's block
    ids is 0 for that block). The cause is the query, the domain or `k`.
  - **Example:** the model searches the `docs` domain for "remove noise from my signal". It gets
    tutorial pages, none naming a filter block, and then invents a block id. A `catalog` search for
    "low pass filter" would have found it.

If no single reasonable search could have surfaced the needed block, set `lookup` = `none` and say
so in `evidence`. The diagnostic "one search can surface all of the reference's blocks" helps here.

## Codes outside the plan's list

These codes are not error types of eval plan §9.

- `no_output`, `none` and `other` are still **model failures**. They are counted with the model
  errors, each in its own row of the report. A cut-off answer, for example, "is scored as wrong"
  (eval plan §7.1).
- `item_defect_suspected` is **not** a model error. It is left out of the error counts until the
  item review decides.

| Code | Use when |
|---|---|
| `no_output` | the model produced nothing to grade: the answer is empty or was cut off at the length cap; Tier F: no `change_graph` call was made within the 3 turns; Tier A: the agent never changed or saved anything relevant. A rejected or refused `change_graph` call is **not** `no_output` (see "Tier-F errors" above). For Tier A, still label `process` |
| `none` (Tier F only) | the first failed rung is V1 and the content of the edit is right, so the only fault is the malformed call. `process` is then `malformed_call` |
| `item_defect_suspected` | you believe the item is wrong, not the model. Examples: the oracle rejects a correct answer; a requirement is hidden (for example an unstated run length); the instruction is ambiguous. Explain in `evidence`. The failure goes to the item review (eval plan §5.3) and is excluded from the error counts until it is resolved |
| `other` | nothing on the list fits; describe it in `evidence`. If many failures end up as `other`, the team discusses adding a type before the labels are final |

## Procedure

1. **Prepare.** The coordinator draws the sample, removes configuration names, fills in
   `auto_facts`, and gives each annotator their own copy of
   [`error_analysis.csv`](../../handoff/review_sheets/templates/error_analysis.csv).
2. **Practice.** Both annotators label the same 10 **dev** failures, then compare and discuss.
   Practice labels are not counted.
3. **Label alone.** Each annotator fills in only their own columns (`_1` or `_2`) and does not
   discuss failures until both copies are in.
4. **Agreement.** Agreement is computed on these **independent** labels, before the expert
   decides anything:
   - Cohen's κ for `primary`, overall and per tier;
   - Cohen's κ for `process` (per tier) and for `lookup` (Tier F);
   - the percentage agreement, next to each κ.
5. **Resolve.** Where both annotators gave the same codes, those are the final codes
   (`resolved_by` = `annotators`). Every row where they differ goes to a Qoherent expert, who
   decides (eval plan Table 1: "A Qoherent expert decides every disagreement"):
   - the annotators do not settle it between themselves first;
   - each annotator re-reads the failure and adds a one-line reason to their `evidence` cell. If
     you see that you simply mistyped a code, say so there; the expert still decides;
   - the expert writes the final codes in `final_primary`, `final_process` and `final_lookup`,
     and their initials in `resolved_by`;
   - the coordinator counts how many rows the expert decided.
6. **Report:**
   - the counts of each final code, per tier and per configuration, with the κ values. `no_output`,
     `none` and `other` are model failures, each in its own row;
   - the `item_defect_suspected` cases and what the item review decided.

## The sheet

[`error_analysis.csv`](../../handoff/review_sheets/templates/error_analysis.csv), one row per
failure:

| Column | Filled by | Content |
|---|---|---|
| `failure_id` | coordinator | opaque id, for example `err-001` |
| `tier`, `item_id`, `split` | coordinator | `F` or `A`; the item; `test` or `dev` |
| `first_failed_rung` | coordinator | `V1`–`V5`, or `cap` for a Tier-A cap |
| `auto_facts` | coordinator | facts from the logs (see above) |
| `trace_ref` | coordinator | where the output, trace and ladder result are |
| `primary_1`, `secondary_1`, `process_1`, `lookup_1`, `evidence_1` | annotator 1 | codes and evidence |
| `primary_2`, `secondary_2`, `process_2`, `lookup_2`, `evidence_2` | annotator 2 | codes and evidence |
| `final_primary`, `final_process`, `final_lookup` | coordinator (same codes) or the expert (different codes) | the final codes |
| `resolved_by` | coordinator or the expert | `annotators` (both gave the same codes), or the expert's initials |
| `note` | anyone | anything else |

## All codes at a glance

| Column | Codes |
|---|---|
| `primary` | `hallucinated_block`, `wrong_port_type`, `missing_connection`, `rate_mismatch`, `parameter_error`, `runtime_error`, `wrong_behaviour`, `no_output`, `item_defect_suspected`, `other`; Tier F only: `none` |
| `process` (Tier A) | `malformed_call`, `unknown_tool`, `edit_without_inspect`, `giving_up`, `looping`, `other`, `none` |
| `process` (Tier F) | `malformed_call`, `rejected_batch`, `other`, `none` |
| `lookup` (Tier F only) | `lookup_not_used`, `wrong_lookup`, `none` |

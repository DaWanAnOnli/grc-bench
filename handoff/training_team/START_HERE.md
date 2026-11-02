# For the training team: what GRC-Bench needs you to keep out of training

From the GRC-Bench evaluation team, 22 Oct 2026. Status: draft for team review; not yet sent.
Claude (Anthropic) wrote the tooling and these files. Claude never judges, repairs or generates
training data.

Every rule below protects the benchmark from training data that overlaps with it. Each file in this
folder covers one kind of data. Start with the table, then read the file for the data you work on.

## The files

| File | Covers | What you do |
|---|---|---|
| `README.md` and `exclude_question_ids.txt` | Tier-K questions (the 107 dev questions, all in the validation split of Qoherent cleaned dataset v3) | Exclude the 107 dev questions, and everything built from them, from every training use; keep the whole validation split out of training, as v3 does. |
| `test_twins.tsv` (and `README.md` §2–3) | training questions that overlap the 185 test questions | Remove the test twins' answers where possible; the team decides on the flagged neighbours. |
| this file, below | generated questions about GNU Radio blocks | Any block may be used, including what a block outputs for a given input. Write your own wording and templates. |
| `flowgraph_seed_split.md` | flowgraph and agent tasks built from GNU Radio example files | Never use a test or dev seed file, also not edited or with a fault planted in it. |
| `flowgraph_composed_chains.md` and `flowgraph_reserved_signatures.txt` | flowgraphs written by hand or by a script | Emit a graph only if its signature hashes to train and its digest is not on the reserved list. |
| this file, below | tool outputs inside training sessions | Cut long tool outputs with the same 4K-token function as the benchmark. |
| this file, below | Qoherent's agent traces of 15-18 October (`new_data/`) | Use all 1,853 traces; they were checked against the benchmark. Render each student's own chat template. |

## Qoherent's agent traces (15-18 October)

`project/new_data/` holds 1,253 one-shot (Tier 1) and 600 two-turn (Tier 2) traces, made by
DeepSeek-V4.1-Flash in GRC-Agent's harness. All of them may be used for training. `./gb traces`
checked them against the benchmark (`reports/agent_traces.md`):

- **Same harness.** Their system prompt and tools match the evaluation harness, apart from the file
  path and two cosmetic schema details.
- **No task-level overlap.** No trace has the goal or the start graph of a benchmark task. 38 traces
  end in a one-block graph whose signature cell is reserved (a lone tone or vector source wired to
  null sinks). For agent traces, the task-level check is the rule, and they pass it.
- **No change on your side.** The GNU Radio test questions ask what a block does to a stated input.
  The traces show the block catalog but never ran a flowgraph, so they do not contain those answers.

## Generated questions about GNU Radio blocks

There is no test/train block split. Training data may cover any GNU Radio block: its catalog entry,
its documentation, and what it outputs for a given input (for example from running it). The
benchmark does not depend on which blocks training uses: its GNU Radio questions come from a seed,
and every evaluation can also draw a fresh set with the same composition, so learning a block is
fine and memorising the published questions shows up as a gap between the two scores.

Two rules remain:

- **Never the published test items.** Items carrying the GRC-Bench canary string, and anything built
  from them, stay out of training.
- **Your own wording.** Write your own question wording and templates; do not copy or imitate
  GRC-Bench's. The overlap audit compares training text with the benchmark, and copied templates
  would make every generated question look like an overlap.

Notes on the data (for your judgement; GRC-Bench does not change training data):

- All 37 `run_flowgraph` calls in the traces got "Flowgraph execution is not wired up in this
  environment". A model trained on them may learn that runs are unavailable. In the benchmark,
  `run_flowgraph` works.
- 53 `run_command` calls read Qoherent's own collection folder (other traces' outputs). The collection
  harness had no sandbox; the benchmark's has one.
- The traces never call `save_block`, `write_file`, `create_directory` or `get_run_log`.
- `new_data/export_unsloth.py` audits and converts the traces, but it wraps reasoning in Qwen3
  `<think>` tags. For Nanbeige4.2-3B and North-Mini-Code, render each model's own chat template from
  the master `tier1_sft_trajectories.jsonl` and `tier2_sft_trajectories.jsonl`, which keep the
  reasoning in a separate field. The script's paths point to Qoherent's `output/` folder, so it does
  not run as-is on this copy.
- Cut long tool outputs with the benchmark's 4K-token function (next section), as for any session.

## Tool outputs: use the benchmark's 4K-token cut

The benchmark cuts every tool output to 4,096 tokens before the model sees it (eval plan §4.4). The
proposal (§4.3) cuts training sessions at 4K tokens too. Use the same function, so that training and
evaluation show the model the same kind of shortened output:

```python
# from the GRC-Bench repository root: PYTHONPATH=src python ...   (needs tiktoken)
from grcbench.common.tokens import TOOL_OUTPUT_CAP, count, truncate_tool_output

short = truncate_tool_output(tool_output_text)        # unchanged if it is 4,096 tokens or fewer
```

- **Tokenizer:** `cl100k_base`, the one Qoherent's SFT release counts with. It is the same for every
  model, so the cut is the same for every model.
- **What is kept:** the beginning, the end and the error lines (lines with "error", "exception",
  "traceback", "failed", "warning", "invalid" and similar), with a marker that says how many tokens
  were removed.
- **Offline use:** `tiktoken` downloads the `cl100k_base` file once. Where there is no network, set
  `TIKTOKEN_CACHE_DIR` to a folder that already holds it (GRC-Bench keeps it in
  `~/grc-bench-cache/tiktoken`).

## Thinking text in Qoherent's SFT release

`data/grc-agent-sft-v0.1.0/.../sft_train.jsonl` still carries the models' thinking. All 12 rows have
a `thinking` field on their assistant messages: 279 messages, about 1.06 million characters. The
release also ships `sft_full_with_thinking.jsonl`, so `sft_train.jsonl` was probably meant to be the
version without it. Check with Qoherent before training on `sft_train.jsonl`, and remove the
thinking fields if training should not see them.

## Versions to check

Each file names the version it belongs to. When a version changes, re-check what you built with the
old one.

- **Tier-K dev split:** the version line at the top of `exclude_question_ids.txt`.
- **Flowgraph signatures:** the header of `flowgraph_reserved_signatures.txt` names the signature
  version (v2), its seed and the date of the class map (22 Oct 2026). The 22 Oct class map gives a
  class to Vector Insert, the bitwise logic blocks and Complex to Mag/Phase, so re-check composed
  graphs that use them.
- **Seed split:** `splits/flowgraph_seed_split.json` (in the GRC-Bench repository).

Questions go to the GRC-Bench evaluation lead.

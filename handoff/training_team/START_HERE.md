# For the training team: what GRC-Bench needs you to keep out of training

From the GRC-Bench evaluation team, 22 Oct 2026. Status: draft for team review; not yet sent.
Claude (Anthropic) wrote the tooling and these files. Claude never judges, repairs or generates
training data.

Every rule below protects the benchmark from training data that overlaps with it. Each file in this
folder covers one kind of data. Start with the table, then read the file for the data you work on.

## The files

| File | Covers | What you do |
|---|---|---|
| `README.md` and `exclude_question_ids.txt` | Tier-K questions (the 200 dev questions) | Exclude the 200 dev questions, and everything built from them, from every training use. |
| `test_twins.tsv` (and `README.md` §2–3) | training questions that overlap the 185 test questions | Remove the test twins' answers where possible; the team decides on the flagged neighbours. |
| `block_partition.md` | generated questions about GNU Radio blocks | Write training questions only about blocks whose `split` is `train`, in your own wording. |
| `flowgraph_seed_split.md` | flowgraph and agent tasks built from GNU Radio example files | Never use a test or dev seed file, also not edited or with a fault planted in it. |
| `flowgraph_composed_chains.md` and `flowgraph_reserved_signatures.txt` | flowgraphs written by hand or by a script | Emit a graph only if its signature hashes to train and its digest is not on the reserved list. |
| this file, below | tool outputs inside training sessions | Cut long tool outputs with the same 4K-token function as the benchmark. |

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

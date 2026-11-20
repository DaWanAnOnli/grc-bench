# GRC-Bench

The execution-grounded benchmark for language models working with GNU Radio (Team JJJP × Qoherent,
UofT MScAC). This folder holds the code, the benchmark items and the reports that implement the
evaluation plan in `../benchmark and evaluation/Benchmark_Evaluation_Plan_to_7Dec.docx` (CAD 504 plan).

> **Status:** the parts of the plan that need no money, outside GPU or API access are built, and the
> G2 pre-registration is drafted for the team's sign-off. Every review sheet is filled in by Claude
> review agents under the review protocol (`docs/guidelines/review_protocol.md`): two independent
> reviewers, an expert agent for every disagreement, and decision rules in code. See
> [Status](#status), `reports/review_summary.md` and `reports/compliance_review.md`.

## How to run

Everything runs on Linux or inside WSL (Ubuntu) through one entry script, which pins the versions:

```bash
cd grc-bench         # this repository's folder
./gb help            # list commands
./gb env             # record versions in reports/environment.json; check the pins
./gb test            # run the unit tests
```

`./gb` uses GRC-Agent's own virtual environment (`~/grc-agent/.venv`, read-only) and refuses to run
unless GRC-Agent is at commit `6c98128` and GNU Radio is 3.10.12. From a Windows prompt, with the
repository on the C: drive: `wsl -e bash -lc "cd '/mnt/c/<path to>/grc-bench' && ./gb help"`.

### Set up a new machine

GRC-Bench runs on Linux or WSL with Ubuntu 26.04, whose `gnuradio` package is the pinned 3.10.12.
Clone GRC-Agent at the pinned commit, create its virtual environment, then run the setup check from
this folder:

```bash
git clone https://github.com/qoherent/grc-agent.git ~/grc-agent && git -C ~/grc-agent checkout 6c98128
cd ~/grc-agent && uv venv --system-site-packages --python /usr/bin/python3 \
  && uv sync --extra dev --locked --python .venv/bin/python
cd <this folder> && ./setup.sh
```

`./setup.sh` checks every prerequisite and prints the fix for each one missing; it is safe to run
again. It builds two things once: GRC-Agent's two lexical knowledge databases, in GRC-Agent's
git-ignored `src/grc_agent/vectors/` folder (GRC-Agent does not ship them and builds them there on
first use), and the tokenizer file. It ends with `./gb`'s pin check. Every command that starts
GRC-Agent needs the databases: Tier-A runs, the Tier-F golden replay gate, the Tier-K retrieval
baseline and `./gb agent`. Model runs also need llama.cpp and the smoke-test model (`./gb fetch`).

## Layout

| Folder | What is in it |
|---|---|
| `src/grcbench/` | the code, one package per part of the plan (below) |
| `items/` | benchmark items: `tier_k/`, `tier_f/`, `tier_a/`, each with `test/` and `dev/` |
| `splits/` | manifests: Tier-K dev split, anchor and CI subsets, label map, block partition, flowgraph seed split |
| `handoff/` | files other people act on: training-team exclusions, the review sheets and their guides, the Qoherent request draft |
| `reviews/` | the review protocol's decided records (one file per sheet) and the declared agent runs (`review_log.jsonl`) |
| `reports/` | generated reports (labels, contamination, seeds, gates, smoke test, compliance review) |
| `docs/` | pre-registration draft, annotation guidelines, review checklists |
| `runs/` | the append-only test-run log (run outputs are not versioned) |
| `tests/` | unit tests and their fixtures |
| `docker/` | the intended Docker image (not built yet) |

Large downloads (llama.cpp b12020, the smoke-test model, the embedding model and its llama.cpp b10333) and per-run working
directories live outside the repository, in `~/grc-bench-cache` and `~/grc-bench-work` inside WSL.

| Package | Plan section | Command |
|---|---|---|
| `common/` | paths, pins, sandbox (bubblewrap), canary, item schemas, 4K tool-output cap; the Colab package | `env`, `baseline`, `fetch`, `reference`, `audit`, `colab-package` |
| `splits/` | §4.1 dev split, anchor and CI subsets | `split-dev` |
| `labels/` | §4.2 task-type labels | `labels` |
| `contamination/` | §6 8-gram, TF-IDF and embedding overlap; audit of Qoherent's agent traces | `contamination`, `traces` |
| `tier_k/` | §4.2, §5.2 knowledge items, GNU Radio behaviour questions, question prompt, rubrics, rule-cascade scoring, B1 retriever | `tier-k-items`, `blocks`, `blockq`, `calc`, `rubrics`, `placeholder-rubrics`, `score-k`, `retriever` |
| `tier_f/` | §4.3, §5.4 validity ladder V1–V5, seeds, tasks, gates | `ladder`, `seeds`, `tier-f` |
| `tier_a/` | §4.4 headless GRC-Agent runner, tasks, grader; the same runner for any request and its shareable zip (`handoff/agent_runner.md`) | `tier-a`, `agent`, `agent-package` |
| `tier_e/` | §4.5 speed and memory protocol | `tier-e` |
| `stats/` | §9 analysis functions; the pre-registration's generated appendices | `stats`, `prereg` |
| `runlog/` | §4.1 test-run log | `runlog` |
| `review/` | §5.2–§5.4 review protocol: blinded inputs, declared agent runs, decision rules, filled sheets, checks | `review` |

## Rules this code enforces

- `./gb` never writes to `data/`, the proposals, the documents or `~/grc-agent` (`setup.sh` builds
  GRC-Agent's knowledge databases there once, only when they are missing). `./gb baseline check`
  compares full file-tree manifests (including git-ignored files) taken before the first run.
- Untrusted flowgraphs and agent tool calls run in a bubblewrap sandbox: read-only filesystem,
  writable run directory only, no network, time and memory limits. The user's files are hidden (the
  home folder apart from GRC-Agent and the cache, and the Windows drives), and code the agent starts
  sees only its own project folder: not the task record, other runs, the repository or any API key.
  A hosted model's key stays on the host (`tier_a/relay.py`).
- Every item carries the canary string, a licence, its source, a cluster id, provenance and a
  review status. The canary never appears in anything a model under test sees.
- Claude (Anthropic) drafts the test content and, under the review protocol
  (`docs/guidelines/review_protocol.md`), also reviews it and writes the Tier-K rubrics:
  - two independent review agents per sheet;
  - a writer agent and two checker agents per rubric;
  - an expert agent that decides every disagreement.

  The decision rules are code (`./gb review`), and every factual claim is checked by a quoted source
  or a run. Every agent run is declared in `reviews/review_log.jsonl`. Claude never changes training
  data. Anthropic models are excluded from the model panel. Pre-registration §10 lists what this
  departs from in the evaluation plan.

## Fresh GNU Radio question sets

The 100 GNU Radio behaviour questions of Tier K are the published reference set. Each asks what a block
outputs for a stated input, and GNU Radio's own output is the key. Anyone can draw a fresh set:

```
./gb blockq --seed <any text> --like-reference      # writes ~/grc-bench-work/fresh/<seed hash>/
```

A fresh set has the same blocks, kinds and number of questions per block as the reference set, repeats
none of its instances, and pairs each question with a reference question (`answers.jsonl`,
`pairs_with`). Report a model's score on both: training on block descriptions and input/output pairs is
fine, and a model that memorised the published questions shows a gap (`stats.functions.fresh_gap`:
reference minus fresh, paired by slot, clustered by block). There is no test/train block split.

## Status

| Part | State | Where |
|---|---|---|
| Tier K | 285 test questions (185 frozen, 100 GNU Radio behaviour questions answered by running GNU Radio) and 107 dev questions (the eligible part of the training team's validation split); fresh GNU Radio question sets from any seed; closed-book question prompt, rule cascade and judge prompt; the reviewed rubrics, with 8 frozen and 3 dev questions removed as underspecified, so 277 test and 104 dev questions are scored | `items/tier_k/`, `reports/tier_k_block_questions.md`, `reports/tier_k_fresh_sets.md` |
| Tier F | 222 tasks, 30 drafted by hand and 192 from the seeded generator (`./gb tier-f generate`), 27 of them built to replace the tasks the review's first round removed; after both rounds 148 test (L1 33, L2 85, L3 30) and 45 dev tasks remain, 29 removed (28 duplicates); all gates passed, in GRC-Agent's harness as one lookup and one edit | `items/tier_f/`, `reports/tier_f_first_set.md` |
| Tier A | 51 tasks, all gates passed: 42 in the first set and 9 built to replace the 7 the review's first round removed (every knowledge task and 3 of 4 save_block tasks, too short for the 8-call floor); 30 test tasks (3 per category) and 12 dev tasks are scored, and 2 accepted replacements are kept as unscored reserves | `items/tier_a/`, `reports/tier_a_first_set.md` |
| GRC-Agent runner | `./gb agent`: GRC-Agent headless on any request, outputs as files, conversations continued across calls, local or hosted OpenAI-compatible models; `./gb agent-package` builds the teammates' zip (setup script, tutorial, self-test; no benchmark content) | `handoff/agent_runner.md`, `handoff/agent_package/TUTORIAL.md` |
| Tier E | llama-bench protocol, checked on a small model (CPU and the local RTX 5090) | `reports/tier_e/` |
| Pipeline tests | placeholder rubrics for the dev questions (`./gb placeholder-rubrics`: the training release's reference answer as one point; `./gb tier-k score DIR --rubrics FILE`, dev only), so the Tier-K route runs end to end apart from the reviewed rubrics; `./gb colab-package`: `./gb`, the code and the dev split (no test item) for a pipeline test on a Colab GPU | `handoff/colab/` |
| Contamination | 8-gram, TF-IDF and embedding audit of every tier, including Qoherent's 1,853 agent traces; task-level check of the traces (`./gb traces`); the flagged pairs are reviewed (confirmed rate: Tier K test 11.9 %, dev 4.7 %, Tier F 0 %) | `reports/contamination.md`, `reports/agent_traces.md` |
| Review | every review sheet filled in by review agents and an expert agent (`./gb review`); decided records in `reviews/`; the expert review of 433 items (H1b counts the original 397, the 36 replacement tasks beside it) | `reports/review_summary.md`, `docs/guidelines/review_protocol.md` |
| Analysis | `grcbench.stats`, self-check 39/39 | `./gb stats selfcheck` |
| Pre-registration | signed off and declared at the freeze (20 Nov 2026, tag `v1.0`); the run list is `status: declared`, so a test run is accepted only for a configuration on it | `docs/preregistration/` |
| Final check | `./gb test`, `./gb audit`, `./gb baseline check` | `reports/compliance_review.md` |

The reviews of every drafted item (`handoff/review_sheets/`) follow the review protocol; their decided
records are in `reviews/` and their summary in `reports/review_summary.md`. Waiting for the team: the
decisions in pre-registration §8. Waiting for access: model runs, judge calibration and the throughput
pilot (`handoff/joel_actions.md`).

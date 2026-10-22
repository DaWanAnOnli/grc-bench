# GRC-Bench

The execution-grounded benchmark for language models working with GNU Radio (Team JJJP × Qoherent,
UofT MScAC). This folder holds the code, the benchmark items and the reports that implement the
evaluation plan in `../benchmark and evaluation/Benchmark_Evaluation_Plan_to_7Dec.docx` (CAD 504 plan).

> **Status (23 Oct 2026, week 3):** the parts of the plan that need no money, outside GPU or API
> access are built, and the G2 pre-registration is drafted for the team's sign-off. Every item drafted
> by Claude (Anthropic) is marked `pending-team-review`: the team reviews, edits and decides. See
> [Status](#status) and `reports/compliance_review.md`.

## How to run

Everything runs inside WSL (Ubuntu) through one entry script, which pins the versions:

```bash
cd "/mnt/c/Users/POPA/Documents/!MScAC/2701/ml_ops_project/project/grc-bench"
./gb help            # list commands
./gb env             # record versions in reports/environment.json; check the pins
./gb test            # run the unit tests
```

`./gb` uses GRC-Agent's own virtual environment (`~/grc-agent/.venv`, read-only) and refuses to run
unless GRC-Agent is at commit `6c98128` and GNU Radio is 3.10.12. From Windows:
`wsl -e bash -lc "cd '/mnt/c/Users/POPA/Documents/!MScAC/2701/ml_ops_project/project/grc-bench' && ./gb help"`.

## Layout

| Folder | What is in it |
|---|---|
| `src/grcbench/` | the code, one package per part of the plan (below) |
| `items/` | benchmark items: `tier_k/`, `tier_f/`, `tier_a/`, each with `test/` and `dev/` |
| `splits/` | manifests: Tier-K dev split, anchor and CI subsets, label map, block partition, flowgraph seed split |
| `handoff/` | files other people act on: training-team exclusions, review sheets for the team, the Qoherent request draft |
| `reports/` | generated reports (labels, contamination, seeds, gates, smoke test, compliance review) |
| `docs/` | pre-registration draft, annotation guidelines, review checklists |
| `runs/` | the append-only test-run log (run outputs are not versioned) |
| `tests/` | unit tests and their fixtures |
| `docker/` | the intended Docker image (not built yet: Docker is not installed on this machine) |

Large downloads (llama.cpp b10333, the smoke-test model, the embedding model) and per-run working
directories live outside the repository, in `~/grc-bench-cache` and `~/grc-bench-work` inside WSL.

| Package | Plan section | Command |
|---|---|---|
| `common/` | paths, pins, sandbox (bubblewrap), canary, item schemas, 4K tool-output cap | `env`, `baseline`, `fetch`, `reference`, `audit` |
| `splits/` | §4.1 dev split, anchor and CI subsets | `split-dev` |
| `labels/` | §4.2 task-type labels | `labels` |
| `contamination/` | §6 8-gram, TF-IDF and embedding overlap; audit of Qoherent's agent traces | `contamination`, `traces` |
| `tier_k/` | §4.2, §5.2 knowledge items, GNU Radio behaviour questions, question prompt, rubrics, rule-cascade scoring, B1 retriever | `tier-k-items`, `blocks`, `blockq`, `calc`, `rubrics`, `score-k`, `retriever` |
| `tier_f/` | §4.3, §5.4 validity ladder V1–V6, seeds, tasks, gates | `ladder`, `seeds`, `tier-f` |
| `tier_a/` | §4.4 headless GRC-Agent runner, tasks, grader; the same runner for any request (`handoff/agent_runner.md`) | `tier-a`, `agent` |
| `tier_e/` | §4.5 speed and memory protocol | `tier-e` |
| `stats/` | §9 analysis functions; the pre-registration's generated appendices | `stats`, `prereg` |
| `runlog/` | §4.1 test-run log | `runlog` |

## Rules this code enforces

- `data/`, the proposals, the documents and `~/grc-agent` are never written to. `./gb baseline check`
  compares full file-tree manifests (including git-ignored files) taken before the first run.
- Untrusted flowgraphs and agent tool calls run in a bubblewrap sandbox: read-only filesystem,
  writable run directory only, no network, time and memory limits. The user's files are hidden (the
  home folder apart from GRC-Agent and the cache, and the Windows drives), and code the agent starts
  sees only its own project folder: not the task record, other runs, the repository or any API key.
  A hosted model's key stays on the host (`tier_a/relay.py`).
- Every item carries the canary string, a licence, its source, a cluster id, provenance and a
  review status. The canary never appears in anything a model under test sees.
- Claude (Anthropic) drafts natural-language test content for the team to review; it never drafts
  reference answers or rubric prose (eval plan Table 5), and it does not touch training data.
  Anthropic models are therefore excluded from the model panel.

## Status

| Part | State (23 Oct) | Where |
|---|---|---|
| Tier K | 285 test questions (185 frozen, 100 GNU Radio behaviour questions answered by running GNU Radio) and 200 dev questions; closed-book question prompt, rule cascade and judge prompt; 385 empty rubric templates for the team | `items/tier_k/`, `reports/tier_k_block_questions.md` |
| Tier F | first set of 30 tasks (20 test, 10 dev), all gates passed, in GRC-Agent's harness as one lookup and one edit | `items/tier_f/`, `reports/tier_f_first_set.md` |
| Tier A | first set of 22 tasks (10 test, 12 dev), all gates passed; 20 more test tasks drafted as specs | `items/tier_a/`, `reports/tier_a_first_set.md` |
| GRC-Agent runner | `./gb agent`: GRC-Agent headless on any request, outputs as files, conversations continued across calls, local or hosted OpenAI-compatible models | `handoff/agent_runner.md` |
| Tier E | llama-bench protocol, checked on a small model (CPU and the local RTX 5090) | `reports/tier_e/` |
| Contamination | 8-gram, TF-IDF and embedding audit of every tier, including Qoherent's 1,853 agent traces; task-level check of the traces (`./gb traces`) | `reports/contamination.md`, `reports/agent_traces.md` |
| Analysis | `grcbench.stats`, self-check 37/37 | `./gb stats selfcheck` |
| Pre-registration | draft for sign-off at G2; the run list is `status: draft`, so every test run is refused until then | `docs/preregistration/` |
| Final check | `./gb test`, `./gb audit`, `./gb baseline check` | `reports/compliance_review.md` |

Waiting for the team: reviews of every drafted item (`handoff/review_sheets/`) and the decisions in
pre-registration §8. Waiting for access: model runs, judge calibration and
the throughput pilot (`handoff/joel_actions.md`).

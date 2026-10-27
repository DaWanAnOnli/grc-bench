# Plan: GRC-Bench build — everything executable now (CAD 504 plan)

## Context
The CAD 504 evaluation plan (`benchmark and evaluation/Benchmark_Evaluation_Plan_to_7Dec.docx`) and the methodology proposal (`Methodology_Proposal_to_7Dec.docx`) are written. Joel asked to:
- execute every part that needs no money, GPU or outside access;
- use this laptop (i7-1360P, 16 GB RAM, no NVIDIA GPU) and its WSL Ubuntu 26.04 (GNU Radio 3.10.12; GRC-Agent 0.6.0 at commit `6c98128`);
- keep a clean, intuitive working directory;
- review every part against the evaluation plan, the proposal and the Data Summary.

Today is Wed 14 Oct (W2). The gates are G1 Fri 16 Oct, G2 Fri 30 Oct (v0.9 end-to-end, OSF pre-registration) and G3 Fri 13 Nov (freeze). Ultracode is on, so execution runs as a sequence of workflows, one per phase, with me reviewing between them.

### User decisions (14 Oct)
1. **Claude drafts and the team decides.**
   - Claude drafts the natural-language test content: agent-task specs, flowgraph-task instructions and question templates.
   - The team reviews, edits and decides.
   - Every drafted item records `provenance.drafted_by: "Claude (Anthropic)"` and `review.status: pending-team-review`.
   - Data Summary §11 D tells Qoherent the team writes the agent tasks with no AI involved. This change is recorded as a deviation for G1.
   - Anthropic models are excluded from the panel.
   - Claude never drafts rubric prose; Table 5 bars the Claude family from drafting references.
   - Calculation answers come from Claude's scripts, and a team member recomputes them blind.
   - Knowledge-task facts are derived mechanically from the graph or the block YAML.
   - Golden graphs and oracles written by Claude are declared in the role table and pass every automated gate plus team review.
   - From now on, Claude never judges, repairs or generates training data. Partition manifests are the only training-side output.
2. **First set first.** Flowgraph tasks start with a first set of ~30 (≈ 10 dev + 20 test) covering build, modify, repair and parameter-tune and both L1 and L2. The set passes every gate and team review before scaling to 150. Tier A follows the same pattern: all specs are drafted, but only a first set is fully built and gated.
3. **Location.** A new `project/grc-bench/` folder, a local git repo committed with Windows git only.
   - Never modified: `data/`, the proposals, the documents and `~/grc-agent` (per Qoherent: "no expected changes to the grc-agent repository").
   - Run directories live on the WSL filesystem (`~/grc-bench-work`), not `/mnt/c`.
4. **Smoke model.** Run inside WSL:
   - llama.cpp **b10333** (the build GRC-Agent already pins) plus `Qwen3-4B-Q4_K_M.gguf` (2.50 GB, SHA-256 `7485fe6f…34fdf5`).
   - Settings: CPU, 16K context, `q8_0` KV cache with `-fa on`, `--jinja`, a pinned `--reasoning-format`, `-np 1`.
   - It tests plumbing only and is never a result.
   - The Windows Ollama can't be reached from WSL, and its `gemma3:12b` has no tool support.

### Decision (16 Oct): Tier F runs in GRC-Agent's harness as "one lookup + one edit"
GRC-Agent never lets the model write a .grc file. Its file tools refuse .grc writes, and the model edits only through `change_graph`. So Tier F is no longer "instruction → .grc YAML". A Tier-F sample is a short episode in the **same harness as Tier A**, with identical inputs:
- **Inputs:** the real system prompt, the 22 tool schemas, the active-flowgraph note, and the chat template.
- **Prefilled turn:** the history already contains one full `inspect_graph` call and GRC-Agent's real result for the start graph. For build tasks, the start graph is an empty graph with a `samp_rate` variable.

Rules:
1. At most **one `query_knowledge`** call, with the query, domain and k chosen by the model. The model may skip it.
2. **One atomic `change_graph`** call, applied through GRC-Agent's own `change_graph`. A rejected batch rolls back with no retry. The task then fails at the matching rung: an unknown block at V2, a validation error at V3. A forced invalid graph is caught by V3.
3. Out-of-budget calls and any other tool (run, shell, files, web) get a fixed reply: "not available in this task; submit your edit with change_graph". The budget is not announced up front.
4. **Cap of 3 model turns.** If no `change_graph` has been applied by then, the task fails.
5. Calls in the same reply (lookup + edit) both run, and the edit cannot use the lookup's result. This is the model's choice and is recorded.

The final graph is graded by the ladder V2–V5. V1 means "a well-formed `change_graph` call matching the tool schema".

**Diagnostics:**
- whether the lookup was used;
- retrieval recall of the reference's block ids in the lookup results;
- whether GRC-Agent accepted the edit;
- whether one search can surface all of the reference's blocks.

**Consequences:**
- The Tier-A runner gets a `tier_f` setting.
- Each Tier-F task stores `golden_calls.json` (a lookup + change_graph args equivalent to start→reference). Solvability becomes golden replay 10/10 through the real runner.
- B1 for Tier F is redefined as oracle grounding: catalog entries for every reference block.
- The Tier-F text in the evaluation plan is to be updated.

**Oracle policy:** V5 = signal checks + **structural graph checks** (`tier_f/graph_checks.py`: block_present / block_absent / path / same_blocks_as_start / max_functional_blocks).
- Every requirement the instruction states must be checked.
- Structural wording ("filter the square wave", "only change parameter values") needs a graph check.
- Otherwise, phrase the instruction behaviourally so any valid structure passes.
- If the oracle needs a long run, the instruction states it ("keeps producing samples until stopped").

### Verified facts the plan relies on
- **Frozen split algorithm, reproduced exactly:**
  - strata are the raw `(origin, task_type, difficulty)` values;
  - each stratum gets the floor of `185·n/1853`, and the leftover slots go to the largest remainders (ties broken by sorting the stratum tuples);
  - within a stratum, ids are ranked by `sha256(seed+"\0"+id)`.
- **Task-type spellings:** there are 23 raw spellings for the 6 canonical types; 134 rows are ambiguous (`conceptual`, the `compare` variants, the info-request variants).
- **Test↔train overlap:** 26 test questions share an 8-gram with a training question, using the tokenizer that lowercases and turns non-word characters into spaces. All 26 are template twins. 33 test questions share an 8-gram with some teacher answer.
- **Training data:** nothing has been exported yet (`11_training_exports` is empty). An export would use 11,128 judged pairs.
- **Examples (230):**
  - 13 are strictly headless and 119 are GUI-only, which need conversion;
  - 98 need hardware or a network, and 28 are invalid as shipped;
  - WSL has no ALSA audio devices;
  - noise sources with a fixed seed are bit-identical across runs.
- **GRC validator gaps:** `is_valid()` misses unknown block ids (they load as dummy blocks), dangling connections (only `import_data` returns them) and invalid enum values (silently replaced).
- **GRC-Agent:**
  - the real executor is `agent_factory.build_agents_from_cfg`, with 22 tools in the SFT rows;
  - running, saving and run logs are only implemented in the GTK `NativeFlowgraphProxy`;
  - the file tools are GTK-free;
  - the `.venv` already has numpy, scipy, yaml, pytest, jsonschema and tiktoken, and is used read-only;
  - `bwrap` is present; there is no sudo and no Docker.
- **Training task specs to stay disjoint from:** Qoherent's collection tasks t01–t15 plus t16/22/24/25, and GRC-Agent's 15 `tests/scenarios`. They are recoverable read-only through `git show 4bd0436^:…`.

## Working directory
```
project/grc-bench/
  README.md        purpose, how to run each step, status table, map to eval-plan sections
  gb               WSL entry script: GRC-Agent .venv python, PYTHONPATH=src, refuses to run unless
                   ~/grc-agent HEAD=6c98128 and GNU Radio=3.10.12; isolation env (below)
  pyproject.toml   metadata + pytest config
  src/grcbench/    common/ (paths, io, canary, schemas, sandbox, tokens) · splits/ · labels/ ·
                   contamination/ · tier_k/ (blockgen, calc, rules, judge_prompt, retriever) ·
                   tier_f/ (ladder, convert, compose, faults, oracles, mutate, tasks, partial_credit) ·
                   tier_a/ (agent_build, headless_proxy, caps, user, runner, grader, replay) ·
                   tier_e/ (llama-bench protocol) · stats/ · runlog/
  items/           tier_k/{test,dev} · tier_f/{test,dev}/<task_id>/ · tier_a/{test,dev}/<task_id>/
  splits/          K dev split, anchor + CI subsets, block partition, flowgraph seed split, label map
  handoff/         training_team/ · qoherent/ · joel_actions.md · review_sheets/ (CSV for the team)
  reports/         labels, contamination, seed inventory, ladder/solvability/mutation, smoke,
                   environment.json, compliance_review.md, reference/ (recovered specs: hashes + refs)
  docs/            preregistration/ (preregistration.md, run_list.yaml) · guidelines/ · checklists/
  runs/            test_run_log.jsonl (append-only); run outputs git-ignored
  tests/           pytest
  docker/          Dockerfile pinning GNU Radio 3.10.12 (not built: no Docker yet)
```
- **Every item record carries** `canary`, `license` (GNU Radio examples are GPL-3.0), `source`, `cluster_id`, `provenance` and `review`.
- **The canary string** stays out of prompts.
- **Frozen items** are never rewritten in place: a change makes a new version and keeps the original.
- **Large downloads** go to `~/grc-bench-cache`: llama.cpp b10333, Qwen3-4B, and EmbeddingGemma at the SHA pinned in `embed_runtime.py`.

### Isolation (applies to every harness and agent run)
- **One OS process per run**, inside `bwrap`:
  - `--unshare-net`;
  - a temporary `HOME`, and `XDG_*` pointing into the run directory;
  - system directories read-only, the run directory writable;
  - rlimits and a timeout.
- **Environment variables:** `GRC_HIER_PATH`, `GRC_AGENT_ENV`, `GRC_AGENT_VECTORS_DIR` (a copy of the lexical DBs), `GRC_AGENT_RUNTIME_DIR`, `PYTHONDONTWRITEBYTECODE=1`, `QT_QPA_PLATFORM=offscreen`.
- **Catalog hash:** the GRC block catalog is hashed at V2.
- **Baseline manifests:** before any run, take a full file-tree manifest of `~/grc-agent` (including git-ignored files; the untracked `projects/` folder is part of the baseline), `data/` and the documents.

## Phases (each = one Workflow run; I review results before the next)

### Phase 0 — Foundation (inline, me)
- Skeleton, `git init`, `.gitattributes` (`eol=lf`), `.gitignore`, `gb`, `common/` (canary GUID, schemas, `sandbox.py` bwrap wrapper, cl100k truncation helper).
- `reports/environment.json` and the baseline manifests.
- Background downloads with checksum verification.
- Read-only recovery of Qoherent's `task_spec.py`, `runner_contract.py`, `export.py` and the t-specs into `reports/reference/`, with only commit/path/hash references in anything releasable.

### Phase 1 — Splits, labels, catalog, seeds (Workflow "foundations")
Agents own disjoint modules. Each output is checked by a separate verifier agent that tries to refute it.

**WS1 — Tier-K dev split** (top priority: it unblocks the training team).
- `splits/` implements the frozen algorithm. A test must regenerate the 185 frozen IDs exactly.
- The dev split is 200 of the 1,668 training questions:
  - **strata:** the raw (origin, task_type, difficulty) values, as in the frozen split;
  - **seed:** `grc-bench-dev-v1-2026-10-14`.
- **Twin groups** are built from train↔train 8-gram twins plus `semantic_duplicate_candidates.jsonl` (5,720 scored pairs) above a threshold.
- **Excluded from dev:**
  - every twin-group member, so no dev item has a sibling in training;
  - the training twins of test questions;
  - the 8 questions every teacher answer of which was rejected.
- **Report:** the dev composition compared with test.
- **Subsets:** an anchor subset of 60 dev questions; a CI smoke subset of 20 Tier-K dev questions now, plus 10 Tier-F dev tasks after Phase 3.
- **Outputs:**
  - `splits/tier_k_dev.json`;
  - `items/tier_k/dev/`;
  - `handoff/training_team/`: dev IDs excluded from every training use (SFT pairs, DPO repair pairs, RAFT examples, SliCK tagging).

**WS2 — Labels and contamination.**
- **Rule map** for the 14 unambiguous task-type spellings.
- **The 134 ambiguous rows** are each classified with a reason, against `task_types.json` and `curriculum.md`. Raw labels are kept.
- **Team labelling:**
  - all ambiguous **test** items are labelled by two team members, and κ is reported;
  - 10% of the training rows are spot-checked.
- **Frozen before any test run.** Merges for groups under 20, and the factual vs. reasoning mapping, go into the G2 pre-registration.
- **Question-family `cluster_id`s** for Tier K come from the twin groups.
- **Contamination audit:**
  - 8-gram overlap and TF-IDF cosine (both lexical);
  - **embedding similarity now**, with EmbeddingGemma on llama.cpp b10333 in embedding mode.
  - Compared sets: every benchmark item against training questions minus dev, teacher answers, both repair sets, the SFT sessions and quarantine, the Qoherent t-specs and GRC-Agent's scenarios.
  - Reported per tier, and re-run after Phases 3–4 and again in W6.
  - The LLM rephrase detector waits for API access.

**WS3a — Block partition.**
- Test vs. train blocks, hash-ranked and stratified by module.
- Excluded: deprecated blocks and the hardware/network modules (uhd, soapy, iio, zeromq, network, audio).
- Only the partition is handed off; the training side must use different wording.

**WS4a — Seed inventory and split.**
- Convert all 132 headless or GUI-only examples. The swap list:
  - `qtgui_*` sinks become **file sinks**, which are the observation points;
  - `variable_qtgui_*` widgets become plain variables;
  - audio sinks become file sinks;
  - `no_gui` / `run` options, and Head-bounded sources.
- Keep the graphs that pass V1–V4, and measure the yield.
- **Inventory table** in `reports/seed_inventory.md`.
- **Seed split** (test / dev / train) at source-file level:
  - Qoherent's and GRC-Agent's seeds are forced to train;
  - L1 chain signatures (see Phase 3) are partitioned by signature.
- Output `splits/flowgraph_seed_split.json`; the training team needs it in W3.

### Phase 2 — Tier-F ladder harness (inline core plus a test-writing workflow)
- **Rungs:**
  - V1: YAML and `.grc` schema.
  - V2: Platform load. Fails on dummy blocks, dangling connections, invalid enum values (linted against the YAML) or a catalog-hash mismatch.
  - V3: compile: `validate()`, then `grcc` in the sandbox.
  - V4: bounded subprocess run. Fails on a traceback, on not stopping cleanly within a grace period, or on the liveness check (≥ N items reaching every observation sink).
  - V5: the task oracle.
- **Fixtures:** one per failure class (unconnected port, IO-size mismatch, failed assert, undefined variable, double upstream, unknown block, dangling connection, bad enum), plus the 28 examples that are invalid as shipped as negative cases.
- **Compact `.grc` form:** no `states`, only non-default parameters; GRC fills in the rest, which is verified. Reference answers must be ≤ 3K tokens. The 8K Tier-F answer cap includes reasoning.
- **Prompt and extraction:** the Tier-F prompt format and the answer-extraction rule (the first fenced `.grc` YAML block).
- **Partial credit [48]:** the structural score over blocks, parameters and connections (order-insensitive).

### Phase 3 — Tier-F first set (Workflow "tier-f-first-set": pipeline per task)
Pipeline per task: draft spec → build the reference and oracle → gates → adversarial review. A completeness critic runs at the end.

- **The first set:** ~30 tasks (≈ 10 dev, 20 test).
  - All four families: build, modify, repair (deleted connection, type mismatch, rate mismatch, undefined variable, nonexistent block) and parameter-tune.
  - Both L1 (from the deterministic composer of ≤ 5-block single-rate chains) and L2 (from converted examples).
  - Numbers perturbed, with provenance.
- **Each task:**
  - a Claude-drafted instruction;
  - an observation contract;
  - a golden reference;
  - for repair/modify/tune tasks, a check that the start state fails the oracle;
  - fail-to-pass **and** pass-to-pass checks;
  - spec-based tolerances;
  - numpy seeded in the runner.
- **Gates:**
  - the reference passes V1–V5 in 10 of 10 runs, under evaluation concurrency (8 workers);
  - ≥ 2 equivalent valid variants pass;
  - ≥ 5 mutants that break the spec (wrong frequency, bypassed filter, wrong sample rate, wrong gain, wrong-tap rewiring) all reach V4, and the oracle rejects ≥ 90%;
  - equivalent mutants are excluded from the denominator;
  - failing tasks are tightened or dropped, and the drops are counted.
- **Team review sheet:** `handoff/review_sheets/tier_f_first_set.csv`. Scaling to 150 happens after the team's decision.

### Phase 4 — Tier-A runner and first set (Workflow "tier-a")
**Agent build** mirrors `build_agents_from_cfg` at `6c98128`, with these changes:
- the model is llama-server via `openai_compatible`;
- step persistence goes to a temporary store;
- compaction is off in the benchmark setting, with a "GRC-Agent defaults" variant kept for the 10-task sensitivity check;
- the web tools are kept in the schema but return a fixed "offline" message;
- the shell tools run under bwrap.

**`HeadlessFlowgraphProxy`** ports the run, save, log and save-block behaviour of `NativeFlowgraphProxy` field by field, reusing the GTK-free `exec_monitor.py`. It runs the unmodified Qt program offscreen.

**Conformance tests:**
- the system prompt and the 22 tool schemas equal the `system`/`tools` of the SFT rows;
- run and log results match the shapes in the SFT rows.
- Fallback if needed: the real app under WSLg.

**Caps:**
- `UsageLimits(request_limit=40, per_request_input_tokens_limit=65536)`, with one usage counter shared across approval resumes and scripted-user turns.
- GRC-Agent's own graceful stop counts as a cap failure.
- The engine context is ≥ 64K per slot for real runs, with context shift off.
- The 4K-token tool-output cap (cl100k; head + tail + error lines) is applied by a `WrapToolExecuteHandler`. Truncation is **not** a failure. The same function goes to the training team.
- Raw tool arguments are logged before the JSON repair runs.

**Scripted user:** pattern → reply; otherwise "please use your best judgement".

**Trace format:** Qoherent's SFT row format, from the recovered `export.py`.

**Guards:**
- the test split needs a `run_list.yaml` config id;
- the run is logged before it starts;
- collection mode refuses test and dev items.

**Tasks:**
- Draft all 30 test and 12 dev specs, covering Qoherent's 9 categories plus 2–3 refusal/abstention tasks.
- Each records its disjointness from the t-specs and the scenarios.
- **The first set fully built and gated:** all 12 dev tasks and 10 test tasks that between them cover every category.
- **Gates:**
  - the golden tool-call sequence replays 10 of 10 via pydantic-ai `FunctionModel`, confirming the 8–20 call range;
  - an agent that does nothing fails every task;
  - the wrong-end-state fixture is rejected;
  - the grader re-runs the final graph itself.

**Smoke test:**
- FunctionModel replay first, then Qwen3-4B on 3 dev tasks.
- Record whether `--jinja` grammar-constrains tool calls, and pin it (pre-registration §8, decision 1: B0 is the server default).

### Phase 5 — Tier K, scoring, Tier E, B1, analysis, docs (Workflow "scoring-and-docs")
**WS3b — Block questions (~100 test).** The question kinds:
- port dtype for a given type setting;
- the type → I/O mapping (`option_attributes`);
- enum choices (scored as a set);
- decimation/interpolation rate (executed);
- assert constraints;
- port counts and message ports.

Default-value, which-block and parameter-meaning questions are dropped. Answers come from the YAML and are checked through the GRC Platform API (instantiate the block, set its type, rewrite, read the ports), with rates confirmed by running a flowgraph. Each question gets `cluster_id` = block family. They are deduplicated against the bank and come with a wording-review sheet.

**WS6 — Rubrics and scoring:**
- **Rubric templates** for the 185 test and 200 dev questions; the prose is left empty.
- **Calculation scripts** for the 25 calculation questions, plus a blind recomputation sheet.
- **Rule cascade:** parse → numeric (pre-registered tiered relative-error credit; unit or order-of-magnitude error = 0) → symbolic → judge.
- **MiniMax-M3 judge prompt template.**

**Tier E:** llama-bench protocol tooling (median of 5, warm, at 4K–64K context), exercised on Qwen3-4B as a plumbing test.

**B1 retriever:** GRC-Agent's lexical BM25 database (a copy), with retrieved pages added to the prompt.

**WS7 — `stats/` and the test-run log:**
- **`stats/` functions:**
  - pass@k, pass^k;
  - cluster bootstrap and clustered SE;
  - McNemar and Wilcoxon, Holm–Bonferroni, effect sizes;
  - MDE (must reproduce Table 8: K 7–9, F 10–13, A 23–28 points);
  - difficulty and point-biserial;
  - Kendall τ stability;
  - κ, Scott's π, Krippendorff's α, judge–human gap.
- **Pre-registered definitions:**
  - McNemar's per-item pass = majority over n samples;
  - split-half = samples {1,2} vs {3,4}, plus a random-half bootstrap;
  - H1c is size-matched by subsampling Tier K to 150;
  - Func@1's denominator is tasks with an oracle;
  - Holm–Bonferroni families are per research question.
- **Test-run log tooling.**

**Docs:**
- `docs/preregistration/` (`preregistration.md` and `run_list.yaml`, which is Table 6), OSF-ready for G2.
- `docs/checklists/`: two-person task review and expert review (needed now).
- `docs/guidelines/`: rubric writing, judge calibration (150 answers, 3 annotators including the Qoherent expert) and the error-analysis taxonomy.

**WS9 — Hand-offs, drafted and not sent:**
- **Training team:** dev exclusions, test twins, the block partition, the seed split, the truncation function, the rule that Claude does not touch training data, and a note that `sft_train.jsonl` still contains thinking.
- **Qoherent request:**
  - ~30 h of expert review plus a calibration annotator;
  - task descriptions for further sessions;
  - the H4 reference hardware;
  - the Data Summary §11 D deviation.
- **`joel_actions.md`:**
  - send the compute request before G1;
  - run `sudo apt install build-essential cmake` in WSL for Nanbeige's llama.cpp fork;
  - install Docker.

### Phase 6 — Review (Workflow "compliance-review")
- **Automated gates:**
  - pytest;
  - 185/185 reproduction;
  - generators deterministic (re-run → identical hashes);
  - Tier-F gates and Tier-A replay gates;
  - disjointness (dev∩train, test∩train, block partition, seed split, Tier-A vs. specs and scenarios);
  - canary, license and schema validation;
  - the before/after file-tree manifests unchanged.
- **Compliance matrix** `reports/compliance_review.md`. One verifier agent per section of eval-plan §1.2, §4–§9, §11 (W2–W4), §12–§14; proposal §4–§6; Data Summary A/C/D; Table 5.
  - Each requirement → artifact → check → status: done, partial, or blocked with the reason.
  - Findings are adversarially verified by 3 skeptics, fixed, and re-checked until a round finds nothing new.
- **Recorded deviations:**
  - native WSL GNU Radio instead of Docker;
  - offline web tools;
  - Data Summary A/C/D;
  - L3 tasks deferred to W5;
  - Tier-F and Tier-A first sets, pending team review.

## Not executable now
- **Needs money or access:** the throughput pilot; judge calibration runs, B2 and the panel (no API access or GPUs); base-model runs and item pruning (no weights or GPU); fine-tuned runs (training not started).
- **Needs hardware:** Tier E on H1 (needs a 32 GB laptop and Nanbeige's fork build) and on H2, H3 and the server.
- **Needs people:** expert review and all team reviews.
- **Later or another team's work:**
  - L3 tasks (W5);
  - scaling Tier F to 150 and Tier A to 30 tasks after review;
  - topic normalization 930 → 14 (data team);
  - the LLM rephrase detector (needs API access).

## Verification (end-to-end)
1. `./gb test` is green. It covers the frozen-split reproduction, stats against known values (including Table 8), every ladder failure class, the rule cascade, agent conformance and FunctionModel replays.
2. `./gb build-all` run twice gives identical item hashes.
3. `./gb tier-f gates` passes: every first-set task at 10/10, ≥ 2 variants pass, mutant rejection ≥ 90%, and start states fail.
4. `./gb tier-a gates` passes: replays at 10/10, the do-nothing agent fails, wrong states are rejected. `./gb tier-a smoke` writes SFT-format traces.
5. `./gb audit` comes back clean on disjointness, contamination, canary, license and schema.
6. The file-tree manifests of `~/grc-agent` (including git-ignored files), `data/` and the documents are identical before and after.
7. `reports/compliance_review.md` has no open "fail" rows, and the README status table is filled in.

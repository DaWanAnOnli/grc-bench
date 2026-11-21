# GRC-Bench pre-registration (benchmark v1.0)

**Status: signed off by the team and declared at the freeze on Friday 20 November 2026 (benchmark
v1.0, git tag `v1.0`).** It is published on the Open Science Framework (OSF) before any fine-tuned
model is run on the test split. Any later deviation is reported in the 7 December report. Amendment 1
(21 November 2026, §11) changes the run list.

- **Team:** JJJP, University of Toronto MScAC: Joel Rompis (evaluation lead), Jiuyue Cai, Patricio
  Pizaña Vela, John Borland. Industry partner: Qoherent.
- **Plans this document follows:** the methodology proposal (`Methodology_Proposal_to_7Dec.docx`) and
  the benchmark evaluation plan for the CAD 504 budget case (`Benchmark_Evaluation_Plan_to_7Dec.docx`,
  cited as "eval plan §n"). Rows of eval plan Table 1 marked "set by budget" follow the CAD 504 case.
- **Companion files:**
  - `docs/preregistration/run_list.yaml`: the declared test-split run list (eval plan Table 6);
  - the analysis code, package `grcbench.stats` of the `grc-bench` repository, at the commit tagged
    `v1.0` (Appendix A lists its analysis choices);
  - the Tier-K question prompt and the judge prompt snapshot (Appendix B).

## 1. What this document fixes in advance

1. The hypotheses and the rule that decides each one (§3).
2. The metrics and the scoring of every tier (§5).
3. The runs on the test split: only the configurations in `run_list.yaml` (§6).
4. The analysis: estimators, intervals, tests, corrections and effect sizes (§7).

Before the freeze, the test split is run only by the base models and the outside models, and only
to compute item statistics for pruning (eval plan §5.3). No fine-tuned model is run on the test split
before this document is published.

## 2. The benchmark

GRC-Bench measures how well a language model develops GNU Radio flowgraphs with GRC-Agent, the
assistant inside GNU Radio Companion. It has four tiers.

| Tier | What it measures | Dev split (for every choice) | Test split v1.0 (frozen 20 Nov) | Built by G2 |
|---|---|---|---|---|
| K: knowledge | Knowledge of RF, signal processing and GNU Radio, shown in written answers to questions | 107 questions: the training team's validation questions that have no twin left in training and are not close to a test question | 285: the 185 frozen questions and 100 GNU Radio behaviour questions, each answered by running GNU Radio (the public reference set; any seed draws a fresh set of the same composition) | all 285 test questions and the dev questions; rubric templates (the prose is written and checked by the review agents, §10) |
| F: flowgraphs | Making a correct flowgraph edit for a stated goal: one documentation lookup and one edit inside GRC-Agent, judged by running the result | 45 tasks | 150 tasks (60 L1, 60 L2, 30 L3) | a first set of 30 (20 test, 10 dev), all gates passed, then reviewed with the rest (§10) |
| A: agent | Completing a realistic multi-step task with GRC-Agent's tools (8–20 tool calls), judged by the end state | 12 tasks | 30 tasks | 10 test and 12 dev tasks built, all gates passed; 20 test tasks drafted as specifications |
| E: speed and memory | Prompt and generation speed, time to first token and peak memory on the target hardware | — | — | the llama-bench protocol, tested on a small model |

**Splits and the test-run log** (eval plan §4.1).
- Every choice (checkpoints, data mixes, adapter placement, early stopping, the CI gate) uses the dev
  split.
- **The training data and the Tier-K dev split.** The training data is the training team's release,
  Qoherent cleaned dataset v3 (1 November 2026, archive SHA-256 `c0705e6b…121a50b`), with its own
  train, validation and test splits. The Tier-K dev split is the part of its validation split that
  passes the dev rule: 107 of its 148 Q&A questions. The other 41 belong to a twin group of
  training questions or are close to a frozen test question. `./gb audit` checks that the release's
  training files hold no Tier-K dev or test question and no canary.
- A run on the test split must name a configuration declared in `run_list.yaml`. The harness refuses
  any other test run.
- Every test run is appended to `runs/test_run_log.jsonl` before it starts, with its model, settings,
  code version, date and reason.
- Items are pruned before the freeze using only the answers of base models and outside models.

**Contamination controls** (eval plan §6).
- Every benchmark item is compared with the training data by 8-word overlap, TF-IDF similarity and
  embedding similarity. The rate is reported for each tier, and flagged pairs are judged by the review agents.
- Tier-K items with a near-twin in the training questions are reported separately from clean items.
- The functional signatures of every test and dev flowgraph are reserved: the training team removes
  any training flowgraph with a reserved signature.
- Every item record carries the canary string; the canary never appears in anything a model sees.
- Benchmark items are never sent to a model that creates training data.
- **Agent traces used for training.** Qoherent's 1,853 agent traces of 15–18 October (DeepSeek-V4.1-Flash
  in GRC-Agent's harness) are training data: the release puts 1,195 of them in its training file
  and 261 more in an optional training file, and holds 396 back for its own validation and test
  (1 is excluded). `./gb traces` checks all 1,853 against the
  benchmark (`reports/agent_traces.md`):
  - no trace has the goal or the start graph of a Tier-F or Tier-A task (the task-level check of eval
    plan §6);
  - their text is a corpus of the contamination audit;
  - they show the catalog entries of almost every block (port types, parameter options, port counts)
    but never ran a flowgraph. So Tier K's GNU Radio questions ask what a block does on a stated input,
    which only execution reveals, and none is about a block a trace built in GNU Radio code.
- **No block split.** Training data may use any GNU Radio block, including what it outputs for a given
  input. The GNU Radio questions come from a seed, and a fresh set of the same composition checks for
  memorisation of the published ones (§5.1, §7).

## 3. Research questions, hypotheses and decision rules

The research questions are those of eval plan Table 3. Each hypothesis is decided by the rule below.
Where the plan leaves a detail open, the rule cites the proposal of Appendix A that fixes it.

| Hypothesis | Statement | Decided by |
|---|---|---|
| **H1a** reliability | Model rankings are stable, the judge agrees with the reference labelling, and reference solutions are reproducible. | Per tier, all four parts must hold. (1) Kendall's τ between the rankings of two independent bootstrap resamples is at least 0.8 in at least 95 % of 1,000 pairs (`H1a-decision`). (2) The split-half τ, samples {1, 2} against {3, 4}, is at least 0.8. (3) The judge passes both calibration gates (§5.1, `judge-gate`). (4) Every remaining Tier-F and Tier-A reference passes 10 of 10 harness runs. Parts 1 and 2 rank the outside models together with the target models, so H1a is decided once the outside models have run (§6); until then parts 3 and 4 are reported on their own. |
| **H1b** validity | GRC-Bench measures GNU Radio development skill, not general ability. | The expert agent of the review protocol accepts at least 90 % of the reviewed items of each tier as correct and relevant (first pass; an item marked `unsure` counts as not accepted). This measures the expert agent's acceptance, not a GNU Radio engineer's (§10). The correlations of each tier with published BFCL and MMLU-Pro engineering scores across the panel are reported with intervals and described, not tested (eval plan §9), once the panel has run (§6). |
| **H1c** separation | Running the outputs separates models better than asking questions: Tier-F scores spread more widely, and more model pairs differ significantly, than Tier-K scores. | Tier K is subsampled to Tier F's 150 items. H1c holds if the 95 % bootstrap intervals of both SD(F) − SD(K₁₅₀) and sig_pairs(F) − sig_pairs(K₁₅₀) lie above 0, with the same test on both tiers (`H1c-test`, `H1c-mixed-n`, `H1c-decision`). H1c is decided once the outside models have run (§6). |
| **H2a** recipe | Mixed SFT (knowledge, trajectories and replay) beats single-source SFT on every tier. | On the test split, Tiers K and F: the best S1 checkpoint against each single-source comparison arm of the same model, with the primary paired test of each tier, Holm-corrected in the RQ2 family. H2a holds for a tier if every comparison favours the mixed recipe at the corrected 5 % level. Tier A is compared on the dev split only, because the arms are not run on Tier A of the test split (Table 6). H2a is decided only if the single-source arms are trained (§8, decision 9). |
| **H2b** preference stage | Defect-aware preference optimisation lowers the hallucinated-block rate by at least 30 % relative to SFT alone. | Tier F: the hallucinated-block rate of S1 + DPO and of S1 + DPOP against the best S1 checkpoint. H2b holds for a method if the relative reduction is at least 30 % and the paired test of the per-task rates is significant after Holm correction in the RQ2 family. H2b is decided only if the preference stage is trained (§8, decision 9). |
| **H3** curation | Adding validated repairs to PASS-only data raises accuracy and lowers the hallucinated-block rate at equal compute. | The curation comparison A2 runs on the dev split (eval plan §7.2): PASS + repaired against PASS-only, on Tier-K score and on the Tier-F hallucinated-block rate, paired tests, Holm-corrected in the RQ3 family. Because the dev split also served for model choices, H3 is reported with that limitation. H3 is decided only if the PASS-only comparison model is trained (§8, decision 9). |
| **H4a** compression cost | 4-bit compression costs more on agentic (A) than on knowledge (K) metrics. | For the final model of each architecture: the drop from its anchor (BF16 for Nanbeige, Q8_0 for North-Mini-Code, §8 decision 15) to Q4_K_M on Tier A minus the drop on Tier K, in percentage points, with a 95 % cluster-bootstrap interval (items resampled within each tier). H4a holds if the interval lies above 0 (`H4a-decision`, §8). Tier A's MDE at its 30 test tasks (23–28 points) is printed beside the result. |
| **H4b** speed | The MoE model (North-Mini-Code) reaches at least 80 tokens/s on the reference hardware at Q4_K_M. | Tier E on setup H2 (a laptop with a 4 GB GPU, MoE experts in CPU memory): the median generation speed of 5 warm llama-bench runs is at least 80 tokens/s, decided at each of 4K, 16K, 32K and 64K tokens of context. The result states the context range over which H4b holds ("holds up to 16K"); a context that cannot be run does not hold (`H4b-context`, §8). Setups H1 and H3 are reported beside it. |

## 4. Item quality before the freeze

- **Two-reviewer review** (the review protocol, `docs/guidelines/review_protocol.md`). Two Claude
  review agents (`claude-opus-5-5`), each in its own context and blind to the other, review every Tier-F
  task, Tier-A task and generated block question, and every sheet of `handoff/review_sheets/`, with the
  checklists and guides of `docs/`. A writer agent writes every Tier-K rubric, and two checker agents
  check it. The expert agent decides every disagreement and audits a seeded 10 % sample of each
  sheet's agreements. The decision rules are code (`src/grcbench/review/decide.py`). Every factual
  claim is checked by a quote or a run, and every agent run is declared in `reviews/review_log.jsonl`.
- **Expert review.** The expert agent reviews every GNU Radio question and every Tier-F and Tier-A
  task for correctness and relevance (`docs/checklists/expert_review.md`).
- **Automated gates.** Every Tier-F and Tier-A reference passes 10 of 10 harness runs. Every Tier-F
  oracle rejects at least 90 % of the mutants that break the specification. Every Tier-A grader
  rejects a deliberately wrong end state, and an agent that does nothing fails every task.
- **Pruning.** An item is removed only for a reason a person can state (a wrong answer, ambiguous
  wording, a broken oracle), never because of its statistics alone. Item difficulty and discrimination
  from the base and outside models only flag items for a second look. The number of removed items and
  the reasons are published.
- **Who wrote what** (eval plan Table 5). The rules of Table 5 apply: Mistral models are not scored on
  Tier K; the judge's family and the family that drafts rubrics are not scored on Tier K; a teacher's
  Tier-K score is also reported without the questions its family wrote. Claude (Anthropic) drafted the
  Tier-F and Tier-A tasks, their reference solutions and oracles, and the settings and wording of the
  GNU Radio behaviour questions, whose answers come from GNU Radio runs. Claude review agents review
  and decide every item, and a Claude writer agent writes the Tier-K rubrics (§10). Anthropic models
  are not on the panel.

## 5. Scoring and metrics

### 5.1 Tier K

- **How a question is posed.** Tier K is closed-book. A model under test gets one message: a fixed
  instruction and the question, with no GRC-Agent system prompt and no tools
  (`grcbench.tier_k.answer_prompt`, version and hash in Appendix B). The instruction asks for a closing
  line "Final answer: …" and for the unit after every number, and it states that a number without a
  unit is never read as decibels. A reply is limited to 16,384 tokens, reasoning included. B1 places
  the retrieved documentation before the same message.
- **GNU Radio behaviour questions.** Each of the 100 asks for one number: the k-th output of a block set
  as stated on a stated finite input, or how many items it produces. The key is GNU Radio's own output.
  The generator runs the block twice in the sandbox and reads the recording, and keeps a question only
  when the runs agree, an independent model of the block's documented behaviour gives the same number,
  and the rule cascade scores the key 1 and a wrong value less (`./gb blockq`;
  `reports/tier_k_block_questions.md`). Floats are accepted within 1 % (with an absolute margin that
  shrinks near zero), integers and counts exactly.
- **Reference set and fresh sets.** The 100 questions are the reference set (seed
  `grc-bench-blockq-v3-2026-10-23`). Any other seed gives a fresh set with the same blocks, the same
  number of questions of each kind per block and none of the reference set's instances; each fresh
  question is paired with the reference question in the same block slot (`./gb blockq --seed S
  --like-reference`; `reports/tier_k_fresh_sets.md`). The fresh set of this study comes from a private
  seed with SHA-256 `4426ac05b766f88634056a9e90550c75a7bb08b1fbd9cf3c749e0f8b9c24ca28`; the seed is
  revealed with the results.
- **Rule cascade first, judge second.** Every answer is parsed by the rule cascade (`grcbench.tier_k.rules`):
  numbers with units and tolerances, option letters, block identifiers and symbols. Numeric answers
  earn tiered credit by relative error, and a wrong unit or order of magnitude earns 0. Only answers
  the rules cannot decide go to the judge.
- **Decibel values** are graded on the difference in dB: by default within 0.1 dB for full credit and
  about 0.27 dB for half (about 2.3 % and 6.4 % in power), against 1 % and 5 % for linear values. This
  is deliberately somewhat more lenient: 0.1 dB is the usual precision of a decibel value, and an
  answer correctly rounded to it errs by up to 0.05 dB. For voltage levels (dBV, dBmV, dBµV), whose
  decibels are 20·log10 of a voltage, the half band ends at about 0.44 dB (1.2 % and 5.2 % in
  voltage).
- **One reading for every number.** A number the rules could read two ways is read by fixed
  conventions (`grcbench.tier_k.rules`, P4, P8 and P11):
  - against a decibel reference a bare number is a linear ratio (1000 is 30 dB for a power ratio);
    against any other reference it is read in the reference's unit, with no other reading;
  - a decibel answer for a plain ratio, a ratio (%, "1000 times") for a decibel value and an angle
    for a plain number are converted (as a power or an amplitude ratio, as the rubric records) and
    graded, unless the question fixes the form: then the other form scores 0;
  - a loss word next to a positive value ("a loss of 3 dB", "10 kHz below the carrier") makes it
    negative when the reference is negative or the rubric marks the quantity as a gain;
  - Vpp and Vpk convert exactly; an RMS voltage converts only when the rubric records a sine;
  - against a reference with no unit, a GNU Radio item word after a number leaves the number, and
    bits, bytes and symbols count items ("34 vectors", "34 short items" and "34 bytes" are 34 items);
  - whatever still holds two readings of different credit gets the lower one (the tie-break).
- **The judge.** MiniMax-M3 grades the remaining answers against the item's rubric, with the reasoning
  text removed and without knowing which model wrote the answer (listed model and vendor names are
  masked before the judge reads the answer). It returns a verdict per rubric point and the values it
  reads for each graded number, which the rules grade (a value in a unit the rules do not read, such
  as "3.84 Mcps", earns 0 for that number); the verdicts are turned into a score in [0, 1].
  Every request uses temperature 0 and a fixed seed. A GNU Radio behaviour question that the rules
  cannot read goes to the judge with a numbers-only rubric built from its key. The prompt version
  and its hash are fixed in Appendix B; any change needs a new version.
- **Judge calibration** (eval plan §8.1; `docs/guidelines/judge_calibration.md`). Before the judge
  grades any test answer, it is compared with a reference labelling of 150 answers by three annotator
  agents, adjudicated by the expert agent (§10). The 150 answers come from the main runs' Tier-K test
  answers of B0, B1 and the fine-tuned models, drawn once the fine-tuned models' runs are logged
  (§10). The judge passes only if both gates hold: Cohen's κ on the per-point
  credits is at least 0.7 with the lower end of its 95 % interval at least 0.6, and the linear-weighted
  κ on the answers' score bands meets the same thresholds. Scott's π and the mean judge–reference gap
  are reported, for each task type too. If MiniMax-M3 fails, Kimi K2.6 is calibrated with the same
  prompt, and both results are reported. In week 8, 50 more answers from the final models are labelled
  and the judge–reference gap is reported for each model.
- **Every answer is scored automatically** (§8, decision 7); no answer is scored by a person. If the
  judge returns no valid verdict in 3 attempts, the fallback judge (Kimi K2.6) is asked once; if it
  also fails, the answer scores 0. A request that fails (an HTTP error or a timeout) is sent again; it
  never counts as a failed reply, and an answer whose requests keep failing is not scored and is listed.
  An answer that carries the benchmark's canary scores 0 and is
  counted. Each model's report gives, next to its Tier-K score, the number of answers decided by the
  tie-break, with a number scored 0 for the wrong form, scored 0 after failed judge replies, with a
  number scored 0 for a unit the rules do not read, and sent to the judge with masked names.
- **Limits.** Answers longer than 16,384 tokens are cut and scored as wrong. Reasoning text is removed
  before scoring.
- **Reported groups.** Results are given by the cleaned task-type labels only, with no group of fewer
  than 20 items: `gnuradio_usage` is reported with `design` as one group, design + gnuradio_usage
  (§8, decision 4). Factual recall and reasoning are reported separately on the 185 frozen questions,
  and the 100 GNU Radio behaviour questions as a group of their own (§8, decision 5). Clean items and
  items with a near-twin in the training questions are reported separately.

### 5.2 Tier F

- **Setting.** A Tier-F sample is a short episode in GRC-Agent's own harness, with its real system
  prompt and tool schemas. The history already holds one `inspect_graph` call on the start graph. The
  model may make at most one `query_knowledge` call and must then submit one atomic `change_graph`
  call, within 3 model turns. GRC-Agent applies the edit; a rejected edit is not retried.
- **B1** (§8, decision 12). B1 on Tier F is oracle grounding: GRC-Agent's catalog entry for every block
  of the task's reference solution is placed before the instruction, as B1 places the retrieved
  documentation before a Tier-K question. The lookup, the edit, the caps and the grading are B0's. The
  model is told which blocks the reference uses, so the result shows how far a model gets once it knows
  the right blocks, not how much retrieval helps, and it is reported as oracle grounding (§10).
- **Tool calls.** Every model served by llama.cpp runs with the server's default tool grammar
  (`--jinja`), which holds tool calls to the tools' JSON schemas (§8, decision 1). API models use the
  provider's tool calling, with strict mode where it is offered; their well-formed-call rates are
  therefore not compared like for like with the llama.cpp models'. For the llama.cpp models, V1 and
  the well-formed-call share mostly check the server rather than the model; the
  hallucinated-block rate stays informative, since the grammar fixes a call's form but not the block
  names in it, and V2–V5 compare as usual.
- **Families and levels** (§8, decision 8). Modify changes the graph's structure or function;
  parameter-tune changes values only, and its checks require the start graph's blocks. L1: at most 5
  functional blocks, one sample rate and no modulator–demodulator chain. L2: 6–15 blocks, or a change
  of sample rate, or modulation or demodulation. L3: message ports, tags or PDUs.
- **Validity ladder.** V1 a well-formed `change_graph` call; V2 the graph loads with only real blocks;
  V3 it compiles (it validates and its Python code is generated); V4 it runs and every observation
  point receives data; V5 the task's oracle passes (signal checks and structural checks). Each check runs only if the previous one
  passed.
- **Metrics.** The pass rate at each rung, shown as a funnel; Exec@1 (reaching V4) and Func@1
  (reaching V5, over the tasks that have a V5 oracle), with the unbiased pass@k estimator; the
  hallucinated-block rate (the share of samples whose edit names a block that does not exist in the
  pinned GNU Radio catalog); a partial-credit score over blocks, parameters and connections. Answers
  are limited to 8K tokens (8,192 per reply, sent with every request), reasoning included; a reply that
  reaches the limit fails the sample.
- **Perturbed examples.** Scores on tasks built from public GNU Radio examples (numbers changed) are
  also reported separately from scores on composed tasks.

### 5.3 Tier A

- **Harness, the same for every model.** The pinned GRC-Agent version (commit 6c98128) with its real
  system prompt and 22 tools; tool outputs cut to 4K tokens (beginning and end kept, plus error lines);
  64K tokens of context, or the model's own context if shorter; at most 40 model requests; 30 minutes per
  episode (15 minutes for a Tier-F sample), with GRC-Agent's HTTP client waiting for the model server as long
  as that (§10); a time limit on each flowgraph run; web tools
  offline; the shell in a sandbox without network. A run that hits a cap is a failure. The episode
  process is limited to 8 GiB of memory and one hour of CPU time, and each shell command to 4 GiB and ten
  minutes of CPU time (§8, decision 13). A shell command stopped at its limit returns its exit code to
  the model; a tool that runs out of memory in the episode process ends the episode as an agent error,
  and its end state is graded. Tool calls as in Tier F: the llama.cpp server's default tool grammar for every model it serves, the provider's
  tool calling (strict mode where offered) for API models (§8, decision 1). For the llama.cpp models
  the share of well-formed calls therefore mostly checks the server, as V1 does in Tier F.
- **Scripted user.** Set replies for the questions a task expects; otherwise "please use your best
  judgement".
- **Grading by end state.** The final saved flowgraph is re-run by the grader through the Tier-F
  ladder and checked against the task's assertions. Knowledge and refusal tasks check the answer
  against facts the grader works out itself: from the graph, from GNU Radio's block definitions or from
  the files the run wrote. A fact with a label is read from the first line of the answer that names its
  label, and a number must be the first one after the label. An empty or missing answer never counts.
- **Metrics.** Task success and pass^k over k runs; the share of tool calls that are well-formed and
  match the tool's schema; calls to tools that do not exist; correct refusals; first-attempt success
  against recovery after an error; steps, tokens and time per task.
- **Invalid episodes.** An episode that ends with an infrastructure error (the model server or the
  harness failed, not the model) is not scored and is re-run (§8, item 10). A reply the model server
  cannot parse (llama.cpp reports that the output does not match the model's tool-call format) is the
  model's failure, not an infrastructure error; the tokens it generated are recorded from the server's
  own count, and in Tier F a reply that used up the 8K limit counts as reaching the answer limit.

### 5.4 Tier E

For each compression level (BF16, Q8_0, Q5_K_M, Q4_K_M): prompt-processing and generation speed,
time to first token and peak RAM and GPU memory, at 4K, 16K, 32K and 64K tokens of context, measured
with llama-bench as the median of 5 warm runs, on H1 (CPU-only laptop), H2 (4 GB GPU, the reference
for H4b) and H3 (Qoherent's RTX 3090). BF16 and Q8_0 of North-Mini-Code are measured on the rented
server GPU.

### 5.5 What every run stores

The full prompt and output of every item, the tool calls, the harness logs, the score at every rung,
the tokens and the time, and the run's GPU-hours. All raw outputs are released with the benchmark.

## 6. The declared run list

`run_list.yaml` lists the 39 configurations that may run on the test split (eval plan Table 6), with
their model, precision, tiers, samples per item (n) and Tier-A runs per task (k), and the hypotheses
they serve. The main runs are 12 configurations: each target model's base model (B0, and B1 on Tiers
K and F), its model fine-tuned on all the data, and that model's compressed versions (§8, decision 9).
The 7 December report compares the fine-tuned models with their base models (RQ2) and answers RQ4 on
them. The other 27 configurations are marked `later: true` and run after the main runs if time
allows: North-Mini-Code in BF16, the training variants outside the main runs and the outside models
(§10). In summary:

- **Nanbeige4.2-3B** (BF16 unless stated). Main runs: B0, B1, the best S1 checkpoint (the final
  model) and the final model compressed to Q8_0, Q5_K_M and Q4_K_M. Later: S1 + DPO and S1 + DPOP,
  seeds 2 and 3 of the final setting, and four comparison arms (knowledge-only, trajectories-only,
  the other LoRA placement, and the arm without general data).
- **North-Mini-Code**, compared at the 4-bit deployment precision with Q8_0 as the anchor (§8,
  decision 15). Main runs: B0 at Q8_0 and Q4, B1 at Q4, and the final model (the best S1 checkpoint) at
  Q8_0, Q5_K_M and Q4. Later: B0 and the final model in BF16, which need a GPU with at least 80 GB; the
  S1 checkpoint as its own Q4 configuration (the same model as the final model at Q4 while there is no
  preference stage), seeds 2 and 3, and one comparison arm (attention-only adapters).
- **References, run later:** Qwen3-30B-A3B at Q4 and the B2 cloud teacher, DeepSeek-V4.1-Flash.
- **The panel, run later** (eval plan §7.3): six small open models, four open teachers and one closed
  frontier model. Its slots are declared now, with the rules for their members (§4 and eval plan §7.3).
  The members are chosen when the panel is to run, and are entered in `run_list.yaml` in one dated
  amendment before the first panel run; the test-run guard refuses a slot with no member.

Every configuration uses n = 5 samples per Tier-K and Tier-F item and, where it runs Tier A, k = 4
runs per task, except the extra seeds (n = 3, k = 2), the comparison arms (n = 3), North-Mini-Code's
BF16 B0 (k = 2) and the closed frontier model (n = 1, k = 2).

**BFCL** (eval plan Table 6, RQ2; §8, decision 16). The configurations marked BFCL in `run_list.yaml`
(in the main runs, each target model's base model and its fine-tuned model) run the Berkeley
Function-Calling Leaderboard with its own package, `bfcl-eval` 2026.3.23 (BFCL v4), on its non-live and
live categories: 3,641 single-turn cases that need no web service. The multi-turn categories are left
out for time, the agentic ones because they call web services. The model is the configuration's
llama.cpp server, called through BFCL's OpenAI-compatible chat handler with native tool calls (the
server's default tool grammar, as in B0), at BFCL's own temperature 0.001, with each reply limited to
16,384 tokens (`./gb bfcl`). BFCL's own checker scores the replies; its non-live and live overall
accuracies are reported for each base model beside its fine-tuned model, described, not tested.

**Engines and settings** (eval plan §7.1). All open models run in llama.cpp b12020 from GGUF files,
BF16 references included. Nanbeige4.2-3B is served with the team's chat template
(`serve/templates/nanbeige42.jinja` in the team's evaluation repository, SHA-256
`3d5fb57d879e88549d48211ba990eb7471084891b0af9cd8e6576351964981b2`): the model's own template, with
the system messages at the start of a conversation merged into one, because its own template refuses
GRC-Agent's two leading system messages. With one system message the prompt is unchanged.
Each model uses its recommended sampling settings; greedy decoding is
checked on the dev split only. North-Mini-Code at Q8_0 keeps the experts of its first four layers in CPU
memory (`--n-cpu-moe 4`), so that it fits the 32 GB GPU of the main runs. Tier-K answers may be
generated through several server slots at once, each holding at least the answer limit plus 4,096
tokens (and the B1 documentation budget for B1); Tier-F and Tier-A episodes run at most two at a time,
each slot holding the full 64K context (§8, decision 14). The slots and the layer placement change the
speed, not the requests. Both target models keep an 8-bit KV cache in every Tier-A run; a check
on 10 tasks compares it with a 16-bit cache. The 4K tool-output cap is checked against GRC-Agent's
default settings on 10 tasks.

## 7. Analysis

- **Estimates.** pass@1 from n samples with the unbiased estimator; pass^k from k runs; Tier-K
  scores as mean credit.
- **Uncertainty.** Means with 95 % intervals. Items from one question family or one source flowgraph
  share a `cluster_id`; standard errors are clustered (CR1, t with G − 1 degrees of freedom) and
  checked with a cluster bootstrap (10,000 resamples, seed 20261014).
- **Paired comparisons** on shared items. Tier K is a score tier: Wilcoxon signed-rank on per-item
  mean scores. Tiers F and A are pass/fail tiers: McNemar on items reduced to the majority of their
  samples (a tie fails). With clusters, both tests take their exact cluster sign-flip form
  (`clustered-tests`). Each comparison has one primary p-value (`primary-test`), and the effect size is
  reported next to it (Cohen's h or the matched odds ratio; the paired difference with its interval).
- **Corrections.** Holm–Bonferroni within each research question: RQ2 (H2a and H2b, and every
  comparison of a trained model with B0 and B1), RQ3 (H3) and RQ4 (every compression level against
  BF16, and H4a). If configurations marked `later: true` add comparisons to a research question after
  the 7 December report, the correction is applied again over the whole family, and both sets of
  adjusted p-values are reported.
- **What the benchmark can detect** (eval plan Table 8: 80 % power, 5 % significance, paired, 20–30 %
  of items decided differently). The MDE is printed beside every claim, with a design-effect
  adjustment when items are clustered (`mde-design-effect`). It is computed at the v1.0 test items, the
  ones the review kept (§10); `./gb stats mde` prints both rows.

  | Tier | Test items in Table 8 | MDE in Table 8 (points) | Test items in v1.0 | MDE in v1.0 (points) |
  |---|--:|---|--:|---|
  | K | 285 | 7–9 | 277 | 8–9 |
  | F | 150 | 10–13 | 148 | 10–13 |
  | A | 30 | 23–28 | 30 | 23–28 |

  A paired difference cannot exceed the share of items the two models decide differently. At its 30
  test tasks, Tier A's MDE is 23 points at Table 8's 20 % and 28 points at its 30 %: larger than the share
  of tasks decided differently at 20 %, and smaller at 30 %. So under Table 8's assumptions a paired
  Tier-A comparison reaches 80 % power only when more than about 26 % of its tasks are decided
  differently. Tier-A differences are reported with their intervals and this MDE, and a non-significant
  Tier-A comparison is not read as evidence of no difference.

- **Seeds.** If time allows, the final setting of each model is trained with 2 more seeds (§8,
  decision 9); the spread between the 3 seeds is then reported separately from the item-level interval.
- **Memorisation check (Tier K).** The headline configurations (B0 and the final model of each target,
  and B2 when it runs) also answer the private fresh set. The gap is the reference-minus-fresh mean item score over
  the paired slots, clustered by block, with both 95 % intervals; memorisation is flagged when both lie
  above 0 (`fresh-set-gap`). The headline Tier-K score stays the reference set's.
- **Item analysis.** Difficulty and discrimination for every item; a Rasch model as an exploratory
  check only (`rasch-solved-min`).
- **Error analysis.** Two annotators label failures with the set list of error types
  (`docs/guidelines/error_analysis.md`); their κ is reported.
- **Exploratory analyses.** Anything not listed here, and any run outside `run_list.yaml`, is
  labelled exploratory and kept out of the main claims. One is planned: Tier-A success of the trained
  models by category, split by whether the training traces cover the category's tool pattern. The
  traces never save a block to the library or write files, and they never ran a flowgraph
  (`reports/agent_traces.md`).

## 8. Decisions at sign-off

Decisions 1–7 were taken by the evaluation lead on 27 October 2026, decisions 8–11 on 19 November 2026
and decisions 12–16 on 20 November 2026; the rule cascade, the judge template, the statistics package, the label groups, the Tier-F
items and the run list implement them.

1. **B0 and the tool grammar.** B0 is the base model served with the default tool grammar
   (`--jinja`) on Tiers A and F, as is every later stage of every model llama.cpp serves; there is
   no separate grammar-constrained baseline (§5.2, §5.3, §10). The cloud teacher is B2.
2. **The analysis proposals** in Appendix A are accepted as written, including the strict bootstrap
   rule of H1a (95 % of pairs at τ ≥ 0.8, not the median τ).
3. **H4a and H4b.** H4a as proposed (`H4a-decision`). H4b is decided at each of 4K, 16K, 32K and 64K
   tokens of context, and the result is stated as a range (`H4b-context`, §3).
4. **The small `gnuradio_usage` group** (11 frozen test items) is reported with `design`: one group,
   design + gnuradio_usage, of 49 frozen test items (`reports/labels.md`).
5. **Recall and reasoning,** on the 185 frozen questions: recall is `factual` (30 items); reasoning is
   `reasoning`, `calculation`, `diagnosis`, `design` and `gnuradio_usage` (155 items). The 100 GNU
   Radio behaviour questions are a group of their own.
6. **Open scoring findings** (`reports/scoring_reviews/round2_status.md`): every finding is resolved
   in the rule cascade, the judge template or the statistics package, with two exceptions. Rules
   finding F7 stands as a rule: decibel values are graded on the difference in dB (§5.1). Finding
   F5, the rule cascade's deferral rate on dev answers, stays open until the dev split has rubrics
   with graded numbers (below).
7. **No human route.** Every Tier-K answer is scored automatically, by the rules' fixed conventions
   and the judge, with the fallback judge's single attempt and a score of 0 after failed replies
   (§5.1).
8. **Tier-F decisions** of `reports/tier_f_reviews/fix_plan.md` §7. The family boundary and the
   level rule are as §5.2 states them: L1 has at most 5 functional blocks, one sample rate and no
   modulator–demodulator chain, and `gbf-test-06` (control through a message port) is L3. Tasks
   composed by hand or by the generator, rather than converted from GNU Radio examples, are recorded
   in §10, and so is the canary policy: the item record carries the canary. Tasks derived from GNU
   Radio examples keep their GPL-3.0-or-later attribution in their licence field. `gbf-test-05` keeps
   its documented limit, an output check that a hard-coded output could pass. The role-table
   declaration (§4, §10), the graph checks of `gbf-dev-01` and `gbf-dev-07`, and the separate faults of
   `gbf-dev-10` and `gbf-test-17` are in the items.
9. **Training variants.** For the main runs each target model is trained once, by supervised
   fine-tuning (SFT) on all the data, so there are two arms per model: the base model and the
   fine-tuned model, whose best S1 checkpoint is the final model. The preference stage (S1 + DPO,
   S1 + DPOP), seeds 2 and 3, the comparison arms of the methodology proposal's Table 7 and the
   curation comparison of H3 are not part of the main runs. Their configurations stay declared with
   `later: true` (§6), and H2a, H2b and H3 are decided only if the variants they need are trained.
10. **Infrastructure re-runs.** An episode that ends with an infrastructure error is re-run up to
    twice; if it fails a third time, it is scored as a failure and listed. A crash of the model server
    or an exception in the harness is an infrastructure error; reaching a cap, a malformed tool call
    and a reply the server cannot parse are the model's failures (§5.3).
11. **The code version:** the commit tagged `v1.0` in the public `grc-bench` repository, which this
    document cites for the analysis code; the manifest of the OSF upload gives its hash.
12. **B1 on Tier F** is oracle grounding (§5.2). Every Tier-F sample, B0 included, already makes its
    one lookup through GRC-Agent's own search, so B1 adds the catalog entries of the reference
    solution's blocks before the instruction.
13. **Resource limits** of an episode, as the harness sets them (§5.3): 8 GiB of memory and one hour of
    CPU time for the episode process, 4 GiB and ten minutes of CPU time for each shell command.
14. **Parallel server slots** (§6): Tier-K answers through several slots, each holding a full answer;
    Tier-F and Tier-A episodes at most two at a time, each with the full context.
15. **North-Mini-Code's anchor is Q8_0** (§6, §10). Its BF16 configurations need a GPU with at least
    80 GB; they stay declared, marked `later: true`.
16. **BFCL** runs on each target model's base model and on its fine-tuned model, with the pinned package,
    categories and settings of §6.

Work that remains before any test run: the judge calibration (§5.1), which needs API access to
MiniMax-M3 and produces the record that the judge runner (`./gb tier-k judge`) requires before it
grades a test answer; and dev-split rubrics with graded numbers, so that the rule cascade's deferral
rate on dev answers (round-2 finding F5) can be measured.

## 9. Fixed at the freeze (20 November)

- The v1.0 items of every tier, after the two-reviewer and expert reviews and pruning, with the number
  of removed items and the reasons (`reports/review_summary.md`).
- The run list: the two target models' configurations as the main runs, and the outside models'
  configurations marked to run after them (§6).
- The MDEs recomputed for the final item counts.
- One amendment of this document, if needed. Every later deviation is reported in the 7 December
  report.

## 10. Known differences from the approved plans

- **Environment.** GNU Radio 3.10.12 runs natively in WSL (Ubuntu) instead of a Docker image;
  `docker/Dockerfile` records the intended image, and `./gb env` records the versions.
- **Tier-F format.** Tier F runs inside GRC-Agent's harness as one lookup and one edit, because
  GRC-Agent never lets a model write a `.grc` file; the evaluation plan's Tier-F text is to be updated.
- **Who drafted the tasks.** Data Summary §11 names other authors for three kinds of item. Tier-F
  instructions (§11 A, DeepSeek-V4.1-Flash) and Tier-A tasks (§11 D, team members with no AI
  involved) were drafted by Claude, and Claude review agents review and decide every one; this is
  raised with Qoherent. The 100 GNU Radio test questions (§11 C, an AI model) are generated by a script, with
  answers from GNU Radio runs.
- **GNU Radio questions answered by execution.** Eval plan §4.2 reads the answers of the generated
  GNU Radio questions from the block catalog. The agent traces used for training show those catalog
  entries, so the questions ask instead what a block does on a stated input, and the key comes from
  running it (§5.1).
- **Agent traces** were collected on GNU Radio 3.10.9.2, where flowgraph execution was not available.
  The benchmark runs on 3.10.12 with execution.
- **Web tools** return a fixed "offline" message in Tier A.
- **The model server's time to answer.** GRC-Agent's HTTP client gives up on a request after 120 seconds
  without data from the model server. In Tiers F and A it waits as long as the episode's time limit
  instead (§5.3). Long reasoning can pass 120 seconds without output: in the dev-split run of 28 October
  on the RTX 5090, 3 of 10 Tier-F samples ended that way, and on a Colab T4 both Tier-F samples of the
  pipeline test. With GRC-Agent's value, a model's Tier-F and Tier-A results would depend on how long it
  reasons and on the speed of the GPU it runs on, not only on the pre-registered limits.
- **The Tier-K dev split.** Eval plan §4.1 (Table 4) holds 200 of the 1,668 training questions back
  as the dev split. Here the dev split comes from the training team's validation split, so that
  training and GRC-Bench hold back the same questions (§2): 107 questions. Every dev-split choice,
  and the curation comparison A2 of H3, rests on those 107 questions, and the anchor (60) and CI
  (20) subsets are drawn from them.
- **The Tier-K answer limit.** The evaluation plan as approved at gate G1 limits Tier-K answers to 4K
  tokens (§7.1). Here the limit is
  16,384 tokens, reasoning included. On the dev split of 14 October (200 questions) the smoke model
  reached 4,096 tokens on 48 of 200 answers while still reasoning; asked again with the larger limit, 9 finished between 4K and 16K
  tokens, 23 within 4K and 16 not at all.
- **The grammar-constrained baseline.** The methodology proposals and the evaluation plan as approved at
  gate G1 (Table 1) list a baseline with grammar-constrained tool calls on Tier A. In llama.cpp b12020 the server's default already constrains tool calls, so
  B0 is that setting and no separate baseline runs; GRC-Bench does not measure the grammar's
  contribution, or whether fine-tuning alone makes tool calls well formed. The cloud teacher upper
  bound, numbered 3 in the methodology proposals and in the evaluation plan as approved at gate G1,
  is B2 here.
- **L3 tasks** of Tier F are added after G2 (week 5), and Tier F and Tier A scale from their first
  sets only after the team's decision on them.
- **The language-model paraphrase detector** of the contamination audit waits for API access; the
  8-word, TF-IDF and embedding checks run now.
- **Review by agents** (`docs/guidelines/review_protocol.md`). Eval plan §5.2–§5.4, §8.1 and Tables
  1, 5 and 7 give the reviews to team members and a Qoherent expert. Here Claude agents do all of
  them:
  - two independent review agents per sheet;
  - a writer agent and two checker agents per rubric;
  - an expert agent that decides every disagreement, audits the agreements and does the expert review
    of §5.3.

  The parts of the plan this changes:
  - **Table 5.** Claude drafts the Tier-K rubrics, although Claude labelled part of Qoherent's data and
    Table 5 keeps every such family from drafting references. Rubrics written by Claude may favour
    answers that read like Claude's, and so the models trained on those labels. Every rubric point
    therefore rests on a quoted source or an executed check, and every rubric is checked for word
    8-grams shared with the question's teacher answers. Anthropic models are not scored.
  - **Table 1 and §5.3 (H1b).** The expert is an agent. H1b reports the expert agent's acceptance,
    which is not evidence of a GNU Radio engineer's acceptance.
  - **§8.1 (H1a, part 3).** The judge is calibrated against agent labels. Its κ measures agreement
    with Claude's labels, not with people's.
  - **Agreement statistics.** The two reviewers are two runs of one model. Their κ measures that
    model's consistency and overstates the agreement two people would reach.
  - **Blinding.** Blinding is by what each agent's brief allows it to open, not by file permissions.
  - **Table 7.** The person-hours of the reviews, the rubrics, the calibration and the error analysis
    become declared agent runs (at most 38 for the reviews before the freeze: 30 in the first round,
    8 in the second).
  - **Similarity sheets.** Both reviewers label every contamination pair, not reviewer 2 on a sample,
    and the agreement is reported on every pair and on the every-fifth sample.
- **The freeze date.** The approved plans freeze benchmark v1.0 at gate G3 on Friday 13 November. The
  review by agents (above), and the second round for the replacement tasks (below), need until Friday
  20 November, so the freeze is then. Gate G4 (27 November) and the 7 December report keep their
  dates.
- **Items the review removed or rewrote** (eval plan §5.3: removed and counted). The test split of
  §2 is counted before the review. A removed item keeps its record, with the reason, and no run uses
  it (`reports/review_summary.md` lists each one).
  - **Tier K.** 8 of the 185 frozen test questions are removed as underspecified (4 `calculation`,
    2 `reasoning`, 2 `gnuradio_usage`), so 277 test questions are scored: 177 frozen and 100 GNU Radio
    behaviour questions. The reported groups of decisions 4 and 5 keep their rule with these counts:
    recall 30, reasoning 147, design + gnuradio_usage 47, all at least 20. 3 of the 107 dev questions
    are removed, so 104 remain; the anchor subset keeps its 60 questions and the CI subset 19 of its 20.
  - **Questions that name an API GNU Radio 3.10 does not have.** Dev question `q_26bef0bdbcea7324`
    gets a version 2 with the real method name, and the original text is kept. Frozen test questions
    are never changed in place, so test question `q_ba378ee64a7cddef` stays as written. Its rubric
    grades it as a premise-correction item: a point for spotting the name, an error for accepting it.
  - **Tier F.** 23 of the 195 tasks are rejected: 22 duplicates (tasks of one shared cell that make the
    same edit) and `gbf-test-22`, whose requested change has no effect on its output. So 129 test and
    43 dev tasks remain. The test levels are L1 25, L2 74 and L3 30, not the planned 60, 60 and 30:
    18 modify tasks with more than 5 blocks are relabelled L2 (eval plan §4.3: L1 has at most 5
    blocks), and all 23 rejected tasks are L1. Results are reported per level, at the counts after the
    replacements below.
  - **Tier A.** 7 of the 42 tasks are rejected (5 test, 2 dev): all 4 knowledge tasks and 3 of the 4
    save_block tasks, because a competent agent finishes them in 1 to 3 tool calls, below the
    8-call floor of eval plan §4.4. So 25 test and 10 dev tasks remain, and the test split keeps no
    knowledge task and one save_block task. The replacements below fill both categories again.
  - **Replacement tasks for Tiers F and A.** Each rejected Tier-F and Tier-A task is replaced by a task
    built after the review and reviewed in a second round under the same protocol: two reviewers, the
    expert agent with its audit, and its expert review. The second round uses 8 more declared agent runs.
    - Tier F: one new task per rejected task, in its family. It is L1 where a cell has room for an L1
      edit no other task of the cell makes, and L2 otherwise. There are 3 test spares and 1 dev spare:
      `gbf-test-151` to `gbf-test-174` and `gbf-dev-46` to `gbf-dev-48`.
    - Tier A: 3 knowledge and 2 save_block test tasks, and one of each in dev, built to need at least
      8 tool calls, plus one spare of each category: `gba-test-31` to `gba-test-37`, `gba-dev-13` and
      `gba-dev-14`. The knowledge tasks are graded with labelled facts (§5), so an answer that gives
      the right values to the wrong items fails.

    The replacements are numbered before the review. The first ones it accepts, in id order, fill the
    slots: per split for Tier F, and per category and split for Tier A. An accepted replacement beyond
    the slots is kept as a reserve and not scored.

    Both reviewers rejected 6 Tier-F replacements as duplicates of first-round tasks of their cells
    (`gbf-test-151`, `-153`, `-154`, `-156`, `-164` and `gbf-dev-48`), and the expert agent's review
    agreed on all six. The other 21 are accepted: 19 test replacements fill 19 of the 21 test slots, and
    2 fill the dev slots. So Tier F has 148 test tasks (L1 33, L2 85, L3 30), 2 fewer than the planned
    150, and 45 dev tasks. All 9 Tier-A replacements are accepted, and `gba-test-36` and `gba-test-37`
    are reserves. So Tier A has 30 test tasks, 3 in each of its 10 categories, and 12 dev tasks, as
    planned.
  - **Edits after the review.** The second round found three faults that first-round tasks share with
    their replacements, and the team decided on 17 November to apply the same edits to the first-round
    tasks. An edited task's version goes up by one (to 2, or to 3 where the review had already edited
    it), its status is `approved-with-edits`, its notes name the edit as a post-review edit, its gates
    are re-run, and the expert agent of the second round checked that each edit is applied and closes
    its gap.
    - Seven Tier-F build tasks (`gbf-test-24`, `-37`, `-94`, `-114`, `-134`, `-135` and `gbf-dev-33`)
      passed a graph that leaves out the tone their filter removes. They get the replacements' two graph
      checks: a count of Signal Sources, and a path through the Add block.
    - In `gba-test-24` the scripted user answered a question about the category with "Yes, go ahead.".
      Its rules take the replacements' order, so the question gets "Use the default.".
    - Fifteen Tier-A tasks check the File Sink's file name with a pattern on the parameter's text, which
      fails a name set through a variable. The grader matches such a pattern against the evaluated value
      too (grader version 4), as it already does for a plain name, so no task file changes.

    Tier K's removed questions are not replaced. Its frozen and dev questions come from the training
    team's held-out splits, and any new question from that source would overlap the training data.
  - **Sizes stated elsewhere in this document.** They are the sizes before the review. The analyses use
    the items that remain: H1c subsamples Tier K to Tier F's 148 test tasks, and the minimum detectable
    effects of §7 are computed at 277 (K), 148 (F) and 30 (A) items.
- **How H1b counts the items the review removed.** The expert review covers every item in scope, the
  ones the two-reviewer review removed included, and H1b's share counts all of them. This rule was
  fixed, in the expert-review scope and the agents' briefs, before any expert verdict existed. The
  expert accepted no removed item: in Tier F every removed task is a duplicate or does nothing, and in
  Tier A every removed task is below the 8-call floor. So under this rule H1b cannot hold for Tier F
  (at most 172 of 195) or Tier A (at most 35 of 42). Beside the test, `reports/review_summary.md`
  gives the share over the items the review kept, as a description, not a test. The replacement tasks
  are not in the test either: they were built after the expert's verdicts were known. So H1b is
  counted on the original pool, and the replacements' acceptance is reported beside it: 21 of 27 in
  Tier F and 9 of 9 in Tier A.
- **Outside models after the main runs.** Eval plan §7.3 runs the panel, Qwen3-30B-A3B and B2
  together with the target models, and chooses the panel's members at the freeze. Here the main runs
  are the two target models' base and fine-tuned configurations, and the 7 December report compares
  the fine-tuned models with their base models. The 13 outside-model configurations run afterwards,
  if time allows, and the panel's members are chosen then (§6). So H1a and H1c, and H1b's correlations with published
  scores, are decided only once the outside models have run, and so are the comparisons that need
  them: the RQ2 upper bound (B2) and the open MoE model of North-Mini-Code's size (Qwen3-30B-A3B).
- **The judge calibration's answers.** Eval plan §8.1 draws the 150 calibration answers from the base
  models, B2 and the panel, and Table 15 labels them in week 4. Here they come from the main runs, in
  three source groups: B0, B1 and the fine-tuned models. They are drawn once the fine-tuned models'
  Tier-K test runs are logged and labelled before the judge grades any test answer (§5.1); until the
  judge passes, no judge score is reported.
- **B1 on Tier F.** The evaluation plan's B1 baseline (its Table 1) runs on Tiers K and F, with the
  retrieved pages added to the prompt. In Tier F every configuration already makes its one lookup through GRC-Agent's own search,
  so B1 there is oracle grounding: the catalog entries of the reference solution's blocks (§5.2). It
  tells the model which blocks to use, and its result is reported as grounding, not as retrieval.
- **North-Mini-Code's anchor.** Eval plan Table 6 anchors North-Mini-Code's compression comparison at
  BF16. Its BF16 weights (61 GB) do not fit the 32 GB GPU of the main runs: with the experts in CPU
  memory it generates about 5 tokens per second, and its two dev Tier-A episodes took 25 and 26 of the
  30 minutes allowed, so the clock rather than the model would decide many episodes. Q8_0 (31 GB) runs
  with the experts of four layers in CPU memory at about 83 tokens per second and is the anchor here
  (§6); H4a's drop for North-Mini-Code is from Q8_0 to Q4_K_M. The BF16 configurations keep their
  declared settings, marked `later: true`, for a GPU with at least 80 GB.
- **Training variants.** The methodology proposal's Table 7 compares data sources (knowledge-only,
  trajectories-only, both, and both with general replay), adapter placements, a preference stage (DPO
  and DPOP) and three seeds, and H3 compares PASS-only with PASS + repaired data. The main runs train each
  target model once, by SFT on all the data (§8, decision 9). The variants keep their declared
  configurations, marked `later: true`; H2a, H2b and H3 are decided only if the variants they need are
  trained, and the spread between seeds is reported only if seeds 2 and 3 are trained.
- **Where the Tier-F tasks come from.** Eval plan §4.3 draws Tier-F tasks from GNU Radio's example
  flowgraphs, permissively licensed community flowgraphs and simple hand-written L1 chains. Of the 148
  test tasks, 128 come from the seeded generator (`tier_f/generate.py`), 14 were composed by hand and
  6 are built from GNU Radio examples; of the 45 dev tasks, 35, 6 and 4 (§8, decision 8). Scores on the
  example-based tasks are reported separately (§5.2).
- **The canary.** Eval plan §6 puts the canary in every released file. Here every item record carries
  it (a Tier-F or Tier-A task's `task.json`, a Tier-K question's record), and the other files of a
  task do not (§8, decision 8). The canary never appears in a file a model under test sees.

## 11. Amendments

- **Amendment 1 (21 November 2026, the evaluation lead).** North-Mini-Code's base model at Q8_0
  (`nmc-q8-b0`) is postponed: it stays declared, marked `later: true`. In its place the main runs gain
  Nanbeige4.2-3B's base model at Q4_K_M (`nb-q4-b0`: Tiers K and F with n = 5, Tier A with k = 4),
  converted and served like the other Nanbeige configurations (§6). It is compared with the fine-tuned
  model at Q4_K_M (`nb-q4-final`), base against fine-tuned at the 4-bit deployment precision (RQ2), and
  with `nb-bf16-b0` for what 4-bit compression costs the base model. With `nmc-q8-b0`, North-Mini-Code's
  base-model anchor at Q8_0 and its base-model BFCL run wait too; its base model is compared with its
  fine-tuned model at Q4_K_M (`nmc-q4-b0`, `nmc-q4-final`). The run list has 40 configurations: 12 main
  runs and 28 marked `later: true`. The amendment is published before the first test run of
  `nb-q4-b0`; no test run before it is affected.

## Appendix A: analysis proposals

Generated by `./gb prereg` from `grcbench.stats.proposals` (also printed by
`./gb stats proposals --format md`).

<!-- BEGIN stats-proposals (generated; do not edit by hand) -->
Approved, pre-registered definitions (implementation plan, Phase 5 WS7):

- **majority-mcnemar**: McNemar's per-item pass: the majority of the item's n samples pass; a tie fails.
- **split-half**: split-half: samples {1,2} vs {3,4} of n = 5, the 5th dropped, plus a random-half bootstrap.
- **h1c-size-matched**: H1c: Tier K subsampled to Tier F's 150 items, repeated, and the distribution reported.
- **func-denominator**: Func@1's denominator: the tasks that have a V5 oracle.
- **holm-families**: Holm-Bonferroni within each research question.

Proposals for G2 (deviation = changes a stated definition; extension = decides an open point):

### clustered-tests (deviation): Paired tests when items are clustered

- Stated: WS7 task spec: McNemar exact (binomial on discordant items) and Wilcoxon signed-rank (scipy.stats.wilcoxon); eval plan §9: items of one question family or source flowgraph are not independent.
- Proposal: Confirmatory comparisons pass every item's cluster_id, and the primary test is then the exact cluster sign-flip form: McNemar on D = sum of per-cluster discordance (exact randomisation form of Durkalski et al. 2003) and the signed-rank test with whole clusters' rank sums sign-flipped. Without clusters both reduce exactly to the binomial McNemar and scipy's Wilcoxon, which stay available.
- Code: functions.mcnemar_clustered, functions.wilcoxon_clustered, functions.compare_paired
- Evidence: tests/test_stats.py::test_clustered_tests_hold_their_size_when_items_are_clustered: with a model-by-family tilt (30 clusters of 5, 300 null datasets) the item-level McNemar and Wilcoxon reject a true null in more than 8% and 10%, the sign-flip forms in at most 7.5%.

### primary-test (extension): Which test is primary on which tier

- Stated: eval plan §9: McNemar for pass/fail, Wilcoxon signed-rank for scores.
- Proposal: Tier K (partial credit) is a score tier: Wilcoxon on per-item mean scores. Tiers F and A are pass/fail tiers: McNemar on majority-reduced items. compare_paired returns ONE primary p-value, the one entered into Holm, with effect sizes on its own estimand; for pass/fail tiers the Wilcoxon test of per-item pass@1 is reported as secondary and kept out of Holm.
- Code: functions.TIER_OUTCOME, functions.PRIMARY_TEST, functions.compare_paired
- Evidence: tests/test_stats.py::test_compare_paired_effects_follow_their_own_test_when_estimands_disagree: majority share and pass@1 can disagree in sign.

### wilcoxon-method (extension): Null distribution of the signed-rank test

- Proposal: Differences are rounded to 12 decimals and zeros dropped (Wilcoxon's rule); exact law without ties and at most 50 non-zero differences, full enumeration with ties and at most 13, otherwise the normal approximation with tie correction and no continuity correction.
- Code: functions.wilcoxon_signed_rank
- Evidence: equal to scipy.stats.wilcoxon in each regime (tests/test_stats.py).

### intervals (extension): Interval methods

- Stated: eval plan §9: means with 95% confidence intervals, clustered standard errors checked with a cluster bootstrap.
- Proposal: Clustered SE: CR1 (G/(G-1)) with t on G-1 degrees of freedom. Bootstrap: percentile intervals, clusters resampled, B = 10,000, seed 20261014. Kappa: percentile bootstrap with whole answers resampled.
- Code: functions.clustered_mean_ci, functions.cluster_bootstrap_ci, functions.BOOTSTRAP_B, functions.SEED, agreement.cohen_kappa_ci
- Evidence: singleton clusters reproduce scipy's sem and t interval; the bootstrap SE matches the CR0 SE within 5% (tests/test_stats.py).

### mde-design-effect (extension): The MDE printed next to a clustered comparison

- Stated: eval plan §9 / Table 8: MDE at 80% power, 5% significance, paired, p_d = 0.2-0.3, printed next to every claim.
- Proposal: compare_paired prints Table 8's value as the independent-items planning value ('mde', with 'mde_note') and, when clusters are given, the design effect (clustered / iid SE of the primary paired difference, squared) and mde * sqrt(max(1, design effect)) ('mde_design_adjusted').
- Code: functions.compare_paired, functions.design_effect, functions.MDE_NOTE
- Evidence: Exact power of the exact McNemar test at Table 8's MDE with independent items (binomial sum, not simulated): K 0.778 / 0.777, F 0.768 / 0.778 at p_d = 0.2 / 0.3, A 0.830 at p_d = 0.3; at p_d = 0.2 Tier A's MDE (22.9 points) exceeds p_d itself and cannot occur. With items correlated in clusters of 5 the cluster test's power at the printed MDE fell to 0.57-0.70 (review simulation, cluster SD of log-odds 0.5-1.0).

### split-half-n (extension): Split-half for configurations with n other than 5

- Stated: split-half = samples {1,2} vs {3,4} of n = 5 (the run list also has n = 3 and the closed frontier model at n = 1, eval plan Table 6 and §7.3).
- Proposal: Each configuration is split at its own n into {1..h} vs {h+1..2h}, h = floor(n/2) (n = 3: {1} vs {2}); n = 1 cannot be split and is listed as excluded. Sensitivity: only configurations with n >= 4.
- Code: functions.split_halves, functions.split_half_panel, functions.random_half_panel, functions.SPLIT_HALF_MIN_SAMPLES
- Evidence: n = 5 gives exactly the approved split (tests/test_stats.py).

### H1a-decision (extension): Decision rule of H1a

- Stated: eval plan Table 3: Kendall's tau >= 0.8 between bootstrap resamples and between halves of the samples; judge kappa >= 0.7 with 95% lower bound >= 0.6; references pass 10 of 10 harness runs.
- Proposal: Bootstrap part: tau >= 0.8 in at least 95% of the pairs of independent bootstrap resamples (undefined taus count as below); the median tau and the share are reported beside it, and 'median tau >= 0.8' is the lenient alternative to decide between at G2. Split-half part: the split-half tau >= 0.8 (random-half reported, not decided on). Judge part: both calibration gates (judge-gate). Harness part: every remaining task passed all its runs.
- Code: functions.h1a_decision, functions.H1A_BOOT_RULE, functions.H1A_BOOT_SHARE
- Evidence: the median rule would let H1a hold with almost half of the resample pairs below 0.8 (methodology review M6).

### H1c-test (extension): The test behind H1c's count of significant pairs

- Proposal: Both tiers use the same test: Wilcoxon signed-rank on per-item mean scores (Tier K's mean credit, Tier F's pass@1), Holm over all pairs, cluster sign-flip form with clusters. Each tier's own primary test (K Wilcoxon, F McNemar) is a sensitivity analysis only, because mixing tests compares test power, not tiers.
- Code: functions.H1C_TEST, functions.separation, functions.size_matched_separation
- Evidence: ./gb stats h1c-sim --ratio 1.6 --spread 1.0 (21 models, F items 1.6x as discriminating): with K Wilcoxon / F McNemar and the earlier share rule, H1c held in 1% of 200 datasets (pairs part 1%), with the same test on both tiers in 100% (review simulation c_h1c_alt.py, reproduced).

### H1c-mixed-n (extension): H1c across a panel with different numbers of samples

- Stated: eval plan §9: across the panel of about 21 models; Table 6 / §7.3: n = 5, 3 and 1.
- Proposal: Every model is reduced on its own samples: per-item mean score (pass@1) for the spread and the Wilcoxon test, majority of its own n (ties fail) for the McNemar sensitivity. No model is dropped and no sample truncated.
- Code: functions.separation, functions.size_matched_separation
- Evidence: tests/test_stats.py::test_h1c_runs_on_a_panel_with_n_5_3_1.

### H1c-decision (extension): Decision rule of H1c

- Stated: subsample Tier K to 150 items, repeated, report the distribution.
- Proposal: The subsample distribution is reported descriptively. H1c holds if the 95% percentile bootstrap intervals of SD(F) - SD(K at 150) AND of sig_pairs(F) - sig_pairs(K at 150) both lie above 0. Each of B = 1,000 replicates resamples Tier F's clusters and a fresh 150-item Tier-K subsample's clusters with replacement.
- Code: functions.h1c_decision, functions.size_matched_separation, functions.H1C_BOOT_B
- Evidence: H1C_SIM_EVIDENCE

### H4a-decision (extension): Decision rule of H4a

- Stated: proposal H4: 4-bit GGUF costs more on agentic (A) than knowledge (K) metrics; measured as the score change from BF16 to Q4.
- Proposal: For the final model of each architecture: the drop from BF16 to Q4_K_M on Tier A (per-task success) minus the drop on Tier K (per-item credit), with a 95% percentile interval from a stratified cluster bootstrap (each tier's clusters resampled independently, B = 10,000). H4a holds if the interval lies above 0. Tier A's MDE at its 30 test tasks (23-28 points) is printed beside it.
- Code: functions.compression_cost_difference
- Evidence: tests/test_preregistration.py::test_h4a_decision_rule: equal drops give an interval around 0; a 30-point larger Tier-A drop on 30 tasks holds.

### H4b-context (extension): The speed that decides H4b

- Stated: proposal H4: the MoE reaches >= 80 tok/s on reference hardware at Q4_K_M; eval plan §3: the reference hardware is setup H2 (4 GB GPU, MoE experts in CPU memory).
- Proposal: Generation speed (llama-bench tg) of North-Mini-Code at Q4_K_M on H2, median of 5 warm runs, decided at each of 4K, 16K, 32K and 64K tokens of context; the result states the context range over which H4b holds. A context that cannot be run (for example out of memory) does not hold. Setups H1 and H3 are reported beside it.
- Code: tier_e.bench.CONTEXTS, tier_e.bench.REPETITIONS, functions.h4b_decision
- Evidence: the protocol of eval plan §4.5 (median of 5 warm runs, 4K-64K).

### judge-gate (extension): What the judge's kappa is computed on

- Stated: eval plan §8.1: the judge passes if Cohen's kappa >= 0.7 and the lower end of its 95% interval >= 0.6.
- Proposal: docs/guidelines/judge_calibration.md: the rule is applied twice and the judge passes only if both hold: (1) unweighted kappa on the credits of every P and N row, judge vs the adjudicated reference labelling (unanimous label, else the adjudicator's), whole answers resampled; (2) linear-weighted kappa on the answers' score bands 0 | (0, 0.5) | [0.5, 1) | 1, each answer one item. Stricter than §8.1 read alone.
- Code: agreement.judge_calibration, agreement.score_band, agreement.human_reference
- Evidence: a judge that wrongly zeroes answers through D, DU or H flags fails gate 2 even when the point credits agree (tests/test_stats.py).

### rasch-solved-min (extension): Exploratory Rasch fit on a partial-credit tier

- Stated: eval plan §9: a Rasch model is fitted as an exploratory check only.
- Proposal: On Tier K a model 'solves' an item when its mean credit is at least 0.5; models and items with extreme scores are removed first, and no estimates are reported when the remaining data are separated.
- Code: items.RASCH_SOLVED_MIN, items.rasch_fit
- Evidence: exploratory only; nothing is kept or dropped because of it.

### fresh-set-gap (extension): Memorisation check on Tier K's GNU Radio questions

- Proposal: The 100 published GNU Radio questions are the reference set. A fresh set of the same composition comes from a private seed whose SHA-256 is published with this document (tier_k.blockgen). The headline configurations (B0 and the final model of each target, and B2) also answer the fresh set. The gap is the reference-minus-fresh mean item score over paired slots, clustered by block, with the clustered t and cluster-bootstrap 95% intervals; memorisation is flagged when both lie above 0. The headline Tier-K score stays the reference set's.
- Code: functions.fresh_gap
- Evidence: ./gb stats selfcheck: equal sets give a gap of 0 and no flag; a fresh set 0.5 lower on 6 of 8 slots gives 0.375, flagged.
<!-- END stats-proposals -->

## Appendix B: Tier-K question prompt and judge prompt

Generated by `./gb prereg` from `grcbench.tier_k.answer_prompt` and `grcbench.tier_k.judge_prompt`
(the judge's full snapshot: `docs/preregistration/judge_prompt_snapshot.json`).

<!-- BEGIN judge-prompt (generated; do not edit by hand) -->
| Item | Value |
|---|---|
| Tier-K question prompt version | `tier-k-answer-v3` |
| Tier-K question prompt SHA-256 | `1fd9a7e2488adc2970742f41382337963b7ab47a670711c260e31ebf8f2f158a` |
| Tier-K answer limit | 16,384 tokens, reasoning included |
| Judge prompt version | `tier-k-judge-v4` |
| Judge prompt SHA-256 | `34ce062b5dd91321294df5c33217f9f8c45ab251a7a23c60c71ad0e46d2a8b35` |
| Judges | MiniMax-M3 (fallback: Kimi K2.6) |
| Rule cascade | `tier-k-rules 0.6` |
| Entity index | 187,829 names, SHA-256 `f2610360dce1da1e2cd2fff7b4fa7d461f4766a81e95b2d458150500944028e6` |
| Probe requests in the hash | 6 toy items x 11 answers |

Tier K is closed-book: a model under test gets one message, the question prompt's instruction and the question (`grcbench.tier_k.answer_prompt`); B1 places the retrieved documentation before it. The fallback judge uses the judge template unchanged, so both judges share one version and one hash. The full template, rubric rendering, verdict schema and aggregation rules are in `docs/preregistration/judge_prompt_snapshot.json`.
<!-- END judge-prompt -->

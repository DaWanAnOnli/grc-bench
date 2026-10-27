# GRC-Bench pre-registration (benchmark v0.9)

**Status: draft for the team's sign-off at gate G2 (Friday 30 October 2026).** After sign-off it is
published on the Open Science Framework (OSF), before any fine-tuned model is run on the test split.
It is amended once, at the freeze (gate G3, Friday 13 November 2026). Any later deviation is reported
in the 7 December report.

- **Team:** JJJP, University of Toronto MScAC: Joel Rompis (evaluation lead), Jiuyue Cai, Patricio
  Pizaña Vela, John Borland. Industry partner: Qoherent.
- **Plans this document follows:** the methodology proposal (`Methodology_Proposal_to_7Dec.docx`) and
  the benchmark evaluation plan for the CAD 504 budget case (`Benchmark_Evaluation_Plan_to_7Dec.docx`,
  cited as "eval plan §n"). Rows of eval plan Table 1 marked "set by budget" follow the CAD 504 case.
- **Companion files:**
  - `docs/preregistration/run_list.yaml`: the declared test-split run list (eval plan Table 6);
  - the analysis code, package `grcbench.stats` of the `grc-bench` repository, at the commit named at
    sign-off (Appendix A lists its open choices);
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

| Tier | What it measures | Dev split (for every choice) | Test split v1.0 (frozen 13 Nov) | Built by G2 |
|---|---|---|---|---|
| K: knowledge | Knowledge of RF, signal processing and GNU Radio, shown in written answers to questions | 200 questions held back from the 1,668 training questions | 285: the 185 frozen questions and 100 GNU Radio behaviour questions, each answered by running GNU Radio (the public reference set; any seed draws a fresh set of the same composition) | all 285 test and 200 dev questions; rubric templates (prose written by the team) |
| F: flowgraphs | Making a correct flowgraph edit for a stated goal: one documentation lookup and one edit inside GRC-Agent, judged by running the result | 45 tasks | 150 tasks (60 L1, 60 L2, 30 L3) | a first set of 30 (20 test, 10 dev), all gates passed, pending team review |
| A: agent | Completing a realistic multi-step task with GRC-Agent's tools (8–20 tool calls), judged by the end state | 12 tasks | 30 tasks | 10 test and 12 dev tasks built, all gates passed; 20 test tasks drafted as specifications |
| E: speed and memory | Prompt and generation speed, time to first token and peak memory on the target hardware | — | — | the llama-bench protocol, tested on a small model |

**Splits and the test-run log** (eval plan §4.1).
- Every choice (checkpoints, data mixes, adapter placement, early stopping, the CI gate) uses the dev
  split.
- A run on the test split must name a configuration declared in `run_list.yaml`. The harness refuses
  any other test run.
- Every test run is appended to `runs/test_run_log.jsonl` before it starts, with its model, settings,
  code version, date and reason.
- Items are pruned before the freeze using only the answers of base models and outside models.

**Contamination controls** (eval plan §6).
- Every benchmark item is compared with the training data by 8-word overlap, TF-IDF similarity and
  embedding similarity. The rate is reported for each tier, and flagged pairs are judged by the team.
- Tier-K items with a near-twin in the training questions are reported separately from clean items.
- The functional signatures of every test and dev flowgraph are reserved: the training team removes
  any training flowgraph with a reserved signature.
- Every item record carries the canary string; the canary never appears in anything a model sees.
- Benchmark items are never sent to a model that creates training data.
- **Agent traces used for training.** Qoherent's 1,853 agent traces of 15–18 October (DeepSeek-V4.1-Flash
  in GRC-Agent's harness) are used in full for training. `./gb traces` checks them against the
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
| **H1a** reliability | Model rankings are stable, the judge agrees with people, and reference solutions are reproducible. | Per tier, all four parts must hold. (1) Kendall's τ between the rankings of two independent bootstrap resamples is at least 0.8 in at least 95 % of 1,000 pairs (`H1a-decision`). (2) The split-half τ, samples {1, 2} against {3, 4}, is at least 0.8. (3) The judge passes both calibration gates (§5.1, `judge-gate`). (4) Every remaining Tier-F and Tier-A reference passes 10 of 10 harness runs. |
| **H1b** validity | GRC-Bench measures GNU Radio development skill, not general ability. | The Qoherent expert accepts at least 90 % of the reviewed items of each tier as correct and relevant. The correlations of each tier with published BFCL and MMLU-Pro engineering scores across the panel are reported with intervals and described, not tested (eval plan §9). |
| **H1c** separation | Running the outputs separates models better than asking questions: Tier-F scores spread more widely, and more model pairs differ significantly, than Tier-K scores. | Tier K is subsampled to Tier F's 150 items. H1c holds if the 95 % bootstrap intervals of both SD(F) − SD(K₁₅₀) and sig_pairs(F) − sig_pairs(K₁₅₀) lie above 0, with the same test on both tiers (`H1c-test`, `H1c-mixed-n`, `H1c-decision`). |
| **H2a** recipe | Mixed SFT (knowledge, trajectories and replay) beats single-source SFT on every tier. | On the test split, Tiers K and F: the best S1 checkpoint against each single-source comparison arm of the same model, with the primary paired test of each tier, Holm-corrected in the RQ2 family. H2a holds for a tier if every comparison favours the mixed recipe at the corrected 5 % level. Tier A is compared on the dev split only, because the arms are not run on Tier A of the test split (Table 6). |
| **H2b** preference stage | Defect-aware preference optimisation lowers the hallucinated-block rate by at least 30 % relative to SFT alone. | Tier F: the hallucinated-block rate of S1 + DPO and of S1 + DPOP against the best S1 checkpoint. H2b holds for a method if the relative reduction is at least 30 % and the paired test of the per-task rates is significant after Holm correction in the RQ2 family. |
| **H3** curation | Adding validated repairs to PASS-only data raises accuracy and lowers the hallucinated-block rate at equal compute. | The curation comparison A2 runs on the dev split (eval plan §7.2): PASS + repaired against PASS-only, on Tier-K score and on the Tier-F hallucinated-block rate, paired tests, Holm-corrected in the RQ3 family. Because the dev split also served for model choices, H3 is reported with that limitation. |
| **H4a** compression cost | 4-bit compression costs more on agentic (A) than on knowledge (K) metrics. | For the final model of each architecture: the drop from BF16 to Q4_K_M on Tier A minus the drop on Tier K, in percentage points, with a 95 % cluster-bootstrap interval (items resampled within each tier). H4a holds if the interval lies above 0 (`H4a-decision`, §8). Tier A's MDE of 23–28 points is printed beside the result. |
| **H4b** speed | The MoE model (North-Mini-Code) reaches at least 80 tokens/s on the reference hardware at Q4_K_M. | Tier E on setup H2 (a laptop with a 4 GB GPU, MoE experts in CPU memory): the median generation speed of 5 warm llama-bench runs is at least 80 tokens/s, decided at each of 4K, 16K, 32K and 64K tokens of context. The result states the context range over which H4b holds ("holds up to 16K"); a context that cannot be run does not hold (`H4b-context`, §8). Setups H1 and H3 are reported beside it. |

## 4. Item quality before the freeze

- **Two-person review.** Two team members review every Tier-K rubric, Tier-F task, Tier-A task and
  generated block question with the checklists in `docs/checklists/`. A Qoherent expert decides every
  disagreement.
- **Expert review.** A Qoherent expert reviews every GNU Radio question and every Tier-F and Tier-A
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
  GNU Radio behaviour questions, whose answers come from GNU Radio runs; the team reviews and decides
  every item. Claude does not draft rubric prose. Anthropic models are not
  on the panel.

## 5. Scoring and metrics

### 5.1 Tier K

- **How a question is posed.** Tier K is closed-book. A model under test gets one message: a fixed
  instruction and the question, with no GRC-Agent system prompt and no tools
  (`grcbench.tier_k.answer_prompt`, version and hash in Appendix B). The instruction asks for a closing
  line "Final answer: …" and for the unit after every number, and it states that a number without a
  unit is never read as decibels. A reply is limited to 4,096 tokens, reasoning included. B1 places
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
  grades any test answer, it is compared with human labels on 150 answers labelled by three annotators,
  one of them a Qoherent expert. The judge passes only if both gates hold: Cohen's κ on the per-point
  credits is at least 0.7 with the lower end of its 95 % interval at least 0.6, and the linear-weighted
  κ on the answers' score bands meets the same thresholds. Scott's π and the mean judge–human gap are
  reported, for each task type too. If MiniMax-M3 fails, Kimi K2.6 is calibrated with the same prompt,
  and both results are reported. In week 8, 50 more answers from the final models are labelled and the
  judge–human gap is reported for each model.
- **Every answer is scored automatically** (§8, decision 7); no answer is scored by a person. If the
  judge returns no valid verdict in 3 attempts, the fallback judge (Kimi K2.6) is asked once; if it
  also fails, the answer scores 0. An answer that carries the benchmark's canary scores 0 and is
  counted. Each model's report gives, next to its Tier-K score, the number of answers decided by the
  tie-break, scored 0 for the wrong form alone, scored 0 after failed judge replies, with a number
  scored 0 for a unit the rules do not read, and sent to the judge with masked names.
- **Limits.** Answers longer than 4K tokens are cut and scored as wrong. Reasoning text is removed
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
- **Tool calls.** Every model served by llama.cpp runs with the server's default tool grammar
  (`--jinja`), which holds tool calls to the tools' JSON schemas (§8, decision 1). API models use the
  provider's tool calling, with strict mode where it is offered; their well-formed-call rates are
  therefore not compared like for like with the llama.cpp models'. For the llama.cpp models, V1 and
  the well-formed-call share mostly check the server rather than the model; the
  hallucinated-block rate stays informative, since the grammar fixes a call's form but not the block
  names in it, and V2–V5 compare as usual.
- **Validity ladder.** V1 a well-formed `change_graph` call; V2 the graph loads with only real blocks;
  V3 it compiles (it validates and its Python code is generated); V4 it runs and every observation
  point receives data; V5 the task's oracle passes (signal checks and structural checks). Each check runs only if the previous one
  passed.
- **Metrics.** The pass rate at each rung, shown as a funnel; Exec@1 (reaching V4) and Func@1
  (reaching V5, over the tasks that have a V5 oracle), with the unbiased pass@k estimator; the
  hallucinated-block rate (the share of samples whose edit names a block that does not exist in the
  pinned GNU Radio catalog); a partial-credit score over blocks, parameters and connections. Answers
  are limited to 8K tokens, reasoning included.
- **Perturbed examples.** Scores on tasks built from public GNU Radio examples (numbers changed) are
  also reported separately from scores on composed tasks.

### 5.3 Tier A

- **Harness, the same for every model.** The pinned GRC-Agent version (commit 6c98128) with its real
  system prompt and 22 tools; tool outputs cut to 4K tokens (beginning and end kept, plus error lines);
  64K tokens of context; at most 40 model requests; a time limit on each flowgraph run; web tools
  offline; the shell in a sandbox without network. A run that hits a cap is a failure. Tool calls as
  in Tier F: the llama.cpp server's default tool grammar for every model it serves, the provider's
  tool calling (strict mode where offered) for API models (§8, decision 1). For the llama.cpp models
  the share of well-formed calls therefore mostly checks the server, as V1 does in Tier F.
- **Scripted user.** Set replies for the questions a task expects; otherwise "please use your best
  judgement".
- **Grading by end state.** The final saved flowgraph is re-run by the grader through the Tier-F
  ladder and checked against the task's assertions; knowledge and refusal tasks check the answer
  against facts read from the graph. An empty or missing answer never counts.
- **Metrics.** Task success and pass^k over k runs; the share of tool calls that are well-formed and
  match the tool's schema; calls to tools that do not exist; correct refusals; first-attempt success
  against recovery after an error; steps, tokens and time per task.
- **Invalid episodes.** An episode that ends with an infrastructure error (the model server or the
  harness failed, not the model) is not scored and is re-run (§8, item 10).

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

`run_list.yaml` lists the 38 configurations that may run on the test split (eval plan Table 6), with
their model, precision, tiers, samples per item (n) and Tier-A runs per task (k), and the hypotheses
they serve. In summary:

- **Nanbeige4.2-3B** (BF16 unless stated): B0, B1, the best S1 checkpoint, S1 + DPO and
  S1 + DPOP, the final model compressed to Q8_0, Q5_K_M and Q4_K_M, seeds 2 and 3 of the final
  setting, and four comparison arms.
- **North-Mini-Code**, compared at the 4-bit deployment precision with BF16 as the anchor: B0 in BF16
  and Q4, B1 and the best S1 checkpoint at Q4, the final model in BF16, Q8_0, Q5_K_M and Q4,
  seeds 2 and 3, and one comparison arm (attention-only adapters).
- **References:** Qwen3-30B-A3B at Q4 and the B2 cloud teacher, DeepSeek-V4.1-Flash.
- **The panel** (eval plan §7.3): six small open models, four open teachers and one closed frontier
  model. The panel's slots are declared now; its members are chosen and added at the freeze.

Every configuration uses n = 5 samples per Tier-K and Tier-F item and, where it runs Tier A, k = 4
runs per task, except the extra seeds (n = 3, k = 2), the comparison arms (n = 3), North-Mini-Code's
BF16 B0 (k = 2) and the closed frontier model (n = 1, k = 2).

**Engines and settings** (eval plan §7.1). All open models run in llama.cpp b10333 from GGUF files,
BF16 references included. Each model uses its recommended sampling settings; greedy decoding is
checked on the dev split only. Both target models keep an 8-bit KV cache in every Tier-A run; a check
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
  BF16, and H4a).
- **What the benchmark can detect** (eval plan Table 8: 80 % power, 5 % significance, paired, 20–30 %
  of items decided differently). The MDE is printed beside every claim, with a design-effect
  adjustment when items are clustered (`mde-design-effect`).

  | Tier | Test items | MDE (percentage points) |
  |---|--:|---|
  | K | 285 | 7–9 |
  | F | 150 | 10–13 |
  | A | 30 | 23–28 |

- **Seeds.** The final setting of each model is trained with 3 seeds; the spread between seeds is
  reported separately from the item-level interval.
- **Memorisation check (Tier K).** The headline configurations (B0 and the final model of each target,
  and B2) also answer the private fresh set. The gap is the reference-minus-fresh mean item score over
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

Decisions 1–7 were taken by the evaluation lead on 27 October 2026; the rule cascade, the judge
template, the statistics package, the label groups and the run list implement them. Decisions 8–11
are open; each states what has to be decided and the option this draft proposes.

1. **B0 and the tool grammar.** B0 is the base model served with the default tool grammar
   (`--jinja`) on Tiers A and F, as is every later stage of every model llama.cpp serves; there is
   no separate grammar-constrained baseline (§5.2, §5.3, §10). The cloud teacher is B2.
2. **The analysis proposals** in Appendix A are accepted as written, including the strict bootstrap
   rule of H1a (95 % of pairs at τ ≥ 0.8, not the median τ).
3. **H4a and H4b.** H4a as proposed (`H4a-decision`). H4b is decided at each of 4K, 16K, 32K and 64K
   tokens of context, and the result is stated as a range (`H4b-context`, §3).
4. **The small `gnuradio_usage` group** (11 frozen test items) is reported with `design`: one group,
   design + gnuradio_usage, of 50 frozen test items (`reports/labels.md`).
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
8. **Tier-F decisions** of `reports/tier_f_reviews/fix_plan.md` §7: the level rule, the boundary
   between modify and parameter-tune, the hand-composed L2 tasks, the role-table declaration, the
   canary policy, GPL attribution, documented output-only limits, and the overlap between dev-10 and
   test-17.
9. **The four Nanbeige comparison arms** on the test split (Table 6): name them (proposal Table 7:
   data source A1 and adapter placement A3).
10. **Infrastructure re-runs:** *proposed:* an episode with an infrastructure error is re-run up to
    twice; if it fails a third time it is scored as a failure and listed.
11. **The code version:** the commit of the `grc-bench` repository made at sign-off, which this
    document cites for the analysis code.

Work that remains before any test run: the judge runner, which sends the answers the rules hand
over to MiniMax-M3, applies the fallback step and counts the failed replies; dev-split rubrics
with graded numbers, so that the rule cascade's deferral rate on dev answers (round-2 finding F5)
can be measured; and, in the Tier-A harness, a context-cap failure for a request that the model
server refuses because the model's own context (below the 64K cap) is full, which the harness
records as an infrastructure error (§5.3).

## 9. Fixed at the freeze (13 November)

- The v1.0 items of every tier, after the team and expert reviews and pruning, with the number of
  removed items and the reasons.
- The panel's members, entered in their declared slots of `run_list.yaml`.
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
  involved) were drafted by Claude, and the team reviews and decides every one; this is raised with
  Qoherent. The 100 GNU Radio test questions (§11 C, an AI model) are generated by a script, with
  answers from GNU Radio runs.
- **GNU Radio questions answered by execution.** Eval plan §4.2 reads the answers of the generated
  GNU Radio questions from the block catalog. The agent traces used for training show those catalog
  entries, so the questions ask instead what a block does on a stated input, and the key comes from
  running it (§5.1).
- **Agent traces** were collected on GNU Radio 3.10.9.2, where flowgraph execution was not available.
  The benchmark runs on 3.10.12 with execution.
- **Web tools** return a fixed "offline" message in Tier A.
- **The grammar-constrained baseline.** Eval plan Table 1 lists a baseline with grammar-constrained
  tool calls on Tier A. In llama.cpp b10333 the server's default already constrains tool calls, so
  B0 is that setting and no separate baseline runs; GRC-Bench does not measure the grammar's
  contribution, or whether fine-tuning alone makes tool calls well formed. The cloud teacher upper
  bound, numbered 3 in the approved plans, is B2 here.
- **L3 tasks** of Tier F are added after G2 (week 5), and Tier F and Tier A scale from their first
  sets only after the team's decision on them.
- **The language-model paraphrase detector** of the contamination audit waits for API access; the
  8-word, TF-IDF and embedding checks run now.

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
- Proposal: For the final model of each architecture: the drop from BF16 to Q4_K_M on Tier A (per-task success) minus the drop on Tier K (per-item credit), with a 95% percentile interval from a stratified cluster bootstrap (each tier's clusters resampled independently, B = 10,000). H4a holds if the interval lies above 0. Tier A's MDE (23-28 points) is printed beside it.
- Code: functions.compression_cost_difference
- Evidence: tests/test_preregistration.py::test_h4a_decision_rule: equal drops give an interval around 0; a 30-point larger Tier-A drop on 30 tasks holds.

### H4b-context (extension): The speed that decides H4b

- Stated: proposal H4: the MoE reaches >= 80 tok/s on reference hardware at Q4_K_M; eval plan §3: the reference hardware is setup H2 (4 GB GPU, MoE experts in CPU memory).
- Proposal: Generation speed (llama-bench tg) of North-Mini-Code at Q4_K_M on H2, median of 5 warm runs, decided at each of 4K, 16K, 32K and 64K tokens of context; the result states the context range over which H4b holds. A context that cannot be run (for example out of memory) does not hold. Setups H1 and H3 are reported beside it.
- Code: tier_e.bench.CONTEXTS, tier_e.bench.REPETITIONS, functions.h4b_decision
- Evidence: the protocol of eval plan §4.5 (median of 5 warm runs, 4K-64K).

### judge-gate (extension): What the judge's kappa is computed on

- Stated: eval plan §8.1: the judge passes if Cohen's kappa >= 0.7 and the lower end of its 95% interval >= 0.6.
- Proposal: docs/guidelines/judge_calibration.md: the rule is applied twice and the judge passes only if both hold: (1) unweighted kappa on the credits of every P and N row, judge vs the adjudicated human reference (unanimous label, else the adjudicator's), whole answers resampled; (2) linear-weighted kappa on the answers' score bands 0 | (0, 0.5) | [0.5, 1) | 1, each answer one item. Stricter than §8.1 read alone.
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
| Tier-K question prompt version | `tier-k-answer-v2` |
| Tier-K question prompt SHA-256 | `e6c73d16e2f40b4ca4e207c2be7405dd222f94e055ee3e25c84b148e691f4b34` |
| Tier-K answer limit | 4,096 tokens, reasoning included |
| Judge prompt version | `tier-k-judge-v3` |
| Judge prompt SHA-256 | `26e0b2285cd832aeb5b64cf150c9db6e632425a406ea98c50f92b648b82005b1` |
| Judges | MiniMax-M3 (fallback: Kimi K2.6) |
| Rule cascade | `tier-k-rules 0.5` |
| Entity index | 187,829 names, SHA-256 `f2610360dce1da1e2cd2fff7b4fa7d461f4766a81e95b2d458150500944028e6` |
| Probe requests in the hash | 6 toy items x 11 answers |

Tier K is closed-book: a model under test gets one message, the question prompt's instruction and the question (`grcbench.tier_k.answer_prompt`); B1 places the retrieved documentation before it. The fallback judge uses the judge template unchanged, so both judges share one version and one hash. The full template, rubric rendering, verdict schema and aggregation rules are in `docs/preregistration/judge_prompt_snapshot.json`.
<!-- END judge-prompt -->

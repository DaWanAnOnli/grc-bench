# Request to Qoherent (draft for Joel)

Draft, 22 Oct 2026. Not sent. Joel edits and sends it, or raises the points at the next client
meeting. Claude (Anthropic) drafted it.

## 1. Expert time

GRC-Bench needs a GNU Radio expert from Qoherent for two jobs before the benchmark is frozen at G3
(Fri 13 Nov):

- **Expert review of the benchmark tasks: about 30 hours.** The expert checks that the Tier-F
  flowgraph tasks and the Tier-A agent tasks are realistic, fair and correctly checked. The
  checklist is `docs/checklists/expert_review.md`. The first set is 30 Tier-F tasks
  (`reports/tier_f_first_set.md`), followed by the Tier-A tasks.
- **One annotator for the judge calibration.** Three people label the same 150 Tier-K answers
  (`docs/guidelines/judge_calibration.md`), and one of the three should be the Qoherent expert.

Could you name the person and say how many hours per week they can give from now until 13 Nov?

## 2. What Claude drafted (a change to Data Summary §11 D)

Data Summary §11 D says the team writes the agent tasks with no AI involved. In the build, Claude
(Anthropic) drafted:

- the instructions of the Tier-F and Tier-A tasks;
- their reference solutions;
- their automatic checks (oracles).

How this is kept fair:

- Every drafted item records `provenance.drafted_by: "Claude (Anthropic)"`.
- Nothing is used until two team members approve it. The expert review in point 1 comes on top.
- Every item passes the automatic quality gates.
- Anthropic models are not on the evaluated panel.
- Claude does not draft rubric text, judge answers, or repair or generate training data.

We would like Qoherent to accept this change. It will be declared in the role table of the
pre-registration (G2, Fri 30 Oct).

## 3. The new agent traces

Your 14 Oct reply says that more, shorter agent traces will be generated. To keep them apart from
the benchmark's test tasks:

- Please send the **task descriptions** of the planned sessions before they are generated, so that
  we can check them against the Tier-A test tasks at task level.
- Please have the generator follow the flowgraph rules in `handoff/training_team/`: no test or dev
  seed files (`flowgraph_seed_split.md`), and no composed graph on the reserved list
  (`flowgraph_composed_chains.md`).

## 4. A question about the SFT release

`grc-agent-sft-v0.1.0/sft_train.jsonl` keeps a `thinking` field on 279 assistant messages (all 12
rows). The release also has `sft_full_with_thinking.jsonl`. Is `sft_train.jsonl` meant to be the
version without thinking?

## 5. Hardware and compute

- **Compute:** the request is in `benchmark and evaluation/Compute_Budget_Note_GPU_Hours.docx`
  (GPU-hours per GPU type, and API tokens).
- **Radios on loan:** two RTL-SDR v4 and a Pluto, as offered on 14 Oct. When can we collect them?

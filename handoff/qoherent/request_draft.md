# Request to Qoherent (draft for Joel)

Draft, 22 Oct 2026. Not sent. Joel edits and sends it, or raises the points at the next client
meeting. Claude (Anthropic) drafted it.

## 1. How the benchmark is reviewed

The evaluation plan asked for a Qoherent expert to:

- review every GNU Radio question and every Tier-F and Tier-A task (about 30 hours);
- decide every disagreement between reviewers;
- label answers for the judge calibration.

GRC-Bench does these reviews with Claude review agents instead (`docs/guidelines/review_protocol.md`):

- **Two reviewers.** Two independent review agents check every item, each alone.
- **An expert agent.** It decides every disagreement, audits a sample of the agreements, and does
  the expert review of every GNU Radio question and every Tier-F and Tier-A task
  (`docs/checklists/expert_review.md`).
- **Evidence.** Every factual claim is checked by a quoted source or by running GNU Radio.
- **Logging.** Every agent run is declared and logged.

So no expert time is needed before the freeze (Fri 20 Nov). The pre-registration states the limits
openly. The expert-acceptance figure (H1b) is the agent's, not a GNU Radio engineer's. If a Qoherent
engineer later wants to look at a sample of items, the sheet and checklist are ready, and we would
report that review next to the agent's.

We would like Qoherent to accept this change.

## 2. What Claude drafted (a change to Data Summary §11 D)

Data Summary §11 D says the team writes the agent tasks with no AI involved. In the build, Claude
(Anthropic) drafted:

- the instructions of the Tier-F and Tier-A tasks;
- their reference solutions;
- their automatic checks (oracles).

How this is kept fair:

- Every drafted item records `provenance.drafted_by: "Claude (Anthropic)"`.
- Nothing is used until the review of point 1 approves it: two independent review agents, and the
  expert agent for every disagreement.
- Every item passes the automatic quality gates.
- Anthropic models are not on the evaluated panel.
- Claude writes the Tier-K rubrics under the same review. It never judges model answers in a scored
  run, and never repairs or generates training data.

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

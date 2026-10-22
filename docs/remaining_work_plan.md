# Remaining work plan: low-token execution (from 18 Oct 2026)

Ultracode used about 44 % of a weekly allowance in a day. From now on the remaining work runs **without
workflows**. The approved plan (`implementation_plan.md`) still sets the scope and rules; this file only
changes how the work is executed.

## Execution rules

1. **No Workflow tool and no parallel subagents.** Claude works inline. A single read-only Explore subagent is allowed only for a broad search where only the conclusion matters.
2. **Automated checks replace reviewer agents.** The evidence is the gates, tests and replay checks that already exist: the V1–V6 ladder, 10/10 solvability, variant and mutation gates, golden replay, conformance tests and `./gb baseline check`. No per-item adversarial reviewers and no second fix rounds unless a gate fails.
3. **The team review is the quality backstop.** It was already required by the plan. Everything Claude drafts stays `pending-team-review`, and the checklists in `docs/checklists/` guide that review.
4. **Generate, don't hand-write.** Similar items (block questions, rubric templates, Tier-F golden calls) come from one generator script per kind. Claude never reasons about each item separately.
5. **Quiet commands.** Commands print short summaries (`tail`, counts, pass/fail tables). Large JSON and transcripts are never read whole. Long runs use one background command with a single completion notice, with no monitors or heartbeats.
6. **One verification pass at the end** (step 10), done as a checklist, not a multi-skeptic loop.

## Where things stand (18 Oct, evening)

The three workflows were stopped. **They will not be resumed:** resuming re-runs most stages. Their loose ends are finished inline in steps 1–3.

| Area | State |
|---|---|
| Tier F | Signature v2, reviews, fix plan, tooling T1–T8 and all 10 fixer groups are done. **Unfinished:** second fix rounds for groups 1, 7, 9 and 10, the fresh review of group 2, and the set critic. |
| Tier A | Runner (with the Tier-F setting), grader and golden replay are done. **Unfinished:** the smoke test, and the review of the five Tier-A parts (never done). |
| Phase 5a | Statistics, rules, calculation answers, judge prompt, retriever and docs are built, reviewed and fixed once. **Unfinished:** second-round fixes for rules, statistics and judge, and the Tier-E builder (the first plumbing report from 18 Oct exists). |
| Not started | Phase 1 review round, full contamination run, Tier-F edit-format conversion, Tier-A tasks, block questions, rubric templates, pre-registration, hand-offs, eval-plan Tier-F text, final check. |

## Steps (in order; G2 is Fri 30 Oct)

| # | Step | How, cheaply | Done when |
|---|---|---|---|
| 1 | **Close Tier F** | Run `./gb tier-f gates all`. Extract the unresolved verify findings for groups 1, 7, 9 and 10 with a script. Fix only high-severity findings and gate failures inline; list the rest in the team sheet. Group 2 gets one checklist pass instead of a reviewer. The set report (`reports/tier_f_first_set.md`: coverage matrix, signature splits, gate table) comes from a small script; then `./gb tier-f sheet`. | 30/30 gates pass; report and sheet written |
| 2 | **Close Phase 5a** | Run its tests. Extract the second-round findings for rules, statistics and judge; fix the high-severity ones inline. Tier E: keep the 18 Oct plumbing report, with no further benchmark runs. | tests green; `./gb stats selfcheck` and `./gb score-k --selftest` pass |
| 3 | **Close Tier A** | Run `./gb tier-a smoke` once, in the background. Read only `reports/tier_a_smoke.md`. Instead of 10 reviewers, write three targeted tests: Tier-F prefill byte-identical to a real `inspect_graph` turn; one cap counter shared across resumes; no write or network escape from the sandbox. Then run `./gb baseline check`. | smoke report exists; the three tests pass; baseline unchanged |
| 4 | **Phase 1 check** (replaces the review round) | Run the existing tests (185/185 reproduction, exclusions, determinism) plus a double-run hash comparison of the split, label and catalog outputs. Run `./gb contamination` once in the background; read only its summary table. | all reproducible; contamination summary updated for Tier K and the new Tier-F tasks |
| 5 | **Tier-F edit format** | Run `./gb tier-a tier-f-golden all` (derives `golden_calls.json`) and the replay gate on all 30 tasks. Fix failures inline. | 30/30 golden replays pass V1–V6 10/10 |
| 6 | **Tier-A first set** (largest item) | One compact spec template. Tasks reuse Tier-F start graphs, as the eval plan's risk mitigation allows. Claude drafts 4–5 tasks per session with `golden_calls.json` and a wrong-end-state fixture, checked by `./gb tier-a gates`. **First set:** 12 dev + 10 test, covering all 9 categories plus 2 refusal tasks. The other 20 test tasks are drafted as specs only, for team review. | first set passes gates (replay 10/10, do-nothing fails, wrong state rejected) |
| 7 | **Block questions + rubric templates** | One generator over the block YAML: port dtypes, type→I/O mapping, enum sets, rates, asserts, port counts. Answers are checked through the GRC Platform API, deduplicated, and listed on a wording sheet. A second script writes rubric templates for the 185 test and 200 dev questions with empty prose fields. | ~100 questions + review sheet; 385 templates |
| 8 | **Pre-registration (G2)** | Write `docs/preregistration/preregistration.md` and `run_list.yaml` (Table 6) in one pass. Definitions come from the `stats/` docstrings and the eval plan, so nothing is re-derived. Include the team decisions still open (fix-plan §7, label merges, level/family rules). | OSF-ready draft for team sign-off |
| 9 | **Hand-offs + plan text** | Update the training-team README (signature v2, seed split, truncation function). Draft the Qoherent request (expert hours, Data Summary §11 D deviation, role-table declaration) and `joel_actions.md`. Give the replacement Tier-F paragraph for the evaluation plan (the docx is rebuilt only if asked). | drafts in `handoff/` |
| 10 | **Final check** | Run `./gb test`, `./gb audit` and `./gb baseline check`, then fill `reports/compliance_review.md` as one table (requirement → artifact → status), done once. | no failing rows |
| 11 | **Checkpoint** | Ask Joel before the first git commit of `grc-bench/`. | — |

**Progress (22 Oct).** Steps 1–7 are done, step 8 is a draft for the team's sign-off and step 9's
drafts are in `handoff/` (details in `../HANDOVER.md`, "22 Oct"):

| # | Result |
|---|---|
| 1 | 30/30 Tier-F tasks pass every gate; `reports/tier_f_first_set.md`; sheet with `known_limits` |
| 2 | 6/6 high findings resolved; 27 medium/low listed for the G2 scoring freeze |
| 3 | three targeted tests pass; smoke reports on CPU and on the RTX 5090; baseline unchanged since 18 Oct |
| 4 | `./gb build-all`: every generated output deterministic; contamination re-run for all tiers |
| 5 | 30/30 Tier-F golden edits replay 10/10 |
| 6 | 22/22 Tier-A tasks (10 test, 12 dev) pass all gates; 20 test tasks drafted as specs; `./gb tier-a sheet` |
| 7 | 100 block questions with a wording sheet; 385 rubric templates |
| 8 | `docs/preregistration/preregistration.md` and `run_list.yaml` (status: draft); `./gb prereg` |
| 9 | training-team, Qoherent and Joel drafts; replacement Tier-F text for the evaluation plan |
| 10 | `./gb test` 1,164 pass; `./gb audit` 8/8; baseline unchanged since 18 Oct; `reports/compliance_review.md` |

Left: step 11 (ask Joel before the first commit), then the team's reviews and G2 decisions.

**Rough schedule** (moved on 21 Oct; nothing ran on 19–20 Oct):
- 21–22 Oct: steps 1–3.
- 23 Oct: steps 4–5.
- 24–27 Oct: step 6.
- 28 Oct: step 7.
- 28–29 Oct: steps 8–9.
- 29 Oct: step 10.
- Steps 6 and 7 can swap if the team wants the Tier-K material earlier.

## What changes in quality, and how it is covered

| Lost | Covered by |
|---|---|
| Independent adversarial review of each item | Gates and mutation tests catch broken oracles and shortcut passes; the team's two-person review (checklists exist) catches wording and fairness problems; the open findings from the earlier rounds are on the team sheets. |
| Loop-until-dry fixing | One fix pass, driven by gate failures and high-severity findings only. |
| Multi-skeptic final review | One checklist pass in step 10, plus the automated audit. |

Scaling Tier F to 150 tasks and Tier A to 30 tasks still waits for the team's decision on the first sets (approved plan). When it comes, it should reuse the generator approach of step 7 wherever possible.

# Actions for Joel

22 Oct 2026. These are the things only Joel (or the team) can do. They are ordered by the date they
are needed. G2 is Fri 30 Oct: benchmark v0.9 runs end to end, and the pre-registration is published.

## Before G2 (Fri 30 Oct)

1. **The reviews follow the review protocol** (`docs/guidelines/review_protocol.md`). Claude review
   agents fill in every sheet below:
   - two independent reviewers per sheet (a writer and two checkers per rubric);
   - the expert agent for every disagreement;
   - decision rules in code (`./gb review`);
   - at most 38 declared agent runs before the freeze, 30 in the first round and 8 in the second
     (`reviews/review_log.jsonl`).

   The decided records are in `reviews/`, and the summary is `reports/review_summary.md`.

   | Sheet (`handoff/review_sheets/`) | Rows | Guide |
   |---|---|---|
   | `tier_f_first_set.csv` | 222 tasks (30 drafted by hand, 192 generated; 27 of them replace tasks the first round rejected) | `docs/checklists/tier_f_task_review.md` |
   | `tier_a_first_set.csv` | 51 tasks (9 of them replace tasks the first round rejected) | `docs/checklists/tier_a_task_review.md` |
   | `tier_k_block_questions.csv` | 100 questions | `tier_k_block_questions_instructions.md` (wording review, each question solved blind with GNU Radio) |
   | `rubric_sheet_test.csv`, `rubric_sheet_dev.csv` | 185 + 107 | `docs/guidelines/rubric_writing.md` (a writer agent and two checker agents) |
   | `task_type_ambiguous_test.csv` | 11 questions | `task_type_instructions.md` (two labels each; κ is reported) |
   | `task_type_spotcheck_train.csv` | 170 rows | `task_type_instructions.md` (10 % spot check) |
   | `tier_k_calc_blind_recompute.csv` | 25 answers | `tier_k_calc_instructions.md` (recompute without looking at the key) |
   | `tier_k_twin_threshold_review.csv` | 235 pairs | `similarity_sheets_instructions.md`, section B (the dev-split twin thresholds) |
   | `tier_k_test_clusters.csv` | 10 rows | `similarity_sheets_instructions.md`, section A |
   | `contamination_tier_k_test.csv`, `contamination_tier_k_dev.csv`, `contamination_tier_f_test.csv`, `contamination_tier_f_dev.csv` | 188 + 30 + 17 + 6 flagged pairs (the Tier-A sheets have none) | `similarity_sheets_instructions.md`, section C |
   | `expert_review.csv` | 433 items (H1b counts the original 397; the 36 replacements are reported beside it) | `docs/checklists/expert_review.md` (the expert agent; H1b) |

2. **Make the team decisions** listed in `reports/tier_f_first_set.md` §6 (set level: 8 points; task
   level: 8 tasks) and in `reports/scoring_reviews/round2_status.md` (the open findings, before the
   rule cascade and judge template are frozen in the pre-registration).
3. **Sign off the pre-registration.** The draft is `docs/preregistration/preregistration.md` with
   `run_list.yaml`. Its §8 lists 11 decisions (B0 and the tool grammar, the analysis proposals in Appendix A,
   label merges, the open scoring findings, the four Nanbeige comparison arms, ...). After the
   decisions: set `status: declared` in `run_list.yaml` (until then the harness refuses every test
   run), run `./gb prereg`, make the first commit (item 9) and upload to a new OSF project.
4. **Send the Qoherent request** (`handoff/qoherent/request_draft.md`): acceptance of the
   Claude-drafting change to Data Summary §11 D and of the review protocol, the task descriptions of
   the new traces, and the SFT-release question.
5. **Send the compute request** (`benchmark and evaluation/Compute_Budget_Note_GPU_Hours.docx`) if it
   has not gone out yet. Model runs stay blocked until there is approval of the CAD 504, MiniMax and
   DeepSeek access, and Qoherent GPU access.
6. **Send the training-team hand-off** (`handoff/training_team/START_HERE.md` and the files it lists).
   The reserved-signature list (`handoff/training_team/flowgraph_reserved_signatures.txt`) has 206
   entries, 17 more than the list the generated training graphs were filtered against. Ask the team to
   check the graphs they have generated against the 206 entries, and to use this list from now on.

## On this machine

7. **Delete the stale migration package** `<home>\Documents\GRC-Bench\GRC-Bench-migration`.
   It is an old copy, and its `wsl-grcbench.tar` contains the GRC-Agent API key. The working copy is
   the `project` folder that holds this repository (`<project>` in the reports).
8. **Install build tools in WSL** (needs your sudo password), for the Tier-E build of Nanbeige's
   llama.cpp fork: `sudo apt install build-essential cmake`. Docker is optional (the Dockerfile in
   `docker/` pins GNU Radio 3.10.12; the benchmark runs natively in WSL today).
9. **First git commit.** `grc-bench/` has never been committed. Claude asks before the first commit.
   The pre-registration should cite this commit.
10. **Retake the baseline** of the read-only inputs
    (`./gb baseline take --root grc_agent_repo,gnuradio_user_cache`) if you accept the two changes
    `./gb baseline check` reports since 18 Oct (`~/grc-agent/.git/index`, `fftw_wisdom`).
11. **The RTX 5090 in this machine.** llama.cpp b12020 runs on it with CUDA (`./gb fetch --cuda`: llama.cpp's own CUDA 12.8 download).
    Some of the GPU-hours in the compute request (about 61 rented RTX 5090 hours in the normal plan) could run here
    instead. Your decision; it changes no benchmark setting.

# Actions for Joel

22 Oct 2026. These are the things only Joel (or the team) can do. They are ordered by the date they
are needed. G2 is Fri 30 Oct: benchmark v0.9 runs end to end, and the pre-registration is published.

## Before G2 (Fri 30 Oct)

1. **Assign the team reviews.** Each sheet below needs two people working alone, then one agreed
   decision. Fill in a copy of a sheet, never the generated file.

   | Sheet (`handoff/review_sheets/`) | Rows | Guide |
   |---|---|---|
   | `tier_f_first_set.csv` | 195 tasks (30 drafted by hand, 165 generated) | `docs/checklists/tier_f_task_review.md` (about 10 minutes per task per reviewer, so about 33 hours each for all 195) |
   | `tier_a_first_set.csv` | 42 tasks | `docs/checklists/tier_a_task_review.md` (about 20 minutes per task per reviewer) |
   | `tier_k_block_questions.csv` | 100 questions | wording review of the GNU Radio behaviour questions: is everything the answer depends on stated? (`docs/guidelines/rubric_writing.md`) |
   | `rubric_sheet_test.csv`, `rubric_sheet_dev.csv` | 185 + 107 | `docs/guidelines/rubric_writing.md` (rubric prose and its two checks; Claude does not draft rubric prose) |
   | `task_type_ambiguous_test.csv` | 11 questions | `task_type_instructions.md` (two labels each; κ is reported) |
   | `task_type_spotcheck_train.csv` | 170 rows | `task_type_instructions.md` (10 % spot check) |
   | `tier_k_calc_blind_recompute.csv` | 25 answers | `tier_k_calc_instructions.md` (recompute without looking at the key) |
   | `tier_k_twin_threshold_review.csv` | 235 pairs | the dev-split twin thresholds |
   | `tier_k_test_clusters.csv` | 10 clusters | confirm each Tier-K test cluster or split it out (`reports/tier_k_items.md`) |
   | `contamination_tier_k_test.csv`, `contamination_tier_k_dev.csv`, `contamination_tier_f_test.csv`, `contamination_tier_f_dev.csv` | 188 + 30 + 17 + 6 flagged pairs (the Tier-A sheets have none) | `reports/contamination.md` |

2. **Make the team decisions** listed in `reports/tier_f_first_set.md` §6 (set level: 8 points; task
   level: 8 tasks) and in `reports/scoring_reviews/round2_status.md` (the open findings, before the
   rule cascade and judge template are frozen in the pre-registration).
3. **Sign off the pre-registration.** The draft is `docs/preregistration/preregistration.md` with
   `run_list.yaml`. Its §8 lists 11 decisions (B0 and the tool grammar, the analysis proposals in Appendix A,
   label merges, the open scoring findings, the four Nanbeige comparison arms, ...). After the
   decisions: set `status: declared` in `run_list.yaml` (until then the harness refuses every test
   run), run `./gb prereg`, make the first commit (item 9) and upload to a new OSF project.
4. **Send the Qoherent request** (`handoff/qoherent/request_draft.md`): the expert's name and hours,
   acceptance of the Claude-drafting change to Data Summary §11 D, the task descriptions of the new
   traces, and the SFT-release question.
5. **Send the compute request** (`benchmark and evaluation/Compute_Budget_Note_GPU_Hours.docx`) if it
   has not gone out yet. Model runs stay blocked until there is approval of the CAD 504, MiniMax and
   DeepSeek access, and Qoherent GPU access.
6. **Send the training-team hand-off** (`handoff/training_team/START_HERE.md` and the files it lists).

## On this machine

7. **Delete the stale migration package** `C:\Users\POPA\Documents\GRC-Bench\GRC-Bench-migration`.
   It is an old copy, and its `wsl-grcbench.tar` contains the GRC-Agent API key. The working copy is
   `C:\Users\POPA\Documents\!MScAC\2701\ml_ops_project\project`.
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

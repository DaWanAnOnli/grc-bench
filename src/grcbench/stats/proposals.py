"""The analysis PROPOSALS of grcbench.stats, listed once for the G2 pre-registration (eval plan §9).

Eval plan §9 "Pre-registration": "Hypotheses, metrics, the declared run list and the analysis
scripts are published on the Open Science Framework at gate G2 (30 October) [...] Any deviation is
reported in the 7 December report." The approved implementation plan (Phase 5, WS7) pre-registers
five definitions (APPROVED). Every other decision the analysis code takes is a PROPOSAL: it is
marked "PROPOSAL <id>" where the code defines it and listed here, so that the team can accept,
change or reject each one at G2 and the docs workstream can copy the list into
docs/preregistration/preregistration.md. `./gb stats proposals [--format text|md|json]` prints it.

Each PROPOSAL has
  id         the marker in the code ("PROPOSAL <id>");
  kind       "deviation": it changes a definition that the approved plan, the eval plan or the WS7
             task spec states; "extension": it decides a point that they leave open;
  topic, approved (what the plan or spec says, "" when silent), proposal, where (module.symbol,
  checked by the tests), evidence (how the choice was checked; simulation figures are reproducible
  with the command given).
"""

from __future__ import annotations

import json

APPROVED = (
    {"id": "majority-mcnemar", "definition": "McNemar's per-item pass: the majority of the item's n "
     "samples pass; a tie fails", "where": ("functions.majority_pass",)},
    {"id": "split-half", "definition": "split-half: samples {1,2} vs {3,4} of n = 5, the 5th "
     "dropped, plus a random-half bootstrap",
     "where": ("functions.SPLIT_HALVES", "functions.split_half_tau", "functions.random_half_tau")},
    {"id": "h1c-size-matched", "definition": "H1c: Tier K subsampled to Tier F's 150 items, "
     "repeated, and the distribution reported", "where": ("functions.size_matched_separation",)},
    {"id": "func-denominator", "definition": "Func@1's denominator: the tasks that have a V5 oracle",
     "where": ("functions.func_at_1", "functions.funnel")},
    {"id": "holm-families", "definition": "Holm-Bonferroni within each research question",
     "where": ("functions.holm_bonferroni",)},
)

PROPOSALS = (
    {"id": "clustered-tests", "kind": "deviation",
     "topic": "Paired tests when items are clustered",
     "approved": "WS7 task spec: McNemar exact (binomial on discordant items) and Wilcoxon "
                 "signed-rank (scipy.stats.wilcoxon); eval plan §9: items of one question family or "
                 "source flowgraph are not independent",
     "proposal": "Confirmatory comparisons pass every item's cluster_id, and the primary test is "
                 "then the exact cluster sign-flip form: McNemar on D = sum of per-cluster "
                 "discordance (exact randomisation form of Durkalski et al. 2003) and the signed-rank "
                 "test with whole clusters' rank sums sign-flipped. Without clusters both reduce "
                 "exactly to the binomial McNemar and scipy's Wilcoxon, which stay available.",
     "where": ("functions.mcnemar_clustered", "functions.wilcoxon_clustered",
               "functions.compare_paired"),
     "evidence": "tests/test_stats.py::test_clustered_tests_hold_their_size_when_items_are_clustered:"
                 " with a model-by-family tilt (30 clusters of 5, 300 null datasets) the item-level "
                 "McNemar and Wilcoxon reject a true null in more than 8% and 10%, the sign-flip "
                 "forms in at most 7.5%."},
    {"id": "primary-test", "kind": "extension",
     "topic": "Which test is primary on which tier",
     "approved": "eval plan §9: McNemar for pass/fail, Wilcoxon signed-rank for scores",
     "proposal": "Tier K (partial credit) is a score tier: Wilcoxon on per-item mean scores. Tiers F "
                 "and A are pass/fail tiers: McNemar on majority-reduced items. compare_paired "
                 "returns ONE primary p-value, the one entered into Holm, with effect sizes on its "
                 "own estimand; for pass/fail tiers the Wilcoxon test of per-item pass@1 is reported "
                 "as secondary and kept out of Holm.",
     "where": ("functions.TIER_OUTCOME", "functions.PRIMARY_TEST", "functions.compare_paired"),
     "evidence": "tests/test_stats.py::test_compare_paired_effects_follow_their_own_test_when_"
                 "estimands_disagree: majority share and pass@1 can disagree in sign."},
    {"id": "wilcoxon-method", "kind": "extension",
     "topic": "Null distribution of the signed-rank test",
     "approved": "",
     "proposal": "Differences are rounded to 12 decimals and zeros dropped (Wilcoxon's rule); exact "
                 "law without ties and at most 50 non-zero differences, full enumeration with ties "
                 "and at most 13, otherwise the normal approximation with tie correction and no "
                 "continuity correction.",
     "where": ("functions.wilcoxon_signed_rank",),
     "evidence": "equal to scipy.stats.wilcoxon in each regime (tests/test_stats.py)."},
    {"id": "intervals", "kind": "extension",
     "topic": "Interval methods",
     "approved": "eval plan §9: means with 95% confidence intervals, clustered standard errors "
                 "checked with a cluster bootstrap",
     "proposal": "Clustered SE: CR1 (G/(G-1)) with t on G-1 degrees of freedom. Bootstrap: percentile "
                 "intervals, clusters resampled, B = 10,000, seed 20261014. Kappa: percentile "
                 "bootstrap with whole answers resampled.",
     "where": ("functions.clustered_mean_ci", "functions.cluster_bootstrap_ci",
               "functions.BOOTSTRAP_B", "functions.SEED", "agreement.cohen_kappa_ci"),
     "evidence": "singleton clusters reproduce scipy's sem and t interval; the bootstrap SE matches "
                 "the CR0 SE within 5% (tests/test_stats.py)."},
    {"id": "mde-design-effect", "kind": "extension",
     "topic": "The MDE printed next to a clustered comparison",
     "approved": "eval plan §9 / Table 8: MDE at 80% power, 5% significance, paired, p_d = 0.2-0.3, "
                 "printed next to every claim",
     "proposal": "compare_paired prints Table 8's value as the independent-items planning value "
                 "('mde', with 'mde_note') and, when clusters are given, the design effect (clustered "
                 "/ iid SE of the primary paired difference, squared) and mde * sqrt(max(1, design "
                 "effect)) ('mde_design_adjusted').",
     "where": ("functions.compare_paired", "functions.design_effect", "functions.MDE_NOTE"),
     "evidence": "Exact power of the exact McNemar test at Table 8's MDE with independent items "
                 "(binomial sum, not simulated): K 0.778 / 0.777, F 0.768 / 0.778 at p_d = 0.2 / "
                 "0.3, A 0.830 at p_d = 0.3; at p_d = 0.2 Tier A's MDE (22.9 points) exceeds p_d "
                 "itself and cannot occur. With items correlated in clusters of 5 the cluster test's "
                 "power at the printed MDE fell to 0.57-0.70 (review simulation, cluster SD of "
                 "log-odds 0.5-1.0)."},
    {"id": "split-half-n", "kind": "extension",
     "topic": "Split-half for configurations with n other than 5",
     "approved": "split-half = samples {1,2} vs {3,4} of n = 5 (the run list also has n = 3 and the "
                 "closed frontier model at n = 1, eval plan Table 6 and §7.3)",
     "proposal": "Each configuration is split at its own n into {1..h} vs {h+1..2h}, h = floor(n/2) "
                 "(n = 3: {1} vs {2}); n = 1 cannot be split and is listed as excluded. "
                 "Sensitivity: only configurations with n >= 4.",
     "where": ("functions.split_halves", "functions.split_half_panel",
               "functions.random_half_panel", "functions.SPLIT_HALF_MIN_SAMPLES"),
     "evidence": "n = 5 gives exactly the approved split (tests/test_stats.py)."},
    {"id": "H1a-decision", "kind": "extension",
     "topic": "Decision rule of H1a",
     "approved": "eval plan Table 3: Kendall's tau >= 0.8 between bootstrap resamples and between "
                 "halves of the samples; judge kappa >= 0.7 with 95% lower bound >= 0.6; references "
                 "pass 10 of 10 harness runs",
     "proposal": "Bootstrap part: tau >= 0.8 in at least 95% of the pairs of independent bootstrap "
                 "resamples (undefined taus count as below); the median tau and the share are "
                 "reported beside it, and 'median tau >= 0.8' is the lenient alternative to decide "
                 "between at G2. Split-half part: the split-half tau >= 0.8 (random-half reported, "
                 "not decided on). Judge part: both calibration gates (judge-gate). Harness part: "
                 "every remaining task passed all its runs.",
     "where": ("functions.h1a_decision", "functions.H1A_BOOT_RULE", "functions.H1A_BOOT_SHARE"),
     "evidence": "the median rule would let H1a hold with almost half of the resample pairs below "
                 "0.8 (methodology review M6)."},
    {"id": "H1c-test", "kind": "extension",
     "topic": "The test behind H1c's count of significant pairs",
     "approved": "",
     "proposal": "Both tiers use the same test: Wilcoxon signed-rank on per-item mean scores (Tier "
                 "K's mean credit, Tier F's pass@1), Holm over all pairs, cluster sign-flip form "
                 "with clusters. Each tier's own primary test (K Wilcoxon, F McNemar) is a "
                 "sensitivity analysis only, because mixing tests compares test power, not tiers.",
     "where": ("functions.H1C_TEST", "functions.separation", "functions.size_matched_separation"),
     "evidence": "./gb stats h1c-sim --ratio 1.6 --spread 1.0 (21 models, F items 1.6x as "
                 "discriminating): with K Wilcoxon / F McNemar and the earlier share rule, H1c held "
                 "in 1% of 200 datasets (pairs part 1%), with the same test on both tiers in 100% "
                 "(review simulation c_h1c_alt.py, reproduced)."},
    {"id": "H1c-mixed-n", "kind": "extension",
     "topic": "H1c across a panel with different numbers of samples",
     "approved": "eval plan §9: across the panel of about 21 models; Table 6 / §7.3: n = 5, 3 and 1",
     "proposal": "Every model is reduced on its own samples: per-item mean score (pass@1) for the "
                 "spread and the Wilcoxon test, majority of its own n (ties fail) for the McNemar "
                 "sensitivity. No model is dropped and no sample truncated.",
     "where": ("functions.separation", "functions.size_matched_separation"),
     "evidence": "tests/test_stats.py::test_h1c_runs_on_a_panel_with_n_5_3_1."},
    {"id": "H1c-decision", "kind": "extension",
     "topic": "Decision rule of H1c",
     "approved": "subsample Tier K to 150 items, repeated, report the distribution",
     "proposal": "The subsample distribution is reported descriptively. H1c holds if the 95% "
                 "percentile bootstrap intervals of SD(F) - SD(K at 150) AND of sig_pairs(F) - "
                 "sig_pairs(K at 150) both lie above 0. Each of B = 1,000 replicates resamples Tier "
                 "F's clusters and a fresh 150-item Tier-K subsample's clusters with replacement.",
     "where": ("functions.h1c_decision", "functions.size_matched_separation",
               "functions.H1C_BOOT_B"),
     "evidence": "H1C_SIM_EVIDENCE"},
    {"id": "H4a-decision", "kind": "extension",
     "topic": "Decision rule of H4a",
     "approved": "proposal H4: 4-bit GGUF costs more on agentic (A) than knowledge (K) metrics; "
                 "measured as the score change from BF16 to Q4",
     "proposal": "For the final model of each architecture: the drop from BF16 to Q4_K_M on Tier A "
                 "(per-task success) minus the drop on Tier K (per-item credit), with a 95% percentile "
                 "interval from a stratified cluster bootstrap (each tier's clusters resampled "
                 "independently, B = 10,000). H4a holds if the interval lies above 0. Tier A's MDE "
                 "(23-28 points) is printed beside it.",
     "where": ("functions.compression_cost_difference",),
     "evidence": "tests/test_preregistration.py::test_h4a_decision_rule: equal drops give an interval "
                 "around 0; a 30-point larger Tier-A drop on 30 tasks holds."},
    {"id": "H4b-context", "kind": "extension",
     "topic": "The speed that decides H4b",
     "approved": "proposal H4: the MoE reaches >= 80 tok/s on reference hardware at Q4_K_M; eval plan "
                 "§3: the reference hardware is setup H2 (4 GB GPU, MoE experts in CPU memory)",
     "proposal": "Generation speed (llama-bench tg) of North-Mini-Code at Q4_K_M on H2, median of 5 warm "
                 "runs, at 4K tokens of context. The 16K, 32K and 64K speeds and setups H1 and H3 are "
                 "reported beside it.",
     "where": ("tier_e.bench.CONTEXTS", "tier_e.bench.REPETITIONS"),
     "evidence": "the protocol of eval plan §4.5 (median of 5 warm runs, 4K-64K)."},
    {"id": "judge-gate", "kind": "extension",
     "topic": "What the judge's kappa is computed on",
     "approved": "eval plan §8.1: the judge passes if Cohen's kappa >= 0.7 and the lower end of its "
                 "95% interval >= 0.6",
     "proposal": "docs/guidelines/judge_calibration.md: the rule is applied twice and the judge "
                 "passes only if both hold: (1) unweighted kappa on the credits of every P and N row, "
                 "judge vs the adjudicated human reference (unanimous label, else the adjudicator's), "
                 "whole answers resampled; (2) linear-weighted kappa on the answers' score bands "
                 "0 | (0, 0.5) | [0.5, 1) | 1, each answer one item. Stricter than §8.1 read alone.",
     "where": ("agreement.judge_calibration", "agreement.score_band", "agreement.human_reference"),
     "evidence": "a judge that wrongly zeroes answers through D, DU or H flags fails gate 2 even "
                 "when the point credits agree (tests/test_stats.py)."},
    {"id": "rasch-solved-min", "kind": "extension",
     "topic": "Exploratory Rasch fit on a partial-credit tier",
     "approved": "eval plan §9: a Rasch model is fitted as an exploratory check only",
     "proposal": "On Tier K a model 'solves' an item when its mean credit is at least 0.5; models "
                 "and items with extreme scores are removed first, and no estimates are reported "
                 "when the remaining data are separated.",
     "where": ("items.RASCH_SOLVED_MIN", "items.rasch_fit"),
     "evidence": "exploratory only; nothing is kept or dropped because of it."},
)


def get(proposal_id: str) -> dict:
    """The PROPOSAL with this id."""
    for p in PROPOSALS:
        if p["id"] == proposal_id:
            return p
    raise KeyError(proposal_id)


def as_markdown(rows: list[dict] | None = None) -> str:
    """The list as Markdown, for docs/preregistration/preregistration.md."""
    rows = list(PROPOSALS) if rows is None else rows
    out = ["## Analysis proposals (grcbench.stats)", "",
           "Approved, pre-registered definitions (implementation plan, Phase 5 WS7):", ""]
    out += [f"- **{a['id']}**: {a['definition']}." for a in APPROVED]
    out += ["", "Proposals for G2 (deviation = changes a stated definition; extension = decides an "
            "open point):", ""]
    for p in rows:
        out += [f"### {p['id']} ({p['kind']}): {p['topic']}", ""]
        if p["approved"]:
            out.append(f"- Stated: {p['approved']}.")
        out += [f"- Proposal: {p['proposal']}", f"- Code: {', '.join(p['where'])}",
                f"- Evidence: {p['evidence']}", ""]
    return "\n".join(out)


def as_json(rows: list[dict] | None = None) -> str:
    rows = list(PROPOSALS) if rows is None else rows
    return json.dumps({"approved": list(APPROVED), "proposals": rows}, indent=2, sort_keys=True)

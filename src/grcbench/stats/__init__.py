"""Analysis functions (eval plan §9).

functions  estimates (pass@k, pass^k, Exec@1, Func@1, funnel, per-item scores), clustered
           uncertainty, paired tests (exact and cluster sign-flip McNemar / Wilcoxon) with effect
           sizes and one primary p-value per comparison, reliability (rank stability, split-half
           for mixed-n panels, harness determinism), H1c, H1a/H1c decision rules, seeds, validity
items      difficulty, point-biserial discrimination, review flags, exploratory Rasch fit (with an
           existence check), for pass/fail and partial-credit tiers
agreement  Cohen's kappa (+ bootstrap CI), Scott's pi, Krippendorff's alpha, the three-annotator
           human reference and judge calibration, judge-human gap, per-task-type breakdown
mde        minimum detectable effect, eval plan Table 8
cli        `./gb stats selfcheck` and `./gb stats mde`
"""

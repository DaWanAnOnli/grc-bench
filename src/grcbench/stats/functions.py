"""Estimates, uncertainty, paired comparisons and reliability for GRC-Bench results (eval plan §9).

Implements eval plan §9 "Analysis Plan" (and proposal §8 "Experimental Protocol and Statistical
Analysis"):
  Estimates     "pass@1 from n samples with the unbiased estimator [45]; pass^k from k runs [50]"
                -> pass_at_k (Chen et al. 2021), pass_hat_k (tau-bench, Yao et al. 2025); Exec@1 and
                Func@1 of Tier F (§4.3 "Metrics") -> exec_at_1, func_at_1, funnel; per-item scores of
                pass/fail and partial-credit tiers -> item_pass_at_1, item_scores.
  Uncertainty   "Means with 95% confidence intervals. Items from the same question family or source
                flowgraph are not independent, so we use clustered standard errors and check them with
                a cluster bootstrap [53]" -> cluster_robust_se, clustered_mean_ci,
                cluster_bootstrap_ci, and the paired versions for model differences.
  Comparisons   "Paired tests on shared items (McNemar for pass/fail, Wilcoxon signed-rank for
                scores), Holm-Bonferroni correction within each research question, and effect sizes
                next to p-values" -> mcnemar_exact, mcnemar_clustered, wilcoxon_signed_rank,
                wilcoxon_clustered, holm_bonferroni, cohens_h, matched_odds_ratio,
                paired_difference, compare_paired.
  Reliability   (RQ1a, H1a: Kendall's tau >= 0.8) "Rank stability between bootstrap resamples and
                between halves of the samples [88]; harness determinism from the solvability runs"
                -> rank_stability_bootstrap, split_half_tau / split_half_panel, random_half_tau /
                random_half_panel, harness_determinism, h1a_decision.
  H1c           (RQ1c) "Tier-F scores spread more widely, and more model pairs differ
                significantly, than on Tier K", size-matched -> separation, size_matched_separation
                (subsample distribution + bootstrap interval of F minus K), h1c_decision.
  Seeds         "we report the spread between seeds separately from the item-level interval"
                -> seed_spread.
  Validity      (RQ1b) "Correlations of each tier with published BFCL and MMLU-Pro engineering scores
                across the panel, reported with intervals and described, not tested"
                -> correlation_ci (no p-value is returned).
Item analysis (stats.items), agreement (stats.agreement) and Table 8's MDE (stats.mde) live in the
sibling modules.

Outcome types. A tier's per-sample outcome is "pass_fail" (0/1: Tier F's ladder rungs, Tier A's task
success) or "score" (partial credit in [0, 1]: Tier K, where the rule cascade gives 1, 0.5 or 0 and
the judge a rubric score; eval plan §4.2 "Calculation answers get partial credit by relative error").
§9 names the test for each: McNemar for pass/fail, Wilcoxon signed-rank for scores (PRIMARY_TEST).

Pre-registered definitions (approved implementation plan, Phase 5 WS7):
  * McNemar's per-item pass: an item counts as passed if the MAJORITY of its n samples pass; a tie
    (2 of 4 runs) counts as a fail (majority_pass).
  * Split-half: samples {1, 2} vs {3, 4} of n = 5, the 5th dropped (SPLIT_HALVES), plus a
    random-half bootstrap (random_half_tau / random_half_panel).
  * Func@1's denominator: the tasks that have a V5 oracle (func_at_1; the funnel's V5 rung too).
  * Holm-Bonferroni families: one per research question (holm_bonferroni takes family -> p-values).
  * H1c: Tier K is subsampled, without replacement, to Tier F's item count (150 in v1.0), repeated
    H1C_REPEATS times, and the distribution is reported.
  * Intervals: 95%; bootstrap intervals are percentile intervals over BOOTSTRAP_B resamples with
    seed SEED; clustered standard errors use the CR1 correction G/(G-1) and t with G-1 degrees of
    freedom.
PROPOSALS of the analysis code, to be fixed in the G2 pre-registration (§9: the analysis scripts are
published on OSF at G2). Each is marked "PROPOSAL <id>" where it is defined, and the full list, with
whether it deviates from or only extends the approved definitions, is stats.proposals.PROPOSALS
(`./gb stats proposals`). In this module:
  * primary-test: TIER_OUTCOME / PRIMARY_TEST: Tier K is a score tier (Wilcoxon), Tiers F and A are
    pass/fail tiers (McNemar). compare_paired returns ONE primary p-value ("p_primary"), the one that
    enters Holm, with effect sizes on that test's own estimand; any other p-value is secondary.
  * clustered-tests: confirmatory comparisons pass every item's cluster_id, and the paired tests are
    then the exact cluster sign-flip versions (mcnemar_clustered, wilcoxon_clustered). The
    unclustered tests assume independent items; with correlated items in a cluster they reject a
    true null too often.
  * mde-design-effect: next to Table 8's MDE (independent items), compare_paired reports a design-
    effect-adjusted MDE when clusters are given.
  * split-half-n: each configuration is split at its own n into {1..h} vs {h+1..2h}, h = floor(n/2)
    (split_halves; n = 5 gives the approved {1,2} vs {3,4}, n = 3 gives {1} vs {2}); configurations
    with n = 1 (the closed frontier model, §7.3) cannot be split and are listed as excluded
    (SPLIT_HALF_MIN_SAMPLES). Sensitivity: min_samples = 4.
  * H1a-decision: the bootstrap part holds if tau >= 0.8 in at least 95% of the pairs of bootstrap
    resamples (H1A_BOOT_RULE; the median is reported and available as the lenient alternative), the
    split-half part if the split-half tau >= 0.8, the judge part if both calibration gates pass.
  * H1c-test, H1c-mixed-n, H1c-decision: both tiers are compared with the same test (H1C_TEST);
    every model is reduced on its own samples; H1c holds if the 95% bootstrap intervals of F minus
    size-matched K exclude 0 from above for the spread AND for the number of significant pairs.
  * H4a-decision: the drop from full precision to Q4_K_M on Tier A minus the drop on Tier K, with a
    stratified cluster-bootstrap interval; H4a holds if it lies above 0 (compression_cost_difference).
Every random function takes `seed` and draws from numpy's default_rng(seed), so its output is the
same on every run. Inputs keyed by item id are processed in sorted id order.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence

import numpy as np
from scipy import special as _special
from scipy import stats as _st

from .mde import P_DISCORDANT
from .mde import mde as _mde

ALPHA = 0.05
SEED = 20261014                 # fixed seed for every resampling procedure (decisions of 14 Oct)
BOOTSTRAP_B = 10_000            # resamples for bootstrap confidence intervals
RANK_B = 1_000                  # pairs of resamples for bootstrap rank stability
HALF_REPEATS = 1_000            # random splits for the random-half bootstrap
SPLIT_HALVES = ((0, 1), (2, 3))  # samples {1,2} vs {3,4} (0-based indices); sample 5 is dropped
SPLIT_HALF_MIN_SAMPLES = 2      # PROPOSAL split-half-n: n >= 2 to be split (n = 1 excluded)
TAU_MIN = 0.8                   # H1a: Kendall's tau >= 0.8
H1A_BOOT_RULES = ("share", "median")
H1A_BOOT_RULE = "share"         # PROPOSAL H1a-decision: tau >= 0.8 in at least H1A_BOOT_SHARE ...
H1A_BOOT_SHARE = 0.95           # ... of the pairs of bootstrap resamples
H1C_ITEMS = 150                 # Tier F's test items in v1.0 (eval plan Table 4)
H1C_REPEATS = 1_000             # size-matched Tier-K subsamples (pre-registered, descriptive)
H1C_BOOT_B = 1_000              # PROPOSAL H1c-decision: bootstrap replicates of F minus K
H1C_TEST = "wilcoxon"           # PROPOSAL H1c-test: the same test on both tiers
SOLVABILITY_RUNS = 10           # eval plan §5.4: the reference passes 10 of 10 harness runs
RUNGS = ("V1", "V2", "V3", "V4", "V5")
EXEC_RUNG = "V4"                # Exec@1: the flowgraph runs
FUNC_RUNG = "V5"                # Func@1: it runs and behaves correctly
OUTCOMES = ("pass_fail", "score")
TIER_OUTCOME = {"K": "score", "F": "pass_fail", "A": "pass_fail"}    # PROPOSAL (module docstring)
PRIMARY_TEST = {"pass_fail": "mcnemar", "score": "wilcoxon"}        # eval plan §9 "Comparisons"


# ---------------------------------------------------------------- input checks -------------------

def _vector(x, name: str = "values") -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _bools(x, name: str = "outcomes") -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    if a.dtype != bool and not np.all(np.isin(a, (0, 1))):
        raise ValueError(f"{name} must hold pass/fail values (bool or 0/1); partial-credit scores "
                         "need the 'score' outcome (Wilcoxon), see PRIMARY_TEST")
    return a.astype(bool)


def _unit_scores(x, name: str = "scores") -> np.ndarray:
    a = _vector(x, name)
    if np.any((a < 0.0) | (a > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1] (pass/fail as 0/1, partial credit in between)")
    return a


def _outcome(outcome: str) -> str:
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, not {outcome!r}")
    return outcome


def _counts(outcomes: Sequence) -> tuple[int, int]:
    a = _bools(list(outcomes))
    return int(a.size), int(a.sum())


def _check_nck(n: int, c: int, k: int) -> None:
    if n < 1 or not 0 <= c <= n:
        raise ValueError(f"need n >= 1 and 0 <= c <= n (n={n}, c={c})")
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n (k={k}, n={n})")


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


# ---------------------------------------------------------------- estimates (§9 "Estimates") -----

def pass_at_k(n: int, c: int, k: int = 1) -> float:
    """Unbiased pass@k from n samples of which c pass (Chen et al. 2021): 1 - C(n-c, k) / C(n, k).

    Exact integer binomials; pass@1 = c / n.
    """
    _check_nck(n, c, k)
    if k == 1:
        return c / n                  # the same value, without rounding through 1 - (n-c)/n
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k (tau-bench): the chance that k runs drawn from the n all pass, C(c, k) / C(n, k).

    With n == k this is 1 if all k runs pass, else 0, so its mean over tasks is the share of tasks
    solved in all k runs; with n > k it is the unbiased per-task estimate.
    """
    _check_nck(n, c, k)
    return math.comb(c, k) / math.comb(n, k)


def mean_pass_at_k(outcomes: Mapping[str, Sequence[bool]], k: int = 1) -> float:
    """Mean of the per-item pass@k over items (item -> its samples' pass/fail)."""
    if not outcomes:
        raise ValueError("no items")
    return _mean([pass_at_k(*_counts(outcomes[i]), k) for i in sorted(outcomes)])


def mean_pass_hat_k(outcomes: Mapping[str, Sequence[bool]], k: int) -> float:
    """Mean of the per-task pass^k over tasks (task -> its runs' pass/fail)."""
    if not outcomes:
        raise ValueError("no tasks")
    return _mean([pass_hat_k(*_counts(outcomes[t]), k) for t in sorted(outcomes)])


def item_pass_at_1(outcomes: Mapping[str, Sequence[bool]]) -> dict[str, float]:
    """Per-item pass@1 = c / n of pass/fail samples."""
    out = {}
    for item in sorted(outcomes):
        n, c = _counts(outcomes[item])
        out[item] = c / n
    return out


def _item_value(samples: Sequence, outcome: str) -> float:
    if outcome == "pass_fail":
        n, c = _counts(samples)
        return c / n
    return float(_unit_scores(list(samples), "samples").mean())


def item_scores(outcomes: Mapping[str, Sequence[float]], outcome: str = "pass_fail"
                ) -> dict[str, float]:
    """Per-item score used by the score test and item analysis: pass@1 = c / n for "pass_fail"
    samples, the mean sample score for "score" samples (partial credit in [0, 1], e.g. Tier K)."""
    outcome = _outcome(outcome)
    if not outcomes:
        raise ValueError("no items")
    return {item: _item_value(outcomes[item], outcome) for item in sorted(outcomes)}


def reached(highest_passed: str | None, rung: str) -> bool:
    """Did a ladder result whose highest consecutive passed rung is `highest_passed` reach `rung`?"""
    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung!r}")
    if highest_passed is None:
        return False
    if highest_passed not in RUNGS:
        raise ValueError(f"unknown rung {highest_passed!r}")
    return RUNGS.index(highest_passed) >= RUNGS.index(rung)


def _rung_outcomes(highest: Mapping[str, Sequence[str | None]], rung: str,
                   tasks: Sequence[str]) -> dict[str, list[bool]]:
    out = {}
    for t in tasks:
        samples = list(highest[t])
        if not samples:
            raise ValueError(f"task {t} has no samples")
        out[t] = [reached(h, rung) for h in samples]
    return out


def exec_at_1(highest: Mapping[str, Sequence[str | None]]) -> dict:
    """Exec@1 (eval plan §4.3): pass@1 of "the flowgraph runs" (ladder rung V4 reached), all tasks.

    `highest` maps task id -> the ladder's `highest_passed` ("V1".."V5" or None) of each sample.
    """
    tasks = sorted(highest)
    if not tasks:
        raise ValueError("no tasks")
    per = _rung_outcomes(highest, EXEC_RUNG, tasks)
    return {"value": mean_pass_at_k(per, 1), "n_tasks": len(tasks),
            "n_samples": sum(len(v) for v in per.values()), "rung": EXEC_RUNG}


def _oracle_list(highest: Mapping, oracle_tasks: Collection[str]) -> list[str]:
    oracle = sorted(set(oracle_tasks))
    missing = [t for t in oracle if t not in highest]
    if missing:
        raise ValueError(f"oracle tasks without samples: {missing}")
    return oracle


def func_at_1(highest: Mapping[str, Sequence[str | None]], oracle_tasks: Collection[str]) -> dict:
    """Func@1 (eval plan §4.3): pass@1 of "runs and behaves correctly" (rung V5 reached).

    Pre-registered denominator: only the tasks that have a V5 oracle (`oracle_tasks`); tasks without
    one are scored up to V4 (§4.3 "Coverage") and are excluded here, not counted as failures. Every
    oracle task must have samples.
    """
    oracle = _oracle_list(highest, oracle_tasks)
    if not oracle:
        raise ValueError("no task has a V5 oracle")
    per = _rung_outcomes(highest, FUNC_RUNG, oracle)
    excluded = sorted(set(highest) - set(oracle))
    return {"value": mean_pass_at_k(per, 1), "n_tasks": len(oracle),
            "n_samples": sum(len(v) for v in per.values()), "rung": FUNC_RUNG,
            "excluded_no_oracle": len(excluded)}


def funnel(highest: Mapping[str, Sequence[str | None]], oracle_tasks: Collection[str]) -> dict:
    """pass@1 of reaching each rung (§4.3: "The pass rate at each rung shown as a funnel").

    Rungs V1-V4 are computed over all tasks; V5 only over the tasks with a V5 oracle, because §4.3
    scores tasks without an oracle "up to V4" and Func@1's pre-registered denominator is the oracle
    tasks (so rates["V5"] == func_at_1(...)["value"] and rates["V4"] == exec_at_1(...)["value"]).
    The V5 bar therefore has its own denominator (n_tasks["V5"]) and is not bounded by V4's. With no
    oracle task, V5 is NaN.
    """
    tasks = sorted(highest)
    if not tasks:
        raise ValueError("no tasks")
    oracle = _oracle_list(highest, oracle_tasks)
    rates = {r: mean_pass_at_k(_rung_outcomes(highest, r, tasks), 1) for r in RUNGS[:-1]}
    rates[FUNC_RUNG] = (mean_pass_at_k(_rung_outcomes(highest, FUNC_RUNG, oracle), 1)
                        if oracle else float("nan"))
    n_tasks = {r: len(tasks) for r in RUNGS[:-1]}
    n_tasks[FUNC_RUNG] = len(oracle)
    return {"rates": rates, "n_tasks": n_tasks, "excluded_from_V6": len(tasks) - len(oracle)}


# ---------------------------------------------------------------- clustered uncertainty ----------

def _missing_label(c) -> bool:
    if c is None:
        return True
    if isinstance(c, (float, np.floating)) and math.isnan(c):
        return True
    return isinstance(c, str) and not c.strip()


def cluster_codes(clusters: Sequence | None, n: int) -> tuple[np.ndarray, int]:
    """Integer cluster codes (sorted cluster labels) and the number of clusters G.

    clusters=None makes every item its own cluster. A missing label (None, NaN, empty string) is an
    error: items without a cluster_id must not be pooled silently into one "None" cluster.
    """
    if clusters is None:
        return np.arange(n), n
    labels = list(clusters)
    if len(labels) != n:
        raise ValueError(f"{len(labels)} cluster labels for {n} items")
    bad = [i for i, c in enumerate(labels) if _missing_label(c)]
    if bad:
        raise ValueError(f"missing cluster label at positions {bad[:10]}"
                         f"{' ...' if len(bad) > 10 else ''}: every item needs its cluster_id")
    labels = [str(c) for c in labels]
    index = {u: i for i, u in enumerate(sorted(set(labels)))}
    return np.array([index[c] for c in labels], dtype=np.int64), len(index)


def cluster_robust_se(values: Sequence[float], clusters: Sequence | None = None,
                      correction: str = "CR1") -> float:
    """Cluster-robust (Liang-Zeger) standard error of a mean.

    Var = c * sum_g (sum_{i in g} (y_i - ybar))^2 / N^2, the sandwich estimator of an
    intercept-only regression (Miller 2024 [53], "clustered standard errors"; Cameron and Miller
    2015). correction "CR1" (default) uses c = G / (G - 1); "CR0" uses c = 1. With every item its
    own cluster, CR1 equals the usual s / sqrt(N). NaN when there are fewer than 2 clusters.
    """
    y = _vector(values)
    codes, g = cluster_codes(clusters, y.size)
    if g < 2:
        return float("nan")
    sums = np.bincount(codes, weights=y - y.mean(), minlength=g)
    var = float(np.sum(sums ** 2)) / y.size ** 2
    if correction == "CR1":
        var *= g / (g - 1)
    elif correction != "CR0":
        raise ValueError("correction must be 'CR1' or 'CR0'")
    return math.sqrt(var)


def clustered_mean_ci(values: Sequence[float], clusters: Sequence | None = None,
                      alpha: float = ALPHA) -> dict:
    """Mean with a (1 - alpha) interval: mean +- t_{G-1} * cluster-robust SE (CR1)."""
    y = _vector(values)
    _, g = cluster_codes(clusters, y.size)
    se = cluster_robust_se(y, clusters, "CR1")
    m = float(y.mean())
    if g < 2:
        lo = hi = float("nan")
    else:
        t = float(_st.t.ppf(1.0 - alpha / 2.0, g - 1))
        lo, hi = m - t * se, m + t * se
    return {"mean": m, "se": se, "lo": lo, "hi": hi, "df": g - 1, "n_items": int(y.size),
            "n_clusters": g, "alpha": alpha}


def bootstrap_weights(n_items: int, clusters: Sequence | None = None, B: int = BOOTSTRAP_B,
                      seed: int = SEED) -> np.ndarray:
    """B x n_items integer weights of a cluster bootstrap: each resample draws G clusters with
    replacement, and every item of a drawn cluster gets weight = times its cluster was drawn.

    With `clusters`, the draws are indexed by the sorted cluster labels, so the result does not
    depend on the order of the items; without, each item is its own cluster in the given order.
    """
    if B < 1:
        raise ValueError("B must be >= 1")
    codes, g = cluster_codes(clusters, n_items)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, g, size=(B, g))
    flat = (draws + g * np.arange(B)[:, None]).ravel()
    cluster_w = np.bincount(flat, minlength=B * g).reshape(B, g)
    return cluster_w[:, codes]


def _percentile_ci(stats_: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.quantile(stats_, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def cluster_bootstrap_ci(values: Sequence[float], clusters: Sequence | None = None,
                         B: int = BOOTSTRAP_B, seed: int = SEED, alpha: float = ALPHA) -> dict:
    """Percentile cluster-bootstrap interval for a mean (clusters resampled with replacement).

    Each resample's mean is the item-weighted mean over the items of its drawn clusters.
    """
    y = _vector(values)
    w = bootstrap_weights(y.size, clusters, B, seed)
    means = (w @ y) / w.sum(axis=1)
    lo, hi = _percentile_ci(means, alpha)
    _, g = cluster_codes(clusters, y.size)
    return {"mean": float(y.mean()), "lo": lo, "hi": hi,
            "se_boot": float(means.std(ddof=1)) if B > 1 else float("nan"),
            "B": B, "seed": seed, "n_items": int(y.size), "n_clusters": g, "alpha": alpha}


def _paired(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    x, y = _vector(a, "a"), _vector(b, "b")
    if x.size != y.size:
        raise ValueError("paired samples must have the same length (shared items, same order)")
    return x - y


def paired_cluster_robust_se(a: Sequence[float], b: Sequence[float],
                             clusters: Sequence | None = None, correction: str = "CR1") -> float:
    """Cluster-robust SE of the mean paired difference a - b on shared items."""
    return cluster_robust_se(_paired(a, b), clusters, correction)


def paired_cluster_bootstrap_ci(a: Sequence[float], b: Sequence[float],
                                clusters: Sequence | None = None, B: int = BOOTSTRAP_B,
                                seed: int = SEED, alpha: float = ALPHA) -> dict:
    """Cluster-bootstrap interval of the mean paired difference a - b (same clusters for both)."""
    return cluster_bootstrap_ci(_paired(a, b), clusters, B, seed, alpha)


def paired_difference(a: Sequence[float], b: Sequence[float], clusters: Sequence | None = None,
                      B: int = BOOTSTRAP_B, seed: int = SEED, alpha: float = ALPHA) -> dict:
    """Effect size "difference with CI": mean(a - b) with the clustered t interval and the
    cluster-bootstrap percentile interval."""
    d = _paired(a, b)
    t = clustered_mean_ci(d, clusters, alpha)
    boot = cluster_bootstrap_ci(d, clusters, B, seed, alpha)
    return {"diff": t["mean"], "se": t["se"], "t_lo": t["lo"], "t_hi": t["hi"],
            "boot_lo": boot["lo"], "boot_hi": boot["hi"], "n_items": t["n_items"],
            "n_clusters": t["n_clusters"], "B": B, "seed": seed, "alpha": alpha}


def compression_cost_difference(a_full: Sequence[float], a_comp: Sequence[float], k_full: Sequence[float],
                                k_comp: Sequence[float], a_clusters: Sequence | None = None,
                                k_clusters: Sequence | None = None, B: int = BOOTSTRAP_B, seed: int = SEED,
                                alpha: float = ALPHA) -> dict:
    """PROPOSAL H4a-decision (G2 pre-registration): does compression cost more on Tier A than on Tier K?

    Proposal H4: "4-bit GGUF costs more on agentic (A) than knowledge (K) metrics". Inputs are per-item
    scores of one model at full precision and compressed, on the same items in the same order: Tier A
    per-task success rates, Tier K mean credits. The drop of a tier is mean(full - compressed); the
    statistic is drop_A - drop_K (score units; x100 for percentage points). Its percentile interval
    comes from a stratified cluster bootstrap: each of B replicates resamples Tier A's clusters and
    Tier K's clusters independently, with replacement (seeds `seed` and `seed + 1`). H4a holds if the
    whole interval lies above 0.
    """
    d_a, d_k = _paired(a_full, a_comp), _paired(k_full, k_comp)
    w_a = bootstrap_weights(d_a.size, a_clusters, B, seed)
    w_k = bootstrap_weights(d_k.size, k_clusters, B, seed + 1)
    stats_ = (w_a @ d_a) / w_a.sum(axis=1) - (w_k @ d_k) / w_k.sum(axis=1)
    lo, hi = _percentile_ci(stats_, alpha)
    return {"drop_A": float(d_a.mean()), "drop_K": float(d_k.mean()),
            "difference": float(d_a.mean() - d_k.mean()), "lo": lo, "hi": hi, "holds": bool(lo > 0),
            "n_items": {"A": int(d_a.size), "K": int(d_k.size)},
            "n_clusters": {"A": cluster_codes(a_clusters, d_a.size)[1], "K": cluster_codes(k_clusters, d_k.size)[1]},
            "B": B, "seed": seed, "alpha": alpha}


# ---------------------------------------------------------------- paired tests -------------------

def majority_pass(samples: Sequence[bool]) -> bool:
    """Pre-registered per-item reduction for McNemar: passed iff MORE than half of the n samples
    pass. A tie (e.g. 2 of 4 runs) is a fail; with n = 5 an item needs 3 passing samples."""
    n, c = _counts(samples)
    return 2 * c > n


def mcnemar_exact_p(a_only, b_only):
    """Two-sided exact McNemar p-value: a binomial test of a_only successes in a_only + b_only
    discordant items at 1/2, p = min(1, 2 * P[X <= min(a_only, b_only)]); 1 with no discordant
    items. Works elementwise on arrays."""
    x = np.asarray(a_only, dtype=np.int64)
    y = np.asarray(b_only, dtype=np.int64)
    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("counts must be >= 0")
    m = x + y
    p = np.minimum(1.0, 2.0 * _st.binom.cdf(np.minimum(x, y), m, 0.5))
    p = np.where(m == 0, 1.0, p)
    return float(p) if p.ndim == 0 else p


def mcnemar_exact(a_pass: Sequence[bool], b_pass: Sequence[bool]) -> dict:
    """Exact McNemar test on per-item pass/fail of two models on the same items (same order).

    Assumes independent items; with clustered items use mcnemar_clustered."""
    a, b = _bools(a_pass, "a_pass"), _bools(b_pass, "b_pass")
    if a.size != b.size:
        raise ValueError("paired outcomes must have the same length")
    a_only, b_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return {"n_items": int(a.size), "a_only": a_only, "b_only": b_only,
            "both": int(np.sum(a & b)), "neither": int(np.sum(~a & ~b)),
            "discordant_share": (a_only + b_only) / a.size,
            "p_value": mcnemar_exact_p(a_only, b_only), "method": "exact binomial (items independent)"}


def mcnemar_from_samples(a: Mapping[str, Sequence[bool]], b: Mapping[str, Sequence[bool]]) -> dict:
    """McNemar on the shared items after the pre-registered majority reduction of each item's
    samples (ties fail)."""
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("no shared items")
    res = mcnemar_exact([majority_pass(a[i]) for i in shared], [majority_pass(b[i]) for i in shared])
    res["n_unshared"] = len(set(a) ^ set(b))
    res["reduction"] = "majority of samples pass; ties fail"
    return res


def _signflip_p(contrib) -> float:
    """Exact two-sided p-value of S = sum_g s_g c_g with independent fair signs s_g = +-1, observed
    at every s_g = +1: P(|S| >= |sum_g c_g|). contrib holds integers (one per cluster).

    The null distribution is built exactly by dynamic programming over the clusters (each step
    halves the mass and shifts it by +-|c_g|), so there is no Monte Carlo error and no seed.
    """
    c = np.asarray(contrib, dtype=np.int64)
    v = np.abs(c[c != 0])
    obs = abs(int(c.sum()))
    if v.size == 0 or obs == 0:
        return 1.0
    g = int(np.gcd.reduce(v))
    v = np.sort(v // g)
    obs //= g
    total = int(v.sum())
    dist = np.zeros(2 * total + 1)
    dist[total] = 1.0
    reach = 0
    for x in v.tolist():
        lo, hi = total - reach, total + reach + 1
        seg = 0.5 * dist[lo:hi]
        dist[lo:hi] = 0.0
        dist[lo - x:hi - x] += seg
        dist[lo + x:hi + x] += seg
        reach += x
    return float(min(1.0, dist[:total - obs + 1].sum() + dist[total + obs:].sum()))


def mcnemar_clustered(a_pass: Sequence[bool], b_pass: Sequence[bool],
                      clusters: Sequence | None = None) -> dict:
    """Cluster-aware exact McNemar test on per-item pass/fail (eval plan §9: items of one question
    family or source flowgraph are not independent).

    Statistic D = a_only - b_only = sum_g D_g, with D_g = sum over cluster g's items of (a_i - b_i).
    Under H0 the two models are exchangeable within each cluster, so each D_g is symmetric about 0
    and, given |D_g|, D is distributed as sum_g s_g |D_g| with independent fair signs. The p-value is
    computed exactly from that distribution (_signflip_p). This is the exact randomisation form of
    the clustered McNemar test of Durkalski et al. (2003, Stat Med 22:2417), whose chi-square
    D^2 / sum_g D_g^2 (reported as "durkalski_chi2") is its normal approximation. With every item
    its own cluster (clusters=None) the p-value equals mcnemar_exact's. With few informative
    clusters the smallest attainable p-value is 2 / 2^G' (G' = clusters with D_g != 0).
    """
    a, b = _bools(a_pass, "a_pass"), _bools(b_pass, "b_pass")
    if a.size != b.size:
        raise ValueError("paired outcomes must have the same length")
    codes, g = cluster_codes(clusters, a.size)
    d_g = np.rint(np.bincount(codes, weights=a.astype(float) - b.astype(float),
                              minlength=g)).astype(np.int64)
    a_only, b_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    ss = int(np.sum(d_g ** 2))
    return {"n_items": int(a.size), "a_only": a_only, "b_only": b_only,
            "both": int(np.sum(a & b)), "neither": int(np.sum(~a & ~b)),
            "discordant_share": (a_only + b_only) / a.size, "n_clusters": g,
            "n_clusters_informative": int(np.sum(d_g != 0)), "statistic": a_only - b_only,
            "durkalski_chi2": (a_only - b_only) ** 2 / ss if ss else 0.0,
            "p_value": _signflip_p(d_g), "method": "cluster sign-flip, exact"}


def _signed_ranks(d: np.ndarray) -> np.ndarray:
    """Signed average ranks of |d| over the non-zero differences (0 for zero differences)."""
    sr = np.zeros(d.size)
    nz = d != 0
    if nz.any():
        sr[nz] = np.sign(d[nz]) * _st.rankdata(np.abs(d[nz]))
    return sr


def _rank_effect(sr: np.ndarray) -> dict:
    t_plus, t_minus = float(sr[sr > 0].sum()), float(-sr[sr < 0].sum())
    tot = t_plus + t_minus
    return {"t_plus": t_plus, "t_minus": t_minus,
            "rank_biserial": (t_plus - t_minus) / tot if tot else 0.0}


def wilcoxon_signed_rank(a_scores: Sequence[float], b_scores: Sequence[float],
                         alternative: str = "two-sided") -> dict:
    """Wilcoxon signed-rank test (scipy.stats.wilcoxon) on paired per-item scores, items assumed
    independent (with clustered items use wilcoxon_clustered).

    Differences are rounded to 12 decimals first, so scores on a grid (pass@1 = c/n) tie exactly,
    and zero differences are dropped (Wilcoxon's "wilcox" rule). The null distribution is chosen
    from the NON-ZERO differences and recorded in "method"; scipy's own "auto" counts the zeros
    too, so it falls back to the normal approximation earlier:
      "exact"        no tied |differences| and at most 50 of them (the exact signed-rank law);
      "permutation"  ties and at most 13 non-zero differences: all 2^n sign flips enumerated (exact
                     given the ties; scipy.stats.PermutationMethod, deterministic);
      "asymptotic"   otherwise: normal approximation with the tie correction, no continuity
                     correction.
    All differences zero -> p = 1. Effect size: the matched-pairs rank-biserial correlation
    (T+ - T-) / (T+ + T-) (Kerby 2014), next to the p-value as §9 asks.
    """
    d = np.round(_paired(a_scores, b_scores), 12)
    nz = d[d != 0]
    base = {"n_items": int(d.size), "n_nonzero": int(nz.size), "mean_diff": float(d.mean()),
            "alternative": alternative, **_rank_effect(_signed_ranks(d))}
    if nz.size == 0:
        return {**base, "statistic": 0.0, "p_value": 1.0, "method": "none (all differences zero)"}
    tied = np.unique(np.abs(nz)).size < nz.size
    if not tied and nz.size <= 50:
        name, method = "exact", "exact"
    elif nz.size <= 13:
        name, method = "permutation", _st.PermutationMethod(n_resamples=9999)
    else:
        name, method = "asymptotic", "asymptotic"
    res = _st.wilcoxon(nz, zero_method="wilcox", correction=False, alternative=alternative,
                       method=method)
    return {**base, "statistic": float(res.statistic), "p_value": float(res.pvalue), "method": name}


def wilcoxon_clustered(a_scores: Sequence[float], b_scores: Sequence[float],
                       clusters: Sequence | None = None) -> dict:
    """Cluster-aware Wilcoxon signed-rank test on paired per-item scores (two-sided, exact).

    The |differences| of all items are ranked together (average ranks; zero differences dropped,
    as in wilcoxon_signed_rank), and S = T+ - T- = sum_g R_g, with R_g the sum of the signed ranks
    in cluster g. Under H0 the two models are exchangeable within each cluster, so the signs of
    whole clusters can be flipped: given the ranks, S is distributed as sum_g s_g |R_g| with
    independent fair signs, enumerated exactly (_signflip_p). This keeps the within-cluster
    correlation that the item-level test ignores (cf. the clustered signed-rank test of Rosner,
    Glynn and Lee 2006, Biometrics 62:185). With every item its own cluster (clusters=None) it is
    the exact signed-rank test with average ranks for ties, for any number of items (it equals
    scipy's "exact" and "permutation" p-values where those apply).
    """
    d = np.round(_paired(a_scores, b_scores), 12)
    codes, g = cluster_codes(clusters, d.size)
    sr = _signed_ranks(d)
    r_g = np.rint(2.0 * np.bincount(codes, weights=sr, minlength=g)).astype(np.int64)
    eff = _rank_effect(sr)
    return {"n_items": int(d.size), "n_nonzero": int(np.sum(d != 0)), "mean_diff": float(d.mean()),
            "n_clusters": g, "n_clusters_informative": int(np.sum(r_g != 0)),
            "statistic": eff["t_plus"] - eff["t_minus"], **eff,
            "p_value": _signflip_p(r_g), "method": "cluster sign-flip, exact"}


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order:
    p_adj_(i) = max_{j <= i} min(1, (m - j + 1) * p_(j)) over the ascending p_(1..m)."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be 1-D")
    if p.size and (np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj
    return out


def _p_of(value) -> float:
    if isinstance(value, Mapping):
        if "p_primary" not in value:
            raise ValueError("a comparison result must carry 'p_primary' (compare_paired)")
        return float(value["p_primary"])
    return float(value)


def holm(p_values: Mapping[str, float | Mapping], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm-Bonferroni within ONE family: name -> {"p", "p_holm", "reject"} (reject iff
    p_holm <= alpha, the same as Holm's step-down rule). A value may be a p-value or a
    compare_paired result, whose "p_primary" (and only that) enters the family."""
    names = sorted(p_values)
    ps = [_p_of(p_values[n]) for n in names]
    adj = holm_adjust(ps)
    return {n: {"p": p, "p_holm": float(a), "reject": bool(a <= alpha)}
            for n, p, a in zip(names, ps, adj)}


def holm_bonferroni(families: Mapping[str, Mapping[str, float | Mapping]],
                    alpha: float = ALPHA) -> dict:
    """Holm-Bonferroni applied separately within each family (pre-registered: one family per
    research question). family -> {hypothesis -> p or compare_paired result} gives
    family -> holm(...)."""
    return {fam: holm(families[fam], alpha) for fam in sorted(families)}


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h between two proportions: 2 asin(sqrt(p1)) - 2 asin(sqrt(p2))."""
    for p in (p1, p2):
        if not 0.0 <= p <= 1.0:
            raise ValueError("proportions must lie in [0, 1]")
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


def matched_odds_ratio(a_only: int, b_only: int, alpha: float = ALPHA) -> dict:
    """Matched-pairs (McNemar) odds ratio a_only / b_only with its exact conditional interval:
    the Clopper-Pearson interval of pi = a_only / (a_only + b_only) mapped by pi / (1 - pi)."""
    if a_only < 0 or b_only < 0:
        raise ValueError("counts must be >= 0")
    m = a_only + b_only
    if m == 0:
        nan = float("nan")
        return {"odds_ratio": nan, "lo": nan, "hi": nan, "a_only": 0, "b_only": 0, "alpha": alpha}
    ci = _st.binomtest(a_only, m, 0.5).proportion_ci(confidence_level=1.0 - alpha, method="exact")

    def odds(p: float) -> float:
        return math.inf if p >= 1.0 else p / (1.0 - p)

    return {"odds_ratio": math.inf if b_only == 0 else a_only / b_only,
            "lo": odds(float(ci.low)), "hi": odds(float(ci.high)),
            "a_only": a_only, "b_only": b_only, "alpha": alpha}


def _mde_row(n: int, alpha: float, scale: float = 1.0) -> dict:
    return {f"pd_{pd}": scale * _mde(n, pd, alpha) for pd in P_DISCORDANT}


MDE_NOTE = ("Table 8 planning value: independent items, normal approximation to the paired test, "
            "80% power at 5% significance; the exact McNemar test reaches slightly less power at "
            "it, and correlated items in clusters lower power further (see mde_design_adjusted)")


def design_effect(d: Sequence[float], clusters: Sequence | None) -> float:
    """Design effect of the mean paired difference: (clustered SE / independent-items SE)^2, both
    CR1 (cluster_robust_se). NaN without clusters, with fewer than 2 clusters, or when the
    differences do not vary."""
    if clusters is None:
        return float("nan")
    x = _vector(d, "differences")
    se_iid = cluster_robust_se(x)
    se_cl = cluster_robust_se(x, clusters)
    if not (math.isfinite(se_cl) and se_iid > 0):
        return float("nan")
    return (se_cl / se_iid) ** 2


def compare_paired(a: Mapping[str, Sequence[float]], b: Mapping[str, Sequence[float]],
                   clusters: Mapping[str, str] | None = None, outcome: str = "pass_fail",
                   B: int = BOOTSTRAP_B, seed: int = SEED, alpha: float = ALPHA) -> dict:
    """One model comparison on shared items (eval plan §9 "Comparisons").

    a, b: item -> the model's per-sample outcomes (0/1 for outcome "pass_fail"; scores in [0, 1]
    for "score", e.g. Tier K). clusters: item -> cluster_id; confirmatory comparisons pass it
    (PROPOSAL), and the primary test is then the exact cluster sign-flip form.

    The primary test is PRIMARY_TEST[outcome], and every effect size under "primary" refers to the
    same estimand as its p-value:
      pass_fail  McNemar on the majority-reduced items (ties fail). Estimand: the difference in the
                 share of items passed by majority, with clustered-t and cluster-bootstrap
                 intervals; Cohen's h on those two shares; the matched-pairs odds ratio
                 a_only / b_only.
      score      Wilcoxon signed-rank on the per-item mean scores. Estimand: the mean per-item score
                 difference with its intervals; the matched-pairs rank-biserial correlation.
    "p_primary" is the p-value that enters holm_bonferroni. For pass_fail, "secondary" is the
    Wilcoxon test of the per-item pass@1 with the pass@1 difference (the headline estimate's own
    comparison). It is labelled secondary and kept out of Holm: the two estimands can disagree in
    sign (a model can pass more items by majority while having the lower mean pass@1).
    "mde" is the a-priori minimum detectable effect for this number of shared items under Table
    8's assumption p_d = 0.2-0.3 (eval plan §9: "We print the MDE next to every claim"). It is the
    planning value for INDEPENDENT items ("mde_note"): the normal approximation behind Table 8, at
    which the exact McNemar test has slightly less than 80% power; for scores in [0, 1] it is
    conservative in its variance, since a paired difference d in [-1, 1] has Var(d) <= P(d != 0).
    With clusters (PROPOSAL mde-design-effect) the primary test is the cluster sign-flip test, whose
    power falls when items in a cluster are correlated, so "design_effect" = (clustered / iid SE of
    the primary estimand's paired difference)^2 and "mde_design_adjusted" = mde * sqrt(max(1,
    design_effect)) are reported next to it (the design effect is floored at 1, so a noisy estimate
    below 1 never claims more sensitivity than Table 8). Without clusters both are None.
    "mde_at_observed_pd" (descriptive) uses this comparison's own share of differing items; it is
    not a power analysis of the observed effect.
    """
    outcome = _outcome(outcome)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("no shared items")
    n = len(shared)
    if clusters is not None:
        missing = [i for i in shared if i not in clusters]
        if missing:
            raise ValueError(f"items without a cluster_id: {missing[:10]}")
    cl = None if clusters is None else [clusters[i] for i in shared]
    _, g = cluster_codes(cl, n)
    sa = [_item_value(a[i], outcome) for i in shared]
    sb = [_item_value(b[i], outcome) for i in shared]
    score_test = wilcoxon_clustered if cl is not None else (lambda x, y, _c: wilcoxon_signed_rank(x, y))
    score_name = "wilcoxon_clustered" if cl is not None else "wilcoxon_signed_rank"
    if outcome == "pass_fail":
        ma = np.array([majority_pass(a[i]) for i in shared], dtype=float)
        mb = np.array([majority_pass(b[i]) for i in shared], dtype=float)
        test = mcnemar_clustered(ma, mb, cl) if cl is not None else mcnemar_exact(ma, mb)
        share_a, share_b = float(ma.mean()), float(mb.mean())
        primary = {"test": "mcnemar_clustered" if cl is not None else "mcnemar_exact",
                   "estimand": "share of items passed by a majority of samples (ties fail), a - b",
                   "p_value": test["p_value"], "share_a": share_a, "share_b": share_b,
                   "difference": paired_difference(ma, mb, cl, B, seed, alpha),
                   "cohens_h": cohens_h(share_a, share_b),
                   "odds_ratio": matched_odds_ratio(test["a_only"], test["b_only"], alpha),
                   "detail": test}
        sec = score_test(sa, sb, cl)
        secondary = {"test": score_name, "estimand": "mean per-item pass@1, a - b",
                     "p_value": sec["p_value"], "difference": paired_difference(sa, sb, cl, B, seed,
                                                                                alpha),
                     "rank_biserial": sec["rank_biserial"], "detail": sec,
                     "note": "secondary: reported, not entered into Holm"}
        p_d = test["discordant_share"]
        d_primary = ma - mb
    else:
        test = score_test(sa, sb, cl)
        primary = {"test": score_name, "estimand": "mean per-item score, a - b",
                   "p_value": test["p_value"], "difference": paired_difference(sa, sb, cl, B, seed,
                                                                               alpha),
                   "rank_biserial": test["rank_biserial"], "detail": test}
        secondary = None
        p_d = test["n_nonzero"] / n
        d_primary = np.asarray(sa) - np.asarray(sb)
    deff = design_effect(d_primary, cl) if cl is not None else None
    if deff is None:
        adjusted = None
    elif math.isfinite(deff):
        adjusted = _mde_row(n, alpha, math.sqrt(max(1.0, deff)))
    else:
        adjusted = {k: float("nan") for k in _mde_row(n, alpha)}
    return {"outcome": outcome, "n_items": n, "n_unshared": len(set(a) ^ set(b)),
            "n_clusters": g, "clustered": cl is not None,
            "mean_a": _mean(sa), "mean_b": _mean(sb),
            "primary": primary, "secondary": secondary, "p_primary": primary["p_value"],
            "mde": _mde_row(n, alpha), "mde_note": MDE_NOTE,
            "design_effect": deff, "mde_design_adjusted": adjusted,
            "mde_at_observed_pd": _mde(n, p_d, alpha) if p_d > 0 else float("nan")}


# ---------------------------------------------------------------- reliability (RQ1a) -------------

def _model_samples(results: Mapping[str, Mapping[str, Sequence[float]]]
                   ) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    """model -> item -> samples  ->  (models, items, model -> items x n array). Every model must have
    the same items, and every item of one model the same number of samples n (n may differ
    between models)."""
    models = sorted(results)
    if not models:
        raise ValueError("no models")
    items = sorted(results[models[0]])
    if not items:
        raise ValueError(f"model {models[0]} has no items")
    out = {}
    for m in models:
        if sorted(results[m]) != items:
            raise ValueError(f"model {m} does not have the same items as {models[0]}")
        ns = sorted({len(results[m][i]) for i in items})
        if len(ns) != 1 or ns[0] == 0:
            raise ValueError(f"model {m} has {ns} samples per item; every item of one model needs "
                             "the same number (>= 1) of samples")
        arr = np.array([[float(x) for x in results[m][i]] for i in items])
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"model {m} has non-finite sample scores")
        out[m] = arr
    return models, items, out


def sample_tensor(results: Mapping[str, Mapping[str, Sequence[float]]]
                  ) -> tuple[list[str], list[str], np.ndarray]:
    """model -> item -> per-sample scores  ->  (models, items, array models x items x samples).

    Every model must have the same items and the same number of samples per item. A panel whose
    configurations have different n (the run list's n = 5, n = 3 and n = 1 configurations) is
    rejected with a clear message: the panel functions take the model -> item -> samples mapping
    itself and reduce each model on its own n (separation / size_matched_separation for H1c,
    split_half_panel / random_half_panel, items.matrix_from_samples for per-item scores).
    """
    models, items, arrs = _model_samples(results)
    ns = {m: arrs[m].shape[1] for m in models}
    if len(set(ns.values())) != 1:
        raise ValueError(f"models have different numbers of samples per item ({ns}); pass the "
                         "model -> item -> samples mapping to separation / "
                         "size_matched_separation, split_half_panel / random_half_panel or "
                         "items.matrix_from_samples(results, 'mean'), which reduce each model on "
                         "its own n")
    return models, items, np.stack([arrs[m] for m in models])


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b between two score vectors over the same models (NaN if undefined, e.g. all
    models tied in one vector). Rounded to 12 decimals: scipy's tau-b of two identical rankings
    can come out as 0.9999999999999999, which must not fail a tau >= threshold rule."""
    res = _st.kendalltau(np.asarray(x, float), np.asarray(y, float))
    return round(float(res.statistic), 12)


def summarize(values: Sequence[float], threshold: float | None = None) -> dict:
    """Distribution summary (NaNs counted, not summarised): mean, median, 2.5/97.5 percentiles."""
    v = np.asarray(values, dtype=float)
    ok = v[np.isfinite(v)]
    out = {"n": int(ok.size), "n_undefined": int(v.size - ok.size)}
    if ok.size:
        lo, hi = np.quantile(ok, [0.025, 0.975])
        out.update(mean=float(ok.mean()), median=float(np.median(ok)), p2_5=float(lo),
                   p97_5=float(hi), min=float(ok.min()), max=float(ok.max()))
        if threshold is not None:
            out["share_ge_threshold"] = float(np.mean(ok >= threshold))
            out["threshold"] = threshold
    return out


def _model_item_matrix(scores) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if s.ndim != 2 or s.shape[0] < 2 or s.shape[1] < 1:
        raise ValueError("scores must be a models x items matrix with at least 2 models")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores contain non-finite values")
    return s


def rank_stability_bootstrap(scores, clusters: Sequence | None = None, B: int = RANK_B,
                             seed: int = SEED) -> dict:
    """Rank stability between bootstrap resamples (H1a: tau >= 0.8).

    scores: models x items matrix of per-item scores (pass@1 or mean score; for a panel whose
    models have different n, build it with items.matrix_from_samples(results, "mean")). Items (or,
    with `clusters`, clusters of items) are resampled 2B times; the primary statistic is Kendall's
    tau between the model means of independent resamples 2j and 2j+1 (B values); tau between each
    resample and the full-data ranking is reported too.
    """
    s = _model_item_matrix(scores)
    w = bootstrap_weights(s.shape[1], clusters, 2 * B, seed)
    means = (w @ s.T) / w.sum(axis=1, keepdims=True)
    full = s.mean(axis=1)
    pairs = [kendall_tau(means[2 * j], means[2 * j + 1]) for j in range(B)]
    vs_full = [kendall_tau(full, means[r]) for r in range(2 * B)]
    _, g = cluster_codes(clusters, s.shape[1])
    return {"tau_between_resamples": summarize(pairs, TAU_MIN),
            "tau_vs_full": summarize(vs_full, TAU_MIN), "B": B, "seed": seed,
            "n_models": int(s.shape[0]), "n_items": int(s.shape[1]), "n_clusters": g}


def _sample_array(samples) -> np.ndarray:
    x = np.asarray(samples, dtype=float)
    if x.ndim != 3 or x.shape[0] < 2:
        raise ValueError("samples must be a models x items x samples array with at least 2 models")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples contain non-finite values")
    return x


def split_halves(n: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The split-half rule for n samples per item (0-based sample indices): {1..h} vs {h+1..2h},
    h = floor(n/2), the rest dropped. n = 5 gives the pre-registered {1,2} vs {3,4} with the 5th
    dropped (SPLIT_HALVES); n = 4 the same with nothing dropped; n = 3 gives {1} vs {2} with the
    3rd dropped (PROPOSAL for the run list's n = 3 configurations). n = 1 cannot be split."""
    if n < 2:
        raise ValueError(f"cannot split {n} sample(s) per item into two halves (needs n >= 2)")
    h = n // 2
    return tuple(range(h)), tuple(range(h, 2 * h))


def split_half_tau(samples, halves: tuple[Sequence[int], Sequence[int]] | None = None) -> dict:
    """Pre-registered split-half rank stability: Kendall's tau between the model rankings computed
    from samples {1,2} and from samples {3,4} (the 5th sample of n = 5 is dropped).

    samples: models x items x samples array of per-sample scores (0/1 or partial credit), the
    same n for every model. halves=None applies split_halves(n); for a panel with different n
    per model use split_half_panel.
    """
    x = _sample_array(samples)
    if halves is None:
        halves = split_halves(x.shape[2])
    h1, h2 = list(halves[0]), list(halves[1])
    if not h1 or not h2 or set(h1) & set(h2) or max(h1 + h2) >= x.shape[2] or min(h1 + h2) < 0:
        raise ValueError(f"halves {halves} do not fit {x.shape[2]} samples per item")
    a = x[:, :, h1].mean(axis=(1, 2))
    b = x[:, :, h2].mean(axis=(1, 2))
    dropped = sorted(set(range(x.shape[2])) - set(h1) - set(h2))
    return {"tau": kendall_tau(a, b), "half_1_samples": [i + 1 for i in h1],
            "half_2_samples": [i + 1 for i in h2], "dropped_samples": [i + 1 for i in dropped],
            "n_models": int(x.shape[0]), "n_items": int(x.shape[1])}


def random_half_tau(samples, repeats: int = HALF_REPEATS, seed: int = SEED) -> dict:
    """Random-half bootstrap: for each repeat, the n samples of every (model, item) are split at
    random into two halves of floor(n/2) (one sample dropped when n is odd), and Kendall's tau
    between the two halves' model rankings is recorded. The same n for every model; for a panel
    with different n use random_half_panel."""
    x = _sample_array(samples)
    h = x.shape[2] // 2
    if h < 1:
        raise ValueError("need at least 2 samples per item")
    rng = np.random.default_rng(seed)
    taus = []
    for _ in range(repeats):
        perm = np.argsort(rng.random(x.shape), axis=-1)
        xp = np.take_along_axis(x, perm, axis=-1)
        taus.append(kendall_tau(xp[:, :, :h].mean(axis=(1, 2)), xp[:, :, h:2 * h].mean(axis=(1, 2))))
    return {"tau": summarize(taus, TAU_MIN), "half_size": h, "repeats": repeats, "seed": seed,
            "n_models": int(x.shape[0]), "n_items": int(x.shape[1])}


def _panel_population(results, min_samples: int) -> tuple[list[str], dict[str, int], list[str],
                                                          dict[str, np.ndarray]]:
    models, items, arrs = _model_samples(results)
    need = max(2, int(min_samples))
    used = [m for m in models if arrs[m].shape[1] >= need]
    excluded = {m: int(arrs[m].shape[1]) for m in models if arrs[m].shape[1] < need}
    if len(used) < 2:
        raise ValueError(f"fewer than 2 configurations have >= {need} samples per item "
                         f"(excluded: {excluded})")
    return used, excluded, items, arrs


def split_half_panel(results: Mapping[str, Mapping[str, Sequence[float]]],
                     min_samples: int = SPLIT_HALF_MIN_SAMPLES) -> dict:
    """Split-half rank stability over a panel whose configurations have different n (the run list:
    n = 5 for Table 6's configurations, n = 3 for seeds, comparison arms and the outside panel, n = 1
    for the closed frontier model).

    results: model -> item -> per-sample scores. Each model is split at its own n by split_halves
    (n = 5: the pre-registered {1,2} vs {3,4}); models with fewer than `min_samples` samples per
    item (default 2: the n = 1 frontier model) cannot be split and are listed under "excluded".
    Kendall's tau is taken between the included models' half-1 and half-2 means. Sensitivity:
    min_samples = 4 keeps only the configurations the approved rule was written for.
    """
    used, excluded, items, arrs = _panel_population(results, min_samples)
    a, b, by_n = [], [], {}
    for m in used:
        n = arrs[m].shape[1]
        h1, h2 = split_halves(n)
        a.append(float(arrs[m][:, list(h1)].mean()))
        b.append(float(arrs[m][:, list(h2)].mean()))
        by_n[str(n)] = {"half_1_samples": [i + 1 for i in h1], "half_2_samples": [i + 1 for i in h2],
                        "dropped_samples": [i + 1 for i in range(2 * len(h1), n)]}
    return {"tau": kendall_tau(a, b), "models": used, "excluded": excluded, "halves_by_n": by_n,
            "min_samples": max(2, int(min_samples)), "n_models": len(used), "n_items": len(items)}


def random_half_panel(results: Mapping[str, Mapping[str, Sequence[float]]],
                      repeats: int = HALF_REPEATS, seed: int = SEED,
                      min_samples: int = SPLIT_HALF_MIN_SAMPLES) -> dict:
    """Random-half bootstrap over a panel with different n per model: in each repeat, every
    (model, item)'s n samples are shuffled and split into two halves of floor(n/2) of that model's
    n; tau between the halves' model means is recorded. Same population rule as split_half_panel.
    """
    used, excluded, items, arrs = _panel_population(results, min_samples)
    rng = np.random.default_rng(seed)
    taus = []
    for _ in range(repeats):
        a, b = [], []
        for m in used:
            x = arrs[m]
            h = x.shape[1] // 2
            xp = np.take_along_axis(x, np.argsort(rng.random(x.shape), axis=1), axis=1)
            a.append(float(xp[:, :h].mean()))
            b.append(float(xp[:, h:2 * h].mean()))
        taus.append(kendall_tau(a, b))
    return {"tau": summarize(taus, TAU_MIN), "models": used, "excluded": excluded,
            "half_sizes": {m: int(arrs[m].shape[1] // 2) for m in used}, "repeats": repeats,
            "seed": seed, "n_models": len(used), "n_items": len(items)}


def harness_determinism(runs: Mapping[str, Sequence[bool]], required_runs: int = SOLVABILITY_RUNS,
                        digests: Mapping[str, Sequence[str]] | None = None) -> dict:
    """Harness determinism from the solvability runs (eval plan §5.4: the reference solution of every
    task passes every check in 10 of 10 harness runs).

    runs: task -> pass/fail of each repeated run. A task is "pass_all" (solvable) only with at least
    `required_runs` runs that all pass; "flaky" when some pass and some fail; "fail_all"; or
    "too_few_runs". digests (optional): task -> an output digest per run; tasks whose digests differ
    between runs are listed under "varying_outputs".
    """
    status: dict[str, str] = {}
    for t in sorted(runs):
        n, c = _counts(runs[t])
        if c and c < n:
            status[t] = "flaky"
        elif c == 0:
            status[t] = "fail_all"
        elif n < required_runs:
            status[t] = "too_few_runs"
        else:
            status[t] = "pass_all"
    varying = sorted(t for t, d in (digests or {}).items() if len(set(d)) > 1)
    counts = {s: sum(1 for v in status.values() if v == s)
              for s in ("pass_all", "flaky", "fail_all", "too_few_runs")}
    n_tasks = len(status)
    return {"n_tasks": n_tasks, **{f"n_{k}": v for k, v in counts.items()},
            "share_pass_all": counts["pass_all"] / n_tasks if n_tasks else float("nan"),
            "required_runs": required_runs,
            "not_solvable": sorted(t for t, s in status.items() if s != "pass_all"),
            "varying_outputs": varying, "status": status}


def h1a_decision(rank_bootstrap: Mapping, split_half: Mapping,
                 judge: Mapping | None = None, determinism: Mapping | None = None,
                 threshold: float = TAU_MIN, bootstrap_rule: str = H1A_BOOT_RULE,
                 share_min: float = H1A_BOOT_SHARE) -> dict:
    """PROPOSAL H1a-decision (G2 pre-registration): the decision rule of H1a (eval plan Table 3:
    "Model rankings are stable (Kendall's tau >= 0.8 between bootstrap resamples and between halves
    of the samples). The judge agrees with people at kappa >= 0.7, with the lower end of its 95%
    interval >= 0.6. Reference solutions pass 10 of 10 harness runs.").

    Parts (each holds or not; H1a holds when every part given holds, "complete" when all four are):
      bootstrap   bootstrap_rule "share" (default): Kendall's tau between independent bootstrap
                  resamples (rank_stability_bootstrap) is >= threshold in at least share_min (95%)
                  of the resample pairs, undefined taus counting as below. "τ >= 0.8 between
                  bootstrap resamples" is read as a statement about resample pairs in general, at
                  the 95% level used everywhere else in §9. "median" (median tau >= threshold) is
                  the lenient alternative: H1a would then hold with almost half the pairs below
                  0.8. The share, the median and the 2.5th percentile are reported either way;
      split_half  the pre-registered split-half tau (split_half_tau or split_half_panel) is
                  >= threshold; the random-half bootstrap is reported beside it, not decided on;
      judge       agreement.judge_calibration's result: BOTH gates of judge_calibration.md (the
                  rubric lines and the answer score bands) pass agreement.judge_passes. A result
                  without the score-band gate is refused, so a judge can never pass on one gate;
      harness     every remaining task's reference passed all its runs (harness_determinism).
    An undefined tau (NaN) does not hold.
    """
    def ge(x) -> bool:
        return x is not None and math.isfinite(float(x)) and float(x) >= threshold

    if bootstrap_rule not in H1A_BOOT_RULES:
        raise ValueError(f"bootstrap_rule must be one of {H1A_BOOT_RULES}")
    tb = rank_bootstrap["tau_between_resamples"]
    if tb.get("n") and tb.get("threshold") != threshold:
        raise ValueError(f"the bootstrap summary counts tau >= {tb.get('threshold')}, not "
                         f">= {threshold}")
    n_ok, n_undef = int(tb.get("n", 0)), int(tb.get("n_undefined", 0))
    share = tb["share_ge_threshold"] * n_ok / (n_ok + n_undef) if n_ok else 0.0
    med = tb.get("median")
    holds = share >= share_min if bootstrap_rule == "share" else ge(med)
    parts = {"bootstrap": {"rule": (f"tau >= {threshold} in >= {share_min:.0%} of resample pairs"
                                    if bootstrap_rule == "share" else f"median tau >= {threshold}"),
                           "share_ge_threshold": share, "median": med, "p2_5": tb.get("p2_5"),
                           "n_pairs": n_ok + n_undef, "holds": bool(holds)},
             "split_half": {"statistic": "split-half tau", "value": split_half["tau"],
                            "holds": ge(split_half["tau"])}}
    if judge is not None:
        if not judge.get("complete"):
            raise ValueError("the judge part needs judge_calibration's result with both gates (rubric "
                             "lines and answer score bands, judge_calibration.md): pass judge_scores "
                             "and human_scores to judge_calibration")
        parts["judge"] = {"statistic": "kappa and its 95% lower bound, lines and score bands",
                          "value": {g: (judge[g]["kappa"], judge[g]["lo"])
                                    for g in ("gate_lines", "gate_bands")},
                          "holds": bool(judge["passes"])}
    if determinism is not None:
        ok = determinism["n_tasks"] > 0 and determinism["n_pass_all"] == determinism["n_tasks"]
        parts["harness"] = {"statistic": "share of tasks passing all runs",
                            "value": determinism["share_pass_all"], "holds": bool(ok)}
    return {"parts": parts, "holds": all(p["holds"] for p in parts.values()),
            "complete": len(parts) == 4, "threshold": threshold}


# ---------------------------------------------------------------- H1c separation (RQ1c) ----------

H1C_TESTS = ("mcnemar", "wilcoxon")


def _wilcoxon_rows_p(d) -> np.ndarray:
    """wilcoxon_signed_rank's two-sided p-value for every row of d (paired differences a - b).

    The normal approximation (the "asymptotic" rule of wilcoxon_signed_rank: tie-corrected variance,
    no continuity correction, zero differences dropped) is computed for all rows at once; rows that
    wilcoxon_signed_rank tests exactly (no tied |differences|, at most 50 non-zero) or by enumeration
    (ties, at most 13 non-zero) are passed to it. Every p-value therefore equals the scalar
    function's (to rounding), and a row of zeros gets p = 1.
    """
    d = np.round(np.asarray(d, dtype=float), 12)
    rows, n_items = d.shape
    nz = d != 0
    n = nz.sum(axis=1)
    absd = np.abs(d)
    ranks = _st.rankdata(absd, axis=1) - (n_items - n)[:, None]   # zeros take the lowest ranks
    r_plus = np.where(d > 0, ranks, 0.0).sum(axis=1)
    s = np.sort(np.where(nz, absd, np.inf), axis=1)               # tie groups of the non-zero |d|
    new = np.ones(s.shape, dtype=bool)
    new[:, 1:] = s[:, 1:] != s[:, :-1]
    gid = np.cumsum(new, axis=1) + (n_items + 1) * np.arange(rows)[:, None]
    t = np.bincount(gid[np.isfinite(s)], minlength=(n_items + 1) * rows).astype(float)
    row_of = np.arange(t.size) // (n_items + 1)
    tie = np.bincount(row_of, weights=t ** 3 - t, minlength=rows)
    tied = np.bincount(row_of, weights=(t > 1).astype(float), minlength=rows) > 0
    asym = (tied & (n > 13)) | (~tied & (n > 50))
    p = np.ones(rows)
    mn = n * (n + 1) / 4.0
    var = (n * (n + 1) * (2.0 * n + 1.0) - tie / 2.0) / 24.0
    z = (r_plus[asym] - mn[asym]) / np.sqrt(var[asym])
    p[asym] = np.minimum(1.0, 2.0 * _special.ndtr(-np.abs(z)))
    zeros = np.zeros(n_items)
    for r in np.flatnonzero(~asym & (n > 0)):
        p[r] = wilcoxon_signed_rank(d[r], zeros)["p_value"]
    return p


def _pair_pvalues(x: np.ndarray, test: str, codes: np.ndarray | None) -> np.ndarray:
    """p-values of `test` for every model pair (i < j, row-major order) on a models x items matrix
    of per-item outcomes; `codes` = one integer cluster code per item (cluster sign-flip tests), or
    None (items independent)."""
    i, j = np.triu_indices(x.shape[0], 1)
    if codes is None:
        if test == "mcnemar":                                       # vectorised exact McNemar
            maj = x.astype(np.int64)
            a_only = maj @ (1 - maj).T
            return np.atleast_1d(mcnemar_exact_p(a_only[i, j], a_only[j, i]))
        return _wilcoxon_rows_p(x[i] - x[j])
    cl = codes.tolist()
    f = mcnemar_clustered if test == "mcnemar" else wilcoxon_clustered
    return np.array([f(x[a], x[b], cl)["p_value"] for a, b in zip(i, j)], dtype=float)


def _tier_matrices(samples, test: str, name: str):
    """(models | None, items | None, outcome matrix, mean matrix, samples per model | None).

    samples: a models x items x samples array (the same n for every model; models in the caller's
    order), or model -> item -> samples, where every model may have its OWN number of samples n
    (the run list: n = 5, 3 and 1). Each model is reduced on its own samples: the mean matrix holds
    the per-item mean score (pass@1 = c/n for pass/fail samples), the outcome matrix the per-item
    outcome `test` works on: that mean for "wilcoxon", the majority pass (ties fail) for "mcnemar".
    """
    if test not in H1C_TESTS:
        raise ValueError(f"test must be one of {H1C_TESTS}, not {test!r}")
    if isinstance(samples, Mapping):
        models, items, arrs = _model_samples(samples)
        per_model = [arrs[m] for m in models]
        n_by_model = {m: int(arrs[m].shape[1]) for m in models}
        if len(models) < 2:
            raise ValueError(f"{name}: need at least 2 models")
    else:
        x = _sample_array(samples)
        models = items = n_by_model = None
        per_model = list(x)
    if test == "mcnemar":
        if not all(np.all(np.isin(a, (0.0, 1.0))) for a in per_model):
            raise ValueError(f"{name}: McNemar needs pass/fail samples (0/1); partial-credit scores "
                             "use test='wilcoxon' (eval plan §9: Wilcoxon signed-rank for scores)")
    elif any(np.any((a < 0.0) | (a > 1.0)) for a in per_model):
        raise ValueError(f"{name}: scores must lie in [0, 1]")
    means = np.stack([a.mean(axis=1) for a in per_model])
    if test == "mcnemar":
        out = np.stack([(2 * a.sum(axis=1) > a.shape[1]).astype(float) for a in per_model])
    else:
        out = means
    return models, items, out, means, n_by_model


def _tier_clusters(clusters, items: list[str] | None, n_items: int, name: str
                   ) -> tuple[np.ndarray | None, int]:
    """Integer cluster codes for a tier's items (None without clusters) and the number of clusters."""
    if clusters is None:
        return None, n_items
    if items is not None:
        if not isinstance(clusters, Mapping):
            raise ValueError(f"{name}: with model -> item samples, clusters must map item -> cluster_id")
        missing = [i for i in items if i not in clusters]
        if missing:
            raise ValueError(f"{name}: items without a cluster_id: {missing[:10]}")
        labels = [clusters[i] for i in items]
    else:
        if isinstance(clusters, Mapping):
            raise ValueError(f"{name}: with an array of samples, clusters must be a sequence in item "
                             "order")
        labels = list(clusters)
    codes, g = cluster_codes(labels, n_items)
    return codes, g


def _separation_stats(out: np.ndarray, means: np.ndarray, test: str, codes: np.ndarray | None,
                      alpha: float, holm_correct: bool) -> tuple[float, float, int, int]:
    """(SD, range, significant pairs, pairs) of one tier on the given item columns."""
    m = means.mean(axis=1)
    p = _pair_pvalues(out, test, codes)
    p_used = holm_adjust(p) if holm_correct else p
    return float(m.std(ddof=1)), float(m.max() - m.min()), int(np.sum(p_used <= alpha)), int(p.size)


def separation(samples, test: str = H1C_TEST, clusters=None, alpha: float = ALPHA,
               holm_correct: bool = True) -> dict:
    """How strongly one tier separates the panel (RQ1c, eval plan Table 3: "Spread of scores; number
    of significantly different pairs").

    samples: models x items x samples array, or model -> item -> samples with each model's own n
    (_tier_matrices). Spread = SD (ddof 1) and range of the models' tier scores (mean over items of
    the per-item mean score: pass@1 on pass/fail tiers, mean credit on Tier K). Significant pairs =
    model pairs whose paired test rejects at alpha, Holm-corrected over all pairs (one family) by
    default. test: "wilcoxon" (default, H1C_TEST: signed-rank on per-item mean scores, any scores in
    [0, 1]) or "mcnemar" (pass/fail samples, majority-reduced items). clusters: item -> cluster_id
    with a mapping, a sequence in item order with an array; the exact cluster sign-flip forms are
    then used.
    """
    models, items, out, means, n_by_model = _tier_matrices(samples, test, "samples")
    codes, g = _tier_clusters(clusters, items, out.shape[1], "samples")
    sd, rng_, sig, n_pairs = _separation_stats(out, means, test, codes, alpha, holm_correct)
    return {"sd": sd, "range": rng_, "n_significant_pairs": sig, "n_pairs": n_pairs,
            "n_models": int(out.shape[0]), "n_items": int(out.shape[1]), "test": test,
            "clustered": codes is not None, "n_clusters": g, "holm": holm_correct,
            "models": models, "samples_per_model": n_by_model}


def _cluster_draw(codes: np.ndarray, g: int, rng: np.random.Generator
                  ) -> tuple[np.ndarray, np.ndarray]:
    """One cluster-bootstrap resample: g clusters drawn with replacement. Returns the item indices
    of the drawn clusters (in draw order) and a fresh cluster code per draw, so that a cluster drawn
    twice enters a cluster-aware test as two clusters."""
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes, minlength=g)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    draws = rng.integers(0, g, size=g)
    idx = np.concatenate([order[starts[c]:starts[c] + sizes[c]] for c in draws])
    return idx, np.repeat(np.arange(g), sizes[draws])


def _interval(boot: np.ndarray, estimate: float, alpha: float) -> dict:
    lo, hi = _percentile_ci(boot, alpha)
    return {"estimate": estimate, "lo": lo, "hi": hi,
            "p_one_sided": float((1 + np.sum(boot <= 0)) / (boot.size + 1))}


def size_matched_separation(k_samples, f_samples, k_test: str = H1C_TEST, f_test: str = H1C_TEST,
                            k_clusters=None, f_clusters=None, n_items: int | None = None,
                            repeats: int = H1C_REPEATS, B: int = H1C_BOOT_B, seed: int = SEED,
                            alpha: float = ALPHA, holm_correct: bool = True) -> dict:
    """H1c at equal size (RQ1c; eval plan Table 3: "Across the panel, Tier-F scores spread more
    widely, and more model pairs differ significantly, than on Tier K").

    Pre-registered (approved plan, WS7): Tier K is subsampled without replacement to Tier F's item
    count (n_items; default F's count, 150 in v1.0), `repeats` times, and the distribution of K's
    spread and significant pairs is reported next to Tier F's ("tier_k_subsampled", with the
    descriptive shares of subsamples in which K separates at least as much as F). Those shares are
    NOT a test: they treat Tier F's value as fixed and the subsamples of one item set vary less
    than independent item sets (PROPOSAL H1c-decision, stats.proposals).

    The inference is "difference" (PROPOSAL H1c-decision): a 95% percentile bootstrap interval of
    F minus size-matched K, for the SD and for the number of significant pairs, from B replicates
    (B = 0 skips it). Each replicate resamples Tier F's clusters (items without clusters) with
    replacement, draws a fresh size-matched subsample of Tier K and resamples ITS clusters the same
    way, and recomputes both statistics on both tiers, so both tiers' item-sampling noise enters and
    both are resampled at the same size. "estimate" is F's value minus the mean over the
    subsamples; "p_one_sided" = (1 + #{replicates <= 0}) / (B + 1). h1c_decision applies the rule.

    Tests (PROPOSAL H1c-test): BOTH tiers use the same test, by default H1C_TEST = "wilcoxon" on
    per-item mean scores (Tier K's mean credit, Tier F's pass@1), so the pairs count compares the
    tiers and not the power of two different tests. Each tier's own primary test (k_test="wilcoxon",
    f_test="mcnemar") is a sensitivity analysis only ("matched_tests" False). With clusters (item ->
    cluster_id for mappings, a sequence in item order for arrays) the cluster sign-flip forms are
    used.

    k_samples / f_samples: model -> item -> samples mappings, in which every model may have its own
    n (each model reduced on its own samples, _tier_matrices; models matched BY NAME), or models x
    items x samples arrays (the caller guarantees the model order). Every random draw comes from
    numpy's default_rng: the subsamples from `seed`, the bootstrap from (seed, 1).
    """
    mk, ik, k_out, k_means, k_ns = _tier_matrices(k_samples, k_test, "Tier K")
    mf, if_, f_out, f_means, f_ns = _tier_matrices(f_samples, f_test, "Tier F")
    if (mk is None) != (mf is None):
        raise ValueError("pass both tiers as model -> item mappings or both as arrays")
    if mk is not None and mk != mf:
        raise ValueError(f"Tier K and Tier F must have the same models: {mk} vs {mf}")
    if k_out.shape[0] != f_out.shape[0]:
        raise ValueError("Tier K and Tier F must have the same models")
    if repeats < 1 or B < 0:
        raise ValueError("need repeats >= 1 and B >= 0")
    kc, gk = _tier_clusters(k_clusters, ik, k_out.shape[1], "Tier K")
    fc, gf = _tier_clusters(f_clusters, if_, f_out.shape[1], "Tier F")
    n_k, n_f = k_out.shape[1], f_out.shape[1]
    n_sub = n_f if n_items is None else int(n_items)
    if not 1 <= n_sub <= n_k:
        raise ValueError(f"cannot subsample {n_sub} of {n_k} Tier-K items")

    def stats(out, means, test, codes, idx, new_codes=None):
        c = None if codes is None else (codes[idx] if new_codes is None else new_codes)
        return _separation_stats(out[:, idx], means[:, idx], test, c, alpha, holm_correct)

    f_sep = separation(f_samples, f_test, f_clusters, alpha, holm_correct)
    k_full = separation(k_samples, k_test, k_clusters, alpha, holm_correct)
    rng = np.random.default_rng(seed)
    sub = np.array([stats(k_out, k_means, k_test, kc,
                          np.sort(rng.choice(n_k, size=n_sub, replace=False)))
                    for _ in range(repeats)], dtype=float)
    sd_k, rng_k, sig_k = sub[:, 0], sub[:, 1], sub[:, 2]
    out = {"n_items": n_sub, "repeats": repeats, "B": B, "seed": seed, "alpha": alpha,
           "models": mk, "k_test": k_test, "f_test": f_test, "matched_tests": k_test == f_test,
           "tier_f": f_sep, "tier_k_full": k_full,
           "tier_k_subsampled": {"sd": summarize(sd_k), "range": summarize(rng_k),
                                 "n_significant_pairs": summarize(sig_k)},
           "share_k_sd_ge_f": float(np.mean(sd_k >= f_sep["sd"])),
           "share_k_sig_pairs_ge_f": float(np.mean(sig_k >= f_sep["n_significant_pairs"])),
           "difference": None}
    if B == 0:
        return out
    f_codes = np.arange(n_f) if fc is None else fc
    k_codes = np.arange(n_k) if kc is None else kc
    rb = np.random.default_rng([seed, 1])
    boot = np.empty((B, 2))
    for b in range(B):
        fi, fnew = _cluster_draw(f_codes, gf, rb)
        sf = stats(f_out, f_means, f_test, fc, fi, fnew)
        si = rb.choice(n_k, size=n_sub, replace=False)
        uniq, inv = np.unique(k_codes[si], return_inverse=True)
        ki, knew = _cluster_draw(inv.astype(np.int64), uniq.size, rb)
        sk = stats(k_out, k_means, k_test, kc, si[ki], knew)
        boot[b] = sf[0] - sk[0], sf[2] - sk[2]
    out["difference"] = {
        "sd": _interval(boot[:, 0], f_sep["sd"] - float(sd_k.mean()), alpha),
        "n_significant_pairs": _interval(boot[:, 1], f_sep["n_significant_pairs"]
                                         - float(sig_k.mean()), alpha),
        "B": B, "alpha": alpha, "seed": [seed, 1],
        "method": "percentile bootstrap of F minus size-matched K; clusters (items) of both tiers "
                  "resampled, a fresh K subsample per replicate"}
    return out


def h1c_decision(size_matched: Mapping) -> dict:
    """PROPOSAL H1c-decision (G2 pre-registration): the decision rule of H1c on
    size_matched_separation's output (eval plan Table 3: "Across the panel, Tier-F scores spread
    more widely, and more model pairs differ significantly, than on Tier K").

    spread  holds if the lower end of the (1 - alpha) bootstrap interval of SD(F) - SD(K at F's
            size) is above 0;
    pairs   holds if the lower end of the interval of sig_pairs(F) - sig_pairs(K at F's size) is
            above 0 (Tier F has strictly more significant pairs).
    H1c holds when both parts hold (an intersection-union rule, so its false-positive rate is at
    most that of either part, alpha / 2 one-sided by construction). "confirmatory" is False when
    the tiers were tested with different tests (sensitivity only) or the subsample distribution was
    run without the bootstrap (B = 0 raises). The descriptive subsample shares are reported beside
    the decision, never decided on.
    """
    diff = size_matched.get("difference")
    if diff is None:
        raise ValueError("h1c_decision needs the bootstrap interval: run size_matched_separation "
                         "with B >= 1 (H1C_BOOT_B)")
    parts = {}
    for part, key, share in (("spread", "sd", "share_k_sd_ge_f"),
                             ("pairs", "n_significant_pairs", "share_k_sig_pairs_ge_f")):
        d = diff[key]
        parts[part] = {"estimate": d["estimate"], "lo": d["lo"], "hi": d["hi"],
                       "p_one_sided": d["p_one_sided"], "holds": bool(d["lo"] > 0),
                       "descriptive_share_k_ge_f": size_matched[share]}
    matched = bool(size_matched["k_test"] == size_matched["f_test"])
    return {**parts, "holds": bool(parts["spread"]["holds"] and parts["pairs"]["holds"]),
            "confirmatory": matched, "alpha": diff["alpha"], "B": diff["B"],
            "tests": (size_matched["k_test"], size_matched["f_test"]),
            "note": "" if matched else "different tests on the two tiers: sensitivity analysis "
                                       "only (PROPOSAL H1c-test)"}


# ---------------------------------------------------------------- seeds and validity -------------

def seed_spread(scores_by_seed: Mapping[str, float]) -> dict:
    """Spread of one configuration's score between training seeds (eval plan §9 "Seeds": 3 seeds of
    each model's final setting, reported separately from the item-level interval, as a limitation).
    """
    seeds = sorted(scores_by_seed)
    if len(seeds) < 2:
        raise ValueError("need at least 2 seeds")
    v = _vector([scores_by_seed[s] for s in seeds], "scores")
    return {"n_seeds": len(seeds), "seeds": seeds, "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()),
            "range": float(v.max() - v.min())}


def correlation_ci(x: Sequence[float], y: Sequence[float], method: str = "pearson",
                   alpha: float = ALPHA) -> dict:
    """Correlation of two score vectors across the panel with a (1 - alpha) Fisher-z interval.

    Eval plan §9 "Validity": described, not tested, so no p-value is returned.
      pearson   scipy.stats.pearsonr and its Fisher-z confidence interval;
      spearman  rho of scipy.stats.spearmanr, Fisher-z interval with the Fieller, Hartley and
                Pearson (1957) standard error sqrt(1.06 / (n - 3)).
    Needs at least 4 models and non-constant scores.
    """
    a, b = _vector(x, "x"), _vector(y, "y")
    if a.size != b.size:
        raise ValueError("x and y need one score per model each")
    n = int(a.size)
    if n < 4:
        raise ValueError("need at least 4 models")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        raise ValueError("a constant score vector has no correlation")
    if method == "pearson":
        res = _st.pearsonr(a, b)
        r = float(res.statistic)
        ci = res.confidence_interval(confidence_level=1.0 - alpha)
        lo, hi = float(ci.low), float(ci.high)
    elif method == "spearman":
        r = float(_st.spearmanr(a, b).statistic)
        if abs(r) >= 1.0:
            lo = hi = r
        else:
            z, se = math.atanh(r), math.sqrt(1.06 / (n - 3))
            q = float(_st.norm.ppf(1.0 - alpha / 2.0))
            lo, hi = math.tanh(z - q * se), math.tanh(z + q * se)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    return {"r": r, "lo": lo, "hi": hi, "n": n, "method": method, "alpha": alpha}

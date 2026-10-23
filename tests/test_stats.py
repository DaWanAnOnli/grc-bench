"""Tests for stats/ (eval plan §9, §8.1, Table 8): known values, scipy, textbook examples."""

from __future__ import annotations

import inspect
import math
import sys
from itertools import combinations, product

import numpy as np
import pytest
from scipy import stats as st

from grcbench.stats import agreement as ag
from grcbench.stats import cli
from grcbench.stats import functions as fn
from grcbench.stats import items as it
from grcbench.stats import mde as md


# ---------------------------------------------------------------- estimates ----------------------

@pytest.mark.parametrize("n", range(1, 9))
def test_pass_at_k_matches_brute_force_over_k_subsets(n):
    for c in range(n + 1):
        runs = [1] * c + [0] * (n - c)
        for k in range(1, n + 1):
            subsets = list(combinations(runs, k))
            brute = sum(any(s) for s in subsets) / len(subsets)
            assert fn.pass_at_k(n, c, k) == pytest.approx(brute, abs=1e-12)
            all_pass = sum(all(s) for s in subsets) / len(subsets)
            assert fn.pass_hat_k(n, c, k) == pytest.approx(all_pass, abs=1e-12)


def test_pass_at_k_known_values_and_checks():
    assert fn.pass_at_k(5, 2, 1) == 0.4
    assert fn.pass_at_k(5, 2, 2) == pytest.approx(0.7)
    assert fn.pass_at_k(5, 2, 4) == 1.0
    assert fn.pass_at_k(5, 0, 3) == 0.0
    for bad in [(0, 0, 1), (5, 6, 1), (5, 2, 0), (5, 2, 6), (5, -1, 1)]:
        with pytest.raises(ValueError):
            fn.pass_at_k(*bad)


def test_pass_hat_k_with_n_equal_k_is_share_solved_in_all_runs():
    runs = {"t1": [1, 1, 1, 1], "t2": [1, 1, 1, 0], "t3": [0, 0, 0, 0], "t4": [True] * 4}
    assert fn.mean_pass_hat_k(runs, 4) == 0.5
    assert fn.pass_hat_k(5, 4, 4) == pytest.approx(0.2)        # C(4,4)/C(5,4)
    assert fn.mean_pass_at_k(runs, 1) == pytest.approx((1 + 0.75 + 0 + 1) / 4)
    assert fn.item_pass_at_1({"b": [1, 0], "a": [1, 1, 1, 0, 0]}) == {"a": 0.6, "b": 0.5}
    with pytest.raises(ValueError):
        fn.mean_pass_at_k({"x": [1, 2]})                      # not pass/fail
    with pytest.raises(ValueError):
        fn.mean_pass_at_k({"x": []})


def test_exec_and_func_at_1_use_the_preregistered_denominators():
    highest = {"a": ["V5", "V4", "V2", None, "V5"], "b": ["V4"] * 5, "c": ["V3"] * 5}
    e = fn.exec_at_1(highest)
    assert e["value"] == pytest.approx((0.6 + 1.0 + 0.0) / 3) and e["n_tasks"] == 3
    f = fn.func_at_1(highest, ["a"])
    assert f["value"] == pytest.approx(0.4) and f["n_tasks"] == 1 and f["excluded_no_oracle"] == 2
    with pytest.raises(ValueError):
        fn.func_at_1(highest, ["a", "zz"])                    # oracle task with no samples
    with pytest.raises(ValueError):
        fn.func_at_1(highest, [])
    funnel = fn.funnel(highest, ["a"])
    vals = [funnel["rates"][r] for r in fn.RUNGS[:-1]]
    assert vals == sorted(vals, reverse=True) and funnel["rates"]["V4"] == e["value"]
    assert fn.reached("V5", "V4") and not fn.reached("V3", "V4") and not fn.reached(None, "V1")
    with pytest.raises(ValueError):
        fn.reached("V7", "V4")


def test_funnel_v6_uses_the_oracle_denominator_like_func_at_1():
    # reviewer probe P4: one oracle task at V5 x5, one task without an oracle at V4 x5
    highest = {"oracle_task": ["V5"] * 5, "no_oracle_task": ["V4"] * 5}
    f = fn.func_at_1(highest, {"oracle_task"})
    fu = fn.funnel(highest, {"oracle_task"})
    assert f["value"] == 1.0 and fu["rates"]["V5"] == f["value"]          # not 0.5
    assert fu["n_tasks"] == {"V1": 2, "V2": 2, "V3": 2, "V4": 2, "V5": 1}
    assert fu["excluded_from_V6"] == 1 and fu["rates"]["V4"] == 1.0
    assert math.isnan(fn.funnel(highest, [])["rates"]["V5"])
    with pytest.raises(ValueError):
        fn.funnel(highest, ["missing_task"])


# ---------------------------------------------------------------- clustered uncertainty ----------

def test_cluster_robust_se_reduces_to_known_forms():
    rng = np.random.default_rng(0)
    y = rng.normal(size=101)
    assert fn.cluster_robust_se(y) == pytest.approx(st.sem(y), rel=1e-12)
    ci = fn.clustered_mean_ci(y)
    lo, hi = st.t.interval(0.95, len(y) - 1, loc=y.mean(), scale=st.sem(y))
    assert (ci["lo"], ci["hi"]) == pytest.approx((lo, hi), rel=1e-12)
    means = rng.normal(size=30)
    y2 = np.repeat(means, 4)
    cl = np.repeat([f"q{g}" for g in range(30)], 4)
    assert fn.cluster_robust_se(y2, cl) == pytest.approx(st.sem(means), rel=1e-12)
    ratio = fn.cluster_robust_se(y2, cl, "CR1") / fn.cluster_robust_se(y2, cl, "CR0")
    assert ratio == pytest.approx(math.sqrt(30 / 29))
    assert math.isnan(fn.cluster_robust_se(y2, ["one"] * len(y2)))


def test_cluster_robust_se_is_order_invariant_and_larger_with_correlated_clusters():
    rng = np.random.default_rng(1)
    g = rng.normal(size=50)
    y = np.repeat(g, 6) + rng.normal(scale=0.3, size=300)
    cl = np.repeat(np.arange(50), 6)
    perm = rng.permutation(300)
    assert fn.cluster_robust_se(y[perm], cl[perm]) == pytest.approx(fn.cluster_robust_se(y, cl))
    assert fn.cluster_robust_se(y, cl) > 1.8 * fn.cluster_robust_se(y)   # ignoring clusters underestimates


def test_cluster_bootstrap_is_deterministic_order_invariant_and_matches_cr0():
    rng = np.random.default_rng(2)
    g = rng.normal(size=80)
    y = np.repeat(g, 3) + rng.normal(scale=0.5, size=240)
    cl = [f"c{i:03d}" for i in np.repeat(np.arange(80), 3)]
    a = fn.cluster_bootstrap_ci(y, cl, B=5000, seed=11)
    assert a == fn.cluster_bootstrap_ci(y, cl, B=5000, seed=11)
    assert a != fn.cluster_bootstrap_ci(y, cl, B=5000, seed=12)
    perm = rng.permutation(240)
    b = fn.cluster_bootstrap_ci(y[perm], [cl[i] for i in perm], B=5000, seed=11)
    assert (b["lo"], b["hi"]) == pytest.approx((a["lo"], a["hi"]), abs=1e-12)
    assert a["se_boot"] == pytest.approx(fn.cluster_robust_se(y, cl, "CR0"), rel=0.05)
    assert a["lo"] < a["mean"] < a["hi"] and a["n_clusters"] == 80
    w = fn.bootstrap_weights(240, cl, B=7, seed=1)
    assert w.shape == (7, 240) and np.all(w.sum(axis=1) == 240)   # equal clusters: N items per draw


def test_paired_versions_are_the_one_sample_versions_on_differences():
    rng = np.random.default_rng(3)
    a, b = rng.random(60), rng.random(60)
    cl = np.repeat(np.arange(20), 3)
    assert fn.paired_cluster_bootstrap_ci(a, b, cl, B=800) == fn.cluster_bootstrap_ci(a - b, cl, B=800)
    assert fn.paired_cluster_robust_se(a, b, cl) == fn.cluster_robust_se(a - b, cl)
    d = fn.paired_difference(a, b, cl, B=800)
    assert d["diff"] == pytest.approx(np.mean(a - b))
    assert d["t_lo"] < d["diff"] < d["t_hi"] and d["boot_lo"] < d["diff"] < d["boot_hi"]
    with pytest.raises(ValueError):
        fn.paired_difference(a, b[:-1])


# ---------------------------------------------------------------- paired tests -------------------

def test_mcnemar_exact_equals_binomtest_and_hand_value():
    assert fn.mcnemar_exact_p(1, 6) == pytest.approx(0.125)        # 2 * (1 + 7) / 2^7
    assert fn.mcnemar_exact_p(0, 0) == 1.0
    for x in range(25):
        for y in range(25):
            if x + y:
                assert fn.mcnemar_exact_p(x, y) == pytest.approx(st.binomtest(x, x + y).pvalue, abs=1e-12)
    arr = fn.mcnemar_exact_p(np.array([1, 3]), np.array([6, 3]))
    assert arr.tolist() == pytest.approx([0.125, 1.0])


def test_mcnemar_majority_reduction_ties_fail():
    assert fn.majority_pass([1, 1, 1, 0, 0]) and not fn.majority_pass([1, 1, 0, 0, 0])
    assert not fn.majority_pass([1, 1, 0, 0]) and fn.majority_pass([1, 1, 1, 0])
    a = {"i1": [1, 1, 1, 0, 0], "i2": [1, 1, 0, 0], "i3": [1] * 5, "i4": [0] * 5, "only_a": [1]}
    b = {"i1": [0] * 5, "i2": [1, 1, 1, 1], "i3": [1] * 5, "i4": [1, 1, 1, 0, 0]}
    r = fn.mcnemar_from_samples(a, b)
    assert (r["a_only"], r["b_only"], r["both"], r["neither"]) == (1, 2, 1, 0)
    assert r["n_items"] == 4 and r["n_unshared"] == 1
    assert r["p_value"] == pytest.approx(st.binomtest(1, 3).pvalue)


def test_signflip_null_distribution_is_exact():
    rng = np.random.default_rng(20)
    for _ in range(200):
        c = rng.integers(-6, 7, int(rng.integers(1, 10)))
        brute = np.mean([abs(np.dot(s, np.abs(c))) >= abs(c.sum())
                         for s in product((-1, 1), repeat=c.size)])
        assert fn._signflip_p(c) == pytest.approx(brute, abs=1e-12)
    assert fn._signflip_p([3, 1, 1]) == pytest.approx(0.25) and fn._signflip_p([0, 0]) == 1.0


def test_clustered_tests_reduce_to_the_exact_tests_on_singletons():
    for x in range(20):
        for y in range(20):
            a, b = [1] * x + [0] * y + [1, 0], [0] * x + [1] * y + [1, 0]
            assert fn.mcnemar_clustered(a, b)["p_value"] == pytest.approx(fn.mcnemar_exact_p(x, y),
                                                                          abs=1e-12)
    rng = np.random.default_rng(21)
    for n in (5, 20, 45):
        u, v = rng.normal(size=n), rng.normal(size=n)
        assert fn.wilcoxon_clustered(u, v)["p_value"] == pytest.approx(
            st.wilcoxon(u, v, method="exact").pvalue, abs=1e-12)
    for _ in range(30):                                            # ties on the pass@1 grid
        a, b = rng.integers(0, 6, 12) / 5, rng.integers(0, 6, 12) / 5
        nz = np.round(a - b, 12)
        nz = nz[nz != 0]
        if nz.size < 2:
            continue
        ref = st.wilcoxon(nz, method=st.PermutationMethod(n_resamples=9999)).pvalue
        assert fn.wilcoxon_clustered(a, b)["p_value"] == pytest.approx(ref, abs=1e-12)


def test_clustered_mcnemar_hand_example_and_durkalski_statistic():
    a = [1, 1, 1, 1, 1, 0]
    b = [0, 0, 0, 0, 0, 0]
    cl = ["c1", "c1", "c1", "c2", "c3", "c3"]
    r = fn.mcnemar_clustered(a, b, cl)
    assert r["p_value"] == pytest.approx(2 / 8)                     # D_g = 3, 1, 1
    assert fn.mcnemar_exact(a, b)["p_value"] == pytest.approx(2 / 32)
    assert r["durkalski_chi2"] == pytest.approx(25 / 11) and r["statistic"] == 5
    w = fn.wilcoxon_clustered([0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.3, 0.4], ["q"] * 4)
    assert w["p_value"] == 1.0 and w["rank_biserial"] == 1.0         # one cluster: no evidence
    with pytest.raises(ValueError, match="cluster label"):
        fn.mcnemar_clustered(a, b, ["c1", None, "c1", "c2", "c3", "c3"])


def test_clustered_tests_hold_their_size_when_items_are_clustered():
    # reviewer probe meth_probe2: models equal in expectation, a cluster-level model x family tilt
    G, m, sims = 30, 5, 300
    cl = np.repeat(np.arange(G), m)
    rng = np.random.default_rng(321)
    rej = {"mcnemar": 0, "mcnemar_clustered": 0, "wilcoxon": 0, "wilcoxon_clustered": 0}
    for _ in range(sims):
        tilt = np.repeat(rng.normal(0, 0.8, size=G), m)
        eta = np.repeat(rng.normal(0, 1.0, size=G), m) + rng.normal(0, 0.5, size=G * m)
        A = (rng.random((G * m, 5)) < (1 / (1 + np.exp(-(eta + tilt / 2))))[:, None]).astype(int)
        B = (rng.random((G * m, 5)) < (1 / (1 + np.exp(-(eta - tilt / 2))))[:, None]).astype(int)
        am, bm = 2 * A.sum(1) > 5, 2 * B.sum(1) > 5
        rej["mcnemar"] += fn.mcnemar_exact(am, bm)["p_value"] <= 0.05
        rej["mcnemar_clustered"] += fn.mcnemar_clustered(am, bm, cl)["p_value"] <= 0.05
        rej["wilcoxon"] += fn.wilcoxon_signed_rank(A.mean(1), B.mean(1))["p_value"] <= 0.05
        rej["wilcoxon_clustered"] += fn.wilcoxon_clustered(A.mean(1), B.mean(1), cl)["p_value"] <= 0.05
    rate = {k: v / sims for k, v in rej.items()}
    assert rate["mcnemar"] > 0.08 and rate["wilcoxon"] > 0.10        # the item-level tests over-reject
    assert rate["mcnemar_clustered"] <= 0.075 and rate["wilcoxon_clustered"] <= 0.075


def test_wilcoxon_method_is_chosen_from_the_nonzero_differences():
    a = np.array([0.2] * 10 + [0.4, 0.6, 0.6, 0.8, 0.4, 0.2, 1.0, 0.8, 0.6, 0.4])
    b = np.array([0.2] * 10 + [0.2, 0.4, 0.8, 0.6, 0.2, 0.4, 0.8, 0.6, 0.4, 0.2])
    r = fn.wilcoxon_signed_rank(a, b)                               # 20 items, 10 non-zero, tied
    assert r["n_nonzero"] == 10 and r["method"] == "permutation"
    nz = np.round(a - b, 12)
    nz = nz[nz != 0]
    assert r["p_value"] == pytest.approx(
        st.wilcoxon(nz, method=st.PermutationMethod(n_resamples=9999)).pvalue, abs=1e-12)
    assert r["p_value"] == pytest.approx(fn.wilcoxon_clustered(a, b)["p_value"], abs=1e-12)
    sr = fn.wilcoxon_signed_rank([1, 2, 3], [0, 0, 4])              # d = 1, 2, -1: ranks 1.5, 3, 1.5
    assert (sr["t_plus"], sr["t_minus"]) == (4.5, 1.5) and sr["rank_biserial"] == pytest.approx(0.5)


def test_wilcoxon_matches_scipy_auto_and_treats_grid_scores_as_ties():
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=20), rng.normal(size=20)
    r = fn.wilcoxon_signed_rank(x, y)
    assert r["method"] == "exact" and r["p_value"] == pytest.approx(st.wilcoxon(x, y).pvalue, abs=1e-12)
    assert fn.wilcoxon_signed_rank(np.arange(1, 11), np.zeros(10))["p_value"] == pytest.approx(2 / 1024)
    for n, method in [(10, "permutation"), (60, "asymptotic")]:
        a = rng.integers(0, 6, n) / 5                          # pass@1 on the grid of n = 5 samples
        b = rng.integers(0, 6, n) / 5
        r = fn.wilcoxon_signed_rank(a, b)
        ia, ib = np.rint(a * 5).astype(int), np.rint(b * 5).astype(int)
        ref = st.wilcoxon(ia - ib, zero_method="wilcox").pvalue  # integers: exact ties, scipy "auto"
        assert r["method"] == method and r["p_value"] == pytest.approx(ref, abs=1e-12)
    tied = fn.wilcoxon_signed_rank([0.6, 0.4, 1.0, 0.8], [0.4, 0.2, 0.8, 0.6])  # all diffs 0.2
    assert tied["method"] == "permutation" and tied["p_value"] == pytest.approx(2 / 16)
    zero = fn.wilcoxon_signed_rank([0.2, 0.4], [0.2, 0.4])
    assert zero["p_value"] == 1.0 and zero["n_nonzero"] == 0


def _holm_reference(p):
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def test_holm_bonferroni_textbook_and_reference_and_families():
    res = fn.holm_bonferroni({"RQ1a": {"h1": 0.01, "h2": 0.04, "h3": 0.03, "h4": 0.005},
                              "RQ2": {"x": 0.04}})
    assert [res["RQ1a"][h]["p_holm"] for h in ("h1", "h2", "h3", "h4")] == pytest.approx(
        [0.03, 0.06, 0.06, 0.02])
    assert [res["RQ1a"][h]["reject"] for h in ("h1", "h2", "h3", "h4")] == [True, False, False, True]
    assert res["RQ2"]["x"] == {"p": 0.04, "p_holm": 0.04, "reject": True}   # its own family
    rng = np.random.default_rng(5)
    for _ in range(50):
        p = rng.random(rng.integers(1, 12)) ** 3
        assert fn.holm_adjust(p).tolist() == pytest.approx(_holm_reference(p.tolist()))
    # step-down rule: reject while p_(i) <= alpha / (m - i + 1)
    p = [0.001, 0.0125, 0.02, 0.3]
    rej = [v["reject"] for v in fn.holm(dict(zip("abcd", p))).values()]
    assert rej == [True, True, True, False]
    with pytest.raises(ValueError):
        fn.holm_adjust([0.5, 1.2])


def test_effect_sizes():
    assert fn.cohens_h(0.6, 0.4) == pytest.approx(0.40272, abs=1e-5)
    assert fn.cohens_h(1, 0) == pytest.approx(math.pi) and fn.cohens_h(0.3, 0.3) == 0.0
    r = fn.matched_odds_ratio(6, 2)
    assert r["odds_ratio"] == 3.0
    ci = st.binomtest(6, 8).proportion_ci(method="exact")
    assert (r["lo"], r["hi"]) == pytest.approx((ci.low / (1 - ci.low), ci.high / (1 - ci.high)))
    assert fn.matched_odds_ratio(4, 0)["odds_ratio"] == math.inf
    assert fn.matched_odds_ratio(4, 0)["hi"] == math.inf
    assert math.isnan(fn.matched_odds_ratio(0, 0)["odds_ratio"])
    # the exact interval excludes 1 exactly when the exact McNemar test rejects at 5%
    for x in range(0, 20):
        for y in range(0, 20):
            if x + y:
                o = fn.matched_odds_ratio(x, y)
                assert (o["lo"] > 1 or o["hi"] < 1) == (fn.mcnemar_exact_p(x, y) < 0.05)


def test_compare_paired_bundles_every_statistic():
    rng = np.random.default_rng(6)
    items = [f"k{i:03d}" for i in range(120)]
    a = {i: (rng.random(5) < 0.7).tolist() for i in items}
    b = {i: (rng.random(5) < 0.5).tolist() for i in items}
    clusters = {i: f"fam{n // 4}" for n, i in enumerate(items)}
    r = fn.compare_paired(a, b, clusters, B=1000)
    pr, se = r["primary"], r["secondary"]
    assert r["n_items"] == 120 and r["n_clusters"] == 30 and r["clustered"]
    assert pr["test"] == "mcnemar_clustered" and se["test"] == "wilcoxon_clustered"
    assert pr["difference"]["n_clusters"] == 30
    pa = np.mean([np.mean(a[i]) for i in items])
    assert r["mean_a"] == pytest.approx(pa)
    assert se["difference"]["diff"] == pytest.approx(r["mean_a"] - r["mean_b"])   # pass@1 estimand
    ma = np.array([fn.majority_pass(a[i]) for i in items], float)
    mb = np.array([fn.majority_pass(b[i]) for i in items], float)
    assert pr["difference"]["diff"] == pytest.approx(ma.mean() - mb.mean())       # McNemar estimand
    assert pr["cohens_h"] == pytest.approx(fn.cohens_h(ma.mean(), mb.mean()))
    det = pr["detail"]
    assert pr["odds_ratio"]["odds_ratio"] == pytest.approx(det["a_only"] / det["b_only"])
    assert r["p_primary"] == pr["p_value"] == fn.mcnemar_clustered(ma, mb, [clusters[i] for i in items])["p_value"]
    assert r["p_primary"] < 0.05 and se["p_value"] < 0.05
    assert r["mde"] == {"pd_0.2": md.mde(120, 0.2), "pd_0.3": md.mde(120, 0.3)}   # a priori, Table 8
    assert r["mde_at_observed_pd"] == pytest.approx(md.mde(120, det["discordant_share"]))
    un = fn.compare_paired(a, b, B=200)
    assert un["primary"]["test"] == "mcnemar_exact" and un["secondary"]["test"] == "wilcoxon_signed_rank"
    assert un["p_primary"] == fn.mcnemar_exact_p(det["a_only"], det["b_only"])


def test_compare_paired_effects_follow_their_own_test_when_estimands_disagree():
    # reviewer probe P1: majority-pass (McNemar) favours B, mean pass@1 favours A
    a = {f"x{i:03d}": [1] * 5 for i in range(100)} | {f"y{i:03d}": [1, 1, 0, 0, 0] for i in range(50)}
    b = {f"x{i:03d}": [1, 1, 1, 0, 0] for i in range(100)} | {f"y{i:03d}": [1, 1, 1, 0, 0] for i in range(50)}
    r = fn.compare_paired(a, b, B=2000)
    pr, se = r["primary"], r["secondary"]
    assert (pr["detail"]["a_only"], pr["detail"]["b_only"]) == (0, 50)
    # every primary effect points the same way as the primary test (towards B) ...
    assert pr["difference"]["diff"] == pytest.approx(-1 / 3) and pr["difference"]["boot_hi"] < 0
    assert pr["cohens_h"] < 0 and pr["odds_ratio"]["odds_ratio"] == 0.0 and pr["odds_ratio"]["hi"] < 1
    # ... and the pass@1 comparison, which points to A, is labelled secondary and kept out of Holm
    assert se["difference"]["diff"] == pytest.approx(0.2) and se["rank_biserial"] > 0
    assert "secondary" in se["note"]
    assert r["p_primary"] == pytest.approx(fn.mcnemar_exact_p(0, 50))
    fam = fn.holm_bonferroni({"RQ2": {"a_vs_b": r, "other": 0.5}})["RQ2"]
    assert fam["a_vs_b"]["p"] == r["p_primary"]
    with pytest.raises(ValueError):
        fn.holm({"bad": {"p_value": 0.01}})                          # results must carry p_primary


def test_compare_paired_on_tier_k_partial_credit_scores():
    rng = np.random.default_rng(16)
    items = [f"k{i:03d}" for i in range(90)]
    a = {i: rng.choice([0.0, 0.5, 1.0], size=5, p=[0.2, 0.3, 0.5]).tolist() for i in items}
    b = {i: rng.choice([0.0, 0.5, 1.0], size=5, p=[0.35, 0.3, 0.35]).tolist() for i in items}
    clusters = {i: f"twin{n // 3}" for n, i in enumerate(items)}
    with pytest.raises(ValueError, match="score"):
        fn.compare_paired(a, b)                                     # 0.5 is not pass/fail
    r = fn.compare_paired(a, b, outcome="score", B=500)
    sa = [np.mean(a[i]) for i in items]
    sb = [np.mean(b[i]) for i in items]
    assert r["primary"]["test"] == "wilcoxon_signed_rank" and r["secondary"] is None
    assert r["p_primary"] == fn.wilcoxon_signed_rank(sa, sb)["p_value"]
    assert r["primary"]["difference"]["diff"] == pytest.approx(np.mean(sa) - np.mean(sb))
    rc = fn.compare_paired(a, b, clusters, outcome="score", B=500)
    assert rc["primary"]["test"] == "wilcoxon_clustered" and rc["n_clusters"] == 30
    assert rc["p_primary"] == fn.wilcoxon_clustered(sa, sb, [clusters[i] for i in items])["p_value"]
    assert fn.item_scores(a, "score")[items[0]] == pytest.approx(np.mean(a[items[0]]))
    with pytest.raises(ValueError):
        fn.compare_paired(a, b, outcome="binary")
    with pytest.raises(ValueError, match="cluster_id"):
        fn.compare_paired(a, b, {items[0]: "twin0"}, outcome="score")


# ---------------------------------------------------------------- MDE / Table 8 ------------------

def test_mde_reproduces_eval_plan_table8():
    got = {(n, pd): round(100 * md.mde(n, pd)) for n in (285, 150, 30) for pd in (0.2, 0.3)}
    assert got == {(285, 0.2): 7, (285, 0.3): 9, (150, 0.2): 10, (150, 0.3): 13,
                   (30, 0.2): 23, (30, 0.3): 28}
    rows = md.table8()
    assert [(r["tier"], r["points"]) for r in rows] == [("K", "7-9"), ("F", "10-13"), ("A", "23-28")]
    assert all(r["matches_plan"] for r in rows)
    z = st.norm.ppf(0.975) + st.norm.ppf(0.8)
    assert md.mde(285, 0.2) == pytest.approx(z * math.sqrt(0.2 / 285))


def test_n_for_mde_inverts_mde():
    for target, pd in [(0.10, 0.2), (0.05, 0.3), (0.25, 0.25)]:
        n = md.n_for_mde(target, pd)
        assert md.mde(n, pd) <= target < md.mde(n - 1, pd)
    with pytest.raises(ValueError):
        md.mde(0, 0.2)
    with pytest.raises(ValueError):
        md.mde(100, 0.0)


# ---------------------------------------------------------------- reliability --------------------

def _panel(seed=0, p=(0.1, 0.3, 0.5, 0.7, 0.9), n_items=285, n=5):
    rng = np.random.default_rng(seed)
    return (rng.random((len(p), n_items, n)) < np.asarray(p)[:, None, None]).astype(float)


def test_rank_stability_bootstrap():
    x = _panel()
    r = fn.rank_stability_bootstrap(x.mean(axis=2), B=300, seed=3)
    assert r["tau_between_resamples"]["median"] == 1.0
    assert r["tau_between_resamples"]["share_ge_threshold"] == 1.0
    assert r == fn.rank_stability_bootstrap(x.mean(axis=2), B=300, seed=3)
    noisy = _panel(seed=1, p=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5), n_items=40)
    rn = fn.rank_stability_bootstrap(noisy.mean(axis=2), B=300, seed=3)
    assert rn["tau_between_resamples"]["median"] < 0.8          # equal models: unstable ranking
    cl = np.repeat(np.arange(57), 5)
    rc = fn.rank_stability_bootstrap(x.mean(axis=2), clusters=cl, B=50)
    assert rc["n_clusters"] == 57
    assert fn.kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0 and fn.kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0


def test_split_half_uses_samples_12_vs_34_and_drops_5():
    rng = np.random.default_rng(7)
    x = (rng.random((6, 50, 5)) < 0.5).astype(float)
    r = fn.split_half_tau(x)
    assert r["half_1_samples"] == [1, 2] and r["half_2_samples"] == [3, 4] and r["dropped_samples"] == [5]
    a, b = x[:, :, :2].mean(axis=(1, 2)), x[:, :, 2:4].mean(axis=(1, 2))
    assert r["tau"] == pytest.approx(st.kendalltau(a, b).statistic, abs=1e-12)
    y = x.copy()
    y[:, :, 4] = 1 - y[:, :, 4]                                  # sample 5 must not matter
    assert fn.split_half_tau(y)["tau"] == r["tau"]
    assert fn.split_halves(5) == fn.SPLIT_HALVES and fn.split_halves(4) == ((0, 1), (2, 3))
    r3 = fn.split_half_tau(x[:, :, :3])                          # n = 3: {1} vs {2}, 3rd dropped
    assert (r3["half_1_samples"], r3["half_2_samples"], r3["dropped_samples"]) == ([1], [2], [3])
    with pytest.raises(ValueError, match="needs n >= 2"):
        fn.split_half_tau(x[:, :, :1])
    with pytest.raises(ValueError):
        fn.split_half_tau(x, halves=((0, 1), (1, 2)))             # overlapping halves


def _ragged_panel(seed=17, n_items=40):
    """The run list's sample counts: n = 5 (Table 6), n = 3 (seeds, arms, outside panel), n = 1."""
    rng = np.random.default_rng(seed)
    spec = {"cfg_a": (5, 0.3), "cfg_b": (5, 0.5), "small_1": (3, 0.4), "small_2": (3, 0.7),
            "teacher": (3, 0.8), "frontier": (1, 0.9)}
    return {m: {f"q{i:02d}": (rng.random(n) < p).astype(float).tolist() for i in range(n_items)}
            for m, (n, p) in spec.items()}


def test_split_half_panel_handles_mixed_sample_counts():
    res = _ragged_panel()
    with pytest.raises(ValueError, match="different numbers of samples per item"):
        fn.sample_tensor(res)                                      # clear message, not a numpy error
    s = fn.split_half_panel(res)
    assert s["excluded"] == {"frontier": 1} and s["n_models"] == 5
    assert s["halves_by_n"]["5"] == {"half_1_samples": [1, 2], "half_2_samples": [3, 4],
                                     "dropped_samples": [5]}
    assert s["halves_by_n"]["3"] == {"half_1_samples": [1], "half_2_samples": [2],
                                     "dropped_samples": [3]}
    # recompute by hand
    a = [np.mean([res[m][i][:len(res[m][i]) // 2] for i in res[m]]) for m in s["models"]]
    b = [np.mean([res[m][i][len(res[m][i]) // 2:2 * (len(res[m][i]) // 2)] for i in res[m]])
         for m in s["models"]]
    assert s["tau"] == pytest.approx(st.kendalltau(a, b).statistic, abs=1e-12)
    s4 = fn.split_half_panel(res, min_samples=4)                   # sensitivity: n >= 4 only
    assert s4["models"] == ["cfg_a", "cfg_b"] and set(s4["excluded"]) == {"small_1", "small_2",
                                                                          "teacher", "frontier"}
    rh = fn.random_half_panel(res, repeats=30, seed=3)
    assert rh == fn.random_half_panel(res, repeats=30, seed=3)
    assert rh["half_sizes"] == {"cfg_a": 2, "cfg_b": 2, "small_1": 1, "small_2": 1, "teacher": 1}
    assert rh["excluded"] == {"frontier": 1} and rh["tau"]["n"] == 30
    uniform = {m: res[m] for m in ("cfg_a", "cfg_b")}
    models, items, arr = fn.sample_tensor(uniform)
    assert fn.split_half_panel(uniform)["tau"] == fn.split_half_tau(arr)["tau"]
    bad = {**uniform, "odd": {i: ([1, 0, 1] if i == "q00" else [1, 0]) for i in res["cfg_a"]}}
    with pytest.raises(ValueError, match="model odd"):
        fn.split_half_panel(bad)
    with pytest.raises(ValueError, match="fewer than 2 configurations"):
        fn.split_half_panel({"cfg_a": res["cfg_a"], "frontier": res["frontier"]})


def test_random_half_bootstrap():
    x = np.repeat(_panel(n=1), 5, axis=2)
    r = fn.random_half_tau(x, repeats=40, seed=2)
    assert r["half_size"] == 2 and r["tau"]["min"] == 1.0
    noisy = _panel(seed=3, p=(0.5,) * 6, n_items=30)
    r1 = fn.random_half_tau(noisy, repeats=40, seed=2)
    assert r1 == fn.random_half_tau(noisy, repeats=40, seed=2) and r1["tau"]["mean"] < 0.8


def test_harness_determinism():
    r = fn.harness_determinism({"a": [True] * 10, "b": [1] * 9 + [0], "c": [0] * 10, "d": [1] * 5},
                               digests={"a": ["h"] * 10, "b": ["h"] * 9 + ["g"]})
    assert r["status"] == {"a": "pass_all", "b": "flaky", "c": "fail_all", "d": "too_few_runs"}
    assert r["share_pass_all"] == 0.25 and r["not_solvable"] == ["b", "c", "d"]
    assert r["varying_outputs"] == ["b"]


# ---------------------------------------------------------------- H1c ----------------------------

def test_separation_counts_pairs_with_exact_mcnemar_and_holm():
    # 2 models x 10 items x 1 sample: model 0 passes items 0-8, model 1 passes item 9 only.
    x = np.zeros((2, 10, 1))
    x[0, :9, 0] = 1
    x[1, 9, 0] = 1
    s = fn.separation(x, test="mcnemar")
    assert s["n_pairs"] == 1 and s["n_significant_pairs"] == 1           # p = binomtest(9, 10)
    assert st.binomtest(9, 10).pvalue < 0.05
    assert s["range"] == pytest.approx(0.8) and s["sd"] == pytest.approx(np.std([0.9, 0.1], ddof=1))
    p = _panel(seed=4)
    a = fn.separation(p, test="mcnemar", holm_correct=False)["n_significant_pairs"]
    b = fn.separation(p, test="mcnemar")["n_significant_pairs"]
    assert a >= b and a == 10
    assert fn.separation(p)["test"] == fn.H1C_TEST == "wilcoxon"     # PROPOSAL H1c-test: one test for both tiers
    with pytest.raises(ValueError, match="test='wilcoxon'"):
        fn.separation(np.full((2, 3, 2), 0.5), test="mcnemar")       # McNemar needs pass/fail
    with pytest.raises(ValueError):
        fn.separation(p, test="t")


def test_separation_with_partial_credit_scores_and_clusters():
    rng = np.random.default_rng(22)
    k = rng.choice([0.0, 0.5, 1.0], size=(4, 60, 5), p=[0.3, 0.3, 0.4])   # Tier-K-like credit
    k[0] = np.minimum(1.0, k[0] + 0.5)                              # model 0 clearly better
    s = fn.separation(k, test="wilcoxon")
    assert s["test"] == "wilcoxon" and not s["clustered"]
    item = k.mean(axis=2)
    ps = {(i, j): fn.wilcoxon_signed_rank(item[i], item[j])["p_value"] for i, j in combinations(range(4), 2)}
    assert s["n_significant_pairs"] == int(np.sum(fn.holm_adjust(list(ps.values())) <= 0.05))
    assert s["sd"] == pytest.approx(np.std(k.mean(axis=(1, 2)), ddof=1))
    cl = [f"f{i // 6}" for i in range(60)]
    sc = fn.separation(k, test="wilcoxon", clusters=cl)
    pc = [fn.wilcoxon_clustered(item[i], item[j], cl)["p_value"] for i, j in combinations(range(4), 2)]
    assert sc["clustered"] and sc["n_clusters"] == 10
    assert sc["n_significant_pairs"] == int(np.sum(fn.holm_adjust(pc) <= 0.05))
    x = (k >= 1.0).astype(float)
    pm = [fn.mcnemar_clustered((2 * x[i].sum(1) > 5), (2 * x[j].sum(1) > 5), cl)["p_value"]
          for i, j in combinations(range(4), 2)]
    assert fn.separation(x, clusters=cl)["n_significant_pairs"] == int(np.sum(fn.holm_adjust(pm) <= 0.05))


def test_size_matched_separation():
    rng = np.random.default_rng(8)
    k = (rng.random((5, 285, 5)) < np.linspace(0.45, 0.55, 5)[:, None, None]).astype(float)
    f = _panel(seed=9, n_items=150)
    r = fn.size_matched_separation(k, f, repeats=30, seed=1)
    assert r["n_items"] == 150 and r["repeats"] == 30
    assert r["share_k_sd_ge_f"] == 0.0 and r["tier_k_subsampled"]["sd"]["n"] == 30
    assert r == fn.size_matched_separation(k, f, repeats=30, seed=1)
    assert (r["k_test"], r["f_test"]) == ("wilcoxon", "wilcoxon")     # the same test on both tiers (M1)
    assert fn.h1c_decision(r)["holds"]
    with pytest.raises(ValueError):
        fn.size_matched_separation(k[:, :100], f)                 # K smaller than F
    with pytest.raises(ValueError):
        fn.size_matched_separation(k[:4], f)


def test_size_matched_separation_accepts_tier_k_scores_clusters_and_aligns_models_by_name():
    # reviewer probe P2: Tier-K rubric fractions next to Tier-F pass/fail
    rng = np.random.default_rng(0)
    k = np.round(rng.random((5, 285, 5)), 2)
    f = (rng.random((5, 150, 5)) < 0.5).astype(float)
    r = fn.size_matched_separation(k, f, repeats=5)
    assert r["tier_k_full"]["test"] == "wilcoxon" and r["tier_k_subsampled"]["sd"]["n"] == 5
    with pytest.raises(ValueError, match="pass/fail"):
        fn.size_matched_separation(f, k[:, :150], k_test="mcnemar", f_test="mcnemar", repeats=2)
    names = ["m_a", "m_b", "m_c", "m_d", "m_e"]
    kd = {m: {f"k{i:03d}": k[n, i].tolist() for i in range(285)} for n, m in enumerate(names)}
    fd = {m: {f"f{i:03d}": f[n, i].tolist() for i in range(150)} for n, m in enumerate(names)}
    kc = {f"k{i:03d}": f"twin{i // 5}" for i in range(285)}
    fc = {f"f{i:03d}": f"src{i // 10}" for i in range(150)}
    rd = fn.size_matched_separation(kd, dict(reversed(list(fd.items()))), k_clusters=kc,
                                    f_clusters=fc, repeats=5)
    assert rd["models"] == names and rd["tier_f"]["clustered"] and rd["tier_f"]["n_clusters"] == 15
    assert rd["tier_k_full"]["n_clusters"] == 57
    ra = fn.size_matched_separation(k, f, k_clusters=[kc[f"k{i:03d}"] for i in range(285)],
                                    f_clusters=[fc[f"f{i:03d}"] for i in range(150)], repeats=5)
    names_only = ("models", "samples_per_model")                 # reported for name-keyed input only
    strip = lambda d: {k: v for k, v in d.items() if k not in names_only}  # noqa: E731
    assert strip(ra["tier_f"]) == strip(rd["tier_f"]) and ra["tier_k_subsampled"] == rd["tier_k_subsampled"]
    assert rd["tier_f"]["samples_per_model"] == {m: 5 for m in names} and ra["tier_f"]["models"] is None
    with pytest.raises(ValueError, match="same models"):
        fn.size_matched_separation(kd, {**fd, "m_extra": fd["m_a"]}, repeats=2)
    with pytest.raises(ValueError, match="both"):
        fn.size_matched_separation(kd, f, repeats=2)
    with pytest.raises(ValueError, match="cluster_id"):
        fn.size_matched_separation(kd, fd, k_clusters={"k000": "x"}, repeats=2)


def test_h1a_and_h1c_decision_rules():
    x = _panel()
    rb = fn.rank_stability_bootstrap(x.mean(axis=2), B=100)
    sh = fn.split_half_tau(x)
    d = fn.h1a_decision(rb, sh)
    assert d["holds"] and not d["complete"] and set(d["parts"]) == {"bootstrap", "split_half"}
    det = fn.harness_determinism({"t1": [1] * 10, "t2": [1] * 9 + [0]})
    judge = {"complete": True, "passes": True, "gate_lines": {"kappa": 0.75, "lo": 0.62},
             "gate_bands": {"kappa": 0.8, "lo": 0.66}}
    full = fn.h1a_decision(rb, sh, judge, det)
    assert full["complete"] and not full["holds"] and not full["parts"]["harness"]["holds"]
    assert full["parts"]["judge"]["holds"]
    with pytest.raises(ValueError, match="both gates"):              # the line gate alone never passes
        fn.h1a_decision(rb, sh, {**judge, "complete": False, "gate_bands": None, "passes": None}, det)
    low = fn.h1a_decision({"tau_between_resamples": {"median": 0.79}}, {"tau": 0.9})
    assert not low["holds"] and low["parts"]["split_half"]["holds"]
    none = fn.h1a_decision({"tau_between_resamples": {"n": 0, "n_undefined": 5}}, {"tau": 0.9})
    assert not none["parts"]["bootstrap"]["holds"]
    sm = {"difference": {"sd": {"estimate": 0.02, "lo": 0.005, "hi": 0.04, "p_one_sided": 0.01},
                         "n_significant_pairs": {"estimate": 2, "lo": -1, "hi": 5, "p_one_sided": 0.2},
                         "alpha": 0.05, "B": 1000},
          "share_k_sd_ge_f": 0.05, "share_k_sig_pairs_ge_f": 0.051, "k_test": "wilcoxon", "f_test": "wilcoxon"}
    c = fn.h1c_decision(sm)                     # PROPOSAL H1c-decision: both interval lower ends above 0
    assert c["spread"]["holds"] and not c["pairs"]["holds"] and not c["holds"] and c["confirmatory"]
    assert c["spread"]["descriptive_share_k_ge_f"] == 0.05             # reported, never decided on
    assert not fn.h1c_decision({**sm, "f_test": "mcnemar"})["confirmatory"]
    with pytest.raises(ValueError, match="bootstrap interval"):
        fn.h1c_decision({k: v for k, v in sm.items() if k != "difference"})


# ---------------------------------------------------------------- seeds, validity ----------------

def test_seed_spread_and_correlation_ci():
    s = fn.seed_spread({"s3": 0.7, "s1": 0.5, "s2": 0.6})
    assert s["seeds"] == ["s1", "s2", "s3"] and s["sd"] == pytest.approx(0.1) and s["range"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        fn.seed_spread({"s1": 0.5})
    rng = np.random.default_rng(14)
    x = rng.random(21)
    y = x + rng.normal(scale=0.3, size=21)
    c = fn.correlation_ci(x, y)
    ref = st.pearsonr(x, y)
    ci = ref.confidence_interval()
    assert (c["r"], c["lo"], c["hi"]) == pytest.approx((ref.statistic, ci.low, ci.high), abs=1e-12)
    assert "p_value" not in c                                    # described, not tested (§9)
    sp = fn.correlation_ci(x, y, "spearman")
    rho = st.spearmanr(x, y).statistic
    half = st.norm.ppf(0.975) * math.sqrt(1.06 / 18)
    assert sp["r"] == pytest.approx(rho) and sp["lo"] == pytest.approx(math.tanh(math.atanh(rho) - half))
    assert fn.correlation_ci([1, 2, 3, 4], [2, 4, 6, 8], "spearman")["lo"] == 1.0
    with pytest.raises(ValueError):
        fn.correlation_ci([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        fn.correlation_ci([1, 1, 1, 1], [1, 2, 3, 4])


# ---------------------------------------------------------------- item analysis ------------------

def test_point_biserial_matches_scipy_and_flags():
    rng = np.random.default_rng(10)
    m = (rng.random((21, 60)) < np.linspace(0.15, 0.85, 21)[:, None]).astype(float)
    disc = it.discrimination(m)
    for j in range(60):
        rest = m.sum(axis=1) - m[:, j]
        if np.ptp(m[:, j]):
            assert disc[j] == pytest.approx(st.pointbiserialr(m[:, j], rest).statistic, abs=1e-12)
    m[:, 0], m[:, 1] = 1.0, 0.0
    m[:, 2] = (np.arange(21) < 8).astype(float)                   # solved only by the weakest
    res = it.item_analysis(m, items=[f"q{j:02d}" for j in range(60)])
    flags = {r["item"]: r["flags"] for r in res["items"]}
    assert flags["q00"] == ["all_solve"] and flags["q01"] == ["none_solve"]
    assert flags["q02"] == ["negative_discrimination"]
    assert res["flag_counts"]["all_solve"] >= 1 and "q02" in res["flagged"]
    assert math.isnan(res["items"][0]["discrimination"])
    assert it.difficulty(m)[2] == pytest.approx(8 / 21)


def test_item_analysis_with_missing_runs():
    m = np.array([[1, 0, 1, np.nan], [1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1.0]])
    d = it.difficulty(m)
    assert d.tolist() == pytest.approx([0.75, 0.5, 0.5, 2 / 3])
    disc = it.discrimination(m)
    rest0 = np.array([1 / 2, 2 / 3, 0, 1])                         # mean of each model's other items
    ref = np.corrcoef(m[:, 0], rest0)[0, 1]
    assert disc[0] == pytest.approx(ref)
    rest3 = np.array([2 / 3, 0, 1])                                # model 0 did not run item 3
    assert disc[3] == pytest.approx(np.corrcoef([1, 0, 1], rest3)[0, 1])
    models, items, mat = it.matrix_from_samples({"m1": {"a": [1, 1, 0], "b": [1, 0]},
                                                 "m2": {"a": [0, 0, 1]}})
    assert models == ["m1", "m2"] and items == ["a", "b"]
    assert mat[0].tolist() == [1.0, 0.0] and mat[1, 0] == 0.0 and math.isnan(mat[1, 1])
    _, _, p1 = it.matrix_from_samples({"m1": {"a": [1, 1, 0]}}, reduce="pass1")
    assert p1[0, 0] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        it.difficulty([[1.5, 0]])


def test_rasch_recovers_parameters_and_drops_extremes():
    rng = np.random.default_rng(11)
    b = np.linspace(-2, 2, 20)
    theta = rng.normal(size=400)
    x = (rng.random((400, 20)) < 1 / (1 + np.exp(-(theta[:, None] - b[None, :])))).astype(float)
    x[rng.random(x.shape) < 0.05] = np.nan                        # some runs missing
    r = it.rasch_fit(x)
    assert r["converged"]
    est = np.array([r["items"][f"item{j}"]["difficulty"] for j in range(20)])
    assert abs(est.mean()) < 1e-9                                 # identification
    assert np.corrcoef(est, b)[0, 1] > 0.98 and np.sqrt(np.mean((est - b) ** 2)) < 0.3
    infit = [v["infit"] for v in r["items"].values()]
    assert 0.8 < np.median(infit) < 1.2                            # data follow the model
    small = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0]], dtype=float)
    rs = it.rasch_fit(small, models=list("abcde"))
    # a (all 1) and d (all 0) go first; then item2 is all 0, which leaves e all 1
    assert rs["dropped_models"] == ["a", "d", "e"] and rs["dropped_items"] == ["item2"]
    assert rs["converged"] and set(rs["models"]) == {"b", "c"}
    with pytest.raises(ValueError, match="solved_min"):
        it.rasch_fit([[0.5, 1], [0, 1]])


def test_rasch_reports_no_estimates_when_the_mle_does_not_exist():
    # correctness probe (f): no all-0/all-1 row or column, but a Guttman-like separated block;
    # the old fit reported converged=True with estimates of +-14.7
    g = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 1]], float)
    assert not it.mle_exists(g)
    r = it.rasch_fit(g)
    assert r["converged"] is False and r["mle_exists"] is False and "items" not in r
    assert "separated" in r["reason"]
    assert it.mle_exists(np.array([[1, 0], [0, 1]], float))         # overlapping: finite (all 0)
    rng = np.random.default_rng(23)
    theta, b = rng.normal(scale=1.5, size=21), rng.normal(scale=1.5, size=150)
    x = (rng.random((21, 150)) < 1 / (1 + np.exp(-(theta[:, None] - b[None, :])))).astype(float)
    fit = it.rasch_fit(x)                                            # panel-sized data: exists
    assert fit["mle_exists"] and fit["converged"] and fit["max_abs_estimate"] < 10


def test_item_analysis_on_partial_credit_scores():
    rng = np.random.default_rng(24)
    res = {f"m{r:02d}": {f"k{i:02d}": rng.choice([0.0, 0.5, 1.0], size=n).tolist()
                         for i in range(30)} for r, n in zip(range(8), [5, 5, 5, 3, 3, 3, 1, 1])}
    with pytest.raises(ValueError):
        it.matrix_from_samples(res)                                  # majority needs pass/fail
    models, items, mat = it.matrix_from_samples(res, "mean")        # mixed n is fine
    assert mat[0, 0] == pytest.approx(np.mean(res["m00"]["k00"]))
    a = it.item_analysis(mat, items, models)
    assert a["items"][0]["difficulty"] == pytest.approx(np.mean(mat[:, 0]))
    rasch = it.rasch_fit(mat, items, models, solved_min=it.RASCH_SOLVED_MIN)
    assert rasch["solved_min"] == 0.5
    with pytest.raises(ValueError):
        it.matrix_from_samples(res, "median")


def test_cluster_labels_must_be_present():
    for bad in ([None, "a"], ["a", float("nan")], ["a", "  "]):
        with pytest.raises(ValueError, match="cluster_id"):
            fn.cluster_codes(bad, 2)
    codes, g = fn.cluster_codes(["fam2", "fam1", "fam2"], 3)
    assert codes.tolist() == [1, 0, 1] and g == 2
    with pytest.raises(ValueError):
        fn.cluster_bootstrap_ci([0.1, 0.2], [None, "x"], B=10)


# ---------------------------------------------------------------- agreement ----------------------

def _table(a, b, c, d):
    return (["y"] * (a + b) + ["n"] * (c + d), ["y"] * a + ["n"] * b + ["y"] * c + ["n"] * d)


@pytest.mark.parametrize("cells, kappa", [((20, 5, 10, 15), 0.4), ((45, 15, 25, 15), 0.1304),
                                          ((25, 35, 5, 35), 0.2593)])
def test_cohen_kappa_textbook_tables(cells, kappa):
    assert round(ag.cohen_kappa(*_table(*cells)), 4) == kappa


def test_weighted_kappa_against_definitions():
    rng = np.random.default_rng(12)
    x = rng.integers(0, 5, 120).tolist()
    y = [min(4, max(0, v + int(rng.integers(-2, 3)))) for v in x]
    cats = list(range(5))
    obs = np.zeros((5, 5))
    for a, b in zip(x, y):
        obs[a, b] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.array([[abs(i - j) / 4 for j in range(5)] for i in range(5)])
    ref_lin = 1 - (w * obs).sum() / (w * exp).sum()
    assert ag.cohen_kappa(x, y, cats, "linear") == pytest.approx(ref_lin, abs=1e-12)
    xa, ya = np.array(x, float), np.array(y, float)
    cov = np.mean((xa - xa.mean()) * (ya - ya.mean()))
    ref_q = 2 * cov / (xa.var() + ya.var() + (xa.mean() - ya.mean()) ** 2)
    assert ag.cohen_kappa(x, y, cats, "quadratic") == pytest.approx(ref_q, abs=1e-12)
    assert ag.cohen_kappa(x, y, cats) == pytest.approx(
        (np.trace(obs) - np.trace(exp)) / (1 - np.trace(exp)), abs=1e-12)
    assert math.isnan(ag.cohen_kappa(["a", "a"], ["a", "a"]))
    with pytest.raises(ValueError):
        ag.cohen_kappa([1, None], [1, 0])
    with pytest.raises(ValueError):
        ag.cohen_kappa([1, 2], [1, 3], categories=[1, 2])


def test_scott_pi_and_krippendorff_binary_example():
    r1 = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]                          # Krippendorff (2011), section A
    r2 = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    assert ag.scott_pi(r1, r2) == pytest.approx(1 / 21)
    alpha = ag.krippendorff_alpha([r1, r2])
    assert round(alpha, 3) == 0.095
    assert alpha == pytest.approx(1 - (19 / 20) * (1 - ag.scott_pi(r1, r2)))
    assert ag.cohen_kappa(r1, r2) == pytest.approx(0.04 / 0.44)


def test_krippendorff_alpha_classic_example_all_levels():
    data = cli.KRIPP_EXAMPLE
    got = {lv: round(ag.krippendorff_alpha(data, lv), 3) for lv in ("nominal", "ordinal", "interval", "ratio")}
    assert got == {"nominal": 0.743, "ordinal": 0.815, "interval": 0.849, "ratio": 0.797}
    nan_data = [[np.nan if v is None else float(v) for v in row] for row in data]
    assert ag.krippendorff_alpha(nan_data, "interval") == pytest.approx(ag.krippendorff_alpha(data, "interval"))
    lettered = [[None if v is None else "abcde"[v - 1] for v in row] for row in data]
    assert ag.krippendorff_alpha(lettered) == pytest.approx(ag.krippendorff_alpha(data))
    with pytest.raises(ValueError):
        ag.krippendorff_alpha(lettered, "interval")
    assert ag.krippendorff_alpha([[1, 2], [1, 2]]) == 1.0
    assert math.isnan(ag.krippendorff_alpha([[1, 1], [1, 1]]))


def test_kappa_ci_judge_rule_gap_and_groups():
    rng = np.random.default_rng(13)
    truth = rng.integers(0, 2, 150).tolist()
    judge = [t if rng.random() < 0.9 else 1 - t for t in truth]
    r = ag.cohen_kappa_ci(truth, judge, B=3000)
    assert r == ag.cohen_kappa_ci(truth, judge, B=3000)
    assert r["kappa"] == pytest.approx(ag.cohen_kappa(truth, judge)) and r["lo"] < r["kappa"] < r["hi"]
    cl = [f"q{i // 3}" for i in range(150)]
    rc = ag.cohen_kappa_ci(truth, judge, clusters=cl, B=3000)
    assert rc["n_clusters"] == 50 and rc["kappa"] == r["kappa"]
    assert ag.judge_passes({"kappa": 0.7, "lo": 0.6}) and not ag.judge_passes({"kappa": 0.69, "lo": 0.65})
    with pytest.raises(ValueError):
        ag.judge_passes({"kappa": 0.9, "lo": 0.8, "alpha": 0.1})
    h = rng.random((40, 3))
    h[0, 1] = np.nan
    hm = np.nanmean(h, axis=1)
    g = ag.judge_human_gap(hm + 0.05, h, B=500)
    assert g["gap"] == pytest.approx(0.05) and g["boot_lo"] == pytest.approx(0.05)
    assert g["mean_abs_gap"] == pytest.approx(0.05)
    groups = ["calc"] * 25 + ["factual"] * 15
    bg = ag.by_group(groups, ag.cohen_kappa, truth[:40], judge[:40])
    assert bg["all"]["result"] == pytest.approx(ag.cohen_kappa(truth[:40], judge[:40]))
    assert bg["calc"]["result"] == pytest.approx(ag.cohen_kappa(truth[:25], judge[:25]))
    assert bg["factual"]["small"] and not bg["calc"]["small"]
    assert bg["factual"]["result"] is None and "§4.2" in bg["factual"]["note"]   # withheld
    bs = ag.by_group(groups, ag.cohen_kappa, truth[:40], judge[:40], include_small=True)
    assert bs["factual"]["result"] == pytest.approx(ag.cohen_kappa(truth[25:40], judge[25:40]))
    bad = ag.by_group(["x", "y"], ag.cohen_kappa, [1, None], [1, 0], min_items=1)
    assert "error" in bad["y"] and "result" in bad["x"]


def _calibration_rows(seed=25, n_answers=50, points=3):
    rng = np.random.default_rng(seed)
    cats = ["met", "partially_met", "not_met"]
    truth = rng.choice(cats, size=n_answers * points, p=[0.5, 0.2, 0.3])

    def noisy(p):
        return [t if rng.random() < p else str(rng.choice(cats)) for t in truth]

    ann = {"T1": noisy(0.9), "T2": noisy(0.9), "E": noisy(0.95)}
    adj = [None if len({ann[k][r] for k in ann}) == 1 else str(truth[r]) for r in range(truth.size)]
    answers = [f"ans{r // points:03d}" for r in range(truth.size)]
    return truth.tolist(), ann, adj, answers, noisy(0.85)


def test_human_reference_is_unanimous_or_adjudicated():
    truth, ann, adj, answers, judge = _calibration_rows()
    hr = ag.human_reference(ann, adj)
    for r, ref in enumerate(hr["reference"]):
        labels = {ann[k][r] for k in ann}
        assert ref == (labels.pop() if len(labels) == 1 else adj[r])
    assert hr["n_unanimous"] + hr["n_adjudicated"] == len(truth) and hr["annotators"] == ["E", "T1", "T2"]
    with pytest.raises(ValueError, match="adjudicated"):
        ag.human_reference(ann, None)                                # disagreements need adjudication
    clash = [None] * len(adj)
    clash[next(r for r, a in enumerate(adj) if a is None)] = "other"
    with pytest.raises(ValueError, match="contradicting"):
        ag.human_reference(ann, [a if a is not None else c for a, c in zip(adj, clash)])
    with pytest.raises(ValueError):
        ag.human_reference({"T1": ann["T1"], "T2": [None] + ann["T2"][1:]}, adj)


def test_judge_calibration_gate_is_kappa_against_the_adjudicated_reference():
    truth, ann, adj, answers, judge = _calibration_rows()
    c = ag.judge_calibration(judge, ann, adj, answers, B=1000)
    ref = ag.human_reference(ann, adj)["reference"]
    gate = ag.cohen_kappa_ci(judge, ref, clusters=answers, B=1000)
    assert c["gate_lines"] == gate and c["passes_lines"] == ag.judge_passes(gate)
    assert c["passes"] is None and not c["complete"]                   # no answer scores: no band gate
    assert c["gate_lines"]["n_clusters"] == 50                         # whole answers resampled
    assert c["scott_pi"] == pytest.approx(ag.scott_pi(judge, ref))
    assert c["judge_vs_annotator"]["E"] == pytest.approx(ag.cohen_kappa(judge, ann["E"]))
    assert set(c["human_human_pairwise"]) == {"E|T1", "E|T2", "T1|T2"}
    assert c["human_human_mean"] == pytest.approx(np.mean(list(c["human_human_pairwise"].values())))
    assert c["human_krippendorff_alpha"] == pytest.approx(
        ag.krippendorff_alpha([ann["E"], ann["T1"], ann["T2"]]))
    scores = {a: 0.25 + 0.5 * (i % 2) for i, a in enumerate(dict.fromkeys(answers))}
    perfect = ag.judge_calibration(ref, ann, adj, answers, judge_scores=scores, human_scores=scores, B=300)
    assert perfect["gate_lines"]["kappa"] == 1.0 and perfect["gate_bands"]["kappa"] == 1.0
    assert perfect["passes"] and perfect["complete"] and perfect["n_answers"] == 50
    with pytest.raises(ValueError):
        ag.judge_calibration(judge[:-1], ann, adj, answers)


def _public_functions():
    out = {}
    for mod in (fn, ag, it, md):
        for name, obj in vars(mod).items():
            if inspect.isfunction(obj) and obj.__module__ == mod.__name__ and not name.startswith("_"):
                out[obj.__code__] = f"{mod.__name__.rsplit('.', 1)[1]}.{name}"
    return out


def test_selfcheck_exercises_every_public_function():
    public = _public_functions()
    called = set()

    def prof(frame, event, arg):
        if event == "call" and frame.f_code in public:
            called.add(public[frame.f_code])

    sys.setprofile(prof)
    try:
        rows = cli.selfcheck(verbose=False)
    finally:
        sys.setprofile(None)
    assert all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
    assert sorted(set(public.values()) - called) == []


# ---------------------------------------------------------------- CLI ----------------------------

def test_cli_selfcheck_and_mde(capsys):
    assert cli.main(["selfcheck"]) == 0
    out = capsys.readouterr().out
    assert f"{len(cli.CHECKS)}/{len(cli.CHECKS)} checks passed" in out and "FAIL" not in out
    assert cli.main(["mde"]) == 0
    out = capsys.readouterr().out
    for line in ("K       285", "F       150", "A        30"):
        assert line in out
    assert out.count(" yes") == 3
    assert cli.main(["mde", "--n", "150", "--pd", "0.2"]) == 0
    assert "MDE = 10.23 points" in capsys.readouterr().out

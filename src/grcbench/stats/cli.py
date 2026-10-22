"""`./gb stats <selfcheck | mde | proposals | h1c-sim>`: check the analysis functions, print Table 8
and the G2 proposals (eval plan §9).

    ./gb stats selfcheck        run every public function of grcbench.stats on synthetic data with a
                                known answer (textbook examples, scipy, closed forms, brute force)
                                and print a pass/fail table; exit status 1 if any check fails
    ./gb stats mde              print eval plan Table 8 recomputed (MDE at p_d = 0.2 and 0.3)
    ./gb stats mde --n 120 --pd 0.25 [--alpha 0.05 --power 0.8]
                                the MDE for another number of items and discordant share
    ./gb stats proposals [--format text|md|json]
                                the analysis PROPOSALS for the G2 pre-registration (stats.proposals)
    ./gb stats h1c-sim [--ratio 1.0 --spread 1.5 --datasets 400 --B 1000 --workers N]
                                how often h1c_decision holds on simulated panels (ratio 1 = the
                                null, its false-positive rate; ratio > 1 = power)

Everything here is synthetic: no benchmark item, answer or model output is read.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import numpy as np
from scipy import stats as st

from . import agreement as ag
from . import functions as fn
from . import items as it
from . import mde as md
from . import proposals as pr

Check = tuple[str, str, Callable[[], tuple[str, str, bool]]]


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return bool(abs(a - b) <= tol)


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def _raises(f: Callable, *args, **kwargs) -> str | None:
    """The ValueError message f raises, or None if it does not raise."""
    try:
        f(*args, **kwargs)
    except ValueError as exc:
        return str(exc)
    return None


# ---------------------------------------------------------------- the checks ---------------------

def _c_pass_at_k():
    got = (fn.pass_at_k(5, 2, 1), fn.pass_at_k(5, 2, 2), fn.pass_at_k(5, 2, 4))
    ok = _close(got[0], 0.4) and _close(got[1], 0.7) and got[2] == 1.0
    for n in range(1, 8):                       # Chen et al.'s numerically stable product form
        for c in range(n + 1):
            for k in range(1, n + 1):
                ref = 1.0 if n - c < k else 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
                ok &= _close(fn.pass_at_k(n, c, k), ref, 1e-12)
    return "0.4, 0.7, 1 (n=5,c=2,k=1/2/4); = Chen product form", ", ".join(map(_fmt, got)), ok


def _c_pass_hat_k():
    got = (fn.pass_hat_k(4, 4, 4), fn.pass_hat_k(4, 3, 4), fn.pass_hat_k(5, 4, 4),
           fn.mean_pass_hat_k({"t1": [1, 1, 1, 1], "t2": [1, 1, 1, 0]}, 4))
    ok = got[0] == 1.0 and got[1] == 0.0 and _close(got[2], 0.2) and _close(got[3], 0.5)
    return "1, 0, 0.2, mean 0.5", ", ".join(map(_fmt, got)), ok


def _c_item_scores():
    p1 = fn.item_pass_at_1({"b": [1, 0], "a": [1, 1, 0, 0, 1]})
    sc = fn.item_scores({"k1": [1.0, 0.5, 0.0], "k2": [0.5, 0.5, 1.0]}, "score")
    m = fn.mean_pass_at_k({"a": [1, 1, 0, 0, 1], "b": [1, 0]})
    err = _raises(fn.item_scores, {"x": [0.5]}, "pass_fail")
    ok = p1 == {"a": 0.6, "b": 0.5} and _close(sc["k1"], 0.5) and _close(sc["k2"], 2 / 3) \
        and _close(m, 0.55) and err is not None and "score" in err
    return "pass@1 0.6/0.5; Tier-K-like mean scores 0.5/0.667; 0.5 rejected as pass/fail", \
        f"{p1}; {sc['k1']:.3f}/{sc['k2']:.3f}; mean {m:.3f}", ok


def _c_exec_func():
    highest = {"a": ["V6", "V5", "V2", None, "V6"], "b": ["V5"] * 5, "c": ["V3"] * 5}
    e = fn.exec_at_1(highest)
    f = fn.func_at_1(highest, {"a"})
    fu = fn.funnel(highest, {"a"})
    ok = _close(e["value"], (0.6 + 1.0 + 0.0) / 3) and _close(f["value"], 0.4) \
        and f["n_tasks"] == 1 and f["excluded_no_oracle"] == 2
    ok &= _close(fu["rates"]["V6"], f["value"]) and _close(fu["rates"]["V5"], e["value"]) \
        and fu["n_tasks"]["V6"] == 1 and fu["n_tasks"]["V1"] == 3 and fn.reached("V6", "V5")
    return "Exec@1 0.5333; Func@1 0.4 over 1 oracle task; funnel V6 = Func@1", \
        f"Exec@1 {e['value']:.4f}; Func@1 {f['value']:.4f} over {f['n_tasks']}; " \
        f"funnel V6 {fu['rates']['V6']:.4f} (n {fu['n_tasks']['V6']})", ok


def _c_cluster_se():
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    a = fn.cluster_robust_se(y)
    ok1 = _close(a, float(st.sem(y)), 1e-12)
    means = rng.normal(size=40)                 # 40 clusters of 5 identical items
    y2 = np.repeat(means, 5)
    cl = np.repeat([f"c{g:02d}" for g in range(40)], 5)
    b = fn.cluster_robust_se(y2, cl)
    ok2 = _close(b, float(st.sem(means)), 1e-12)
    ci = fn.clustered_mean_ci(y)
    lo, hi = st.t.interval(0.95, 199, loc=y.mean(), scale=st.sem(y))
    ok3 = _close(ci["lo"], lo, 1e-12) and _close(ci["hi"], hi, 1e-12)
    codes, g = fn.cluster_codes(["b", "a", "b"], 3)
    ok4 = codes.tolist() == [1, 0, 1] and g == 2 and _raises(fn.cluster_codes, ["a", None], 2)
    return "singletons = scipy sem and t interval; constant clusters = sem(means); None label refused", \
        f"{a:.6f} vs {st.sem(y):.6f}; {b:.6f} vs {st.sem(means):.6f}", bool(ok1 and ok2 and ok3 and ok4)


def _c_bootstrap():
    rng = np.random.default_rng(2)
    g = rng.normal(size=60)
    y = np.repeat(g, 4) + rng.normal(scale=0.5, size=240)
    cl = np.repeat(np.arange(60), 4)
    r1 = fn.cluster_bootstrap_ci(y, cl, B=4000, seed=7)
    r2 = fn.cluster_bootstrap_ci(y, cl, B=4000, seed=7)
    cr0 = fn.cluster_robust_se(y, cl, "CR0")
    w = fn.bootstrap_weights(240, cl, B=5, seed=1)
    ok = r1 == r2 and abs(r1["se_boot"] / cr0 - 1) < 0.05 and r1["lo"] < r1["mean"] < r1["hi"] \
        and w.shape == (5, 240) and bool(np.all(w.sum(axis=1) == 240))
    return "same seed -> same CI; bootstrap SE ~ CR0 SE (5%)", \
        f"se_boot {r1['se_boot']:.4f} vs CR0 {cr0:.4f}; CI [{r1['lo']:.3f}, {r1['hi']:.3f}]", ok


def _c_paired():
    a = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.6])
    d = fn.paired_difference(a, a - 0.1, B=500)
    ok = _close(d["diff"], 0.1) and _close(d["boot_lo"], 0.1) and _close(d["boot_hi"], 0.1)
    rng = np.random.default_rng(12)
    x, y = rng.random(30), rng.random(30)
    cl = np.repeat(np.arange(10), 3)
    ok &= fn.paired_cluster_robust_se(x, y, cl) == fn.cluster_robust_se(x - y, cl)
    ok &= fn.paired_cluster_bootstrap_ci(x, y, cl, B=300) == fn.cluster_bootstrap_ci(x - y, cl, B=300)
    return "constant +0.1 shift: diff 0.1, CI [0.1, 0.1]; paired = one-sample on a - b", \
        f"{d['diff']:.4f} [{d['boot_lo']:.4f}, {d['boot_hi']:.4f}]", ok


def _c_mcnemar():
    p = fn.mcnemar_exact_p(1, 6)
    ok = _close(p, 2 * 8 / 128)
    for x in range(0, 15):
        for y in range(0, 15):
            ref = 1.0 if x + y == 0 else st.binomtest(x, x + y, 0.5).pvalue
            ok &= _close(fn.mcnemar_exact_p(x, y), ref, 1e-12)
    maj = (fn.majority_pass([1, 1, 0, 0]), fn.majority_pass([1, 1, 1, 0, 0]))
    ok &= maj == (False, True)
    r = fn.mcnemar_exact([1, 1, 0, 0, 1], [0, 1, 1, 1, 1])
    ok &= (r["a_only"], r["b_only"], r["both"], r["neither"]) == (1, 2, 2, 0)
    s = fn.mcnemar_from_samples({"i1": [1, 1, 1, 0, 0], "i2": [1, 1, 0, 0]},
                                {"i1": [0] * 5, "i2": [1, 1, 1, 1]})
    ok &= (s["a_only"], s["b_only"]) == (1, 1) and _close(s["p_value"], 1.0)
    return "p(1,6) = 0.125 = binomtest; 2/4 fails, 3/5 passes; counts 1/2/2/0", f"{p:.4f}; {maj}", ok


def _c_clustered_tests():
    # three clusters, every discordant item favours a: D_g = 3, 1, 1 -> P(|S| >= 5) = 2/8
    a = [1, 1, 1, 1, 1, 0]
    b = [0, 0, 0, 0, 0, 0]
    cl = ["c1", "c1", "c1", "c2", "c3", "c3"]
    r = fn.mcnemar_clustered(a, b, cl)
    ok = _close(r["p_value"], 0.25) and _close(fn.mcnemar_exact_p(5, 0), 0.0625)
    ok &= r["n_clusters_informative"] == 3 and _close(r["durkalski_chi2"], 25 / 11)
    for x in range(0, 12):                     # singletons: exactly the exact McNemar test
        for y in range(0, 12):                 # (plus two concordant items)
            ok &= _close(fn.mcnemar_clustered([1] * x + [0] * y + [1, 0],
                                              [0] * x + [1] * y + [1, 0])["p_value"],
                         fn.mcnemar_exact_p(x, y), 1e-12)
    rng = np.random.default_rng(13)
    u, v = rng.normal(size=25), rng.normal(size=25)
    w = fn.wilcoxon_clustered(u, v)
    ok &= _close(w["p_value"], float(st.wilcoxon(u - v, method="exact").pvalue), 1e-12)
    g = rng.integers(0, 6, 12) / 5, rng.integers(0, 6, 12) / 5
    nz = np.round(g[0] - g[1], 12)
    nz = nz[nz != 0]
    ref = float(st.wilcoxon(nz, method=st.PermutationMethod(n_resamples=9999)).pvalue)
    ok &= _close(fn.wilcoxon_clustered(*g)["p_value"], ref, 1e-12)
    wc = fn.wilcoxon_clustered([0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.3, 0.4], ["q", "q", "q", "q"])
    ok &= _close(wc["p_value"], 1.0) and _close(wc["rank_biserial"], 1.0)
    return "clusters D_g=3,1,1: p 0.25 (items alone 0.0625); singletons = exact tests; 1 cluster -> 1", \
        f"p {r['p_value']:.4f}; one-cluster p {wc['p_value']:.3f}", ok


def _c_wilcoxon():
    r = fn.wilcoxon_signed_rank(np.arange(1, 11), np.zeros(10))
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=30), rng.normal(size=30)
    r2 = fn.wilcoxon_signed_rank(a, b)
    ref = st.wilcoxon(a - b).pvalue
    ok = _close(r["p_value"], 2 / 1024) and _close(r2["p_value"], float(ref), 1e-12) \
        and _close(r["rank_biserial"], 1.0)
    return "10 positive ranks: p = 2/1024, r_rb = 1; = scipy", \
        f"{r['p_value']:.6f}; {r2['p_value']:.4f}", ok


def _c_holm():
    res = fn.holm_bonferroni({"RQ1a": {"h1": 0.01, "h2": 0.04, "h3": 0.03, "h4": 0.005}})["RQ1a"]
    got = [round(res[h]["p_holm"], 10) for h in ("h1", "h2", "h3", "h4")]
    rej = [res[h]["reject"] for h in ("h1", "h2", "h3", "h4")]
    ok = got == [0.03, 0.06, 0.06, 0.02] and rej == [True, False, False, True]
    adj = fn.holm_adjust([0.04, 0.01])
    ok &= _close(float(adj[0]), 0.04) and _close(float(adj[1]), 0.02)
    via = fn.holm({"cmp": {"p_primary": 0.01, "secondary": {"p_value": 1e-9}}, "x": 0.04})
    ok &= _close(via["cmp"]["p_holm"], 0.02)
    return "adjusted 0.03 0.06 0.06 0.02; reject h1 h4; a comparison enters by p_primary", \
        f"{got}; {rej}", ok


def _c_effects():
    h = fn.cohens_h(0.6, 0.4)
    orr = fn.matched_odds_ratio(6, 2)
    ref = 2 * (math.asin(math.sqrt(0.6)) - math.asin(math.sqrt(0.4)))
    ok = _close(h, ref) and _close(fn.cohens_h(1, 0), math.pi) and _close(orr["odds_ratio"], 3.0) \
        and orr["lo"] < 1 < orr["hi"]
    return "h(0.6,0.4) = 0.4027, h(1,0) = pi; OR(6,2) = 3 with CI over 1", \
        f"{h:.4f}; OR {orr['odds_ratio']:.2f} [{orr['lo']:.3f}, {orr['hi']:.2f}]", ok


def _c_compare_pass_fail():
    # majority and pass@1 disagree in sign: A 5/5 vs B 3/5 on 100 items, A 2/5 vs B 3/5 on 50 items
    a = {f"x{i:03d}": [1] * 5 for i in range(100)} | {f"y{i:03d}": [1, 1, 0, 0, 0] for i in range(50)}
    b = {f"x{i:03d}": [1, 1, 1, 0, 0] for i in range(100)} | {f"y{i:03d}": [1, 1, 1, 0, 0] for i in range(50)}
    r = fn.compare_paired(a, b, B=500)
    pr, se = r["primary"], r["secondary"]
    ok = pr["test"] == "mcnemar_exact" and _close(r["p_primary"], fn.mcnemar_exact_p(0, 50)) \
        and _close(pr["difference"]["diff"], -1 / 3) and pr["odds_ratio"]["odds_ratio"] == 0.0 \
        and pr["cohens_h"] < 0 and _close(se["difference"]["diff"], 0.2) and se["p_value"] < 0.05
    ok &= _close(r["mde"]["pd_0.2"], md.mde(150, 0.2)) and _close(r["mde"]["pd_0.3"], md.mde(150, 0.3))
    clusters = {i: i[0] + i[1:3] for i in a}                      # 10 clusters of x, 5 of y
    rc = fn.compare_paired(a, b, clusters, B=500)
    ok &= rc["primary"]["test"] == "mcnemar_clustered" and rc["n_clusters"] == 15 \
        and _close(rc["p_primary"], 2 / 2 ** 5)                   # 5 informative clusters, all one way
    return "primary McNemar: share diff -1/3, OR 0 (favour B); secondary pass@1 +0.2; MDE Table 8", \
        f"diff {pr['difference']['diff']:.4f}, p {r['p_primary']:.2e}; pass@1 diff " \
        f"{se['difference']['diff']:.3f}; clustered p {rc['p_primary']:.4f}", ok


def _c_compare_score():
    rng = np.random.default_rng(14)
    items = [f"k{i:03d}" for i in range(60)]
    a = {i: rng.choice([0.0, 0.5, 1.0], size=5, p=[0.2, 0.3, 0.5]).tolist() for i in items}
    b = {i: rng.choice([0.0, 0.5, 1.0], size=5, p=[0.4, 0.3, 0.3]).tolist() for i in items}
    r = fn.compare_paired(a, b, outcome="score", B=500)
    ma = [np.mean(a[i]) for i in items]
    mb = [np.mean(b[i]) for i in items]
    ref = fn.wilcoxon_signed_rank(ma, mb)
    ok = r["primary"]["test"] == "wilcoxon_signed_rank" and _close(r["p_primary"], ref["p_value"]) \
        and r["secondary"] is None and _close(r["primary"]["difference"]["diff"], np.mean(ma) - np.mean(mb))
    ok &= _raises(fn.compare_paired, a, b) is not None              # partial credit is not pass/fail
    return "Tier-K-like 0/0.5/1 scores: primary = Wilcoxon on item means; refused as pass/fail", \
        f"diff {r['primary']['difference']['diff']:.4f}, p {r['p_primary']:.2e}", ok


def _c_mde():
    rows = md.table8()
    ok = all(r["matches_plan"] for r in rows)
    n = md.n_for_mde(0.10, 0.2)
    ok &= md.mde(n, 0.2) <= 0.10 < md.mde(n - 1, 0.2) and md.points(0.0742) == 7 and md.points(0.125) == 13
    return "K 7-9, F 10-13, A 23-28; n_for_mde inverts mde", \
        ", ".join(f"{r['tier']} {r['points']}" for r in rows) + f"; n(10 pts, 0.2) = {n}", ok


def _panel(seed: int = 4, n_models: int = 5, n_items: int = 285, n: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = np.linspace(0.1, 0.9, n_models)
    return (rng.random((n_models, n_items, n)) < p[:, None, None]).astype(float)


def _c_rank_stability():
    x = _panel()
    r = fn.rank_stability_bootstrap(x.mean(axis=2), B=200, seed=5)
    s = fn.summarize([0.7, 0.9, float("nan")], 0.8)
    ok = r["tau_between_resamples"]["median"] == 1.0 and r["tau_vs_full"]["min"] == 1.0 \
        and s["n_undefined"] == 1 and s["share_ge_threshold"] == 0.5 \
        and fn.kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0
    return "well-separated models: tau = 1", \
        f"median {r['tau_between_resamples']['median']}, min vs full {r['tau_vs_full']['min']}", ok


def _c_split_half():
    base = _panel(n=1)
    x = np.repeat(base, 5, axis=2)              # every sample identical -> halves agree exactly
    s = fn.split_half_tau(x)
    rh = fn.random_half_tau(x, repeats=50)
    ok = s["tau"] == 1.0 and s["dropped_samples"] == [5] and rh["tau"]["min"] == 1.0 \
        and fn.split_halves(5) == fn.SPLIT_HALVES and fn.split_halves(3) == ((0,), (1,))
    return "identical halves: tau = 1; sample 5 dropped; n = 3 -> {1} vs {2}", \
        f"{s['tau']}; dropped {s['dropped_samples']}; random-half min {rh['tau']['min']}", ok


def _c_split_half_panel():
    items = [f"i{j}" for j in range(30)]
    p = {"m5a": 0.2, "m5b": 0.4, "m3a": 0.6, "m3b": 0.8, "frontier": 0.9}
    n = {"m5a": 5, "m5b": 5, "m3a": 3, "m3b": 3, "frontier": 1}
    rng = np.random.default_rng(15)
    res = {m: {i: [float(rng.random() < p[m])] * n[m] for i in items} for m in p}   # identical samples
    s = fn.split_half_panel(res)
    rh = fn.random_half_panel(res, repeats=20)
    err = _raises(fn.sample_tensor, res)
    ok = s["tau"] == 1.0 and s["excluded"] == {"frontier": 1} and s["n_models"] == 4 \
        and s["halves_by_n"]["3"] == {"half_1_samples": [1], "half_2_samples": [2],
                                      "dropped_samples": [3]} \
        and s["halves_by_n"]["5"]["dropped_samples"] == [5] and rh["tau"]["min"] == 1.0 \
        and err is not None and "different numbers of samples" in err
    s4 = fn.split_half_panel(res, min_samples=4)
    ok &= s4["models"] == ["m5a", "m5b"]
    m5 = {m: res[m] for m in ("m5a", "m5b")}
    ok &= fn.sample_tensor(m5)[2].shape == (2, 30, 5)
    return "n = 5, 3, 1 panel: frontier (n=1) excluded, n=3 split {1}|{2}; mixed n refused clearly", \
        f"tau {s['tau']}, excluded {s['excluded']}, n>=4 models {s4['models']}", ok


def _c_determinism():
    r = fn.harness_determinism({"a": [1] * 10, "b": [1] * 9 + [0], "c": [0] * 10, "d": [1] * 5},
                               digests={"a": ["x"] * 10, "b": ["x"] * 9 + ["y"]})
    got = (r["n_pass_all"], r["n_flaky"], r["n_fail_all"], r["n_too_few_runs"], r["varying_outputs"])
    return "1 pass_all, 1 flaky, 1 fail_all, 1 too_few; b varies", _fmt(got), \
        got == (1, 1, 1, 1, ["b"])


def _c_h1c():
    rng = np.random.default_rng(6)
    k = (rng.random((5, 285, 5)) < np.linspace(0.45, 0.55, 5)[:, None, None]).astype(float)
    f = _panel(seed=7, n_items=150)
    r = fn.size_matched_separation(k, f, repeats=50, B=200)
    d = fn.h1c_decision(r)
    ok = r["n_items"] == 150 and r["share_k_sd_ge_f"] == 0.0 and r["matched_tests"] \
        and r["k_test"] == r["f_test"] == "wilcoxon" and d["holds"] and d["confirmatory"] \
        and r["difference"]["sd"]["lo"] > 0 and r["difference"]["n_significant_pairs"]["lo"] > 0
    kscore = np.round(np.clip(k * rng.choice([1.0, 0.5], size=k.shape), 0, 1), 1)  # 0 / 0.5 / 1
    rs = fn.size_matched_separation(kscore, f, repeats=10, B=0)
    ok &= rs["tier_k_full"]["test"] == "wilcoxon" and rs["difference"] is None \
        and fn.separation(f, "mcnemar")["n_significant_pairs"] == 10 \
        and _raises(fn.h1c_decision, rs) is not None
    same = fn.size_matched_separation(f[:, :100], f[:, 100:], n_items=50, repeats=5, B=100)
    ok &= not fn.h1c_decision(same)["holds"]                       # one item set split in two
    mixed = fn.size_matched_separation(k, f, "wilcoxon", "mcnemar", repeats=5, B=20)
    ok &= not fn.h1c_decision(mixed)["confirmatory"]
    return "narrow K vs spread F: interval of F - K above 0, H1c holds; same tier: does not hold", \
        f"F sd {r['tier_f']['sd']:.3f}, sig {r['tier_f']['n_significant_pairs']}/" \
        f"{r['tier_f']['n_pairs']}; F - K sd [{r['difference']['sd']['lo']:.3f}, " \
        f"{r['difference']['sd']['hi']:.3f}]", ok


def _c_wilcoxon_rows():
    rng = np.random.default_rng(16)
    worst, seen = 0.0, set()
    for n_items, grid in ((150, 5), (40, 0), (12, 5), (14, 2), (6, 0)):
        x = rng.integers(0, grid + 1, (6, n_items)) / grid if grid else rng.random((6, n_items))
        i, j = np.triu_indices(6, 1)
        p = fn._wilcoxon_rows_p(x[i] - x[j])
        for r, (a, b) in enumerate(zip(i, j)):
            s = fn.wilcoxon_signed_rank(x[a], x[b])
            seen.add(s["method"])
            worst = max(worst, abs(s["p_value"] - p[r]))
    ok = worst <= 1e-12 and {"exact", "permutation", "asymptotic"} <= seen
    return "vectorised pair p-values = wilcoxon_signed_rank in every regime", \
        f"max |dp| {worst:.1e}; regimes {sorted(seen)}", ok


def _c_h1c_mixed_n():
    rng = np.random.default_rng(17)
    ns = {"cfg_a": 5, "cfg_b": 5, "arm": 3, "small": 3, "frontier": 1}
    p = {"cfg_a": 0.3, "cfg_b": 0.5, "arm": 0.6, "small": 0.7, "frontier": 0.9}

    def tier(n_items, tag):
        return {m: {f"{tag}{j:03d}": (rng.random(ns[m]) < p[m]).astype(float).tolist()
                    for j in range(n_items)} for m in ns}

    kd, fd = tier(60, "k"), tier(30, "f")
    s = fn.separation(fd, "mcnemar")
    models = sorted(ns)
    means = [np.mean([np.mean(v) for v in fd[m].values()]) for m in models]
    maj = np.array([[fn.majority_pass(fd[m][i]) for i in sorted(fd[m])] for m in models], float)
    i, j = np.triu_indices(len(models), 1)
    pv = fn.holm_adjust([fn.mcnemar_exact(maj[a], maj[b])["p_value"] for a, b in zip(i, j)])
    ok = s["samples_per_model"] == ns and _close(s["sd"], float(np.std(means, ddof=1)), 1e-12) \
        and s["n_significant_pairs"] == int(np.sum(pv <= 0.05)) and s["models"] == models
    r = fn.size_matched_separation(kd, fd, repeats=5, B=20)
    ok &= r["tier_k_full"]["samples_per_model"]["frontier"] == 1 and r["n_items"] == 30
    return "n = 5, 3, 1 panel: each model reduced on its own n; H1c runs", \
        f"sd {s['sd']:.4f}, sig pairs {s['n_significant_pairs']}/{s['n_pairs']}", ok


def _calibration_result(passes: bool = True) -> dict:
    gate = {"kappa": 0.75 if passes else 0.5, "lo": 0.65, "alpha": 0.05}
    return {"complete": True, "passes": passes, "gate_lines": gate, "gate_bands": gate}


def _c_h1_decisions():
    good = fn.summarize([0.9] * 96 + [0.7] * 4, 0.8)
    rb = {"tau_between_resamples": good}
    a = fn.h1a_decision(rb, {"tau": 0.8}, _calibration_result(),
                        {"n_tasks": 30, "n_pass_all": 30, "share_pass_all": 1.0})
    b = fn.h1a_decision(rb, {"tau": float("nan")})
    half = {"tau_between_resamples": fn.summarize([0.9] * 60 + [0.7] * 40, 0.8)}
    strict = fn.h1a_decision(half, {"tau": 0.9})
    lenient = fn.h1a_decision(half, {"tau": 0.9}, bootstrap_rule="median")
    one_gate = _raises(fn.h1a_decision, rb, {"tau": 0.9}, {"complete": False, "passes": None})
    diff = {"sd": {"estimate": 0.05, "lo": 0.01, "hi": 0.09, "p_one_sided": 0.001},
            "n_significant_pairs": {"estimate": 3.0, "lo": 0.0, "hi": 6.0, "p_one_sided": 0.04},
            "alpha": 0.05, "B": 1000}
    c = fn.h1c_decision({"difference": diff, "k_test": "wilcoxon", "f_test": "wilcoxon",
                         "share_k_sd_ge_f": 0.01, "share_k_sig_pairs_ge_f": 0.2})
    ok = a["holds"] and a["complete"] and not b["holds"] and not b["complete"] \
        and not strict["holds"] and lenient["holds"] and one_gate is not None \
        and c["spread"]["holds"] and not c["pairs"]["holds"] and not c["holds"]
    return "H1a: 4 parts hold; 60% of pairs >= 0.8 fails the share rule, passes median; " \
           "one judge gate refused; H1c needs both lower bounds > 0", \
        f"H1a {a['holds']}/{b['holds']}, share {strict['holds']}, median {lenient['holds']}; " \
        f"H1c {c['holds']}", ok


def _c_seeds_validity():
    s = fn.seed_spread({"seed1": 0.5, "seed2": 0.6, "seed3": 0.7})
    x = [0.21, 0.35, 0.40, 0.52, 0.61, 0.70, 0.74, 0.83]
    y = [0.30, 0.28, 0.45, 0.50, 0.58, 0.77, 0.69, 0.90]
    c = fn.correlation_ci(x, y)
    r = float(np.corrcoef(x, y)[0, 1])
    q = st.norm.ppf(0.975) / math.sqrt(len(x) - 3)
    ok = _close(s["mean"], 0.6) and _close(s["sd"], 0.1) and _close(s["range"], 0.2) \
        and _close(c["r"], r, 1e-12) and _close(c["lo"], math.tanh(math.atanh(r) - q), 1e-9) \
        and _close(c["hi"], math.tanh(math.atanh(r) + q), 1e-9)
    return "seeds sd 0.1; Pearson r with Fisher-z interval", \
        f"sd {s['sd']:.4f}; r {c['r']:.4f} [{c['lo']:.3f}, {c['hi']:.3f}]", ok


def _c_item_stats():
    rng = np.random.default_rng(8)
    m = (rng.random((21, 40)) < np.linspace(0.2, 0.8, 21)[:, None]).astype(float)
    disc = it.discrimination(m)
    ok = True
    for j in range(m.shape[1]):
        rest = m.sum(axis=1) - m[:, j]
        if np.ptp(m[:, j]) > 0:
            ok &= _close(disc[j], float(st.pointbiserialr(m[:, j], rest).statistic), 1e-12)
    m[:, 0], m[:, 1] = 1.0, 0.0
    m[:, 2] = (np.arange(21) < 10).astype(float)          # the weakest models solve it
    res = {r["item"]: r["flags"] for r in it.item_analysis(m)["items"]}
    ok &= res["item0"] == ["all_solve"] and res["item1"] == ["none_solve"] \
        and res["item2"] == ["negative_discrimination"]
    ok &= _close(float(it.difficulty(m)[2]), 10 / 21)
    models, items, mat = it.matrix_from_samples({"m1": {"a": [1, 1, 0], "b": [1, 0, 0, 0]},
                                                 "m2": {"a": [1.0, 0.5, 0.0, 0.5, 1.0]}}, "mean")
    ok &= models == ["m1", "m2"] and _close(mat[0, 0], 2 / 3) and _close(mat[1, 0], 0.6) \
        and math.isnan(mat[1, 1])
    _, _, maj = it.matrix_from_samples({"m1": {"a": [1, 1, 0], "b": [1, 1, 0, 0]}})
    ok &= maj[0].tolist() == [1.0, 0.0]
    return "point-biserial = scipy; flags all/none/negative; majority and mean reductions", \
        f"flags {res['item0']}, {res['item1']}, {res['item2']}", ok


def _c_rasch():
    rng = np.random.default_rng(9)
    b = np.linspace(-2, 2, 25)
    theta = rng.normal(size=300)
    x = (rng.random((300, 25)) < 1 / (1 + np.exp(-(theta[:, None] - b[None, :])))).astype(float)
    r = it.rasch_fit(x)
    est = np.array([r["items"][f"item{j}"]["difficulty"] for j in range(25)])
    corr = float(np.corrcoef(est, b)[0, 1])
    rmse = float(np.sqrt(np.mean((est - b) ** 2)))
    guttman = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 1]], float)
    g = it.rasch_fit(guttman)
    ok = r["converged"] and r["mle_exists"] and corr > 0.98 and rmse < 0.3 \
        and not it.mle_exists(guttman) and not g["converged"] and not g["mle_exists"] \
        and it.mle_exists(np.array([[1, 0], [0, 1]], float))
    return "recovers simulated difficulties (r > 0.98, RMSE < 0.3); separated data -> no estimates", \
        f"r {corr:.4f}, RMSE {rmse:.3f}; separated: mle_exists {g['mle_exists']}", ok


def _kappa_table(a: int, b: int, c: int, d: int) -> tuple[list[str], list[str]]:
    r1 = ["y"] * (a + b) + ["n"] * (c + d)
    r2 = ["y"] * a + ["n"] * b + ["y"] * c + ["n"] * d
    return r1, r2


def _c_kappa():
    got = [round(ag.cohen_kappa(*_kappa_table(*t)), 4)
           for t in ((20, 5, 10, 15), (45, 15, 25, 15), (25, 35, 5, 35))]
    cats, table = ag.confusion(*_kappa_table(20, 5, 10, 15))
    ok = got == [0.4, 0.1304, 0.2593] and cats == ["n", "y"] and table.tolist() == [[15, 10], [5, 20]]
    return "0.4, 0.1304, 0.2593 (textbook 2x2 tables)", _fmt(got), ok


def _c_weighted_kappa():
    rng = np.random.default_rng(10)
    x = rng.integers(0, 4, 80)
    y = np.clip(x + rng.integers(-1, 2, 80), 0, 3)
    kq = ag.cohen_kappa(x.tolist(), y.tolist(), categories=[0, 1, 2, 3], weights="quadratic")
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    ref = 2 * cov / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    return "quadratic kappa = 2cov/(var+var+mean gap^2)", f"{kq:.6f} vs {ref:.6f}", _close(kq, ref, 1e-12)


def _c_scott_alpha_binary():
    r1 = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    r2 = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    pi = ag.scott_pi(r1, r2)
    alpha = ag.krippendorff_alpha([r1, r2], "nominal")
    ok = _close(pi, 1 / 21) and _close(alpha, 1 - 19 * 8 / 168) and \
        _close(alpha, 1 - (19 / 20) * (1 - pi))
    return "pi 0.0476; alpha 0.095 (Krippendorff 2011 A)", f"pi {pi:.4f}; alpha {alpha:.4f}", ok


KRIPP_EXAMPLE = [  # Krippendorff (2011) "Computing Krippendorff's Alpha-Reliability", C: 4 observers
    [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
    [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
]


def _c_kripp():
    got = {lv: round(ag.krippendorff_alpha(KRIPP_EXAMPLE, lv), 3)
           for lv in ("nominal", "ordinal", "interval", "ratio")}
    want = {"nominal": 0.743, "ordinal": 0.815, "interval": 0.849, "ratio": 0.797}
    return "nominal .743 ordinal .815 interval .849 ratio .797", _fmt(got), got == want


def _c_kappa_ci_gap():
    rng = np.random.default_rng(11)
    truth = rng.integers(0, 2, 150)
    judge = np.where(rng.random(150) < 0.9, truth, 1 - truth)
    r = ag.cohen_kappa_ci(truth.tolist(), judge.tolist(), B=2000)
    r2 = ag.cohen_kappa_ci(truth.tolist(), judge.tolist(), B=2000)
    ok = r == r2 and r["lo"] < r["kappa"] < r["hi"] and \
        _close(r["kappa"], ag.cohen_kappa(truth.tolist(), judge.tolist()))
    ok &= ag.judge_passes({"kappa": 0.75, "lo": 0.61}) and not ag.judge_passes({"kappa": 0.75, "lo": 0.59})
    h = rng.random(40)
    g = ag.judge_human_gap(h + 0.1, np.column_stack([h, h, h]), B=500)
    ok &= _close(g["gap"], 0.1) and _close(g["boot_lo"], 0.1) and _close(g["boot_hi"], 0.1)
    return "CI brackets kappa, deterministic; pass rule; gap 0.1 exactly", \
        f"kappa {r['kappa']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]; gap {g['gap']:.4f}", ok


def _c_calibration():
    t1 = [1, 1, 0, 0.5, 1, 0] * 10                                  # credits of P / N rows
    t2 = [1, 1, 0, 1, 1, 0] * 10
    ex = [1, 0, 0, 0.5, 1, 0] * 10
    adj = [None, 1, None, 0.5, None, None] * 10
    ann = {"T1": t1, "T2": t2, "E": ex}
    hr = ag.human_reference(ann, adj)
    answers = [f"a{i // 3:02d}" for i in range(60)]
    human = {f"a{a:02d}": [1.0, 0.5, 0.0, 0.25, 0.75][a % 5] for a in range(20)}
    c = ag.judge_calibration(hr["reference"], ann, adj, answers, human, human, B=300)
    ok = hr["n_adjudicated"] == 20 and hr["reference"][:6] == [1, 1, 0, 0.5, 1, 0]
    ok &= c["gate_lines"]["kappa"] == 1.0 and c["gate_bands"]["kappa"] == 1.0 and c["passes"] \
        and c["complete"] and c["gate_lines"]["n_clusters"] == 20 and c["n_answers"] == 20 \
        and _close(c["judge_vs_annotator"]["T1"], 1.0) and c["human_human_mean"] < 1.0
    zeroed = {a: (0.0 if s == 1.0 else s) for a, s in human.items()}   # false D flags: 1 -> 0
    z = ag.judge_calibration(hr["reference"], ann, adj, answers, zeroed, human, B=300)
    ok &= z["passes_lines"] and not z["passes_bands"] and not z["passes"]
    only_lines = ag.judge_calibration(hr["reference"], ann, adj, answers, B=50)
    ok &= only_lines["passes"] is None and not only_lines["complete"]
    miss = _raises(ag.human_reference, {"T1": t1, "T2": t2}, None)
    ok &= miss is not None and "adjudicated" in miss
    return "reference = unanimous or adjudicated; both gates pass; zeroed answers fail gate 2", \
        f"lines {c['gate_lines']['kappa']}, bands {c['gate_bands']['kappa']}, passes " \
        f"{c['passes']}; zeroing judge: bands kappa {z['gate_bands']['kappa']:.3f}", ok


def _c_bands_presence():
    bands = [ag.score_band(s) for s in (0.0, 1e-12, 0.2, 0.49999999999999994, 0.5, 0.99, 1.0)]
    pa = ag.presence_agreement(["present", "absent", "absent", "present"],
                               ["present", "absent", "present", "present"])
    rare = ag.presence_agreement(["absent"] * 5, ["absent"] * 5)
    ok = bands == [0, 0, 1, 2, 2, 2, 3] and _raises(ag.score_band, 1.5) is not None \
        and _close(pa["percent_agreement"], 0.75) and _close(pa["kappa"], 0.5) \
        and (pa["n_present_judge"], pa["n_present_reference"]) == (2, 3) \
        and math.isnan(rare["kappa"]) and rare["percent_agreement"] == 1.0
    return "bands 0 | (0, .5) | [.5, 1) | 1; presence kappa 0.5, 75% agreement", \
        f"{bands}; kappa {pa['kappa']:.3f}", ok


def _c_design_effect():
    rng = np.random.default_rng(18)
    g = np.repeat(rng.normal(size=40), 5)                          # clusters of 5 identical values
    cl = np.repeat(np.arange(40), 5)
    deff = fn.design_effect(g, cl)
    ok = _close(deff, 5 * (199 / 200) * (40 / 39), 1e-9) \
        and math.isnan(fn.design_effect(g, None))
    a = {f"q{i:03d}": [1] * 5 if g[i] > 0 else [0] * 5 for i in range(200)}
    b = {f"q{i:03d}": [0] * 5 for i in range(200)}
    r = fn.compare_paired(a, b, {f"q{i:03d}": f"c{cl[i]}" for i in range(200)}, B=200)
    ok &= r["design_effect"] > 1 and _close(r["mde_design_adjusted"]["pd_0.2"],
                                            r["mde"]["pd_0.2"] * math.sqrt(r["design_effect"]))
    return "identical clusters of 5: design effect ~5; adjusted MDE = MDE * sqrt(deff)", \
        f"deff {deff:.3f}; compare_paired deff {r['design_effect']:.2f}", ok


def _c_proposals():
    import importlib

    bad = []
    for p in list(pr.PROPOSALS) + list(pr.APPROVED):
        for ref in p["where"]:
            mod, _, name = ref.rpartition(".")      # "functions.x" in grcbench.stats, "tier_e.bench.Y" outside
            pkg = f"grcbench.{mod}" if "." in mod else f"grcbench.stats.{mod}"
            if not hasattr(importlib.import_module(pkg), name):
                bad.append(ref)
    ids = [p["id"] for p in pr.PROPOSALS]
    ok = not bad and len(ids) == len(set(ids)) \
        and all(p["kind"] in ("deviation", "extension") for p in pr.PROPOSALS) \
        and pr.get("H1c-test")["kind"] == "extension" and "## Analysis proposals" in pr.as_markdown() \
        and '"proposals"' in pr.as_json()
    return "every PROPOSAL names code that exists; ids unique; kinds valid", \
        f"{len(ids)} proposals, {sum(p['kind'] == 'deviation' for p in pr.PROPOSALS)} deviation(s)" \
        f"{'; missing ' + ', '.join(bad) if bad else ''}", ok


def _c_h4a():
    k_full = [0.8, 0.6, 0.7, 0.9] * 25
    k_comp = [x - 0.05 for x in k_full]
    a_full = [1.0, 0.75, 1.0, 0.5, 1.0] * 6
    same = fn.compression_cost_difference(a_full, [x - 0.05 for x in a_full], k_full, k_comp, B=500)
    worse = fn.compression_cost_difference(a_full, [x - 0.4 for x in a_full], k_full, k_comp,
                                          a_clusters=[i // 3 for i in range(30)], B=500)
    ok = abs(same["difference"]) < 1e-12 and not same["holds"] and same["lo"] <= 0 <= same["hi"]         and abs(worse["difference"] - 0.35) < 1e-12 and worse["holds"] and worse["n_clusters"]["A"] == 10
    return "equal drops: no; Tier-A drop 35 points larger: holds",         f"same {same['difference']:.3f} [{same['lo']:.3f}, {same['hi']:.3f}]; worse {worse['difference']:.3f} "         f"[{worse['lo']:.3f}, {worse['hi']:.3f}]", ok


def _c_by_group():
    groups = ["calc"] * 30 + ["factual"] * 10
    r = ag.by_group(groups, ag.cohen_kappa, [1, 0] * 20, [1, 0] * 20)
    ok = r["all"]["result"] == 1.0 and r["calc"]["n"] == 30 and not r["calc"]["small"] \
        and r["factual"]["small"] and r["factual"]["result"] is None
    rs = ag.by_group(groups, ag.cohen_kappa, [1, 0] * 20, [1, 0] * 20, include_small=True)
    ok &= rs["factual"]["result"] == 1.0
    return "per-type results; factual (10 items) small and withheld", \
        f"all {r['all']['result']}, calc n {r['calc']['n']}, factual {r['factual']['result']}", ok


CHECKS: list[Check] = [
    ("pass@k", "§9 estimates", _c_pass_at_k),
    ("pass^k", "§9 estimates", _c_pass_hat_k),
    ("per-item scores", "§9 estimates, §4.2", _c_item_scores),
    ("Exec@1 / Func@1 / funnel", "§4.3 metrics", _c_exec_func),
    ("cluster-robust SE", "§9 uncertainty", _c_cluster_se),
    ("cluster bootstrap CI", "§9 uncertainty", _c_bootstrap),
    ("paired difference CI", "§9 effect sizes", _c_paired),
    ("McNemar exact + majority", "§9 comparisons", _c_mcnemar),
    ("clustered McNemar/Wilcoxon", "§9 comparisons", _c_clustered_tests),
    ("Wilcoxon signed-rank", "§9 comparisons", _c_wilcoxon),
    ("Holm-Bonferroni", "§9 comparisons", _c_holm),
    ("Cohen's h, matched OR", "§9 effect sizes", _c_effects),
    ("compare_paired pass/fail", "§9 comparisons", _c_compare_pass_fail),
    ("compare_paired scores", "§9 comparisons", _c_compare_score),
    ("MDE Table 8", "§9 Table 8", _c_mde),
    ("design-effect MDE", "§9 Table 8", _c_design_effect),
    ("bootstrap rank stability", "§9 reliability", _c_rank_stability),
    ("split-half / random-half", "§9 reliability", _c_split_half),
    ("split-half, mixed-n panel", "§9 reliability", _c_split_half_panel),
    ("harness determinism", "§5.4, §9", _c_determinism),
    ("vectorised pair Wilcoxon", "§9 comparisons", _c_wilcoxon_rows),
    ("H1c size-matched", "§3 RQ1c", _c_h1c),
    ("H1c, mixed-n panel", "§3 RQ1c, §7.3", _c_h1c_mixed_n),
    ("H1a / H1c decisions", "§3 Table 3", _c_h1_decisions),
    ("seed spread, correlation CI", "§9 seeds/validity", _c_seeds_validity),
    ("difficulty, point-biserial", "§5.3, §9 items", _c_item_stats),
    ("Rasch (exploratory)", "§9 items", _c_rasch),
    ("Cohen's kappa", "§8.1 agreement", _c_kappa),
    ("weighted kappa", "§8.1 agreement", _c_weighted_kappa),
    ("Scott's pi, alpha (2 coders)", "§8.1 agreement", _c_scott_alpha_binary),
    ("Krippendorff's alpha", "§8.1 agreement", _c_kripp),
    ("kappa CI, pass rule, gap", "§8.1 agreement", _c_kappa_ci_gap),
    ("judge calibration (3 humans)", "§8.1 agreement", _c_calibration),
    ("score bands, presence rows", "§8.1 agreement", _c_bands_presence),
    ("per-task-type breakdown", "§8.1, §4.2", _c_by_group),
    ("PROPOSALS registry", "§9 pre-registration", _c_proposals),
    ("H4a compression cost", "§3 H4a", _c_h4a),
]


def selfcheck(verbose: bool = True) -> list[dict]:
    rows = []
    for name, section, check in CHECKS:
        try:
            expected, got, ok = check()
        except Exception as exc:  # a crash is a failed check, reported with its message
            expected, got, ok = "-", f"{type(exc).__name__}: {exc}", False
        rows.append({"check": name, "section": section, "expected": expected, "got": got,
                     "ok": bool(ok)})
    if verbose:
        print(f"{'check':30s} {'section':18s} {'result':6s} expected | got")
        for r in rows:
            print(f"{r['check']:30s} {r['section']:18s} {'PASS' if r['ok'] else 'FAIL':6s} "
                  f"{r['expected']} | {r['got']}")
        n_ok = sum(r["ok"] for r in rows)
        print(f"\n{n_ok}/{len(rows)} checks passed")
    return rows


# ---------------------------------------------------------------- H1c operating characteristics ---

H1C_SIM = {"n_models": 21, "k_items": 285, "f_items": 150, "n": 5, "difficulty_sd": 1.2}


def _h1c_sim_one(args: tuple) -> tuple[bool, bool, bool]:
    s, ratio, spread, B, repeats, k_test, f_test, seed = args
    rng = np.random.default_rng([seed, s])
    theta = np.linspace(-spread, spread, H1C_SIM["n_models"])

    def tier(n_items: int, a: float) -> np.ndarray:
        b = rng.normal(0.0, H1C_SIM["difficulty_sd"], n_items)
        p = 1.0 / (1.0 + np.exp(-a * (theta[:, None] - b[None, :])))
        return (rng.random((theta.size, n_items, H1C_SIM["n"])) < p[:, :, None]).astype(float)

    k, f = tier(H1C_SIM["k_items"], 1.0), tier(H1C_SIM["f_items"], ratio)
    r = fn.size_matched_separation(k, f, k_test, f_test, repeats=repeats, B=B, seed=seed + s)
    d = fn.h1c_decision(r)
    return d["spread"]["holds"], d["pairs"]["holds"], d["holds"]


def simulate_h1c(datasets: int = 400, ratio: float = 1.0, spread: float = 1.5,
                 B: int = fn.H1C_BOOT_B, repeats: int = 20, k_test: str = fn.H1C_TEST,
                 f_test: str = fn.H1C_TEST, seed: int = fn.SEED, workers: int | None = None) -> dict:
    """How often h1c_decision's parts hold on simulated panels (PROPOSAL H1c-decision evidence).

    Each dataset is a Rasch panel shaped like the run list (H1C_SIM: 21 models with abilities evenly
    spread over +-spread logits, item difficulties N(0, 1.2), n = 5 samples, 285 Tier-K and 150
    Tier-F items). ratio = 1 is the null: both tiers come from the same process, so neither
    separates more and a valid rule holds in at most alpha/2 = 2.5% of datasets. ratio > 1: Tier F's
    items discriminate `ratio` times as strongly (the alternative; the share is the power).
    Dataset s draws from default_rng([seed, s]), so the result does not depend on `workers`.
    `repeats` only changes the reported point estimate, not the decision.
    """
    from multiprocessing import Pool

    jobs = [(s, ratio, spread, B, repeats, k_test, f_test, seed) for s in range(datasets)]
    if workers == 1:
        res = [_h1c_sim_one(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            res = pool.map(_h1c_sim_one, jobs)
    a = np.array(res, dtype=float)
    rates = a.mean(axis=0)
    mcse = np.sqrt(rates * (1 - rates) / datasets)
    return {"datasets": datasets, "ratio": ratio, "spread": spread, "B": B, "k_test": k_test,
            "f_test": f_test, "seed": seed, **H1C_SIM,
            **{f"share_{p}_holds": float(r) for p, r in zip(("spread", "pairs", "h1c"), rates)},
            **{f"mcse_{p}": float(m) for p, m in zip(("spread", "pairs", "h1c"), mcse)}}


def print_table8(alpha: float, power: float) -> bool:
    rows = md.table8(alpha, power)
    print(f"Table 8. Minimum detectable effect ({power:.0%} power, {alpha:.0%} significance, paired)")
    print(f"{'tier':5s} {'items':>5s} {'p_d=0.2':>8s} {'p_d=0.3':>8s} {'points':>7s} {'plan':>6s} match")
    for r in rows:
        print(f"{r['tier']:5s} {r['n_items']:5d} {100 * r['mde_pd_0_2']:8.2f} "
              f"{100 * r['mde_pd_0_3']:8.2f} {r['points']:>7s} {r['printed']:>6s} "
              f"{'yes' if r['matches_plan'] else 'NO'}")
    return all(r["matches_plan"] for r in rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gb stats", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck", help="run every analysis function on synthetic data with known answers")
    p = sub.add_parser("mde", help="print Table 8, or the MDE for --n and --pd")
    p.add_argument("--n", type=int, help="number of paired items")
    p.add_argument("--pd", type=float, nargs="+", help="share(s) of discordant items")
    p.add_argument("--alpha", type=float, default=md.ALPHA)
    p.add_argument("--power", type=float, default=md.POWER)
    q = sub.add_parser("proposals", help="print the analysis PROPOSALS for the G2 pre-registration")
    q.add_argument("--format", choices=("text", "md", "json"), default="text")
    s = sub.add_parser("h1c-sim", help="false-positive rate / power of the H1c decision rule")
    s.add_argument("--ratio", type=float, default=1.0,
                   help="Tier F's item discrimination relative to Tier K (1 = the null)")
    s.add_argument("--spread", type=float, default=1.5, help="model abilities span +-spread logits")
    s.add_argument("--datasets", type=int, default=400)
    s.add_argument("--B", type=int, default=fn.H1C_BOOT_B)
    s.add_argument("--k-test", default=fn.H1C_TEST, choices=fn.H1C_TESTS)
    s.add_argument("--f-test", default=fn.H1C_TEST, choices=fn.H1C_TESTS)
    s.add_argument("--workers", type=int, default=None, help="processes (default: all CPUs)")
    args = ap.parse_args(argv)
    if args.cmd == "selfcheck":
        return 0 if all(r["ok"] for r in selfcheck()) else 1
    if args.cmd == "proposals":
        if args.format == "md":
            print(pr.as_markdown())
        elif args.format == "json":
            print(pr.as_json())
        else:
            for a in pr.APPROVED:
                print(f"approved   {a['id']:18s} {a['definition']}")
            for p_ in pr.PROPOSALS:
                print(f"{p_['kind']:10s} {p_['id']:18s} {p_['topic']}  [{', '.join(p_['where'])}]")
        return 0
    if args.cmd == "h1c-sim":
        r = simulate_h1c(args.datasets, args.ratio, args.spread, args.B, k_test=args.k_test,
                         f_test=args.f_test, workers=args.workers)
        print(f"H1c rule on {r['datasets']} simulated panels ({r['n_models']} models, K "
              f"{r['k_items']} / F {r['f_items']} items, n = {r['n']}, abilities +-{r['spread']}, "
              f"F discrimination x{r['ratio']}, tests K {r['k_test']} / F {r['f_test']}, B {r['B']}):")
        for part in ("spread", "pairs", "h1c"):
            print(f"  {part:7s} holds in {r[f'share_{part}_holds']:.4f} "
                  f"(Monte Carlo SE {r[f'mcse_{part}']:.4f})")
        return 0
    if args.n is None and args.pd is None:
        default = args.alpha == md.ALPHA and args.power == md.POWER
        ok = print_table8(args.alpha, args.power)
        return 0 if ok or not default else 1
    if args.n is None:
        ap.error("--n is required with --pd")
    for pd in args.pd or list(md.P_DISCORDANT):
        v = md.mde(args.n, pd, args.alpha, args.power)
        print(f"n = {args.n}, p_d = {pd}: MDE = {100 * v:.2f} points "
              f"(alpha {args.alpha}, power {args.power})")
    return 0

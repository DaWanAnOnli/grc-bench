"""Agreement between the judge and people, and between annotators (eval plan §8.1, §9).

Implements eval plan §8.1 "Calibration": "The judge passes if Cohen's κ is at least 0.7 and the
lower end of its 95% interval is at least 0.6. We also report Scott's π and the average gap between
the judge's and the humans' scores, because a judge can agree often and still be off on scores
[73]. We report agreement for each task type, not only overall." Also RQ1a / H1a (κ ≥ 0.7, lower
95% bound ≥ 0.6), proposal §8 "Judge validity: κ/α against humans", and eval plan §9 "Error
analysis ... We report their agreement (κ)".

    cohen_kappa(r1, r2, categories=None, weights=None)   two raters; weights None | "linear" |
                                                         "quadratic" (ordered categories)
    cohen_kappa_ci(r1, r2, ...)                          κ with a percentile bootstrap interval
                                                         (items, or clusters of items, resampled)
    judge_passes(kappa_ci)                               the §8.1 rule: κ >= 0.7 and lower >= 0.6
    scott_pi(r1, r2)                                     two raters, pooled marginals
    krippendorff_alpha(data, level)                      any number of annotators, missing values;
                                                         level nominal | ordinal | interval | ratio
    human_reference(annotators, adjudicated)             the three annotators reduced to one
                                                         reference label per row (adjudication)
    score_band(score)                                    an answer score's band 0-3
    judge_calibration(judge, annotators, adjudicated,    the §8.1 rule applied twice (rubric lines,
                      answers, judge_scores,             answer score bands), plus π, judge-
                      human_scores)                      annotator and human-human agreement
    presence_agreement(judge, reference)                 κ, % agreement and "present" counts on
                                                         the error and H rows (reported only)
    judge_human_gap(judge, human, clusters)              mean(judge - human) with CIs
    by_group(groups, fn, *columns, min_items)            per-task-type breakdown of any of these

The judge is calibrated against THREE annotators (§8.1: "150 answers labelled by three annotators,
one of them a Qoherent expert"), but Cohen's κ compares two raters. The reduction is the one in
docs/guidelines/judge_calibration.md ("Adjudication"): the human reference of a rubric line is the
label all three gave, or else the adjudicator's label. §8.1 does not say what κ is computed on, so
the guideline's proposal for G2 ("Analysis and pass rule"; PROPOSAL judge-gate in stats.proposals)
applies the rule TWICE and the judge passes only if both hold:
  1. rubric lines: unweighted Cohen's κ between the judge's and the reference credit (1, 0.5, 0) on
     every P and N row, with a bootstrap interval that resamples whole answers ("gate_lines");
  2. answer scores: Cohen's κ with linear weights between the judge's and the human score band of
     each answer (score_band), each answer one item ("gate_bands").
judge_calibration computes both, and reports the judge's κ against each annotator and the
human-human agreement beside them for information.

Labels may be any hashable values (numbers or strings); a missing rating is None or NaN.
Krippendorff's alpha uses the coincidence-matrix method of Krippendorff (2011), "Computing
Krippendorff's Alpha-Reliability".
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations

import numpy as np

from .functions import (ALPHA, BOOTSTRAP_B, SEED, _percentile_ci, bootstrap_weights,
                        cluster_codes, paired_difference)

KAPPA_MIN = 0.7          # eval plan §8.1 / H1a
KAPPA_CI_LOW_MIN = 0.6   # lower end of the 95% interval
MIN_GROUP_ITEMS = 20     # eval plan §4.2: no group with fewer than 20 items is reported on its own
SCORE_BANDS = (0, 1, 2, 3)   # judge_calibration.md: 0 | (0, 0.5) | [0.5, 1) | 1
BAND_TOL = 1e-9              # scores within 1e-9 of 0, 0.5 or 1 count as that value


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or \
        (isinstance(v, np.floating) and bool(np.isnan(v)))


def _sort_key(v):
    if isinstance(v, (bool, np.bool_)):
        return (0, float(v), "")
    if isinstance(v, (int, float, np.integer, np.floating)):
        return (0, float(v), "")
    return (1, 0.0, str(v))


def _pairs(r1: Sequence, r2: Sequence) -> tuple[list, list]:
    a, b = list(r1), list(r2)
    if len(a) != len(b) or not a:
        raise ValueError("two non-empty rating sequences of the same length are needed")
    if any(_missing(v) for v in a + b):
        raise ValueError("Cohen's kappa and Scott's pi need a rating from both raters on every item")
    return a, b


def _check_order(a: list, b: list, categories: Sequence | None, weights: str | None) -> None:
    """Weights need ordered categories: text ratings carry no order, so they need `categories`
    (round-2 finding C5: "met" < "not_met" < "partially_met" alphabetically)."""
    if weights is not None and categories is None and not all(
            isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
            for v in a + b):
        raise ValueError("weighted kappa needs ordered categories: pass `categories` for non-numeric ratings")


def _categories(a: list, b: list, categories: Sequence | None) -> list:
    if categories is None:
        return sorted(set(a) | set(b), key=_sort_key)
    cats = list(categories)
    unknown = (set(a) | set(b)) - set(cats)
    if unknown:
        raise ValueError(f"ratings outside the given categories: {sorted(unknown, key=_sort_key)}")
    return cats


def _disagreement_weights(k: int, weights: str | None) -> np.ndarray:
    i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    if weights is None:
        return (i != j).astype(float)
    if k < 2:
        return np.zeros((k, k))
    if weights == "linear":
        return np.abs(i - j) / (k - 1)
    if weights == "quadratic":
        return ((i - j) / (k - 1)) ** 2
    raise ValueError("weights must be None, 'linear' or 'quadratic'")


def confusion(r1: Sequence, r2: Sequence, categories: Sequence | None = None
              ) -> tuple[list, np.ndarray]:
    """Categories (sorted, or as given) and the K x K count table (rows rater 1, columns rater 2)."""
    a, b = _pairs(r1, r2)
    cats = _categories(a, b, categories)
    idx = {c: n for n, c in enumerate(cats)}
    table = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        table[idx[x], idx[y]] += 1
    return cats, table


def _kappa_tables(tables: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted kappa 1 - sum(w p_obs) / sum(w p_exp) for a stack of K x K tables (NaN if the
    expected disagreement is 0, e.g. both raters used one and the same category)."""
    n = tables.sum(axis=(1, 2))
    p = tables / n[:, None, None]
    e = p.sum(axis=2)[:, :, None] * p.sum(axis=1)[:, None, :]
    do = (w * p).sum(axis=(1, 2))
    de = (w * e).sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(de > 1e-15, 1.0 - do / de, np.nan)


def cohen_kappa(r1: Sequence, r2: Sequence, categories: Sequence | None = None,
                weights: str | None = None) -> float:
    """Cohen's kappa (Po - Pe) / (1 - Pe); with weights, 1 - sum(w Po) / sum(w Pe) over ordered
    categories (linear |i-j|/(K-1) or quadratic ((i-j)/(K-1))^2 disagreement weights). Weights with
    non-numeric ratings need `categories` (ValueError otherwise)."""
    a, b = _pairs(r1, r2)
    _check_order(a, b, categories, weights)
    cats, table = confusion(a, b, categories)
    return float(_kappa_tables(table[None], _disagreement_weights(len(cats), weights))[0])


def cohen_kappa_ci(r1: Sequence, r2: Sequence, categories: Sequence | None = None,
                   weights: str | None = None, clusters: Sequence | None = None,
                   B: int = BOOTSTRAP_B, seed: int = SEED, alpha: float = ALPHA) -> dict:
    """Cohen's kappa with a percentile bootstrap (1 - alpha) interval: items (or, with `clusters`,
    clusters of items) are resampled with replacement; resamples whose kappa is undefined are
    dropped and counted. Fewer than 2 clusters (or items) give no interval (NaN bounds)."""
    a, b = _pairs(r1, r2)
    _check_order(a, b, categories, weights)
    cats, table = confusion(a, b, categories)
    k = len(cats)
    idx = {c: n for n, c in enumerate(cats)}
    onehot = np.zeros((len(a), k * k))
    onehot[np.arange(len(a)), [idx[x] * k + idx[y] for x, y in zip(a, b)]] = 1.0
    wts = bootstrap_weights(len(a), clusters, B, seed)
    w = _disagreement_weights(k, weights)
    boot = _kappa_tables((wts @ onehot).reshape(B, k, k), w)
    ok = boot[np.isfinite(boot)]
    _, g = cluster_codes(clusters, len(a))
    lo, hi = _percentile_ci(ok, alpha) if ok.size and g >= 2 else (float("nan"), float("nan"))
    return {"kappa": float(_kappa_tables(table[None], w)[0]), "lo": lo, "hi": hi,
            "B": B, "seed": seed, "alpha": alpha, "n_items": len(a), "n_clusters": g,
            "n_undefined_resamples": int(boot.size - ok.size), "weights": weights,
            "categories": cats}


def judge_passes(kappa_ci: dict) -> bool:
    """Eval plan §8.1: the judge passes if kappa >= 0.7 and the lower end of its 95% interval >= 0.6.

    kappa_ci is one gate statistic, a cohen_kappa_ci result: judge_calibration(...)["gate_lines"]
    or ["gate_bands"]. The judge passes calibration only if BOTH gates pass
    (judge_calibration(...)["passes"])."""
    if kappa_ci.get("alpha", ALPHA) != 0.05:
        raise ValueError("the §8.1 rule is defined on the 95% interval")
    return bool(kappa_ci["kappa"] >= KAPPA_MIN and kappa_ci["lo"] >= KAPPA_CI_LOW_MIN)


def scott_pi(r1: Sequence, r2: Sequence, categories: Sequence | None = None) -> float:
    """Scott's pi (Po - Pe) / (1 - Pe) with Pe = sum_k ((n1_k + n2_k) / 2N)^2 (pooled marginals)."""
    _, table = confusion(r1, r2, categories)
    n = table.sum()
    po = np.trace(table) / n
    pooled = (table.sum(axis=1) + table.sum(axis=0)) / (2 * n)
    pe = float(np.sum(pooled ** 2))
    return float((po - pe) / (1 - pe)) if pe < 1 - 1e-15 else float("nan")


def _delta2(values: np.ndarray, n_c: np.ndarray, level: str) -> np.ndarray:
    c, k = np.meshgrid(values, values, indexing="ij")
    if level == "nominal":
        return (c != k).astype(float)
    if level == "interval":
        return (c - k) ** 2
    if level == "ratio":
        s = c + k
        return np.divide((c - k) ** 2, s ** 2, out=np.zeros_like(s), where=s != 0)
    if level == "ordinal":
        cum = np.concatenate([[0.0], np.cumsum(n_c)])      # cum[g] = sum of n over ranks < g
        g, h = np.meshgrid(np.arange(len(values)), np.arange(len(values)), indexing="ij")
        lo, hi = np.minimum(g, h), np.maximum(g, h)
        between = cum[hi + 1] - cum[lo]                     # sum_{r=lo..hi} n_r
        d = between - (n_c[g] + n_c[h]) / 2.0
        return np.where(g == h, 0.0, d ** 2)
    raise ValueError("level must be nominal, ordinal, interval or ratio")


def krippendorff_alpha(data: Sequence[Sequence], level: str = "nominal") -> float:
    """Krippendorff's alpha = 1 - D_o / D_e for annotators x units data (None / NaN = missing).

    Coincidences: every unit with m >= 2 values adds n_uc n_uk / (m - 1) to o_ck (c != k) and
    n_uc (n_uc - 1) / (m - 1) to o_cc; then alpha = 1 - (n - 1) sum o_ck d2_ck / sum n_c n_k d2_ck.
    Units with fewer than 2 values are not pairable and are ignored. NaN if only one value occurs.
    """
    rows = [list(r) for r in data]
    if not rows or len({len(r) for r in rows}) != 1:
        raise ValueError("data must be annotators x units with the same number of units per row")
    units = [[v for v in col if not _missing(v)] for col in zip(*rows)]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    values = sorted({v for u in units for v in u}, key=_sort_key)
    if level != "nominal" and any(_sort_key(v)[0] != 0 for v in values):
        raise ValueError(f"level {level!r} needs numeric ratings")
    idx = {v: i for i, v in enumerate(values)}
    k = len(values)
    o = np.zeros((k, k))
    for u in units:
        n_u = np.bincount([idx[v] for v in u], minlength=k).astype(float)
        o += (np.outer(n_u, n_u) - np.diag(n_u)) / (len(u) - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    numeric = np.array([float(v) for v in values]) if level != "nominal" else np.arange(k, dtype=float)
    d2 = _delta2(numeric, n_c, level)
    de = float(np.sum(np.outer(n_c, n_c) * d2))
    if de <= 0:
        return float("nan")
    return float(1.0 - (n - 1) * float(np.sum(o * d2)) / de)


def human_reference(annotators: Mapping[str, Sequence], adjudicated: Sequence | None = None
                    ) -> dict:
    """One human reference label per row from several annotators (judge_calibration.md,
    "Adjudication"): the label every annotator gave where they all agree, otherwise the
    adjudicated label.

    annotators: name -> one label per row (every annotator labels every row, §8.1).
    adjudicated: the adjudicated_label column (None / NaN allowed on unanimous rows). A row whose
    labels differ and that has no adjudicated label, or an adjudicated label that contradicts a
    unanimous one, is an error (listed by row index).
    """
    names = sorted(annotators)
    if len(names) < 2:
        raise ValueError("need at least 2 annotators")
    cols = [list(annotators[n]) for n in names]
    n_rows = len(cols[0])
    if n_rows == 0 or any(len(c) != n_rows for c in cols):
        raise ValueError("every annotator needs one label per row")
    if any(_missing(v) for c in cols for v in c):
        raise ValueError("every annotator must label every row")
    adj = [None] * n_rows if adjudicated is None else list(adjudicated)
    if len(adj) != n_rows:
        raise ValueError("adjudicated needs one entry per row")
    ref, missing, conflict, n_adj = [], [], [], 0
    for r in range(n_rows):
        labels = {c[r] for c in cols}
        if len(labels) == 1:
            (lab,) = labels
            if not _missing(adj[r]) and adj[r] != lab:
                conflict.append(r)
            ref.append(lab)
        elif _missing(adj[r]):
            missing.append(r)
            ref.append(None)
        else:
            ref.append(adj[r])
            n_adj += 1
    if missing or conflict:
        raise ValueError(f"rows needing an adjudicated label: {missing[:20]}; adjudicated labels "
                         f"contradicting a unanimous one: {conflict[:20]}")
    return {"reference": ref, "annotators": names, "n_rows": n_rows,
            "n_unanimous": n_rows - n_adj, "n_adjudicated": n_adj}


def score_band(score: float) -> int:
    """The score band of an answer (judge_calibration.md, pass-rule condition 2): 0 for a score of
    0; 1 above 0 and below 0.5; 2 from 0.5 to below 1; 3 for a score of 1. Scores within BAND_TOL
    of 0, 0.5 or 1 count as that value (a weighted mean can land a rounding error off)."""
    s = float(score)
    if not math.isfinite(s) or s < -BAND_TOL or s > 1.0 + BAND_TOL:
        raise ValueError(f"an answer score must lie in [0, 1], not {score!r}")
    if s <= BAND_TOL:
        return 0
    if s >= 1.0 - BAND_TOL:
        return 3
    return 1 if s < 0.5 - BAND_TOL else 2


def _answer_bands(judge_scores: Mapping[str, float], human_scores: Mapping[str, float],
                  answers: Sequence) -> tuple[list[str], list[int], list[int]]:
    if set(judge_scores) != set(human_scores):
        raise ValueError("judge_scores and human_scores need the same answers")
    missing = sorted(set(map(str, answers)) - set(map(str, judge_scores)))
    if missing:
        raise ValueError(f"answers with rubric rows but no score: {missing[:10]}")
    ids = sorted(judge_scores, key=str)
    return ids, [score_band(judge_scores[a]) for a in ids], [score_band(human_scores[a]) for a in ids]


def judge_calibration(judge: Sequence, annotators: Mapping[str, Sequence],
                      adjudicated: Sequence | None, answers: Sequence,
                      judge_scores: Mapping[str, float] | None = None,
                      human_scores: Mapping[str, float] | None = None,
                      categories: Sequence | None = None, B: int = BOOTSTRAP_B, seed: int = SEED,
                      alpha: float = ALPHA) -> dict:
    """The eval plan §8.1 calibration statistics and the pass rule of judge_calibration.md.

    Rows (condition 1): one per rubric line that enters the line gate, i.e. every P row and every N
    row the judge reads, labelled with its credit 1, 0.5 or 0 (the D, DU and H rows are reported
    with presence_agreement, by kind, and are not passed here). judge: the judge's credit per row;
    annotators: name -> credits (three in §8.1); adjudicated: the adjudicated_label column;
    answers: the answer id of each row (the bootstrap cluster, since the lines of one answer are not
    independent). A reply that stays malformed after the retries is entered by the caller as a
    disagreement on every line and on its band (judge_calibration.md, "The judge's run").
    Answers (condition 2): judge_scores / human_scores map answer id -> the answer's score in [0, 1]
    (explain_score on the judge's verdict and on the adjudicated labels); every answer with rows
    needs one.
      gate_lines  unweighted Cohen's kappa, judge vs the adjudicated human reference
                  (human_reference), percentile bootstrap resampling whole answers;
      gate_bands  Cohen's kappa with linear weights on the score bands (score_band, categories
                  0-3), judge vs human, each answer one item;
      passes      both gates pass judge_passes (None, with "complete" False, when the answer scores
                  are not given: the line gate alone never passes the judge);
      also        Scott's pi on the line pairs (§8.1); the judge's kappa against each annotator and
                  their mean; human-human agreement on the independent labels before adjudication:
                  pairwise Cohen's kappa, their mean, and Krippendorff's alpha (nominal) -- the ceiling
                  the judge can reasonably reach (judge_calibration.md, "Also reported").
    """
    hr = human_reference(annotators, adjudicated)
    ref = hr["reference"]
    j = list(judge)
    answers = list(answers)
    if len(j) != hr["n_rows"] or len(answers) != hr["n_rows"]:
        raise ValueError("judge, annotators and answers need one entry per row")
    if (judge_scores is None) != (human_scores is None):
        raise ValueError("pass both judge_scores and human_scores, or neither")
    gate_lines = cohen_kappa_ci(j, ref, categories, clusters=answers, B=B, seed=seed, alpha=alpha)
    gate_bands, n_answers = None, None
    if judge_scores is not None:
        ids, jb, hb = _answer_bands(judge_scores, human_scores, answers)
        gate_bands = cohen_kappa_ci(jb, hb, SCORE_BANDS, weights="linear", B=B, seed=seed,
                                    alpha=alpha)
        n_answers = len(ids)
    names = hr["annotators"]
    per = {n: cohen_kappa(j, annotators[n], categories) for n in names}
    hh = {f"{x}|{y}": cohen_kappa(annotators[x], annotators[y], categories)
          for x, y in combinations(names, 2)}

    def mean(d: dict) -> float:
        v = [x for x in d.values() if math.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    p_lines = judge_passes(gate_lines)
    p_bands = None if gate_bands is None else judge_passes(gate_bands)
    return {"gate_lines": gate_lines, "gate_bands": gate_bands,
            "passes_lines": p_lines, "passes_bands": p_bands,
            "passes": None if p_bands is None else bool(p_lines and p_bands),
            "complete": gate_bands is not None,
            "scott_pi": scott_pi(j, ref, categories),
            "judge_vs_annotator": per, "judge_vs_annotator_mean": mean(per),
            "human_human_pairwise": hh, "human_human_mean": mean(hh),
            "human_krippendorff_alpha": krippendorff_alpha([list(annotators[n]) for n in names]),
            "n_rows": hr["n_rows"], "n_answers": n_answers, "n_unanimous": hr["n_unanimous"],
            "n_adjudicated": hr["n_adjudicated"],
            "statistic": "two gates (judge_calibration.md): unweighted kappa on rubric-line "
                         "credits vs the adjudicated reference, answers resampled; linear-weighted "
                         "kappa on answer score bands, answers resampled"}


def presence_agreement(judge: Sequence, reference: Sequence, present: Sequence = ("present",)
                       ) -> dict:
    """Agreement on the error and H rows (judge_calibration.md, "Also reported"; not a pass
    condition): each label counts as present if it is in `present`, else absent. Returns Cohen's
    kappa (NaN when both sides use one label only, as on rows that are almost always absent),
    percentage agreement, and how many rows each side marked present. Use by_group with the row kind
    for the per-kind figures."""
    a, b = _pairs(judge, reference)
    pres = set(present)
    ja, rb = [x in pres for x in a], [x in pres for x in b]
    return {"kappa": cohen_kappa(ja, rb, categories=[False, True]),
            "percent_agreement": float(np.mean([x == y for x, y in zip(ja, rb)])),
            "n_present_judge": int(sum(ja)), "n_present_reference": int(sum(rb)), "n": len(a)}


def judge_human_gap(judge: Sequence[float], human, clusters: Sequence | None = None,
                    B: int = BOOTSTRAP_B, seed: int = SEED, alpha: float = ALPHA) -> dict:
    """Average gap between the judge's and the humans' scores (eval plan §8.1), judge - human.

    human: one score per answer, or answers x annotators (NaN = not labelled), averaged over the
    annotators who labelled the answer. The calibration's reported gap uses the score computed from
    the ADJUDICATED labels (judge_calibration.md: "A human score is computed for each answer from
    them"), i.e. a 1-D vector; the annotators x answers form is for information. Reports the mean
    gap with the clustered t interval and the cluster-bootstrap interval
    (functions.paired_difference) and the mean absolute gap.
    """
    h = np.asarray(human, dtype=float)
    if h.ndim == 2:
        cnt = (~np.isnan(h)).sum(axis=1)
        if np.any(cnt == 0):
            raise ValueError("every answer needs at least one human score")
        h = np.nansum(h, axis=1) / cnt
    j = np.asarray(judge, dtype=float)
    diff = paired_difference(j, h, clusters, B, seed, alpha)
    return {"gap": diff["diff"], "se": diff["se"], "t_lo": diff["t_lo"], "t_hi": diff["t_hi"],
            "boot_lo": diff["boot_lo"], "boot_hi": diff["boot_hi"],
            "mean_abs_gap": float(np.mean(np.abs(j - h))), "judge_mean": float(j.mean()),
            "human_mean": float(h.mean()), "n_answers": int(j.size),
            "n_clusters": diff["n_clusters"], "B": B, "seed": seed, "alpha": alpha}


def by_group(groups: Sequence[str], fn: Callable, *columns: Sequence,
             min_items: int = MIN_GROUP_ITEMS, include_small: bool = False) -> dict:
    """Apply fn(*columns) overall ("all") and within each group (e.g. task type).

    columns are per-answer sequences aligned with `groups` (one entry per answer; a row of an
    answers x annotators matrix is one entry). Groups with fewer than min_items answers are marked
    "small" and, because eval plan §4.2 reports no group with fewer than 20 items on its own (it is
    merged with its nearest neighbour, decided before the final runs), get no result unless
    include_small=True (diagnostics only). A ValueError inside fn is recorded as "error".
    """
    g = list(groups)
    cols = [list(c) for c in columns]
    if any(len(c) != len(g) for c in cols):
        raise ValueError("every column needs one entry per group label")

    def run(idx: list[int]) -> dict:
        small = len(idx) < min_items
        if small and not include_small:
            return {"n": len(idx), "small": True, "result": None,
                    "note": f"fewer than {min_items} items: not reported on its own (eval plan §4.2)"}
        try:
            return {"n": len(idx), "small": small, "result": fn(*[[c[i] for i in idx] for c in cols])}
        except ValueError as exc:
            return {"n": len(idx), "small": small, "error": str(exc)}

    out = {"all": run(list(range(len(g))))}
    for label in sorted(set(g), key=str):
        out[str(label)] = run([i for i, x in enumerate(g) if x == label])
    return out

"""Item analysis: difficulty, discrimination, review flags and an exploratory Rasch fit.

Implements eval plan §9 "Item analysis (RQ1b, RQ1c). Difficulty and discrimination for every item
[87]. A Rasch model is fitted as an exploratory check only, because a panel of about 21 models
(outside models plus our own configurations) is too small to fit richer models reliably." and §5.3
"Item statistics": difficulty = "the share of models that solve it", discrimination = "whether
models that do well overall also solve it"; "Items that every model solves, that no model solves,
or that strong models fail more often than weak ones are re-checked by a person." Flags only route
an item to a person: §5.3 "Pruning" removes an item only for a reason a person can state, never
only because of its statistics.

Input: a models x items matrix of item scores in [0, 1]; NaN = the model did not run the item.
  pass/fail tiers (F, A)  0/1 (solved = functions.majority_pass of its samples) or per-item pass@1;
  score tiers (K)         the model's mean score on the item (partial credit, eval plan §4.2). The
                          difficulty is then the mean item score over models, i.e. the classical
                          difficulty of a partial-credit item (the share of the maximum score), and
                          "all_solve" / "none_solve" mean full / zero credit for every model.

    matrix_from_samples(results, reduce)  model -> item -> samples  ->  (models, items, matrix)
    difficulty(matrix)                    column means over the models that ran the item
    discrimination(matrix)                point-biserial item-rest correlation over models: the item
                                          score vs the mean of the model's OTHER items (for complete
                                          0/1 data this equals scipy.stats.pointbiserialr against
                                          the rest-sum score; for partial-credit scores it is the
                                          corrected item-total correlation); NaN with < 3 models or
                                          no variance
    item_analysis(matrix, items, models)  per-item records with flags all_solve, none_solve,
                                          negative_discrimination
    rasch_fit(matrix, items, models)      EXPLORATORY joint maximum likelihood Rasch fit
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize
from scipy.special import expit

from .functions import _counts, _unit_scores, majority_pass

FLAGS = ("all_solve", "none_solve", "negative_discrimination")
REDUCTIONS = ("majority", "pass1", "mean")
RASCH_SOLVED_MIN = 0.5   # PROPOSAL (G2): exploratory Rasch on a score tier: solved iff mean >= 0.5


def _matrix(matrix) -> np.ndarray:
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError("matrix must be models x items")
    obs = ~np.isnan(a)
    if np.any(np.isinf(a)) or np.any((a[obs] < 0) | (a[obs] > 1)):
        raise ValueError("item scores must lie in [0, 1] (NaN = not run)")
    return a


def matrix_from_samples(results: Mapping[str, Mapping[str, Sequence[float]]],
                        reduce: str = "majority") -> tuple[list[str], list[str], np.ndarray]:
    """Build the models x items matrix from model -> item -> per-sample outcomes:
      "majority"  pre-registered per-item pass of pass/fail samples (ties fail);
      "pass1"     c / n of pass/fail samples;
      "mean"      the mean sample score, for any samples in [0, 1] (Tier K's partial credit; equals
                  "pass1" on pass/fail samples).
    Models may have different numbers of samples. Items a model did not run are NaN."""
    if reduce not in REDUCTIONS:
        raise ValueError(f"reduce must be one of {REDUCTIONS}")
    models = sorted(results)
    items = sorted({i for m in models for i in results[m]})
    out = np.full((len(models), len(items)), np.nan)
    for r, m in enumerate(models):
        for c, i in enumerate(items):
            if i in results[m]:
                if reduce == "majority":
                    out[r, c] = float(majority_pass(results[m][i]))
                elif reduce == "pass1":
                    n, k = _counts(results[m][i])
                    out[r, c] = k / n
                else:
                    out[r, c] = float(_unit_scores(list(results[m][i]), "samples").mean())
    return models, items, out


def difficulty(matrix) -> np.ndarray:
    """Share of models solving each item (the mean item score over the models that ran it)."""
    a = _matrix(matrix)
    obs = ~np.isnan(a)
    cnt = obs.sum(axis=0)
    tot = np.where(obs, a, 0.0).sum(axis=0)
    return np.divide(tot, cnt, out=np.full(a.shape[1], np.nan), where=cnt > 0)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    return float(np.sum(xc * yc) / np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2)))


def discrimination(matrix) -> np.ndarray:
    """Point-biserial (Pearson) correlation between each item's score and the rest score (mean of
    the model's other observed items), over the models that ran the item."""
    a = _matrix(matrix)
    obs = ~np.isnan(a)
    filled = np.where(obs, a, 0.0)
    row_sum, row_cnt = filled.sum(axis=1), obs.sum(axis=1)
    out = np.full(a.shape[1], np.nan)
    for i in range(a.shape[1]):
        ok = obs[:, i] & (row_cnt > 1)
        x = a[ok, i]
        rest = (row_sum[ok] - x) / (row_cnt[ok] - 1)
        out[i] = _pearson(x, rest)
    return out


def item_analysis(matrix, items: Sequence[str] | None = None,
                  models: Sequence[str] | None = None) -> dict:
    """Difficulty, discrimination and review flags for every item (eval plan §5.3, §9)."""
    a = _matrix(matrix)
    items = [f"item{i}" for i in range(a.shape[1])] if items is None else list(items)
    if len(items) != a.shape[1] or (models is not None and len(models) != a.shape[0]):
        raise ValueError("item or model names do not match the matrix")
    diff, disc = difficulty(a), discrimination(a)
    n_models = (~np.isnan(a)).sum(axis=0)
    rows = []
    for j, item in enumerate(items):
        flags = []
        if n_models[j] and diff[j] == 1.0:
            flags.append("all_solve")
        if n_models[j] and diff[j] == 0.0:
            flags.append("none_solve")
        if np.isfinite(disc[j]) and disc[j] < 0:
            flags.append("negative_discrimination")
        rows.append({"item": item, "difficulty": float(diff[j]), "discrimination": float(disc[j]),
                     "n_models": int(n_models[j]), "flags": flags})
    rows.sort(key=lambda r: r["item"])
    return {"items": rows, "n_items": len(rows), "n_models": int(a.shape[0]),
            "flag_counts": {f: sum(f in r["flags"] for r in rows) for f in FLAGS},
            "flagged": [r["item"] for r in rows if r["flags"]],
            "note": "flags send an item to a person for re-checking; they never remove it (§5.3)"}


def _drop_extremes(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively drop models and items with no observations or an extreme (all 0 / all 1) score,
    whose joint-ML estimates are infinite. Returns boolean keep masks (models, items)."""
    keep_m = np.ones(x.shape[0], bool)
    keep_i = np.ones(x.shape[1], bool)
    while True:
        sub = x[np.ix_(keep_m, keep_i)]
        obs = ~np.isnan(sub)
        s = np.where(obs, sub, 0.0)
        n_m, s_m = obs.sum(1), s.sum(1)
        n_i, s_i = obs.sum(0), s.sum(0)
        bad_m = (n_m == 0) | (s_m == 0) | (s_m == n_m)
        bad_i = (n_i == 0) | (s_i == 0) | (s_i == n_i)
        if not bad_m.any() and not bad_i.any():
            return keep_m, keep_i
        keep_m[np.flatnonzero(keep_m)[bad_m]] = False
        keep_i[np.flatnonzero(keep_i)[bad_i]] = False
        if not keep_m.any() or not keep_i.any():
            return keep_m, keep_i


def mle_exists(matrix) -> bool:
    """Do finite joint maximum-likelihood Rasch estimates exist for this 0/1 matrix (NaN = not run)?

    The Rasch JML is a logistic regression with model and item effects, and its estimates are
    finite exactly when there is no (quasi-)complete separation (Albert and Anderson 1984): no
    direction (d_theta, d_b) with (2x - 1)(d_theta_m - d_b_i) >= 0 on every observed cell and > 0
    on at least one. Checked exactly by a linear programme (scipy.optimize.linprog, HiGHS):
    maximise sum (2x - 1)(d_theta_m - d_b_i) under those constraints in the box [-1, 1]; a positive
    optimum means the likelihood keeps rising along that direction, so some estimates diverge.
    All-0 / all-1 rows and columns are the simplest cases; patterns such as a Guttman block are not
    caught by them.
    """
    a = np.asarray(matrix, dtype=float)
    obs = ~np.isnan(a)
    m_idx, i_idx = np.nonzero(obs)
    if m_idx.size == 0:
        return False
    n_m, n_i = a.shape
    sgn = 2.0 * a[m_idx, i_idx] - 1.0
    rows = np.arange(m_idx.size)
    con = sparse.csr_matrix((np.concatenate([-sgn, sgn]),
                             (np.concatenate([rows, rows]), np.concatenate([m_idx, n_m + i_idx]))),
                            shape=(m_idx.size, n_m + n_i))
    c = np.asarray(con.sum(axis=0)).ravel()               # minimise -sum (2x-1) eta
    res = linprog(c, A_ub=con, b_ub=np.zeros(m_idx.size), bounds=(-1.0, 1.0), method="highs")
    if res.status != 0:
        raise RuntimeError(f"separation check failed: {res.message}")
    return bool(-res.fun <= 1e-7)


def rasch_fit(matrix, items: Sequence[str] | None = None, models: Sequence[str] | None = None,
              max_iter: int = 2000, solved_min: float | None = None) -> dict:
    """EXPLORATORY Rasch model P(model m solves item i) = 1 / (1 + exp(-(theta_m - b_i))), fitted by
    joint maximum likelihood (scipy.optimize L-BFGS-B, analytic gradient). Eval plan §9 uses it as
    an exploratory check only; no item is kept or dropped because of it.

    Needs 0/1 scores (NaN = not run). For a score tier (Tier K's partial credit) pass
    solved_min=RASCH_SOLVED_MIN: a model solves an item when its mean score is at least that value
    (PROPOSAL, exploratory). Models and items with an all-0 or all-1 pattern (infinite estimates)
    are removed first, repeatedly; if the rest is still separated (mle_exists is False), no
    estimates are reported ("mle_exists": False), because they would diverge. Identification: mean
    item difficulty = 0. JMLE item difficulties are known to be spread out by about I/(I-1) (no
    correction applied). Reports abilities, difficulties with model-based standard errors, and item
    infit/outfit mean-squares.
    """
    a = _matrix(matrix)
    obs_all = ~np.isnan(a)
    if solved_min is not None:
        a = np.where(obs_all, (a >= solved_min).astype(float), np.nan)
    if np.any((a[obs_all] != 0) & (a[obs_all] != 1)):
        raise ValueError("the Rasch fit needs 0/1 item scores (for partial credit pass solved_min)")
    items = [f"item{i}" for i in range(a.shape[1])] if items is None else list(items)
    models = [f"model{m}" for m in range(a.shape[0])] if models is None else list(models)
    keep_m, keep_i = _drop_extremes(a)
    dropped = {"dropped_models": [models[i] for i in np.flatnonzero(~keep_m)],
               "dropped_items": [items[i] for i in np.flatnonzero(~keep_i)],
               "solved_min": solved_min, "note": "exploratory only (eval plan §9)"}
    if keep_m.sum() < 2 or keep_i.sum() < 2:
        return {"converged": False, "mle_exists": False,
                "reason": "fewer than 2 models or items without extreme scores", **dropped}
    x = a[np.ix_(keep_m, keep_i)]
    if not mle_exists(x):
        return {"converged": False, "mle_exists": False,
                "reason": "the joint ML estimates do not exist: the remaining data are "
                          "quasi-completely separated, so some estimates diverge", **dropped}
    obs = ~np.isnan(x)
    xf = np.where(obs, x, 0.0)
    n_m, n_i = x.shape

    def nll(params: np.ndarray) -> tuple[float, np.ndarray]:
        eta = params[:n_m, None] - params[None, n_m:]
        ll = np.where(obs, xf * eta - np.logaddexp(0.0, eta), 0.0)
        r = np.where(obs, xf - expit(eta), 0.0)
        return -float(ll.sum()), np.concatenate([-r.sum(axis=1), r.sum(axis=0)])

    def logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 0.02, 0.98)
        return np.log(p / (1 - p))

    x0 = np.concatenate([logit(xf.sum(1) / obs.sum(1)), -logit(xf.sum(0) / obs.sum(0))])
    res = minimize(nll, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "gtol": 1e-8, "ftol": 1e-14})
    shift = res.x[n_m:].mean()
    theta, b = res.x[:n_m] - shift, res.x[n_m:] - shift
    p = expit(theta[:, None] - b[None, :])
    var = np.where(obs, p * (1 - p), 0.0)
    resid2 = np.where(obs, (xf - p) ** 2, 0.0)
    z2 = np.divide(resid2, var, out=np.zeros_like(var), where=var > 0)
    infit = resid2.sum(0) / var.sum(0)
    outfit = z2.sum(0) / obs.sum(0)
    kept_items = [items[i] for i in np.flatnonzero(keep_i)]
    kept_models = [models[i] for i in np.flatnonzero(keep_m)]
    return {"converged": bool(res.success), "mle_exists": True, "n_iter": int(res.nit),
            "log_likelihood": -float(res.fun),
            "max_abs_estimate": float(np.max(np.abs(np.concatenate([theta, b])))),
            "items": {it: {"difficulty": float(b[j]), "se": float(1 / np.sqrt(var[:, j].sum())),
                           "infit": float(infit[j]), "outfit": float(outfit[j])}
                      for j, it in enumerate(kept_items)},
            "models": {m: {"ability": float(theta[r]), "se": float(1 / np.sqrt(var[r].sum()))}
                       for r, m in enumerate(kept_models)},
            **dropped}

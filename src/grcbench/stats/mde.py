"""Minimum detectable effect of a paired comparison (eval plan §9, Table 8).

Eval plan §9: "Table 8 gives the smallest difference between two models that each tier can detect
reliably. It assumes that 20-30% of the items are decided differently by the two models [...] We
print the MDE next to every claim." Table 8: 80% power, 5% significance, paired;
K 285 items -> 7-9 points, F 150 -> 10-13, A 30 -> 23-28.

Formula (paired binary outcomes, normal approximation to McNemar's test; the variance of the mean
paired difference is at most p_d / n when a share p_d of the items is discordant):

    mde(n, p_d) = (z_{1 - alpha/2} + z_{power}) * sqrt(p_d / n)

    mde(n_items, p_discordant, alpha=0.05, power=0.8) -> float (a proportion; x100 = points)
    n_for_mde(target, p_discordant, ...)               -> smallest n with mde(n, p_d) <= target
    table8()                                           -> the rows of Table 8, recomputed
"""

from __future__ import annotations

import math

from scipy.stats import norm

ALPHA = 0.05
POWER = 0.8
P_DISCORDANT = (0.2, 0.3)          # Table 8's assumption: 20-30% of items decided differently

# Table 8 as printed in the eval plan: tier -> (test items in v1.0, MDE low, MDE high) in points.
TABLE8 = {"K": (285, 7, 9), "F": (150, 10, 13), "A": (30, 23, 28)}


def mde(n_items: int, p_discordant: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """Minimum detectable difference in pass rate (a proportion) for a paired two-sided test."""
    if n_items < 1:
        raise ValueError("n_items must be >= 1")
    if not 0.0 < p_discordant <= 1.0:
        raise ValueError("p_discordant must be in (0, 1]")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must be in (0, 1)")
    z = norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)
    return float(z * math.sqrt(p_discordant / n_items))


def n_for_mde(target: float, p_discordant: float, alpha: float = ALPHA, power: float = POWER) -> int:
    """Smallest number of items whose MDE is at most `target` (a proportion)."""
    if not 0.0 < target <= 1.0:
        raise ValueError("target must be in (0, 1]")
    z = norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)
    n = max(1, math.ceil(p_discordant * (z / target) ** 2))
    while n > 1 and mde(n - 1, p_discordant, alpha, power) <= target:   # guard against float edges
        n -= 1
    while mde(n, p_discordant, alpha, power) > target:
        n += 1
    return n


def points(x: float) -> int:
    """A proportion as whole percentage points, rounded half up (Table 8 prints whole points)."""
    return int(math.floor(100.0 * x + 0.5))


def table8(alpha: float = ALPHA, power: float = POWER) -> list[dict]:
    """Recompute Table 8: one row per tier with the MDE at p_d = 0.2 and 0.3 and the printed range."""
    rows = []
    for tier, (n, lo_printed, hi_printed) in TABLE8.items():
        lo = mde(n, P_DISCORDANT[0], alpha, power)
        hi = mde(n, P_DISCORDANT[1], alpha, power)
        rows.append({"tier": tier, "n_items": n, "mde_pd_0_2": lo, "mde_pd_0_3": hi,
                     "points": f"{points(lo)}-{points(hi)}",
                     "printed": f"{lo_printed}-{hi_printed}",
                     "matches_plan": (points(lo), points(hi)) == (lo_printed, hi_printed)})
    return rows

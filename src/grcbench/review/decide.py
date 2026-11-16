"""The decision rules (code, not judgement) and the expert agent's audit sample.

  same verdict from both reviewers            final, decided by R1;R2
  both `approve with edits`                   expert: merges the two lists (contradicting edits are a
                                              disagreement, which the expert decides)
  both `reject` on a rubric sheet             expert: rewrites the rubric (a checker's reject means
                                              "wrong in a way that needs rewriting"), or decides
                                              `underspecified`
  different verdicts, or either `unsure`      expert decides, after reading both reviewers' reasons
  the expert                                  may not answer `unsure`; when the evidence cannot settle
                                              a row it takes the sheet's conservative default

Audit: the expert re-reviews, blind, a seeded sample of each sheet's agreed rows (10 %, at least 5).
An audit disagreement is decided by the expert. If the expert disagrees on more than 5 % of the
sample, it reviews every agreed row of that sheet.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from grcbench.review.sheets import Sheet

AUDIT_SEED = "grcbench-review-2026-11"
AUDIT_SHARE = 0.10
AUDIT_MIN = 5
AUDIT_FULL_REVIEW_ABOVE = 0.05


@dataclass(frozen=True)
class Route:
    to: str                 # "final" or "expert"
    value: str | None       # the agreed value when final
    reason: str             # agreement | disagreement | unsure | edits to merge


def _value(spec: Sheet, out: Mapping[str, Any], who: str) -> str:
    v = (out.get("value") or "").strip()
    if v not in spec.values:
        raise ValueError(f"{spec.name}: {who} wrote {v!r}; allowed: {list(spec.values)}")
    return v


def route(spec: Sheet, r1: Mapping[str, Any], r2: Mapping[str, Any]) -> Route:
    a, b = _value(spec, r1, spec.roles[0]), _value(spec, r2, spec.roles[1])
    if a == "unsure" or b == "unsure":
        return Route("expert", None, "unsure")
    if a != b:
        return Route("expert", None, "disagreement")
    if a == "approve with edits":
        return Route("expert", None, "edits to merge")
    if a == "reject" and spec.kind == "rubric":           # rubric_writing.md: "wrong in a way that needs rewriting"
        return Route("expert", None, "rejected: the expert rewrites the rubric")
    return Route("final", a, "agreement")


def expert_value(spec: Sheet, ex: Mapping[str, Any]) -> str:
    v = (ex.get("value") or "").strip()
    if v not in spec.expert_values:
        raise ValueError(f"{spec.name}: the expert wrote {v!r}; allowed: {list(spec.expert_values)}")
    return v


def _rank(sheet: str, key: Sequence[str]) -> str:
    return hashlib.sha256(f"{AUDIT_SEED}|{sheet}|{'|'.join(key)}".encode("utf-8")).hexdigest()


def audit_sample(sheet: str, agreed: Sequence[Sequence[str]]) -> list[tuple[str, ...]]:
    """The agreed rows the expert re-reviews: the first n by seeded hash, n = max(5, ceil(10 %))."""
    keys = sorted({tuple(k) for k in agreed}, key=lambda k: _rank(sheet, k))
    n = min(len(keys), max(AUDIT_MIN, math.ceil(AUDIT_SHARE * len(keys))))
    return keys[:n]


def audit_outcome(results: Sequence[tuple[Any, str, str]]) -> dict[str, Any]:
    """results: (key, agreed value, expert's value). Full review when the expert disagrees on > 5 %."""
    n = len(results)
    d = sum(1 for _, agreed, ex in results if agreed != ex)
    rate = d / n if n else 0.0
    return {"n": n, "disagreements": d, "rate": rate, "full_review": rate > AUDIT_FULL_REVIEW_ABOVE}

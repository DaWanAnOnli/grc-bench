"""Order-insensitive structural partial credit over blocks, parameters and connections.

Proposal §5.2 / eval plan §4.3 metric [48]: a score in [0, 1] comparing a generated flowgraph with
the task's reference, so that a graph that fails the ladder can still be credited for what it got
right. Three parts, averaged:
  blocks       F1 between the multisets of block types (ids) of the two graphs;
  parameters   for blocks matched one-to-one (same type, Hungarian assignment maximising parameter
               agreement), the share of the reference's explicitly set parameters reproduced;
  connections  F1 between the connection sets after renaming matched blocks to reference names.
Disabled blocks, the options block, notes and GUI coordinates are ignored. Values are compared as
normalised strings, and numerically when both parse as numbers (relative tolerance 1e-6).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

IGNORE_IDS = {"options", "note", "import"}
IGNORE_PARAMS = {"comment", "alias", "affinity", "maxoutbuf", "minoutbuf"}


def _blocks(data: dict) -> list[dict]:
    out = []
    for b in data.get("blocks", []) or []:
        if not isinstance(b, dict) or b.get("id") in IGNORE_IDS:
            continue
        if (b.get("states") or {}).get("state", "enabled") == "disabled":
            continue
        params = {k: v for k, v in (b.get("parameters") or {}).items() if k not in IGNORE_PARAMS}
        out.append({"name": str(b.get("name")), "id": str(b.get("id")), "params": params})
    return out


def _norm(v: Any) -> str:
    return str(v).strip().strip("'\"").replace(" ", "")


def _eq(a: Any, b: Any) -> bool:
    sa, sb = _norm(a), _norm(b)
    if sa == sb:
        return True
    try:
        fa, fb = float(sa), float(sb)
        return abs(fa - fb) <= 1e-6 * max(1.0, abs(fb))
    except ValueError:
        return False


def _param_agreement(model: dict, ref: dict) -> float:
    keys = [k for k in ref["params"]]
    if not keys:
        return 1.0
    return sum(1 for k in keys if k in model["params"] and _eq(model["params"][k], ref["params"][k])) / len(keys)


def score(model_data: dict, ref_data: dict) -> dict[str, float]:
    mb, rb = _blocks(model_data), _blocks(ref_data)
    cm, cr = Counter(b["id"] for b in mb), Counter(b["id"] for b in rb)
    inter = sum((cm & cr).values())
    block_f1 = 2 * inter / (len(mb) + len(rb)) if (mb or rb) else 1.0

    mapping: dict[str, str] = {}
    agreements: list[float] = []
    for bid in sorted(set(cm) & set(cr)):
        ms = [b for b in mb if b["id"] == bid]
        rs = [b for b in rb if b["id"] == bid]
        cost = np.array([[-_param_agreement(m, r) for r in rs] for m in ms])
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            mapping[ms[i]["name"]] = rs[j]["name"]
            agreements.append(-cost[i, j])
    n_ref_matchable = len(rb)
    param_score = (sum(agreements) / n_ref_matchable) if n_ref_matchable else 1.0

    def conns(data: dict, rename: dict[str, str] | None) -> set[tuple]:
        names = {b["name"] for b in _blocks(data)}
        out = set()
        for c in data.get("connections", []) or []:
            if not (isinstance(c, (list, tuple)) and len(c) == 4):
                continue
            s, sp, d, dp = (str(x) for x in c)
            if s not in names or d not in names:
                continue
            if rename is not None:
                s, d = rename.get(s, f"?{s}"), rename.get(d, f"?{d}")
            out.add((s, sp, d, dp))
        return out

    mc, rc = conns(model_data, mapping), conns(ref_data, None)
    conn_f1 = 2 * len(mc & rc) / (len(mc) + len(rc)) if (mc or rc) else 1.0
    total = (block_f1 + param_score + conn_f1) / 3
    return {"blocks": round(block_f1, 6), "parameters": round(param_score, 6),
            "connections": round(conn_f1, 6), "score": round(total, 6)}

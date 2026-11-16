"""Executed comparisons: a blind value against the key, scored by the Tier-K rule cascade.

A recomputed calculation answer or a blind GNU Radio answer agrees with the key when the rules
(`tier_k/rules.py`, the code that scores models) give it full credit against the key's reference.
"""

from __future__ import annotations

from typing import Any, Mapping

from grcbench.tier_k import rules


def _parts(x: Any) -> list[Any]:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _split(text: str) -> list[str]:
    return [p.strip() for p in (text or "").split(";")]


def _reference(value: float, unit: str | None, tol: float | None, tol_db: float | None) -> dict[str, Any]:
    ref: dict[str, Any] = {"value": float(value), "unit": unit or ""}
    if tol_db is not None:
        ref["abs_tolerance"] = float(tol_db)
    elif tol is not None:
        ref["tolerance"] = float(tol)
    return ref


def full_credit(text: str, reference: Mapping[str, Any]) -> bool:
    if not (text or "").strip():
        return False
    return rules.score(text, dict(reference)).get("credit") == 1.0


def calc_status(key: Mapping[str, Any], out: Mapping[str, Any]) -> str:
    """agrees | disagrees | underspecified | no value, for one recomputer's row against the key."""
    value = (out.get("value") or "").strip()
    notes = (out.get("notes") or "").strip().lower()
    if not value:
        return "underspecified" if "underspecified" in notes else "no value"
    if key.get("specification") != "determined":
        return "disagrees"
    values, units = _split(value), _split(out.get("unit") or "")
    kv, ku = _parts(key["value"]), _parts(key.get("unit"))
    kt, kdb = _parts(key.get("tolerance")), _parts(key.get("tolerance_db"))
    if len(values) != len(kv):
        return "disagrees"
    units += [""] * (len(values) - len(units))
    for v, u, ref_v, ref_u, t, tdb in zip(values, units, kv, ku, kt + [None] * len(kv), kdb + [None] * len(kv)):
        if ref_v is None or not full_credit(f"{v} {u}".strip(), _reference(ref_v, ref_u, t, tdb)):
            return "disagrees"
    return "agrees"


AGREED = {"determined": "agrees", "underspecified": "underspecified", "team reference": "no value"}


def calc_route(specification: str, s1: str, s2: str) -> str:
    """final when both recomputers reach what the key says; otherwise the expert decides."""
    want = AGREED.get(specification)
    return "final" if want is not None and s1 == want and s2 == want else "expert"


def block_matches(reference: Mapping[str, Any], blind_value: str) -> bool:
    return full_credit(str(blind_value), reference)

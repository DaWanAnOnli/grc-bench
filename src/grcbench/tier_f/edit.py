"""Small, explicit edits on flowgraph dicts, shared by fault injection (repair tasks) and mutation
tests (eval plan §5.4: "we break the reference solution in several ways (a wrong frequency, a
missing filter, a wrong sample rate)"). Every function returns a new dict; inputs are never changed.
"""

from __future__ import annotations

import copy
from typing import Any


def _blk(data: dict, name: str) -> dict:
    for b in data["blocks"]:
        if b["name"] == name:
            return b
    raise KeyError(f"no block named {name}")


def set_param(data: dict, block: str, param: str, value: Any) -> dict:
    d = copy.deepcopy(data)
    _blk(d, block).setdefault("parameters", {})[param] = str(value)
    return d


def scale_param(data: dict, block: str, param: str, factor: float) -> dict:
    d = copy.deepcopy(data)
    prm = _blk(d, block).setdefault("parameters", {})
    old = prm.get(param)
    try:
        prm[param] = repr(float(eval(str(old), {"__builtins__": {}}, {})) * factor)   # literal expressions only
    except Exception:
        prm[param] = f"({old})*{factor}"
    return d


def set_id(data: dict, block: str, new_id: str) -> dict:
    d = copy.deepcopy(data)
    _blk(d, block)["id"] = new_id
    return d


def delete_connection(data: dict, src: str, dst: str) -> dict:
    d = copy.deepcopy(data)
    d["connections"] = [c for c in d["connections"] if not (c[0] == src and c[2] == dst)]
    return d


def bypass(data: dict, block: str) -> dict:
    """Remove a single-input/single-output block and connect its upstream directly to its downstream."""
    d = copy.deepcopy(data)
    ins = [c for c in d["connections"] if c[2] == block]
    outs = [c for c in d["connections"] if c[0] == block]
    d["connections"] = [c for c in d["connections"] if block not in (c[0], c[2])]
    for i in ins:
        for o in outs:
            d["connections"].append([i[0], i[1], o[2], o[3]])
    d["blocks"] = [b for b in d["blocks"] if b["name"] != block]
    return d


def rewire_input(data: dict, dst: str, dst_port: str, new_src: str, new_src_port: str = "0") -> dict:
    """Feed `dst:dst_port` from a different block output (the "wrong tap" mutation).

    If the block that used to feed it is left with an unconnected output, that output is parked on a
    Null Sink of the destination's type, so the mutant stays a valid flowgraph (it must reach V4)."""
    d = copy.deepcopy(data)
    old = [c for c in d["connections"] if c[2] == dst and str(c[3]) == str(dst_port)]
    d["connections"] = [c for c in d["connections"] if not (c[2] == dst and str(c[3]) == str(dst_port))]
    d["connections"].append([new_src, str(new_src_port), dst, str(dst_port)])
    for src, sport, _, _ in old:
        if not any(c[0] == src and str(c[1]) == str(sport) for c in d["connections"]):
            typ = (_blk(d, dst).get("parameters") or {}).get("type", "float")
            name = f"{src}_parked"
            d["blocks"].append({"name": name, "id": "blocks_null_sink",
                                "parameters": {"type": typ, "vlen": "1", "num_inputs": "1"}})
            d["connections"].append([src, str(sport), name, "0"])
    return d


OPERATORS = {
    "set_param": set_param, "scale_param": scale_param, "set_id": set_id,
    "delete_connection": delete_connection, "bypass": bypass, "rewire_input": rewire_input,
}


def apply(data: dict, op: dict) -> dict:
    """Apply one edit given as {"op": name, **arguments}."""
    args = {k: v for k, v in op.items() if k not in ("op", "breaks", "note")}
    return OPERATORS[op["op"]](data, **args)

"""Golden tool calls of a Tier-F task in the edit format (build plan, Decision 16 Oct).

A Tier-F sample is answered inside GRC-Agent's harness as "one lookup + one edit"
(grcbench.tier_a.tier_f_mode), so each Tier-F task stores the golden pair of calls that solves it:

    <task_dir>/golden_calls.json = {"query_knowledge": {"query": ..., "domain": "catalog", "k": ...},
                                    "change_graph": {"reason": ..., "remove_connections": [...],
                                                     "remove_blocks": [...], "add_blocks": [...],
                                                     "update_params": [...], "update_states": [...],
                                                     "add_connections": [...]}}

`derive_golden_calls(task_dir)` derives both from the task's files alone, deterministically:

  query_knowledge  a catalog search naming the reference's non-trivial blocks (every block id other
                   than options / variables / imports / notes / parameters, in the reference's
                   order, without repeats); k = min(20, max(5, 2 x their number)).
  change_graph     the edit turning the start graph (start.grc, or the empty build start graph of
                   tier_f_mode.start_graph_text) into reference.grc, in GRC-Agent's change_graph
                   schema (agent.py change_graph_func; empty operation lists are left out), blocks
                   matched by instance name:
                     remove_connections  start connections the reference lacks, and every start
                                         connection of a block that is removed or replaced
                     remove_blocks       start blocks the reference lacks, or whose block id changes
                     add_blocks          reference blocks the start lacks (or replaced ones), with
                                         the reference's parameters and its state if not enabled
                     update_params       on blocks kept: every reference parameter whose text
                                         differs from the start's, and every start parameter the
                                         reference omits reset to its catalog default; the options
                                         block's parameters except its id
                     update_states       blocks kept whose state changes
                     add_connections     reference connections not already present
                   Connections are "src:port->dst:port" with GRC's port keys.

`verify_golden_calls(task_dir, calls)` applies the edit with GRC-Agent's OWN change_graph engine
(grc_agent.adapter.graph.change_graph, on GRC-Agent's platform, in an isolated bubblewrap process
like an episode) to the start graph as tier_f_mode loads it, and checks the result is EQUIVALENT
to reference.grc: the same block ids and states, parameters as GRC resolves them (defaults filled
in) and the same connections, up to renaming of instance names (the options block is compared
without its id). The replay gate (grcbench.tier_a.replay.tier_f_gates) then runs the pair through
the real runner (run_tier_f_sample) with a FunctionModel 10/10.

    ./gb tier-a tier-f-golden <task_dir|all> [--no-write] [--no-gate]
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from itertools import permutations
from pathlib import Path
from typing import Any

import yaml

from ..common.io import read_json, write_text
from ..common.paths import GNURADIO_BLOCKS, WORK
from .graph_checks import NON_FUNCTIONAL

GOLDEN_FILE = "golden_calls.json"
LOOKUP, EDIT = "query_knowledge", "change_graph"
EDIT_OPS = ("remove_connections", "remove_blocks", "add_blocks", "update_params", "update_states",
            "add_connections")
REASON = "Edit the flowgraph as the instruction asks."
VERIFY_TIMEOUT = 300.0


class GoldenCallsError(ValueError):
    """golden_calls.json is missing or malformed, or the derived edit is not equivalent."""


# --------------------------------------------------------------------------------------------
# Derivation (pure: YAML and the catalog's block YAML files)
# --------------------------------------------------------------------------------------------

def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _blocks(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {b["name"]: b for b in data.get("blocks") or [] if isinstance(b, dict) and b.get("name")}


def _state(block: dict[str, Any]) -> str:
    return _text((block.get("states") or {}).get("state") or "enabled")


def _conn(c: list[Any]) -> str:
    return f"{_text(c[0])}:{_text(c[1])}->{_text(c[2])}:{_text(c[3])}"


def catalog_default(block_id: str, param: str, blocks_dir: Path = GNURADIO_BLOCKS) -> str | None:
    """A parameter's default in the block's catalog YAML (an enum's first option when it has no
    default); None if the YAML does not declare a usable one (e.g. a ${...} template)."""
    path = Path(blocks_dir) / f"{block_id}.block.yml"
    if not path.is_file():
        return None
    for p in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("parameters") or []:
        if isinstance(p, dict) and p.get("id") == param:
            if "default" in p:
                value = _text(p["default"])
            elif isinstance(p.get("options"), list) and p["options"]:
                value = _text(p["options"][0])
            else:
                value = ""
            return None if "${" in value else value
    return None


def lookup_args(reference: dict[str, Any]) -> dict[str, Any]:
    """The golden query_knowledge arguments (module docstring)."""
    ids: list[str] = []
    for b in reference.get("blocks") or []:
        bid = _text(b.get("id"))
        if bid and not bid.startswith(NON_FUNCTIONAL) and bid not in ids:
            ids.append(bid)
    return {"query": " ".join(ids), "domain": "catalog", "k": min(20, max(5, 2 * len(ids)))}


def edit_args(start: dict[str, Any], reference: dict[str, Any], *, reason: str = REASON) -> dict[str, Any]:
    """The golden change_graph arguments turning `start` into `reference` (module docstring)."""
    sb, rb = _blocks(start), _blocks(reference)
    replaced = {n for n in sb if n in rb and sb[n].get("id") != rb[n].get("id")}
    removed = [n for n in sb if n not in rb or n in replaced]
    added = [n for n in rb if n not in sb or n in replaced]
    gone = set(removed)
    ref_conns = [_conn(c) for c in reference.get("connections") or []]
    start_conns = [_conn(c) for c in start.get("connections") or []]
    remove_connections = [c for c in start_conns
                          if c not in ref_conns or c.split(":")[0] in gone or c.split("->")[1].split(":")[0] in gone]
    kept = [c for c in start_conns if c not in remove_connections]
    add_connections = [c for c in ref_conns if c not in kept]
    add_blocks = []
    for n in added:
        b = rb[n]
        item: dict[str, Any] = {"block_id": _text(b["id"]), "instance_name": n,
                                "params": {k: _text(v) for k, v in (b.get("parameters") or {}).items() if k != "id"}}
        if not item["params"]:
            del item["params"]
        if _state(b) != "enabled":
            item["state"] = _state(b)
        add_blocks.append(item)
    update_params, update_states = [], []
    so = ((start.get("options") or {}).get("parameters") or {})
    ro = ((reference.get("options") or {}).get("parameters") or {})
    opts = {k: _text(v) for k, v in ro.items() if k != "id" and _text(so.get(k)) != _text(v)}
    if opts:
        update_params.append({"instance_name": _text(so.get("id") or "default"), "params": opts})
    for n in rb:
        if n not in sb or n in replaced:
            continue
        s, r = sb[n], rb[n]
        sp, rp = s.get("parameters") or {}, r.get("parameters") or {}
        params = {k: _text(v) for k, v in rp.items() if k != "id" and (k not in sp or _text(sp[k]) != _text(v))}
        for k, v in sp.items():
            if k in rp or k == "id":
                continue
            default = catalog_default(_text(r["id"]), k)
            if default is not None and default != _text(v):
                params[k] = default
        if params:
            update_params.append({"instance_name": n, "params": params})
        if _state(s) != _state(r):
            update_states.append({"instance_name": n, "state": _state(r)})
    ops = {"remove_connections": remove_connections, "remove_blocks": removed, "add_blocks": add_blocks,
           "update_params": update_params, "update_states": update_states, "add_connections": add_connections}
    out: dict[str, Any] = {"reason": reason}
    out.update({k: v for k, v in ops.items() if v})
    return out


def task_graphs(task_dir: Path | str) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """(task record, start graph compact text, start data, reference data) of a Tier-F task."""
    from ..tier_a.tier_f_mode import load_tier_f_task, start_graph_text
    task = load_tier_f_task(task_dir)
    start_text, _ = start_graph_text(task)
    reference = yaml.safe_load((Path(task["_dir"]) / task["reference_grc"]).read_text(encoding="utf-8"))
    return task, start_text, yaml.safe_load(start_text), reference


def derive_golden_calls(task_dir: Path | str) -> dict[str, Any]:
    """{"query_knowledge": args, "change_graph": args} for the task (module docstring)."""
    _, _, start, reference = task_graphs(task_dir)
    return {LOOKUP: lookup_args(reference), EDIT: edit_args(start, reference)}


def load_golden_calls(task_dir: Path | str) -> dict[str, Any]:
    path = Path(task_dir) / GOLDEN_FILE
    if not path.is_file():
        raise GoldenCallsError(f"{Path(task_dir).name}: no {GOLDEN_FILE}")
    data = read_json(path)
    if not isinstance(data, dict) or set(data) != {LOOKUP, EDIT} \
            or not all(isinstance(data[k], dict) for k in (LOOKUP, EDIT)):
        raise GoldenCallsError(f"{path}: must hold exactly a {LOOKUP} and a {EDIT} argument object")
    return data


def golden_turns(calls: dict[str, Any]) -> list[dict[str, Any]]:
    """FunctionModel turns replaying the pair: the lookup in the first reply, the edit in the next
    (so the edit could have used the lookup's result: 2 of the 3 model requests)."""
    return [{"tool_calls": [{"name": LOOKUP, "args": calls[LOOKUP], "id": "golden_lookup"}]},
            {"tool_calls": [{"name": EDIT, "args": calls[EDIT], "id": "golden_edit"}]},
            {"text": "Done."}]


def write_golden_calls(task_dir: Path | str, calls: dict[str, Any]) -> Path:
    path = Path(task_dir) / GOLDEN_FILE
    write_text(path, json.dumps({LOOKUP: calls[LOOKUP], EDIT: calls[EDIT]}, indent=1, ensure_ascii=False))
    return path


# --------------------------------------------------------------------------------------------
# Equivalence of two GRC views (grcbench.tier_a._grc_view)
# --------------------------------------------------------------------------------------------

def _sig(block: dict[str, Any]) -> tuple:
    return (block["key"], block["state"], tuple(sorted((k, v) for k, v in block["params"].items() if k != "id")))


def equivalence_diffs(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Why view `a` is not equivalent to view `b` (empty: equivalent): same block ids and states,
    parameters as GRC resolves them and connections, up to renaming of instance names; the options
    block is compared without its id."""
    if not a.get("ok") or not b.get("ok"):
        return [f"graph does not load: {a.get('error') or b.get('error')}"]
    oa = [x for x in a["blocks"].values() if x["key"] == "options"]
    ob = [x for x in b["blocks"].values() if x["key"] == "options"]
    diffs: list[str] = []
    if len(oa) != 1 or len(ob) != 1:
        diffs.append("options block missing")
    else:
        for k in sorted(set(oa[0]["params"]) | set(ob[0]["params"])):
            if k != "id" and oa[0]["params"].get(k) != ob[0]["params"].get(k):
                diffs.append(f"options.{k}: {oa[0]['params'].get(k)!r} != {ob[0]['params'].get(k)!r}")
    ba = {n: x for n, x in a["blocks"].items() if x["key"] != "options"}
    bb = {n: x for n, x in b["blocks"].items() if x["key"] != "options"}
    ca = {tuple(c) for c in a["connections"]}
    cb = {tuple(c) for c in b["connections"]}
    ident = []
    for n in sorted(set(ba) | set(bb)):
        if n not in ba or n not in bb:
            ident.append(f"block {n!r} only in the {'reference' if n not in ba else 'edited graph'}")
        elif _sig(ba[n]) != _sig(bb[n]):
            x, y = ba[n], bb[n]
            if x["key"] != y["key"] or x["state"] != y["state"]:
                ident.append(f"{n}: {x['key']}/{x['state']} != {y['key']}/{y['state']}")
            for k in sorted(set(x["params"]) | set(y["params"])):
                if k != "id" and x["params"].get(k) != y["params"].get(k):
                    ident.append(f"{n}.{k}: {x['params'].get(k)!r} != {y['params'].get(k)!r}")
    ident += [f"connection {c} only in the edited graph" for c in sorted(ca - cb)]
    ident += [f"connection {c} only in the reference" for c in sorted(cb - ca)]
    if ident and _renaming(ba, bb, ca, cb) is None:
        diffs += ident
    return diffs


def _renaming(ba: dict, bb: dict, ca: set, cb: set) -> dict[str, str] | None:
    """A bijection a-name -> b-name between blocks of equal signature that maps ca onto cb."""
    if len(ba) != len(bb) or len(ca) != len(cb):
        return None
    groups: dict[tuple, tuple[list[str], list[str]]] = {}
    for n, x in sorted(ba.items()):
        groups.setdefault(_sig(x), ([], []))[0].append(n)
    for n, x in sorted(bb.items()):
        if _sig(x) not in groups:
            return None
        groups[_sig(x)][1].append(n)
    if any(len(l) != len(r) for l, r in groups.values()):
        return None
    keys = sorted(groups, key=lambda s: len(groups[s][0]))
    total = 1
    for s in keys:
        for i in range(2, len(groups[s][0]) + 1):
            total *= i
        if total > 100_000:
            return None                                        # too ambiguous to search

    def search(i: int, mapping: dict[str, str]) -> dict[str, str] | None:
        if i == len(keys):
            mapped = {(mapping[s], sp, mapping[d], dp) for s, sp, d, dp in ca}
            return dict(mapping) if mapped == cb else None
        left, right = groups[keys[i]]
        for perm in permutations(right):
            found = search(i + 1, {**mapping, **dict(zip(left, perm))})
            if found is not None:
                return found
        return None
    return search(0, {})


# --------------------------------------------------------------------------------------------
# Verification through GRC-Agent's own change_graph (isolated process)
# --------------------------------------------------------------------------------------------

def _verify_main(spec_path: str) -> int:
    """Episode-process side: apply the edit with GRC-Agent's change_graph, write both views."""
    from grc_agent.adapter import get_platform
    from grc_agent.adapter.graph import change_graph

    from ..tier_a._grc_view import view_flow_graph
    from ._worker import _normalise
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    platform = get_platform()

    def load(text: str) -> Any:
        fg = platform.make_flow_graph()
        fg.import_data(_normalise(yaml.safe_load(text)))
        fg.rewrite()
        fg.validate()
        return fg
    out: dict[str, Any] = {}
    fg = load(Path(spec["start"]).read_text(encoding="utf-8"))
    out["start"] = view_flow_graph(fg)
    args = {k: v for k, v in spec["edit"].items() if k in EDIT_OPS and v}
    try:
        res = change_graph(fg, **args)
    except Exception as exc:                     # noqa: BLE001 - reported as a failed edit
        res = {"ok": False, "error_type": "exception", "errors": [f"{type(exc).__name__}: {exc}"]}
    out["result"] = json.loads(json.dumps(res, default=str))
    out["edited"] = view_flow_graph(fg)
    out["reference"] = view_flow_graph(load(Path(spec["reference"]).read_text(encoding="utf-8")))
    Path(spec["out"]).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 0


def verify_golden_calls(task_dir: Path | str, calls: dict[str, Any] | None = None, *, keep: bool = False
                        ) -> dict[str, Any]:
    """Apply `calls["change_graph"]` (default: the derived calls) with GRC-Agent's change_graph to
    the start graph and compare the result with reference.grc (module docstring)."""
    from ..tier_a import isolation
    task, start_text, _, _ = task_graphs(task_dir)
    calls = calls or derive_golden_calls(task_dir)
    run_dir = isolation.check_run_dir(WORK / "tier_f_golden" / f"{task['id']}-{uuid.uuid4().hex[:8]}")
    run_dir.mkdir(parents=True)
    try:
        env = isolation.make_run_env(run_dir)
        write_text(run_dir / "start_compact.grc", start_text)
        shutil.copyfile(Path(task["_dir"]) / task["reference_grc"], run_dir / "reference.grc")
        spec = {"start": str(run_dir / "start_compact.grc"), "reference": str(run_dir / "reference.grc"),
                "edit": calls[EDIT], "out": str(run_dir / "verify.json")}
        (run_dir / "verify_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        proc = isolation.run_episode_process(run_dir, "grcbench.tier_f.golden",
                                             ["verify", str(run_dir / "verify_spec.json")], env,
                                             timeout=VERIFY_TIMEOUT, network="none")
        if not (run_dir / "verify.json").is_file():
            return {"task": task["id"], "equivalent": False, "accepted": False,
                    "diffs": [f"verification process failed (exit {proc.returncode}): {proc.stderr[-1500:]}"]}
        views = json.loads((run_dir / "verify.json").read_text(encoding="utf-8"))
    finally:
        if not keep:
            shutil.rmtree(run_dir, ignore_errors=True)
    result = views["result"]
    accepted = bool(result.get("ok"))
    diffs = equivalence_diffs(views["edited"], views["reference"]) if accepted else \
        [f"change_graph rejected the edit: {result.get('errors') or result.get('error_type')}"]
    return {"task": task["id"], "accepted": accepted, "equivalent": accepted and not diffs, "diffs": diffs,
            "still_invalid": bool(result.get("still_invalid")), "reference_valid": views["reference"]["valid"],
            "edited_valid": views["edited"]["valid"]}


def derive_and_verify(task_dir: Path | str, *, write: bool = True) -> dict[str, Any]:
    """Derive the golden calls, verify them through GRC-Agent's change_graph and (if equivalent and
    `write`) store golden_calls.json in the task directory."""
    calls = derive_golden_calls(task_dir)
    out = verify_golden_calls(task_dir, calls)
    out["calls"] = calls
    out["written"] = None
    if out["equivalent"] and write:
        out["written"] = str(write_golden_calls(task_dir, calls))
    return out


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "verify":
        return _verify_main(argv[1])
    print("usage: python -m grcbench.tier_f.golden verify <spec.json>   (use ./gb tier-a tier-f-golden)")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

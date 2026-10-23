"""Structural checks on the final flowgraph, evaluated in V5 next to the signal checks.

Signal checks alone accept any flowgraph that produces the right output. When an instruction
requires a structure ("filter the square wave", "only change parameter values"), that requirement
must be checked too (eval plan §5.4: every stated requirement is checked; nothing unstated is).
Checks are name-independent: they refer to block ids (types), never to instance names. Only
enabled blocks count (state "disabled" blocks and their connections are ignored); a block in state
"bypassed" is present for block_present / path but differs from an enabled one for the
*_as_start comparisons.

Check kinds (KINDS below; each may carry "group": f2p | p2p | main, like signal checks):
  block_present    {"ids": [...], "min": 1, "max": null, "params": {"decim": "4"}}
                   at least `min` (and at most `max`) enabled blocks whose id is in `ids` and whose
                   parameters equal the given values (numeric comparison when both parse as numbers)
  block_absent     {"ids": [...]}                       no enabled block with one of these ids
  path             {"from_ids": [...], "to_ids": [...], "via_ids": [...] optional,
                    "from_upstream_of": {"ids": [...], "params": {...}} optional,
                    "to_params": {...} optional}
                   a directed stream/message path from a block in from_ids to a block in to_ids,
                   passing through a block in via_ids when given. "from_upstream_of" anchors the
                   start on an observation point: only from_ids blocks that reach a block matching
                   it (e.g. the File Sink of stream_a.dat) count, so "the source behind stream_a.dat
                   reaches mux_out.dat" is checked whatever the source is named. "to_params" keeps
                   only to_ids blocks with these parameter values. A "file" parameter compares by
                   file name (the directory is ignored), as the observation contract does.
  same_blocks_as_start  {"variables": "count", "ignore_ids": []}
                   the multiset of block ids equals the start graph's (parameter-only tasks: no
                   blocks added or removed). A bypassed block counts as "<id> (bypassed)".
                   "variables" says how variable blocks (id "variable" or "variable_*") count:
                     "count"      (default) like any other block: by id only, so a variable may be
                                  renamed, but not added or removed
                     "value_only" by (id, name): each start variable must still exist under the
                                  same name and type, none may be added; its value may change
                     "ignore"     not compared: variables may be added, removed or renamed
                   "ignore_ids": block ids left out of the comparison on both sides (e.g. "note",
                   "import"), so a harmless helper block does not fail the check
  same_connections_as_start  {"mode": "isomorphic"}
                   the connections equal the start graph's up to block renaming, so a value-only
                   answer passes and any rewiring fails. A connection is (source block, source
                   port, sink block, sink port) between two enabled blocks; ports compare as
                   strings (stream port numbers, message port names). Blocks without connections
                   (variables, notes) are not part of this check; same_blocks_as_start covers them.
                   "mode": "isomorphic" (default) there is a one-to-one mapping between the
                                  connected blocks of the answer and of the start graph that keeps
                                  every block id and maps the connections exactly (catches a
                                  rewiring between two blocks of the same type)
                           "multiset" only the multiset of (source id, source port, sink id, sink
                                  port) must be equal (weaker; implied by "isomorphic")
  max_functional_blocks {"max": n}                      enabled blocks other than options/variables
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any

NON_FUNCTIONAL = ("options", "variable", "import", "note", "parameter")
KINDS = ("block_present", "block_absent", "path", "same_blocks_as_start", "same_connections_as_start",
         "max_functional_blocks")
VARIABLE_MODES = ("count", "value_only", "ignore")
CONNECTION_MODES = ("isomorphic", "multiset")
ISO_STEP_LIMIT = 200_000


def _enabled(data: dict) -> list[dict]:
    return [b for b in data.get("blocks", []) or [] if isinstance(b, dict)
            and (b.get("states") or {}).get("state", "enabled") != "disabled"]


def _eq(a: Any, b: Any) -> bool:
    sa, sb = str(a).strip().strip("'\""), str(b).strip().strip("'\"")
    if sa == sb:
        return True
    try:
        fa, fb = float(sa), float(sb)
        return abs(fa - fb) <= 1e-9 * max(1.0, abs(fb))
    except ValueError:
        return False


def _param_eq(key: str, a: Any, b: Any) -> bool:
    if key == "file":                                  # by file name, like the observation contract
        a, b = (str(x).strip().strip("'\"").replace("\\", "/").rsplit("/", 1)[-1] for x in (a, b))
    return _eq(a, b)


def _matches(block: dict, ids: list[str], params: dict | None) -> bool:
    if block.get("id") not in ids:
        return False
    bp = block.get("parameters") or {}
    return all(k in bp and _param_eq(k, bp[k], v) for k, v in (params or {}).items())


def _graph(data: dict) -> dict[str, set[str]]:
    names = {b["name"] for b in _enabled(data)}
    adj: dict[str, set[str]] = {n: set() for n in names}
    for c in data.get("connections", []) or []:
        if isinstance(c, (list, tuple)) and len(c) == 4 and c[0] in names and c[2] in names:
            adj[c[0]].add(c[2])
    return adj


def _reachable(adj: dict[str, set[str]], start: str) -> set[str]:
    seen, q = {start}, deque([start])
    while q:
        for nxt in adj.get(q.popleft(), ()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _is_variable(block_id: str) -> bool:
    return block_id == "variable" or block_id.startswith("variable_")


def _block_keys(data: dict, variables: str, ignore_ids: set[str]) -> Counter:
    keys: Counter = Counter()
    for b in _enabled(data):
        bid = str(b.get("id", ""))
        if bid in ignore_ids:
            continue
        key = bid
        if _is_variable(bid):
            if variables == "ignore":
                continue
            if variables == "value_only":
                key = f"{bid} '{b.get('name')}'"
        if (b.get("states") or {}).get("state") == "bypassed":
            key += " (bypassed)"
        keys[key] += 1
    return keys


def same_blocks(data: dict, start: dict, variables: str = "count",
                ignore_ids: list[str] | None = None) -> tuple[bool, str]:
    if variables not in VARIABLE_MODES:
        return False, f"unknown variables mode {variables!r} (use one of {VARIABLE_MODES})"
    ign = set(ignore_ids or [])
    a, s = _block_keys(data, variables, ign), _block_keys(start, variables, ign)
    if a == s:
        return True, "same block types"
    return False, f"added {dict(a - s)}, removed {dict(s - a)}"


def _edges(data: dict) -> tuple[list[tuple[str, str, str, str]], dict[str, str]]:
    """Connections between enabled blocks as (src name, src port, dst name, dst port), plus name -> id."""
    label = {b["name"]: str(b.get("id", "")) for b in _enabled(data) if "name" in b}
    out = []
    for c in data.get("connections", []) or []:
        if isinstance(c, (list, tuple)) and len(c) == 4 and c[0] in label and c[2] in label:
            out.append((str(c[0]), str(c[1]).strip(), str(c[2]), str(c[3]).strip()))
    return out, label


def _typed(edges, label) -> Counter:
    return Counter((label[s], sp, label[d], dp) for s, sp, d, dp in edges)


def _fmt(edge_counter: Counter) -> dict[str, int]:
    return {f"{s}:{sp}->{d}:{dp}": n for (s, sp, d, dp), n in sorted(edge_counter.items())}


def _isomorphic(ea, la, es, ls) -> bool | None:
    """Is there a label-preserving bijection between the connected blocks mapping ea onto es exactly?
    None if the search exceeds ISO_STEP_LIMIT (treated as a failure)."""
    nodes_a = sorted({n for s, _, d, _ in ea for n in (s, d)})
    nodes_s = sorted({n for s, _, d, _ in es for n in (s, d)})
    if len(nodes_a) != len(nodes_s):
        return False
    if set(nodes_a) == set(nodes_s) and all(la[n] == ls[n] for n in nodes_a) and Counter(ea) == Counter(es):
        return True                                       # fast path: same names, same wiring

    def tables(edges, lab):
        pair: dict[tuple[str, str], Counter] = defaultdict(Counter)
        sig: dict[str, Counter] = defaultdict(Counter)
        nbr: dict[str, set[str]] = defaultdict(set)
        for s, sp, d, dp in edges:
            pair[(s, d)][(sp, dp)] += 1
            sig[s][("out", sp, lab[d], dp)] += 1
            sig[d][("in", dp, lab[s], sp)] += 1
            nbr[s].add(d)
            nbr[d].add(s)
        return pair, {n: tuple(sorted(c.items())) for n, c in sig.items()}, nbr

    pa, sa, na = tables(ea, la)
    ps, ss, ns = tables(es, ls)
    cands = {n: [m for m in nodes_s if ls[m] == la[n] and ss[m] == sa[n]] for n in nodes_a}
    if any(not c for c in cands.values()):
        return False
    f: dict[str, str] = {}
    used: set[str] = set()
    steps = 0

    def pick() -> str:
        free = [n for n in nodes_a if n not in f]
        return min(free, key=lambda n: (-sum(m in f for m in na[n]), len(cands[n]), n))

    def ok(n: str, c: str) -> bool:
        if pa.get((n, n), Counter()) != ps.get((c, c), Counter()):
            return False
        mapped_a = [m for m in na[n] if m in f and m != n]
        if len(mapped_a) != sum(1 for m in ns[c] if m in used and m != c):
            return False
        return all(pa.get((n, m), Counter()) == ps.get((c, f[m]), Counter())
                   and pa.get((m, n), Counter()) == ps.get((f[m], c), Counter()) for m in mapped_a)

    def search() -> bool | None:
        nonlocal steps
        if len(f) == len(nodes_a):
            return True
        n = pick()
        for c in cands[n]:
            if c in used:
                continue
            steps += 1
            if steps > ISO_STEP_LIMIT:
                return None
            if ok(n, c):
                f[n] = c
                used.add(c)
                r = search()
                if r is not False:
                    return r
                del f[n]
                used.discard(c)
        return False

    return search()


def same_connections(data: dict, start: dict, mode: str = "isomorphic") -> tuple[bool, str]:
    if mode not in CONNECTION_MODES:
        return False, f"unknown mode {mode!r} (use one of {CONNECTION_MODES})"
    ea, la = _edges(data)
    es, ls = _edges(start)
    ta, ts = _typed(ea, la), _typed(es, ls)
    if ta != ts:
        return False, f"added {_fmt(ta - ts)}, removed {_fmt(ts - ta)}"
    if mode == "multiset":
        return True, "same connections (multiset of block-id edges)"
    iso = _isomorphic(ea, la, es, ls)
    if iso is None:
        return False, "connection comparison undecided (search limit reached)"
    return (True, "same connections") if iso else (
        False, "same block-id edges, but wired between different blocks than in the start graph")


def evaluate(checks: list[dict], data: dict, start: dict | None = None) -> list[dict]:
    blocks = _enabled(data)
    by_name = {b["name"]: b for b in blocks}
    adj = _graph(data)
    out = []
    for chk in checks:
        kind = chk["kind"]
        ok, detail = False, ""
        if kind == "block_present":
            n = sum(_matches(b, chk["ids"], chk.get("params")) for b in blocks)
            ok = n >= chk.get("min", 1) and (chk.get("max") is None or n <= chk["max"])
            detail = f"{n} matching blocks"
        elif kind == "block_absent":
            n = sum(b.get("id") in chk["ids"] for b in blocks)
            ok, detail = n == 0, f"{n} forbidden blocks"
        elif kind == "path":
            froms = [n for n, b in by_name.items() if b.get("id") in chk["from_ids"]]
            anchor = chk.get("from_upstream_of")
            if anchor:
                marks = {n for n, b in by_name.items() if _matches(b, anchor["ids"], anchor.get("params"))}
                froms = [f for f in froms if _reachable(adj, f) & marks]
            tos = {n for n, b in by_name.items() if _matches(b, chk["to_ids"], chk.get("to_params"))}
            vias = {n for n, b in by_name.items() if b.get("id") in chk.get("via_ids", [])}
            for f in froms:
                reach = _reachable(adj, f)
                if not chk.get("via_ids"):
                    if reach & tos:
                        ok = True
                        break
                else:
                    if any(tos & _reachable(adj, v) for v in vias & reach):
                        ok = True
                        break
            detail = "path found" if ok else "no such path"
        elif kind == "same_blocks_as_start":
            if start is None:
                detail = "no start graph"
            else:
                ok, detail = same_blocks(data, start, chk.get("variables", "count"), chk.get("ignore_ids"))
        elif kind == "same_connections_as_start":
            if start is None:
                detail = "no start graph"
            else:
                ok, detail = same_connections(data, start, chk.get("mode", "isomorphic"))
        elif kind == "max_functional_blocks":
            n = sum(not str(b.get("id", "")).startswith(NON_FUNCTIONAL) for b in blocks)
            ok, detail = n <= chk["max"], f"{n} functional blocks"
        else:
            detail = f"unknown check kind {kind}"
        out.append({"kind": kind, "group": chk.get("group", "main"), "passed": bool(ok), "detail": detail,
                    "spec": chk})
    return out

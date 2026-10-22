"""`./gb traces`: audit of Qoherent's agent traces of 15-18 Oct against the benchmark (eval plan §6).

The traces (project/new_data/, read-only) are training data: 1,253 one-shot Tier-1 and 600 two-turn
Tier-2 conversations that DeepSeek-V4.1-Flash produced through GRC-Agent's own harness. They are used
in full for training (Joel, 23 Oct), so the benchmark is checked against them:

  harness      the system prompt and tool schemas of the traces against the SFT release that the
               Tier-A conformance check uses (tier_a/conformance.py): training and evaluation must
               present the same harness
  tool use     families, tool calls per trace, tools never called, the replies to run_flowgraph (the
               collection harness could not execute flowgraphs), validation-feedback retries, reads of
               the collection's own folder, tool names outside the schema
  exposure     for every test-side block of splits/block_partition.json: the traces in which its
               catalog entry (a knowledge-base result or its printed block YAML) appears, and whether a
               trace built or ran it in GNU Radio code. Catalog facts are exposed, so Tier K asks only
               questions whose answers come from execution (tier_k/blockgen.py)
  task level   eval plan §6: "a session with the same goal or the same starting flowgraph counts as
               overlap": every trace's start graph and final functional blocks against the start,
               reference and golden graphs of every Tier-F and Tier-A task
  signatures   the trace's final graph cell (tier_f/compose.signature) against the reserved cells

Outputs: reports/agent_traces.md and reports/agent_traces.json (the audit), splits/agent_traces.json
(the manifest the block-question generator reads: catalog-exposed and executed blocks). The text
overlap of the traces with every benchmark item is the contamination audit's `agent_traces` corpus.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import yaml

from grcbench.common.io import read_json, sha256_file, write_json, write_text
from grcbench.common.paths import AGENT_TRACES, GNURADIO_BLOCKS, ITEMS, NEW_DATA, REPO, REPORTS, SPLITS

OUT_REPORT = REPORTS / "agent_traces.md"
OUT_JSON = REPORTS / "agent_traces.json"
OUT_MANIFEST = SPLITS / "agent_traces.json"
FEEDBACK = "Validation feedback"
MIN_MATCH_BLOCKS = 2        # a task-level match needs at least two functional blocks in common
NOT_FUNCTIONAL = ("variable", "import", "options", "parameter", "note", "epy_module", "blocks_null_source",
                  "blocks_null_sink", "blocks_throttle", "blocks_throttle2", "blocks_file_sink", "blocks_head")
GR_MODULES = ("blocks", "analog", "digital", "filter", "fec", "vocoder", "channels", "dtv", "fft", "trellis", "pdu")
_EXEC = re.compile(r"top_block|\.run\(\)|vector_source|vector_sink|firdes\.|\.work\(|from gnuradio import ("
                   + "|".join(GR_MODULES) + r")")
_API = re.compile(r"\b(" + "|".join(GR_MODULES) + r")\.([a-z][a-z0-9_]*)")
_RESULT_ID = re.compile(r'"block_id":\s*"([a-z0-9_]+)"')
_YAML_ID = re.compile(r"(?m)^id:\s*([a-z0-9_]+)\s*$")


# --------------------------------------------------------------------------- reading
def traces() -> Iterator[tuple[str, dict]]:
    """(tier, row) for every trace, Tier 1 first, in file order."""
    for tier, path in AGENT_TRACES.items():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield tier, json.loads(line)


def family(scenario_id: str) -> str:
    return re.match(r"(t\d_[a-z]+)", scenario_id).group(1)


def _calls(row: dict) -> list[dict]:
    return [tc for m in row["messages"] for tc in (m.get("tool_calls") or [])]


def _args(tc: dict) -> dict:
    try:
        a = json.loads(tc["function"]["arguments"])
    except (TypeError, ValueError):
        return {}
    return a if isinstance(a, dict) else {}


def start_graph(row: dict, block_ids: set[str]) -> str | None:
    """None for the empty canvas, else the block id of the one-block scaffold the trace starts from."""
    m = re.search(r"Active flowgraph file path: (\S+)", row["system"])
    stem = Path(m.group(1)).stem if m else ""
    if stem in ("", "t1_empty"):
        return None
    return stem[3:] if stem.startswith("t1_") else stem


def final_blocks(row: dict, start: str | None) -> list[str]:
    """Block ids of the trace's final graph: the scaffold block plus every block a change_graph added."""
    out = [start] if start else []
    for tc in _calls(row):
        if tc["function"]["name"] == "change_graph":
            for b in _args(tc).get("add_blocks") or []:
                if isinstance(b, dict) and b.get("block_id"):
                    out.append(str(b["block_id"]))
    return out


def functional(ids) -> tuple[str, ...]:
    return tuple(sorted(i for i in ids if i and not i.startswith(NOT_FUNCTIONAL)))


# --------------------------------------------------------------------------- the checks
def harness(rows: list[dict]) -> dict[str, Any]:
    from grcbench.tier_a.conformance import SFT_FILE
    with open(SFT_FILE, encoding="utf-8") as fh:
        ref = json.loads(fh.readline())
    old = {t["name"]: t for t in ref["tools"]}

    def body(system: str) -> str:
        return "\n".join(line for line in system.splitlines()
                         if not line.startswith(("Session ID", "Active flowgraph file path")))

    systems = Counter(body(r["system"]) for r in rows)
    toolsets = {json.dumps({t["function"]["name"]: t["function"] for t in r["tools"]}, sort_keys=True) for r in rows}
    new = json.loads(next(iter(toolsets)))
    differing = {}
    for name in sorted(set(new) & set(old)):
        diffs = []

        def walk(a, b, path=""):
            if isinstance(a, dict) and isinstance(b, dict):
                for k in sorted(set(a) | set(b)):
                    if k not in a:
                        diffs.append(f"added {path}/{k}")
                    elif k not in b:
                        diffs.append(f"removed {path}/{k}")
                    else:
                        walk(a[k], b[k], f"{path}/{k}")
            elif a != b:
                diffs.append(f"changed {path}")
        walk(old[name], new[name])
        if diffs:
            differing[name] = diffs
    return {"rows": len(rows), "distinct_system_prompts": len(systems), "distinct_tool_sets": len(toolsets),
            "system_prompt_equal": len(systems) == 1 and next(iter(systems)) == body(ref["system"]),
            "tools_only_in_traces": sorted(set(new) - set(old)), "tools_only_in_reference": sorted(set(old) - set(new)),
            "schemas_differing": differing, "reference": str(SFT_FILE.relative_to(SFT_FILE.parents[2]))}


def tool_use(rows: list[tuple[str, dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tier in AGENT_TRACES:
        sub = [r for t, r in rows if t == tier]
        calls = [len(_calls(r)) for r in sub]
        totals = Counter(tc["function"]["name"] for r in sub for tc in _calls(r))
        schema = {t["function"]["name"] for t in sub[0]["tools"]}
        run_replies: Counter = Counter()
        outside = feedback = malformed = guardrail = 0
        for r in sub:
            ids = {tc["id"]: tc["function"]["name"] for tc in _calls(r)}
            for m in r["messages"]:
                if m["role"] == "tool" and ids.get(m.get("tool_call_id")) == "run_flowgraph":
                    run_replies[str(m["content"])[:90]] += 1
                if m["role"] == "user" and str(m.get("content", "")).startswith(FEEDBACK):
                    feedback += 1
            for tc in _calls(r):
                if tc["function"]["name"] not in schema:
                    malformed += 1
                if tc["function"]["name"] in ("run_command", "start_command") and "atomic_dataset" in str(
                        tc["function"]["arguments"]):
                    outside += 1
            guardrail += len(r.get("guardrail_events") or [])
        out[tier] = {"rows": len(sub), "families": dict(sorted(Counter(family(r["scenario_id"]) for r in sub).items())),
                     "subcategories": len({r["category"] for r in sub}),
                     "calls_mean": round(statistics.mean(calls), 2), "calls_median": round(statistics.median(calls), 1),
                     "calls_max": max(calls), "rows_with_8_to_20_calls": sum(8 <= c <= 20 for c in calls),
                     "tool_totals": dict(totals.most_common()), "tools_never_called": sorted(schema - set(totals)),
                     "run_flowgraph_replies": dict(run_replies), "validation_feedback_turns": feedback,
                     "guardrail_events": guardrail, "collection_folder_reads": outside,
                     "tool_names_outside_schema": malformed}
    return out


def _make_map() -> dict[str, list[str]]:
    """GNU Radio Python constructor ("blocks.keep_m_in_n") -> GRC block ids, from each block YAML's make."""
    out: dict[str, list[str]] = defaultdict(list)
    for p in sorted(Path(GNURADIO_BLOCKS).glob("*.block.yml")):
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        make = str(((d.get("templates") or {}).get("make")) or "")
        m = re.match(r"\s*([a-z][a-z0-9_]*)\.([a-z][a-z0-9_${}.]*)\s*\(", make)
        if m and d.get("id"):
            out[f"{m.group(1)}.{m.group(2).split('$')[0]}"].append(str(d["id"]))
    return out


def exposure(rows: list[tuple[str, dict]], test_blocks: set[str]) -> dict[str, Any]:
    shown: dict[str, set[str]] = defaultdict(set)
    executed: dict[str, set[str]] = defaultdict(set)
    commands = []
    make = _make_map()
    for _, r in rows:
        sid = r["scenario_id"]
        for m in r["messages"]:
            if m["role"] == "tool":
                text = str(m.get("content") or "")
                for bid in set(_RESULT_ID.findall(text)) | set(_YAML_ID.findall(text)):
                    shown[bid].add(sid)
        for tc in _calls(r):
            if tc["function"]["name"] not in ("run_command", "start_command"):
                continue
            cmd = str(_args(tc).get("command", ""))
            if "python" not in cmd or not _EXEC.search(cmd):
                continue
            apis = sorted({f"{a}.{b}" for a, b in _API.findall(cmd)})
            blocks = sorted({bid for api in apis for key, ids in make.items() if api == key or (
                key.endswith("_") and api.startswith(key)) for bid in ids})
            for bid in blocks:
                executed[bid].add(sid)
            commands.append({"scenario_id": sid, "python_api": apis, "blocks": blocks})
    return {"catalog_shown": {b: sorted(shown[b]) for b in sorted(test_blocks) if shown.get(b)},
            "catalog_shown_any": len({b for b in shown}),
            "executed_in_code": {b: sorted(s) for b, s in sorted(executed.items())},
            "gnuradio_code_commands": len(commands), "commands": commands}


def _task_graphs() -> list[dict]:
    """Start, reference and golden graphs of every Tier-F and Tier-A task, as functional block-id sets."""
    out = []
    for task in sorted((ITEMS / "tier_f").glob("*/*/task.json")) + sorted((ITEMS / "tier_a").glob("*/*/task.json")):
        d = task.parent
        files = [("start", d / "start.grc"), ("reference", d / "reference.grc"),
                 ("golden", d / "golden_end_state.grc")] + [("start", p) for p in sorted((d / "start").glob("*.grc"))]
        for role, p in files:
            if p.is_file():
                g = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                out.append({"task": d.name, "role": role,
                            "blocks": functional(b.get("id") for b in g.get("blocks") or [])})
    return out


def task_level(rows: list[tuple[str, dict]], block_ids: set[str]) -> dict[str, Any]:
    graphs = _task_graphs()
    starts = defaultdict(list)
    ends = defaultdict(list)
    for g in graphs:
        if g["blocks"]:
            (starts if g["role"] == "start" else ends)[g["blocks"]].append(g["task"])
    same_start, same_end, generic = [], [], []
    for _, r in rows:
        st = start_graph(r, block_ids)
        st_set = functional([st]) if st else ()
        fin = functional(final_blocks(r, st))
        for key, index, out in ((st_set, starts, same_start), (fin, ends, same_end)):
            if key and key in index:
                # one functional block (a lone source or sink) is no specific flowgraph: listed, not counted
                (generic if len(key) < MIN_MATCH_BLOCKS else out).append(
                    {"scenario_id": r["scenario_id"], "blocks": list(key), "tasks": index[key]})
    return {"task_graphs": len(graphs), "same_start_graph": same_start, "same_final_blocks": same_end,
            "one_block_matches": generic}


def signatures(rows: list[tuple[str, dict]], block_ids: set[str]) -> dict[str, Any]:
    from grcbench.tier_f import compose
    audit = compose.audit_items(REPO)
    owners = {tuple(r["v2_signature"]): r["tasks"] for r in audit["reserved"]}
    hits: dict[tuple, list[str]] = defaultdict(list)
    sizes: Counter = Counter()
    for _, r in rows:
        st = start_graph(r, block_ids)
        graph = {"blocks": [{"name": f"b{i}", "id": b, "parameters": {}}
                            for i, b in enumerate(sorted(set(final_blocks(r, st))))], "connections": []}
        sig = compose.signature(graph)
        sizes[len(sig)] += 1
        if sig in owners:
            hits[sig].append(r["scenario_id"])
    return {"cell_sizes": {str(k): v for k, v in sorted(sizes.items())},
            "reserved_cells_hit": [{"cell": list(k), "owners": owners[k], "traces": len(v), "scenario_ids": v}
                                   for k, v in sorted(hits.items())]}


# --------------------------------------------------------------------------- report
def build() -> dict[str, Any]:
    rows = list(traces())
    part = read_json(SPLITS / "block_partition.json")["blocks"]
    block_ids = {b["id"] for b in part}
    test_blocks = {b["id"] for b in part if b.get("split") == "test"}
    return {"inputs": {tier: {"path": str(p.relative_to(NEW_DATA.parent)), "sha256": sha256_file(p)}
                       for tier, p in AGENT_TRACES.items()},
            "model": sorted({r.get("model") for _, r in rows}),
            "harness": harness([r for _, r in rows]), "tool_use": tool_use(rows),
            "exposure": exposure(rows, test_blocks), "test_side_blocks": len(test_blocks),
            "task_level": task_level(rows, block_ids), "signatures": signatures(rows, block_ids)}


def manifest(doc: dict) -> dict[str, Any]:
    ex = doc["exposure"]
    return {"generator": "grcbench.contamination.agent_traces", "inputs": doc["inputs"],
            "catalog_shown_test_blocks": sorted(ex["catalog_shown"]),
            "executed_blocks": sorted(ex["executed_in_code"]),
            "note": ("Blocks whose catalog entry a training trace shows, and blocks a trace built or ran in "
                     "GNU Radio code. Tier-K block questions are execution-derived and avoid executed blocks.")}


def report(doc: dict) -> str:
    h, tu, ex, tl, sg = doc["harness"], doc["tool_use"], doc["exposure"], doc["task_level"], doc["signatures"]
    t1, t2 = tu["tier1"], tu["tier2"]
    runs = Counter()
    for t in (t1, t2):
        runs.update(t["run_flowgraph_replies"])
    L = ["# Qoherent's agent traces (15-18 Oct) against GRC-Bench", "",
         "Generated by `./gb traces` (`grcbench.contamination.agent_traces`). The traces are training data and are "
         "used in full for training; this report checks the benchmark against them (eval plan §6).", "",
         "## 1. The data", "",
         f"- **Tier 1:** {t1['rows']} one-shot traces, {t1['subcategories']} GNU Radio subcategories. Families: "
         + ", ".join(f"{k} {v}" for k, v in t1["families"].items()) + ".",
         f"- **Tier 2:** {t2['rows']} two-turn traces. Families: "
         + ", ".join(f"{k} {v}" for k, v in t2["families"].items()) + ".",
         f"- **Teacher:** {', '.join(doc['model'])}, through GRC-Agent's own harness, on GNU Radio 3.10.9.2 "
         "(the benchmark pins 3.10.12).", "",
         "## 2. Harness", "",
         f"Reference: Qoherent's SFT release (`{h['reference']}`), which the Tier-A conformance check uses.", "",
         "- System prompt: " + ("every trace has the same prompt, identical to the reference apart from the session "
                                "line and the active flowgraph's path." if h["system_prompt_equal"] else
                                f"**{h['distinct_system_prompts']} prompts, different from the reference**.")]
    L.append(f"- Tools: {'one tool set in every trace' if h['distinct_tool_sets'] == 1 else str(h['distinct_tool_sets']) + ' tool sets'}. Only in the traces: "
             f"{', '.join(h['tools_only_in_traces']) or 'none'} (GRC-Agent's output tool, which the evaluation harness "
             f"also offers). Only in the reference: {', '.join(h['tools_only_in_reference']) or 'none'}.")
    for name, diffs in h["schemas_differing"].items():
        L.append(f"- `{name}` schema: " + "; ".join(diffs[:6]) + (" ..." if len(diffs) > 6 else ""))
    L += ["", "## 3. Tool use", "",
          "| | Tier 1 | Tier 2 |", "|---|--:|--:|",
          f"| tool calls per trace (mean / median / max) | {t1['calls_mean']} / {t1['calls_median']} / {t1['calls_max']} "
          f"| {t2['calls_mean']} / {t2['calls_median']} / {t2['calls_max']} |",
          f"| traces with 8-20 tool calls (the Tier-A range) | {t1['rows_with_8_to_20_calls']} | {t2['rows_with_8_to_20_calls']} |",
          f"| validation-feedback retries (GRC-Agent's own) | {t1['validation_feedback_turns']} | {t2['validation_feedback_turns']} |",
          f"| guardrail events | {t1['guardrail_events']} | {t2['guardrail_events']} |",
          f"| run_command calls reading the collection's own folder | {t1['collection_folder_reads']} | {t2['collection_folder_reads']} |",
          f"| tool names outside the schema | {t1['tool_names_outside_schema']} | {t2['tool_names_outside_schema']} |",
          "", f"- Tools never called: Tier 1 {', '.join(t1['tools_never_called'])}; Tier 2 "
          f"{', '.join(t2['tools_never_called'])}.",
          f"- `run_flowgraph`: {sum(runs.values())} calls, every reply an error: "
          + "; ".join(f"\"{k}...\" ({v})" for k, v in runs.items()) + ". No trace ran a flowgraph.", "",
          "## 4. Exposure of benchmark facts", "",
          f"- **Catalog facts.** The catalog entry (a knowledge-base result, or the printed block YAML) of "
          f"**{len(ex['catalog_shown'])} of the {doc['test_side_blocks']} test-side blocks** appears in at least one "
          "trace: its parameter options, port types and port counts. Questions answered by such facts would not be "
          "held out, so Tier K's GNU Radio questions are answered by execution instead (`tier_k/blockgen.py`).",
          f"- **Execution.** No trace ran a flowgraph. {ex['gnuradio_code_commands']} `run_command` calls build "
          f"GNU Radio objects in Python; the blocks they touch ({len(ex['executed_in_code'])}: "
          f"{', '.join(ex['executed_in_code']) or 'none'}) are listed in `splits/agent_traces.json`, and the "
          "block-question generator asks nothing about them.", "",
          "## 5. Task-level check (eval plan §6)", "",
          f"Each trace's start graph (the empty canvas or a one-block scaffold) and final functional blocks against "
          f"the start, reference and golden graphs of every Tier-F and Tier-A task ({tl['task_graphs']} graphs).", "",
          f"- Same start graph: **{len(tl['same_start_graph'])}** traces.",
          f"- Same final functional blocks: **{len(tl['same_final_blocks'])}** traces."]
    L += [f"  - {x['scenario_id']}: {', '.join(x['tasks'])}" for x in (tl["same_start_graph"] + tl["same_final_blocks"])[:20]]
    if tl["one_block_matches"]:
        L += [f"- Not counted: {len(tl['one_block_matches'])} trace(s) share only one functional block with a task "
              "graph, which is no specific flowgraph (a match needs at least "
              f"{MIN_MATCH_BLOCKS} functional blocks in common):"]
        L += [f"  - {x['scenario_id']} ({', '.join(x['blocks'])}): {', '.join(x['tasks'])}" for x in tl["one_block_matches"][:20]]
    L += ["", "## 6. Reserved flowgraph signatures", "",
          "Cell sizes of the traces' final graphs: " + ", ".join(f"{k} classes: {v}" for k, v in sg["cell_sizes"].items()) + "."]
    for hit in sg["reserved_cells_hit"]:
        L.append(f"- Cell `{'|'.join(hit['cell'])}` (reserved by {', '.join(hit['owners'])}): {hit['traces']} traces, "
                 "each a lone block wired to null sources and sinks.")
    L += ["", "These traces share a one-class cell but not a task: none has a task's goal or start graph (section 5). "
          "For agent sessions the eval plan's rule is the task-level check; the reserved cells remain the rule for "
          "synthetic training flowgraphs (`handoff/training_team/flowgraph_composed_chains.md`).", ""]
    return "\n".join(L)


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="gb traces", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO, help="root folder for the outputs (default: the repo)")
    args = ap.parse_args(argv)
    missing = [str(p) for p in AGENT_TRACES.values() if not p.is_file()]
    if missing:
        print("agent traces not found: " + ", ".join(missing))
        return 2
    doc = build()
    write_json(args.out / OUT_JSON.relative_to(REPO), doc)
    write_json(args.out / OUT_MANIFEST.relative_to(REPO), manifest(doc))
    write_text(args.out / OUT_REPORT.relative_to(REPO), report(doc))
    ex = doc["exposure"]
    print(f"traces: {doc['tool_use']['tier1']['rows']} + {doc['tool_use']['tier2']['rows']}; catalog facts of "
          f"{len(ex['catalog_shown'])}/{doc['test_side_blocks']} test-side blocks shown; "
          f"{len(ex['executed_in_code'])} blocks built in code; task-level overlaps "
          f"{len(doc['task_level']['same_start_graph']) + len(doc['task_level']['same_final_blocks'])}")
    for p in (OUT_REPORT, OUT_JSON, OUT_MANIFEST):
        print("wrote", args.out / p.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""The Tier-F setting of the Tier-A runner: "one lookup + one edit" (build plan, Decision 16 Oct).

GRC-Agent never lets the model write a .grc file (its file tools refuse .grc writes; the model edits
only through change_graph), so a Tier-F sample is a short episode in the SAME harness as Tier A,
with IDENTICAL inputs: GRC-Agent's real executor (agent_build), its real system prompt, the 22 tool
schemas, the per-turn active-flowgraph note and the model's own chat template.

The episode
  start graph  the task's start.grc (modify / repair / parameter-tune), converted from the compact
               form to GRC's full file format by GRC's own writer and opened as the active page
               <run>/project/flowgraph.grc (the original is never edited). Build tasks start from
               an empty graph holding only a `samp_rate` variable (rule below).
  prefill      the history already holds the user's instruction and one FULL inspect_graph call
               (no arguments, i.e. no `targets`) with id `prefill_call_id(task id)`. The runner
               continues that history with no new prompt, so pydantic-ai executes the pending call
               through GRC-Agent's real tool and the whole capability chain (the 4K tool-output cap
               included) and sends the model exactly the request it would have received had it made
               the call itself (tests/test_tier_a_runner.py checks this byte for byte on the wire).
  budget       at most ONE query_knowledge call (query, domain and k are the model's) and ONE
               change_graph call, the first of each in the order the model emitted them. The edit
               is applied atomically by GRC-Agent's own change_graph (approved as the runner
               approves every call); a rejected batch rolls back and there is no retry. Every
               other call (a second lookup or edit, inspect_graph, generate_python, run_flowgraph,
               get_run_log, save_*, file, shell and web tools) gets BLOCKED_TEXT instead of
               running. The budget is not announced up front: the instruction is the task's own.
               Calls in one reply all run, so a lookup in the same reply as the edit cannot inform
               it; that is the model's choice and is recorded (`edit_same_reply_as_lookup`).
  end          the episode ends as soon as the in-budget change_graph call has been processed
               (applied, rejected by GRC-Agent, or refused at argument validation), or when the
               model answers without editing, or at the cap of 3 model requests (TIER_F_REQUESTS;
               the prefilled call is not a model request). No applied change_graph = failure.

Grading (host side, `grade`)
  V1       the in-budget change_graph call is well formed: its arguments, as the model emitted them
           (before GRC-Agent's JSON repair), are a JSON object valid against the change_graph tool
           schema (caps.schema_check).
  V2-V6    the final graph (exported from GRC's memory with GRC's writer to <run>/final.grc) goes
           through grcbench.tier_f.ladder.check(final_text, task), graph_checks included.
  rejected a rejected edit fails at the rung its error codes map to (REJECTION_RUNG: an unknown
           block or port at V2, a validation error at V3); arguments GRC-Agent refused to validate
           fail at V1.
  A sample passes when V1-V6 all pass.

Build-task start graph rule (`start_graph_text`): options {id: default, generate_options: no_gui,
output_language: python} plus one variable `samp_rate`. Its value is the task's sample rate when
the instruction states one, i.e. the FIRST match in the instruction of "sample/sampling rate
[of|is|at|=|:] <number> [k|M]Hz|S/s|sps|samples per second", "<number> [k|M]Hz sample rate" or
"<number> samples per second" (k = 1e3, M = 1e6; written as an integer when integral); otherwise
32000, GRC's default for File > New. A task record may override the rule with
`tier_a_setting.start_samp_rate`. no_gui (instead of GRC's qt_gui default) matches every Tier-F
reference graph and the ladder's headless run.

Recorded per sample (EpisodeResult.tier_f): the prefilled call, the start-graph rule, the lookup
(used, raw arguments, query/domain/k, schema validity, retrieved block ids, recall of the reference's
block ids), the edit (called, raw arguments before JSON repair, well formed, accepted / rejected,
GRC-Agent's error codes, rejection rung), whether lookup and edit shared a reply, every blocked
call, the model requests used and the final graph; after grading also the rungs and `passed`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..common.canary import CANARY_GUID
from ..common.io import read_json, sha256_file, write_json, write_text
from ..common.paths import ITEMS
from ..common.schemas import validate as schema_validate
from ..runlog import log as runlog
from . import isolation
from .caps import Caps, EpisodeMeter
from .models import ModelSpec, as_spec
from .task import Task

BLOCKED_TEXT = "not available in this task; submit your edit with change_graph"
TIER_F_REQUESTS = 3
TIER_F_EPISODE_SECONDS = 900.0
START_GRAPH_NAME = "flowgraph.grc"
DEFAULT_SAMP_RATE = "32000"                      # GRC's File > New default
LOOKUP, EDIT, INSPECT = "query_knowledge", "change_graph", "inspect_graph"
BUDGETED = (LOOKUP, EDIT)
STATE_DIR = "tier_f"                             # <run>/tier_f/: start_compact.grc, start_full.grc
FINAL_GRAPH = "final.grc"
PREFILL_ARGS = "{}"                              # a full inspection: no `targets`

# change_graph error code (GRC-Agent adapter/graph.py) -> the rung a rejected edit fails at.
REJECTION_RUNG = {
    "invalid_request": "V1", "invalid_connection_format": "V1", "args_invalid": "V1",
    "block_type_not_found": "V2", "block_not_found": "V2", "connection_not_found": "V2",
    "add_connection_failed": "V2", "connection_silently_dropped": "V2", "duplicate_block_name": "V2",
    "remove_block_failed": "V2",
    "gnu_validation": "V3", "set_param_failed": "V3", "update_params_failed": "V3", "set_state_failed": "V3",
    "update_states_failed": "V3", "auto_resolve_failed": "V3", "mutation_failed": "V3",
}
HARNESS_CODES = ("save_failed", "rollback_failed")
RUNGS = ("V1", "V2", "V3", "V4", "V5", "V6")

_NUM = r"(?P<num>\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?)"
_UNIT = r"(?P<unit>[kKM]?)(?:Hz|S/s|sps|Sps)\b"
_RATE_PATTERNS = (
    re.compile(r"sampl(?:e|ing)[ -]rate\s*(?:of|is|at|=|:)?\s*(?:about\s+)?" + _NUM + r"\s*" + _UNIT, re.IGNORECASE),
    re.compile(r"sampl(?:e|ing)[ -]rate\s*(?:of|is|at|=|:)?\s*(?:about\s+)?" + _NUM + r"\s*(?P<unit>)samples?\s+per\s+second",
               re.IGNORECASE),
    re.compile(_NUM + r"\s*" + _UNIT + r"\s+sampl(?:e|ing)[ -]rate", re.IGNORECASE),
    re.compile(_NUM + r"\s*(?P<unit>)samples?\s+per\s+second", re.IGNORECASE),
)
_CODE = re.compile(r"""['"]code['"]\s*:\s*['"]([A-Za-z_]+)['"]""")


class TierFTaskError(ValueError):
    """A Tier-F task directory cannot be run in this setting."""


# --------------------------------------------------------------------------------------------
# Start graph and prefill (pure functions, shared by host and episode)
# --------------------------------------------------------------------------------------------

def stated_sample_rate(instruction: str) -> str | None:
    """The sample rate an instruction states (module docstring rule), as GRC parameter text."""
    best: tuple[int, str] | None = None
    for pattern in _RATE_PATTERNS:
        m = pattern.search(instruction or "")
        if m is None:
            continue
        unit = (m.group("unit") or "").strip()
        value = float(m.group("num").replace(",", "")) * {"k": 1e3, "K": 1e3, "M": 1e6}.get(unit, 1.0)
        text = str(int(round(value))) if abs(value - round(value)) < 1e-9 else repr(value)
        if best is None or m.start() < best[0]:
            best = (m.start(), text)
    return best[1] if best else None


def start_graph_text(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """(compact .grc text of the start graph, the rule record) for a Tier-F task record with `_dir`."""
    import yaml
    start = task.get("start_grc")
    if start:
        path = Path(task["_dir"]) / start
        return path.read_text(encoding="utf-8"), {"kind": "task_start_graph", "file": start,
                                                 "sha256": sha256_file(path)}
    override = (task.get("tier_a_setting") or {}).get("start_samp_rate")
    stated = stated_sample_rate(task.get("instruction", ""))
    value = str(override) if override is not None else (stated or DEFAULT_SAMP_RATE)
    source = "task override" if override is not None else ("instruction" if stated else "GRC default")
    data = {"options": {"parameters": {"id": "default", "generate_options": "no_gui", "output_language": "python"}},
            "blocks": [{"name": "samp_rate", "id": "variable", "parameters": {"value": value}}],
            "connections": [], "metadata": {"file_format": 1}}
    return yaml.safe_dump(data, sort_keys=False), {"kind": "empty_build_graph", "samp_rate": value,
                                                    "samp_rate_source": source}


def prefill_call_id(task_id: str) -> str:
    """The prefilled inspect_graph call's id: 32 alphanumerics derived from the task id (llama.cpp's
    server gives tool calls 32 random alphanumerics; this keeps the format and makes it repeatable)."""
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    n = int.from_bytes(hashlib.sha256(f"grcbench tier-f prefill {task_id}".encode()).digest(), "big")
    out = []
    for _ in range(32):
        n, r = divmod(n, len(alphabet))
        out.append(alphabet[r])
    return "".join(out)


def prefill_messages(instruction: str, call_id: str) -> list[Any]:
    """The prefilled history: the instruction, then a model response holding one full inspect_graph
    call that has not been answered yet (the runner's first step executes it for real)."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
    return [ModelRequest(parts=[UserPromptPart(instruction)]),
            ModelResponse(parts=[ToolCallPart(INSPECT, PREFILL_ARGS, tool_call_id=call_id)])]


def tier_f_caps(setting: str = "benchmark", **overrides: Any) -> Caps:
    """The caps of the Tier-F setting: the setting's caps with 3 model requests."""
    base: dict[str, Any] = {"requests": TIER_F_REQUESTS, "episode_seconds": TIER_F_EPISODE_SECONDS}
    base.update(overrides)
    return Caps.for_setting(setting, **base)


# --------------------------------------------------------------------------------------------
# The budget (a HarnessCapability policy)
# --------------------------------------------------------------------------------------------

@dataclass
class TierFBudget:
    """Which tool calls run in the Tier-F setting (module docstring)."""

    prefill_id: str
    meter: EpisodeMeter
    blocked_text: str = BLOCKED_TEXT
    allowed: dict[str, str] = field(default_factory=dict)     # tool -> the in-budget call id
    decisions: dict[str, bool] = field(default_factory=dict)  # call id -> blocked?

    def _first_calls(self, messages: list[Any]) -> None:
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        for msg in messages:
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if (isinstance(part, ToolCallPart) and part.tool_name in BUDGETED
                            and part.tool_call_id != self.prefill_id):
                        self.allowed.setdefault(part.tool_name, part.tool_call_id)

    def blocked(self, ctx: Any, call: Any) -> bool:
        cid = call.tool_call_id
        if cid not in self.decisions:
            self._first_calls(list(getattr(ctx, "messages", None) or []))
            self.decisions[cid] = not (cid == self.prefill_id or self.allowed.get(call.tool_name) == cid)
        return self.decisions[cid]

    def edit_processed(self) -> bool:
        """The in-budget change_graph call has been applied, rejected or refused at validation."""
        cid = self.allowed.get(EDIT)
        rec = self.meter.calls.get(cid) if cid else None
        return bool(rec and (rec.get("execution") or rec.get("validation") == "invalid"))


# --------------------------------------------------------------------------------------------
# Episode process side
# --------------------------------------------------------------------------------------------

def task_from_spec(spec: dict[str, Any]) -> Task:
    """The Task the episode runs: the Tier-F record, its start graph opened as flowgraph.grc, no
    scripted user (a Tier-F sample is one turn)."""
    record, task_dir = spec["task"]["record"], Path(spec["task"]["task_dir"])
    return Task(id=record["id"], split=record["split"], version=int(record["version"]),
                category=record["family"], instruction=record["instruction"],
                start_state={"active_graph": START_GRAPH_NAME}, scripted_user={"max_turns": 0},
                acceptance={}, task_dir=task_dir, record=record)


def materialise_start_graph(spec: dict[str, Any], lay: isolation.RunLayout) -> Path:
    """Load the compact start graph with GRC (GRC-Agent's platform) and write it in GRC's full file
    format to <run>/project/flowgraph.grc; a copy goes to <run>/tier_f/start_full.grc."""
    import yaml

    from grc_agent.adapter import get_platform

    from ..tier_f._worker import _normalise
    text = Path(spec["tier_f"]["start_compact"]).read_text(encoding="utf-8")
    platform = get_platform()
    fg = platform.make_flow_graph()
    fg.import_data(_normalise(yaml.safe_load(text)))
    fg.rewrite()
    fg.validate()
    dest = lay.project / START_GRAPH_NAME
    platform.save_flow_graph(str(dest), fg)
    shutil.copyfile(dest, lay.run_dir / STATE_DIR / "start_full.grc")
    return dest


async def conversation(agent: Any, deps: Any, task: Task, state: Any, meter: EpisodeMeter, capability: Any,
                       conversation_id: str, t0: float) -> str:
    """The Tier-F episode: the prefilled history, continued once (approvals included) until the edit
    has been processed, the model answers, or a cap is hit. Returns the end reason."""
    from .runner import HarnessStopped, _agent_turn
    from .user import reply_text
    budget: TierFBudget = capability.policy
    state.history = prefill_messages(task.instruction, budget.prefill_id)
    turn: dict[str, Any] = {"index": 0, "source": "instruction", "user_message": task.instruction,
                            "prefilled_tool_calls": [budget.prefill_id], "requests_before": meter.usage.requests,
                            "started_s": round(time.monotonic() - t0, 3)}
    state.turns.append(turn)
    output = await _agent_turn(agent, deps, None, state, meter, capability, conversation_id, turn,
                               stop=budget.edit_processed)
    stopped = isinstance(output, HarnessStopped)
    turn.update(requests_after=meter.usage.requests, agent_reply=reply_text(state.history, None if stopped else output),
                output_kind=type(output).__name__, seconds=round(time.monotonic() - t0 - turn["started_s"], 3))
    if meter.hits:
        return f"cap:{meter.hits[0].cap}"
    if budget.edit_processed():
        rec = meter.calls[budget.allowed[EDIT]]
        return "tier_f_edit_applied" if rec.get("execution") == "ok" else "tier_f_edit_rejected"
    return "agent_finished"


def _tool_results(history: list[Any]) -> dict[str, Any]:
    """tool_call_id -> the ToolReturnPart / RetryPromptPart answering it."""
    from pydantic_ai.messages import ModelRequest, RetryPromptPart, ToolReturnPart
    out: dict[str, Any] = {}
    for msg in history:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, (ToolReturnPart, RetryPromptPart)) and part.tool_call_id:
                    out[part.tool_call_id] = part
    return out


def _response_of(history: list[Any]) -> dict[str, int]:
    """tool_call_id -> index of the model response that made the call."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    out: dict[str, int] = {}
    for i, msg in enumerate(history):
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    out[part.tool_call_id] = i
    return out


def _block_ids(obj: Any) -> list[str]:
    """Every "block_id" value in a (JSON) structure, in order, without repeats."""
    found: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "block_id" and isinstance(v, str) and v not in found:
                    found.append(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(obj)
    return found


def _content_text(part: Any) -> str:
    content = getattr(part, "content", None)
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _parsed_args(raw: str) -> Any:
    try:
        return json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, ValueError):
        return None


def episode_record(spec: dict[str, Any], state: Any, meter: EpisodeMeter, lay: isolation.RunLayout,
                   proxy: Any, budget: TierFBudget) -> dict[str, Any]:
    """The Tier-F sample record (module docstring), and the final graph written to <run>/final.grc
    from GRC's memory with GRC's own writer. Runs in the episode process before the proxy closes."""
    from pydantic_ai.messages import RetryPromptPart
    prefill = spec["tier_f"]["prefill_id"]
    history = list(state.history)
    results = _tool_results(history)
    response_of = _response_of(history)
    calls = [c for c in meter.calls.values() if not c.get("prefilled")]
    allowed = {tool: cid for tool, cid in budget.allowed.items() if cid in meter.calls}

    final_path = lay.run_dir / FINAL_GRAPH
    page = proxy.page if proxy is not None else None
    if page is not None and page.flow_graph is not None:
        page.flow_graph.parent_platform.save_flow_graph(str(final_path), page.flow_graph)

    lookup: dict[str, Any] = {"used": LOOKUP in allowed}
    if LOOKUP in allowed:
        rec = meter.calls[allowed[LOOKUP]]
        args = _parsed_args(rec["raw_args"]) if rec["raw_args_kind"] == "str" else json.loads(rec["raw_args"] or "{}")
        part = results.get(rec["tool_call_id"])
        retrieved: list[str] = []
        if part is not None and not isinstance(part, RetryPromptPart):
            try:
                retrieved = _block_ids(json.loads(_content_text(part)))
            except (json.JSONDecodeError, ValueError):
                retrieved = []
        lookup.update(tool_call_id=rec["tool_call_id"], raw_args=rec["raw_args"], schema_valid=rec["schema_valid"],
                      query=(args or {}).get("query") if isinstance(args, dict) else None,
                      domain=(args or {}).get("domain") if isinstance(args, dict) else None,
                      k=(args or {}).get("k") if isinstance(args, dict) else None,
                      outcome=rec.get("execution") or rec.get("validation"),
                      retrieved_block_ids=retrieved, truncated=bool(rec.get("truncated")),
                      request=rec.get("request"))

    edit: dict[str, Any] = {"called": EDIT in allowed}
    if EDIT in allowed:
        rec = meter.calls[allowed[EDIT]]
        part = results.get(rec["tool_call_id"])
        accepted = rec.get("execution") == "ok"
        codes: list[str] = []
        message = None
        if part is not None and not accepted:
            message = _content_text(part)[:4000]
            if rec.get("validation") == "invalid":
                codes = ["args_invalid"]
            else:
                codes = list(dict.fromkeys(_CODE.findall(message)))
        edit.update(tool_call_id=rec["tool_call_id"], raw_args=rec["raw_args"], raw_args_kind=rec["raw_args_kind"],
                    json_object=rec["json_object"], well_formed=rec["schema_valid"], schema_errors=rec["schema_errors"],
                    repair_changed_args=rec.get("repair_changed_args"), validation=rec.get("validation"),
                    execution=rec.get("execution"), accepted=accepted,
                    rejected=bool(rec.get("execution") in ("retry", "failed", "error")
                                  or rec.get("validation") == "invalid"),
                    error_codes=codes, error_message=message, request=rec.get("request"))
        if accepted and part is not None:
            try:
                edit["result"] = json.loads(_content_text(part))
            except (json.JSONDecodeError, ValueError):
                edit["result"] = _content_text(part)[:2000]
    same = (LOOKUP in allowed and EDIT in allowed
            and response_of.get(allowed[LOOKUP]) == response_of.get(allowed[EDIT]))
    blocked = [{"tool": c["tool"], "tool_call_id": c["tool_call_id"], "request": c.get("request")}
               for c in calls if c.get("blocked")]
    return {
        "setting": "one lookup + one edit", "prefill": {"tool": INSPECT, "args": PREFILL_ARGS, "tool_call_id": prefill,
                                                        "truncated": bool(meter.calls.get(prefill, {}).get("truncated"))},
        "start": spec["tier_f"]["start"], "lookup": lookup, "edit": edit,
        "edit_same_reply_as_lookup": bool(same), "blocked_calls": blocked,
        "model_requests": meter.usage.requests, "request_cap": meter.caps.requests,
        "final_graph": str(final_path) if final_path.is_file() else None,
        "final_graph_sha256": sha256_file(final_path) if final_path.is_file() else None,
    }


# --------------------------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------------------------

def load_tier_f_task(task_dir: Path | str) -> dict[str, Any]:
    """A Tier-F task record (taskio.load: the record plus `_dir`), checked against the TIER_F
    schema. Unit-test fixtures (source.kind "test_fixture", never benchmark items) may omit the
    canary; everything else the schema requires is still checked."""
    from ..tier_f import taskio
    task_dir = Path(task_dir).resolve()
    if not (task_dir / "task.json").is_file():
        raise TierFTaskError(f"{task_dir} has no task.json")
    task = taskio.load(task_dir)
    record = {k: v for k, v in task.items() if not k.startswith("_")}
    if (record.get("source") or {}).get("kind") == "test_fixture":
        from ..common.canary import CANARY
        record.setdefault("canary", CANARY)
    errors = schema_validate(record)
    if task.get("tier") != "F":
        errors.append("not a Tier-F record")
    for name in filter(None, (task.get("start_grc"), task.get("reference_grc"))):
        if not (task_dir / name).is_file():
            errors.append(f"{name} is missing")
    if CANARY_GUID in task.get("instruction", ""):
        errors.append("instruction contains the canary")
    start = task.get("start_grc")
    if start and (task_dir / start).is_file() and CANARY_GUID in (task_dir / start).read_text(encoding="utf-8"):
        errors.append(f"{start} contains the canary (the model sees it through inspect_graph)")
    if errors:
        raise TierFTaskError(f"{task_dir.name}: " + "; ".join(errors))
    return task


def reference_block_ids(task: dict[str, Any]) -> dict[str, list[str]]:
    """Block ids of the task's reference graph (all, and those its start graph lacks)."""
    import yaml
    ref = yaml.safe_load((Path(task["_dir"]) / task["reference_grc"]).read_text(encoding="utf-8")) or {}
    ids = sorted({b.get("id") for b in ref.get("blocks") or [] if b.get("id")})
    start_ids: set[str] = set()
    if task.get("start_grc"):
        start = yaml.safe_load((Path(task["_dir"]) / task["start_grc"]).read_text(encoding="utf-8")) or {}
        start_ids = {b.get("id") for b in start.get("blocks") or [] if b.get("id")}
    else:
        start_ids = {"variable"}
    return {"all": ids, "new": [i for i in ids if i not in start_ids]}


def rejection_rung(codes: list[str]) -> str | None:
    """The lowest rung any of a rejected edit's error codes maps to (None: harness failure only)."""
    rungs = [REJECTION_RUNG[c] for c in codes if c in REJECTION_RUNG]
    if rungs:
        return min(rungs)
    if any(c in HARNESS_CODES for c in codes):
        return None
    return "V3"                                       # an unlisted GRC-Agent code: a failed batch


def grade(record: dict[str, Any], task: dict[str, Any], *, ladder_check: Callable[..., dict] | None = None
          ) -> dict[str, Any]:
    """V1 (well-formed edit) + V2-V6 (ladder on the final graph); returns the grading record."""
    from ..tier_f import ladder
    check = ladder_check or ladder.check
    edit = record.get("edit") or {}
    out: dict[str, Any] = {"rungs": {}, "first_failed": None, "highest_passed": None, "passed": False,
                           "ladder": None, "failure": None}
    refs = reference_block_ids(task)
    lookup = record.get("lookup") or {}
    if lookup.get("used"):
        got = set(lookup.get("retrieved_block_ids") or [])
        lookup["reference_recall"] = (round(len(got & set(refs["all"])) / len(refs["all"]), 4) if refs["all"] else None)
        lookup["new_block_recall"] = (round(len(got & set(refs["new"])) / len(refs["new"]), 4) if refs["new"] else None)
    out["reference_block_ids"] = refs
    if not edit.get("called"):
        out["rungs"]["V1"] = {"passed": False, "errors": ["no change_graph call"]}
        out.update(first_failed="V1", failure="no_edit")
        return out
    out["rungs"]["V1"] = {"passed": bool(edit.get("well_formed")), "errors": edit.get("schema_errors") or []}
    if edit.get("accepted"):
        final = record.get("final_graph")
        if not final or not Path(final).is_file():
            out.update(first_failed="V2", failure="final_graph_missing")
            out["rungs"]["V2"] = {"passed": False, "errors": ["final graph missing"]}
            return out
        lad = check(Path(final).read_text(encoding="utf-8"), task)
        out["ladder"] = {k: lad.get(k) for k in ("highest_passed", "first_failed", "hallucinated_blocks")}
        for rung in RUNGS[1:]:
            if rung in lad.get("rungs", {}):
                out["rungs"][rung] = lad["rungs"][rung]
        if not lad["rungs"].get("V1", {}).get("passed", False):
            out["rungs"]["V2"] = {"passed": False, "errors": ["final graph does not parse: "
                                                              + "; ".join(lad["rungs"].get("V1", {}).get("errors", []))]}
    else:
        rung = rejection_rung(edit.get("error_codes") or []) or "harness"
        out["rejection_rung"] = rung
        if rung in RUNGS and rung != "V1":
            out["rungs"][rung] = {"passed": False, "errors": [f"change_graph rejected: {', '.join(edit.get('error_codes') or [])}"]}
        elif rung == "V1":
            out["rungs"]["V1"] = {"passed": False, "errors": (out["rungs"]["V1"]["errors"]
                                                              + [f"change_graph rejected: {', '.join(edit.get('error_codes') or [])}"])}
        out["failure"] = "edit_rejected"
    highest = None
    first_failed = None
    for rung in RUNGS:
        res = out["rungs"].get(rung)
        if res is None or res.get("passed") is not True:
            first_failed = rung if (res is not None or first_failed is None) else first_failed
            break
        highest = rung
    out["highest_passed"] = highest
    out["first_failed"] = None if highest == "V6" else first_failed
    out["passed"] = highest == "V6"
    if not out["passed"] and out["failure"] is None:
        out["failure"] = f"failed_{out['first_failed']}"
    return out


def run_dir_for(run_root: Path | None, config_id: str, task_id: str, setting: str, sample: int) -> Path:
    from .runner import _safe
    root = Path(run_root) if run_root is not None else isolation.RUNS_ROOT
    return root / _safe(config_id) / _safe(task_id) / f"tier_f-{setting}-s{sample:02d}"


def run_tier_f_sample(task_dir: Path | str, model_factory: ModelSpec | str | Callable[..., Any],
                      run_root: Path | None = None, *, config_id: str = "adhoc", sample: int = 0,
                      setting: str = "benchmark", caps: Caps | None = None, mode: str = "evaluation",
                      reason: str = "", grade_sample: bool = True, invalid_gate: str = "pinned",
                      run_log_entry: dict[str, Any] | None = None, log_path: Path = runlog.LOG,
                      run_list: Path = runlog.RUN_LIST, items_root: Path = ITEMS,
                      include_thinking: bool = True) -> Any:
    """Run one Tier-F sample of the task in `task_dir` (module docstring); returns the EpisodeResult
    with `tier_f` filled in (graded unless grade_sample=False). The run-log guard applies as for
    Tier A (test split: declared configuration id, logged before the run starts)."""
    from .runner import (
        HARNESS_VERSION, PROCESS_MARGIN, RESULT_FILE, SPEC_FILE, STDERR_FILE, TRACE_FILE, EpisodeResult,
        _failed_result, run_id_for,
    )
    task = load_tier_f_task(task_dir)
    spec_model = as_spec(model_factory)
    caps = caps or tier_f_caps(setting)
    settings = {"setting": setting, "tier_f_setting": "one lookup + one edit", "caps": caps.to_dict(),
                "invalid_gate": invalid_gate, "approvals": "all approved (GRC-Agent yolo)",
                "streaming": spec_model.streams(),
                "harness": HARNESS_VERSION}
    if run_log_entry is None:
        run_log_entry = runlog.start_run(split=task["split"], config_id=config_id, model=spec_model.describe(),
                                         settings=settings, items=[task["id"]], item_splits=[task["split"]],
                                         reason=reason or f"Tier-F sample {task['id']}", tier="F", mode=mode,
                                         log_path=log_path, run_list=run_list, items_root=items_root)
    run_dir = isolation.check_run_dir(run_dir_for(run_root, config_id, task["id"], setting, sample))
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    with contextlib.ExitStack() as stack:
        binds: list[Path] = []
        if spec_model.relay is not None:                  # a hosted API (runner.py, "a relay")
            from .relay import LOCAL_BASE_URL, Relay
            relay = stack.enter_context(Relay(spec_model.relay["upstream"], spec_model.relay["api_key_env"],
                                              allow_http_loopback=bool(spec_model.relay.get("allow_http_loopback"))))
            spec_model = dataclasses.replace(spec_model, base_url=LOCAL_BASE_URL, uds_path=str(relay.socket_path))
            binds.append(relay.dir)
        env = isolation.make_run_env(run_dir, base_url=spec_model.base_url, model_name=spec_model.model_name)
        compact, start_rule = start_graph_text(task)
        write_text(run_dir / STATE_DIR / "start_compact.grc", compact)
        record = {k: v for k, v in task.items() if not k.startswith("_")}
        episode = {"kind": "tier_f", "run_id": run_id_for(task["id"], config_id, f"tier_f-{setting}", sample),
                   "task": {"record": record, "task_dir": task["_dir"]}, "model": spec_model.to_dict(),
                   "setting": setting, "caps": caps.to_dict(), "config_id": config_id, "sample": sample, "mode": mode,
                   "invalid_gate": invalid_gate, "run_log_entry": run_log_entry,
                   "tier_f": {"prefill_id": prefill_call_id(task["id"]), "start": start_rule,
                              "start_compact": str(run_dir / STATE_DIR / "start_compact.grc")}}
        write_json(run_dir / SPEC_FILE, episode)
        t0 = time.monotonic()
        proc = isolation.run_episode_process(run_dir, "grcbench.tier_a.runner",
                                             ["episode", "--spec", str(run_dir / SPEC_FILE)], env,
                                             timeout=caps.episode_seconds + PROCESS_MARGIN,
                                             network=spec_model.network(), extra_ro_binds=binds)
    write_text(run_dir / STDERR_FILE, proc.stderr or "")
    process = {"returncode": proc.returncode, "timed_out": proc.timed_out, "seconds": round(time.monotonic() - t0, 3),
               "network": spec_model.network(), "stderr_tail": (proc.stderr or "")[-4000:]}
    if (run_dir / RESULT_FILE).is_file():
        result = EpisodeResult.from_dict(read_json(run_dir / RESULT_FILE))
    else:
        result = _failed_result(episode, run_dir, task_from_spec(episode), spec_model, caps, proc.timed_out, process)
        result.tier = "F"
    result.process = process
    result.run_log_entry = run_log_entry
    if result.tier_f is not None and grade_sample and result.valid and "record_error" not in result.tier_f:
        if result.cap_failure:
            result.tier_f["grade"] = {"passed": False, "failure": f"cap:{result.caps_hit[0]['cap'] if result.caps_hit else '?'}",
                                      "rungs": {}, "first_failed": None, "highest_passed": None}
        else:
            result.tier_f["grade"] = grade(result.tier_f, task)
    write_json(run_dir / RESULT_FILE, result.to_dict())
    from .trace import export_run_dir, write_trace
    passed = (result.tier_f or {}).get("grade", {}).get("passed") if grade_sample else None
    write_trace(run_dir / TRACE_FILE, [export_run_dir(run_dir, include_thinking=include_thinking,
                                                      acceptance_passed=passed)])
    return result


def run_tier_f_samples(task_dirs: list[Path | str], model_factory: ModelSpec | str | Callable[..., Any], *,
                       config_id: str, reason: str, samples: int = 1, setting: str = "benchmark",
                       run_root: Path | None = None, mode: str = "evaluation", caps: Caps | None = None,
                       log_path: Path = runlog.LOG, run_list: Path = runlog.RUN_LIST, items_root: Path = ITEMS,
                       on_result: Callable[[Any], None] | None = None) -> list[Any]:
    """`samples` Tier-F samples of every task, with one guard and one run-log entry for the batch."""
    tasks = [load_tier_f_task(d) for d in task_dirs]
    if not tasks:
        return []
    spec_model = as_spec(model_factory)
    caps = caps or tier_f_caps(setting)
    splits = sorted({t["split"] for t in tasks})
    entry = runlog.start_run(split=splits[0] if len(splits) == 1 else "+".join(splits), config_id=config_id,
                             model=spec_model.describe(),
                             settings={"setting": setting, "tier_f_setting": "one lookup + one edit",
                                       "caps": caps.to_dict(), "samples": samples},
                             items=[t["id"] for t in tasks], item_splits=splits, reason=reason, tier="F", mode=mode,
                             log_path=log_path, run_list=run_list, items_root=items_root)
    out = []
    for t in tasks:
        for k in range(samples):
            res = run_tier_f_sample(t["_dir"], spec_model, run_root, config_id=config_id, sample=k, setting=setting,
                                    caps=caps, mode=mode, reason=reason, run_log_entry=entry or {"logged": False},
                                    log_path=log_path, run_list=run_list, items_root=items_root)
            out.append(res)
            if on_result is not None:
                on_result(res)
    return out


"""Export a Tier-A episode as one row of Qoherent's SFT row format (proposal §4.4; eval plan §7.4).

The proposal: collected sessions "are exported in the row format of Qoherent's v0.1.0 release (task,
system prompt, messages with tool calls, teacher, usage, quality), so they merge with the existing
12". Benchmark episodes are written in the same format, so the traces released with the benchmark
(eval plan §7.4 "All raw outputs are released") read like the SFT rows.

Row = the release's keys, in the release's order, then GRC-Bench's metadata under "grcbench":
    session_id, task_id, category, verdict, teacher, system, tools, messages, usage, grcbench

How each field is made
  messages   `map_messages`: the recovered exporter's rules (experiments/dataset_collection/
             export.py map_messages L52-87, read-only copy in ~/grc-bench-cache/reference), plus four
             conventions every row of the v0.1.0 release shows and the recovered file lacks (the
             release was assembled by a later exporter; checked on all 32 released rows by
             tests/test_tier_a_runner.py):
               R1 tool calls carry their "id", and tool messages a "tool_call_id" before "name";
               R2 every assistant message has "content" ("" when it has only tool calls);
               R3 a retry prompt is a tool message too: its text (ModelRetry message) or, for a
                  validation error, the JSON list of errors; "name" is the tool, or "unknown" when
                  the retry is not about a tool call (e.g. "Please return text or include your
                  response in a tool call.");
               R4 key order role, content, tool_calls, thinking / role, tool_call_id, name, content.
             The export.py rules kept as they are: user prompts only when text; a tool return's
             non-text content as json.dumps(..., default=str); thinking joined and stripped, kept
             in a parallel "thinking" field (include_thinking); an assistant message with neither
             text nor tool calls is dropped unless it carries thinking; "arguments" is str(args),
             i.e. the argument string exactly as the model emitted it (a dict, which only scripted
             test models produce, is written as JSON, as the release README requires).
  system     the first request's instructions (export.py L178-186).
  tools      the episode's tool snapshot (agent_build.freeze_tools, = collect.py freeze_tools).
  usage      export.py sum_usage: input_tokens, output_tokens, requests (+ reasoning_tokens).
  teacher    {"provider", "model"} of the model under test.
  verdict    "completed" / "failed_acceptance" when the episode has been graded; otherwise
             "ungraded", "cap_failure", "agent_error" or "infrastructure_abort" (Qoherent's verdicts
             where one fits; GRC-Bench's caps have no counterpart there).
Tool results in `messages` are what the model saw, i.e. after the 4K tool-output cap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..common.io import read_json

ROW_KEYS = ("session_id", "task_id", "category", "verdict", "teacher", "system", "tools", "messages", "usage")


def _json_default(obj: Any) -> str:
    return str(obj)


def _args_text(args: Any) -> str:
    if isinstance(args, str):
        return args
    if args is None:
        return "{}"
    return json.dumps(args, ensure_ascii=False)


def _content_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, default=_json_default)


def map_messages(messages: list[Any], *, include_thinking: bool = True) -> list[dict[str, Any]]:
    """pydantic-ai ModelMessages -> the release's OpenAI-style messages (module docstring)."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        RetryPromptPart,
        TextPart,
        ThinkingPart,
        ToolCallPart,
        ToolReturnPart,
    )
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if part.__class__.__name__ == "UserPromptPart" and isinstance(getattr(part, "content", None), str):
                    out.append({"role": "user", "content": part.content})
                elif isinstance(part, ToolReturnPart):
                    out.append({"role": "tool", "tool_call_id": part.tool_call_id, "name": part.tool_name,
                                "content": _content_text(part.content)})
                elif isinstance(part, RetryPromptPart):
                    out.append({"role": "tool", "tool_call_id": part.tool_call_id,
                                "name": part.tool_name or "unknown", "content": _content_text(part.content)})
        elif isinstance(msg, ModelResponse):
            thinking = "".join(p.content for p in msg.parts if isinstance(p, ThinkingPart)).strip()
            text = "".join(p.content for p in msg.parts if isinstance(p, TextPart)).strip()
            calls = [{"id": c.tool_call_id, "name": c.tool_name, "arguments": _args_text(c.args)}
                     for c in msg.parts if isinstance(c, ToolCallPart)]
            if not text and not calls and not (include_thinking and thinking):
                continue
            row: dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                row["tool_calls"] = calls
            if include_thinking and thinking:
                row["thinking"] = thinking
            out.append(row)
    return out


def sum_usage(messages: list[Any]) -> dict[str, int]:
    """export.py sum_usage, verbatim: model-reported usage summed over the responses."""
    from pydantic_ai.messages import ModelResponse
    totals: dict[str, int] = {}
    for msg in messages:
        if isinstance(msg, ModelResponse) and msg.usage is not None:
            u = msg.usage
            for name in ("input_tokens", "output_tokens", "requests"):
                value = getattr(u, name, None)
                if isinstance(value, int):
                    totals[name] = totals.get(name, 0) + value
            details = getattr(u, "details", None)
            reasoning = getattr(details, "reasoning_tokens", None)
            if isinstance(reasoning, int):
                totals["reasoning_tokens"] = totals.get("reasoning_tokens", 0) + reasoning
    return totals


def system_prompt(messages: list[Any]) -> str:
    """export.py: the first ModelRequest's instructions."""
    from pydantic_ai.messages import ModelRequest
    return next((m.instructions for m in messages
                 if isinstance(m, ModelRequest) and getattr(m, "instructions", None)), None) or ""


def verdict(result: dict[str, Any], acceptance_passed: bool | None = None) -> str:
    status = result.get("status")
    if status == "infrastructure_error":
        return "infrastructure_abort"
    if status == "cap_failure":
        return "cap_failure"
    if status == "agent_error":
        return "agent_error"
    if acceptance_passed is None:
        return "ungraded"
    return "completed" if acceptance_passed else "failed_acceptance"


def load_messages(path: Path) -> list[Any]:
    from pydantic_ai.messages import ModelMessagesTypeAdapter
    return list(ModelMessagesTypeAdapter.validate_json(Path(path).read_bytes()))


def episode_row(result: dict[str, Any], messages: list[Any], tools: list[dict[str, Any]], *,
                include_thinking: bool = True, acceptance_passed: bool | None = None) -> dict[str, Any]:
    """One SFT-format row for an episode (`result` = EpisodeResult.to_dict())."""
    model = (result.get("agent_info") or {}).get("model") or {}
    row: dict[str, Any] = {
        "session_id": f"grcbench-{result['run_id']}",
        "task_id": result["task_id"],
        "category": result["category"],
        "verdict": verdict(result, acceptance_passed),
        "teacher": {"provider": model.get("provider") or result.get("model", {}).get("kind"),
                    "model": model.get("model_name") or result.get("model", {}).get("model_name")},
        "system": system_prompt(messages),
        "tools": [{"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
                  for t in tools],
        "messages": map_messages(messages, include_thinking=include_thinking),
        "usage": sum_usage(messages),
    }
    row["grcbench"] = {
        "run_id": result["run_id"], "split": result["split"], "task_version": result.get("task_version"),
        "config_id": result["config_id"], "setting": result["setting"], "sample": result.get("sample"),
        "mode": result.get("mode"), "status": result["status"], "end_reason": result.get("end_reason"),
        "caps": result.get("caps"), "caps_hit": result.get("caps_hit", []),
        "acceptance_passed": acceptance_passed, "include_thinking": include_thinking,
        "grc_agent_commit": (result.get("agent_info") or {}).get("grc_agent_commit"),
        "harness": result.get("harness"),
    }
    return row


def export_run_dir(run_dir: Path, *, include_thinking: bool = True,
                   acceptance_passed: bool | None = None) -> dict[str, Any]:
    """Build the row from an episode's run directory (result.json, messages.json, tools.json)."""
    run_dir = Path(run_dir)
    result = read_json(run_dir / "result.json")
    messages = load_messages(run_dir / "messages.json") if (run_dir / "messages.json").exists() else []
    tools = read_json(run_dir / "tools.json") if (run_dir / "tools.json").exists() else []
    return episode_row(result, messages, tools, include_thinking=include_thinking,
                       acceptance_passed=acceptance_passed)


def write_trace(path: Path, rows: list[dict[str, Any]]) -> int:
    """Write rows as the release writes them: one json.dumps(row) per line with the keys in row
    order (NOT sorted: the order is part of the format, R4) and non-ASCII escaped (the release
    files contain no raw non-ASCII bytes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return len(rows)


def row_format_errors(row: dict[str, Any]) -> list[str]:
    """Structural check of one row against the release format (used on the release itself and on
    our rows). Returns problems; empty when the row has the release's shape."""
    errs: list[str] = []
    keys = [k for k in row if k not in ("grcbench", "quarantine_reason")]
    if tuple(keys) != ROW_KEYS:
        errs.append(f"row keys {keys} != {list(ROW_KEYS)}")
    if not isinstance(row.get("system"), str):
        errs.append("system is not a string")
    for t in row.get("tools") or []:
        if list(t) != ["name", "description", "parameters"]:
            errs.append(f"tool keys {list(t)}")
    pending: list[str] = []
    for i, m in enumerate(row.get("messages") or []):
        role = m.get("role")
        if role == "user":
            if list(m) != ["role", "content"] or not isinstance(m["content"], str):
                errs.append(f"messages[{i}] user shape {list(m)}")
        elif role == "assistant":
            order = [k for k in ("role", "content", "tool_calls", "thinking") if k in m]
            if list(m) != order or "content" not in m or not isinstance(m["content"], str):
                errs.append(f"messages[{i}] assistant shape {list(m)}")
            for c in m.get("tool_calls") or []:
                if list(c) != ["id", "name", "arguments"] or not isinstance(c["arguments"], str):
                    errs.append(f"messages[{i}] tool_call shape {list(c)}")
                pending.append(c.get("id"))
            if "thinking" in m and not (isinstance(m["thinking"], str) and m["thinking"]):
                errs.append(f"messages[{i}] empty thinking")
        elif role == "tool":
            if list(m) != ["role", "tool_call_id", "name", "content"] or not isinstance(m["content"], str):
                errs.append(f"messages[{i}] tool shape {list(m)}")
            if m.get("tool_call_id") in pending:
                pending.remove(m["tool_call_id"])
        else:
            errs.append(f"messages[{i}] unknown role {role!r}")
    return errs

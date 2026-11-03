"""Tier-A harness caps (eval plan §4.4, §1.2; build plan Phase 4 "Caps").

Eval plan §4.4: "tool outputs capped at 4K tokens (beginning and end kept, plus error lines);
context capped at 64K tokens; at most 40 turns; a set time limit for each flowgraph run. A run that
hits a cap is a failure." This module holds those settings and the one meter that enforces and
records them for a whole episode.

Caps and how each is enforced
  requests        "40 turns" = at most 40 model requests per task. Every agent.iter of the episode
                  (the instruction, every approval resume, every scripted-user turn) is given the
                  SAME pydantic-ai RunUsage object and UsageLimits(request_limit=40), so the count
                  is shared across all of them (pydantic-ai 2.37 uses a passed RunUsage in place).
                  GRC-Agent's own StopGracefully(max_requests=40) counts per agent run only (its
                  for_run() returns a fresh counter), so it cannot bound an episode on its own.
  context         per-request input tokens <= 65,536: UsageLimits(per_request_input_tokens_limit),
                  checked against the provider-reported input tokens of each response (the model's
                  own tokenizer; llama-server reports the full prompt). An engine that refuses an
                  over-long prompt (HTTP 400, llama.cpp "exceed_context_size_error") is the same cap.
  tool output     every tool result is cut to 4,096 cl100k tokens by
                  grcbench.common.tokens.truncate_tool_output (head + tail + error lines). Truncating
                  is NOT a failure; it only shortens what the model sees (build plan Phase 4).
  episode time    a wall-clock limit for the whole episode (asyncio timeout in the episode process,
                  plus a process timeout on the host).
  reply tokens    the Tier-F answer limit (pre-registration §5.2: 8K tokens, reasoning included;
                  tier_f_mode.TIER_F_REPLY_TOKENS), none in Tier A. Every model request carries it as
                  pydantic-ai's max_tokens setting (sent as max_completion_tokens, which llama-server
                  honours); a reply the server ends at the limit (finish_reason "length") is the cap,
                  raised in wrap_model_request before pydantic-ai's own token-limit errors. When the
                  cut reply cannot be parsed, llama-server sends an error instead of it (UNPARSEABLE);
                  a failed request then records the server's own count (EpisodeMeter.server_last_reply:
                  llama_server.last_reply, GET /slots), and a reply that used up the limit is the cap
                  (reply_limit_hit). Any other unparseable reply is the model's (runner: status
                  agent_error, end reason "unparseable_reply"), never an infrastructure error.
  flowgraph run   a wall-clock limit for each flowgraph run (HeadlessFlowgraphProxy max_run_seconds,
                  headless_proxy D6): the run's sandbox is killed at the limit, and the episode ends
                  as a cap failure at the next step.
  StopGracefully  GRC-Agent's own graceful stop (agent.py L109-193: more than 40 requests in one
                  agent run, or one tool failing terminally 3 times in a row) ends the agent run with
                  a canned final answer. It counts as a cap failure. It is detected structurally: a
                  ModelRequestNode never ends a pydantic-ai graph by itself, so an End returned for
                  one through the capability chain is a capability's short-circuit. The canned text
                  (STOP_TEXT_*) gives the reason.

The harness capability (HarnessCapability) is placed OUTERMOST in GRC-Agent's capability chain
(CapabilityOrdering(position="outermost")), so that
  * before_tool_validate sees the arguments exactly as the model emitted them, before GRC-Agent's
    JsonRepairCapability (agent.py L196-233) rewrites JSON-stringified values: needed for the
    well-formed-call metric of eval plan §4.4;
  * wrap_tool_execute truncates the result after every other capability (the prompt-injection
    defender included) has produced it, i.e. exactly what the model would see;
  * wrap_node_run sees StopGracefully's End.

In the "grc_agent_defaults" setting (the 10-task check of §4.4) the context and tool-output caps are
off (agent_build.CAPS); the request, episode-time and flowgraph-run caps stay.

Per tool call the meter records (EpisodeMeter.calls): the raw arguments and their kind, whether they
are a JSON object, `schema_valid` (schema_check: the raw arguments validate against the tool's JSON
schema, i.e. the well-formed-call metric; JSON repair does not count), whether GRC-Agent's JSON
repair changed them, the validation / approval / execution events, the result size in cl100k tokens
and whether the 4K cap truncated it.

A policy object (HarnessCapability.policy) may refuse calls before validation; a refused call is
not validated, repaired or executed and gets the policy's fixed text as its result. The Tier-F
setting's budget (tier_f_mode.TierFBudget) is the only policy.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import CapabilityOrdering
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage, UsageLimits

from ..common.tokens import count, truncate_tool_output
from . import agent_build

# Cap names used in results, traces and the run log.
CAP_REQUESTS = "requests"
CAP_CONTEXT = "context"
CAP_EPISODE_TIME = "episode_time"
CAP_FLOWGRAPH_RUN_TIME = "flowgraph_run_time"
CAP_STOP_GRACEFULLY = "grc_agent_stop_gracefully"
CAP_REPLY_TOKENS = "reply_tokens"
CAP_NAMES = (CAP_REQUESTS, CAP_CONTEXT, CAP_EPISODE_TIME, CAP_FLOWGRAPH_RUN_TIME, CAP_STOP_GRACEFULLY,
             CAP_REPLY_TOKENS)

DEFAULT_EPISODE_SECONDS = 1800.0
DEFAULT_FLOWGRAPH_RUN_SECONDS = 300.0       # = headless_proxy.DEFAULT_MAX_RUN_SECONDS

# GRC-Agent 6c98128 agent.py L164-191: the two final answers StopGracefully ends a run with.
STOP_TEXT_MAX_REQUESTS = (
    "I've made a lot of tool calls without finishing, so I'm stopping here rather than looping "
    "further. Please check what's changed so far and try a more specific follow-up.")
STOP_TEXT_REPEATED_FAILURE = re.compile(
    r"\A`(?P<tool>[^`]*)` is not available in this environment and kept failing, so I'm stopping "
    r"rather than retrying it further\. The rest of the session is unaffected\.\Z")

# llama.cpp server: HTTP 400 {"error": {"type": "exceed_context_size_error", "message": "the request
# exceeds the available context size, ..."}} when a prompt does not fit the slot's context.
_ENGINE_CONTEXT = re.compile(r"exceed_context_size|exceeds? the available context size|context length|"
                             r"maximum context|context window", re.IGNORECASE)


@dataclass(frozen=True)
class Caps:
    """The harness caps of one setting (None = no cap)."""

    requests: int | None = 40
    context_tokens: int | None = 65536
    tool_output_tokens: int | None = 4096
    episode_seconds: float = DEFAULT_EPISODE_SECONDS
    flowgraph_run_seconds: float = DEFAULT_FLOWGRAPH_RUN_SECONDS
    reply_tokens: int | None = None

    @classmethod
    def for_setting(cls, setting: str, **overrides: Any) -> "Caps":
        """The caps of `setting` ("benchmark" or "grc_agent_defaults", agent_build.CAPS); keyword
        overrides exist for tests and are recorded with every result."""
        if setting not in agent_build.CAPS:
            raise ValueError(f"unknown setting {setting!r}; expected one of {sorted(agent_build.CAPS)}")
        base = agent_build.CAPS[setting]
        caps = cls(requests=base["requests"], context_tokens=base["context_tokens"],
                   tool_output_tokens=base["tool_output_tokens"])
        return dataclasses.replace(caps, **overrides) if overrides else caps

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Caps":
        return cls(**{f.name: data[f.name] for f in dataclasses.fields(cls) if f.name in data})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def usage_limits(self) -> UsageLimits:
        """pydantic-ai limits for every agent run of the episode (with the shared RunUsage)."""
        return UsageLimits(request_limit=self.requests, per_request_input_tokens_limit=self.context_tokens)

    def model_settings(self) -> dict[str, Any] | None:
        """pydantic-ai model settings for every agent run of the episode: the answer limit, if any."""
        return {"max_tokens": self.reply_tokens} if self.reply_tokens else None


@dataclass
class CapHit:
    cap: str
    detail: str
    limit: Any = None
    at_request: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class CapExceeded(Exception):
    """Raised by the harness to end an agent run as soon as a harness cap is hit."""

    def __init__(self, hit: CapHit) -> None:
        super().__init__(f"{hit.cap}: {hit.detail}")
        self.hit = hit


def stop_gracefully_reason(output: Any) -> str | None:
    """'max_requests' or 'repeated_failure:<tool>' if `output` is one of StopGracefully's canned
    final answers, else None."""
    if not isinstance(output, str):
        return None
    if output == STOP_TEXT_MAX_REQUESTS:
        return "max_requests"
    m = STOP_TEXT_REPEATED_FAILURE.match(output)
    return f"repeated_failure:{m.group('tool')}" if m else None


ENGINE_REFUSAL = "engine refused the prompt"      # the detail of a context cap the model server reported
# llama.cpp's message when its chat parser cannot read what the model generated (common/chat parsers)
UNPARSEABLE = re.compile(r"model produced output that does not match the expected", re.IGNORECASE)


def unparseable_reply(exc: BaseException) -> bool:
    """The model server reported that it could not parse the model's reply (UNPARSEABLE, anywhere in the
    exception chain): the model's output, not a server failure."""
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if UNPARSEABLE.search(str(e)):
            return True
        e = e.__cause__ if e.__cause__ is not None else (None if e.__suppress_context__ else e.__context__)
    return False


def reply_limit_hit(meter: "EpisodeMeter") -> "CapHit | None":
    """The answer-limit cap for a request that failed after its reply used up the limit, by the server's count
    (the last request's `server_reply`, recorded by HarnessCapability.wrap_model_request)."""
    limit = meter.caps.reply_tokens
    reply = meter.requests[-1].get("server_reply") if meter.requests else None
    if not limit or not reply or reply.get("processing") or reply.get("n_remain") != 0:
        return None
    n = reply.get("n_decoded")
    if not isinstance(n, int) or n < limit:
        return None
    return CapHit(CAP_REPLY_TOKENS, f"the reply reached the {limit:,}-token answer limit ({n:,} tokens generated, "
                  "by the server's count; the server could not return the cut reply)", limit, meter.usage.requests)


def _context_refusal(exc: BaseException) -> str | None:
    """An HTTP 400/413 reply in the exception's chain (__cause__, and __context__ unless suppressed)
    whose read body says the prompt does not fit the engine's context: GRC-Agent's retrying client
    raises httpx2.HTTPStatusError, which the OpenAI SDK and pydantic-ai wrap as "Connection error."
    A body nobody read (httpx2.ResponseNotRead) is not a refusal."""
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        response = getattr(e, "response", None)
        status = getattr(response, "status_code", None)
        if status in (400, 413):
            try:
                body = response.text
            except Exception:              # noqa: BLE001 - httpx2.ResponseNotRead: a streamed body never read
                body = ""
            if body and _ENGINE_CONTEXT.search(body):
                return f"HTTP {status} {body[:300]}"
        e = e.__cause__ if e.__cause__ is not None else (None if e.__suppress_context__ else e.__context__)
    return None


def classify_exception(exc: BaseException, caps: Caps, requests: int | None = None) -> CapHit | None:
    """The cap an exception from an agent run stands for, or None if it is not a cap."""
    from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
    if isinstance(exc, CapExceeded):
        return exc.hit
    if isinstance(exc, UsageLimitExceeded):
        text = str(exc)
        if "per_request_input_tokens_limit" in text:
            return CapHit(CAP_CONTEXT, text.split(". Consider")[0], caps.context_tokens, requests)
        if "request_limit" in text:
            return CapHit(CAP_REQUESTS, text.split(". Consider")[0], caps.requests, requests)
        return CapHit("usage", text, None, requests)
    if isinstance(exc, ModelHTTPError) and exc.status_code in (400, 413) and caps.context_tokens is not None:
        body = exc.body if isinstance(exc.body, str) else json.dumps(exc.body, default=str)
        if _ENGINE_CONTEXT.search(f"{exc.message} {body}"):
            return CapHit(CAP_CONTEXT, f"{ENGINE_REFUSAL}: HTTP {exc.status_code} {body[:300]}",
                          caps.context_tokens, requests)
    refused = _context_refusal(exc) if caps.context_tokens is not None else None
    if refused:
        return CapHit(CAP_CONTEXT, f"{ENGINE_REFUSAL}: {refused}", caps.context_tokens, requests)
    if isinstance(exc, TimeoutError):
        return CapHit(CAP_EPISODE_TIME, "episode wall-clock limit reached", caps.episode_seconds, requests)
    return None


# --------------------------------------------------------------------------------------------
# Tool-output cap
# --------------------------------------------------------------------------------------------

def model_view(result: Any) -> str:
    """The text the model reads for a tool result (pydantic-ai ToolReturnPart.model_response_str):
    strings as they are, anything else as pydantic-ai's JSON dump."""
    from pydantic_ai.messages import tool_return_ta
    if isinstance(result, str):
        return result
    return tool_return_ta.dump_json(result, by_alias=True).decode()


def cap_tool_result(result: Any, cap: int | None) -> tuple[Any, dict[str, Any]]:
    """Apply the tool-output cap to one tool result; returns (result for the model, record).

    A result within the cap is returned unchanged (same object, so the model sees exactly what it
    would without the harness). A longer one is replaced by the truncated text of its model view.
    `ToolReturn` objects keep their metadata and multimodal content; only the return value is cut.
    """
    from pydantic_ai.messages import ToolReturn
    if isinstance(result, ToolReturn):
        inner, rec = cap_tool_result(result.return_value, cap)
        if rec["truncated"]:
            result = dataclasses.replace(result, return_value=inner)
        return result, rec
    text = model_view(result)
    n = count(text)
    if cap is None or n <= cap:
        return result, {"tokens": n, "truncated": False}
    cut = truncate_tool_output(text, cap)
    return cut, {"tokens": n, "truncated": True, "tokens_after": count(cut)}


# --------------------------------------------------------------------------------------------
# The episode meter and the harness capability
# --------------------------------------------------------------------------------------------

def _raw_args(args: Any) -> tuple[str, str]:
    """(raw text, kind) of a tool call's arguments as the model emitted them."""
    if isinstance(args, str):
        return args, "str"
    if args is None:
        return "", "none"
    return json.dumps(args, ensure_ascii=False), type(args).__name__


def _json_object(raw: str) -> bool:
    if raw.strip() == "":
        return True             # pydantic-ai treats empty arguments as {}
    try:
        return isinstance(json.loads(raw), dict)
    except (json.JSONDecodeError, ValueError):
        return False


def schema_check(args: Any, schema: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """Whether a tool call's arguments, AS THE MODEL EMITTED THEM (before GRC-Agent's JSON repair),
    are a JSON object that validates against the tool's JSON schema (the one sent to the model).

    This is the well-formed-call metric of eval plan §4.4 and V1 of the Tier-F setting. JSON-
    stringified arrays/objects, which GRC-Agent's JsonRepairCapability would decode, and the
    pydantic aliases GRC-Agent accepts (e.g. "id" for "block_id") do NOT count as well formed: the
    schema is the contract. Empty arguments count as {} (pydantic-ai's rule)."""
    import jsonschema
    if isinstance(args, str):
        if args.strip() == "":
            obj: Any = {}
        else:
            try:
                obj = json.loads(args)
            except (json.JSONDecodeError, ValueError) as exc:
                return False, [f"arguments are not valid JSON: {exc}"]
    else:
        obj = {} if args is None else args
    if not isinstance(obj, dict):
        return False, ["arguments are not a JSON object"]
    if not schema:
        return True, []
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.absolute_path))
    return not errors, [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"[:300] for e in errors[:5]]


@dataclass
class EpisodeMeter:
    """Shared usage, cap state and per-call / per-request records of one episode."""

    caps: Caps
    usage: RunUsage = field(default_factory=RunUsage)
    hits: list[CapHit] = field(default_factory=list)
    calls: dict[str, dict[str, Any]] = field(default_factory=dict)       # by tool_call_id, in order
    requests: list[dict[str, Any]] = field(default_factory=list)
    tool_names: set[str] = field(default_factory=set)
    output_tool_names: set[str] = field(default_factory=set)
    flowgraph_cap_probe: Callable[[], dict[str, Any] | None] | None = None
    # the model server's own record of its last reply (llama_server.last_reply), read after a failed request
    server_last_reply: Callable[[], dict[str, Any] | None] | None = None
    prefilled_ids: set[str] = field(default_factory=set)     # harness-made calls (Tier-F prefill)
    turn_index: int = 0
    started: float = field(default_factory=time.monotonic)

    def hit(self, cap_hit: CapHit) -> CapHit:
        if not any(h.cap == cap_hit.cap for h in self.hits):
            self.hits.append(cap_hit)
        return cap_hit

    def call_record(self, call: Any, tool_def: Any = None) -> dict[str, Any]:
        cid = call.tool_call_id
        rec = self.calls.get(cid)
        if rec is None:
            raw, kind = _raw_args(call.args)
            schema = getattr(tool_def, "parameters_json_schema", None) if tool_def is not None else None
            well_formed, schema_errors = schema_check(call.args, schema)
            rec = {"tool_call_id": cid, "tool": call.tool_name, "turn": self.turn_index,
                   "request": self.usage.requests, "raw_args": raw, "raw_args_kind": kind,
                   "json_object": _json_object(raw) if kind == "str" else True,
                   "schema_valid": well_formed, "schema_errors": schema_errors,
                   "known_tool": tool_def is not None, "events": []}
            if cid in self.prefilled_ids:
                rec["prefilled"] = True
            self.calls[cid] = rec
        return rec

    def flowgraph_cap(self) -> CapHit | None:
        probe = self.flowgraph_cap_probe() if self.flowgraph_cap_probe is not None else None
        if not probe:
            return None
        return CapHit(CAP_FLOWGRAPH_RUN_TIME,
                      f"flowgraph run killed at the {self.caps.flowgraph_run_seconds:g} s limit: "
                      f"{probe.get('command')!r}", self.caps.flowgraph_run_seconds, self.usage.requests)

    def summary(self) -> dict[str, Any]:
        calls = list(self.calls.values())
        return {
            "requests": self.usage.requests, "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens, "tool_calls": self.usage.tool_calls,
            "max_request_input_tokens": max((r["input_tokens"] or 0 for r in self.requests), default=0),
            "tool_results_truncated": sum(1 for c in calls if c.get("truncated")),
            "model_tool_calls": sum(1 for c in calls if not c.get("prefilled")),
            "model_tool_calls_schema_valid": sum(1 for c in calls if not c.get("prefilled") and c.get("schema_valid")),
        }


@dataclass
class HarnessCapability(AbstractCapability[Any]):
    """The harness's own capability (outermost): caps, tool-output truncation, raw-argument log."""

    meter: EpisodeMeter = field(default_factory=lambda: EpisodeMeter(Caps()))
    # Optional tool-access policy (the Tier-F budget, tier_f_mode.TierFBudget): `blocked(ctx, call)`
    # decides, before validation, that a call gets `blocked_text` instead of running.
    policy: Any = None

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(position="outermost")

    async def for_run(self, ctx: Any) -> "HarnessCapability":   # noqa: ARG002 - one meter per episode
        return self

    # -- model requests ------------------------------------------------------------------------
    async def wrap_model_request(self, ctx: Any, *, request_context: Any, handler: Any) -> Any:
        params = request_context.model_request_parameters
        self.meter.tool_names.update(t.name for t in params.function_tools)
        self.meter.output_tool_names.update(t.name for t in params.output_tools)
        t0 = time.monotonic()
        rec: dict[str, Any] = {"index": len(self.meter.requests) + 1, "turn": self.meter.turn_index,
                               "started_s": round(t0 - self.meter.started, 3)}
        self.meter.requests.append(rec)
        try:
            response = await handler(request_context)
        except BaseException as exc:
            rec.update(seconds=round(time.monotonic() - t0, 3), error=f"{type(exc).__name__}: {exc}"[:500],
                       input_tokens=None, output_tokens=None)
            # a streamed reply the server could not send shows up here as a cancelled request: its count is
            # the server's (a limit the harness hit before the request, e.g. the request cap, is not)
            if self.meter.server_last_reply is not None and not isinstance(exc, (UsageLimitExceeded, CapExceeded)):
                rec["server_reply"] = self.meter.server_last_reply()
            raise
        usage = response.usage
        rec.update(seconds=round(time.monotonic() - t0, 3), input_tokens=usage.input_tokens,
                   output_tokens=usage.output_tokens, model_name=response.model_name,
                   finish_reason=getattr(response, "finish_reason", None))
        limit = self.meter.caps.reply_tokens
        if limit and rec["finish_reason"] == "length":
            raise CapExceeded(self.meter.hit(CapHit(
                CAP_REPLY_TOKENS, f"the reply reached the {limit:,}-token answer limit (reasoning included)",
                limit, self.meter.usage.requests)))
        return response

    # -- graph nodes ---------------------------------------------------------------------------
    async def wrap_node_run(self, ctx: Any, *, node: Any, handler: Any) -> Any:
        from pydantic_ai import Agent
        from pydantic_graph import End
        fg = self.meter.flowgraph_cap()
        if fg is not None:
            raise CapExceeded(self.meter.hit(fg))
        result = await handler(node)
        if Agent.is_model_request_node(node) and isinstance(result, End):
            output = getattr(result.data, "output", None)
            reason = stop_gracefully_reason(output) or "a capability ended the run at a model request"
            self.meter.hit(CapHit(CAP_STOP_GRACEFULLY, reason, 40, self.meter.usage.requests))
        return result

    # -- tool validation (raw arguments, before GRC-Agent's JSON repair) -----------------------
    async def before_tool_validate(self, ctx: Any, *, call: Any, tool_def: Any, args: Any) -> Any:
        from pydantic_ai.exceptions import SkipToolValidation
        rec = self.meter.call_record(call, tool_def)
        if self.policy is not None and self.policy.blocked(ctx, call):
            # Not validated, not repaired, not executed: wrap_tool_execute returns the fixed text.
            rec["blocked"] = True
            rec["events"].append({"stage": "validate", "request": self.meter.usage.requests, "outcome": "blocked"})
            raise SkipToolValidation({})
        return args

    async def wrap_tool_validate(self, ctx: Any, *, call: Any, tool_def: Any, args: Any, handler: Any) -> Any:
        from pydantic_ai.exceptions import ApprovalRequired, CallDeferred
        rec = self.meter.call_record(call, tool_def)
        raw = rec["raw_args"]
        repaired = json.dumps(args, ensure_ascii=False, sort_keys=True) if isinstance(args, dict) else args
        original = None
        if rec["raw_args_kind"] == "str" and rec["json_object"]:
            original = json.dumps(json.loads(raw) if raw.strip() else {}, ensure_ascii=False, sort_keys=True)
        elif isinstance(call.args, dict):
            original = json.dumps(call.args, ensure_ascii=False, sort_keys=True)
        rec.setdefault("repair_changed_args", original is None or repaired != original)
        event: dict[str, Any] = {"stage": "validate", "request": self.meter.usage.requests}
        rec["events"].append(event)
        try:
            validated = await handler(args)
        except (ApprovalRequired, CallDeferred) as exc:
            event["outcome"] = "deferred" if isinstance(exc, CallDeferred) else "approval_required"
            rec.setdefault("validation", "ok")
            raise
        except Exception as exc:
            event.update(outcome="invalid", error=_error_text(exc))
            rec.setdefault("validation", "invalid")
            rec.setdefault("validation_error", _error_text(exc))
            raise
        event["outcome"] = "ok"
        rec.setdefault("validation", "ok")
        return validated

    # -- tool execution (the 4K tool-output cap) -----------------------------------------------
    async def wrap_tool_execute(self, ctx: Any, *, call: Any, tool_def: Any, args: Any, handler: Any) -> Any:
        from pydantic_ai.exceptions import ToolFailedError, ToolRetryError
        rec = self.meter.call_record(call, tool_def)
        cap = self.meter.caps.tool_output_tokens
        t0 = time.monotonic()
        event: dict[str, Any] = {"stage": "execute", "request": self.meter.usage.requests,
                                 "approved": bool(getattr(ctx, "tool_call_approved", False))}
        rec["events"].append(event)
        if rec.get("blocked"):
            event.update(outcome="blocked", seconds=0.0)
            rec["execution"] = "blocked"
            return self.policy.blocked_text
        try:
            result = await handler(args)
        except ToolRetryError as exc:
            part = exc.tool_retry
            if isinstance(part.content, str):
                part.content, info = cap_tool_result(part.content, cap)
                event.update(info)
                rec["truncated"] = rec.get("truncated", False) or info["truncated"]
            event.update(outcome="retry", seconds=round(time.monotonic() - t0, 3),
                         error=_error_text(exc))
            rec["execution"] = "retry"
            raise
        except ToolFailedError as exc:
            part = exc.tool_failed
            part.content, info = cap_tool_result(part.content, cap)
            event.update(info, outcome="failed", seconds=round(time.monotonic() - t0, 3))
            rec["truncated"] = rec.get("truncated", False) or info["truncated"]
            rec["execution"] = "failed"
            raise
        except BaseException as exc:
            event.update(outcome="error", seconds=round(time.monotonic() - t0, 3), error=_error_text(exc))
            rec["execution"] = "error"
            raise
        capped, info = cap_tool_result(result, cap)
        event.update(info, outcome="ok", seconds=round(time.monotonic() - t0, 3))
        rec["truncated"] = rec.get("truncated", False) or info["truncated"]
        rec["execution"] = "ok"
        return capped


def _error_text(exc: BaseException, limit: int = 1000) -> str:
    return f"{type(exc).__name__}: {exc}"[:limit]

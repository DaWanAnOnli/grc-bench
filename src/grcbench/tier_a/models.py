"""The model under test of a Tier-A episode, as a JSON-serialisable spec (eval plan §7.1).

Every episode runs in its own OS process (isolation.py explains why), so the host cannot hand the
episode a Python model object; it hands it a ModelSpec, and the episode builds the model:

  openai_compatible  the real-run path: any OpenAI-compatible server (llama.cpp llama-server).
                     Without `uds_path`, `build_model` returns None and GRC-Agent builds the model
                     itself from the run's .env (provider openai_compatible -> OpenAIChatModel), on
                     GRC-Agent's retrying HTTP client with error bodies read (agent_build patches
                     _retrying_http_client with retrying_client); with `uds_path` the same
                     construction goes over a UNIX socket (agent_build.make_openai_compatible_model).
                     `hosted(...)` is the same path to a hosted OpenAI-compatible API: the spec
                     carries `relay` = {"upstream": <https base URL>, "api_key_env": <variable
                     NAME>}, and the runner starts relay.Relay for the episode and points the model
                     at its socket. The key itself is never part of a spec.
  scripted           a pydantic-ai FunctionModel that replays fixed turns (tests, golden-sequence
                     replays). Each model request takes the next turn:
                         {"text": str, "thinking": str,
                          "tool_calls": [{"name": str, "args": dict | str, "id": str}],
                          "usage": {"input_tokens": int, "output_tokens": int}}
                     A string "args" is sent verbatim, so malformed JSON can be scripted. When the
                     turns run out, "on_exhausted" decides: "text" (a closing text reply) or
                     "repeat_last" (the last turn again, with fresh call ids: a looping agent).
  wire_scripted      the REAL llama-server path minus the network: GRC-Agent's own OpenAIChatModel
                     construction (agent_build.make_openai_compatible_model) over an httpx2
                     MockTransport that answers each POST /chat/completions with the next scripted
                     chat completion and appends the exact request body to <run>/wire.jsonl. Turns:
                         {"text": str, "reasoning_content": str,
                          "tool_calls": [{"id": str, "name": str, "arguments": str | dict}],
                          "usage": {"prompt_tokens": int, "completion_tokens": int},
                          "finish_reason": str}   (default "tool_calls" or "stop"; "length" = a
                                                   reply the server ended at max_completion_tokens)
                     Used for the byte-level conformance of the Tier-F setting (the body is what
                     llama-server would receive) and for golden replays through the wire format.
  factory            "module:function" or "/abs/path/file.py:function", called with **kwargs and
                     returning a pydantic-ai Model (any custom test model).

Streaming (`ModelSpec.streams()`). GRC-Agent's desktop turn driver streams every model request
(chat/stream_view.py `_stream_request`: `node.stream(run.ctx)`), and Qoherent's collection drove that
same turn driver (experiments/dataset_collection/runner_contract.py). The runner does the same for
the real-run path: `openai_compatible` streams by default. This matters beyond rendering: GRC-Agent's
HTTP client has a 120 s READ timeout (agent_factory._HTTP_TIMEOUT) and retries on it. A streamed
request only needs a byte every 120 s, whereas a non-streamed one must be completely generated within
120 s. A stream can still be quiet that long while llama-server keeps generating reasoning: GRC-Agent's
client ended 3 of 10 Tier-F samples of the 28 Oct dev run (RTX 5090) and both of a Colab T4 run
(2 Nov 2026) that way. So an
episode's client reads for as long as the episode's wall-clock cap (build_model and
agent_build.build_benchmark_agent with `read_seconds`): only the pre-registered cap ends a request. What the model is sent is the
same either way except `"stream": true` (instead of false) and `"stream_options": {"include_usage":
true}` in the body (tests/test_tier_a_smoke.py compares both bodies). `scripted` never streams (a FunctionModel without
a stream function); `wire_scripted` streams when asked (`stream=True`: it then answers with
server-sent events, as llama-server does); `factory` streams only when asked.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from .isolation import DEFAULT_BASE_URL, DEFAULT_MODEL

Kind = Literal["openai_compatible", "scripted", "wire_scripted", "factory"]
EXHAUSTED_TEXT = "Scripted model: no further turns."
WIRE_FILE = "wire.jsonl"            # request bodies recorded by the wire_scripted model (run dir)


@dataclass(frozen=True)
class ModelSpec:
    kind: Kind
    base_url: str = DEFAULT_BASE_URL
    model_name: str = DEFAULT_MODEL
    uds_path: str | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    on_exhausted: Literal["text", "repeat_last"] = "text"
    factory: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    stream: bool | None = None          # None = the kind's default (module docstring, "Streaming")
    relay: dict[str, Any] | None = None  # hosted API through relay.Relay (`hosted`)

    def __post_init__(self) -> None:
        if self.kind == "scripted" and self.stream:
            raise ValueError("scripted models do not stream (FunctionModel without a stream function)")
        if self.relay is not None:
            if self.kind != "openai_compatible":
                raise ValueError("a relay serves openai_compatible models only")
            unknown = set(self.relay) - {"upstream", "api_key_env", "allow_http_loopback"}
            if unknown or not self.relay.get("upstream") or not self.relay.get("api_key_env"):
                raise ValueError("relay needs 'upstream' and 'api_key_env' (the variable's name)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        return cls(**data)

    def streams(self) -> bool:
        """Whether the runner streams this model's requests (module docstring, "Streaming")."""
        if self.stream is not None:
            return bool(self.stream)
        return self.kind == "openai_compatible"

    def network(self) -> Literal["host", "none"]:
        """The episode process needs the host network only for a llama-server on a TCP port."""
        return "host" if self.kind == "openai_compatible" and not self.uds_path else "none"

    def describe(self) -> dict[str, Any]:
        """What the run log and results record about the model (no secrets exist to leak)."""
        if self.kind == "openai_compatible":
            out = {"kind": self.kind, "base_url": self.base_url, "model_name": self.model_name,
                   "uds_path": self.uds_path, "stream": self.streams()}
            if self.relay is not None:
                out["relay"] = {"upstream": self.relay["upstream"], "api_key_env": self.relay["api_key_env"]}
            return out
        if self.kind in ("scripted", "wire_scripted"):
            return {"kind": self.kind, "turns": len(self.turns), "on_exhausted": self.on_exhausted,
                    **({"model_name": self.model_name, "stream": self.streams()}
                       if self.kind == "wire_scripted" else {})}
        return {"kind": self.kind, "factory": self.factory, "kwargs": self.kwargs, "stream": self.streams()}


def openai_compatible(base_url: str = DEFAULT_BASE_URL, model_name: str = DEFAULT_MODEL,
                      uds_path: str | None = None, stream: bool | None = None) -> ModelSpec:
    return ModelSpec("openai_compatible", base_url=base_url, model_name=model_name, uds_path=uds_path,
                     stream=stream)


def hosted(api_base: str, model_name: str, api_key_env: str, stream: bool | None = None) -> ModelSpec:
    """A hosted OpenAI-compatible API (OpenAI, OpenRouter, DeepSeek, ...) through relay.Relay: the
    key is read on the host from the environment variable `api_key_env` when the episode starts."""
    from .relay import Upstream
    Upstream.parse(api_base)                  # https only; fails early on a bad URL
    return ModelSpec("openai_compatible", model_name=model_name, stream=stream,
                     relay={"upstream": api_base, "api_key_env": api_key_env})


def scripted(turns: list[dict[str, Any]], on_exhausted: Literal["text", "repeat_last"] = "text") -> ModelSpec:
    for i, turn in enumerate(turns):
        unknown = set(turn) - {"text", "thinking", "tool_calls", "usage"}
        if unknown:
            raise ValueError(f"turn {i}: unknown keys {sorted(unknown)}")
    json.dumps(turns)                         # must survive the trip to the episode process
    return ModelSpec("scripted", turns=list(turns), on_exhausted=on_exhausted)


def wire_scripted(turns: list[dict[str, Any]], model_name: str = DEFAULT_MODEL,
                  on_exhausted: Literal["text", "repeat_last"] = "text", stream: bool = False) -> ModelSpec:
    """GRC-Agent's OpenAIChatModel over a scripted chat-completions transport (module docstring);
    stream=True makes the runner stream and the transport answer with server-sent events."""
    for i, turn in enumerate(turns):
        unknown = set(turn) - {"text", "reasoning_content", "tool_calls", "usage", "finish_reason"}
        if unknown:
            raise ValueError(f"turn {i}: unknown keys {sorted(unknown)}")
        for call in turn.get("tool_calls") or []:
            if not call.get("id") or not call.get("name"):
                raise ValueError(f"turn {i}: wire tool calls need an id and a name")
    json.dumps(turns)
    return ModelSpec("wire_scripted", model_name=model_name, turns=list(turns), on_exhausted=on_exhausted,
                     stream=stream)


def factory(ref: str | Callable[..., Any], **kwargs: Any) -> ModelSpec:
    """A factory spec from "module:function", "/path/file.py:function" or a top-level function."""
    if callable(ref):
        if "<locals>" in ref.__qualname__:
            raise ValueError("model factories must be top-level functions (the episode process imports them)")
        ref = f"{Path(inspect.getfile(ref)).resolve()}:{ref.__qualname__}"
    json.dumps(kwargs)
    return ModelSpec("factory", factory=ref, kwargs=dict(kwargs))


def as_spec(model: ModelSpec | str | Callable[..., Any]) -> ModelSpec:
    if isinstance(model, ModelSpec):
        return model
    return factory(model)


def _load_ref(ref: str) -> Callable[..., Any]:
    target, _, name = ref.rpartition(":")
    if not target or not name:
        raise ValueError(f"factory reference must be 'module:function' or 'file.py:function', got {ref!r}")
    if target.endswith(".py"):
        spec = importlib.util.spec_from_file_location(f"grcbench_model_factory_{abs(hash(target))}", target)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load {target}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    obj: Any = module
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def build_model(spec: ModelSpec, *, read_seconds: float | None = None) -> Any:
    """The pydantic-ai model for `spec` (None = let GRC-Agent build it from the run's .env).
    read_seconds: the HTTP read timeout of an OpenAI-compatible model's client (agent_build.retrying_client)."""
    if spec.kind == "openai_compatible":
        if spec.uds_path:
            from .agent_build import make_openai_compatible_model
            return make_openai_compatible_model(spec.base_url, spec.model_name, uds_path=spec.uds_path,
                                                read_seconds=read_seconds)
        return None
    if spec.kind == "scripted":
        from pydantic_ai.models.function import FunctionModel
        return FunctionModel(ScriptedTurns(spec.turns, spec.on_exhausted), model_name="grcbench-scripted")
    if spec.kind == "wire_scripted":
        import os

        import httpx2

        from .agent_build import make_openai_compatible_model
        record = Path(os.environ.get("GRC_BENCH_RUN_DIR") or ".") / WIRE_FILE
        server = WireScript(spec.turns, spec.on_exhausted, spec.model_name, record)
        return make_openai_compatible_model(spec.base_url, spec.model_name,
                                            transport=httpx2.MockTransport(server), read_seconds=read_seconds)
    if spec.kind == "factory":
        return _load_ref(str(spec.factory))(**spec.kwargs)
    raise ValueError(f"unknown model kind {spec.kind!r}")


class ScriptedTurns:
    """FunctionModel function replaying fixed turns, one per model request (module docstring)."""

    def __init__(self, turns: list[dict[str, Any]], on_exhausted: str = "text") -> None:
        self.turns = turns
        self.on_exhausted = on_exhausted
        self.n = 0

    def __call__(self, messages: list[Any], info: Any) -> Any:   # noqa: ARG002 - FunctionModel API
        from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart
        from pydantic_ai.usage import RequestUsage
        index = self.n
        self.n += 1
        if index < len(self.turns):
            turn = self.turns[index]
        elif self.on_exhausted == "repeat_last" and self.turns:
            turn = self.turns[-1]
        else:
            turn = {"text": EXHAUSTED_TEXT}
        parts: list[Any] = []
        if turn.get("thinking"):
            parts.append(ThinkingPart(turn["thinking"]))
        if turn.get("text"):
            parts.append(TextPart(turn["text"]))
        for j, call in enumerate(turn.get("tool_calls") or []):
            cid = call.get("id") if index < len(self.turns) and call.get("id") else f"call_{index:03d}_{j}"
            parts.append(ToolCallPart(call["name"], call.get("args", {}), tool_call_id=cid))
        if not parts:
            parts.append(TextPart(""))
        usage = RequestUsage(**turn["usage"]) if turn.get("usage") else RequestUsage()
        return ModelResponse(parts=parts, usage=usage, model_name="grcbench-scripted")


class WireScript:
    """httpx2 MockTransport handler: one scripted chat completion per request (wire_scripted).

    Every request body is appended verbatim (one JSON line {"path", "body": <the body text, byte for
    byte>} per request) to `record` before it is answered, so tests compare exactly what an
    OpenAI-compatible server receives. A request with "stream": true is answered with server-sent
    events (`stream_chunks`), a non-streamed one with the chat.completion object.
    """

    def __init__(self, turns: list[dict[str, Any]], on_exhausted: str, model_name: str, record: Path) -> None:
        self.turns = turns
        self.on_exhausted = on_exhausted
        self.model_name = model_name
        self.record = Path(record)
        self.n = 0

    def completion(self, index: int) -> dict[str, Any]:
        if index < len(self.turns):
            turn, fresh = self.turns[index], False
        elif self.on_exhausted == "repeat_last" and self.turns:
            turn, fresh = self.turns[-1], True
        else:
            turn, fresh = {"text": EXHAUSTED_TEXT}, False
        message: dict[str, Any] = {"role": "assistant", "content": turn.get("text") or None}
        if turn.get("reasoning_content"):
            message["reasoning_content"] = turn["reasoning_content"]
        calls = []
        for j, call in enumerate(turn.get("tool_calls") or []):
            args = call.get("arguments", {})
            calls.append({"id": f"call_{index:03d}_{j}" if fresh else call["id"], "type": "function",
                          "function": {"name": call["name"],
                                       "arguments": args if isinstance(args, str) else json.dumps(args)}})
        if calls:
            message["tool_calls"] = calls
        usage = turn.get("usage") or {"prompt_tokens": 1, "completion_tokens": 1}
        usage = {**usage, "total_tokens": int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))}
        return {"id": f"chatcmpl-grcbench-{index:03d}", "object": "chat.completion", "created": 0,
                "model": self.model_name, "usage": usage,
                "choices": [{"index": 0, "message": message,
                             "finish_reason": turn.get("finish_reason") or ("tool_calls" if calls else "stop")}]}

    @staticmethod
    def stream_chunks(completion: dict[str, Any]) -> list[dict[str, Any]]:
        """The chat.completion.chunk objects a streaming OpenAI-compatible server (llama-server)
        sends for `completion`: the role, the reasoning and the text deltas, one tool-call header
        delta (id, name) and one arguments delta per call, the finish reason, then the usage chunk
        (stream_options.include_usage)."""
        message = completion["choices"][0]["message"]
        base = {"id": completion["id"], "object": "chat.completion.chunk", "created": 0,
                "model": completion["model"]}

        def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        out = [chunk({"role": "assistant", "content": None})]
        if message.get("reasoning_content"):
            out.append(chunk({"reasoning_content": message["reasoning_content"]}))
        if message.get("content"):
            out.append(chunk({"content": message["content"]}))
        for j, call in enumerate(message.get("tool_calls") or []):
            out.append(chunk({"tool_calls": [{"index": j, "id": call["id"], "type": "function",
                                              "function": {"name": call["function"]["name"], "arguments": ""}}]}))
            out.append(chunk({"tool_calls": [{"index": j, "function": {"arguments": call["function"]["arguments"]}}]}))
        out.append(chunk({}, completion["choices"][0]["finish_reason"]))
        out.append({**base, "choices": [], "usage": completion["usage"]})
        return out

    def __call__(self, request: Any) -> Any:
        import httpx2
        body = request.content.decode("utf-8")
        self.record.parent.mkdir(parents=True, exist_ok=True)
        with open(self.record, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"path": request.url.path, "body": body}, ensure_ascii=False) + "\n")
        index = self.n
        self.n += 1
        completion = self.completion(index)
        try:
            streamed = bool(json.loads(body).get("stream"))
        except (json.JSONDecodeError, ValueError, AttributeError):
            streamed = False
        if not streamed:
            return httpx2.Response(200, json=completion)
        events = "".join(f"data: {json.dumps(c)}\n\n" for c in self.stream_chunks(completion)) + "data: [DONE]\n\n"
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=events.encode("utf-8"))

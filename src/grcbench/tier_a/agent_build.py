"""Build GRC-Agent's real executor agent, headless, for Tier A.

Eval plan §4.4: "One pinned GRC-Agent version and its real tool definitions". So the benchmark
does not re-create GRC-Agent: `build_benchmark_agent` calls GRC-Agent's own
`agent_factory.build_agents_from_cfg` at the pinned commit (6c98128) and keeps the executor role,
so the system prompt, output types, output validator, capabilities and all 22 model-visible tools
are GRC-Agent's own objects. The factory runs with these scoped substitutions, each undone when it
returns:

  1. `grc_agent.native_canvas` imports GTK. The factory only takes `NativeFlowgraphProxy` from it,
     as `deps_type` (a type annotation). A GTK-free stand-in module provides the episode's deps
     class under that name; GTK is never imported.
  2. `agent_factory._build_model` returns the given pydantic-ai model (tests: FunctionModel or
     TestModel). With model=None the factory builds the model itself from the run's .env
     (provider openai_compatible -> OpenAIChatModel on OpenAIProvider(base_url=llama-server), with
     GRC-Agent's retrying HTTP client): the real-run path. `make_openai_compatible_model` builds the
     same object explicitly (optionally over a UNIX socket or a test transport).
  3. `agent_factory._build_compaction_capability` returns `CompactionOff` in the "benchmark"
     setting (eval plan §4.4: context capped at 64K tokens, a run that hits a cap fails). The
     "grc_agent_defaults" setting keeps GRC-Agent's own compaction (the 10-task check of §4.4).
  4. `agent_factory.web_search_cap` / `web_fetch_cap` are replaced by local-only copies of the same
     two tools: identical name, description, JSON schema and argument validator, but the body
     returns OFFLINE_MESSAGE. Local-only (native=False) also keeps the two tools in the request for
     every model class: pydantic-ai drops a local web tool whenever the model class claims native
     web support, which FunctionModel and TestModel always do and OpenAIChatModel does only for
     model names containing "-search-preview" (pydantic_ai/profiles/openai.py).
  5. Process-wide (not scoped): the harness shell module's process launcher is routed through the
     nested no-network bubblewrap sandbox (isolation.install_shell_sandbox). GrcShell, its
     denylist, approval flags and tool schemas are untouched.

Substitutions 1, 2, 4 and 5 apply in both settings; the settings differ only in compaction (3)
and in the harness caps the runner applies (CAPS). In "grc_agent_defaults" GRC-Agent sizes its
compaction from the context_length its endpoint reports on /v1/models; for an http:// endpoint
that reports none it falls back to a 32,000-token window (agent_factory._build_compaction_capability).

Kept exactly as GRC-Agent builds them: instructions build_system_prompt("pai-desktop-chat") plus
the per-turn active-flowgraph note, output types [GrcAgentResponse, str, DeferredToolRequests]
with validate_flowgraph_state, JsonRepairCapability, StopGracefully(max_requests=40,
max_repeated_failures=3), ModelRequestLogger, StepPersistence (its store is
<run>/.grc_agent/chat_sessions.db because GRC_AGENT_ENV=<run>/.env), SystemReminders, GrcFileSystem,
GrcShell, the prompt-injection defender, model settings and retries {"tools": 3, "output": 3}.
Approvals (change_graph, run_flowgraph start, run_command, start_command) are not decided here:
the runner resolves the agent's DeferredToolRequests output, as GRC-Agent's sidebar does.

An agent must be built inside an episode process prepared by isolation.make_run_env; the build
refuses to run otherwise (isolation.check_process_isolation).
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic_ai.capabilities import AbstractCapability

from ..common.paths import GRC_AGENT_COMMIT
from . import isolation

Setting = Literal["benchmark", "grc_agent_defaults"]
SETTINGS: tuple[str, ...] = ("benchmark", "grc_agent_defaults")
OFFLINE_MESSAGE = "offline: web access is not available in this benchmark"
NATIVE_CANVAS = "grc_agent.native_canvas"

# Harness caps of eval plan §4.4, recorded in `info`; they are applied by the runner at run time
# (usage limits and a tool-output wrapper), not baked into the agent.
CAPS = {
    "benchmark": {"requests": 40, "context_tokens": 65536, "tool_output_tokens": 4096},
    "grc_agent_defaults": {"requests": 40, "context_tokens": None, "tool_output_tokens": None},
}


@dataclasses.dataclass
class CompactionOff(AbstractCapability[Any]):
    """Stands in for GRC-Agent's TranscriptPreservingTieredCompaction in the benchmark setting.

    It has no hooks, so the history reaches the model unmodified; the 64K-token context cap
    (eval plan §4.4) decides instead, and a run that exceeds it fails.
    """


def _offline_copy(tool: Any) -> Any:
    """The same pydantic-ai Tool (name, description, schema, argument validator, retries), but a
    body that returns OFFLINE_MESSAGE without touching the network."""
    async def offline(*_args: Any, **_kwargs: Any) -> str:
        return OFFLINE_MESSAGE

    clone = copy.copy(tool)
    clone.function = offline
    clone.function_schema = dataclasses.replace(tool.function_schema, function=offline, is_async=True)
    return clone


def offline_web_capabilities() -> tuple[Any, Any]:
    """Local-only, offline versions of GRC-Agent's web_search_cap and web_fetch_cap."""
    from pydantic_ai.capabilities import WebFetch, WebSearch

    from grc_agent.agent import web_fetch_cap, web_search_cap
    return (WebSearch(native=False, local=_offline_copy(web_search_cap.local)),
            WebFetch(native=False, local=_offline_copy(web_fetch_cap.local)))


@contextlib.contextmanager
def _swapped(obj: Any, **attrs: Any) -> Iterator[None]:
    old = {name: getattr(obj, name) for name in attrs}
    for name, value in attrs.items():
        setattr(obj, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(obj, name, value)


@contextlib.contextmanager
def gtk_free_native_canvas(proxy_cls: type) -> Iterator[types.ModuleType]:
    """Put a GTK-free `grc_agent.native_canvas` into sys.modules for the duration of the block.

    It exposes only `NativeFlowgraphProxy` (= `proxy_cls`), the one name the factory imports;
    any other attribute raises, so a GTK code path can never run by accident.
    """
    stub = types.ModuleType(NATIVE_CANVAS, "GRC-Bench stand-in for grc_agent.native_canvas (no GTK).")
    stub.NativeFlowgraphProxy = proxy_cls          # type: ignore[attr-defined]
    stub.__grcbench_stub__ = True                  # type: ignore[attr-defined]

    def _missing(name: str) -> Any:
        raise AttributeError(f"{NATIVE_CANVAS} is a GTK-free GRC-Bench stand-in; {name!r} needs GTK")

    stub.__getattr__ = _missing                    # type: ignore[attr-defined]
    saved = sys.modules.get(NATIVE_CANVAS)
    sys.modules[NATIVE_CANVAS] = stub
    try:
        yield stub
    finally:
        if saved is None:
            sys.modules.pop(NATIVE_CANVAS, None)
        else:
            sys.modules[NATIVE_CANVAS] = saved


def flow_graph_of(deps: Any) -> Any:
    """The live GRC FlowGraph behind `deps` (a proxy with `_get_target`, like
    NativeFlowgraphProxy, or a bare FlowGraph as in GRC-Agent's scenario harness)."""
    target = getattr(type(deps), "_get_target", None)
    return target(deps) if callable(target) else deps


def _graph_path(deps: Any) -> str | None:
    """desktop_app's `canvas.path`: the active graph's file, None when unsaved or no graph is open.
    Read from `deps.grc_file_path`, the surface GRC-Agent's own active-flowgraph note reads
    (agent_factory.add_active_flowgraph_context), so the note and the file tools always agree."""
    try:
        return str(deps.grc_file_path or "") or None
    except Exception:
        return None


def _live_flow_graph(deps: Any) -> Any:
    """desktop_app's `canvas.current_flow_graph`: the live FlowGraph, None when no graph is open
    (NativeFlowgraphProxy._get_target raises RuntimeError then; the canvas returns None)."""
    try:
        return flow_graph_of(deps)
    except RuntimeError:
        return None


def error_body_transport(inner: Any) -> Any:
    """`inner` (an httpx2 async transport) with the body of every response of status >= 400 read before
    it is returned, so the HTTPStatusError that GRC-Agent's retrying transport raises carries the
    server's reason (caps.classify_exception reads llama-server's exceed_context_size_error from it).
    Other responses, streamed replies among them, pass through unread."""
    import httpx2

    class ReadErrorBodies(httpx2.AsyncBaseTransport):
        async def handle_async_request(self, request: Any) -> Any:
            response = await inner.handle_async_request(request)
            if response.status_code >= 400:
                await response.aread()
            return response

        async def aclose(self) -> None:
            await inner.aclose()

    return ReadErrorBodies()


def retrying_client(inner: Any = None, *, read_seconds: float | None = None) -> Any:
    """GRC-Agent's retrying httpx2 client (agent_factory._retrying_http_client: its timeouts, retry
    config and raise_for_status check) around `inner` (default httpx2.AsyncHTTPTransport()), with the
    bodies of error replies read (error_body_transport). `read_seconds` replaces GRC-Agent's 120 s read
    timeout: an episode passes its wall-clock cap, so a streamed request whose server is quiet for a
    while (long reasoning on a slow GPU) is ended by the pre-registered cap only."""
    import httpx2
    from grc_agent import agent_factory as af
    from pydantic_ai.retries import AsyncHTTPX2TenacityTransport
    inner = httpx2.AsyncHTTPTransport() if inner is None else inner
    timeout = {**af._HTTP_TIMEOUT, **({"read": read_seconds} if read_seconds else {})}
    return httpx2.AsyncClient(
        timeout=httpx2.Timeout(**timeout),
        transport=AsyncHTTPX2TenacityTransport(
            config=af._retry_config(httpx2.TransportError, httpx2.HTTPStatusError),
            wrapped=error_body_transport(inner), validate_response=lambda r: r.raise_for_status()))


def make_openai_compatible_model(base_url: str, model_name: str, *, uds_path: str | None = None,
                                 transport: Any = None, read_seconds: float | None = None) -> Any:
    """The model GRC-Agent itself builds for provider openai_compatible (agent_factory._build_model),
    on GRC-Agent's retrying client with error bodies read (retrying_client). `uds_path` reaches a
    llama-server listening on a UNIX socket; `transport` substitutes the innermost httpx2 transport
    (tests capture the wire request with httpx2.MockTransport). The retry wrapper and timeouts are
    GRC-Agent's in every case, apart from the read timeout when `read_seconds` is given.
    """
    from grc_agent import agent_factory as af
    cfg = {"provider": "openai_compatible", "model": model_name, "openai_compatible_base_url": base_url}
    if uds_path is None and transport is None:
        return af._build_model(cfg, retrying_client(read_seconds=read_seconds))
    import httpx2
    inner = transport if transport is not None else httpx2.AsyncHTTPTransport(uds=uds_path)
    return af._build_model(cfg, retrying_client(inner, read_seconds=read_seconds))


def capability_names(agent: Any) -> list[str]:
    """Class names of the agent's capabilities, in order (flattening combined capabilities)."""
    out: list[str] = []

    def walk(cap: Any) -> None:
        inner = getattr(cap, "capabilities", None)
        if isinstance(inner, (list, tuple)):
            for c in inner:
                walk(c)
        elif cap is not None:
            out.append(type(cap).__name__)

    walk(getattr(agent, "_root_capability", None))
    return out


async def freeze_tools(agent: Any, deps: Any = None) -> list[dict[str, Any]]:
    """Qoherent's tool snapshot (collect.py freeze_tools): the agent's toolset, sorted by name,
    as {name, description, parameters}. This is the `tools` field of the SFT row format.

    pydantic-ai 2.37 binds capability-contributed toolsets to the run's capability registry, so
    the context carries that registry (built by pydantic-ai's own `_build_run_capabilities`);
    collect.py's bare RunContext only worked on the older pydantic-ai of the collection. Output
    tools (GRC-Agent's `final_result` for GrcAgentResponse) are left out, as in the SFT release
    ("The `final_result` terminator tool isn't in the frozen tool schema", its README).
    """
    from pydantic_ai import RunContext
    from pydantic_ai.agent import _build_run_capabilities
    from pydantic_ai.usage import RunUsage
    ctx = RunContext(deps=deps, model=agent.model, usage=RunUsage(),
                     capabilities=_build_run_capabilities(agent._root_capability))
    tools = await agent._get_toolset().get_tools(ctx)
    frozen = []
    for name, tool in sorted(tools.items()):
        tool_def = getattr(tool, "tool_def", None)
        if getattr(tool_def, "kind", None) == "output":
            continue
        frozen.append({"name": str(getattr(tool_def, "name", name) or name),
                       "description": str(getattr(tool_def, "description", "") or ""),
                       "parameters": getattr(tool_def, "parameters_json_schema", None)})
    return frozen


def build_benchmark_agent(model: Any, deps: Any, project_dir: Path,
                          setting: Setting = "benchmark", *,
                          read_seconds: float | None = None) -> tuple[Any, dict[str, Any]]:
    """Build GRC-Agent's executor for one episode; returns (agent, info).

    model        a pydantic-ai Model, or None to let GRC-Agent build it from the run's .env
    deps         the episode's flowgraph deps (headless proxy or a bare GRC FlowGraph); the agent
                 is run with `deps=deps`
    project_dir  the episode project folder (inside the run directory): root of the file tools,
                 working directory of the (sandboxed) shell tools
    setting      "benchmark" (compaction off) or "grc_agent_defaults" (GRC-Agent's compaction)
    read_seconds the HTTP read timeout of GRC-Agent's client (retrying_client); the runner passes the
                 episode's wall-clock cap

    `info` records the pinned commit, the setting, the model, the capability list, the store and
    settings paths, and every substitution made, for the run record.
    """
    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}, got {setting!r}")
    layout = isolation.check_process_isolation()
    project_dir = Path(project_dir).resolve()
    if not project_dir.is_dir() or not isolation.is_inside(project_dir, layout.run_dir):
        raise isolation.IsolationError(f"project directory {project_dir} must be a folder inside "
                                       f"the run directory {layout.run_dir}")

    from grc_agent import agent_factory as af
    from grc_agent import db as grc_db
    from grc_agent import fs_tools
    from grc_agent import settings as grc_settings

    if grc_settings.env_path().resolve() != layout.env_file.resolve():
        raise isolation.IsolationError(f"GRC-Agent reads {grc_settings.env_path()}, not {layout.env_file}")
    cfg = grc_settings.load_settings()
    if cfg["provider"] != "openai_compatible":
        raise isolation.IsolationError(f"run .env selects provider {cfg['provider']!r}, "
                                       "expected openai_compatible")

    # The file tools' root and the shell's cwd follow these providers, as in GRC-Agent's desktop
    # app (desktop_app.py L274-278: canvas.path, canvas.current_flow_graph, sidebar project
    # directory). They are (re)installed for `deps` on every build, superseding any earlier wiring
    # in this process (HeadlessFlowgraphProxy installs the same desktop-app wiring for itself).
    fs_tools.set_active_graph_providers(grc_path_fn=lambda: _graph_path(deps),
                                        flow_graph_fn=lambda: _live_flow_graph(deps),
                                        project_dir_fn=lambda: project_dir)
    isolation.install_shell_sandbox(project_dir)

    substitutions = [
        f"{NATIVE_CANVAS}: GTK-free stand-in exposing NativeFlowgraphProxy = {type(deps).__name__}",
        "web_search_cap/web_fetch_cap: same tools, local-only, body returns the offline message",
        "shell commands: nested bubblewrap sandbox (project dir writable, no network)",
    ]
    patches: dict[str, Any] = {}
    grc_client = af._retrying_http_client
    patches["_retrying_http_client"] = (
        lambda provider="": grc_client(provider) if provider in af._HTTPX1_ONLY_PROVIDERS
        else retrying_client(read_seconds=read_seconds))
    substitutions.append("HTTP client: GRC-Agent's retrying client with the body of an error reply read "
                         "(a model server's context refusal is the context cap)")
    if read_seconds:
        substitutions.append(f"HTTP read timeout: {read_seconds:g} s, the episode's wall-clock cap, instead of "
                             f"GRC-Agent's {af._HTTP_TIMEOUT['read']:g} s")
    if model is not None:
        patches["_build_model"] = lambda _cfg, _http_client: model
        substitutions.append(f"model: given {type(model).__name__} instead of GRC-Agent's own build")
    if setting == "benchmark":
        patches["_build_compaction_capability"] = lambda _cfg, agent_name="grc_chat": CompactionOff()
        substitutions.append("compaction: off (CompactionOff)")
    search_cap, fetch_cap = offline_web_capabilities()
    patches["web_search_cap"] = search_cap
    patches["web_fetch_cap"] = fetch_cap

    with gtk_free_native_canvas(type(deps)), _swapped(af, **patches):
        bundle = af.build_agents_from_cfg(cfg)
    if bundle.model_build_error:
        raise RuntimeError(f"GRC-Agent could not build the model: {bundle.model_build_error}")
    agent = bundle.executor

    provider, base_url, model_name = af.describe_model(agent.model)
    info = {
        "grc_agent_commit": GRC_AGENT_COMMIT,
        "factory": "grc_agent.agent_factory.build_agents_from_cfg (executor role)",
        "agent_name": agent.name,
        "setting": setting,
        "caps": CAPS[setting],
        "model": {"class": type(agent.model).__name__, "provider": provider, "base_url": base_url,
                  "model_name": model_name},
        "capabilities": capability_names(agent),
        "substitutions": substitutions,
        "env_file": str(grc_settings.env_path()),
        "step_store": str(grc_db.get_db_path()),
        "project_dir": str(project_dir),
        "gtk_loaded": any((m == "gi" or m.startswith("gi.")) and mod is not None
                          for m, mod in list(sys.modules.items())),
    }
    return agent, info

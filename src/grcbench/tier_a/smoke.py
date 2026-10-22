"""The Tier-A smoke test (build plan Phase 4 "Smoke test"): a PLUMBING test of the harness, never a
model result.

Build plan: "FunctionModel replay first, then Qwen3-4B on 3 dev tasks. Record whether --jinja
grammar-constrains tool calls, and pin it, since that is the B0 vs. B2 distinction" (eval plan
Table 6: B2 = "grammar-constrained tool calls"). The project rule is stricter than the plan's
wording: no benchmark item (test OR dev) is ever run through a model before the run list allows it,
so the smoke test uses the unit-test fixture tasks under tests/fixtures only (their records say
source.kind "test_fixture"; `fixture_tasks` refuses anything else):

  1. golden replays   every Tier-A fixture task with a golden_calls.json, replayed by a FunctionModel
                      through the real runner and graded (grader.grade); every Tier-F fixture task
                      with a golden_calls.json, replayed in the Tier-F setting (run_tier_f_sample)
                      and graded V1-V6. All must pass.
  2. grammar probe    (with start_server) three short-lived llama-servers with the smoke flags and full
                      logging (-lv 5): the DEFAULT one, a CONTROL with --skip-chat-parsing (a pure
                      content parser: no tool-call parsing, no tool grammar) and one with --no-jinja.
                      Each gets the same tiny chat request whose instruction asks for tool arguments
                      that violate the tool's JSON schema (an enum value outside the enum, a string for
                      an integer), greedy, thinking off and on, with tool_choice auto and required;
                      plus one request that sends its own `grammar` together with tools. If the
                      default server's tool calls always satisfy the schema while the control writes
                      the violating values, the default build constrains tool calls to the schema with
                      a grammar. The server logs' chat-format and grammar lines (the compiled GBNF
                      holds the tool's enum) are kept as the second source. `tool_call_settings` turns
                      the verdict into the B0 vs. B2 flag settings (eval plan Table 6).
  3. real model       (with base_url, or start_server: the pinned llama.cpp build with the smoke model)
                      the server's /props and /v1/models, then `tier_f_samples` Tier-F samples ("one
                      lookup + one edit") of each Tier-F fixture in `tier_f_tasks`, then one graded
                      episode per Tier-A fixture task (A_ORDER), all through the real runner with
                      the benchmark caps. These are plumbing checks: a fixture a small model fails
                      is reported, not an error.

The smoke llama-server (`server_command`): llama.cpp LLAMA_CPP_BUILD with LD_LIBRARY_PATH set to the
build directory, the smoke model (Qwen3-4B Q4_K_M), 127.0.0.1 on a free port, one slot (-np 1),
q8_0 K and V cache with flash attention (eval plan §7.1: 8-bit KV cache in every Tier-A run),
Jinja chat templates (--jinja), reasoning returned separately (--reasoning-format deepseek: the
thoughts go to message.reasoning_content, which pydantic-ai turns into ThinkingParts, so the trace's
"thinking" field and the answer text stay apart), Qwen3's recommended thinking-mode sampling as the
server default (SAMPLING; GRC-Agent sends none), context shift off, SSE pings every 30 s, and the
context size `server_context(native)`: the model's native maximum (GGUF <arch>.context_length) up to
the 64K cap of eval plan §4.4. Real runs need at least 65,536 tokens per slot with context shift
off; a smaller smoke context is recorded as such. LLAMA_ARG_* variables are removed from the
server's environment, so only the command line sets it.

Every result goes to <WORK>/tier_a/smoke/<UTC time>/: summary.json (rewritten after every episode),
traces.jsonl (the model episodes' rows, Qoherent's SFT row format), golden_traces.jsonl, the server
logs and every run directory. `render_report` turns a summary into reports/tier_a_smoke.md.

Requests are streamed (models.py "Streaming"). The first real-model attempt (18 Oct) ran them
unstreamed, so GRC-Agent's 120 s HTTP read timeout cut off every response that took longer than
120 s to generate and the client re-sent it until its retries ran out.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import socket
import statistics
import struct
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from ..common.io import read_json, sha256_file, write_json, write_text
from ..common.paths import CACHE, FIXTURES, LLAMA_CPP_BUILD, REPORTS, WORK
from . import isolation, models

FIXTURE_A = FIXTURES / "tier_a" / "tasks"
FIXTURE_F = FIXTURES / "tier_f_task"
SMOKE_ROOT = WORK / "tier_a" / "smoke"
REPORT = REPORTS / "tier_a_smoke.md"
SMOKE_CONFIG = "smoke"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
PROPS_KEYS = ("model_path", "build_info", "chat_template_caps", "total_slots", "modalities")
SMOKE_MODEL = CACHE / "models" / "Qwen3-4B-Q4_K_M.gguf"
LLAMA_DIR = CACHE / "llama.cpp" / LLAMA_CPP_BUILD               # the official CPU build (./gb fetch)
LLAMA_DIR_CUDA = CACHE / "llama.cpp" / f"{LLAMA_CPP_BUILD}-cuda"   # the same tag built with CUDA (sm_120)
WSL_GPU_LIBS = Path("/usr/lib/wsl/lib")                           # libcuda in WSL (the Windows driver)
BACKENDS = ("auto", "cpu", "cuda")
GPU_LAYERS = "999"                                                # every layer on the GPU
CTX_CAP = 65536                                  # eval plan §4.4: context capped at 64K tokens
KV_CACHE_TYPE = "q8_0"                           # eval plan §7.1: 8-bit KV cache in every Tier-A run
REASONING_FORMAT = "deepseek"                    # thoughts in message.reasoning_content (module docstring)
SSE_PING_SECONDS = 30
SERVER_ALIAS = "Qwen3-4B-Q4_K_M"
# GRC-Agent sends no sampling settings (agent_factory: ModelSettings()), so the engine's defaults
# apply; eval plan §7.1: each model uses its recommended settings. Qwen3 model card, thinking mode:
# temperature 0.6, top-p 0.95, top-k 20, min-p 0 (no greedy decoding).
SAMPLING = ("--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0")
DEFAULT_TIER_F_TASKS = ("fx_modify_tone",)
DEFAULT_TIER_F_SAMPLES = 3
# Real-model order: the Tier-F samples first, then the Tier-A fixtures, expected shortest first.
A_ORDER = ("fx-a-knowledge-tone", "fx-a-modify-tone", "fx-a-repair-type", "fx-a-save-block", "fx-a-build-file")
LOG_MARKERS = ("chat format", "chat_format", "grammar", "jinja", "n_ctx", "context shift", "reasoning",
               "chat template", "thinking")
TOOL_CALL_TAG = "<tool_call>"


class NotAFixture(ValueError):
    """The smoke test runs fixture tasks only."""


def _is_fixture(task_dir: Path) -> bool:
    try:
        record = read_json(task_dir / "task.json")
    except (OSError, ValueError):
        return False
    return isolation.is_inside(task_dir, FIXTURES) and (record.get("source") or {}).get("kind") == "test_fixture"


def fixture_tasks(golden_file: str = "golden_calls.json") -> tuple[list[Path], list[Path]]:
    """(Tier-A fixture task dirs, Tier-F fixture task dirs) that have a golden_calls.json."""
    a = sorted(d for d in FIXTURE_A.iterdir() if (d / golden_file).is_file()) if FIXTURE_A.is_dir() else []
    f = sorted(d for d in FIXTURE_F.iterdir() if (d / golden_file).is_file()) if FIXTURE_F.is_dir() else []
    for d in a + f:
        if not _is_fixture(d):
            raise NotAFixture(f"{d} is not a unit-test fixture task")
    return a, f


def _local(base_url: str) -> str:
    host = urllib.parse.urlsplit(base_url).hostname
    if host not in LOCAL_HOSTS:
        raise ValueError(f"the smoke test only talks to a local llama-server, not {host!r}")
    return base_url


def _root_url(base_url: str) -> str:
    root = _local(base_url).rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def _get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:      # noqa: S310 - local only (_local)
        return json.loads(resp.read().decode("utf-8"))


def server_props(base_url: str, timeout: float = 10.0) -> dict[str, Any]:
    """The llama-server's /props and /v1/models, reduced to what the smoke record needs (local
    servers only)."""
    root = _root_url(base_url)
    props = _get_json(root + "/props", timeout)
    gen = props.get("default_generation_settings") or {}
    template = props.get("chat_template") or ""
    out = {k: props.get(k) for k in PROPS_KEYS if k in props}
    out.update(n_ctx=gen.get("n_ctx"), chat_template_chars=len(template),
               chat_template_has_tools="tools" in template)
    try:
        listed = _get_json(root + "/v1/models", timeout).get("data") or []
        meta = (listed[0].get("meta") or {}) if listed else {}
        out["model_id"] = listed[0].get("id") if listed else None
        out["model_meta"] = {k: meta.get(k) for k in ("n_ctx_train", "n_params", "size", "n_vocab") if k in meta}
    except (urllib.error.URLError, OSError, ValueError, AttributeError, IndexError) as exc:
        out["model_meta"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------------------------
# The model file and the server
# --------------------------------------------------------------------------------------------

_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q",
                 12: "<d"}
_GGUF_STRING, _GGUF_ARRAY = 8, 9


def gguf_metadata(path: Path, wanted: Callable[[str], bool] | None = None) -> dict[str, Any]:
    """Scalar and string key/value metadata of a GGUF file (format v2/v3), read from its header.
    `wanted(key)` limits what is kept; arrays (the tokenizer's) are skipped, never kept."""
    out: dict[str, Any] = {}
    with open(path, "rb") as fh:
        def read(fmt: str) -> Any:
            size = struct.calcsize(fmt)
            data = fh.read(size)
            if len(data) != size:
                raise ValueError(f"{path}: truncated GGUF header")
            return struct.unpack(fmt, data)[0]

        def string() -> str:
            return fh.read(read("<Q")).decode("utf-8", errors="replace")

        def skip(vtype: int) -> None:
            if vtype in _GGUF_SCALARS:
                fh.seek(struct.calcsize(_GGUF_SCALARS[vtype]), os.SEEK_CUR)
            elif vtype == _GGUF_STRING:
                fh.seek(read("<Q"), os.SEEK_CUR)
            elif vtype == _GGUF_ARRAY:
                etype, n = read("<I"), read("<Q")
                if etype in _GGUF_SCALARS:
                    fh.seek(n * struct.calcsize(_GGUF_SCALARS[etype]), os.SEEK_CUR)
                else:
                    for _ in range(n):
                        skip(etype)
            else:
                raise ValueError(f"{path}: unknown GGUF value type {vtype}")

        if fh.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = read("<I")
        if version < 2:
            raise ValueError(f"{path}: GGUF version {version} is not supported")
        read("<Q")                                          # tensor count
        for _ in range(read("<Q")):
            key, vtype = string(), read("<I")
            keep = wanted is None or wanted(key)
            if keep and vtype in _GGUF_SCALARS:
                out[key] = read(_GGUF_SCALARS[vtype])
            elif keep and vtype == _GGUF_STRING:
                out[key] = string()
            else:
                skip(vtype)
    return out


def native_context(model: Path = SMOKE_MODEL) -> int | None:
    """The model's trained context length (GGUF <general.architecture>.context_length)."""
    meta = gguf_metadata(model, lambda k: k == "general.architecture" or k.endswith(".context_length"))
    arch = meta.get("general.architecture")
    value = meta.get(f"{arch}.context_length") if arch else None
    return int(value) if value else None


def server_context(native: int | None, cap: int = CTX_CAP) -> int:
    """The smoke server's context: the native maximum up to the 64K cap (module docstring)."""
    return min(int(native), cap) if native else cap


def default_threads() -> int:
    return max(1, min(16, (os.cpu_count() or 8) - 4))


def llama_backend(choice: str | None = None) -> str:
    """"cpu" or "cuda". The choice comes from the argument, else GRC_BENCH_LLAMA_BACKEND, else "auto":
    CUDA when the CUDA build of the pinned tag and the WSL GPU driver library are both present."""
    choice = choice or os.environ.get("GRC_BENCH_LLAMA_BACKEND", "auto")
    if choice not in BACKENDS:
        raise ValueError(f"llama backend {choice!r}: use one of {BACKENDS}")
    if choice == "auto":
        cuda = (LLAMA_DIR_CUDA / "llama-server").is_file() and (WSL_GPU_LIBS / "libcuda.so.1").exists()
        return "cuda" if cuda else "cpu"
    return choice


def llama_dir(backend: str | None = None) -> Path:
    return LLAMA_DIR_CUDA if llama_backend(backend) == "cuda" else LLAMA_DIR


def server_command(port: int, *, ctx: int, model: Path = SMOKE_MODEL, threads: int | None = None,
                   extra: tuple[str, ...] = (), backend: str | None = None) -> list[str]:
    """The smoke llama-server command line (module docstring); with the CUDA build every layer is
    offloaded to the GPU (-ngl)."""
    threads = threads or default_threads()
    gpu = ("-ngl", GPU_LAYERS) if llama_backend(backend) == "cuda" else ()
    return [str(llama_dir(backend) / "llama-server"), "-m", str(model), *gpu, "--host", "127.0.0.1", "--port", str(port),
            "-np", "1", "-c", str(ctx), "-ctk", KV_CACHE_TYPE, "-ctv", KV_CACHE_TYPE, "-fa", "on",
            "--jinja", "--reasoning-format", REASONING_FORMAT, "--no-context-shift",
            "--sse-ping-interval", str(SSE_PING_SECONDS), *SAMPLING, "-t", str(threads), "-tb", str(threads),
            "--no-webui", "-a", SERVER_ALIAS, *extra]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def server_env(backend: str | None = None) -> dict[str, str]:
    """The server's environment: LD_LIBRARY_PATH = the build directory (plus WSL's GPU driver
    libraries for the CUDA build), no LLAMA_ARG_* settings."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
    dirs = [str(llama_dir(backend))] + ([str(WSL_GPU_LIBS)] if llama_backend(backend) == "cuda" else [])
    env["LD_LIBRARY_PATH"] = ":".join(dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    return env


def gpu_name() -> str | None:
    """The GPU nvidia-smi reports (name, memory, driver), or None without one."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None


_TIMING = re.compile(r"(prompt eval|eval) time =\s*([\d.]+) ms /\s*(\d+) tokens")


def server_timing(log_text: str) -> dict[str, Any]:
    """The engine's own speed over a run, from llama-server's per-request print_timing lines:
    prompt tokens processed (the prefix cache already skips the rest) and tokens generated, with
    their totals in seconds and tokens per second."""
    sums = {"prompt eval": [0.0, 0, 0], "eval": [0.0, 0, 0]}          # ms, tokens, requests
    for line in log_text.splitlines():
        m = _TIMING.search(line)
        if m:
            s = sums[m.group(1)]
            s[0] += float(m.group(2))
            s[1] += int(m.group(3))
            s[2] += 1
    out: dict[str, Any] = {}
    for key, name in (("prompt eval", "prompt"), ("eval", "generation")):
        ms, tokens, n = sums[key]
        out[name] = {"requests": n, "tokens": tokens, "seconds": round(ms / 1000, 1),
                     "tokens_per_second": round(tokens / (ms / 1000), 2) if ms else None}
    return out


_PR_SET_PDEATHSIG = 1


def _die_with_parent() -> Callable[[], None] | None:
    """A preexec_fn that makes the child receive SIGTERM when the process that started it dies
    (Linux prctl PR_SET_PDEATHSIG), so a killed smoke run never leaves a llama-server behind. libc
    and prctl are resolved here, in the parent; the child only makes the call."""
    try:
        import ctypes
        import signal
        prctl = ctypes.CDLL("libc.so.6", use_errno=True).prctl
    except (OSError, AttributeError):
        return None
    sig = int(signal.SIGTERM)

    def preexec() -> None:
        prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0)
    return preexec


class LocalServer:
    """`with LocalServer(log, ctx=...) as base_url:` runs a smoke llama-server and always stops it:
    on leaving the block, and (PR_SET_PDEATHSIG) when the starting thread's process dies, so start
    it from the main thread. The CLI turns SIGTERM / SIGHUP into SystemExit so the block is left."""

    def __init__(self, log_path: Path, *, ctx: int, threads: int | None = None, extra: tuple[str, ...] = (),
                 model: Path = SMOKE_MODEL, ready_timeout: float = 600.0) -> None:
        self.log_path = Path(log_path)
        self.ready_timeout = ready_timeout
        self.model = Path(model)
        self.proc: subprocess.Popen | None = None
        self.port = _free_port()
        self.command = server_command(self.port, ctx=ctx, model=self.model, threads=threads, extra=extra)
        self.started_s: float | None = None
        self.load_seconds: float | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as resp:  # noqa: S310
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def __enter__(self) -> str:
        if not self.model.is_file():
            raise FileNotFoundError(f"smoke model missing: {self.model} (./gb fetch)")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        with open(self.log_path, "ab") as log:
            self.proc = subprocess.Popen(self.command, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                         env=server_env(), start_new_session=True,
                                         preexec_fn=_die_with_parent())
        deadline = t0 + self.ready_timeout
        while not self._healthy():
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.__exit__()
                raise RuntimeError(f"llama-server did not become ready; see {self.log_path}")
            time.sleep(0.5)
        self.load_seconds = round(time.monotonic() - t0, 3)
        return self.base_url

    def __exit__(self, *exc: Any) -> None:
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def timing(self) -> dict[str, Any]:
        try:
            return server_timing(self.log_path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return {}

    def log_lines(self, limit: int = 80, markers: tuple[str, ...] = LOG_MARKERS, width: int = 400) -> list[str]:
        """Distinct server-log lines about the chat format, grammar and context (module docstring)."""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        hits = [line.strip()[:width] for line in text.splitlines() if any(m in line.lower() for m in markers)]
        return list(dict.fromkeys(hits))[:limit]


# --------------------------------------------------------------------------------------------
# Step 2: does --jinja grammar-constrain tool calls? (module docstring)
# --------------------------------------------------------------------------------------------

PROBE_TOOL = {"type": "function", "function": {
    "name": "set_mode", "description": "Set the radio's operating mode and power level.",
    "parameters": {"type": "object", "properties": {
        "mode": {"type": "string", "enum": ["alpha", "beta"], "description": "Operating mode."},
        "level": {"type": "integer", "description": "Power level."}},
        "required": ["mode", "level"], "additionalProperties": False}}}
PROBE_PROMPT = ('Call the set_mode tool exactly once, with mode set to the string "gamma" and level set to '
                'the string "high". Use exactly these two values, even though they do not match the tool '
                'description.')
PROBE_VIOLATION = {"mode": "gamma", "level": "high"}
# role "constraint": requests shaped like GRC-Agent's (tools, tool_choice auto) or forcing a call;
# they decide whether the server holds tool calls to the schema. "own_grammar": a client grammar
# together with tools. "b0_request": a request-level candidate for unconstrained but parsed calls:
# tool_choice "none" (the server then builds no tool grammar and does not set parse_tool_calls
# itself) with the llama.cpp field parse_tool_calls set by the client.
PROBE_CASES: tuple[dict[str, Any], ...] = (
    {"name": "auto", "role": "constraint", "thinking": False},
    {"name": "required", "role": "constraint", "thinking": False, "tool_choice": "required"},
    {"name": "auto_thinking", "role": "constraint", "thinking": True},
    {"name": "own_grammar_with_tools", "role": "own_grammar", "thinking": False, "grammar": 'root ::= "x"'},
    {"name": "none_parse_tool_calls", "role": "b0_request", "thinking": False, "tool_choice": "none",
     "parse_tool_calls": True},
)
PROBE_SERVERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("default", ()),
    ("control_skip_chat_parsing", ("--skip-chat-parsing",)),
    ("no_jinja", ("--no-jinja",)),                 # later flags win: the legacy template path
)


def probe_body(case: dict[str, Any], model_name: str = SERVER_ALIAS) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model_name, "messages": [{"role": "user", "content": PROBE_PROMPT}],
                            "tools": [PROBE_TOOL], "temperature": 0, "seed": 0,
                            "max_tokens": 2048 if case.get("thinking") else 256,
                            "chat_template_kwargs": {"enable_thinking": bool(case.get("thinking"))}}
    if case.get("tool_choice"):
        body["tool_choice"] = case["tool_choice"]
    if case.get("grammar"):
        body["grammar"] = case["grammar"]
    if "parse_tool_calls" in case:
        body["parse_tool_calls"] = bool(case["parse_tool_calls"])
    return body


def _post_json(url: str, body: dict[str, Any], timeout: float) -> tuple[int, Any, float]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:        # noqa: S310 - local only
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    seconds = round(time.monotonic() - t0, 3)
    try:
        return status, json.loads(raw.decode("utf-8")), seconds
    except (ValueError, UnicodeDecodeError):
        return status, raw.decode("utf-8", errors="replace")[:2000], seconds


def _schema_errors(args: Any) -> list[str]:
    import jsonschema
    if not isinstance(args, dict):
        return ["arguments are not a JSON object"]
    validator = jsonschema.Draft202012Validator(PROBE_TOOL["function"]["parameters"])
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"[:200]
            for e in sorted(validator.iter_errors(args), key=lambda e: list(e.absolute_path))]


def calls_in_text(text: str) -> list[dict[str, Any]]:
    """Tool calls written as text (Qwen3/Hermes `<tool_call>{json}</tool_call>`), as the control
    server returns them unparsed in the content."""
    out = []
    for chunk in (text or "").split(TOOL_CALL_TAG)[1:]:
        body = chunk.split("</tool_call>")[0].strip()
        try:
            obj = json.loads(body)
        except (ValueError, json.JSONDecodeError):
            out.append({"name": None, "arguments": body[:500], "parsed": False})
            continue
        if isinstance(obj, dict):
            out.append({"name": obj.get("name"), "arguments": obj.get("arguments"), "parsed": True})
    return out


def probe_record(case: dict[str, Any], status: int, reply: Any, seconds: float) -> dict[str, Any]:
    """What one probe request showed (pure: tests feed it canned replies)."""
    rec: dict[str, Any] = {"case": case["name"], "role": case.get("role", "constraint"), "http_status": status,
                           "seconds": seconds, "tool_choice": case.get("tool_choice") or "auto",
                           "thinking": bool(case.get("thinking"))}
    if status != 200 or not isinstance(reply, dict):
        err = reply.get("error") if isinstance(reply, dict) else reply
        rec["error"] = (err.get("message") if isinstance(err, dict) else str(err))[:500]
        return rec
    choice = (reply.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    calls = []
    for c in msg.get("tool_calls") or []:
        raw = (c.get("function") or {}).get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, json.JSONDecodeError):
            args = None
        calls.append({"name": (c.get("function") or {}).get("name"), "arguments": raw,
                      "schema_errors": _schema_errors(args) if args is not None else ["arguments are not valid JSON"]})
    text_calls = calls_in_text(content)
    for c in text_calls:
        args = c["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, json.JSONDecodeError):
                args = None
        c["schema_errors"] = _schema_errors(args) if args is not None else ["arguments are not valid JSON"]
    rec.update(finish_reason=choice.get("finish_reason"), tool_calls=calls, text_tool_calls=text_calls,
               content=content[:600], reasoning_chars=len(msg.get("reasoning_content") or ""),
               usage=reply.get("usage"))
    return rec


def request_log_evidence(text: str, width: int = 300) -> dict[str, Any]:
    """What the server logged while handling ONE request (the slice of its -lv 5 log): the grammar
    kinds it built ("Grammar (<kind>): ..."), the compiled root rule, whether a lazy grammar was
    triggered or still awaited its trigger, and the chat format."""
    lines = [ln.strip() for ln in text.splitlines()]
    kinds = []
    for ln in lines:
        if "Grammar (" in ln and "):" in ln:
            kinds.append(ln.split("Grammar (", 1)[1].split(")", 1)[0])
    roots = [ln[:width] for ln in lines if ln.startswith("root ::=")]
    return {"grammar_kinds": list(dict.fromkeys(kinds)), "root_rules": list(dict.fromkeys(roots))[:2],
            "triggered": any("grammar triggered" in ln.lower() for ln in lines),
            "awaiting_trigger": any("awaiting trigger" in ln.lower() for ln in lines),
            "chat_format": _chat_format([ln for ln in lines if "chat format" in ln.lower()])}


def run_probe_cases(base_url: str, timeout: float = 900.0, log_path: Path | None = None) -> list[dict[str, Any]]:
    """Every PROBE_CASES request, in order; with `log_path` (the server's log) each record also
    gets the log evidence of its own request (request_log_evidence)."""
    url = _root_url(base_url) + "/v1/chat/completions"
    out = []
    for case in PROBE_CASES:
        start = log_path.stat().st_size if log_path is not None and log_path.is_file() else 0
        status, reply, seconds = _post_json(url, probe_body(case), timeout)
        rec = probe_record(case, status, reply, seconds)
        if log_path is not None:
            time.sleep(1.0)                      # the server's logger thread writes asynchronously
            with open(log_path, "rb") as fh:
                fh.seek(start)
                rec["log"] = request_log_evidence(fh.read().decode("utf-8", errors="replace"))
        out.append(rec)
    return out


def _b0_request_record(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """The request-level B0 candidate (role "b0_request") on one server, read from its record."""
    rec = next((r for r in cases if r.get("role") == "b0_request"), None)
    if rec is None:
        return {}
    parsed = rec.get("tool_calls") or []
    log = rec.get("log") or {}
    return {"case": rec["case"], "http_status": rec.get("http_status"), "error": rec.get("error"),
            "parsed_tool_calls": len(parsed), "parsed_calls_violating": sum(1 for c in parsed if c["schema_errors"]),
            "text_tool_calls": len(rec.get("text_tool_calls") or []),
            "grammar_kinds": log.get("grammar_kinds"), "chat_format": log.get("chat_format"),
            "arguments": [c.get("arguments") for c in parsed][:2]}


def probe_verdict(servers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The probe's reading (module docstring): is the DEFAULT server's tool-call output held to the
    tool schema while the CONTROL (no tool grammar) writes the schema-violating values? Only the
    "constraint" cases decide it; the request-level B0 candidate is read separately."""
    default = (servers.get("default") or {}).get("cases") or []
    control = (servers.get("control_skip_chat_parsing") or {}).get("cases") or []

    def calls(rec: dict[str, Any]) -> list[dict[str, Any]]:
        return (rec.get("tool_calls") or []) + (rec.get("text_tool_calls") or [])

    def constraint(rec: dict[str, Any]) -> bool:
        return rec.get("role", "constraint") == "constraint"
    tool_cases = [r for r in default if constraint(r) and r.get("http_status") == 200]
    default_calls = [c for r in tool_cases for c in r.get("tool_calls") or []]
    default_text_calls = [c for r in tool_cases for c in r.get("text_tool_calls") or []]
    control_calls = [c for r in control if constraint(r) for c in calls(r)]
    own = next((r for r in default if r.get("role") == "own_grammar" or r["case"] == "own_grammar_with_tools"), {})

    def grammar_lines(name: str) -> list[str]:
        srv = servers.get(name) or {}
        if srv.get("log_evidence"):
            return srv["log_evidence"].get("grammar_lines") or []
        return [ln for ln in srv.get("log_lines") or [] if "grammar" in ln.lower()]
    out = {
        "default_parsed_tool_calls": len(default_calls),
        "default_calls_schema_valid": sum(1 for c in default_calls if not c["schema_errors"]),
        "default_unparsed_text_calls": len(default_text_calls),
        "control_text_calls": len(control_calls),
        "control_calls_schema_valid": sum(1 for c in control_calls if not c.get("schema_errors")),
        "control_calls_violating": sum(1 for c in control_calls if c.get("schema_errors")),
        "own_grammar_with_tools_refused": bool(own) and own.get("http_status") != 200,
        "own_grammar_with_tools_error": own.get("error"),
        "default_grammar_log_lines": len(grammar_lines("default")),
        "control_grammar_log_lines": len(grammar_lines("control_skip_chat_parsing")),
        "default_enum_rule_lines": len(((servers.get("default") or {}).get("log_evidence") or {}).get("enum_rule_lines") or []),
        "control_enum_rule_lines": len(((servers.get("control_skip_chat_parsing") or {}).get("log_evidence") or {})
                                       .get("enum_rule_lines") or []),
    }
    out["default_schema_constrained"] = bool(default_calls) and out["default_calls_schema_valid"] == len(default_calls)
    out["control_unconstrained"] = out["control_calls_violating"] > 0
    out["grammar_constrains_tool_calls"] = (out["default_schema_constrained"] and out["control_unconstrained"]
                                            and out["default_grammar_log_lines"] > 0)
    out["control_parsed_tool_calls"] = sum(len(r.get("tool_calls") or []) for r in control if constraint(r))
    out["default_case_grammars"] = {r["case"]: {k: (r.get("log") or {}).get(k) for k in
                                                ("grammar_kinds", "triggered", "awaiting_trigger", "root_rules")}
                                    for r in default if r.get("log")}
    out["default_b0_request"] = _b0_request_record(default)
    nj = [r for r in (servers.get("no_jinja") or {}).get("cases") or [] if constraint(r)]
    out["no_jinja_cases"] = len(nj)
    out["no_jinja_tool_requests_refused"] = sum(1 for r in nj if r.get("http_status") != 200)
    out["no_jinja_parsed_tool_calls"] = sum(len(r.get("tool_calls") or []) for r in nj)
    out["no_jinja_error"] = next((r.get("error") for r in nj if r.get("error")), None)
    out["chat_formats"] = {name: _chat_format((srv.get("log_evidence") or {}).get("chat_format_lines") or [])
                           for name, srv in servers.items()}
    return out


def _chat_format(lines: list[str]) -> str | None:
    """The format name of the first "chat format: <name>" server-log line."""
    for line in lines:
        low = line.lower()
        for key in ("chat format:", "chat_format:"):
            if key in low:
                return line[low.index(key) + len(key):].strip() or None
    return None


def tool_call_settings(verdict: dict[str, Any]) -> dict[str, Any]:
    """B0 (tool calls as the model writes them) vs. B2 ("grammar-constrained tool calls", eval plan
    Table 6) on this build, read from the probe verdict. Only the tool-call flags differ; every
    other server flag is the smoke command's (or the run's) own. A setting is usable by GRC-Agent
    as is only if the server returns the calls parsed (OpenAI `tool_calls`): GRC-Agent never reads
    a call out of the answer text."""
    b2 = {"server_flags": "--jinja (Jinja chat template, chat parsing on: the build's default)",
          "request": "tools with tool_choice \"auto\", as GRC-Agent sends them",
          "chat_format": (verdict.get("chat_formats") or {}).get("default"),
          "tool_calls_schema_constrained": bool(verdict.get("default_schema_constrained")),
          "grammar_in_log": (verdict.get("default_grammar_log_lines") or 0) > 0,
          "server_parses_tool_calls": (verdict.get("default_parsed_tool_calls") or 0) > 0}
    b0 = {"server_flags": "--jinja --skip-chat-parsing",
          "chat_format": (verdict.get("chat_formats") or {}).get("control_skip_chat_parsing"),
          "tool_calls_schema_constrained": not verdict.get("control_unconstrained"),
          "grammar_in_log": (verdict.get("control_grammar_log_lines") or 0) > 0,
          "server_parses_tool_calls": (verdict.get("control_parsed_tool_calls") or 0) > 0}
    no_jinja = {"server_flags": "--no-jinja", "cases": verdict.get("no_jinja_cases"),
                "tool_requests_refused": verdict.get("no_jinja_tool_requests_refused"),
                "server_parses_tool_calls": (verdict.get("no_jinja_parsed_tool_calls") or 0) > 0,
                "error": verdict.get("no_jinja_error")}
    b2["usable_by_grc_agent_as_is"] = b2["server_parses_tool_calls"]
    b0["usable_by_grc_agent_as_is"] = b0["server_parses_tool_calls"]
    flag_only_b0 = (not b0["tool_calls_schema_constrained"]) and b0["server_parses_tool_calls"]
    r = verdict.get("default_b0_request") or {}
    parsed, violating = r.get("parsed_tool_calls") or 0, r.get("parsed_calls_violating") or 0
    b0_request = {"server_flags": "--jinja (the B2 server)",
                  "request": "tool_choice \"none\" + \"parse_tool_calls\": true (a llama.cpp request field)",
                  "ran": bool(r), "http_status": r.get("http_status"), "error": r.get("error"),
                  "chat_format": r.get("chat_format"),
                  "server_parses_tool_calls": parsed > 0, "parsed_calls_violating_schema": violating,
                  "grammar_in_log": bool(r.get("grammar_kinds")), "calls_left_in_text": r.get("text_tool_calls"),
                  "arguments": r.get("arguments"),
                  # GRC-Agent sends tool_choice "auto"; the harness would have to rewrite the two fields
                  "usable_by_grc_agent_as_is": False}
    request_b0 = bool(r) and r.get("http_status") == 200 and parsed > 0 and violating > 0
    return {"B2": b2, "B0_candidate": b0, "B0_request_candidate": b0_request, "no_jinja": no_jinja,
            "client_grammar_with_tools_refused": bool(verdict.get("own_grammar_with_tools_refused")),
            "default_is_grammar_constrained": bool(verdict.get("grammar_constrains_tool_calls")),
            "flag_only_unconstrained_setting_with_parsed_calls": flag_only_b0,
            "request_level_unconstrained_with_parsed_calls": request_b0}


def probe_log_evidence(text: str, width: int = 600, limit: int = 12) -> dict[str, list[str]]:
    """Grammar evidence in a probe server's full log: lines that name a grammar, GBNF rule lines
    (`::=`) holding the probe tool's enum values (a grammar built from the tool schema), lines about
    grammar triggers / lazy grammars, and the chat-format lines."""
    lines = [ln.strip() for ln in text.splitlines()]

    def pick(pred: Callable[[str], bool]) -> list[str]:
        return list(dict.fromkeys(ln[:width] for ln in lines if pred(ln)))[:limit]
    return {"grammar_lines": pick(lambda ln: "grammar" in ln.lower()),
            "enum_rule_lines": pick(lambda ln: "::=" in ln and "alpha" in ln and "beta" in ln),
            "trigger_lines": pick(lambda ln: "trigger" in ln.lower() or "lazy" in ln.lower()),
            "chat_format_lines": pick(lambda ln: "chat format" in ln.lower() or "chat_format" in ln.lower())}


def grammar_probe(root: Path, *, ctx: int, threads: int | None = None) -> dict[str, Any]:
    """Step 2 (module docstring): both probe servers, each started and always stopped."""
    servers: dict[str, dict[str, Any]] = {}
    for name, extra in PROBE_SERVERS:
        server = LocalServer(root / f"probe_{name}.log", ctx=ctx, threads=threads, extra=(*extra, "-lv", "5"))
        rec: dict[str, Any] = {"command": server.command, "log": str(server.log_path)}
        try:
            base_url = server.__enter__()
            rec["load_seconds"] = server.load_seconds
            rec["props"] = server_props(base_url)
            rec["cases"] = run_probe_cases(base_url, log_path=server.log_path)
        except Exception as exc:                 # noqa: BLE001 - recorded; the smoke run goes on
            rec["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            server.__exit__()
            try:
                rec["log_evidence"] = probe_log_evidence(server.log_path.read_text(encoding="utf-8", errors="replace"))
            except OSError as exc:
                rec["log_evidence"] = {"error": [str(exc)]}
        servers[name] = rec
    verdict = probe_verdict(servers)
    return {"servers": servers, "verdict": verdict, "tool_call_settings": tool_call_settings(verdict),
            "prompt": PROBE_PROMPT, "tool": PROBE_TOOL}


# --------------------------------------------------------------------------------------------
# Steps 1 and 3
# --------------------------------------------------------------------------------------------

def _stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def machine_state() -> dict[str, Any]:
    """CPU count, load averages and the other llama.cpp processes running now: the smoke CPU is
    shared with other jobs, so the wall times are read against this."""
    from ..tier_e.bench import llama_processes
    try:
        load = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        load = None
    return {"time": _utc(), "cpus": os.cpu_count(), "load_1_5_15": load,
            "other_llama_processes": [f"{p['name']} (pid {p['pid']})" for p in llama_processes()]}


def _a_line(kind: str, res: Any, grade: dict[str, Any]) -> str:
    return (f"{kind:7s} A {res.task_id:24s} {'PASS' if grade.get('passed') else 'fail':4s} {res.status:20s} "
            f"requests={res.usage.get('requests')} calls={len(res.tool_calls)}")


def _f_line(kind: str, res: Any) -> str:
    g = (res.tier_f or {}).get("grade") or {}
    return (f"{kind:7s} F {res.task_id:24s} s{res.sample:02d} {'PASS' if g.get('passed') else 'fail':4s} "
            f"{res.status:20s} {g.get('failure') or ''}")


def _golden_replays(a_dirs: list[Path], f_dirs: list[Path], root: Path, reps: int, workers: int,
                    summary: dict[str, Any]) -> list[str]:
    """Step 1 of the smoke test (module docstring); returns the run directories."""
    from ..tier_f import golden as tf_golden
    from .replay import golden_model, replay, scripted_run_log_entry
    from .tier_f_mode import run_tier_f_sample

    def golden_a(job: tuple[Path, int]) -> tuple[Any, dict]:
        d, k = job
        return replay(d, golden_model(d), root / "golden", sample=k, what="smoke golden replay")

    def golden_f(job: tuple[Path, int]) -> Any:
        d, k = job
        spec = models.scripted(tf_golden.golden_turns(tf_golden.load_golden_calls(d)))
        return run_tier_f_sample(d, spec, root / "golden", config_id=SMOKE_CONFIG, sample=k,
                                 run_log_entry=scripted_run_log_entry(spec, "smoke golden replay"))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        a_res = list(pool.map(golden_a, [(d, k) for d in a_dirs for k in range(reps)]))
        f_res = list(pool.map(golden_f, [(d, k) for d in f_dirs for k in range(reps)]))
    for res, g in a_res:
        summary["golden"].append({"tier": "A", "task": res.task_id, "sample": res.sample, "passed": g["passed"],
                                  "status": res.status, "tool_calls": len(res.tool_calls),
                                  "reasons": g["reasons"][:5], "run_dir": res.run_dir})
        summary["lines"].append(_a_line("golden", res, g))
    for res in f_res:
        g = (res.tier_f or {}).get("grade") or {}
        summary["golden"].append({"tier": "F", "task": res.task_id, "sample": res.sample,
                                  "passed": bool(g.get("passed")), "status": res.status,
                                  "failure": g.get("failure"), "run_dir": res.run_dir})
        summary["lines"].append(_f_line("golden", res))
    return [r.run_dir for r, _ in a_res] + [r.run_dir for r in f_res]


def _unparsed_tool_call_texts(run_dir: Path) -> int:
    """Assistant text parts that contain a tool-call tag: calls the server did not parse."""
    from pydantic_ai.messages import ModelResponse, TextPart

    from .trace import load_messages
    path = Path(run_dir) / "messages.json"
    if not path.is_file():
        return 0
    return sum(1 for m in load_messages(path) if isinstance(m, ModelResponse)
               for p in m.parts if isinstance(p, TextPart) and TOOL_CALL_TAG in (p.content or ""))


def _request_stats(res: Any) -> dict[str, Any]:
    reqs = [r for r in res.model_requests if r.get("seconds") is not None]
    secs = [r["seconds"] for r in reqs]
    return {"requests": len(res.model_requests), "request_errors": sum(1 for r in res.model_requests if r.get("error")),
            "median_request_seconds": round(statistics.median(secs), 1) if secs else None,
            "max_request_seconds": round(max(secs), 1) if secs else None,
            "output_tokens": res.usage.get("output_tokens"), "input_tokens": res.usage.get("input_tokens"),
            "max_request_input_tokens": res.usage.get("max_request_input_tokens")}


def tier_a_record(res: Any) -> dict[str, Any]:
    """The smoke record of one Tier-A model episode (EpisodeResult with .grade)."""
    g = res.grade or {}
    calls = [c for c in res.tool_calls if not c.get("prefilled")]
    malformed = [c for c in calls if not c.get("schema_valid")]
    return {"tier": "A", "task": res.task_id, "sample": res.sample, "status": res.status,
            "completed": res.status == "completed", "end_reason": res.end_reason,
            "passed": g.get("passed"), "grade_valid": g.get("valid", True),
            "tool_calls": len(calls), "tools_used": [c["tool"] for c in calls],
            "malformed_calls": len(malformed), "invalid_json_calls": sum(1 for c in calls if not c.get("json_object")),
            "unknown_tool_calls": len(res.unknown_tool_calls),
            "repaired_calls": sum(1 for c in calls if c.get("repair_changed_args")),
            "malformed_examples": [{"tool": c["tool"], "errors": (c.get("schema_errors") or [])[:2],
                                    "raw_args": (c.get("raw_args") or "")[:300]} for c in malformed[:3]],
            "unparsed_tool_call_texts": _unparsed_tool_call_texts(Path(res.run_dir)),
            "caps_hit": [h.get("cap") for h in res.caps_hit], "truncated_tool_results": res.usage.get("tool_results_truncated"),
            "wall_seconds": (res.process or {}).get("seconds"), "agent_seconds": res.timing.get("agent_seconds"),
            **_request_stats(res), "reasons": (g.get("reasons") or [])[:6],
            "error": (res.error or {}).get("message"), "final_text": (res.final_text or "")[:400],
            "run_dir": res.run_dir}


def tier_f_record(res: Any) -> dict[str, Any]:
    """The smoke record of one Tier-F sample (EpisodeResult with .tier_f)."""
    rec = res.tier_f or {}
    g = rec.get("grade") or {}
    lookup, edit = rec.get("lookup") or {}, rec.get("edit") or {}
    return {"tier": "F", "task": res.task_id, "sample": res.sample, "status": res.status,
            "end_reason": res.end_reason, "passed": g.get("passed"), "failure": g.get("failure"),
            "highest_passed": g.get("highest_passed"), "first_failed": g.get("first_failed"),
            "rungs": {k: bool(v.get("passed")) for k, v in (g.get("rungs") or {}).items()},
            "lookup_used": lookup.get("used"), "lookup_query": lookup.get("query"),
            "lookup_domain": lookup.get("domain"), "lookup_k": lookup.get("k"),
            "lookup_reference_recall": lookup.get("reference_recall"),
            "lookup_retrieved": (lookup.get("retrieved_block_ids") or [])[:8],
            "edit_called": edit.get("called"), "edit_well_formed": edit.get("well_formed"),
            "edit_accepted": edit.get("accepted"), "edit_error_codes": edit.get("error_codes"),
            "edit_raw_args": (edit.get("raw_args") or "")[:600],
            "edit_same_reply_as_lookup": rec.get("edit_same_reply_as_lookup"),
            "blocked_calls": [b["tool"] for b in rec.get("blocked_calls") or []],
            "model_requests": rec.get("model_requests"), "caps_hit": [h.get("cap") for h in res.caps_hit],
            "unparsed_tool_call_texts": _unparsed_tool_call_texts(Path(res.run_dir)),
            "wall_seconds": (res.process or {}).get("seconds"), **_request_stats(res),
            "error": (res.error or {}).get("message"), "final_text": (res.final_text or "")[:400],
            "run_dir": res.run_dir}


def ordered_a_dirs(a_dirs: list[Path]) -> list[Path]:
    """Tier-A fixtures in A_ORDER (expected shortest episodes first), then the rest by name."""
    rank = {name: i for i, name in enumerate(A_ORDER)}
    return sorted(a_dirs, key=lambda d: (rank.get(d.name, len(rank)), d.name))


def _model_episodes(base_url: str, model_name: str, a_dirs: list[Path], f_dirs: list[Path], f_samples: int,
                    root: Path, summary: dict[str, Any], save: Callable[[], None],
                    episode_seconds: float | None = None) -> dict[str, Any]:
    """Step 3 of the smoke test: graded Tier-F samples, then Tier-A episodes, with the real model."""
    from .caps import Caps
    from .runner import run_episode
    from .tier_f_mode import run_tier_f_sample, tier_f_caps
    spec = models.openai_compatible(_local(base_url), model_name)
    a_caps = Caps.for_setting("benchmark", **({"episode_seconds": episode_seconds} if episode_seconds else {}))
    record: dict[str, Any] = {"base_url": base_url, "model_name": model_name, "model": spec.describe(),
                              "caps_tier_a": a_caps.to_dict(), "caps_tier_f": tier_f_caps().to_dict(),
                              "episodes": [], "tier_f": []}
    summary["model"] = {**(summary.get("model") or {}), **record}
    try:
        summary["model"]["props"] = server_props(base_url)
    except Exception as exc:                     # noqa: BLE001 - recorded; the episodes show the rest
        summary["model"]["props"] = {"error": f"{type(exc).__name__}: {exc}"}
    save()
    for d in f_dirs:
        for k in range(f_samples):
            res = run_tier_f_sample(d, spec, root / "model", config_id=SMOKE_CONFIG, sample=k,
                                    reason="smoke test on a fixture task")
            summary["model"]["tier_f"].append(tier_f_record(res))
            summary["lines"].append(_f_line("model", res))
            save()
    for d in ordered_a_dirs(a_dirs):
        res = run_episode(d, spec, "benchmark", root / "model", config_id=SMOKE_CONFIG, caps=a_caps,
                          reason="smoke test on a fixture task", grade_episode=True)
        summary["model"]["episodes"].append(tier_a_record(res))
        summary["lines"].append(_a_line("model", res, res.grade or {}))
        save()
    return summary["model"]


def collect_traces(run_dirs: list[str], path: Path) -> dict[str, Any]:
    """Concatenate the episodes' trace.jsonl rows into one file; count rows off the release format."""
    from .trace import row_format_errors, write_trace
    rows = []
    for run_dir in run_dirs:
        f = Path(run_dir) / "trace.jsonl"
        if f.is_file():
            rows += [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    write_trace(path, rows)
    bad = {r.get("session_id"): errs for r in rows if (errs := row_format_errors(r))}
    return {"path": str(path), "rows": len(rows), "rows_off_format": len(bad), "format_errors": dict(list(bad.items())[:3])}


def _tier_f_dirs(names: tuple[str, ...] | list[str]) -> list[Path]:
    dirs = [FIXTURE_F / n for n in names]
    for d in dirs:
        if not _is_fixture(d):
            raise NotAFixture(f"{d} is not a unit-test fixture task")
    return dirs


def run_smoke(*, base_url: str | None = None, model_name: str = SERVER_ALIAS, reps: int = 1, workers: int = 5,
              run_root: Path | None = None, start_server: bool = False, only: list[str] | None = None,
              golden: bool = True, probe: bool = True, tier_f_tasks: tuple[str, ...] | list[str] = DEFAULT_TIER_F_TASKS,
              tier_f_samples: int = DEFAULT_TIER_F_SAMPLES, threads: int | None = None,
              notes: list[str] | None = None, report: Path | None = None,
              episode_seconds: float | None = None) -> dict[str, Any]:
    """Run the smoke test (module docstring); returns the summary (also written to disk, and rendered
    to `report` when given). `only` keeps the fixture tasks whose directory name is listed;
    golden=False skips step 1, probe=False step 2; episode_seconds overrides the Tier-A episode
    wall-clock cap for the real-model episodes (recorded)."""
    a_dirs, f_dirs = fixture_tasks()
    model_f_dirs = _tier_f_dirs(tier_f_tasks)
    if only:
        a_dirs = [d for d in a_dirs if d.name in only]
        f_dirs = [d for d in f_dirs if d.name in only]
        model_f_dirs = [d for d in model_f_dirs if d.name in only]
    root = isolation.check_run_dir(Path(run_root) if run_root is not None else SMOKE_ROOT / _stamp())
    root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"started": _utc(), "root": str(root), "notes": list(notes or []),
                               "fixtures": {"tier_a": [d.name for d in a_dirs], "tier_f_golden": [d.name for d in f_dirs],
                                            "tier_f_model": [d.name for d in model_f_dirs],
                                            "tier_f_samples": tier_f_samples},
                               "machine": {"start": machine_state()},
                               "golden": [], "probe": None, "model": None, "lines": [], "path": str(root / "summary.json")}

    def save() -> None:
        write_json(root / "summary.json", summary)
    golden_dirs: list[str] = []
    if golden:
        golden_dirs = _golden_replays(a_dirs, f_dirs, root, reps, workers, summary)
        summary["golden_traces"] = collect_traces(golden_dirs, root / "golden_traces.jsonl")
    summary["golden_passed"] = all(r["passed"] for r in summary["golden"]) and (bool(summary["golden"]) or not golden)
    save()
    server = None
    if start_server and not base_url:
        native = native_context(SMOKE_MODEL)
        ctx = server_context(native)
        summary["server"] = {"model_file": str(SMOKE_MODEL), "model_sha256": sha256_file(SMOKE_MODEL),
                             "llama_cpp_build": LLAMA_CPP_BUILD, "native_context": native, "context": ctx,
                             "context_cap": CTX_CAP, "kv_cache": KV_CACHE_TYPE, "reasoning_format": REASONING_FORMAT,
                             "sse_ping_seconds": SSE_PING_SECONDS, "threads": threads or default_threads(),
                             "backend": llama_backend(), "llama_dir": str(llama_dir()),
                             "gpu": gpu_name() if llama_backend() == "cuda" else None,
                             "context_shift": False, "slots": 1, "sampling": " ".join(SAMPLING)}
        save()
        if probe:
            summary["machine"]["before_probe"] = machine_state()
            summary["probe"] = grammar_probe(root, ctx=ctx, threads=threads)
            save()
        server = LocalServer(root / "llama-server.log", ctx=ctx, threads=threads)
        summary["server"]["command"] = server.command
    try:
        if base_url or server is not None:
            summary["machine"]["before_model"] = machine_state()
        if server is not None:
            base_url = server.__enter__()
            summary["server"]["load_seconds"] = server.load_seconds
        if base_url:
            t0 = time.monotonic()
            _model_episodes(base_url, model_name, a_dirs, model_f_dirs, tier_f_samples, root, summary, save,
                            episode_seconds=episode_seconds)
            summary["model"]["seconds"] = round(time.monotonic() - t0, 1)
    except Exception as exc:                     # noqa: BLE001 - recorded; the report says the step broke
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if server is not None:
            server.__exit__()
            summary["server"]["log"] = str(server.log_path)
            summary["server"]["log_lines"] = server.log_lines()
            summary["server"]["timing"] = server.timing()
        save()
    if summary.get("model"):
        dirs = [e["run_dir"] for e in summary["model"]["episodes"] + summary["model"]["tier_f"]]
        summary["traces"] = collect_traces(dirs, root / "traces.jsonl")
    summary["machine"]["end"] = machine_state()
    summary["finished"] = _utc()
    save()
    if report is not None:
        write_text(Path(report), render_report(summary))
    return summary


# --------------------------------------------------------------------------------------------
# The report (reports/tier_a_smoke.md)
# --------------------------------------------------------------------------------------------

BANNER = ("> **Plumbing test of the Tier-A harness, not a model result.** It runs the unit-test fixture "
          "tasks (never a benchmark item) with Qwen3-4B Q4_K_M, the smoke-test model (build plan, decision 4), "
          "on a CPU shared with other jobs. Its pass/fail lines show that the harness runs, records and grades "
          "an episode end to end; they say nothing about GRC-Bench models.")


def _yn(value: Any) -> str:
    return "n/a" if value is None else ("yes" if value else "no")


def _cell(value: Any, width: int = 120) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= width else text[: width - 1] + "…"


def _mins(seconds: Any) -> str:
    return "n/a" if seconds is None else f"{float(seconds) / 60:.1f} min"


def _probe_section(probe: dict[str, Any] | None) -> list[str]:
    if not probe:
        return ["The grammar probe did not run (no `--start-server`, or `--no-probe`).", ""]
    v = probe.get("verdict") or {}
    lines = ["Probe request (every server, greedy, seed 0): one user message asking for tool arguments that "
             "violate the tool's JSON schema, " + f"`{json.dumps(PROBE_VIOLATION)}`" + " for "
             "`set_mode(mode: enum [\"alpha\", \"beta\"], level: integer)`, thinking off and on, `tool_choice` "
             "auto and required, one request that sends its own `grammar` together with the tools, and one "
             "with `tool_choice` none plus the llama.cpp field `parse_tool_calls: true`. The last column is "
             "what the server logged while handling that request.", "",
             "| Server | Case | HTTP | Parsed tool calls (arguments) | Calls left in the text | Schema-valid | Seconds | "
             "Log: grammar (trigger) |",
             "|---|---|--:|---|---|---|--:|---|"]
    for name, srv in (probe.get("servers") or {}).items():
        if srv.get("error") and not srv.get("cases"):
            lines.append(f"| {name} | (server failed) | | {_cell(srv['error'])} | | | |")
            continue
        for c in srv.get("cases") or []:
            parsed = "; ".join(f"{t['name']} {t['arguments']}" for t in c.get("tool_calls") or []) or "none"
            texts = "; ".join(f"{t.get('name')} {json.dumps(t.get('arguments'))}" for t in c.get("text_tool_calls") or [])
            allc = (c.get("tool_calls") or []) + (c.get("text_tool_calls") or [])
            valid = f"{sum(1 for t in allc if not t.get('schema_errors'))}/{len(allc)}" if allc else "n/a"
            if c.get("error"):
                parsed = "refused: " + c["error"]
            log = c.get("log")
            if log is None:
                logged = "n/a"
            else:
                trig = ("triggered" if log.get("triggered") else
                        "awaiting trigger" if log.get("awaiting_trigger") else "no trigger")
                logged = (f"{', '.join(log.get('grammar_kinds') or []) or 'none'} ({trig})"
                          + (f"; format {log['chat_format']}" if log.get("chat_format") else ""))
            lines.append(f"| {name} | {c['case']} | {c.get('http_status')} | {_cell(parsed, 160)} | "
                         f"{_cell(texts or 'none', 160)} | {valid} | {c.get('seconds')} | {logged} |")
    lines += ["", "Evidence in the server logs (`-lv 5`; first lines of each kind, truncated):", ""]
    for name, srv in (probe.get("servers") or {}).items():
        ev = srv.get("log_evidence") or {}
        lines.append(f"- **{name}** (`{' '.join(Path(a).name if a.startswith('/') else a for a in srv.get('command', [])[1:])}`):")
        for kind, label in (("chat_format_lines", "chat format"), ("grammar_lines", "grammar"),
                            ("trigger_lines", "trigger / lazy"), ("enum_rule_lines", "GBNF rule with the enum")):
            found = ev.get(kind) or []
            lines.append(f"  - {label}: " + (f"`{_cell(found[0], 300)}`" + (f" (+{len(found) - 1} more)" if len(found) > 1 else "")
                                             if found else "none"))
    lines += ["", "Reading (computed from the table and the logs):", "",
              f"- Default server: {v.get('default_calls_schema_valid')}/{v.get('default_parsed_tool_calls')} parsed tool "
              f"calls satisfy the schema although the instruction asks for violating values; "
              f"{v.get('default_unparsed_text_calls')} calls were left unparsed in the text.",
              f"- Control (`--skip-chat-parsing`): {v.get('control_calls_violating')}/{v.get('control_text_calls')} "
              f"calls written in the text violate the schema (the model's unconstrained choice).",
              f"- A request that brings its own `grammar` with tools is refused: {_yn(v.get('own_grammar_with_tools_refused'))}"
              + (f" (`{_cell(v.get('own_grammar_with_tools_error'), 160)}`)." if v.get("own_grammar_with_tools_error") else "."),
              f"- Grammar lines in the default server's log: {v.get('default_grammar_log_lines')}, GBNF rules holding "
              f"the enum: {v.get('default_enum_rule_lines')}; in the control's: {v.get('control_grammar_log_lines')} and "
              f"{v.get('control_enum_rule_lines')}.",
              f"- **--jinja grammar-constrains tool calls for this model in this build: "
              f"{_yn(v.get('grammar_constrains_tool_calls'))}.**", ""]
    return lines + _settings_section(probe.get("tool_call_settings") or tool_call_settings(v))


def _settings_section(s: dict[str, Any]) -> list[str]:
    """The B0 vs. B2 flag settings (eval plan Table 6), from tool_call_settings."""
    b2, b0, nj = s.get("B2") or {}, s.get("B0_candidate") or {}, s.get("no_jinja") or {}
    br = s.get("B0_request_candidate") or {}
    lines = ["### Tool-call settings for B0 and B2 (eval plan Table 6)", "",
             "Only the tool-call flags differ between the rows; every other server flag is the run's own "
             "(here the smoke command above). GRC-Agent reads tool calls only from the server's parsed "
             "`tool_calls`, never from the answer text.", "",
             "| Setting | Server flags | Chat format (log) | Tool grammar in the log | Calls held to the schema | "
             "Server returns parsed `tool_calls` | Usable by GRC-Agent as is |",
             "|---|---|---|---|---|---|---|",
             f"| B2: grammar-constrained tool calls | `{b2.get('server_flags')}` | {b2.get('chat_format') or 'n/a'} | "
             f"{_yn(b2.get('grammar_in_log'))} | {_yn(b2.get('tool_calls_schema_constrained'))} | "
             f"{_yn(b2.get('server_parses_tool_calls'))} | {_yn(b2.get('usable_by_grc_agent_as_is'))} |",
             f"| B0 candidate: calls as the model writes them | `{b0.get('server_flags')}` | "
             f"{b0.get('chat_format') or 'n/a'} | {_yn(b0.get('grammar_in_log'))} | "
             f"{_yn(b0.get('tool_calls_schema_constrained'))} | {_yn(b0.get('server_parses_tool_calls'))} | "
             f"{_yn(b0.get('usable_by_grc_agent_as_is'))} |",
             f"| No Jinja | `{nj.get('server_flags')}` | n/a | n/a | n/a | {_yn(nj.get('server_parses_tool_calls'))} "
             f"({nj.get('tool_requests_refused')}/{nj.get('cases')} tool requests refused"
             + (f": `{_cell(nj.get('error'), 120)}`" if nj.get("error") else "") + ") | "
             f"{_yn(nj.get('server_parses_tool_calls'))} |"]
    if br.get("ran"):
        held = ("n/a" if not br.get("server_parses_tool_calls") else
                "no" if br.get("parsed_calls_violating_schema") else "yes (or the model complied)")
        status = f"HTTP {br.get('http_status')}" + (f": `{_cell(br.get('error'), 100)}`" if br.get("error") else "")
        lines.append(f"| B0 request-level candidate: {br.get('request')} | `{br.get('server_flags')}` | "
                     f"{br.get('chat_format') or 'n/a'} ({status}) | {_yn(br.get('grammar_in_log'))} | {held} | "
                     f"{_yn(br.get('server_parses_tool_calls'))}"
                     + (f" ({_cell(json.dumps(br.get('arguments')), 120)})" if br.get("arguments") else "")
                     + " | no: GRC-Agent sends tool_choice \"auto\" |")
    lines.append("")
    if s.get("request_level_unconstrained_with_parsed_calls"):
        lines += ["**Reading.** With `--jinja` (the default), llama.cpp b10333 compiles the request's tool "
                  "schemas into a grammar and applies it once the model opens a tool call, so the server's "
                  "default IS the B2 setting (grammar-constrained tool calls)." if s.get("default_is_grammar_constrained")
                  else "**Reading.** See the table for the default server.",
                  "", "An unconstrained B0 is possible on the same server without a flag, at the request level: "
                  "`tool_choice: \"none\"` with the llama.cpp field `parse_tool_calls: true` gave parsed calls that "
                  "break the schema. GRC-Agent sends `tool_choice: \"auto\"`, so B0 needs the harness to rewrite "
                  "these two request fields (for example through the model's `extra_body`), checked like the "
                  "other harness parts before B0 runs; until then B0 and B2 run the same configuration.", ""]
        return lines
    if s.get("default_is_grammar_constrained") and not s.get("flag_only_unconstrained_setting_with_parsed_calls"):
        lines += ["**Reading.** With `--jinja` (the default), llama.cpp b10333 compiles the request's tool "
                  "schemas into a grammar and applies it once the model opens a tool call, so the server's "
                  "default IS the B2 setting (grammar-constrained tool calls). No flag of this build gives "
                  "parsed but unconstrained calls: the one setting without the tool grammar "
                  "(`--skip-chat-parsing`) also leaves the calls unparsed in the text, `--no-jinja` "
                  + ("refused the tool requests" if nj.get("tool_requests_refused") else
                     "returned no parsed calls" if not nj.get("server_parses_tool_calls") else
                     "returned parsed calls (see its row)")
                  + ", and a request may not bring its own grammar together with tools"
                  + (" (HTTP 400)" if s.get("client_grammar_with_tools_refused") else "")
                  + (", and the request-level candidate (`tool_choice: \"none\"` + `parse_tool_calls: true`) gave no "
                     "parsed calls that break the schema" if br.get("ran") else "") + ". An unconstrained "
                  "B0 therefore needs harness work (an open decision for the team): llama-server with "
                  "`--jinja --skip-chat-parsing` plus a parser in the harness that turns the model's own "
                  "tool-call text into `tool_calls` (keeping malformed calls as malformed calls), or B0 is "
                  "defined as the default server, in which case B0 and B2 are the same configuration.", ""]
    elif s.get("flag_only_unconstrained_setting_with_parsed_calls"):
        lines += ["**Reading.** B0 and B2 differ by flags alone: the B0 row returns parsed calls that "
                  "are not held to the schema.", ""]
    else:
        lines += ["**Reading.** The probe did not show the default server holding tool calls to the schema; "
                  "see the table above before pinning B0 and B2.", ""]
    return lines


def _timing_lines(timing: dict[str, Any]) -> list[str]:
    p, g = timing.get("prompt") or {}, timing.get("generation") or {}
    if not (p.get("requests") or g.get("requests")):
        return []
    return [f"| Engine speed in this run (server log) | {p.get('requests')} requests; prompt {p.get('tokens')} tokens "
            f"processed in {p.get('seconds')} s ({p.get('tokens_per_second')} tokens/s; cached prefixes are not "
            f"re-processed); generation {g.get('tokens')} tokens in {g.get('seconds')} s "
            f"({g.get('tokens_per_second')} tokens/s) |"]


def _machine_lines(machine: dict[str, Any]) -> list[str]:
    out = []
    for when in ("start", "before_probe", "before_model", "end"):
        m = machine.get(when)
        if m:
            others = ", ".join(m.get("other_llama_processes") or []) or "none"
            out.append(f"| CPU load ({when.replace('_', ' ')}, {m.get('time')}) | load average 1/5/15 min "
                       f"{m.get('load_1_5_15')} on {m.get('cpus')} CPUs; other llama.cpp processes: {others} |")
    return out


def render_report(summary: dict[str, Any]) -> str:
    """reports/tier_a_smoke.md from a smoke summary (deterministic for a given summary)."""
    srv = summary.get("server") or {}
    model = summary.get("model") or {}
    props = model.get("props") or {}
    out = ["# Tier-A smoke test: plumbing with a real local model", "", BANNER, "",
           f"Generated by `./gb tier-a smoke` (`grcbench.tier_a.smoke`; build plan Phase 4 \"Smoke test\"; eval plan "
           f"§4.4, §7.1, Table 6). Run started {summary.get('started')}, finished {summary.get('finished')} (UTC). "
           f"Everything it wrote: `{summary.get('root')}`.", ""]
    if summary.get("notes"):
        out += ["## Notes", ""] + [f"- {n}" for n in summary["notes"]] + [""]
    out += ["## Engine", "",
            "| Setting | Value |", "|---|---|",
            f"| llama.cpp | {srv.get('llama_cpp_build', 'n/a')} (`{props.get('build_info', 'n/a')}`) |",
            f"| Model | `{Path(srv.get('model_file', '')).name or model.get('model_name')}` "
            f"(SHA-256 `{(srv.get('model_sha256') or 'n/a')[:16]}…`) |",
            f"| Context per slot | {srv.get('context', props.get('n_ctx'))} tokens (native maximum "
            f"{srv.get('native_context', 'n/a')}, cap {srv.get('context_cap', CTX_CAP)}; server reports n_ctx "
            f"{props.get('n_ctx')}). Real runs need ≥ 65,536 per slot with context shift off"
            + (" (this smoke context is below that: a smoke-only setting, and a prompt over it is refused by the "
               "server and recorded as the context cap)"
               if isinstance(srv.get("context"), int) and srv["context"] < CTX_CAP else "") + ". |",
            f"| Slots / KV cache / context shift | {srv.get('slots', 'n/a')} / {srv.get('kv_cache', 'n/a')} K and V, "
            f"flash attention on / {'off' if srv.get('context_shift') is False else 'n/a'} |",
            f"| Chat template | Jinja (`--jinja`), template caps {_cell(json.dumps(props.get('chat_template_caps')), 200)} |",
            f"| Reasoning | `--reasoning-format {srv.get('reasoning_format', 'n/a')}` (thoughts in "
            f"`reasoning_content`, kept as the trace's `thinking`) |",
            f"| Requests | streamed (`stream: true`), SSE pings every {srv.get('sse_ping_seconds', 'n/a')} s; "
            f"GRC-Agent's own HTTP client (120 s read timeout, retries) |",
            f"| Sampling (server defaults; GRC-Agent sends none) | `{srv.get('sampling', 'n/a')}` (Qwen3 model card, "
            f"thinking mode) |",
            f"| Backend | {srv.get('backend', 'cpu')}"
            + (f" (`{srv['llama_dir']}`; GPU {srv['gpu']}, every layer offloaded)" if srv.get("gpu") else "") + " |",
            f"| Threads | {srv.get('threads', 'n/a')} |",
            *_timing_lines(srv.get("timing") or {}),
            f"| Command | `{' '.join(srv.get('command') or []) or 'n/a (external server: ' + str(model.get('base_url')) + ')'}` |",
            *_machine_lines(summary.get("machine") or {}),
            ""]
    if summary.get("error"):
        out += [f"**The model step stopped with an error:** `{_cell(summary['error'], 400)}` (the server was "
                "stopped; the tables below hold what finished).", ""]
    out += ["## Does `--jinja` grammar-constrain tool calls? (B0 vs. B2)", ""] + _probe_section(summary.get("probe"))
    out += ["## Step 1: golden replays (FunctionModel through the real runner)", "",
            f"All passed: **{_yn(summary.get('golden_passed'))}**.", "",
            "| Tier | Task | Sample | Passed | Status | Tool calls |", "|---|---|--:|---|---|--:|"]
    for g in summary.get("golden") or []:
        out.append(f"| {g['tier']} | {g['task']} | {g.get('sample')} | {_yn(g.get('passed'))} | {g.get('status')} | "
                   f"{g.get('tool_calls', '')} |")
    eps = model.get("episodes") or []
    ca = model.get("caps_tier_a") or {}
    caps_text = (f"{ca.get('requests')} requests, {ca.get('context_tokens')}-token context cap, "
                 f"{ca.get('tool_output_tokens')}-token tool outputs, {_mins(ca.get('episode_seconds'))} episode limit, "
                 f"{_mins(ca.get('flowgraph_run_seconds'))} per flowgraph run") if ca else "the benchmark caps"
    out += ["", "## Step 3, Tier A: the fixture tasks with the real model", "",
            f"One episode per fixture task through the real runner (GRC-Agent's executor, benchmark setting: "
            f"{caps_text}), graded by "
            "`grader.grade` (end state only). \"Malformed\" = model tool calls whose arguments, as emitted, do not "
            "validate against the tool's schema (before GRC-Agent's JSON repair); \"unparsed\" = assistant text "
            "holding a `<tool_call>` the server did not parse.", "",
            "| Task | Completed? | End reason | Requests | Tool calls | Malformed (invalid JSON) | Unknown tools | "
            "Unparsed | Caps hit | Grader | Wall time |",
            "|---|---|---|--:|--:|---|--:|--:|---|---|--:|"]
    for e in eps:
        verdict = "PASS" if e.get("passed") else ("fail" if e.get("passed") is False else "n/a")
        out.append(f"| {e['task']} | {_yn(e.get('completed'))} ({e.get('status')}) | {e.get('end_reason')} | "
                   f"{e.get('requests')} | {e.get('tool_calls')} | {e.get('malformed_calls')} ({e.get('invalid_json_calls')}) | "
                   f"{e.get('unknown_tool_calls')} | {e.get('unparsed_tool_call_texts')} | "
                   f"{', '.join(e.get('caps_hit') or []) or 'none'} | {verdict} | {_mins(e.get('wall_seconds'))} |")
    if eps:
        out += ["", "Per episode (tools in call order; the grader's first reasons when it failed):", ""]
        for e in eps:
            out.append(f"- **{e['task']}**: tools {', '.join(e.get('tools_used') or []) or 'none'}; median request "
                       f"{e.get('median_request_seconds')} s (max {e.get('max_request_seconds')} s); output tokens "
                       f"{e.get('output_tokens')}; largest prompt {e.get('max_request_input_tokens')} tokens.")
            for r in (e.get("reasons") or [])[:4] if not e.get("passed") else []:
                out.append(f"  - grader: {_cell(r, 240)}")
            for m in e.get("malformed_examples") or []:
                out.append(f"  - malformed `{m['tool']}`: {_cell('; '.join(m.get('errors') or []), 200)}")
            if e.get("error"):
                out.append(f"  - error: {_cell(e['error'], 300)}")
    fs = model.get("tier_f") or []
    out += ["", "## Step 3, Tier F: the \"one lookup + one edit\" setting with the real model", "",
            "Each sample: the history holds the instruction and one prefilled full `inspect_graph` call with its "
            "real result; at most one `query_knowledge`, then one atomic `change_graph`; 3 model requests; other "
            "calls get the fixed reply; graded V1 (well-formed edit) + ladder V2-V6.", "",
            "| Task | Sample | Lookup used? (query) | Edit called? | Well formed (V1) | Accepted? | Ladder verdict | "
            "Requests | End reason | Wall time |",
            "|---|--:|---|---|---|---|---|--:|---|--:|"]
    for f in fs:
        verdict = ("PASS (V1-V6)" if f.get("passed") else
                   f"fail: {f.get('failure') or 'n/a'} (highest {f.get('highest_passed') or 'none'})")
        query = f" ({_cell(f.get('lookup_query'), 60)}, {f.get('lookup_domain')}, k={f.get('lookup_k')})" if f.get("lookup_used") else ""
        out.append(f"| {f['task']} | {f['sample']} | {_yn(f.get('lookup_used'))}{query} | {_yn(f.get('edit_called'))} | "
                   f"{_yn(f.get('edit_well_formed'))} | {_yn(f.get('edit_accepted'))}"
                   f"{' (' + ', '.join(f.get('edit_error_codes') or []) + ')' if f.get('edit_error_codes') else ''} | "
                   f"{verdict} | {f.get('model_requests')} | {f.get('end_reason')} | {_mins(f.get('wall_seconds'))} |")
    if fs:
        out += ["", "What the model did in each sample:", ""]
        for f in fs:
            what = []
            if f.get("lookup_used"):
                what.append(f"looked up {f.get('lookup_retrieved')} (reference recall {f.get('lookup_reference_recall')})")
            if f.get("edit_called"):
                what.append(f"edit `{_cell(f.get('edit_raw_args'), 300)}`")
            if f.get("edit_same_reply_as_lookup"):
                what.append("lookup and edit in the same reply")
            if f.get("blocked_calls"):
                what.append(f"blocked calls {f['blocked_calls']}")
            if not f.get("edit_called"):
                what.append(f"no edit; final text `{_cell(f.get('final_text'), 200)}`")
            if f.get("error"):
                what.append(f"error {_cell(f['error'], 200)}")
            out.append(f"- sample {f['sample']}: " + "; ".join(what))
    tr = summary.get("traces") or {}
    out += ["", "## Traces", "",
            f"- Model episodes: `{tr.get('path', 'n/a')}`, {tr.get('rows', 0)} rows in Qoherent's SFT row format "
            f"({tr.get('rows_off_format', 0)} off format).",
            f"- Golden replays: `{(summary.get('golden_traces') or {}).get('path', 'n/a')}`, "
            f"{(summary.get('golden_traces') or {}).get('rows', 0)} rows "
            f"({(summary.get('golden_traces') or {}).get('rows_off_format', 0)} off format).",
            f"- Summary: `{summary.get('path')}`; server log `{srv.get('log', 'n/a')}`.", ""]
    return "\n".join(out).rstrip() + "\n"

"""The model relay: a hosted, OpenAI-compatible model API for an episode, with the API key kept
outside the sandbox.

An episode process has no network ("network": "none") and must never hold an API key: code the
agent writes (an Embedded Python block, a flowgraph, a shell command) runs next to it, and anything
it can read can end up in a trace. So the host runs this relay for the duration of the episode:

    episode process  --HTTP over a UNIX socket-->  relay (host)  --HTTPS-->  provider
    (GRC-Agent's own openai_compatible            adds "Authorization: Bearer <key>"
     model, models.openai_compatible(uds_path))

  * the key is read once from the host environment variable the caller names (`api_key_env`) and is
    never written anywhere: not to the run folder, episode.json, the results or the logs;
  * the socket lives in its own directory under ~/grc-bench-work/.relay/, which the episode
    sandbox binds read-only (connecting to a socket needs no write access) and the sandboxes for
    code the agent starts do not see (sandbox.agent_hidden_roots hides the run directories);
  * only requests under the local base path (LOCAL_BASE, "/v1") are forwarded, onto the
    upstream's base path (e.g. https://openrouter.ai/api/v1/chat/completions); anything else gets
    a local 404, and a path with "." or ".." segments a 400;
  * one request per connection: the request is sent upstream with "Connection: close", the
    response head gets "Connection: close", and the response bytes are passed on as they arrive,
    so streamed (server-sent events) responses stream and the provider's status codes (429, 5xx)
    reach GRC-Agent's own retry logic unchanged;
  * standard library only (asyncio, ssl); the relay runs on its own thread, next to the blocking
    episode-process wait of runner.run_episode.

`requests` records method, path and status of every forwarded request (never headers or bodies).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import ssl
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Any, Self

from ..common.paths import WORK

LOCAL_BASE = "/v1"
LOCAL_BASE_URL = "http://grc-bench-relay" + LOCAL_BASE     # what the episode's model is pointed at
SOCKET_NAME = "model.sock"
RELAY_ROOT = WORK / ".relay"
MAX_HEAD = 64 << 10
IDLE_TIMEOUT = 600.0               # seconds without a byte from the provider before giving up
_DROP = {"host", "authorization", "connection", "keep-alive", "proxy-authorization", "proxy-connection",
         "te", "trailer", "upgrade", "content-length", "transfer-encoding"}


class RelayError(RuntimeError):
    """The relay cannot start (no key, bad upstream URL)."""


@dataclass
class Upstream:
    scheme: str
    host: str
    port: int
    base_path: str

    @classmethod
    def parse(cls, url: str, *, allow_http_loopback: bool = False) -> Upstream:
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ("https", "http") or not u.hostname:
            raise RelayError(f"upstream must be an https:// URL, got {url!r}")
        if u.scheme == "http" and not (allow_http_loopback and u.hostname in ("127.0.0.1", "localhost", "::1")):
            raise RelayError("upstream must use https (plain http only for a loopback test server)")
        if u.query or u.fragment or u.username or u.password:
            raise RelayError("upstream URL must not carry a query, fragment or credentials")
        return cls(u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80), u.path.rstrip("/"))

    @property
    def host_header(self) -> str:
        default = 443 if self.scheme == "https" else 80
        return self.host if self.port == default else f"{self.host}:{self.port}"

    def describe(self) -> str:
        return f"{self.scheme}://{self.host_header}{self.base_path}"


def _local_path(target: str) -> str | None:
    """The upstream-relative path of a request target, or None when it is not under LOCAL_BASE."""
    if not target.startswith("/"):
        return None
    path, _, query = target.partition("?")
    if path != LOCAL_BASE and not path.startswith(LOCAL_BASE + "/"):
        return None
    segments = urllib.parse.unquote(path).split("/")
    if any(s in (".", "..") for s in segments):
        raise ValueError("dot segments")
    rest = path[len(LOCAL_BASE):]
    return rest + (f"?{query}" if query else "")


def _response(status: int, reason: str, body: str) -> bytes:
    data = body.encode("utf-8")
    return (f"HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\n"
            f"content-length: {len(data)}\r\nconnection: close\r\n\r\n").encode("ascii") + data


@dataclass
class Relay:
    """One relay: `with Relay(upstream_url, api_key_env=NAME) as relay:` serves relay.socket_path."""

    upstream_url: str
    api_key_env: str
    allow_http_loopback: bool = False          # tests only: a fake upstream on 127.0.0.1
    requests: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.upstream = Upstream.parse(self.upstream_url, allow_http_loopback=self.allow_http_loopback)
        if not self.api_key_env or not os.environ.get(self.api_key_env):
            raise RelayError(f"environment variable {self.api_key_env!r} is not set (it must hold the API key)")
        self.dir = RELAY_ROOT / uuid.uuid4().hex[:12]
        self.socket_path = self.dir / SOCKET_NAME
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._ssl = ssl.create_default_context() if self.upstream.scheme == "https" else None

    def describe(self) -> dict[str, Any]:
        """What results record: the upstream and the variable's NAME (never its value)."""
        return {"upstream": self.upstream.describe(), "api_key_env": self.api_key_env}

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> Self:
        self.dir.mkdir(parents=True, exist_ok=False)
        os.chmod(self.dir, 0o700)
        self._thread = threading.Thread(target=self._serve, name="grcbench-relay", daemon=True)
        self._thread.start()
        self._ready.wait(30)
        if self._error is not None or not self.socket_path.exists():
            self.__exit__(None, None, None)
            raise RelayError(f"relay did not start: {self._error!r}")
        return self

    def __exit__(self, *exc: object) -> None:
        loop, thread = self._loop, self._thread
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._stop.set)
        if thread is not None:
            thread.join(30)
        shutil.rmtree(self.dir, ignore_errors=True)

    def _serve(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:          # noqa: BLE001 - reported to __enter__
            self._error = exc
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        server = await asyncio.start_unix_server(self._handle, path=str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        self._ready.set()
        async with server:
            await self._stop.wait()

    # ------------------------------------------------------------------ one request
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        record: dict[str, Any] = {"t": round(time.time(), 3)}
        try:
            await self._forward(reader, writer, record)
        except Exception as exc:              # noqa: BLE001 - the client gets a 502, the record the type
            record.setdefault("status", 502)
            record["error"] = type(exc).__name__
            try:
                writer.write(_response(502, "Bad Gateway", '{"error": {"message": "model relay: upstream failed"}}'))
                await writer.drain()
            except Exception:                 # noqa: BLE001, S110 - the client is gone
                pass
        finally:
            self.requests.append(record)
            writer.close()

    async def _forward(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                       record: dict[str, Any]) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        if len(head) > MAX_HEAD:
            raise ValueError("request head too large")
        lines = head.decode("latin-1").split("\r\n")
        method, target, _version = lines[0].split(" ", 2)
        headers = [tuple(x.strip() for x in line.split(":", 1)) for line in lines[1:] if ":" in line]
        record.update(method=method, path=target.partition("?")[0])
        try:
            rest = _local_path(target)
        except ValueError:
            record["status"] = 400
            writer.write(_response(400, "Bad Request", '{"error": {"message": "model relay: bad path"}}'))
            return
        if rest is None:
            record["status"] = 404
            writer.write(_response(404, "Not Found", '{"error": {"message": "model relay: not found"}}'))
            return
        names = {k.lower(): v for k, v in headers}
        if "chunked" in names.get("transfer-encoding", "").lower():
            record["status"] = 411
            writer.write(_response(411, "Length Required", '{"error": {"message": "model relay: chunked body"}}'))
            return
        body = await reader.readexactly(int(names.get("content-length", "0") or 0))
        out = [f"{method} {self.upstream.base_path}{rest} HTTP/1.1", f"Host: {self.upstream.host_header}",
               f"Authorization: Bearer {os.environ.get(self.api_key_env, '')}", "Connection: close",
               f"Content-Length: {len(body)}"]
        out += [f"{k}: {v}" for k, v in headers if k.lower() not in _DROP]
        up_reader, up_writer = await asyncio.wait_for(asyncio.open_connection(
            self.upstream.host, self.upstream.port, ssl=self._ssl,
            server_hostname=self.upstream.host if self._ssl else None), 30)
        try:
            up_writer.write(("\r\n".join(out) + "\r\n\r\n").encode("latin-1") + body)
            await up_writer.drain()
            resp_head = await asyncio.wait_for(up_reader.readuntil(b"\r\n\r\n"), IDLE_TIMEOUT)
            resp_lines = resp_head.decode("latin-1").split("\r\n")
            parts = resp_lines[0].split(" ", 2)
            record["status"] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            kept = [line for line in resp_lines[1:] if line and line.split(":", 1)[0].strip().lower()
                    not in ("connection", "keep-alive")]
            writer.write(("\r\n".join([resp_lines[0], *kept, "Connection: close"]) + "\r\n\r\n").encode("latin-1"))
            while True:
                chunk = await asyncio.wait_for(up_reader.read(65536), IDLE_TIMEOUT)
                if not chunk:
                    break
                writer.write(chunk)
                await writer.drain()
        finally:
            up_writer.close()

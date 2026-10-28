"""What a model server reports about itself: llama-server's GET /props (its context per slot,
default_generation_settings.n_ctx, and its slots, total_slots) and GET /v1/models (the model's own
context, meta.n_ctx_train), over TCP or a UNIX socket. Anything a server does not report is None: an API
without these endpoints, a reply that is not JSON or not of the expected shape, a listener that is not HTTP."""

from __future__ import annotations

import http.client
import json
import socket
import urllib.parse
from typing import Any


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._path)
        self.sock = sock


def _get_json(base_url: str | None, path: str, uds_path: str | None, timeout: float) -> Any:
    """The parsed JSON reply to GET <root><path>, or None (any failure)."""
    try:
        if uds_path:
            conn: http.client.HTTPConnection = _UnixHTTPConnection(uds_path, timeout)
        else:
            parts = urllib.parse.urlsplit(base_url or "")
            if parts.scheme not in ("http", "https") or not parts.hostname:
                return None
            cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
            conn = cls(parts.hostname, parts.port, timeout=timeout)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            return json.loads(resp.read().decode("utf-8")) if resp.status == 200 else None
        finally:
            conn.close()
    except (http.client.HTTPException, OSError, ValueError):
        return None


def _positive(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None


def server_info(base_url: str | None, *, uds_path: str | None = None, timeout: float = 5.0) -> dict[str, int | None]:
    """{"n_ctx", "total_slots", "n_ctx_train"} as the server reports them (module docstring); base_url is the
    OpenAI-compatible base URL (".../v1"), whose root holds /props."""
    props = _get_json(base_url, "/props", uds_path, timeout)
    props = props if isinstance(props, dict) else {}
    gen = props.get("default_generation_settings")
    listed = _get_json(base_url, "/v1/models", uds_path, timeout)
    data = listed.get("data") if isinstance(listed, dict) else None
    first = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else {}
    meta = first.get("meta") if isinstance(first.get("meta"), dict) else {}
    return {"n_ctx": _positive(gen.get("n_ctx")) if isinstance(gen, dict) else None,
            "total_slots": _positive(props.get("total_slots")),
            "n_ctx_train": _positive(meta.get("n_ctx_train"))}

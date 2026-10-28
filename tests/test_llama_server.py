"""Tests for grcbench.common.llama_server: what a model server reports about itself (stub servers only)."""

from __future__ import annotations

import json
import socket
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from grcbench.common import llama_server

PROPS = {"default_generation_settings": {"n_ctx": 40960}, "total_slots": 2}
MODELS = {"data": [{"id": "m", "meta": {"n_ctx_train": 40960}}]}
NONE = {"n_ctx": None, "total_slots": None, "n_ctx_train": None}


def _handler(replies: dict[str, object]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def address_string(self) -> str:                  # a UNIX socket has no client address
            return "stub"

        def do_GET(self) -> None:                         # noqa: N802 - http.server API
            if self.path not in replies:
                self.send_error(404)
                return
            data = json.dumps(replies[self.path]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def _serve(server) -> None:
    threading.Thread(target=server.serve_forever, daemon=True).start()


def test_a_server_reports_its_context_its_slots_and_the_models_context():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler({"/props": PROPS, "/v1/models": MODELS}))
    _serve(srv)
    try:
        info = llama_server.server_info(f"http://127.0.0.1:{srv.server_address[1]}/v1")
    finally:
        srv.shutdown()
    assert info == {"n_ctx": 40960, "total_slots": 2, "n_ctx_train": 40960}


def test_a_server_on_a_unix_socket_is_read_too(tmp_path):
    class UnixHTTP(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    path = str(tmp_path / "llama.sock")
    srv = UnixHTTP(path, _handler({"/props": PROPS, "/v1/models": MODELS}))
    _serve(srv)
    try:
        info = llama_server.server_info("http://localhost/v1", uds_path=path)
    finally:
        srv.shutdown()
    assert info == {"n_ctx": 40960, "total_slots": 2, "n_ctx_train": 40960}


def test_what_a_server_does_not_report_is_none():
    api = ThreadingHTTPServer(("127.0.0.1", 0), _handler({}))                     # no /props: an API
    bad = ThreadingHTTPServer(("127.0.0.1", 0), _handler({"/props": {"default_generation_settings": [1, 2],
                                                                     "total_slots": True},
                                                          "/v1/models": {"data": "x"}}))
    raw = socket.socket()                                                        # a listener that is not HTTP
    raw.bind(("127.0.0.1", 0))
    raw.listen()

    def garbage() -> None:
        while True:
            try:
                conn, _ = raw.accept()
            except OSError:
                return
            conn.sendall(b"not http at all\r\n\r\n")
            conn.close()

    threading.Thread(target=garbage, daemon=True).start()
    for s in (api, bad):
        _serve(s)
    try:
        assert llama_server.server_info(f"http://127.0.0.1:{api.server_address[1]}/v1") == NONE
        assert llama_server.server_info(f"http://127.0.0.1:{bad.server_address[1]}/v1") == NONE
        assert llama_server.server_info(f"http://127.0.0.1:{raw.getsockname()[1]}/v1", timeout=2) == NONE
    finally:
        api.shutdown()
        bad.shutdown()
        raw.close()

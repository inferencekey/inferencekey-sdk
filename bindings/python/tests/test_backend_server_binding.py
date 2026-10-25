"""Smoke tests for the native HTTP/SSE binding (``_inferencekey.BackendServer``).

These drive the pyo3 classes directly with a tiny application object, below
``serve.py``. The parity and golden tests over the real runtime live in
``test_backend_native.py``. Skipped when the native module lacks the server
(``IK_REQUIRE_NATIVE=1`` turns the skip into a failure).
"""

from __future__ import annotations

import errno
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

try:
    from inferencekey import _inferencekey as native
except (ImportError, OSError):  # pragma: no cover - depends on the build
    native = None

HAS_SERVER = getattr(native, "BACKEND_SERVER_ABI", 0) == 1

if not HAS_SERVER:
    if os.environ.get("IK_REQUIRE_NATIVE") == "1":
        raise RuntimeError("IK_REQUIRE_NATIVE=1 but the native backend server is missing")
    pytestmark = pytest.mark.skip(reason="native module has no BACKEND_SERVER_ABI")


class App:
    """Echoes JSON routes; streams the request body's ``n`` events."""

    def __init__(self) -> None:
        self.json_calls = []
        self.pushed = []

    def handle_json(self, route: str, body: bytes):
        self.json_calls.append((route, body, threading.current_thread().name))
        return 200, json.dumps({"route": route, "len": len(body)}).encode()

    def handle_stream(self, body: bytes, sink) -> None:
        request = json.loads(body or b"{}")
        if request.get("reject"):
            sink.reply(400, b'{"error": "rejected"}')
            return
        if request.get("raise"):
            raise RuntimeError("boom")
        sink.start()
        for i in range(request.get("n", 3)):
            payload = json.dumps({"n": i}).encode()
            self.pushed.append(sink.push(payload))
        assert sink.keepalive()
        assert sink.push_many([b'{"a": 1}', b'{"b": 2}'])
        sink.done()


@pytest.fixture
def server():
    app = App()
    srv = native.BackendServer(0, app, shutdown_grace=1.0)
    yield srv, app
    srv.close()


def _post(port: int, path: str, body: bytes, timeout: float = 5.0):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def _get(port: int, path: str):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_module_exposes_the_abi_and_classes():
    assert native.BACKEND_SERVER_ABI == 1
    assert native.BackendServer.__name__ == "BackendServer"
    assert native.StreamSink.__name__ == "StreamSink"


def test_start_serve_and_stop_from_python():
    app = App()
    srv = native.BackendServer(0, app, shutdown_grace=1.0)
    port = srv.local_port
    assert srv.server_address == ("127.0.0.1", port)
    assert srv.backlog == 4096 and srv.max_inflight == 1024 and srv.io_threads >= 1

    assert _get(port, "/healthz") == (503, b'{"status": "loading"}')
    srv.set_ready(True)
    assert _get(port, "/healthz") == (200, b'{"status": "ok"}')
    srv.set_manifest(json.dumps({"name": "smoke", "api": "openai-chat"}).encode())
    assert _get(port, "/meta") == (200, b'{"name": "smoke", "api": "openai-chat"}')
    status, body = _get(port, "/v1/models")
    assert status == 200 and json.loads(body)["data"][0]["id"] == "smoke"

    status, _headers, body = _post(port, "/process", b'{"id": "1"}')
    assert (status, json.loads(body)) == (200, {"route": "/process", "len": 11})
    assert app.json_calls[0][2] != threading.current_thread().name

    stats = json.loads(srv.stats())
    assert stats["requests"] >= 5

    started = time.monotonic()
    srv.shutdown()
    assert srv.wait(timeout=0) is True
    srv.close()
    assert time.monotonic() - started < 3
    # The port is free again at once.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def test_stream_through_the_sink(server):
    srv, app = server
    srv.set_ready(True)
    status, headers, body = _post(srv.local_port, "/v1/chat/completions", b'{"n": 3}')
    assert status == 200
    assert headers["Content-Type"] == "text/event-stream"
    assert body == (
        b'data: {"n": 0}\n\ndata: {"n": 1}\n\ndata: {"n": 2}\n\n'
        b": keepalive\n\n"
        b'data: {"a": 1}\n\ndata: {"b": 2}\n\n'
        b"data: [DONE]\n\n"
    )
    assert app.pushed == [True, True, True]


def test_reply_before_start_is_plain_json(server):
    srv, _app = server
    status, headers, body = _post(srv.local_port, "/v1/chat/completions", b'{"reject": true}')
    assert (status, body) == (400, b'{"error": "rejected"}')
    assert headers["Content-Type"] == "application/json"


def test_exception_escaping_the_app_is_a_500(server, capfd):
    srv, _app = server
    status, _headers, body = _post(srv.local_port, "/v1/chat/completions", b'{"raise": true}')
    assert status == 500
    assert json.loads(body) == {"error": {"message": "internal server error", "type": "server_error"}}
    assert "RuntimeError: boom" in capfd.readouterr().err


def test_bad_json_reply_shape_is_a_500(capfd):
    class Broken:
        def handle_json(self, route, body):
            return "not a tuple"

        def handle_stream(self, body, sink):
            sink.reply(200, b"{}")

    srv = native.BackendServer(0, Broken(), shutdown_grace=1.0)
    try:
        status, _headers, body = _post(srv.local_port, "/forecast", b"{}")
        assert (status, body) == (500, b'{"error": "internal server error"}')
    finally:
        srv.close()
    assert "handle_json raised" in capfd.readouterr().err


def test_bind_conflict_raises_eaddrinuse():
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]
        with pytest.raises(OSError) as excinfo:
            native.BackendServer(port, App())
        assert excinfo.value.errno == errno.EADDRINUSE


def test_wait_times_out_while_running(server):
    srv, _app = server
    started = time.monotonic()
    assert srv.wait(timeout=0.3) is False
    assert 0.25 <= time.monotonic() - started < 2


def test_dropping_the_server_closes_it():
    srv = native.BackendServer(0, App(), shutdown_grace=1.0)
    port = srv.local_port
    del srv
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))

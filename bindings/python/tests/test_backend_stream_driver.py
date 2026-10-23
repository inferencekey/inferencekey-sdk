"""The transport-independent request functions of ``inferencekey.backend.serve``.

``handle_process``, ``handle_forecast``, ``prepare_chat`` and
``drive_chat_stream`` hold what a request means; every HTTP transport calls
them. These tests drive them directly with a recording ``out``, and check the
stdlib transport's passthrough and disconnect handling over a real socket.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, List, Tuple

import pytest

from inferencekey.backend import (
    BackendContext,
    CustomBackend,
    InvalidInput,
    Job,
    Result,
    chat_completion,
)
from inferencekey.backend import serve as serve_mod

MESSAGES = [{"role": "user", "content": "hi"}]
CHUNK = {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}


class ScriptedBackend(CustomBackend):
    """Streams whatever ``request["script"]`` names; records generator close."""

    name = "scripted"

    def __init__(self) -> None:
        self.closed = threading.Event()

    def setup(self, ctx: BackendContext) -> None:
        pass

    def process(self, job: Job) -> Result:
        if job.input.get("bad"):
            raise InvalidInput("bad job")
        return Result(output={"ok": True})

    def chat(self, request):
        return chat_completion("hello", model=self.name)

    def chat_stream(self, request):
        script = request.get("script", "plain")
        if script == "refuse":
            exc = InvalidInput("too long")
            exc.code = "context_length_exceeded"
            raise exc
        if script == "unsupported":
            raise NotImplementedError
        if script == "explode":
            raise RuntimeError("no GPU")
        if script == "empty":
            return iter(())
        return self._stream(script)

    def _stream(self, script):
        try:
            if script == "mixed":
                yield {"choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": None}]}
                yield '{"choices":[{"index":0,"delta":{"content":"b"},"finish_reason":null}]}'
                yield b'{"choices":[{"index":0,"delta":{"content":"\\u00e9"},"finish_reason":null}]}'
                yield None
                yield bytearray(b'{"choices":[],"usage":{"completion_tokens":3}}')
            elif script == "fail_midway":
                yield CHUNK
                raise RuntimeError("lost the GPU")
            elif script == "endless":
                while True:
                    yield CHUNK
                    time.sleep(0.005)
            elif script == "slow_start":
                time.sleep(0.3)
                while True:
                    yield CHUNK
                    time.sleep(0.005)
            else:
                yield CHUNK
        finally:
            self.closed.set()


class RecordingOut:
    """A ``StreamOut`` that records calls; the client "leaves" after ``accept`` writes."""

    def __init__(self, accept: int = 10_000) -> None:
        self.calls: List[Tuple[str, Any]] = []
        self.accept = accept

    def reply(self, status: int, body: bytes) -> None:
        self.calls.append(("reply", (status, json.loads(body))))

    def start(self) -> None:
        self.calls.append(("start", None))

    def _write(self, kind: str, payload: Any) -> bool:
        if self.accept <= 0:
            return False
        self.accept -= 1
        self.calls.append((kind, payload))
        return True

    def push(self, payload: bytes) -> bool:
        return self._write("push", payload)

    def keepalive(self) -> bool:
        return self._write("keepalive", None)

    def done(self) -> None:
        self.calls.append(("done", None))


def _state(ready: bool = True) -> Tuple[ScriptedBackend, Any]:
    backend = ScriptedBackend()
    state = serve_mod._BackendState(backend)
    if ready:
        state.mark_ready()
    return backend, state


def _body(obj: Any) -> bytes:
    return json.dumps(obj).encode()


def test_content_length_rules() -> None:
    assert serve_mod.content_length(None) == (0, None)
    assert serve_mod.content_length("12") == (12, None)
    assert serve_mod.content_length("abc") == (0, (400, b'{"error": "invalid Content-Length"}'))
    too_large = (0, (400, b'{"error": "request body too large"}'))
    assert serve_mod.content_length("-1") == too_large
    assert serve_mod.content_length(str(16 * 1024 * 1024 + 1)) == too_large
    assert serve_mod.content_length(str(16 * 1024 * 1024)) == (16 * 1024 * 1024, None)


def test_process_and_forecast_map_errors_to_status_codes() -> None:
    _, state = _state()
    assert serve_mod.handle_process(state, _body({"id": "j", "input": {}})) == (200, b'{"output": {"ok": true}}')
    assert serve_mod.handle_process(state, _body({"id": "j", "input": {"bad": True}})) == (400, b'{"error": "bad job"}')
    status, body = serve_mod.handle_process(state, b"{not json")
    assert status == 400 and json.loads(body)["error"].startswith("invalid JSON body: ")
    assert serve_mod.handle_process(state, b"")[0] == 400  # {} is not a job
    forecast = {"id": "f", "target": [1.0, 2.0], "horizon": 1}
    assert serve_mod.handle_forecast(state, _body(forecast)) == (
        501, b'{"error": "backend does not support forecast"}'
    )
    _, loading = _state(ready=False)
    assert serve_mod.handle_process(loading, b"") == (503, b'{"error": "backend not ready"}')
    assert serve_mod.handle_forecast(loading, b"") == (503, b'{"error": "backend not ready"}')


def test_prepare_chat_answers_json_or_hands_over_the_stream() -> None:
    _, state = _state()
    kind, status, body = serve_mod.prepare_chat(state, _body({"messages": MESSAGES}))
    assert (kind, status) == ("json", 200)
    assert json.loads(body)["choices"][0]["message"]["content"] == "hello"
    assert serve_mod.prepare_chat(state, _body({"messages": MESSAGES, "stream": True})) == (
        "stream", {"messages": MESSAGES, "stream": True}
    )
    kind, status, body = serve_mod.prepare_chat(state, _body({"prompt": "x"}))
    assert (kind, status, json.loads(body)["error"]["type"]) == ("json", 400, "invalid_request_error")
    kind, status, body = serve_mod.prepare_chat(state, b"\xff")
    assert (kind, status) == ("json", 400) and json.loads(body)["error"].startswith("invalid JSON body: ")
    _, loading = _state(ready=False)
    assert serve_mod.prepare_chat(loading, b"") == (
        "json", 503, b'{"error": {"message": "backend not ready", "type": "server_error"}}'
    )


@pytest.mark.parametrize(
    "script, status, error",
    [
        ("refuse", 400, {"message": "too long", "type": "invalid_request_error", "code": "context_length_exceeded"}),
        ("unsupported", 501, {"message": "backend does not support chat", "type": "invalid_request_error"}),
        ("explode", 500, {"message": "no GPU", "type": "server_error"}),
    ],
)
def test_stream_errors_before_the_first_chunk_are_plain_replies(script, status, error) -> None:
    _, state = _state()
    out = RecordingOut()
    serve_mod.drive_chat_stream(state, {"messages": MESSAGES, "script": script}, out)
    assert out.calls == [("reply", (status, {"error": error}))]


def test_empty_stream_is_just_done() -> None:
    _, state = _state()
    out = RecordingOut()
    serve_mod.drive_chat_stream(state, {"messages": MESSAGES, "script": "empty"}, out)
    assert out.calls == [("start", None), ("done", None)]


def test_str_and_bytes_chunks_pass_through_and_none_is_a_keepalive() -> None:
    backend, state = _state()
    out = RecordingOut()
    serve_mod.drive_chat_stream(state, {"messages": MESSAGES, "script": "mixed"}, out)
    assert out.calls == [
        ("start", None),
        ("push", b'{"choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": null}]}'),
        ("push", b'{"choices":[{"index":0,"delta":{"content":"b"},"finish_reason":null}]}'),
        ("push", b'{"choices":[{"index":0,"delta":{"content":"\\u00e9"},"finish_reason":null}]}'),
        ("keepalive", None),
        ("push", b'{"choices":[],"usage":{"completion_tokens":3}}'),
        ("done", None),
    ]
    assert backend.closed.is_set()


def test_mid_stream_failure_sends_an_error_event_then_done() -> None:
    backend, state = _state()
    out = RecordingOut()
    serve_mod.drive_chat_stream(state, {"messages": MESSAGES, "script": "fail_midway"}, out)
    assert out.calls[0] == ("start", None)
    assert out.calls[-2] == ("push", b'{"error": {"message": "lost the GPU", "type": "server_error"}}')
    assert out.calls[-1] == ("done", None)
    assert backend.closed.is_set()


def test_a_client_that_leaves_closes_the_generator_and_gets_no_done() -> None:
    backend, state = _state()
    out = RecordingOut(accept=3)
    serve_mod.drive_chat_stream(state, {"messages": MESSAGES, "script": "endless"}, out)
    assert backend.closed.is_set()
    assert [kind for kind, _ in out.calls] == ["start", "push", "push", "push"]


def test_encode_chunk() -> None:
    assert serve_mod.encode_chunk({"a": 1}) == b'{"a": 1}'
    assert serve_mod.encode_chunk('{"a":1}') == b'{"a":1}'
    assert serve_mod.encode_chunk("\u00e9") == "\u00e9".encode("utf-8")
    assert serve_mod.encode_chunk(b"raw") == b"raw"
    assert serve_mod.encode_chunk(memoryview(b"view")) == b"view"


# --------------------------------------------------------------------------
# Over the stdlib transport
# --------------------------------------------------------------------------


@pytest.fixture()
def served():
    started = []

    def start() -> Tuple[ScriptedBackend, int]:
        backend = ScriptedBackend()
        httpd, _state = serve_mod.serve(backend, 0, {})
        started.append(httpd)
        return backend, httpd.server_address[1]

    yield start
    for httpd in started:
        httpd.shutdown()
        httpd.server_close()


def _raw_stream(port: int, request: dict) -> socket.socket:
    body = json.dumps(request).encode()
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
    )
    return s


def _read_all(s: socket.socket) -> bytes:
    data = b""
    while True:
        part = s.recv(65536)
        if not part:
            return data
        data += part


def test_stdlib_writes_str_and_bytes_chunks_verbatim(served) -> None:
    _, port = served()
    s = _raw_stream(port, {"messages": MESSAGES, "stream": True, "script": "mixed"})
    try:
        raw = _read_all(s)
    finally:
        s.close()
    head, _, body = raw.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.0 200")
    assert b"Content-Type: text/event-stream" in head
    assert body == (
        b'data: {"choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":"b"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":"\\u00e9"},"finish_reason":null}]}\n\n'
        b": keepalive\n\n"
        b'data: {"choices":[],"usage":{"completion_tokens":3}}\n\n'
        b"data: [DONE]\n\n"
    )


def test_stdlib_closes_the_generator_when_the_client_leaves_before_the_first_chunk(served) -> None:
    backend, port = served()
    s = _raw_stream(port, {"messages": MESSAGES, "stream": True, "script": "slow_start"})
    s.close()
    assert backend.closed.wait(5)


def test_stdlib_closes_the_generator_when_the_client_leaves_mid_stream(served) -> None:
    backend, port = served()
    s = _raw_stream(port, {"messages": MESSAGES, "stream": True, "script": "endless"})
    assert s.recv(64).startswith(b"HTTP/1.0 200")
    s.close()
    assert backend.closed.wait(5)


def test_stdlib_listens_with_a_backlog_of_1024() -> None:
    assert serve_mod._HTTPServer.request_queue_size == 1024

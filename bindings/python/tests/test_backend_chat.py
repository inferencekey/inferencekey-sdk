"""OpenAI chat on custom backends: ``chat()``, ``/v1/chat/completions`` and SSE.

Torch-free fake backends, real server over loopback HTTP, like
``test_backend_serve.py``.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from inferencekey.backend import (
    API_OPENAI_CHAT,
    API_PROCESS,
    BackendContext,
    CustomBackend,
    InvalidInput,
    Job,
    Result,
    chat_completion,
    completion_to_chunks,
    package_backend,
)
from inferencekey.backend import serve as serve_mod
from inferencekey.errors import PackagingError


class ToolChatBackend(CustomBackend):
    """Answers text, or a tool call when the request offers tools."""

    name = "tool-chat"

    def setup(self, ctx: BackendContext) -> None:
        self.config = ctx.config

    def process(self, job: Job) -> Result:  # pragma: no cover
        return Result(output={})

    def chat(self, request):
        if request.get("boom"):
            raise RuntimeError("model exploded")
        if request.get("bad"):
            raise InvalidInput("bad request")
        if request.get("tools"):
            return chat_completion(
                model="tool-chat",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                    }
                ],
                prompt_tokens=10,
                completion_tokens=3,
            )
        last = request["messages"][-1]["content"]
        return chat_completion(f"echo: {last}", model="tool-chat", prompt_tokens=5, completion_tokens=2)


class TokenStreamBackend(ToolChatBackend):
    """Overrides chat_stream to yield token by token, then fails if asked."""

    def chat_stream(self, request):
        for piece in ("hel", "lo"):
            yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
        if request.get("fail_midway"):
            raise RuntimeError("lost the GPU")
        yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


class EndlessStreamBackend(ToolChatBackend):
    """Streams until the client goes away; records that its generator was closed."""

    closed = threading.Event()
    # Strong references to every generator handed out, so ``closed`` is set
    # only by an explicit close from the runtime, not by garbage collection.
    generators: list = []

    def chat_stream(self, request):
        gen = self._stream()
        EndlessStreamBackend.generators.append(gen)
        return gen

    def _stream(self):
        try:
            while True:
                yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}
                time.sleep(0.01)
        finally:
            EndlessStreamBackend.closed.set()


class ProcessOnlyBackend(CustomBackend):
    def setup(self, ctx: BackendContext) -> None:
        pass

    def process(self, job: Job) -> Result:
        return Result(output={})


class ForcedProcessBackend(ToolChatBackend):
    """Implements chat() but pins api to process explicitly."""

    api = API_PROCESS


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def server():
    started = []

    def start(backend_cls):
        port = _free_port()
        httpd, _state = serve_mod.serve(backend_cls(), port, {"device": "cpu"})
        started.append(httpd)
        return port

    yield start
    for httpd in started:
        httpd.shutdown()
        httpd.server_close()


def _post_raw(port: int, body: dict):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read().decode()


def _sse_events(body: str) -> list:
    return [line[len("data: "):] for line in body.splitlines() if line.startswith("data: ")]


def test_api_is_inferred_from_chat_override() -> None:
    assert ToolChatBackend().resolved_api() == API_OPENAI_CHAT
    assert ToolChatBackend().manifest().to_wire()["api"] == API_OPENAI_CHAT
    assert ProcessOnlyBackend().resolved_api() == API_PROCESS
    assert ForcedProcessBackend().resolved_api() == API_PROCESS


def test_chat_returns_completion_with_usage(server) -> None:
    port = server(ToolChatBackend)
    status, ctype, body = _post_raw(port, {"messages": [{"role": "user", "content": "hi"}]})
    assert status == 200 and ctype.startswith("application/json")
    data = json.loads(body)
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "echo: hi"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


def test_chat_returns_tool_calls(server) -> None:
    port = server(ToolChatBackend)
    status, _, body = _post_raw(
        port,
        {"messages": [{"role": "user", "content": "read it"}], "tools": [{"type": "function", "function": {"name": "read_file"}}]},
    )
    choice = json.loads(body)["choices"][0]
    assert status == 200
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "read_file"


def test_stream_replays_chat_as_sse_chunks(server) -> None:
    port = server(ToolChatBackend)
    status, ctype, body = _post_raw(
        port,
        {
            "stream": True,
            "messages": [{"role": "user", "content": "read it"}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        },
    )
    assert status == 200 and ctype.startswith("text/event-stream")
    events = _sse_events(body)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    calls = [c["choices"][0]["delta"]["tool_calls"] for c in chunks if c["choices"] and "tool_calls" in c["choices"][0]["delta"]]
    assert calls[0][0]["index"] == 0 and calls[0][0]["function"]["name"] == "read_file"
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["total_tokens"] == 13


def test_stream_yields_tokens_from_chat_stream(server) -> None:
    port = server(TokenStreamBackend)
    status, _, body = _post_raw(port, {"stream": True, "messages": [{"role": "user", "content": "x"}]})
    events = _sse_events(body)
    text = "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1])
    assert status == 200 and text == "hello" and events[-1] == "[DONE]"


def test_stream_failure_midway_sends_error_event_then_done(server) -> None:
    port = server(TokenStreamBackend)
    status, _, body = _post_raw(port, {"stream": True, "fail_midway": True, "messages": [{"role": "user", "content": "x"}]})
    events = _sse_events(body)
    assert status == 200
    assert json.loads(events[-2])["error"]["message"] == "lost the GPU"
    assert events[-1] == "[DONE]"


@pytest.mark.parametrize("stream", [False, True])
def test_process_only_backend_answers_501(server, stream) -> None:
    port = server(ProcessOnlyBackend)
    status, _, body = _post_raw(port, {"stream": stream, "messages": [{"role": "user", "content": "x"}]})
    assert status == 501
    assert json.loads(body)["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("stream", [False, True])
def test_invalid_input_is_400_and_model_failure_is_500(server, stream) -> None:
    port = server(ToolChatBackend)
    msgs = [{"role": "user", "content": "x"}]
    assert _post_raw(port, {"stream": stream, "bad": True, "messages": msgs})[0] == 400
    assert _post_raw(port, {"stream": stream, "boom": True, "messages": msgs})[0] == 500
    # The server survived both.
    assert _post_raw(port, {"messages": msgs})[0] == 200


def test_missing_messages_is_400(server) -> None:
    port = server(ToolChatBackend)
    assert _post_raw(port, {"prompt": "x"})[0] == 400


#: A 200 status line on either transport (stdlib speaks HTTP/1.0, native HTTP/1.1).
OK_STATUS_LINES = (b"HTTP/1.0 200", b"HTTP/1.1 200")


def test_server_listens_with_a_large_backlog_and_daemon_threads(server) -> None:
    port = server(ToolChatBackend)
    assert serve_mod.LISTEN_BACKLOG >= 1024
    assert serve_mod._HTTPServer.request_queue_size == serve_mod.LISTEN_BACKLOG
    assert serve_mod._HTTPServer.daemon_threads
    # A burst of simultaneous connections is accepted, none reset.
    socks = [socket.create_connection(("127.0.0.1", port), timeout=5) for _ in range(64)]
    try:
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        for s in socks:
            s.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
            )
        for s in socks:
            # stdlib answers with an HTTP/1.0 status line, native with HTTP/1.1.
            assert s.recv(64).startswith(OK_STATUS_LINES)
    finally:
        for s in socks:
            s.close()


def test_client_disconnect_closes_the_backend_stream(server) -> None:
    EndlessStreamBackend.closed.clear()
    EndlessStreamBackend.generators.clear()
    port = server(EndlessStreamBackend)
    body = json.dumps({"stream": True, "messages": [{"role": "user", "content": "x"}]}).encode()
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
    )
    assert s.recv(64).startswith(OK_STATUS_LINES)
    s.close()
    assert EndlessStreamBackend.closed.wait(5)


def test_models_lists_the_backend(server) -> None:
    port = server(ToolChatBackend)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as resp:
        data = json.loads(resp.read().decode())
    assert data["data"][0]["id"] == "tool-chat"


def test_completion_to_chunks_without_tools() -> None:
    chunks = list(completion_to_chunks(chat_completion("hi")))
    assert [c["choices"][0]["delta"] for c in chunks] == [{"role": "assistant", "content": ""}, {"content": "hi"}, {}]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    with_usage = list(completion_to_chunks(chat_completion("hi", prompt_tokens=1, completion_tokens=1)))
    assert with_usage[-1] == {**{k: with_usage[-1][k] for k in ("id", "object", "created", "model")}, "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def test_packaging_writes_api_only_for_openai_chat(tmp_path: Path) -> None:
    src = tmp_path / "backend.py"
    src.write_text("x = 1\n", encoding="utf-8")
    chat_pkg = package_backend(src=src, entrypoint="backend:B", name="c", version="1", out_dir=tmp_path / "a", api="openai-chat")
    plain_pkg = package_backend(src=src, entrypoint="backend:B", name="p", version="1", out_dir=tmp_path / "b")
    assert chat_pkg.manifest["api"] == "openai-chat"
    assert "api" not in plain_pkg.manifest
    with pytest.raises(PackagingError):
        package_backend(src=src, entrypoint="backend:B", name="x", version="1", out_dir=tmp_path / "c", api="grpc")

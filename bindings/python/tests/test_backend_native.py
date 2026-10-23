"""The native HTTP/SSE transport against the stdlib one (design section 9.2).

* Golden: one scripted backend served by both transports; status codes and
  body bytes (JSON bodies, every 4xx and 5xx, de-chunked SSE bodies) must be
  equal. Known transport differences are listed and pinned, not ignored.
* Generator close on disconnect, health while ``setup()`` holds the GIL, the
  fallback rules of ``IK_BACKEND_HTTP``, HTTP/1.0 framing, a burst of 500
  streams, pass-through chunks, port reuse, a GIL-hog watchdog, and a CLI
  run whose listener belongs to the serving process.

Native runs are skipped only when the native module has no backend server
(``IK_REQUIRE_NATIVE=1`` makes that a failure).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import pytest

import inferencekey
from inferencekey.backend import (
    CustomBackend,
    ForecastResult,
    InvalidInput,
    Result,
    chat_completion,
)
from inferencekey.backend import serve as serve_mod

from conftest import TRANSPORTS, native_available, require_native

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _select(name: str, monkeypatch: pytest.MonkeyPatch, **env: str) -> Any:
    if name == "native":
        require_native()
    monkeypatch.setenv("IK_BACKEND_HTTP", name)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    transport = serve_mod._select_transport()
    assert transport.kind == name
    return transport


def _start(backend: CustomBackend, name: str, monkeypatch: pytest.MonkeyPatch, **env: str):
    transport = _select(name, monkeypatch, **env)
    httpd, state = serve_mod.serve(backend, 0, {}, transport=transport)
    return httpd, state


def _stop(httpd: Any) -> None:
    httpd.shutdown()
    httpd.server_close()


class Raw:
    """One HTTP response as it came off the socket."""

    def __init__(self, data: bytes) -> None:
        head, _, raw_body = data.partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        self.status_line = lines[0]
        parts = self.status_line.split(" ", 2)
        self.version = parts[0]
        self.status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        self.headers: Dict[str, str] = {}
        for line in lines[1:]:
            key, _, value = line.partition(":")
            self.headers[key.strip().lower()] = value.strip()
        self.raw_body = raw_body
        self.chunked = self.headers.get("transfer-encoding", "").lower() == "chunked"
        self.body = _dechunk(raw_body) if self.chunked else raw_body


def _dechunk(data: bytes) -> bytes:
    out = bytearray()
    while data:
        size_line, _, rest = data.partition(b"\r\n")
        size = int(size_line.split(b";")[0], 16)
        if size == 0:
            break
        out += rest[:size]
        data = rest[size + 2:]
    return bytes(out)


def raw_request(
    port: int,
    method: str,
    path: str,
    body: Optional[bytes] = None,
    *,
    headers: Optional[List[Tuple[str, str]]] = None,
    version: str = "HTTP/1.1",
    content_length: Optional[str] = "auto",
    timeout: float = 10.0,
) -> Raw:
    """Send one request with ``Connection: close`` and read the reply until EOF."""
    lines = [f"{method} {path} {version}", "Host: 127.0.0.1", "Connection: close"]
    for key, value in headers or []:
        lines.append(f"{key}: {value}")
    if content_length == "auto":
        if body is not None:
            lines.append(f"Content-Length: {len(body)}")
    elif content_length is not None:
        lines.append(f"Content-Length: {content_length}")
    request = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8") + (body or b"")
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(request)
        chunks = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
    return Raw(b"".join(chunks))


def _chat(script: str, stream: bool = True, **extra: Any) -> bytes:
    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": stream, "script": script}
    payload.update(extra)
    return json.dumps(payload).encode()


def _delta(text: str) -> Dict[str, Any]:
    return {"id": "c1", "object": "chat.completion.chunk", "choices": [
        {"index": 0, "delta": {"content": text}, "finish_reason": None}]}


# --------------------------------------------------------------------------
# The scripted backend
# --------------------------------------------------------------------------


class GoldenBackend(CustomBackend):
    """Answers each route according to the request body, for byte comparison."""

    name = "golden-é"
    version = "1.0"
    api = "openai-chat"

    def setup(self, ctx) -> None:
        gate = getattr(type(self), "gate", None)
        if gate is not None:
            gate.wait(10)

    def process(self, job):
        mode = job.input.get("mode")
        if mode == "invalid":
            raise InvalidInput("no audio in job")
        if mode == "boom":
            raise RuntimeError("process exploded ☃")
        if mode == "not-result":
            return {"output": 1}
        return Result(output={"echo": job.input, "id": job.id, "text": "café \U0001f600"})

    def forecast(self, request):
        if request.target and request.target[0] == 0:
            raise NotImplementedError
        return ForecastResult(median=[float(v) for v in request.target[: request.horizon]])

    def chat(self, request):
        script = request.get("script")
        if script == "invalid":
            err = InvalidInput("prompt too long")
            err.code = "context_length_exceeded"
            raise err
        if script == "boom":
            raise RuntimeError("chat exploded")
        if script == "not-dict":
            return ["nope"]
        if script == "not-implemented":
            raise NotImplementedError
        # Fixed id and timestamp, so both transports send the same bytes.
        completion = chat_completion("hello é", model=self.name, completion_id="chatcmpl-golden")
        completion["created"] = 1700000000
        return completion

    def chat_stream(self, request):
        script = request.get("script")
        if script == "not-implemented":
            raise NotImplementedError
        if script == "invalid":
            raise InvalidInput("bad stream request")
        if script == "boom-early":
            raise RuntimeError("failed before the first chunk")
        return self._stream(script)

    def _stream(self, script):
        if script == "empty":
            return
        if script == "multi-line":
            yield _delta("ok")
            yield '{"choices":\n[]}'
            return
        if script == "boom-late":
            yield _delta("a")
            raise RuntimeError("failed mid-stream é")
        if script == "big":
            for i in range(2000):
                yield _delta(f"token {i} " + "x" * (i % 50))
                if i % 100 == 0:
                    yield None
            return
        # The default script: dict chunks, keepalives, str and bytes passthrough.
        yield {"id": "c1", "object": "chat.completion.chunk", "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
        yield None
        yield _delta("hello é \U0001f600 \"quoted\"\\n")
        yield '{"choices":[{"index":0,"delta":{"content":"str chunk"},"finish_reason":null}]}'
        yield None
        yield b'{"choices":[{"index":0,"delta":{"content":"bytes chunk"},"finish_reason":null}]}'
        yield {"id": "c1", "object": "chat.completion.chunk", "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield {"id": "c1", "object": "chat.completion.chunk", "choices": [],
               "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4}}


class ProcessOnlyBackend(CustomBackend):
    name = "process-only"

    def setup(self, ctx) -> None:
        pass

    def process(self, job):
        return Result(output={})


#: Requests answered the same way by both transports: (label, method, path, body, extra kwargs).
GOLDEN_CASES: List[Tuple[str, str, str, Optional[bytes], Dict[str, Any]]] = [
    ("healthz", "GET", "/healthz", None, {}),
    ("healthz with query", "GET", "/healthz?x=1", None, {}),
    ("meta", "GET", "/meta", None, {}),
    ("models", "GET", "/v1/models", None, {}),
    ("get unknown", "GET", "/nope", None, {}),
    ("leading double slash", "GET", "//healthz", None, {}),
    ("post leading double slash", "POST", "//process", json.dumps({"id": "j", "input": {}}).encode(), {}),
    ("inner double slash", "GET", "/v1//models", None, {}),
    ("post unknown", "POST", "/nope", b"{}", {}),
    ("process ok", "POST", "/process", json.dumps({"id": "j1", "input": {"t": "é"}}).encode(), {}),
    ("process query", "POST", "/process?trace=1", json.dumps({"id": "j1", "input": {}}).encode(), {}),
    ("process no id", "POST", "/process", b'{"input": {}}', {}),
    ("process not an object", "POST", "/process", b"[1, 2]", {}),
    ("process empty body", "POST", "/process", None, {"content_length": None}),
    ("process invalid json", "POST", "/process", b"{nope", {}),
    ("process invalid utf-8", "POST", "/process", b"\xff\xfe", {}),
    ("process invalid input", "POST", "/process", json.dumps({"id": "j", "input": {"mode": "invalid"}}).encode(), {}),
    ("process raises", "POST", "/process", json.dumps({"id": "j", "input": {"mode": "boom"}}).encode(), {}),
    ("process not a Result", "POST", "/process", json.dumps({"id": "j", "input": {"mode": "not-result"}}).encode(), {}),
    ("process too large", "POST", "/process", b"", {"content_length": str(16 * 1024 * 1024 + 1)}),
    ("forecast ok", "POST", "/forecast", json.dumps({"id": "f", "target": [1, 2, 3], "horizon": 2}).encode(), {}),
    ("forecast 501", "POST", "/forecast", json.dumps({"id": "f", "target": [0, 1], "horizon": 1}).encode(), {}),
    ("forecast bad horizon", "POST", "/forecast", json.dumps({"id": "f", "target": [1], "horizon": 0}).encode(), {}),
    ("forecast too large", "POST", "/forecast", b"", {"content_length": "99999999999999"}),
    ("chat ok", "POST", "/v1/chat/completions", _chat("ok", stream=False), {}),
    ("chat invalid input with code", "POST", "/v1/chat/completions", _chat("invalid", stream=False), {}),
    ("chat raises", "POST", "/v1/chat/completions", _chat("boom", stream=False), {}),
    ("chat not a dict", "POST", "/v1/chat/completions", _chat("not-dict", stream=False), {}),
    ("chat 501", "POST", "/v1/chat/completions", _chat("not-implemented", stream=False), {}),
    ("chat no messages", "POST", "/v1/chat/completions", b'{"prompt": "x"}', {}),
    ("chat invalid json", "POST", "/v1/chat/completions", b"{", {}),
    ("chat too large", "POST", "/v1/chat/completions", b"", {"content_length": str(16 * 1024 * 1024 + 1)}),
    ("stream default script", "POST", "/v1/chat/completions", _chat("default"), {}),
    ("stream empty", "POST", "/v1/chat/completions", _chat("empty"), {}),
    ("stream multi-line passthrough", "POST", "/v1/chat/completions", _chat("multi-line"), {}),
    ("stream fails mid-way", "POST", "/v1/chat/completions", _chat("boom-late"), {}),
    ("stream 400 before first chunk", "POST", "/v1/chat/completions", _chat("invalid"), {}),
    ("stream 500 before first chunk", "POST", "/v1/chat/completions", _chat("boom-early"), {}),
    ("stream 501", "POST", "/v1/chat/completions", _chat("not-implemented"), {}),
    ("stream 2000 chunks", "POST", "/v1/chat/completions", _chat("big"), {}),
    ("put 501", "PUT", "/process", b"{}", {}),
    ("delete 501", "DELETE", "/healthz", None, {}),
    ("patch 501", "PATCH", "/v1/chat/completions", b"{}", {}),
]

#: Content-Length values hyper answers itself, with a bodyless 400, before the
#: native transport's body rules run; stdlib answers them with a JSON 400.
#: Same status, different body: a known transport difference (DESIGN 3.2 step 4).
HYPER_CONTENT_LENGTH_400 = ["abc", "+5", "1_0", "0x10", "-1", "-", "5.0", "٥"]

#: Paths the transports route differently: ``{path: (native status, stdlib status)}``.
#: ``/healthz#x``: the http crate drops the fragment; stdlib compares the whole target.
PATH_DIFFERENCES = {"/healthz#x": (200, 404)}


def _serve_golden(name: str, monkeypatch: pytest.MonkeyPatch, backend: Optional[CustomBackend] = None):
    httpd, _state = _start(backend or GoldenBackend(), name, monkeypatch)
    return httpd


@pytest.fixture
def both(monkeypatch: pytest.MonkeyPatch):
    """The golden backend on both transports: ``{"stdlib": port, "native": port}``."""
    require_native()
    servers = {name: _serve_golden(name, monkeypatch) for name in TRANSPORTS}
    yield {name: httpd.server_address[1] for name, httpd in servers.items()}
    for httpd in servers.values():
        _stop(httpd)


# --------------------------------------------------------------------------
# 1. Golden byte equality
# --------------------------------------------------------------------------


@pytest.mark.parametrize("label,method,path,body,kwargs", GOLDEN_CASES, ids=[c[0] for c in GOLDEN_CASES])
def test_golden_bytes_match(both, capfd, label, method, path, body, kwargs) -> None:
    stdlib = raw_request(both["stdlib"], method, path, body, **kwargs)
    native = raw_request(both["native"], method, path, body, **kwargs)
    capfd.readouterr()  # tracebacks of the failing scripts
    assert stdlib.status == native.status, label
    assert stdlib.body == native.body, label
    assert stdlib.headers.get("content-type") == native.headers.get("content-type"), label
    if stdlib.headers.get("content-type") == "text/event-stream":
        assert native.headers.get("cache-control") == "no-cache"
        assert native.headers.get("connection") == "close"
        assert native.chunked, "an HTTP/1.1 client gets chunked framing"
        assert native.body.endswith(b"data: [DONE]\n\n")


def test_golden_stream_contents_are_what_the_backend_yielded(both) -> None:
    native = raw_request(both["native"], "POST", "/v1/chat/completions", _chat("default"))
    events = native.body.split(b"\n\n")
    assert events[1] == b": keepalive"
    assert events[3] == b'data: {"choices":[{"index":0,"delta":{"content":"str chunk"},"finish_reason":null}]}'
    assert events[5] == b'data: {"choices":[{"index":0,"delta":{"content":"bytes chunk"},"finish_reason":null}]}'
    assert events[-2] == b"data: [DONE]"
    multi = raw_request(both["native"], "POST", "/v1/chat/completions", _chat("multi-line"))
    assert multi.body.endswith(
        b'data: {"error": {"message": "pre-serialized chunk must be one line of JSON", '
        b'"type": "server_error"}}\n\ndata: [DONE]\n\n'
    )


def test_golden_unready_post_routes_match(monkeypatch: pytest.MonkeyPatch) -> None:
    require_native()

    class Gated(GoldenBackend):
        gate = threading.Event()

    servers: Dict[str, Any] = {}
    threads = []
    ports: Dict[str, int] = {}
    for name in TRANSPORTS:
        transport = _select(name, monkeypatch)
        port = _free_port()
        ports[name] = port

        def run(name=name, transport=transport, port=port):
            servers[name] = serve_mod.serve(Gated(), port, {}, transport=transport)[0]

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        threads.append(thread)
    try:
        for name, port in ports.items():
            _wait_for_status(port, 503)
        for path, body in (
            ("/process", json.dumps({"id": "j", "input": {}}).encode()),
            ("/forecast", json.dumps({"id": "f", "target": [1], "horizon": 1}).encode()),
            ("/v1/chat/completions", _chat("ok", stream=False)),
            ("/v1/chat/completions", _chat("default")),
        ):
            stdlib = raw_request(ports["stdlib"], "POST", path, body)
            native = raw_request(ports["native"], "POST", path, body)
            assert (stdlib.status, stdlib.body) == (native.status, native.body)
            assert native.status == 503
        stdlib = raw_request(ports["stdlib"], "GET", "/healthz")
        native = raw_request(ports["native"], "GET", "/healthz")
        assert (stdlib.status, stdlib.body) == (native.status, native.body) == (503, b'{"status": "loading"}')
    finally:
        Gated.gate.set()
        for thread in threads:
            thread.join(10)
        for httpd in servers.values():
            _stop(httpd)


@pytest.mark.parametrize("value", HYPER_CONTENT_LENGTH_400)
def test_known_difference_malformed_content_length(both, value) -> None:
    for path in ("/process", "/forecast", "/v1/chat/completions"):
        stdlib = raw_request(both["stdlib"], "POST", path, b"", content_length=value)
        native = raw_request(both["native"], "POST", path, b"", content_length=value)
        assert stdlib.status == native.status == 400, (path, value)
        expected = b'{"error": "request body too large"}' if value == "-1" else b'{"error": "invalid Content-Length"}'
        assert stdlib.body == expected
        assert native.body == b"", "hyper's own 400 has no body"


@pytest.mark.parametrize("pad", [" {}", "  {}"])
def test_content_length_leading_whitespace_is_trimmed_by_both(both, pad) -> None:
    # The stdlib header parser drops leading whitespace before content_length()
    # runs, and hyper trims it as optional whitespace.
    body = json.dumps({"id": "j", "input": {}}).encode()
    stdlib = raw_request(both["stdlib"], "POST", "/process", body, content_length=pad.format(len(body)))
    native = raw_request(both["native"], "POST", "/process", body, content_length=pad.format(len(body)))
    assert (stdlib.status, stdlib.body) == (native.status, native.body)
    assert stdlib.status == 200


@pytest.mark.parametrize("pad", ["{} ", "  {}  "])
def test_known_difference_content_length_trailing_whitespace(both, pad) -> None:
    # hyper trims trailing optional whitespace too (RFC 9110) and reads the
    # body; the stdlib header parser keeps it, so content_length() rejects it.
    body = json.dumps({"id": "j", "input": {}}).encode()
    stdlib = raw_request(both["stdlib"], "POST", "/process", body, content_length=pad.format(len(body)))
    native = raw_request(both["native"], "POST", "/process", body, content_length=pad.format(len(body)))
    assert (stdlib.status, stdlib.body) == (400, b'{"error": "invalid Content-Length"}')
    assert native.status == 200


@pytest.mark.parametrize("path", sorted(PATH_DIFFERENCES))
def test_known_difference_paths(both, path) -> None:
    native_status, stdlib_status = PATH_DIFFERENCES[path]
    assert raw_request(both["stdlib"], "GET", path).status == stdlib_status
    assert raw_request(both["native"], "GET", path).status == native_status


def test_known_difference_json_routes_keep_alive_on_native(both) -> None:
    """stdlib answers HTTP/1.0 and closes; native answers HTTP/1.1 and keeps JSON connections open."""
    with socket.create_connection(("127.0.0.1", both["native"]), timeout=5) as sock:
        for _ in range(3):
            sock.sendall(b"GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n")
            reply = b""
            while not reply.endswith(b'{"status": "ok"}'):
                reply += sock.recv(4096)
            assert reply.startswith(b"HTTP/1.1 200")
    assert raw_request(both["stdlib"], "GET", "/healthz").version == "HTTP/1.0"


# --------------------------------------------------------------------------
# 2. GeneratorExit on disconnect
# --------------------------------------------------------------------------


class SlowStreamBackend(CustomBackend):
    """Streams forever (chunks or keepalives only); keeps every generator alive."""

    name = "slow-stream"
    api = "openai-chat"
    generators: List[Any] = []
    closed = threading.Event()
    started = threading.Event()

    def setup(self, ctx) -> None:
        pass

    def process(self, job):
        return Result(output={})

    def chat(self, request):
        return chat_completion("x", model=self.name)

    def chat_stream(self, request):
        gen = self._stream(request.get("script"))
        # A strong reference: only an explicit close() runs the finally.
        type(self).generators.append(gen)
        return gen

    def _stream(self, script):
        try:
            yield _delta("first")
            type(self).started.set()
            while True:
                time.sleep(0.02)
                yield None if script == "idle" else _delta("more")
        finally:
            type(self).closed.set()


def _open_stream(port: int, script: str) -> socket.socket:
    body = _chat(script)
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.sendall(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
    )
    assert sock.recv(64).split(b" ", 2)[1] == b"200"
    return sock


@pytest.mark.parametrize("name", TRANSPORTS)
@pytest.mark.parametrize("script", ["chunks", "idle"])
def test_generator_closed_within_a_second_of_disconnect(monkeypatch, name, script) -> None:
    SlowStreamBackend.generators.clear()
    SlowStreamBackend.closed.clear()
    SlowStreamBackend.started.clear()
    httpd, _ = _start(SlowStreamBackend(), name, monkeypatch)
    try:
        sock = _open_stream(httpd.server_address[1], script)
        assert SlowStreamBackend.started.wait(5)
        time.sleep(0.1)
        sock.close()
        started = time.monotonic()
        assert SlowStreamBackend.closed.wait(1.0), f"{name}/{script}: generator not closed within 1 s"
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert SlowStreamBackend.generators, "the backend kept its generator"
    finally:
        _stop(httpd)


# --------------------------------------------------------------------------
# 3. Health while setup() holds the GIL (separate process)
# --------------------------------------------------------------------------

GIL_HOG_SETUP = textwrap.dedent(
    '''
    import time
    from inferencekey.backend import CustomBackend, Result

    class HogBackend(CustomBackend):
        name = "hog"

        def setup(self, ctx):
            end = time.monotonic() + float(ctx.config.get("seconds", 5))
            x = 0
            while time.monotonic() < end:
                # Pure Python, plus C calls that keep the GIL for a while.
                for _ in range(1000):
                    x += 1
                sum(range(3_000_000))

        def process(self, job):
            return Result(output={})
    '''
)


def _spawn_cli(tmp_path, module_source: str, entry: str, transport: str, config: Optional[dict] = None,
               extra_env: Optional[Dict[str, str]] = None):
    (tmp_path / "cli_backend.py").write_text(module_source)
    env = dict(os.environ)
    env["IK_BACKEND_HTTP"] = transport
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path)] + [p for p in sys.path if p])
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra_env or {})
    port = _free_port()
    argv = [sys.executable, "-X", "ik_workload_id=test", "-m", "inferencekey.backend.serve",
            "--port", str(port), "--backend", f"cli_backend:{entry}"]
    if config is not None:
        argv += ["--config-json", json.dumps(config)]
    proc = subprocess.Popen(argv, cwd=str(tmp_path), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    return proc, port


def _probe(port: int, timeout: float = 3.0) -> Tuple[Optional[int], float]:
    started = time.monotonic()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=timeout) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except OSError:
        status = None
    return status, time.monotonic() - started


@pytest.mark.parametrize("name", TRANSPORTS)
def test_healthz_answers_while_setup_holds_the_gil(tmp_path, name, capsys) -> None:
    if name == "native":
        require_native()
    proc, port = _spawn_cli(tmp_path, GIL_HOG_SETUP, "HogBackend", name, config={"seconds": 5})
    latencies = []
    try:
        deadline = time.monotonic() + 20
        status = None
        while status != 503 and time.monotonic() < deadline:
            status, _ = _probe(port, 0.5)
            time.sleep(0.02)
        assert status == 503, "the server never answered while loading"
        while time.monotonic() < deadline:
            status, latency = _probe(port)
            if status == 200:
                break
            assert status == 503
            latencies.append(latency)
            time.sleep(0.05)
        assert status == 200
    finally:
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=20)
    assert len(latencies) >= 20, latencies
    worst = max(latencies)
    with capsys.disabled():
        print(f"\n[{name}] healthz during a GIL-holding setup: {len(latencies)} probes, "
              f"p50 {sorted(latencies)[len(latencies) // 2] * 1000:.1f} ms, max {worst * 1000:.1f} ms")
    assert f"http transport: {name}" in err
    if name == "native":
        assert worst < 0.1, f"native /healthz took {worst * 1000:.0f} ms while setup() held the GIL"


# --------------------------------------------------------------------------
# 4. Fallback and selection
# --------------------------------------------------------------------------


def _block_native(monkeypatch: pytest.MonkeyPatch, replacement: Any = None) -> None:
    """Make ``from inferencekey import _inferencekey`` fail (or return ``replacement``)."""
    monkeypatch.delattr(inferencekey, "_inferencekey", raising=False)
    if replacement is None:
        monkeypatch.setitem(sys.modules, "inferencekey._inferencekey", None)
    else:
        monkeypatch.setattr(inferencekey, "_inferencekey", replacement, raising=False)
        monkeypatch.setitem(sys.modules, "inferencekey._inferencekey", replacement)


def _serve_and_capture(monkeypatch, capfd) -> Tuple[str, Any]:
    httpd, _ = serve_mod.serve(ProcessOnlyBackend(), 0, {})
    try:
        status, body = _get(httpd.server_address[1], "/healthz")
        assert (status, body) == (200, b'{"status": "ok"}')
    finally:
        _stop(httpd)
    return capfd.readouterr().err, httpd


def test_auto_falls_back_to_stdlib_when_the_import_fails(monkeypatch, capfd) -> None:
    _block_native(monkeypatch)
    monkeypatch.setenv("IK_BACKEND_HTTP", "auto")
    err, httpd = _serve_and_capture(monkeypatch, capfd)
    lines = [line for line in err.splitlines() if line.startswith("http transport:")]
    assert len(lines) == 1
    assert lines[0].startswith("http transport: stdlib (reason: _inferencekey could not be imported")
    assert isinstance(httpd, serve_mod.ThreadingHTTPServer)


def test_auto_falls_back_to_stdlib_without_the_abi(monkeypatch, capfd) -> None:
    _block_native(monkeypatch, types.ModuleType("inferencekey._inferencekey"))
    monkeypatch.setenv("IK_BACKEND_HTTP", "auto")
    err, _ = _serve_and_capture(monkeypatch, capfd)
    lines = [line for line in err.splitlines() if line.startswith("http transport:")]
    assert lines == ["http transport: stdlib (reason: _inferencekey has no BACKEND_SERVER_ABI)"]


def test_auto_falls_back_on_an_abi_mismatch(monkeypatch) -> None:
    stale = types.ModuleType("inferencekey._inferencekey")
    stale.BACKEND_SERVER_ABI = 2
    _block_native(monkeypatch, stale)
    monkeypatch.setenv("IK_BACKEND_HTTP", "auto")
    transport = serve_mod._select_transport()
    assert transport.kind == "stdlib"
    assert "BACKEND_SERVER_ABI is 2" in transport.reason


def test_auto_picks_native_when_available(monkeypatch) -> None:
    require_native()
    monkeypatch.setenv("IK_BACKEND_HTTP", "auto")
    assert serve_mod._select_transport().kind == "native"


def test_default_is_stdlib_in_this_release(monkeypatch) -> None:
    monkeypatch.delenv("IK_BACKEND_HTTP", raising=False)
    transport = serve_mod._select_transport()
    assert transport.kind == "stdlib"
    assert transport.reason == "IK_BACKEND_HTTP=stdlib (default)"


def _main(monkeypatch, *env: Tuple[str, str]) -> int:
    for key, value in env:
        monkeypatch.setenv(key, value)
    return serve_mod.main(["--port", str(_free_port()), "--backend", f"{__name__}:ProcessOnlyBackend"])


def test_native_without_the_native_module_exits_2(monkeypatch, capfd) -> None:
    _block_native(monkeypatch)
    assert _main(monkeypatch, ("IK_BACKEND_HTTP", "native")) == 2
    err = capfd.readouterr().err
    assert "IK_BACKEND_HTTP=native but the native transport is unavailable" in err


@pytest.mark.parametrize("env", [
    ("IK_BACKEND_HTTP", "fast"),
    ("IK_BACKEND_MAX_INFLIGHT", "0"),
    ("IK_BACKEND_IO_THREADS", "four"),
    ("IK_BACKEND_BACKLOG", "-1"),
    ("IK_BACKEND_STREAM_MAX_BYTES", "1e6"),
    ("IK_BACKEND_FLUSH_TICK_US", "1ms"),
    ("IK_BACKEND_FLUSH_TICK_US", "250"),
    ("IK_BACKEND_FLUSH_TICK_US", "100001"),
    ("IK_BACKEND_FLUSH_TICK_US", "99999999999999999999999"),
    ("IK_SSE_CONNECTION_CLOSE", "maybe"),
])
def test_invalid_transport_settings_exit_2(monkeypatch, capfd, env) -> None:
    if env[0] != "IK_BACKEND_HTTP":
        require_native()
        monkeypatch.setenv("IK_BACKEND_HTTP", "native")
    assert _main(monkeypatch, env) == 2
    assert env[0] in capfd.readouterr().err


def test_env_knobs_reach_the_native_server(monkeypatch, capfd) -> None:
    httpd, _ = _start(ProcessOnlyBackend(), "native", monkeypatch,
                      IK_BACKEND_MAX_INFLIGHT="7", IK_BACKEND_IO_THREADS="2", IK_BACKEND_BACKLOG="128",
                      IK_BACKEND_FLUSH_TICK_US="2500")
    try:
        err = capfd.readouterr().err
        assert "http transport: native (abi 1, io_threads=2, backlog=128, max_inflight=7, flush_tick_us=2500)" in err
    finally:
        _stop(httpd)


def test_flush_tick_zero_streams_the_same_bytes(monkeypatch) -> None:
    # IK_BACKEND_FLUSH_TICK_US=0 wakes each stream on every push instead of in
    # batches; the bytes on the wire must not depend on it.
    bodies = []
    for tick in ("0", "1000"):
        httpd, _ = _start(GoldenBackend(), "native", monkeypatch, IK_BACKEND_FLUSH_TICK_US=tick)
        try:
            assert httpd._server.flush_tick_us == int(tick)
            reply = raw_request(httpd.server_address[1], "POST", "/v1/chat/completions", _chat("default"))
        finally:
            _stop(httpd)
        assert reply.status_line.endswith("200 OK")
        bodies.append(reply.body)
    assert bodies[0] == bodies[1]
    assert bodies[0].endswith(b"data: [DONE]\n\n")


# --------------------------------------------------------------------------
# 5. HTTP/1.0 and IK_SSE_CONNECTION_CLOSE framing
# --------------------------------------------------------------------------


def test_http10_client_gets_a_close_delimited_stream(monkeypatch) -> None:
    httpd = _serve_golden("native", monkeypatch)
    try:
        reply = raw_request(httpd.server_address[1], "POST", "/v1/chat/completions", _chat("default"),
                            version="HTTP/1.0")
    finally:
        _stop(httpd)
    assert reply.status_line == "HTTP/1.0 200 OK"
    assert "transfer-encoding" not in reply.headers and "content-length" not in reply.headers
    assert reply.raw_body.endswith(b"data: [DONE]\n\n")


def test_sse_connection_close_forces_close_delimited_streams(monkeypatch) -> None:
    httpd, _ = _start(GoldenBackend(), "native", monkeypatch, IK_SSE_CONNECTION_CLOSE="1")
    try:
        reply = raw_request(httpd.server_address[1], "POST", "/v1/chat/completions", _chat("default"))
    finally:
        _stop(httpd)
    assert reply.version == "HTTP/1.0" and not reply.chunked
    assert reply.raw_body.endswith(b"data: [DONE]\n\n")


# --------------------------------------------------------------------------
# 6. A burst of 500 streams
# --------------------------------------------------------------------------


class TwentyChunkBackend(CustomBackend):
    name = "twenty"
    api = "openai-chat"

    def setup(self, ctx) -> None:
        pass

    def process(self, job):
        return Result(output={})

    def chat(self, request):
        return chat_completion("x", model=self.name)

    def chat_stream(self, request):
        for i in range(20):
            yield _delta(str(i))


def _stream_urllib(port: int, results: List[Any], index: int) -> None:
    body = _chat("default")
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            data = resp.read()
        results[index] = data.count(b"data: ") if data.endswith(b"data: [DONE]\n\n") else f"no DONE: {data[-80:]!r}"
    except Exception as exc:  # noqa: BLE001 (counted as an error)
        results[index] = repr(exc)


def _run_clients(port: int, count: int, target=_stream_urllib) -> List[Any]:
    results: List[Any] = [None] * count
    threads = [threading.Thread(target=target, args=(port, results, i), daemon=True) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(90)
    return results


def test_500_parallel_streams_all_complete(monkeypatch) -> None:
    httpd, _ = _start(TwentyChunkBackend(), "native", monkeypatch)
    try:
        results = _run_clients(httpd.server_address[1], 500)
        stats = httpd.stats()
    finally:
        _stop(httpd)
    errors = [r for r in results if r != 21]
    assert not errors, f"{len(errors)} of 500 failed, e.g. {errors[:3]}"
    assert stats["accept_errors"] == 0
    assert stats["chunks_pushed"] == 500 * 20


# --------------------------------------------------------------------------
# 7. Pass-through chunks are written verbatim
# --------------------------------------------------------------------------


def test_str_and_bytes_chunks_pass_through_verbatim(monkeypatch) -> None:
    httpd = _serve_golden("native", monkeypatch)
    try:
        reply = raw_request(httpd.server_address[1], "POST", "/v1/chat/completions", _chat("default"))
    finally:
        _stop(httpd)
    assert b'data: {"choices":[{"index":0,"delta":{"content":"str chunk"},"finish_reason":null}]}\n\n' in reply.body
    assert b'data: {"choices":[{"index":0,"delta":{"content":"bytes chunk"},"finish_reason":null}]}\n\n' in reply.body


# --------------------------------------------------------------------------
# 8. shutdown() and server_close() free the port
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", TRANSPORTS)
def test_shutdown_frees_the_port_for_an_immediate_rebind(monkeypatch, name) -> None:
    _select(name, monkeypatch)
    port = _free_port()
    for _ in range(3):
        httpd, _state = serve_mod.serve_backend(ProcessOnlyBackend, port, block=False)
        assert httpd.server_address == ("127.0.0.1", port)
        assert _get(port, "/healthz")[0] == 200
        started = time.monotonic()
        _stop(httpd)
        assert time.monotonic() - started < 2
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


@pytest.mark.parametrize("name", TRANSPORTS)
def test_server_close_releases_the_backend(monkeypatch, name) -> None:
    """A REPL or a test suite restarting a backend must get its memory (weights, VRAM) back."""
    import gc
    import weakref

    _select(name, monkeypatch)
    refs = []
    for _ in range(3):
        backend = ProcessOnlyBackend()
        refs.append(weakref.ref(backend))
        httpd, state = serve_mod.serve_backend(backend, 0, block=False)
        port = httpd.server_address[1]
        reply = raw_request(port, "POST", "/process", json.dumps({"id": "j", "input": {}}).encode())
        assert reply.status == 200, reply.body
        _stop(httpd)
        del backend, httpd, state
        gc.collect()
    alive = sum(ref() is not None for ref in refs)
    assert alive == 0, f"{alive} of 3 backends still alive after server_close() and gc.collect()"


def test_native_shutdown_closes_open_streams(monkeypatch) -> None:
    SlowStreamBackend.closed.clear()
    SlowStreamBackend.started.clear()
    httpd, _ = _start(SlowStreamBackend(), "native", monkeypatch)
    sock = _open_stream(httpd.server_address[1], "chunks")
    try:
        assert SlowStreamBackend.started.wait(5)
        started = time.monotonic()
        _stop(httpd)
        assert time.monotonic() - started < 2
        assert SlowStreamBackend.closed.wait(1)
    finally:
        sock.close()


# --------------------------------------------------------------------------
# 9. GIL hog under a watchdog
# --------------------------------------------------------------------------


class GilHogBackend(CustomBackend):
    """Burns pure-Python CPU for every chunk."""

    name = "gil-hog"
    api = "openai-chat"

    def setup(self, ctx) -> None:
        pass

    def process(self, job):
        return Result(output={})

    def chat(self, request):
        return chat_completion("x", model=self.name)

    def chat_stream(self, request):
        for i in range(10):
            x = 0
            for _ in range(20000):
                x += 1
            yield _delta(str(x + i))


def test_gil_hog_backend_at_200_streams_does_not_deadlock(monkeypatch) -> None:
    httpd, _ = _start(GilHogBackend(), "native", monkeypatch)
    outcome: Dict[str, Any] = {}

    def run() -> None:
        outcome["results"] = _run_clients(httpd.server_address[1], 200)

    watchdog = threading.Thread(target=run, daemon=True)
    started = time.monotonic()
    watchdog.start()
    watchdog.join(60)
    try:
        assert not watchdog.is_alive(), "deadlock: 200 GIL-hog streams did not finish within 60 s"
        errors = [r for r in outcome["results"] if r != 11]
        assert not errors, f"{len(errors)} of 200 failed, e.g. {errors[:3]}"
        assert _get(httpd.server_address[1], "/healthz")[0] == 200
    finally:
        _stop(httpd)
    assert time.monotonic() - started < 60


# --------------------------------------------------------------------------
# CLI: the listener belongs to the serving process; the transport line
# --------------------------------------------------------------------------

CLI_BACKEND = textwrap.dedent(
    '''
    from inferencekey.backend import CustomBackend, Result

    class CliBackend(CustomBackend):
        name = "cli"

        def setup(self, ctx):
            pass

        def process(self, job):
            return Result(output={"ok": True})
    '''
)


@pytest.mark.parametrize("name", TRANSPORTS)
def test_cli_owns_its_listener_and_names_the_transport(tmp_path, name) -> None:
    if name == "native":
        require_native()
    proc, port = _spawn_cli(tmp_path, CLI_BACKEND, "CliBackend", name)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and _probe(port, 0.5)[0] != 200:
            time.sleep(0.05)
        assert _probe(port)[0] == 200
        listeners = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True,
                                   text=True, timeout=10).stdout
        assert f"pid={proc.pid}," in listeners, listeners
        if name == "native":
            names = {
                open(f"/proc/{proc.pid}/task/{tid}/comm").read().strip()
                for tid in os.listdir(f"/proc/{proc.pid}/task")
            }
            assert any(n.startswith("ik-http-io") for n in names), names
    finally:
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=20)
    assert proc.returncode == 0, err
    lines = [line for line in err.splitlines() if line.startswith("http transport:")]
    assert len(lines) == 1 and lines[0].startswith(f"http transport: {name} ("), err
    assert out.count("model loaded") == 1
    assert f"serving on http://127.0.0.1:{port}" in out


SLEEPY_CLI_BACKEND = textwrap.dedent(
    '''
    import sys
    import time

    from inferencekey.backend import CustomBackend, Result

    class SleepyBackend(CustomBackend):
        name = "sleepy"

        def setup(self, ctx):
            pass

        def process(self, job):
            print("process started", file=sys.stderr, flush=True)
            time.sleep(30)
            return Result(output={})
    '''
)


@pytest.mark.parametrize("name", TRANSPORTS)
def test_sigterm_with_a_call_in_flight_exits_within_the_worker_budget(tmp_path, name) -> None:
    """The worker sends SIGTERM and kills 5 s later; the backend's own cleanup must get to run."""
    if name == "native":
        require_native()
    proc, port = _spawn_cli(tmp_path, SLEEPY_CLI_BACKEND, "SleepyBackend", name)
    client: Optional[threading.Thread] = None
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and _probe(port, 0.5)[0] != 200:
            time.sleep(0.05)
        assert _probe(port)[0] == 200

        def post() -> None:
            try:
                raw_request(port, "POST", "/process", json.dumps({"id": "j", "input": {}}).encode(), timeout=40)
            except OSError:
                pass

        client = threading.Thread(target=post, daemon=True)
        client.start()
        # Wait until the backend is inside process().
        line = proc.stderr.readline()
        while line and "process started" not in line:
            line = proc.stderr.readline()
        assert "process started" in line
        started = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=20)
        took = time.monotonic() - started
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=20)
    print(f"\n[{name}] SIGTERM with one sleeping /process: exit after {took:.2f} s, rc={proc.returncode}")
    assert proc.returncode == 0
    assert took < 3.0, f"{name} took {took:.2f} s to exit after SIGTERM"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(port: int, path: str) -> Tuple[int, bytes]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _wait_for_status(port: int, status: int, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if _get(port, "/healthz")[0] == status:
                return
        except OSError:
            pass
        time.sleep(0.02)
    raise AssertionError(f"port {port} never answered {status}")


def test_native_is_available_when_required() -> None:
    if os.environ.get("IK_REQUIRE_NATIVE") == "1":
        assert native_available()

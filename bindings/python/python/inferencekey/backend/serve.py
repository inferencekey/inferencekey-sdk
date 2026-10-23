"""Runtime: serve a :class:`~inferencekey.backend.CustomBackend` over HTTP.

Long-lived, single-process server bound to ``127.0.0.1`` (loopback only, never
``0.0.0.0``). The SDK runtime pulls in no HTTP framework and **no** ``torch``
(``torch`` belongs to the developer's backend).

Two transports serve the same request semantics (the functions below):

* ``native``: the Rust server in the SDK's native module (``_inferencekey``).
  It accepts, parses, answers ``/healthz``, ``/meta`` and ``/v1/models`` and
  writes every byte without the GIL; Python only runs the backend methods.
* ``stdlib``: the standard library :mod:`http.server`, one thread per
  connection. It needs no native module.

``IK_BACKEND_HTTP`` picks one: ``stdlib`` (the default for now), ``auto``
(native when the native module has the server, else stdlib) or ``native``
(exit code 2 when it is unavailable). One ``http transport: ...`` line on
stderr names the transport in use. The native transport also reads
``IK_BACKEND_MAX_INFLIGHT``, ``IK_BACKEND_IO_THREADS``, ``IK_BACKEND_BACKLOG``,
``IK_BACKEND_STREAM_MAX_BYTES``, ``IK_BACKEND_FLUSH_TICK_US`` and
``IK_SSE_CONNECTION_CLOSE``.

Run it::

    python -m inferencekey.backend.serve --port <port> --backend <module:Class>

The entrypoint may also come from the ``IK_BACKEND_ENTRYPOINT`` env var and the
model config from ``--config-json`` or the ``IK_BACKEND_CONFIG`` env var (JSON).

Endpoints (a minimal, SDK-owned schema — **not** OpenAI-compatible):

* ``GET /healthz`` — ``503`` until :meth:`setup` finishes, then ``200``
  ``{"status": "ok"}``.
* ``GET /meta`` — ``200`` with the backend's declarative metadata
  (``name``/``version``/``task_type``/``requirements``); available regardless of
  readiness.
* ``POST /process`` — body ``{"id": str, "input": {...}}``; returns ``200``
  ``{"output": {...}}``. A bad body is ``400``; a raising ``process()`` is
  ``500`` ``{"error": "..."}`` and the server stays alive.
* ``POST /forecast`` — body ``{"id": str, "target": [...], "horizon": int}``
  (see :class:`~inferencekey.backend.ForecastRequest`); returns ``200``
  ``{"forecast": {...}}``. Same body/readiness handling as ``/process``: a bad
  body is ``400``, a raising ``forecast()`` is ``500`` and the server stays
  alive. If the backend does not implement ``forecast()`` it answers ``501``
  ``{"error": "backend does not support forecast"}`` — the capability is
  optional, so ``process``-only backends keep working unchanged.
* ``POST /v1/chat/completions``: an OpenAI chat completion request, served by
  :meth:`~inferencekey.backend.CustomBackend.chat` (or ``chat_stream`` when the
  body has ``"stream": true``, answered as server-sent events ending in
  ``data: [DONE]``; a ``None`` item from ``chat_stream`` is written as the SSE
  comment ``: keepalive``, which clients ignore, so a backend can find out that
  a client left while it has nothing to send; a ``str`` or ``bytes`` item is
  taken as an already-serialized chunk and written as is, without
  ``json.dumps``; when the client leaves, the generator is closed, so its
  ``finally`` runs). Only for backends that implement ``chat()`` (``api`` is
  ``"openai-chat"`` in ``/meta``); the others answer ``501``. Errors use the
  OpenAI shape ``{"error": {"message", "type"}}``, plus ``"code"`` when the
  raised :class:`~inferencekey.backend.InvalidInput` has a ``code`` attribute
  (for example ``"context_length_exceeded"``).
* ``GET /v1/models``: the one model this backend serves, OpenAI-shaped.

To start a backend from code instead of the CLI, use :func:`serve_backend`.

Lifecycle: :meth:`setup` is called once before traffic. If it raises, the
traceback goes to stderr and the process exits non-zero (the worker supervisor
sees it). On success exactly one ``model loaded`` line is logged. ``SIGTERM``
(and ``SIGINT``) shut the server down cleanly.
"""

from __future__ import annotations

import argparse
import atexit
import html
import importlib
import json
import os
import re
import signal
import sys
import threading
import traceback
import weakref
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, Optional, Protocol, Tuple, Union

from ..errors import BackendEntrypointError, BackendSetupError
from .base import (
    BackendContext,
    CustomBackend,
    ForecastRequest,
    ForecastResult,
    InvalidInput,
    Job,
    Result,
)

#: Loopback only — the runtime never binds a routable interface.
HOST = "127.0.0.1"

#: Cap request bodies so a misbehaving client cannot exhaust memory.
_MAX_BODY_BYTES = 16 * 1024 * 1024

#: What ``next()`` returns for a chunk generator with no chunk at all.
_NO_CHUNK = object()

#: Listen backlog. The stdlib default (5) resets connections as soon as a few
#: hundred clients open streams at once (one LLM backend serving 400 developers
#: lost about 300 of 400 that way); the kernel caps it at ``net.core.somaxconn``.
LISTEN_BACKLOG = 1024


class _HTTPServer(ThreadingHTTPServer):
    """One daemon thread per connection, with a backlog sized for bursts of streams."""

    request_queue_size = LISTEN_BACKLOG
    daemon_threads = True


def _log(message: str) -> None:
    """Emit one line to stdout (operational logs; no secrets)."""
    print(message, file=sys.stdout, flush=True)


def _log_err(message: str) -> None:
    """Emit one line to stderr."""
    print(message, file=sys.stderr, flush=True)


def load_backend(entrypoint: str) -> CustomBackend:
    """Import ``module:Class`` and instantiate the backend.

    Raises :class:`~inferencekey.errors.BackendEntrypointError` if the
    entrypoint is malformed, cannot be imported, or is not a
    :class:`CustomBackend` subclass.
    """
    if ":" not in entrypoint:
        raise BackendEntrypointError(
            f"entrypoint must be 'module:Class', got {entrypoint!r}"
        )
    module_name, _, class_name = entrypoint.partition(":")
    if not module_name or not class_name:
        raise BackendEntrypointError(
            f"entrypoint must be 'module:Class', got {entrypoint!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 — wrap any import failure
        raise BackendEntrypointError(
            f"could not import backend module {module_name!r}: {exc}"
        ) from exc
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise BackendEntrypointError(
            f"module {module_name!r} has no attribute {class_name!r}"
        ) from exc
    if not (isinstance(cls, type) and issubclass(cls, CustomBackend)):
        raise BackendEntrypointError(
            f"{entrypoint!r} is not a CustomBackend subclass"
        )
    return cls()


class _BackendState:
    """Shared, thread-safe state between the lifecycle and the HTTP handlers."""

    def __init__(self, backend: CustomBackend) -> None:
        self._backend = backend
        self._ready = threading.Event()
        self._ready_hooks: list = []

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def on_ready(self, hook: Any) -> None:
        """Call ``hook()`` when the backend becomes ready (the native transport's health flag)."""
        self._ready_hooks.append(hook)

    def mark_ready(self) -> None:
        self._ready.set()
        hooks, self._ready_hooks = self._ready_hooks, []
        for hook in hooks:
            hook()

    def process(self, job: Job) -> Result:
        return self._backend.process(job)

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        return self._backend.forecast(request)

    def chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self._backend.chat(request)

    def chat_stream(self, request: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        return self._backend.chat_stream(request)

    def manifest(self) -> Dict[str, Any]:
        """The backend's declarative metadata, as a JSON-serializable dict."""
        return self._backend.manifest().to_wire()


# --------------------------------------------------------------------------
# Request semantics, shared by every HTTP transport
# --------------------------------------------------------------------------
#
# The transport (today the standard library server below) owns sockets,
# framing and the Content-Length rules; these functions own what a request
# means. Each takes the raw request body and returns the status and the JSON
# bytes to send, so two transports built on them answer byte for byte alike.

#: A JSON response: HTTP status and the exact body bytes.
JsonReply = Tuple[int, bytes]

#: An acceptable ``Content-Length`` value (see :func:`content_length`).
_CONTENT_LENGTH = re.compile(r"-?[0-9]+")


def _json_reply(status: int, payload: Any) -> JsonReply:
    return status, json.dumps(payload).encode("utf-8")


def _openai_error(status: int, message: str, kind: str, code: Optional[str] = None) -> JsonReply:
    error: Dict[str, Any] = {"message": message, "type": kind}
    if code:
        error["code"] = code
    return _json_reply(status, {"error": error})


def content_length(header: Optional[str]) -> Tuple[int, Optional[JsonReply]]:
    """Validate a ``Content-Length`` header: ``(length, None)`` or ``(0, error reply)``.

    A missing header counts as an empty body. The value must be ASCII digits,
    optionally after a minus sign (a negative length is "too large", as it has
    always been); forms ``int()`` would also take, such as ``+5``, ``1_0``, inner
    spaces or non-ASCII digits, are invalid, so every transport applies one
    exact rule. Every POST route uses the same rules and the same plain
    ``{"error": ...}`` shape, ``/v1/chat/completions`` included.
    """
    if header is None:
        return 0, None
    if not _CONTENT_LENGTH.fullmatch(header):
        return 0, _json_reply(400, {"error": "invalid Content-Length"})
    length = int(header)
    if length < 0 or length > _MAX_BODY_BYTES:
        return 0, _json_reply(400, {"error": "request body too large"})
    return length, None


def _decode_body(body: bytes) -> Tuple[Any, Optional[JsonReply]]:
    """Decode a JSON request body (empty means ``{}``): ``(data, None)`` or ``(None, 400 reply)``."""
    try:
        return (json.loads(body.decode("utf-8")) if body else {}), None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, _json_reply(400, {"error": f"invalid JSON body: {exc}"})


def healthz_reply(state: _BackendState) -> JsonReply:
    """``GET /healthz``: 200 once :meth:`setup` finished, 503 while it runs."""
    if state.ready:
        return _json_reply(200, {"status": "ok"})
    return _json_reply(503, {"status": "loading"})


def models_payload(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """``GET /v1/models``: the one model this backend serves, OpenAI-shaped."""
    return {
        "object": "list",
        "data": [{"id": manifest.get("name", ""), "object": "model", "owned_by": "custom"}],
    }


def handle_process(state: _BackendState, body: bytes) -> JsonReply:
    """``POST /process``: run :meth:`CustomBackend.process` on a job body."""
    if not state.ready:
        return _json_reply(503, {"error": "backend not ready"})
    data, error = _decode_body(body)
    if error is not None:
        return error

    try:
        job = Job.from_wire(data)
    except ValueError as exc:
        return _json_reply(400, {"error": str(exc)})

    try:
        result = state.process(job)
    except InvalidInput as exc:
        # The client's input was malformed (e.g. a transcription request with
        # no audio). That is a 4xx (the caller's fault, not the model's), so
        # return 400, not the misleading 500 a generic failure gets. Caught
        # BEFORE `except Exception` on purpose.
        return _json_reply(400, {"error": str(exc)})
    except Exception as exc:  # noqa: BLE001 (keep the server alive)
        # Resilience: log the traceback for operators, return a clean 500 to
        # the caller, and stay alive for the next job. A bare ValueError from
        # the user's model is a real bug (500), NOT a client error; only
        # InvalidInput (above) is reclassified.
        _log_err(f"process failed for job {job.id}:")
        traceback.print_exc(file=sys.stderr)
        return _json_reply(500, {"error": str(exc)})

    if not isinstance(result, Result):
        return _json_reply(
            500,
            {"error": "process() must return a Result; wrap your value as Result(output=...)"},
        )
    return _json_reply(200, result.to_wire())


def handle_forecast(state: _BackendState, body: bytes) -> JsonReply:
    """``POST /forecast``: run :meth:`CustomBackend.forecast` on a forecast body."""
    if not state.ready:
        return _json_reply(503, {"error": "backend not ready"})
    data, error = _decode_body(body)
    if error is not None:
        return error

    try:
        request = ForecastRequest.from_wire(data)
    except ValueError as exc:
        return _json_reply(400, {"error": str(exc)})

    try:
        result = state.forecast(request)
    except NotImplementedError:
        # The backend does not implement forecast(): a distinct, expected
        # condition (501), not a per-request failure (500).
        return _json_reply(501, {"error": "backend does not support forecast"})
    except InvalidInput as exc:
        # Malformed client input surfaced from an input accessor: 400, caught
        # BEFORE the generic handler; a bare ValueError from the user's model
        # stays a real 500 below.
        return _json_reply(400, {"error": str(exc)})
    except Exception as exc:  # noqa: BLE001 (keep the server alive)
        _log_err(f"forecast failed for request {request.id}:")
        traceback.print_exc(file=sys.stderr)
        return _json_reply(500, {"error": str(exc)})

    if not isinstance(result, ForecastResult):
        return _json_reply(
            500,
            {
                "error": "forecast() must return a ForecastResult; wrap your value as "
                "ForecastResult(median=...)"
            },
        )
    return _json_reply(200, result.to_wire())


def prepare_chat(
    state: _BackendState, body: bytes
) -> Union[Tuple[str, int, bytes], Tuple[str, Dict[str, Any]]]:
    """``POST /v1/chat/completions`` up to the point where a stream would start.

    Returns ``("json", status, body)`` for a non-streaming request and for every
    early error (503 not ready, invalid JSON, no ``messages`` list, and the
    ``chat()`` error mapping), or ``("stream", request)`` when the request asks
    to stream; the transport then calls :func:`drive_chat_stream`.
    """
    if not state.ready:
        return ("json",) + _openai_error(503, "backend not ready", "server_error")
    data, error = _decode_body(body)
    if error is not None:
        return ("json",) + error
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        return ("json",) + _openai_error(
            400, "request must be a JSON object with a 'messages' list", "invalid_request_error"
        )

    if data.get("stream"):
        return ("stream", data)

    try:
        completion = state.chat(data)
    except NotImplementedError:
        return ("json",) + _openai_error(501, "backend does not support chat", "invalid_request_error")
    except InvalidInput as exc:
        return ("json",) + _openai_error(
            400, str(exc), "invalid_request_error", getattr(exc, "code", None)
        )
    except Exception as exc:  # noqa: BLE001 (keep the server alive)
        _log_err("chat failed:")
        traceback.print_exc(file=sys.stderr)
        return ("json",) + _openai_error(500, str(exc), "server_error")
    if not isinstance(completion, dict):
        return ("json",) + _openai_error(
            500,
            "chat() must return a chat.completion dict; build it with "
            "inferencekey.backend.chat_completion(...)",
            "server_error",
        )
    return ("json",) + _json_reply(200, completion)


class StreamOut(Protocol):
    """Where :func:`drive_chat_stream` writes a stream; each transport implements it.

    ``reply`` answers with a plain JSON response instead of a stream (only
    before ``start``). ``start`` sends the 200 event-stream head. ``push`` sends
    one ``data:`` event with the given payload and ``keepalive`` the SSE comment
    ``: keepalive``; both return ``False`` once the client is gone. ``done``
    sends ``data: [DONE]`` and ends the stream.
    """

    def reply(self, status: int, body: bytes) -> None: ...

    def start(self) -> None: ...

    def push(self, payload: bytes) -> bool: ...

    def keepalive(self) -> bool: ...

    def done(self) -> None: ...


def encode_chunk(chunk: Any) -> bytes:
    """The ``data:`` payload of one chunk.

    A dict (or any JSON value) is serialized with :func:`json.dumps`. A ``str``
    or ``bytes`` chunk is taken as an already-serialized chunk object and passed
    through as is, which saves the ``json.dumps`` on hot streaming paths. It
    must be one line of JSON: a line break would end the ``data:`` line early
    and the client would get truncated JSON, so it raises ``ValueError``
    instead (which the stream driver turns into an error event).
    """
    if isinstance(chunk, bytes):
        data = chunk
    elif isinstance(chunk, str):
        data = chunk.encode("utf-8")
    elif isinstance(chunk, (bytearray, memoryview)):
        data = bytes(chunk)
    else:
        return json.dumps(chunk).encode("utf-8")
    if b"\n" in data or b"\r" in data:
        raise ValueError("pre-serialized chunk must be one line of JSON")
    return data


def drive_chat_stream(state: _BackendState, request: Dict[str, Any], out: StreamOut) -> None:
    """Answer a streaming chat request through ``out`` as server-sent events.

    The first chunk is pulled before anything is sent, so a backend that
    refuses the request (not implemented, invalid input, a failure) still gets
    a proper status code instead of a 200 stream carrying an error. A ``None``
    chunk is a keepalive. When the client goes away (``push`` or ``keepalive``
    returns ``False``) the generator is closed, so its ``finally`` runs (the
    hook a backend uses to abort the generation). A failure after the first
    chunk is sent as an OpenAI error event, followed by ``[DONE]``.
    """
    try:
        chunks = iter(state.chat_stream(request))
        first = next(chunks, _NO_CHUNK)
    except NotImplementedError:
        out.reply(*_openai_error(501, "backend does not support chat", "invalid_request_error"))
        return
    except InvalidInput as exc:
        out.reply(*_openai_error(400, str(exc), "invalid_request_error", getattr(exc, "code", None)))
        return
    except Exception as exc:  # noqa: BLE001 (keep the server alive)
        _log_err("chat stream failed before the first chunk:")
        traceback.print_exc(file=sys.stderr)
        out.reply(*_openai_error(500, str(exc), "server_error"))
        return

    out.start()

    def send(chunk: Any) -> bool:
        # A keepalive has nothing to send, but a write is how a client that
        # left is noticed.
        return out.keepalive() if chunk is None else out.push(encode_chunk(chunk))

    try:
        if first is not _NO_CHUNK and not send(first):
            _close_quietly(chunks)
            return
        for chunk in chunks:
            if not send(chunk):
                # The client went away; nothing left to tell it. Closing the
                # generator lets the backend cancel the generation now (its
                # ``finally`` runs) instead of whenever it is garbage collected.
                _close_quietly(chunks)
                return
    except Exception as exc:  # noqa: BLE001 (keep the server alive)
        _log_err("chat stream failed mid-way:")
        traceback.print_exc(file=sys.stderr)
        _close_quietly(chunks)
        event = {"error": {"message": str(exc), "type": "server_error"}}
        if not out.push(json.dumps(event).encode("utf-8")):
            return
    out.done()


def _close_quietly(chunks: Iterator[Any]) -> None:
    """Close a backend's chunk generator, logging (not raising) what its cleanup raises."""
    close = getattr(chunks, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:  # noqa: BLE001 (keep the server alive)
        _log_err("chat stream cleanup failed:")
        traceback.print_exc(file=sys.stderr)


# --------------------------------------------------------------------------
# Standard library transport
# --------------------------------------------------------------------------


class _StdlibStream:
    """:class:`StreamOut` over a :class:`BaseHTTPRequestHandler`: HTTP/1.0, close-delimited.

    Every event is written and flushed at once. Any ``OSError`` from the socket
    (``ConnectionError`` covers a broken pipe, a reset and an abort; a timeout
    means the same) marks the client gone, and from then on ``push`` and
    ``keepalive`` return ``False`` without writing.
    """

    def __init__(self, handler: BaseHTTPRequestHandler) -> None:
        self._handler = handler
        self.closed = False

    def reply(self, status: int, body: bytes) -> None:
        _send_body(self._handler, status, body)

    def start(self) -> None:
        handler = self._handler
        # No Content-Length: the response ends when the connection closes
        # (HTTP/1.0 semantics of BaseHTTPRequestHandler).
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.close_connection = True
        try:
            handler.end_headers()
        except OSError:
            self.closed = True

    def _write(self, data: bytes) -> bool:
        if self.closed:
            return False
        try:
            self._handler.wfile.write(data)
            self._handler.wfile.flush()
        except OSError:
            self.closed = True
            return False
        return True

    def push(self, payload: bytes) -> bool:
        return self._write(b"data: " + payload + b"\n\n")

    def keepalive(self) -> bool:
        return self._write(b": keepalive\n\n")

    def done(self) -> None:
        self._write(b"data: [DONE]\n\n")


def _send_body(handler: BaseHTTPRequestHandler, status: int, body: bytes) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _make_handler(state: _BackendState) -> type:
    """Build a request handler bound to ``state`` (avoids global mutable state).

    A thin adapter: it reads the request and writes the response, and leaves
    what the request means to the shared functions above.
    """

    class _Handler(BaseHTTPRequestHandler):
        # Quieten the default per-request stderr logging; we log deliberately.
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802 (stdlib API)
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                _send_body(self, *healthz_reply(state))
            elif path == "/meta":
                # Static metadata; available regardless of readiness so an
                # operator can introspect the backend while the model loads.
                _send_body(self, *_json_reply(200, state.manifest()))
            elif path == "/v1/models":
                _send_body(self, *_json_reply(200, models_payload(state.manifest())))
            else:
                _send_body(self, *_json_reply(404, {"error": "not found"}))

        def _read_body(self) -> Optional[bytes]:
            """Read the request body, or send the 400 and return ``None``."""
            length, error = content_length(self.headers.get("Content-Length"))
            if error is not None:
                _send_body(self, *error)
                return None
            return self.rfile.read(length) if length else b""

        def do_POST(self) -> None:  # noqa: N802 (stdlib API)
            path = self.path.split("?", 1)[0]
            if path not in ("/process", "/forecast", "/v1/chat/completions"):
                _send_body(self, *_json_reply(404, {"error": "not found"}))
                return
            body = self._read_body()
            if body is None:
                return
            if path == "/process":
                _send_body(self, *handle_process(state, body))
            elif path == "/forecast":
                _send_body(self, *handle_forecast(state, body))
            else:
                prepared = prepare_chat(state, body)
                if prepared[0] == "stream":
                    drive_chat_stream(state, prepared[1], _StdlibStream(self))
                else:
                    _send_body(self, prepared[1], prepared[2])

    return _Handler

# --------------------------------------------------------------------------
# Native transport (Rust, in the SDK's native module)
# --------------------------------------------------------------------------

#: The native server interface this runtime speaks (``_inferencekey.BACKEND_SERVER_ABI``).
_NATIVE_ABI = 1

#: ``IK_BACKEND_HTTP`` when it is unset. ``auto`` once the native transport is
#: the default.
_DEFAULT_TRANSPORT = "stdlib"

_TRANSPORT_MODES = ("auto", "native", "stdlib")


class _Transport:
    """The transport :func:`serve` uses, and why (see :func:`_select_transport`)."""

    def __init__(
        self,
        kind: str,
        reason: str = "",
        module: Any = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.kind = kind
        self.reason = reason
        self.module = module
        self.options = options or {}

    def __repr__(self) -> str:
        return f"_Transport({self.kind!r}, {self.reason!r})"


def _env_int(name: str, minimum: int) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    if not re.fullmatch(r"[0-9]+", raw) or int(raw) < minimum:
        raise BackendSetupError(f"{name} must be an integer >= {minimum}, got {raw!r}")
    return int(raw)


# The flush tick in microseconds: 0 (a wake per push) or a value the server's
# 1 ms timer can honor (it rounds deadlines up to whole milliseconds, so ticks
# from 1 to 999 would all behave like about 1000), up to 100 ms (a larger tick
# would hold a busy stream's events that long).
_FLUSH_TICK_MIN_US = 1000
_FLUSH_TICK_MAX_US = 100_000


def _env_flush_tick() -> Optional[int]:
    name = "IK_BACKEND_FLUSH_TICK_US"
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    if not re.fullmatch(r"[0-9]+", raw) or not (
        int(raw) == 0 or _FLUSH_TICK_MIN_US <= int(raw) <= _FLUSH_TICK_MAX_US
    ):
        raise BackendSetupError(
            f"{name} must be 0 or an integer from {_FLUSH_TICK_MIN_US} to {_FLUSH_TICK_MAX_US}, got {raw!r}"
        )
    return int(raw)


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise BackendSetupError(f"{name} must be 0 or 1, got {raw!r}")


def _native_options() -> Dict[str, Any]:
    """The native server settings from the environment (unset ones keep the server's defaults)."""
    options: Dict[str, Any] = {}
    for env, key, minimum in (
        ("IK_BACKEND_MAX_INFLIGHT", "max_inflight", 1),
        ("IK_BACKEND_IO_THREADS", "io_threads", 1),
        ("IK_BACKEND_BACKLOG", "backlog", 1),
        ("IK_BACKEND_STREAM_MAX_BYTES", "stream_max_bytes", 1),
    ):
        value = _env_int(env, minimum)
        if value is not None:
            options[key] = value
    flush_tick = _env_flush_tick()
    if flush_tick is not None:
        options["flush_tick_us"] = flush_tick
    if _env_flag("IK_SSE_CONNECTION_CLOSE"):
        options["sse_connection_close"] = True
    return options


def _load_native() -> Tuple[Any, str]:
    """``(module, "")`` when the native server is usable, else ``(None, reason)``."""
    try:
        from .. import _inferencekey as module
    except (ImportError, OSError) as exc:
        return None, f"_inferencekey could not be imported ({type(exc).__name__}: {exc})"
    abi = getattr(module, "BACKEND_SERVER_ABI", None)
    if abi is None:
        return None, "_inferencekey has no BACKEND_SERVER_ABI"
    if abi != _NATIVE_ABI:
        return None, f"_inferencekey BACKEND_SERVER_ABI is {abi!r}, this runtime needs {_NATIVE_ABI}"
    return module, ""


def _select_transport() -> _Transport:
    """Pick the HTTP transport from ``IK_BACKEND_HTTP`` (``auto``, ``native`` or ``stdlib``).

    ``auto`` uses the native server when the native module imports and has a
    matching ``BACKEND_SERVER_ABI``, and the standard library otherwise.
    ``native`` raises :class:`~inferencekey.errors.BackendSetupError` when it
    is unavailable (the CLI exits 2). Invalid knob values raise the same.
    """
    raw = os.environ.get("IK_BACKEND_HTTP", "").strip().lower()
    mode = raw or _DEFAULT_TRANSPORT
    if mode not in _TRANSPORT_MODES:
        raise BackendSetupError(
            f"IK_BACKEND_HTTP must be one of {', '.join(_TRANSPORT_MODES)}, got {raw!r}"
        )
    if mode == "stdlib":
        return _Transport("stdlib", "IK_BACKEND_HTTP=stdlib" + ("" if raw else " (default)"))
    module, why_not = _load_native()
    if module is None:
        if mode == "native":
            raise BackendSetupError(f"IK_BACKEND_HTTP=native but the native transport is unavailable: {why_not}")
        return _Transport("stdlib", why_not)
    return _Transport("native", module=module, options=_native_options())


class _NativeApp:
    """What the Rust server calls for each POST request (on its request threads)."""

    def __init__(self, state: _BackendState) -> None:
        self._state = state

    def handle_json(self, route: str, body: bytes) -> JsonReply:
        if route == "/process":
            return handle_process(self._state, body)
        if route == "/forecast":
            return handle_forecast(self._state, body)
        return _json_reply(404, {"error": "not found"})

    def handle_stream(self, body: bytes, sink: Any) -> None:
        """``POST /v1/chat/completions``: a plain JSON reply or an event stream, through ``sink``."""
        prepared = prepare_chat(self._state, body)
        if prepared[0] == "stream":
            drive_chat_stream(self._state, prepared[1], sink)
        else:
            sink.reply(prepared[1], prepared[2])


class _NativeHttpd:
    """The native server with the parts of :class:`ThreadingHTTPServer` callers use.

    ``server_address``, ``server_port``, ``serve_forever()``, ``shutdown()``
    and ``server_close()``. ``shutdown()`` stops accepting, closes open streams
    (their generators are closed at the next chunk) and lets in-flight
    responses finish for up to 2 s (the worker kills a backend 5 s after
    SIGTERM); ``server_close()`` releases the runtime and the reference to the
    backend.
    Both wait with the GIL released and are safe to call more than once. An
    ``atexit`` hook closes a server nobody closed.
    """

    def __init__(self, server: Any) -> None:
        self._server = server
        self.server_address = tuple(server.server_address)
        self.server_port = self.server_address[1]
        atexit.register(self._close_at_exit)

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        del poll_interval  # the native server needs no polling
        self._server.wait()

    def shutdown(self) -> None:
        self._server.shutdown()

    def server_close(self) -> None:
        self._server.close()
        atexit.unregister(self._close_at_exit)

    def stats(self) -> Dict[str, Any]:
        """The ``/debug/stats`` counters."""
        return json.loads(self._server.stats())

    def _close_at_exit(self) -> None:
        self._server.close()


def _start_native(state: _BackendState, port: int, transport: _Transport) -> _NativeHttpd:
    server = transport.module.BackendServer(
        port,
        _NativeApp(state),
        unsupported_method_page=_unsupported_method_page(),
        **transport.options,
    )
    _set_native_manifest(server, state)
    # Weak: the server holds the app, the app holds the state, and a strong
    # hook would close the cycle back to the server (whose native object the
    # GC cannot traverse), leaking the backend after server_close().
    # _NativeHttpd keeps the server alive while it is in use.
    server_ref = weakref.ref(server)

    def _set_ready() -> None:
        live = server_ref()
        if live is not None:
            live.set_ready(True)

    state.on_ready(_set_ready)
    return _NativeHttpd(server)


def _unsupported_method_page() -> Tuple[str, str]:
    """The 501 page this interpreter's ``http.server`` sends, with a ``%(message)s`` placeholder.

    ``(page, content_type)`` for the native server. Rendered here because the
    page differs between CPython versions (``DEFAULT_ERROR_MESSAGE`` changed in
    3.11, and so did ``str()`` of the ``HTTPStatus`` it is formatted with).
    Mirrors ``BaseHTTPRequestHandler.send_error``.
    """
    handler = BaseHTTPRequestHandler
    code = HTTPStatus.NOT_IMPLEMENTED
    page = handler.error_message_format % {
        "code": code,
        "message": "%(message)s",
        "explain": html.escape(handler.responses[code][1], quote=False),
    }
    return page, handler.error_content_type


def _set_native_manifest(server: Any, state: _BackendState) -> None:
    try:
        manifest = json.dumps(state.manifest()).encode("utf-8")
    except Exception:  # noqa: BLE001 (keep serving; /meta then shows {})
        _log_err("manifest() failed:")
        traceback.print_exc(file=sys.stderr)
        return
    server.set_manifest(manifest)


def _transport_line(transport: _Transport, httpd: Any) -> str:
    if transport.kind == "native":
        server = httpd._server
        return (
            f"http transport: native (abi {_NATIVE_ABI}, io_threads={server.io_threads}, "
            f"backlog={server.backlog}, max_inflight={server.max_inflight}, "
            f"flush_tick_us={server.flush_tick_us})"
        )
    return f"http transport: stdlib (reason: {transport.reason})"


def _parse_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve model config from ``--config-json`` or ``IK_BACKEND_CONFIG``."""
    raw: Optional[str] = args.config_json
    if raw is None:
        raw = os.environ.get("IK_BACKEND_CONFIG")
    if raw is None or raw == "":
        return {}
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BackendSetupError(f"config is not valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise BackendSetupError("config must be a JSON object")
    return config


def _resolve_entrypoint(args: argparse.Namespace) -> str:
    """Resolve the entrypoint from ``--backend`` or ``IK_BACKEND_ENTRYPOINT``."""
    entrypoint = args.backend or os.environ.get("IK_BACKEND_ENTRYPOINT")
    if not entrypoint:
        raise BackendEntrypointError(
            "no backend entrypoint: pass --backend module:Class or set "
            "IK_BACKEND_ENTRYPOINT"
        )
    return entrypoint


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inferencekey.backend.serve",
        description="Serve a CustomBackend over a loopback HTTP server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="loopback TCP port to bind (127.0.0.1 only)",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="entrypoint 'module:Class'; defaults to $IK_BACKEND_ENTRYPOINT",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="model config as a JSON object; defaults to $IK_BACKEND_CONFIG",
    )
    return parser


def serve(
    backend: CustomBackend,
    port: int,
    config: Dict[str, Any],
    *,
    transport: Optional[_Transport] = None,
) -> Tuple[Any, _BackendState]:
    """Bind the server, run :meth:`setup`, then return the running server.

    Binding happens *before* :meth:`setup` so ``GET /healthz`` answers ``503``
    while the model loads. Returns the bound, already-serving server and its
    state so callers (and tests) can drive shutdown: a
    :class:`ThreadingHTTPServer` on the stdlib transport, or an object with
    the same ``server_address``, ``shutdown()`` and ``server_close()`` on the
    native one. Raises :class:`~inferencekey.errors.BackendSetupError` if
    :meth:`setup` raises, or if ``IK_BACKEND_HTTP`` asks for a transport that
    is unavailable. ``transport`` (private) is an already-selected transport.
    """
    if transport is None:
        transport = _select_transport()
    state = _BackendState(backend)
    httpd: Any
    if transport.kind == "native":
        httpd = _start_native(state, port, transport)
    else:
        httpd = _HTTPServer((HOST, port), _make_handler(state))
        serve_thread = threading.Thread(
            target=httpd.serve_forever, name="ik-backend-http", daemon=True
        )
        serve_thread.start()
    _log_err(_transport_line(transport, httpd))

    ctx = BackendContext(config=config, port=httpd.server_address[1])
    try:
        backend.setup(ctx)
    except Exception as exc:  # noqa: BLE001 — wrap into the contract error
        httpd.shutdown()
        httpd.server_close()
        raise BackendSetupError(str(exc)) from exc

    if transport.kind == "native":
        # In case setup() changed what manifest() returns.
        _set_native_manifest(httpd._server, state)
    state.mark_ready()
    # Exactly one 'model loaded' line, per the contract.
    _log("model loaded")
    return httpd, state


def serve_backend(
    backend_or_class: Union[CustomBackend, type],
    port: int,
    config: Optional[Dict[str, Any]] = None,
    *,
    block: bool = True,
) -> Tuple[Any, _BackendState]:
    """Start a backend from code — the ergonomic alternative to the CLI.

    Accepts either a :class:`CustomBackend` instance or its class (instantiated
    with no arguments, matching the CLI's contract). Reuses the internal
    :func:`serve` for binding, ``setup()`` and the single ``model loaded`` log,
    so behaviour is identical to ``python -m inferencekey.backend.serve``.

    By default it blocks until ``SIGTERM``/``SIGINT`` (handy in a script).
    Pass ``block=False`` to get the running server back immediately (handy in a
    REPL or test); call ``httpd.shutdown(); httpd.server_close()`` to stop it.
    """
    if isinstance(backend_or_class, CustomBackend):
        backend = backend_or_class
    elif isinstance(backend_or_class, type) and issubclass(
        backend_or_class, CustomBackend
    ):
        backend = backend_or_class()
    else:
        raise BackendEntrypointError(
            "serve_backend expects a CustomBackend instance or subclass, got "
            f"{backend_or_class!r}"
        )

    httpd, state = serve(backend, port, config or {})
    if not block:
        return httpd, state

    _log(f"serving on http://{HOST}:{httpd.server_address[1]}")
    _run_until_signalled(httpd)
    return httpd, state


def _run_until_signalled(httpd: Any) -> None:
    """Block until SIGTERM/SIGINT, then shut the server down cleanly."""
    stop = threading.Event()

    def _on_signal(signum: int, _frame: Any) -> None:
        _log(f"received signal {signum}, shutting down")
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    try:
        stop.wait()
    finally:
        httpd.shutdown()
        httpd.server_close()


def main(argv: Optional[list] = None) -> int:
    """CLI entrypoint. Returns a process exit code (0 ok, non-zero on failure)."""
    args = _build_parser().parse_args(argv)

    try:
        entrypoint = _resolve_entrypoint(args)
        config = _parse_config(args)
        # Before loading the backend: an unavailable transport is a
        # configuration error (exit 2), not a setup failure.
        transport = _select_transport()
        backend = load_backend(entrypoint)
    except (BackendEntrypointError, BackendSetupError) as exc:
        _log_err(f"backend startup failed: {exc}")
        return 2

    try:
        httpd, _state = serve(backend, args.port, config, transport=transport)
    except BackendSetupError as exc:
        # setup() failed: traceback already chained; surface it to stderr and
        # exit non-zero so the supervisor knows the backend never came up.
        _log_err(f"setup() failed: {exc}")
        traceback.print_exc(file=sys.stderr)
        return 1

    _log(f"serving on http://{HOST}:{httpd.server_address[1]}")
    _run_until_signalled(httpd)
    return 0


if __name__ == "__main__":
    sys.exit(main())

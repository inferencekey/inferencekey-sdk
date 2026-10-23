"""Runtime: serve a :class:`~inferencekey.backend.CustomBackend` over HTTP.

Long-lived, single-process server bound to ``127.0.0.1`` (loopback only, never
``0.0.0.0``). Built on the standard library :mod:`http.server` so the SDK
runtime pulls in no HTTP framework and **no** ``torch`` — ``torch`` belongs to
the developer's backend.

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
  ``data: [DONE]``). Only for backends that implement ``chat()`` (``api`` is
  ``"openai-chat"`` in ``/meta``); the others answer ``501``. Errors use the
  OpenAI shape ``{"error": {"message", "type"}}``.
* ``GET /v1/models``: the one model this backend serves, OpenAI-shaped.

To start a backend from code instead of the CLI, use :func:`serve_backend`.

Lifecycle: :meth:`setup` is called once before traffic. If it raises, the
traceback goes to stderr and the process exits non-zero (the worker supervisor
sees it). On success exactly one ``model loaded`` line is logged. ``SIGTERM``
(and ``SIGINT``) shut the server down cleanly.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, Optional, Tuple, Union

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

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def mark_ready(self) -> None:
        self._ready.set()

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


def _make_handler(state: _BackendState) -> type:
    """Build a request handler bound to ``state`` (avoids global mutable state)."""

    class _Handler(BaseHTTPRequestHandler):
        # Quieten the default per-request stderr logging; we log deliberately.
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                if state.ready:
                    self._send_json(200, {"status": "ok"})
                else:
                    self._send_json(503, {"status": "loading"})
                return
            if path == "/meta":
                # Static metadata; available regardless of readiness so an
                # operator can introspect the backend while the model loads.
                self._send_json(200, state.manifest())
                return
            if path == "/v1/models":
                model_id = state.manifest().get("name", "")
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": model_id, "object": "model", "owned_by": "custom"}],
                    },
                )
                return
            self._send_json(404, {"error": "not found"})

        def _read_json_body(self) -> "Optional[Dict[str, Any]]":
            """Read and decode the request body, or send a 4xx and return ``None``.

            Enforces the same limits for every POST endpoint: a valid
            ``Content-Length`` within :data:`_MAX_BODY_BYTES` and a well-formed
            JSON body. On any violation it sends the error response itself and
            returns ``None`` so the caller just bails.
            """
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length"})
                return None
            if length < 0 or length > _MAX_BODY_BYTES:
                self._send_json(400, {"error": "request body too large"})
                return None

            raw = self.rfile.read(length) if length else b""
            try:
                return json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": f"invalid JSON body: {exc}"})
                return None

        def do_POST(self) -> None:  # noqa: N802 — stdlib API
            path = self.path.split("?", 1)[0]
            if path == "/process":
                self._handle_process()
            elif path == "/forecast":
                self._handle_forecast()
            elif path == "/v1/chat/completions":
                self._handle_chat()
            else:
                self._send_json(404, {"error": "not found"})

        def _handle_process(self) -> None:
            if not state.ready:
                self._send_json(503, {"error": "backend not ready"})
                return

            data = self._read_json_body()
            if data is None:
                return

            try:
                job = Job.from_wire(data)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            try:
                result = state.process(job)
            except InvalidInput as exc:
                # The client's input was malformed (e.g. a transcription request
                # with no audio). That is a 4xx — the caller's fault, not the
                # model's — so return 400, not the misleading 500 a generic
                # failure gets. Caught BEFORE `except Exception` on purpose.
                self._send_json(400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001 — keep the server alive
                # Resilience: log the traceback for operators, return a clean
                # 500 to the caller, and stay alive for the next job. A bare
                # ValueError from the user's model is a real bug (500), NOT a
                # client error — only InvalidInput (above) is reclassified.
                _log_err(f"process failed for job {job.id}:")
                traceback.print_exc(file=sys.stderr)
                self._send_json(500, {"error": str(exc)})
                return

            if not isinstance(result, Result):
                self._send_json(
                    500,
                    {
                        "error": "process() must return a Result; wrap your "
                        "value as Result(output=...)"
                    },
                )
                return
            self._send_json(200, result.to_wire())

        def _handle_forecast(self) -> None:
            if not state.ready:
                self._send_json(503, {"error": "backend not ready"})
                return

            data = self._read_json_body()
            if data is None:
                return

            try:
                request = ForecastRequest.from_wire(data)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            try:
                result = state.forecast(request)
            except NotImplementedError:
                # The backend does not implement forecast() — a distinct,
                # expected condition (501), not a per-request failure (500).
                self._send_json(
                    501, {"error": "backend does not support forecast"}
                )
                return
            except InvalidInput as exc:
                # Malformed client input surfaced from an input accessor → 400,
                # caught BEFORE the generic handler; a bare ValueError from the
                # user's model stays a real 500 below.
                self._send_json(400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001 — keep the server alive
                _log_err(f"forecast failed for request {request.id}:")
                traceback.print_exc(file=sys.stderr)
                self._send_json(500, {"error": str(exc)})
                return

            if not isinstance(result, ForecastResult):
                self._send_json(
                    500,
                    {
                        "error": "forecast() must return a ForecastResult; wrap "
                        "your value as ForecastResult(median=...)"
                    },
                )
                return
            self._send_json(200, result.to_wire())

        def _send_openai_error(self, status: int, message: str, kind: str) -> None:
            self._send_json(status, {"error": {"message": message, "type": kind}})

        def _handle_chat(self) -> None:
            if not state.ready:
                self._send_openai_error(503, "backend not ready", "server_error")
                return

            data = self._read_json_body()
            if data is None:
                return
            if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
                self._send_openai_error(
                    400, "request must be a JSON object with a 'messages' list",
                    "invalid_request_error",
                )
                return

            if data.get("stream"):
                self._stream_chat(data)
                return

            try:
                completion = state.chat(data)
            except NotImplementedError:
                self._send_openai_error(
                    501, "backend does not support chat", "invalid_request_error"
                )
                return
            except InvalidInput as exc:
                self._send_openai_error(400, str(exc), "invalid_request_error")
                return
            except Exception as exc:  # noqa: BLE001 (keep the server alive)
                _log_err("chat failed:")
                traceback.print_exc(file=sys.stderr)
                self._send_openai_error(500, str(exc), "server_error")
                return
            if not isinstance(completion, dict):
                self._send_openai_error(
                    500,
                    "chat() must return a chat.completion dict; build it with "
                    "inferencekey.backend.chat_completion(...)",
                    "server_error",
                )
                return
            self._send_json(200, completion)

        def _stream_chat(self, data: Dict[str, Any]) -> None:
            """Answer a streaming chat request as server-sent events.

            The first chunk is pulled before the headers go out, so a backend
            that refuses the request (not implemented, invalid input) still gets
            a proper status code instead of a 200 stream carrying an error.
            """
            try:
                chunks = iter(state.chat_stream(data))
                first = next(chunks, None)
            except NotImplementedError:
                self._send_openai_error(
                    501, "backend does not support chat", "invalid_request_error"
                )
                return
            except InvalidInput as exc:
                self._send_openai_error(400, str(exc), "invalid_request_error")
                return
            except Exception as exc:  # noqa: BLE001 (keep the server alive)
                _log_err("chat stream failed before the first chunk:")
                traceback.print_exc(file=sys.stderr)
                self._send_openai_error(500, str(exc), "server_error")
                return

            # No Content-Length: the response ends when the connection closes
            # (HTTP/1.0 semantics of BaseHTTPRequestHandler).
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def emit(payload: Any) -> None:
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()

            try:
                if first is not None:
                    emit(json.dumps(first))
                for chunk in chunks:
                    emit(json.dumps(chunk))
            except (BrokenPipeError, ConnectionResetError):
                # The client went away; nothing left to tell it. Closing the
                # generator lets the backend cancel the generation now (its
                # ``finally`` runs) instead of whenever it is garbage collected.
                _close_quietly(chunks)
                return
            except Exception as exc:  # noqa: BLE001 (keep the server alive)
                _log_err("chat stream failed mid-way:")
                traceback.print_exc(file=sys.stderr)
                try:
                    emit(json.dumps({"error": {"message": str(exc), "type": "server_error"}}))
                except (BrokenPipeError, ConnectionResetError):
                    return
            try:
                emit("[DONE]")
            except (BrokenPipeError, ConnectionResetError):
                return

    return _Handler


def _close_quietly(chunks: Iterator[Dict[str, Any]]) -> None:
    """Close a backend's chunk generator, logging (not raising) what its cleanup raises."""
    close = getattr(chunks, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:  # noqa: BLE001 (keep the server alive)
        _log_err("chat stream cleanup failed:")
        traceback.print_exc(file=sys.stderr)


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
    backend: CustomBackend, port: int, config: Dict[str, Any]
) -> Tuple[ThreadingHTTPServer, _BackendState]:
    """Bind the server, run :meth:`setup`, then return the running server.

    Binding happens *before* :meth:`setup` so ``GET /healthz`` answers ``503``
    while the model loads. Returns the bound, already-serving server and its
    state so callers (and tests) can drive shutdown. Raises
    :class:`~inferencekey.errors.BackendSetupError` if :meth:`setup` raises.
    """
    state = _BackendState(backend)
    handler_cls = _make_handler(state)
    httpd = _HTTPServer((HOST, port), handler_cls)

    serve_thread = threading.Thread(
        target=httpd.serve_forever, name="ik-backend-http", daemon=True
    )
    serve_thread.start()

    ctx = BackendContext(config=config, port=httpd.server_address[1])
    try:
        backend.setup(ctx)
    except Exception as exc:  # noqa: BLE001 — wrap into the contract error
        httpd.shutdown()
        httpd.server_close()
        raise BackendSetupError(str(exc)) from exc

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
) -> Tuple[ThreadingHTTPServer, _BackendState]:
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


def _run_until_signalled(httpd: ThreadingHTTPServer) -> None:
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
        backend = load_backend(entrypoint)
    except (BackendEntrypointError, BackendSetupError) as exc:
        _log_err(f"backend startup failed: {exc}")
        return 2

    try:
        httpd, _state = serve(backend, args.port, config)
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

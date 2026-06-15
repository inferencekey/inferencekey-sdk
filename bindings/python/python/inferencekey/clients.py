"""Ergonomic clients over the native ``_inferencekey.Client``.

Two planes, two tokens, least privilege:

* :class:`ManagementClient` (``ik_sdk_``) provisions; it has no inference.
* :class:`DataClient` (``ik_live_`` per workload) calls inference; it cannot
  provision.

All JSON marshalling and error remapping happens here so the native layer stays
minimal and the public surface is typed.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Callable, Iterator, List, Optional, Union

from . import _inferencekey  # native extension
from .enums import OnDrift
from .errors import (
    ApiError,
    AuthError,
    ConfigurationError,
    InferenceKeyError,
    PermissionDenied,
    ValidationError,
)
from .types import EmbedResult, EndpointRef, ReadinessEvent, TextChunk, TextResult, WorkloadSpec

_DEFAULT_BASE_URL = "https://api.inferencekey.com"


def _resolve(explicit: Optional[str], env: str) -> Optional[str]:
    """Explicit value wins over the environment variable."""
    if explicit:
        return explicit
    value = os.environ.get(env)
    return value or None


def _call(fn, *args):
    """Invoke a native method, remapping its builtin exceptions to SDK ones."""
    try:
        return fn(*args)
    except PermissionError as e:
        raise PermissionDenied(str(e)) from None
    except ValueError as e:
        raise _value_error(str(e)) from None
    except RuntimeError as e:
        raise ApiError(str(e)) from None


def _value_error(message: str) -> Exception:
    """Disambiguate auth vs config vs validation from the message prefix."""
    if message.startswith("authentication failed"):
        return AuthError(message)
    if message.startswith("configuration error"):
        return ConfigurationError(message)
    return ValidationError(message)


class ManagementClient:
    """Control-plane client (``ik_sdk_`` token), scoped to one project."""

    def __init__(self, *, base_url: str, sdk_token: str, project: Optional[str] = None) -> None:
        if not sdk_token:
            raise ConfigurationError("ManagementClient requires an ik_sdk_ token.")
        self._native = _inferencekey.Client(base_url)
        self._sdk_token = sdk_token
        self._project = project

    @classmethod
    def from_env(
        cls,
        *,
        base_url: Optional[str] = None,
        project: Optional[str] = None,
        sdk_token: Optional[str] = None,
    ) -> "ManagementClient":
        """Resolve config (explicit > env) and construct."""
        return cls(
            base_url=_resolve(base_url, "INFERENCEKEY_BASE_URL") or _DEFAULT_BASE_URL,
            project=_resolve(project, "INFERENCEKEY_PROJECT"),
            sdk_token=_resolve(sdk_token, "INFERENCEKEY_SDK_TOKEN") or "",
        )

    def ensure(
        self,
        spec: WorkloadSpec,
        *,
        on_drift: Union[OnDrift, str] = OnDrift.RECONCILE,
        project: Optional[str] = None,
    ) -> EndpointRef:
        """Idempotently provision/reconcile ``spec``; returns an :class:`EndpointRef`."""
        project_id = project or spec.project or self._project
        if not project_id:
            raise ConfigurationError(
                "No project configured. Set INFERENCEKEY_PROJECT, pass project=, or set spec.project."
            )
        policy = on_drift.value if hasattr(on_drift, "value") else on_drift
        raw = _call(
            self._native.ensure,
            self._sdk_token,
            project_id,
            json.dumps(spec.to_wire()),
            policy,
        )
        data = json.loads(raw)
        return EndpointRef(project_slug=data["project_slug"], workload_slug=data["workload_slug"])

    def delete(self, workload_slug: str, *, project: Optional[str] = None) -> bool:
        """Delete a workload by slug. Returns ``True`` if it existed, ``False`` if
        it was already gone — idempotent, so it's safe to call on cleanup/exit
        without checking first.

        Cloud GPUs the autoscaler provisioned for the workload are torn down on
        the platform (private workers are only unassigned), so deleting also
        stops the billable capacity. Uses the ``ik_sdk_`` control token::

            import signal
            signal.signal(signal.SIGINT, lambda *_: (mgmt.delete(slug), sys.exit(0)))
        """
        project_id = project or self._project
        if not project_id:
            raise ConfigurationError(
                "No project configured. Set INFERENCEKEY_PROJECT or pass project=."
            )
        return _call(self._native.delete, self._sdk_token, project_id, workload_slug)

    def wait_until_ready(
        self,
        workload_slug: str,
        *,
        project: Optional[str] = None,
        timeout: float = 600.0,
        on_progress: Optional[Callable[[ReadinessEvent], None]] = None,
        silent: bool = False,
    ) -> None:
        """Wait until ``workload_slug`` is serving, reporting progress as the
        platform schedules a worker, provisions a cloud GPU, and boots the runtime.

        Returns when the platform reports the ``ready`` phase; raises on an
        ``error`` phase or after ``timeout`` seconds. By default it prints a live
        progress view to the terminal; pass your own ``on_progress`` to handle
        events yourself, or ``silent=True`` to suppress output. Progress is
        streamed over the ``ik_sdk_`` control token, so it lives here on the
        management client (no data key needed)::

            ref = mgmt.ensure(spec)
            mgmt.wait_until_ready(ref.workload_slug, timeout=600)
        """
        project_id = project or self._project
        if not project_id:
            raise ConfigurationError(
                "No project configured. Set INFERENCEKEY_PROJECT or pass project=."
            )
        # The built-in view animates a clock between events via a background
        # ticker; a caller's on_progress and the silent path get events only.
        renderer = None if (on_progress or silent) else _make_progress_renderer()
        on_event = on_progress or _noop_progress

        native_iter = _call(
            self._native.readiness_events,
            self._sdk_token,
            project_id,
            workload_slug,
        )
        # Advance the on-screen seconds counter ~1×/s between server events so a
        # long cold start doesn't look frozen while the main thread blocks on
        # next(). Keyed off a local clock (the server only restamps elapsed per
        # event). A daemon thread so it never keeps the process alive.
        started = time.monotonic()
        ticker_stop = threading.Event()

        def _run_ticker() -> None:
            while not ticker_stop.wait(1.0):
                renderer.tick(round((time.monotonic() - started) * 1000))

        ticker: Optional[threading.Thread] = None
        if renderer is not None:
            ticker = threading.Thread(target=_run_ticker, name="ik-readiness-ticker", daemon=True)
            ticker.start()

        try:
            deadline = started + timeout
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f'workload "{workload_slug}" not ready after {timeout:.0f}s'
                    )
                try:
                    raw = next(native_iter)
                except StopIteration:
                    return  # stream ended without an explicit ready
                except PermissionError as e:
                    raise PermissionDenied(str(e)) from None
                except ValueError as e:
                    raise _value_error(str(e)) from None
                except RuntimeError as e:
                    raise ApiError(str(e)) from None
                data = json.loads(raw)
                event = ReadinessEvent(
                    phase=data["phase"],
                    message=data["message"],
                    elapsed_ms=data.get("elapsed_ms", 0),
                    step=data.get("step"),
                )
                # The built-in renderer shares the ticker's local clock so the
                # counter never jumps; a caller's on_progress gets the event verbatim.
                if renderer is not None:
                    renderer.render(event, round((time.monotonic() - started) * 1000))
                else:
                    on_event(event)
                if event.phase == "ready":
                    return
                if event.phase == "error":
                    raise ApiError(
                        f'workload "{workload_slug}" failed to become ready: {event.message}'
                    )
        finally:
            ticker_stop.set()
            if ticker is not None:
                ticker.join(timeout=2.0)


class DataClient:
    """Data-plane client. Derive an :class:`Endpoint` per workload, each with its
    own ``ik_live_`` key — so one app can drive several workloads with different keys."""

    def __init__(self, *, base_url: str, project: str, api_key: Optional[str] = None) -> None:
        if not project:
            raise ConfigurationError("DataClient requires a project slug.")
        self._native = _inferencekey.Client(base_url)
        self._project = project
        self._default_key = api_key

    @classmethod
    def from_env(
        cls,
        *,
        base_url: Optional[str] = None,
        project: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "DataClient":
        return cls(
            base_url=_resolve(base_url, "INFERENCEKEY_BASE_URL") or _DEFAULT_BASE_URL,
            project=_resolve(project, "INFERENCEKEY_PROJECT") or "",
            api_key=_resolve(api_key, "INFERENCEKEY_API_KEY"),
        )

    def endpoint(self, workload_slug: str, *, api_key: Optional[str] = None) -> "Endpoint":
        """Bind an endpoint to ``workload_slug`` and its own ``ik_live_`` key."""
        key = api_key or self._default_key
        if not key:
            raise ConfigurationError(
                f'No ik_live_ key for workload "{workload_slug}". Pass api_key= or set INFERENCEKEY_API_KEY.'
            )
        return Endpoint(self._native, self._project, workload_slug, key)


class Endpoint:
    """A single workload's OpenAI-compatible endpoint, bound to one ``ik_live_`` key."""

    def __init__(self, native, project_slug: str, workload_slug: str, api_key: str) -> None:
        self._native = native
        self._project_slug = project_slug
        self.workload_slug = workload_slug
        self._api_key = api_key

    def generate_text(
        self,
        *,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> TextResult:
        """Run a (non-streaming) chat completion."""
        params = _drop_none(
            {
                "prompt": prompt,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        raw = _call(
            self._native.generate_text,
            self._project_slug,
            self.workload_slug,
            self._api_key,
            json.dumps(params),
        )
        data = json.loads(raw)
        return TextResult(
            text=data["text"],
            model=data["model"],
            finish_reason=data.get("finish_reason"),
            raw=data.get("raw", {}),
        )

    def generate_text_stream(
        self,
        *,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[TextChunk]:
        """Run a streaming chat completion, yielding one :class:`TextChunk` per
        SSE frame as tokens are produced.

        The connection is opened eagerly (so auth/validation errors raise here,
        not mid-iteration); each chunk is then pulled lazily as you iterate::

            for chunk in ep.generate_text_stream(prompt="Hola"):
                print(chunk.text, end="", flush=True)
        """
        params = _drop_none(
            {
                "prompt": prompt,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        native_iter = _call(
            self._native.generate_text_stream,
            self._project_slug,
            self.workload_slug,
            self._api_key,
            json.dumps(params),
        )
        return self._iter_chunks(native_iter)

    @staticmethod
    def _iter_chunks(native_iter) -> Iterator[TextChunk]:
        """Adapt the native chunk-JSON iterator into typed ``TextChunk``s,
        remapping any native error raised mid-stream to an SDK exception."""
        while True:
            try:
                raw = next(native_iter)
            except StopIteration:
                return
            except PermissionError as e:
                raise PermissionDenied(str(e)) from None
            except ValueError as e:
                raise _value_error(str(e)) from None
            except RuntimeError as e:
                raise ApiError(str(e)) from None
            data = json.loads(raw)
            yield TextChunk(
                text=data["text"],
                finish_reason=data.get("finish_reason"),
                raw=data.get("raw", {}),
            )

    def embed(self, *, input: Union[str, List[str]]) -> EmbedResult:
        """Create embeddings for one or more inputs."""
        items = [input] if isinstance(input, str) else list(input)
        raw = _call(
            self._native.embed,
            self._project_slug,
            self.workload_slug,
            self._api_key,
            json.dumps({"input": items}),
        )
        data = json.loads(raw)
        return EmbedResult(
            embeddings=data["embeddings"],
            model=data["model"],
            raw=data.get("raw", {}),
        )


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


_READINESS_PHASES = ["scheduling", "provisioning", "bootstrapping", "ready"]


def _noop_progress(_event: ReadinessEvent) -> None:
    pass


class _ProgressRenderer:
    """Default terminal progress view for :meth:`ManagementClient.wait_until_ready`.

    ``render`` paints on each readiness event; ``tick`` repaints the same line
    with a freshly elapsed clock between events so the seconds counter keeps
    moving during long waits (cold starts can go minutes between server events).
    On a TTY it redraws a single line of phase dots; off a TTY (CI / piped) it
    prints one plain line per event and ``tick`` is a no-op (no clock to animate).
    """

    def __init__(self) -> None:
        self._is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        # The most recent event, retained so ``tick`` can repaint the same
        # phase/message with an updated clock. Guards stdout against the main
        # thread (``render``) and the ticker thread (``tick``) interleaving.
        self._last: Optional[ReadinessEvent] = None
        self._done = False
        self._lock = threading.Lock()

    def _paint(self, event: ReadinessEvent, elapsed_ms: int) -> None:
        secs = round(elapsed_ms / 1000)
        if event.phase == "error":
            reached = len(_READINESS_PHASES) - 1
        else:
            reached = _READINESS_PHASES.index(event.phase) if event.phase in _READINESS_PHASES else 0
        dots = []
        for i, _p in enumerate(_READINESS_PHASES):
            if event.phase == "ready" or i < reached:
                dots.append("\x1b[32m●\x1b[0m")      # done
            elif i == reached:
                dots.append("\x1b[36m◐\x1b[0m")      # in progress
            else:
                dots.append("\x1b[90m○\x1b[0m")      # pending
        label = f"\x1b[31m{event.message}\x1b[0m" if event.phase == "error" else event.message
        # \r returns to line start; \x1b[K clears to end of line before redraw.
        sys.stdout.write(f"\r\x1b[K{' '.join(dots)}  {label} ({secs}s)")
        if event.phase in ("ready", "error"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    def render(self, event: ReadinessEvent, elapsed_ms: Optional[int] = None) -> None:
        """Paint on a new event. ``elapsed_ms``, when given, overrides the
        server's ``event.elapsed_ms`` for the on-screen clock (TTY only) so it
        stays monotonic with ``tick``'s local clock."""
        if not self._is_tty:
            secs = round(event.elapsed_ms / 1000)
            step = f" [{event.step}]" if event.step else ""
            print(f"[{secs}s] {event.phase}{step}: {event.message}")
            return
        with self._lock:
            self._last = event
            self._paint(event, event.elapsed_ms if elapsed_ms is None else elapsed_ms)
            if event.phase in ("ready", "error"):
                self._done = True

    def tick(self, elapsed_ms: int) -> None:
        """Repaint the last phase/message with an updated clock (TTY only)."""
        if not self._is_tty:
            return
        with self._lock:
            if self._done or self._last is None:
                return
            self._paint(self._last, elapsed_ms)


def _make_progress_renderer() -> _ProgressRenderer:
    return _ProgressRenderer()

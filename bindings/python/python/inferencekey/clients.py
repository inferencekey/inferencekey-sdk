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
import time
from typing import Iterator, List, Optional, Union

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
from .types import EmbedResult, EndpointRef, TextChunk, TextResult, WorkloadSpec

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

    def wait_until_ready(
        self,
        *,
        timeout: float = 300.0,
        interval: float = 2.0,
        max_interval: float = 15.0,
    ) -> None:
        """Block until the workload's endpoint is serving, by probing it.

        The control plane has no readiness flag, so this sends a tiny chat probe
        and treats a transport/5xx failure as "still starting" — retrying with
        exponential backoff up to ``max_interval`` — while a 401/403/404 (bad
        key, wrong token type, unknown workload) is a real error that raises
        immediately. Raises :class:`TimeoutError` if ``timeout`` elapses first.

        Call this after ``ensure()`` provisions a cold worker, before streaming::

            ep.wait_until_ready(timeout=600)
            for chunk in ep.generate_text_stream(prompt="..."):
                ...
        """
        deadline = time.monotonic() + timeout
        delay = interval
        while True:
            try:
                self.generate_text(prompt="ping", max_tokens=1)
                return  # a successful completion means the worker is up.
            except (PermissionDenied, AuthError, ValidationError) as e:
                # Fatal: a wrong/instufficient credential or unknown workload
                # will never become ready — surface it, do not spin.
                raise e
            except (ApiError, InferenceKeyError):
                # Transport error or 5xx: the worker is likely still booting.
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f'workload "{self.workload_slug}" not ready after {timeout:.0f}s'
                    ) from None
                time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
                delay = min(delay * 2, max_interval)

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

"""The custom-backend contract.

A *custom backend* is the unit of work a developer ships to run their own
PyTorch inference: load a model **once**, then process many jobs against it.
This module is the contract only — three serializable dataclasses and one
abstract base class — and it deliberately imports **neither** a HTTP framework
**nor** ``torch``. The HTTP runtime lives in :mod:`inferencekey.backend.serve`;
``torch`` is a dependency of the developer's backend, never of the SDK.

A backend is two methods:

* :meth:`CustomBackend.setup` — called once at startup with a
  :class:`BackendContext`; instantiate the ``nn.Module`` here and stash it on
  ``self``.
* :meth:`CustomBackend.process` — called per :class:`Job`; reuse the model
  loaded in :meth:`setup` and return a :class:`Result`.

Example::

    import torch
    from inferencekey.backend import BackendContext, CustomBackend, Job, Result

    class EchoBackend(CustomBackend):
        def setup(self, ctx: BackendContext) -> None:
            self.device = ctx.config.get("device", "cpu")
            self.model = torch.nn.Identity().to(self.device)

        def process(self, job: Job) -> Result:
            x = torch.tensor(job.input["values"], device=self.device)
            return Result(output={"values": self.model(x).tolist()})
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


#: The task types the product supports. ``text2text`` is the default; a backend
#: declares which one it implements via :attr:`CustomBackend.task_type`.
TASK_TYPES = (
    "text2text",
    "embedding",
    "text2image",
    "text2audio",
    "audio2text",
    "reranker",
    "classification",
    "reward",
    "forecast",
)

#: Default quantile levels a forecast reports when the request omits them: a
#: symmetric 80% interval around the median.
DEFAULT_QUANTILE_LEVELS = (0.1, 0.5, 0.9)


@dataclass
class BackendManifest:
    """Declarative metadata describing a backend.

    A minimal descriptor the runtime can expose (via ``GET /meta``) without
    importing the backend's heavy deps. ``requirements`` is a *reference* to the
    backend's ``requirements.txt`` (a path, by convention), not its contents —
    packaging and upload are future tasks, out of scope here.
    """

    name: str = ""
    version: str = ""
    task_type: str = ""
    requirements: str = ""
    #: PEP 440 version specifier the backend's deps need (e.g. ``">=3.9,<3.12"``
    #: or an exact ``"3.11"``). Empty means no constraint. The worker feeds this
    #: to ``uv`` so the venv gets the pinned Python regardless of the node's.
    requires_python: str = ""

    def to_wire(self) -> Dict[str, Any]:
        """Render to the JSON-serializable dict returned by ``GET /meta``."""
        return {
            "name": self.name,
            "version": self.version,
            "task_type": self.task_type,
            "requirements": self.requirements,
            "requires_python": self.requires_python,
        }


@dataclass
class Job:
    """One unit of work handed to :meth:`CustomBackend.process`.

    ``id`` correlates the job across logs and responses; ``input`` is a free,
    JSON-serializable dict whose shape the backend defines.

    **Job/Result are free dicts** — this contract does not fix a schema; the
    worker fills ``input`` and reads ``output`` in a future task. What follows is
    the *agreed mapping* for the product's dominant case, **text** (``text2text``
    / ``classification``); respect it, do not redefine ``Job``/``Result``:

    * A text-generation job arrives as either ::

          {"id": "j1", "input": {"prompt": "Translate to French: hello"}}

      or the chat shape ::

          {"id": "j1", "input": {"messages": [{"role": "user", "content": "..."}]}}

      and the backend returns ::

          {"output": {"text": "bonjour"}}

    * A classification job returns a label (and optionally a score) ::

          {"output": {"label": "positive", "score": 0.98}}

    The backend owns the exact keys; these are the conventions the rest of the
    product builds on.
    """

    id: str
    input: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, data: Dict[str, Any]) -> "Job":
        """Build a :class:`Job` from a decoded ``/process`` request body.

        Raises :class:`ValueError` if required fields are missing or mistyped;
        the runtime maps that to a 400 response.
        """
        if not isinstance(data, dict):
            raise ValueError("job must be a JSON object")
        job_id = data.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job 'id' must be a non-empty string")
        job_input = data.get("input", {})
        if not isinstance(job_input, dict):
            raise ValueError("job 'input' must be a JSON object")
        return cls(id=job_id, input=job_input)

    def to_wire(self) -> Dict[str, Any]:
        """Render to the JSON-serializable dict for transport."""
        return {"id": self.id, "input": self.input}

    # --- Typed accessors -------------------------------------------------
    #
    # ``input`` stays a free dict (the contract is deliberately schema-less),
    # but the *same job* reaches ``process()`` with different envelopes
    # depending on the caller — a raw ``/process`` curl vs. the platform's
    # OpenAI-compatible data plane. These accessors hide that difference so a
    # backend reads one shape and works from both, instead of hard-coding a
    # single key and 500-ing when the other caller shows up (which is exactly
    # how the audio2text example first broke).

    def text(self) -> str:
        """Return the prompt text for a text2text/classification job.

        Accepts both the ``{"prompt": ...}`` shape and the chat shape
        ``{"messages": [{"role", "content"}, ...]}`` (last user message wins).
        Raises :class:`ValueError` if neither is present.
        """
        prompt = self.input.get("prompt")
        if isinstance(prompt, str) and prompt:
            return prompt
        for msg in reversed(self.messages()):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
        # Fall back to the last message with string content, whatever its role.
        for msg in reversed(self.messages()):
            if isinstance(msg.get("content"), str) and msg["content"]:
                return msg["content"]
        raise ValueError(
            "job input has no text: expected 'prompt' or 'messages' with content"
        )

    def messages(self) -> "list[Dict[str, Any]]":
        """Return the chat ``messages`` list (empty if the job carries none)."""
        msgs = self.input.get("messages")
        return msgs if isinstance(msgs, list) else []

    def audio_bytes(self) -> bytes:
        """Return the raw audio bytes for an audio2text job, from EITHER shape.

        * Native ``/process``: ``input.audio_b64`` (base64 of the audio file).
        * Public API (``POST .../v1/audio/transcriptions``): the manager forwards
          the OpenAI multipart body base64-encoded in ``input.raw_body_b64``
          (with ``input.raw_body_content_type`` carrying the boundary); we parse
          the multipart and return the ``file`` part's bytes.

        Raises :class:`ValueError` if no audio can be found or decoded — mapped
        to a clean error by the runtime.
        """
        import base64
        import binascii

        b64 = self.input.get("audio_b64")
        if isinstance(b64, str) and b64:
            try:
                return base64.b64decode(b64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"input.audio_b64 is not valid base64: {exc}")

        raw_b64 = self.input.get("raw_body_b64")
        content_type = self.input.get("raw_body_content_type") or ""
        if isinstance(raw_b64, str) and raw_b64 and content_type.startswith("multipart/"):
            try:
                raw = base64.b64decode(raw_b64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"input.raw_body_b64 is not valid base64: {exc}")
            data = _multipart_file_bytes(raw, content_type)
            if data:
                return data
            raise ValueError(
                "multipart body carried no file part; send audio as the 'file' field"
            )

        raise ValueError(
            "job input carries no audio: expected 'audio_b64' (native) or a "
            "multipart body in 'raw_body_b64' (public /v1/audio/transcriptions)"
        )


def _multipart_file_bytes(raw: bytes, content_type: str) -> "bytes | None":
    """First file part's bytes from a multipart/form-data body (stdlib parser).

    Prefers the part named ``file`` (OpenAI's transcription field); falls back to
    the first part with a filename or an ``audio/*`` content type.
    """
    import email

    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    msg = email.message_from_bytes(header + raw)
    if not msg.is_multipart():
        return None
    fallback = None
    for part in msg.walk():
        if part.is_multipart():
            continue
        disp = part.get("Content-Disposition", "") or ""
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        if 'name="file"' in disp or "name=file" in disp:
            return payload
        if fallback is None and (
            "filename=" in disp or (part.get_content_type() or "").startswith("audio/")
        ):
            fallback = payload
    return fallback


@dataclass
class Result:
    """The outcome of :meth:`CustomBackend.process`.

    ``output`` is a free, JSON-serializable dict — this contract is **not**
    OpenAI-compatible; the backend owns its own schema.
    """

    output: Dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> Dict[str, Any]:
        """Render to the JSON-serializable dict returned by ``/process``."""
        return {"output": self.output}


def _as_float_list(value: Any, field_name: str) -> List[float]:
    """Coerce a JSON array of numbers to ``List[float]`` or raise ``ValueError``.

    Accepts ``int`` and ``float`` (JSON has no int/float distinction), rejects
    ``bool`` (which is an ``int`` subclass in Python) and anything non-numeric,
    with a message naming ``field_name`` so the 400 says *what* is wrong.
    """
    if not isinstance(value, list):
        raise ValueError(f"'{field_name}' must be a JSON array of numbers")
    out: List[float] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(
                f"'{field_name}[{i}]' must be a number, got {type(item).__name__}"
            )
        out.append(float(item))
    return out


@dataclass
class ForecastRequest:
    """One time-series forecast request handed to :meth:`CustomBackend.forecast`.

    A single series per request (batching multiple series is out of scope for
    v1). ``id`` correlates the request across logs and responses, mirroring
    :class:`Job`.

    Fields:

    * ``target`` — the observed history of the series (non-empty list of
      numbers).
    * ``horizon`` — how many future steps to predict (``int > 0``).
    * ``timestamps`` — optional ISO8601 timestamps for the history; when given
      the backend may echo forward-dated timestamps in the result.
    * ``freq`` — optional pandas-style frequency string for the series (e.g.
      ``"D"``, ``"H"``, ``"5min"``, ``"W"``, ``"M"``). Foundation time-series
      models (Chronos-2, TimesFM, Moirai) use it to (a) generate the output
      forecast window's timestamps and (b) feed the frequency to the model. The
      SDK does not validate the exact pandas syntax — any non-empty string is
      accepted and the backend interprets it.
    * ``quantile_levels`` — quantiles to report, each in the open interval
      ``(0, 1)``; defaults to ``[0.1, 0.5, 0.9]``.
    * ``past_covariates`` — optional named series aligned to the history; each
      list must have ``len(target)`` values.
    * ``future_covariates`` — optional named series known for the future
      window; each list must have ``horizon`` values.

    Minimal valid request::

        {"id": "f1", "target": [1, 2, 3, 4, 5], "horizon": 3}
    """

    id: str
    target: List[float]
    horizon: int
    timestamps: Optional[List[str]] = None
    freq: Optional[str] = None
    quantile_levels: List[float] = field(
        default_factory=lambda: list(DEFAULT_QUANTILE_LEVELS)
    )
    past_covariates: Optional[Dict[str, List[float]]] = None
    future_covariates: Optional[Dict[str, List[float]]] = None

    @classmethod
    def from_wire(cls, data: Dict[str, Any]) -> "ForecastRequest":
        """Build a :class:`ForecastRequest` from a decoded ``/forecast`` body.

        Raises :class:`ValueError` (mapped to a ``400`` by the runtime) with a
        message that says *what* is missing or malformed — never a traceback.
        """
        if not isinstance(data, dict):
            raise ValueError("forecast request must be a JSON object")

        req_id = data.get("id")
        if not isinstance(req_id, str) or not req_id:
            raise ValueError("forecast 'id' must be a non-empty string")

        if "target" not in data:
            raise ValueError("forecast 'target' is required (the series history)")
        target = _as_float_list(data.get("target"), "target")
        if not target:
            raise ValueError("forecast 'target' must be a non-empty list of numbers")

        if "horizon" not in data:
            raise ValueError("forecast 'horizon' is required (steps to predict)")
        horizon = data.get("horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise ValueError("forecast 'horizon' must be an integer")
        if horizon <= 0:
            raise ValueError(f"forecast 'horizon' must be > 0, got {horizon}")

        timestamps = data.get("timestamps")
        if timestamps is not None:
            if not isinstance(timestamps, list) or not all(
                isinstance(t, str) for t in timestamps
            ):
                raise ValueError("forecast 'timestamps' must be a list of strings")
            if len(timestamps) != len(target):
                raise ValueError(
                    f"forecast 'timestamps' has {len(timestamps)} values but "
                    f"target has {len(target)}"
                )

        freq = data.get("freq")
        if freq is not None and (not isinstance(freq, str) or not freq):
            raise ValueError("forecast 'freq' must be a non-empty string")

        raw_levels = data.get("quantile_levels")
        if raw_levels is None:
            quantile_levels = list(DEFAULT_QUANTILE_LEVELS)
        else:
            quantile_levels = _as_float_list(raw_levels, "quantile_levels")
            for q in quantile_levels:
                if not (0.0 < q < 1.0):
                    raise ValueError(
                        f"forecast 'quantile_levels' must each be in (0, 1), got {q}"
                    )

        past_covariates = cls._validate_covariates(
            data.get("past_covariates"),
            "past_covariate",
            len(target),
            "target length",
        )
        future_covariates = cls._validate_covariates(
            data.get("future_covariates"),
            "future_covariate",
            horizon,
            "horizon",
        )

        return cls(
            id=req_id,
            target=target,
            horizon=horizon,
            timestamps=timestamps,
            freq=freq,
            quantile_levels=quantile_levels,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

    @staticmethod
    def _validate_covariates(
        raw: Any, label: str, expected_len: int, expected_desc: str
    ) -> Optional[Dict[str, List[float]]]:
        """Validate a covariates map: dict of name → list of ``expected_len`` numbers.

        Returns ``None`` when ``raw`` is absent. ``label`` names the covariate in
        error messages (e.g. ``future_covariate 'temp' has 3 values but horizon
        is 24``).
        """
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError(f"forecast '{label}s' must be a JSON object")
        out: Dict[str, List[float]] = {}
        for name, series in raw.items():
            values = _as_float_list(series, f"{label} '{name}'")
            if len(values) != expected_len:
                raise ValueError(
                    f"{label} '{name}' has {len(values)} values but "
                    f"{expected_desc} is {expected_len}"
                )
            out[name] = values
        return out

    def to_wire(self) -> Dict[str, Any]:
        """Render to the JSON-serializable dict for transport.

        Omits optional fields that are ``None`` so the wire form of a minimal
        request stays minimal.
        """
        payload: Dict[str, Any] = {
            "id": self.id,
            "target": self.target,
            "horizon": self.horizon,
            "quantile_levels": self.quantile_levels,
        }
        if self.timestamps is not None:
            payload["timestamps"] = self.timestamps
        if self.freq is not None:
            payload["freq"] = self.freq
        if self.past_covariates is not None:
            payload["past_covariates"] = self.past_covariates
        if self.future_covariates is not None:
            payload["future_covariates"] = self.future_covariates
        return payload


@dataclass
class ForecastResult:
    """The outcome of :meth:`CustomBackend.forecast`.

    * ``median`` — the point forecast, one value per future step
      (``len == horizon``).
    * ``quantiles`` — optional map of quantile level (as a string key, e.g.
      ``"0.1"``) to a list of predicted values (``len == horizon``); the median
      quantile (``"0.5"``) need not be duplicated here.
    * ``timestamps`` — optional ISO8601 timestamps for the forecast window
      (``len == horizon``).

    Wire form::

        {"forecast": {"median": [...], "timestamps": [...],
                      "quantiles": {"0.1": [...], "0.9": [...]}}}
    """

    median: List[float] = field(default_factory=list)
    quantiles: Optional[Dict[str, List[float]]] = None
    timestamps: Optional[List[str]] = None

    def to_wire(self) -> Dict[str, Any]:
        """Render to the JSON-serializable dict returned by ``/forecast``.

        Optional fields (``timestamps``, ``quantiles``) are omitted when unset so
        a bare median forecast stays compact.
        """
        forecast: Dict[str, Any] = {"median": self.median}
        if self.timestamps is not None:
            forecast["timestamps"] = self.timestamps
        if self.quantiles is not None:
            forecast["quantiles"] = self.quantiles
        return {"forecast": forecast}


@dataclass
class BackendContext:
    """What the runtime passes to :meth:`CustomBackend.setup`.

    ``config`` is where the developer resolves model knobs such as
    ``model_name``, ``device`` (default ``"cpu"``; no GPU autodetection) and any
    weights path. ``port`` is the loopback port the runtime is serving on.
    """

    config: Dict[str, Any] = field(default_factory=dict)
    port: int = 0


class CustomBackend(ABC):
    """Base class a developer subclasses to define a custom inference backend.

    Subclasses implement :meth:`setup` (load the model once) and
    :meth:`process` (run one job). The runtime instantiates the subclass with no
    arguments, so do not require constructor parameters — read everything from
    the :class:`BackendContext` in :meth:`setup`.

    Optionally declare metadata as class attributes (:attr:`name`,
    :attr:`version`, :attr:`task_type`, :attr:`requirements`); the runtime
    exposes them via ``GET /meta``. Leaving them unset is fine — :meth:`manifest`
    falls back to the class name and empty fields.
    """

    #: Human-readable backend name. Defaults to the class name in :meth:`manifest`.
    name: str = ""
    #: Backend version string (e.g. ``"0.1.0"``); free-form.
    version: str = ""
    #: One of :data:`TASK_TYPES`; empty means "unspecified".
    task_type: str = ""
    #: By convention, a path reference to the backend's ``requirements.txt``.
    requirements: str = ""
    #: PEP 440 version specifier the backend's deps need (e.g. ``">=3.9,<3.12"``
    #: or an exact ``"3.11"``). Empty means "no constraint" — the runtime picks a
    #: sane default Python. Declare this when a pinned dep only ships wheels for
    #: certain Python versions so the worker provisions the right interpreter.
    requires_python: str = ""

    def manifest(self) -> BackendManifest:
        """Return this backend's declarative metadata for ``GET /meta``.

        Reads the class attributes; if :attr:`name` is unset it falls back to the
        class name so ``/meta`` always returns *something* without breaking.
        Override only if a backend computes metadata dynamically.
        """
        return BackendManifest(
            name=self.name or type(self).__name__,
            version=self.version,
            task_type=self.task_type,
            requirements=self.requirements,
            requires_python=self.requires_python,
        )

    @abstractmethod
    def setup(self, ctx: BackendContext) -> None:
        """Load the model once. Called a single time before serving traffic.

        Instantiate the ``nn.Module`` (or whatever the backend needs) here and
        store it on ``self``. Raising from here means the backend never becomes
        ready: the runtime logs the traceback and exits non-zero.
        """
        raise NotImplementedError

    @abstractmethod
    def process(self, job: Job) -> Result:
        """Process one :class:`Job` using the model loaded in :meth:`setup`.

        Raising from here yields a 500 ``{"error": ...}`` for *this* job while
        the server stays alive for the next one.
        """
        raise NotImplementedError

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        """Predict a time series (optional capability, not abstract).

        This is **opt-in**: the default raises :class:`NotImplementedError`, so
        backends that only implement :meth:`process` stay valid and the runtime
        answers ``501`` on ``POST /forecast``. A forecasting backend reuses the
        model loaded in :meth:`setup` (no new lifecycle) and overrides this to
        return a :class:`ForecastResult`.

        Raising a *generic* exception here yields a ``500`` for this request
        while the server stays alive, mirroring :meth:`process`.

        A forecasting backend reads ``request.freq`` (the pandas-style frequency,
        e.g. ``"D"``) when it needs to date the output window's timestamps or feed
        the frequency to the model; it is ``None`` when the caller omits it.

        Example (a trivial persistence forecaster)::

            def forecast(self, request: ForecastRequest) -> ForecastResult:
                last = request.target[-1]
                return ForecastResult(median=[last] * request.horizon)
        """
        raise NotImplementedError("backend does not support forecast")

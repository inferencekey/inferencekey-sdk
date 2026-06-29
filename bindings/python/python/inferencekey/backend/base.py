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
from typing import Any, Dict


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
)


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

    def to_wire(self) -> Dict[str, Any]:
        """Render to the JSON-serializable dict returned by ``GET /meta``."""
        return {
            "name": self.name,
            "version": self.version,
            "task_type": self.task_type,
            "requirements": self.requirements,
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

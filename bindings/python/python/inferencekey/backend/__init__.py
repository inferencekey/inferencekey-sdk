"""Custom-backend contract (T01).

The pure-Python surface a developer uses to ship their own PyTorch inference:
subclass :class:`CustomBackend`, implement :meth:`~CustomBackend.setup` and
:meth:`~CustomBackend.process`, then serve it with::

    python -m inferencekey.backend.serve --port <port> --backend <module:Class>

This subpackage imports neither a HTTP framework nor ``torch``; see
:mod:`inferencekey.backend.serve` for the runtime.

For a quick start from code (a script or REPL) instead of the CLI, use
:func:`~inferencekey.backend.serve.serve_backend`::

    from inferencekey.backend import serve_backend
    from mybackend import MyBackend

    serve_backend(MyBackend, port=8099, config={"device": "cpu"})
"""

from __future__ import annotations

from .base import (
    TASK_TYPES,
    BackendContext,
    BackendManifest,
    CustomBackend,
    Job,
    Result,
)
from .serve import serve_backend

__all__ = [
    "CustomBackend",
    "Job",
    "Result",
    "BackendContext",
    "BackendManifest",
    "TASK_TYPES",
    "serve_backend",
]

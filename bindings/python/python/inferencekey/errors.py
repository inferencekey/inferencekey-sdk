"""Exception hierarchy.

The native layer raises Python builtins (``PermissionError`` / ``ValueError`` /
``RuntimeError``) keyed off the core error variant; this module re-exports SDK
aliases so callers can ``except inferencekey.PermissionDenied`` and friends.
"""

from __future__ import annotations


class InferenceKeyError(Exception):
    """Base class for every SDK error."""


class PermissionDenied(InferenceKeyError):
    """403 — the credential may not perform the operation
    (wrong_credential_type / project_scope_mismatch / scope_insufficient)."""


class AuthError(InferenceKeyError):
    """401 — missing or invalid credentials."""


class ValidationError(InferenceKeyError):
    """A request argument failed local or server validation (400)."""


class ConfigurationError(InferenceKeyError):
    """Client-side misconfiguration before any request."""


class ApiError(InferenceKeyError):
    """Any other non-2xx response or transport failure."""


class BackendError(InferenceKeyError):
    """Base class for custom-backend (T01) contract/runtime failures."""


class BackendSetupError(BackendError):
    """``CustomBackend.setup()`` failed — the backend never became ready and the
    runtime exits with a non-zero status."""


class BackendEntrypointError(BackendError):
    """The ``module:Class`` entrypoint could not be imported or is not a
    :class:`~inferencekey.backend.CustomBackend` subclass."""


class PackagingError(BackendError):
    """Packaging a custom backend into a distributable artifact failed —
    a missing/invalid source, a malformed ``module:Class`` entrypoint, an
    unknown ``task_type``, or an unsafe member while reading an archive. No
    artifact is produced when this is raised."""

"""Publish a packaged custom backend to the Manager (control plane).

A backend artifact built by :func:`inferencekey.backend.package_backend` is a
``.tar.gz`` with a root ``manifest.json``. :func:`publish_custom_backend` reads
that manifest (without importing the backend or ``torch``, reusing
:func:`~inferencekey.backend.packaging.read_manifest_from_archive`) and uploads
the artifact to::

    POST {base_url}/api/tenants/{tenant_id}/custom-backends

as a ``multipart/form-data`` body: the file part ``file`` (sent as
``application/gzip``) plus text parts ``name``, ``slug``, ``version``,
``task_type`` and ``entrypoint`` taken from the manifest. The ``Authorization``
header carries ``Bearer <token>`` with the token passed **verbatim** — this
function makes no assumption about its format (the Manager authenticates the
upload as a user via ``AuthUser``; the SDK does not mint or reshape it).

Implementation note: the multipart body is built with the standard library
(:mod:`urllib.request`) — no third-party HTTP dependency is added. The control
plane already lives in pure Python (:mod:`inferencekey.clients`), so keeping the
upload here, off the native extension, is the lower-risk path and matches the
"thin binding" rule.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .backend.packaging import read_manifest_from_archive
from .errors import (
    ApiError,
    AuthError,
    ConfigurationError,
    PermissionDenied,
    ValidationError,
)

__all__ = ["publish_custom_backend"]

#: Default control-plane base URL, kept in lockstep with
#: :data:`inferencekey.clients._DEFAULT_BASE_URL`.
_DEFAULT_BASE_URL = "https://api.inferencekey.com"


def publish_custom_backend(
    tenant_id: str,
    package_path: str,
    *,
    token: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload a packaged custom backend to the Manager and return its record.

    Reads the artifact's ``manifest.json`` for the upload metadata (``slug``
    falls back to ``name`` when the manifest omits it), then POSTs the artifact
    as multipart form data to
    ``{base_url}/api/tenants/{tenant_id}/custom-backends``.

    :param tenant_id: the tenant UUID the backend is registered under.
    :param package_path: path to the ``.tar.gz`` built by ``package_backend``.
    :param token: the bearer token, forwarded **verbatim** in
        ``Authorization: Bearer <token>``. Required.
    :param base_url: control-plane base URL; defaults to
        ``INFERENCEKEY_BASE_URL`` then the SDK default.
    :returns: the parsed JSON response (``id``, ``slug``, ``sha256``, …).
    :raises ConfigurationError: if ``token`` is missing or the manifest lacks a
        required field.
    :raises PermissionDenied: on a 401/403 response.
    :raises ValidationError: on a 400 response.
    :raises ApiError: on any other non-2xx response or transport failure.
    :raises PackagingError: if the artifact's manifest cannot be read.
    """
    if not token:
        raise ConfigurationError("publish_custom_backend requires a token.")

    resolved_base = base_url or os.environ.get("INFERENCEKEY_BASE_URL") or _DEFAULT_BASE_URL

    manifest = read_manifest_from_archive(package_path)
    name = manifest.get("name")
    if not name:
        raise ConfigurationError(
            f"manifest in {package_path!r} is missing a 'name'."
        )
    slug = manifest.get("slug") or name
    fields = {
        "name": str(name),
        "slug": str(slug),
        "version": str(manifest.get("version", "")),
        "task_type": str(manifest.get("task_type", "")),
        "entrypoint": str(manifest.get("entrypoint", "")),
    }

    with open(package_path, "rb") as fh:
        file_bytes = fh.read()
    filename = os.path.basename(package_path)

    body, content_type = _encode_multipart(fields, "file", filename, file_bytes)

    url = f"{resolved_base.rstrip('/')}/api/tenants/{tenant_id}/custom-backends"
    request = Request(url, data=body, method="POST")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Content-Type", content_type)
    request.add_header("Content-Length", str(len(body)))

    try:
        with urlopen(request) as response:  # noqa: S310 - explicit known URL
            raw = response.read()
    except HTTPError as exc:
        raise _http_error(exc) from None
    except URLError as exc:
        raise ApiError(f"could not reach {url}: {exc.reason}") from None

    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError(f"invalid JSON in response from {url}: {exc}") from None


def _http_error(exc: HTTPError) -> Exception:
    """Map an HTTP status to the matching SDK exception, with the response body."""
    detail = _read_error_body(exc)
    message = f"{exc.code} {exc.reason}: {detail}" if detail else f"{exc.code} {exc.reason}"
    if exc.code == 401:
        return AuthError(message)
    if exc.code == 403:
        return PermissionDenied(message)
    if exc.code == 400:
        return ValidationError(message)
    return ApiError(message)


def _read_error_body(exc: HTTPError) -> str:
    """Best-effort decode of an error response body (empty string on failure)."""
    try:
        return exc.read().decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001 - body is diagnostic only, never fatal
        return ""


def _encode_multipart(
    fields: Dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
) -> tuple[bytes, str]:
    """Build a ``multipart/form-data`` body and its ``Content-Type`` header.

    Text ``fields`` come first, then one file part under ``file_field`` sent with
    ``Content-Type: application/gzip`` (the Manager keys the artifact off the
    field name, not this type, but it labels the part accurately).
    """
    boundary = f"----inferencekey-{uuid.uuid4().hex}"
    crlf = b"\r\n"
    parts: list[bytes] = []

    for key, value in fields.items():
        parts.append(f"--{boundary}".encode("utf-8"))
        parts.append(
            f'Content-Disposition: form-data; name="{key}"'.encode("utf-8")
        )
        parts.append(b"")
        parts.append(value.encode("utf-8"))

    parts.append(f"--{boundary}".encode("utf-8"))
    parts.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"'
        ).encode("utf-8")
    )
    parts.append(b"Content-Type: application/gzip")
    parts.append(b"")
    parts.append(file_bytes)

    parts.append(f"--{boundary}--".encode("utf-8"))
    parts.append(b"")

    body = crlf.join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type

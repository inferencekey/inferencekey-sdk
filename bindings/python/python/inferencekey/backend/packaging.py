"""Package a custom backend into a distributable ``.tar.gz`` artifact.

A *backend package* is what a future task uploads to the Manager and the worker
downloads: a gzip-compressed tarball (``kind='archive'``, aligned with the
Manager which accepts ``.tar.gz``/``.zip``) bundling

* the developer's backend **code** (a single ``.py`` file or a package/dir),
* a ``requirements.txt`` with the backend's dependencies, and
* a ``manifest.json`` at the **root** of the archive with the static metadata
  (:data:`MANIFEST_NAME`).

The ``manifest.json`` lives at the archive root **by design** (committee
decision P4): the Manager must read it *statically*, without starting or
importing the backend — and crucially without importing ``torch``. Hence this
module is pure stdlib (:mod:`tarfile`, :mod:`hashlib`, :mod:`json`,
:mod:`pathlib`); it never imports the developer's backend nor ``torch``. The
metadata comes from the caller's arguments, not from introspecting the code.

Public surface:

* :func:`package_backend` — build the artifact, return a :class:`BackendPackage`.
* :func:`read_manifest_from_archive` — extract *only* ``manifest.json`` from a
  package, safely (guarding against path traversal), and parse it.
* :class:`BackendPackage` — the dataclass returned by :func:`package_backend`.
* :class:`~inferencekey.errors.PackagingError` — raised on any validation or
  archive-safety failure (re-exported here for convenience).
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..errors import PackagingError
from .base import TASK_TYPES

__all__ = [
    "BackendPackage",
    "PackagingError",
    "SDK_PROTOCOL",
    "MANIFEST_NAME",
    "package_backend",
    "read_manifest_from_archive",
]

#: The SDK packaging-protocol version embedded in every manifest. A simple
#: constant the Manager/worker can branch on; bump only on a breaking layout
#: change. Kept a string for forward-compatibility.
SDK_PROTOCOL = "1"

#: The manifest filename, always at the root of the artifact.
MANIFEST_NAME = "manifest.json"

#: The requirements filename inside the artifact. Always present (empty if the
#: caller passes no ``requirements``) so the layout is stable.
REQUIREMENTS_NAME = "requirements.txt"

#: Read the final ``.tar.gz`` in blocks when hashing, so large artifacts do not
#: load wholly into memory.
_HASH_BLOCK = 1 << 20


@dataclass
class BackendPackage:
    """The result of :func:`package_backend`.

    ``path`` is the artifact on disk; ``sha256``/``size_bytes`` describe that
    exact file (the Manager registers both); ``manifest`` is the metadata dict
    written to ``manifest.json`` inside the archive.
    """

    path: str
    sha256: str
    size_bytes: int
    manifest: Dict[str, Any]


def _parse_entrypoint(entrypoint: str) -> Tuple[str, str]:
    """Validate ``module:Class`` and return ``(module, class)``.

    Raises :class:`PackagingError` if the form is wrong (missing ``:`` or an
    empty side) — mirroring the loader in :mod:`inferencekey.backend.serve`, but
    statically (no import).
    """
    if not isinstance(entrypoint, str) or ":" not in entrypoint:
        raise PackagingError(
            f"entrypoint must be 'module:Class', got {entrypoint!r}"
        )
    module_name, _, class_name = entrypoint.partition(":")
    if not module_name or not class_name:
        raise PackagingError(
            f"entrypoint must be 'module:Class', got {entrypoint!r}"
        )
    return module_name, class_name


def _collect_code_members(src: Path) -> List[Tuple[Path, str]]:
    """Return ``(filesystem_path, arcname)`` pairs for the backend code.

    A single ``.py`` file maps to its bare filename at the archive root; a
    directory is added recursively under its own name, in sorted order for a
    deterministic layout. ``manifest.json``/``requirements.txt`` at the source
    root are skipped — this module owns those names in the artifact.
    """
    if src.is_file():
        return [(src, src.name)]

    members: List[Tuple[Path, str]] = []
    base = src.name
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        arcname = f"{base}/{rel.as_posix()}"
        members.append((path, arcname))
    return members


def package_backend(
    *,
    src: str,
    entrypoint: str,
    name: str,
    version: str,
    out_dir: str,
    slug: Optional[str] = None,
    task_type: Optional[str] = None,
    requirements: Optional[str] = None,
    description: Optional[str] = None,
) -> BackendPackage:
    """Bundle a custom backend into ``<out_dir>/<name>-<version>.tar.gz``.

    The artifact contains, in deterministic (sorted) order, the backend code
    from ``src`` (a ``.py`` file or a directory), a ``requirements.txt`` (the
    file at ``requirements`` if given, otherwise an empty one so the layout is
    stable), and a root ``manifest.json`` carrying ``name``, ``slug``,
    ``version``, ``task_type``, ``entrypoint``, ``sdk_protocol`` and optional
    ``description``. ``slug`` is the publish identifier the Manager registers the
    backend under; it defaults to ``name`` when omitted.

    The metadata is taken verbatim from the arguments: this never imports the
    backend or ``torch``. Validation happens *before* any file is written, so a
    failure leaves no artifact behind.

    :raises PackagingError: if ``src`` does not exist, ``entrypoint`` is not
        ``module:Class``, or ``task_type`` is not one of
        :data:`~inferencekey.backend.base.TASK_TYPES`.
    """
    src_path = Path(src)
    if not src_path.exists():
        raise PackagingError(f"src does not exist: {src!r}")

    _parse_entrypoint(entrypoint)

    if task_type is not None and task_type not in TASK_TYPES:
        raise PackagingError(
            f"unknown task_type {task_type!r}; expected one of {TASK_TYPES}"
        )

    req_path: Optional[Path] = None
    if requirements is not None:
        req_path = Path(requirements)
        if not req_path.is_file():
            raise PackagingError(
                f"requirements is not a file: {requirements!r}"
            )

    code_members = _collect_code_members(src_path)
    if not code_members:
        raise PackagingError(f"src contains no files to package: {src!r}")

    manifest: Dict[str, Any] = {
        "name": name,
        "slug": slug or name,
        "version": version,
        "task_type": task_type or "",
        "entrypoint": entrypoint,
        "sdk_protocol": SDK_PROTOCOL,
    }
    if description is not None:
        manifest["description"] = description

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    artifact = out_path / f"{name}-{version}.tar.gz"

    manifest_bytes = json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    req_bytes = req_path.read_bytes() if req_path is not None else b""

    with tarfile.open(artifact, "w:gz") as tar:
        _add_bytes(tar, MANIFEST_NAME, manifest_bytes)
        _add_bytes(tar, REQUIREMENTS_NAME, req_bytes)
        for fs_path, arcname in code_members:
            tar.add(str(fs_path), arcname=arcname, recursive=False)

    sha256, size_bytes = _hash_and_size(artifact)
    return BackendPackage(
        path=str(artifact),
        sha256=sha256,
        size_bytes=size_bytes,
        manifest=manifest,
    )


def _add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    """Add an in-memory blob to ``tar`` as ``arcname``."""
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _hash_and_size(path: Path) -> Tuple[str, int]:
    """Return ``(sha256_hex, size_bytes)`` of ``path``, read in blocks."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_HASH_BLOCK), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _is_safe_member_name(name: str) -> bool:
    """True if ``name`` is a relative, traversal-free archive path.

    Rejects absolute paths and any component equal to ``..`` — the guard that
    keeps a malicious tar from writing outside an extraction root.
    """
    pure = Path(name)
    if pure.is_absolute():
        return False
    return not any(part == ".." for part in pure.parts)


def read_manifest_from_archive(path: str) -> Dict[str, Any]:
    """Extract and parse ``manifest.json`` from a backend package.

    Opens the ``.tar.gz`` at ``path``, reads *only* the root ``manifest.json``
    member, parses it as JSON and returns the dict. Never imports the backend or
    ``torch``. Members are validated against path traversal before being read.

    :raises PackagingError: if the archive cannot be opened, holds no readable
        ``manifest.json``, contains an unsafe member name, or the manifest is not
        valid JSON.
    """
    try:
        tar = tarfile.open(path, "r:gz")
    except (tarfile.TarError, OSError) as exc:
        raise PackagingError(f"could not open archive {path!r}: {exc}") from exc

    with tar:
        for member in tar.getmembers():
            if not _is_safe_member_name(member.name):
                raise PackagingError(
                    f"unsafe member in archive {path!r}: {member.name!r}"
                )
        try:
            member = tar.getmember(MANIFEST_NAME)
        except KeyError as exc:
            raise PackagingError(
                f"{MANIFEST_NAME} not found in archive {path!r}"
            ) from exc
        if not member.isfile():
            raise PackagingError(
                f"{MANIFEST_NAME} in archive {path!r} is not a regular file"
            )
        extracted = tar.extractfile(member)
        if extracted is None:
            raise PackagingError(
                f"could not read {MANIFEST_NAME} from archive {path!r}"
            )
        raw = extracted.read()

    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackagingError(
            f"{MANIFEST_NAME} in archive {path!r} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise PackagingError(
            f"{MANIFEST_NAME} in archive {path!r} must be a JSON object"
        )
    return manifest

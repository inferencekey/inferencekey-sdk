"""Unit tests for :func:`inferencekey.publish.publish_custom_backend`.

Torch-free and dependency-free: the artifact is built with the pure-stdlib
:func:`package_backend`, and the HTTP round-trip runs against a throwaway
:mod:`http.server` on localhost — exercising the real ``urllib`` multipart path,
no mocking of the network layer.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from inferencekey import (
    ApiError,
    AuthError,
    ConfigurationError,
    PermissionDenied,
    ValidationError,
    publish_custom_backend,
)
from inferencekey.backend import package_backend

_TENANT = "4668440b-73c6-41b0-b4b8-a5912d221851"

_DUMMY_BACKEND = '''\
from inferencekey.backend import CustomBackend, Job, Result

class DummyBackend(CustomBackend):
    def process(self, job: Job) -> Result:
        return Result(output=job.input)
'''


def _build_package(tmp_path: Path, *, with_slug: bool) -> str:
    src = tmp_path / "backend.py"
    src.write_text(_DUMMY_BACKEND)
    out = tmp_path / "out"
    pkg = package_backend(
        src=str(src),
        entrypoint="backend:DummyBackend",
        name="echo",
        slug="echo-pub" if with_slug else None,
        version="9.9.9",
        task_type="text2text",
        out_dir=str(out),
    )
    return pkg.path


class _Capture:
    """Holds what the fake server received for assertions after the request."""

    def __init__(self) -> None:
        self.headers: Dict[str, str] = {}
        self.body: bytes = b""
        self.path: str = ""


def _serve(status: int, response: bytes, capture: _Capture) -> Tuple[HTTPServer, str]:
    """Start a one-request localhost server returning ``status``/``response``."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            capture.path = self.path
            capture.headers = {k.lower(): v for k, v in self.headers.items()}
            length = int(self.headers.get("Content-Length", "0"))
            capture.body = self.rfile.read(length)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args: Any) -> None:  # silence test output
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


def test_publish_sends_multipart_with_manifest_fields(tmp_path: Path) -> None:
    package = _build_package(tmp_path, with_slug=True)
    body = json.dumps({"id": "abc-123", "slug": "echo-pub", "sha256": "deadbeef"}).encode()
    capture = _Capture()
    server, base_url = _serve(201, body, capture)
    try:
        resp = publish_custom_backend(_TENANT, package, token="raw-token-123", base_url=base_url)
    finally:
        server.server_close()

    assert resp == {"id": "abc-123", "slug": "echo-pub", "sha256": "deadbeef"}
    assert capture.path == f"/api/tenants/{_TENANT}/custom-backends"
    # Token is forwarded verbatim, prefixed only with the bearer scheme.
    assert capture.headers["authorization"] == "Bearer raw-token-123"
    assert capture.headers["content-type"].startswith("multipart/form-data; boundary=")
    text = capture.body.decode("utf-8", errors="replace")
    for fragment in (
        'name="name"',
        'name="slug"',
        "echo-pub",
        'name="version"',
        "9.9.9",
        'name="task_type"',
        "text2text",
        'name="entrypoint"',
        "backend:DummyBackend",
        'name="file"; filename="echo-9.9.9.tar.gz"',
        "Content-Type: application/gzip",
    ):
        assert fragment in text, fragment


def test_publish_slug_falls_back_to_name(tmp_path: Path) -> None:
    # Force a manifest without a slug by stripping it from a package; simplest is
    # to build with the legacy path (no slug) — package_backend defaults slug to
    # name, so assert the wire carries the name as the slug.
    package = _build_package(tmp_path, with_slug=False)
    capture = _Capture()
    server, base_url = _serve(201, b"{}", capture)
    try:
        publish_custom_backend(_TENANT, package, token="t", base_url=base_url)
    finally:
        server.server_close()
    text = capture.body.decode("utf-8", errors="replace")
    assert 'name="slug"' in text
    # slug part value is the name ("echo") since none was given.
    assert "\r\necho\r\n" in text


@pytest.mark.parametrize(
    "status,exc",
    [
        (401, AuthError),
        (403, PermissionDenied),
        (400, ValidationError),
        (500, ApiError),
    ],
)
def test_publish_maps_http_errors(tmp_path: Path, status: int, exc: type) -> None:
    package = _build_package(tmp_path, with_slug=True)
    capture = _Capture()
    server, base_url = _serve(status, b'{"error":"boom"}', capture)
    try:
        with pytest.raises(exc):
            publish_custom_backend(_TENANT, package, token="t", base_url=base_url)
    finally:
        server.server_close()


def test_publish_requires_token(tmp_path: Path) -> None:
    package = _build_package(tmp_path, with_slug=True)
    with pytest.raises(ConfigurationError):
        publish_custom_backend(_TENANT, package, token=None)

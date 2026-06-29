"""Integration tests for the runtime — start the real server, hit it over HTTP.

These use torch-free fake backends so the suite runs without torch installed.
The example's torch backend is exercised by the README's manual acceptance steps.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from inferencekey.backend import BackendContext, CustomBackend, Job, Result
from inferencekey.backend import serve as serve_mod
from inferencekey.errors import BackendEntrypointError, BackendSetupError


# --- torch-free fake backends (module-level so the entrypoint loader finds them) ---


class SlowEchoBackend(CustomBackend):
    """Echoes input; setup sleeps so /healthz is observably 503 first."""

    setup_calls = 0

    def setup(self, ctx: BackendContext) -> None:
        type(self).setup_calls += 1
        time.sleep(0.3)
        self.device = ctx.config.get("device", "cpu")

    def process(self, job: Job) -> Result:
        if job.input.get("boom"):
            raise ValueError("forced failure")
        return Result(output={"echo": job.input, "device": self.device})


class FailingSetupBackend(CustomBackend):
    def setup(self, ctx: BackendContext) -> None:
        raise RuntimeError("setup blew up")

    def process(self, job: Job) -> Result:  # pragma: no cover
        return Result(output={})


class MetaBackend(CustomBackend):
    """Declares metadata via class attributes — exercises GET /meta."""

    name = "meta-demo"
    version = "1.2.3"
    task_type = "classification"
    requirements = "requirements.txt"

    def setup(self, ctx: BackendContext) -> None:
        pass

    def process(self, job: Job) -> Result:
        return Result(output={})


class BareBackend(CustomBackend):
    """No metadata declared — /meta must fall back to the class name."""

    def setup(self, ctx: BackendContext) -> None:
        pass

    def process(self, job: Job) -> Result:
        return Result(output={})


class NotAResultBackend(CustomBackend):
    """process() returns the wrong type — exercises the C-5 error message."""

    def setup(self, ctx: BackendContext) -> None:
        pass

    def process(self, job: Job) -> Result:
        return {"output": {}}  # type: ignore[return-value]


class NotABackend:
    pass


# --- helpers ---


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _get(port: int, path: str):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _post(port: int, path: str, body: dict):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


# --- entrypoint loading ---


def test_load_backend_resolves_subclass() -> None:
    b = serve_mod.load_backend(f"{__name__}:SlowEchoBackend")
    assert isinstance(b, SlowEchoBackend)


@pytest.mark.parametrize("bad", ["no_colon", ":Class", "mod:", f"{__name__}:Missing"])
def test_load_backend_rejects_bad_entrypoints(bad: str) -> None:
    with pytest.raises(BackendEntrypointError):
        serve_mod.load_backend(bad)


def test_load_backend_rejects_non_backend_class() -> None:
    with pytest.raises(BackendEntrypointError):
        serve_mod.load_backend(f"{__name__}:NotABackend")


# --- readiness, processing, resilience, single load ---


def test_readiness_and_process_lifecycle() -> None:
    SlowEchoBackend.setup_calls = 0
    port = _free_port()
    backend = SlowEchoBackend()

    # serve() binds then runs setup() (which sleeps). Start it in a thread so we
    # can observe the 503 window before setup completes.
    server_box = {}

    def _run():
        server_box["httpd"], server_box["state"] = serve_mod.serve(
            backend, port, {"device": "cpu"}
        )

    t = threading.Thread(target=_run)
    t.start()

    # While setup sleeps, /healthz must be 503.
    time.sleep(0.1)
    status, body = _get(port, "/healthz")
    assert status == 503, body

    t.join(timeout=5)
    assert "httpd" in server_box, "serve() did not return"
    httpd = server_box["httpd"]
    try:
        # After setup: 200.
        status, body = _get(port, "/healthz")
        assert status == 200 and body == {"status": "ok"}

        # Process N>=3 valid jobs.
        for i in range(3):
            status, body = _post(port, "/process", {"id": f"j{i}", "input": {"n": i}})
            assert status == 200
            assert body["output"]["echo"] == {"n": i}

        # setup() ran exactly once across all jobs.
        assert SlowEchoBackend.setup_calls == 1

        # Resilience: a job that raises -> 500 {"error":...}, server stays alive.
        status, body = _post(port, "/process", {"id": "boom", "input": {"boom": True}})
        assert status == 500 and "error" in body

        # A valid job afterwards still works (200).
        status, body = _post(port, "/process", {"id": "after", "input": {"n": 99}})
        assert status == 200 and body["output"]["echo"] == {"n": 99}

        # Bad job body -> 400.
        status, body = _post(port, "/process", {"input": {}})
        assert status == 400 and "error" in body
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_setup_failure_raises_and_never_ready() -> None:
    port = _free_port()
    with pytest.raises(BackendSetupError):
        serve_mod.serve(FailingSetupBackend(), port, {})
    # Server was torn down; nothing listens -> connection refused.
    with pytest.raises(urllib.error.URLError):
        _get(port, "/healthz")


def test_main_returns_nonzero_on_setup_failure() -> None:
    port = _free_port()
    rc = serve_mod.main(["--port", str(port), "--backend", f"{__name__}:FailingSetupBackend"])
    assert rc == 1


def test_main_returns_nonzero_on_bad_entrypoint() -> None:
    rc = serve_mod.main(["--port", str(_free_port()), "--backend", "does.not.exist:Nope"])
    assert rc == 2


# --- C-1: GET /meta ---


def test_meta_returns_declared_metadata() -> None:
    port = _free_port()
    httpd, _state = serve_mod.serve(MetaBackend(), port, {})
    try:
        status, body = _get(port, "/meta")
        assert status == 200
        assert body == {
            "name": "meta-demo",
            "version": "1.2.3",
            "task_type": "classification",
            "requirements": "requirements.txt",
        }
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_meta_falls_back_to_class_name_without_metadata() -> None:
    port = _free_port()
    httpd, _state = serve_mod.serve(BareBackend(), port, {})
    try:
        status, body = _get(port, "/meta")
        assert status == 200
        assert body == {
            "name": "BareBackend",
            "version": "",
            "task_type": "",
            "requirements": "",
        }
    finally:
        httpd.shutdown()
        httpd.server_close()


# --- C-4: serve_backend helper ---


def test_serve_backend_from_class_non_blocking() -> None:
    port = _free_port()
    httpd, state = serve_mod.serve_backend(BareBackend, port, block=False)
    try:
        assert state.ready
        status, _body = _get(port, "/healthz")
        assert status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_serve_backend_accepts_instance() -> None:
    port = _free_port()
    httpd, _state = serve_mod.serve_backend(
        MetaBackend(), port, {"device": "cpu"}, block=False
    )
    try:
        status, body = _get(port, "/meta")
        assert status == 200 and body["name"] == "meta-demo"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_serve_backend_rejects_non_backend() -> None:
    with pytest.raises(BackendEntrypointError):
        serve_mod.serve_backend(NotABackend, _free_port(), block=False)


def test_serve_backend_importable_from_package() -> None:
    from inferencekey.backend import serve_backend as exported

    assert exported is serve_mod.serve_backend


# --- C-5: clearer error when process() does not return a Result ---


def test_process_not_returning_result_yields_helpful_500() -> None:
    port = _free_port()
    httpd, _state = serve_mod.serve_backend(NotAResultBackend, port, block=False)
    try:
        status, body = _post(port, "/process", {"id": "j1", "input": {}})
        assert status == 500
        assert "must return a Result" in body["error"]
        assert "Result(output=...)" in body["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()

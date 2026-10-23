"""Shared test setup: run the runtime's contract tests on both HTTP transports.

Every test in the modules of ``PARITY_MODULES`` runs twice, once per value of
``IK_BACKEND_HTTP`` (``stdlib`` and ``native``), without any change to the test
itself: the autouse ``transport`` fixture sets the variable, and
``serve.serve()`` (in process, or in a subprocess that inherits the
environment) picks the transport from it.

The native run is skipped only when the native module has no backend server
(``BACKEND_SERVER_ABI``). With ``IK_REQUIRE_NATIVE=1`` (CI) a missing native
server fails instead.
"""

from __future__ import annotations

import os

import pytest

TRANSPORTS = ("stdlib", "native")

#: Modules whose tests must pass unchanged on both transports.
PARITY_MODULES = {
    "test_backend_serve",
    "test_backend_chat",
    "test_backend_contract",
    "test_example_deepinfer",
    "test_example_text_sentiment",
}


def native_available() -> bool:
    """Whether the native module has the backend server this runtime speaks."""
    try:
        from inferencekey import _inferencekey
    except (ImportError, OSError):
        return False
    return getattr(_inferencekey, "BACKEND_SERVER_ABI", 0) == 1


def require_native() -> None:
    """Skip (or, with ``IK_REQUIRE_NATIVE=1``, fail) when native is unavailable."""
    if native_available():
        return
    if os.environ.get("IK_REQUIRE_NATIVE") == "1":
        pytest.fail("IK_REQUIRE_NATIVE=1 but the native backend server is unavailable")
    pytest.skip("native module has no BACKEND_SERVER_ABI")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    module = metafunc.module.__name__.rsplit(".", 1)[-1]
    if module in PARITY_MODULES and "transport" in metafunc.fixturenames:
        metafunc.parametrize("transport", TRANSPORTS, indirect=True)


@pytest.fixture(autouse=True)
def transport(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """The HTTP transport a parity test runs on (``None`` outside the parity modules)."""
    name = getattr(request, "param", None)
    if name is None:
        yield None
        return
    if name == "native":
        require_native()
    monkeypatch.setenv("IK_BACKEND_HTTP", name)
    yield name

"""Test the text-sentiment example backend (C-3).

This exercises the *text* example end to end over the real runtime. It needs
torch (the example's dep), so the whole module is skipped if torch is not
installed — exactly like the echo example, whose torch path is covered by its
README's manual acceptance steps.
"""

from __future__ import annotations

import json
import pathlib
import sys
import urllib.error
import urllib.request

import pytest

pytest.importorskip("torch")

from inferencekey.backend import serve as serve_mod  # noqa: E402

# Make the example folder importable so `backend:SentimentBackend` resolves.
_EXAMPLE_DIR = (
    pathlib.Path(__file__).resolve().parents[3]
    / "examples"
    / "custom-backend-text-sentiment"
)
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _get(port: int, path: str):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode())


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


def test_sentiment_example_serves_text_jobs_and_meta() -> None:
    from backend import SentimentBackend  # imported lazily after sys.path setup

    port = _free_port()
    httpd, state = serve_mod.serve_backend(
        SentimentBackend, port, {"device": "cpu"}, block=False
    )
    try:
        assert state.ready

        # /meta reflects the declared metadata.
        status, meta = _get(port, "/meta")
        assert status == 200
        assert meta["name"] == "tiny-sentiment"
        assert meta["task_type"] == "classification"

        # Positive text via input.text.
        status, body = _post(
            port, "/process", {"id": "p", "input": {"text": "good great love best"}}
        )
        assert status == 200
        assert body["output"]["label"] == "positive"
        assert 0.0 <= body["output"]["score"] <= 1.0

        # Negative text via input.prompt (the alternate text key).
        status, body = _post(
            port,
            "/process",
            {"id": "n", "input": {"prompt": "bad terrible hate worst"}},
        )
        assert status == 200
        assert body["output"]["label"] == "negative"

        # Missing text/prompt -> process() raises -> 500, server stays alive.
        status, body = _post(port, "/process", {"id": "bad", "input": {}})
        assert status == 500 and "error" in body

        status, _body = _get(port, "/healthz")
        assert status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()

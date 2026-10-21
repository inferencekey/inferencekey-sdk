"""Tests for the DeepInfer custom backend example, with its fake engine.

No GPU and no DeepInfer: ``engine: "fake"`` answers in the DeepSeek-V4.1
completion format, so this covers the OpenAI to DeepSeek-V4.1 message mapping,
DSML tool-call parsing, streaming (including held-back markup) and usage, over
the real SDK runtime.
"""

from __future__ import annotations

import json
import pathlib
import socket
import sys
import urllib.request

import pytest

from inferencekey.backend import API_OPENAI_CHAT, BackendContext
from inferencekey.backend import serve as serve_mod

_EXAMPLE_DIR = pathlib.Path(__file__).resolve().parents[3] / "examples" / "custom-backend-deepinfer"
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

import deepinfer_backend as db  # noqa: E402
import dsv41_encoding as dsv41  # noqa: E402

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
        },
    }
]


def _backend(**config):
    backend = db.DeepInferBackend()
    backend.setup(BackendContext(config={"engine": "fake", **config}))
    return backend


def test_backend_declares_openai_chat() -> None:
    assert db.DeepInferBackend().resolved_api() == API_OPENAI_CHAT


def test_messages_and_tools_map_to_the_dsv41_prompt() -> None:
    request = {
        "messages": [
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "read a.txt"}]},
        ],
        "tools": TOOLS,
    }
    messages = db.to_dsv41_messages(request)
    assert messages[0] == {"role": "system", "content": "Be brief.", "tools": TOOLS}
    prompt = dsv41.encode_messages(messages, thinking_mode="thinking")
    assert prompt.startswith(dsv41.bos_token)
    assert '"name": "Read"' in prompt and "<｜User｜>read a.txt<｜Assistant｜><think>" in prompt


def test_tools_without_system_message_create_one() -> None:
    messages = db.to_dsv41_messages({"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS})
    assert messages[0]["role"] == "system" and messages[0]["tools"] == TOOLS


def test_chat_returns_tool_call_reasoning_and_usage() -> None:
    backend = _backend(fake_tool_arguments={"file_path": "/tmp/a.txt"})
    out = backend.chat({"messages": [{"role": "user", "content": "read it"}], "tools": TOOLS})
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["id"].startswith("call_") and call["type"] == "function"
    assert call["function"]["name"] == "Read"
    assert json.loads(call["function"]["arguments"]) == {"file_path": "/tmp/a.txt"}
    assert choice["message"]["reasoning_content"] == "Let me look at the request."
    assert out["usage"]["prompt_tokens"] > 0 and out["usage"]["completion_tokens"] > 0


def test_tool_result_round_trip() -> None:
    backend = _backend(fake_tool_arguments={"file_path": "/tmp/a.txt"})
    first = backend.chat({"messages": [{"role": "user", "content": "read it"}], "tools": TOOLS})
    assistant = first["choices"][0]["message"]
    messages = [
        {"role": "user", "content": "read it"},
        assistant,
        {"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"], "content": "hello world\nsecond"},
    ]
    second = backend.chat({"messages": messages, "tools": TOOLS})
    assert second["choices"][0]["finish_reason"] == "stop"
    assert second["choices"][0]["message"]["content"] == "The tool returned: hello world"


def test_chat_mode_has_no_reasoning() -> None:
    out = _backend(thinking_mode="chat").chat({"messages": [{"role": "user", "content": "hi"}]})
    message = out["choices"][0]["message"]
    assert message["content"] == "echo: hi" and "reasoning_content" not in message


def test_stream_sends_reasoning_then_tool_call_then_usage() -> None:
    backend = _backend(fake_tool_arguments={"file_path": "/tmp/a.txt", "limit": 5})
    chunks = list(backend.chat_stream({"messages": [{"role": "user", "content": "read it"}], "tools": TOOLS}))
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") or "" for d in deltas)
    assert reasoning == "Let me look at the request."
    assert content == ""  # no DSML markup leaks into content
    calls = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
    assert len(calls) == 1 and json.loads(calls[0]["function"]["arguments"]) == {"file_path": "/tmp/a.txt", "limit": 5}
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["total_tokens"] > 0


def test_splitter_holds_back_a_split_marker() -> None:
    splitter = db.StreamSplitter(thinking=True)
    out = []
    for piece in ["thinking", " hard</th", "ink>Answer", " here", "\n\n<｜DS", "ML｜ calls>..."]:
        out += splitter.feed(piece)
    assert "".join(d.get("reasoning_content", "") for d in out) == "thinking hard"
    assert "".join(d.get("content", "") for d in out) == "Answer here"
    assert splitter.in_tool_calls


def test_truncated_completion_is_finish_length() -> None:
    backend = _backend(thinking_mode="chat")
    out = backend.chat({"messages": [{"role": "user", "content": "a long question"}], "max_tokens": 1})
    assert out["choices"][0]["finish_reason"] == "length"


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(ValueError):
        _backend(engine="vllm")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_served_over_the_runtime_with_sse() -> None:
    port = _free_port()
    httpd, _ = serve_mod.serve(db.DeepInferBackend(), port, {"engine": "fake", "fake_tool_arguments": {"file_path": "/x"}})
    try:
        body = {"stream": True, "messages": [{"role": "user", "content": "read"}], "tools": TOOLS}
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            events = [l[6:] for l in resp.read().decode().splitlines() if l.startswith("data: ")]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        assert any(c["choices"] and "tool_calls" in c["choices"][0]["delta"] for c in chunks)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_deepinfer_venv_extends_path_and_sets_rank_interpreter(tmp_path, monkeypatch) -> None:
    import deepinfer_engines as de

    minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = tmp_path / "lib" / minor / "site-packages"
    site.mkdir(parents=True)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python").write_text("")
    monkeypatch.setattr(sys, "path", list(sys.path))

    class Ctx:
        executable = None

        def set_executable(self, path: str) -> None:
            self.executable = path

    ctx = Ctx()
    de.use_deepinfer_venv(str(tmp_path), ctx)
    assert sys.path[-1] == str(site)
    assert ctx.executable == str(tmp_path / "bin" / "python")

    de.use_deepinfer_venv(None, ctx)  # no venv configured: nothing changes
    with pytest.raises(FileNotFoundError):
        de.use_deepinfer_venv(str(tmp_path / "missing"), Ctx())

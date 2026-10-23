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


# --------------------------------------------------------------------------
# DeepInferEngine without a GPU: detokenization, the rank loop, the parent side
# --------------------------------------------------------------------------

import queue  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import deepinfer_engines as de  # noqa: E402

_ALPHABET = de._byte_level_alphabet()
_BYTE_CHAR = {b: c for c, b in _ALPHABET.items()}
_ADDED = ["<｜end▁of▁sentence｜>", "</think>"]


def _bl(data: bytes) -> str:
    """Bytes as a byte-level BPE token string."""
    return "".join(_BYTE_CHAR[b] for b in data)


class ToyByteLevelBackend:
    """A tiny Hugging Face ``tokenizers.Tokenizer`` look-alike with a ByteLevel decoder.

    Ids 0 to 255 are single bytes; then a few merges, including the first two
    bytes of a CJK character (so a character splits across tokens), then the
    added tokens. ``decode`` follows the Rust ByteLevel decoder: per token, the
    bytes its characters stand for (or its UTF-8 when a character is outside
    the byte alphabet), unknown ids dropped, UTF-8 with replacement.
    """

    def __init__(self, decoder: str = "ByteLevel") -> None:
        self.decoder = decoder
        merges = [b"Hello", b" world", "你".encode()[:2], " 🎉".encode()[:3], b"\n\n"]
        self.vocab = [_bl(bytes([b])) for b in range(256)] + [_bl(m) for m in merges] + _ADDED

    def to_str(self) -> str:
        return json.dumps({"decoder": {"type": self.decoder}})

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        return len(self.vocab)

    def id_to_token(self, token_id: int):
        return self.vocab[token_id] if 0 <= token_id < len(self.vocab) else None

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        out = b""
        for i in ids:
            token = self.id_to_token(i)
            if token is None:
                continue
            try:
                out += bytes(_ALPHABET[c] for c in token)
            except KeyError:
                out += token.encode()
        return out.decode("utf-8", "replace")

    def decode_batch(self, batch, skip_special_tokens: bool = False):
        return [self.decode(ids) for ids in batch]

    def encode(self, text: str):
        return types.SimpleNamespace(ids=self.encode_ids(text))

    def encode_ids(self, text: str):
        ids, rest = [], text
        while rest:
            added = next((a for a in _ADDED if rest.startswith(a)), None)
            if added:
                ids.append(self.vocab.index(added))
                rest = rest[len(added):]
                continue
            ids.extend(rest[0].encode())
            rest = rest[1:]
        return ids


class ToyTokenizer:
    """``PreTrainedTokenizerFast`` look-alike over :class:`ToyByteLevelBackend`."""

    clean_up_tokenization_spaces = False

    def __init__(self, backend=None) -> None:
        self.backend_tokenizer = backend or ToyByteLevelBackend()

    def encode(self, text: str):
        return self.backend_tokenizer.encode_ids(text)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return self.backend_tokenizer.decode(ids)


class ToySentencePiece:
    """A tokenizer without a byte table: SentencePiece style, the leading space of the
    whole text is dropped and byte-fallback tokens assemble multi-byte characters."""

    PIECES = ["▁Hello", "▁world", ",", "▁naïve", "<0xF0>", "<0x9F>", "<0x8E>", "<0x89>", "!"]

    def encode(self, text: str):  # pragma: no cover (prompts are not tokenized with it)
        return [0]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        out = b""
        for i in ids:
            piece = self.PIECES[i]
            out += bytes([int(piece[3:5], 16)]) if piece.startswith("<0x") else piece.replace("▁", " ").encode()
        text = out.decode("utf-8", "replace")
        return text[1:] if text.startswith(" ") else text


def _feed_one_by_one(detok, ids):
    pieces = [detok.feed([i]) for i in ids]
    return pieces, "".join(pieces) + detok.flush()


def test_byte_level_table_is_used_and_exact_across_split_characters() -> None:
    tok = ToyTokenizer()
    new = de.detokenizer_factory(tok)
    detok = new()
    assert isinstance(detok, de.ByteLevelDetokenizer)
    vocab = tok.backend_tokenizer.vocab
    # "Hello 你好 🎉</think>\n\n" with 你 and 🎉 split across tokens, then EOS.
    ids = [vocab.index(_bl(b"Hello")), 32, vocab.index(_bl("你".encode()[:2])), "你".encode()[2]]
    ids += list("好".encode()) + [vocab.index(_bl(" 🎉".encode()[:3])), *" 🎉".encode()[3:]]
    ids += [vocab.index("</think>"), vocab.index(_bl(b"\n\n")), vocab.index(_ADDED[0])]
    pieces, text = _feed_one_by_one(detok, ids)
    assert text == tok.decode(ids) == "Hello 你好 🎉</think>\n\n<｜end▁of▁sentence｜>"
    assert not any("�" in p for p in pieces)  # half characters are held back, not sent broken
    assert pieces[2] == "" and pieces[3] == "你"


def test_byte_level_detokenizer_matches_decode_at_the_edges() -> None:
    tok = ToyTokenizer()
    # Ends inside a character: decode() shows U+FFFD, and so does the final flush.
    ids = list("é".encode())[:1]
    assert _feed_one_by_one(de.detokenizer_factory(tok)(), ids)[1] == tok.decode(ids) == "�"
    # Invalid bytes in the middle and unknown ids (dropped by decode) on the way.
    ids = [0xFF, 65, 10_000, -1, 66]
    assert _feed_one_by_one(de.detokenizer_factory(tok)(), ids)[1] == tok.decode(ids) == "�AB"
    # Several tokens in one feed (a DSpark step) give the same text.
    ids = tok.encode("naïve 你 🎉 </think>")
    detok = de.detokenizer_factory(tok)()
    assert detok.feed(ids[:5]) + detok.feed(ids[5:]) + detok.flush() == "naïve 你 🎉 </think>"


def test_no_byte_table_when_decode_is_not_plain_bytes() -> None:
    assert de.byte_level_table(ToyTokenizer(ToyByteLevelBackend(decoder="Metaspace"))) is None
    cleaned = ToyTokenizer()
    cleaned.clean_up_tokenization_spaces = True
    assert de.byte_level_table(cleaned) is None
    assert de.byte_level_table(ToySentencePiece()) is None
    # A backend whose decode disagrees with the table is caught by the check.
    lying = ToyByteLevelBackend()
    lying.decode_batch = lambda batch, skip_special_tokens=False: ["x" for _ in batch]
    assert de.byte_level_table(ToyTokenizer(lying)) is None
    assert isinstance(de.detokenizer_factory(ToySentencePiece())(), de.WindowDetokenizer)


def test_window_detokenizer_matches_full_decode() -> None:
    tok = ToySentencePiece()
    ids = [0, 2, 1, 3, 4, 5, 6, 7, 8]  # "Hello, world naïve🎉!" with 🎉 in four byte tokens
    pieces, text = _feed_one_by_one(de.WindowDetokenizer(tok), ids)
    assert text == tok.decode(ids) == "Hello, world naïve🎉!"
    assert not any("�" in p for p in pieces)
    assert pieces[1] == "," and pieces[2] == " world"  # the space survives the sliding window
    truncated = [0, 4, 5]
    assert _feed_one_by_one(de.WindowDetokenizer(tok), truncated)[1] == tok.decode(truncated)


@pytest.mark.skipif(not __import__("os").environ.get("BENCH_TOKENIZER_DIR"), reason="needs BENCH_TOKENIZER_DIR")
def test_byte_table_with_the_real_dsv41_tokenizer() -> None:  # pragma: no cover (optional)
    transformers = pytest.importorskip("transformers")
    import os
    import random

    tok = transformers.AutoTokenizer.from_pretrained(os.environ["BENCH_TOKENIZER_DIR"])
    assert de.byte_level_table(tok) is not None
    new = de.detokenizer_factory(tok)
    rng = random.Random(0)
    text = "Hola, ¿qué tal? 你好世界 🎉🚀 </think>\n\n<｜DSML｜calls> café\n\tdef f(x): return x"
    size = len(tok)  # slow in transformers: computed once
    sequences = [tok.encode(text)] + [[rng.randrange(size) for _ in range(40)] for _ in range(300)]
    for ids in sequences:
        assert _feed_one_by_one(new(), ids)[1] == tok.decode(ids, skip_special_tokens=False)


# --- the rank loop, in-process, against a fake engine ----------------------


class FakeRankEngine:
    """The ``llm.llm_engine`` surface the rank loop uses, stepping instantly."""

    multi_modal_processor = None

    def __init__(self, completions, tokens_per_step: int = 1) -> None:
        self.completions = completions  # request id -> token ids it will generate
        self.tokens_per_step = tokens_per_step
        self.added = {}
        self.running = {}
        self.aborted = []
        self.steps_with_output = 0

    def add_request(self, user_id, request_id, sampling_params, prompt_token_ids, multi_modal_data=None):
        self.added[request_id] = list(prompt_token_ids)
        self.running[request_id] = list(self.completions[request_id])

    def has_unfinished_requests(self) -> bool:
        return bool(self.running)

    def step(self):
        outputs = []
        for rid, left in list(self.running.items()):
            for _ in range(min(self.tokens_per_step, len(left))):
                token = left.pop(0)
                outputs.append(types.SimpleNamespace(request_id=rid, token_id=token, finished=not left,
                                                     prompt_len=len(self.added[rid])))
            if not left:
                del self.running[rid]
        self.steps_with_output += bool(outputs)
        time.sleep(0.002)
        return outputs

    def flush_inflight_reqs(self):
        return []

    def abort_request(self, request_id):
        self.aborted.append(request_id)
        self.running.pop(request_id, None)


def _start_rank(engine, tokenizer=None, **settings):
    requests, results = queue.Queue(), queue.Queue()
    thread = threading.Thread(
        target=de._rank_loop,
        args=(0, engine, tokenizer or ToyTokenizer(), requests, results,
              {"idle_poll_seconds": 0.001, "tokenizer_threads": 2, **settings}),
        kwargs={"clone": lambda raw: raw, "sampling_params": dict, "token_array": list,
                "multi_modal_data": lambda: None},
        daemon=True,
    )
    thread.start()
    assert results.get(timeout=5) == ("ready", 0, None)
    return requests, results, thread


def _collect(results, until_finished, timeout=5.0):
    """Out messages until every request id in ``until_finished`` has finished."""
    messages, done = [], set()
    deadline = time.time() + timeout
    while done < set(until_finished):
        kind, rank, items = results.get(timeout=max(deadline - time.time(), 0.01))
        assert kind == "out" and rank == 0
        messages.append(items)
        done |= {item[0] for item in items if item[4]}
    return messages


def test_rank_tokenizes_detokenizes_and_batches_per_step() -> None:
    tok = ToyTokenizer()
    answers = {"a": "Hello 你好 🎉 world", "b": "naïve </think>done"}
    engine = FakeRankEngine({rid: tok.encode(text) for rid, text in answers.items()}, tokens_per_step=2)
    requests, results, thread = _start_rank(engine, tok)
    requests.put(("add", "a", "prompt A", {"max_tokens": 64}))
    requests.put(("add", "b", "a longer prompt B", {"max_tokens": 64}))
    messages = _collect(results, {"a", "b"})
    requests.put(None)
    thread.join(5)
    assert not thread.is_alive()

    # The prompts were tokenized in the rank and reached the engine as token ids.
    assert engine.added == {"a": tok.encode("prompt A"), "b": tok.encode("a longer prompt B")}
    # One message per engine step with output, and at most one item per request in it.
    assert len(messages) <= engine.steps_with_output
    assert all(len({item[0] for item in items}) == len(items) for items in messages)
    for rid, text in answers.items():
        items = [item for batch in messages for item in batch if item[0] == rid]
        assert "".join(item[1] for item in items) == text
        last = items[-1]
        assert last[2] == len(tok.encode(text))  # completion_tokens, exact
        assert last[3] == len(tok.encode("prompt A" if rid == "a" else "a longer prompt B"))
        assert last[4] and last[5] is None
        assert not any("�" in item[1] for item in items)


def test_rank_aborts_a_request_and_keeps_serving_the_others() -> None:
    tok = ToyTokenizer()
    engine = FakeRankEngine({"long": [65] * 500, "short": tok.encode("ok")})
    requests, results, thread = _start_rank(engine, tok)
    requests.put(("add", "long", "p", {}))
    kind, _, items = results.get(timeout=5)
    assert items[0][0] == "long"
    requests.put(("abort", "long"))
    requests.put(("add", "short", "p", {}))
    messages = _collect(results, {"short"})
    requests.put(None)
    thread.join(5)
    assert engine.aborted == ["long"]
    late = [item for batch in messages for item in batch if item[0] == "long"]
    assert len(late) < 10 and not any(item[4] for item in late)  # stopped, never "finished"


def test_rank_drops_a_request_aborted_while_tokenizing() -> None:
    release = threading.Event()

    class SlowTokenizer(ToyTokenizer):
        def encode(self, text):
            release.wait(5)
            return super().encode(text)

    engine = FakeRankEngine({"gone": [65], "next": [66]})
    requests, results, thread = _start_rank(engine, SlowTokenizer(), tokenizer_threads=1)
    requests.put(("add", "gone", "p", {}))
    requests.put(("abort", "gone"))
    time.sleep(0.05)
    release.set()
    requests.put(("add", "next", "p", {}))
    _collect(results, {"next"})
    requests.put(None)
    thread.join(5)
    assert "gone" not in engine.added and engine.aborted == []


def test_rank_reports_tokenization_and_engine_errors_on_the_request() -> None:
    class BrokenTokenizer(ToyTokenizer):
        def encode(self, text):
            if text == "bad":
                raise ValueError("cannot tokenize")
            return super().encode(text)

    class PickyEngine(FakeRankEngine):
        def add_request(self, user_id, request_id, **kwargs):
            if request_id == "refused":
                raise RuntimeError("prompt too long")
            super().add_request(user_id, request_id, **kwargs)

    engine = PickyEngine({"fine": [65, 66]})
    requests, results, thread = _start_rank(engine, BrokenTokenizer())
    for rid, prompt in (("broken", "bad"), ("refused", "p"), ("fine", "p")):
        requests.put(("add", rid, prompt, {}))
    items = {item[0]: item for batch in _collect(results, {"broken", "refused", "fine"}) for item in batch if item[4]}
    requests.put(None)
    thread.join(5)
    assert "cannot tokenize" in items["broken"][5]
    assert items["refused"][5] == "prompt too long"
    assert items["fine"][5] is None and items["fine"][2] == 2


# --- the parent side, with in-process queues --------------------------------


def _parent_engine(stream_interval: float = 0.0, dp: int = 1):
    engine = object.__new__(de.DeepInferEngine)
    engine._results = queue.Queue()
    engine._requests = [queue.Queue() for _ in range(dp)]
    engine._procs = []
    engine._streams, engine._owner = {}, {}
    engine._in_flight = [0] * dp
    engine._lock = threading.Lock()
    engine._ids = __import__("itertools").count()
    engine._closed = False
    engine._stream_interval = stream_interval
    engine._chunk_budget = 0.0
    engine._reader = threading.Thread(target=engine._read_results, daemon=True)
    engine._reader.start()
    return engine


def test_parent_merges_what_piled_up_into_one_event() -> None:
    engine = _parent_engine()
    gen = engine.generate(de.GenRequest(prompt="p", max_tokens=8))
    events = []
    reader = threading.Thread(target=lambda: events.extend(gen), daemon=True)
    reader.start()
    message = engine._requests[0].get(timeout=5)
    assert message[0] == "add" and message[2] == "p" and message[3]["max_tokens"] == 8
    rid = message[1]
    engine._results.put(("out", 0, [(rid, "Hel", 1, 5, False, None)]))
    time.sleep(0.05)
    engine._results.put(("out", 0, [(rid, "lo ", 2, 5, False, None)]))
    engine._results.put(("out", 0, [(rid, "wor", 3, 5, False, None)]))
    engine._results.put(("out", 0, [(rid, "ld", 4, 5, True, None)]))
    reader.join(5)
    assert "".join(e.text for e in events) == "Hello world"
    assert len(events) < 4  # the last three arrived together and went out together
    assert events[-1].finished and events[-1].completion_tokens == 4 and events[-1].prompt_tokens == 5
    assert engine._in_flight == [0] and not engine._streams


def test_parent_stream_interval_batches_steps_without_delaying_the_end() -> None:
    engine = _parent_engine(stream_interval=0.2)
    gen = engine.generate(de.GenRequest(prompt="p"))
    events = []
    started = time.time()
    reader = threading.Thread(target=lambda: events.extend(gen), daemon=True)
    reader.start()
    rid = engine._requests[0].get(timeout=5)[1]
    for i in range(10):
        engine._results.put(("out", 0, [(rid, "x", i + 1, 3, i == 9, None)]))
        time.sleep(0.005)
    reader.join(5)
    assert "".join(e.text for e in events) == "x" * 10
    assert len(events) <= 3 and events[-1].completion_tokens == 10
    assert time.time() - started < 0.2 + 0.15  # the last item cut the interval short


def test_parent_chunk_budget_stretches_the_interval_with_many_streams() -> None:
    engine = _parent_engine(stream_interval=0.0)
    engine._chunk_budget = 10.0  # one stream may send 10 chunks per second
    gen = engine.generate(de.GenRequest(prompt="p"))
    events = []
    reader = threading.Thread(target=lambda: events.extend(gen), daemon=True)
    reader.start()
    rid = engine._requests[0].get(timeout=5)[1]
    for i in range(20):
        engine._results.put(("out", 0, [(rid, "y", i + 1, 3, i == 19, None)]))
        time.sleep(0.005)
    reader.join(5)
    assert "".join(e.text for e in events) == "y" * 20
    assert len(events) <= 3 and events[-1].completion_tokens == 20


def test_parent_cancels_on_close_and_reports_errors() -> None:
    engine = _parent_engine(dp=2)
    gen = engine.generate(de.GenRequest(prompt="p"))
    holder = []
    reader = threading.Thread(target=lambda: holder.append(next(gen)), daemon=True)
    reader.start()
    _, rid, _, _ = engine._requests[0].get(timeout=5)
    assert engine._in_flight == [1, 0]
    engine._results.put(("out", 0, [(rid, "a", 1, 1, False, None)]))
    reader.join(5)
    gen.close()  # the client went away
    assert engine._requests[0].get(timeout=5) == ("abort", rid)
    assert engine._in_flight == [0, 0] and not engine._streams
    engine._results.put(("out", 0, [(rid, "late", 2, 1, True, None)]))  # ignored

    gen = engine.generate(de.GenRequest(prompt="p"))
    holder = []
    reader = threading.Thread(target=lambda: holder.extend(gen), daemon=True)
    reader.start()
    _, rid, _, _ = engine._requests[0].get(timeout=5)
    engine._results.put(("out", 0, [(rid, "", 0, 0, True, "tokenization failed: boom")]))
    reader.join(5)
    assert holder[-1].finished and holder[-1].error == "tokenization failed: boom"
    assert engine._requests[0].empty()  # a finished request is not aborted


# --- end to end: spawned ranks on the benchmark's simulated DeepInfer ---------


def test_deepinfer_engine_end_to_end_with_simulated_ranks(tmp_path, monkeypatch) -> None:
    import multiprocessing as mp

    fake_modules = _EXAMPLE_DIR / "bench" / "fake_modules"
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        import numpy  # noqa: F401
    except ImportError:  # the ranks only need np.asarray and np.int32
        stub = tmp_path / "stub"
        stub.mkdir()
        (stub / "numpy.py").write_text("int32 = 'int32'\n\ndef asarray(x, dtype=None):\n    return list(x)\n")
        sys.path.append(str(stub))
    sys.path.append(str(fake_modules))  # spawned ranks inherit sys.path
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    ctx = mp.get_context("spawn")
    sim = {"step_rate": 400.0, "tokenizer": "fake", "barrier": ctx.Barrier(2), "flags": ctx.Array("b", 2, lock=False)}
    backend = db.DeepInferBackend()
    backend.setup(BackendContext(config={
        "engine": "deepinfer", "model_dir": str(model), "ep": 2, "dp": 2, "thinking_mode": "chat",
        "llm_kwargs": {"bench_sim": sim}, "load_timeout_seconds": 60, "stream_interval_ms": 10,
    }))
    engine = backend.engine
    try:
        messages = [{"role": "user", "content": "hello there"}]
        out = backend.chat({"messages": messages, "max_tokens": 30})
        usage = out["usage"]
        assert usage["completion_tokens"] == 30 and usage["prompt_tokens"] > 0
        assert out["choices"][0]["message"]["content"].count(" w") == 30  # one fake word per token
        assert out["choices"][0]["finish_reason"] == "length"

        chunks = list(backend.chat_stream({"messages": messages, "max_tokens": 40}))
        content = "".join(c["choices"][0]["delta"].get("content", "") or "" for c in chunks if c["choices"])
        assert content.count(" w") == 40 and chunks[-1]["usage"]["completion_tokens"] == 40

        # Many requests at once spread over both ranks, each with exact usage.
        results = {}

        def one(i: int) -> None:
            results[i] = backend.chat({"messages": [{"role": "user", "content": f"q{i}"}], "max_tokens": 10 + i})

        threads = [threading.Thread(target=one, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert all(results[i]["usage"]["completion_tokens"] == 10 + i for i in range(12))

        # A client that leaves early frees its slot.
        stream = backend.chat_stream({"messages": messages, "max_tokens": 5000})
        next(stream), next(stream)
        stream.close()
        assert engine._in_flight == [0, 0] and not engine._streams
    finally:
        engine.close()
    assert all(not p.is_alive() for p in engine._procs)

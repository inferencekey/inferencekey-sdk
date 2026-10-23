"""Prefix cache of the DeepInfer custom backend: the index, the rank side with the
simulated engine, session routing in the parent, usage, and two turns end to end."""

from __future__ import annotations

import pathlib
import queue
import sys
import threading
import time
import types

import pytest

from inferencekey.backend import BackendContext

_EXAMPLE_DIR = pathlib.Path(__file__).resolve().parents[3] / "examples" / "custom-backend-deepinfer"
for path in (_EXAMPLE_DIR, _EXAMPLE_DIR / "bench" / "fake_modules"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import deepinfer_backend as db  # noqa: E402
import deepinfer_engines as de  # noqa: E402
import prefix_cache as pc  # noqa: E402
from deepinfer.entrypoints.llm import SimEngine  # noqa: E402  (the bench's simulated rank)
from deepinfer.sampling_params import SamplingParams  # noqa: E402


# --- the index ----------------------------------------------------------------


def test_chain_hashes_name_the_whole_prefix() -> None:
    a = list(range(1000))
    b = a[:500] + [7] * 500
    ha, hb = pc.block_hashes(a), pc.block_hashes(b)
    assert len(ha) == 1000 // pc.BLOCK
    shared = 500 // pc.BLOCK
    assert ha[:shared] == hb[:shared] and ha[shared] != hb[shared]
    # The same block content after a different prefix gets a different name.
    c = [9] * pc.BLOCK + a[: 2 * pc.BLOCK]
    assert pc.block_hashes(c)[1] != ha[0]


def test_usable_hit_leaves_the_replay_window_and_skips_tiny_hits() -> None:
    assert pc.usable_hit(10, 10 * pc.BLOCK + 500) == 10 * pc.BLOCK
    # Whole prompt cached: keep at least REPLAY_GUARD tokens to compute.
    assert pc.usable_hit(10, 10 * pc.BLOCK) == (10 * pc.BLOCK - pc.REPLAY_GUARD) // pc.BLOCK * pc.BLOCK
    assert pc.usable_hit(1, 5000) == 0
    assert pc.usable_hit(0, 5000) == 0


class _Pins:
    def __init__(self) -> None:
        self.refs: dict = {}

    def pin(self, blocks) -> None:
        for b in blocks:
            self.refs[b] = self.refs.get(b, 0) + 1

    def unpin(self, blocks) -> None:
        for b in blocks:
            self.refs[b] -= 1
            assert self.refs[b] >= 0, "double unpin"


def test_index_shares_prefixes_and_evicts_leaves_lru() -> None:
    pins = _Pins()
    index = pc.PrefixIndex(pins.pin, pins.unpin)
    sys_prompt = [1] * (3 * pc.BLOCK)
    s1 = pc.block_hashes(sys_prompt + [2] * (2 * pc.BLOCK))
    s2 = pc.block_hashes(sys_prompt + [3] * (2 * pc.BLOCK))
    assert index.insert(s1, [10, 11, 12, 13, 14]) == 5
    assert index.insert(s2, [20, 21, 22, 23, 24]) == 2  # the system prompt is stored once
    assert index.lookup(s2) == [10, 11, 12, 23, 24]
    assert all(pins.refs[b] == 1 for b in (10, 11, 12, 13, 14, 23, 24)) and 20 not in pins.refs
    index.lookup(s2)  # s2 is now the most recent
    assert index.evict(2) == 2
    assert pins.refs[13] == 0 and pins.refs[14] == 0  # s1's own tail went first
    assert index.lookup(s1) == [10, 11, 12]
    # A protected chain (a running sequence uses it) is never evicted.
    index.protect(s2[:5])
    assert index.evict(100) == 0
    index.unprotect(s2[:5])
    assert index.evict(100) == 5 and len(index) == 0
    assert all(n == 0 for n in pins.refs.values())


# --- the rank, with the simulated engine ------------------------------------------


class CharTokenizer:
    """One token per character (prefix-stable, so chain hashing applies)."""

    def encode(self, text: str, add_special_tokens: bool = True):
        return [ord(c) % 5000 for c in text]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(chr(97 + i % 26) for i in ids)

    def __len__(self) -> int:
        return 5000


def _start_rank(engine, kv_blocks: int):
    requests, results = queue.Queue(), queue.Queue()
    settings = {"idle_poll_seconds": 0.001, "tokenizer_threads": 2, "stats_interval_seconds": 0.05,
                "prefix_cache_free_fraction": 0.0}
    cache = de._RankCache(0, engine, settings)
    thread = threading.Thread(
        target=de._rank_loop,
        args=(0, engine, CharTokenizer(), requests, results, settings),
        kwargs={"clone": lambda raw: raw, "sampling_params": SamplingParams, "token_array": list,
                "multi_modal_data": lambda: None, "vocab_size": 5000, "cache": cache},
        daemon=True,
    )
    thread.start()
    assert results.get(timeout=5) == ("ready", 0, None)
    return requests, results, cache


def _run(requests, results, request_id: str, prompt: str, max_tokens: int = 8, timeout: float = 10.0):
    """Send one request; return (cached tokens, prompt tokens, last stats seen)."""
    requests.put(("add", request_id, prompt, {"max_tokens": max_tokens, "session_key": "s"}))
    cached, prompt_len, stats = None, None, None
    deadline = time.time() + timeout
    while True:
        kind, _, payload = results.get(timeout=max(deadline - time.time(), 0.01))
        if kind == "meta":
            cached = dict(payload).get(request_id, cached)
        elif kind == "stats":
            stats = payload
        elif kind == "out":
            for item in payload:
                if item[0] == request_id and item[4]:
                    assert item[5] is None, item
                    return cached, item[3], stats


def _sim(kv_blocks: int) -> SimEngine:
    return SimEngine(0, max_num_seqs=16, max_num_batched_tokens=4096,
                     sim={"step_rate": 2000.0, "kv_blocks": kv_blocks}, token_ids=list(range(100)), vocab_size=5000)


def test_second_turn_starts_from_the_first_turns_prompt() -> None:
    engine = _sim(kv_blocks=1000)
    requests, results, cache = _start_rank(engine, 1000)
    turn1 = "x" * 2000
    cached, prompt_len, _ = _run(requests, results, "t1", turn1)
    assert cached == 0 and prompt_len == 2000
    turn2 = turn1 + "y" * 700
    cached, prompt_len, _ = _run(requests, results, "t2", turn2)
    # Turn 1 cached (2000 - 1) // 128 = 15 blocks; the hit drops the last one (replayed).
    assert cached == 14 * pc.BLOCK and prompt_len == 2700
    assert engine.hit_tokens == cached
    requests.put(None)


def test_eviction_makes_room_and_frees_every_block_at_the_end() -> None:
    engine = _sim(kv_blocks=60)  # 60 blocks: three 2,000-token sessions do not fit together
    requests, results, cache = _start_rank(engine, 60)
    for i in range(4):
        _run(requests, results, f"s{i}", chr(65 + i) * 2000)
    assert cache.index.evicted > 0
    # The newest session is still cached.
    cached, _, _ = _run(requests, results, "s3b", chr(68) * 2000 + "z" * 300)
    assert cached > 0
    requests.put(None)
    time.sleep(0.05)
    cache.index.clear()
    assert engine.block_manager.get_num_free_blocks() == 60


def test_aborted_request_releases_its_protection_and_blocks() -> None:
    engine = _sim(kv_blocks=200)
    requests, results, cache = _start_rank(engine, 200)
    _run(requests, results, "a", "q" * 1500)
    requests.put(("add", "b", "q" * 1500 + "w" * 200, {"max_tokens": 100000, "session_key": "s"}))
    deadline = time.time() + 5
    while "b" not in cache.live and time.time() < deadline:
        time.sleep(0.005)
    assert cache.index.protected
    requests.put(("abort", "b"))
    deadline = time.time() + 5
    while cache.index.protected and time.time() < deadline:
        time.sleep(0.005)
    assert not cache.index.protected and "b" not in cache.live
    requests.put(None)


# --- the parent: session routing ---------------------------------------------------


def _router(dp: int = 4):
    engine = object.__new__(de.DeepInferEngine)
    engine._in_flight = [0] * dp
    engine._affinity = True
    engine._sessions = de.OrderedDict()
    engine._max_sessions = 3
    engine._migrate_at = 2
    engine._used_blocks = [0] * dp
    engine._assigned = [0] * dp
    engine.migrations = 0
    return engine


def test_router_keeps_a_session_on_its_rank_and_balances_new_ones_by_kv() -> None:
    engine = _router()
    engine._used_blocks = [500, 100, 300, 900]
    req = de.GenRequest(prompt="p" * 3000, session_key="alpha")
    rank = engine._route(req)
    assert rank == 1
    engine._used_blocks[1] = 10_000  # even when its rank fills up, the session stays
    assert engine._route(req) == 1
    # Busy past session_migrate_inflight: the session moves (one cold turn).
    engine._in_flight[1] = 2
    assert engine._route(req) != 1 and engine.migrations == 1
    # The session table is an LRU with a cap.
    for key in ("b", "c", "d"):
        engine._route(de.GenRequest(prompt="p", session_key=key))
    assert "alpha" not in engine._sessions and len(engine._sessions) == 3


def test_session_key_is_the_user_field_or_the_conversation_head() -> None:
    turn1 = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "first"}]}
    turn2 = {"messages": turn1["messages"] + [{"role": "assistant", "content": "ok"},
                                              {"role": "user", "content": "second"}]}
    other = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "else"}]}
    assert db.session_key(turn1) == db.session_key(turn2) != db.session_key(other)
    assert db.session_key({**turn1, "user": "dev-7"}) == "u:dev-7"


def test_usage_reports_cached_tokens_only_when_the_engine_has_a_cache() -> None:
    with_cache = db.DeepInferBackend._usage(de.GenEvent(prompt_tokens=100, completion_tokens=5, cached_tokens=64))
    assert with_cache["prompt_tokens_details"] == {"cached_tokens": 64}
    assert "prompt_tokens_details" not in db.DeepInferBackend._usage(de.GenEvent(prompt_tokens=1))


# --- end to end over simulated ranks -------------------------------------------------


def test_two_turns_end_to_end_report_cached_tokens(tmp_path, monkeypatch) -> None:
    import multiprocessing as mp

    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        import numpy  # noqa: F401
    except ImportError:
        pytest.skip("needs numpy for the spawned ranks")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    ctx = mp.get_context("spawn")
    sim = {"step_rate": 400.0, "tokenizer": "fake", "kv_blocks": 5000,
           "barrier": ctx.Barrier(2), "flags": ctx.Array("b", 2, lock=False)}
    backend = db.DeepInferBackend()
    backend.setup(BackendContext(config={
        "engine": "deepinfer", "model_dir": str(model), "ep": 2, "dp": 2, "thinking_mode": "chat",
        "llm_kwargs": {"bench_sim": sim}, "load_timeout_seconds": 60, "stream_interval_ms": 10,
    }))
    engine = backend.engine
    try:
        messages = [{"role": "system", "content": "repository " * 400}, {"role": "user", "content": "go"}]
        first = backend.chat({"messages": messages, "max_tokens": 10, "user": "dev-1"})
        assert first["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        messages += [first["choices"][0]["message"], {"role": "user", "content": "and then " * 50}]
        chunks = list(backend.chat_stream({"messages": messages, "max_tokens": 10, "user": "dev-1"}))
        usage = chunks[-1]["usage"]
        assert usage["prompt_tokens_details"]["cached_tokens"] >= 4 * pc.BLOCK
        assert usage["prompt_tokens_details"]["cached_tokens"] < usage["prompt_tokens"]
        deadline = time.time() + 10  # the ranks report every stats_interval_seconds
        while sum(s.get("hits", 0) for s in engine.rank_stats) < 1 and time.time() < deadline:
            time.sleep(0.05)
        assert sum(s.get("hits", 0) for s in engine.rank_stats) == 1
    finally:
        engine.close()

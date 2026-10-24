"""L1 host tier of the DeepInfer prefix cache: the host index, the prefix index hooks
it needs, demotion and promotion in the rank with the simulated engine and mover,
failure fallbacks, the parent side (meta, budget, log line) and the real mover's
tensor logic on CPU tensors."""

from __future__ import annotations

import pathlib
import queue
import sys
import threading
import time

import pytest

_EXAMPLE_DIR = pathlib.Path(__file__).resolve().parents[3] / "examples" / "custom-backend-deepinfer"
_FAKE_MODULES = _EXAMPLE_DIR / "bench" / "fake_modules"
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

import deepinfer_engines as de  # noqa: E402
import kv_mover as km  # noqa: E402
import l1_cache as lc  # noqa: E402
import prefix_cache as pc  # noqa: E402


def _import_sim():
    """The bench's simulated rank, imported without leaving its stand-in torch
    and deepinfer on sys.path or in sys.modules (other tests skip when torch
    is missing and must keep doing so)."""
    before = set(sys.modules)
    sys.path.append(str(_FAKE_MODULES))
    try:
        from deepinfer.entrypoints import llm as sim_llm
        from deepinfer.sampling_params import SamplingParams
    finally:
        sys.path.remove(str(_FAKE_MODULES))
        for name in set(sys.modules) - before:
            if name.split(".")[0] in ("deepinfer", "torch", "bench_fake_tokenizer"):
                del sys.modules[name]
    return sim_llm, SamplingParams


sim_llm, SamplingParams = _import_sim()
SimEngine = sim_llm.SimEngine
B = pc.BLOCK


# --- helpers ------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _engine(kv_blocks: int, host_blocks: int = 1000, gbps: float = 50.0, **extra) -> SimEngine:
    sim = {"step_rate": 5000.0, "kv_blocks": kv_blocks, "l1_pcie_gbps": gbps, "l1_host_blocks": host_blocks,
           "l1_copy_latency_us": 10.0, **extra}
    return SimEngine(0, max_num_seqs=16, max_num_batched_tokens=1 << 20, sim=sim, token_ids=list(range(100)),
                     vocab_size=5000)


def _cache(engine: SimEngine, l1: bool = True, **settings) -> "de._RankCache":
    cache = de._RankCache(0, engine, {"prefix_cache_free_fraction": 0.0, **settings})
    if l1:
        info = cache.start_l1(0)  # the simulated mover takes l1_host_blocks
        assert info["l1"], info
    return cache


def _ids(session: int, blocks: int, extra: int = 0) -> list:
    """Token ids of a session prompt: ``blocks`` full blocks plus ``extra`` tokens."""
    return [(session * 7919 + i) % 4000 for i in range(blocks * B + extra)]


def _cached_chain(engine: SimEngine, cache: "de._RankCache", ids: list) -> list:
    """Put a finished prompt's blocks into L0 the way the engine would: allocate,
    'prefill' (write the content tags), insert, drop the sequence's reference."""
    hashes = pc.block_hashes(ids)
    bm = engine.block_manager
    blocks = bm.allocate(len(hashes))
    for block, tag in zip(blocks, sim_llm._block_tags(ids)):
        bm.kv_tag[block] = tag
    cache.index.insert(hashes, blocks)
    bm.unpin_blocks(blocks)
    return blocks


def _adder(engine: SimEngine, request_id: str, ids: list, max_tokens: int = 4):
    def add(enqueue: bool = True):
        return engine.add_request(user_id="u", request_id=request_id,
                                  sampling_params=SamplingParams(max_tokens=max_tokens),
                                  prompt_token_ids=ids, enqueue=enqueue)
    return add


def _add(engine: SimEngine, cache: "de._RankCache", request_id: str, ids: list, max_tokens: int = 4):
    return cache.add(_adder(engine, request_id, ids, max_tokens), request_id, ids, pc.block_hashes(ids), max_tokens)


def _evict_all(cache: "de._RankCache") -> None:
    cache.make_room(cache.total_blocks * 10)


# --- the host index ---------------------------------------------------------------------


def test_host_pool_hands_out_largest_runs_and_merges_on_release() -> None:
    pool = lc.HostPool(100)
    a = pool.alloc(30)
    assert a == list(range(30)) and pool.free == 70
    b = pool.alloc(20)
    assert b == list(range(30, 50))
    pool.release(a[10:20])
    assert pool.runs() == 2  # [10, 20) and [50, 100)
    c = pool.alloc(55)  # largest run first: 50 from [50, 100), then 5 from [10, 20)
    assert c[:50] == list(range(50, 100)) and len(c) == 55 and pool.free == 5
    pool.release(c)
    pool.release(b)
    pool.release(a[:10] + a[20:])
    assert pool.runs() == 1 and pool.free == 100
    assert pool.alloc(200) == list(range(100))


class _FakeMover:
    """Done after ``polls`` calls to done(); records what was copied."""

    def __init__(self, capacity: int, polls: int = 0) -> None:
        self.bytes_per_block = 1000
        self.capacity_slots = capacity
        self.polls = polls
        self.issued = []

    def d2h(self, gpu_ids, slots, checksum=False):
        ticket = {"left": self.polls, "n": len(gpu_ids)}
        self.issued.append(("d2h", list(gpu_ids), list(slots)))
        return ticket

    def h2d(self, slots, gpu_ids, checksum=False):
        ticket = {"left": self.polls, "n": len(gpu_ids)}
        self.issued.append(("h2d", list(slots), list(gpu_ids)))
        return ticket

    def done(self, ticket):
        if ticket["left"] > 0:
            ticket["left"] -= 1
            return False
        return True

    def checksums(self, ticket):
        return None

    def elapsed(self, ticket):
        return 0.001


def test_host_index_lru_is_leaf_first_and_continuation_starts_at_the_l0_boundary() -> None:
    unpinned = []
    l1 = lc.L1(_FakeMover(capacity=6), unpinned.extend)
    a = pc.block_hashes(_ids(1, 4))
    b = pc.block_hashes(_ids(2, 4))
    # L0 hands leaves over deepest first.
    l1.demote(a[::-1], [13, 12, 11, 10], [3, 2, 1, 0])
    assert l1.continuation(a, 0) == 4 and l1.continuation(a, 1) == 3
    assert not unpinned  # pinned until the copies completed
    l1.poll()
    assert sorted(unpinned) == [10, 11, 12, 13]
    assert [l1.entries[h].state for h in a] == [lc.READY] * 4
    # b needs 4 slots, 2 are free: the two deepest of a (the LRU end) go.
    l1.demote(b[::-1], [23, 22, 21, 20], [3, 2, 1, 0])
    assert l1.counters["l1_evicted"] == 2
    assert all(h in l1.entries for h in a[:2]) and not any(h in l1.entries for h in a[2:])
    assert l1.continuation(a, 0) == 2
    # A block already in L1 is just unpinned again (inclusive L1).
    l1.poll()
    unpinned.clear()
    l1.demote([a[0]], [99], [0])
    assert unpinned == [99] and l1.counters["l1_demote_skipped"] == 1


def test_host_index_keeps_the_shallowest_blocks_when_it_cannot_take_all() -> None:
    unpinned = []
    mover = _FakeMover(capacity=2, polls=5)
    l1 = lc.L1(mover, unpinned.extend)
    a = pc.block_hashes(_ids(3, 4))
    l1.demote(a[:2][::-1], [11, 10], [1, 0])  # in flight: WRITING entries are never evicted
    l1.demote(a[2:][::-1], [13, 12], [3, 2])
    assert sorted(unpinned) == [12, 13] and l1.counters["l1_dropped"] == 2
    mover2 = _FakeMover(capacity=2)
    l1b = lc.L1(mover2, unpinned.extend)
    unpinned.clear()
    l1b.demote(a[::-1], [13, 12, 11, 10], [3, 2, 1, 0])
    assert sorted(unpinned) == [12, 13]  # the deepest were dropped, the shallowest copied
    assert mover2.issued[0][1] == [11, 10]


# --- the prefix index hooks ----------------------------------------------------------------


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


def test_insert_adopts_a_restored_suffix_without_touching_the_l0_prefix() -> None:
    pins = _Pins()
    index = pc.PrefixIndex(pins.pin, pins.unpin)
    h = pc.block_hashes(_ids(4, 6))
    index.insert(h[:2], [10, 11])
    other = pc.block_hashes(_ids(4, 6))[:5]
    index.insert(other, [10, 11, 50, 51, 52])  # someone else cached positions 2..4 meanwhile
    before = dict(pins.refs)
    dups: list = []
    added = index.insert(h, [10, 11, 60, 61, 62, 63], adopt_from=2, duplicates=dups)
    assert added == 1 and dups == [60, 61, 62]  # each duplicate exactly once
    assert pins.refs == before  # blocks0 untouched, the adopted block 63 not pinned again
    assert index.lookup(h) == [10, 11, 50, 51, 52, 63]


def test_evict_hands_leaves_to_demote_with_their_depths() -> None:
    pins = _Pins()
    index = pc.PrefixIndex(pins.pin, pins.unpin)
    h = pc.block_hashes(_ids(5, 4))
    index.insert(h, [10, 11, 12, 13])
    got = []
    assert index.evict(3, demote=lambda hs, ids, depths: got.append((hs, ids, depths))) == 3
    assert got == [(h[3:0:-1], [13, 12, 11], [3, 2, 1])]
    assert pins.refs[13] == 1  # the pins passed to demote, not released
    assert index.lookup(h) == [10]


def test_evict_is_incremental() -> None:
    pins = _Pins()
    index = pc.PrefixIndex(lambda b: None, lambda b: None)
    n = 226_000
    for s in range(n // 100):
        hashes = [bytes([s % 256, s // 256, i]) + b"x" * 13 for i in range(100)]
        index.insert(hashes, range(s * 100, s * 100 + 100))
    t0 = time.perf_counter()
    assert index.evict(8000) == 8000
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05, elapsed  # the design target is 5 ms; generous for a loaded CI machine
    del pins


# --- demotion and promotion in the simulated rank ----------------------------------------------


def test_no_block_is_unpinned_before_its_demotion_copy_completed() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    mover = engine.l1_kv_mover
    clock = mover.clock = _Clock()
    ids = _ids(1, 20, 50)
    blocks = _cached_chain(engine, cache, ids)
    _evict_all(cache)
    assert len(cache.index) == 0 and cache.l1.demoting
    bm = engine.block_manager
    assert all(bm.refs.get(b) == 1 for b in blocks)  # still pinned, now by the pending demotion
    cache.tick([])
    assert all(bm.refs.get(b) == 1 for b in blocks)
    clock.t += 1.0
    cache.tick([])
    assert not any(b in bm.refs for b in blocks) and not cache.l1.demoting
    assert all(e.state == lc.READY for e in cache.l1.entries.values())
    assert bm.get_num_free_blocks() == 100


def test_an_early_unpin_is_caught_by_the_content_tags(monkeypatch) -> None:
    """The simulated rank must notice the bug L1 is designed to avoid."""
    engine = _engine(kv_blocks=60)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    ids = _ids(1, 20, 50)
    _cached_chain(engine, cache, ids)
    original = lc.L1._copy_out

    def buggy(self, need, slots, unpin_now):
        original(self, need, slots, unpin_now)
        for copy in self.demoting:
            self.unpin(copy.gpu_ids)  # unpinned at issue instead of at completion
            for h in copy.hashes:
                self.entries[h].owner = None

    monkeypatch.setattr(lc.L1, "_copy_out", buggy)
    _evict_all(cache)
    engine.block_manager.allocate(55)  # the freed blocks are reused (and poisoned) before the copy lands
    clock.t += 1.0
    cache.tick([])
    engine.block_manager.unpin_blocks(list(engine.block_manager.refs))
    seq, _ = _add(engine, cache, "r", ids + [1] * 300)
    clock.t += 1.0
    out = cache.tick([])
    assert out and engine.kv_mismatch > 0


def test_a_returning_session_hits_through_l1_after_its_copy_completed() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    ids = _ids(1, 20, 50)
    _cached_chain(engine, cache, ids)
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    assert len(cache.index) == 0 and len(cache.l1.entries) == 20
    turn2 = ids + [7] * 400
    seq, cached = _add(engine, cache, "t2", turn2)
    assert seq is de.PROMOTING and cached == 0
    # Not admitted while the copy is in flight.
    assert cache.tick([]) == [] and not engine.waiting
    clock.t += 1.0
    (request_id, seq, cached, l1_tokens), = cache.tick([])
    hit = pc.usable_hit(20, len(turn2))
    assert request_id == "t2" and seq is not None and seq in engine.waiting
    assert cached == hit - B  # the hit computed at lookup, not hit - 256
    assert l1_tokens == cached  # L0 was empty: everything came from L1
    assert engine.kv_mismatch == 0 and engine.hit_tokens == cached
    stats = cache.stats()
    assert stats["l1_promotions"] == 1 and stats["l1_promoted_blocks"] == hit // B - 1
    assert stats["l1_hits"] == 1 and stats["l1_hit_tokens"] == cached and stats["hits"] == 1
    # The restored chain is in L0 now, and the host copies stay (inclusive).
    assert len(cache.index) == hit // B - 1 and len(cache.l1.entries) == 20
    assert not cache.promoting and not cache.l1.restoring
    assert all(e.busy == 0 for e in cache.l1.entries.values())


def test_l0_hit_continued_in_l1_promotes_only_the_missing_blocks() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    ids = _ids(2, 20, 50)
    blocks = _cached_chain(engine, cache, ids)
    cache.index.evict(12, demote=cache.l1.demote)  # the 12 deepest go to L1
    clock.t += 1.0
    cache.tick([])
    assert len(cache.index) == 8
    refs_before = {b: engine.block_manager.refs[b] for b in blocks[:8]}
    turn2 = ids + [9] * 400
    assert _add(engine, cache, "t2", turn2)[0] is de.PROMOTING
    clock.t += 1.0
    (_, seq, cached, l1_tokens), = cache.tick([])
    assert cached == 19 * B and l1_tokens == 11 * B  # positions 8..18; block 19 is replayed
    assert engine.kv_mismatch == 0
    # blocks0 carry their L0 pin plus the sequence's own reference, nothing more.
    assert all(engine.block_manager.refs[b] == refs_before[b] + 1 for b in blocks[:8])
    assert seq.block_table[:8] == blocks[:8]


def test_admission_waits_for_free_blocks_then_times_out_to_recompute() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine, l1_promote_wait_seconds=0.05)
    clock = engine.l1_kv_mover.clock = _Clock()
    ids = _ids(3, 20, 50)
    _cached_chain(engine, cache, ids)
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    held = engine.block_manager.allocate(80)  # running sequences hold most of the pool
    turn2 = ids + [5] * 300
    assert _add(engine, cache, "t2", turn2)[0] is de.PROMOTING
    assert cache.tick([]) == []  # no blocks yet
    time.sleep(0.06)
    (_, seq, cached, l1_tokens), = cache.tick([])
    assert cached == 0 and l1_tokens == 0 and seq in engine.waiting  # recomputes
    assert cache.l1.counters["l1_promote_timeouts"] == 1
    assert not cache.l1.restoring and all(e.busy == 0 for e in cache.l1.entries.values())
    engine.block_manager.unpin_blocks(held)


def test_a_promotion_never_overruns_the_free_list() -> None:
    """With an almost empty pool the free check under the lock prevents the
    halfway throw of allocate_nolock."""
    engine = _engine(kv_blocks=40)
    cache = _cache(engine, l1_promote_wait_seconds=10.0)
    clock = engine.l1_kv_mover.clock = _Clock()
    ids = _ids(4, 20, 50)
    _cached_chain(engine, cache, ids)
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    held = engine.block_manager.allocate(30)
    assert _add(engine, cache, "t2", ids + [3] * 300)[0] is de.PROMOTING
    for _ in range(3):
        assert cache.tick([]) == []
    assert engine.block_manager.get_num_free_blocks() == 10  # nothing leaked
    engine.block_manager.unpin_blocks(held)
    assert cache.tick([]) == []  # room now: the copy is issued
    clock.t += 1.0
    (_, seq, cached, _), = cache.tick([])
    assert seq is not None and cached == 19 * B and engine.kv_mismatch == 0


def test_copy_failure_before_issue_falls_back_cleanly() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    mover = engine.l1_kv_mover
    clock = mover.clock = _Clock()
    ids = _ids(5, 20, 50)
    _cached_chain(engine, cache, ids)
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    mover.fail_before_issue = True
    seq, cached = _add(engine, cache, "t2", ids + [1] * 300)
    assert seq is not de.PROMOTING and seq in engine.waiting and cached == 0
    assert cache.l1.counters["l1_fallbacks"] == 1
    assert not cache.promoting and not cache.l1.restoring and not cache.index.protected
    assert all(e.busy == 0 for e in cache.l1.entries.values())
    # Only the new sequence holds blocks (its prompt is allocated at admission, not yet).
    assert engine.block_manager.get_num_free_blocks() == 100


def test_partial_copy_failure_quarantines_until_the_copy_completed() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    mover = engine.l1_kv_mover
    clock = mover.clock = _Clock()
    ids = _ids(6, 20, 50)
    _cached_chain(engine, cache, ids)
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    mover.staging_rows = 4
    mover.fail_after_chunks = 2
    seq, cached = _add(engine, cache, "t2", ids + [1] * 300)
    assert seq is not de.PROMOTING and cached == 0  # admitted at once, recomputes
    bm = engine.block_manager
    q_blocks, _ = cache.l1.quarantined()
    assert q_blocks == 19 and bm.get_num_free_blocks() == 100 - 19
    busy = [e for e in cache.l1.entries.values() if e.busy]
    assert len(busy) == 19  # their slots may still be read
    cache.tick([])
    assert cache.l1.quarantined() == (19, 0)
    clock.t += 1.0
    cache.tick([])
    assert cache.l1.quarantined() == (0, 0) and bm.get_num_free_blocks() == 100
    assert all(e.busy == 0 for e in cache.l1.entries.values())


def test_a_big_demotion_is_split_and_a_full_index_ring_defers_it(monkeypatch) -> None:
    engine = _engine(kv_blocks=100)
    mover = engine.l1_kv_mover
    mover.max_ticket_blocks = 6  # like the real mover: half its index ring
    cache = _cache(engine)
    clock = mover.clock = _Clock()
    ids = _ids(8, 20, 50)
    blocks = _cached_chain(engine, cache, ids)
    issued = []
    original = mover.d2h
    full = {"on": True}

    def d2h(gpu_ids, slots, checksum=False):
        if full["on"] and issued:
            raise km.RingFull("the L1 index ring is full")
        issued.append(len(gpu_ids))
        return original(gpu_ids, slots, checksum)

    monkeypatch.setattr(mover, "d2h", d2h)
    _evict_all(cache)
    bm = engine.block_manager
    assert issued == [6] and len(cache.l1.deferred) == 3
    assert cache.l1.pending_d2h_blocks == 20 and all(bm.refs.get(b) == 1 for b in blocks)
    cache.tick([])  # still full: nothing unpinned, order kept
    assert issued == [6] and len(cache.l1.deferred) == 3
    full["on"] = False
    clock.t += 1.0
    cache.tick([])  # the first copy completed, the rest is issued now
    assert issued == [6, 6, 6, 2] and not cache.l1.deferred
    assert sum(1 for b in blocks if b in bm.refs) == 14
    clock.t += 1.0
    cache.tick([])
    assert not any(b in bm.refs for b in blocks) and cache.l1.pending_d2h_blocks == 0
    assert all(e.state == lc.READY for e in cache.l1.entries.values())
    assert cache.l1.counters["l1_fallbacks"] == 0 and cache.l1.counters["l1_dropped"] == 0
    # A returning session longer than one copy recomputes instead of promoting.
    seq, cached = _add(engine, cache, "t2", ids + [1] * 300)
    assert seq is not de.PROMOTING and cached == 0


def test_a_promotion_retries_when_the_index_ring_is_full(monkeypatch) -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    mover = engine.l1_kv_mover
    clock = mover.clock = _Clock()
    ids = _ids(9, 20, 50)
    _cached_chain(engine, cache, ids)
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    original = mover.h2d
    full = {"on": True}

    def h2d(slots, gpu_ids, checksum=False):
        if full["on"]:
            raise km.RingFull("the L1 index ring is full")
        return original(slots, gpu_ids, checksum)

    monkeypatch.setattr(mover, "h2d", h2d)
    assert _add(engine, cache, "t2", ids + [1] * 300)[0] is de.PROMOTING
    bm = engine.block_manager
    assert bm.get_num_free_blocks() == 100 - 19  # the destination blocks stay ours
    assert cache.tick([]) == [] and bm.get_num_free_blocks() == 100 - 19  # not allocated twice
    full["on"] = False
    assert cache.tick([]) == []  # issued now
    clock.t += 1.0
    (_, seq, cached, l1_tokens), = cache.tick([])
    assert cached == 19 * B and l1_tokens == cached and engine.kv_mismatch == 0
    assert cache.l1.counters["l1_fallbacks"] == 0


def test_rescue_takes_over_blocks_whose_demotion_is_still_running() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    ids = _ids(7, 20, 50)
    blocks = _cached_chain(engine, cache, ids)
    _evict_all(cache)  # the demotion copy is still running
    assert _add(engine, cache, "t2", ids + [1] * 300)[0] is de.PROMOTING
    (_, seq, cached, l1_tokens), = cache.tick([])  # nothing to copy: rescued
    assert cached == 19 * B and l1_tokens == 19 * B and engine.kv_mismatch == 0
    assert cache.l1.counters["l1_rescued_blocks"] == 19 and cache.l1.counters["l1_promoted_blocks"] == 0
    bm = engine.block_manager
    assert all(bm.refs[b] == 2 for b in blocks[:19])  # L0 pin + the sequence
    assert bm.refs[blocks[19]] == 1  # not rescued: its demotion still holds it
    clock.t += 1.0
    cache.tick([])
    assert all(bm.refs[b] == 2 for b in blocks[:19]) and blocks[19] not in bm.refs
    assert all(e.state == lc.READY for e in cache.l1.entries.values())


def test_a_rescued_block_evicted_again_is_unpinned_by_its_own_demotion() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    ids = _ids(8, 20, 50)
    blocks = _cached_chain(engine, cache, ids)
    _evict_all(cache)
    assert _add(engine, cache, "t2", ids + [1] * 300)[0] is de.PROMOTING
    (_, seq, _, _), = cache.tick([])
    cache.forget("t2")
    engine.abort_request("t2")  # the sequence drops its references
    _evict_all(cache)  # L0 evicts the rescued blocks while their first copy still runs
    bm = engine.block_manager
    assert all(bm.refs.get(b) == 1 for b in blocks[:19])
    clock.t += 1.0
    cache.tick([])
    assert bm.get_num_free_blocks() == 100


def test_a_second_promotion_of_the_same_prefix_waits_for_the_first() -> None:
    engine = _engine(kv_blocks=200)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    ids = _ids(9, 20, 50)
    _cached_chain(engine, cache, ids)
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    assert _add(engine, cache, "a", ids + [1] * 300)[0] is de.PROMOTING
    assert _add(engine, cache, "b", ids + [2] * 300)[0] is de.PROMOTING
    assert cache.promoting["b"].waits_for == {"a"} and cache.promoting["b"].ticket is None
    clock.t += 1.0
    out = {item[0]: item for item in cache.tick([])}
    assert set(out) == {"a", "b"}
    assert out["a"][2] == out["b"][2] == 19 * B
    assert out["a"][3] == 19 * B and out["b"][3] == 0  # b hit through L0 after a restored it
    assert cache.l1.counters["l1_promoted_blocks"] == 19 and engine.kv_mismatch == 0


def test_abort_during_the_copy_keeps_the_restored_chain() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    ids = _ids(10, 20, 50)
    _cached_chain(engine, cache, ids)
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    assert _add(engine, cache, "t2", ids + [1] * 300)[0] is de.PROMOTING
    cache.forget("t2")
    clock.t += 1.0
    assert cache.tick([]) == []
    assert len(cache.index) == 19 and not engine.waiting and not cache.index.protected


def test_l1_polling_runs_with_an_empty_l0_and_nothing_pending() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    _cached_chain(engine, cache, _ids(11, 20, 50))
    _evict_all(cache)
    assert len(cache.index) == 0
    clock.t += 1.0
    cache.tick([])  # nothing pending, L0 empty: the copies are still polled
    assert not cache.l1.demoting and engine.block_manager.get_num_free_blocks() == 100


def test_verify_compares_restored_bytes_with_the_demoted_and_cached_ones() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine, l1_verify_fraction=1.0)
    clock = engine.l1_kv_mover.clock = _Clock()
    ids = _ids(12, 20, 50)
    turn1 = _adder(engine, "t1", ids)
    seq, _ = cache.add(turn1, "t1", ids, pc.block_hashes(ids), 4)
    for _ in range(50):
        engine.step()  # prefill writes the tags; finalize inserts into L0 and samples checksums
        if seq.generated >= 4:
            break
    clock.t += 1.0
    cache.tick([])
    assert len(cache.l1.l0_sums) == 20
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    assert _add(engine, cache, "t2", ids + [1] * 300)[0] is de.PROMOTING
    clock.t += 1.0
    (_, seq2, cached, _), = cache.tick([])
    assert cached == 19 * B and cache.l1.counters["l1_verify_failures"] == 0
    assert cache.l1.counters["l1_checksums"] >= 39
    # A corrupted host slot is caught, L1 turns itself off and the request recomputes.
    engine.abort_request("t2")
    cache.forget("t2")
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    entry = cache.l1.entries[pc.block_hashes(ids)[3]]
    engine.l1_kv_mover.host_tag[entry.slot] ^= 1
    assert _add(engine, cache, "t3", ids + [2] * 300)[0] is de.PROMOTING
    clock.t += 1.0
    (_, seq3, cached, _), = cache.tick([])
    assert cached == 0 and cache.l1.counters["l1_verify_failures"] == 1 and not cache.l1.enabled


def _recompute_into_l0(engine: SimEngine, cache: "de._RankCache", ids: list) -> list:
    """Another request recomputes ``ids`` with other bytes (batch nondeterminism),
    and L0 caches that block with a sampled checksum."""
    hashes = pc.block_hashes(ids)
    bm = engine.block_manager
    blocks = bm.allocate(len(hashes))
    for block, tag in zip(blocks, sim_llm._block_tags(ids)):
        bm.kv_tag[block] = tag ^ 1
    cache.index.insert(hashes, blocks)
    bm.unpin_blocks(blocks)
    cache.l1.checksum_l0(hashes, blocks)
    return blocks


def test_verify_never_compares_a_slot_with_another_block_of_the_same_hash() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    l1 = cache.l1
    ids = _ids(1, 20, 50)
    _cached_chain(engine, cache, ids)  # turn 1, no checksum sampled
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])  # demoted unsampled: the slots carry no checksum
    assert len(l1.entries) == 20 and all(e.checksum is None for e in l1.entries.values())
    _recompute_into_l0(engine, cache, ids)
    clock.t += 1.0
    cache.tick([])
    assert len(l1.l0_sums) == 20
    _evict_all(cache)  # L1 already holds these hashes: unpinned, and their L0 sums go with them
    clock.t += 1.0
    cache.tick([])
    assert not l1.l0_sums
    l1.verify_fraction = 1.0
    assert _add(engine, cache, "t2", ids + [7] * 400)[0] is de.PROMOTING
    clock.t += 1.0
    (_, seq, cached, l1_tokens), = cache.tick([])
    assert l1.counters["l1_verify_failures"] == 0 and l1.enabled
    assert cached == 19 * B and l1_tokens == cached and engine.kv_mismatch == 0


def test_verify_at_demotion_uses_the_l0_sum_of_the_copied_block_only() -> None:
    engine = _engine(kv_blocks=100, host_blocks=20)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    l1 = cache.l1
    ids = _ids(2, 20, 50)
    first = _recompute_into_l0(engine, cache, ids)  # block X per hash, sum sampled
    clock.t += 1.0
    cache.tick([])
    assert l1.l0_sums[pc.block_hashes(ids)[0]] == (first[0], engine.block_manager.kv_tag[first[0]])
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    assert all(e.checksum is not None for e in l1.entries.values())  # the L0 sum of the very block copied
    # L1 loses the chain (another session takes its slots), and the prefix is
    # recomputed with other bytes into L0, unsampled this time.
    _cached_chain(engine, cache, _ids(3, 20, 50))
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    assert not any(h in l1.entries for h in pc.block_hashes(ids))
    _cached_chain(engine, cache, ids)
    l1.verify_fraction = 1.0
    _evict_all(cache)  # sampled demotion: there is no L0 sum of this block to compare with
    clock.t += 1.0
    cache.tick([])
    assert l1.counters["l1_verify_failures"] == 0 and l1.enabled
    # A block that does not hold what the prefill wrote into it is still caught.
    ids4 = _ids(4, 20, 50)
    blocks4 = _recompute_into_l0(engine, cache, ids4)
    clock.t += 1.0
    cache.tick([])
    engine.block_manager.kv_tag[blocks4[5]] ^= 2
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    assert l1.counters["l1_verify_failures"] == 1 and not l1.enabled and l1.suspect


def test_make_room_counts_only_blocks_their_demotion_will_free() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    l1 = cache.l1
    ids = _ids(5, 20, 50)
    _cached_chain(engine, cache, ids)
    _evict_all(cache)
    assert l1.pending_d2h_blocks == 20 and l1.freeing_blocks == 20
    assert _add(engine, cache, "t2", ids + [1] * 300)[0] is de.PROMOTING
    (_, seq, cached, _), = cache.tick([])  # 19 blocks rescued: L0 and the sequence hold them now
    assert cached == 19 * B
    assert l1.pending_d2h_blocks == 20 and l1.freeing_blocks == 1
    # The sequence ends and L0 evicts the rescued blocks again: their copy unpins them.
    cache.forget("t2")
    engine.abort_request("t2")
    _evict_all(cache)
    assert l1.freeing_blocks == 20
    clock.t += 1.0
    cache.tick([])
    assert l1.pending_d2h_blocks == 0 and l1.freeing_blocks == 0
    # A stuck copy may never complete: its blocks are not counted either.
    engine2 = _engine(kv_blocks=100)
    cache2 = _cache(engine2, l1_copy_timeout_seconds=0.0)
    clock2 = engine2.l1_kv_mover.clock = _Clock()
    _cached_chain(engine2, cache2, ids)
    _evict_all(cache2)
    cache2.tick([])
    assert cache2.l1.counters["l1_stuck"] == 1
    assert cache2.l1.pending_d2h_blocks == 20 and cache2.l1.freeing_blocks == 0
    clock2.t += 1.0
    cache2.tick([])
    assert cache2.l1.pending_d2h_blocks == 0 and cache2.l1.freeing_blocks == 0
    assert engine2.block_manager.get_num_free_blocks() == 100


def test_a_failing_promotion_start_leaves_nothing_held() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    ids = _ids(6, 20, 50)
    blocks = _cached_chain(engine, cache, ids)
    cache.index.evict(12, demote=cache.l1.demote)
    clock.t += 1.0
    cache.tick([])
    bm = engine.block_manager

    def broken(n):
        raise RuntimeError("allocator broke")

    bm.allocate_nolock = broken
    with pytest.raises(RuntimeError, match="allocator broke"):
        _add(engine, cache, "t2", ids + [1] * 300)
    assert not cache.promoting and not cache.l1.restoring and not cache.index.protected
    assert all(e.busy == 0 for e in cache.l1.entries.values())
    assert all(bm.refs[b] == 1 for b in blocks[:8]) and bm.get_num_free_blocks() == 92
    del bm.allocate_nolock
    assert cache.tick([]) == []  # nothing left to admit later


def test_an_admission_that_raises_after_the_copy_is_reported_and_unprotected() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    ids = _ids(7, 20, 50)
    _cached_chain(engine, cache, ids)
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    turn2 = ids + [1] * 300

    def raising(enqueue: bool = True):
        raise RuntimeError("add_request broke")

    assert cache.add(raising, "t2", turn2, pc.block_hashes(turn2), 4)[0] is de.PROMOTING
    other = ids[: 6 * B] + [4] * 300  # waits for t2, then is admitted in the same tick
    assert _add(engine, cache, "t3", other)[0] is de.PROMOTING
    clock.t += 1.0
    out = {item[0]: item for item in cache.tick([])}
    assert out["t2"][:4] == ("t2", None, 0, 0) and "add_request broke" in out["t2"][4]
    assert out["t3"][1] is not None and out["t3"][2] == 5 * B
    assert not cache.promoting and "t2" not in cache.live
    # Only the running t3 protects anything.
    assert cache.index.protected == {h: 1 for h in cache.live["t3"][2]}
    assert len(cache.index) == 19  # the restored chain stays for the next turn


def test_rank_loop_reports_an_admission_that_raises_after_the_copy() -> None:
    engine = _engine(kv_blocks=60, host_blocks=500)
    requests, results, cache, _ = _start_rank(engine)
    for i in range(6):
        _run(requests, results, f"s{i}", chr(65 + i) * 2000)
    original = engine.add_request

    def add_request(**kwargs):
        if kwargs["request_id"] == "back":
            raise RuntimeError("add_request broke")
        return original(**kwargs)

    engine.add_request = add_request
    requests.put(("add", "back", "A" * 2000 + "z" * 300, {"max_tokens": 8, "session_key": "s"}))
    deadline = time.time() + 10
    error = None
    while error is None:
        kind, _, payload = results.get(timeout=max(deadline - time.time(), 0.01))
        if kind == "out":
            error = next((item for item in payload if item[0] == "back" and item[4]), None)
    assert "add_request broke" in error[5] and error[6] is None
    assert cache.l1.counters["l1_promotions"] >= 1  # it went through L1 first
    assert not cache.promoting and not cache.index.protected
    engine.add_request = original
    meta, _ = _run(requests, results, "again", "A" * 2000 + "z" * 300)
    assert meta[1] == 14 * B
    requests.put(None)


def test_a_disabled_l1_never_unpins_a_block_its_copy_still_reads() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    ids = _ids(8, 20, 50)
    blocks = _cached_chain(engine, cache, ids)
    _evict_all(cache)
    assert _add(engine, cache, "t2", ids + [1] * 300)[0] is de.PROMOTING
    cache.tick([])  # rescued into L0 while the demotion copy still reads them
    cache.forget("t2")
    engine.abort_request("t2")
    cache.l1.disable("test")
    _evict_all(cache)
    bm = engine.block_manager
    assert len(cache.index) == 0
    assert all(bm.refs.get(b) == 1 for b in blocks[:19])  # held for the copy
    bm.allocate(bm.get_num_free_blocks())  # nothing it could hand out is still being read
    clock.t += 1.0
    cache.tick([])
    assert all(e.state == lc.READY and cache.l1.mover.host_tag[e.slot] != sim_llm.POISON
               for e in cache.l1.entries.values())
    assert not any(b in bm.refs for b in blocks)


def test_eviction_scan_gives_up_when_nothing_is_evictable(monkeypatch) -> None:
    monkeypatch.setattr(lc.L1, "SCAN_LIMIT", 3)
    unpinned = []
    l1 = lc.L1(_FakeMover(capacity=10, polls=100), unpinned.extend)
    a = pc.block_hashes(_ids(9, 8))
    l1.demote(a[::-1], list(range(18, 10, -1)), list(range(7, -1, -1)))  # 8 copies in flight
    b = pc.block_hashes(_ids(10, 4))
    l1.demote(b[::-1], [23, 22, 21, 20], [3, 2, 1, 0])
    assert l1.counters["l1_dropped"] == 2 and l1.counters["l1_evicted"] == 0
    assert sorted(unpinned) == [22, 23]


def test_a_disabled_l1_restores_nothing_more() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine, l1_promote_wait_seconds=60.0)
    clock = engine.l1_kv_mover.clock = _Clock()
    mover = engine.l1_kv_mover
    ids = _ids(11, 20, 50)
    _cached_chain(engine, cache, ids)
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    held = engine.block_manager.allocate(80)
    assert _add(engine, cache, "t2", ids + [5] * 300)[0] is de.PROMOTING  # claimed, waiting for blocks
    assert cache.tick([]) == []
    cache.l1.disable("verify failure", suspect=True)
    (_, seq, cached, l1_tokens), = cache.tick([])  # at once, not after l1_promote_wait_seconds
    assert seq in engine.waiting and cached == 0 and l1_tokens == 0 and mover.copied["h2d"] == 0
    assert not cache.promoting and not cache.l1.restoring
    assert all(e.busy == 0 for e in cache.l1.entries.values())
    engine.block_manager.unpin_blocks(held)


def test_an_unchecked_copy_from_a_suspect_l1_is_not_used() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine)
    clock = engine.l1_kv_mover.clock = _Clock()
    ids = _ids(12, 20, 50)
    _cached_chain(engine, cache, ids)
    _evict_all(cache)
    clock.t += 1.0
    cache.tick([])
    assert _add(engine, cache, "t2", ids + [5] * 300)[0] is de.PROMOTING
    assert cache.promoting["t2"].ticket is not None and not cache.promoting["t2"].checked
    cache.l1.disable("verify failure", suspect=True)
    clock.t += 1.0
    (_, seq, cached, l1_tokens), = cache.tick([])
    assert cached == 0 and l1_tokens == 0 and len(cache.index) == 0
    assert engine.block_manager.get_num_free_blocks() == 100
    assert all(e.busy == 0 for e in cache.l1.entries.values())


def test_without_l1_eviction_is_the_old_behaviour() -> None:
    engine = _engine(kv_blocks=100)
    cache = _cache(engine, l1=False)
    blocks = _cached_chain(engine, cache, _ids(13, 20, 50))
    _evict_all(cache)
    assert not any(b in engine.block_manager.refs for b in blocks)
    assert cache.l1 is None and "l1_blocks" not in cache.stats()
    seq, cached = _add(engine, cache, "t2", _ids(13, 20, 50) + [1] * 300)
    assert seq is not de.PROMOTING and cached == 0


def test_start_l1_turns_itself_off_when_the_engine_lacks_a_mover() -> None:
    engine = _engine(kv_blocks=100, gbps=0.0)
    cache = _cache(engine, l1=False)
    assert cache.start_l1(1 << 30) == {"l1": False, "reason": "the engine has no KV mover"}
    assert cache.l1 is None
    cp2 = de._RankCache(0, _engine(kv_blocks=100), {"cp": 2})
    assert cp2.start_l1(1 << 30)["reason"] == "L1 needs context_parallel_size 1"


# --- randomized stress ------------------------------------------------------------------------


def _stress(seed: int, kv_blocks: int = 120, host_blocks: int = 150, sessions: int = 10, iters: int = 1200,
            fail: bool = True, timeout: float = 10.0, verify: float = 0.0, max_ticket=None, monkeypatch=None) -> dict:
    """Random turns, aborts, copy failures (before and after issue), timeouts, index
    ring splits and L1 eviction over the simulated rank; after draining, every
    reference and every content tag must balance."""
    import random

    rng = random.Random(seed)
    engine = _engine(kv_blocks=kv_blocks, host_blocks=host_blocks)
    mover = engine.l1_kv_mover
    if max_ticket:
        mover.max_ticket_blocks = max_ticket
    mover.staging_rows = 3
    clock = mover.clock = _Clock()
    errors = []
    print_exc = de.traceback.print_exc

    def record(*args, **kwargs):
        errors.append(sys.exc_info()[1])
        print_exc(*args, **kwargs)

    monkeypatch.setattr(de.traceback, "print_exc", record)
    cache = _cache(engine, l1_copy_timeout_seconds=timeout, l1_verify_fraction=verify,
                   l1_promote_wait_seconds=0.0)
    l1 = cache.l1
    system = [(i * 31) % 4000 for i in range(4 * B)]
    prompts = {s: system + [(s * 7919 + i) % 4000 for i in range(rng.randint(1, 6) * B + rng.randint(0, 127))]
               for s in range(sessions)}
    tag_of = {}
    live = {}  # request id -> session
    n = 0
    for it in range(iters + 400):
        draining = it >= iters
        r = rng.random()
        if not draining and r < 0.25:
            idle = [s for s in range(sessions) if s not in live.values()]
            if idle:
                s = rng.choice(idle)
                ids = prompts[s] = prompts[s] + [rng.randrange(4000) for _ in range(rng.randint(10, 3 * B))]
                if len(ids) > 30 * B:
                    ids = prompts[s] = system + [rng.randrange(4000) for _ in range(rng.randint(1, 4) * B)]
                for h, tag in zip(pc.block_hashes(ids), sim_llm._block_tags(ids)):
                    tag_of[h] = tag
                n += 1
                mover.fail_before_issue = fail and rng.random() < 0.05
                mover.fail_after_chunks = 1 if fail and rng.random() < 0.05 else None
                seq, _ = _add(engine, cache, f"r{n}", ids, max_tokens=rng.randint(1, 6))
                if seq is not None:
                    live[f"r{n}"] = s
        elif not draining and r < 0.30 and live:
            request_id = rng.choice(list(live))
            cache.forget(request_id)
            engine.abort_request(request_id)
            del live[request_id]
        if rng.random() < 0.5 or draining:
            clock.t += rng.choice([0.0, 1e-5, 1e-4, 1e-3, 1.0])
        mover.fail_before_issue = fail and rng.random() < 0.02
        mover.fail_after_chunks = 1 if fail and rng.random() < 0.02 else None
        cache.tick(list(live))
        mover.fail_before_issue = False
        mover.fail_after_chunks = None
        assert l1.freeing_blocks == sum(1 for e in l1.entries.values() if e.freeing())
        if engine.has_unfinished_requests():
            for out in engine.step():
                if out.finished and out.request_id in live:
                    del live[out.request_id]
                    cache.forget(out.request_id)
    for _ in range(20):
        clock.t += 100
        cache.tick([])
        while engine.has_unfinished_requests():
            engine.step()
    time.sleep(0.01)  # past l1_copy_timeout_seconds 0: a copy flagged stuck still completes
    for _ in range(3):
        clock.t += 100
        cache.tick([])
    assert not errors, errors
    assert engine.kv_mismatch == 0
    assert not cache.promoting and not l1.demoting and not l1.quarantine and not l1.deferred and not l1.restoring
    assert l1.pending_d2h_blocks == 0 and l1.freeing_blocks == 0
    bm = engine.block_manager
    expected: dict = {}
    for node in cache.index._nodes.values():
        expected[node.block_id] = expected.get(node.block_id, 0) + 1
    assert bm.refs == expected  # exactly one reference per L0 node, nothing leaked or freed twice
    for h, node in cache.index._nodes.items():
        assert bm.kv_tag.get(node.block_id) == tag_of[h], "an L0 block holds another hash's content"
    slots = set()
    for h, entry in l1.entries.items():
        assert entry.state == lc.READY and entry.busy == 0 and entry.slot not in slots
        assert mover.host_tag.get(entry.slot) == tag_of[h], "an L1 slot holds another hash's content"
        slots.add(entry.slot)
    assert l1.pool.free + len(l1.entries) == l1.pool.capacity
    return cache.stats()


@pytest.mark.parametrize("seed", range(4))
def test_stress_keeps_every_reference_and_content_tag_balanced(seed, monkeypatch) -> None:
    stats = _stress(seed, monkeypatch=monkeypatch)
    assert stats["l1_promotions"] > 0


@pytest.mark.parametrize("seed", range(2))
def test_stress_with_a_tight_host_pool_and_split_copies(seed, monkeypatch) -> None:
    _stress(seed, kv_blocks=70, host_blocks=40, sessions=14, max_ticket=5, monkeypatch=monkeypatch)


@pytest.mark.parametrize("seed", range(2))
def test_stress_with_verify_on_never_fails_verify(seed, monkeypatch) -> None:
    stats = _stress(seed, fail=False, verify=1.0, monkeypatch=monkeypatch)
    assert stats["l1_verify_failures"] == 0 and stats["l1_enabled"]


@pytest.mark.parametrize("seed", range(2))
def test_stress_with_every_copy_timing_out(seed, monkeypatch) -> None:
    _stress(seed, timeout=0.0, monkeypatch=monkeypatch)


# --- the rank loop ------------------------------------------------------------------------


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True):
        return [ord(c) % 5000 for c in text]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(chr(97 + i % 26) for i in ids)

    def __len__(self) -> int:
        return 5000


def _start_rank(engine, **settings):
    requests, results = queue.Queue(), queue.Queue()
    settings = {"idle_poll_seconds": 0.001, "tokenizer_threads": 2, "stats_interval_seconds": 0.05,
                "prefix_cache_free_fraction": 0.0, **settings}
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
    requests.put(("l1_budget", 0))
    while True:
        kind, _, info = results.get(timeout=5)
        if kind == "l1_ready":
            break
    return requests, results, cache, info


def _run(requests, results, request_id: str, prompt: str, max_tokens: int = 8, timeout: float = 10.0):
    """Send one request; return (meta item, prompt tokens)."""
    requests.put(("add", request_id, prompt, {"max_tokens": max_tokens, "session_key": "s"}))
    meta = None
    deadline = time.time() + timeout
    while True:
        kind, _, payload = results.get(timeout=max(deadline - time.time(), 0.01))
        if kind == "meta":
            meta = next((item for item in payload if item[0] == request_id), meta)
        elif kind == "out":
            for item in payload:
                if item[0] == request_id and item[4]:
                    assert item[5] is None, item
                    return meta, item[3]


def test_rank_loop_serves_returning_sessions_from_l1_with_a_small_gpu_pool() -> None:
    engine = _engine(kv_blocks=60, host_blocks=500)
    requests, results, cache, info = _start_rank(engine)
    assert info["l1"] and info["l1_tokens_capacity"] == 500 * B
    # Six 2,000-token sessions; the GPU pool holds about three of them.
    for i in range(6):
        _run(requests, results, f"s{i}", chr(65 + i) * 2000)
    assert cache.index.evicted > 0
    l1_tokens = 0
    for i in range(6):
        meta, prompt_len = _run(requests, results, f"s{i}b", chr(65 + i) * 2000 + "z" * 300)
        assert prompt_len == 2300
        assert meta[1] == 14 * B, meta  # every returning session hits, L0 or L1
        l1_tokens += meta[2]
    assert l1_tokens > 0
    assert engine.kv_mismatch == 0 and engine.scheduler.preempted == 0
    deadline = time.time() + 5
    stats = {}
    while time.time() < deadline and not stats.get("l1_hits"):
        kind, _, payload = results.get(timeout=5)
        if kind == "stats":
            stats = payload
    assert stats["l1_hits"] > 0 and stats["l1_demoted_blocks"] > 0 and "l1_h2d_ms_p99" in stats
    requests.put(None)


def test_rank_loop_abort_while_promoting() -> None:
    engine = _engine(kv_blocks=60, host_blocks=500)
    requests, results, cache, _ = _start_rank(engine)
    for i in range(6):
        _run(requests, results, f"s{i}", chr(65 + i) * 2000)
    mover = engine.l1_kv_mover
    issue = mover._issue

    def slow_h2d(kind, src, dst, want_sums):
        ticket = issue(kind, src, dst, want_sums)
        if kind == "h2d":
            ticket.t_done += 3600.0  # from now on a promotion copy does not land by itself
        return ticket

    mover._issue = slow_h2d
    requests.put(("add", "back", "A" * 2000 + "z" * 300, {"max_tokens": 8, "session_key": "s"}))
    deadline = time.time() + 10
    promo = None
    while time.time() < deadline:
        promo = cache.promoting.get("back")
        if promo is not None and promo.ticket is not None:
            break
        time.sleep(0.005)
    assert promo is not None and promo.ticket is not None
    requests.put(("abort", "back"))
    deadline = time.time() + 5
    while not promo.aborted and time.time() < deadline:
        time.sleep(0.005)
    assert promo.aborted
    promo.ticket.t_done = 0.0  # let the copy land now
    deadline = time.time() + 5
    while "back" in cache.promoting and time.time() < deadline:
        time.sleep(0.005)
    assert "back" not in cache.promoting and "back" not in cache.live
    assert not any(s.request_id == "back" for s in engine.waiting)  # never admitted
    assert not cache.index.protected or all(h not in cache.index.protected for h in promo.hashes)
    requests.put(None)


# --- the parent -------------------------------------------------------------------------------


def test_parent_reader_takes_l1_tokens_from_meta() -> None:
    engine = object.__new__(de.DeepInferEngine)
    engine._results = queue.Queue()
    stream = de._Stream()
    engine._streams = {"r": stream}
    engine._last_message = time.monotonic()
    engine._closed = True
    engine._failed = None
    engine._results.put(("meta", 0, [("r", 1792, 768)]))
    engine._results.put(("meta", 0, [("gone", 1, 0, "future field")]))
    engine._results.put(None)
    engine._read_results()
    assert stream.cached == 1792 and stream.l1_tokens == 768


def test_rank_stats_log_line_has_the_l1_part(capsys) -> None:
    engine = object.__new__(de.DeepInferEngine)
    engine._in_flight = [0]
    engine._sessions = de.OrderedDict()
    engine._session_idle = 1800.0
    engine.migrations = 0
    engine._lock = threading.Lock()
    engine.rank_stats = [{"rank": 0, "total_blocks": 100, "free_blocks": 40, "l1_blocks": 41203,
                          "l1_capacity_blocks": 602496, "l1_pinned_bytes": 602496 * 114048, "l1_hits": 120,
                          "l1_hit_tokens": 9_100_000, "l1_promoted_blocks": 71000, "l1_h2d_ms_p50": 6.1,
                          "l1_h2d_ms_p99": 17.9, "l1_h2d_gbps": 49.8, "l1_demoted_blocks": 88000,
                          "l1_d2h_gbps": 50.3, "l1_demote_skipped": 12000}]
    engine._log_rank_stats()
    err = capsys.readouterr().err
    assert "L1 41,203/602,496 blocks (4.7 GB), hits 120 (9,100,000 tokens), promoted 71,000 blocks" in err
    assert "h2d p50 6.1 ms p99 17.9 ms, 49.8 GB/s" in err


def test_host_budget_comes_from_the_cgroup_minus_what_cannot_be_reclaimed(tmp_path) -> None:
    gib = 2**30
    (tmp_path / "memory.max").write_text(str(1500 * gib))
    (tmp_path / "memory.stat").write_text(
        f"anon {420 * gib}\nfile {600 * gib}\nshmem {12 * gib}\nunevictable 0\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"engram_layer_ids": [1, 14], "engram_max_ngram_size": 4, '
                                       '"engram_n_heads": 8, "engram_vocab_size": 16000000, "engram_head_dim": 256}')
    per_rank, line = de.l1_host_budget({}, 4, str(model), cgroup=str(tmp_path))
    engram = 2 * 2 * 3 * 8 * 16_000_000 * 256  # tables and buffer storage, about 393 GB
    spare = 1500 * gib - 432 * gib - 75 * gib - engram
    assert per_rank == int(min(64 * gib, spare / 4))
    assert "cgroup v2" in line and "Engram host 366 GiB reserved" in line
    small, _ = de.l1_host_budget({"l1_host_gb": 8, "l1_engram_host_gb": 0}, 4, str(model), cgroup=str(tmp_path))
    assert small == 8 * gib
    tight, _ = de.l1_host_budget({}, 4, str(model), cgroup=str(tmp_path / "none"),
                                 meminfo=str(_meminfo(tmp_path, 40, 20)))
    assert tight == 0  # 40 GiB machine, 20 available, 32 GiB reserve: nothing left


def _meminfo(tmp_path, total_gib: int, available_gib: int) -> pathlib.Path:
    path = tmp_path / "meminfo"
    path.write_text(f"MemTotal: {total_gib * 2**20} kB\nMemAvailable: {available_gib * 2**20} kB\n")
    return path


# --- the real mover on CPU tensors -------------------------------------------------------------


def test_torch_mover_moves_rows_byte_exact_on_cpu_tensors() -> None:
    torch = pytest.importorskip("torch")
    bpb = 1152 * 2
    gpu = torch.randint(0, 256, (64, bpb), dtype=torch.uint8)
    ref = gpu.clone()
    # Several chunks per ticket, and runs split across host segments.
    mover = km.TorchKVMover(gpu, bpb, index_ring=256, log=lambda m: None, staging_rows=5, slots_per_segment=7)
    assert mover.pin_host(40 * bpb) == 40
    ids = [3, 9, 10, 11, 30, 31, 2, 63, 17, 18, 19, 20]
    slots = [0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 33, 34]
    t = mover.d2h(ids, slots, checksum=True)
    assert mover.done(t)
    sums = mover.checksums(t)
    for block, slot, s in zip(ids, slots, sums):
        row = mover._host_rows(slot, 1)[0]
        assert torch.equal(row.view(torch.uint8), ref[block])
        assert s == int(ref[block].view(torch.int64).sum())
    gpu[ids] = 0  # the blocks are reused for something else
    dst = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    t = mover.h2d(slots, dst, checksum=True)
    assert mover.done(t)
    for block, d in zip(ids, dst):
        assert torch.equal(gpu[d], ref[block])
    assert mover.checksums(t) == sums
    with pytest.raises(ValueError):
        mover.d2h([0], [0])  # block 0 is never handed out
    with pytest.raises(ValueError):
        mover.h2d([40], [5])  # slot out of range
    assert not mover.rings["d2h"].ranges and not mover.rings["h2d"].ranges


def test_index_ring_wraps_and_frees_in_order() -> None:
    ring = km._Ring(10)
    a = ring.alloc(4)
    b = ring.alloc(4)
    assert (a, b) == (0, 4)
    with pytest.raises(RuntimeError):
        ring.alloc(3)
    ring.free(a)
    assert ring.alloc(3) == 0  # wraps to the start
    ring.free(b)
    with pytest.raises(ValueError):
        ring.alloc(11)

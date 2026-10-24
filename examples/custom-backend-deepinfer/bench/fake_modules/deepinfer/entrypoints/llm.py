"""Simulated ``deepinfer.entrypoints.llm.LLM``: a rank that decodes at a fixed step rate.

Model of one data-parallel rank, per ``step()``:

* admit waiting requests while fewer than ``max_num_seqs`` are in the batch;
* prefill up to ``max_num_batched_tokens`` (minus this step's decode tokens) of
  prompt tokens, costing ``tokens / prefill_tokens_per_s`` seconds (0 means
  free); a request emits its first token in the step that finishes its prompt;
* decode: every running request emits ``tokens_per_step`` tokens, one
  ``RequestOutput`` each;
* the step lasts ``1 / step_rate`` seconds plus its prefill time, on a fixed
  grid while the rank keeps up; a rank whose loop falls more than a step behind
  (a slow host, a long tokenization) loses that time, like a GPU waiting on its
  host.

Lockstep: with ``barrier`` and ``flags`` (multiprocessing objects made by
``sim_server.py``), ``has_unfinished_requests()`` is an OR over the ranks and
``step()`` ends on a barrier, like DeepInfer's all-reduce and expert-parallel
collectives: one slow rank (tokenizing a long prompt, prefilling) slows all.

KV cache (``kv_blocks``, 0 = unlimited): a pool of 128-token blocks with
refcounts, the surface of DeepInfer plus the local prefix-cache patch that the
backend's ``_RankCache`` uses: ``block_manager.pin_blocks / unpin_blocks /
get_num_free_blocks``, ``cache_config.num_gpu_blocks``,
``add_request(enqueue=False)``, ``scheduler.add_sequence_with_prefix(seq,
blocks, num_cached, swa_valid_start)`` / ``enqueue_with_prefix`` (only the uncached suffix is
prefilled), ``Sequence.block_table`` and ``finalize_seq(seq)``, called for a
finished sequence before its blocks are freed. A request is admitted when its
prompt and ``max_tokens`` fit in the free blocks and the pool is under the
90 % prefill watermark; otherwise it waits. ``decode_ctx_beta`` adds that many seconds per
step for every token of context in the running batch (attention cost).

Host tier (``l1_pcie_gbps`` > 0): ``SimEngine.l1_kv_mover`` is a
:class:`SimKVMover`, the stand-in for the backend's ``TorchKVMover``: a host
pool of slots and one link per direction with a bandwidth and a per-copy
latency, so a copy takes simulated time and the backend must wait for it. To
catch ordering bugs without a GPU, the rank keeps a content tag per block (a
chained crc32 of the prompt tokens up to the end of the block, written when
the prefill finishes): a block is poisoned when it is allocated or freed, a
device-to-host copy reads the tags of its source blocks only when it
completes, a host-to-device copy poisons its destination blocks when issued
and writes them only when it completes, and ``add_sequence_with_prefix``
counts every reused block whose tag does not match the sequence's own tokens
(``kv_mismatch``). An early unpin, an early admission or any hash, slot or
block mix-up shows up there.

Every ``stats_interval_s`` the rank appends one JSON line with its counters to
``<stats_path>.rank<N>`` so the load driver can compare what the ranks produce
with what the clients receive.
"""

from __future__ import annotations

import json
import time
import types
import zlib
from array import array
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence

from deepinfer._C.engine import RequestOutput


class StepOutputs(list):
    """What ``step()`` / ``flush_inflight_reqs()`` return: the real engine returns a
    numpy array, whose truth value raises when it is empty or has more than one
    element. Mirror that so ``if outputs:`` fails here and not on a GPU."""

    def __bool__(self) -> bool:
        raise ValueError("The truth value of an array is ambiguous (DeepInfer returns np.ndarray): use len()")

# Plain prose without markup characters, so the backend's stream splitter never
# holds text back waiting for a possible </think> or DSML marker.
_PASSAGE = (
    "The scheduler keeps a batch of running sequences and advances each of them by one "
    "token per step. When a sequence reaches its length limit it leaves the batch and a "
    "waiting one takes its place. Throughput grows with the batch until the step time "
    "starts to grow with it, which on a large mixture of experts model happens late, "
    "because every expert sees more tokens per step and the matrix multiplications get "
    "more efficient. Latency for each user is the step time, so a server that keeps its "
    "steps short and its batches full serves many developers at once. "
)


BLOCK = 128
PREFILL_WATERMARK = 0.9
POISON = -1  # the content tag of a block nobody wrote (fresh, freed, or a copy still in flight)


def _block_tags(token_ids: Any) -> List[int]:
    """Content tag of each full prompt block: a crc32 chained over the tokens up to its end."""
    data = token_ids.tobytes() if hasattr(token_ids, "tobytes") else array("i", token_ids).tobytes()
    step = BLOCK * 4
    tags: List[int] = []
    crc = 0
    for start in range(0, len(data) - step + 1, step):
        crc = zlib.crc32(data[start : start + step], crc)
        tags.append(crc)
    return tags


class _Seq:
    __slots__ = ("request_id", "prompt_len", "prefill_left", "max_tokens", "generated", "offset",
                 "block_table", "num_cached_tokens", "engine", "token_ids", "_tags")

    @property
    def status(self) -> str:
        return self.engine.seq_status(self) if self.engine is not None else "?"

    def __init__(self, request_id: str, prompt_len: int, max_tokens: int, offset: int) -> None:
        self.request_id = request_id
        self.prompt_len = prompt_len
        self.prefill_left = prompt_len
        self.max_tokens = max_tokens
        self.generated = 0
        self.offset = offset
        self.block_table: List[int] = []
        self.num_cached_tokens = 0
        self.engine: Any = None
        self.token_ids: Any = None  # kept only when the engine tracks content tags
        self._tags: Optional[List[int]] = None

    def tags(self) -> List[int]:
        if self._tags is None:
            self._tags = _block_tags(self.token_ids) if self.token_ids is not None else []
        return self._tags


class _BlockManager:
    """Refcounted block ids; ids are handed out lazily so a huge pool costs nothing.

    ``kv_tag`` holds the content tag of every written block (absent = poisoned).
    ``lock`` / ``unlock`` / ``allocate_nolock`` mirror the real block manager:
    ``allocate_nolock`` has no free check and raises halfway when the free list
    runs dry, keeping the blocks it already took (the real one leaks them too).
    """

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.refs: Dict[int, int] = {}
        self.released: List[int] = []
        self.next_fresh = 1  # block 0 is never handed out, as in DeepInfer
        self.kv_tag: Dict[int, int] = {}
        self.locked = False

    def get_num_free_blocks(self) -> int:
        return self.num_blocks - len(self.refs)

    def allocate(self, n: int) -> List[int]:
        out = []
        for _ in range(n):
            out.append(self._take())
        return out

    def _take(self) -> int:
        if self.released:
            block = self.released.pop()
        else:
            block = self.next_fresh
            self.next_fresh += 1
        self.refs[block] = 1
        self.kv_tag.pop(block, None)
        return block

    def lock(self) -> None:
        assert not self.locked, "block manager lock re-entered"
        self.locked = True

    def unlock(self) -> None:
        assert self.locked, "unlock without lock"
        self.locked = False

    def allocate_nolock(self, n: int) -> List[int]:
        assert self.locked, "allocate_nolock without the lock"
        out = []
        for _ in range(n):
            if self.get_num_free_blocks() <= 0:
                raise RuntimeError("no free KV blocks")  # like TORCH_CHECK(_head != 0); taken blocks leak
            out.append(self._take())
        return out

    def pin_blocks(self, block_ids: List[int]) -> None:
        for block in block_ids:
            self.refs[block] += 1

    def unpin_blocks(self, block_ids: List[int]) -> None:
        for block in block_ids:
            n = self.refs[block] - 1
            if n:
                self.refs[block] = n
            else:
                del self.refs[block]
                self.kv_tag.pop(block, None)  # a freed block's bytes are fair game
                self.released.append(block)


class _Scheduler:
    def __init__(self, engine: "SimEngine") -> None:
        self.engine = engine
        self.preempted = 0

    def add_sequence_with_prefix(self, seq: _Seq, blocks: Any, num_cached: int, swa_valid_start: int) -> None:
        blocks = [int(b) for b in blocks]
        assert len(blocks) == num_cached // BLOCK, (len(blocks), num_cached)
        engine = self.engine
        if engine.kv_tags:
            tags, kv_tag = seq.tags(), engine.block_manager.kv_tag
            bad = sum(1 for i, b in enumerate(blocks) if i >= len(tags) or kv_tag.get(b, POISON) != tags[i])
            if bad:
                engine.kv_mismatch += bad
                engine.kv_mismatch_requests += 1
        self.engine.block_manager.pin_blocks(blocks)
        seq.block_table = blocks
        seq.num_cached_tokens = num_cached
        seq.prefill_left = seq.prompt_len - num_cached
        self.engine.hit_tokens += num_cached
        self.engine.waiting.append(seq)

    def add_sequence(self, seq: _Seq) -> None:
        self.engine.waiting.append(seq)

    def get_num_preempted_seqs(self) -> int:
        return self.preempted

    def get_sum_prompt_len_waiting(self) -> "tuple[int, int]":
        waiting = self.engine.waiting
        return len(waiting), sum(s.prompt_len for s in waiting)


class _ModelConfig:
    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size

    def get_vocab_size(self) -> int:
        return self._vocab_size


class SimEngine:
    """The ``llm.llm_engine`` surface the rank loop uses."""

    multi_modal_processor = None

    def __init__(self, rank: int, max_num_seqs: int, max_num_batched_tokens: int, sim: Dict[str, Any],
                 token_ids: List[int], vocab_size: int) -> None:
        self.rank = rank
        self.model_config = _ModelConfig(vocab_size)
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.step_s = 1.0 / float(sim.get("step_rate", 70.0))
        self.tokens_per_step = int(sim.get("tokens_per_step", 1))
        self.prefill_rate = float(sim.get("prefill_tokens_per_s", 0.0))
        self.decode_ctx_beta = float(sim.get("decode_ctx_beta", 0.0))
        kv_blocks = int(sim.get("kv_blocks", 0)) or 1 << 40
        self.block_manager = _BlockManager(kv_blocks)
        self.cache_config = types.SimpleNamespace(num_gpu_blocks=kv_blocks, block_size=BLOCK)
        gbps = float(sim.get("l1_pcie_gbps", 0.0))
        self.kv_tags = bool(gbps > 0 or sim.get("kv_tags"))
        self.kv_mismatch = 0
        self.kv_mismatch_requests = 0
        self.l1_kv_mover = (SimKVMover(self, gbps, float(sim.get("l1_copy_latency_us", 10.0)),
                                       int(sim.get("l1_bytes_per_block", 114048)),
                                       host_blocks=int(sim.get("l1_host_blocks", 0)))
                            if gbps > 0 else None)
        self.scheduler = _Scheduler(self)
        self.hit_tokens = 0
        self.prompt_tokens = 0
        self.barrier = sim.get("barrier")
        self.flags = sim.get("flags")
        self.stats_path: Optional[str] = sim.get("stats_path")
        self.stats_interval = float(sim.get("stats_interval_s", 1.0))
        self.token_ids = token_ids
        self.waiting: Deque[_Seq] = deque()
        self.prefilling: List[_Seq] = []
        self.running: List[_Seq] = []
        self.to_finalize: List[_Seq] = []  # finished, finalized at the start of the next step
        self.last_end = 0.0
        self.steps = 0
        self.tokens = 0
        self.prefilled = 0
        self.next_stats = time.time()

    # --- the calls _rank_main makes -------------------------------------------

    def add_request(self, user_id: str, request_id: str, sampling_params: Any, prompt_token_ids: Any,
                    multi_modal_data: Optional[dict] = None, enqueue: bool = True, **_: Any) -> _Seq:
        # Like DeepInfer, the new sequence is returned (None would mean "dropped").
        offset = zlib.crc32(request_id.encode()) % len(self.token_ids)
        seq = _Seq(request_id, len(prompt_token_ids), sampling_params.max_tokens, offset)
        seq.engine = self
        if self.kv_tags:
            seq.token_ids = prompt_token_ids
        self.prompt_tokens += seq.prompt_len
        if enqueue:
            self.waiting.append(seq)
        return seq

    def enqueue_with_prefix(self, seq: _Seq, blocks: Any, num_cached: int, swa_valid_start: int) -> None:
        """The patched engine's entry point (multimodal and structured-output handling aside)."""
        self.scheduler.add_sequence_with_prefix(seq, blocks, num_cached, swa_valid_start)

    def seq_status(self, seq: _Seq) -> str:
        if any(s is seq for s in self.waiting):
            return f"WAITING (position {next(i for i, s in enumerate(self.waiting) if s is seq)} of {len(self.waiting)})"
        if seq in self.prefilling:
            return f"PREFILLING ({seq.prefill_left} tokens left)"
        if seq in self.running:
            return "RUNNING"
        return "NOT IN ENGINE"

    def finalize_seq(self, seq: _Seq) -> None:
        """DeepInfer's hook for a finished sequence; frees its blocks (the backend wraps it)."""
        self.block_manager.unpin_blocks(seq.block_table)
        seq.block_table = []

    def _admit(self, seq: _Seq) -> bool:
        bm = self.block_manager
        need = -(-(seq.prompt_len + seq.max_tokens) // BLOCK) - len(seq.block_table)
        used = bm.num_blocks - bm.get_num_free_blocks()
        # As DeepInfer: no new prefill above the watermark, and the blocks must exist.
        if used > PREFILL_WATERMARK * bm.num_blocks or need > bm.get_num_free_blocks():
            return False
        seq.block_table = seq.block_table + bm.allocate(need)
        return True

    def has_unfinished_requests(self) -> bool:
        # A finished sequence stays in the scheduler until the next step finalizes it.
        local = bool(self.waiting or self.prefilling or self.running or self.to_finalize)
        self._maybe_stats()
        if self.barrier is None:
            return local
        self.flags[self.rank] = 1 if local else 0
        self.barrier.wait()
        unfinished = any(self.flags[:])
        # Second barrier: nobody overwrites its flag before everyone has read them.
        self.barrier.wait()
        return unfinished

    def step(self) -> StepOutputs:
        self._finalize_pending()
        while self.waiting and len(self.prefilling) + len(self.running) < self.max_num_seqs:
            if not self._admit(self.waiting[0]):
                break  # FIFO: wait for blocks to free up
            self.prefilling.append(self.waiting.popleft())

        outputs: List[RequestOutput] = []
        decode_tokens = 0
        still_running: List[_Seq] = []
        for seq in self.running:
            done = self._emit(seq, outputs)
            decode_tokens += self.tokens_per_step
            if not done:
                still_running.append(seq)

        budget = max(self.max_num_batched_tokens - decode_tokens, 0)
        prefill_tokens = 0
        still_prefilling: List[_Seq] = []
        for seq in self.prefilling:
            take = min(seq.prefill_left, budget - prefill_tokens)
            seq.prefill_left -= take
            prefill_tokens += take
            if seq.prefill_left == 0:
                if self.kv_tags:
                    self._write_tags(seq)
                # The step that finishes a prompt samples its first token.
                if not self._emit(seq, outputs, first=True):
                    still_running.append(seq)
            else:
                still_prefilling.append(seq)
        self.running = still_running
        self.prefilling = still_prefilling

        work = self.step_s + (prefill_tokens / self.prefill_rate if self.prefill_rate > 0 else 0.0)
        if self.decode_ctx_beta:
            work += self.decode_ctx_beta * sum(s.prompt_len + s.generated for s in self.running)
        now = time.perf_counter()
        # Steps follow a fixed grid while the rank keeps up, so the loop's own
        # overhead (barriers, queue polling) is absorbed like GPU time overlapping
        # host work; a rank that falls a whole step behind restarts the grid.
        start = self.last_end if now - self.last_end < self.step_s else now
        end = start + work
        time.sleep(max(end - now, 0.0))
        self.last_end = end
        if self.barrier is not None:
            self.barrier.wait()

        self.steps += 1
        self.tokens += len(outputs)
        self.prefilled += prefill_tokens
        return StepOutputs(outputs)

    def flush_inflight_reqs(self) -> StepOutputs:
        self._finalize_pending()
        return StepOutputs()

    def abort_request(self, request_id: Any) -> tuple:
        ids = {request_id} if isinstance(request_id, str) else set(request_id)
        before = len(self.waiting) + len(self.prefilling) + len(self.running)
        for seq in [*self.waiting, *self.prefilling, *self.running]:
            if seq.request_id in ids:
                # Aborts free the blocks directly, without finalize_seq (as the C++ abort does).
                self.block_manager.unpin_blocks(seq.block_table)
                seq.block_table = []
        self.waiting = deque(s for s in self.waiting if s.request_id not in ids)
        self.prefilling = [s for s in self.prefilling if s.request_id not in ids]
        self.running = [s for s in self.running if s.request_id not in ids]
        return before - len(self.waiting) - len(self.prefilling) - len(self.running), 0

    # --- helpers --------------------------------------------------------------

    def _write_tags(self, seq: _Seq) -> None:
        """The prefill wrote the KV of the prompt blocks it computed."""
        tags, table, kv_tag = seq.tags(), seq.block_table, self.block_manager.kv_tag
        for i in range(seq.num_cached_tokens // BLOCK, min(len(tags), len(table))):
            kv_tag[table[i]] = tags[i]

    def _emit(self, seq: _Seq, outputs: List[RequestOutput], first: bool = False) -> bool:
        """Append this step's tokens of ``seq``; True when it finished."""
        n = 1 if first else self.tokens_per_step
        ids = self.token_ids
        for _ in range(n):
            token = ids[(seq.offset + seq.generated) % len(ids)]
            seq.generated += 1
            finished = seq.generated >= seq.max_tokens
            outputs.append(RequestOutput(seq.request_id, token, finished, seq.prompt_len,
                                         "length" if finished else ""))
            if finished:
                # DeepInfer (zero-overhead scheduling) finalizes a finished sequence in the
                # NEXT step's pre_sampler_sync, after its last token went out: mirror that.
                self.to_finalize.append(seq)
                return True
        return False

    def _finalize_pending(self) -> None:
        pending, self.to_finalize = self.to_finalize, []
        for seq in pending:
            self.finalize_seq(seq)

    def _maybe_stats(self) -> None:
        if not self.stats_path:
            return
        now = time.time()
        if now < self.next_stats:
            return
        self.next_stats = now + self.stats_interval
        line = {
            "rank": self.rank, "t": now, "steps": self.steps, "tokens": self.tokens,
            "prefilled": self.prefilled, "running": len(self.running),
            "prefilling": len(self.prefilling), "waiting": len(self.waiting),
            "free_blocks": self.block_manager.get_num_free_blocks(), "hit_tokens": self.hit_tokens,
            "prompt_tokens": self.prompt_tokens, "kv_mismatch": self.kv_mismatch,
        }
        if self.l1_kv_mover is not None:
            line.update(self.l1_kv_mover.counters())
        with open(f"{self.stats_path}.rank{self.rank}", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")


class SimPartialIssue(Exception):
    """Like ``kv_mover.PartialIssue``: part of the copy is queued; ``.ticket`` completes later."""

    def __init__(self, ticket: "_SimTicket") -> None:
        super().__init__("simulated copy failure after the first chunks")
        self.ticket = ticket


class _SimTicket:
    __slots__ = ("kind", "src", "dst", "t_issue", "t_done", "applied", "sums", "want_sums")

    def __init__(self, kind: str, src: List[int], dst: List[int], t_issue: float, t_done: float,
                 want_sums: bool) -> None:
        self.kind = kind
        self.src = src
        self.dst = dst
        self.t_issue = t_issue
        self.t_done = t_done
        self.applied = False
        self.sums: Optional[List[int]] = None
        self.want_sums = want_sums


class SimKVMover:
    """Simulated ``KVMover``: a host pool of tagged slots and one link per direction.

    A ticket of ``n`` blocks completes at ``max(now, link free) + latency + n *
    bytes_per_block / bandwidth`` and keeps the link busy until then. ``done()``
    compares with ``clock()`` and, the first time it returns True, applies the
    copy: the content tags move only at completion (see the module docstring).
    Test hooks: ``fail_before_issue`` (raise before anything is queued) and
    ``fail_after_chunks`` (raise :class:`SimPartialIssue` after that many chunks
    of ``staging_rows`` blocks, whose copy still completes later).
    """

    def __init__(self, engine: "SimEngine", gbps: float, latency_us: float, bytes_per_block: int,
                 host_blocks: int = 0, staging_rows: int = 588) -> None:
        self.engine = engine
        self.gbps = gbps
        self.latency = latency_us * 1e-6
        self.bytes_per_block = bytes_per_block
        self.host_blocks = host_blocks
        self.staging_rows = staging_rows
        self.capacity_slots = 0
        self.host_tag: Dict[int, int] = {}
        self.link_free = {"d2h": 0.0, "h2d": 0.0}
        self.clock = time.perf_counter
        self.fail_before_issue = False
        self.fail_after_chunks: Optional[int] = None
        self.copied = {"d2h": 0, "h2d": 0}

    def pin_host(self, nbytes: int) -> int:
        self.capacity_slots = self.host_blocks or max(0, int(nbytes) // self.bytes_per_block)
        return self.capacity_slots

    def _check(self, gpu_ids: Sequence[int], slots: Optional[Sequence[int]]) -> None:
        if self.fail_before_issue:
            raise RuntimeError("simulated copy failure before issue")
        if slots is not None and len(slots) != len(gpu_ids):
            raise ValueError("blocks and slots differ in length")
        num = self.engine.block_manager.num_blocks
        for b in gpu_ids:
            if not 0 < b <= num:  # ids 1..num_blocks (block 0 is never handed out)
                raise ValueError(f"GPU block {b} out of range")
        for s in slots or ():
            if not 0 <= s < self.capacity_slots:
                raise ValueError(f"host slot {s} out of range")

    def _issue(self, kind: str, src: List[int], dst: List[int], want_sums: bool) -> _SimTicket:
        direction = "h2d" if kind == "h2d" else "d2h"
        n = len(src)
        partial = (self.fail_after_chunks is not None and n > self.fail_after_chunks * self.staging_rows)
        if partial:
            n = self.fail_after_chunks * self.staging_rows
            src, dst = src[:n], dst[:n]
        now = self.clock()
        cost = self.latency + (n * self.bytes_per_block / (self.gbps * 1e9) if kind != "sum" else 0.0)
        t_done = max(now, self.link_free[direction]) + cost
        self.link_free[direction] = t_done
        ticket = _SimTicket(kind, src, dst, now, t_done, want_sums)
        if kind == "h2d":
            kv_tag = self.engine.block_manager.kv_tag
            for b in dst:
                kv_tag.pop(b, None)  # the destination holds garbage until the copy lands
        if partial:
            raise SimPartialIssue(ticket)
        return ticket

    def d2h(self, gpu_ids: Sequence[int], slots: Sequence[int], checksum: bool = False) -> _SimTicket:
        self._check(gpu_ids, slots)
        return self._issue("d2h", [int(b) for b in gpu_ids], [int(s) for s in slots], checksum)

    def h2d(self, slots: Sequence[int], gpu_ids: Sequence[int], checksum: bool = False) -> _SimTicket:
        self._check(gpu_ids, slots)
        return self._issue("h2d", [int(s) for s in slots], [int(b) for b in gpu_ids], checksum)

    def checksum_blocks(self, gpu_ids: Sequence[int]) -> _SimTicket:
        self._check(gpu_ids, None)
        return self._issue("sum", [int(b) for b in gpu_ids], [], True)

    def done(self, ticket: _SimTicket) -> bool:
        if ticket.applied:
            return True
        if self.clock() < ticket.t_done:
            return False
        ticket.applied = True
        kv_tag = self.engine.block_manager.kv_tag
        if ticket.kind == "d2h":
            sums = [kv_tag.get(b, POISON) for b in ticket.src]
            for s, tag in zip(ticket.dst, sums):
                self.host_tag[s] = tag
        elif ticket.kind == "h2d":
            sums = [self.host_tag.get(s, POISON) for s in ticket.src]
            for b, tag in zip(ticket.dst, sums):
                if tag == POISON:
                    kv_tag.pop(b, None)
                else:
                    kv_tag[b] = tag
        else:
            sums = [kv_tag.get(b, POISON) for b in ticket.src]
        if ticket.kind != "sum":
            self.copied["h2d" if ticket.kind == "h2d" else "d2h"] += len(ticket.src)
        ticket.sums = sums
        return True

    def checksums(self, ticket: _SimTicket) -> Optional[List[int]]:
        return ticket.sums if ticket.want_sums else None

    def elapsed(self, ticket: _SimTicket) -> float:
        return ticket.t_done - ticket.t_issue

    def counters(self) -> Dict[str, int]:
        return {"l1_sim_d2h_blocks": self.copied["d2h"], "l1_sim_h2d_blocks": self.copied["h2d"]}

    def close(self) -> None:
        pass


class LLM:
    """Accepts the constructor arguments of DeepInfer's ``LLM``; the simulation reads
    ``model``, ``engine_rank``, ``max_num_seqs``, ``max_num_batched_tokens`` and
    ``bench_sim`` (passed through the backend's ``llm_kwargs`` config key)."""

    def __init__(self, model: str, engine_rank: int = 0, max_num_seqs: int = 64,
                 max_num_batched_tokens: int = 32768, bench_sim: Optional[Dict[str, Any]] = None,
                 **_: Any) -> None:
        sim = dict(bench_sim or {})
        if sim.get("tokenizer") == "fake":
            from bench_fake_tokenizer import FakeTokenizer  # noqa: PLC0415

            self._tokenizer = FakeTokenizer()
        else:
            from transformers import AutoTokenizer  # noqa: PLC0415

            self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        token_ids = list(self._tokenizer.encode(_PASSAGE * 4, add_special_tokens=False))
        self.llm_engine = SimEngine(engine_rank, int(max_num_seqs), int(max_num_batched_tokens), sim, token_ids,
                                    len(self._tokenizer))

    def get_tokenizer(self) -> Any:
        return self._tokenizer

    def close(self) -> None:
        pass

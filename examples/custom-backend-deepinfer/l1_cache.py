"""Host RAM tier (L1) of the DeepInfer prefix cache (pure Python, no torch).

The GPU prefix cache (``prefix_cache.PrefixIndex``, "L0") evicts idle sessions'
blocks when the KV pool fills, and their next turn recomputes the whole context
(up to 1M tokens, seconds of prefill that stall every data-parallel rank). L1
keeps a copy of every evicted block in pinned host memory and copies it back on
the next hit instead:

* An entry is one row of the engine's paged KV tensor (``gpu_blocks``), named by
  the same chain hash as in L0, so an entry is valid whichever GPU block it
  came from. No per-sequence state is stored: SWA bounded replay recomputes the
  last cached block with a zeroed state slot anyway.
* Demotion (L0 eviction) copies the evicted leaves to host slots on a side
  stream; the GPU blocks stay pinned until the copy completed, so the scheduler
  never hands out a block that is still being read.
* Promotion (a lookup whose L0 hit continues in L1) is driven by ``_RankCache``
  in ``deepinfer_engines.py``: it claims the entries here, copies them into
  fresh GPU blocks and admits the request only after that copy completed.
* L1 is inclusive: a promoted block keeps its host copy, so evicting it again
  is just an unpin.
* Nothing is freed while a copy may still touch it: a copy that failed after
  part of it was queued, or that got stuck, keeps its GPU blocks and host slots
  in quarantine until its done event completed.

Copies go through a KV mover (``kv_mover.py``: the real PyTorch one, or the
bench's simulated one), which this module only calls through ``d2h``, ``h2d``,
``done``, ``checksums`` and ``elapsed``. Everything runs on the rank loop's
thread, between engine steps, like L0.
"""

from __future__ import annotations

import random
import sys
import time
from collections import OrderedDict, deque
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from kv_mover import RingFull
from prefix_cache import BLOCK

READY = 0
WRITING = 1  # the demotion copy is in flight: the host bytes are not valid yet


class _Owner:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


#: Who holds the GPU block's reference while an entry is ``WRITING``: L1 (the
#: demotion unpins it when the copy completes), L0 (a promotion put the block
#: back into the GPU index), or a promotion (its request id, a ``str``).
OWNER_L1 = _Owner("L1")
OWNER_L0 = _Owner("L0")


class Entry:
    __slots__ = ("slot", "state", "busy", "gpu_block", "owner", "checksum", "stuck")

    def __init__(self, slot: int, gpu_block: int) -> None:
        self.slot = slot
        self.state = WRITING
        self.busy = 0  # promotion copies reading the slot; never evicted while > 0
        self.gpu_block = gpu_block  # only while WRITING
        self.owner: Any = OWNER_L1  # only while WRITING
        #: Checksum of the bytes this slot holds (verify): the sampled source sum
        #: of its own demotion copy, else the L0 sum of the very GPU block it was
        #: copied from. Never a sum of another block cached under the same hash.
        self.checksum: Optional[int] = None
        self.stuck = False  # its demotion copy is flagged stuck: not counted as freeing

    def freeing(self) -> bool:
        """Its GPU block returns to the free list when its demotion copy completes."""
        return self.state == WRITING and self.owner is OWNER_L1 and not self.stuck


class HostPool:
    """Host slots: a run-length free list (start -> length), largest runs first.

    A chain demoted in one batch then gets few contiguous runs (copies are
    issued per run), and freeing merges a run with its neighbours in O(1)
    instead of an O(n) sorted insert per slot.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._starts: Dict[int, int] = {}  # start -> length
        self._ends: Dict[int, int] = {}  # end (exclusive) -> start
        self._buckets: Dict[int, Set[int]] = {}  # length -> starts
        self.free = 0
        if capacity > 0:
            self._add(0, capacity)
            self.free = capacity

    def _add(self, start: int, length: int) -> None:
        self._starts[start] = length
        self._ends[start + length] = start
        self._buckets.setdefault(length, set()).add(start)

    def _remove(self, start: int) -> int:
        length = self._starts.pop(start)
        del self._ends[start + length]
        bucket = self._buckets[length]
        bucket.discard(start)
        if not bucket:
            del self._buckets[length]
        return length

    def runs(self) -> int:
        return len(self._starts)

    def alloc(self, n: int) -> List[int]:
        """Up to ``n`` free slots (fewer when the pool has fewer)."""
        out: List[int] = []
        while n > 0 and self._buckets:
            length = max(self._buckets)
            start = next(iter(self._buckets[length]))
            self._remove(start)
            take = min(length, n)
            out.extend(range(start, start + take))
            if take < length:
                self._add(start + take, length - take)
            n -= take
        self.free -= len(out)
        return out

    def release(self, slots: Iterable[int]) -> None:
        ordered = sorted(slots)
        i = 0
        while i < len(ordered):
            start = ordered[i]
            j = i + 1
            while j < len(ordered) and ordered[j] == ordered[j - 1] + 1:
                j += 1
            length = j - i
            self.free += length
            left = self._ends.get(start)
            if left is not None:
                length += self._remove(left)
                start = left
            if start + length in self._starts:
                length += self._remove(start + length)
            self._add(start, length)
            i = j


class _Copy:
    """A demotion copy in flight."""

    __slots__ = ("ticket", "hashes", "gpu_ids", "slots", "t0", "warned", "checked")

    def __init__(self, ticket: Any, hashes: List[bytes], gpu_ids: List[int], slots: List[int],
                 checked: bool) -> None:
        self.ticket = ticket
        self.hashes = hashes
        self.gpu_ids = gpu_ids
        self.slots = slots
        self.t0 = time.monotonic()
        self.warned = False
        self.checked = checked


class _Quarantine:
    """Memory a copy may still touch: released only after its done event."""

    __slots__ = ("ticket", "unpin", "slots", "unbusy")

    def __init__(self, ticket: Any, unpin: List[int], slots: List[int], unbusy: List[bytes]) -> None:
        self.ticket = ticket
        self.unpin = unpin
        self.slots = slots
        self.unbusy = unbusy


def _percentiles(values: Sequence[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2], ordered[min(n - 1, int(n * 0.99))], ordered[-1]


class L1:
    """The host tier of one rank: chain hash -> host slot, LRU, demotions and quarantine.

    ``unpin`` is the block manager's ``unpin_blocks``. ``mover`` is a KV mover
    whose host pool is already pinned (``capacity_slots``).
    """

    #: Stuck copies after which L1 stops taking new work for the rest of the run.
    MAX_STUCK = 3
    #: Entries an eviction scan skips (in flight or in use) before it gives up:
    #: they gather at the recent end, so a long run of them means nothing is evictable.
    SCAN_LIMIT = 4096

    def __init__(self, mover: Any, unpin: Callable[[List[int]], None], *, rank: int = 0,
                 verify_fraction: float = 0.0, copy_timeout: float = 10.0,
                 rng: Optional[random.Random] = None,
                 l0_block: Optional[Callable[[bytes], Optional[int]]] = None) -> None:
        self.mover = mover
        self.unpin = unpin
        #: The GPU block L0 caches a hash in (``PrefixIndex.block_of``), or None.
        self.l0_block = l0_block
        self.rank = rank
        self.bytes_per_block = int(mover.bytes_per_block)
        self.pool = HostPool(int(mover.capacity_slots))
        self.entries: "OrderedDict[bytes, Entry]" = OrderedDict()  # LRU, oldest first
        self.restoring: Dict[bytes, str] = {}  # hash -> request id of the promotion bringing it back
        self.demoting: List[_Copy] = []
        #: Demotions waiting for room in the mover's index ring: (hashes, blocks, slots).
        self.deferred: List[Tuple[List[bytes], List[int], List[int]]] = []
        #: Largest batch per copy (the real mover's index ring bounds it).
        self.max_ticket: Optional[int] = getattr(mover, "max_ticket_blocks", None)
        self.quarantine: List[_Quarantine] = []
        self.verify_fraction = float(verify_fraction)
        self.copy_timeout = float(copy_timeout)
        self.rng = rng or random.Random(0)
        self.enabled = True
        self.disabled_reason = ""
        self.suspect = False  # disabled by a verify failure: host contents are not trusted
        self.stuck = 0
        #: Checksums of blocks as the prefill wrote them (sampled), by hash: (GPU
        #: block, sum), valid while L0 holds that block for the hash (dropped when
        #: L0 evicts it), so a sum is never compared with another block's bytes.
        self.l0_sums: "OrderedDict[bytes, Tuple[int, int]]" = OrderedDict()
        self.l0_sum_tickets: List[Tuple[Any, List[bytes], List[int]]] = []
        self.pending_d2h_blocks = 0  # blocks with a demotion copy in flight or deferred (stats)
        #: Blocks the scheduler gets back when their demotion copies complete: entries
        #: still WRITING, owned by L1 (not rescued or adopted again), not stuck.
        self.freeing_blocks = 0
        self.h2d_ms: Deque[float] = deque(maxlen=2048)
        self.d2h_ms: Deque[float] = deque(maxlen=2048)
        self._window = {"h2d_bytes": 0, "h2d_s": 0.0, "d2h_bytes": 0, "d2h_s": 0.0}
        self.counters: Dict[str, int] = {k: 0 for k in (
            "l1_hits", "l1_hit_tokens", "l1_promotions", "l1_promoted_blocks", "l1_rescued_blocks",
            "l1_demotions", "l1_demoted_blocks", "l1_demote_skipped", "l1_dropped", "l1_evicted",
            "l1_promote_timeouts", "l1_fallbacks", "l1_verify_failures", "l1_checksums", "l1_stuck")}

    # --- helpers ------------------------------------------------------------------

    def log(self, message: str) -> None:
        print(f"deepinfer rank {self.rank}: {message}", file=sys.stderr, flush=True)

    def sample(self) -> bool:
        return self.verify_fraction > 0 and self.rng.random() < self.verify_fraction

    def disable(self, reason: str, suspect: bool = False) -> None:
        """Stop taking new work; ``suspect`` when the host contents may be wrong
        (verify failure), so nothing more is restored from them."""
        if suspect:
            self.suspect = True
        if self.enabled:
            self.enabled = False
            self.disabled_reason = reason
            self.log(f"L1 disabled: {reason}")

    def set_owner(self, entry: Entry, owner: Any) -> None:
        """Change who holds a ``WRITING`` entry's GPU block, keeping ``freeing_blocks``."""
        before = entry.freeing()
        entry.owner = owner
        self.freeing_blocks += entry.freeing() - before

    def _touch_chain(self, hashes: Sequence[bytes]) -> None:
        """Mark a chain used, deepest first, so the block next to the L0 boundary is
        the most recent and the deepest one is evicted first (leaf-first, like L0)."""
        entries = self.entries
        for h in reversed(hashes):
            if h in entries:
                entries.move_to_end(h)

    def _alloc_slots(self, n: int) -> List[int]:
        """Up to ``n`` slots, evicting old ``READY`` entries that nothing uses."""
        short = n - self.pool.free
        if short > 0:
            victims: List[bytes] = []
            freed: List[int] = []
            skipped = 0
            for h, entry in self.entries.items():
                if len(freed) >= short:
                    break
                if entry.state == READY and entry.busy == 0 and h not in self.restoring:
                    victims.append(h)
                    freed.append(entry.slot)
                else:
                    skipped += 1
                    if skipped > self.SCAN_LIMIT:
                        break  # the rest is dropped (l1_dropped) instead of scanning every entry
            for h in victims:
                del self.entries[h]
            if freed:
                self.pool.release(freed)
                self.counters["l1_evicted"] += len(freed)
        return self.pool.alloc(n)

    # --- demotion -------------------------------------------------------------------

    def demote(self, hashes: List[bytes], gpu_ids: List[int], depths: List[int]) -> None:
        """L0 evicted these leaves and passed their pins here: copy them to host
        slots and unpin each block once its copy completed (or right away when
        no copy is needed). Also called when L1 is disabled: a block whose
        demotion copy still reads it is then left to that copy, never unpinned
        early."""
        entries = self.entries
        l0_sums = self.l0_sums
        if not self.enabled:
            unpin_off: List[int] = []
            for h, block in zip(hashes, gpu_ids):
                l0_sums.pop(h, None)
                entry = entries.get(h)
                if entry is not None and entry.state == WRITING and entry.gpu_block == block:
                    self.set_owner(entry, OWNER_L1)  # its copy unpins it when it completes
                else:
                    unpin_off.append(block)
            if unpin_off:
                self.unpin(unpin_off)
            return
        unpin_now: List[int] = []
        need: List[Tuple[int, bytes, int, Optional[int]]] = []
        for h, block, depth in zip(hashes, gpu_ids, depths):
            # L0 no longer holds this block for the hash: its sum goes with it.
            known = l0_sums.pop(h, None)
            entry = entries.get(h)
            if entry is None:
                need.append((depth, h, block, known[1] if known is not None and known[0] == block else None))
            elif entry.state == READY:
                unpin_now.append(block)  # inclusive L1: the host copy is still there
                self.counters["l1_demote_skipped"] += 1
            elif entry.gpu_block == block:
                # A promotion put this block back into L0 while its own demotion
                # copy still reads it: that copy unpins it when it completes.
                self.set_owner(entry, OWNER_L1)
                self.counters["l1_demote_skipped"] += 1
            else:
                unpin_now.append(block)  # another block with the same content is being copied
                self.counters["l1_demote_skipped"] += 1
        if need:
            slots = self._alloc_slots(len(need))
            if len(slots) < len(need):
                # Keep the shallowest blocks: a continuation must start at the L0 boundary.
                need.sort(key=lambda item: item[0])
                for _, _, block, _ in need[len(slots):]:
                    unpin_now.append(block)
                self.counters["l1_dropped"] += len(need) - len(slots)
                need = need[: len(slots)]
            if need:
                self._copy_out(need, slots, unpin_now)
        if unpin_now:
            self.unpin(unpin_now)

    def _copy_out(self, need: List[Tuple[int, bytes, int, Optional[int]]], slots: List[int],
                  unpin_now: List[int]) -> None:
        # Deepest first into the LRU, so the shallowest is the most recent.
        order = sorted(range(len(need)), key=lambda i: -need[i][0])
        hashes = [need[i][1] for i in order]
        blocks = [need[i][2] for i in order]
        for i, h, block, slot in zip(order, hashes, blocks, slots):
            entry = Entry(slot, block)
            entry.checksum = need[i][3]  # the L0 sum of this very block, if one was sampled
            self.entries[h] = entry
        self.pending_d2h_blocks += len(blocks)
        self.freeing_blocks += len(blocks)
        step = self.max_ticket or len(blocks)
        for i in range(0, len(blocks), step):
            self._issue_d2h(hashes[i : i + step], blocks[i : i + step], slots[i : i + step], unpin_now)

    def _issue_d2h(self, hashes: List[bytes], blocks: List[int], slots: List[int], unpin_now: List[int]) -> bool:
        """Queue one demotion copy of entries already added as ``WRITING`` (and
        counted in ``pending_d2h_blocks``); False when it was deferred because
        the mover's index ring is full (``poll`` retries it)."""
        checked = self.sample()
        try:
            ticket = self.mover.d2h(blocks, slots, checksum=checked)
        except RingFull:
            self.deferred.append((hashes, blocks, slots))
            return False
        except Exception as exc:  # noqa: BLE001 (fall back to plain L0 eviction)
            self.pending_d2h_blocks -= len(blocks)
            # Only the references L1 still holds are dropped here: a promotion
            # may have taken a block over (rescue) while the copy was deferred.
            mine = []
            for h, block in zip(hashes, blocks):
                entry = self.entries.pop(h, None)
                if entry is not None and entry.freeing():
                    self.freeing_blocks -= 1
                if entry is None or entry.owner is OWNER_L1:
                    mine.append(block)
            partial = getattr(exc, "ticket", None)
            self.counters["l1_fallbacks"] += 1
            if partial is not None:
                # Part of the copy is queued: keep the blocks pinned and the slots
                # out of use until it completed.
                self.quarantine.append(_Quarantine(partial, mine, slots, []))
                self.log(f"L1 demotion of {len(blocks)} blocks failed after it started ({exc}); "
                         "quarantined until it completes")
            else:
                self.pool.release(slots)
                unpin_now.extend(mine)
                self.log(f"L1 demotion of {len(blocks)} blocks failed ({exc}); dropped")
            return True
        self.demoting.append(_Copy(ticket, hashes, blocks, slots, checked))
        self.counters["l1_demotions"] += 1
        self.counters["l1_demoted_blocks"] += len(blocks)
        return True

    def _retry_deferred(self) -> None:
        deferred, self.deferred = self.deferred, []
        unpin: List[int] = []
        for i, (hashes, blocks, slots) in enumerate(deferred):
            if not self._issue_d2h(hashes, blocks, slots, unpin):
                self.deferred.extend(deferred[i + 1 :])  # still full: keep the order
                break
        if unpin:
            self.unpin(unpin)

    # --- polling ----------------------------------------------------------------------

    def poll(self) -> None:
        """Between steps: finish completed demotions, release quarantine, flag stuck copies."""
        mover = self.mover
        if self.demoting:
            still: List[_Copy] = []
            unpin: List[int] = []
            for copy in self.demoting:
                if mover.done(copy.ticket):
                    self._finish_demotion(copy, unpin)
                else:
                    if not copy.warned and time.monotonic() - copy.t0 > self.copy_timeout:
                        copy.warned = True
                        self._mark_stuck(copy)
                        self._stuck(len(copy.gpu_ids), "device to host")
                    still.append(copy)
            self.demoting = still
            if unpin:
                self.unpin(unpin)
        if self.quarantine:
            still_q: List[_Quarantine] = []
            for item in self.quarantine:
                if mover.done(item.ticket):
                    self._release_quarantine(item)
                else:
                    still_q.append(item)
            self.quarantine = still_q
        if self.l0_sum_tickets:
            still_s = []
            l0_block = self.l0_block
            for ticket, hashes, blocks in self.l0_sum_tickets:
                if mover.done(ticket):
                    sums = mover.checksums(ticket) or []
                    for h, block, s in zip(hashes, blocks, sums):
                        if l0_block is not None and l0_block(h) != block:
                            continue  # L0 evicted it meanwhile: the sum would outlive the block
                        self.l0_sums[h] = (block, s)
                        self.l0_sums.move_to_end(h)
                    while len(self.l0_sums) > max(4 * self.pool.capacity, 1 << 16):
                        self.l0_sums.popitem(last=False)
                else:
                    still_s.append((ticket, hashes, blocks))
            self.l0_sum_tickets = still_s
        if self.deferred:
            self._retry_deferred()

    def _finish_demotion(self, copy: _Copy, unpin: List[int]) -> None:
        sums = self.mover.checksums(copy.ticket) if copy.checked else None
        if sums is not None:
            self.counters["l1_checksums"] += len(sums)
        for i, (h, block, slot) in enumerate(zip(copy.hashes, copy.gpu_ids, copy.slots)):
            entry = self.entries.get(h)
            if entry is None or entry.slot != slot or entry.state != WRITING:
                continue  # cannot happen: WRITING entries are never evicted or dropped
            owner = entry.owner
            if entry.freeing():
                self.freeing_blocks -= 1
            entry.state = READY
            entry.gpu_block = -1
            entry.owner = None
            if sums is not None:
                # entry.checksum, when set, is the L0 sum of this same GPU block.
                expected = entry.checksum
                entry.checksum = sums[i]
                if expected is not None and expected != sums[i]:
                    self.counters["l1_verify_failures"] += 1
                    self.log(f"L1 verify FAILED at demotion: block {block} does not hold what the prefill "
                             "wrote into it for its hash")
                    self.disable("verify failure", suspect=True)
            if owner is OWNER_L1:
                unpin.append(block)
            # OWNER_L0 or a promotion: the block's reference stays where it is.
        self.pending_d2h_blocks -= len(copy.gpu_ids)
        elapsed = float(self.mover.elapsed(copy.ticket))
        self.d2h_ms.append(elapsed * 1000.0)
        self._window["d2h_bytes"] += len(copy.gpu_ids) * self.bytes_per_block
        self._window["d2h_s"] += elapsed

    def _release_quarantine(self, item: _Quarantine) -> None:
        if item.unpin:
            self.unpin(item.unpin)
        if item.slots:
            self.pool.release(item.slots)
        for h in item.unbusy:
            entry = self.entries.get(h)
            if entry is not None and entry.busy > 0:
                entry.busy -= 1

    def _mark_stuck(self, copy: _Copy) -> None:
        """A stuck demotion may never complete: stop counting its blocks as freeing."""
        for h, slot in zip(copy.hashes, copy.slots):
            entry = self.entries.get(h)
            if entry is not None and entry.slot == slot and entry.state == WRITING and not entry.stuck:
                before = entry.freeing()
                entry.stuck = True
                self.freeing_blocks -= before

    def quarantine_copy(self, ticket: Any, unpin: List[int], slots: List[int], unbusy: List[bytes]) -> None:
        self.quarantine.append(_Quarantine(ticket, unpin, slots, unbusy))

    def _stuck(self, blocks: int, direction: str) -> None:
        self.stuck += 1
        self.counters["l1_stuck"] += 1
        self.log(f"L1 copy stuck for {self.copy_timeout:.0f} s ({blocks} blocks, {direction}); "
                 "keeping its blocks out of use")
        if self.stuck >= self.MAX_STUCK:
            self.disable(f"{self.stuck} stuck copies")

    def record_h2d(self, ticket: Any, blocks: int) -> None:
        elapsed = float(self.mover.elapsed(ticket))
        self.h2d_ms.append(elapsed * 1000.0)
        self._window["h2d_bytes"] += blocks * self.bytes_per_block
        self._window["h2d_s"] += elapsed

    def checksum_l0(self, hashes: List[bytes], gpu_ids: List[int]) -> None:
        """Sample checksums of blocks entering L0, as the prefill wrote them (verify)."""
        if not self.enabled or not hashes:
            return
        try:
            ticket = self.mover.checksum_blocks(gpu_ids)
        except Exception as exc:  # noqa: BLE001 (verification is best effort)
            self.log(f"L1 checksum of {len(gpu_ids)} L0 blocks failed: {exc}")
            return
        self.l0_sum_tickets.append((ticket, list(hashes), [int(b) for b in gpu_ids]))

    # --- promotion primitives (driven by _RankCache) -----------------------------------------

    def continuation(self, hashes: Sequence[bytes], n0: int) -> int:
        """How many consecutive hashes from position ``n0`` L1 has (or is restoring)."""
        n = 0
        entries, restoring = self.entries, self.restoring
        for h in hashes[n0:]:
            if h in entries or h in restoring:
                n += 1
            else:
                break
        return n

    def restorers(self, hashes: Iterable[bytes]) -> Set[str]:
        """Request ids of the pending promotions that bring any of ``hashes`` back."""
        restoring = self.restoring
        return {restoring[h] for h in hashes if h in restoring}

    def claim(self, request_id: str, hashes: Sequence[bytes], first: int
              ) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
        """Claim the entries of ``hashes`` (chain positions ``first``, ``first + 1``, ...)
        for a promotion: ``READY`` entries need a copy (``busy += 1``), ``WRITING``
        entries whose block L1 holds are rescued (the promotion takes the still
        pinned GPU block over). Returns ``(h2d, rescued)``, lists of
        ``(position, slot)`` and ``(position, gpu_block)``, or ``None`` (nothing
        claimed) when an entry is gone or held by someone else."""
        h2d: List[Tuple[int, int]] = []
        rescued: List[Tuple[int, int]] = []
        for i, h in enumerate(hashes):
            entry = self.entries.get(h)
            if entry is None or h in self.restoring or (entry.state == WRITING and entry.owner is not OWNER_L1):
                self.unclaim(request_id, hashes[:i], first, h2d, rescued)
                return None
            if entry.state == WRITING:
                self.set_owner(entry, request_id)
                rescued.append((first + i, entry.gpu_block))
            else:
                entry.busy += 1
                h2d.append((first + i, entry.slot))
            self.restoring[h] = request_id
        self._touch_chain(hashes)
        return h2d, rescued

    def unclaim(self, request_id: str, hashes: Sequence[bytes], first: int, h2d: List[Tuple[int, int]],
                rescued: List[Tuple[int, int]], keep_busy: bool = False) -> List[int]:
        """Undo :meth:`claim`; returns the rescued GPU blocks the caller must unpin now
        (the others go back to their demotion copy, which unpins them)."""
        unpin: List[int] = []
        for pos, block in rescued:
            h = hashes[pos - first]
            entry = self.entries.get(h)
            if entry is not None and entry.state == WRITING and entry.owner == request_id:
                self.set_owner(entry, OWNER_L1)
            else:
                unpin.append(block)
        if not keep_busy:
            for pos, _ in h2d:
                entry = self.entries.get(hashes[pos - first])
                if entry is not None and entry.busy > 0:
                    entry.busy -= 1
        for h in hashes:
            if self.restoring.get(h) == request_id:
                del self.restoring[h]
        return unpin

    def settle(self, request_id: str, hashes: Sequence[bytes], first: int, h2d: List[Tuple[int, int]],
               rescued: List[Tuple[int, int]], dropped: Set[int]) -> List[int]:
        """A promotion completed and its blocks went into L0 (except ``dropped``, the
        duplicates L0 already held under other block ids). Returns the rescued
        blocks among ``dropped`` the caller may unpin now."""
        unpin: List[int] = []
        for pos, block in rescued:
            entry = self.entries.get(hashes[pos - first])
            writing = entry is not None and entry.state == WRITING and entry.owner == request_id
            if block in dropped:
                if writing:
                    self.set_owner(entry, OWNER_L1)  # its demotion copy unpins it when done
                else:
                    unpin.append(block)
            elif writing:
                self.set_owner(entry, OWNER_L0)  # L0 holds the reference; the copy just completes
        for pos, _ in h2d:
            entry = self.entries.get(hashes[pos - first])
            if entry is not None and entry.busy > 0:
                entry.busy -= 1
        for h in hashes:
            if self.restoring.get(h) == request_id:
                del self.restoring[h]
        self._touch_chain(hashes)
        return unpin

    def entry_sums(self, hashes: Sequence[bytes]) -> List[Optional[int]]:
        """The checksum of what each hash's slot holds (``Entry.checksum``), or None.

        Only the entry's own sums: ``l0_sums`` may describe another block computed
        for the same hash (batch nondeterminism gives other bytes)."""
        out: List[Optional[int]] = []
        for h in hashes:
            entry = self.entries.get(h)
            out.append(entry.checksum if entry is not None else None)
        return out

    # --- stats ------------------------------------------------------------------------

    def quarantined(self) -> Tuple[int, int]:
        return (sum(len(q.unpin) for q in self.quarantine), sum(len(q.slots) for q in self.quarantine))

    def stats(self) -> Dict[str, Any]:
        h2d = _percentiles(list(self.h2d_ms))
        d2h = _percentiles(list(self.d2h_ms))
        window = self._window
        q_blocks, q_slots = self.quarantined()
        out: Dict[str, Any] = dict(self.counters)
        out.update({
            "l1_enabled": self.enabled,
            "l1_blocks": len(self.entries),
            "l1_capacity_blocks": self.pool.capacity,
            "l1_pinned_bytes": self.pool.capacity * self.bytes_per_block,
            "l1_tokens_capacity": self.pool.capacity * BLOCK,
            "l1_pending_d2h_blocks": self.pending_d2h_blocks,
            "l1_freeing_blocks": self.freeing_blocks,
            "l1_quarantined_blocks": q_blocks,
            "l1_quarantined_slots": q_slots,
            "l1_h2d_ms_p50": round(h2d[0], 3), "l1_h2d_ms_p99": round(h2d[1], 3), "l1_h2d_ms_max": round(h2d[2], 3),
            "l1_d2h_ms_p50": round(d2h[0], 3), "l1_d2h_ms_p99": round(d2h[1], 3), "l1_d2h_ms_max": round(d2h[2], 3),
            "l1_h2d_gbps": round(window["h2d_bytes"] / window["h2d_s"] / 1e9, 2) if window["h2d_s"] else 0.0,
            "l1_d2h_gbps": round(window["d2h_bytes"] / window["d2h_s"] / 1e9, 2) if window["d2h_s"] else 0.0,
        })
        self._window = {"h2d_bytes": 0, "h2d_s": 0.0, "d2h_bytes": 0, "d2h_s": 0.0}
        return out

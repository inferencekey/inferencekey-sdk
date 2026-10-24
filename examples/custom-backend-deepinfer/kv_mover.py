"""KV movers: copy rows of a rank's paged KV tensor to pinned host slots and back.

All paged KV of a DeepInfer rank lives in one GPU tensor,
``cache_engine.gpu_blocks``, shaped ``[num_gpu_blocks, bytes_per_block]``
uint8: a block id is a row, and every layer's cache (``ca_cache``,
``ica_cache``) is a column slice of those rows. So moving row ``b`` moves every
layer's KV of block ``b``, and nothing else is indexed by the block id.

The interface (:class:`KVMover`) is what ``l1_cache.L1`` and ``_RankCache``
call. Two implementations: :class:`TorchKVMover` here (pure PyTorch, in the
rank process) and ``SimKVMover`` in the bench's simulated engine
(``bench/fake_modules/deepinfer/entrypoints/llm.py``). Torch is imported only
inside :class:`TorchKVMover`, so this module loads anywhere.

Ordering (the whole point): every copy batch first waits on an event recorded
on the engine's current stream, so it runs after every GPU write already queued
(a pending zero-overhead-scheduling forward, the prefill that wrote a block).
The caller unpins demoted blocks, or admits a promoted request, only after the
ticket's done event completed (``done()``, polled between steps), and a ticket
that failed after part of it was queued still records its done event (in a
``finally`` block), so its memory can be released safely later.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import platform
import sys
import time
from contextlib import nullcontext
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

try:  # Python 3.8+: typing.Protocol
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment,misc]


class KVMover(Protocol):
    bytes_per_block: int
    capacity_slots: int  # host slots, all pinned by pin_host() at startup

    def pin_host(self, nbytes: int) -> int:
        """Pin the host pool once, before serving; returns the number of slots."""

    def d2h(self, gpu_ids: Sequence[int], slots: Sequence[int], checksum: bool = False) -> Any:
        """Copy GPU rows to host slots; returns an opaque ticket."""

    def h2d(self, slots: Sequence[int], gpu_ids: Sequence[int], checksum: bool = False) -> Any:
        """Copy host slots to GPU rows; returns an opaque ticket."""

    def checksum_blocks(self, gpu_ids: Sequence[int]) -> Any:
        """Row sums of GPU blocks, taken on the copy stream (verify); returns a ticket."""

    def done(self, ticket: Any) -> bool:
        """Non-blocking: whether the ticket's copies completed."""

    def checksums(self, ticket: Any) -> Optional[List[int]]:
        """After done(): per-block sums taken when the bytes moved, or None."""

    def elapsed(self, ticket: Any) -> float:
        """Issue to completion, seconds."""

    def close(self) -> None:
        """Release the host pool (after every copy completed)."""


class PartialIssue(Exception):
    """Issuing failed after some chunks were queued; ``.ticket`` has a done event
    recorded on the copy stream, so the caller quarantines the ticket's memory
    until it completes."""

    def __init__(self, ticket: Any, cause: BaseException) -> None:
        super().__init__(f"copy failed after it started: {cause}")
        self.ticket = ticket


def make_kv_mover(engine: Any, settings: dict, rank: int = 0) -> Tuple[Optional[Any], str]:
    """The engine's own mover when it has one (the bench's ``SimEngine.l1_kv_mover``),
    else a :class:`TorchKVMover` over ``engine.driver_worker.cache_engine.gpu_blocks``.
    Returns ``(mover, "")`` or ``(None, reason)``."""
    if hasattr(engine, "l1_kv_mover"):
        mover = engine.l1_kv_mover
        return (mover, "") if mover is not None else (None, "the engine has no KV mover")
    try:
        gpu_blocks = engine.driver_worker.cache_engine.gpu_blocks
        bytes_per_block = int(engine.cache_config.bytes_per_block)
    except AttributeError as exc:
        return None, f"the engine does not expose its KV tensor ({exc})"
    try:
        mover = TorchKVMover(gpu_blocks, bytes_per_block, rank=rank,
                             staging_mb=int(settings.get("l1_staging_mb", 64)),
                             index_ring=int(settings.get("l1_index_ring", 65536)),
                             segment_mb=int(settings.get("l1_segment_mb", 1024)))
    except Exception as exc:  # noqa: BLE001 (L1 is optional: serve without it)
        return None, f"KV mover setup failed: {exc}"
    return mover, ""


class RingFull(RuntimeError):
    """The index ring has no room for a ticket right now (nothing was queued):
    the caller retries on a later tick, after earlier tickets completed."""


class _Ring:
    """A FIFO range allocator over a pinned index buffer (one per direction).

    Tickets on one stream complete in issue order, so ranges are freed in order
    too; a ticket reserves ``n`` contiguous entries, wrapping to 0 when the end
    has no room."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.ranges: List[List[int]] = []  # [start, end, freed]

    def alloc(self, n: int) -> int:
        if n > self.size:
            raise ValueError(f"{n} blocks exceed the index ring ({self.size})")
        if not self.ranges:
            start = 0
        else:
            head, tail = self.ranges[-1][1], self.ranges[0][0]
            if head > tail:
                start = head if self.size - head >= n else (0 if tail >= n else -1)
            elif head < tail:
                start = head if tail - head >= n else -1
            else:  # head == tail: full
                start = -1
            if start < 0:
                raise RingFull("the L1 index ring is full")
        self.ranges.append([start, start + n, 0])
        return start

    def free(self, start: int) -> None:
        for r in self.ranges:
            if r[0] == start and not r[2]:
                r[2] = 1
                break
        while self.ranges and self.ranges[0][2]:
            self.ranges.pop(0)


class _Ticket:
    __slots__ = ("kind", "n", "ring", "r0", "start", "done_evt", "t_issue", "t_done", "seconds", "sums",
                 "want_sums", "finished")

    def __init__(self, kind: str, n: int) -> None:
        self.kind = kind
        self.n = n
        self.ring: Optional[_Ring] = None
        self.r0 = -1
        self.start: Any = None
        self.done_evt: Any = None
        self.t_issue = time.perf_counter()
        self.t_done = 0.0
        self.seconds = 0.0
        self.sums: Optional[List[int]] = None
        self.want_sums = False
        self.finished = False


def _runs(slots: Sequence[int], per_segment: int) -> Iterator[Tuple[int, int, int]]:
    """``(first slot, offset in the list, count)`` for each run of consecutive slots
    that stays inside one host segment."""
    i, n = 0, len(slots)
    while i < n:
        j = i + 1
        while (j < n and slots[j] == slots[j - 1] + 1 and slots[j] % per_segment != 0):
            j += 1
        yield slots[i], i, j - i
        i = j


class TorchKVMover:
    """The real mover: ``index_select`` / ``index_copy_`` on an int64 view of the
    KV tensor, through a GPU staging buffer, to and from pinned host segments,
    on two side streams (one per direction, PCIe is full duplex).

    The host pool is a set of anonymous ``mmap`` segments registered with
    ``cudaHostRegister`` once at startup (``pin_host``): no power-of-two
    rounding of the caching host allocator, ordinary anonymous pages for the
    cgroup and for NUMA policy, and nothing pinned while serving. Block
    indices go through a pinned ring per direction, so issuing a ticket
    allocates no torch memory.

    With a CPU tensor (unit tests) there are no streams or events: copies run
    synchronously and every ticket is done at once. The tensor logic is the same.
    """

    def __init__(self, gpu_blocks: Any, bytes_per_block: int, *, rank: int = 0, staging_mb: int = 64,
                 index_ring: int = 65536, segment_mb: int = 1024,
                 log: Optional[Callable[[str], None]] = None, staging_rows: Optional[int] = None,
                 slots_per_segment: Optional[int] = None) -> None:
        import torch  # noqa: PLC0415

        self.torch = torch
        self.rank = rank
        self.log = log or (lambda m: print(f"deepinfer rank {rank}: {m}", file=sys.stderr, flush=True))
        if bytes_per_block % 8:
            raise ValueError(f"bytes_per_block {bytes_per_block} is not a multiple of 8")
        if tuple(gpu_blocks.shape[1:]) != (bytes_per_block,) or gpu_blocks.dtype != torch.uint8:
            raise ValueError(f"unexpected KV tensor {tuple(gpu_blocks.shape)} {gpu_blocks.dtype}, "
                             f"expected [N, {bytes_per_block}] uint8")
        self.bytes_per_block = bytes_per_block
        self.width = bytes_per_block // 8
        self.num_gpu_blocks = int(gpu_blocks.shape[0])
        self.blocks64 = gpu_blocks.view(torch.int64)  # 8 bytes per thread instead of 1
        self.device = gpu_blocks.device
        self.cuda = self.device.type == "cuda"
        self.segment_mb = segment_mb
        self.staging_rows = staging_rows or max(1, staging_mb * 2**20 // bytes_per_block)
        self.staging = {d: torch.empty((self.staging_rows, self.width), dtype=torch.int64, device=self.device)
                        for d in ("d2h", "h2d")}
        self.rings = {d: _Ring(index_ring) for d in ("d2h", "h2d")}
        #: Callers split bigger batches, so one ticket never takes more than half the ring.
        self.max_ticket_blocks = max(1, index_ring // 2)
        self._keep: List[Any] = []  # mmaps and ctypes views kept alive with their registrations
        self._registered: List[Tuple[int, int]] = []
        self.ring_host = {d: self._host_buffer(index_ring) for d in ("d2h", "h2d")}
        self.sums_host = {d: self._host_buffer(index_ring) for d in ("d2h", "h2d")}
        self.ring_dev = {d: torch.empty(index_ring, dtype=torch.int64, device=self.device) for d in ("d2h", "h2d")}
        self.sums_dev = {d: torch.empty(index_ring, dtype=torch.int64, device=self.device) for d in ("d2h", "h2d")}
        self.streams = {d: torch.cuda.Stream(device=self.device) for d in ("d2h", "h2d")} if self.cuda else {}
        self.segments: List[Any] = []
        self.slots_per_segment = slots_per_segment or max(1, segment_mb * 2**20 // bytes_per_block)
        self.capacity_slots = 0
        self._orphans: List[_Ticket] = []

    # --- host memory ------------------------------------------------------------------

    def _register(self, mm: mmap.mmap, size: int) -> None:
        if not self.cuda:
            return
        view = ctypes.c_char.from_buffer(mm)
        ptr = ctypes.addressof(view)
        err = self.torch.cuda.cudart().cudaHostRegister(ptr, size, 0)
        if int(err) != 0:
            del view
            raise RuntimeError(f"cudaHostRegister of {size:,} bytes failed with CUDA error {int(err)}")
        self._keep.append(view)
        self._registered.append((ptr, size))

    def _host_buffer(self, entries: int) -> Any:
        """A small pinned int64 buffer (index ring, checksums)."""
        size = max(entries * 8, mmap.PAGESIZE)
        mm = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self._register(mm, size)
        self._keep.append(mm)
        return self.torch.frombuffer(mm, dtype=self.torch.int64, count=entries)

    def pin_host(self, nbytes: int) -> int:
        """Map and register the whole host pool now; raises (with what was pinned
        so far) if a registration fails."""
        torch = self.torch
        total = max(0, int(nbytes) // self.bytes_per_block)
        node = self._numa_node()
        pinned = 0
        t0 = time.perf_counter()
        while self.capacity_slots < total:
            rows = min(self.slots_per_segment, total - self.capacity_slots)
            size = rows * self.bytes_per_block
            mm = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            if node is not None:
                _mbind(mm, size, node)
            try:
                self._register(mm, size)
            except Exception as exc:
                mm.close()
                raise RuntimeError(f"pinning L1 segment {len(self.segments)} ({size:,} bytes) failed after "
                                   f"{pinned:,} bytes: {exc}") from exc
            self._keep.append(mm)
            self.segments.append(torch.frombuffer(mm, dtype=torch.int64).view(rows, self.width))
            self.capacity_slots += rows
            pinned += size
        self.log(f"L1 pinned {pinned / 2**30:.1f} GiB in {len(self.segments)} segments "
                 f"in {time.perf_counter() - t0:.1f} s")
        return self.capacity_slots

    def _numa_node(self) -> Optional[int]:
        """The GPU's NUMA node when this process is not bound to it (the segments are
        then bound explicitly); logs what it found. Best effort."""
        if not self.cuda:
            return None
        try:
            props = self.torch.cuda.get_device_properties(self.device)
            bdf = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
            with open(f"/sys/bus/pci/devices/{bdf}/numa_node", encoding="utf-8") as fh:
                gpu_node = int(fh.read().strip())
            allowed = ""
            with open("/proc/self/status", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("Mems_allowed_list:"):
                        allowed = line.split(":", 1)[1].strip()
        except Exception as exc:  # noqa: BLE001
            self.log(f"L1 NUMA check skipped ({exc})")
            return None
        if gpu_node < 0:
            self.log(f"L1 NUMA: GPU {bdf} reports no node; process Mems_allowed_list {allowed}")
            return None
        bound = allowed == str(gpu_node)
        self.log(f"L1 NUMA: GPU {bdf} on node {gpu_node}, process Mems_allowed_list {allowed} "
                 f"({'bound by engine' if bound else 'binding the host segments explicitly'})")
        return None if bound else gpu_node

    def _host_rows(self, slot: int, n: int) -> Any:
        seg, row = divmod(slot, self.slots_per_segment)
        return self.segments[seg][row : row + n]

    # --- copies ---------------------------------------------------------------------------

    def _validate(self, gpu_ids: Sequence[int], slots: Optional[Sequence[int]]) -> None:
        if slots is not None and len(slots) != len(gpu_ids):
            raise ValueError(f"{len(gpu_ids)} blocks but {len(slots)} slots")
        for b in gpu_ids:
            if not 0 < b < self.num_gpu_blocks:
                raise ValueError(f"GPU block {b} out of range")
        if slots is not None:
            for s in slots:
                if not 0 <= s < self.capacity_slots:
                    raise ValueError(f"host slot {s} out of range")

    def _reap(self) -> None:
        self._orphans = [t for t in self._orphans if not self.done(t)]

    def _issue(self, kind: str, gpu_ids: Sequence[int], slots: Optional[Sequence[int]], checksum: bool) -> _Ticket:
        torch = self.torch
        direction = "h2d" if kind == "h2d" else "d2h"
        self._reap()
        self._validate(gpu_ids, slots)
        n = len(gpu_ids)
        ticket = _Ticket(kind, n)
        ticket.want_sums = checksum
        ring = self.rings[direction]
        r0 = ring.alloc(n)  # raises before anything is queued
        ticket.ring, ticket.r0 = ring, r0
        ring_host, ring_dev = self.ring_host[direction], self.ring_dev[direction]
        sums_host, sums_dev = self.sums_host[direction], self.sums_dev[direction]
        ring_host.numpy()[r0 : r0 + n] = list(gpu_ids)
        staging = self.staging[direction]
        stream = self.streams.get(direction)
        queued = False  # anything on the stream (the done event must then be recorded)
        touched = False  # anything that reads or writes blocks or slots
        if stream is not None:
            order = torch.cuda.Event()
            order.record(torch.cuda.current_stream(self.device))
        try:
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                if stream is not None:
                    stream.wait_event(order)
                    ticket.start = torch.cuda.Event(enable_timing=True)
                    ticket.start.record(stream)
                idx = ring_dev[r0 : r0 + n]
                queued = True
                idx.copy_(ring_host[r0 : r0 + n], non_blocking=True)
                rows = self.staging_rows
                for c0 in range(0, n, rows):
                    k = min(rows, n - c0)
                    touched = True
                    if kind == "h2d":
                        for slot0, off, cnt in _runs(slots[c0 : c0 + k], self.slots_per_segment):
                            staging[off : off + cnt].copy_(self._host_rows(slot0, cnt), non_blocking=True)
                        self.blocks64.index_copy_(0, idx[c0 : c0 + k], staging[:k])
                    else:
                        torch.index_select(self.blocks64, 0, idx[c0 : c0 + k], out=staging[:k])
                        if slots is not None:
                            for slot0, off, cnt in _runs(slots[c0 : c0 + k], self.slots_per_segment):
                                self._host_rows(slot0, cnt).copy_(staging[off : off + cnt], non_blocking=True)
                    if checksum:
                        sums_dev[r0 + c0 : r0 + c0 + k] = staging[:k].sum(dim=1)
                if checksum:
                    sums_host[r0 : r0 + n].copy_(sums_dev[r0 : r0 + n], non_blocking=True)
        except Exception as exc:
            if touched:
                raise PartialIssue(ticket, exc) from exc
            if queued:
                self._orphans.append(ticket)  # only the index copy is queued: free its ring range later
            else:
                ring.free(r0)
            raise
        finally:
            if queued and stream is not None:
                ticket.done_evt = torch.cuda.Event(enable_timing=True)
                ticket.done_evt.record(stream)
        return ticket

    def d2h(self, gpu_ids: Sequence[int], slots: Sequence[int], checksum: bool = False) -> _Ticket:
        return self._issue("d2h", list(gpu_ids), list(slots), checksum)

    def h2d(self, slots: Sequence[int], gpu_ids: Sequence[int], checksum: bool = False) -> _Ticket:
        return self._issue("h2d", list(gpu_ids), list(slots), checksum)

    def checksum_blocks(self, gpu_ids: Sequence[int]) -> _Ticket:
        return self._issue("sum", list(gpu_ids), None, True)

    def done(self, ticket: _Ticket) -> bool:
        if ticket.finished:
            return True
        if ticket.done_evt is not None and not ticket.done_evt.query():
            return False
        ticket.finished = True
        ticket.t_done = time.perf_counter()
        if ticket.start is not None:
            ticket.seconds = ticket.start.elapsed_time(ticket.done_evt) / 1000.0
        else:
            ticket.seconds = ticket.t_done - ticket.t_issue
        if ticket.ring is not None:
            if ticket.want_sums:
                direction = "h2d" if ticket.kind == "h2d" else "d2h"
                ticket.sums = self.sums_host[direction][ticket.r0 : ticket.r0 + ticket.n].tolist()
            ticket.ring.free(ticket.r0)
        return True

    def checksums(self, ticket: _Ticket) -> Optional[List[int]]:
        return ticket.sums

    def elapsed(self, ticket: _Ticket) -> float:
        return ticket.seconds

    def close(self) -> None:
        if self.cuda:
            self.torch.cuda.synchronize(self.device)
            cudart = self.torch.cuda.cudart()
            for ptr, _ in self._registered:
                cudart.cudaHostUnregister(ptr)
        self._registered.clear()
        self.segments.clear()
        self._keep.clear()


def _mbind(mm: mmap.mmap, size: int, node: int) -> None:
    """Bind an untouched anonymous mapping to one NUMA node (MPOL_BIND), best effort."""
    numbers = {"x86_64": 237, "aarch64": 235}
    number = numbers.get(platform.machine())
    if number is None:
        return
    view = ctypes.c_char.from_buffer(mm)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        mask = (ctypes.c_ulong * 16)()
        mask[node // 64] = 1 << (node % 64)
        rc = libc.syscall(number, ctypes.c_void_p(ctypes.addressof(view)), ctypes.c_ulong(size), 2,
                          mask, ctypes.c_ulong(16 * 64), 0)
        if rc != 0:
            print(f"L1: mbind to node {node} failed (errno {ctypes.get_errno()}: "
                  f"{os.strerror(ctypes.get_errno())})", file=sys.stderr, flush=True)
    finally:
        del view

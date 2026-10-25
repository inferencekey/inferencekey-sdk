"""Exact GIL-held fraction of another CPython process, read from its memory.

py-spy's ``--gil`` either pauses every thread at each sample (which halves the
throughput of a server with hundreds of threads) or, with ``--nonblocking``,
reads the GIL's last holder, so a thread that released the GIL around a
``time.sleep`` or a socket write still counts. This module reads the GIL's
``locked`` flag itself: it finds the main interpreter's GIL in the target, then
reads that one int from ``/proc/<pid>/mem`` a couple of thousand times a
second without stopping the target. The fraction of reads that find it locked
is the fraction of time some thread held the GIL.

The struct layout is not hard-coded. The sampler calibrates on its own
interpreter, which must be the same CPython build as the target (load.py runs
both with the bench venv's python): it looks for the pointer in the main
interpreter state whose ``last_holder`` is the calling thread, and checks that
its ``locked`` flag reads 1 while that thread holds the GIL. CPython 3.12 layout
of ``struct _gil_runtime_state``: ``interval`` (unsigned long),
``last_holder`` (pointer), ``locked`` (int).

Reading another process's memory needs ptrace permission: the target is a
child of load.py, and the bench backends call ``allow_ptrace()`` as well.
"""

from __future__ import annotations

import ctypes
import os
import struct
import time
from typing import Dict, List, Optional, Tuple

_PTR = struct.Struct("<Q")
_INT = struct.Struct("<i")
_LAST_HOLDER_OFFSET = 8
_LOCKED_OFFSET = 16


def _mapping_of(pid: str, addr: int) -> Tuple[str, int, int]:
    """(path, mapping start, file offset) of the mapping that holds ``addr``."""
    with open(f"/proc/{pid}/maps", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            lo, hi = (int(x, 16) for x in parts[0].split("-"))
            if lo <= addr < hi:
                return (parts[5] if len(parts) > 5 else ""), lo, int(parts[2], 16)
    raise LookupError(f"address {addr:#x} not mapped in {pid}")


def _find_mapping(pid: int, path: str, offset: int) -> int:
    with open(f"/proc/{pid}/maps", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) > 5 and parts[5] == path and int(parts[2], 16) == offset:
                return int(parts[0].split("-")[0], 16)
    raise LookupError(f"{path} at offset {offset:#x} not mapped in {pid}")


def calibrate() -> Dict[str, object]:
    """Where the main interpreter state is and where its GIL pointer sits, in this interpreter."""
    api = ctypes.pythonapi
    api.PyInterpreterState_Main.restype = ctypes.c_void_p
    api.PyThreadState_Get.restype = ctypes.c_void_p
    interp = api.PyInterpreterState_Main()
    tstate = api.PyThreadState_Get()
    # ctypes.pythonapi calls keep the GIL, so right now this thread holds it.
    # Candidate pointers are read through /proc/self/mem, which fails cleanly
    # (instead of crashing) on an address that is not mapped.
    fd = os.open("/proc/self/mem", os.O_RDONLY)
    try:
        state = os.pread(fd, 4096, interp)
        for off in range(0, len(state) - 7, 8):
            (ptr,) = _PTR.unpack_from(state, off)
            if ptr < 4096 or ptr % 8:
                continue
            try:
                head = os.pread(fd, _LOCKED_OFFSET + 4, ptr)
            except OSError:
                continue
            if len(head) < _LOCKED_OFFSET + 4:
                continue
            (holder,) = _PTR.unpack_from(head, _LAST_HOLDER_OFFSET)
            if holder != tstate:
                continue
            # os.pread released the GIL, so check ``locked`` with a direct read
            # (this address is known to be mapped) while holding it.
            if ctypes.c_int.from_address(ptr + _LOCKED_OFFSET).value == 1:
                path, lo, file_off = _mapping_of("self", interp)
                return {"path": path, "file_offset": file_off, "interp_rel": interp - lo, "gil_ptr_offset": off}
    finally:
        os.close(fd)
    raise LookupError("could not find the GIL in the main interpreter state")


class GilReader:
    """Reads the target's GIL ``locked`` flag without stopping it."""

    def __init__(self, pid: int, layout: Optional[Dict[str, object]] = None) -> None:
        layout = layout or calibrate()
        base = _find_mapping(pid, str(layout["path"]), int(layout["file_offset"]))
        interp = base + int(layout["interp_rel"])
        self.fd = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
        (gil,) = _PTR.unpack(os.pread(self.fd, 8, interp + int(layout["gil_ptr_offset"])))
        if not gil:
            raise LookupError("the target has no GIL yet")
        self.locked_addr = gil + _LOCKED_OFFSET

    def locked(self) -> bool:
        return _INT.unpack(os.pread(self.fd, 4, self.locked_addr))[0] != 0

    def close(self) -> None:
        os.close(self.fd)


def sample(pid: int, stop: object, interval: float = 0.0005, bucket_s: float = 0.1) -> Dict[str, List[int]]:
    """Sample until ``stop.is_set()``; returns {bucket start (ms since epoch): [locked reads, reads]}."""
    reader = GilReader(pid)
    buckets: Dict[str, List[int]] = {}
    step = int(bucket_s * 1000)
    try:
        while not stop.is_set():  # type: ignore[attr-defined]
            key = str(int(time.time() * 1000) // step * step)
            cell = buckets.setdefault(key, [0, 0])
            try:
                cell[0] += reader.locked()
            except OSError:
                break  # the target exited
            cell[1] += 1
            time.sleep(interval)
    finally:
        reader.close()
    return buckets


def sample_into(pid: int, stop: object, out: object) -> None:
    """Process entry point: ``sample()`` and put the buckets (or an error string) on ``out``."""
    try:
        out.put(sample(pid, stop))  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 (report, do not kill the bench)
        out.put(f"{type(exc).__name__}: {exc}")  # type: ignore[attr-defined]


def fraction(buckets: Dict[str, List[int]], t_lo: float, t_hi: float) -> Optional[float]:
    """GIL-held fraction over the buckets that start inside [t_lo, t_hi] (time.time() seconds)."""
    locked = reads = 0
    for key, (k_locked, k_reads) in buckets.items():
        if t_lo <= int(key) / 1000.0 <= t_hi:
            locked += k_locked
            reads += k_reads
    return round(locked / reads, 3) if reads else None

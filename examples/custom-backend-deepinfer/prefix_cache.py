"""GPU prefix cache of one DeepInfer data-parallel rank (pure Python, no torch).

DeepInfer's open release frees every KV block when a request finishes, so the
next turn of a coding-agent session (the same conversation plus a few thousand
new tokens) prefills its whole context again. This index keeps the blocks of
finished prompts alive, pinned in the engine's block manager, and lets a new
request start from the longest cached prefix.

* A prompt is cut into blocks of ``BLOCK`` tokens. Block ``i`` is named by a
  chain hash: ``h_i = blake2b(h_{i-1} || tokens of block i)``, so a hash names
  the whole prefix up to the end of its block, and two sessions that share a
  system prompt share its nodes (and its GPU blocks) once.
* Only full blocks are cached, and only up to the end of the prompt (with
  ``drop_thinking`` the next turn re-renders the previous answer, so its KV is
  never reused).
* Nodes form a tree; only leaves are evicted (least recently used first), so
  every cached chain stays a valid prefix.
* The index holds one pin per cached block (``pin``/``unpin`` are the block
  manager's refcount calls); a running sequence that reuses a block holds its
  own reference, so evicting a block in use frees nothing until it finishes.

The rank loop calls everything from one thread (lookup and eviction between
engine steps, ``insert`` from the ``finalize_seq`` hook inside ``step()``, which
runs on the same thread), so there is no locking here. Tokenizer threads only
call :func:`block_hashes`.
"""

from __future__ import annotations

import hashlib
from array import array
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Sequence

BLOCK = 128
#: Tokens a cache hit must leave to recompute: the 128-token sliding window plus
#: 6 backtracking tokens (DeepInfer's SWA bounded replay; see F0-PLAN 1.3).
REPLAY_GUARD = 134


def block_hashes(token_ids: Sequence[int], block: int = BLOCK) -> List[bytes]:
    """Chain hashes of the full blocks of ``token_ids``."""
    out: List[bytes] = []
    prev = b""
    ids = token_ids if isinstance(token_ids, array) else array("i", token_ids)
    for start in range(0, len(ids) - block + 1, block):
        h = hashlib.blake2b(prev, digest_size=16)
        h.update(ids[start : start + block].tobytes())
        prev = h.digest()
        out.append(prev)
    return out


def usable_hit(hit_blocks: int, prompt_len: int, block: int = BLOCK) -> int:
    """Tokens of a cache hit that can be used: a multiple of ``block``, leaving at
    least ``REPLAY_GUARD`` tokens of the prompt to compute. 0 when too short to
    be worth it (under two blocks)."""
    cap = (prompt_len - REPLAY_GUARD) // block * block
    hit = min(hit_blocks * block, max(cap, 0))
    return hit if hit >= 2 * block else 0


class _Node:
    __slots__ = ("block_id", "parent", "children", "depth")

    def __init__(self, block_id: int, parent: Optional[bytes], depth: int) -> None:
        self.block_id = block_id
        self.parent = parent
        self.children = 0
        self.depth = depth  # position of the block in its chain


class PrefixIndex:
    """Chain hash -> GPU block, with leaf-only LRU eviction.

    The evictable leaves (no children, not protected) are kept in an
    ``OrderedDict``, oldest first, so eviction pops from the front instead of
    sorting every node on each call (about 0.2 s for 226K nodes, on the
    lockstep thread, and with the host tier eviction runs on almost every step).
    """

    def __init__(self, pin: Callable[[List[int]], None], unpin: Callable[[List[int]], None]) -> None:
        self._pin = pin
        self._unpin = unpin
        self._nodes: Dict[bytes, _Node] = {}
        self._leaves: "OrderedDict[bytes, None]" = OrderedDict()  # evictable leaves, least recently used first
        self.protected: Dict[bytes, int] = {}  # node -> running sequences using it
        self.hits = 0
        self.hit_tokens = 0
        self.lookups = 0
        self.inserted = 0
        self.evicted = 0

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, h: bytes) -> bool:
        return h in self._nodes

    def lookup(self, hashes: Sequence[bytes], count: bool = True) -> List[int]:
        """GPU blocks of the longest cached prefix of ``hashes`` (touched as used)."""
        if count:
            self.lookups += 1
        blocks: List[int] = []
        leaves = self._leaves
        for h in hashes:
            node = self._nodes.get(h)
            if node is None:
                break
            if h in leaves:
                leaves.move_to_end(h)
            blocks.append(node.block_id)
        return blocks

    def protect(self, hashes: Iterable[bytes]) -> None:
        for h in hashes:
            n = self.protected.get(h, 0)
            if not n:
                self._leaves.pop(h, None)
            self.protected[h] = n + 1

    def unprotect(self, hashes: Iterable[bytes]) -> None:
        for h in hashes:
            n = self.protected.get(h, 0) - 1
            if n > 0:
                self.protected[h] = n
            elif self.protected.pop(h, None) is not None:
                node = self._nodes.get(h)
                if node is not None and node.children == 0:
                    self._leaves[h] = None

    def insert(self, hashes: Sequence[bytes], block_ids: Sequence[int], adopt_from: Optional[int] = None,
               duplicates: Optional[List[int]] = None) -> int:
        """Cache ``block_ids[i]`` under ``hashes[i]`` for the blocks not cached yet;
        returns how many were added.

        Pass the whole chain from its first block, so parent links stay right.
        Without ``adopt_from`` each added block gets one pin. With
        ``adopt_from = k``, positions ``>= k`` carry a reference the caller
        already holds (a block restored from the host tier), which the index
        takes over instead of pinning; a position ``>= k`` that is already
        cached under another block id is appended to ``duplicates`` so the
        caller drops its reference. Positions ``< k`` behave as without it.
        """
        new_blocks: List[int] = []
        added = 0
        parent: Optional[bytes] = None
        nodes, leaves = self._nodes, self._leaves
        for i, (h, block_id) in enumerate(zip(hashes, block_ids)):
            adopted = adopt_from is not None and i >= adopt_from
            node = nodes.get(h)
            if node is None:
                node = _Node(int(block_id), parent, i)
                nodes[h] = node
                if parent is not None:
                    parent_node = nodes[parent]
                    if parent_node.children == 0:
                        leaves.pop(parent, None)
                    parent_node.children += 1
                if h not in self.protected:
                    leaves[h] = None
                if not adopted:
                    new_blocks.append(int(block_id))
                added += 1
            else:
                if h in leaves:
                    leaves.move_to_end(h)
                if adopted and node.block_id != int(block_id) and duplicates is not None:
                    duplicates.append(int(block_id))
            parent = h
        if new_blocks:
            self._pin(new_blocks)
        self.inserted += added
        return added

    def evict(self, blocks_wanted: int,
              demote: Optional[Callable[[List[bytes], List[int], List[int]], None]] = None) -> int:
        """Drop up to ``blocks_wanted`` least recently used leaves that no running
        sequence uses; returns how many left the index.

        Without ``demote`` their pins are released. With it, ``demote(hashes,
        block_ids, depths)`` takes the pins over (the host tier copies the blocks
        out first), deepest leaves first.
        """
        if blocks_wanted <= 0 or not self._leaves:
            return 0
        nodes, leaves = self._nodes, self._leaves
        hashes: List[bytes] = []
        freed: List[int] = []
        depths: List[int] = []
        while len(freed) < blocks_wanted and leaves:
            h, _ = leaves.popitem(last=False)
            node = nodes.pop(h)
            hashes.append(h)
            freed.append(node.block_id)
            depths.append(node.depth)
            if node.parent is not None:
                parent = nodes.get(node.parent)
                if parent is not None:
                    parent.children -= 1
                    if parent.children == 0 and node.parent not in self.protected:
                        # The parent became a leaf; it is older than anything
                        # touched after it, so it goes next in line.
                        leaves[node.parent] = None
                        leaves.move_to_end(node.parent, last=False)
        if freed:
            self.evicted += len(freed)
            if demote is not None:
                demote(hashes, freed, depths)
            else:
                self._unpin(freed)
        return len(freed)

    def clear(self) -> int:
        blocks = [n.block_id for n in self._nodes.values()]
        self._nodes.clear()
        self._leaves.clear()
        if blocks:
            self._unpin(blocks)
        return len(blocks)

    def stats(self) -> Dict[str, int]:
        return {"cached_blocks": len(self._nodes), "lookups": self.lookups, "hits": self.hits,
                "hit_tokens": self.hit_tokens, "inserted": self.inserted, "evicted": self.evicted}


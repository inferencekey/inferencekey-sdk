"""Engines behind the DeepInfer custom backend.

An engine turns one already-encoded DeepSeek-V4.1 prompt into a stream of
completion text. Two implementations:

* :class:`FakeEngine` (``engine: "fake"``): deterministic, no GPU, no
  DeepInfer. It answers in the DeepSeek-V4.1 completion format (reasoning,
  ``</think>``, text or a DSML tool call, end of sentence) so the backend's
  message mapping, tool-call parsing, streaming and usage can be tested
  anywhere.
* :class:`DeepInferEngine` (``engine: "deepinfer"``, the default): DeepSeek's
  DeepInfer engine. It starts one process per data-parallel rank, each holding a
  ``deepinfer.entrypoints.llm.LLM`` on its own GPUs, and spreads requests over
  the ranks.

What :class:`DeepInferEngine` relies on, as read from the DeepInfer source on
2026-10-21 (github.com/deepseek-ai/DeepInfer, ``deepinfer/entrypoints/llm.py``,
``deepinfer/engine/llm_engine.py``, ``deepinfer/sampling_params.py`` and
``examples/_config.py`` / ``_launch.py`` / ``offline_inference.py``):

* ``LLM(model=..., expert_parallel_size=..., context_parallel_size=...,
  engine_rank=..., data_parallel_size=..., trust_remote_code=True,
  chunked_prefill=True, decode_sequence=False, max_model_len=...,
  max_num_seqs=..., max_num_batched_tokens=..., gpu_memory_utilization=...,
  lookahead_tokens=...)``, built in its own process after
  ``CUDA_VISIBLE_DEVICES`` is set (one process per data-parallel rank, spawn
  context, like ``examples/_launch.py``).
* ``llm.llm_engine.add_request(user_id=..., request_id=..., sampling_params=...,
  prompt_token_ids=np.int32 array, multi_modal_data=...)``,
  ``llm.llm_engine.step()``, ``has_unfinished_requests()`` and
  ``flush_inflight_reqs()``, each step output cloned with
  ``RequestOutput.clone(raw)`` and carrying ``request_id``, ``token_id``,
  ``finished`` and ``prompt_len``: the loop of ``LLM._run_engine``.
* ``has_unfinished_requests()`` all-reduces across the data-parallel group, so
  every rank must call it, and ``step()``, in lockstep. The rank loop below does
  exactly that, also when a rank has nothing to do.
* ``llm.llm_engine.abort_request(request_id)`` (``scheduler.abort_sequence``,
  local to the rank) to drop the request of a client that disconnected.
* ``llm.get_tokenizer()``: the checkpoint's Hugging Face tokenizer, used in the
  rank to tokenize prompts and to detokenize completions. DeepInfer's release
  tokenizer forces ``add_bos_token=True`` and the encoded prompt already starts
  with the BOS text, so prompts are encoded with ``add_special_tokens=False``
  (as DeepInfer's ``eval/run_eval.py`` does).
* ``add_request`` returns ``None``, without raising, when a prompt id is outside
  ``[0, model_config.get_vocab_size())``; such a request never produces output.
  The rank rejects those ids itself and also checks the return value.
* A request the scheduler gives up on (prompt longer than ``max_model_len`` or
  than the KV cache, preempted and stopped, structured output or vision
  failures) comes back as ONE finished ``RequestOutput`` whose ``token_id`` is
  just the sequence's last token and whose ``finish_reason`` says why
  (``csrc/engine/outputs.hpp``, ``sequence.hpp get_finished_reason``). The rank
  turns it into an error instead of completion text.

What has NOT been checked on hardware is marked ``UNVERIFIED`` below and listed
in the README.
"""

from __future__ import annotations

import codecs
import glob
import hashlib
import itertools
import json
import os
import queue
import site
import sys
import threading
import time
import traceback
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional

from prefix_cache import BLOCK, PrefixIndex, block_hashes, usable_hit


@dataclass
class GenRequest:
    """One generation: an encoded prompt and its sampling settings."""

    prompt: str
    max_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 1.0
    seed: Optional[int] = None
    #: Identifies the conversation (the OpenAI ``user`` field, else a hash of
    #: its first messages): its turns go to the rank that caches its prefix.
    session_key: Optional[str] = None
    ignore_eos: bool = False


@dataclass
class GenEvent:
    """A piece of completion text; the last event of a request has ``finished``."""

    text: str = ""
    finished: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    #: Why the request failed, when the backend should answer something other
    #: than a server error: ``"context_length_exceeded"`` or ``"invalid_prompt"``.
    error_code: Optional[str] = None
    #: Prompt tokens served from the prefix cache (None: the engine has no cache).
    cached_tokens: Optional[int] = None


class Engine:
    """Interface: stream the completion of one request."""

    def generate(self, request: GenRequest) -> Iterator[GenEvent]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# Fake engine
# --------------------------------------------------------------------------

EOS = "<｜end▁of▁sentence｜>"
THINK_END = "</think>"
DSML = "｜DSML｜"


class FakeEngine(Engine):
    """Deterministic stand-in that speaks the DeepSeek-V4.1 completion format.

    It looks at the encoded prompt: if the prompt declares tools and holds no
    tool result yet, it calls the first declared tool with ``fake_tool_arguments``
    (config) or a placeholder; otherwise it answers with text. In thinking mode
    (the prompt ends with ``<think>``) it reasons first. The completion is
    streamed a few characters at a time, so streaming is exercised too.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.tool_arguments: Dict[str, Any] = config.get("fake_tool_arguments") or {}
        self.chunk_chars = int(config.get("fake_chunk_chars", 7))

    def generate(self, request: GenRequest) -> Iterator[GenEvent]:
        text = self._completion(request.prompt)
        prompt_tokens = max(1, len(request.prompt) // 4)
        pieces = [text[i : i + self.chunk_chars] for i in range(0, len(text), self.chunk_chars)]
        pieces = pieces[: max(1, request.max_tokens)]
        for i, piece in enumerate(pieces):
            last = i == len(pieces) - 1
            yield GenEvent(
                text=piece,
                finished=last,
                prompt_tokens=prompt_tokens,
                completion_tokens=i + 1,
            )

    def _completion(self, prompt: str) -> str:
        thinking = prompt.endswith("<think>")
        reasoning = "Let me look at the request." + THINK_END if thinking else ""
        tool = self._first_tool(prompt)
        if tool and "<tool_result>" not in prompt:
            params = "".join(
                f'<{DSML} parameter name="{name}" string="{"true" if isinstance(value, str) else "false"}">'
                f"{value if isinstance(value, str) else json.dumps(value)}</{DSML} parameter>\n"
                for name, value in (self.tool_arguments or {"input": "placeholder"}).items()
            )
            call = (
                f"\n\n<{DSML} calls>\n<{DSML} invoke name=\"{tool}\">\n{params}"
                f"</{DSML} invoke>\n</{DSML} calls>"
            )
            return reasoning + call + EOS
        if "<tool_result>" in prompt:
            result = prompt.rsplit("<tool_result>", 1)[1].split("</tool_result>", 1)[0]
            first_line = result.strip().splitlines()[0] if result.strip() else "(empty)"
            return reasoning + f"The tool returned: {first_line}" + EOS
        user = prompt.rsplit("<｜User｜>", 1)[-1].split("<｜Assistant｜>", 1)[0]
        return reasoning + f"echo: {user}" + EOS

    @staticmethod
    def _first_tool(prompt: str) -> Optional[str]:
        marker = "### Available Tool Schemas"
        if marker not in prompt:
            return None
        schemas = prompt.split(marker, 1)[1]
        for line in schemas.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)["name"]
                except (ValueError, KeyError):
                    continue
        return None


# --------------------------------------------------------------------------
# Incremental detokenization (runs in the rank processes)
# --------------------------------------------------------------------------


class IncrementalDetokenizer:
    """Turns the token ids of one completion into text, a few tokens at a time.

    ``feed`` returns the text that the new tokens make final; text that could
    still change (the first bytes of a multi-byte character split across tokens)
    is held back. ``flush`` returns what is left at the end. The concatenation of
    every ``feed`` and the final ``flush`` equals ``tokenizer.decode(all_ids,
    skip_special_tokens=False)``.
    """

    def feed(self, token_ids: List[int]) -> str:
        raise NotImplementedError

    def flush(self) -> str:
        raise NotImplementedError


def _byte_level_alphabet() -> Dict[str, int]:
    """The byte each character of a byte-level BPE vocabulary stands for (GPT-2's
    ``bytes_to_unicode``, inverted)."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    alphabet = {chr(b): b for b in printable}
    shifted = 0
    for b in range(256):
        if b not in alphabet.values():
            alphabet[chr(256 + shifted)] = b
            shifted += 1
    return alphabet


_TABLE_CHECK_TEXT = "Hello, ¿qué tal? 你好世界 🎉 naïve </think>\n\n\tdef f(x):  return x"


def byte_level_table(tokenizer: Any) -> Optional[List[bytes]]:
    """The bytes of every token id, when decoding is just "join the bytes, then UTF-8".

    That holds for a Hugging Face fast tokenizer with a ``ByteLevel`` decoder and
    no ``clean_up_tokenization_spaces`` (DeepSeek-V4.1's): the decoder maps each
    token's characters back to bytes (a token with a character outside the
    byte alphabet, like an added special token, contributes its UTF-8), drops
    unknown ids, and decodes the whole with replacement characters. The table is
    checked against the tokenizer's own ``decode`` for every single id and for a
    multilingual sample; any difference, or any other tokenizer, returns
    ``None`` and the caller falls back to :class:`WindowDetokenizer`.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or getattr(tokenizer, "clean_up_tokenization_spaces", False):
        return None
    try:
        decoder = json.loads(backend.to_str()).get("decoder") or {}
        if decoder.get("type") != "ByteLevel":
            return None
        alphabet = _byte_level_alphabet()
        size = backend.get_vocab_size(with_added_tokens=True)
        table: List[bytes] = []
        for token_id in range(size):
            token = backend.id_to_token(token_id)
            if token is None:
                table.append(b"")
                continue
            try:
                table.append(bytes(alphabet[c] for c in token))
            except KeyError:
                table.append(token.encode("utf-8"))
        singles = backend.decode_batch([[i] for i in range(size)], skip_special_tokens=False)
        if any(piece.decode("utf-8", "replace") != text for piece, text in zip(table, singles)):
            return None
        sample = list(backend.encode(_TABLE_CHECK_TEXT).ids)
        joined = b"".join(table[i] for i in sample).decode("utf-8", "replace")
        if joined != backend.decode(sample, skip_special_tokens=False):
            return None
    except Exception:  # noqa: BLE001 (any surprise: use the generic path)
        return None
    return table


class ByteLevelDetokenizer(IncrementalDetokenizer):
    """Table lookups plus an incremental UTF-8 decoder: about a microsecond per token."""

    __slots__ = ("_table", "_size", "_utf8")

    def __init__(self, table: List[bytes]) -> None:
        self._table = table
        self._size = len(table)
        self._utf8 = codecs.getincrementaldecoder("utf-8")("replace")

    def feed(self, token_ids: List[int]) -> str:
        table, size = self._table, self._size
        data = b"".join(table[t] for t in token_ids if 0 <= t < size)
        return self._utf8.decode(data)

    def flush(self) -> str:
        # An unfinished multi-byte character at the very end becomes U+FFFD, as in decode().
        return self._utf8.decode(b"", True)


class WindowDetokenizer(IncrementalDetokenizer):
    """Generic incremental decoding for any tokenizer with ``decode``.

    The prefix and read offsets of vLLM and TGI: decode a short window that
    ends at the last emitted token and the same window with the new tokens, and
    emit the difference, unless it ends in U+FFFD (a multi-byte character still
    incomplete) or did not grow (tokens that merge with the next ones). Two
    small decodes per feed, never the whole completion.
    """

    __slots__ = ("_decode", "_ids", "_read")

    def __init__(self, tokenizer: Any) -> None:
        self._decode = tokenizer.decode
        self._ids: List[int] = []  # the window: context tokens, then tokens not yet emitted
        self._read = 0  # how many tokens of the window were emitted already

    def feed(self, token_ids: List[int]) -> str:
        self._ids.extend(token_ids)
        return self._advance(final=False)

    def flush(self) -> str:
        return self._advance(final=True)

    def _advance(self, final: bool) -> str:
        ids = self._ids
        if self._read == len(ids):
            return ""
        before = self._decode(ids[: self._read], skip_special_tokens=False) if self._read else ""
        after = self._decode(ids, skip_special_tokens=False)
        if not final and (len(after) <= len(before) or after.endswith("�")):
            return ""
        # Slide: the tokens just emitted become the context of the next window.
        self._ids = ids[self._read :]
        self._read = len(self._ids)
        return after[len(before) :]


def detokenizer_factory(tokenizer: Any) -> Callable[[], IncrementalDetokenizer]:
    """A constructor of per-request detokenizers: the byte table when it is exact, else windows."""
    table = byte_level_table(tokenizer)
    if table is not None:
        return lambda: ByteLevelDetokenizer(table)
    return lambda: WindowDetokenizer(tokenizer)


# --------------------------------------------------------------------------
# DeepInfer engine
# --------------------------------------------------------------------------


#: ``GenEvent.error_code`` values: client errors the backend answers with a 400.
CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
INVALID_PROMPT = "invalid_prompt"

#: ``RequestOutput.finish_reason`` values of requests the scheduler gave up on
#: (``sequence.hpp get_finished_reason``), with the error code each maps to.
#: Structured-output failures start with ``"xgrammar"``.
_ENGINE_GAVE_UP = {
    "prompt_length_exceeded": CONTEXT_LENGTH_EXCEEDED,
    "insufficient_system_resource": None,
    "ve_error": None,
}


def _gave_up(reason: str) -> bool:
    return reason in _ENGINE_GAVE_UP or reason.startswith("xgrammar")


def _rank_devices(visible: List[str], rank: int, cp_size: int) -> List[str]:
    """GPUs of one data-parallel rank: ``cp_size`` consecutive entries.

    Mirrors ``setup_cuda_visible_devices`` in DeepInfer's ``examples/_config.py``,
    but over the devices the workload was given (``config.env.CUDA_VISIBLE_DEVICES``)
    instead of a ``DEVICE_OFFSET``.
    """
    start = rank * cp_size % len(visible)
    return visible[start : start + cp_size]


def _rank_main(
    rank: int,
    settings: Dict[str, Any],
    requests: Any,
    results: Any,
) -> None:
    """One data-parallel rank: build an LLM, then loop in lockstep with the others."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(settings["devices"][rank])
    os.environ.update(settings["env"])

    import numpy as np  # noqa: PLC0415 (DeepInfer must be imported after the env is set)

    import torch  # noqa: F401,PLC0415 (loads libc10 before deepinfer._C)
    from deepinfer._C.engine import RequestOutput  # type: ignore[import-not-found]
    from deepinfer.entrypoints.llm import LLM  # type: ignore[import-not-found]
    from deepinfer.sampling_params import SamplingParams  # type: ignore[import-not-found]

    # UNVERIFIED: constructor arguments copied from DeepInfer's examples/_config.py
    # create_llm() and offline_inference.py; not run on hardware by us.
    llm = LLM(
        model=settings["model_path"],
        expert_parallel_size=settings["ep"],
        context_parallel_size=settings["cp"],
        engine_rank=rank,
        data_parallel_size=settings["dp"],
        trust_remote_code=True,
        chunked_prefill=True,
        decode_sequence=False,
        disable_log_stats=False,
        **settings["llm_kwargs"],
    )
    engine = llm.llm_engine
    needs_mm = engine.multi_modal_processor is not None
    vocab_size = int(engine.model_config.get_vocab_size())
    try:
        _rank_loop(
            rank,
            engine,
            llm.get_tokenizer(),
            requests,
            results,
            settings,
            clone=RequestOutput.clone,
            sampling_params=SamplingParams,
            token_array=lambda ids: np.asarray(ids, dtype=np.int32),
            multi_modal_data=lambda: dict(image_bytes=[], image_sizes=[]) if needs_mm else None,
            vocab_size=vocab_size,
            cache=_RankCache(rank, engine, settings) if settings.get("prefix_cache", True) else None,
        )
    finally:
        llm.close()


class _RankCache:
    """The GPU prefix cache of one rank: a :class:`PrefixIndex` and the engine calls around it.

    Needs DeepInfer with the local prefix-cache patch
    (``deepinfer-patch/0002-local-prefix-cache.patch``):
    ``scheduler.add_sequence_with_prefix(seq, blocks, num_cached, swa_valid_start)``
    and ``Sequence.block_table``. Uses the engine's own ``block_manager``
    (``pin_blocks`` / ``unpin_blocks`` / ``get_num_free_blocks``),
    ``cache_config.num_gpu_blocks``, ``add_request(enqueue=False)`` and
    ``finalize_seq``, which this class wraps to cache a finished prompt's blocks
    before the engine frees them. Everything runs on the rank loop's thread.
    """

    def __init__(self, rank: int, engine: Any, settings: Dict[str, Any]) -> None:
        self.rank = rank
        self.engine = engine
        self.blocks = engine.block_manager
        self.total_blocks = int(engine.cache_config.num_gpu_blocks)
        self.index = PrefixIndex(self.blocks.pin_blocks, self.blocks.unpin_blocks)
        # Free blocks kept on top of what waiting prompts need: DeepInfer stops
        # admitting prefill above 90 % use (GPU_CACHE_PREFILL_WATERMARK), and a
        # sequence that runs out of blocks during decode is preempted.
        self.headroom = int((0.10 + float(settings.get("prefix_cache_free_fraction", 0.05))) * self.total_blocks)
        self.waiting_len = getattr(engine.scheduler, "get_sum_prompt_len_waiting", None)
        self.live: Dict[str, tuple] = {}  # request id -> (hashes, prompt_len, hashes the sequence reuses)
        self.preempt_check = getattr(engine.scheduler, "get_num_preempted_seqs", None)
        finalize = engine.finalize_seq

        def finalize_seq(seq: Any) -> None:
            try:
                self._finished(seq)
            except Exception:  # noqa: BLE001 (never break the engine step)
                traceback.print_exc(file=sys.stderr)
            finalize(seq)

        engine.finalize_seq = finalize_seq
        print(f"deepinfer rank {rank}: prefix cache on, {self.total_blocks} KV blocks of {BLOCK} tokens "
              f"({self.total_blocks * BLOCK:,} tokens), headroom {self.headroom} blocks", file=sys.stderr, flush=True)

    def add(self, add_request: Callable[..., Any], request_id: str, token_ids: Any, hashes: List[bytes],
            max_tokens: int) -> "tuple[Any, int]":
        """Add a request, starting from its longest cached prefix; returns (sequence, cached tokens)."""
        prompt_len = len(token_ids)
        blocks = self.index.lookup(hashes)
        hit = usable_hit(len(blocks), prompt_len)
        # The block [hit - BLOCK, hit) is recomputed with the new tokens (SWA
        # bounded replay), so it gets a fresh block and the cached one is never written.
        reused = hit // BLOCK - 1 if hit else 0
        used = hashes[:reused]
        self.index.protect(used)  # before making room, so the hit is not the block evicted
        self.make_room(prompt_len - hit + max_tokens)
        if not hit:
            seq = add_request(enqueue=True)
            if seq is not None:
                self.live[request_id] = (hashes, prompt_len, [])
            return seq, 0
        seq = add_request(enqueue=False)
        if seq is None:
            self.index.unprotect(used)
            return None, 0
        self.engine.scheduler.add_sequence_with_prefix(seq, blocks[:reused], hit - BLOCK, hit - BLOCK)
        if self.engine.multi_modal_processor is not None:
            self.engine.multi_modal_processor.add(request_id, None)
        self.index.hits += 1
        self.index.hit_tokens += hit - BLOCK
        self.live[request_id] = (hashes, prompt_len, used)
        return seq, hit - BLOCK

    def make_room(self, new_tokens: int) -> None:
        want = -(-max(new_tokens, 0) // BLOCK) + self.headroom
        free = int(self.blocks.get_num_free_blocks())
        if free < want:
            self.index.evict(want - free)

    def tick(self) -> None:
        """Between steps: evict for the prompts already waiting in the engine's queue."""
        if not len(self.index) or self.waiting_len is None:
            return
        waiting, prompt_len = self.waiting_len()
        if waiting:
            self.make_room(min(int(prompt_len), self.total_blocks * BLOCK))

    def forget(self, request_id: str) -> None:
        """A request left without finishing in the engine (aborted): drop its protection."""
        entry = self.live.pop(request_id, None)
        if entry is not None:
            self.index.unprotect(entry[2])

    def _finished(self, seq: Any) -> None:
        entry = self.live.pop(seq.request_id, None)
        if entry is None:
            return
        hashes, prompt_len, used = entry
        self.index.unprotect(used)
        # Prompt blocks only, and never the last prompt token's block: with
        # drop_thinking the next turn re-renders everything after the prompt.
        n = min(len(hashes), (prompt_len - 1) // BLOCK)
        table = list(seq.block_table)[:n]
        if n and len(table) == n:
            self.index.insert(hashes[:n], table)

    def stats(self) -> Dict[str, Any]:
        out = {"rank": self.rank, "total_blocks": self.total_blocks,
               "free_blocks": int(self.blocks.get_num_free_blocks()), "live": len(self.live), **self.index.stats()}
        if self.preempt_check is not None:
            out["preempted"] = int(self.preempt_check())
        return out


class _Live:
    """A request this rank has handed to the engine."""

    __slots__ = ("detokenizer", "tokens")

    def __init__(self, detokenizer: IncrementalDetokenizer) -> None:
        self.detokenizer = detokenizer
        self.tokens = 0


def _rank_loop(
    rank: int,
    engine: Any,
    tokenizer: Any,
    requests: Any,
    results: Any,
    settings: Dict[str, Any],
    *,
    clone: Callable[[Any], Any],
    sampling_params: Callable[..., Any],
    token_array: Callable[[List[int]], Any],
    multi_modal_data: Callable[[], Any],
    vocab_size: Optional[int] = None,
    cache: Optional[_RankCache] = None,
) -> None:
    """The rank's serving loop, apart from building the engine (so tests can drive it).

    Messages in (``requests``): ``("add", request_id, prompt, params)``,
    ``("abort", request_id)`` and ``None`` (stop). Messages out (``results``):
    ``("ready", rank, None)`` once, then at most one ``("out", rank, items)``
    per engine step, ``items`` being one tuple per request that advanced:
    ``(request_id, text, completion_tokens, prompt_tokens, finished, error)``,
    with the text these tokens made final and the totals so far. A failed
    request's last item carries a seventh field, its error code (see
    ``CONTEXT_LENGTH_EXCEEDED`` and ``INVALID_PROMPT``) or ``None``.
    With a prefix ``cache``: ``("meta", rank, [(request_id, cached_tokens), ...])``
    for the requests added in an iteration (before their first output), and
    ``("stats", rank, {...})`` every ``stats_interval_seconds``.

    The parent never sees a token id: prompts are tokenized here, on a few
    threads (the Hugging Face fast tokenizer releases the GIL while it encodes,
    so a 137K-token prompt does not stall the lockstep loop), and completions
    are detokenized here, incrementally, per request.
    """
    new_detokenizer = detokenizer_factory(tokenizer)
    idle_sleep = float(settings["idle_poll_seconds"])
    stats_every = float(settings.get("stats_interval_seconds", 2.0))
    next_stats = time.monotonic()
    abort_on_disconnect = bool(settings.get("abort_on_disconnect", True))
    pool = ThreadPoolExecutor(
        max_workers=max(1, int(settings.get("tokenizer_threads", 2))),
        thread_name_prefix=f"deepinfer-rank{rank}-tokenize",
    )
    tokenized: "Deque[tuple]" = deque()  # filled by the pool, drained by the loop
    tokenizing: Dict[str, bool] = {}  # request id -> aborted while tokenizing
    live: Dict[str, _Live] = {}

    def tokenize(request_id: str, prompt: str, params: Dict[str, Any]) -> None:
        try:
            # The encoded prompt starts with the BOS text already; the release
            # tokenizer would add a second BOS id otherwise.
            ids = tokenizer.encode(prompt, add_special_tokens=False)
        except Exception as exc:  # noqa: BLE001 (report it on the request)
            tokenized.append((request_id, None, params, (f"tokenization failed: {exc}", None), []))
            return
        if vocab_size is not None and ids and max(ids) >= vocab_size:
            # The release tokenizer registers multimodal placeholders as added
            # tokens past the vocabulary; a message quoting one would encode to
            # such an id, and the engine would drop the request silently.
            bad = next(i for i in ids if i >= vocab_size)
            error = (f"the prompt holds token id {bad}, outside the model vocabulary ({vocab_size}); "
                     f"the messages contain reserved special-token text: {tokenizer.decode([bad])!r}")
            tokenized.append((request_id, None, params, (error, INVALID_PROMPT), []))
            return
        hashes = block_hashes(ids) if cache is not None else []
        tokenized.append((request_id, token_array(ids), params, None, hashes))

    results.put(("ready", rank, None))
    pending_flush = False
    try:
        while True:
            items: List[tuple] = []
            # Messages from the parent since the last iteration.
            while True:
                try:
                    message = requests.get_nowait()
                except queue.Empty:
                    break
                if message is None:
                    return
                if message[0] == "add":
                    _, request_id, prompt, params = message
                    tokenizing[request_id] = False
                    pool.submit(tokenize, request_id, prompt, params)
                elif message[0] == "abort":
                    request_id = message[1]
                    if request_id in tokenizing:
                        tokenizing[request_id] = True
                    elif live.pop(request_id, None) is not None and abort_on_disconnect:
                        if cache is not None:
                            cache.forget(request_id)
                        try:
                            # UNVERIFIED: aborting between steps while other requests run.
                            engine.abort_request(request_id)
                        except Exception as exc:  # noqa: BLE001 (keep the rank alive)
                            print(f"deepinfer rank {rank}: abort of {request_id} failed: {exc}",
                                  file=sys.stderr, flush=True)

            # Prompts tokenized since the last iteration go to the engine.
            metas: List[tuple] = []
            while tokenized:
                request_id, token_ids, params, error, hashes = tokenized.popleft()
                if tokenizing.pop(request_id, False):
                    continue  # the client left while its prompt was being tokenized
                if error is None:
                    params = dict(params)
                    user_id = (params.pop("session_key", None) or "0")[:64]
                    try:
                        # UNVERIFIED: adding requests between steps while others are in flight
                        # (the offline example adds them all before the first step).
                        def add(enqueue: bool = True) -> Any:
                            return engine.add_request(
                                user_id=user_id,
                                request_id=request_id,
                                sampling_params=sampling_params(**params),
                                prompt_token_ids=token_ids,
                                multi_modal_data=multi_modal_data(),
                                **({} if enqueue else {"enqueue": False}),
                            )

                        if cache is not None:
                            seq, cached = cache.add(add, request_id, token_ids, hashes, int(params["max_tokens"]))
                            metas.append((request_id, cached))
                        else:
                            seq = add()
                        if seq is None:
                            # Dropped without an exception (a token id out of range): it
                            # would never produce an output, so it must not wait for one.
                            error = ("the engine rejected the prompt (token id out of range)", INVALID_PROMPT)
                    except Exception as exc:  # noqa: BLE001 (report and keep the rank alive)
                        error = (str(exc), None)
                if error is None:
                    live[request_id] = _Live(new_detokenizer())
                else:
                    items.append((request_id, "", 0, 0, True, *error))

            # Collective across the data-parallel group: every rank calls it.
            # UNVERIFIED: that idle ranks may keep calling it (and step() with an
            # empty local batch) indefinitely, as a long-lived server needs.
            if cache is not None:
                cache.tick()
            unfinished = engine.has_unfinished_requests()
            if unfinished:
                outputs = engine.step()
                pending_flush = True
            elif pending_flush:
                # Same value on every rank (they all stepped), so the flush stays in lockstep.
                outputs = engine.flush_inflight_reqs()
                pending_flush = False
            else:
                if metas:
                    results.put(("meta", rank, metas))
                if items:
                    results.put(("out", rank, items))
                if cache is not None and time.monotonic() >= next_stats:
                    next_stats = time.monotonic() + stats_every
                    results.put(("stats", rank, cache.stats()))
                time.sleep(idle_sleep)
                continue

            if outputs:
                _collect_step(outputs, clone, live, items)
            if metas:
                results.put(("meta", rank, metas))
            if items:
                results.put(("out", rank, items))
            if cache is not None:
                for item in items:
                    if item[4]:
                        cache.forget(item[0])  # finished or failed: no-op once finalize ran
                if time.monotonic() >= next_stats:
                    next_stats = time.monotonic() + stats_every
                    results.put(("stats", rank, cache.stats()))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _collect_step(outputs: Any, clone: Callable[[Any], Any], live: Dict[str, _Live], items: List[tuple]) -> None:
    """Group one step's outputs by request, detokenize them and append one item per request.

    A request the scheduler gave up on finishes with a ``finish_reason`` such as
    ``prompt_length_exceeded`` and a token that was never generated (the last
    prompt token, or a repeat of the last one sent): that token is dropped and
    the request ends with an error instead.
    """
    step: Dict[str, list] = {}
    for raw in outputs:
        # clone: the step's outputs are released by the next step.
        out = clone(raw)
        entry = step.get(out.request_id)
        if entry is None:
            entry = step[out.request_id] = [[], False, 0, None]
        finished = bool(out.finished)
        reason = getattr(out, "finish_reason", "") if finished else ""
        if reason and _gave_up(reason):
            entry[3] = reason
        else:
            entry[0].append(int(out.token_id))
        entry[1] = entry[1] or finished
        entry[2] = int(out.prompt_len)
    for request_id, (token_ids, finished, prompt_len, gave_up) in step.items():
        state = live.get(request_id)
        if state is None:
            continue  # aborted
        state.tokens += len(token_ids)
        text = state.detokenizer.feed(token_ids) if token_ids else ""
        if finished:
            text += state.detokenizer.flush()
            del live[request_id]
        elif not text:
            continue  # nothing final yet (half a character); the totals ride on the next item
        if gave_up:
            error = f"DeepInfer finished the request: {gave_up}"
            items.append((request_id, text, state.tokens, prompt_len, True, error, _ENGINE_GAVE_UP.get(gave_up)))
        else:
            items.append((request_id, text, state.tokens, prompt_len, finished, None))


class _Stream:
    """Parent-side mailbox of one request: the reader appends, the request's thread drains."""

    __slots__ = ("items", "wake", "done", "cached")

    def __init__(self) -> None:
        self.cached: Optional[int] = None
        self.items: "Deque[tuple]" = deque()
        self.wake = threading.Event()  # items arrived
        self.done = threading.Event()  # the last item arrived

    def drain(self) -> List[tuple]:
        items = []
        try:
            while True:
                items.append(self.items.popleft())
        except IndexError:
            return items


def _merge(items: List[tuple]) -> GenEvent:
    """Everything a request received since its thread last ran, as one event."""
    last = items[-1]
    failed = next((item for item in items if item[5]), None)
    return GenEvent(
        text=last[1] if len(items) == 1 else "".join(item[1] for item in items),
        finished=bool(last[4]) or failed is not None,
        prompt_tokens=last[3],
        completion_tokens=last[2],
        error=failed[5] if failed else None,
        error_code=failed[6] if failed and len(failed) > 6 else None,
    )


def use_deepinfer_venv(venv: Optional[str], ctx: Any) -> None:
    """Run DeepInfer from a prebuilt venv instead of the backend's own.

    DeepInfer is compiled in place by its ``build.sh`` (no installable wheel)
    and pulls torch, its vendored CUDA kernels and transformers. The worker
    gives every custom backend a fresh venv, so the heavy one is built once on
    the node (``deepinfer_venv``, e.g. ``/workspace/di-venv`` on a network
    volume) and used from here: its site-packages go at the END of
    ``sys.path`` (the backend's own packages keep precedence) and the rank
    processes start with its interpreter. Both venvs must use the same Python
    minor version. No-op when ``venv`` is empty.
    """
    if not venv:
        return
    python = os.path.join(venv, "bin", "python")
    if not os.path.exists(python):
        raise FileNotFoundError(f"deepinfer_venv {venv!r} has no bin/python")
    want = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sites = sorted(glob.glob(os.path.join(venv, "lib", want, "site-packages")))
    if not sites:
        raise FileNotFoundError(f"deepinfer_venv {venv!r} has no lib/{want}/site-packages (Python mismatch?)")
    for site_dir in sites:
        if site_dir not in sys.path:
            # addsitedir also runs the venv's .pth files (DeepInfer's build
            # makes its source tree importable through one).
            site.addsitedir(site_dir)
    ctx.set_executable(python)


class DeepInferEngine(Engine):
    """DeepInfer with one process per data-parallel rank and a session-affinity router.

    The open release of DeepInfer ships no load balancer across data-parallel
    replicas; this router is ours. With the prefix cache on (``prefix_cache``,
    default), every turn of a conversation goes to the rank that holds its
    prefix (a new conversation goes to the rank with the least KV in use), and
    moves only when that rank already has ``session_migrate_inflight`` requests
    in flight: a move costs a full cold prefill on the new rank. Without the
    cache, the rank with the fewest requests in flight wins.

    The work is split so this process (one GIL for every HTTP stream) stays
    cheap: the ranks tokenize prompts, detokenize completions incrementally and
    send one message per engine step with the new text of every request that
    advanced. Here a reader thread hands each item to its request's mailbox,
    and the request's thread takes everything that arrived since it last ran as
    one event, so a busy server sends fewer, larger SSE chunks instead of
    falling behind.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        import multiprocessing as mp  # noqa: PLC0415

        ctx = mp.get_context("spawn")
        use_deepinfer_venv(config.get("deepinfer_venv"), ctx)

        ep = int(config.get("ep", 4))
        cp = int(config.get("cp", 1))
        dp = int(config.get("dp") or max(ep // cp, 1))
        if (dp * cp) % ep != 0:
            raise ValueError(f"dp * cp ({dp} * {cp}) must be a multiple of ep ({ep})")
        visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
        if not visible:
            visible = [str(i) for i in range(dp * cp)]
        if len(visible) < dp * cp:
            raise ValueError(f"{dp * cp} GPUs needed (dp {dp} x cp {cp}), CUDA_VISIBLE_DEVICES lists {len(visible)}")

        model_path = resolve_model_path(config)
        env = {
            # Defaults of DeepInfer's examples/_config.py setup_common_env().
            "DEEPINFER_USE_NEXTN_DECODE": str(config.get("nextn_decode", 5)),
            "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS": str(config.get("extra_experts", 32)),
            "DEEPINFER_EXPERT_COUNT_WINDOW_SIZE": "3",
            # DSpark speculative decoding and SWA bounded replay (README "Features").
            "DEEPINFER_ENABLE_MTP_VARLEN": "1" if config.get("dspark", True) else "0",
        }
        if config.get("swa_bounded_replay", True):
            env["DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE"] = "compact"
            env["DEEPINFER_YOCO_PREFILL_KEEP_TOKENS"] = str(config.get("swa_keep_tokens", 140))
        if config.get("eplb", True):
            env["DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL"] = "30"
            env["DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE"] = "1"
        env.update({k: str(v) for k, v in (config.get("deepinfer_env") or {}).items()})

        # UNVERIFIED: the context length and batch sizes that fit 4 or 8 B200.
        max_model_len = int(config.get("max_model_len", 262144))
        settings = {
            "model_path": model_path,
            "ep": ep,
            "cp": cp,
            "dp": dp,
            "devices": [_rank_devices(visible, r, cp) for r in range(dp)],
            "env": env,
            "idle_poll_seconds": float(config.get("idle_poll_seconds", 0.005)),
            "tokenizer_threads": int(config.get("tokenizer_threads", 2)),
            "abort_on_disconnect": bool(config.get("abort_on_disconnect", True)),
            "prefix_cache": bool(config.get("prefix_cache", True)),
            "prefix_cache_free_fraction": float(config.get("prefix_cache_free_fraction", 0.05)),
            "stats_interval_seconds": float(config.get("stats_interval_seconds", 2.0)),
            "llm_kwargs": {
                "max_model_len": max_model_len,
                "max_num_seqs": int(config.get("max_num_seqs", 64)),
                "max_num_batched_tokens": int(config.get("max_num_batched_tokens", 32768)),
                "gpu_memory_utilization": float(config.get("gpu_memory_utilization", 0.85)),
                # As in offline_inference.py. UNVERIFIED: its meaning with DSpark on.
                "lookahead_tokens": min(1, int(env["DEEPINFER_USE_NEXTN_DECODE"])),
                **(config.get("llm_kwargs") or {}),
            },
        }

        self._results = ctx.Queue()
        self._requests = [ctx.Queue() for _ in range(dp)]
        self._procs = [
            ctx.Process(target=_rank_main, args=(r, settings, self._requests[r], self._results), daemon=False)
            for r in range(dp)
        ]
        try:
            for proc in self._procs:
                proc.start()
            self._wait_ready(dp, float(config.get("load_timeout_seconds", 3600)))
        except BaseException:
            # The ranks that did start would keep their GPUs, and as non-daemon
            # processes they would keep this process from exiting.
            self._stop_ranks(grace=0.0)
            raise

        self._streams: Dict[str, _Stream] = {}
        self._owner: Dict[str, int] = {}
        self._in_flight = [0] * dp
        self._affinity = bool(config.get("prefix_cache", True))
        self._sessions: "OrderedDict[str, int]" = OrderedDict()  # session key -> rank, LRU
        self._max_sessions = int(config.get("session_table_size", 100_000))
        self._migrate_at = int(config.get("session_migrate_inflight", 4 * int(settings["llm_kwargs"]["max_num_seqs"])))
        self._used_blocks = [0] * dp  # from the ranks' stats messages
        self._assigned = [0] * dp  # estimated prompt blocks routed since the last stats message
        self.rank_stats: List[Dict[str, Any]] = [{} for _ in range(dp)]
        self.migrations = 0
        self._lock = threading.Lock()
        self._ids = itertools.count()
        self._closed = False
        self._failed: Optional[str] = None
        self._last_message = time.monotonic()
        self._stream_interval = float(config.get("stream_interval_ms", 50)) / 1000.0
        self._chunk_budget = float(config.get("stream_chunks_per_second", 8000))
        self._heartbeat = float(config.get("stream_heartbeat_seconds", 1.0))
        self._request_stall = float(config.get("request_stall_seconds", 1800))
        self._stall_timeout = float(config.get("stall_timeout_seconds", 600))
        self._close_timeout = float(config.get("close_timeout_seconds", 30))
        self._reader = threading.Thread(target=self._read_results, name="deepinfer-results", daemon=True)
        self._reader.start()

    def _wait_ready(self, dp: int, load_timeout: float) -> None:
        ready = set()
        deadline = time.time() + load_timeout
        while len(ready) < dp:
            if any(not p.is_alive() for p in self._procs):
                raise RuntimeError("a DeepInfer rank exited while loading the model; see its log above")
            try:
                kind, rank, _ = self._results.get(timeout=5)
            except queue.Empty:
                if time.time() > deadline:
                    raise TimeoutError(f"DeepInfer ranks not ready after {load_timeout:.0f} s")
                continue
            if kind == "ready":
                ready.add(rank)

    def generate(self, request: GenRequest) -> Iterator[GenEvent]:
        """Stream one request's completion.

        Besides the completion events it yields an empty event (no text, not
        finished) after ``stream_heartbeat_seconds`` without news, so the HTTP
        layer can write a keepalive and notice a client that left while the
        request is queued, prefilling or inside a tool call.
        """
        request_id = str(next(self._ids))
        stream = _Stream()
        params: Dict[str, Any] = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            params["seq_seed"] = request.seed
        if request.ignore_eos:
            params["ignore_eos"] = True
        if request.session_key:
            params["session_key"] = request.session_key
        with self._lock:
            failed = self._failed
            if not failed:
                rank = self._route(request)
                self._in_flight[rank] += 1
                self._streams[request_id] = stream
                self._owner[request_id] = rank
        if failed:
            yield GenEvent(finished=True, error=failed)
            return
        self._requests[rank].put(("add", request_id, request.prompt, params))
        finished = False
        interval, budget = self._stream_interval, self._chunk_budget
        heartbeat, request_stall = self._heartbeat or None, self._request_stall
        last_item = time.monotonic()
        try:
            while not finished:
                if not stream.wake.wait(heartbeat):
                    if request_stall and time.monotonic() - last_item > request_stall:
                        # A backstop: a request the engine lost must not hold its
                        # client and its slot forever.
                        self._cancel(request_id)
                        finished = True
                        yield GenEvent(finished=True, error=f"no output from DeepInfer for {request_stall:.0f} s")
                        return
                    yield GenEvent()
                    continue
                stream.wake.clear()
                items = stream.drain()
                if not items:
                    continue
                last_item = time.monotonic()
                event = _merge(items)
                event.cached_tokens = stream.cached
                finished = event.finished
                yield event
                if not finished:
                    # Let a few steps pile up: one wake-up and one SSE write per
                    # interval instead of per step keeps this process cheap, and
                    # the interval grows with the number of streams so the total
                    # stays within the chunk budget. The last item cuts the wait
                    # short, so a request never ends late.
                    wait = max(interval, len(self._streams) / budget) if budget else interval
                    if wait:
                        stream.done.wait(wait)
        finally:
            if not finished:
                # The consumer stopped early (client disconnected): free the slot.
                self._cancel(request_id)

    def _route(self, request: GenRequest) -> int:
        """The rank for a request (called with the lock held)."""
        ranks = range(len(self._in_flight))
        if not self._affinity:
            return min(ranks, key=lambda r: self._in_flight[r])
        key = request.session_key
        if key is not None:
            rank = self._sessions.get(key)
            if rank is not None:
                if self._in_flight[rank] < self._migrate_at:
                    self._sessions.move_to_end(key)
                    return rank
                self.migrations += 1
        # Prompt characters / 3: a rough token count, refined by the next stats message.
        rank = min(ranks, key=lambda r: (self._used_blocks[r] + self._assigned[r], self._in_flight[r]))
        self._assigned[rank] += len(request.prompt) // 3 // BLOCK + 1
        if key is not None:
            self._sessions[key] = rank
            self._sessions.move_to_end(key)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
        return rank

    def _read_results(self) -> None:
        streams = self._streams
        next_check = time.monotonic() + 1.0
        while True:
            try:
                try:
                    message = self._results.get(timeout=1.0)
                except queue.Empty:
                    message = ()
                if message is None:
                    return
                now = time.monotonic()
                if message or not streams:
                    self._last_message = now
                if message and message[0] == "meta":
                    for request_id, cached in message[2]:
                        stream = streams.get(request_id)
                        if stream is not None:
                            stream.cached = cached
                elif message and message[0] == "stats":
                    rank, info = message[1], message[2]
                    with self._lock:
                        self.rank_stats[rank] = info
                        self._used_blocks[rank] = int(info.get("total_blocks", 0)) - int(info.get("free_blocks", 0))
                        self._assigned[rank] = 0
                elif message and message[0] == "out":
                    for item in message[2]:
                        stream = streams.get(item[0])
                        if stream is None:
                            continue  # cancelled
                        stream.items.append(item)
                        if item[4]:
                            self._release(item[0])
                            stream.done.set()
                        if not stream.wake.is_set():  # a set() per item would cost a lock each
                            stream.wake.set()
                # On a timer, not only when the queue is quiet: with independent
                # ranks a dead one would go unnoticed while the others produce.
                if now >= next_check:
                    next_check = now + 1.0
                    self._check_ranks(now)
            except Exception as exc:  # noqa: BLE001 (fail loudly instead of hanging every stream)
                traceback.print_exc(file=sys.stderr)
                self._fail(f"the DeepInfer results reader failed: {exc!r}")
                return

    def _check_ranks(self, now: float) -> None:
        """Fail the engine when a rank died or every rank went quiet with requests in flight.

        The data-parallel ranks run in lockstep (``has_unfinished_requests()``
        all-reduces across them), so once one rank is gone the others block in
        that collective for good: every request fails, not only the dead rank's.
        """
        if self._closed or self._failed:
            return
        for rank, proc in enumerate(self._procs):
            if not proc.is_alive():
                self._fail(f"DeepInfer rank {rank} exited (code {proc.exitcode}); see its log. "
                           "The ranks run in lockstep, so the engine stopped: restart the backend")
                return
        quiet = now - self._last_message
        if self._stall_timeout and self._streams and quiet > self._stall_timeout:
            self._fail(f"no output from any DeepInfer rank for {quiet:.0f} s with "
                       f"{len(self._streams)} requests in flight; the engine looks hung")

    def _fail(self, message: str) -> None:
        """Stop serving: fail every request now and later ones at once, and stop the ranks."""
        with self._lock:
            if self._failed:
                return
            self._failed = message
            streams = list(self._streams.items())
        print(f"deepinfer: {message}", file=sys.stderr, flush=True)
        for request_id, stream in streams:
            self._release(request_id)
            stream.items.append((request_id, "", 0, 0, True, message))
            stream.done.set()
            stream.wake.set()
        # The ranks left are blocked in a collective or useless without a reader.
        self._stop_ranks(grace=0.0)

    def _release(self, request_id: str) -> Optional[int]:
        """Forget a request; returns the rank it was on, or None if it was already gone."""
        with self._lock:
            self._streams.pop(request_id, None)
            rank = self._owner.pop(request_id, None)
            if rank is not None:
                self._in_flight[rank] -= 1
        return rank

    def _cancel(self, request_id: str) -> None:
        rank = self._release(request_id)
        if rank is not None and not self._closed and not self._failed:
            self._requests[rank].put(("abort", request_id))

    def _stop_ranks(self, grace: float) -> None:
        """Wait up to ``grace`` seconds for the ranks to exit, then terminate, then kill.

        A clean exit is not guaranteed: each rank stops at whatever iteration it
        reads the stop message, and one still inside the lockstep collective
        waits for a rank that already left.
        """
        procs = [p for p in self._procs if p.pid is not None]
        deadline = time.monotonic() + grace
        for proc in procs:
            proc.join(timeout=max(deadline - time.monotonic(), 0.0))
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)

    def close(self) -> None:
        self._closed = True
        for q in self._requests:
            q.put(None)
        self._stop_ranks(grace=self._close_timeout)
        self._results.put(None)


def resolve_model_path(config: Dict[str, Any]) -> str:
    """Local checkpoint directory: ``model_dir`` in the config, else
    ``DEEPINFER_HF_MODEL_DIR`` (the variable DeepInfer's examples read), else an
    error. Downloading 510 GB from the backend's ``setup()`` is not something to
    do by accident, so there is no implicit Hub download here.
    """
    path = config.get("model_dir") or os.environ.get("DEEPINFER_HF_MODEL_DIR")
    if not path:
        raise ValueError(
            "set model_dir in the workload config or DEEPINFER_HF_MODEL_DIR in config.env "
            "to a local copy of deepseek-ai/DeepSeek-V4.1-Flash-1017"
        )
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise FileNotFoundError(f"{path} has no config.json: not a Hugging Face checkpoint directory")
    return os.path.abspath(path)


def build_engine(config: Dict[str, Any]) -> Engine:
    """The engine named by ``config["engine"]``: ``"deepinfer"`` (default) or ``"fake"``."""
    kind = config.get("engine", "deepinfer")
    if kind == "fake":
        return FakeEngine(config)
    if kind == "deepinfer":
        return DeepInferEngine(config)
    raise ValueError(f"unknown engine {kind!r}; expected 'deepinfer' or 'fake'")

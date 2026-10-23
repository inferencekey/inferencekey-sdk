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
  rank to tokenize prompts and to detokenize completions.

What has NOT been checked on hardware is marked ``UNVERIFIED`` below and listed
in the README.
"""

from __future__ import annotations

import codecs
import glob
import itertools
import json
import os
import queue
import site
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional


@dataclass
class GenRequest:
    """One generation: an encoded prompt and its sampling settings."""

    prompt: str
    max_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 1.0
    seed: Optional[int] = None


@dataclass
class GenEvent:
    """A piece of completion text; the last event of a request has ``finished``."""

    text: str = ""
    finished: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None


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
        )
    finally:
        llm.close()


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
) -> None:
    """The rank's serving loop, apart from building the engine (so tests can drive it).

    Messages in (``requests``): ``("add", request_id, prompt, params)``,
    ``("abort", request_id)`` and ``None`` (stop). Messages out (``results``):
    ``("ready", rank, None)`` once, then at most one ``("out", rank, items)``
    per engine step, ``items`` being one tuple per request that advanced:
    ``(request_id, text, completion_tokens, prompt_tokens, finished, error)``,
    with the text these tokens made final and the totals so far.

    The parent never sees a token id: prompts are tokenized here, on a few
    threads (the Hugging Face fast tokenizer releases the GIL while it encodes,
    so a 137K-token prompt does not stall the lockstep loop), and completions
    are detokenized here, incrementally, per request.
    """
    new_detokenizer = detokenizer_factory(tokenizer)
    idle_sleep = float(settings["idle_poll_seconds"])
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
            tokenized.append((request_id, token_array(tokenizer.encode(prompt)), params, None))
        except Exception as exc:  # noqa: BLE001 (report it on the request)
            tokenized.append((request_id, None, params, f"tokenization failed: {exc}"))

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
                        try:
                            # UNVERIFIED: aborting between steps while other requests run.
                            engine.abort_request(request_id)
                        except Exception as exc:  # noqa: BLE001 (keep the rank alive)
                            print(f"deepinfer rank {rank}: abort of {request_id} failed: {exc}",
                                  file=sys.stderr, flush=True)

            # Prompts tokenized since the last iteration go to the engine.
            while tokenized:
                request_id, token_ids, params, error = tokenized.popleft()
                if tokenizing.pop(request_id, False):
                    continue  # the client left while its prompt was being tokenized
                if error is None:
                    try:
                        # UNVERIFIED: adding requests between steps while others are in flight
                        # (the offline example adds them all before the first step).
                        engine.add_request(
                            user_id="0",
                            request_id=request_id,
                            sampling_params=sampling_params(**params),
                            prompt_token_ids=token_ids,
                            multi_modal_data=multi_modal_data(),
                        )
                    except Exception as exc:  # noqa: BLE001 (report and keep the rank alive)
                        error = str(exc)
                if error is None:
                    live[request_id] = _Live(new_detokenizer())
                else:
                    items.append((request_id, "", 0, 0, True, error))

            # Collective across the data-parallel group: every rank calls it.
            # UNVERIFIED: that idle ranks may keep calling it (and step() with an
            # empty local batch) indefinitely, as a long-lived server needs.
            unfinished = engine.has_unfinished_requests()
            if unfinished:
                outputs = engine.step()
                pending_flush = True
            elif pending_flush:
                # Same value on every rank (they all stepped), so the flush stays in lockstep.
                outputs = engine.flush_inflight_reqs()
                pending_flush = False
            else:
                if items:
                    results.put(("out", rank, items))
                time.sleep(idle_sleep)
                continue

            if outputs:
                _collect_step(outputs, clone, live, items)
            if items:
                results.put(("out", rank, items))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _collect_step(outputs: Any, clone: Callable[[Any], Any], live: Dict[str, _Live], items: List[tuple]) -> None:
    """Group one step's outputs by request, detokenize them and append one item per request."""
    step: Dict[str, list] = {}
    for raw in outputs:
        # clone: the step's outputs are released by the next step.
        out = clone(raw)
        entry = step.get(out.request_id)
        if entry is None:
            entry = step[out.request_id] = [[], False, 0]
        entry[0].append(int(out.token_id))
        entry[1] = entry[1] or bool(out.finished)
        entry[2] = int(out.prompt_len)
    for request_id, (token_ids, finished, prompt_len) in step.items():
        state = live.get(request_id)
        if state is None:
            continue  # aborted
        state.tokens += len(token_ids)
        text = state.detokenizer.feed(token_ids)
        if finished:
            text += state.detokenizer.flush()
            del live[request_id]
        elif not text:
            continue  # nothing final yet (half a character); the totals ride on the next item
        items.append((request_id, text, state.tokens, prompt_len, finished, None))


class _Stream:
    """Parent-side mailbox of one request: the reader appends, the request's thread drains."""

    __slots__ = ("items", "wake", "done")

    def __init__(self) -> None:
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
    error = next((item[5] for item in items if item[5]), None)
    return GenEvent(
        text=last[1] if len(items) == 1 else "".join(item[1] for item in items),
        finished=bool(last[4]) or error is not None,
        prompt_tokens=last[3],
        completion_tokens=last[2],
        error=error,
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
    """DeepInfer with one process per data-parallel rank and a least-loaded router.

    The open release of DeepInfer ships no load balancer across data-parallel
    replicas; this router (fewest requests in flight wins) is ours and simple on
    purpose.

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
        for proc in self._procs:
            proc.start()

        ready = set()
        load_timeout = float(config.get("load_timeout_seconds", 3600))
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

        self._streams: Dict[str, _Stream] = {}
        self._owner: Dict[str, int] = {}
        self._in_flight = [0] * dp
        self._lock = threading.Lock()
        self._ids = itertools.count()
        self._closed = False
        self._stream_interval = float(config.get("stream_interval_ms", 50)) / 1000.0
        self._reader = threading.Thread(target=self._read_results, name="deepinfer-results", daemon=True)
        self._reader.start()

    def generate(self, request: GenRequest) -> Iterator[GenEvent]:
        request_id = str(next(self._ids))
        stream = _Stream()
        params: Dict[str, Any] = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            params["seq_seed"] = request.seed
        with self._lock:
            rank = min(range(len(self._in_flight)), key=lambda r: self._in_flight[r])
            self._in_flight[rank] += 1
            self._streams[request_id] = stream
            self._owner[request_id] = rank
        self._requests[rank].put(("add", request_id, request.prompt, params))
        finished = False
        interval = self._stream_interval
        try:
            while not finished:
                stream.wake.wait()
                stream.wake.clear()
                items = stream.drain()
                if not items:
                    continue
                event = _merge(items)
                finished = event.finished
                yield event
                if interval and not finished:
                    # Let a few steps pile up: one wake-up and one SSE write per
                    # interval instead of per step keeps this process cheap. The
                    # last item cuts the wait short, so a request never ends late.
                    stream.done.wait(interval)
        finally:
            if not finished:
                # The consumer stopped early (client disconnected): free the slot.
                self._cancel(request_id)

    def _read_results(self) -> None:
        streams = self._streams
        while True:
            try:
                message = self._results.get(timeout=1.0)
            except queue.Empty:
                self._check_ranks()
                continue
            if message is None:
                return
            kind, _rank, items = message
            if kind != "out":
                continue
            for item in items:
                stream = streams.get(item[0])
                if stream is None:
                    continue  # cancelled
                stream.items.append(item)
                if item[4]:
                    self._release(item[0])
                    stream.done.set()
                if not stream.wake.is_set():  # a set() per item would cost a lock each
                    stream.wake.set()

    def _check_ranks(self) -> None:
        """Fail the requests of a rank that died, so their clients get an error instead of hanging."""
        if self._closed:
            return
        for rank, proc in enumerate(self._procs):
            if proc.is_alive():
                continue
            with self._lock:
                dead = [rid for rid, owner in self._owner.items() if owner == rank]
            for request_id in dead:
                stream = self._streams.get(request_id)
                if stream is None:
                    continue
                self._release(request_id)
                message = f"DeepInfer rank {rank} exited (code {proc.exitcode}); see its log"
                stream.items.append((request_id, "", 0, 0, True, message))
                stream.done.set()
                stream.wake.set()

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
        if rank is not None and not self._closed:
            self._requests[rank].put(("abort", request_id))

    def close(self) -> None:
        self._closed = True
        for q in self._requests:
            q.put(None)
        for proc in self._procs:
            proc.join(timeout=30)
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

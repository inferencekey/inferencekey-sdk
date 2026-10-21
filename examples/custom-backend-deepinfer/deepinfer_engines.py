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

What has NOT been checked on hardware is marked ``UNVERIFIED`` below and listed
in the README.
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


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
    tokenizer = llm.get_tokenizer()
    needs_mm = engine.multi_modal_processor is not None
    results.put(("ready", rank, None))

    idle_sleep = settings["idle_poll_seconds"]
    pending_flush = False
    while True:
        # Take every request routed to this rank since the last iteration.
        while True:
            try:
                item = requests.get_nowait()
            except queue.Empty:
                break
            if item is None:
                llm.close()
                return
            request_id, prompt, params = item
            try:
                token_ids = np.asarray(tokenizer.encode(prompt), dtype=np.int32)
                # UNVERIFIED: adding requests between steps while others are in flight
                # (the offline example adds them all before the first step).
                engine.add_request(
                    user_id="0",
                    request_id=request_id,
                    sampling_params=SamplingParams(**params),
                    prompt_token_ids=token_ids,
                    multi_modal_data=dict(image_bytes=[], image_sizes=[]) if needs_mm else None,
                )
            except Exception as exc:  # noqa: BLE001 (report and keep the rank alive)
                results.put(("error", request_id, str(exc)))

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
            time.sleep(idle_sleep)
            continue

        tokens = []
        for raw in outputs:
            out = RequestOutput.clone(raw)
            tokens.append((out.request_id, int(out.token_id), bool(out.finished), int(out.prompt_len)))
        if tokens:
            results.put(("tokens", rank, tokens))


@dataclass
class _Stream:
    events: "queue.Queue[GenEvent]" = field(default_factory=queue.Queue)
    token_ids: List[int] = field(default_factory=list)
    text: str = ""
    prompt_tokens: int = 0


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
    for site in sites:
        if site not in sys.path:
            sys.path.append(site)
    ctx.set_executable(python)


class DeepInferEngine(Engine):
    """DeepInfer with one process per data-parallel rank and a least-loaded router.

    The open release of DeepInfer ships no load balancer across data-parallel
    replicas; this router (fewest requests in flight wins) is ours and simple on
    purpose. Text is decoded in this process with the checkpoint's tokenizer.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        import multiprocessing as mp  # noqa: PLC0415

        ctx = mp.get_context("spawn")
        use_deepinfer_venv(config.get("deepinfer_venv"), ctx)
        from transformers import AutoTokenizer  # type: ignore[import-not-found]  # noqa: PLC0415

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

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._results = ctx.Queue()
        self._requests = [ctx.Queue() for _ in range(dp)]
        self._procs = [
            ctx.Process(target=_rank_main, args=(r, settings, self._requests[r], self._results), daemon=True)
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
        self._requests[rank].put((request_id, request.prompt, params))
        while True:
            event = stream.events.get()
            yield event
            if event.finished:
                return

    def _read_results(self) -> None:
        while True:
            kind, who, payload = self._results.get()
            if kind == "error":
                self._finish(who, GenEvent(finished=True, error=payload))
                continue
            if kind != "tokens":
                continue
            for request_id, token_id, finished, prompt_len in payload:
                stream = self._streams.get(request_id)
                if stream is None:
                    continue
                stream.token_ids.append(token_id)
                stream.prompt_tokens = prompt_len
                # Decode the whole completion and emit what is new: simple and
                # correct for multi-byte characters split across tokens.
                text = self.tokenizer.decode(stream.token_ids, skip_special_tokens=False)
                delta, stream.text = text[len(stream.text) :], text
                event = GenEvent(
                    text=delta,
                    finished=finished,
                    prompt_tokens=prompt_len,
                    completion_tokens=len(stream.token_ids),
                )
                if finished:
                    self._finish(request_id, event)
                else:
                    stream.events.put(event)

    def _finish(self, request_id: str, event: GenEvent) -> None:
        with self._lock:
            stream = self._streams.pop(request_id, None)
            rank = self._owner.pop(request_id, None)
            if rank is not None:
                self._in_flight[rank] -= 1
        if stream is not None:
            stream.events.put(event)

    def close(self) -> None:
        for q in self._requests:
            q.put(None)
        for proc in self._procs:
            proc.join(timeout=30)


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

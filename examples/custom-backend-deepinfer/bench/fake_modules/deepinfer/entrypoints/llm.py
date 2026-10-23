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

Every ``stats_interval_s`` the rank appends one JSON line with its counters to
``<stats_path>.rank<N>`` so the load driver can compare what the ranks produce
with what the clients receive.
"""

from __future__ import annotations

import json
import time
import zlib
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from deepinfer._C.engine import RequestOutput

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


class _Seq:
    __slots__ = ("request_id", "prompt_len", "prefill_left", "max_tokens", "generated", "offset")

    def __init__(self, request_id: str, prompt_len: int, max_tokens: int, offset: int) -> None:
        self.request_id = request_id
        self.prompt_len = prompt_len
        self.prefill_left = prompt_len
        self.max_tokens = max_tokens
        self.generated = 0
        self.offset = offset


class SimEngine:
    """The ``llm.llm_engine`` surface the rank loop uses."""

    multi_modal_processor = None

    def __init__(self, rank: int, max_num_seqs: int, max_num_batched_tokens: int, sim: Dict[str, Any],
                 token_ids: List[int]) -> None:
        self.rank = rank
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.step_s = 1.0 / float(sim.get("step_rate", 70.0))
        self.tokens_per_step = int(sim.get("tokens_per_step", 1))
        self.prefill_rate = float(sim.get("prefill_tokens_per_s", 0.0))
        self.barrier = sim.get("barrier")
        self.flags = sim.get("flags")
        self.stats_path: Optional[str] = sim.get("stats_path")
        self.stats_interval = float(sim.get("stats_interval_s", 1.0))
        self.token_ids = token_ids
        self.waiting: Deque[_Seq] = deque()
        self.prefilling: List[_Seq] = []
        self.running: List[_Seq] = []
        self.last_end = 0.0
        self.steps = 0
        self.tokens = 0
        self.prefilled = 0
        self.next_stats = time.time()

    # --- the calls _rank_main makes -------------------------------------------

    def add_request(self, user_id: str, request_id: str, sampling_params: Any, prompt_token_ids: Any,
                    multi_modal_data: Optional[dict] = None, **_: Any) -> None:
        offset = zlib.crc32(request_id.encode()) % len(self.token_ids)
        self.waiting.append(_Seq(request_id, len(prompt_token_ids), sampling_params.max_tokens, offset))

    def has_unfinished_requests(self) -> bool:
        local = bool(self.waiting or self.prefilling or self.running)
        self._maybe_stats()
        if self.barrier is None:
            return local
        self.flags[self.rank] = 1 if local else 0
        self.barrier.wait()
        unfinished = any(self.flags[:])
        # Second barrier: nobody overwrites its flag before everyone has read them.
        self.barrier.wait()
        return unfinished

    def step(self) -> List[RequestOutput]:
        while self.waiting and len(self.prefilling) + len(self.running) < self.max_num_seqs:
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
                # The step that finishes a prompt samples its first token.
                if not self._emit(seq, outputs, first=True):
                    still_running.append(seq)
            else:
                still_prefilling.append(seq)
        self.running = still_running
        self.prefilling = still_prefilling

        work = self.step_s + (prefill_tokens / self.prefill_rate if self.prefill_rate > 0 else 0.0)
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
        return outputs

    def flush_inflight_reqs(self) -> List[RequestOutput]:
        return []

    def abort_request(self, request_id: Any) -> tuple:
        ids = {request_id} if isinstance(request_id, str) else set(request_id)
        before = len(self.waiting) + len(self.prefilling) + len(self.running)
        self.waiting = deque(s for s in self.waiting if s.request_id not in ids)
        self.prefilling = [s for s in self.prefilling if s.request_id not in ids]
        self.running = [s for s in self.running if s.request_id not in ids]
        return before - len(self.waiting) - len(self.prefilling) - len(self.running), 0

    # --- helpers --------------------------------------------------------------

    def _emit(self, seq: _Seq, outputs: List[RequestOutput], first: bool = False) -> bool:
        """Append this step's tokens of ``seq``; True when it finished."""
        n = 1 if first else self.tokens_per_step
        ids = self.token_ids
        for _ in range(n):
            token = ids[(seq.offset + seq.generated) % len(ids)]
            seq.generated += 1
            finished = seq.generated >= seq.max_tokens
            outputs.append(RequestOutput(seq.request_id, token, finished, seq.prompt_len))
            if finished:
                return True
        return False

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
        }
        with open(f"{self.stats_path}.rank{self.rank}", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")


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
        self.llm_engine = SimEngine(engine_rank, int(max_num_seqs), int(max_num_batched_tokens), sim, token_ids)

    def get_tokenizer(self) -> Any:
        return self._tokenizer

    def close(self) -> None:
        pass

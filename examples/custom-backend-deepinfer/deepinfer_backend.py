"""DeepSeek-V4.1-Flash on DeepSeek's DeepInfer engine, as an InferenceKey custom backend.

The backend implements ``chat()`` and ``chat_stream()``, so the SDK runtime
serves OpenAI ``/v1/chat/completions`` and the platform routes chat to it like
it routes chat to vLLM: streaming, tool calls and usage included, and the
workload's Anthropic ``/v1/messages`` endpoint works for Claude Code.

Per request:

1. OpenAI messages and tools become a DeepSeek-V4.1 prompt with the model's own
   reference encoder (``dsv41_encoding.encode_messages``, vendored from the model
   repository): tools go on the system message, tool results become
   ``<tool_result>`` blocks, thinking mode is ``<think>``.
2. The engine (:mod:`deepinfer_engines`) streams the completion text.
3. The completion is split back: reasoning (before ``</think>``) becomes
   ``reasoning_content``, text becomes ``content``, and a DSML ``calls`` block
   becomes OpenAI ``tool_calls`` (``dsv41_encoding.parse_message_from_completion_text``).
   While streaming, reasoning and text go out as they arrive; a tool call is
   sent once complete, because its arguments are only valid JSON at the end.
   When nothing goes out for ``stream_heartbeat_seconds`` (a queued or
   prefilling request, a tool call being generated) the stream yields ``None``,
   which the runtime writes as an SSE comment: a client that left is noticed
   and its request aborted.

A prompt longer than the engine accepts is answered with a 400 whose error
code is ``context_length_exceeded``, so a client such as Claude Code can
compact the conversation and retry.

Config (the workload ``config``, received as ``ctx.config``): see the README.
"""

from __future__ import annotations

import copy
import itertools
import json
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional

import dsv41_encoding as dsv41
from deepinfer_engines import (
    CONTEXT_LENGTH_EXCEEDED,
    INVALID_PROMPT,
    Engine,
    GenEvent,
    GenRequest,
    build_engine,
)

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

TOOL_CALLS_START = f"\n\n<{dsv41.dsml_token}{dsv41.tool_calls_block_name}"
THINK_END = dsv41.thinking_end_token
EOS = dsv41.eos_token


class ContextLengthExceeded(InvalidInput):
    """The prompt does not fit the engine's context: a 400 with an OpenAI error code."""

    code = CONTEXT_LENGTH_EXCEEDED


def _raise_for(event: GenEvent) -> None:
    """Raise the exception that matches a failed engine event (no-op without an error)."""
    if not event.error:
        return
    if event.error_code == CONTEXT_LENGTH_EXCEEDED:
        raise ContextLengthExceeded(f"the prompt is too long for this model's context ({event.error})")
    if event.error_code == INVALID_PROMPT:
        raise InvalidInput(event.error)
    raise RuntimeError(f"DeepInfer failed: {event.error}")


def _text_of(content: Any) -> str:
    """Flatten OpenAI content (a string or a list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def to_dsv41_messages(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """OpenAI request (messages + tools) to the message list the encoder expects.

    The encoder reads tools from the system message's ``tools`` field, so the
    request's tools are attached to the first system message (one is created if
    the conversation has none). Text-only: image parts are dropped.
    """
    messages: List[Dict[str, Any]] = []
    for raw in request.get("messages") or []:
        if not isinstance(raw, dict) or "role" not in raw:
            raise InvalidInput("every message needs a role")
        role = raw["role"]
        if role == "developer":
            role = "system"
        msg: Dict[str, Any] = {"role": role, "content": _text_of(raw.get("content"))}
        if role == "assistant":
            if raw.get("tool_calls"):
                msg["tool_calls"] = copy.deepcopy(raw["tool_calls"])
            if raw.get("reasoning_content"):
                msg["reasoning_content"] = raw["reasoning_content"]
        if role == "tool" and raw.get("tool_call_id"):
            msg["tool_call_id"] = raw["tool_call_id"]
        messages.append(msg)
    if not messages:
        raise InvalidInput("request has no messages")
    tools = request.get("tools") or []
    if tools:
        system = next((m for m in messages if m["role"] == "system"), None)
        if system is None:
            system = {"role": "system", "content": ""}
            messages.insert(0, system)
        system["tools"] = tools
    return messages


class StreamSplitter:
    """Splits a streamed DeepSeek-V4.1 completion into reasoning and content deltas.

    Text that could still turn out to be the start of ``</think>`` or of a DSML
    tool-call block is held back until it is clear, so no markup leaks to the
    client. Once a tool-call block starts, nothing more is emitted: the caller
    parses the full completion at the end.
    """

    def __init__(self, thinking: bool) -> None:
        self.in_reasoning = thinking
        self.in_tool_calls = False
        self._pending = ""

    def feed(self, text: str) -> List[Dict[str, str]]:
        """Return the deltas (``{"reasoning_content": ...}`` or ``{"content": ...}``) now safe to send."""
        if self.in_tool_calls:
            return []
        self._pending += text
        out: List[Dict[str, str]] = []
        while self._pending and not self.in_tool_calls:
            marker = THINK_END if self.in_reasoning else TOOL_CALLS_START
            stops = [marker, TOOL_CALLS_START, EOS] if self.in_reasoning else [TOOL_CALLS_START, EOS]
            hit = min(((self._pending.find(s), s) for s in stops if s in self._pending), default=None)
            key = "reasoning_content" if self.in_reasoning else "content"
            if hit is not None:
                index, stop = hit
                if index:
                    out.append({key: self._pending[:index]})
                self._pending = self._pending[index + len(stop) :]
                if stop == THINK_END:
                    self.in_reasoning = False
                    continue
                if stop == TOOL_CALLS_START:
                    self.in_tool_calls = True
                self._pending = ""
                break
            safe = len(self._pending) - self._longest_prefix_tail(stops)
            if safe > 0:
                out.append({key: self._pending[:safe]})
                self._pending = self._pending[safe:]
            break
        return out

    def _longest_prefix_tail(self, stops: List[str]) -> int:
        """Length of the longest end of the pending text that starts any stop marker."""
        tail = self._pending[1 - max(len(stop) for stop in stops) :]
        if not any(stop[0] in tail for stop in stops):
            return 0  # the common case: no "<" or newline near the end
        best = 0
        for stop in stops:
            for n in range(min(len(stop) - 1, len(self._pending)), 0, -1):
                if self._pending.endswith(stop[:n]):
                    best = max(best, n)
                    break
        return best


class DeepInferBackend(CustomBackend):
    name = "deepseek-v4.1-flash-deepinfer"
    version = "0.1.0"
    task_type = "text2text"
    requires_python = ">=3.12,<3.13"

    def setup(self, ctx: BackendContext) -> None:
        self.config: Dict[str, Any] = dict(ctx.config)
        self.model_name = self.config.get("served_model_name", "deepseek-v4.1-flash")
        self.thinking_mode = self.config.get("thinking_mode", "thinking")
        self.reasoning_effort = self.config.get("reasoning_effort", "high")
        self.default_max_tokens = int(self.config.get("default_max_tokens", 8192))
        self.heartbeat_seconds = float(self.config.get("stream_heartbeat_seconds", 1.0))
        self.engine: Engine = build_engine(self.config)

    def process(self, job: Job) -> Result:
        completion = self.chat({"messages": job.messages() or [{"role": "user", "content": job.text()}]})
        message = completion["choices"][0]["message"]
        return Result(output={"text": message.get("content") or "", "tool_calls": message.get("tool_calls") or []})

    # --- OpenAI chat ---------------------------------------------------------

    def _gen_request(self, request: Dict[str, Any]) -> GenRequest:
        mode = request.get("thinking_mode") or self.thinking_mode
        prompt = dsv41.encode_messages(
            to_dsv41_messages(request),
            thinking_mode=mode,
            reasoning_effort=request.get("reasoning_effort") or self.reasoning_effort,
        )
        max_tokens = request.get("max_completion_tokens") or request.get("max_tokens") or self.default_max_tokens
        return GenRequest(
            prompt=prompt,
            max_tokens=int(max_tokens),
            temperature=float(request.get("temperature", self.config.get("temperature", 0.6))),
            top_p=float(request.get("top_p", self.config.get("top_p", 1.0))),
            seed=request.get("seed"),
        )

    def chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        events = list(self.engine.generate(self._gen_request(request)))
        for event in events:
            _raise_for(event)
        text = "".join(e.text for e in events)
        last = events[-1] if events else GenEvent(finished=True)
        message, finish_reason = self._parse(text, request)
        return self._completion(message, finish_reason, last)

    def chat_stream(self, request: Dict[str, Any]) -> Iterator[Optional[Dict[str, Any]]]:
        """Stream OpenAI chunks; ``None`` items are keepalives (no delta for a while).

        The first chunk waits for the engine's first event or first heartbeat,
        so a prompt the engine refuses at once (too long, invalid) still gets a
        400 before the 200 headers go out.
        """
        gen = self._gen_request(request)
        thinking = gen.prompt.endswith(dsv41.thinking_start_token)
        splitter = StreamSplitter(thinking=thinking)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        base = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": self.model_name}

        def chunk(delta: Dict[str, Any], finish: Optional[str] = None) -> Dict[str, Any]:
            return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        events = self.engine.generate(gen)
        try:
            first = next(events, None)
            if first is not None:
                _raise_for(first)
            yield chunk({"role": "assistant", "content": ""})
            parts: List[str] = []
            last = GenEvent()
            heartbeat = self.heartbeat_seconds
            sent_at = time.monotonic()
            for event in itertools.chain([first] if first is not None else [], events):
                _raise_for(event)
                if event.text:
                    parts.append(event.text)
                if event.finished or event.text:
                    last = event
                deltas = splitter.feed(event.text) if event.text else []
                for delta in deltas:
                    yield chunk(delta)
                if deltas:
                    sent_at = time.monotonic()
                elif heartbeat and not event.finished and time.monotonic() - sent_at >= heartbeat:
                    # Nothing for the client for a while: a keepalive lets the
                    # runtime find out whether the client is still there.
                    yield None
                    sent_at = time.monotonic()
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()  # a client that left frees its engine slot now
        message, finish_reason = self._parse("".join(parts), request)
        for index, call in enumerate(message.get("tool_calls") or []):
            yield chunk({"tool_calls": [{"index": index, **call}]})
        yield chunk({}, finish_reason)
        yield {**base, "choices": [], "usage": self._usage(last)}

    # --- helpers -------------------------------------------------------------

    def _parse(self, text: str, request: Dict[str, Any]) -> "tuple[Dict[str, Any], str]":
        """Completion text to an OpenAI assistant message and finish reason."""
        mode = request.get("thinking_mode") or self.thinking_mode
        finished_by_eos = text.endswith(EOS)
        candidate = text if finished_by_eos else text + EOS
        try:
            parsed = dsv41.parse_message_from_completion_text(candidate, thinking_mode=mode)
        except (AssertionError, ValueError):
            # Truncated or malformed output (the parser raises both): return what can be read as text.
            reasoning, _, content = candidate.partition(THINK_END) if mode == "thinking" else ("", "", candidate)
            parsed = {
                "content": content.split(TOOL_CALLS_START, 1)[0].replace(EOS, ""),
                "reasoning_content": reasoning,
                "tool_calls": [],
            }
        message: Dict[str, Any] = {"role": "assistant", "content": parsed.get("content") or None}
        if parsed.get("reasoning_content"):
            message["reasoning_content"] = parsed["reasoning_content"]
        calls = []
        for call in parsed.get("tool_calls") or []:
            function = call["function"]
            name = function["name"] if not call.get("namespace") else f"{call['namespace']}::{function['name']}"
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": function["arguments"]},
            })
        if calls:
            message["tool_calls"] = calls
            return message, "tool_calls"
        return message, "stop" if finished_by_eos else "length"

    @staticmethod
    def _usage(last: GenEvent) -> Dict[str, int]:
        return {
            "prompt_tokens": last.prompt_tokens,
            "completion_tokens": last.completion_tokens,
            "total_tokens": last.prompt_tokens + last.completion_tokens,
        }

    def _completion(self, message: Dict[str, Any], finish_reason: str, last: GenEvent) -> Dict[str, Any]:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": self._usage(last),
        }

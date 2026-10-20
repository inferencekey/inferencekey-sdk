"""An OpenAI-chat custom backend: the template for serving your own LLM engine.

Implementing ``chat()`` makes the backend speak OpenAI ``/v1/chat/completions``
(``api = "openai-chat"`` is inferred). The platform then routes chat to it the
same way it routes chat to vLLM: streaming, tool calls and usage included, and
the workload's Anthropic ``/v1/messages`` endpoint works for Claude Code.

Two modes, chosen by the workload config (``ctx.config``):

* **upstream** (``upstream_base_url`` set): forward every request to an
  OpenAI-compatible server and stream its answer back token by token. Replace
  the two ``_upstream_*`` calls with your engine (DeepInfer, a custom PyTorch
  loop...) and you have a production backend.
* **scripted** (default): a deterministic stand-in that needs no model, so the
  whole chain can be tested offline. With tools it reads ``demo_file``, then
  edits it (``demo_edit_old`` to ``demo_edit_new``), then reports, which is
  enough for Claude Code to run two real tool calls end to end.

No third-party dependency: only the standard library.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

from inferencekey.backend import (
    BackendContext,
    CustomBackend,
    InvalidInput,
    Job,
    Result,
    chat_completion,
)


def _rough_tokens(text: str) -> int:
    """About four characters per token: good enough for the scripted mode."""
    return max(1, len(text) // 4)


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


class OpenAIChatBackend(CustomBackend):
    name = "openai-chat-example"
    version = "0.1.0"
    task_type = "text2text"

    def setup(self, ctx: BackendContext) -> None:
        self.config = ctx.config
        self.upstream = (self.config.get("upstream_base_url") or "").rstrip("/")
        self.upstream_model = self.config.get("upstream_model", "")
        key_env = self.config.get("upstream_api_key_env", "")
        self.upstream_key = os.environ.get(key_env, "") if key_env else ""
        # Lets the platform's readiness timeout be exercised: a real engine
        # loading hundreds of GB takes minutes before /healthz turns 200.
        time.sleep(float(self.config.get("demo_setup_seconds", 0)))

    def process(self, job: Job) -> Result:
        # Non-chat callers of /process still get an answer.
        return Result(output={"text": self.chat({"messages": job.messages()})["choices"][0]["message"]["content"]})

    # --- chat --------------------------------------------------------------

    def chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if self.upstream:
            return self._upstream_chat(request)
        return self._scripted(request)

    def chat_stream(self, request: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        if self.upstream:
            yield from self._upstream_stream(request)
            return
        seconds = float(self.config.get("demo_stream_seconds", 0))
        if seconds <= 0:
            yield from super().chat_stream(request)
            return
        # Slow stream: the answer trickles out over `seconds`, to exercise
        # request timeouts longer than the five-minute defaults.
        words = [f"word{i} " for i in range(10)]
        base = {"object": "chat.completion.chunk", "model": self.name}
        yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
        for word in words:
            time.sleep(seconds / len(words))
            yield {**base, "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}]}
        yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

    # --- scripted mode -----------------------------------------------------

    def _scripted(self, request: Dict[str, Any]) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = request["messages"]
        tools = {t.get("function", {}).get("name") for t in request.get("tools") or []}
        prompt_tokens = sum(_rough_tokens(_message_text(m)) for m in messages)
        tool_results = [m for m in messages if m.get("role") == "tool"]
        demo_file = self.config.get("demo_file", "")
        old = self.config.get("demo_edit_old", "")
        new = self.config.get("demo_edit_new", "")

        def call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
            return chat_completion(
                model=self.name,
                tool_calls=[{
                    "id": f"call_{len(tool_results) + 1}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }],
                prompt_tokens=prompt_tokens,
                completion_tokens=20,
            )

        read_tool = "Read" if "Read" in tools else ("read_file" if "read_file" in tools else None)
        if demo_file and read_tool and not tool_results:
            key = "file_path" if read_tool == "Read" else "path"
            return call(read_tool, {key: demo_file})
        if demo_file and old and "Edit" in tools and len(tool_results) == 1:
            return call("Edit", {"file_path": demo_file, "old_string": old, "new_string": new})
        if tool_results:
            first_line = _message_text(tool_results[0]).strip().splitlines()[:1]
            text = f"Done. I read {demo_file}"
            if len(tool_results) > 1:
                text += f" and replaced {old!r} with {new!r}"
            text += f". Its first line was: {first_line[0] if first_line else '(empty)'}"
        else:
            last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if last_user is None:
                raise InvalidInput("no user message")
            text = f"echo: {_message_text(last_user)}"
        return chat_completion(text, model=self.name, prompt_tokens=prompt_tokens, completion_tokens=_rough_tokens(text))

    # --- upstream mode (replace these two with your engine) ----------------

    def _upstream_request(self, request: Dict[str, Any], stream: bool) -> urllib.request.Request:
        body = dict(request, stream=stream)
        if self.upstream_model:
            body["model"] = self.upstream_model
        headers = {"Content-Type": "application/json"}
        if self.upstream_key:
            headers["Authorization"] = f"Bearer {self.upstream_key}"
        return urllib.request.Request(
            f"{self.upstream}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )

    def _upstream_chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        timeout: Optional[float] = self.config.get("upstream_timeout_secs", 600)
        with urllib.request.urlopen(self._upstream_request(request, False), timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _upstream_stream(self, request: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        timeout: Optional[float] = self.config.get("upstream_timeout_secs", 600)
        with urllib.request.urlopen(self._upstream_request(request, True), timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    return
                yield json.loads(payload)

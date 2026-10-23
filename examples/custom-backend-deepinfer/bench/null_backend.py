"""A backend that does nothing but stream: it isolates the runtime's HTTP/SSE layer.

``chat_stream`` yields ``max_tokens`` one-token content chunks paced by
``time.sleep`` at ``tok_s`` tokens per second (the first one at once, then on an
absolute schedule so the pace does not drift), then a finish chunk and a usage
chunk with empty ``choices``, like the DeepInfer backend. No multiprocessing,
no tokenizer, no engine: whatever limits a run of this backend is the runtime.

``load.py --backend null`` serves it with the real CLI::

    python -m inferencekey.backend.serve --port N --backend null_backend:NullBackend \\
        --config-json '{"tok_s": 50}'

The first token chunk carries ``bench_t``: ``[t_enter, t_yield]``, the wall
clock when the runtime first resumed the generator and when it yielded that
token. ``load.py`` uses them to split the time to first token into the part
before the backend runs, the backend's own latency and the delivery.

Config keys: ``tok_s`` (50), ``default_tokens`` (256, when the request has no
``max_tokens``) and ``chunk_format``: ``"dict"`` (the default) or ``"str"``,
which yields each chunk already serialized (the runtime passes ``str`` chunks
through without ``json.dumps``).
"""

from __future__ import annotations

import ctypes
import json
import time
import uuid
from typing import Any, Dict, Iterator

from inferencekey.backend import BackendContext, CustomBackend, Job, Result, chat_completion

#: prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY): with kernel.yama.ptrace_scope = 1
#: only an ancestor may attach, and load.py's py-spy is a sibling of the server.
_PR_SET_PTRACER = 0x59616D61
_PR_SET_PTRACER_ANY = ctypes.c_ulong(-1).value


def allow_ptrace() -> None:
    """Let any process of this user attach a profiler to this one (Linux; a no-op elsewhere)."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PTRACER, _PR_SET_PTRACER_ANY, 0, 0, 0)
    except (OSError, AttributeError):
        pass


class NullBackend(CustomBackend):
    name = "null"
    version = "1"
    task_type = "text2text"
    api = "openai-chat"

    def setup(self, ctx: BackendContext) -> None:
        allow_ptrace()
        self.tok_s = float(ctx.config.get("tok_s", 50.0))
        self.default_tokens = int(ctx.config.get("default_tokens", 256))
        self.chunk_format = str(ctx.config.get("chunk_format", "dict"))
        if self.chunk_format not in ("dict", "str"):
            raise ValueError("chunk_format must be 'dict' or 'str'")

    def process(self, job: Job) -> Result:
        return Result(output={"text": "null"})

    def chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        n = int(request.get("max_tokens") or self.default_tokens)
        return chat_completion("tok " * n, model=self.name, prompt_tokens=1, completion_tokens=n)

    def chat_stream(self, request: Dict[str, Any]) -> Iterator[Any]:
        n = int(request.get("max_tokens") or self.default_tokens)
        base = {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": self.name}

        def chunk(choices: list, **extra: Any) -> Any:
            obj = {**base, "choices": choices, **extra}
            return json.dumps(obj) if self.chunk_format == "str" else obj

        interval = 1.0 / self.tok_s if self.tok_s > 0 else 0.0
        t_enter = time.time()
        start = time.monotonic()
        yield chunk([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
        for i in range(n):
            if interval:
                delay = start + i * interval - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            if i == 0:
                # Wall-clock stamps for load.py: when the runtime first resumed
                # this generator and when the first token was yielded, so the
                # time the runtime adds to the first token can be told apart
                # from the backend's own first-yield latency.
                yield chunk([{"index": 0, "delta": {"content": "tok "}, "finish_reason": None}],
                            bench_t=[t_enter, time.time()])
                continue
            yield chunk([{"index": 0, "delta": {"content": "tok "}, "finish_reason": None}])
        yield chunk([{"index": 0, "delta": {}, "finish_reason": "length"}])
        yield chunk([], usage={"prompt_tokens": 1, "completion_tokens": n, "total_tokens": n + 1})

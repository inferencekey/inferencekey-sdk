"""Limite 1B - Violetto (Paradigma) as an InferenceKey custom backend.

Why a custom backend: Violetto is a new architecture (``LimiteForCausalLM``)
that stock vLLM does not know. Paradigma ships it as an out-of-tree vLLM
plugin (``limite-vllm``, registered with ``VLLM_PLUGINS=limite``). The built-in
``vllm`` backend installs vLLM only, so this backend brings the plugin along:

* ``requirements.txt`` installs vLLM 0.26.0 (torch 2.11, CUDA 13.0 wheels),
  Transformers 5.6.2 and the plugin from Paradigma's GitHub, pinned to a commit.
* ``setup()`` starts ``vllm serve paradigma-inc/limite-1b-violetto`` on a free
  loopback port with ``VLLM_PLUGINS=limite`` and waits for ``/health``. vLLM
  downloads the 2.1 GB checkpoint from Hugging Face on first use.
* ``chat()`` / ``chat_stream()`` forward OpenAI chat requests to vLLM, so the
  workload endpoint speaks ``/v1/chat/completions`` with token streaming.
  Streaming matters here: a competition problem can take tens of thousands of
  tokens, more than any non-streamed request timeout.

Violetto's chat template accepts one user turn and only its own fixed system
prompt, so the backend drops any other system message before forwarding.

Settings are class attributes. Override them with the workload ``config``
(keys in lower case) or ``LIMITE_*`` environment variables.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

# Violetto's fixed system prompt (from its chat template). Any other system
# message makes the template raise, so other system messages are dropped.
CANONICAL_SYSTEM = (
    "You are a helpful assistant.\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def _log(msg: str) -> None:
    print(f"[limite-backend] {msg}", file=sys.stderr, flush=True)


def _die_with_parent() -> None:
    """Child-side hook: get SIGTERM when this backend process dies (Linux).

    The worker may stop the backend while ``setup()`` is still waiting for vLLM
    (a failed start, a deleted workload). Without this, ``vllm serve`` would be
    left running as an orphan holding the GPU memory.
    """
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LimiteVioletto(CustomBackend):
    name = "limite-1b-violetto"
    version = "0.1.2"
    task_type = "text2text"
    description = "Limite 1B - Violetto (Paradigma) on vLLM 0.26 with the Limite plugin"

    model = "paradigma-inc/limite-1b-violetto"
    served_name = "limite-1b-violetto"
    # vLLM engine settings. 131072 is the model's native context; the
    # authors' evaluator gives a completion everything but a 4096-token
    # prompt reserve.
    max_model_len = 131072
    gpu_memory_utilization = 0.90
    max_num_seqs = 64
    extra_args = ""  # appended verbatim to `vllm serve`
    # Sampling the authors recommend (also in generation_config.json).
    temperature = 0.6
    top_p = 0.95
    ready_timeout_s = 1500
    request_timeout_s = 3600

    def _setting(self, ctx: BackendContext, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"LIMITE_{key.upper()}")
        if raw is None:
            raw = (ctx.config or {}).get(key)
        if raw is None:
            return default
        return type(default)(raw)

    # ------------------------------------------------------------------ setup

    def setup(self, ctx: BackendContext) -> None:
        for key in ("model", "served_name", "max_model_len", "gpu_memory_utilization",
                    "max_num_seqs", "extra_args", "temperature", "top_p",
                    "ready_timeout_s", "request_timeout_s"):
            setattr(self, key, self._setting(ctx, key))
        self.port = _free_port()
        vllm_bin = os.path.join(os.path.dirname(sys.executable), "vllm")
        cmd = [
            vllm_bin, "serve", self.model,
            "--host", "127.0.0.1", "--port", str(self.port),
            "--served-model-name", self.served_name,
            "--dtype", "bfloat16",
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-num-seqs", str(self.max_num_seqs),
        ] + self.extra_args.split()
        env = dict(os.environ)
        env["VLLM_PLUGINS"] = "limite"  # load only Paradigma's plugin
        # FlashInfer JIT-compiles its sampling kernels with `ninja` during
        # vLLM's warmup. Stock GPU images lack it; the `ninja` wheel from
        # requirements.txt puts it in this venv's bin, so put that on PATH.
        env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
        _log("starting: VLLM_PLUGINS=limite " + " ".join(cmd))
        t0 = time.time()
        # Same stdout/stderr as this process, so vLLM's log lands in the worker log.
        self.proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, preexec_fn=_die_with_parent)
        atexit.register(self._stop)
        deadline = t0 + float(self.ready_timeout_s)
        while True:
            code = self.proc.poll()
            if code is not None:
                raise RuntimeError(f"vllm serve exited with code {code} before it was ready")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            if time.time() > deadline:
                self._stop()
                raise RuntimeError(f"vllm serve not ready after {self.ready_timeout_s} s")
            time.sleep(1.0)
        self.ready_after_s = round(time.time() - t0, 1)
        _log(f"vLLM ready on 127.0.0.1:{self.port} after {self.ready_after_s} s")

    def _stop(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    # --------------------------------------------------------------- requests

    def _body(self, request: Dict[str, Any], stream: bool) -> Dict[str, Any]:
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise InvalidInput("expected a non-empty 'messages' list")
        kept: List[Dict[str, Any]] = []
        for m in messages:
            if not isinstance(m, dict):
                raise InvalidInput("each message must be an object")
            if m.get("role") == "system" and m.get("content") != CANONICAL_SYSTEM:
                continue  # the template only accepts its own system prompt
            kept.append(m)
        body = {k: v for k, v in request.items() if k not in ("messages", "model", "stream", "stream_options")}
        extra = body.pop("extra_body", None)
        if isinstance(extra, dict):
            body.update(extra)
        body["messages"] = kept
        body["model"] = self.served_name
        body.setdefault("temperature", self.temperature)
        body.setdefault("top_p", self.top_p)
        body["stream"] = stream
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def _post(self, body: Dict[str, Any]):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=float(self.request_timeout_s))
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if 400 <= e.code < 500:
                raise InvalidInput(f"vLLM {e.code}: {detail}") from e
            raise RuntimeError(f"vLLM {e.code}: {detail}") from e

    def chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with self._post(self._body(request, stream=False)) as resp:
            return json.loads(resp.read().decode())

    def chat_stream(self, request: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        with self._post(self._body(request, stream=True)) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    return
                yield json.loads(payload)

    def process(self, job: Job) -> Result:
        # Callers of /process (the async jobs API) get the text and the usage.
        t0 = time.time()
        out = self.chat({"messages": job.messages(), **{k: v for k, v in (job.input or {}).items()
                                                         if k in ("max_tokens", "temperature", "top_p", "seed")}})
        choice = (out.get("choices") or [{}])[0]
        return Result(output={
            "text": (choice.get("message") or {}).get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "usage": out.get("usage"),
            "seconds": round(time.time() - t0, 2),
        })

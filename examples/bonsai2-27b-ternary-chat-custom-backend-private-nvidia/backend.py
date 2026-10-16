"""Ternary Bonsai 2 27B (Prism ML) as an InferenceKey custom backend.

Why a custom backend: the model ships as GGUF with two new weight types,
PTQ1_0 (dense trits, 1.75 bits/weight) and PQ2_0 (2-bit slots). Only Prism
ML's llama.cpp fork can read them; stock llama.cpp rejects them as unknown
types, and the built-in `llamacpp` backend installs stock llama.cpp. So this
backend brings its own runtime:

``setup()`` (once per process)
    1. downloads the fork's prebuilt Linux CUDA 12.8 tarball from its GitHub
       release (about 200 MB) and unpacks it,
    2. downloads the GGUF (5.95 GB for PTQ1_0) and, for image input, the
       vision projector (0.63 GB), in parallel with the tarball,
    3. starts the fork's ``llama-server`` on a free loopback port with the
       model card's flags and waits for ``/health``.

``process(job)`` (per request)
    forwards the chat request (messages, images, sampling, ``reasoning_effort``
    and other llama.cpp fields from ``extra_body``) to
    ``llama-server``'s OpenAI route and returns
    ``{"text", "reasoning", "finish_reason", "usage", "timings"}``.

The tarball does not bundle the CUDA runtime (``libcudart.so.12``,
``libcublas.so.12``). They come from the ``nvidia-*-cu12`` wheels in
requirements.txt, so the backend works on any host with a driver for CUDA
12.8 or newer, whatever CUDA toolkit the container image has.

Settings are class attributes. Override them with ``BONSAI_*`` environment
variables or the runtime's ``--config-json`` (keys in lower case).
"""

from __future__ import annotations

import atexit
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

RELEASE_BASE = "https://github.com/PrismML-Eng/llama.cpp/releases/download"

# Fields of an OpenAI chat body that are passed to llama-server untouched.
_SAMPLING_KEYS = ("top_p", "stop", "presence_penalty", "frequency_penalty", "seed")


def _log(msg: str) -> None:
    print(f"[bonsai-backend] {msg}", file=sys.stderr, flush=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _download(url: str, dest: Path) -> float:
    """Stream ``url`` to ``dest`` (atomic rename). Returns seconds spent."""
    if dest.is_file() and dest.stat().st_size > 0:
        return 0.0
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "inferencekey-bonsai-backend"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=8 << 20)
    tmp.rename(dest)
    return time.time() - t0


def _cuda_wheel_lib_dirs() -> List[str]:
    """``lib`` folders of the nvidia-*-cu12 wheels installed in this venv."""
    dirs: List[str] = []
    try:
        import nvidia  # namespace package from nvidia-cuda-runtime-cu12 / nvidia-cublas-cu12
    except ImportError:
        return dirs
    for root in list(getattr(nvidia, "__path__", [])):
        dirs.extend(sorted(d for d in glob.glob(os.path.join(root, "*", "lib")) if os.path.isdir(d)))
    return dirs


class TernaryBonsai2Backend(CustomBackend):
    name = "ternary-bonsai-2-27b"
    version = "0.1.0"
    task_type = "text2text"
    description = "Ternary Bonsai 2 27B (Prism ML) GGUF on Prism ML's llama.cpp fork"

    # Runtime: Prism ML's llama.cpp fork, prebuilt for Linux x64 + CUDA 12.8.
    # (Their KNOWN_ISSUES.md: the CUDA 13.3 build segfaults on some Linux systems.)
    release_tag = "prism-b10777-e3de715"
    release_asset = "linux-cuda-12.8-x64"
    # Weights.
    hf_repo = "prism-ml/Ternary-Bonsai-2-27B-gguf"
    gguf_file = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"  # or Ternary-Bonsai-2-27B-PQ2_0.gguf
    mmproj_file = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"  # "" = text only
    # Server flags. Context and sampling follow the model card.
    ctx_size = 65536
    parallel = 0  # 0 = llama-server's default slot count
    server_args = "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05"
    # Request defaults (the caller can override both per request).
    default_reasoning_effort = "medium"  # "medium" or "xhigh"; "high" is a 500 in this template
    default_max_tokens = 16384
    request_timeout_s = 290  # the worker gives each /process call 300 s
    ready_timeout_s = 900

    # ------------------------------------------------------------------ setup

    def _setting(self, ctx: BackendContext, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"BONSAI_{key.upper()}")
        if raw is None:
            raw = (ctx.config or {}).get(key)
        if raw is None:
            return default
        return type(default)(raw) if not isinstance(default, str) else str(raw)

    def setup(self, ctx: BackendContext) -> None:
        for key in ("release_tag", "release_asset", "hf_repo", "gguf_file", "mmproj_file",
                    "ctx_size", "parallel", "server_args", "default_reasoning_effort",
                    "default_max_tokens", "request_timeout_s", "ready_timeout_s"):
            setattr(self, key, self._setting(ctx, key))
        cache = Path(os.environ.get("BONSAI_CACHE_DIR", Path.home() / ".cache" / "bonsai-backend"))
        self.timings: Dict[str, float] = {}
        t_setup = time.time()

        tag, asset = self.release_tag, self.release_asset
        tarball = cache / f"llama-{tag}-bin-{asset}.tar.gz"
        bin_dir = cache / f"llama-{tag}-{asset}"
        url = f"{RELEASE_BASE}/{tag}/llama-{tag}-bin-{asset}.tar.gz"

        def fetch_runtime() -> float:
            if (bin_dir / "llama-server").is_file():
                return 0.0
            took = _download(url, tarball)
            tmp = Path(tempfile.mkdtemp(dir=cache, prefix=".extract-"))
            with tarfile.open(tarball) as tf:
                try:
                    tf.extractall(tmp, filter="data")
                except TypeError:  # Python without extraction filters
                    tf.extractall(tmp)
            # The archive holds one top-level folder (llama-<tag>/).
            inner = [p for p in tmp.iterdir() if p.is_dir()]
            src = inner[0] if len(inner) == 1 and not (tmp / "llama-server").exists() else tmp
            src.rename(bin_dir)
            shutil.rmtree(tmp, ignore_errors=True)
            tarball.unlink(missing_ok=True)
            return took

        def fetch_hf(filename: str) -> str:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(repo_id=self.hf_repo, filename=filename)

        _log(f"fetching {url} and {self.hf_repo}/{self.gguf_file}"
             + (f" + {self.mmproj_file}" if self.mmproj_file else ""))
        with ThreadPoolExecutor(max_workers=3) as pool:
            t0 = time.time()
            f_rt = pool.submit(fetch_runtime)
            f_model = pool.submit(fetch_hf, self.gguf_file)
            f_mm = pool.submit(fetch_hf, self.mmproj_file) if self.mmproj_file else None
            f_rt.result()
            self.timings["runtime_ready_s"] = round(time.time() - t0, 1)
            model_path = f_model.result()
            mmproj_path = f_mm.result() if f_mm else None
            self.timings["downloads_s"] = round(time.time() - t0, 1)
        _log(f"downloads done in {self.timings['downloads_s']} s (runtime {self.timings['runtime_ready_s']} s)")

        env = dict(os.environ)
        lib_dirs = [str(bin_dir)] + _cuda_wheel_lib_dirs()
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p])

        self.port = _free_port()
        cmd = [str(bin_dir / "llama-server"), "-m", model_path,
               "--host", "127.0.0.1", "--port", str(self.port),
               "-ngl", "99", "-fa", "on", "-c", str(self.ctx_size), "--jinja",
               "--alias", "ternary-bonsai-2-27b"]
        if mmproj_path:
            cmd += ["--mmproj", mmproj_path]
        if int(self.parallel) > 0:
            cmd += ["-np", str(self.parallel)]
        cmd += self.server_args.split()
        _log("starting: " + " ".join(cmd))
        # Same stdout/stderr as this process, so llama-server's log lands in the worker log.
        self.proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL)
        # The runtime's SIGTERM handler exits through atexit; the worker also
        # stops the whole process group, so llama-server never outlives us.
        atexit.register(self._stop)

        t_load = time.time()
        deadline = t_load + float(self.ready_timeout_s)
        while True:
            code = self.proc.poll()
            if code is not None:
                raise RuntimeError(f"llama-server exited with code {code} before it was ready")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            if time.time() > deadline:
                self._stop()
                raise RuntimeError(f"llama-server not ready after {self.ready_timeout_s} s")
            time.sleep(1.0)
        self.timings["server_load_s"] = round(time.time() - t_load, 1)
        self.timings["setup_s"] = round(time.time() - t_setup, 1)
        _log(f"llama-server ready on 127.0.0.1:{self.port}: {json.dumps(self.timings)}")
        self._lock = threading.Lock()

    def _stop(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    # ---------------------------------------------------------------- process

    def _body(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        messages = inp.get("messages")
        if not isinstance(messages, list) or not messages:
            prompt = inp.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise InvalidInput("expected 'messages' (OpenAI chat) or 'prompt'")
            messages = [{"role": "user", "content": prompt}]
        # The chat template wants exactly one system message, first (KNOWN_ISSUES.md).
        systems = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
        if len(systems) > 1 or (systems and messages[0] is not systems[0]):
            text = "\n\n".join(str(m.get("content", "")) for m in systems)
            messages = [{"role": "system", "content": text}] + [m for m in messages if m not in systems]
        body: Dict[str, Any] = {
            "model": "ternary-bonsai-2-27b",
            "messages": messages,
            "max_tokens": inp.get("max_tokens") or int(self.default_max_tokens),
            "stream": False,
        }
        if inp.get("temperature") is not None:
            body["temperature"] = inp["temperature"]
        for key in _SAMPLING_KEYS:
            if inp.get(key) is not None:
                body[key] = inp[key]
        for key in ("tools", "tool_choice"):
            if inp.get(key) is not None:
                body[key] = inp[key]
        # The platform carries non-OpenAI fields (reasoning_effort, top_k, min_p,
        # chat_template_kwargs, ...) in `extra_body`; the async jobs API sends them
        # at the top level. Accept both.
        extra = inp.get("extra_body") if isinstance(inp.get("extra_body"), dict) else {}
        for key in ("reasoning_effort", "top_k", "min_p", "repeat_penalty", "chat_template_kwargs",
                    "thinking_budget_tokens", "reasoning_budget", "n_probs"):
            if key in inp and key not in extra:
                extra[key] = inp[key]
        body.update(extra)
        body.setdefault("reasoning_effort", self.default_reasoning_effort)
        if body.get("reasoning_effort") == "high":
            body["reasoning_effort"] = "xhigh"  # "high" is an HTTP 500 in this chat template
        return body

    def process(self, job: Job) -> Result:
        body = self._body(job.input)
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=float(self.request_timeout_s)) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if 400 <= e.code < 500:
                raise InvalidInput(f"llama-server {e.code}: {detail}") from e
            raise RuntimeError(f"llama-server {e.code}: {detail}") from e
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        t = resp.get("timings") or {}
        out: Dict[str, Any] = {
            "text": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason"),
            "usage": resp.get("usage"),
            "timings": {
                "prompt_tokens": t.get("prompt_n"),
                "prompt_tokens_per_s": round(t["prompt_per_second"], 1) if t.get("prompt_per_second") else None,
                "generated_tokens": t.get("predicted_n"),
                "generation_tokens_per_s": round(t["predicted_per_second"], 1) if t.get("predicted_per_second") else None,
                "total_s": round(time.time() - t0, 2),
            },
        }
        if msg.get("tool_calls"):
            out["tool_calls"] = msg["tool_calls"]
        return Result(output=out)


class TernaryBonsai2PQ2Backend(TernaryBonsai2Backend):
    """The PQ2_0 packing (7.21 GB): faster decode on H100, A100 and Blackwell, faster prefill everywhere."""

    name = "ternary-bonsai-2-27b-pq2"
    gguf_file = "Ternary-Bonsai-2-27B-PQ2_0.gguf"

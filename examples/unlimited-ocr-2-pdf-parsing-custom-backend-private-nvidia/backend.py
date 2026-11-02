"""Baidu Unlimited OCR 2 as an InferenceKey custom backend (whole-document parsing).

Unlimited OCR 2 (Baidu, MIT) is a 3B mixture-of-experts document parser
(about 0.5B active parameters, DeepEncoder vision tower in front of a
DeepSeek-V2 style decoder). It reads every page of a document in ONE request
and writes one Markdown answer, with one ``<PAGE>`` marker per input page, so a
table or a paragraph that continues on the next page can be merged instead of
being cut at the page break. Its Sparse Reference Sliding Window Attention
(SR-SWA) keeps the per-token decoding cost almost flat as the document grows.

The authors serve it with a customised SGLang build that ships inside the model
repository (``wheel/sglang-0.5.20.dev13+g75b9b06b02-py3-none-any.whl``). No
built-in InferenceKey backend can install that wheel, so this backend does it:

``setup()`` (once per process, cached on disk)
    1. downloads the checkpoint (6.7 GB of BF16 weights) at a pinned revision,
    2. builds a Python 3.10 environment with uv and installs the authors'
       SGLang wheel, PyMuPDF and fastokens (the model card's recipe),
    3. starts ``sglang.launch_server`` on a loopback port with the authors'
       flags and environment, sized by the knobs below, and waits for it.

``chat()`` / ``chat_stream()`` (per request) turn the request into page images
and forward ONE multi-page request to that server, with the authors' n-gram
repetition guard, streaming the Markdown back.

Callers
-------
``POST .../v1/chat/completions`` (OpenAI shape). The user message may carry:

* a PDF as an OpenAI ``file`` part:
  ``{"type": "file", "file": {"filename": "doc.pdf", "file_data": "data:application/pdf;base64,..."}}``
* page images as ``image_url`` parts (``data:`` URIs or http(s) URLs), in page order,
* or a text part with an http(s) link to a PDF or an image (handy from a chat
  playground), optionally followed by ``pages 3-9``.

Options, as top-level request fields (direct calls) or written in the message
text (through the platform, which forwards the standard chat fields only, e.g.
"https://example.org/report.pdf pages 1-10 crop_mode=false"): ``pages`` ("1-10" or [1, 2, 5]),
``dpi`` (default 300, as the authors), ``crop_mode`` (true = gundam tiles, the
authors' default; false = one global view per page, fewer tokens),
``tokens_per_page`` (output budget per page, default 8000).

Server knobs (workload ``config`` keys or ``UOCR_*`` environment variables):
``page_window`` (pages per scheduling window, default 10; 0 = the whole
document in one window), ``mem_fraction_static`` (0.7, as the authors; it is
also how we emulated smaller cards), ``max_running_requests`` (8),
``max_prefill_tokens`` (2048), ``cuda_graph`` (true), ``extra_server_args``,
``hf_repo`` / ``hf_revision``, ``ready_timeout_s``, ``cache_dir``.

Attribution: Unlimited OCR 2 by Baidu, MIT license
(https://huggingface.co/baidu/Unlimited-OCR-2, https://github.com/baidu/Unlimited-OCR);
the SGLang build is the authors' fork of SGLang (Apache-2.0).
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result, chat_completion

WHEEL = "wheel/sglang-0.5.20.dev13+g75b9b06b02-py3-none-any.whl"
PAGE_TOKEN_ID = 128827  # <PAGE> (config.json sparse_attention_config.page_token_id)
EOS_TOKEN_ID = 1
URL_RE = re.compile(r"https?://\S+")
PAGES_RE = re.compile(r"\bpages?\s+(\d+)\s*(?:-|to)\s*(\d+)", re.I)
# Options written in the message text (the platform forwards the standard chat fields only):
# "crop_mode=false" or "base mode", "dpi=200", "tokens_per_page=4000".
OPT_RE = re.compile(r"\b(crop_mode|dpi|tokens_per_page)\s*=\s*(\w+)", re.I)


def _log(msg: str) -> None:
    print(f"[unlimited-ocr-2] {msg}", file=sys.stderr, flush=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _child_setup() -> None:
    """Runs in the server child before exec: its own process group, and SIGTERM when this
    backend process dies (the worker may stop a custom backend with SIGKILL, which no handler
    sees; without this the server would outlive the backend and keep the GPU)."""
    os.setsid()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - not Linux
        pass


def _parse_pages(spec: Any, total: int) -> List[int]:
    """1-based page list from "3-9", "1,4,7", [1, 2] or None (all pages)."""
    if spec in (None, "", "all"):
        return list(range(1, total + 1))
    if isinstance(spec, list):
        pages = [int(p) for p in spec]
    else:
        pages = []
        for part in str(spec).split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                pages += list(range(int(a), int(b) + 1))
            elif part:
                pages.append(int(part))
    pages = [p for p in pages if 1 <= p <= total]
    if not pages:
        raise InvalidInput(f"no valid pages in {spec!r} (document has {total})")
    return pages


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (inferencekey unlimited-ocr-2 backend)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _data_uri_bytes(uri: str) -> bytes:
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    return _fetch(uri)


class UnlimitedOcr2Backend(CustomBackend):
    name = "unlimited-ocr-2"
    version = "0.1.2"
    task_type = "text2text"
    api = "openai-chat"
    requirements = "requirements.txt"
    requires_python = ">=3.10,<3.13"
    description = "Baidu Unlimited OCR 2 (authors' SGLang build): a whole PDF to Markdown in one request"

    # Pinned upstream: what this example was tested with (2026-11-02).
    hf_repo = "baidu/Unlimited-OCR-2"
    hf_revision = "c14071b549f4fcab10909f3c4acc6a2edb366947"
    page_window = 10
    mem_fraction_static = 0.7
    max_running_requests = 8
    max_prefill_tokens = 2048
    cuda_graph = True
    extra_server_args = ""
    max_page_tokens = 20000
    ready_timeout_s = 1800.0
    request_timeout_s = 3600.0
    cache_dir = ""

    # ------------------------------------------------------------------ setup

    def _setting(self, ctx: BackendContext, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"UOCR_{key.upper()}")
        if raw is None:
            raw = (ctx.config or {}).get(key)
        if raw is None:
            return default
        if isinstance(default, bool):
            return str(raw).strip().lower() in ("1", "true", "yes", "on")
        return type(default)(raw)

    def setup(self, ctx: BackendContext) -> None:
        for key in ("hf_repo", "hf_revision", "page_window", "mem_fraction_static", "max_running_requests",
                    "max_prefill_tokens", "cuda_graph", "extra_server_args", "max_page_tokens",
                    "ready_timeout_s", "request_timeout_s", "cache_dir"):
            setattr(self, key, self._setting(ctx, key))
        # Keep everything on the container's local disk: loading weights from a network
        # mount (/workspace on many GPU clouds) is many times slower.
        cache = Path(self.cache_dir or Path.home() / ".cache" / "unlimited-ocr-2-backend")
        cache.mkdir(parents=True, exist_ok=True)
        self.timings: Dict[str, float] = {}
        t_setup = time.time()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        # 1. The checkpoint (the 156 MB replay GIF and the paper are skipped).
        from huggingface_hub import snapshot_download

        t0 = time.time()
        self.model_dir = Path(snapshot_download(
            self.hf_repo, revision=self.hf_revision,
            local_dir=str(cache / "models" / self.hf_repo.split("/")[-1]),
            ignore_patterns=["assets/*", "*.pdf"]))
        self.timings["download_s"] = round(time.time() - t0, 1)

        # 2. The authors' SGLang build in its own Python 3.10 environment.
        t0 = time.time()
        venv = self._sglang_env(cache)
        self.timings["env_s"] = round(time.time() - t0, 1)
        self.py = str(venv / "bin" / "python")
        self.ngram_processor = subprocess.run(
            [self.py, "-c", "from sglang.srt.sampling.custom_logit_processor import "
             "DeepseekOCRNoRepeatNGramLogitProcessor as P; print(P.to_str())"],
            capture_output=True, text=True, check=True).stdout.strip()
        _log(f"weights in {self.timings['download_s']} s, environment in {self.timings['env_s']} s")

        # 3. The server, with the authors' environment and flags.
        self.port = _free_port()
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update({
            "VIRTUAL_ENV": str(venv),
            "PATH": f"{venv / 'bin'}:{env.get('PATH', '')}",
            # Page-window scheduling: the KV tail of the previous window is reused.
            "SGLANG_UOCR_EANABLE_DOC_LIMIT": "1" if self.page_window > 0 else "0",  # sic, the authors' name
            "SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW": str(max(self.page_window, 1)),
            "SGLANG_UOCR_DOC_BASE_PAGE_WINDOW": str(max(self.page_window, 1)),
            "SGLANG_UOCR_MAX_PAGE_TOKENS": str(self.max_page_tokens),
            "SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB": "64",
            "SGLANG_UOCR_SHM_WRAP_MODEL_DATA": "1",
            "SGLANG_UOCR_SHM_WRAP_MIN_MB": "1",
            "SGLANG_VLM_CACHE_SIZE_MB": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        mpt = str(self.max_prefill_tokens)
        cmd = [self.py, "-m", "sglang.launch_server",
               "--model-path", str(self.model_dir), "--served-model-name", "Unlimited-OCR-2",
               "--chat-template", "unlimited-ocr", "--tokenizer-backend", "fastokens",
               "--tokenizer-worker-num", "4",
               "--mem-fraction-static", str(self.mem_fraction_static),
               "--max-running-requests", str(self.max_running_requests),
               "--max-prefill-tokens", mpt,
               "--enable-custom-logit-processor", "--enable-streaming-session",
               "--allow-auto-truncate", "--disable-radix-cache", "--mm-preprocess-cache-size-mb", "0",
               "--host", "127.0.0.1", "--port", str(self.port)]
        if self.cuda_graph:
            bs = str(self.max_running_requests)
            cmd += ["--cuda-graph-backend-decode", "full", "--cuda-graph-max-bs-decode", bs,
                    "--cuda-graph-backend-prefill", "full", "--cuda-graph-max-bs-prefill", "16384",
                    "--cuda-graph-bs-prefill", str(int(mpt) // 2), mpt,
                    "--cuda-graph-config", json.dumps({"prefill": {"backend": "full", "full_prefill_max_req": int(bs)}})]
        else:
            cmd += ["--disable-cuda-graph"]
        if self.extra_server_args.strip():
            cmd += self.extra_server_args.split()
        self.pidfile = cache / "sglang.pid"
        self._kill_stale()
        self.server_log = cache / f"sglang-{self.port}.log"
        _log("starting: " + " ".join(cmd))
        log = open(self.server_log, "ab")
        self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(cache),
                                     preexec_fn=_child_setup)
        self.pidfile.write_text(str(self.proc.pid))
        atexit.register(self._stop_server)
        t0 = time.time()
        self._wait_ready(t0)
        self.timings["server_start_s"] = round(time.time() - t0, 1)
        self.timings["setup_s"] = round(time.time() - t_setup, 1)
        self.tmp_root = cache / "pages"
        self.tmp_root.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        _log(f"ready on port {self.port}, page_window={self.page_window}; timings {self.timings}")

    def _sglang_env(self, cache: Path) -> Path:
        venv = cache / f"sglang-venv-{self.hf_revision[:8]}"
        done = venv / ".inferencekey-ready"
        if done.is_file():
            return venv
        uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
        if not Path(uv).exists():
            raise RuntimeError("uv not found: it is in requirements.txt; is the backend venv complete?")
        env = dict(os.environ, UV_LINK_MODE="copy")
        env.pop("VIRTUAL_ENV", None)

        def run(*args: str) -> None:
            _log("$ " + " ".join(args))
            subprocess.run(list(args), env=env, check=True)

        run(uv, "venv", "--python", "3.10", str(venv))
        py = str(venv / "bin" / "python")
        run(uv, "pip", "install", "--python", py, str(self.model_dir / WHEEL),
            "pymupdf==1.27.2.2", "fastokens==0.3.2")
        done.write_text(self.hf_revision)
        return venv

    def _kill_stale(self) -> None:
        try:
            pid = int(self.pidfile.read_text().strip())
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, ValueError):
            return
        if "sglang.launch_server" in cmdline:
            _log(f"stopping an SGLang server left by an earlier process (pid {pid})")
            try:
                os.killpg(pid, signal.SIGTERM)
                for _ in range(60):
                    if not Path(f"/proc/{pid}").exists():
                        break
                    time.sleep(1)
                else:
                    os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _stop_server(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass

    def _wait_ready(self, t0: float) -> None:
        url = f"http://127.0.0.1:{self.port}/v1/models"
        while True:
            if self.proc.poll() is not None:
                tail = self.server_log.read_text(errors="replace")[-4000:]
                raise RuntimeError(f"sglang exited with {self.proc.returncode}:\n{tail}")
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception:  # noqa: BLE001 - not up yet
                pass
            if time.time() - t0 > self.ready_timeout_s:
                raise RuntimeError(f"sglang not ready after {self.ready_timeout_s} s; log {self.server_log}")
            time.sleep(3)

    # ---------------------------------------------------------------- request

    def _pages(self, request: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Page images (data URIs, in order) from the request, plus what was done."""
        pdfs: List[bytes] = []
        images: List[str] = []
        page_spec = request.get("pages")
        for msg in request.get("messages") or []:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
            for part in parts:
                kind = part.get("type")
                if kind in ("file", "input_file"):
                    f = part.get("file") or part
                    data = f.get("file_data") or f.get("data") or ""
                    pdfs.append(_data_uri_bytes(data) if data.startswith(("data:", "http")) else base64.b64decode(data))
                elif kind == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    images.append(url if url.startswith("data:") else
                                  "data:image/png;base64," + base64.b64encode(_fetch(url)).decode())
                elif kind == "text":
                    text = part.get("text") or ""
                    for url in URL_RE.findall(text):
                        url = url.rstrip(").,;")
                        blob = _fetch(url)
                        if blob[:4] == b"%PDF":
                            pdfs.append(blob)
                        else:
                            images.append("data:image/png;base64," + base64.b64encode(blob).decode())
                    m = PAGES_RE.search(text)
                    if m and page_spec is None:
                        page_spec = f"{m.group(1)}-{m.group(2)}"
                    for key, value in OPT_RE.findall(text):
                        request.setdefault(key.lower(), value)
                    if re.search(r"\bbase mode\b", text, re.I):
                        request.setdefault("crop_mode", False)
        info: Dict[str, Any] = {}
        if pdfs:
            import fitz  # PyMuPDF

            dpi = int(request.get("dpi") or 300)
            t0 = time.time()
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            for blob in pdfs:
                doc = fitz.open(stream=blob, filetype="pdf")
                for p in _parse_pages(page_spec, len(doc)):
                    png = doc[p - 1].get_pixmap(matrix=mat).tobytes("png")
                    images.append("data:image/png;base64," + base64.b64encode(png).decode())
                info["pdf_pages_total"] = len(doc)
                doc.close()
            info["render_s"] = round(time.time() - t0, 2)
            info["dpi"] = dpi
        if not images:
            raise InvalidInput("no document: send a PDF as a 'file' part, page images as 'image_url' parts, "
                               "or a link to a PDF in the text")
        info["pages"] = len(images)
        return images, info

    def _payload(self, request: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        images, info = self._pages(request)
        n = len(images)
        crop = request.get("crop_mode", True)
        crop = crop if isinstance(crop, bool) else str(crop).lower() in ("1", "true", "yes")
        tokens_per_page = int(request.get("tokens_per_page") or 8000)
        content = [{"type": "image_url", "image_url": {"url": u}} for u in images]
        content.append({"type": "text", "text": "<PAGE><image>" * n + "Multi page parsing."})
        payload = {
            "model": "Unlimited-OCR-2",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(request.get("max_tokens") or tokens_per_page * n),
            "temperature": 0,
            "skip_special_tokens": False,
            "images_config": {"crop_mode": crop},
            "custom_logit_processor": self.ngram_processor,
            "custom_params": {"ngram_size": 35, "window_size": 1024 if n > 1 else 128,
                              "page_token_id": PAGE_TOKEN_ID, "eos_token_id": EOS_TOKEN_ID,
                              "max_page_tokens": self.max_page_tokens},
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        info.update({"crop_mode": crop, "page_window": self.page_window})
        return payload, info

    def _stream_upstream(self, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=self.request_timeout_s)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:2000]
            if 400 <= e.code < 500:
                raise InvalidInput(f"sglang rejected the request ({e.code}): {detail}")
            raise RuntimeError(f"sglang error {e.code}: {detail}")
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                yield json.loads(data)

    def chat_stream(self, request: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        payload, info = self._payload(request)
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        t0 = time.time()
        first: Optional[float] = None
        usage: Dict[str, Any] = {}
        with self.lock if self.max_running_requests <= 1 else _nullcontext():
            for event in self._stream_upstream(payload):
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content") or ""
                    finish = choice.get("finish_reason")
                    if piece and first is None:
                        first = time.time() - t0
                    if piece or finish:
                        yield {"id": cid, "object": "chat.completion.chunk", "model": "Unlimited-OCR-2",
                               "choices": [{"index": 0, "delta": {"content": piece} if piece else {},
                                            "finish_reason": finish}]}
        seconds = time.time() - t0
        info.update({"seconds": round(seconds, 2), "first_token_s": round(first or 0, 2),
                     "pages_per_minute": round(info["pages"] / seconds * 60, 1) if seconds else None})
        _log(f"done: {info} usage {usage}")
        yield {"id": cid, "object": "chat.completion.chunk", "model": "Unlimited-OCR-2", "choices": [],
               "usage": {"prompt_tokens": usage.get("prompt_tokens"),
                         "completion_tokens": usage.get("completion_tokens"),
                         "total_tokens": usage.get("total_tokens")}}

    def chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        text: List[str] = []
        usage: Dict[str, Any] = {}
        finish = None
        for chunk in self.chat_stream(request):
            if chunk.get("usage"):
                usage = chunk["usage"]
            for c in chunk.get("choices") or []:
                text.append((c.get("delta") or {}).get("content") or "")
                finish = c.get("finish_reason") or finish
        return chat_completion("".join(text), model="Unlimited-OCR-2", finish_reason=finish or "stop",
                               prompt_tokens=usage.get("prompt_tokens"),
                               completion_tokens=usage.get("completion_tokens"))

    def process(self, job: Job) -> Result:
        request = dict(job.input or {})
        if "messages" not in request and request.get("pdf_b64"):
            request["messages"] = [{"role": "user", "content": [
                {"type": "file", "file": {"file_data": "data:application/pdf;base64," + request["pdf_b64"]}}]}]
        t0 = time.time()
        out = self.chat(request)
        text = out["choices"][0]["message"]["content"]
        return Result({"text": text, "pages": text.count("<PAGE>"), "seconds": round(time.time() - t0, 2),
                       "usage": out.get("usage")})


class _nullcontext:
    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, *exc):
        return False

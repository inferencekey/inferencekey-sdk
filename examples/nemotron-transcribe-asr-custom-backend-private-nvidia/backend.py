"""NVIDIA Nemotron 3.5 Transcribe as an InferenceKey custom backend (audio2text).

Nemotron 3.5 Transcribe (NVIDIA, OpenMDW-1.1) is a speech LLM: an audio
encoder (ASR transformer + Sortformer speaker encoder) in front of the
Nemotron 3.5 Lightning 30B-A3B hybrid Mamba-Transformer mixture-of-experts
language model, about 4.3B active parameters. The text instruction picks the
transcript: verbatim with fillers, cleaned, speaker-tagged, with word
timestamps (a CTC alignment head), translated, or an answer about the audio.
40 languages, recordings up to one hour.

The authors serve it with vLLM 0.32.0 plus the NeMo Speech vLLM plugin
(``VLLM_PLUGINS=nemo_speechlm``), installed together from the NeMo Speech
repository with uv. No built-in InferenceKey backend can install that pair, so
this backend does it:

``setup()`` (once per process, cached on disk)
    1. downloads NeMo Speech at a pinned commit and builds its environment the
       way the model card says (``uv sync --extra asr --extra vllm --locked``),
    2. downloads the checkpoint (NVFP4 by default: 21.6 GB, Blackwell only),
    3. starts ``vllm serve`` on a loopback port with the authors' flags, sized
       for one 32 GB card, and waits until it answers.

``process(job)`` (per request) sends one recording and one instruction to
that server and returns the transcript.

Callers
-------
* ``POST .../v1/audio/transcriptions`` (OpenAI multipart: ``file``, ``model``,
  and optionally ``prompt``, ``timestamp_granularities[]=word``,
  ``temperature``). ``prompt`` is the instruction (or a style name, below).
* a JSON body ``{"audio_b64": ..., "style" | "prompt": ..., "timestamps": bool,
  "language": "Spanish", "max_tokens": N}``.

Styles (the authors' own prompts, from the model card's prompt guide):
``verbatim`` (default), ``clean``, ``speakers`` (speaker tags; add
``timestamps`` for word times and diarization), ``nearfield``,
``translate`` (needs ``language``). Any other text is sent as the instruction.

Output: ``{"text", "style", "prompt", "ctc_timestamps" (when asked),
"duration", "seconds", "rtfx", "completion_tokens", "finish_reason", "model"}``.

Knobs (workload ``config`` keys or ``NEMOTRON_*`` environment variables):
``hf_repo`` / ``hf_revision`` (checkpoint), ``gpu_memory_utilization``
(default 0.92), ``max_model_len`` (65536), ``max_num_seqs`` (8), ``mtp``
(speculative tokens, 8; 0 turns MTP off), ``enforce_eager`` (false),
``extra_vllm_args``, ``ready_timeout_s`` (1800), ``cache_dir``.
``gpu_memory_utilization`` is also how we emulated smaller cards.

Attribution: NVIDIA-Nemotron-3.5-Transcribe-30B-A4B by NVIDIA, OpenMDW-1.1
(https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B);
NeMo Speech (Apache-2.0, https://github.com/NVIDIA-NeMo/Speech); vLLM
(Apache-2.0). The generation-loop guardrail is the authors' function, adapted
from Qwen3-ASR (Apache-2.0).
"""

from __future__ import annotations

import atexit
import base64
import email
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

NEMO_REPO = "NVIDIA-NeMo/Speech"

STYLES = {
    "verbatim": "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. "
                "Formatting: punctuation and capitalization. Output: transcript only.",
    "clean": "Audio transcription. Language: as spoken. Wording: omit fillers and accidental repetitions; keep the "
             "remaining spoken wording unchanged. Formatting: punctuation and capitalization. Output: transcript only.",
    "speakers": "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal "
                "fillers. Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> "
                "<spk:3> on the transcription.",
    "nearfield": "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. No "
                 "punctuation or capitalization. Transcribe only the speaker nearest to the microphone. Do not "
                 "transcribe any other voices.",
    "translate": "Please translate the given speech to {language}.",
}

# The authors' generation-loop guardrail (model card, "Generation-Loop Guardrail").
_CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
_BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)


def cleanup_generation_loops(text: str) -> str:
    text = _CHAR_RUN.sub(lambda m: m.group(1), text)
    return _BLOCK_RUN.sub(lambda m: m.group(1), text)


def _log(msg: str) -> None:
    print(f"[nemotron-transcribe] {msg}", file=sys.stderr, flush=True)


def _mime(data: bytes) -> str:
    if data[:4] == b"RIFF":
        return "audio/wav"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if data[4:8] == b"ftyp":
        return "audio/mp4"
    return "application/octet-stream"


def _multipart_fields(raw: bytes, content_type: str) -> Dict[str, Any]:
    """All parts of an OpenAI transcription multipart body: text fields and the file."""
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    msg = email.message_from_bytes(header + raw)
    out: Dict[str, Any] = {}
    for part in msg.walk():
        if part.is_multipart():
            continue
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True) or b""
        if not name:
            continue
        if name == "file" or part.get_filename():
            out["file"] = payload
        else:
            value = payload.decode("utf-8", "replace")
            if name.endswith("[]"):
                out.setdefault(name[:-2], []).append(value)
            else:
                out[name] = value
    return out


def _duration(path: Path) -> Optional[float]:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001 - formats libsndfile cannot read
        pass
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                  str(path)], capture_output=True, text=True, timeout=30)
            return float(out.stdout.strip())
        except Exception:  # noqa: BLE001
            return None
    return None


def _child_setup() -> None:
    """Runs in the vLLM child before exec: its own process group (so we can stop the whole server),
    and SIGTERM when this backend process dies. The worker stops a custom backend with SIGKILL,
    which no handler sees; without this the server outlives the backend and keeps the GPU."""
    os.setsid()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - not Linux
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fetch_tarball(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".dl-"))
    req = urllib.request.Request(url, headers={"User-Agent": "inferencekey-nemotron-transcribe-backend"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        try:
            tf.extractall(tmp, filter="data")
        except TypeError:  # Python < 3.12
            tf.extractall(tmp)
    inner = next(p for p in tmp.iterdir() if p.is_dir())
    if dest.exists():
        shutil.rmtree(dest)
    inner.rename(dest)
    tmp.rmdir()


class NemotronTranscribeBackend(CustomBackend):
    name = "nemotron-transcribe"
    version = "0.1.4"
    task_type = "audio2text"
    requirements = "requirements.txt"
    requires_python = ">=3.12,<3.13"
    description = "NVIDIA Nemotron 3.5 Transcribe (vLLM 0.32 + NeMo Speech): the prompt picks the transcript style"

    # Pinned upstream: what this example was tested with (2026-11-01).
    nemo_commit = "e5d79d5a49cbec49d79440b0070aac0c32a8dbb8"
    hf_repo = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
    hf_revision = "66c6adc4020179567074d0399cda2a4eeb2bb347"
    # vLLM sizing for one 32 GB card (the authors' profile is for 96 GB: 0.85, 64 sequences).
    gpu_memory_utilization = 0.92
    max_model_len = 65536
    max_num_seqs = 8
    mtp = 8
    enforce_eager = False
    extra_vllm_args = ""
    ready_timeout_s = 1800.0
    request_timeout_s = 1500.0
    cache_dir = ""

    # ------------------------------------------------------------------ setup

    def _setting(self, ctx: BackendContext, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"NEMOTRON_{key.upper()}")
        if raw is None:
            raw = (ctx.config or {}).get(key)
        if raw is None:
            return default
        if isinstance(default, bool):
            return str(raw).strip().lower() in ("1", "true", "yes", "on")
        return type(default)(raw)

    def setup(self, ctx: BackendContext) -> None:
        for key in ("nemo_commit", "hf_repo", "hf_revision", "gpu_memory_utilization", "max_model_len",
                    "max_num_seqs", "mtp", "enforce_eager", "extra_vllm_args", "ready_timeout_s",
                    "request_timeout_s", "cache_dir"):
            setattr(self, key, self._setting(ctx, key))
        cache = Path(self.cache_dir or Path.home() / ".cache" / "nemotron-transcribe-backend")
        cache.mkdir(parents=True, exist_ok=True)
        self.timings: Dict[str, float] = {}
        t_setup = time.time()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        cuda_bin = "/usr/local/cuda/bin"
        if Path(cuda_bin, "nvcc").exists() and cuda_bin not in os.environ.get("PATH", ""):
            # FlashInfer JIT-compiles the quantized kernels with nvcc (CUDA 13 toolkit).
            os.environ["PATH"] = f"{cuda_bin}:{os.environ.get('PATH', '')}"
            os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

        # 1. NeMo Speech + vLLM 0.32.0 in their own environment (Python 3.13, uv, the repo's lock file).
        t0 = time.time()
        venv = self._nemo_env(cache)
        self._flashinfer_kernels(venv)
        self.timings["env_s"] = round(time.time() - t0, 1)

        # 2. The checkpoint.
        from huggingface_hub import snapshot_download

        t0 = time.time()
        model_dir = Path(snapshot_download(self.hf_repo, revision=self.hf_revision,
                                           local_dir=str(cache / "models" / self.hf_repo.split("/")[-1])))
        self.timings["download_s"] = round(time.time() - t0, 1)
        _log(f"environment in {self.timings['env_s']} s, weights in {self.timings['download_s']} s")

        # 3. vLLM with the NeMo Speech plugin, the authors' flags sized for one 32 GB card.
        self.port = _free_port()
        self.served_name = self.hf_repo
        env = dict(os.environ)
        env.update({
            "VLLM_PLUGINS": "nemo_speechlm",
            "VLLM_MAX_AUDIO_DECODE_DURATION_S": "3600",
            "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "180",
            "VLLM_USE_V2_MODEL_RUNNER": "0",  # word timestamps need the V1 model runner
            "VLLM_DISABLED_KERNELS": "FlashInferFP8ScaledMMLinearKernel",
            "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
            "MAX_JOBS": os.environ.get("MAX_JOBS", "6"),
            "VIRTUAL_ENV": str(venv),
            "PATH": f"{venv / 'bin'}:{env.get('PATH', '')}",
        })
        env.pop("PYTHONPATH", None)
        cmd = [str(venv / "bin" / "vllm"), "serve", str(model_dir),
               "--host", "127.0.0.1", "--port", str(self.port),
               "--served-model-name", self.served_name,
               "--trust-remote-code", "--dtype", "bfloat16", "--kv-cache-dtype", "fp8",
               "--gpu-memory-utilization", str(self.gpu_memory_utilization),
               "--max-model-len", str(self.max_model_len), "--max-num-batched-tokens", "4096",
               "--max-num-seqs", str(self.max_num_seqs),
               "--enable-chunked-prefill", "--no-enable-prefix-caching",
               "--use-replayssm", "--mamba-backend", "flashinfer", "--mamba-cache-mode", "none",
               "--mm-processor-cache-gb", "0", "--no-async-scheduling",
               "--limit-mm-per-prompt", json.dumps({"audio": {"count": 1, "length": 57600000}}),
               "--reasoning-parser", "nemotron_v3", "--trust-request-mm-kwargs",
               "--middleware", "nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware"]
        if self.mtp > 0:
            cmd += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": self.mtp})]
        if self.enforce_eager:
            cmd += ["--enforce-eager"]
        if self.extra_vllm_args.strip():
            cmd += self.extra_vllm_args.split()
        # A vLLM server left behind by an earlier backend process (the worker stopped it with
        # SIGKILL) would still hold the GPU: stop it first.
        self.pidfile = cache / "vllm.pid"
        self._kill_stale()
        self.vllm_log = cache / f"vllm-{self.port}.log"
        _log("starting: " + " ".join(cmd))
        log = open(self.vllm_log, "ab")
        self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(cache),
                                     preexec_fn=_child_setup)
        self.pidfile.write_text(str(self.proc.pid))
        atexit.register(self._stop_server)
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous = signal.getsignal(sig)

            def _on_signal(signum, frame, previous=previous):
                self._stop_server()
                if callable(previous):
                    previous(signum, frame)
                else:
                    raise SystemExit(128 + signum)

            try:
                signal.signal(sig, _on_signal)
            except ValueError:  # not the main thread
                pass
        t0 = time.time()
        self._wait_ready(t0)
        self.timings["vllm_start_s"] = round(time.time() - t0, 1)

        # Warm-up: the first requests compile and tune kernels.
        t0 = time.time()
        silence = cache / "warmup.wav"
        if not silence.exists():
            import wave

            with wave.open(str(silence), "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
                w.writeframes(b"\x00\x00" * 32000)
        self._ask(silence.read_bytes(), STYLES["verbatim"], timestamps=False, max_tokens=32)
        self.timings["warmup_s"] = round(time.time() - t0, 1)
        self.timings["setup_s"] = round(time.time() - t_setup, 1)
        self.lock = threading.Lock()
        _log(f"ready: {self.hf_repo} on port {self.port}; timings {self.timings}")

    def _nemo_env(self, cache: Path) -> Path:
        src = cache / f"nemo-speech-{self.nemo_commit[:12]}"
        venv = src / ".venv"
        done = venv / ".inferencekey-ready"
        if done.is_file():
            return venv
        uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
        if not Path(uv).exists():
            raise RuntimeError("uv not found: it is in requirements.txt; is the backend venv complete?")
        if not (src / "pyproject.toml").is_file():
            _fetch_tarball(f"https://codeload.github.com/{NEMO_REPO}/tar.gz/{self.nemo_commit}", src)
        env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(venv), UV_LINK_MODE="copy")
        env.pop("VIRTUAL_ENV", None)

        def run(*args: str) -> None:
            _log("$ " + " ".join(args))
            subprocess.run(list(args), cwd=str(src), env=env, check=True)

        run(uv, "venv", "--python", "3.13", str(venv))
        run(uv, "sync", "--extra", "asr", "--extra", "vllm", "--locked", "--no-install-project")
        py = str(venv / "bin" / "python")
        run(uv, "pip", "install", "--python", py, "--no-deps", "-e", ".")
        # PyAV decodes compressed audio without a system FFmpeg; vLLM needs OpenCV, which NeMo excludes.
        run(uv, "pip", "install", "--python", py, "--no-deps", "av==19.0.2", "opencv-python-headless==5.0.0.93")
        done.write_text(self.nemo_commit)
        return venv

    def _flashinfer_kernels(self, venv: Path) -> None:
        """Prebuilt FlashInfer kernels for this GPU (flashinfer-jit-cache-<arch>), if FlashInfer publishes them.

        Without them FlashInfer compiles its NVFP4 GEMMs with nvcc on first start: several minutes,
        and ninja runs one nvcc per core, which ran a 62 GB container out of memory in our test
        ("FAILED: [code=137] ... Killed"). MAX_JOBS (default 6) caps what is still compiled.
        """
        marker = venv / ".inferencekey-flashinfer-kernels"
        if marker.is_file():
            return
        try:
            cap = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30).stdout.split()[0]
        except Exception:  # noqa: BLE001
            return
        major, minor = cap.split(".")
        version = subprocess.run([str(venv / "bin" / "python"), "-c", "import flashinfer; print(flashinfer.__version__)"],
                                 capture_output=True, text=True).stdout.strip()
        cuda = "cu130"
        uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
        for suffix in ("f", "a", ""):
            # The top-level package (it carries __version__, which FlashInfer checks) plus the
            # provider with the kernels for this architecture.
            pkgs = [f"flashinfer-jit-cache=={version}+{cuda}", f"flashinfer-jit-cache-sm{major}{minor}{suffix}=={version}+{cuda}"]
            pkg = pkgs[1]
            r = subprocess.run([uv, "pip", "install", "--python", str(venv / "bin" / "python"), "--no-deps",
                                "--index-url", f"https://flashinfer.ai/whl/{cuda}", *pkgs], capture_output=True, text=True)
            if r.returncode == 0:
                _log(f"installed {pkg}")
                marker.write_text(pkg)
                return
        _log(f"no prebuilt FlashInfer kernels for sm{major}{minor} {version}; they compile on first start")
        marker.write_text("none")

    def _kill_stale(self) -> None:
        try:
            pid = int(self.pidfile.read_text().strip())
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, ValueError):
            return
        if "vllm" in cmdline and " serve " in cmdline:
            _log(f"stopping a vLLM server left by an earlier process (pid {pid})")
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
                tail = self.vllm_log.read_text(errors="replace")[-4000:]
                raise RuntimeError(f"vllm serve exited with {self.proc.returncode}:\n{tail}")
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception:  # noqa: BLE001 - not up yet
                pass
            if time.time() - t0 > self.ready_timeout_s:
                raise RuntimeError(f"vllm serve not ready after {self.ready_timeout_s} s; log {self.vllm_log}")
            time.sleep(3)

    # ---------------------------------------------------------------- request

    def _ask(self, audio: bytes, prompt: str, timestamps: bool, max_tokens: int,
             temperature: float = 0.0) -> Dict[str, Any]:
        b64 = base64.b64encode(audio).decode()
        body: Dict[str, Any] = {
            "model": self.served_name,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "audio_url", "audio_url": {"url": f"data:{_mime(audio)};base64,{b64}"}},
            ]}],
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "skip_special_tokens": False,  # keep speaker and event tags
            "spaces_between_special_tokens": False,
        }
        if timestamps:
            body["mm_processor_kwargs"] = {"capture_ctc_timestamps": True}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:2000]
            if 400 <= e.code < 500:
                raise InvalidInput(f"vLLM rejected the request ({e.code}): {detail}")
            raise RuntimeError(f"vLLM error {e.code}: {detail}")

    def process(self, job: Job) -> Result:
        inp = job.input or {}
        fields: Dict[str, Any] = {}
        if isinstance(inp.get("raw_body_b64"), str) and str(inp.get("raw_body_content_type", "")).startswith("multipart/"):
            fields = _multipart_fields(base64.b64decode(inp["raw_body_b64"]), inp["raw_body_content_type"])
            audio = fields.get("file") or b""
        else:
            audio = job.audio_bytes()
        if not audio:
            raise InvalidInput("no audio: send it as the multipart 'file' field or as 'audio_b64'")

        def get(key: str, default: Any = None) -> Any:
            return inp.get(key, fields.get(key, default))

        style = str(get("style") or "").strip().lower()
        prompt = str(get("prompt") or "").strip()
        if not style and prompt.lower() in STYLES:
            style, prompt = prompt.lower(), ""
        language = str(get("language") or "English")
        if not prompt:
            style = style or "verbatim"
            if style not in STYLES:
                raise InvalidInput(f"unknown style {style!r}; use one of {sorted(STYLES)} or send a prompt")
            prompt = STYLES[style].format(language=language)
        else:
            style = style or "custom"
        grans = get("timestamp_granularities") or []
        if isinstance(grans, str):
            grans = [grans]
        ts_raw = get("timestamps", False)
        timestamps = (str(ts_raw).lower() in ("1", "true", "yes", "word")) or "word" in grans
        max_tokens = int(get("max_tokens") or 16384)
        temperature = float(get("temperature") or 0.0)

        tmp = Path(tempfile.mkstemp(suffix=".audio")[1])
        try:
            tmp.write_bytes(audio)
            duration = _duration(tmp)
        finally:
            tmp.unlink(missing_ok=True)

        with self.lock:
            t0 = time.perf_counter()
            out = self._ask(audio, prompt, timestamps, max_tokens, temperature)
            seconds = time.perf_counter() - t0
        choice = (out.get("choices") or [{}])[0]
        text = cleanup_generation_loops((choice.get("message") or {}).get("content") or "").strip()
        usage = out.get("usage") or {}
        result: Dict[str, Any] = {
            "text": text,
            "style": style,
            "prompt": prompt,
            "duration": round(duration, 3) if duration else None,
            "seconds": round(seconds, 3),
            "rtfx": round(duration / seconds, 1) if duration and seconds > 0 else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "model": self.hf_repo,
        }
        if timestamps:
            result["ctc_timestamps"] = out.get("ctc_timestamps")
        return Result(result)

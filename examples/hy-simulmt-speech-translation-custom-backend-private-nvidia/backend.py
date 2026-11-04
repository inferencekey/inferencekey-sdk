"""Tencent Hy-SimulMT as an InferenceKey custom backend (simultaneous speech translation).

Hy-SimulMT-30B-A3B (Tencent Hunyuan, Apache-2.0) is a speech LLM for
simultaneous translation between Chinese and English: an audio tower in front
of a Hunyuan v3 mixture-of-experts language model (30B total, about 3B active
per token). It listens in 1.6 s blocks (then 0.8 s or 1.6 s) and, after each
block, either waits or writes the next piece of the translation, so the
listener gets the text while the speaker is still talking. The ``Interleave``
checkpoints also write the source transcript next to the translation. An
optional Qwen3-TTS sidecar reads the translation aloud in a clone of the
speaker's voice.

The authors serve it with stock vLLM (>= 0.31) plus their ``simulst`` plugin,
and put **SimulInfer** in front of it: the service that cuts the audio into
blocks, runs voice activity detection, keeps the conversation context and
streams the results (``POST /translate/stream`` and ``/translate/ws``). No
built-in InferenceKey backend runs that stack, so this backend does:

``setup()`` (once per process, cached on disk)
    1. builds a Python 3.12 environment with uv: vLLM, simulst and SimulInfer
       (pinned versions below),
    2. downloads the checkpoint (FP8 Interleave by default: 32 GB) to the
       local disk and writes the authors' inference template into it
       (``simulinfer-prepare-model``),
    3. starts ``vllm serve`` with the authors' launcher and flags, then the
       SimulInfer service on loopback ports,
    4. optionally (``tts=true``) builds a second environment with vLLM-Omni
       and starts Qwen3-TTS-12Hz-0.6B-Base for voice cloning.

``process(job)`` (per request) streams one recording through SimulInfer and
returns every partial result with the time it arrived.

Callers
-------
* ``POST .../v1/audio/transcriptions`` (multipart: ``file`` and optionally
  ``direction`` = ``zh-en`` | ``en-zh``, or ``language`` = source language
  ``zh`` | ``en``; ``transport`` = ``realtime`` | ``file``; ``tts`` = true).
* a JSON body ``{"audio_b64": ..., "direction": ..., "transport": ..., "tts": ...}``.

``transport=realtime`` (default) sends the audio to SimulInfer's WebSocket at
the speed it was spoken (100 ms packets), the way a microphone would, so the
timings are what a live listener sees. ``transport=file`` uploads the whole
file to ``/translate/stream`` and SimulInfer feeds it as fast as the GPU goes.

Output: ``{"text" (translation), "source_text", "direction", "transport",
"audio_s", "wall_s", "first_text_ms", "first_text_audio_ms",
"delay_ms" {median, p90, max} (realtime: how long after a block was spoken its
result arrived), "events" [{t_ms, start_ms, end_ms, action, text,
source_text}], "complete", "tts" {first_audio_ms, seconds, sample_rate,
wav_b64} when asked}``.

Knobs (workload ``config`` keys or ``HYSIMULMT_*`` environment variables):
``hf_repo`` / ``hf_revision`` (checkpoint), ``vllm_version`` (0.32.0),
``gpu_memory_utilization`` (0.78), ``max_model_len`` (8192),
``max_num_seqs`` (4), ``extra_vllm_args``, ``tts`` (false), ``tts_model``,
``tts_gpu_memory_utilization`` (0.15), ``ready_timeout_s`` (2400),
``cache_dir``. ``gpu_memory_utilization`` is also how we emulated smaller cards.

Attribution: Hy-SimulMT by Tencent Hunyuan, Apache-2.0
(https://huggingface.co/tencent/Hy-SimulMT-30B-A3B,
https://github.com/Tencent-Hunyuan/Hy-SimulMT, SimulInfer Apache-2.0);
simulst (Apache-2.0); vLLM and vLLM-Omni (Apache-2.0); Qwen3-TTS by the Qwen
team (Apache-2.0).
"""

from __future__ import annotations

import atexit
import base64
import email
import io
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

REPO = "Tencent-Hunyuan/Hy-SimulMT"


def _log(msg: str) -> None:
    print(f"[hy-simulmt] {msg}", file=sys.stderr, flush=True)


def _multipart_fields(raw: bytes, content_type: str) -> Dict[str, Any]:
    """Text fields and the file of a multipart body (OpenAI transcription shape)."""
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
            out["filename"] = part.get_filename() or "audio"
        else:
            out[name] = payload.decode("utf-8", "replace")
    return out


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _child_setup() -> None:
    """In each server child before exec: its own process group, and SIGTERM when this
    backend process dies (the worker stops a custom backend with SIGKILL)."""
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
    req = urllib.request.Request(url, headers={"User-Agent": "inferencekey-hy-simulmt-backend"})
    with urllib.request.urlopen(req, timeout=900) as resp:
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


def _pcm16_mono_16k(audio: bytes) -> bytes:
    """Decode wav/flac/ogg/mp3 to 16 kHz mono PCM16, what SimulInfer's WebSocket takes by default."""
    import numpy as np
    import soundfile as sf
    import soxr

    data, rate = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != 16000:
        mono = soxr.resample(mono, rate, 16000)
    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class HySimulMTBackend(CustomBackend):
    name = "hy-simulmt"
    version = "0.1.2"
    task_type = "audio2text"
    requirements = "requirements.txt"
    requires_python = ">=3.12,<3.13"
    description = "Tencent Hy-SimulMT (vLLM + simulst + SimulInfer): Chinese-English simultaneous speech translation"

    # Pinned upstream: what this example was tested with (2026-11-04).
    simulmt_commit = "7efcb18c521a7025b37d4bc2230a1df8ade83950"
    simulst_version = "0.1.0"
    vllm_version = "0.32.0"
    # SimulInfer's voice-clone launcher imports Qwen3TTSSpeakerEncoder from the Qwen3-TTS talker
    # module, which vLLM-Omni moved after 0.31: 0.32.0 and 0.32.0rc1 fail with ImportError.
    tts_vllm_version = "0.31.0"
    vllm_omni_version = "0.31.0rc1"
    hf_repo = "tencent/Hy-SimulMT-30B-A3B-Interleave-FP8"
    hf_revision = "7f37cb76119dbdf06db69d00d73e75d264f6aa85"
    # 0.78 of a 48 GB card is about 35 GiB: the FP8 weights (about 30 GiB) plus the KV cache.
    gpu_memory_utilization = 0.78
    max_model_len = 8192
    max_num_seqs = 4
    # The authors' launcher uses 16384; vLLM then profiles the audio encoder with 3,276 dummy
    # clips, which on a 48 GB card threw thousands of out-of-memory retries before it recovered.
    max_num_batched_tokens = 4096
    extra_vllm_args = ""
    tts = False
    tts_model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    tts_gpu_memory_utilization = 0.15
    ready_timeout_s = 2400.0
    request_timeout_s = 1200.0
    cache_dir = ""

    # ------------------------------------------------------------------ setup

    def _setting(self, ctx: BackendContext, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"HYSIMULMT_{key.upper()}")
        if raw is None:
            raw = (ctx.config or {}).get(key)
        if raw is None:
            return default
        if isinstance(default, bool):
            return _truthy(raw)
        return type(default)(raw)

    def setup(self, ctx: BackendContext) -> None:
        for key in ("simulmt_commit", "simulst_version", "vllm_version", "tts_vllm_version", "vllm_omni_version", "hf_repo",
                    "hf_revision", "gpu_memory_utilization", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
                    "extra_vllm_args",
                    "tts", "tts_model", "tts_gpu_memory_utilization", "ready_timeout_s", "request_timeout_s",
                    "cache_dir"):
            setattr(self, key, self._setting(ctx, key))
        cache = Path(self.cache_dir or Path.home() / ".cache" / "hy-simulmt-backend")
        cache.mkdir(parents=True, exist_ok=True)
        self.cache = cache
        self.timings: Dict[str, float] = {}
        self.procs: List[subprocess.Popen] = []
        t_setup = time.time()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        atexit.register(self._stop_servers)
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous = signal.getsignal(sig)

            def _on_signal(signum, frame, previous=previous):
                self._stop_servers()
                if callable(previous):
                    previous(signum, frame)
                else:
                    raise SystemExit(128 + signum)

            try:
                signal.signal(sig, _on_signal)
            except ValueError:  # not the main thread
                pass
        self._kill_stale()

        # 1. Source (SimulInfer) and the model environment.
        t0 = time.time()
        src = cache / f"hy-simulmt-{self.simulmt_commit[:12]}"
        if not (src / "SimulInfer" / "pyproject.toml").is_file():
            _fetch_tarball(f"https://codeload.github.com/{REPO}/tar.gz/{self.simulmt_commit}", src)
            for heavy in ("assets", "SimulTrain"):  # demo videos and training code: not needed to serve
                shutil.rmtree(src / heavy, ignore_errors=True)
        self.simulinfer_src = src / "SimulInfer"
        self.model_env = self._env(cache / "env-model", [
            f"vllm=={self.vllm_version}", f"simulst[audio]=={self.simulst_version}", f"{self.simulinfer_src}[vad]"])
        self.timings["env_s"] = round(time.time() - t0, 1)

        # 2. The checkpoint, on the local disk, with the authors' inference template.
        from huggingface_hub import snapshot_download

        t0 = time.time()
        model_dir = Path(snapshot_download(self.hf_repo, revision=self.hf_revision,
                                           local_dir=str(cache / "models" / self.hf_repo.split("/")[-1]),
                                           ignore_patterns=["assets/*", "*.mp4", "*.png"]))
        self.timings["download_s"] = round(time.time() - t0, 1)
        self._run([str(self.model_env / "bin" / "simulinfer-prepare-model"), "--model", str(model_dir)])
        # SimulInfer reads the output format (translation only, or source + translation) from
        # export_manifest.json or from a known checkpoint name. The FP8 and GGUF repos ship
        # neither (2026-11-04), so declare it: the Interleave checkpoints write both.
        manifest = model_dir / "export_manifest.json"
        if not manifest.is_file():
            enabled = "interleave" in self.hf_repo.lower()
            manifest.write_text(json.dumps({"source_target_format": {
                "enabled": enabled, "source_token": "<\uff5chy_place\u2581holder\u2581no\u2581790\uff5c>",
                "target_token": "<\uff5chy_place\u2581holder\u2581no\u2581791\uff5c>"}}))
        _log(f"environment in {self.timings['env_s']} s, weights in {self.timings['download_s']} s")

        # 3. vLLM with the authors' launcher (prints the exact command), then SimulInfer.
        t0 = time.time()
        self.vllm_port = _free_port()
        launch = [str(self.model_env / "bin" / "python"), "-m", "simulinfer.launch", "--model-dir", str(model_dir),
                  "--host", "127.0.0.1", "--port", str(self.vllm_port),
                  "--gpu-memory-utilization", str(self.gpu_memory_utilization),
                  "--max-model-len", str(self.max_model_len), "--max-num-seqs", str(self.max_num_seqs),
                  "--max-num-batched-tokens", str(self.max_num_batched_tokens), "--dry-run"]
        printed = subprocess.run(launch, capture_output=True, text=True, check=True, env=self._env_vars(self.model_env))
        line = printed.stdout.strip().splitlines()[-1]
        runner, _, command = line.partition(" ")
        import shlex

        cmd = shlex.split(command) + (self.extra_vllm_args.split() if self.extra_vllm_args.strip() else [])
        env = self._env_vars(self.model_env)
        env[runner.split("=", 1)[0]] = runner.split("=", 1)[1]
        self._start("vllm", cmd, env)
        self._wait_http(f"http://127.0.0.1:{self.vllm_port}/v1/models", "vllm", t0)
        self.timings["vllm_start_s"] = round(time.time() - t0, 1)

        tts_url = ""
        if self.tts:
            t0 = time.time()
            tts_env = self._env(cache / "env-tts", [
                f"vllm=={self.tts_vllm_version}", f"vllm-omni=={self.vllm_omni_version}", str(self.simulinfer_src)])
            self.timings["tts_env_s"] = round(time.time() - t0, 1)
            t0 = time.time()
            self.tts_port = _free_port()
            self._start("tts", [str(tts_env / "bin" / "simulinfer-tts-omni"), "--model", self.tts_model,
                                "--host", "127.0.0.1", "--port", str(self.tts_port),
                                "--gpu-memory-utilization", str(self.tts_gpu_memory_utilization)],
                        self._env_vars(tts_env))
            self._wait_http(f"http://127.0.0.1:{self.tts_port}/v1/models", "tts", t0)
            self.timings["tts_start_s"] = round(time.time() - t0, 1)
            tts_url = f"http://127.0.0.1:{self.tts_port}/v1"

        t0 = time.time()
        self.app_port = _free_port()
        config = cache / f"simulinfer-{self.app_port}.toml"
        config.write_text(self._config(tts_url))
        self._start("simulinfer", [str(self.model_env / "bin" / "simulinfer"), "--config", str(config)],
                    self._env_vars(self.model_env))
        self._wait_http(f"http://127.0.0.1:{self.app_port}/health", "simulinfer", t0)
        with urllib.request.urlopen(f"http://127.0.0.1:{self.app_port}/health", timeout=10) as r:
            self.health = json.loads(r.read())
        self.timings["simulinfer_start_s"] = round(time.time() - t0, 1)
        self.timings["setup_s"] = round(time.time() - t_setup, 1)
        self.lock = threading.Lock()
        _log(f"ready: {self.hf_repo}; health {self.health}; timings {self.timings}")

    def _env_vars(self, venv: Path) -> Dict[str, str]:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update({"VIRTUAL_ENV": str(venv), "PATH": f"{venv / 'bin'}:{env.get('PATH', '')}"})
        return env

    def _run(self, args: List[str], env: Optional[Dict[str, str]] = None) -> None:
        _log("$ " + " ".join(args))
        subprocess.run(args, env=env, check=True)

    def _env(self, venv: Path, packages: List[str]) -> Path:
        done = venv / ".inferencekey-ready"
        stamp = json.dumps(packages)
        if done.is_file() and done.read_text() == stamp:
            return venv
        uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
        env = dict(os.environ, UV_LINK_MODE="copy", UV_HTTP_TIMEOUT="300")
        env.pop("VIRTUAL_ENV", None)
        self._run([uv, "venv", "--clear", "--python", "3.12", str(venv)], env)
        self._run([uv, "pip", "install", "--python", str(venv / "bin" / "python"), *packages], env)
        done.write_text(stamp)
        return venv

    def _config(self, tts_url: str) -> str:
        """The authors' default SimulInfer configuration, pointed at our loopback servers."""
        text = (self.simulinfer_src / "simulinfer" / "default_config.toml").read_text()
        text = text.replace('port = 8080', f'port = {self.app_port}', 1)
        text = text.replace('base_url = "http://127.0.0.1:8800/v1"', f'base_url = "http://127.0.0.1:{self.vllm_port}/v1"', 1)
        if tts_url:
            head, _, _ = text.partition("[tts]")
            text = head + "\n".join([
                "[tts]", 'tts_backend = "vllm_omni"', f'tts_url = "{tts_url}"', f'tts_model = "{self.tts_model}"',
                'tts_task_type = "Base"', "tts_sample_rate = 24000", 'tts_history_mode = "paired"',
                "tts_history_window_s = 10", "tts_clone_source = true", ""])
        return text

    def _start(self, name: str, cmd: List[str], env: Dict[str, str]) -> None:
        log_path = self.cache / f"{name}.log"
        _log(f"starting {name}: " + " ".join(cmd))
        log = open(log_path, "ab")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(self.cache),
                                preexec_fn=_child_setup)
        proc.name, proc.log_path = name, log_path  # type: ignore[attr-defined]
        self.procs.append(proc)
        with open(self.cache / "servers.pid", "a") as f:
            f.write(f"{proc.pid}\n")

    def _wait_http(self, url: str, name: str, t0: float) -> None:
        proc = next(p for p in self.procs if p.name == name)  # type: ignore[attr-defined]
        while True:
            if proc.poll() is not None:
                tail = proc.log_path.read_text(errors="replace")[-4000:]  # type: ignore[attr-defined]
                raise RuntimeError(f"{name} exited with {proc.returncode}:\n{tail}")
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception:  # noqa: BLE001 - not up yet
                pass
            if time.time() - t0 > self.ready_timeout_s:
                raise RuntimeError(f"{name} not ready after {self.ready_timeout_s} s; log {proc.log_path}")  # type: ignore[attr-defined]
            time.sleep(3)

    def _kill_stale(self) -> None:
        """Servers left by an earlier backend process (stopped with SIGKILL) still hold the GPU."""
        pidfile = self.cache / "servers.pid"
        if not pidfile.is_file():
            return
        for raw in pidfile.read_text().split():
            try:
                pid = int(raw)
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            except (OSError, ValueError):
                continue
            if any(k in cmdline for k in ("vllm", "simulinfer")):
                _log(f"stopping a server left by an earlier process (pid {pid})")
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
        pidfile.unlink(missing_ok=True)

    def _stop_servers(self) -> None:
        for proc in reversed(getattr(self, "procs", [])):
            if proc.poll() is not None:
                continue
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass

    # ---------------------------------------------------------------- request

    def _realtime(self, audio: bytes, direction: str, tts: bool) -> Dict[str, Any]:
        """Send the audio over SimulInfer's WebSocket at speaking speed; time every result."""
        from websockets.sync.client import connect

        pcm = _pcm16_mono_16k(audio)
        events: List[Dict[str, Any]] = []
        tts_pcm = bytearray()
        tts_rate = 24000
        first_audio_ms: Optional[float] = None
        with connect(f"ws://127.0.0.1:{self.app_port}/translate/ws", max_size=None,
                     open_timeout=30, close_timeout=10) as ws:
            ws.send(json.dumps({"direction": direction, "input_format": "s16le", "input_sample_rate": 16000,
                                "input_channels": 1, "tts": tts, "audio_transport": "base64"}))
            ready = json.loads(ws.recv(timeout=60))
            if ready.get("event") == "error":
                raise InvalidInput(f"SimulInfer refused the session: {ready.get('detail')}")
            t0 = time.perf_counter()
            stop = threading.Event()

            def sender() -> None:
                step = 3200  # 100 ms of 16 kHz PCM16
                for i, pos in enumerate(range(0, len(pcm), step)):
                    delay = t0 + i * 0.1 - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    if stop.is_set():
                        return
                    ws.send(pcm[pos:pos + step])
                ws.send(json.dumps({"eof": True}))

            th = threading.Thread(target=sender, daemon=True)
            th.start()
            try:
                while True:
                    msg = ws.recv(timeout=self.request_timeout_s)
                    t_ms = round((time.perf_counter() - t0) * 1000, 1)
                    if isinstance(msg, bytes):
                        continue
                    ev = json.loads(msg)
                    kind = ev.get("event")
                    if kind == "audio":
                        if first_audio_ms is None:
                            first_audio_ms = t_ms
                        tts_rate = int(ev.get("sample_rate") or tts_rate)
                        tts_pcm.extend(base64.b64decode(ev.get("data") or ""))
                        continue
                    ev["t_ms"] = t_ms
                    events.append(ev)
                    if kind in ("done", "error"):
                        break
            finally:
                stop.set()
                th.join(timeout=5)
        out = self._summary(events, len(pcm) / 32000.0, "realtime", direction)
        if tts:
            out["tts"] = {"first_audio_ms": first_audio_ms, "seconds": round(len(tts_pcm) / 2 / tts_rate, 2),
                          "sample_rate": tts_rate, "wav_b64": base64.b64encode(_wav(bytes(tts_pcm), tts_rate)).decode()}
        return out

    def _file(self, audio: bytes, filename: str, direction: str) -> Dict[str, Any]:
        """Upload the file to /translate/stream (fed as fast as the GPU goes); time every SSE event."""
        boundary = uuid.uuid4().hex
        parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"direction\"\r\n\r\n{direction}\r\n".encode(),
                 f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                 f"Content-Type: application/octet-stream\r\n\r\n".encode() + audio + b"\r\n",
                 f"--{boundary}--\r\n".encode()]
        req = urllib.request.Request(f"http://127.0.0.1:{self.app_port}/translate/stream", data=b"".join(parts),
                                     method="POST", headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        events: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    ev = json.loads(line[5:])
                    ev["t_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    events.append(ev)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:1000]
            if 400 <= e.code < 500:
                raise InvalidInput(f"SimulInfer rejected the audio ({e.code}): {detail}")
            raise RuntimeError(f"SimulInfer error {e.code}: {detail}")
        done = next((e for e in events if e.get("event") == "done"), {})
        audio_s = float(done.get("duration_ms") or 0) / 1000.0
        return self._summary(events, audio_s, "file", direction)

    @staticmethod
    def _summary(events: List[Dict[str, Any]], audio_s: float, transport: str, direction: str) -> Dict[str, Any]:
        err = next((e for e in events if e.get("event") == "error"), None)
        if err:
            raise RuntimeError(f"SimulInfer error: {err.get('detail')}")
        done = next((e for e in events if e.get("event") == "done"), {})
        chunks = [e for e in events if e.get("event") in ("chunk", "flush")]
        texts = [e for e in chunks if (e.get("text") or "").strip()]
        delays = [e["t_ms"] - e["end_ms"] for e in chunks if e.get("end_ms") is not None]
        compact = [{k: e.get(k) for k in ("t_ms", "event", "start_ms", "end_ms", "action", "text", "source_text")
                    if e.get(k) not in (None, "")} for e in events if e.get("event") in ("chunk", "flush", "done")]

        def pct(values: List[float], q: float) -> Optional[float]:
            if not values:
                return None
            ordered = sorted(values)
            return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1)

        return {
            "text": (done.get("full_text") or "".join(e.get("text") or "" for e in chunks)).strip(),
            "source_text": (done.get("source_text") or "").strip(),
            "direction": direction,
            "transport": transport,
            "complete": done.get("complete"),
            "audio_s": round(audio_s, 2),
            "wall_s": round((events[-1]["t_ms"] if events else 0) / 1000.0, 2),
            "chunks": len(chunks),
            "first_text_ms": texts[0]["t_ms"] if texts else None,
            "first_text_audio_ms": texts[0].get("end_ms") if texts else None,
            "delay_ms": {"median": round(statistics.median(delays), 1) if delays else None,
                         "p90": pct(delays, 0.9), "max": round(max(delays), 1) if delays else None},
            "events": compact,
        }

    def process(self, job: Job) -> Result:
        inp = job.input or {}
        fields: Dict[str, Any] = {}
        filename = "audio"
        if isinstance(inp.get("raw_body_b64"), str) and str(inp.get("raw_body_content_type", "")).startswith("multipart/"):
            fields = _multipart_fields(base64.b64decode(inp["raw_body_b64"]), inp["raw_body_content_type"])
            audio = fields.get("file") or b""
            filename = fields.get("filename") or filename
        else:
            audio = job.audio_bytes()
        if not audio:
            raise InvalidInput("no audio: send it as the multipart 'file' field or as 'audio_b64'")

        def get(key: str, default: Any = None) -> Any:
            value = inp.get(key, fields.get(key))
            return default if value in (None, "") else value

        direction = str(get("direction") or "").strip().lower()
        if not direction:
            language = str(get("language", "zh")).strip().lower()
            direction = "en-zh" if language.startswith("en") else "zh-en"
        if direction not in ("zh-en", "en-zh"):
            raise InvalidInput(f"direction must be zh-en or en-zh, got {direction!r}")
        transport = str(get("transport", "realtime")).strip().lower()
        if transport not in ("realtime", "file"):
            raise InvalidInput(f"transport must be realtime or file, got {transport!r}")
        tts = _truthy(get("tts", False))
        if tts and not self.health.get("tts_available"):
            raise InvalidInput("tts=true needs the workload started with the tts knob (HYSIMULMT_TTS=1)")
        if tts and transport != "realtime":
            raise InvalidInput("tts needs transport=realtime")

        with self.lock:
            if transport == "realtime":
                out = self._realtime(audio, direction, tts)
            else:
                out = self._file(audio, filename, direction)
        out["model"] = self.hf_repo
        return Result(out)

"""ZONOS3-Mini (Zyphra) text-to-speech, served by SGLang-Omni, as an InferenceKey custom backend.

ZONOS3-Mini is Zyphra's Apache-2.0 text-to-speech model: an 11-layer hybrid
Gated DeltaNet / attention mixture-of-experts backbone (800M parameters in all,
about 100M active per token) that writes 15 Hz codec frames, rendered to
44.1 kHz audio by a causal iSTFT head as they are produced. It speaks 43
languages and clones a voice from a short clip with no transcript.

The upstream server for NVIDIA GPUs is SGLang-Omni (``sgl-omni serve``), which
answers the OpenAI-compatible ``/v1/audio/speech`` route. No built-in
InferenceKey backend runs SGLang-Omni, so this backend wraps it:

``setup()`` (once per process)
    1. builds a separate Python 3.12 virtualenv with SGLang-Omni at a pinned
       commit (it pins torch 2.14.1, SGLang 0.5.23 and CUDA 13 wheels, which
       need pre-release resolution, so it does not share the backend's venv),
    2. downloads, in parallel with step 1, only the checkpoint files SGLang-Omni
       reads (about 14.5 GB: the language model, the codec generator, the
       iSTFT head, the w2v-BERT teacher and the registries) plus the 27 preset
       voice clips, at a pinned Hugging Face revision, to the local disk,
    3. starts ``sgl-omni serve`` on a loopback port and waits for ``/health``.

``process(job)`` (per request, thread-safe: requests run concurrently)
    forwards the text and an optional reference clip to the local server,
    streams the PCM back to measure the time to first audio, and returns
    ``{"audio_b64", "format": "wav", "sample_rate", "seconds", "timings", ...}``.

Request shapes accepted (options can sit at the top level or in ``extra_body``):

* OpenAI TTS (``/v1/audio/speech``, the body arrives in ``audio_body``):
  ``input`` is the text, ``voice`` a preset voice name (``Imogen_en-British_F_1``
  ...) or ``default``, ``ref_audio`` a base64 clip (or data URI) to clone.
* OpenAI chat: the last user message carries the text and, for cloning, an
  ``input_audio`` part ``{"data": <base64>, "format": "wav"}``.
* Plain fields: ``text`` and ``audio_b64``.

Options: ``seed``, ``temperature``, ``top_k``, ``top_p``, ``repetition_penalty``,
``max_new_tokens``, ``condition_values`` (``{"speaking_rate": <UTF-8 bytes per
second>, "loudness": <LUFS>, "fmax": <Hz>}``), ``stream`` (default true: stream
from the server to measure the time to first audio; the reply is still one WAV).

Settings are class attributes; override them with ``ZONOS3_*`` environment
variables or the workload config (keys in lower case).
``engine_gpu_memory_fraction`` is the share of the card's memory the pipeline
may plan for; SGLang-Omni's default (0.3) is sized for an 80 GB H100, so a
24 GB or smaller card needs a larger share (see README).
"""

from __future__ import annotations

import atexit
import base64
import ctypes
import hashlib
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

SAMPLE_RATE = 44100
# The files SGLang-Omni's Zonos3 pipeline reads (sglang_omni/models/zonos3), and the preset voices.
ALLOW_PATTERNS = [
    "lm/zonos3_mini/*",
    "lm/zonos3_mini/**",
    "autoencoder/generator.pt",
    "autoencoder/heads/full_mr/*",
    "autoencoder/teacher/*",
    "registries/*",
    "gguf/voices/**",
]


def _log(msg: str) -> None:
    print(f"[zonos3-backend] {msg}", file=sys.stderr, flush=True)


def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """16-bit little-endian mono PCM -> WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _die_with_parent() -> None:
    """Runs in the server child before exec: SIGTERM it when the backend process dies (Linux)."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        pass


def _gpu_used_mib() -> Optional[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


class Zonos3MiniBackend(CustomBackend):
    name = "zonos3-mini-sglang-omni"
    version = "0.1.3"
    task_type = "text2audio"
    description = "ZONOS3-Mini (Zyphra, Apache-2.0): 43-language TTS and voice cloning, served by SGLang-Omni"

    # Pinned upstream: the SGLang-Omni commit and the Hugging Face revision this example was tested with.
    sglang_omni_commit = "bac76771a0aa0a8fe849dc158836c22dca542f81"
    hf_repo = "Zyphra/ZONOS3-Mini"
    hf_revision = "d2d142a86b1e6aeb4252679673c86ecd86177bf8"
    python_version = "3.12"
    # Share of the GPU's TOTAL memory the colocated pipeline process may use (weights of all stages
    # plus the language model's state pools). 0 = SGLang-Omni's default, 0.3, sized for an 80 GB H100:
    # on a 24 GB card it fails with "Colocated GPU memory budget leaves no KV-cache headroom".
    engine_gpu_memory_fraction = 0.0
    # 0 = SGLang-Omni's default. Otherwise SGLang's mem_fraction_static for the engine stage.
    mem_fraction_static = 0.0
    # Extra arguments for `sgl-omni serve`, split on spaces.
    extra_server_args = ""
    # How long setup() waits for the server's /health after starting it.
    server_start_timeout_s = 1500
    # Input guards.
    max_chars = 3000
    max_reference_seconds = 60.0

    # ------------------------------------------------------------------ setup

    def _setting(self, ctx: BackendContext, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"ZONOS3_{key.upper()}")
        if raw is None:
            raw = (ctx.config or {}).get(key)
        if raw is None:
            return default
        if isinstance(default, bool):
            return str(raw).lower() in ("1", "true", "yes", "on")
        return type(default)(raw)

    @staticmethod
    def _uv() -> str:
        found = shutil.which("uv")
        if found:
            return found
        from uv import find_uv_bin  # the `uv` wheel from requirements.txt

        return find_uv_bin()

    def _build_server_venv(self, cache: Path) -> Path:
        """Python 3.12 venv with SGLang-Omni at the pinned commit; reused when already complete."""
        venv = cache / f"sgl-omni-{self.sglang_omni_commit[:12]}"
        marker = venv / ".ik-install-complete"
        if marker.is_file():
            return venv
        uv = self._uv()
        env = {**os.environ, "UV_PYTHON_PREFERENCE": "managed", "UV_LINK_MODE": "copy"}
        subprocess.run([uv, "venv", "--clear", "-p", self.python_version, str(venv)], check=True, env=env)
        overrides = cache / "sgl-omni-overrides.txt"
        # SGLang-Omni's own [tool.uv] override, which uv applies only when installing the project itself.
        overrides.write_text("protobuf>=6.31.1,<7.0.0\n")
        spec = f"sglang-omni @ https://github.com/sgl-project/sglang-omni/archive/{self.sglang_omni_commit}.tar.gz"
        subprocess.run([uv, "pip", "install", "--python", str(venv / "bin" / "python"), "--prerelease=allow",
                        "--override", str(overrides), spec], check=True, env=env)
        marker.write_text(spec + "\n")
        return venv

    def _download(self, cache: Path) -> Path:
        from huggingface_hub import snapshot_download

        local = cache / f"weights-{self.hf_revision[:12]}"
        snapshot_download(self.hf_repo, revision=self.hf_revision, allow_patterns=ALLOW_PATTERNS,
                          local_dir=str(local), max_workers=16)
        return local

    def _load_voices(self, weights: Path) -> Dict[str, Path]:
        voices: Dict[str, Path] = {}
        for p in sorted((weights / "gguf" / "voices").rglob("*")):
            if p.suffix.lower() in (".wav", ".mp3", ".ogg", ".flac"):
                voices[p.stem] = p
        return voices

    def _wait_healthy(self, deadline: float) -> None:
        url = f"http://127.0.0.1:{self.port}/health"
        while time.time() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(f"sgl-omni serve exited with code {self.server.returncode}; "
                                   f"last log lines:\n{self._log_tail()}")
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    status = json.loads(r.read().decode(errors="replace") or "{}").get("status")
                    if r.status == 200 and status == "healthy":
                        return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        raise RuntimeError(f"sgl-omni serve not healthy after {self.server_start_timeout_s} s; "
                           f"last log lines:\n{self._log_tail()}")

    def _log_tail(self, n: int = 40) -> str:
        try:
            return "\n".join(self.server_log.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def setup(self, ctx: BackendContext) -> None:
        for key in ("sglang_omni_commit", "hf_repo", "hf_revision", "python_version", "engine_gpu_memory_fraction", "mem_fraction_static",
                    "extra_server_args", "server_start_timeout_s", "max_chars", "max_reference_seconds"):
            setattr(self, key, self._setting(ctx, key))
        # Keep everything on the local disk: weights on a network volume load many times slower.
        cache = Path(os.environ.get("ZONOS3_CACHE_DIR", Path.home() / ".cache" / "zonos3-backend"))
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        self.timings: Dict[str, Any] = {}
        t_setup = time.time()

        # The install (a few GB of wheels) and the weights (about 14.5 GB) download in parallel.
        result: Dict[str, Any] = {}

        def download() -> None:
            t = time.time()
            try:
                result["weights"] = self._download(cache)
            except Exception as e:  # noqa: BLE001
                result["download_error"] = e
            result["download_s"] = round(time.time() - t, 1)

        dl = threading.Thread(target=download, daemon=True)
        dl.start()
        t0 = time.time()
        venv = self._build_server_venv(cache)
        self.timings["install_s"] = round(time.time() - t0, 1)
        dl.join()
        if "download_error" in result:
            raise RuntimeError(f"weights download failed: {result['download_error']}")
        weights: Path = result["weights"]
        self.timings["download_s"] = result["download_s"]
        self.voices = self._load_voices(weights)
        _log(f"server venv in {self.timings['install_s']} s, weights in {self.timings['download_s']} s "
             f"({weights}), {len(self.voices)} preset voices")

        self.refs_dir = cache / "references"
        self.refs_dir.mkdir(exist_ok=True)
        self.port = _free_port()
        args = [str(venv / "bin" / "sgl-omni"), "serve", "--model-path", str(weights),
                "--host", "127.0.0.1", "--port", str(self.port), "--model-name", "zonos3-mini"]
        if self.engine_gpu_memory_fraction > 0:
            args += ["--tts_engine.gpu_memory_fraction", str(self.engine_gpu_memory_fraction)]
        if self.mem_fraction_static > 0:
            args += ["--mem-fraction-static", str(self.mem_fraction_static)]
        if self.extra_server_args.strip():
            args += self.extra_server_args.split()
        logs = cache / "logs"
        logs.mkdir(exist_ok=True)
        self.server_log = logs / f"sgl-omni-{time.strftime('%Y%m%dT%H%M%S')}-{self.port}.log"
        _log(f"starting: {' '.join(args)} (log {self.server_log})")
        t0 = time.time()
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
        env["PATH"] = f"{venv / 'bin'}:{env.get('PATH', '')}"
        # Own process group (so teardown can stop the server and its stage processes together), and
        # a parent-death signal: when the worker stops this backend, the server goes with it.
        self.server = subprocess.Popen(args, stdout=self.server_log.open("ab"), stderr=subprocess.STDOUT,
                                       env=env, cwd=str(cache), start_new_session=True,
                                       preexec_fn=_die_with_parent)
        atexit.register(self.teardown)
        previous = signal.getsignal(signal.SIGTERM)

        def on_term(signum, frame):  # the worker stops the backend with SIGTERM
            self.teardown()
            if callable(previous):
                previous(signum, frame)
            else:
                raise SystemExit(0)

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, on_term)
        self._wait_healthy(time.time() + self.server_start_timeout_s)
        self.timings["server_start_s"] = round(time.time() - t0, 1)
        self.timings["gpu_used_mib_after_start"] = _gpu_used_mib()

        t0 = time.time()
        self._synthesize({"text": "Ready.", "seed": 1, "stream": True})
        self.timings["warmup_s"] = round(time.time() - t0, 2)
        self.timings["setup_s"] = round(time.time() - t_setup, 1)
        _log(f"ZONOS3-Mini ready on 127.0.0.1:{self.port}: {self.timings}")

    def teardown(self) -> None:
        """Stop the sgl-omni server and its stage processes (the whole process group)."""
        server = getattr(self, "server", None)
        if server is None:
            return
        for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
            try:
                os.killpg(server.pid, sig)
            except (ProcessLookupError, PermissionError):
                return
            try:
                server.wait(timeout=wait)
                time.sleep(1)  # let the stage processes in the group exit too
            except subprocess.TimeoutExpired:
                continue

    # ---------------------------------------------------------------- inputs

    @staticmethod
    def _from_messages(messages: Any) -> Tuple[str, Optional[str]]:
        """Text and base64 audio from the last user message."""
        if not isinstance(messages, list):
            return "", None
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content, None
            texts: List[str] = []
            audio: Optional[str] = None
            for part in content or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
                elif part.get("type") == "input_audio":
                    audio = (part.get("input_audio") or {}).get("data")
                elif part.get("type") == "audio_url":
                    url = (part.get("audio_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        audio = url
            return " ".join(t for t in texts if t).strip(), audio
        return "", None

    def _parse(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        f: Dict[str, Any] = dict(inp)
        speech = f.get("audio_body") if isinstance(f.get("audio_body"), dict) else None
        if speech:  # OpenAI /v1/audio/speech body
            f = {**speech, **{k: v for k, v in f.items() if k != "audio_body"}}
        if isinstance(f.get("extra_body"), dict):
            f = {**f, **f["extra_body"]}
        text, audio = self._from_messages(f.get("messages"))
        text = str(f.get("input") or f.get("text") or text or "").strip()
        audio = f.get("ref_audio") or f.get("audio_b64") or audio
        if not text:
            raise InvalidInput("expected the text to say: 'input' (OpenAI TTS), 'text', or a user message")
        if len(text) > self.max_chars:
            raise InvalidInput(f"text is {len(text)} characters; the limit is {self.max_chars}")
        f["text"] = text
        f["ref_audio_b64"] = audio
        return f

    def _reference_path(self, f: Dict[str, Any]) -> Tuple[Optional[Path], Optional[str]]:
        """Reference clip on disk: the request's audio, or a preset voice. (path, label)."""
        audio = f.get("ref_audio_b64")
        if isinstance(audio, str) and audio:
            if audio.startswith("data:"):
                audio = audio.split(",", 1)[-1]
            try:
                raw = base64.b64decode(audio, validate=False)
            except Exception as e:  # noqa: BLE001
                raise InvalidInput(f"reference audio is not valid base64: {e}") from e
            if len(raw) < 1000:
                raise InvalidInput("reference audio is too short to be a clip")
            digest = hashlib.sha256(raw).hexdigest()[:24]
            ext = (".wav" if raw[:4] == b"RIFF" else ".flac" if raw[:4] == b"fLaC" else ".ogg" if raw[:4] == b"OggS"
                   else ".mp3" if raw[:3] == b"ID3" or raw[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2") else ".wav")
            path = self.refs_dir / f"{digest}{ext}"
            if not path.is_file():
                path.write_bytes(raw)
            return path, "uploaded"
        voice = str(f.get("voice") or "").strip()
        if voice and voice.lower() not in ("default", "alloy"):
            if voice not in self.voices:
                raise InvalidInput(f"unknown voice {voice!r}; preset voices: {', '.join(sorted(self.voices))}")
            return self.voices[voice], voice
        return None, None

    # ---------------------------------------------------------------- synth

    def _synthesize(self, f: Dict[str, Any]) -> Dict[str, Any]:
        import requests

        ref, ref_label = self._reference_path(f)
        stream = str(f.get("stream", True)).lower() not in ("0", "false", "no")
        body: Dict[str, Any] = {"input": f["text"], "response_format": "pcm" if stream else "wav", "stream": stream}
        for key in ("seed", "max_new_tokens", "top_k"):
            if f.get(key) is not None:
                body[key] = int(f[key])
        for key in ("temperature", "top_p", "repetition_penalty"):
            if f.get(key) is not None:
                body[key] = float(f[key])
        if isinstance(f.get("condition_values"), dict):
            body["condition_values"] = f["condition_values"]
        if ref is not None:
            body["references"] = [{"audio_path": str(ref)}]
        url = f"http://127.0.0.1:{self.port}/v1/audio/speech"
        t0 = time.time()
        ttfa = None
        with requests.post(url, json=body, stream=stream, timeout=900) as r:
            if r.status_code >= 400:
                msg = r.text[:500]
                if r.status_code < 500:
                    raise InvalidInput(f"SGLang-Omni rejected the request ({r.status_code}): {msg}")
                raise RuntimeError(f"SGLang-Omni error {r.status_code}: {msg}")
            if stream:
                sample_rate = int(r.headers.get("X-Sample-Rate") or r.headers.get("x-sample-rate") or SAMPLE_RATE)
                chunks: List[bytes] = []
                for chunk in r.iter_content(chunk_size=None):
                    if chunk:
                        if ttfa is None:
                            ttfa = time.time() - t0
                        chunks.append(chunk)
                pcm = b"".join(chunks)
                if len(pcm) % 2:
                    pcm = pcm[:-1]
                wav = _wav_bytes(pcm, sample_rate)
                seconds = len(pcm) / 2 / sample_rate
            else:
                wav = r.content
                with wave.open(io.BytesIO(wav)) as w:
                    sample_rate = w.getframerate()
                    seconds = w.getnframes() / float(sample_rate)
        synth_s = time.time() - t0
        return {"wav": wav, "sample_rate": sample_rate, "seconds": seconds, "synth_s": synth_s,
                "ttfa_s": ttfa, "stream": stream, "reference": ref_label}

    # ---------------------------------------------------------------- process

    def process(self, job: Job) -> Result:
        t0 = time.time()
        f = self._parse(job.input)
        out = self._synthesize(f)
        seconds = out["seconds"]
        return Result(output={
            "audio_b64": base64.b64encode(out["wav"]).decode(),
            "format": "wav",
            "sample_rate": out["sample_rate"],
            "seconds": round(seconds, 2),
            "text": f["text"],
            "reference": out["reference"],
            "timings": {
                "ttfa_s": round(out["ttfa_s"], 3) if out["ttfa_s"] is not None else None,
                "synth_s": round(out["synth_s"], 3),
                "total_s": round(time.time() - t0, 3),
                "realtime_factor": round(out["synth_s"] / seconds, 3) if seconds > 0 else None,
                "x_realtime": round(seconds / out["synth_s"], 2) if out["synth_s"] > 0 else None,
                "streamed": out["stream"],
            },
            "model": self.hf_repo,
            "server": f"sglang-omni@{self.sglang_omni_commit[:12]}",
        })

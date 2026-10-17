"""AuK-Flash (Tencent Hunyuan) as an InferenceKey custom backend.

AuK is a 1.5B diffusion transformer for speech generation AND editing: one
natural-language instruction picks the task (zero-shot voice cloning, TTS from a
voice description, replacing words inside a recording, pitch / speed / volume,
emotion, whisper, denoising, speaker separation...). AuK-Flash is the 4-step
distilled checkpoint (no CFG). No built-in backend serves it (it is neither an
LLM nor a vllm-omni TTS model), so this backend drives the authors' own PyTorch
inference code:

``setup()`` (once per process)
    1. downloads the AuK source at a pinned commit from GitHub (about 10 MB
       tarball, only ``src/auk`` is used) and puts it on ``sys.path``,
    2. downloads, in parallel, the AuK-Flash checkpoint + VAE (6.8 GB) and the
       Qwen2.5-Omni-3B encoder it conditions on (12 GB; only its "Thinker"
       half is loaded),
    3. builds ``AukInfer`` on the GPU (bf16 encoder, fp32 DiT).

``process(job)`` (per request)
    reads one instruction (text) and an optional input audio (base64 WAV),
    runs ``AukInfer.generate`` and returns ``{"audio_b64", "format",
    "sample_rate", "seconds", "timings", ...}`` (16-bit PCM WAV, 24 kHz mono).

Request shapes accepted (all fields can also sit in ``extra_body``):

* OpenAI chat (``/v1/chat/completions``): the last user message carries the
  instruction as text and, for editing / cloning, the source audio as an
  ``input_audio`` part ``{"data": <base64 wav>, "format": "wav"}``.
* Plain fields: ``instruction`` (or ``prompt``) and ``audio_b64``.
* OpenAI TTS (``/v1/audio/speech`` body in ``audio_body``): ``input`` is the
  text to say; ``instructions`` a voice description; ``ref_audio`` (base64) a
  voice to clone. The backend turns these into AuK's instruction templates.

Options: ``gen_seconds`` (output length), ``gen_text`` + ``ref_text`` (let the
backend estimate the length the way the AuK CLI does), ``seed``.

LICENCE NOTE: AuK's code and weights are MIT, but the Qwen2.5-Omni-3B encoder
it needs is under the Qwen RESEARCH licence (non-commercial). See the README.

Settings are class attributes; override them with ``AUK_*`` environment
variables or the runtime's ``--config-json`` (keys in lower case).
"""

from __future__ import annotations

import base64
import io
import math
import os
import re
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

AUK_REPO = "Tencent-Hunyuan/AuK"


def _log(msg: str) -> None:
    print(f"[auk-backend] {msg}", file=sys.stderr, flush=True)


def _wav_bytes(samples, sample_rate: int) -> bytes:
    """float tensor/array in [-1, 1] -> 16-bit PCM mono WAV bytes."""
    import numpy as np

    arr = samples.detach().cpu().float().numpy() if hasattr(samples, "detach") else np.asarray(samples)
    arr = arr.reshape(-1)
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _wav_seconds(path: str) -> float:
    import soundfile as sf

    info = sf.info(path)
    return info.frames / float(info.samplerate)


class AukFlashBackend(CustomBackend):
    name = "auk-flash-speech"
    version = "0.1.3"
    task_type = "text2audio"
    description = "AuK-Flash (Tencent Hunyuan, MIT): speech generation and editing from one instruction, 4 steps"

    # Pinned upstream: code commit and Hugging Face revisions (the ones this example was tested with).
    auk_commit = "9481474f107ee4d8dec959b3aed05b0627cebbf8"
    hf_repo = "tencent/AuK-Flash"
    hf_revision = "575b92f0895f75180bf2cbd35f2e176c5732b8ed"
    ckpt_file = "auk_flash.safetensors"
    qwen_repo = "Qwen/Qwen2.5-Omni-3B"
    qwen_revision = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
    dtype = "bf16"
    # Moves the Qwen encoder and the DiT to CPU RAM between uses (AuK's own
    # option). Needed on 16 GB cards; slower per request.
    cpu_offload = False
    # Output length guard: one request answers at most this many seconds of audio.
    max_gen_seconds = 30.0
    # Length estimate for text-only requests when the caller gives no gen_seconds.
    chars_per_second = 14.0

    # ------------------------------------------------------------------ setup

    def _setting(self, ctx: BackendContext, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"AUK_{key.upper()}")
        if raw is None:
            raw = (ctx.config or {}).get(key)
        if raw is None:
            return default
        if isinstance(default, bool):
            return str(raw).lower() in ("1", "true", "yes", "on")
        return type(default)(raw)

    def _fetch_source(self, cache: Path) -> Path:
        src = cache / f"auk-{self.auk_commit[:12]}" / "src"
        if (src / "auk" / "infer" / "infer_auk.py").is_file():
            return src
        url = f"https://codeload.github.com/{AUK_REPO}/tar.gz/{self.auk_commit}"
        cache.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=cache, prefix=".src-"))
        req = urllib.request.Request(url, headers={"User-Agent": "inferencekey-auk-backend"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            members = [m for m in tf.getmembers() if "/src/auk/" in m.name or m.name.endswith("/src")]
            try:
                tf.extractall(tmp, members=members, filter="data")
            except TypeError:  # Python without extraction filters
                tf.extractall(tmp, members=members)
        inner = next(p for p in tmp.iterdir() if p.is_dir())
        dest = src.parent
        if dest.exists():
            import shutil

            shutil.rmtree(dest)
        inner.rename(dest)
        tmp.rmdir()
        return src

    def setup(self, ctx: BackendContext) -> None:
        for key in ("auk_commit", "hf_repo", "hf_revision", "ckpt_file", "qwen_repo", "qwen_revision",
                    "dtype", "cpu_offload", "max_gen_seconds", "chars_per_second"):
            setattr(self, key, self._setting(ctx, key))
        cache = Path(os.environ.get("AUK_CACHE_DIR", Path.home() / ".cache" / "auk-backend"))
        self.timings: Dict[str, float] = {}
        t_setup = time.time()

        # Two snapshot_download calls run in parallel threads; tqdm's shared class
        # lock races when both progress bars close (AttributeError: type object
        # 'tqdm' has no attribute '_lock'). The bars only spam the worker log anyway.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from huggingface_hub import snapshot_download

        def fetch_flash() -> str:
            return snapshot_download(self.hf_repo, revision=self.hf_revision,
                                     allow_patterns=["*.safetensors", "config.yaml"])

        def fetch_qwen() -> str:
            return snapshot_download(self.qwen_repo, revision=self.qwen_revision,
                                     ignore_patterns=["*.md", "LICENSE", ".gitattributes"])

        _log(f"fetching AuK source @ {self.auk_commit[:12]}, {self.hf_repo} and {self.qwen_repo}")
        with ThreadPoolExecutor(max_workers=3) as pool:
            t0 = time.time()
            f_src = pool.submit(self._fetch_source, cache)
            f_flash = pool.submit(fetch_flash)
            f_qwen = pool.submit(fetch_qwen)
            src = f_src.result()
            flash_dir = f_flash.result()
            qwen_dir = f_qwen.result()
            self.timings["downloads_s"] = round(time.time() - t0, 1)
        _log(f"downloads done in {self.timings['downloads_s']} s")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        import torch
        from auk.infer.infer_auk import AukInfer

        t_load = time.time()
        self.engine = AukInfer(
            os.path.join(flash_dir, "config.yaml"),
            os.path.join(flash_dir, self.ckpt_file),
            dtype=self.dtype,
            qwen_path=qwen_dir,
            cpu_offload=bool(self.cpu_offload),
        )
        self.timings["load_s"] = round(time.time() - t_load, 1)
        self.timings["setup_s"] = round(time.time() - t_setup, 1)
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        loaded = torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0
        self.timings["vram_loaded_mib"] = round(loaded)
        _log(f"AuK-Flash ready on {self.device_name}: {self.timings}")
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- inputs

    @staticmethod
    def _merged(inp: Dict[str, Any]) -> Dict[str, Any]:
        """Top-level fields plus ``extra_body`` (the platform carries non-OpenAI fields there)."""
        merged = dict(inp)
        extra = inp.get("extra_body")
        if isinstance(extra, dict):
            merged.update(extra)
        return merged

    @staticmethod
    def _from_messages(messages: Any) -> Tuple[str, Optional[str]]:
        """Instruction text and base64 audio from the last user message."""
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
                        audio = url.split(",", 1)[-1]
            return " ".join(t for t in texts if t).strip(), audio
        return "", None

    def _parse(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        f = self._merged(inp)
        instruction, audio_b64 = self._from_messages(f.get("messages"))
        speech = f.get("audio_body") if isinstance(f.get("audio_body"), dict) else None
        if speech:  # OpenAI /v1/audio/speech body
            f = {**speech, **{k: v for k, v in f.items() if k != "audio_body"}}
            text = str(speech.get("input") or "").strip()
            audio_b64 = speech.get("ref_audio") or audio_b64
            if speech.get("instruction"):
                instruction = str(speech["instruction"])
            elif text and audio_b64:
                instruction = f'Say the following with the same voice: "{text}"'
            elif text:
                # This phrasing (the AuK CLI example's) kept the voice description out of
                # the audio in our tests; the cookbook's "Generate speech based on ...
                # The content to speak is ..." often spoke words of the description first.
                voice = speech.get("instructions") or "a clear, natural adult voice"
                instruction = f'Based on the following description: "{voice}", generate speech content "{text}".'
            f.setdefault("gen_text", text)
        instruction = str(f.get("instruction") or f.get("prompt") or instruction or "").strip()
        audio_b64 = f.get("audio_b64") or audio_b64
        if not instruction:
            raise InvalidInput("expected an instruction: a user message (chat), 'instruction' or an audio_body 'input'")
        if isinstance(audio_b64, str) and audio_b64.startswith("data:"):
            audio_b64 = audio_b64.split(",", 1)[-1]
        return {"instruction": instruction, "audio_b64": audio_b64, "fields": f}

    def _gen_seconds(self, instruction: str, audio_path: Optional[str], f: Dict[str, Any]) -> Optional[float]:
        if f.get("gen_seconds"):
            return float(f["gen_seconds"])
        src_s = _wav_seconds(audio_path) if audio_path else None
        gen_text, ref_text = f.get("gen_text"), f.get("ref_text")
        if gen_text and ref_text and src_s:  # the AuK CLI's estimate
            return src_s * len(str(gen_text).encode()) / max(1, len(str(ref_text).encode()))
        quoted = re.findall(r'["“‘\']([^"”’]{2,})["”’\']', instruction)
        say = re.search(r"(same voice|content to speak|generate speech content|生成语音内容)", instruction, re.I)
        if not audio_path or say:
            text = str(gen_text or (quoted[-1] if quoted else ""))
            if not text:
                raise InvalidInput("text-only request: set gen_seconds or gen_text (the text to say)")
            return max(1.0, len(text) / float(self.chars_per_second) + 0.6)
        speed = re.search(r"speed to ([0-9.]+)\s*x", instruction, re.I)
        if speed and src_s:
            return src_s / float(speed.group(1))
        repl = re.search(r"Replace '([^']+)' with '([^']+)'", instruction)
        if repl and src_s:
            return max(0.5, src_s + (len(repl.group(2)) - len(repl.group(1))) / float(self.chars_per_second))
        return None  # same length as the source (pitch, emotion, denoise, separation...)

    # ---------------------------------------------------------------- process

    def process(self, job: Job) -> Result:
        import torch

        req = self._parse(job.input)
        f = req["fields"]
        t0 = time.time()
        audio_path = None
        tmpdir = tempfile.mkdtemp(prefix="auk-")
        try:
            if req["audio_b64"]:
                try:
                    raw = base64.b64decode(req["audio_b64"], validate=False)
                except Exception as e:  # noqa: BLE001
                    raise InvalidInput(f"audio is not valid base64: {e}") from e
                audio_path = os.path.join(tmpdir, "input.wav")
                with open(audio_path, "wb") as fh:
                    fh.write(raw)
                try:
                    in_s = _wav_seconds(audio_path)
                except Exception as e:  # noqa: BLE001
                    raise InvalidInput(f"could not read the input audio (send WAV or FLAC): {e}") from e
            else:
                in_s = None
            gen_s = self._gen_seconds(req["instruction"], audio_path, f)
            limit = float(self.max_gen_seconds)
            if (gen_s or in_s or 0) > limit:
                raise InvalidInput(f"output would be {(gen_s or in_s):.1f} s; this backend caps one request at {limit:.0f} s")
            content: List[Dict[str, Any]] = [{"type": "text", "text": req["instruction"]}]
            if audio_path:
                content.append({"type": "audio", "audio": audio_path})
            messages = [{"role": "user", "content": content}]
            seed = f.get("seed")
            with self._lock:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                t_gpu = time.time()
                audio, sr = self.engine.generate(
                    messages, audio=audio_path, gen_seconds=gen_s,
                    seed=int(seed) if seed is not None else None,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gpu_s = time.time() - t_gpu
                peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
            wav = _wav_bytes(audio, sr)
            out_s = audio.shape[-1] / float(sr)
            return Result(output={
                "audio_b64": base64.b64encode(wav).decode(),
                "format": "wav",
                "sample_rate": sr,
                "seconds": round(out_s, 2),
                "instruction": req["instruction"],
                "input_seconds": round(in_s, 2) if in_s else None,
                "gen_seconds": round(gen_s, 2) if gen_s else None,
                "timings": {
                    "gpu_s": round(gpu_s, 3),
                    "total_s": round(time.time() - t0, 3),
                    "realtime_factor": round(out_s / gpu_s, 1) if gpu_s > 0 else None,
                    "peak_vram_mib": round(peak),
                },
                "model": "tencent/AuK-Flash",
                "device": self.device_name,
            })
        finally:
            for p in Path(tmpdir).glob("*"):
                p.unlink(missing_ok=True)
            os.rmdir(tmpdir)


class AukFlashOffloadBackend(AukFlashBackend):
    """Same model with AuK's CPU offload on: fits 16 GB cards, slower per request."""

    name = "auk-flash-speech-offload"
    cpu_offload = True

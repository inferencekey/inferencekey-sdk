"""LFM3-Omni-2B (Liquid AI) as an InferenceKey custom backend.

LFM3-Omni-2B is a 2.17B omni model: one checkpoint takes text, images, video
(with its soundtrack) and 16 kHz audio, and answers in text, in 24 kHz speech,
or in text and speech interleaved. The built-in backends do not cover the
speech output (the authors serve speech with Transformers or with llama.cpp),
so this backend drives the checkpoint's own Transformers code:

``setup()`` (once per process)
    1. downloads the checkpoint (about 4.5 GB) at a pinned revision,
    2. loads the model in bf16 with ``trust_remote_code`` (the modeling code
       ships with the checkpoint), the processor and the LiquidCodec decoder,
    3. runs one short warm-up in text and in speech.

``process(job)`` (per request)
    reads an OpenAI-style chat (``messages``: text, ``input_audio``,
    ``image_url``, ``video_url`` parts), runs ``model.generate`` and returns
    ``{"text", "audio_b64", "audio_seconds", "tool_calls", "timings", ...}``
    (the speech as 16-bit PCM WAV, 24 kHz mono).

Options (top level or in ``extra_body``):

* ``mode``: ``text``, ``audio`` (speech only) or ``interleaved`` (text and
  speech). Omitted: the model picks, usually matching the input.
* ``max_new_tokens`` (a speech frame of 80 ms counts as one), ``temperature``
  (text; 0 or missing = greedy, as the authors recommend), ``audio_temperature``
  (0.4) and ``audio_repetition_penalty`` (1.2), the authors' speech settings.
* ``use_audio_in_video`` (default true): the model hears the video soundtrack
  together with the frames. ``fps`` (default 2) for the video frames.
* ``tools``: OpenAI function schemas. The model writes
  ``<tool_call><function=...>`` blocks, returned parsed in ``tool_calls``.

Settings are class attributes; override them with ``LFM3_*`` environment
variables or the runtime's ``--config-json`` (keys in lower case).
``LFM3_VRAM_LIMIT_GIB`` caps the memory PyTorch may use on the GPU (to check
that the model fits a smaller card).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from typing import Any, Dict, List, Optional, Tuple

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

MODES = ("text", "audio", "interleaved")
CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
AUDIO_EXT = {"wav": ".wav", "mp3": ".mp3", "flac": ".flac", "ogg": ".ogg", "m4a": ".m4a", "webm": ".webm"}


def _log(msg: str) -> None:
    print(f"[lfm3-omni-backend] {msg}", file=sys.stderr, flush=True)


def _wav_bytes(samples, sample_rate: int) -> bytes:
    """float tensor in [-1, 1] -> 16-bit PCM mono WAV bytes."""
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


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """The authors' parser: `<tool_call><function=name><parameter=k>v</parameter>...` blocks."""
    calls = []
    for name, body in CALL_RE.findall(text or ""):
        arguments: Dict[str, Any] = {}
        for key, value in PARAM_RE.findall(body):
            try:
                arguments[key] = json.loads(value)
            except json.JSONDecodeError:
                arguments[key] = value
        calls.append({"type": "function", "function": {"name": name.strip(), "arguments": arguments}})
    return calls


class _FirstTick:
    """A generate() streamer that only records when the first item arrives (skip=1 drops the prompt)."""

    def __init__(self, skip: int = 0):
        self.skip = skip
        self.count = 0
        self.first: Optional[float] = None

    def put(self, value) -> None:  # noqa: ANN001
        self.count += 1
        if self.count > self.skip and self.first is None:
            self.first = time.time()

    def end(self) -> None:
        pass


class Lfm3OmniBackend(CustomBackend):
    name = "lfm3-omni-2b"
    version = "0.1.0"
    task_type = "text2audio"
    description = "LFM3-Omni-2B (Liquid AI): text, image, video and audio in; text, 24 kHz speech or both out"

    # Pinned upstream: the Hugging Face revision this example was tested with.
    hf_repo = "LiquidAI/LFM3-Omni-2B"
    hf_revision = "21d75123f86c1f7ffb4bef1ea202710c4e7ebe67"
    # 0 = no cap. Otherwise the GiB PyTorch may allocate on the GPU.
    vram_limit_gib = 0.0
    # Most video frames the processor keeps (the authors' default is 384).
    max_frames = 384
    # Upper bound for max_new_tokens in one request.
    max_new_tokens_cap = 2048

    # ------------------------------------------------------------------ setup

    def _setting(self, ctx: BackendContext, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"LFM3_{key.upper()}")
        if raw is None:
            raw = (ctx.config or {}).get(key)
        if raw is None:
            return default
        if isinstance(default, bool):
            return str(raw).lower() in ("1", "true", "yes", "on")
        return type(default)(raw)

    @staticmethod
    def _ensure_ffmpeg() -> None:
        """Video decoding (torchcodec) needs the FFmpeg shared libraries on the host."""
        if shutil.which("ffmpeg"):
            return
        if hasattr(os, "geteuid") and os.geteuid() == 0 and shutil.which("apt-get"):
            _log("ffmpeg not found: installing it with apt-get (needed for video input)")
            env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
            subprocess.run(["apt-get", "update", "-qq"], check=False, env=env, timeout=600)
            subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=False, env=env, timeout=900)
        if not shutil.which("ffmpeg"):
            _log("WARNING: ffmpeg is missing: video input will fail (install ffmpeg on the worker)")

    def setup(self, ctx: BackendContext) -> None:
        for key in ("hf_repo", "hf_revision", "vram_limit_gib", "max_frames", "max_new_tokens_cap"):
            setattr(self, key, self._setting(ctx, key))
        self.timings: Dict[str, float] = {}
        t_setup = time.time()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        self._ensure_ffmpeg()

        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if torch.cuda.is_available() and self.vram_limit_gib > 0:
            total = torch.cuda.get_device_properties(0).total_memory
            frac = min(1.0, self.vram_limit_gib * 2**30 / total)
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            _log(f"VRAM capped at {self.vram_limit_gib} GiB (fraction {frac:.3f} of {total / 2**30:.1f} GiB)")

        t0 = time.time()
        model_dir = snapshot_download(self.hf_repo, revision=self.hf_revision, ignore_patterns=["assets/*", "*.md"])
        self.timings["download_s"] = round(time.time() - t0, 1)
        _log(f"weights in {self.timings['download_s']} s at {model_dir}")

        t0 = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_dir, dtype=dtype, device_map=self.device, trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, max_frames=int(self.max_frames))
        self.codec = self.processor.audio_tokenizer.to(device=self.device).eval()
        self.sample_rate = 24_000
        self.timings["load_s"] = round(time.time() - t0, 1)
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        self.timings["vram_loaded_mib"] = round(torch.cuda.memory_allocated() / 2**20) if torch.cuda.is_available() else 0
        self._lock = threading.Lock()

        # Warm-up: the first calls pay for kernels and the codec.
        t0 = time.time()
        self._generate([{"role": "user", "content": "Say hi."}], {"mode": "text", "max_new_tokens": 8})
        self._generate([{"role": "system", "content": "Perform TTS in American English using a female voice."},
                        {"role": "user", "content": "Ready."}], {"mode": "audio", "max_new_tokens": 40})
        self.timings["warmup_s"] = round(time.time() - t0, 1)
        self.timings["setup_s"] = round(time.time() - t_setup, 1)
        _log(f"LFM3-Omni-2B ready on {self.device_name}: {self.timings}")

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
    def _data_from(url_or_data: str) -> Tuple[bytes, str]:
        """base64 (raw or a data: URL) -> bytes and the MIME subtype when the data URL names it."""
        mime = ""
        if url_or_data.startswith("data:"):
            head, _, url_or_data = url_or_data.partition(",")
            mime = head[5:].split(";")[0].split("/")[-1]
        elif url_or_data.startswith(("http://", "https://")):
            raise InvalidInput("send media inline as base64 (data: URLs); this backend does not fetch URLs")
        try:
            return base64.b64decode(url_or_data, validate=False), mime
        except Exception as e:  # noqa: BLE001
            raise InvalidInput(f"media is not valid base64: {e}") from e

    def _conversation(self, messages: Any, tmpdir: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """OpenAI chat messages -> the processor's conversation, media written to temp files."""
        if not isinstance(messages, list) or not messages:
            raise InvalidInput("expected 'messages': an OpenAI-style chat")
        counts = {"image": 0, "audio": 0, "video": 0}
        conv: List[Dict[str, Any]] = []
        n = 0
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") not in ("system", "user", "assistant", "tool"):
                raise InvalidInput("each message needs a role: system, user, assistant or tool")
            content = msg.get("content")
            if msg.get("role") != "user" or isinstance(content, str) or content is None:
                out = {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")}
                if isinstance(content, list):  # system/assistant parts: keep the text
                    out["content"] = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
                conv.append(out)
                continue
            parts: List[Dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = part.get("type")
                n += 1
                if kind == "text":
                    parts.append({"type": "text", "text": str(part.get("text", ""))})
                elif kind == "input_audio":
                    a = part.get("input_audio") or {}
                    raw, _ = self._data_from(str(a.get("data", "")))
                    path = os.path.join(tmpdir, f"a{n}{AUDIO_EXT.get(str(a.get('format', 'wav')).lower(), '.wav')}")
                    open(path, "wb").write(raw)
                    parts.append({"type": "audio", "audio": path})
                    counts["audio"] += 1
                elif kind in ("image_url", "image"):
                    url = (part.get("image_url") or {}).get("url") if kind == "image_url" else part.get("image")
                    raw, mime = self._data_from(str(url or ""))
                    path = os.path.join(tmpdir, f"i{n}.{mime or 'png'}")
                    open(path, "wb").write(raw)
                    parts.append({"type": "image", "image": path})
                    counts["image"] += 1
                elif kind in ("video_url", "video"):
                    url = (part.get("video_url") or {}).get("url") if kind == "video_url" else part.get("video")
                    raw, mime = self._data_from(str(url or ""))
                    path = os.path.join(tmpdir, f"v{n}.{mime or 'mp4'}")
                    open(path, "wb").write(raw)
                    parts.append({"type": "video", "video": path})
                    counts["video"] += 1
                else:
                    raise InvalidInput(f"unsupported content part {kind!r}")
            conv.append({"role": "user", "content": parts})
        return conv, counts

    # -------------------------------------------------------------- generate

    def _generate(self, conv: List[Dict[str, Any]], f: Dict[str, Any], has_video: bool = False) -> Dict[str, Any]:
        import torch

        mode = f.get("mode")
        if mode is not None and mode not in MODES:
            raise InvalidInput(f"mode must be one of {', '.join(MODES)}")
        max_new = int(f.get("max_new_tokens") or (512 if mode in (None, "text") else 1024))
        max_new = max(1, min(max_new, int(self.max_new_tokens_cap)))
        tmpl: Dict[str, Any] = {}
        pk: Dict[str, Any] = {}
        if has_video:
            if f.get("use_audio_in_video", True):
                tmpl["load_audio_from_video"] = True
                pk["use_audio_in_video"] = True
            if f.get("fps"):
                pk["fps"] = float(f["fps"])
        if pk:
            tmpl["processor_kwargs"] = pk
        tools = f.get("tools") or None

        t0 = time.time()
        inputs = self.processor.apply_chat_template(
            conv, tools=tools, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", **tmpl,
        ).to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        gen: Dict[str, Any] = {"max_new_tokens": max_new}
        if mode:
            gen["generation_mode"] = mode
        if mode in ("audio", "interleaved") or f.get("audio_temperature") is not None:
            gen["audio_temperature"] = float(f.get("audio_temperature", 0.4))
            gen["audio_repetition_penalty"] = float(f.get("audio_repetition_penalty", 1.2))
        temp = float(f.get("temperature") or 0)
        if temp > 0:
            gen.update(do_sample=True, temperature=temp)
        else:
            gen["do_sample"] = False
        text_tick, audio_tick = _FirstTick(skip=1), _FirstTick()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t_gen = time.time()
        prep_s = t_gen - t0
        with torch.no_grad():
            out = self.model.generate(**inputs, streamer=text_tick, audio_streamer=audio_tick, **gen)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_s = time.time() - t_gen
            end = int(out.sequence_lengths[0])
            ids = out.sequences[0, prompt_tokens:end]
            text = self.processor.decode(ids, skip_special_tokens=True)
            frames = int(out.audio_lengths[0]) if out.audio_lengths is not None else 0
            wav_b64, audio_s, dec_s = None, 0.0, 0.0
            if frames > 0:
                t_dec = time.time()
                waveform = self.codec.decode(out.audio_codes[:1, :, :frames]).audio_values[0]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dec_s = time.time() - t_dec
                wav = _wav_bytes(waveform.squeeze(0), self.sample_rate)
                audio_s = waveform.shape[-1] / float(self.sample_rate)
                wav_b64 = base64.b64encode(wav).decode()
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
        text_tokens = int(len(ids)) - frames
        return {
            "text": text.strip(),
            "audio_b64": wav_b64,
            "format": "wav" if wav_b64 else None,
            "sample_rate": self.sample_rate if wav_b64 else None,
            "audio_seconds": round(audio_s, 2),
            "tool_calls": parse_tool_calls(text) if tools else [],
            "timings": {
                "prep_s": round(prep_s, 3),
                "generate_s": round(gen_s, 3),
                "first_text_s": round(text_tick.first - t_gen, 3) if text_tick.first else None,
                "first_audio_frame_s": round(audio_tick.first - t_gen, 3) if audio_tick.first else None,
                "codec_decode_s": round(dec_s, 3),
                "prompt_tokens": prompt_tokens,
                "text_tokens": text_tokens,
                "audio_frames": frames,
                "slots_per_s": round(len(ids) / gen_s, 1) if gen_s > 0 else None,
                "realtime_factor": round(audio_s / gen_s, 2) if (gen_s > 0 and audio_s) else None,
                "peak_vram_mib": round(peak),
            },
        }

    # ---------------------------------------------------------------- process

    def process(self, job: Job) -> Result:
        f = self._merged(job.input)
        t0 = time.time()
        tmpdir = tempfile.mkdtemp(prefix="lfm3-")
        try:
            conv, counts = self._conversation(f.get("messages"), tmpdir)
            with self._lock:
                try:
                    out = self._generate(conv, f, has_video=counts["video"] > 0)
                except (ValueError, RuntimeError) as e:
                    msg = str(e)
                    if "out of memory" in msg.lower():
                        raise
                    raise InvalidInput(msg[:500]) from e
            out["timings"]["total_s"] = round(time.time() - t0, 3)
            out.update({"inputs": counts, "mode": f.get("mode") or "auto", "model": self.hf_repo,
                        "device": self.device_name})
            return Result(output=out)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

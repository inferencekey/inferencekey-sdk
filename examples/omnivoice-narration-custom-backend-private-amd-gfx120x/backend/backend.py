"""OmniVoice (k2-fsa) as an InferenceKey custom backend for voice narration.

OmniVoice is a zero-shot multilingual TTS model (600+ languages, Apache-2.0,
~0.8B parameters, about 3.5 GB of VRAM in float16). No built-in backend serves
it, so this backend drives the authors' ``omnivoice`` package directly:

``setup()`` (once per process)
    1. downloads the OmniVoice weights at a pinned Hugging Face revision,
    2. loads the model on the best device available (CUDA or ROCm through
       ``torch.cuda``, then Apple MPS, then CPU),
    3. loads the saved voices shipped in ``voices/*.pt`` (OmniVoice voice clone
       prompts, a few KB each: audio tokens of a reference clip plus its text).

``process(job)`` (per request)
    reads the text to narrate, splits it into sentence chunks, synthesizes them
    with one saved voice (or a voice designed from ``instruct``), joins them
    with a short pause and returns ``{"audio_b64", "format", "sample_rate",
    "seconds", "voice", "timings", ...}``.

Request shapes accepted (every option can also sit in ``extra_body``):

* OpenAI chat (``/v1/chat/completions``): the last user message is the text.
* Plain fields: ``text`` (or ``input``).
* OpenAI TTS body (``audio_body``): ``input`` is the text, ``voice`` the voice.

Options: ``voice`` (a saved voice name: ``voz1`` or ``voz2``; ``voz`` is accepted
as the field name too), ``instruct`` (voice design instead of a
saved voice, e.g. "male, elderly, very low pitch"), ``speed`` (default 0.85),
``language`` (default "es"), ``num_step`` (diffusion steps, default 32),
``seed``, ``pause_ms`` (silence between chunks), ``response_format``
(``wav``, ``flac`` or ``mp3``).

Settings are class attributes; override them with ``OMNIVOICE_*`` environment
variables (keys in upper case).
"""

from __future__ import annotations

import base64
import io
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

HERE = Path(__file__).resolve().parent
SAMPLE_RATE = 24000
FORMATS = {"wav": ("WAV", "PCM_16"), "flac": ("FLAC", "PCM_16"), "mp3": ("MP3", None)}


def _log(msg: str) -> None:
    print(f"[omnivoice-backend] {msg}", file=sys.stderr, flush=True)


def _split_sentences(text: str, max_chars: int) -> List[str]:
    """Split text into chunks of whole sentences, each at most ``max_chars`` long when possible."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…;:])\s+", text) if s.strip()]
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def _encode(samples, fmt: str) -> bytes:
    import soundfile as sf

    container, subtype = FORMATS[fmt]
    buf = io.BytesIO()
    sf.write(buf, samples, SAMPLE_RATE, format=container, subtype=subtype)
    return buf.getvalue()


class OmniVoiceNarrationBackend(CustomBackend):
    name = "omnivoice-narration"
    version = "0.1.1"
    task_type = "text2audio"
    description = "OmniVoice (k2-fsa, Apache-2.0): voice narration with saved voices, 600+ languages"

    hf_repo = "k2-fsa/OmniVoice"
    hf_revision = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
    default_voice = "voz1"
    # Older names that callers may still send, mapped to a file in voices/.
    voice_aliases = {"narrador_trailer": "voz1"}
    default_language = "es"
    default_speed = 0.85
    num_step = 32
    pause_ms = 350
    # Sentences are grouped into chunks of about this many characters per generate() call.
    chunk_chars = 280
    # One request narrates at most this many characters (a few minutes of audio).
    max_chars = 6000
    # The result travels back over NATS (8 MB max payload); base64 adds about a third.
    max_audio_bytes = 5_500_000

    # ------------------------------------------------------------------ setup

    def _setting(self, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"OMNIVOICE_{key.upper()}")
        if raw is None:
            return default
        return type(default)(raw)

    @staticmethod
    def _pick_device() -> str:
        import torch

        if torch.cuda.is_available():  # NVIDIA CUDA and AMD ROCm builds both answer here
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def setup(self, ctx: BackendContext) -> None:
        for key in ("hf_repo", "hf_revision", "default_voice", "default_language", "default_speed",
                    "num_step", "pause_ms", "chunk_chars", "max_chars", "max_audio_bytes"):
            setattr(self, key, self._setting(key))
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        self.timings: Dict[str, float] = {}

        import torch
        from huggingface_hub import snapshot_download
        from omnivoice import OmniVoice, VoiceClonePrompt

        t0 = time.time()
        model_dir = snapshot_download(self.hf_repo, revision=self.hf_revision)
        self.timings["download_s"] = round(time.time() - t0, 1)

        self.device = self._pick_device()
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        t0 = time.time()
        self.model = OmniVoice.from_pretrained(model_dir, device_map=self.device, dtype=dtype)
        self.timings["load_s"] = round(time.time() - t0, 1)
        self.device_name = torch.cuda.get_device_name(0) if self.device.startswith("cuda") else self.device

        self.voices = {
            path.stem: VoiceClonePrompt.load(str(path)) for path in sorted((HERE / "voices").glob("*.pt"))
        }
        if self.default_voice not in self.voices:
            raise RuntimeError(f"default voice {self.default_voice!r} not found in voices/: {sorted(self.voices)}")
        self._lock = threading.Lock()
        _log(f"OmniVoice ready on {self.device_name}, voices={sorted(self.voices)}, timings={self.timings}")

    # ---------------------------------------------------------------- inputs

    @staticmethod
    def _text_from_messages(messages: Any) -> str:
        if not isinstance(messages, list):
            return ""
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            return " ".join(
                str(part.get("text", "")) for part in content or [] if isinstance(part, dict) and part.get("type") == "text"
            )
        return ""

    def _parse(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        fields = dict(inp)
        if isinstance(inp.get("extra_body"), dict):
            fields.update(inp["extra_body"])
        speech = fields.get("audio_body")
        if isinstance(speech, dict):  # OpenAI /v1/audio/speech body
            fields = {**speech, **{k: v for k, v in fields.items() if k != "audio_body"}}
        text = str(fields.get("text") or fields.get("input") or self._text_from_messages(fields.get("messages"))).strip()
        if not text:
            raise InvalidInput("expected the text to narrate: a user message (chat), 'text' or an audio_body 'input'")
        if len(text) > self.max_chars:
            raise InvalidInput(f"text is {len(text)} characters; this backend caps one request at {self.max_chars}")

        instruct = fields.get("instruct")
        voice = fields.get("voice") or fields.get("voz") or self.default_voice
        voice = self.voice_aliases.get(voice, voice)
        if not instruct and voice not in self.voices:
            raise InvalidInput(f"unknown voice {voice!r}; available: {sorted(self.voices)}")
        fmt = str(fields.get("response_format") or "wav").lower()
        if fmt not in FORMATS:
            raise InvalidInput(f"response_format must be one of {sorted(FORMATS)}")
        return {
            "text": text,
            "voice": None if instruct else voice,
            "instruct": instruct,
            "format": fmt,
            "speed": float(fields.get("speed") or self.default_speed),
            "language": fields.get("language") or self.default_language,
            "num_step": int(fields.get("num_step") or self.num_step),
            "pause_ms": int(fields.get("pause_ms") if fields.get("pause_ms") is not None else self.pause_ms),
            "seed": fields.get("seed"),
        }

    # ---------------------------------------------------------------- process

    def process(self, job: Job) -> Result:
        import numpy as np
        import torch

        req = self._parse(job.input)
        chunks = _split_sentences(req["text"], self.chunk_chars)
        voice_kwargs = (
            {"instruct": req["instruct"]} if req["instruct"] else {"voice_clone_prompt": self.voices[req["voice"]]}
        )
        pause = np.zeros(int(SAMPLE_RATE * req["pause_ms"] / 1000), dtype=np.float32)

        t0 = time.time()
        pieces: List[Any] = []
        with self._lock:
            if req["seed"] is not None:
                torch.manual_seed(int(req["seed"]))
            for i, chunk in enumerate(chunks):
                audio = self.model.generate(
                    text=chunk, language=req["language"], speed=req["speed"],
                    num_step=req["num_step"], **voice_kwargs,
                )
                if i:
                    pieces.append(pause)
                pieces.append(np.asarray(audio[0], dtype=np.float32).reshape(-1))
        gpu_s = time.time() - t0

        samples = np.concatenate(pieces)
        encoded = _encode(samples, req["format"])
        if len(encoded) > self.max_audio_bytes:
            raise InvalidInput(
                f"the {req['format']} result is {len(encoded) // 1024} KB, over the {self.max_audio_bytes // 1024} KB "
                "transport limit; ask for response_format=mp3 or send a shorter text"
            )
        seconds = len(samples) / float(SAMPLE_RATE)
        return Result(output={
            "audio_b64": base64.b64encode(encoded).decode(),
            "format": req["format"],
            "sample_rate": SAMPLE_RATE,
            "seconds": round(seconds, 2),
            "voice": req["voice"],
            "instruct": req["instruct"],
            "chunks": len(chunks),
            "timings": {
                "gpu_s": round(gpu_s, 3),
                "realtime_factor": round(seconds / gpu_s, 2) if gpu_s > 0 else None,
            },
            "model": self.hf_repo,
            "device": self.device_name,
        })

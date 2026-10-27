"""A custom audio2text backend wrapping Underdog Bark 3.0 Pro (Conway Research).

Bark 3.0 Pro is a 0.6B English speech-to-text model (CC-BY-4.0), fine-tuned from
NVIDIA Parakeet TDT 0.6B v3: a FastConformer encoder with a TDT decoder, 1.25 GB
of bf16 weights. It loads as a stock Transformers model (``ParakeetForTDT``,
transformers 5.19), with no custom code. No built-in InferenceKey backend
(vLLM, SGLang, llama.cpp) runs a Parakeet TDT model, so this thin backend
loads it with Transformers and follows the authors' own runner
(``runtime/transcribe_bark_pro.py``): mono 16 kHz audio, 0.25 s of leading
silence, greedy decoding.

Contract, accepts BOTH callers:

* public API ``POST .../v1/audio/transcriptions`` with the OpenAI multipart
  body (``file`` + ``model``), exactly what ``openai.audio.transcriptions.create``
  sends; the manager forwards it base64-encoded and ``Job.audio_bytes()`` parses it.
* native ``/process`` or a JSON body on the same route:
  ``{"audio_b64": "<base64 of a wav/flac/ogg/mp3 file>"}``.
* output: the OpenAI transcription shape ``{"text": "..."}`` plus
  ``duration`` (seconds of audio), ``seconds`` (GPU time), ``rtfx`` (audio
  seconds per second of compute), ``chunks``, ``vram_peak_gib``, ``device``.

Knobs (environment, through the workload ``config.env``):

* ``BARK_DEVICE``: ``cuda`` (default) or ``cpu`` (float32, as the authors advise).
* ``BARK_DTYPE``: ``bfloat16`` (default on CUDA), ``float16`` or ``float32``.
* ``BARK_VRAM_LIMIT_GIB``: caps this process with
  ``torch.cuda.set_per_process_memory_fraction``; how we emulated smaller cards.
* ``BARK_CHUNK_SECONDS``: audio longer than this is cut into pieces of this
  length and transcribed one by one (default 600; the encoder attends over the
  whole input, so memory grows with length).

Run it by hand (from this folder, with requirements installed)::

    python -m inferencekey.backend.serve --port 8099 --backend backend:BarkProBackend

Attribution: Underdog Bark 3.0 Pro by Conway Research, CC-BY-4.0
(https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Pro), derived from
nvidia/parakeet-tdt-0.6b-v3 (NVIDIA, CC-BY-4.0) and moondream/parakeet-ultra
(Moondream, CC-BY-4.0).
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time

from inferencekey.backend import BackendContext, CustomBackend, Job, Result

try:  # newer SDKs answer 400 for bad client input
    from inferencekey.backend import InvalidInput
except ImportError:  # pragma: no cover - older SDK
    InvalidInput = ValueError

MODEL_ID = "ConwayResearch/Underdog-Bark-3.0-Pro"
#: The revision this backend was tested with (2026-10-27).
REVISION = "4b5c51e04aa8d16dda9ecdff71bc29f7360149e4"
#: Only what the PyTorch path needs: skips the Core ML and MLX copies (~1.2 GB).
ALLOW = [
    "model.safetensors", "config.json", "generation_config.json", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json", "NOTICE",
]
SR = 16000
LEAD_SILENCE = 4000  # 0.25 s at 16 kHz, as in the authors' runner


def _decode(data: bytes):
    """Bytes of an audio file -> mono float32 at 16 kHz."""
    import numpy as np
    import soundfile as sf
    import soxr

    try:
        wave, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)
    except Exception:  # noqa: BLE001 - formats libsndfile cannot read
        try:
            import librosa

            wave, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
        except Exception as exc:  # noqa: BLE001
            raise InvalidInput(f"could not decode the audio: {exc}")
    if not len(wave):
        raise InvalidInput("the audio is empty")
    if sr != SR:
        wave = soxr.resample(wave, sr, SR)
    return np.ascontiguousarray(wave, dtype=np.float32)


class BarkProBackend(CustomBackend):
    """Underdog Bark 3.0 Pro: English speech to text."""

    name = "underdog-bark-pro"
    version = "0.1.0"
    task_type = "audio2text"
    requirements = "requirements.txt"
    requires_python = ">=3.12,<3.13"

    def setup(self, ctx: BackendContext) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, ParakeetForTDT

        t0 = time.time()
        local = snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=ALLOW)
        t_download = time.time()

        self.device = (os.environ.get("BARK_DEVICE") or "cuda").strip().lower()
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; set BARK_DEVICE=cpu to run on the CPU")
        default_dtype = "bfloat16" if self.device == "cuda" else "float32"
        dtype_name = (os.environ.get("BARK_DTYPE") or default_dtype).strip().lower()
        if self.device == "cuda" and dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
            dtype_name = "float16"  # pre-Ampere cards: the authors' documented fallback
        self.dtype = getattr(torch, dtype_name)
        self.dtype_name = dtype_name
        self.chunk_s = float(os.environ.get("BARK_CHUNK_SECONDS") or 600)

        limit = float(os.environ.get("BARK_VRAM_LIMIT_GIB") or 0)
        total = 0.0
        if self.device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            if limit > 0:
                torch.cuda.set_per_process_memory_fraction(min(1.0, limit / total), 0)
        else:
            torch.set_num_threads(int(os.environ.get("BARK_CPU_THREADS") or os.cpu_count() or 4))

        self.processor = AutoProcessor.from_pretrained(local)
        self.model = ParakeetForTDT.from_pretrained(local, dtype=self.dtype).to(self.device).eval()
        self.torch = torch
        self.lock = threading.Lock()
        # Warm-up on one second of silence (first CUDA kernels, allocator).
        self._transcribe(__import__("numpy").zeros(SR, dtype="float32"))
        if self.device == "cuda":
            torch.cuda.synchronize()
            weights = torch.cuda.memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
            where = f"{torch.cuda.get_device_name(0)} ({total:.1f} GiB)" + (f", cap {limit} GiB" if limit > 0 else "")
        else:
            weights = 0.0
            where = f"CPU ({torch.get_num_threads()} threads)"
        self.ready_detail = (
            f"{MODEL_ID} {dtype_name} on {where}: download {t_download - t0:.1f}s, "
            f"load {time.time() - t_download:.1f}s, weights on GPU {weights:.2f} GiB"
        )
        print("bark-pro ready: " + self.ready_detail, file=sys.stderr, flush=True)

    def _transcribe(self, wave) -> str:
        import numpy as np

        torch = self.torch
        wave = np.concatenate([np.zeros(LEAD_SILENCE, np.float32), wave])
        inputs = self.processor(wave, sampling_rate=SR, return_tensors="pt").to(self.device, self.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs)
        ids = out.sequences if hasattr(out, "sequences") else out
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def process(self, job: Job) -> Result:
        torch = self.torch
        wave = _decode(job.audio_bytes())
        duration = len(wave) / SR
        step = int(self.chunk_s * SR) if self.chunk_s > 0 else len(wave)
        pieces = [wave[i:i + step] for i in range(0, len(wave), step)] or [wave]

        with self.lock:  # one request at a time on this GPU
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            text = " ".join(t for t in (self._transcribe(p) for p in pieces) if t)
            if self.device == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30 if self.device == "cuda" else 0.0

        return Result(
            {
                "text": text,
                "duration": round(duration, 3),
                "seconds": round(seconds, 3),
                "rtfx": round(duration / seconds, 1) if seconds > 0 else None,
                "chunks": len(pieces),
                "vram_peak_gib": round(peak, 3),
                "device": self.device,
                "dtype": self.dtype_name,
                "model": MODEL_ID,
            }
        )

"""EmbeddingGemma 2 as a custom backend: text, code, images, video and audio in one 768-d space.

Google DeepMind's EmbeddingGemma 2 (``google/embeddinggemma-2``, Apache-2.0,
740M parameters) maps every modality into the same vector space. The built-in
embedding runtimes (sglang, llama.cpp, Ollama) only take text, and vLLM has no
``EmbeddingGemma2Model`` architecture yet, so this backend runs the model the
way its card documents: with ``sentence-transformers``.

It answers the OpenAI embeddings shape on ``/v1/embeddings``. The gateway puts
the caller's JSON in ``input.passthrough_body``; the backend's ``output`` is
returned to the caller as the HTTP body.

Request (OpenAI fields plus three optional extensions)::

    {
      "model": "default",
      "input": "a string" | ["strings", ...] | [item, ...],
      "dimensions": 768 | 512 | 256 | 128,     # Matryoshka truncation, re-normalized
      "prompt_name": "SearchQuery",             # task prefix for plain-string inputs
      "encoding_format": "float" | "base64"
    }

An ``item`` is a string or an object with ``text`` and/or one media key:
``image``, ``audio``, ``video`` (each a ``data:`` URI, raw base64 or an
``https://`` URL; ``image`` and ``video`` also accept a list). Text inside an
item may carry ``<|image|>``, ``<|audio|>`` and ``<|video|>`` placeholders to
interleave media and text into ONE embedding, exactly as the model card shows.
Media items get no task prefix (the card's rule); text items get
``prompt_name`` when set.

Decoding is done here, not by the model's processor: images with Pillow, audio
and video with PyAV (FFmpeg bundled in the wheel), so the worker needs no
system FFmpeg and no ``torchcodec``. Video is sampled at 1 frame per second, at
most 32 frames (the processor's defaults); audio is resampled to 16 kHz mono.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

MODEL_ID = "google/embeddinggemma-2"
ALLOWED_DIMENSIONS = (768, 512, 256, 128)
MAX_ITEMS = 256
MAX_MEDIA_BYTES = 20 * 1024 * 1024
VIDEO_FPS = 1.0
VIDEO_MAX_FRAMES = 32
AUDIO_RATE = 16000
MEDIA_KEYS = ("image", "audio", "video")
ENCODER_FOR = {"image": "vision", "video": "vision", "audio": "audio"}
# Token cost per media unit, from the model card, only to fill `usage`.
IMAGE_TOKENS = 280
FRAME_TOKENS = 140
AUDIO_TOKENS_PER_SECOND = 25
# Release cached GPU memory after requests larger than this (see process()).
RELEASE_CACHE_ABOVE_TOKENS = 16384


def _cpu_quota() -> int:
    """CPUs this container may really use (cgroup v2 or v1 quota), not the host's count.

    Rented GPU containers often see every host core (64 here) but are capped
    at a handful. torch sizes its CPU thread pool from the host count, and the
    image and video preprocessing (which runs on the CPU) then thrashes: on an
    8-vCPU RunPod container one image took 0.50 s with 64 threads and 0.10 s
    with 8. `EG2_TORCH_THREADS` overrides the detection.
    """
    override = os.environ.get("EG2_TORCH_THREADS")
    if override:
        return max(1, int(override))
    quota = None
    try:  # cgroup v2: "max 100000" or "680000 100000"
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max":
            quota = int(q) / int(p)
    except (OSError, ValueError):
        pass
    if quota is None:
        try:  # cgroup v1
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0 and p > 0:
                quota = q / p
        except (OSError, ValueError):
            pass
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if quota is not None:
        cpus = min(cpus, max(1, int(quota)))
    return max(1, cpus)


def _read_media(ref: Any) -> bytes:
    """Bytes of a media reference: data URI, raw base64, or an https URL."""
    if not isinstance(ref, str) or not ref:
        raise InvalidInput("a media value must be a non-empty string (data URI, base64 or https URL)")
    if ref.startswith("https://") or ref.startswith("http://"):
        import requests

        with requests.get(ref, timeout=30, stream=True) as resp:
            if resp.status_code != 200:
                raise InvalidInput(f"could not fetch media URL: HTTP {resp.status_code}")
            buf = io.BytesIO()
            for chunk in resp.iter_content(1 << 16):
                buf.write(chunk)
                if buf.tell() > MAX_MEDIA_BYTES:
                    raise InvalidInput("media URL is larger than 20 MB")
            return buf.getvalue()
    payload = ref.split(",", 1)[1] if ref.startswith("data:") else ref
    try:
        data = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise InvalidInput(f"media is not valid base64: {exc}") from exc
    if not data:
        raise InvalidInput("media decoded to zero bytes")
    if len(data) > MAX_MEDIA_BYTES:
        raise InvalidInput("media is larger than 20 MB")
    return data


def _image(ref: Any):
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(_read_media(ref)))
        return img.convert("RGB")
    except InvalidInput:
        raise
    except Exception as exc:  # noqa: BLE001 - any decoder error is the caller's input
        raise InvalidInput(f"could not decode image: {exc}") from exc


def _audio(ref: Any) -> Tuple[Any, float]:
    """Decode any container PyAV knows to 16 kHz mono float32; returns (array, seconds)."""
    import av
    import numpy as np

    try:
        container = av.open(io.BytesIO(_read_media(ref)))
        resampler = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_RATE)
        chunks = []
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
        container.close()
    except InvalidInput:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InvalidInput(f"could not decode audio: {exc}") from exc
    if not chunks:
        raise InvalidInput("audio has no samples")
    samples = np.concatenate(chunks).astype(np.float32)
    return samples, len(samples) / AUDIO_RATE


def _video(ref: Any):
    """Decode a video to an RGB frame array (T, H, W, 3) sampled at 1 fps, at most 32 frames."""
    import av
    import numpy as np

    try:
        container = av.open(io.BytesIO(_read_media(ref)))
        stream = container.streams.video[0]
        frames, next_t = [], 0.0
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else next_t
            if t + 1e-6 >= next_t:
                frames.append(frame.to_ndarray(format="rgb24"))
                next_t += 1.0 / VIDEO_FPS
        container.close()
    except InvalidInput:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InvalidInput(f"could not decode video: {exc}") from exc
    if not frames:
        raise InvalidInput("video has no frames")
    if len(frames) > VIDEO_MAX_FRAMES:
        idx = np.linspace(0, len(frames) - 1, VIDEO_MAX_FRAMES).round().astype(int)
        frames = [frames[i] for i in idx]
    return np.stack(frames)


class EmbeddingGemma2Backend(CustomBackend):
    """Full multimodal model: text, images, video and audio (740M parameters)."""

    name = "embeddinggemma2-multimodal"
    version = "0.1.3"
    task_type = "embedding"
    requirements = "requirements.txt"
    # Encoders to drop at load time (the card's "selective encoder loading").
    drop_encoders: Tuple[str, ...] = ()

    def setup(self, ctx: BackendContext) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.threads = _cpu_quota()
        torch.set_num_threads(self.threads)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # bf16 where the GPU supports it, float32 otherwise. Never float16:
        # the card warns it returns NaN or silently degraded vectors.
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        self.dtype_name = str(dtype).replace("torch.", "")
        config_kwargs = {f"{enc}_config": None for enc in self.drop_encoders}
        t0 = time.time()
        self.model = SentenceTransformer(
            MODEL_ID,
            device=self.device,
            model_kwargs={"torch_dtype": dtype},
            config_kwargs=config_kwargs,
            token=os.environ.get("HF_TOKEN") or None,
        )
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.prompts = dict(getattr(self.model, "prompts", {}) or {})
        load_s = time.time() - t0
        vram = ""
        if self.device == "cuda":
            torch.cuda.synchronize()
            vram = f", torch allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB"
        print(
            f"embeddinggemma-2 ready on {self.device} ({self.dtype_name}), "
            f"encoders dropped: {list(self.drop_encoders) or 'none'}, torch threads {self.threads}, "
            f"load {load_s:.1f}s{vram}",
            flush=True,
        )

    # -- request parsing -------------------------------------------------
    def _parse(self, body: Dict[str, Any]) -> Tuple[List[Any], List[int]]:
        raw = body.get("input")
        if raw is None:
            raise InvalidInput("'input' is required")
        items = [raw] if isinstance(raw, (str, dict)) else raw
        if not isinstance(items, list) or not items:
            raise InvalidInput("'input' must be a string, an object or a non-empty list")
        if len(items) > MAX_ITEMS:
            raise InvalidInput(f"at most {MAX_ITEMS} inputs per request")
        prompt_name = body.get("prompt_name")
        prefix = ""
        if prompt_name:
            if prompt_name not in self.prompts:
                raise InvalidInput(
                    f"unknown prompt_name {prompt_name!r}; known: {sorted(self.prompts)}"
                )
            prefix = self.prompts[prompt_name]

        parsed, tokens = [], []
        for i, item in enumerate(items):
            if isinstance(item, str):
                text = prefix + item
                parsed.append(text)
                tokens.append(len(self.tokenizer(text)["input_ids"]))
                continue
            if not isinstance(item, dict):
                raise InvalidInput(f"input[{i}] must be a string or an object")
            unknown = set(item) - {"text", *MEDIA_KEYS}
            if unknown:
                raise InvalidInput(f"input[{i}] has unknown keys {sorted(unknown)}")
            if not any(k in item for k in MEDIA_KEYS) and "text" not in item:
                raise InvalidInput(f"input[{i}] needs 'text' or one of {list(MEDIA_KEYS)}")
            for k in MEDIA_KEYS:
                if k in item and ENCODER_FOR[k] in self.drop_encoders:
                    raise InvalidInput(f"this backend was loaded without the {ENCODER_FOR[k]} encoder")
            out: Dict[str, Any] = {}
            n_tok = 0
            if "text" in item:
                if not isinstance(item["text"], str):
                    raise InvalidInput(f"input[{i}].text must be a string")
                out["text"] = item["text"]
                n_tok += len(self.tokenizer(item["text"])["input_ids"])
            if "image" in item:
                refs = item["image"] if isinstance(item["image"], list) else [item["image"]]
                imgs = [_image(r) for r in refs]
                out["image"] = imgs if len(imgs) > 1 else imgs[0]
                n_tok += IMAGE_TOKENS * len(imgs)
            if "video" in item:
                refs = item["video"] if isinstance(item["video"], list) else [item["video"]]
                vids = [_video(r) for r in refs]
                out["video"] = vids if len(vids) > 1 else vids[0]
                n_tok += sum(FRAME_TOKENS * len(v) for v in vids)
            if "audio" in item:
                samples, seconds = _audio(item["audio"])
                out["audio"] = {"array": samples, "sampling_rate": AUDIO_RATE}
                n_tok += int(round(seconds * AUDIO_TOKENS_PER_SECOND))
            # A bare media item ({"image": ...}) goes in as is; a text-only object
            # keeps its text verbatim (the caller chose the prefix).
            parsed.append(out["text"] if list(out) == ["text"] else out)
            tokens.append(n_tok)
        return parsed, tokens

    # -- job ---------------------------------------------------------------
    def process(self, job: Job) -> Result:
        import numpy as np
        import torch

        body = job.input.get("passthrough_body")
        if not isinstance(body, dict) or not body:
            body = job.input
        dims = body.get("dimensions")
        if dims is not None and dims not in ALLOWED_DIMENSIONS:
            raise InvalidInput(f"'dimensions' must be one of {list(ALLOWED_DIMENSIONS)}")
        encoding = body.get("encoding_format", "float")
        if encoding not in ("float", "base64"):
            raise InvalidInput("'encoding_format' must be 'float' or 'base64'")

        t0 = time.time()
        items, tokens = self._parse(body)
        t1 = time.time()
        with torch.inference_mode():
            vecs = self.model.encode(
                items,
                batch_size=32,
                truncate_dim=dims,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        if self.device == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        if self.device == "cuda" and sum(tokens) > RELEASE_CACHE_ABOVE_TOKENS:
            # A batch of long documents makes PyTorch's allocator grow (8 docs of
            # about 6,000 tokens: about 10 GB) and it keeps that memory cached.
            # Hand it back so a shared GPU does not stay full after one big call.
            torch.cuda.empty_cache()
        vecs = np.asarray(vecs, dtype=np.float32).reshape(len(items), -1)
        if not np.isfinite(vecs).all():
            raise RuntimeError("the model returned non-finite values")

        data = []
        for i, v in enumerate(vecs):
            if encoding == "base64":
                emb: Any = base64.b64encode(struct.pack(f"<{len(v)}f", *v.tolist())).decode()
            else:
                emb = [round(float(x), 7) for x in v]
            data.append({"object": "embedding", "index": i, "embedding": emb})
        total = int(sum(tokens))
        return Result(
            output={
                "object": "list",
                "data": data,
                "model": MODEL_ID,
                "usage": {"prompt_tokens": total, "total_tokens": total},
                # Extension fields: what the backend measured for this request.
                "timings": {
                    "decode_s": round(t1 - t0, 4),
                    "encode_s": round(t2 - t1, 4),
                    "device": self.device,
                    "dtype": self.dtype_name,
                    "threads": self.threads,
                },
            }
        )


class EmbeddingGemma2TextBackend(EmbeddingGemma2Backend):
    """Text and code only: the vision and audio encoders are never loaded (270M parameters)."""

    name = "embeddinggemma2-text"
    drop_encoders = ("vision", "audio")

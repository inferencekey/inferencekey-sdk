"""A custom **text2image** backend wrapping SANA 1.5 (diffusers).

Generates an image from a text prompt using ``SanaPipeline``
(``Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers``). SANA is a linear-
attention diffusion transformer, which is what makes it viable on GPUs where
heavier pipelines are not: the 4.8B checkpoint is ~16 GB of weights sharded in
~5 GB pieces, so the *host RAM* peak while loading stays near one shard instead
of the whole model.

That property is the reason this backend exists. On an AMD RDNA4 node
(gfx1201, R9700) sglang cannot be installed at all, and vllm-omni needs a vLLM
newer than the only build shipping ROCm kernels for that arch — so the
supported path for text2image there is a custom backend driving diffusers
directly.

Contract shape — accepts BOTH callers:

* input, native ``/process`` — ``{"prompt": "...", "steps": 20,
  "height": 1024, "width": 1024, "guidance_scale": 4.5, "seed": 1234}``
  (only ``prompt`` is required).
* input, public API (``POST .../v1/images/generations``) — the manager forwards
  the caller's OpenAI JSON in ``input.passthrough_body``, so the fields are read
  from there instead. OpenAI's ``size: "1024x1024"`` is honoured alongside the
  native ``height``/``width``.
* output — the OpenAI images shape ``{"created": ..., "data": [{"b64_json":
  "<base64 PNG>"}]}``, plus ``image_b64``/``width``/``height``/``steps``/
  ``seed``/``model`` for native callers.

The weights are **not** shipped with this example: ``setup()`` loads them from
the Hugging Face cache (downloading once if absent) and keeps the pipeline in
memory, so ``process()`` reuses it for every job.

Run it (from the SDK repo root, with this folder importable)::

    python -m inferencekey.backend.serve \\
        --port 8099 \\
        --backend backend:SanaText2ImageBackend

Pin a different checkpoint or dtype via config::

    python -m inferencekey.backend.serve \\
        --port 8099 \\
        --backend backend:SanaText2ImageBackend \\
        --config-json '{"model_id": "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"}'

Attribution: SANA by NVIDIA / Efficient-Large-Model
(https://github.com/NVlabs/Sana).
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time

import torch

from inferencekey.backend import (
    BackendContext,
    CustomBackend,
    Job,
    Result,
    pick_device,
)

#: Default checkpoint: the 1.6B variant. It is the one verified end-to-end on a
#: small-RAM node (see README) — ~9.7 GB of weights, ~40 s for the first
#: generation and ~6 s once warm. The 4.8B variant is a drop-in alternative
#: (same pipeline class, same contract) for nodes with more host RAM: set
#: ``{"model_id": "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers"}``.
_DEFAULT_MODEL_ID = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
#: SANA's native training resolution. Off-resolution requests still work but
#: quality degrades, so we default to it.
_DEFAULT_SIZE = 1024
_DEFAULT_STEPS = 20
_DEFAULT_GUIDANCE = 4.5
#: Hard ceiling so a single job can't try to allocate an absurd canvas and OOM
#: a GPU that is shared with other workloads on the same node.
_MAX_SIZE = 2048
_MAX_STEPS = 100


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


class SanaText2ImageBackend(CustomBackend):
    """SANA 1.5 text-to-image backend (prompt in, PNG out)."""

    # Declarative metadata exposed via GET /meta.
    name = "sana-text2image"
    version = "0.2.0"
    task_type = "text2image"
    requirements = "requirements.txt"
    # diffusers' SanaPipeline needs a recent torch; ROCm wheels for RDNA4
    # (gfx120X) only exist for torch >= 2.7, which is also what CUDA nodes get.
    requires_python = ">=3.9"

    def setup(self, ctx: BackendContext) -> None:
        # Some nodes preset HF_HUB_ENABLE_HF_TRANSFER=1 (the accelerated
        # downloader), which hard-fails if `hf_transfer` is absent in this
        # venv. Disable it so from_pretrained falls back to plain HTTP.
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        # Reduce allocator fragmentation: diffusion allocates and frees large
        # activation buffers every denoising step, and on a GPU shared with
        # other workloads the fragmented tail is what triggers a spurious OOM.
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        # Import here, not at module import time: the worker imports this file
        # to read the class metadata before the venv has diffusers installed.
        from diffusers import SanaPipeline

        self.device = pick_device(str(ctx.config.get("device", "auto")))
        self.model_id = str(ctx.config.get("model_id", _DEFAULT_MODEL_ID))

        # bfloat16 halves the weight footprint versus fp32 and is what the SANA
        # release targets. On CPU we fall back to float32, where bf16 kernels
        # are either missing or far slower.
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        pipe = SanaPipeline.from_pretrained(self.model_id, torch_dtype=dtype)
        pipe.to(self.device)
        # from_pretrained can leave the text encoder / VAE in their checkpoint
        # dtype; align them so no step silently upcasts and blows the VRAM
        # budget mid-generation.
        if self.device != "cpu":
            pipe.text_encoder.to(dtype)
            pipe.vae.to(dtype)

        # NOTE: deliberately NOT using enable_model_cpu_offload(). It keeps
        # weights resident in host RAM, and on a small-RAM node the amdgpu SVM
        # mapping fails ("exceeds resident system memory limit") and the process
        # dies with a segfault. Weights live in VRAM; the host only stages them.
        self.pipe = pipe
        print(f"sana pipeline loaded on {self.device}", file=sys.stderr, flush=True)

    def process(self, job: Job) -> Result:
        # Read BOTH request shapes, like the SDK's own `Job.text()` /
        # `Job.audio_bytes()` accessors do:
        #
        # * Native `/process`: the fields sit directly in `job.input`.
        # * Public API (`POST .../v1/images/generations`): the manager puts the
        #   caller's OpenAI JSON in `input.passthrough_body` and forwards it
        #   verbatim, so `input.prompt` is absent.
        #
        # Hard-coding only the first shape is how this backend 500-ed with
        # "`prompt` is required" on every call from the platform UI — the same
        # trap the audio2text example fell into.
        src = job.input
        passthrough = job.input.get("passthrough_body")
        if isinstance(passthrough, dict) and passthrough:
            src = passthrough

        prompt = str(src.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("`prompt` is required and must be a non-empty string")

        negative_prompt = src.get("negative_prompt")
        steps = _clamp(int(src.get("steps", _DEFAULT_STEPS)), 1, _MAX_STEPS)
        guidance = float(src.get("guidance_scale", _DEFAULT_GUIDANCE))

        # OpenAI's images API sizes the canvas with `size: "1024x1024"`; the
        # native shape uses separate `height`/`width`. Accept either.
        height = _clamp(int(src.get("height", _DEFAULT_SIZE)), 256, _MAX_SIZE)
        width = _clamp(int(src.get("width", _DEFAULT_SIZE)), 256, _MAX_SIZE)
        size = src.get("size")
        if isinstance(size, str) and "x" in size:
            try:
                w_raw, h_raw = size.lower().split("x", 1)
                width = _clamp(int(w_raw), 256, _MAX_SIZE)
                height = _clamp(int(h_raw), 256, _MAX_SIZE)
            except ValueError:
                raise ValueError(f"`size` must look like '1024x1024', got {size!r}")

        # A seed makes a job reproducible; without one each call is fresh.
        seed = src.get("seed")
        generator = None
        if seed is not None:
            seed = int(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # SanaPipeline builds "complex human instruction" prompt embeds and
        # trips over an explicit `negative_prompt=None`, so only pass the
        # argument when the caller actually supplied one.
        kwargs = {}
        if negative_prompt:
            kwargs["negative_prompt"] = str(negative_prompt)

        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance_scale=guidance,
                generator=generator,
                **kwargs,
            ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # Answer in the OpenAI images shape (`data: [{b64_json}]`) so anything
        # speaking that API — the platform's endpoint tester, the OpenAI SDKs —
        # renders the result without special-casing this backend. The manager
        # also counts `data.len()` from this shape for billing units
        # (`count_images_in_response`). `image_b64` and the size/seed fields are
        # kept alongside it for native `/process` callers.
        return Result(
            {
                "created": int(time.time()),
                "data": [{"b64_json": image_b64}],
                "image_b64": image_b64,
                "width": image.width,
                "height": image.height,
                "steps": steps,
                "seed": seed,
                "model": self.model_id,
            }
        )

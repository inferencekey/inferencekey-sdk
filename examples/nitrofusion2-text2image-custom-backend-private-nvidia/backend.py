"""A custom **text2image** backend for NitroFusion2 (1 and 2 step images).

NitroFusion2 (Dar-Yen Chen and Yi-Zhe Song, Apache-2.0) is a 6.2B one- and
two-step distillation of Tongyi-MAI's Z-Image-Turbo. It keeps Z-Image-Turbo's
pipeline (Qwen3 text encoder, Flux-style VAE, flow-matching scheduler) and
swaps in its own diffusion transformer. For two steps the authors add a
"de-distillation" LoRA at scale 0.5 on the first step only.

No built-in InferenceKey backend serves it. This backend runs the diffusers
path from the authors' ``inference.py``: it downloads the NitroFusion2
transformer and LoRA plus Z-Image-Turbo's text encoder, tokenizer, VAE and
scheduler (not Z-Image-Turbo's own transformer), loads the pipeline once and
generates one image per request.

Contract shape, accepts BOTH callers:

* input, native ``/process``: ``{"prompt": "...", "size": "1024x1024",
  "steps": 2, "seed": 7}`` (only ``prompt`` is required). Optional:
  ``width``/``height`` (instead of ``size``), ``steps`` (or
  ``num_inference_steps``; 1 to 4, default 2), ``dedistill`` (default true:
  the authors' LoRA schedule for 2 to 4 steps; ignored at 1 step),
  ``output_format`` (``png`` default, ``jpeg`` or ``webp``).
* input, public API (``POST .../v1/images/generations``): the manager forwards
  the caller's OpenAI JSON in ``input.passthrough_body``; fields are read from
  there.
* output: the OpenAI images shape ``{"created": ..., "data": [{"b64_json":
  ...}]}`` plus ``width``/``height``/``seed``/``steps``, ``seconds`` (GPU
  time of the whole generation), ``encode_seconds`` (text encoder),
  ``vae_seconds`` (VAE decode), ``vram_peak_gib`` (torch peak allocated) and
  the memory mode.

Memory (``NF2_MEMORY_MODE``, default ``auto``):

* ``gpu``: everything resident (about 21 GB of bf16 weights: 12.3 GB
  transformer, 8 GB text encoder, VAE). Fits a 24 GB card.
* ``model``: diffusers ``enable_model_cpu_offload()`` (what the authors' own
  script uses): one component on the GPU at a time.
* ``sequential``: diffusers ``enable_sequential_cpu_offload()``: weights
  stream layer by layer. Slowest, smallest.

``auto`` picks ``gpu`` when at least 23 GiB are free, otherwise ``model``.
``NF2_TRANSFORMER`` picks the transformer weights: ``bf16`` (default, the
diffusers folder) or the name of one of the authors' GGUF files (for example
``nf2-Q4_K_M.gguf``), loaded with diffusers' GGUF support.
``NF2_VRAM_LIMIT_GIB`` caps the memory this process may use
(``torch.cuda.set_per_process_memory_fraction``), which is how we emulated
smaller cards on a 24 GB one.

Run it by hand (from this folder, with requirements installed)::

    python -m inferencekey.backend.serve --port 8099 \\
        --backend backend:NitroFusion2Backend

Attribution: NitroFusion2 by Dar-Yen Chen and Yi-Zhe Song, Apache-2.0
(https://huggingface.co/ChenDY/NitroFusion2, arXiv 2611.05648); Z-Image-Turbo
by Tongyi-MAI (Alibaba), Apache-2.0 (https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).
"""

from __future__ import annotations

import base64
import io
import os
import sys
import threading
import time
from pathlib import Path

from inferencekey.backend import BackendContext, CustomBackend, Job, Result, pick_device

try:  # newer SDKs answer 400 for bad client input
    from inferencekey.backend import InvalidInput
except ImportError:  # pragma: no cover - older SDK
    InvalidInput = ValueError

#: Repos and the revisions this backend was tested with (2026-11-12).
NF2_REPO = "ChenDY/NitroFusion2"
NF2_REVISION = "62d1a7000af9c46a0c4348d936bd7670ff33c1b9"
BASE_REPO = "Tongyi-MAI/Z-Image-Turbo"
BASE_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
DEDISTILL_LORA = "nf2-dedistill-lora.safetensors"

#: The authors' schedule (inference.py): scheduler shift and LoRA scale per step.
SHIFT = {1: 3.0, 2: 6.0, 3: 8.0, 4: 9.0}
LORA_SCALES = {1: (0.0,), 2: (0.5, 0.0), 3: (0.5, 0.6, 0.0), 4: (0.5, 0.6, 0.7, 0.0)}

_MIN_SIDE, _MAX_SIDE = 256, 2048
_GPU_MIN_GIB = 23.0
#: The result travels through an 8 MB message: keep the base64 under this.
_MAX_B64_BYTES = 7_500_000


class NitroFusion2Backend(CustomBackend):
    """NitroFusion2 text-to-image backend (prompt in, image out, 1 to 4 steps)."""

    name = "nitrofusion2-text2image"
    version = "0.1.0"
    task_type = "text2image"
    requirements = "requirements.txt"
    requires_python = ">=3.11,<3.13"

    def setup(self, ctx: BackendContext) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        t0 = time.time()

        import torch
        from diffusers import ZImagePipeline, ZImageTransformer2DModel
        from huggingface_hub import hf_hub_download, snapshot_download

        self.torch = torch
        self.lock = threading.Lock()
        self.device = torch.device(pick_device(str(ctx.config.get("device", "auto"))))
        if self.device.type != "cuda":
            raise RuntimeError("NitroFusion2 needs a CUDA GPU")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        budget_gib = self._vram_budget_gib()
        self.transformer_kind = (os.environ.get("NF2_TRANSFORMER") or "bf16").strip()

        # Only what the pipeline needs: NOT Z-Image-Turbo's own transformer.
        base = Path(
            snapshot_download(
                BASE_REPO,
                revision=BASE_REVISION,
                allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "vae/*"],
                max_workers=16,
            )
        )
        patterns = ["transformer/config.json", DEDISTILL_LORA]
        if self.transformer_kind == "bf16":
            patterns.append("transformer/*")
        nf2 = Path(snapshot_download(NF2_REPO, revision=NF2_REVISION, allow_patterns=patterns, max_workers=16))
        gguf_path = None
        if self.transformer_kind != "bf16":
            gguf_path = hf_hub_download(NF2_REPO, self.transformer_kind, revision=NF2_REVISION)
        t_download = time.time()

        if gguf_path is None:
            transformer = ZImageTransformer2DModel.from_pretrained(
                str(nf2), subfolder="transformer", torch_dtype=torch.bfloat16
            )
        else:
            from diffusers import GGUFQuantizationConfig

            transformer = ZImageTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                config=str(nf2),
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            )
        pipe = ZImagePipeline.from_pretrained(str(base), transformer=transformer, torch_dtype=torch.bfloat16)
        pipe.load_lora_weights(str(nf2), weight_name=DEDISTILL_LORA, adapter_name="dedistill")
        self.base_scheduler_config = dict(pipe.scheduler.config)

        self.memory_mode = self._pick_mode(budget_gib)
        if self.memory_mode == "gpu":
            pipe.to(self.device)
        elif self.memory_mode == "model":
            pipe.enable_model_cpu_offload(device=self.device)
        else:
            pipe.enable_sequential_cpu_offload(device=self.device)
        self.pipe = pipe
        self._time_vae()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        loaded_gib = torch.cuda.memory_allocated() / 2**30
        print(
            f"nitrofusion2 ready on {self.device}: download {t_download - t0:.1f}s, "
            f"load {time.time() - t_download:.1f}s, transformer {self.transformer_kind}, "
            f"memory mode {self.memory_mode} (budget {budget_gib:.1f} GiB), weights in VRAM {loaded_gib:.1f} GiB",
            file=sys.stderr,
            flush=True,
        )

    def _vram_budget_gib(self) -> float:
        """GiB this process may use on the GPU (``NF2_VRAM_LIMIT_GIB`` caps it)."""
        torch = self.torch
        index = self.device.index
        free, total = torch.cuda.mem_get_info(index)
        limit = float(os.environ.get("NF2_VRAM_LIMIT_GIB") or 0)
        if limit > 0:
            # Behave like a smaller card: the allocator refuses to go past it.
            torch.cuda.set_per_process_memory_fraction(min(1.0, limit * 2**30 / total), index)
            return min(limit, free / 2**30)
        return free / 2**30

    def _pick_mode(self, budget_gib: float) -> str:
        mode = (os.environ.get("NF2_MEMORY_MODE") or "auto").lower()
        if mode == "auto":
            return "gpu" if budget_gib >= _GPU_MIN_GIB else "model"
        if mode not in ("gpu", "model", "sequential"):
            raise ValueError(f"NF2_MEMORY_MODE must be auto, gpu, model or sequential, got {mode!r}")
        return mode

    def _time_vae(self) -> None:
        """Wrap the VAE decode so each request can report its share of the time."""
        vae = self.pipe.vae
        original = vae.decode
        torch = self.torch
        self._vae_seconds = 0.0

        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            t = time.time()
            out = original(*args, **kwargs)
            torch.cuda.synchronize()
            self._vae_seconds += time.time() - t
            return out

        vae.decode = timed

    def process(self, job: Job) -> Result:
        torch = self.torch
        from diffusers import FlowMatchEulerDiscreteScheduler

        src = job.input
        passthrough = job.input.get("passthrough_body")
        if isinstance(passthrough, dict) and passthrough:
            src = passthrough

        prompt = str(src.get("prompt") or "").strip()
        if not prompt:
            raise InvalidInput("`prompt` is required")
        width = int(src.get("width") or 1024)
        height = int(src.get("height") or 1024)
        size = src.get("size")
        if isinstance(size, str) and "x" in size.lower():
            try:
                w_raw, h_raw = size.lower().split("x", 1)
                width, height = int(w_raw), int(h_raw)
            except ValueError:
                raise InvalidInput(f"`size` must look like '1024x1024', got {size!r}")
        if not (_MIN_SIDE <= width <= _MAX_SIDE and _MIN_SIDE <= height <= _MAX_SIDE):
            raise InvalidInput(f"width and height must be {_MIN_SIDE} to {_MAX_SIDE}, got {width}x{height}")
        if width % 16 or height % 16:
            raise InvalidInput("width and height must be multiples of 16")
        if int(src.get("n", 1)) != 1:
            raise InvalidInput("`n` must be 1 (one image per request)")
        steps = int(src.get("steps") or src.get("num_inference_steps") or 2)
        if steps not in SHIFT:
            raise InvalidInput("NitroFusion2 is distilled for 1 to 4 steps")
        dedistill = src.get("dedistill")
        dedistill = True if dedistill is None else bool(dedistill)
        fmt = str(src.get("output_format") or "png").lower()
        if fmt not in ("png", "jpeg", "jpg", "webp"):
            raise InvalidInput(f"`output_format` must be png, jpeg or webp, got {fmt!r}")
        seed = src.get("seed")
        seed = int(seed) if seed is not None else int.from_bytes(os.urandom(2), "big")

        pipe = self.pipe
        scales = LORA_SCALES[steps] if dedistill else (0.0,) * steps
        sigmas = [1.0 - i / steps for i in range(steps)]
        with self.lock, torch.inference_mode():  # one generation at a time
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.base_scheduler_config, shift=SHIFT[steps])
            pipe.set_adapters("dedistill", scales[0])

            def callback(p, i, t, kwargs):
                if i + 1 < steps:
                    p.set_adapters("dedistill", scales[i + 1])
                return kwargs

            torch.cuda.reset_peak_memory_stats()
            self._vae_seconds = 0.0
            torch.cuda.synchronize()
            t0 = time.time()
            embeds = pipe.encode_prompt(prompt, device=pipe._execution_device, do_classifier_free_guidance=False)[0]
            torch.cuda.synchronize()
            t_enc = time.time() - t0
            image = pipe(
                prompt_embeds=embeds,
                height=height,
                width=width,
                num_inference_steps=steps,
                sigmas=sigmas,
                guidance_scale=0.0,
                generator=torch.Generator("cuda").manual_seed(seed),
                callback_on_step_end=callback,
            ).images[0]
            torch.cuda.synchronize()
            seconds = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            peak_reserved = torch.cuda.max_memory_reserved() / 2**30
            vae_s = self._vae_seconds
            pipe.set_adapters("dedistill", 0.0)

        payload, used_fmt = self._encode(image, fmt)
        return Result(
            {
                "created": int(time.time()),
                "data": [{"b64_json": payload}],
                "width": image.width,
                "height": image.height,
                "seed": seed,
                "steps": steps,
                "dedistill": dedistill and steps > 1,
                "seconds": round(seconds, 3),
                "encode_seconds": round(t_enc, 3),
                "vae_seconds": round(vae_s, 3),
                "vram_peak_gib": round(peak, 2),
                "vram_peak_reserved_gib": round(peak_reserved, 2),
                "memory_mode": self.memory_mode,
                "transformer": self.transformer_kind,
                "output_format": used_fmt,
                "model": NF2_REPO,
            }
        )

    @staticmethod
    def _encode(image, fmt: str):
        buf = io.BytesIO()
        if fmt == "png":
            image.save(buf, format="PNG")
        elif fmt == "webp":
            image.save(buf, format="WEBP", quality=92, method=4)
        else:
            fmt = "jpeg"
            image.save(buf, format="JPEG", quality=95)
        data = base64.b64encode(buf.getvalue())
        if len(data) > _MAX_B64_BYTES:
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=90)
            data, fmt = base64.b64encode(buf.getvalue()), "jpeg"
        return data.decode("ascii"), fmt

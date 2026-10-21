"""A custom **text2image** backend wrapping MOSS-Image-SFT (OpenMOSS).

MOSS-Image is a 5B single-stream diffusion transformer from OpenMOSS
(Apache-2.0). It conditions on a Qwen3.5-9B text encoder and decodes with the
FLUX.2 small VAE. Its strengths are text rendering (English and Chinese,
multi-line, infographics) and **JSON layout prompts** with bounding boxes for
each text block or object.

No built-in InferenceKey backend serves it: the authors ship it as a diffusers
*custom pipeline* (``custom_pipeline="pipeline_moss_image"``,
``trust_remote_code=True``), so this backend loads that pipeline directly.

Two ways to sample, chosen PER REQUEST on one loaded model:

* ``"mode": "sft"`` (default) - the release recipe: 50 Euler steps, CFG 3,
  negative branch with block 22 skipped (two transformer forwards per step).
* ``"mode": "turbo"`` (also picked when ``mode`` is absent and the request asks
  for ``guidance_scale`` 1 with 16 steps or fewer) - the community 8-step DMD distillation by lodestones
  (``lodestones/MOSS-Image-SFT-8-steps-dmd``, ``lora_latest.safetensors``),
  applied UNMERGED at strength 0.7 on the bf16 transformer, 8 Euler steps and
  no CFG (one forward per step). This is the same route as the authors' demo
  Space for that LoRA. Set ``MOSS_TURBO_LORA=0`` to skip loading it.

Contract shape, accepts BOTH callers:

* input, native ``/process``: ``{"prompt": "...", "size": "1024x1024",
  "seed": 7, "mode": "sft", "output_format": "png"}`` (only ``prompt`` is
  required). ``prompt`` may also be a JSON layout object (it is serialized with
  ``ensure_ascii=False``, as the user guide asks). Optional: ``height``,
  ``width``, ``steps`` (or ``num_inference_steps``), ``guidance_scale``,
  ``negative_prompt``,
  ``lora_strength`` (turbo), ``n`` (1 for PNG, up to 4 for JPEG/WebP: the result
  travels through an 8 MB message limit).
* input, public API (``POST .../v1/images/generations``): the manager forwards
  the caller's OpenAI JSON in ``input.passthrough_body``; fields are read from
  there.
* output: the OpenAI images shape ``{"created": ..., "data": [{"b64_json":
  ...}]}`` plus ``width``/``height``/``seed``/``steps``/``mode``/``seconds``
  (GPU time of the generation) and ``vram_peak_gib``.

Nothing heavy ships in the package: ``setup()`` downloads the pinned model
revision (about 27 GB: bf16 transformer 10.2 GB, Qwen3.5-9B text encoder
16.8 GB, VAE 0.25 GB) and the 1.2 GB LoRA into the Hugging Face cache.

Run it by hand (from this folder, with requirements installed)::

    python -m inferencekey.backend.serve --port 8099 \\
        --backend backend:MossImageText2ImageBackend

Attribution: MOSS-Image by OpenMOSS / MOSI, Apache-2.0
(https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT, arXiv:2610.23374).
8-step DMD LoRA by lodestones, Apache-2.0
(https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd).
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time

from inferencekey.backend import BackendContext, CustomBackend, Job, Result, pick_device

try:  # newer SDKs answer 400 for bad client input
    from inferencekey.backend import InvalidInput
except ImportError:  # pragma: no cover - older SDK
    InvalidInput = ValueError

#: Model repo and the revision this backend was tested with (2026-10-21).
_MODEL_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
_MODEL_REVISION = "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9"
#: Community 8-step DMD LoRA (kohya keys, rank 128 on every block projection).
_LORA_REPO = "lodestones/MOSS-Image-SFT-8-steps-dmd"
_LORA_REVISION = "f95df83b5e90ae91ea412434b3f9b7f1c8c6314f"
_LORA_FILE = "lora_latest.safetensors"

_DEFAULTS = {
    #       steps, cfg, lora strength
    "sft": (50, 3.0, 0.0),
    "turbo": (8, 1.0, 0.7),
}
#: The user guide recommends 1 to 2 megapixels and allows 0.25 to 4.
_MIN_SIDE, _MAX_SIDE, _MAX_PIXELS, _ALIGN = 256, 2048, 4 * 1024 * 1024, 16
_MAX_N = 4
_FORMATS = {"png": "PNG", "jpeg": "JPEG", "jpg": "JPEG", "webp": "WEBP"}


def _align(value: int) -> int:
    value = max(_MIN_SIDE, min(_MAX_SIDE, int(value)))
    return int(round(value / _ALIGN) * _ALIGN)


class MossImageText2ImageBackend(CustomBackend):
    """MOSS-Image-SFT text-to-image backend (prompt or JSON layout in, image out)."""

    name = "moss-image-text2image"
    version = "0.2.0"
    task_type = "text2image"
    requirements = "requirements.txt"
    requires_python = ">=3.10"

    def setup(self, ctx: BackendContext) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        t0 = time.time()

        import torch
        from torch import nn
        from torch.nn import functional as F
        from diffusers import DiffusionPipeline
        from huggingface_hub import hf_hub_download, snapshot_download

        self.torch = torch
        self.lock = threading.Lock()
        self.device = torch.device(pick_device(str(ctx.config.get("device", "auto"))))

        local = snapshot_download(
            _MODEL_REPO,
            revision=_MODEL_REVISION,
            ignore_patterns=["assets/*", "*.webp", "*.svg"],
            max_workers=16,
        )
        lora_path = None
        if os.environ.get("MOSS_TURBO_LORA", "1") != "0":
            lora_path = hf_hub_download(_LORA_REPO, _LORA_FILE, revision=_LORA_REVISION)
        t_download = time.time()

        pipe = DiffusionPipeline.from_pretrained(
            local,
            custom_pipeline="pipeline_moss_image",
            trust_remote_code=True,
            torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
        )
        # Same as the model card: a fresh scheduler built from the shipped config.
        pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)

        self.lora_layers = 0
        if lora_path:
            from safetensors.torch import load_file

            class LoRALinear(nn.Module):
                """y = W x + strength * (alpha / rank) * up(down(x)), kept unmerged.

                The distillation delta is below bf16 rounding, so merging it into
                the bf16 weights would throw most of it away.
                """

                strength = 0.0

                def __init__(self, base, down, up, alpha):
                    super().__init__()
                    self.base = base
                    self.register_buffer("down", down.to(base.weight.dtype), persistent=False)
                    self.register_buffer("up", up.to(base.weight.dtype), persistent=False)
                    self.scale = alpha / down.shape[0]

                @property
                def weight(self):  # the pipeline reads proj[...].weight.dtype
                    return self.base.weight

                def forward(self, x):
                    y = self.base(x)
                    if LoRALinear.strength == 0.0:
                        return y
                    return y + F.linear(F.linear(x, self.down), self.up) * (
                        self.scale * LoRALinear.strength
                    )

            state = load_file(lora_path)
            prefix = "diffusion_model."
            names = sorted(
                k[len(prefix):-len(".lora_down.weight")]
                for k in state
                if k.endswith(".lora_down.weight")
            )
            for name in names:
                parent_name, _, child = name.rpartition(".")
                parent = pipe.transformer.get_submodule(parent_name)
                base = parent[child] if isinstance(parent, nn.ModuleDict) else getattr(parent, child)
                key = prefix + name
                wrapped = LoRALinear(
                    base,
                    state[f"{key}.lora_down.weight"],
                    state[f"{key}.lora_up.weight"],
                    float(state[f"{key}.alpha"]),
                )
                if isinstance(parent, nn.ModuleDict):
                    parent[child] = wrapped
                else:
                    setattr(parent, child, wrapped)
            del state
            self.LoRALinear = LoRALinear
            self.lora_layers = len(names)

        self.pipe = pipe.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t_ready = time.time()
        loaded_gib = (
            torch.cuda.memory_allocated() / 2**30 if self.device.type == "cuda" else 0.0
        )
        print(
            f"moss-image ready on {self.device}: download {t_download - t0:.1f}s, "
            f"load {t_ready - t_download:.1f}s, lora layers {self.lora_layers}, "
            f"weights in VRAM {loaded_gib:.1f} GiB",
            file=sys.stderr,
            flush=True,
        )

    def process(self, job: Job) -> Result:
        torch = self.torch
        src = job.input
        passthrough = job.input.get("passthrough_body")
        if isinstance(passthrough, dict) and passthrough:
            src = passthrough

        prompt = src.get("prompt")
        if isinstance(prompt, (dict, list)):
            # A JSON layout: the user guide serializes it with ensure_ascii=False.
            prompt = json.dumps(prompt, ensure_ascii=False)
        prompt = str(prompt or "").strip()
        if not prompt:
            raise InvalidInput("`prompt` is required: a string or a JSON layout object")

        # `steps` or the OpenAI-images-style `num_inference_steps` (the panel's
        # image tester sends the latter, plus `guidance_scale`).
        steps_in = src.get("steps") or src.get("num_inference_steps")
        cfg_in = src.get("guidance_scale")
        mode = src.get("mode")
        if not mode:
            # No explicit mode: a few-step, no-CFG request is a turbo request
            # (the SFT weights alone need CFG and about 50 steps).
            few_step = steps_in is not None and int(steps_in) <= 16
            no_cfg = cfg_in is not None and float(cfg_in) == 1.0
            mode = "turbo" if self.lora_layers and few_step and no_cfg else "sft"
        mode = str(mode).lower()
        if mode not in _DEFAULTS:
            raise InvalidInput(f"`mode` must be 'sft' or 'turbo', got {mode!r}")
        if mode == "turbo" and not self.lora_layers:
            raise InvalidInput("turbo mode needs the DMD LoRA (MOSS_TURBO_LORA=0 disabled it)")
        d_steps, d_cfg, d_strength = _DEFAULTS[mode]

        height = int(src.get("height", 1024))
        width = int(src.get("width", 1024))
        size = src.get("size")
        if isinstance(size, str) and "x" in size:
            try:
                w_raw, h_raw = size.lower().split("x", 1)
                width, height = int(w_raw), int(h_raw)
            except ValueError:
                raise InvalidInput(f"`size` must look like '1024x1024', got {size!r}")
        height, width = _align(height), _align(width)
        if height * width > _MAX_PIXELS:
            raise InvalidInput(f"{width}x{height} is over 4 megapixels, the model's limit")

        steps = max(1, min(100, int(steps_in or d_steps)))
        cfg = float(cfg_in if cfg_in is not None else d_cfg)
        strength = float(src.get("lora_strength", d_strength)) if mode == "turbo" else 0.0
        negative = str(src.get("negative_prompt") or "")
        n = max(1, min(_MAX_N, int(src.get("n", 1))))
        fmt_key = str(src.get("output_format") or "png").lower()
        if fmt_key not in _FORMATS:
            raise InvalidInput(f"`output_format` must be png, jpeg or webp, got {fmt_key!r}")
        fmt = _FORMATS[fmt_key]
        if fmt == "PNG" and n > 1:
            raise InvalidInput("n > 1 needs output_format jpeg or webp (8 MB result limit)")
        seed = src.get("seed")
        seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "big") >> 1

        with self.lock:  # one generation at a time; the LoRA strength is shared state
            if self.lora_layers:
                self.LoRALinear.strength = strength
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            images = self.pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=cfg,
                negative_prompt=negative,
                num_images_per_prompt=n,
                # The release default: CFG's negative branch skips block 22.
                uncond_skip_layers=[22] if cfg != 1.0 else None,
                # Noise drawn on the CPU: the same seed gives the same image on any GPU.
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            seconds = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30 if self.device.type == "cuda" else 0.0

        data = []
        for image in images:
            buf = io.BytesIO()
            if fmt == "PNG":
                image.save(buf, format="PNG")
            else:
                image.convert("RGB").save(buf, format=fmt, quality=92)
            data.append({"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")})

        return Result(
            {
                "created": int(time.time()),
                "data": data,
                "width": width,
                "height": height,
                "seed": seed,
                "steps": steps,
                "guidance_scale": cfg,
                "mode": mode,
                "lora_strength": strength,
                "seconds": round(seconds, 3),
                "vram_peak_gib": round(peak, 2),
                "output_format": fmt.lower(),
                "model": _MODEL_REPO,
            }
        )

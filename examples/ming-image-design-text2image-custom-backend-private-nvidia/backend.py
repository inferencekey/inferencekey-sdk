"""A custom **text2image** backend wrapping Ming-Image-0.1-Design (inclusionAI).

Ming-Image-0.1-Design is an MIT-licensed text-to-image model for text-rich
visual design: UI mockups, posters, infographics. Its image head is a 6B
diffusion transformer conditioned on the hidden states of a Ling MoE
multimodal LLM (through a small Qwen2 connector), decoding with a 4-channel
Qwen-Image VAE, so every image comes out as **RGBA**: with one of the
authors' trigger phrases at the start of the prompt the background is
transparent (alpha 0), otherwise the alpha channel is opaque.

No built-in InferenceKey backend serves it. The authors ship a Python
inference path (``infer.py`` in https://github.com/inclusionAI/Ming-Image);
this backend downloads that code at a pinned commit, loads the model once with
the authors' own loader and calls their ``run_generation`` for each request.

Contract shape, accepts BOTH callers:

* input, native ``/process``: ``{"prompt": "...", "size": "1024x1024",
  "seed": 7, "background": "transparent"}`` (only ``prompt`` is required).
  Optional: ``width``/``height`` (instead of ``size``), ``steps`` (or
  ``num_inference_steps``, default 12), ``guidance_scale`` (default 1.0),
  ``background`` (``transparent`` prepends the authors' RGBA phrase unless the
  prompt already starts with one; ``auto``/``opaque`` leave the prompt alone),
  ``output_format`` (``png`` default, or ``webp``; both keep the alpha
  channel). The model renders in a 1024 or a 2048 bucket (nearest to
  sqrt(width x height)) and resizes to the requested size.
* input, public API (``POST .../v1/images/generations``): the manager forwards
  the caller's OpenAI JSON in ``input.passthrough_body``; fields are read from
  there. ``background`` is the same field OpenAI's images API uses.
* output: the OpenAI images shape ``{"created": ..., "data": [{"b64_json":
  ...}]}`` plus ``width``/``height``/``bucket``/``seed``/``steps``/
  ``seconds`` (GPU time of the generation), ``vram_peak_gib`` and
  ``transparent_fraction`` (share of pixels with alpha 0). A PNG whose base64
  would not fit the 8 MB message limit is re-encoded as lossless WebP, then as
  lossy WebP at a falling quality (``output_format`` says which).

Memory (``MING_MEMORY_MODE``, default ``auto``):

* ``gpu``: everything resident, as the authors validated (one 80 GB card).
* ``lowvram``: the MoE LLM layers (about 34 GB of the 53 GB) stay in CPU RAM
  and stream to the GPU once per prompt; diffusion transformer, connector and
  VAE stay resident.
* ``minimal``: as ``lowvram``, plus the connector streams from CPU RAM, only as many diffusion blocks as fit stay resident, and the VAE
  decodes in tiles.

``auto`` picks by the free VRAM. ``MING_VRAM_LIMIT_GIB`` caps the memory this
process may use (``torch.cuda.set_per_process_memory_fraction``), which is how
we emulated smaller cards. The offloaded modes need about 45 GB of free
system RAM.

Run it by hand (from this folder, with requirements installed)::

    python -m inferencekey.backend.serve --port 8099 \\
        --backend backend:MingImageDesignBackend

Attribution: Ming-Image-0.1-Design by inclusionAI (Ant Group), MIT
(https://huggingface.co/inclusionAI/Ming-Image-0.1-Design); inference code
from https://github.com/inclusionAI/Ming-Image (MIT).
"""

from __future__ import annotations

import base64
import io
import os
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

from inferencekey.backend import BackendContext, CustomBackend, Job, Result, pick_device

try:  # newer SDKs answer 400 for bad client input
    from inferencekey.backend import InvalidInput
except ImportError:  # pragma: no cover - older SDK
    InvalidInput = ValueError

#: Model repo and the revision this backend was tested with (2026-10-23).
_MODEL_REPO = "inclusionAI/Ming-Image-0.1-Design"
_MODEL_REVISION = "c57c49689d12ef7d4e24a170014f1dc43d3ce258"
#: The authors' inference code (infer.py and the model classes), pinned.
_CODE_REPO = "inclusionAI/Ming-Image"
_CODE_COMMIT = "3515cd7b1e3049fc517eed7d35020251cd06519a"

#: The authors' fixed RGBA trigger phrases ("place exactly one at the start").
RGBA_PHRASES = (
    "带透明通道，4通道RGBA图像",
    "透明背景，alpha通道，无底图",
    "抠图素材，背景alpha=0",
    "孤立主体，透明PNG图层",
    "不要白底，不要棋盘格，只要透明通道",
    "RGBA, 4-channel, transparent background",
    "isolated subject, alpha matte, no background",
    "cutout PNG, alpha=0 outside the object",
    "transparent canvas, not white, not checkerboard",
    "production RGBA layer for compositing",
)
_DEFAULT_RGBA_PHRASE = "RGBA, 4-channel, transparent background"

_DEFAULT_STEPS, _DEFAULT_CFG = 12, 1.0
_MIN_SIDE, _MAX_SIDE = 256, 4096
#: The result travels through an 8 MB message: keep the base64 under this.
_MAX_B64_BYTES = 7_500_000
#: Memory mode thresholds (GiB this process may use), see ``_device_map``.
_GPU_MIN_GIB = 60.0
_LOWVRAM_MIN_GIB = 24.0
_ACTIVATION_RESERVE_GIB = 6.0


def _fetch_code() -> Path:
    """Download the authors' inference code at the pinned commit (once)."""
    root = Path(os.environ.get("MING_CODE_DIR") or Path.home() / ".cache" / "ming-image-code")
    target = root / _CODE_COMMIT
    marker = target / "infer.py"
    if not marker.is_file():
        root.mkdir(parents=True, exist_ok=True)
        url = f"https://codeload.github.com/{_CODE_REPO}/tar.gz/{_CODE_COMMIT}"
        tmp = root / f"{_CODE_COMMIT}.tar.gz"
        urllib.request.urlretrieve(url, tmp)
        staging = root / f"{_CODE_COMMIT}.staging"
        with tarfile.open(tmp) as tar:
            tar.extractall(staging, filter="data")
        (inner,) = list(staging.iterdir())
        inner.rename(target)
        staging.rmdir()
        tmp.unlink()
    return target


class MingImageDesignBackend(CustomBackend):
    """Ming-Image-0.1-Design text-to-image backend (prompt in, RGBA image out)."""

    name = "ming-image-design-text2image"
    version = "0.1.4"
    task_type = "text2image"
    requirements = "requirements.txt"
    requires_python = ">=3.11,<3.12"

    def setup(self, ctx: BackendContext) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        t0 = time.time()

        code = _fetch_code()
        if str(code) not in sys.path:
            sys.path.insert(0, str(code))

        import torch
        from huggingface_hub import snapshot_download

        import infer  # the authors' module, from the pinned code
        from inference_profile import load_checkpoint_capabilities
        from mllm_device_map import build_mllm_device_plan, load_mllm_num_hidden_layers
        from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
        from processing_bailingmm2 import load_bailingmm2_processor

        self.torch = torch
        self.infer = infer
        self.lock = threading.Lock()
        self.device = torch.device(pick_device(str(ctx.config.get("device", "auto"))))
        if self.device.type != "cuda":
            raise RuntimeError("Ming-Image-0.1-Design needs a CUDA GPU")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        budget_gib = self._vram_budget_gib()

        local = Path(
            snapshot_download(
                _MODEL_REPO,
                revision=_MODEL_REVISION,
                ignore_patterns=["assets/*", "*.webp"],
                max_workers=16,
            )
        )
        t_download = time.time()

        self.profile = load_checkpoint_capabilities(local)
        self.processor = load_bailingmm2_processor(local / "mllm")
        n_layers = load_mllm_num_hidden_layers(local / "mllm")
        device_map = dict(build_mllm_device_plan(n_layers, 1).device_map)
        self.memory_mode = self._pick_mode(budget_gib)
        index = self.device.index
        for key in list(device_map):
            device_map[key] = index
        if self.memory_mode in ("lowvram", "minimal"):
            for key in device_map:
                if key.startswith("model.model.layers."):
                    device_map[key] = "cpu"

        model = BailingMM2NativeForConditionalGeneration.from_pretrained(
            str(local),
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # the authors' default; no flash-attn needed
            load_image_gen=True,
            image_gen_device=str(self.device),
            device_map=device_map,
        )
        model.eval()
        detail = f"budget {budget_gib:.1f} GiB"
        if self.memory_mode == "lowvram":
            detail += ", LLM layers streamed from CPU RAM"
        elif self.memory_mode == "minimal":
            detail += ", " + self._squeeze(model, budget_gib)
        self.model = model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        loaded_gib = torch.cuda.memory_allocated() / 2**30
        print(
            f"ming-image ready on {self.device}: download {t_download - t0:.1f}s, "
            f"load {time.time() - t_download:.1f}s, memory mode {self.memory_mode} ({detail}), "
            f"weights in VRAM {loaded_gib:.1f} GiB",
            file=sys.stderr,
            flush=True,
        )

    def _vram_budget_gib(self) -> float:
        """GiB this process may use on the GPU (``MING_VRAM_LIMIT_GIB`` caps it)."""
        torch = self.torch
        index = self.device.index
        free, total = torch.cuda.mem_get_info(index)
        limit = float(os.environ.get("MING_VRAM_LIMIT_GIB") or 0)
        if limit > 0:
            # Behave like a smaller card: the allocator refuses to go past it.
            torch.cuda.set_per_process_memory_fraction(min(1.0, limit * 2**30 / total), index)
            return min(limit, free / 2**30)
        return free / 2**30 - 0.5

    def _pick_mode(self, budget_gib: float) -> str:
        mode = (os.environ.get("MING_MEMORY_MODE") or "auto").lower()
        if mode == "auto":
            if budget_gib >= _GPU_MIN_GIB:
                return "gpu"
            return "lowvram" if budget_gib >= _LOWVRAM_MIN_GIB else "minimal"
        if mode not in ("gpu", "lowvram", "minimal"):
            raise ValueError(f"MING_MEMORY_MODE must be auto, gpu, lowvram or minimal, got {mode!r}")
        return mode

    def _squeeze(self, model, budget_gib: float) -> str:
        """``minimal`` mode: stream the connector and part of the diffusion transformer."""
        from accelerate import cpu_offload

        torch = self.torch
        device = self.device
        connector = getattr(model, "connector", None)
        if connector is not None:
            cpu_offload(connector, execution_device=device)
        generator = model.diffusion_loss
        transformer = generator.train_model.transformer
        if hasattr(generator.vae, "enable_tiling"):
            generator.vae.enable_tiling()
        torch.cuda.empty_cache()
        layers = list(transformer.layers)
        block_gib = sum(p.numel() * p.element_size() for p in layers[0].parameters()) / 2**30
        used_gib = torch.cuda.memory_allocated(device) / 2**30
        reserve = float(os.environ.get("MING_ACTIVATION_RESERVE_GIB") or _ACTIVATION_RESERVE_GIB)
        layers_gib = block_gib * len(layers)
        # Everything else stays; keep `reserve` free for activations.
        room = budget_gib - reserve - (used_gib - layers_gib)
        resident = max(0, min(len(layers), int(room / block_gib)))
        # Offload from the end, so the first resident blocks keep the device
        # that the generator reads from its first parameter.
        for layer in layers[resident:]:
            cpu_offload(layer, execution_device=device)
        torch.cuda.empty_cache()
        return (
            f"LLM layers and connector streamed, {resident}/{len(layers)} "
            f"diffusion blocks resident, VAE tiling"
        )

    def process(self, job: Job) -> Result:
        torch = self.torch
        src = job.input
        passthrough = job.input.get("passthrough_body")
        if isinstance(passthrough, dict) and passthrough:
            src = passthrough

        prompt = str(src.get("prompt") or "").strip()
        if not prompt:
            raise InvalidInput("`prompt` is required")
        background = str(src.get("background") or "auto").lower()
        if background not in ("auto", "opaque", "transparent"):
            raise InvalidInput(f"`background` must be auto, opaque or transparent, got {background!r}")
        if background == "transparent" and not prompt.startswith(RGBA_PHRASES):
            # Measured: the phrase on its own first line works; "phrase, prompt" did not.
            prompt = f"{_DEFAULT_RGBA_PHRASE}\n{prompt}"

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
        if not 0.25 <= width / height <= 4.0:
            raise InvalidInput("width/height ratio must be between 1:4 and 4:1")
        if int(src.get("n", 1)) != 1:
            raise InvalidInput("`n` must be 1 (one image per request)")

        steps_in = src.get("steps") or src.get("num_inference_steps")
        steps = max(1, min(50, int(steps_in or _DEFAULT_STEPS)))
        cfg_in = src.get("guidance_scale")
        cfg = float(cfg_in if cfg_in is not None else _DEFAULT_CFG)
        fmt = str(src.get("output_format") or "png").lower()
        if fmt not in ("png", "webp"):
            raise InvalidInput(f"`output_format` must be png or webp (both keep alpha), got {fmt!r}")
        seed = src.get("seed")
        seed = int(seed) if seed is not None else int.from_bytes(os.urandom(2), "big")

        out_w, out_h, bucket = self.infer.resolve_t2i_size("text-to-image", None, width, height)
        sampling = self.profile.resolve_sampling_parameters(steps=steps, cfg=cfg)
        with self.lock, torch.inference_mode():  # one generation at a time
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            images = self.infer.run_generation(
                self.model,
                self.processor,
                self.profile,
                task="text-to-image",
                prompt=prompt,
                input_image=None,
                resolution=bucket,
                width=width,
                height=height,
                sampling=sampling,
                seed=seed,
                num_layers=1,
                dtype=torch.bfloat16,
            )
            torch.cuda.synchronize()
            seconds = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            peak_reserved = torch.cuda.max_memory_reserved() / 2**30

        image = images[0]
        transparent = 0.0
        if image.mode == "RGBA":
            alpha = image.getchannel("A")
            hist = alpha.histogram()
            transparent = hist[0] / float(image.width * image.height)
        payload, used_fmt = self._encode(image, fmt)

        return Result(
            {
                "created": int(time.time()),
                "data": [{"b64_json": payload}],
                "width": image.width,
                "height": image.height,
                "bucket": bucket,
                "mode": image.mode,
                "seed": seed,
                "steps": sampling.steps,
                "guidance_scale": sampling.cfg,
                "background": background,
                "transparent_fraction": round(transparent, 4),
                "seconds": round(seconds, 3),
                "vram_peak_gib": round(peak, 2),
                "vram_peak_reserved_gib": round(peak_reserved, 2),
                "memory_mode": self.memory_mode,
                "output_format": used_fmt,
                "model": _MODEL_REPO,
            }
        )

    @staticmethod
    def _encode(image, fmt: str):
        """Encode keeping alpha; fall back to lossless WebP if PNG is too big."""
        buf = io.BytesIO()
        if fmt == "png":
            image.save(buf, format="PNG", optimize=False)
            data = base64.b64encode(buf.getvalue())
            if len(data) <= _MAX_B64_BYTES:
                return data.decode("ascii"), "png"
            buf = io.BytesIO()
        image.save(buf, format="WEBP", lossless=True, quality=80, method=4)
        data = base64.b64encode(buf.getvalue())
        if len(data) <= _MAX_B64_BYTES:
            return data.decode("ascii"), "webp"
        # Measured: a detailed 2048 x 2048 image can stay over the limit even as
        # lossy WebP at quality 92 (results of 10.6 and 15.5 MB were refused), so
        # step the quality down until it fits.
        for quality in (90, 82, 74, 66, 58, 50):
            buf = io.BytesIO()
            image.save(buf, format="WEBP", quality=quality, method=4)
            data = base64.b64encode(buf.getvalue())
            if len(data) <= _MAX_B64_BYTES:
                return data.decode("ascii"), f"webp-q{quality}"
        raise RuntimeError("image does not fit the 8 MB result limit even as WebP quality 50")

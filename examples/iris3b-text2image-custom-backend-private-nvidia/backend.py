"""A custom **text2image** backend wrapping Iris-3B (Sperid Labs).

Iris-3B is a 3B-parameter text-to-image transformer that generates directly in
**pixel space**: there is no VAE and no latent, the network outputs every pixel
of a 1024x1024 image. No built-in InferenceKey backend serves it (it is not a
diffusers pipeline and not a GGUF), so this backend drives the authors' own
PyTorch inference code.

Two checkpoints are supported. On the platform each one is its own published
backend (the worker does not pass per-workload config to a custom backend):
``Iris3BText2ImageBackend`` serves Turbo and ``Iris3BBaseText2ImageBackend``
serves the base model. When you serve it by hand, ``IRIS_VARIANT`` or
``{"variant": ...}`` in the config also picks one:

* ``turbo`` (default) - ``turbo/model.safetensors`` (about 6 GB), a 4-step
  distilled model with guidance baked in: 4 forward passes per image, no CFG.
* ``base`` - ``model.safetensors`` (about 12 GB, fp32), 100 DPM-Solver++ steps
  with classifier-free guidance (CFG 3 by default).

Both use Qwen3-VL-4B-Instruct as the text encoder (about 9 GB, downloaded on the
first start).

Contract shape, accepts BOTH callers:

* input, native ``/process``: ``{"prompt": "...", "size": "1024x1024",
  "seed": 7, "n": 1, "output_format": "png"}`` (only ``prompt`` is required).
  ``height``/``width`` work too. Turbo snaps the size to its nearest training
  bucket (1024x1024, 1344x752, 1344x896, 1200x896 and the transposes). ``n``
  above 1 needs ``output_format`` ``jpeg`` or ``webp`` (8 MB result limit). The base variant also reads ``steps``, ``guidance_scale`` and
  ``negative_prompt``.
* input, public API (``POST .../v1/images/generations``): the manager forwards
  the caller's OpenAI JSON in ``input.passthrough_body``; the fields are read
  from there.
* output: the OpenAI images shape ``{"created": ..., "data": [{"b64_json":
  ...}]}`` plus ``width``/``height``/``seed``/``steps``/``variant``/
  ``seconds`` (GPU time of the generation) for native callers.

The model code is NOT vendored: ``setup()`` downloads the Apache-2.0 source of
https://github.com/speridlabs/iris-3b at a pinned commit (about 1 MB) and puts
its ``src/`` on ``sys.path``. The weights come from the Hugging Face cache
(downloaded once if absent). Nothing heavy ships in the package.

Run it by hand (from this folder, with requirements installed)::

    python -m inferencekey.backend.serve --port 8099 \\
        --backend backend:Iris3BText2ImageBackend

Attribution: Iris-3B by Sperid Labs, Apache-2.0
(https://huggingface.co/speridlabs/iris-3b, arXiv:2610.09450).
"""

from __future__ import annotations

import base64
import io
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from inferencekey.backend import BackendContext, CustomBackend, Job, Result, pick_device

try:  # newer SDKs answer 400 for bad client input
    from inferencekey.backend import InvalidInput
except ImportError:  # pragma: no cover - older SDK
    InvalidInput = ValueError

#: Hugging Face repo with the weights.
_MODEL_REPO = "speridlabs/iris-3b"
#: Upstream source, pinned so a change upstream cannot break a running worker.
_SRC_REPO = "speridlabs/iris-3b"
_SRC_COMMIT = "73bcdf59adf96b189a8916139612feb54337011e"

#: Turbo was distilled at about one megapixel in these (height, width) buckets.
#: Any other size is snapped to the bucket with the closest aspect ratio.
_TURBO_BUCKETS = [
    (1024, 1024),
    (752, 1344),
    (1344, 752),
    (896, 1344),
    (1344, 896),
    (896, 1200),
    (1200, 896),
]
#: The base model is not tied to buckets; keep the canvas sane and a multiple
#: of the 16-pixel patch.
_MIN_SIDE, _MAX_SIDE, _PATCH = 256, 1536, 16
#: One request can ask for a small batch. The result travels back through NATS,
#: whose message limit is 8 MB. A detailed 1024x1024 PNG is 1.5 to 3.5 MB, so
#: two PNGs can already go over it (we measured a 9.1 MB result that the worker
#: had to drop). PNG is therefore one image per request; JPEG/WebP allow up to 4.
_MAX_N = 4
_FORMATS = {"png": "PNG", "jpeg": "JPEG", "jpg": "JPEG", "webp": "WEBP"}


def _snap_turbo(height: int, width: int) -> tuple[int, int]:
    ratio = width / max(height, 1)
    return min(_TURBO_BUCKETS, key=lambda hw: abs(hw[1] / hw[0] - ratio))


def _snap_free(value: int) -> int:
    value = max(_MIN_SIDE, min(_MAX_SIDE, value))
    return int(round(value / _PATCH) * _PATCH)


def _fetch_source(cache_root: Path) -> Path:
    """Download and unpack the pinned upstream source; return its ``src/``."""
    target = cache_root / _SRC_COMMIT
    src = target / "src"
    if (src / "iris3b" / "sampling.py").is_file():
        return src
    target.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{_SRC_REPO}/tar.gz/{_SRC_COMMIT}"
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - pinned URL
        data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        prefix = f"iris-3b-{_SRC_COMMIT}/"
        members = []
        for m in tar.getmembers():
            if not m.name.startswith(prefix) or not (m.isfile() or m.isdir()):
                continue
            m.name = m.name[len(prefix):]
            if not m.name or m.name.startswith("/") or ".." in Path(m.name).parts:
                continue
            members.append(m)
        tar.extractall(target, members=members)  # noqa: S202 - filtered above
    return src


class Iris3BText2ImageBackend(CustomBackend):
    """Iris-3B text-to-image backend (prompt in, PNG out)."""

    name = "iris3b-text2image"
    version = "0.2.0"
    task_type = "text2image"
    #: Checkpoint used when neither the config nor IRIS_VARIANT picks one.
    default_variant = "turbo"
    requirements = "requirements.txt"
    # The upstream code uses 3.11+ syntax.
    requires_python = ">=3.11"

    def setup(self, ctx: BackendContext) -> None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        t0 = time.time()

        import torch
        from huggingface_hub import snapshot_download
        from omegaconf import OmegaConf

        self.torch = torch
        self.variant = str(
            ctx.config.get("variant") or os.environ.get("IRIS_VARIANT") or self.default_variant
        ).lower()
        if self.variant not in ("turbo", "base"):
            raise ValueError(f"IRIS_VARIANT must be 'turbo' or 'base', got {self.variant!r}")

        cache_root = Path(
            os.environ.get("IRIS_SRC_CACHE", Path.home() / ".cache" / "iris3b-src")
        )
        src = _fetch_source(cache_root)
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        import iris3b.text  # noqa: F401  (registers the text encoders)
        from iris3b.config import inference_config
        from iris3b.models.dit import IrisDiT
        from iris3b.registry import TEXT_ENCODERS
        from iris3b.sampling import generate, generate_few_step, load_for_inference

        self._generate = generate
        self._generate_few_step = generate_few_step

        if self.variant == "turbo":
            patterns = ["turbo/*"]
        else:
            patterns = ["config.yaml", "model.safetensors"]
        repo_dir = Path(snapshot_download(_MODEL_REPO, allow_patterns=patterns))
        ckpt = repo_dir / "turbo" if self.variant == "turbo" else repo_dir
        t_weights = time.time()

        raw, weights = load_for_inference(ckpt)
        # The null-embedding cache (base variant, CFG) defaults to a relative
        # path; keep it in the user cache instead of the backend's cwd.
        null_dir = Path.home() / ".cache" / "iris3b-null"
        self.cfg = inference_config(raw, [f"text_encoder.null_embed_dir={null_dir}"])
        self.sigmas = None
        if self.variant == "turbo":
            self.sigmas = list(OmegaConf.load(ckpt / "distill.yaml").dmd.sigmas)

        self.device = torch.device(pick_device(str(ctx.config.get("device", "auto"))))
        with torch.device("meta"):
            model = IrisDiT(self.cfg.model)
        model.load_state_dict(weights, strict=True, assign=True)
        del weights
        # Same as the upstream sampler: fp32 weights, bf16 autocast at inference.
        self.model = model.eval().to(device=self.device, dtype=torch.float32)
        self.text_encoder = TEXT_ENCODERS.build(
            self.cfg.text_encoder.name, self.cfg.text_encoder, device=self.device
        )
        t_ready = time.time()
        print(
            f"iris3b {self.variant} ready on {self.device}: weights {t_weights - t0:.1f}s, "
            f"load {t_ready - t_weights:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    def process(self, job: Job) -> Result:
        torch = self.torch
        src = job.input
        passthrough = job.input.get("passthrough_body")
        if isinstance(passthrough, dict) and passthrough:
            src = passthrough

        prompt = str(src.get("prompt") or "").strip()
        if not prompt:
            raise InvalidInput("`prompt` is required and must be a non-empty string")

        height = int(src.get("height", 1024))
        width = int(src.get("width", 1024))
        size = src.get("size")
        if isinstance(size, str) and "x" in size:
            try:
                w_raw, h_raw = size.lower().split("x", 1)
                width, height = int(w_raw), int(h_raw)
            except ValueError:
                raise InvalidInput(f"`size` must look like '1024x1024', got {size!r}")
        if self.variant == "turbo":
            height, width = _snap_turbo(height, width)
        else:
            height, width = _snap_free(height), _snap_free(width)

        n = max(1, min(_MAX_N, int(src.get("n", 1))))
        fmt_key = str(src.get("output_format") or "png").lower()
        if fmt_key not in _FORMATS:
            raise InvalidInput(f"`output_format` must be png, jpeg or webp, got {fmt_key!r}")
        fmt = _FORMATS[fmt_key]
        if fmt == "PNG" and n > 1:
            raise InvalidInput(
                "n > 1 needs output_format jpeg or webp: several PNGs go over the 8 MB result limit"
            )
        seed = src.get("seed")
        seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "big") >> 1
        # Noise drawn on the CPU, like the authors' demo: the same seed gives the
        # same image on any GPU model.
        noise = torch.randn(
            (n, self.cfg.model.in_channels, height, width),
            generator=torch.Generator().manual_seed(seed),
        )
        prompts = [prompt] * n

        t0 = time.time()
        common = dict(
            noise=noise,
            device=self.device,
            num_train_timesteps=self.cfg.flow.num_train_timesteps,
            prediction=self.cfg.flow.prediction,
        )
        with torch.inference_mode(), torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            if self.variant == "turbo":
                steps = len(self.sigmas)
                images = self._generate_few_step(
                    self.model, self.text_encoder, prompts,
                    height=height, width=width, sigmas=self.sigmas, **common,
                )
            else:
                steps = max(1, min(200, int(src.get("steps", self.cfg.sample.steps))))
                images = self._generate(
                    self.model, self.text_encoder, prompts,
                    height=height, width=width, steps=steps,
                    order=self.cfg.sample.order,
                    cfg_scale=float(src.get("guidance_scale", self.cfg.sample.cfg_scale)),
                    cfg_interval=tuple(self.cfg.sample.cfg_interval),
                    shift=self.cfg.flow.shift,
                    negative_prompt=str(src.get("negative_prompt") or self.cfg.sample.negative_prompt or ""),
                    **common,
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.time() - t0

        from PIL import Image

        data = []
        for image in images:
            pixels = (
                ((image.float().clamp(-1, 1) + 1) * 127.5)
                .round().byte().permute(1, 2, 0).cpu().numpy()
            )
            buf = io.BytesIO()
            if fmt == "PNG":
                Image.fromarray(pixels).save(buf, format="PNG")
            else:
                Image.fromarray(pixels).save(buf, format=fmt, quality=92)
            data.append({"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")})

        return Result(
            {
                "created": int(time.time()),
                "data": data,
                "width": width,
                "height": height,
                "seed": seed,
                "steps": steps,
                "variant": self.variant,
                "seconds": round(seconds, 3),
                "output_format": fmt.lower(),
                "model": _MODEL_REPO,
            }
        )


class Iris3BBaseText2ImageBackend(Iris3BText2ImageBackend):
    """The undistilled base checkpoint (100 steps, CFG 3), published under its own slug.

    The worker does not pass a per-workload config to a custom backend, so the
    variant is fixed by the entrypoint: publish this class as a second backend
    (``python publish.py --variant base``) to run Turbo and base side by side.
    """

    name = "iris3b-base-text2image"
    default_variant = "base"

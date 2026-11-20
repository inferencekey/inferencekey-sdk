"""A custom backend that turns one picture of an object into a 3D Gaussian splat with **Idol**.

Idol (Speri Labs, Apache-2.0) reads one image (a render, a generated image or a
photo), removes the background, frames the object and generates up to
1,048,576 3D Gaussians, written as a standard 3DGS ``.ply``. The pipeline is
the authors' ``IdolPipeline`` (https://github.com/speridlabs/idol-sdk): a 370M
flow-matching DiT, an octree decoder, a Gaussian decoder, BiRefNet for the
background, and two image encoders (DINOv3 ViT-H+/16 and the FLUX.2 VAE).

Idol Turbo (``speridlabs/idol-turbo``) adds two distilled phase LoRAs (a slow
expert for the high-noise steps, a fast one for the low-noise steps) that fold
the guidance into the weights: 5 steps without guidance instead of 50 guided
steps (100 network evaluations down to 5). Both run from the same loaded
model; each request picks one with ``"turbo": true``.

The image encoders: the authors download them from two gated repos
(``facebook/dinov3-vith16plus-pretrain-lvd1689m`` and
``black-forest-labs/FLUX.2-dev``). By default this backend needs no gated
access (``IDOL_ENCODERS=open``): it takes the DINOv3 weights from Comfy-Org's
Apache-2.0 repack (``Comfy-Org/Idol``, bf16, the precision Idol runs it in)
with the model config from ``onnx-community``, and the FLUX.2 VAE from
``black-forest-labs/FLUX.2-klein-4B`` (Apache-2.0, the same VAE in bf16, the
precision Idol casts it to). ``IDOL_ENCODERS=gated`` uses the authors' repos
instead (needs ``HF_TOKEN`` of an account that accepted both licenses).

Contract (``task_type`` text2image, so the workload answers on the OpenAI
images route ``POST .../v1/images/generations``; the caller's JSON arrives in
``input.passthrough_body``):

* generate: ``{"image": "<base64 or data URL>"}`` or ``{"example": "<name>"}``
  (one of the authors' example pictures in idol-sdk ``demo/examples``; a bare
  ``"prompt"`` that names an example works too, so the dashboard's image
  tester can run it). Optional: ``mode`` (``synthetic`` for renders and
  generated images, ``real`` for photos), ``seed`` (42), ``turbo`` (false),
  ``steps`` (50, or 5 with turbo), ``cfg_strength`` (3.0, ignored by turbo),
  ``rescale_t`` (3.0), ``num_gaussians`` (262,144; 32,768 to 1,048,576),
  ``vram_limit_gib`` (cap the GPU memory for this request, to see whether a
  smaller card would do), ``offload_encoders`` (keep BiRefNet, DINOv3 and the
  VAE in system RAM and move them to the GPU only to read the picture: about
  2.3 GB less VRAM for a little more time), ``vae_tiling`` (the FLUX.2 VAE
  reads the 1024 x 1024 picture in tiles: its activations are the request's
  memory peak). From the dashboard's image tester, write the example name as
  the prompt, plus the words ``turbo``, ``real``, ``offload`` or ``tiling``
  (for example ``obj-backpack turbo``).
* answer: ``data[0].b64_json`` is the image the model saw (background removed,
  cropped, JPEG), so an image viewer shows something; ``ply_id``, ``ply_bytes``,
  ``ply_sha256``, ``parts`` and ``ply_b64`` (part 0 of the ``.ply``); the
  predicted ``camera``; ``seconds`` and ``stage_seconds`` (preprocess,
  encoders, sampling, decode, ply); ``peak_allocated_mib`` /
  ``peak_reserved_mib`` (PyTorch) and ``card_peak_mib`` (``nvidia-smi``).
* fetch: ``{"ply_id": "...", "part": n}`` returns part ``n``. A result
  travels in one 8 MB message, and a 262,144-Gaussian ``.ply`` is 17.8 MB
  (68 bytes per Gaussian), so the file comes back in parts of 2 MB. The
  last 16 files are kept on the worker's disk.

Environment (the workload's ``config.env``):

* ``IDOL_ENCODERS``: ``open`` (default) or ``gated``.
* ``IDOL_TURBO``: ``1`` (default) loads the Idol Turbo LoRAs; ``0`` skips them.
* ``IDOL_OPTIMIZE``: ``1`` (default) is the authors' fast path (torch.compile,
  batched guidance, cuDNN attention, TF32); ``0`` runs eager.
* ``IDOL_VRAM_LIMIT_GIB``: cap GPU memory for the whole process.
* ``IDOL_OFFLOAD_ENCODERS`` / ``IDOL_VAE_TILING``: ``1`` turns the option on for every request.
* ``IDOL_WARMUP``: ``1`` (default) runs one short generation of each kind in
  ``setup()`` so the first real request does not pay for compiling.

Run it by hand (from this folder, with requirements installed)::

    python -m inferencekey.backend.serve --port 8099 --backend backend:IdolBackend

Attribution: Idol and Idol Turbo by Speri Labs (Apache-2.0,
https://huggingface.co/speridlabs/idol); DINOv3 by Meta (DINOv3 License);
FLUX.2 VAE by Black Forest Labs (Apache-2.0 in FLUX.2 [klein]); BiRefNet by
its authors (MIT); repack by Comfy-Org.
"""

from __future__ import annotations

import base64
import collections
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from inferencekey.backend import BackendContext, CustomBackend, Job, Result

try:  # newer SDKs answer 400 for bad client input
    from inferencekey.backend import InvalidInput
except ImportError:  # pragma: no cover - older SDK
    InvalidInput = ValueError

#: Pinned revisions this backend was tested with (2026-11-20).
IDOL_REPO, IDOL_REVISION = "speridlabs/idol", "2731c2291fc65a797c5cb627a5e5526fc1490902"
TURBO_REPO, TURBO_REVISION = "speridlabs/idol-turbo", "45cdd41f1f4d6d70496daa7d619f57ad79307a37"
COMFY_REPO, COMFY_REVISION = "Comfy-Org/Idol", "e907d8dca6c465735eaf4ce9b3a3c2c6401d5705"
DINO_CONFIG_REPO, DINO_CONFIG_REVISION = (
    "onnx-community/dinov3-vith16plus-pretrain-lvd1689m-ONNX", "f115b99bd0a7cdd4ce87ff120df949c80364889e")
VAE_REPO, VAE_REVISION = "black-forest-labs/FLUX.2-klein-4B", "e7b7dc27f91deacad38e78976d1f2b499d76a294"
SDK_COMMIT = "8db7003"
EXAMPLES_URL = "https://raw.githubusercontent.com/speridlabs/idol-sdk/{commit}/demo/examples/{name}.webp"
WEIGHT_FILES = ["config.json", "denoiser.safetensors", "decoder.safetensors", "decoder_gs.safetensors",
                "birefnet.safetensors"]

#: One result message is limited to 8 MB and carries the payload about twice,
#: so each part stays at 2 MB of raw bytes (2.7 MB of base64).
PART_BYTES = 2_000_000
KEEP_FILES = 16
MIN_GAUSSIANS, MAX_GAUSSIANS = 32_768, 1_048_576


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class _VramSampler:
    """Polls ``nvidia-smi`` twice a second and keeps the peak of memory.used (the whole card)."""

    def __init__(self) -> None:
        self.peak = 0
        self.last = 0
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0", "-lms", "500"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except OSError:
            self._proc = None
            return
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        for line in self._proc.stdout:
            try:
                self.last = int(line.strip())
            except ValueError:
                continue
            self.peak = max(self.peak, self.last)

    def reset(self) -> None:
        self.peak = self.last


class IdolBackend(CustomBackend):
    """Idol + Idol Turbo: one image in, a 3D Gaussian splat (.ply) out."""

    name = "idol-image-to-3d"
    version = "0.1.3"
    task_type = "text2image"
    requirements = "requirements.txt"
    requires_python = ">=3.11,<3.13"

    # ------------------------------------------------------------------ setup

    def setup(self, ctx: BackendContext) -> None:
        import torch

        t0 = time.time()
        self.lock = threading.Lock()
        # Local disk of the container, never a network volume.
        self.work = Path(os.environ.get("IDOL_WORKDIR") or Path.home() / ".cache" / "inferencekey" / "idol")
        self.results = self.work / "results"
        self.results.mkdir(parents=True, exist_ok=True)
        self.kept: collections.OrderedDict[str, Path] = collections.OrderedDict()
        self.sha: dict[str, str] = {}

        weights, encoders, lora, turbo_py = self._download()
        t_download = time.time()

        process_cap = float(os.environ.get("IDOL_VRAM_LIMIT_GIB") or 0)
        if process_cap > 0:
            self._cap(process_cap)
        from idol import IdolPipeline

        self.optimize = (os.environ.get("IDOL_OPTIMIZE") or "1") != "0"
        self.pipe = IdolPipeline(weights, image_encoders=encoders, optimize=self.optimize)
        torch.cuda.synchronize()
        t_load = time.time()

        self.turbo = None
        if (os.environ.get("IDOL_TURBO") or "1") != "0":
            spec = importlib.util.spec_from_file_location("idol_turbo", turbo_py)
            self.turbo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.turbo)
            self._patch_turbo(lora)
            self.turbo.set_turbo_mode(self.pipe, False)
        self.vram = _VramSampler()
        _log(
            f"idol: downloads {t_download - t0:.1f}s, load {t_load - t_download:.1f}s, "
            f"weights in VRAM {torch.cuda.memory_allocated() / 2**20:.0f} MiB, encoders {self.encoder_source}, "
            f"turbo {'on' if self.turbo else 'off'}, optimize {self.optimize}"
        )
        if (os.environ.get("IDOL_WARMUP") or "1") != "0":
            image = self._example("toy-robot-tin")
            for turbo in ([False, True] if self.turbo else [False]):
                t = time.time()
                self._generate(image, mode="synthetic", seed=0, turbo=turbo, steps=2, cfg_strength=3.0,
                               rescale_t=3.0, num_gaussians=MIN_GAUSSIANS, limit_gib=0)
                _log(f"idol: warm-up {'turbo' if turbo else 'base'} (compile) {time.time() - t:.1f}s")
        _log(f"idol ready: setup {time.time() - t0:.1f}s")

    def _download(self):
        from huggingface_hub import hf_hub_download, snapshot_download

        weights = snapshot_download(IDOL_REPO, revision=IDOL_REVISION, allow_patterns=WEIGHT_FILES, max_workers=8)
        lora = turbo_py = None
        if (os.environ.get("IDOL_TURBO") or "1") != "0":
            lora = hf_hub_download(TURBO_REPO, "idol_turbo_lora.safetensors", revision=TURBO_REVISION)
            turbo_py = hf_hub_download(TURBO_REPO, "turbo.py", revision=TURBO_REVISION)
        source = (os.environ.get("IDOL_ENCODERS") or "open").lower()
        self.encoder_source = source
        if source == "gated":
            return weights, None, lora, turbo_py  # config.json's facebook/ and black-forest-labs/ repos
        # DINOv3 ViT-H+/16: Comfy-Org's bf16 weights (transformers key names) + the model config.
        dino = self.work / "encoders" / "dinov3-vith16plus"
        dino.mkdir(parents=True, exist_ok=True)
        weights_file = hf_hub_download(COMFY_REPO, "clip_vision/dino_v3_vit_h_bf16.safetensors",
                                       revision=COMFY_REVISION)
        config_file = hf_hub_download(DINO_CONFIG_REPO, "config.json", revision=DINO_CONFIG_REVISION)
        config = json.loads(Path(config_file).read_text())
        config.pop("transformers.js_config", None)
        (dino / "config.json").write_text(json.dumps(config, indent=1))
        link = dino / "model.safetensors"
        if not link.exists():
            link.symlink_to(weights_file)
        # FLUX.2 VAE: the diffusers folder of FLUX.2 [klein] 4B (same VAE as FLUX.2 [dev]).
        vae_root = Path(snapshot_download(VAE_REPO, revision=VAE_REVISION, allow_patterns=["vae/*"]))
        return weights, {"dinov3": self._load_dinov3(dino), "flux2_vae": str(vae_root)}, lora, turbo_py

    @staticmethod
    def _load_dinov3(folder: Path):
        """Load DINOv3 and refuse to start if any weight is missing (a silent random init would ruin the output)."""
        import torch
        from transformers import AutoModel

        model, info = AutoModel.from_pretrained(str(folder), dtype=torch.bfloat16, output_loading_info=True)
        missing = [k for k in info.get("missing_keys", []) if "rope" not in k]
        if missing or info.get("unexpected_keys") or info.get("mismatched_keys"):
            raise RuntimeError(f"DINOv3 weights do not match the config: {info}")
        return model

    def _patch_turbo(self, lora: str) -> None:
        """The authors' patch_pipeline_for_turbo (turbo.py), fed the LoRA file downloaded at the pinned revision."""
        import huggingface_hub

        download = huggingface_hub.hf_hub_download
        huggingface_hub.hf_hub_download = lambda *args, **kwargs: lora
        try:
            self.turbo.patch_pipeline_for_turbo(self.pipe, TURBO_REPO)
        finally:
            huggingface_hub.hf_hub_download = download
        # The LoRA experts are created on the CPU: move them next to the base weights.
        self.pipe.denoiser.to(self.pipe.device)

    @staticmethod
    def _cap(gib: float) -> None:
        import torch

        torch.cuda.empty_cache()
        total = torch.cuda.mem_get_info(0)[1]
        torch.cuda.set_per_process_memory_fraction(min(1.0, gib * 2**30 / total), 0)

    def _example(self, name: str):
        from PIL import Image

        name = Path(name).stem
        cache = self.work / "examples" / f"{name}.webp"
        if not cache.is_file():
            cache.parent.mkdir(parents=True, exist_ok=True)
            url = EXAMPLES_URL.format(commit=SDK_COMMIT, name=name)
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    cache.write_bytes(r.read())
            except Exception:
                raise InvalidInput(f"no example called {name!r} in idol-sdk demo/examples")
        image = Image.open(cache)
        image.load()
        return image

    # ------------------------------------------------------------- generation

    def _encoders(self):
        """The modules used only before sampling: BiRefNet, DINOv3 and the FLUX.2 VAE (about 2.3 GB)."""
        enc = self.pipe.image_encoder
        return [self.pipe.background_remover, enc.dinov3, enc.flux2_vae]

    def _place_encoders(self, device) -> None:
        import torch

        for module in self._encoders():
            module.to(device)
        self.pipe.image_encoder.device = torch.device(device)
        if device == "cpu":
            torch.cuda.empty_cache()

    def _generate(self, image, *, mode, seed, turbo, steps, cfg_strength, rescale_t, num_gaussians, limit_gib,
                  offload=False, tiling=False):
        """IdolPipeline.__call__, split into timed stages. offload: keep the encoders in system RAM and move them to
        the GPU only for the preprocessing and encoding stages (trades about a second for about 2.3 GB of VRAM)."""
        import torch

        if self.turbo:
            self.turbo.set_turbo_mode(self.pipe, turbo)
        vae = self.pipe.image_encoder.flux2_vae
        if hasattr(vae, "enable_tiling"):
            if tiling:
                # diffusers only tiles inputs larger than tile_sample_min_size (1024 here): use 512 px tiles.
                vae.enable_tiling()
                vae.tile_sample_min_size = 512
                vae.tile_latent_min_size = 512 // (2 ** (len(vae.config.block_out_channels) - 1))
            else:
                vae.disable_tiling()
        if offload:
            self._place_encoders("cpu")
        if limit_gib:
            self._cap(limit_gib)
        try:
            return self._run(image, mode=mode, seed=seed, turbo=turbo, steps=steps, cfg_strength=cfg_strength,
                             rescale_t=rescale_t, num_gaussians=num_gaussians, offload=offload)
        finally:
            if limit_gib:
                self._cap(1e6)  # lift the per-request cap again, also after an out-of-memory error
            if offload:
                self._place_encoders(str(self.pipe.device))
            torch.cuda.empty_cache()

    def _run(self, image, *, mode, seed, turbo, steps, cfg_strength, rescale_t, num_gaussians, offload):
        import numpy as np
        import torch
        from tensordict import TensorDict

        import idol.pipeline as ip
        from idol.camera import decode_camera

        pipe = self.pipe
        sync = torch.cuda.synchronize
        stages = {}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.vram.reset()
        t = t0 = time.time()
        if offload:
            self._place_encoders(str(pipe.device))
        prepared = pipe.preprocess(image, mode=mode)
        pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(pipe.device)
        sync(); stages["preprocess"] = round(time.time() - t, 3); t = time.time()
        torch.manual_seed(seed)
        cond = pipe.image_encoder(pixels)
        if offload:
            self._place_encoders("cpu")
        sync(); stages["encoders"] = round(time.time() - t, 3); t = time.time()
        noise = TensorDict({
            "latent": torch.randn(1, pipe.denoiser.q_token_length, pipe.denoiser.in_channels, device=pipe.device),
            "camera": torch.randn(1, 1, pipe.denoiser.cam_channels, device=pipe.device),
        }, batch_size=1, device=pipe.device)
        with pipe._sampling_settings():
            sample = ip.sample_flow(pipe.denoiser, noise, cond=cond, neg_cond=cond.apply(torch.zeros_like),
                                    steps=steps, rescale_t=rescale_t, cfg_strength=0.0 if turbo else cfg_strength,
                                    verbose=False, batch_cfg=pipe.optimize and not turbo)
        sync(); stages["sampling"] = round(time.time() - t, 3); t = time.time()
        gaussian = pipe.decode(sample["latent"], num_gaussians)
        sync(); stages["decode"] = round(time.time() - t, 3)
        seconds = time.time() - t0
        return {"gaussian": gaussian, "prepared": prepared, "camera": decode_camera(sample["camera"][0, 0]),
                "seconds": round(seconds, 3), "stage_seconds": stages,
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                "card_peak_mib": self.vram.peak}

    # ---------------------------------------------------------------- process

    def process(self, job: Job) -> Result:
        src = job.input
        passthrough = job.input.get("passthrough_body")
        if isinstance(passthrough, dict) and passthrough:
            src = passthrough
        if src.get("ply_id"):
            return Result(self._part(str(src["ply_id"]), int(src.get("part") or 0)))

        words = str(src.get("prompt") or "").lower().split()
        image = self._input_image(src)
        turbo = bool(src.get("turbo")) or "turbo" in words
        tiling = bool(src.get("vae_tiling")) or "tiling" in words or (os.environ.get("IDOL_VAE_TILING") or "0") == "1"
        offload = bool(src.get("offload_encoders")) or "offload" in words or (
            (os.environ.get("IDOL_OFFLOAD_ENCODERS") or "0") == "1")
        if turbo and not self.turbo:
            raise InvalidInput("this workload was started with IDOL_TURBO=0")
        mode = str(src.get("mode") or ("real" if "real" in words else "synthetic"))
        if mode not in ("synthetic", "real"):
            raise InvalidInput("`mode` must be 'synthetic' (renders) or 'real' (photos)")
        steps = int(src.get("steps") or (5 if turbo else 50))  # the dashboard's num_inference_steps is ignored
        if not 1 <= steps <= 200:
            raise InvalidInput("`steps` must be between 1 and 200")
        num = int(src.get("num_gaussians") or 262_144)
        if not MIN_GAUSSIANS <= num <= MAX_GAUSSIANS:
            raise InvalidInput(f"`num_gaussians` must be between {MIN_GAUSSIANS} and {MAX_GAUSSIANS}")
        num -= num % 32
        seed = int(src["seed"]) if src.get("seed") is not None else 42
        cfg = float(src.get("cfg_strength") if src.get("cfg_strength") is not None else 3.0)
        rescale_t = float(src.get("rescale_t") or 3.0)
        limit = float(src.get("vram_limit_gib") or 0)

        with self.lock:  # one generation at a time
            out = self._generate(image, mode=mode, seed=seed, turbo=turbo, steps=steps, cfg_strength=cfg,
                                 rescale_t=rescale_t, num_gaussians=num, limit_gib=limit, offload=offload,
                                 tiling=tiling)
            t = time.time()
            ply_id = uuid.uuid4().hex[:16]
            path = self.results / f"{ply_id}.ply"
            out["gaussian"].save_ply(str(path))
            out["stage_seconds"]["ply"] = round(time.time() - t, 3)
        self.sha[ply_id] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.kept[ply_id] = path
        while len(self.kept) > KEEP_FILES:
            old_id, old = self.kept.popitem(last=False)
            self.sha.pop(old_id, None)
            old.unlink(missing_ok=True)

        buf = io.BytesIO()
        out["prepared"].save(buf, format="JPEG", quality=88)
        first = self._part(ply_id, 0)
        return Result({
            "created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")}],
            "output_format": "jpeg",
            "model": "speridlabs/idol" + (" + idol-turbo" if turbo else ""),
            "mode": mode, "seed": seed, "turbo": turbo, "steps": steps,
            "cfg_strength": 0.0 if turbo else cfg, "rescale_t": rescale_t,
            "num_gaussians": num, "vram_limit_gib": limit or None, "offload_encoders": offload, "vae_tiling": tiling,
            "camera": out["camera"], "seconds": out["seconds"], "stage_seconds": out["stage_seconds"],
            "peak_allocated_mib": out["peak_allocated_mib"], "peak_reserved_mib": out["peak_reserved_mib"],
            "card_peak_mib": out["card_peak_mib"],
            **first,
        })

    def _part(self, ply_id: str, part: int) -> dict:
        path = self.kept.get(ply_id)
        if path is None or not path.is_file():
            raise InvalidInput(f"unknown or expired ply_id {ply_id!r}")
        size = path.stat().st_size
        parts = (size + PART_BYTES - 1) // PART_BYTES
        if not 0 <= part < parts:
            raise InvalidInput(f"`part` must be between 0 and {parts - 1}")
        with path.open("rb") as f:
            f.seek(part * PART_BYTES)
            chunk = f.read(PART_BYTES)
        return {"ply_id": ply_id, "part": part, "parts": parts, "ply_bytes": size, "ply_sha256": self.sha[ply_id],
                "ply_b64": base64.b64encode(chunk).decode("ascii")}

    def _input_image(self, src: dict):
        from PIL import Image

        raw = src.get("image") if src.get("image") is not None else src.get("images")
        if isinstance(raw, list):
            if len(raw) != 1:
                raise InvalidInput("send exactly one image")
            raw = raw[0]
        if isinstance(raw, dict):
            raw = raw.get("b64_json") or raw.get("image_url") or raw.get("url")
        if raw:
            if not isinstance(raw, str):
                raise InvalidInput("`image` must be a base64 string or a data URL")
            if raw.startswith("data:"):
                raw = raw.split(",", 1)[-1]
            try:
                image = Image.open(io.BytesIO(base64.b64decode(raw)))
                image.load()
            except Exception:
                raise InvalidInput("`image` is not a readable image")
            return image
        name = str(src.get("example") or (str(src.get("prompt") or "").split() or [""])[0]).strip()
        if not name:
            raise InvalidInput("send `image` (base64) or `example` (an idol-sdk demo example name)")
        if not all(c.isalnum() or c in "-_" for c in name):
            raise InvalidInput("`example` must be a name such as 'toy-robot-tin'")
        return self._example(name)

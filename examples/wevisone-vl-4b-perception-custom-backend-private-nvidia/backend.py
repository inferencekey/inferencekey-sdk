"""A custom backend wrapping WeVisOne-VL (Tencent WeChat Vision).

WeVisOne-VL is an Apache-2.0 vision-language model built on Qwen3-VL that does
open-vocabulary detection, referring, counting, pointing, OCR grounding,
referring segmentation and alpha matting in one model. Boxes and points come out
as coordinate tokens; for pixel-level tasks the model emits an ``<OBJ>`` token
after each box and a 9.6M-parameter pixel decoder (``mask_decoder.pt`` in the
model repo) turns its hidden state into a full-resolution alpha matte. No SAM.

No built-in InferenceKey backend runs that decoder, so this backend downloads
the authors' inference package (https://github.com/Tencent/WeVisOne-VL) at a
pinned commit and drives it:

* ``WEVIS_ENGINE=auto`` (default) picks ``vllm`` on cards of 22 GiB or more and
  ``transformers`` below.
* ``WEVIS_ENGINE=vllm``: the authors' vLLM path generates the text
  (boxes, points, ``<OBJ>``), and a second, Transformers copy of the VLM is
  teacher-forced on that answer to feed the pixel decoder. Two copies of the
  weights: about 20 GB on the GPU for the 4B model.
* ``WEVIS_ENGINE=transformers``: one Transformers copy does both. Slower text
  generation, about half the memory.

Contract, accepts BOTH callers:

* public API ``POST .../v1/images/generations`` (the manager forwards the JSON
  body in ``input.passthrough_body``) or native ``/process`` (``job.input``):
  ``{"task": "matting", "query": "person", "image": "<base64 | data URL | https URL>"}``.
  ``task`` is one of detection, referring, point, count_box, count_point, ocr,
  ocr_grounding, segmentation, matting (default detection). ``query`` is a
  string or a list of labels. Optional: ``max_new_tokens`` (default 2048),
  ``return_masks`` (default true: one grayscale PNG matte per instance).
* a bare ``prompt`` also works (the dashboard image tester only sends a prompt):
  ``"matting: the dog on the left https://example.com/photo.jpg"``, i.e. an
  optional ``<task>:`` prefix, the query, and the image URL anywhere in it.
* output: the OpenAI images shape ``{"created", "data": [{"b64_json"}]}`` where
  ``data[0]`` is the annotated image (boxes, points, mask overlay) and, for
  segmentation/matting, ``data[1]`` is the RGBA cutout; plus the structured
  answer: ``response`` (raw model text), ``predictions`` (pixel boxes/points by
  label), ``count``, ``rejected`` (true when the model answered ``None``: the
  target is not in the image), ``masks`` (PNG mattes), and timings
  ``seconds`` / ``seconds_generate`` / ``seconds_decode`` / ``vram_peak_gib``.

Memory knobs (environment, through the workload ``config.env``):
``WEVIS_ENGINE``, ``WEVIS_GPU_MEMORY_UTILIZATION`` (vLLM share of the card;
default sized to the weights plus 1.75 GiB, 0.465 on a 24 GB card),
``WEVIS_MAX_MODEL_LEN`` (default 3072), ``WEVIS_MAX_NUM_SEQS`` (default 4), ``WEVIS_VRAM_LIMIT_GIB`` (caps the
Transformers process with ``torch.cuda.set_per_process_memory_fraction``: how
we emulated smaller cards), ``WEVIS_MAX_PIXELS`` (input resolution cap, default
the authors' 2025 x 32 x 32), ``WEVIS_MODEL`` (``tencent/WeVisOne-VL-4B`` or
``tencent/WeVisOne-VL-2B``).

Run it by hand (from this folder, with requirements installed)::

    python -m inferencekey.backend.serve --port 8099 --backend backend:WeVisOneVLBackend

Attribution: WeVisOne-VL by WeChat Vision, Tencent Inc., Apache-2.0
(https://huggingface.co/tencent/WeVisOne-VL-4B); inference code from
https://github.com/Tencent/WeVisOne-VL (Apache-2.0).
"""

from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

from inferencekey.backend import BackendContext, CustomBackend, Job, Result

try:  # newer SDKs answer 400 for bad client input
    from inferencekey.backend import InvalidInput
except ImportError:  # pragma: no cover - older SDK
    InvalidInput = ValueError

#: Model repos and the revisions this backend was tested with (2026-10-25).
_REVISIONS = {
    "tencent/WeVisOne-VL-4B": "4a7bed92750ac9d8b8a124cede14759036932dc2",
    "tencent/WeVisOne-VL-2B": "3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf",
}
#: The authors' inference package, pinned.
_CODE_REPO = "Tencent/WeVisOne-VL"
_CODE_COMMIT = "40f735b8daed437ece982db9718e9c9856f52a79"

_TASKS = (
    "detection", "referring", "point", "count_box", "count_point",
    "ocr", "ocr_grounding", "segmentation", "matting",
)
_PIXEL_TASKS = ("segmentation", "matting")
_URL_RE = re.compile(r"https?://\S+")
#: Results travel through an 8 MB message, and the worker base64-encodes this
#: JSON once more (x1.33) on the images route. Measured: overlay + cutout as
#: 1600 px PNGs of a 1620 x 1080 photo made a 9,145,132-byte message that was
#: refused. Keep the images under this many base64 bytes in total.
_MAX_PAYLOAD_B64 = 5_200_000
_MAX_SIDE_OUT = 1600
_MAX_SIDE_CUTOUT = 1280


def _fetch_code() -> Path:
    """Download the authors' inference package at the pinned commit (once)."""
    root = Path(os.environ.get("WEVIS_CODE_DIR") or Path.home() / ".cache" / "wevisone-vl-code")
    target = root / _CODE_COMMIT
    if not (target / "wevisone_vl" / "engine.py").is_file():
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / f"{_CODE_COMMIT}.tar.gz"
        urllib.request.urlretrieve(f"https://codeload.github.com/{_CODE_REPO}/tar.gz/{_CODE_COMMIT}", tmp)
        staging = root / f"{_CODE_COMMIT}.staging"
        with tarfile.open(tmp) as tar:
            tar.extractall(staging, filter="data")
        (inner,) = list(staging.iterdir())
        inner.rename(target)
        staging.rmdir()
        tmp.unlink()
    return target


def _gpu_total_gib() -> float:
    """Total memory of GPU 0 without initialising CUDA in this process
    (vLLM must start its engine before CUDA is touched here)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    return float(out) / 1024.0


def _load_image(src: str):
    from PIL import Image

    src = src.strip()
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": "inferencekey-wevisone-backend"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read(40 * 1024 * 1024)
    else:
        if src.startswith("data:"):
            src = src.split(",", 1)[1]
        try:
            data = base64.b64decode(src, validate=False)
        except Exception as exc:  # noqa: BLE001
            raise InvalidInput(f"`image` is not valid base64: {exc}")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise InvalidInput(f"could not decode the image: {exc}")
    return img.convert("RGB")


def _png_b64(img, max_side: int = _MAX_SIDE_OUT, fmt: str = "PNG") -> str:
    from PIL import Image

    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=88)
    else:
        img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class WeVisOneVLBackend(CustomBackend):
    """WeVisOne-VL: detection, referring, counting, pointing, OCR, segmentation, matting."""

    name = "wevisone-vl-perception"
    version = "0.1.3"
    task_type = "text2image"  # the SDK packager has no image2image yet; text2image opens the images route
    requirements = "requirements.txt"
    requires_python = ">=3.12,<3.13"

    def setup(self, ctx: BackendContext) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # vLLM's FlashInfer JIT calls `ninja`; the wheel puts it in the venv bin.
        venv_bin = str(Path(sys.executable).parent)
        if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
        t0 = time.time()

        code = _fetch_code()
        if str(code) not in sys.path:
            sys.path.insert(0, str(code))
        from huggingface_hub import snapshot_download

        self.model_id = os.environ.get("WEVIS_MODEL") or str(ctx.config.get("model_id") or "tencent/WeVisOne-VL-4B")
        if self.model_id not in _REVISIONS:
            raise ValueError(f"WEVIS_MODEL must be one of {sorted(_REVISIONS)}, got {self.model_id!r}")
        local = snapshot_download(self.model_id, revision=_REVISIONS[self.model_id], max_workers=16)
        t_download = time.time()

        total = _gpu_total_gib()
        self.engine = (os.environ.get("WEVIS_ENGINE") or "auto").strip().lower()
        if self.engine == "auto":
            # Two copies of the 4B weights (vLLM + Transformers) need a 24 GB card.
            self.engine = "vllm" if total >= 22.0 else "transformers"
        if self.engine not in ("vllm", "transformers"):
            raise ValueError(f"WEVIS_ENGINE must be auto, vllm or transformers, got {self.engine!r}")
        # bf16 weights as vLLM loads them (measured: "Model loading took 9.37 GiB" for 4B)
        weights_gib = 9.4 if "4B" in self.model_id else 4.6
        util = float(os.environ.get("WEVIS_GPU_MEMORY_UTILIZATION") or 0)
        if self.engine == "vllm" and util <= 0:
            # Measured on an RTX 4090: 0.465 (11.2 GiB) leaves the Transformers
            # copy room for a 1620 x 1080 matte (card peak 23.5 GB); 0.5 did not.
            util = min(0.9, round((weights_gib + 1.75) / total, 3))
        max_pixels = int(os.environ.get("WEVIS_MAX_PIXELS") or 2025 * 32 * 32)
        if self.engine == "vllm":
            self._shape_vllm(max_pixels)

        from wevisone_vl import WeVisOneVL

        limit = float(os.environ.get("WEVIS_VRAM_LIMIT_GIB") or 0)
        if limit > 0 and self.engine == "transformers":
            import torch  # CUDA may be initialised now: no vLLM engine on this path

            torch.cuda.set_per_process_memory_fraction(min(1.0, limit / total), 0)
        self.model = WeVisOneVL(
            local,
            backend=self.engine,
            device="cuda:0",
            dtype="bfloat16",
            max_pixels=max_pixels,
            gpu_memory_utilization=util if self.engine == "vllm" else 0.45,
            max_model_len=8192,
        )
        import torch

        from wevisone_vl import visualization

        self.torch = torch
        self.vis = visualization
        self.lock = threading.Lock()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.ready_detail = (
            f"{self.model_id} on {torch.cuda.get_device_name(0)} ({total:.1f} GiB), engine {self.engine}"
            + (f" (vLLM share {util})" if self.engine == "vllm" else "")
            + (f", cap {limit} GiB" if limit > 0 else "")
            + f": download {t_download - t0:.1f}s, load {time.time() - t_download:.1f}s, "
            f"transformers weights {torch.cuda.memory_allocated() / 2**30:.2f} GiB"
        )
        print("wevisone-vl ready: " + self.ready_detail, file=sys.stderr, flush=True)

    @staticmethod
    def _shape_vllm(max_pixels: int) -> None:
        """Size the authors' vLLM engine for one image per request.

        Their ``LLM(...)`` call leaves the multimodal limits at Qwen3-VL's
        defaults, so vLLM profiles a 16,384-token image (and 10 images plus 10
        videos per prompt) and, on a 24 GB card shared with the Transformers
        copy, finds no memory left for the KV cache (measured:
        ``Available KV cache memory: -0.87 GiB``). Images reach vLLM already
        resized to ``max_pixels`` by qwen_vl_utils, so profiling for that size
        is exact. Patched by wrapping ``vllm.LLM`` before the authors build it.
        """
        import vllm

        if getattr(vllm.LLM, "_wevis_shaped", False):
            return
        original = vllm.LLM
        side = int(max_pixels ** 0.5)
        max_len = int(os.environ.get("WEVIS_MAX_MODEL_LEN") or 3072)
        max_seqs = int(os.environ.get("WEVIS_MAX_NUM_SEQS") or 4)

        def shaped_llm(*args, **kwargs):
            kwargs["limit_mm_per_prompt"] = {"image": {"count": 1, "width": side, "height": side}, "video": 0}
            kwargs["mm_processor_kwargs"] = {"size": {"longest_edge": int(max_pixels), "shortest_edge": 16 * 32 * 32}}
            kwargs["max_model_len"] = max_len
            kwargs["max_num_seqs"] = max_seqs
            return original(*args, **kwargs)

        shaped_llm._wevis_shaped = True
        vllm.LLM = shaped_llm

    @staticmethod
    def _request(src: dict):
        task = str(src.get("task") or "").strip().lower()
        query = src.get("query")
        image = src.get("image") or src.get("image_url")
        if isinstance(image, dict):  # {"url": ...}
            image = image.get("url")
        prompt = str(src.get("prompt") or "").strip()
        if prompt and (not task or query is None or not image):
            # Dashboard form: "<task>: <query> <image url>"
            url = _URL_RE.search(prompt)
            if url and not image:
                image = url.group(0)
            text = _URL_RE.sub("", prompt).strip()
            head, sep, rest = text.partition(":")
            if sep and head.strip().lower() in _TASKS:
                task = task or head.strip().lower()
                text = rest.strip()
            if query is None:
                query = text
        task = task or "detection"
        if task not in _TASKS:
            raise InvalidInput(f"`task` must be one of {', '.join(_TASKS)}, got {task!r}")
        if not image:
            raise InvalidInput("`image` (base64, data URL or https URL) is required")
        if isinstance(query, list):
            query = [str(q) for q in query if str(q).strip()]
        query = query if query is not None else ""
        if task not in ("ocr",) and not query:
            raise InvalidInput(f"`query` is required for task {task}")
        return task, query, str(image)

    def process(self, job: Job) -> Result:
        torch = self.torch
        src = job.input
        passthrough = job.input.get("passthrough_body")
        if isinstance(passthrough, dict) and passthrough:
            src = passthrough
        task, query, image_src = self._request(src)
        image = _load_image(image_src)
        max_new = max(16, min(8192, int(src.get("max_new_tokens") or 2048)))
        want_masks = bool(src.get("return_masks", True))

        with self.lock:  # one request at a time on this GPU
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            from wevisone_vl.prompts import build_prompt

            prompt = build_prompt(task, query, image_size=image.size)
            response = self.model.generate_text(image, prompt, max_new_tokens=max_new, temperature=0.0)
            t_gen = time.perf_counter() - t0
            boxes, masks = [], []
            if "<OBJ>" in response:
                boxes, masks = self.model.decode_masks(image, prompt, response)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30

        from wevisone_vl.engine import InferenceResult

        result = InferenceResult(response=response, prompt=prompt, boxes=boxes, masks=masks, latency_s=seconds)
        preds = self.vis.parse_predictions(response, image.width, image.height)
        n_inst = sum(len(v) for v in preds.values())
        count = None
        if task.startswith("count"):
            m = re.search(r"(\d+)\s*$", re.sub(r"<[^>]*>", " ", response).strip())
            count = int(m.group(1)) if m else n_inst
        rejected = response.strip().lower().rstrip(".") == "none" or (n_inst == 0 and "none" in response.lower())

        # Annotated image as JPEG (small); the cutout keeps its alpha as PNG.
        images = [{"b64_json": _png_b64(self.vis.render_overlay(image, result), fmt="JPEG"), "kind": "overlay", "format": "jpeg"}]
        if masks:
            alpha = self.vis.combined_alpha(masks, image.width, image.height)
            cut = _png_b64(self.vis.make_cutout(image, alpha), _MAX_SIDE_CUTOUT)
            if len(images[0]["b64_json"]) + len(cut) <= _MAX_PAYLOAD_B64:
                images.append({"b64_json": cut, "kind": "cutout", "format": "png"})
        mask_list = []
        if masks and want_masks and task in _PIXEL_TASKS:
            import numpy as np
            from PIL import Image

            used = sum(len(i["b64_json"]) for i in images)
            for m in masks:
                b64 = _png_b64(Image.fromarray(np.uint8(np.clip(m, 0, 1) * 255), mode="L"), _MAX_SIDE_CUTOUT)
                if used + len(b64) > _MAX_PAYLOAD_B64:
                    break
                used += len(b64)
                mask_list.append(b64)

        return Result(
            {
                "created": int(time.time()),
                "data": images,
                "task": task,
                "query": query,
                "prompt": prompt,
                "response": response,
                "predictions": preds,
                "instances": n_inst,
                "count": count,
                "rejected": bool(rejected),
                "masks": mask_list,
                "masks_total": len(masks),
                "width": image.width,
                "height": image.height,
                "seconds": round(seconds, 3),
                "seconds_generate": round(t_gen, 3),
                "seconds_decode": round(seconds - t_gen, 3),
                "vram_peak_gib": round(peak, 2),
                "engine": self.engine,
                "model": self.model_id,
            }
        )

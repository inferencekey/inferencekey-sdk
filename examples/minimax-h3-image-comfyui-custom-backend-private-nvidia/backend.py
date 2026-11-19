"""A custom **text2image** backend for MiniMax-H3-Image on quantized weights.

MiniMax-H3-Image (MiniMax, MiniMax-H3-Image Community License) is a 33.1B
single-stream diffusion transformer that generates and edits images (up to 16
reference pictures) at about 16 megapixels in 6 steps, with the Qwen3-VL-32B
text encoder of MiniMax-H3. The official BF16 checkpoint needs about 135 GB
with its encoder. Comfy-Org publishes repackaged, pruned and quantized files
that bring it down to one card:

* ``diffusion_models/minimax_h3_image_pruned_w4a8.safetensors`` (12.5 GB)
* ``text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`` (15.7 GB)
* ``vae/minimax_h3_image_vae_fp16.safetensors`` (1.2 GB)

Those files are in ComfyUI's format, so this backend runs ComfyUI headless on
a loopback port and drives it through its HTTP API: it downloads the ComfyUI
release and the three files, starts the server, and turns each request into
the graph of the official template (UNETLoader, CLIPLoader ``minimax``,
VAELoader, MiniMaxH3ImageConditioning, KSampler 6 steps euler/simple cfg 1,
VAEDecode). ComfyUI's own memory manager decides what stays on the GPU: on a
48 GB card it used 31 GB, and under an 8 GB cap it produced the same images.

Contract shape, accepts BOTH callers:

* input, native ``/process``: ``{"prompt": "...", "size": "2048x2048",
  "seed": 7}`` (only ``prompt`` is required). Optional: ``width``/``height``
  (instead of ``size``), ``image`` (one base64 string or data URL, or a list
  of up to 16: the reference pictures of an edit, ``<Picture 1>``,
  ``<Picture 2>``, ... in the prompt), ``output_format`` (``jpeg`` default,
  ``png`` or ``webp``).
* input, public API (``POST .../v1/images/generations``): the manager forwards
  the caller's OpenAI JSON in ``input.passthrough_body``; fields are read from
  there.
* output: the OpenAI images shape ``{"created": ..., "data": [{"b64_json":
  ...}]}`` plus ``width``/``height``/``seed``, ``seconds`` (the whole graph),
  ``stage_seconds`` (text encoder, sampler, VAE decode, as ComfyUI reported
  each node), ``vram_peak_mib`` (the card's ``nvidia-smi`` peak during the
  request) and ``output_format``.

Like the authors' servers, the backend answers 400 to ``num_inference_steps``
other than 6, ``guidance_scale`` other than 1 and to a ``negative_prompt``:
the checkpoint is distilled for exactly that.

A 16-megapixel PNG is 20 to 30 MB and the result travels through an 8 MB
message that carries the image twice, so the encoded image has to stay under
about 2.5 MB: the default output is JPEG and the quality drops until it fits
(the response says which quality was used).

Environment (set through the workload's ``config.env``):

* ``H3_DIT`` / ``H3_TEXT_ENCODER`` / ``H3_VAE``: file names inside the
  Comfy-Org repo folders (defaults above; for example
  ``minimax_h3_image_pruned_int8_convrot.safetensors``).
* ``H3_VRAM_LIMIT_GIB``: cap the GPU memory ComfyUI may use (a hard torch
  cap plus ``--reserve-vram`` for the rest of the card). This is how we
  emulated smaller cards on a 48 GB one.
* ``H3_COMFY_ARGS``: extra ComfyUI flags (for example ``--lowvram``).
* ``H3_WARMUP``: ``1`` (default) generates one small image in ``setup()`` so
  the first real request does not pay for loading the weights.

Run it by hand (from this folder, with requirements installed)::

    python -m inferencekey.backend.serve --port 8099 \\
        --backend backend:MiniMaxH3ImageBackend

Attribution: MiniMax-H3-Image by MiniMax
(https://huggingface.co/MiniMaxAI/MiniMax-H3-Image, MiniMax-H3-Image Community
License Agreement: read section IV before you offer it as a hosted service);
quantized repack by Comfy-Org (https://huggingface.co/Comfy-Org/MiniMax-H3-Image);
ComfyUI by Comfy-Org, GPL-3.0 (https://github.com/Comfy-Org/ComfyUI).
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import io
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tarfile
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

#: What this backend was tested with (2026-11-19).
COMFYUI_TAG = "v0.45.0"
COMFYUI_TARBALL = f"https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/{COMFYUI_TAG}.tar.gz"
WEIGHTS_REPO = "Comfy-Org/MiniMax-H3-Image"
WEIGHTS_REVISION = "89c635085f539d03ad35485ed542c5a286e3c6d6"
DEFAULT_DIT = "minimax_h3_image_pruned_w4a8.safetensors"
DEFAULT_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VAE = "minimax_h3_image_vae_fp16.safetensors"

STEPS = 6
_MULTIPLE = 32
_MIN_AREA, _MAX_AREA = 512 * 512, 20_500_000
_MAX_REFERENCES = 16
#: The result travels through an 8 MB message and the worker's result message
#: is a bit over twice the base64 (we measured 14.4 MB for 7.2 MB and 8.6 MB
#: for 3.9 MB), so the base64 has to stay well under half of the limit.
_MAX_B64_BYTES = 3_400_000
_JPEG_QUALITIES = (95, 90, 85, 80, 75, 70, 60, 50)

#: Node ids of the graph, used to name the stage timings.
_STAGES = {"2": "text_encoder_load", "5": "text_encoder", "1": "dit_load", "7": "sampler", "3": "vae_load",
           "8": "vae_decode", "9": "save"}

_VRAM_LIMIT_NODE = '''"""Hard cap on the GPU memory this ComfyUI process may allocate (H3_VRAM_LIMIT_GIB)."""
import os
import torch

_limit = float(os.environ.get("H3_VRAM_LIMIT_GIB") or 0)
if _limit > 0 and torch.cuda.is_available():
    _total = torch.cuda.mem_get_info(0)[1]
    torch.cuda.set_per_process_memory_fraction(min(1.0, _limit * 2**30 / _total), 0)
    print(f"ik_vram_limit: capped at {_limit} GiB of {_total / 2**30:.1f} GiB")
NODE_CLASS_MAPPINGS = {}
'''


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class _VramSampler:
    """Polls ``nvidia-smi`` twice a second and keeps the peak of memory.used."""

    def __init__(self) -> None:
        self.peak = 0
        self.last = 0
        self._proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0", "-lms", "500"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
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

    def stop(self) -> None:
        self._proc.terminate()


class MiniMaxH3ImageBackend(CustomBackend):
    """MiniMax-H3-Image (quantized Comfy-Org files) behind a headless ComfyUI."""

    name = "minimax-h3-image-comfyui"
    version = "0.1.2"
    task_type = "text2image"
    requirements = "requirements.txt"
    requires_python = ">=3.11,<3.13"

    # ------------------------------------------------------------------ setup

    def setup(self, ctx: BackendContext) -> None:
        t0 = time.time()
        self.lock = threading.Lock()
        self.dit = os.environ.get("H3_DIT") or DEFAULT_DIT
        self.text_encoder = os.environ.get("H3_TEXT_ENCODER") or DEFAULT_TEXT_ENCODER
        self.vae = os.environ.get("H3_VAE") or DEFAULT_VAE
        # Local disk of the container, never a network volume: the weights are
        # read back in full every time ComfyUI moves a model.
        work = Path(os.environ.get("H3_WORKDIR") or Path.home() / ".cache" / "inferencekey" / "minimax-h3-image")
        self.comfy = work / f"ComfyUI-{COMFYUI_TAG}"
        self.out_dir = work / "output"
        self.in_dir = work / "input"
        for d in (self.out_dir, self.in_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._fetch_comfyui(work)
        t_code = time.time()
        self._fetch_weights()
        t_weights = time.time()
        self.vram = _VramSampler()
        self._start_comfyui(work)
        t_server = time.time()
        _log(
            f"minimax-h3-image: ComfyUI {COMFYUI_TAG} up on :{self.port} (code {t_code - t0:.1f}s, "
            f"weights {t_weights - t_code:.1f}s, server start {t_server - t_weights:.1f}s); "
            f"DiT {self.dit}, text encoder {self.text_encoder}"
        )
        if (os.environ.get("H3_WARMUP") or "1") != "0":
            out = self._generate("A red apple on a white table, studio light.", 1024, 1024, 1, [])
            _log(
                f"minimax-h3-image: warm-up 1024x1024 in {out['seconds']} s, stages {out['stage_seconds']}, "
                f"card peak {out['vram_peak_mib']} MiB"
            )
        _log(f"minimax-h3-image ready: setup {time.time() - t0:.1f}s")

    def _fetch_comfyui(self, work: Path) -> None:
        if (self.comfy / "main.py").is_file():
            return
        work.mkdir(parents=True, exist_ok=True)
        tmp = work / "comfyui.tar.gz"
        urllib.request.urlretrieve(COMFYUI_TARBALL, tmp)
        staging = work / "staging"
        with tarfile.open(tmp) as tar:
            tar.extractall(staging, filter="data")
        (next(staging.iterdir())).rename(self.comfy)
        staging.rmdir()
        tmp.unlink()

    def _fetch_weights(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            WEIGHTS_REPO,
            revision=WEIGHTS_REVISION,
            allow_patterns=[
                f"diffusion_models/{self.dit}",
                f"text_encoders/{self.text_encoder}",
                f"vae/{self.vae}",
            ],
            local_dir=str(self.comfy / "models"),
            max_workers=16,
        )
        for sub, name in (("diffusion_models", self.dit), ("text_encoders", self.text_encoder), ("vae", self.vae)):
            if not (self.comfy / "models" / sub / name).is_file():
                raise RuntimeError(f"{sub}/{name} is not in {WEIGHTS_REPO}@{WEIGHTS_REVISION[:8]}")

    def _start_comfyui(self, work: Path) -> None:
        self.port = _free_port()
        args = [
            sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(self.port),
            "--disable-auto-launch", "--output-directory", str(self.out_dir), "--input-directory", str(self.in_dir),
        ]
        limit = float(os.environ.get("H3_VRAM_LIMIT_GIB") or 0)
        if limit > 0:
            (self.comfy / "custom_nodes" / "ik_vram_limit.py").write_text(_VRAM_LIMIT_NODE)
            total_mib = int(subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True, check=True).stdout.strip())
            reserve = max(0.0, total_mib / 1024 - limit)
            args += ["--reserve-vram", f"{reserve:.1f}"]
        args += shlex.split(os.environ.get("H3_COMFY_ARGS") or "")
        self.log_path = work / "comfyui.log"
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen(
            args, cwd=str(self.comfy), env=env, stdout=self.log_path.open("ab"), stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # ComfyUI must not outlive this process (the worker restarts backends).
        subprocess.Popen(
            ["bash", "-c", f"while kill -0 {os.getpid()} 2>/dev/null; do sleep 2; done; kill -TERM -{self.proc.pid}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        atexit.register(self.teardown)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited with {self.proc.returncode}: {self._log_tail()}")
            try:
                self._get("/system_stats", timeout=3)
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError(f"ComfyUI did not answer in 600 s: {self._log_tail()}")
        info = self._get("/object_info/MiniMaxH3ImageConditioning")
        if "MiniMaxH3ImageConditioning" not in info:
            raise RuntimeError(f"ComfyUI {COMFYUI_TAG} has no MiniMaxH3ImageConditioning node")

    def teardown(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        vram = getattr(self, "vram", None)
        if vram is not None:
            vram.stop()

    def _log_tail(self, lines: int = 25) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return "(no log)"

    # ------------------------------------------------------------- ComfyUI API

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def _get(self, path: str, timeout: float = 30):
        with urllib.request.urlopen(self._url(path), timeout=timeout) as r:
            return json.loads(r.read())

    def _post(self, path: str, body: dict):
        req = urllib.request.Request(
            self._url(path), data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI {path} answered {e.code}: {e.read().decode(errors='replace')[:800]}")

    def _graph(self, prompt: str, width: int, height: int, seed: int, references: list[str], prefix: str) -> dict:
        cond = {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "width": width, "height": height}
        graph = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": self.dit, "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": self.text_encoder, "type": "minimax", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": self.vae}},
            "5": {"class_type": "MiniMaxH3ImageConditioning", "inputs": cond},
            "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
            "7": {"class_type": "KSampler",
                  "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["5", 1],
                             "seed": seed, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler",
                             "scheduler": "simple", "denoise": 1.0}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
        }
        for i, name in enumerate(references, start=1):
            node = str(100 + i)
            graph[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
            cond[f"images.image_{i}"] = [node, 0]
        return graph

    async def _run_graph(self, graph: dict, client_id: str) -> tuple[str, dict]:
        """Queue the graph and follow it on the websocket: per-node seconds."""
        import aiohttp

        marks: list[tuple[str, float]] = []
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self._url(f"/ws?clientId={client_id}"), max_msg_size=0) as ws:
                prompt_id = self._post("/prompt", {"prompt": graph, "client_id": client_id})["prompt_id"]
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    event = json.loads(msg.data)
                    data = event.get("data") or {}
                    if data.get("prompt_id") != prompt_id:
                        continue
                    if event["type"] == "executing":
                        if data.get("node") is None:
                            break
                        marks.append((str(data["node"]), time.time()))
                    elif event["type"] == "execution_error":
                        raise RuntimeError(
                            f"ComfyUI node {data.get('node_type')}: {data.get('exception_type')}: "
                            f"{str(data.get('exception_message'))[:600]}"
                        )
                    elif event["type"] == "execution_interrupted":
                        raise RuntimeError("ComfyUI interrupted the generation")
                    elif event["type"] == "execution_success":
                        break
        end = time.time()
        stages: dict[str, float] = {}
        for (node, start), (_, stop) in zip(marks, marks[1:] + [("", end)]):
            key = _STAGES.get(node)
            if key:
                stages[key] = round(stages.get(key, 0.0) + stop - start, 2)
        return prompt_id, stages

    def _generate(self, prompt: str, width: int, height: int, seed: int, references: list[bytes]) -> dict:
        from PIL import Image

        job = uuid.uuid4().hex[:12]
        names = []
        for i, raw in enumerate(references, start=1):
            try:
                img = Image.open(io.BytesIO(raw))
                img.load()
            except Exception:
                raise InvalidInput(f"reference image {i} is not a readable image")
            name = f"ik_{job}_{i}.png"
            img.convert("RGB").save(self.in_dir / name)
            names.append(name)
        graph = self._graph(prompt, width, height, seed, names, f"ik_h3_{job}")
        with self.lock:  # one generation at a time
            self.vram.reset()
            t0 = time.time()
            prompt_id, stages = asyncio.run(self._run_graph(graph, job))
            seconds = time.time() - t0
            peak = self.vram.peak
        history = self._get(f"/history/{prompt_id}").get(prompt_id) or {}
        images = [im for out in (history.get("outputs") or {}).values() for im in out.get("images", [])]
        if not images:
            raise RuntimeError(f"ComfyUI returned no image: {json.dumps(history.get('status'))[:600]}")
        path = self.out_dir / images[0].get("subfolder", "") / images[0]["filename"]
        image = Image.open(path)
        image.load()
        path.unlink(missing_ok=True)
        for name in names:
            (self.in_dir / name).unlink(missing_ok=True)
        return {"image": image, "seconds": round(seconds, 2), "stage_seconds": stages, "vram_peak_mib": peak}

    # ---------------------------------------------------------------- process

    def process(self, job: Job) -> Result:
        src = job.input
        passthrough = job.input.get("passthrough_body")
        if isinstance(passthrough, dict) and passthrough:
            src = passthrough

        prompt = str(src.get("prompt") or "").strip()
        if not prompt:
            raise InvalidInput("`prompt` is required")
        references = self._references(src)
        width = int(src.get("width") or 0)
        height = int(src.get("height") or 0)
        size = src.get("size")
        if isinstance(size, str) and "x" in size.lower():
            try:
                w_raw, h_raw = size.lower().split("x", 1)
                width, height = int(w_raw), int(h_raw)
            except ValueError:
                raise InvalidInput(f"`size` must look like '2048x2048', got {size!r}")
        if not (width and height):
            if references:
                width = height = 0  # the node sizes the canvas from <Picture 1>
            else:
                width = height = 2048
        if width or height:
            if width % _MULTIPLE or height % _MULTIPLE:
                raise InvalidInput(f"width and height must be multiples of {_MULTIPLE}")
            if not (_MIN_AREA <= width * height <= _MAX_AREA):
                raise InvalidInput("the image area must be between 512x512 and about 20 megapixels")
        if int(src.get("n", 1)) != 1:
            raise InvalidInput("`n` must be 1 (one image per request)")
        steps = src.get("steps") or src.get("num_inference_steps")
        if steps is not None and int(steps) != STEPS:
            raise InvalidInput(f"MiniMax-H3-Image is distilled for exactly {STEPS} steps")
        guidance = src.get("guidance_scale")
        if guidance is not None and float(guidance) != 1.0:
            raise InvalidInput("MiniMax-H3-Image runs without classifier-free guidance (guidance_scale 1)")
        if src.get("negative_prompt"):
            raise InvalidInput("MiniMax-H3-Image does not support a negative prompt")
        fmt = str(src.get("output_format") or "jpeg").lower()
        if fmt not in ("png", "jpeg", "jpg", "webp"):
            raise InvalidInput(f"`output_format` must be png, jpeg or webp, got {fmt!r}")
        seed = src.get("seed")
        seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "big")

        out = self._generate(prompt, width, height, seed, references)
        image = out["image"]
        payload, used_fmt, quality = self._encode(image, fmt)
        return Result(
            {
                "created": int(time.time()),
                "data": [{"b64_json": payload}],
                "width": image.width,
                "height": image.height,
                "seed": seed,
                "steps": STEPS,
                "references": len(references),
                "seconds": out["seconds"],
                "stage_seconds": out["stage_seconds"],
                "vram_peak_mib": out["vram_peak_mib"],
                "output_format": used_fmt,
                "output_quality": quality,
                "dit": self.dit,
                "text_encoder": self.text_encoder,
                "model": "MiniMaxAI/MiniMax-H3-Image",
            }
        )

    @staticmethod
    def _references(src: dict) -> list[bytes]:
        raw = src.get("image") if src.get("image") is not None else src.get("images")
        if raw is None:
            return []
        items = raw if isinstance(raw, list) else [raw]
        if len(items) > _MAX_REFERENCES:
            raise InvalidInput(f"at most {_MAX_REFERENCES} reference images")
        out = []
        for i, item in enumerate(items, start=1):
            if isinstance(item, dict):
                item = item.get("b64_json") or item.get("image_url") or item.get("url") or ""
            if not isinstance(item, str) or not item:
                raise InvalidInput(f"reference image {i} must be a base64 string or a data URL")
            if item.startswith("data:"):
                item = item.split(",", 1)[-1]
            try:
                out.append(base64.b64decode(item, validate=False))
            except Exception:
                raise InvalidInput(f"reference image {i} is not valid base64")
        return out

    @staticmethod
    def _encode(image, fmt: str):
        """Encode under the message limit: PNG or WebP if they fit, else JPEG at falling quality."""
        image = image.convert("RGB")
        if fmt in ("png", "webp"):
            buf = io.BytesIO()
            if fmt == "png":
                image.save(buf, format="PNG", compress_level=3)
            else:
                image.save(buf, format="WEBP", quality=92, method=2)
            data = base64.b64encode(buf.getvalue())
            if len(data) <= _MAX_B64_BYTES:
                return data.decode("ascii"), fmt, None
        for quality in _JPEG_QUALITIES:
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=quality, subsampling=0 if quality >= 90 else 2)
            data = base64.b64encode(buf.getvalue())
            if len(data) <= _MAX_B64_BYTES:
                return data.decode("ascii"), "jpeg", quality
        raise RuntimeError("the image does not fit the result message even as JPEG quality 50")

"""RelayVSR as an InferenceKey custom backend: streaming 4x video super-resolution.

RelayVSR (Xijun Wang, Xin Li, Zirui Lang, Suhang Yao, Haoran Li and Zhibo Chen;
Apache-2.0, arXiv 2609.37850) splits the work between two models:

* a **reference generator** adapted from Wan2.2-TI2V-5B (``wan.safetensors``,
  10.1 GB bf16, prompt embeddings included) that produces a detailed latent
  for sparse keyframes only, one every ``window_size - 1`` frames (every 15 by
  default), in a single denoising step;
* a **small VSR network** (``flash.safetensors``, 128 MB) that upscales EVERY
  frame 4x, reading the two keyframe references at the ends of its window
  (dual-endpoint) plus the previous frame.

No built-in InferenceKey backend serves it, so this backend runs the authors'
own inference code (``refvsr_inference`` from github.com/kopperx/RelayVSR,
fetched at a pinned commit) exactly like their ``inference.py``: bf16, one
step at timestep 1000, seed 42, ``CollaborativeFrameStream``.

``setup()`` (once per process)
    downloads the authors' source and the two checkpoints, loads both models
    on the GPU and, unless ``RVSR_WARMUP=0``, runs a tiny clip so the
    FlexAttention / Triton kernels are compiled before the first request.

``process(job)`` (one clip per request)
    input: the clip as ``video_b64`` (MP4/WebM/MOV bytes, base64), or
    ``video_url`` (https), or ``"demo": true`` (the authors' bundled demo
    clip, 624 x 360 at 16 fps). Optional: ``window_size`` (2 to 64, default
    16: a reference every 15 frames), ``max_frames`` (default 96, up to
    ``RVSR_MAX_FRAMES``), ``start_frame``, ``seed``, ``fps``, ``crf``.
    output: ``video_b64`` (H.264 MP4, exactly 4x the input size, no audio)
    plus ``width``/``height``/``frames``/``fps``/``keyframes`` and
    ``timings``: ``model_fps`` (frames per second of model time, the
    authors' measure), ``wan_ms`` / ``flash_ms`` (GPU time per stage),
    ``first_frame_ms``, ``wall_s`` (with decode and encode), and
    ``peak_vram_mib``.

Send the request as JSON in the last user message of ``/v1/chat/completions``
(or as top-level fields). The answer travels through an 8 MB message: the
backend raises the H.264 CRF until the base64 fits and reports the CRF it
used. A full-quality copy (CRF 18) is kept on the worker in
``RVSR_OUTPUT_DIR`` when that variable is set.

Settings (environment, or the workload ``config.env``):
``RVSR_VRAM_LIMIT_GIB`` caps the GPU memory this process may use
(``torch.cuda.set_per_process_memory_fraction``: that is how we emulated
smaller cards), ``RVSR_DTYPE`` (bf16 | fp16), ``RVSR_MAX_FRAMES``,
``RVSR_MAX_PIXELS`` (largest input frame, width x height).

Run it by hand (from this folder, with requirements installed)::

    python -m inferencekey.backend.serve --port 8099 --backend backend:RelayVSRBackend
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict

from inferencekey.backend import BackendContext, CustomBackend, Job, Result

try:  # newer SDKs answer 400 for bad client input
    from inferencekey.backend import InvalidInput
except ImportError:  # pragma: no cover - older SDK
    InvalidInput = ValueError

#: Upstream pins this example was tested with (2026-11-14).
SOURCE_REPO = "kopperx/RelayVSR"
SOURCE_COMMIT = "e856773350f53036d825404c9ea9228e6dffadca"
HF_REPO = "kopper/RelayVSR"
HF_REVISION = "751954dc8adf34aa1d636a39fc548b9089703c85"

#: The result travels through an 8 MB message: keep the base64 under this.
_MAX_B64_BYTES = 7_500_000
_MAX_INPUT_BYTES = 5_000_000


def _log(msg: str) -> None:
    print(f"[relayvsr-backend] {msg}", file=sys.stderr, flush=True)


def _fetch_source(dest: Path) -> None:
    """Download the authors' inference code (not on PyPI) at the pinned commit."""
    url = f"https://codeload.github.com/{SOURCE_REPO}/tar.gz/{SOURCE_COMMIT}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".dl-"))
    req = urllib.request.Request(url, headers={"User-Agent": "inferencekey-relayvsr-backend"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        keep = ("/refvsr_inference/", "/assets/demo/input.mp4", "/LICENSE")
        members = [m for m in tf.getmembers() if any(k in m.name or m.name.endswith(k) for k in keep)]
        try:
            tf.extractall(tmp, members=members, filter="data")
        except TypeError:  # Python < 3.12
            tf.extractall(tmp, members=members)
    inner = next(p for p in tmp.iterdir() if p.is_dir())
    if dest.exists():
        shutil.rmtree(dest)
    inner.rename(dest)
    tmp.rmdir()


class RelayVSRBackend(CustomBackend):
    """RelayVSR 4x video super-resolution (one clip in, one 4x clip out)."""

    name = "relayvsr-video-upscale"
    version = "0.1.0"
    # The SDK packager has no video task type yet; the output is still MP4 video on the chat route.
    task_type = "text2image"
    requirements = "requirements.txt"
    requires_python = ">=3.12,<3.13"

    # ------------------------------------------------------------------ setup

    def setup(self, ctx: BackendContext) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        t0 = time.time()
        cache = Path(os.environ.get("RVSR_CACHE_DIR", Path.home() / ".cache" / "relayvsr-backend"))
        src = cache / f"src-{SOURCE_COMMIT[:12]}"
        if not (src / "refvsr_inference" / "runtime" / "streaming.py").is_file() or not (src / "assets").is_dir():
            _fetch_source(src)
        sys.path.insert(0, str(src))

        import torch
        from huggingface_hub import hf_hub_download

        wan_path = hf_hub_download(HF_REPO, "wan.safetensors", revision=HF_REVISION)
        flash_path = hf_hub_download(HF_REPO, "flash.safetensors", revision=HF_REVISION)
        t_download = time.time()

        from refvsr_inference.checkpoints import load_checkpoints
        from refvsr_inference.models.wan.sampling import WanVSRPipeline

        self.torch = torch
        self.lock = threading.Lock()
        if not torch.cuda.is_available():
            raise RuntimeError("RelayVSR needs a CUDA GPU")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.device_name = torch.cuda.get_device_name(self.device)
        self.vram_limit_gib = float(os.environ.get("RVSR_VRAM_LIMIT_GIB") or 0)
        free, total = torch.cuda.mem_get_info(self.device)
        if self.vram_limit_gib > 0:
            # Behave like a smaller card: the allocator refuses to go past the cap.
            torch.cuda.set_per_process_memory_fraction(min(1.0, self.vram_limit_gib * 2**30 / total), self.device)
        dtype_name = (os.environ.get("RVSR_DTYPE") or "bf16").lower()
        dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
        if dtype_name not in dtypes:
            raise ValueError("RVSR_DTYPE must be bf16 or fp16")
        self.dtype_name, self.dtype = dtype_name, dtypes[dtype_name]
        self.max_frames = int(os.environ.get("RVSR_MAX_FRAMES") or 241)
        self.max_pixels = int(os.environ.get("RVSR_MAX_PIXELS") or 960 * 544)
        self.output_dir = os.environ.get("RVSR_OUTPUT_DIR") or ""
        self.demo_clip = src / "assets" / "demo" / "input.mp4"

        self.wan, self.flash = load_checkpoints(wan_path, flash_path, device=self.device, dtype=self.dtype)
        self.sampler = WanVSRPipeline(
            scheduler_config={"num_train_timesteps": 1000, "shift": 5.0},
            sampling={
                "execution_mode": "streaming",
                "num_inference_steps": 1,
                "timesteps": [1000],
                "guidance_scale": 1.0,
                "seed": 42,
                "kv_cache_window_size": self.wan.temporal_context_frames,
            },
        )
        self.sampler.prepare_streaming(
            model=self.wan, prompt_embeds=self.wan.prompt_embeds, device=self.device, compute_dtype=self.dtype
        )
        torch.cuda.synchronize()
        t_load = time.time()
        weights_gib = torch.cuda.memory_allocated(self.device) / 2**30

        warmup_s = 0.0
        if (os.environ.get("RVSR_WARMUP") or "1") != "0":
            frames = [torch.randint(0, 255, (3, 128, 224), dtype=torch.uint8) for _ in range(17)]
            self._run(frames, window_size=16, seed=42, writer=None)
            warmup_s = time.time() - t_load
        torch.cuda.empty_cache()
        self.timings = {
            "download_s": round(t_download - t0, 1),
            "load_s": round(t_load - t_download, 1),
            "warmup_s": round(warmup_s, 1),
            "weights_gib": round(weights_gib, 2),
        }
        _log(
            f"ready on {self.device_name}: download {self.timings['download_s']}s, load {self.timings['load_s']}s, "
            f"warmup {self.timings['warmup_s']}s, weights in VRAM {weights_gib:.2f} GiB, dtype {dtype_name}, "
            f"VRAM cap {self.vram_limit_gib or 'none'} (card {total / 2**30:.1f} GiB)"
        )

    # ---------------------------------------------------------------- inputs

    @staticmethod
    def _fields(inp: Dict[str, Any]) -> Dict[str, Any]:
        """Top-level fields, ``extra_body``, and a JSON object sent as the last user message."""
        f = dict(inp)
        if isinstance(inp.get("extra_body"), dict):
            f.update(inp["extra_body"])
        if isinstance(inp.get("passthrough_body"), dict):
            f.update(inp["passthrough_body"])
        for msg in reversed(inp.get("messages") or []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
                if isinstance(content, str) and content.strip().startswith("{"):
                    try:
                        f.update(json.loads(content))
                    except json.JSONDecodeError as e:
                        raise InvalidInput(f"user message is not valid JSON: {e}") from e
                elif isinstance(content, str) and content.strip().lower() in ("demo", "run the demo"):
                    f["demo"] = True
                break
        return f

    def _input_file(self, f: Dict[str, Any], tmp: Path) -> Path:
        if f.get("video_b64"):
            try:
                data = base64.b64decode(str(f["video_b64"]), validate=False)
            except Exception as e:  # noqa: BLE001
                raise InvalidInput(f"video_b64 is not valid base64: {e}") from e
        elif f.get("video_url"):
            url = str(f["video_url"])
            if not url.startswith("https://"):
                raise InvalidInput("video_url must be https")
            req = urllib.request.Request(url, headers={"User-Agent": "inferencekey-relayvsr-backend"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read(_MAX_INPUT_BYTES + 1)
        elif f.get("demo"):
            return self.demo_clip
        else:
            raise InvalidInput("send the clip as video_b64 (base64 MP4), video_url (https) or demo: true")
        if len(data) > _MAX_INPUT_BYTES:
            raise InvalidInput(f"input clip is larger than {_MAX_INPUT_BYTES} bytes; send a shorter clip")
        path = tmp / "input.bin"
        path.write_bytes(data)
        return path

    # --------------------------------------------------------------- running

    def _run(self, frames, *, window_size: int, seed: int, writer) -> Dict[str, Any]:
        """Stream ``frames`` (uint8 CHW tensors) through RelayVSR; write each 4x frame to ``writer``."""
        torch = self.torch
        from refvsr_inference.runtime.streaming import CollaborativeFrameStream, InputFrame

        stream = CollaborativeFrameStream(
            wan_model=self.wan,
            flash_model=self.flash,
            wan_pipeline=self.sampler,
            prompt_embeds=self.wan.prompt_embeds,
            device=self.device,
            compute_dtype=self.dtype,
            window_size=window_size,
            seed=seed,
        )
        spans: Dict[str, float] = {}
        first_frame_ms = None
        n = 0
        size = None
        t0 = time.perf_counter()
        packets = (InputFrame(i, rgb, source_time=0.0) for i, rgb in enumerate(frames))
        for result in stream.frames(packets):
            rgb = result.rgb[0].add(1).mul_(127.5).clamp_(0, 255).to(torch.uint8)
            host = rgb.permute(1, 2, 0).contiguous().cpu()
            if first_frame_ms is None:
                first_frame_ms = (time.perf_counter() - t0) * 1000
            for stage, span in result.timings:
                spans[stage] = spans.get(stage, 0.0) + span.milliseconds()
            if writer is not None:
                writer(host.numpy())
            size = (host.shape[1], host.shape[0])
            n += 1
        torch.cuda.synchronize()
        model_ms = sum(spans.values())
        return {
            "frames": n,
            "size": size,
            "keyframes": stream.keyframe_count,
            "keyframe_indices": list(stream.last_keyframe_indices),
            "spans_ms": {k: round(v, 1) for k, v in spans.items()},
            "model_ms": model_ms,
            "first_frame_ms": first_frame_ms,
            "stream_s": time.perf_counter() - t0,
        }

    @staticmethod
    def _encoder(path: Path, fps: Fraction, crf: int):
        import av

        container = av.open(str(path), mode="w", format="mp4")
        state: Dict[str, Any] = {"stream": None}

        def write(rgb) -> None:
            if state["stream"] is None:
                h, w = rgb.shape[:2]
                s = container.add_stream("libx264", rate=fps)
                s.height, s.width = h, w
                s.pix_fmt = "yuv420p"
                s.options = {"crf": str(crf), "preset": "fast"}
                s.codec_context.thread_count = 0
                state["stream"] = s
            for packet in state["stream"].encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)

        def close() -> None:
            if state["stream"] is not None:
                for packet in state["stream"].encode():
                    container.mux(packet)
            container.close()

        return write, close

    @staticmethod
    def _reencode(src: Path, dst: Path, crf: int) -> None:
        import av

        with av.open(str(src)) as inp, av.open(str(dst), mode="w", format="mp4") as out:
            vin = inp.streams.video[0]
            s = out.add_stream("libx264", rate=vin.average_rate)
            s.height, s.width, s.pix_fmt = vin.height, vin.width, "yuv420p"
            s.options = {"crf": str(crf), "preset": "fast"}
            s.codec_context.thread_count = 0
            for frame in inp.decode(vin):
                for packet in s.encode(frame.reformat(format="yuv420p")):
                    out.mux(packet)
            for packet in s.encode():
                out.mux(packet)

    def process(self, job: Job) -> Result:
        torch = self.torch
        from refvsr_inference.media import input_fps, iter_frames

        f = self._fields(job.input)
        window_size = int(f.get("window_size") or 16)
        if not 2 <= window_size <= 64:
            raise InvalidInput("window_size must be between 2 and 64 (a reference every window_size - 1 frames)")
        max_frames = int(f.get("max_frames") or 96)
        if not 2 <= max_frames <= self.max_frames:
            raise InvalidInput(f"max_frames must be between 2 and {self.max_frames}")
        start_frame = max(0, int(f.get("start_frame") or 0))
        seed = int(f.get("seed") if f.get("seed") is not None else 42)
        crf = f.get("crf")

        with tempfile.TemporaryDirectory(prefix="relayvsr-") as tmpdir:
            tmp = Path(tmpdir)
            src = self._input_file(f, tmp)
            t_wall = time.perf_counter()
            try:
                fps = input_fps(src, f.get("fps"))
                frames = list(iter_frames(src, max_frames=max_frames, start_frame=start_frame))
            except Exception as e:  # noqa: BLE001 - bad media is the caller's problem
                raise InvalidInput(f"could not read the clip: {e}") from e
            h, w = frames[0].shape[-2:]
            if h * w > self.max_pixels:
                raise InvalidInput(f"input is {w}x{h}; the largest accepted frame is {self.max_pixels} pixels")
            decode_s = time.perf_counter() - t_wall

            full = tmp / "full.mp4"
            write, close = self._encoder(full, fps, 18)
            with self.lock, torch.inference_mode():  # one clip at a time
                torch.cuda.reset_peak_memory_stats(self.device)
                try:
                    stats = self._run(frames, window_size=window_size, seed=seed, writer=write)
                finally:
                    close()
                peak = torch.cuda.max_memory_allocated(self.device) / 2**20
                peak_reserved = torch.cuda.max_memory_reserved(self.device) / 2**20
                torch.cuda.empty_cache()

            # Fit the answer in one message: start at CRF 18 (or the caller's), raise until it fits.
            t_enc = time.perf_counter()
            out, used_crf = full, 18
            tries = [int(crf)] if crf is not None else [18, 22, 26, 30, 34]
            for c in tries:
                if c != 18:
                    out = tmp / f"crf{c}.mp4"
                    self._reencode(full, out, c)
                used_crf = c
                if len(out.read_bytes()) * 4 / 3 <= _MAX_B64_BYTES:
                    break
            data = out.read_bytes()
            if len(data) * 4 / 3 > _MAX_B64_BYTES:
                raise InvalidInput(f"the 4x clip is too large for one answer even at CRF {used_crf}; send fewer frames")
            if self.output_dir:
                keep = Path(self.output_dir)
                keep.mkdir(parents=True, exist_ok=True)
                shutil.copy(full, keep / f"{job.id or int(time.time())}-crf18.mp4")
            reencode_s = time.perf_counter() - t_enc
            wall_s = time.perf_counter() - t_wall

        n = stats["frames"]
        model_s = stats["model_ms"] / 1000
        spans = stats["spans_ms"]
        return Result(
            output={
                "video_b64": base64.b64encode(data).decode("ascii"),
                "format": "mp4",
                "crf": used_crf,
                "input_width": int(w),
                "input_height": int(h),
                "width": stats["size"][0],
                "height": stats["size"][1],
                "frames": n,
                "fps": float(fps),
                "window_size": window_size,
                "keyframes": stats["keyframes"],
                "seed": seed,
                "timings": {
                    "model_s": round(model_s, 3),
                    "model_fps": round(n / model_s, 2) if model_s else None,
                    "wan_ms": round(spans.get("wan", 0.0), 1),
                    "flash_ms": round(spans.get("flash", 0.0), 1),
                    "condition_ms": round(spans.get("flash_condition", 0.0), 1),
                    "first_frame_ms": round(stats["first_frame_ms"] or 0.0, 1),
                    "stream_s": round(stats["stream_s"], 2),
                    "decode_s": round(decode_s, 2),
                    "reencode_s": round(reencode_s, 2),
                    "wall_s": round(wall_s, 2),
                    "peak_vram_mib": round(peak),
                    "peak_reserved_mib": round(peak_reserved),
                },
                "spans_ms": spans,
                "dtype": self.dtype_name,
                "vram_limit_gib": self.vram_limit_gib or None,
                "device": self.device_name,
                "model": HF_REPO,
            }
        )

"""Hy-Nomas (Tencent Hunyuan) as an InferenceKey custom backend: an interactive video world.

Hy-Nomas is a segment-autoregressive video world model (Apache-2.0): it
generates a world one 5-second segment at a time, 720 x 1280 at 24 fps in 8
distilled denoising steps, and keeps a key/value cache of what it already
generated, so every new segment can be steered with a new instruction while
the world stays the same. No built-in backend serves it, so this backend
drives the authors' own resident generation service
(``apps/seg_ar_web_server.GenerationService``, the GPU side of their web demo)
with their **single-GPU low-memory preset**
(``configs/experiment/single_gpu_low_mem_inference.yaml``: INT8 DiT, INT8 KV
cache, FP8 PixelUMM decoder, CPU offload), the one they ship for one
GeForce RTX 4090.

``setup()`` (once per process)
    1. downloads the Hy-Nomas source at a pinned commit (not on PyPI) and the
       CUTLASS headers the FP8 decoder kernel is compiled against,
    2. downloads the INT8 DiT, the FP8 PixelUMM decoder (about 18 GB) and the
       Qwen2.5-VL-7B-Instruct text encoder (about 16.6 GB),
    3. builds the authors' pipeline with the low-memory preset.

``process(job)`` (per request) generates ONE segment:
    * ``{"step": "start", ...}`` opens a new world (new session),
    * ``{"step": "continue", "session_id": ..., ...}`` extends it, reusing
      the KV cache of every earlier segment,
    and returns ``{"session_id", "segment", "video_b64" (MP4 of this segment,
    about 1 to 3 MB), "timings", ...}``. Join the segments on the client
    (``ffmpeg -f concat``): the whole world in one answer would pass the
    platform's 8 MB message limit after a few segments.

The structured prompt is the authors' JSON format (``Style``,
``perspective_view``, ``subject_list``, ``segments`` with
``visual_description_long`` / ``description_change`` and
``contained_actions``). Send it whole as ``world`` / ``segment``, or use the
short fields: ``style``, ``perspective``, ``subjects`` and ``scene`` +
``action`` to start, ``change`` + ``action`` to continue. The authors' web
demo writes this JSON with a separate 27B prompt enhancer (Nomas-PE); this
backend does not run it, the caller writes the description.

Settings are class attributes; override them with ``NOMAS_*`` environment
variables or the runtime's ``--config-json`` (keys in lower case).
``NOMAS_VRAM_LIMIT_GIB`` caps the memory PyTorch may use on the GPU (to check
smaller cards); ``NOMAS_HEIGHT`` / ``NOMAS_WIDTH`` lower the resolution.
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
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

NOMAS_REPO = "Tencent-Hunyuan/Hy-Nomas"
CUTLASS_TAG = "v4.8.1"


def _log(msg: str) -> None:
    print(f"[hy-nomas-backend] {msg}", file=sys.stderr, flush=True)


def _ts(seconds: float) -> str:
    """Seconds -> the authors' MM:SS.mmm timestamp."""
    ms = int(round(seconds * 1000))
    return f"{ms // 60000:02d}:{(ms // 1000) % 60:02d}.{ms % 1000:03d}"


def _fetch_tarball(url: str, dest: Path, skip: tuple = ()) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".dl-"))
    req = urllib.request.Request(url, headers={"User-Agent": "inferencekey-hy-nomas-backend"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if not any(s in m.name for s in skip)]
        try:
            tf.extractall(tmp, members=members, filter="data")
        except TypeError:
            tf.extractall(tmp, members=members)
    inner = next(p for p in tmp.iterdir() if p.is_dir())
    if dest.exists():
        shutil.rmtree(dest)
    inner.rename(dest)
    tmp.rmdir()


class HyNomasBackend(CustomBackend):
    name = "hy-nomas-world"
    version = "0.1.2"
    task_type = "text2image"  # package and workload type: the SDK has no text2video custom backends yet
    description = "Hy-Nomas (Tencent Hunyuan, Apache-2.0): interactive video world, one 5 s segment per call, single-GPU low-memory preset"

    # Pinned upstream: code commit and Hugging Face revisions this example was tested with.
    nomas_commit = "effb8a49b83e6f1d35e26710db35194f2fd952a7"
    hf_repo = "tencent/Hy-Nomas"
    hf_revision = "main"
    text_encoder_repo = "Qwen/Qwen2.5-VL-7B-Instruct"
    # Output size (the preset's default is 720 x 1280) and the longest segment one call may ask for.
    height = 720
    width = 1280
    max_segment_seconds = 5.0
    # 0 = no cap. Otherwise the GiB PyTorch may allocate on the GPU.
    vram_limit_gib = 0.0
    # Extra Hydra overrides, separated by spaces (e.g. "inference.infer_steps_video=6").
    extra_overrides = ""

    # ------------------------------------------------------------------ setup

    def _setting(self, ctx: BackendContext, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"NOMAS_{key.upper()}")
        if raw is None:
            raw = (ctx.config or {}).get(key)
        if raw is None:
            return default
        return type(default)(raw)

    def setup(self, ctx: BackendContext) -> None:
        for key in ("nomas_commit", "hf_repo", "hf_revision", "text_encoder_repo", "height", "width",
                    "max_segment_seconds", "vram_limit_gib", "extra_overrides"):
            setattr(self, key, self._setting(ctx, key))
        cache = Path(os.environ.get("NOMAS_CACHE_DIR", Path.home() / ".cache" / "hy-nomas-backend"))
        self.timings: Dict[str, float] = {}
        t_setup = time.time()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        # 1. Source and CUTLASS headers.
        t0 = time.time()
        src = cache / f"src-{self.nomas_commit[:12]}"
        if not (src / "apps" / "seg_ar_web_server.py").is_file():
            _fetch_tarball(f"https://codeload.github.com/{NOMAS_REPO}/tar.gz/{self.nomas_commit}", src,
                           skip=(".pdf", "/assets/", "/apps/frontend/"))
        cutlass = cache / f"cutlass-{CUTLASS_TAG.lstrip('v')}-h"
        if not (cutlass / "include/cutlass/gemm/device/gemm_blockwise.h").is_file():
            _fetch_tarball(f"https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/{CUTLASS_TAG}", cutlass,
                           skip=("/examples/", "/test/", "/media/", "/python/", "/docs/", "/tools/"))
        self.timings["source_s"] = round(time.time() - t0, 1)

        # 2. Weights: only what the low-memory preset reads.
        from huggingface_hub import snapshot_download

        t0 = time.time()
        models = cache / "models"
        nomas_dir = Path(snapshot_download(
            self.hf_repo, revision=self.hf_revision, local_dir=str(models / "Hy-Nomas"),
            allow_patterns=["DiT/distilled_int8/*", "VAE/PixelUMM_FT_FP8/*", "examples/*", "config.json", "LICENSE"],
        ))
        te_dir = Path(snapshot_download(self.text_encoder_repo, local_dir=str(models / self.text_encoder_repo.split("/")[-1])))
        self.timings["download_s"] = round(time.time() - t0, 1)
        _log(f"source in {self.timings['source_s']} s, weights in {self.timings['download_s']} s")

        # 3. Environment the authors' launcher sets, then the pipeline.
        cuda_bin = "/usr/local/cuda/bin"
        if Path(cuda_bin, "nvcc").exists() and cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{cuda_bin}:{os.environ.get('PATH', '')}"
            os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
        # torch's JIT build of the FP8 decoder kernel calls `ninja`, installed in this venv's bin/.
        venv_bin = str(Path(sys.executable).parent)
        if venv_bin not in os.environ.get("PATH", "").split(":"):
            os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
        os.environ["CUTLASS_PATH"] = str(cutlass)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("TORCH_INDUCTOR_FX_GRAPH_CACHE", "1")
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / "inductor"))
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(cache / "torch_extensions"))
        os.environ.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        os.chdir(src)

        import torch

        if torch.cuda.is_available() and self.vram_limit_gib > 0:
            total = torch.cuda.get_device_properties(0).total_memory
            frac = min(1.0, self.vram_limit_gib * 2**30 / total)
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            _log(f"VRAM capped at {self.vram_limit_gib} GiB (fraction {frac:.3f} of {total / 2**30:.1f} GiB)")

        from apps.seg_ar_demo_config import load_demo_config
        from apps.seg_ar_web_server import GenerateRequest, GenerationService
        from hymm.inference import _validate_inference_config
        from hymm.models.dit.modules.int8_stored_bf16_linear import resolve_w8_bf16_checkpoint

        out_dir = cache / "outputs"
        overrides = [
            "+experiment=single_gpu_low_mem_inference",
            # The authors' online worker also sets inference.seg_ar_segment_vae_decode=true and
            # model.vae.stable_prefix_streaming=true; with this preset on a 24 GB card that decodes
            # while the DiT is still resident and runs out of memory, so decode after denoising.
            f"dit_weight={json.dumps(str(nomas_dir / 'DiT/distilled_int8/diffusion_pytorch_model_ema_0_995_int8.safetensors'))}",
            f"model.vae.checkpoint={json.dumps(str(nomas_dir / 'VAE/PixelUMM_FT_FP8/diffusion_pytorch_model_fp8.safetensors'))}",
            f"model.text_encoder.model_name={json.dumps(str(te_dir))}",
            f"inference.save_path={json.dumps(str(out_dir))}",
            f"inference.video_size=[{int(self.height)},{int(self.width)}]",
        ] + [o for o in str(self.extra_overrides).split() if o]
        t_load = time.time()
        config = load_demo_config(overrides)
        _validate_inference_config(config)
        resolve_w8_bf16_checkpoint(config.inference, config.dit_weight)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        torch.set_grad_enabled(False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._GenerateRequest = GenerateRequest
        self.service = GenerationService(
            config=config, device=device, output_dir=out_dir, rank=0, world_size=1,
            world_mesh=None, command_group=None, rollout_type="seg_ar",
        )
        self.fps = int(config.inference.video_fps)
        self.timings["load_s"] = round(time.time() - t_load, 1)
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        self.timings["setup_s"] = round(time.time() - t_setup, 1)
        self._lock = threading.Lock()
        # session_id -> {"segments": [paths], "end_s": float, "world": dict}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        _log(f"Hy-Nomas ready on {self.device_name}: {self.timings}")

    # ---------------------------------------------------------------- inputs

    @staticmethod
    def _fields(inp: Dict[str, Any]) -> Dict[str, Any]:
        """Top-level fields, ``extra_body``, and a JSON object sent as the last user message."""
        f = dict(inp)
        if isinstance(inp.get("extra_body"), dict):
            f.update(inp["extra_body"])
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
                elif isinstance(content, str) and content.strip() and "scene" not in f and "change" not in f:
                    f["_text"] = content.strip()
                break
        return f

    def _segment_json(self, f: Dict[str, Any], start_s: float, first: bool) -> Dict[str, Any]:
        seconds = float(f.get("seconds", self.max_segment_seconds))
        if not 1.0 <= seconds <= float(self.max_segment_seconds):
            raise InvalidInput(f"seconds must be between 1 and {self.max_segment_seconds}")
        end_s = start_s + seconds
        if isinstance(f.get("segment"), dict):
            seg = dict(f["segment"])
        else:
            text = f.get("scene") if first else f.get("change")
            text = text or f.get("_text")
            if not text:
                raise InvalidInput("start needs 'scene' (or a whole 'world'); continue needs 'change'")
            seg = {("visual_description_long" if first else "description_change"): str(text)}
            if f.get("action"):
                seg["contained_actions"] = [{"start_time": _ts(start_s), "end_time": _ts(end_s),
                                             "action_description": str(f["action"])}]
        seg["start_time"], seg["end_time"] = _ts(start_s), _ts(end_s)
        for act in seg.get("contained_actions") or []:
            act.setdefault("start_time", _ts(start_s))
            act.setdefault("end_time", _ts(end_s))
        return seg

    def _world_json(self, f: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(f.get("world"), dict):
            world = json.loads(json.dumps(f["world"]))
            segs = world.get("segments") or []
            if not segs:
                raise InvalidInput("'world' needs one entry in 'segments'")
            world["segments"] = [self._segment_json({**f, "segment": segs[0]}, 0.0, True)]
            return world
        subjects = f.get("subjects") or []
        if isinstance(subjects, str):
            subjects = [{"display_name": subjects}]
        subject_list = []
        for i, s in enumerate(subjects):
            s = {"display_name": s} if isinstance(s, str) else dict(s)
            subject_list.append({"subject_id": s.get("subject_id") or f"subject_{i + 1}",
                                 "display_name": s.get("display_name") or s.get("name") or "",
                                 "subject_attribute": s.get("subject_attribute") or s.get("attribute") or ""})
        return {
            "test_id": f.get("test_id") or "world",
            "Style": f.get("style") or "Cinematic realism",
            "perspective_view": f.get("perspective") or "Third-person view",
            "subject_list": subject_list,
            "segments": [self._segment_json(f, 0.0, True)],
        }

    # ---------------------------------------------------------------- process

    def process(self, job: Job) -> Result:
        import torch

        f = self._fields(job.input)
        step = str(f.get("step") or ("continue" if f.get("session_id") else "start")).lower()
        if step not in ("start", "continue", "end"):
            raise InvalidInput("step must be 'start', 'continue' or 'end'")
        with self._lock:
            if step == "end":
                sid = str(f.get("session_id") or "")
                self.sessions.pop(sid, None)
                return Result(output={"session_id": sid, "ended": True})
            if step == "start":
                sid = f"w{uuid.uuid4().hex[:12]}"
                prompt = self._world_json(f)
                state = {"segments": [], "end_s": 0.0, "prompts": [], "seed": int(f.get("seed", 42))}
                reset = True
            else:
                sid = str(f.get("session_id") or "")
                state = self.sessions.get(sid)
                if state is None:
                    raise InvalidInput(f"unknown session_id {sid!r}: start a new world (the backend keeps one world at a time)")
                if self.service._active_session_id != sid:
                    raise InvalidInput("this world is no longer resident (another world was started); start again")
                prompt = {"segments": [self._segment_json(f, state["end_s"], False)]}
                reset = False
            seg = prompt["segments"][0]
            seg_s = (float(f.get("seconds", self.max_segment_seconds)))
            num_frames, boundaries, _ = self.service._seg_ar_timing_from_prompts([json.dumps(prompt)], self.fps)
            req = self._GenerateRequest(
                prompt=json.dumps(prompt, ensure_ascii=False), session_id=sid, preserve_kv_cache=True,
                # A world keeps the seed it started with (the authors' pipeline rejects a change).
                reset_session=reset, seed=int(state["seed"]), num_frames=num_frames,
                segment_boundaries=boundaries,
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.time()
            try:
                out = self.service.generate(req)
            except ValueError as e:
                raise InvalidInput(str(e)) from e
            gpu_s = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
            path = self.service.output_dir / out["video_filename"] if "video_filename" in out else \
                self.service._resolve_output_url(out["video_url"])
            state["segments"].append(str(path))
            state["end_s"] += seg_s
            state["prompts"].append(seg)
            self.sessions = {sid: state}  # one resident world at a time (one KV cache on the GPU)
            result: Dict[str, Any] = {
                "session_id": sid,
                "segment": len(state["segments"]),
                "world_seconds": round(state["end_s"], 3),
                "video_b64": base64.b64encode(Path(path).read_bytes()).decode(),
                "format": "mp4",
                "width": out.get("width"), "height": out.get("height"), "fps": out.get("fps"),
                "frames": out.get("frames"), "seconds": out.get("duration_seconds"),
                "prompt": prompt,
                "timings": {
                    "gpu_s": round(gpu_s, 2),
                    "generation_s": round(float(out.get("generation_seconds", 0)), 2),
                    "encoding_s": round(float(out.get("encoding_seconds", 0)), 2),
                    "peak_vram_mib": round(peak),
                },
                "model": self.hf_repo,
                "device": self.device_name,
            }
            return Result(output=result)

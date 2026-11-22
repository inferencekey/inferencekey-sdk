"""A custom backend that runs **Matrix-Game 4.0** (Skywork AI), a causal video renderer, on one NVIDIA GPU.

Matrix-Game 4.0 (Apache-2.0, https://huggingface.co/Skywork/Matrix-Game-4.0)
re-renders the frames of a real, executable low-poly game into detailed video,
480 x 864, as they stream in. The game engine keeps gameplay, physics and
input; the model only repaints the clean low-poly RGB stream, four frames per
latent chunk, with a 21-latent attention cache. Each scene (Medieval Castle,
Ancient Egypt, City Racing) has its own 5B-parameter BF16 checkpoint built on
Wan2.2-Fun-5B-Control. This backend serves ONE scene (``castle`` by default)
with the authors' own inference source (github.com/SkyworkAI/Matrix-Game,
``Matrix-Game-4/``, pinned below): it is not a Diffusers pipeline.

Presets (the authors' names):

* ``realtime``: TAEHV encode/decode, one denoising step, ring/fused DiT and
  CUDA graphs (the authors' A100 fast path).
* ``rtx5090``: the same, with DiT linears converted to FP8 at load time
  (needs a GPU with FP8 tensor cores: Ada or Blackwell).
* ``quality``: Wan2.2 VAE encode/decode, two denoising steps (1000, 500),
  native DiT. Slower; the CLI default.

Contract (``task_type`` text2image, so the workload answers on the OpenAI
images route ``POST .../v1/images/generations``; the caller's JSON arrives in
``input.passthrough_body``):

* upload the low-poly input video (any format ffmpeg reads; 1280 x 720 game
  captures are center-cropped and resized to 864 x 480 like in training) in
  parts of at most 2 MB: ``{"upload_id", "part", "parts", "chunk_b64"}``. The
  last part answers ``{"input_id"}`` (the SHA-256 prefix of the file).
* render: ``{"input_id", "preset", "seed", "max_frames", "vram_limit_gib"}``.
  ``vram_limit_gib`` caps the GPU memory of the whole process (weights
  included: the engine is reloaded under the cap) to emulate a smaller card.
  From the dashboard's image tester write words as the prompt: a preset name
  and optionally a length, for example ``realtime 3s`` (it renders the last
  uploaded input).
* answer: one request waits at most ``wait_s`` (default 270 s). If the render
  is not done by then the answer is ``{"clip_id", "status": "running"}``; ask
  again with ``{"clip_id": "..."}``. When done: ``status: "done"``,
  ``data[0].b64_json`` (the middle frame as JPEG, so an image viewer shows
  something), the MP4 in parts of 2 MB (``mp4_b64`` = part 0, ``parts``,
  ``mp4_bytes``, ``mp4_sha256``; fetch part n with ``{"clip_id", "part": n}``),
  ``timings`` (model frames per second over the whole clip and after the
  first chunk, per-chunk milliseconds, load and warm-up of the engine) and
  ``peak_allocated_mib`` / ``peak_reserved_mib`` (PyTorch) and
  ``card_peak_mib`` (``nvidia-smi``).

Environment (the workload's ``config.env``):

* ``MG4_SCENE``: ``castle`` (default), ``ancient-egypt`` or ``racing``.
* ``MG4_PRESET``: the engine loaded in ``setup()`` (default ``realtime``).
* ``MG4_HOME``: where the source and the ~25 GB of resources go (default
  ``~/.cache/matrix-game-4``). Keep it on a local disk, not a network volume.

Run it by hand (from this folder, with requirements installed)::

    python -m inferencekey.backend.serve --port 8099 --backend backend:MatrixGame4Backend

Attribution: Matrix-Game 4.0 by Skywork AI (Apache-2.0). Wan2.2-Fun-5B-Control
(umT5 encoder, Wan2.2 VAE) by Alibaba PAI / Wan (Apache-2.0). TAEHV by Ollin
Boer Bohan (MIT).
"""

from __future__ import annotations

import base64
import collections
import concurrent.futures
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from inferencekey.backend import BackendContext, CustomBackend, Job, Result

try:  # newer SDKs answer 400 for bad client input
    from inferencekey.backend import InvalidInput
except ImportError:  # pragma: no cover - older SDK
    InvalidInput = ValueError

#: Pinned revisions this backend was tested with (2026-11-22).
HF_REPO, HF_REVISION = "Skywork/Matrix-Game-4.0", "2c53100457da7abcf0e966f75637c644d3423a63"
SOURCE_REPO = "https://github.com/SkyworkAI/Matrix-Game.git"
SOURCE_COMMIT = "51913b2bfa78a6d54de912b8348b1d318f4d36e2"
PRESETS = ("realtime", "quality", "rtx5090", "rtx5080", "rtx5070ti")

PART_BYTES = 2_000_000
KEEP_FILES = 16
MAX_UPLOAD_BYTES = 200_000_000


def _log(message: str) -> None:
    print(f"[mg4-backend] {message}", file=sys.stderr, flush=True)


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


class _Clip:
    def __init__(self, clip_id: str, params: dict) -> None:
        self.id = clip_id
        self.params = params
        self.started = time.time()
        self.done = threading.Event()
        self.result: dict | None = None
        self.error: str | None = None
        self.path: Path | None = None
        self.sha256 = ""


class MatrixGame4Backend(CustomBackend):
    """Matrix-Game 4.0: a low-poly game video in, the same video re-rendered in detail out."""

    name = "matrix-game-4"
    version = "0.1.0"
    task_type = "text2image"
    requirements = "requirements.txt"
    requires_python = ">=3.12,<3.13"

    # ------------------------------------------------------------------ setup

    def setup(self, ctx: BackendContext) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        t0 = time.time()
        self.home = Path(os.environ.get("MG4_HOME") or Path.home() / ".cache" / "matrix-game-4")
        self.home.mkdir(parents=True, exist_ok=True)
        self.scene = os.environ.get("MG4_SCENE") or "castle"
        src = self._fetch_source()
        t_src = time.time()
        self.resources = self.home / "resources"
        size = self._fetch_resources(src)
        t_dl = time.time()
        _log(f"source {t_src - t0:.1f}s, resources {t_dl - t_src:.1f}s ({size / 1e9:.1f} GB verified)")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Matrix-Game 4.0 needs a CUDA GPU")
        self.torch = torch
        self.card = torch.cuda.get_device_name(0)
        self.total_gib = torch.cuda.mem_get_info(0)[1] / 2**30
        self.capability = torch.cuda.get_device_capability(0)
        _log(f"card {self.card} {self.total_gib:.1f} GiB, sm_{self.capability[0]}{self.capability[1]}, torch "
             f"{torch.__version__}")
        self.vram = _VramSampler()
        self.lock = threading.Lock()
        # Every CUDA call (load, warm-up, graph capture, render) runs on this one thread.
        self.gpu = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mg4-gpu")
        self.engine = None
        self.engine_key: tuple | None = None
        self.clips: "collections.OrderedDict[str, _Clip]" = collections.OrderedDict()
        self.uploads: dict[str, dict] = {}
        self.inputs_dir = Path(tempfile.mkdtemp(prefix="mg4-inputs-"))
        self.results = Path(tempfile.mkdtemp(prefix="mg4-clips-"))
        self.last_input: str | None = None
        preset = os.environ.get("MG4_PRESET") or "realtime"
        load = self.gpu.submit(self._ensure_engine, preset, 0.0).result()
        self.setup_timings = {"source_s": round(t_src - t0, 1), "resources_s": round(t_dl - t_src, 1),
                              "resources_gb": round(size / 1e9, 2), **load}
        _log(f"ready: setup {time.time() - t0:.1f}s {json.dumps(self.setup_timings)}")

    def _fetch_source(self) -> Path:
        """Sparse checkout of Matrix-Game-4/ at the pinned commit (the repo itself is large)."""
        repo = self.home / "Matrix-Game"
        src = repo / "Matrix-Game-4"
        if not (src / "matrix_game" / "inference" / "offline.py").is_file():
            subprocess.run(["rm", "-rf", str(repo)], check=True)
            run = lambda *a: subprocess.run(list(a), check=True)  # noqa: E731
            try:
                run("git", "init", "-q", str(repo))
                run("git", "-C", str(repo), "remote", "add", "origin", SOURCE_REPO)
                run("git", "-C", str(repo), "sparse-checkout", "set", "Matrix-Game-4/matrix_game", "Matrix-Game-4/releases")
                run("git", "-C", str(repo), "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", SOURCE_COMMIT)
                run("git", "-C", str(repo), "checkout", "-q", "FETCH_HEAD")
            except (OSError, subprocess.CalledProcessError) as e:  # no git: the commit tarball, two folders only
                _log(f"git checkout failed ({e}); using the GitHub tarball")
                import tarfile
                import urllib.request

                url = f"https://codeload.github.com/SkyworkAI/Matrix-Game/tar.gz/{SOURCE_COMMIT}"
                with urllib.request.urlopen(url, timeout=300) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
                    for member in tar:
                        parts = member.name.split("/", 1)
                        if len(parts) == 2 and parts[1].startswith(("Matrix-Game-4/matrix_game/", "Matrix-Game-4/releases/")):
                            member.name = parts[1]
                            tar.extract(member, repo, filter="data")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        return src

    def _fetch_resources(self, src: Path) -> int:
        """Download the scene and its shared files from the pinned manifest, in parallel, and check every SHA-256."""
        from huggingface_hub import hf_hub_download
        from matrix_game.resources import checked_path, digest, files_for

        manifest = json.loads((src / "releases" / "models.json").read_text())
        entries = files_for(manifest, self.scene)
        cache = self.home / "hub"

        def one(entry: dict) -> int:
            dest = checked_path(self.resources, entry["path"])
            ok = self.resources / ".verified" / (entry["sha256"] + ".ok")
            if dest.is_file() and dest.stat().st_size == entry["size"] and ok.is_file():
                return entry["size"]
            blob = Path(hf_hub_download(entry.get("repo_id", HF_REPO), entry.get("repo_path", entry["path"]),
                                        repo_type=entry.get("repo_type", "model"),
                                        revision=entry.get("revision", HF_REVISION), cache_dir=str(cache))).resolve()
            if blob.stat().st_size != entry["size"] or digest(blob) != entry["sha256"]:
                raise RuntimeError(f"checksum mismatch: {entry['path']}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.unlink(missing_ok=True)
            os.link(blob, dest)  # same disk: no copy
            ok.parent.mkdir(parents=True, exist_ok=True)
            ok.touch()
            return entry["size"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            return sum(pool.map(one, entries))

    # --------------------------------------------------------------- engine

    def _ensure_engine(self, preset: str, cap_gib: float) -> dict:
        """(Re)load the model for a preset; runs on the GPU thread."""
        key = (preset, cap_gib)
        if self.engine is not None and self.engine_key == key:
            return {}
        torch = self.torch
        if self.engine is not None:
            self.engine = None
            import gc

            gc.collect()
            torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gib / self.total_gib) if cap_gib > 0 else 1.0, 0)
        torch.cuda.reset_peak_memory_stats(0)
        self.vram.reset()
        from matrix_game.inference.loader import ModelComponents
        from matrix_game.scenes import add_model_arguments, resolve_model_arguments
        import argparse

        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        args = parser.parse_args(["--scene", self.scene, "--resources", str(self.resources), "--preset", preset,
                                  "--prompt-mode", "fixed"])
        resolve_model_arguments(args)
        t = time.time()
        comp = ModelComponents(args)
        comp.load()
        torch.cuda.synchronize()
        load_s = time.time() - t
        self.engine = comp
        self.engine_key = key
        info = {"preset": preset, "engine_load_s": round(load_s, 1),
                "weights_gib_on_gpu": round(torch.cuda.memory_allocated(0) / 2**30, 2),
                "load_peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 2**20),
                "load_card_peak_mib": self.vram.peak, "vram_limit_gib": cap_gib or None,
                "config": {k: comp.config()[k] for k in ("vae_backend", "steps", "dit_runtime", "attention_backend",
                                                          "vae_graph", "sink_rope_mode")}}
        _log(f"engine {json.dumps(info)}")
        return info

    def _render(self, p: dict, path: Path) -> dict:
        """Stream the input through the model chunk by chunk (1 frame, then 4 per chunk), like the authors' offline path."""
        torch = self.torch
        import imageio_ffmpeg
        import numpy as np
        from PIL import Image

        from matrix_game.inference.offline import frame_blocks
        from matrix_game.inference.preprocessing import crop_and_resize, frames_to_tensor, tensor_to_uint8

        load = self._ensure_engine(p["preset"], p["vram_limit_gib"])
        torch.cuda.reset_peak_memory_stats(0)
        self.vram.reset()
        reader = imageio_ffmpeg.read_frames(str(p["input_path"]), pix_fmt="rgb24")
        info = next(reader)
        width, height = info["size"]
        fps = info["fps"]
        limit = p["max_frames"]

        def frames():
            for i, raw in enumerate(reader):
                if limit and i >= limit:
                    return
                yield raw

        writer = imageio_ffmpeg.write_frames(str(path), (864, 480), fps=fps, codec="libx264", pix_fmt_in="rgb24",
                                             pix_fmt_out="yuv420p", macro_block_size=1,
                                             output_params=["-crf", "16", "-preset", "fast", "-movflags", "+faststart"])
        writer.send(None)
        stream = self.engine.stream(seed=p["seed"])
        chunk_ms, written, mid = [], 0, None
        started = time.perf_counter()
        try:
            for raw, valid in frame_blocks(frames()):
                imgs = [crop_and_resize(Image.fromarray(np.frombuffer(r, np.uint8).reshape(height, width, 3)), 480, 864)
                        for r in raw]
                t = time.perf_counter()
                result = stream.push(frames_to_tensor(imgs), frame_start=written)
                rgb = tensor_to_uint8(result.pixels)[:valid]
                chunk_ms.append(round((time.perf_counter() - t) * 1000, 1))
                for frame in rgb:
                    writer.send(frame.tobytes())
                    if mid is None and written >= (limit or 240) // 2:
                        mid = frame.copy()
                    written += 1
        finally:
            stream.close()
            writer.close()
            reader.close()
        torch.cuda.synchronize()
        total_s = time.perf_counter() - started
        model_s = sum(chunk_ms) / 1000
        steady = chunk_ms[1:]
        steady_frames = written - 1
        return {
            "mid": mid, "frames": written, "fps": fps, "input_size": [width, height],
            "timings": {
                "wall_s": round(total_s, 2), "model_s": round(model_s, 2),
                "model_fps": round(written / model_s, 2) if model_s else None,
                "steady_model_fps": round(steady_frames / (sum(steady) / 1000), 2) if steady else None,
                "first_chunk_ms": chunk_ms[0] if chunk_ms else None,
                "chunk_ms_median": sorted(steady)[len(steady) // 2] if steady else None,
                "chunk_ms_max": max(steady) if steady else None, "chunks": len(chunk_ms),
                "end_to_end_fps": round(written / total_s, 2) if total_s else None, **load,
            },
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 2**20),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 2**20),
            "card_peak_mib": self.vram.peak,
        }

    # ---------------------------------------------------------------- inputs

    def _upload(self, src: dict) -> dict:
        uid = re.sub(r"[^A-Za-z0-9_-]", "", str(src["upload_id"]))[:64]
        part, parts = int(src.get("part") or 0), int(src.get("parts") or 1)
        if not uid or not 0 <= part < parts <= 200:
            raise InvalidInput("`upload_id`, `part` (0..parts-1) and `parts` (1..200) are required")
        try:
            chunk = base64.b64decode(str(src.get("chunk_b64") or ""))
        except Exception:
            raise InvalidInput("`chunk_b64` is not base64")
        up = self.uploads.setdefault(uid, {"parts": parts, "chunks": {}})
        up["chunks"][part] = chunk
        if sum(len(c) for c in up["chunks"].values()) > MAX_UPLOAD_BYTES:
            self.uploads.pop(uid, None)
            raise InvalidInput("upload too large")
        if len(up["chunks"]) < up["parts"]:
            return {"status": "uploading", "upload_id": uid, "received": len(up["chunks"]), "parts": up["parts"],
                    "data": []}
        blob = b"".join(up["chunks"][i] for i in range(up["parts"]))
        self.uploads.pop(uid, None)
        input_id = hashlib.sha256(blob).hexdigest()[:16]
        (self.inputs_dir / f"{input_id}.video").write_bytes(blob)
        self.last_input = input_id
        return {"status": "uploaded", "input_id": input_id, "bytes": len(blob), "data": []}

    def _params(self, src: dict) -> dict:
        p: dict = {}
        words = str(src.get("prompt") or "").strip().lower().split()
        for w in words:  # dashboard words: "realtime 3s"
            if w in PRESETS:
                p["preset"] = w
            elif w.endswith("s") and w[:-1].replace(".", "", 1).isdigit():
                p["seconds"] = float(w[:-1])
        for key in ("input_id", "preset", "seed", "max_frames", "seconds", "vram_limit_gib"):
            if src.get(key) is not None:
                p[key] = src[key]
        p["input_id"] = str(p.get("input_id") or self.last_input or "")
        path = self.inputs_dir / f"{p['input_id']}.video"
        if not p["input_id"] or not path.is_file():
            raise InvalidInput("upload a low-poly video first (upload_id/part/parts/chunk_b64), then send its input_id")
        p["input_path"] = path
        p["preset"] = str(p.get("preset") or os.environ.get("MG4_PRESET") or "realtime")
        if p["preset"] not in PRESETS:
            raise InvalidInput(f"`preset` must be one of {PRESETS}")
        p["seed"] = int(p.get("seed") or 0)
        p["max_frames"] = int(p.get("max_frames") or 0)
        if p.get("seconds"):
            p["max_frames"] = max(1, int(round(float(p["seconds"]) * 48)))
        p["vram_limit_gib"] = float(p.get("vram_limit_gib") or 0)
        return p

    # --------------------------------------------------------------- running

    def _run_clip(self, clip: _Clip) -> None:
        try:
            with self.lock:
                p = clip.params
                _log(f"clip {clip.id}: input {p['input_id']}, preset {p['preset']}, max_frames {p['max_frames'] or 'all'}, "
                     f"cap {p['vram_limit_gib'] or 'none'}")
                path = self.results / f"{clip.id}.mp4"
                out = self.gpu.submit(self._render, p, path).result()
            from PIL import Image

            buf = io.BytesIO()
            if out["mid"] is not None:
                Image.fromarray(out.pop("mid")).save(buf, format="JPEG", quality=88)
            clip.path = path
            clip.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            clip.result = {
                "created": int(time.time()), "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")}],
                "output_format": "jpeg", "model": HF_REPO, "scene": self.scene, "status": "done", "clip_id": clip.id,
                "preset": p["preset"], "seed": p["seed"], "frames": out["frames"], "fps": out["fps"],
                "seconds_of_video": round(out["frames"] / out["fps"], 2), "input_size": out["input_size"],
                "output_size": [864, 480], "vram_limit_gib": p["vram_limit_gib"] or None, "card": self.card,
                "timings": out["timings"], "peak_allocated_mib": out["peak_allocated_mib"],
                "peak_reserved_mib": out["peak_reserved_mib"], "card_peak_mib": out["card_peak_mib"],
                "request_wall_s": round(time.time() - clip.started, 2),
            }
            _log(f"clip {clip.id} done: {json.dumps({k: clip.result[k] for k in ('preset', 'frames', 'timings', 'peak_allocated_mib', 'peak_reserved_mib', 'card_peak_mib')})}")
        except Exception as e:  # noqa: BLE001 - reported to the caller on the next poll
            clip.error = f"{type(e).__name__}: {e}"
            _log(f"clip {clip.id} failed: {clip.error}")
            try:
                self.gpu.submit(self.torch.cuda.empty_cache).result()
            except Exception:  # noqa: BLE001
                pass
        finally:
            clip.done.set()

    def _answer(self, clip: _Clip, part: int = 0) -> dict:
        if not clip.done.is_set():
            return {"status": "running", "clip_id": clip.id, "elapsed_s": round(time.time() - clip.started, 1),
                    "data": []}
        if clip.error:
            return {"status": "failed", "clip_id": clip.id, "error": clip.error, "data": []}
        size = clip.path.stat().st_size
        parts = (size + PART_BYTES - 1) // PART_BYTES
        if not 0 <= part < parts:
            raise InvalidInput(f"`part` must be between 0 and {parts - 1}")
        with clip.path.open("rb") as f:
            f.seek(part * PART_BYTES)
            chunk = f.read(PART_BYTES)
        head = clip.result if part == 0 else {"status": "done", "clip_id": clip.id, "data": []}
        return {**head, "part": part, "parts": parts, "mp4_bytes": size, "mp4_sha256": clip.sha256,
                "mp4_b64": base64.b64encode(chunk).decode("ascii")}

    # ---------------------------------------------------------------- process

    def process(self, job: Job) -> Result:
        src = job.input
        passthrough = job.input.get("passthrough_body")
        if isinstance(passthrough, dict) and passthrough:
            src = passthrough
        if src.get("upload_id"):
            return Result(self._upload(src))
        if src.get("info"):
            return Result({"status": "info", "card": self.card, "setup": self.setup_timings, "engine": list(self.engine_key or ()),
                           "last_input": self.last_input, "data": []})
        wait_s = float(src.get("wait_s") if src.get("wait_s") is not None else 270)
        if src.get("clip_id"):
            clip = self.clips.get(str(src["clip_id"]))
            if clip is None:
                raise InvalidInput(f"unknown or expired clip_id {src['clip_id']!r}")
            if "part" not in src:
                clip.done.wait(timeout=max(0.0, min(wait_s, 270)))
            return Result(self._answer(clip, int(src.get("part") or 0)))
        params = self._params(src)
        if self.lock.locked():
            busy = next((c.id for c in reversed(self.clips.values()) if not c.done.is_set()), None)
            raise InvalidInput(f"busy with clip {busy}; poll it with clip_id and send this one after")
        clip = _Clip(uuid.uuid4().hex[:16], params)
        self.clips[clip.id] = clip
        while len(self.clips) > KEEP_FILES:
            _, old = self.clips.popitem(last=False)
            if old.path is not None:
                old.path.unlink(missing_ok=True)
        threading.Thread(target=self._run_clip, args=(clip,), daemon=True).start()
        clip.done.wait(timeout=max(0.0, min(wait_s, 270)))
        return Result(self._answer(clip))

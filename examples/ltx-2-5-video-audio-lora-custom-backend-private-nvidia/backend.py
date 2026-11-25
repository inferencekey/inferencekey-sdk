"""A custom backend that runs **LTX-2.5** (Lightricks): video and its soundtrack from one prompt, on one NVIDIA GPU.

LTX-2.5 (https://huggingface.co/Lightricks/LTX-2.5, LTX-2.x Community License)
generates a video clip and its audio (speech, foley, ambience) together in one
pass, from text or from text plus a first image. The weights ship as separate
files: a 22B diffusion transformer (this backend uses the *distilled* one, in
bf16, cast to FP8 on the fly), a Gemma 4 12B text encoder fine-tuned by
Lightricks, a convolutional video VAE, an audio VAE and an x2 latent spatial
upscaler: about 66 GiB in all.

This backend runs the authors' own code (github.com/Lightricks/LTX-2,
``ltx_pipelines.distilled``, the ``DistilledPipeline`` of their quick start,
Apache-2.0), one subprocess per request, so an out-of-memory error ends that
request and never the backend. It can also apply an official LoRA (the
Cinemagraph LoRA is downloaded by default) and start a short LoRA training
run with the authors' ``ltx-trainer`` and its 32 GB low-VRAM config, to check
that customization works on your card.

Contract (``task_type`` text2image, so the workload answers on the OpenAI
images route ``POST .../v1/images/generations``; the caller's JSON arrives in
``input.passthrough_body``):

* generate: ``{"prompt", "seed", "width", "height", "num_frames",
  "frame_rate", "image_b64", "image_strength", "lora", "lora_strength",
  "quantization", "offload", "vram_limit_gib"}``. Defaults: 1536 x 1024,
  121 frames (8k+1), 24 fps, ``quantization`` ``fp8-cast``, ``offload``
  ``none``. ``image_b64`` (PNG/JPEG) makes it image-to-video (the image is
  frame 0). ``lora`` is ``cinemagraph`` or a file name in ``LTX_HOME/loras``.
  ``vram_limit_gib`` caps the PyTorch memory of the run to emulate a smaller
  card. From the dashboard's image tester, the prompt text is enough (it uses
  ``LTX_DASH_WIDTH`` x ``LTX_DASH_HEIGHT``, ``LTX_DASH_FRAMES``).
* train: ``{"train": {"clip_ids": [...], "steps": 30, "width": 512,
  "height": 320, "frames": 49}}`` fine-tunes a LoRA for a few steps on clips
  this backend generated earlier (their prompts are the captions), with the
  authors' ``t2v_lora_low_vram.yaml`` (INT8 transformer, 8-bit Gemma, rank
  16). It installs ``ltx-trainer`` the first time. It answers seconds per
  step, peak memory and the log; it is a smoke test, not a finished LoRA.
* answer: one request waits at most ``wait_s`` (default 270 s). If the task
  is not done the answer is ``{"clip_id", "status": "running"}``; ask again
  with ``{"clip_id": "..."}``. When a generation is done: ``status: "done"``,
  ``data[0].b64_json`` (the middle frame as JPEG), the MP4 (with its AAC audio)
  in parts of 2 MB (``mp4_b64`` = part 0, ``parts``, ``mp4_bytes``,
  ``mp4_sha256``; fetch part n with ``{"clip_id", "part": n}``), ``timings``
  and ``peak_allocated_mib`` / ``peak_reserved_mib`` (PyTorch) and
  ``card_peak_mib`` (``nvidia-smi``).

Environment (the workload's ``config.env``):

* ``HF_TOKEN``: a Hugging Face read token whose account accepted the
  LTX-2.5 terms (the repositories are gated). With it every file comes from
  ``Lightricks/*``. Without it the backend fetches the same bytes from the
  public copies listed in ``FILES`` (``LTX_SOURCE=mirror``); either way each
  file is checked against the SHA-256 Lightricks publishes, so a modified
  copy is rejected.
* ``LTX_HOME``: where the ~66 GiB go (default ``~/.cache/ltx-2-5``). Use a
  local disk: a network volume makes every load minutes slower.
* ``LTX_DASH_WIDTH`` / ``LTX_DASH_HEIGHT`` / ``LTX_DASH_FRAMES``: size used
  for dashboard prompts (default 768 x 512, 49 frames).

Run it by hand (from this folder, with requirements installed)::

    python -m inferencekey.backend.serve --port 8099 --backend backend:Ltx25Backend

License: the code here and the LTX-2 code are Apache-2.0; the LTX-2.5 weights
and any LoRA you train on them are under the LTX-2.x Community License (free
under 10 million USD annual revenue, affiliates included; read it before you
use the outputs commercially).
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
import shutil
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

#: The authors' code this backend was tested with (2026-11-25).
LTX_REPO_URL = "https://github.com/Lightricks/LTX-2.git"
LTX_COMMIT = "c3a7efb18256cba2d76cc32f5b47811d7c9ad642"
CUDNN_PIN = "9.24.0.43"

#: Every file, with the SHA-256 Lightricks publishes for it (Hugging Face LFS oid) and a public copy of the same bytes.
FILES = {
    "transformer": {
        "repo": "Lightricks/LTX-2.5", "revision": "49b093a39c19169a8a81721dcc541e07e945232f",
        "path": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", "size": 42018190584,
        "sha256": "31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4",
        "mirror": ("George-Costanza/ltx25-model-files", "262133f3e7feb84def871e51eed4a1159c957e13", None)},
    "text_encoder": {
        "repo": "Lightricks/LTX-2.5", "revision": "49b093a39c19169a8a81721dcc541e07e945232f",
        "path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", "size": 26263858182,
        "sha256": "ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1",
        "mirror": ("George-Costanza/ltx25-model-files", "262133f3e7feb84def871e51eed4a1159c957e13", None)},
    "video_vae": {
        "repo": "Lightricks/LTX-2.5", "revision": "49b093a39c19169a8a81721dcc541e07e945232f",
        "path": "vae/ltx-2.5-video-vae-conv-bf16.safetensors", "size": 1452269922,
        "sha256": "685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d",
        "mirror": ("George-Costanza/ltx25-model-files", "262133f3e7feb84def871e51eed4a1159c957e13", None)},
    "audio_vae": {
        "repo": "Lightricks/LTX-2.5", "revision": "49b093a39c19169a8a81721dcc541e07e945232f",
        "path": "vae/ltx-2.5-audio-vae-bf16.safetensors", "size": 364866540,
        "sha256": "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5",
        "mirror": ("George-Costanza/ltx25-model-files", "262133f3e7feb84def871e51eed4a1159c957e13", None)},
    "spatial_upsampler": {
        "repo": "Lightricks/LTX-2.5", "revision": "49b093a39c19169a8a81721dcc541e07e945232f",
        "path": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "size": 995778752,
        "sha256": "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8",
        "mirror": ("George-Costanza/ltx25-model-files", "262133f3e7feb84def871e51eed4a1159c957e13", None)},
    "lora_cinemagraph": {
        "repo": "Lightricks/LTX-2.5-22b-LoRA-Cinemagraph", "revision": "a3f23fd142aa8a11e624e12c407a0be383c2f758",
        "path": "ltx-2.5-22b-lora-cinemagraph-0.9.safetensors", "size": 201462650,
        "sha256": "fcd06225e1646080cbb21958a25c9f339ee484519e798bfb00c9b91022cfccfa",
        "mirror": ("Woovee/ltx25-studio-runpod", "0914916c244b0b39401b397f90fe899ffcdbff38",
                   "loras/ltx-2.5-22b-lora-cinemagraph-0.9.safetensors")},
}
LORAS = {"cinemagraph": "lora_cinemagraph"}

PART_BYTES = 2_000_000
KEEP_FILES = 24
MAX_IMAGE_BYTES = 20_000_000

#: Runs one authors' CLI in its own process; caps PyTorch memory first when asked and reports the peaks.
RUNNER = r'''
import json, os, runpy, sys, time
import torch
cap = float(os.environ.get("LTX_VRAM_CAP_GIB") or 0)
if cap > 0:
    total = torch.cuda.mem_get_info(0)[1]
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / total), 0)
module = sys.argv[1]
sys.argv = [module] + sys.argv[2:]
rc = 0
try:
    runpy.run_module(module, run_name="__main__", alter_sys=True)
except SystemExit as e:
    rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    if rc:
        raise
finally:
    print("LTX_RUNNER_PEAK " + json.dumps({
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 2**20),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 2**20)}), file=sys.stderr, flush=True)
'''


def _log(message: str) -> None:
    print(f"[ltx-backend] {message}", file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 * 2**20):
            h.update(chunk)
    return h.hexdigest()


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


class _Task:
    def __init__(self, task_id: str, kind: str, params: dict) -> None:
        self.id = task_id
        self.kind = kind
        self.params = params
        self.started = time.time()
        self.done = threading.Event()
        self.result: dict | None = None
        self.error: str | None = None
        self.path: Path | None = None
        self.sha256 = ""


class Ltx25Backend(CustomBackend):
    """LTX-2.5 distilled: a prompt (and optionally a first image) in, an MP4 with video and audio out."""

    name = "ltx-2-5"
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
        self.home = Path(os.environ.get("LTX_HOME") or Path.home() / ".cache" / "ltx-2-5")
        self.home.mkdir(parents=True, exist_ok=True)
        self._fix_cudnn()
        self.source = os.environ.get("LTX_SOURCE") or ("official" if os.environ.get("HF_TOKEN") else "mirror")
        self.paths, self.download = self._fetch_files()
        t_dl = time.time()
        self.runner = self.home / "ltx_runner.py"
        self.runner.write_text(RUNNER)

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("LTX-2.5 needs a CUDA GPU")
        self.card = torch.cuda.get_device_name(0)
        self.total_gib = torch.cuda.mem_get_info(0)[1] / 2**30
        self.capability = torch.cuda.get_device_capability(0)
        _log(f"card {self.card} {self.total_gib:.1f} GiB, sm_{self.capability[0]}{self.capability[1]}, torch "
             f"{torch.__version__}")
        self.vram = _VramSampler()
        self.lock = threading.Lock()
        self.tasks: "collections.OrderedDict[str, _Task]" = collections.OrderedDict()
        self.results = Path(tempfile.mkdtemp(prefix="ltx-clips-"))
        self.setup_timings = {"download_s": round(t_dl - t0, 1), "source": self.source, **self.download}
        _log(f"ready: setup {time.time() - t0:.1f}s {json.dumps(self.setup_timings)}")

    @staticmethod
    def _fix_cudnn() -> None:
        """Apply the authors' cuDNN override for torch 2.13 (their pyproject.toml), which a requirements.txt cannot.

        torch 2.13.0+cu132 pins nvidia-cudnn-cu13 9.20.0.48, which lacks
        libcudnn_engines_tensor_ir; on images with a system cuDNN in /usr/lib
        (we saw 9.12 on a cloud pod) that soname then resolves to the wrong
        library. 9.24.0.43 is the wheel the authors pin instead.
        """
        from importlib import metadata

        try:
            if metadata.version("nvidia-cudnn-cu13") == CUDNN_PIN:
                return
        except metadata.PackageNotFoundError:
            pass
        uv = shutil.which("uv")
        cmd = ([uv, "pip", "install", "--python", sys.executable] if uv else [sys.executable, "-m", "pip", "install", "-q"])
        subprocess.run(cmd + ["--no-deps", f"nvidia-cudnn-cu13=={CUDNN_PIN}"], check=True)
        _log(f"nvidia-cudnn-cu13 set to {CUDNN_PIN} (the LTX-2 authors' override for torch 2.13)")

    def _fetch_files(self) -> tuple[dict, dict]:
        """Download every file (in parallel) and check each SHA-256 against the one Lightricks publishes."""
        from huggingface_hub import hf_hub_download

        cache = self.home / "hub"
        stats: dict = {"files": {}}

        def one(item: tuple[str, dict]) -> tuple[str, Path]:
            key, f = item
            dest = self.home / "models" / Path(f["path"]).name
            ok = self.home / ".verified" / (f["sha256"] + ".ok")
            if dest.is_file() and dest.stat().st_size == f["size"] and ok.is_file():
                return key, dest
            t = time.time()
            if self.source == "official":
                repo, rev, path = f["repo"], f["revision"], f["path"]
            else:
                repo, rev, path = f["mirror"][0], f["mirror"][1], f["mirror"][2] or f["path"]
            blob = Path(hf_hub_download(repo, path, revision=rev, cache_dir=str(cache))).resolve()
            t_get = time.time()
            if blob.stat().st_size != f["size"] or _sha256(blob) != f["sha256"]:
                raise RuntimeError(f"checksum mismatch for {f['path']} from {repo}: not the bytes Lightricks published")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.unlink(missing_ok=True)
            os.link(blob, dest)  # same disk: no copy
            ok.parent.mkdir(parents=True, exist_ok=True)
            ok.touch()
            stats["files"][key] = {"from": repo, "gb": round(f["size"] / 1e9, 2), "download_s": round(t_get - t, 1),
                                   "sha256_s": round(time.time() - t_get, 1)}
            _log(f"{key}: {f['size'] / 1e9:.2f} GB from {repo} in {t_get - t:.1f}s, sha256 ok in {time.time() - t_get:.1f}s")
            return key, dest

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(FILES)) as pool:
            paths = dict(pool.map(one, FILES.items()))
        stats["total_gb"] = round(sum(f["size"] for f in FILES.values()) / 1e9, 2)
        stats["wall_s"] = round(time.time() - t0, 1)
        return paths, stats

    # ------------------------------------------------------------- generation

    def _command(self, p: dict, out: Path, image: Path | None) -> list[str]:
        cmd = [sys.executable, "-u", str(self.runner), "ltx_pipelines.distilled",
               "--transformer-path", str(self.paths["transformer"]),
               "--text-encoder-path", str(self.paths["text_encoder"]),
               "--video-vae-path", str(self.paths["video_vae"]),
               "--audio-vae-path", str(self.paths["audio_vae"]),
               "--spatial-upsampler-path", str(self.paths["spatial_upsampler"]),
               "--width", str(p["width"]), "--height", str(p["height"]),
               "--num-frames", str(p["num_frames"]), "--frame-rate", str(p["frame_rate"]),
               "--seed", str(p["seed"]), "--output-path", str(out), "--prompt", p["prompt"]]
        if p["quantization"] != "none":
            cmd += ["--quantization", p["quantization"]]
        if p["offload"] != "none":
            cmd += ["--offload", p["offload"]]
        if p["lora"]:
            cmd += ["--lora", str(p["lora_path"]), str(p["lora_strength"])]
        if image is not None:
            cmd += ["--image", str(image), "0", str(p["image_strength"])]
        return cmd

    def _run_logged(self, cmd: list[str], env: dict, cwd: Path | None = None) -> tuple[int, list[str], dict]:
        """Run a command, keep every stderr/stdout line with the seconds since start, and the runner's peak line."""
        t0 = time.time()
        lines: list[str] = []
        peak: dict = {}
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                cwd=str(cwd) if cwd else None, bufsize=1)
        for raw in proc.stdout:
            line = raw.rstrip()
            if line.startswith("LTX_RUNNER_PEAK "):
                peak = json.loads(line.split(" ", 1)[1])
                continue
            lines.append(f"{time.time() - t0:7.1f}s {line[:400]}")
            print(f"[ltx-run] {line[:400]}", file=sys.stderr, flush=True)
        return proc.wait(), lines, peak

    def _generate(self, task: _Task) -> dict:
        p = task.params
        out = self.results / f"{task.id}.mp4"
        image = None
        if p.get("image_bytes"):
            image = self.results / f"{task.id}-input.png"
            image.write_bytes(p.pop("image_bytes"))
        env = dict(os.environ)
        env["LTX_VRAM_CAP_GIB"] = str(p["vram_limit_gib"] or 0)
        self.vram.reset()
        t = time.time()
        rc, lines, peak = self._run_logged(self._command(p, out, image), env)
        wall = time.time() - t
        if rc != 0 or not out.is_file():
            tail = [ln for ln in lines if "Error" in ln or "error" in ln][-3:] or lines[-3:]
            raise RuntimeError(f"generation exited with {rc} after {wall:.0f}s: {' | '.join(tail)[-900:]}")
        probe = self._probe(out)
        return {"wall_s": round(wall, 1), "log": lines[-80:], **peak, "card_peak_mib": self.vram.peak, **probe}

    @staticmethod
    def _probe(path: Path) -> dict:
        import av

        with av.open(str(path)) as c:
            v = next((s for s in c.streams if s.type == "video"), None)
            a = next((s for s in c.streams if s.type == "audio"), None)
            info = {"seconds_of_video": round(float(c.duration or 0) / 1e6, 2)}
            if v is not None:
                info.update(width=v.codec_context.width, height=v.codec_context.height,
                            fps=float(v.average_rate or 0), frames=v.frames)
            info["audio"] = None if a is None else {"codec": a.codec_context.name, "sample_rate": a.codec_context.sample_rate,
                                                    "channels": a.codec_context.channels}
        return info

    @staticmethod
    def _middle_frame(path: Path) -> bytes:
        import av

        with av.open(str(path)) as c:
            frames = [f for f in c.decode(video=0)]
        buf = io.BytesIO()
        frames[len(frames) // 2].to_image().save(buf, format="JPEG", quality=88)
        return buf.getvalue()

    def _params(self, src: dict) -> dict:
        dash = not any(k in src for k in ("width", "height", "num_frames"))
        p = {
            "prompt": str(src.get("prompt") or "").strip(),
            "seed": int(src.get("seed") if src.get("seed") is not None else 42),
            "width": int(src.get("width") or (os.environ.get("LTX_DASH_WIDTH", 768) if dash and not src.get("full") else 1536)),
            "height": int(src.get("height") or (os.environ.get("LTX_DASH_HEIGHT", 512) if dash and not src.get("full") else 1024)),
            "num_frames": int(src.get("num_frames") or (os.environ.get("LTX_DASH_FRAMES", 49) if dash and not src.get("full") else 121)),
            "frame_rate": float(src.get("frame_rate") or 24),
            "quantization": str(src.get("quantization") or "fp8-cast"),
            "offload": str(src.get("offload") or "none"),
            "lora": str(src.get("lora") or ""),
            "lora_strength": float(src.get("lora_strength") or 1.0),
            "image_strength": float(src.get("image_strength") or 1.0),
            "vram_limit_gib": float(src.get("vram_limit_gib") or 0),
        }
        if not p["prompt"]:
            raise InvalidInput("`prompt` is required")
        if p["width"] % 64 or p["height"] % 64:
            raise InvalidInput("`width` and `height` must be multiples of 64")
        if (p["num_frames"] - 1) % 8:
            raise InvalidInput("`num_frames` must be 8k+1 (for example 49, 97 or 121)")
        if p["quantization"] not in ("fp8-cast", "fp8-scaled-mm", "none"):
            raise InvalidInput("`quantization` must be fp8-cast, fp8-scaled-mm or none")
        if p["offload"] not in ("none", "cpu", "disk"):
            raise InvalidInput("`offload` must be none, cpu or disk")
        if p["lora"]:
            key = LORAS.get(p["lora"])
            path = self.paths.get(key) if key else self.home / "loras" / Path(p["lora"]).name
            if path is None or not Path(path).is_file():
                raise InvalidInput(f"unknown LoRA {p['lora']!r} (built in: {sorted(LORAS)}; or a file in LTX_HOME/loras)")
            p["lora_path"] = path
        if src.get("image_b64"):
            try:
                raw = base64.b64decode(str(src["image_b64"]))
            except Exception:
                raise InvalidInput("`image_b64` is not base64")
            if len(raw) > MAX_IMAGE_BYTES:
                raise InvalidInput("image too large")
            from PIL import Image

            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                raise InvalidInput("`image_b64` is not a PNG/JPEG image")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            p["image_bytes"] = buf.getvalue()
        return p

    # ----------------------------------------------------------------- train

    def _ensure_trainer(self) -> dict:
        """Fetch the trainer scripts and configs at the pinned commit and install ltx-trainer into this venv."""
        repo = self.home / "LTX-2"
        t0 = time.time()
        if not (repo / "packages" / "ltx-trainer" / "scripts" / "train.py").is_file():
            shutil.rmtree(repo, ignore_errors=True)
            run = lambda *a: subprocess.run(list(a), check=True)  # noqa: E731
            run("git", "init", "-q", str(repo))
            run("git", "-C", str(repo), "remote", "add", "origin", LTX_REPO_URL)
            run("git", "-C", str(repo), "fetch", "-q", "--depth", "1", "origin", LTX_COMMIT)
            run("git", "-C", str(repo), "checkout", "-q", "FETCH_HEAD")
        t_src = time.time()
        try:
            import ltx_trainer  # noqa: F401
        except ImportError:
            spec = f"ltx-trainer @ git+{LTX_REPO_URL}@{LTX_COMMIT}#subdirectory=packages/ltx-trainer"
            uv = shutil.which("uv")
            pins = ["torch==2.13.0+cu132", "torchvision==0.28.0+cu132", "torchaudio==2.11.1+cu132"]
            cmd = ([uv, "pip", "install", "--python", sys.executable] if uv else [sys.executable, "-m", "pip", "install", "-q"])
            cmd += ["--extra-index-url", "https://download.pytorch.org/whl/cu132", spec, *pins]
            if uv:
                cmd += ["--index-strategy", "unsafe-best-match"]
            subprocess.run(cmd, check=True)
        return {"trainer_source_s": round(t_src - t0, 1), "trainer_install_s": round(time.time() - t_src, 1)}

    def _train(self, task: _Task) -> dict:
        import yaml

        p = task.params
        prep = self._ensure_trainer()
        trainer = self.home / "LTX-2" / "packages" / "ltx-trainer"
        work = Path(tempfile.mkdtemp(prefix="ltx-train-"))
        rows = []
        for cid in p["clip_ids"]:
            t = self.tasks.get(cid)
            if t is None or t.path is None or t.kind != "generate":
                raise InvalidInput(f"unknown clip {cid!r}: train on clips this backend generated")
            dest = work / "videos" / f"{cid}.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(t.path, dest)
            rows.append({"video": f"videos/{cid}.mp4", "caption": t.params["prompt"]})
        (work / "dataset.json").write_text(json.dumps(rows, indent=1))
        cfg = yaml.safe_load((trainer / "configs" / "t2v_lora_low_vram.yaml").read_text())
        ts = cfg["training_strategy"]
        ts["model"].update(model_path=str(self.paths["transformer"]), text_encoder_path=str(self.paths["text_encoder"]),
                           video_vae_path=str(self.paths["video_vae"]), audio_vae_path=str(self.paths["audio_vae"]))
        ts["data"].update(dataset_file=str(work / "dataset.json"), preprocessed_data_root=str(work / "precomputed"),
                          resolution_buckets=[[p["width"], p["height"], p["frames"]]])
        ts["validation"]["samples"] = []
        cfg["optimization"]["steps"] = p["steps"]
        cfg["validation"]["skip_initial_validation"] = True
        cfg["validation"]["interval"] = None
        cfg["checkpoints"]["interval"] = None
        cfg["output_dir"] = str(work / "out")
        (work / "run.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        env = dict(os.environ)
        env["LTX_VRAM_CAP_GIB"] = str(p["vram_limit_gib"] or 0)
        self.vram.reset()
        t = time.time()
        rc, pre_log, _ = self._run_logged([sys.executable, "-u", "scripts/preprocess.py", str(work / "run.yaml")], env,
                                          cwd=trainer)
        pre_s = time.time() - t
        pre_card = self.vram.peak
        if rc != 0:
            raise RuntimeError(f"preprocess exited with {rc} after {pre_s:.0f}s: {' | '.join(pre_log[-4:])[-900:]}")
        self.vram.reset()
        t = time.time()
        rc, log, peak = self._run_logged([sys.executable, "-u", "scripts/train.py", str(work / "run.yaml")], env,
                                         cwd=trainer)
        train_s = time.time() - t
        steps = [float(m.group(1)) for ln in log for m in [re.search(r"(\d+(?:\.\d+)?)\s*s/it", ln)] if m]
        its = [float(m.group(1)) for ln in log for m in [re.search(r"(\d+(?:\.\d+)?)\s*it/s", ln)] if m]
        result = {"rc": rc, **prep, "preprocess_s": round(pre_s, 1), "preprocess_card_peak_mib": pre_card,
                  "train_s": round(train_s, 1), "card_peak_mib": self.vram.peak, "steps": p["steps"],
                  "s_per_step_last": steps[-1] if steps else (round(1 / its[-1], 2) if its else None),
                  "outputs": sorted(str(x.relative_to(work)) for x in (work / "out").rglob("*") if x.is_file())[:20]
                  if (work / "out").is_dir() else [], "log": log[-60:], "preprocess_log": pre_log[-20:]}
        if rc != 0:
            result["error"] = " | ".join(log[-4:])[-900:]
        return result

    # --------------------------------------------------------------- running

    def _run_task(self, task: _Task) -> None:
        try:
            with self.lock:
                _log(f"task {task.id} ({task.kind}): {json.dumps({k: v for k, v in task.params.items() if k not in ('image_bytes', 'lora_path')})[:600]}")
                if task.kind == "generate":
                    out = self._generate(task)
                    task.path = self.results / f"{task.id}.mp4"
                    task.sha256 = hashlib.sha256(task.path.read_bytes()).hexdigest()
                    jpeg = self._middle_frame(task.path)
                    p = task.params
                    task.result = {
                        "created": int(time.time()), "data": [{"b64_json": base64.b64encode(jpeg).decode("ascii")}],
                        "output_format": "jpeg", "model": "Lightricks/LTX-2.5 (22B distilled)", "status": "done",
                        "clip_id": task.id, "prompt": p["prompt"], "seed": p["seed"],
                        "requested": {k: p[k] for k in ("width", "height", "num_frames", "frame_rate", "quantization",
                                                         "offload", "lora", "lora_strength", "vram_limit_gib")},
                        "image_to_video": "image_bytes" in p or (self.results / f"{task.id}-input.png").is_file(),
                        "card": self.card, "request_wall_s": round(time.time() - task.started, 2), **out,
                    }
                else:
                    task.result = {"status": "done", "clip_id": task.id, "kind": "train", "card": self.card,
                                   "data": [], **self._train(task)}
            _log(f"task {task.id} done: {json.dumps({k: v for k, v in task.result.items() if k not in ('data', 'log', 'preprocess_log')})[:1500]}")
        except Exception as e:  # noqa: BLE001 - reported to the caller on the next poll
            task.error = f"{type(e).__name__}: {e}"
            _log(f"task {task.id} failed: {task.error}")
        finally:
            task.done.set()

    def _answer(self, task: _Task, part: int = 0) -> dict:
        if not task.done.is_set():
            return {"status": "running", "clip_id": task.id, "kind": task.kind,
                    "elapsed_s": round(time.time() - task.started, 1), "data": []}
        if task.error:
            return {"status": "failed", "clip_id": task.id, "error": task.error, "data": []}
        if task.path is None:
            return task.result
        size = task.path.stat().st_size
        parts = (size + PART_BYTES - 1) // PART_BYTES
        if not 0 <= part < parts:
            raise InvalidInput(f"`part` must be between 0 and {parts - 1}")
        with task.path.open("rb") as f:
            f.seek(part * PART_BYTES)
            chunk = f.read(PART_BYTES)
        head = task.result if part == 0 else {"status": "done", "clip_id": task.id, "data": []}
        return {**head, "part": part, "parts": parts, "mp4_bytes": size, "mp4_sha256": task.sha256,
                "mp4_b64": base64.b64encode(chunk).decode("ascii")}

    # ---------------------------------------------------------------- process

    def process(self, job: Job) -> Result:
        src = job.input
        passthrough = job.input.get("passthrough_body")
        if isinstance(passthrough, dict) and passthrough:
            src = passthrough
        if src.get("info"):
            return Result({"status": "info", "card": self.card, "setup": self.setup_timings,
                           "tasks": [{"clip_id": t.id, "kind": t.kind, "done": t.done.is_set()} for t in self.tasks.values()],
                           "data": []})
        wait_s = float(src.get("wait_s") if src.get("wait_s") is not None else 270)
        if src.get("clip_id"):
            task = self.tasks.get(str(src["clip_id"]))
            if task is None:
                raise InvalidInput(f"unknown or expired clip_id {src['clip_id']!r}")
            if "part" not in src:
                task.done.wait(timeout=max(0.0, min(wait_s, 270)))
            return Result(self._answer(task, int(src.get("part") or 0)))
        if isinstance(src.get("train"), dict):
            tr = src["train"]
            params = {"clip_ids": [str(c) for c in tr.get("clip_ids") or []], "steps": int(tr.get("steps") or 30),
                      "width": int(tr.get("width") or 512), "height": int(tr.get("height") or 320),
                      "frames": int(tr.get("frames") or 49), "vram_limit_gib": float(tr.get("vram_limit_gib") or 0)}
            if not params["clip_ids"]:
                raise InvalidInput("`train.clip_ids` lists clips this backend generated")
            kind = "train"
        else:
            params = self._params(src)
            kind = "generate"
        if self.lock.locked():
            busy = next((t.id for t in reversed(self.tasks.values()) if not t.done.is_set()), None)
            raise InvalidInput(f"busy with {busy}; poll it with clip_id and send this one after")
        task = _Task(uuid.uuid4().hex[:16], kind, params)
        self.tasks[task.id] = task
        while len(self.tasks) > KEEP_FILES:
            _, old = self.tasks.popitem(last=False)
            if old.path is not None:
                old.path.unlink(missing_ok=True)
        threading.Thread(target=self._run_task, args=(task,), daemon=True).start()
        task.done.wait(timeout=max(0.0, min(wait_s, 270)))
        return Result(self._answer(task))

"""A custom backend that runs **PAN 2** (MBZUAI IFM), a world model, on one NVIDIA GPU.

PAN 2 (Apache-2.0, https://huggingface.co/IFM/PAN2) simulates how a scene
changes when actions happen: given a structured, time-stamped description of
the next world state (and, optionally, the current frame), it renders the
outcome as video, 704 x 1248 at 24 fps, up to 20 s. Its perceptual backbone is
a 29B-parameter MMDiT; a frozen Qwen3.5-9B encodes the description and the
Wan 2.2 VAE decodes the latents. All three ship in the ``IFM/PAN2`` repo
(about 86 GB in bf16). This backend runs them with the authors' Diffusers
integration (``ModularPipeline.from_pretrained("IFM/PAN2")``, diffusers
0.43.0).

The structured description is written by a separate "conceptual reasoner"
(any instruction-following LLM with the authors' rewriter policy:
``inference/pan-prompt-rewriter`` in github.com/ifm-ai/PAN2). This backend
does not rewrite prompts: send the structured caption as ``prompt``. A plain
sentence works too, but the model was trained on structured captions.

Contract (``task_type`` text2image, so the workload answers on the OpenAI
images route ``POST .../v1/images/generations``; the caller's JSON arrives in
``input.passthrough_body``):

* generate: ``prompt`` (structured caption, JSON string or object) or
  ``preset`` (``house``: the authors' first frame, a 3 s walk; ``glass``: a 2 s
  text-to-video physics clip). Optional: ``image`` (base64 first frame, makes
  it image-to-video) or ``example: "house"`` (the authors' first frame),
  ``negative_prompt`` (default: PAN 2's own), ``duration_sec`` or
  ``num_frames`` (4n+1), ``steps`` (default 20; the authors use 50),
  ``guidance_scale`` (2.5), ``seed`` (123), ``height`` / ``width`` (704 x 1248),
  ``vram_limit_gib`` (cap GPU memory for this request, to emulate a smaller
  card), ``text_encoder`` (``gpu``, ``offload``: Qwen3.5-9B waits in system
  RAM and visits the GPU while it runs, or ``cpu``: it runs on the CPU, about
  15 GiB less VRAM at the peak), ``vae_tiling`` (decode in tiles).
  From the dashboard's image tester write a preset name as the prompt, plus
  words like ``1s`` or ``8steps`` (for example ``glass 1s 8steps``).
* answer: a clip takes minutes and one request waits at most ``wait_s``
  (default 270 s, under the platform's 300 s limit). If the clip is not done
  by then the answer is ``{"clip_id", "status": "running"}``; ask again with
  ``{"clip_id": "..."}``. When done: ``status: "done"``, ``data[0].b64_json``
  (the middle frame as JPEG, so an image viewer shows something), the MP4 in
  parts of 2 MB (``mp4_b64`` = part 0, ``parts``, ``mp4_bytes``,
  ``mp4_sha256``; fetch part n with ``{"clip_id", "part": n}``), ``timings``
  (text encoding, denoising, seconds per step, VAE decode, MP4 encode) and
  ``peak_allocated_mib`` / ``peak_reserved_mib`` (PyTorch) and
  ``card_peak_mib`` (``nvidia-smi``).

Environment (the workload's ``config.env``):

* ``PAN2_TEXT_ENCODER``: ``auto`` (default: on the GPU when the card has
  90 GiB or more, else on the CPU), ``gpu``, ``offload`` (system RAM, moved to
  the GPU only while it runs) or ``cpu`` (runs on the CPU; the GPU then holds
  only the transformer and the VAE).
* ``PAN2_VAE_TILING``: ``1`` decodes in tiles on every request.
* ``PAN2_GUARDRAIL``: ``1`` turns on the authors' guardrail (needs
  ``pip install git+https://github.com/ifm-ai/pan2-guardrail`` in the
  requirements, plus Qwen3Guard-Gen-4B). Off by default here, see README.
* ``PAN2_WEIGHTS_DIR``: where the 86 GB go. Default
  ``~/.cache/huggingface``; keep it on a local disk, not a network volume.
* ``PAN2_WARMUP``: ``1`` runs a tiny clip in ``setup()`` (off by default).

Run it by hand (from this folder, with requirements installed)::

    python -m inferencekey.backend.serve --port 8099 --backend backend:PAN2Backend

Attribution: PAN 2 by the Institute of Foundation Models (IFM), MBZUAI
(Apache-2.0). Qwen3.5-9B by the Qwen team (Apache-2.0), Wan 2.2 VAE by
Wan-AI (Apache-2.0), Diffusers by Hugging Face (Apache-2.0).
"""

from __future__ import annotations

import base64
import collections
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
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

#: Pinned revisions this backend was tested with (2026-11-21).
PAN2_REPO, PAN2_REVISION = "IFM/PAN2", "82f54c6cea3e20a9c451b61d7360016e5970503d"
PAN2_CODE_COMMIT = "0fb907b6e16a82ead6d691d1de2eef6299c71e9a"
EXAMPLE_IMAGES = {
    "house": f"https://raw.githubusercontent.com/ifm-ai/PAN2/{PAN2_CODE_COMMIT}/inference/prompts/i2v_house_tour.jpg",
}
WEIGHT_PATTERNS = ["model_index.json", "modular_model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*",
                   "transformer/*", "vae/*", "safety_checker/*"]
COMPONENTS = ["text_encoder", "tokenizer", "transformer", "vae", "scheduler"]

#: One result message is limited to 8 MB and carries the payload about twice,
#: so each part stays at 2 MB of raw bytes (2.7 MB of base64).
PART_BYTES = 2_000_000
KEEP_FILES = 16
FPS = 24

#: Two structured captions written with the authors' rewriter policy
#: (inference/pan-prompt-rewriter/system_prompt.md), for the dashboard tester.
PRESETS = {
    "house": {
        "example": "house", "duration_sec": 3.0,
        "prompt": {
            "video_style": "Photorealistic first-person interior walkthrough filmed handheld at natural walking speed, one uninterrupted continuous shot.",
            "lighting": "Soft daylight from tall windows on the left and the glass front door on the right; warm glow from the iron lantern and the table lamp; gentle shadows on white plaster and pale oak.",
            "perspective_view": "Eye-level first-person view of a person walking, wide lens, starting in the entry hall facing the staircase.",
            "subjects_lifecycle": [],
            "init_frame_visual_description": "A bright entry hall with white plaster walls and pale wide-plank oak floors. A beige patterned rug lies in the foreground. A black iron lantern with candle bulbs hangs from the ceiling. On the left, a tall rounded arch opens into a living room with a cream sofa, a lamp on a side table, built-in niche shelves, a landscape painting above a white fireplace and a potted olive tree by the window. Straight ahead a doorway shows the kitchen with oak cabinets, and a wooden staircase with a black iron handrail rises to the right of it beside a large clay vase with olive branches. On the right, a second arch opens into a dining room with a wooden table, chairs and a pendant light, and a dark wooden front door with glass panes stands at the right edge.",
            "events": [
                {"event_index": 0, "time_range_sec": "0 to 1.5", "event_description": "The walk starts at once: the view moves forward across the rug toward the middle of the hall, the lantern rising toward the top of the frame and the staircase and kitchen doorway growing larger, with slight natural head bob.", "camera_motion": [{"time_range_sec": "0 to 1.5", "camera_motion": ["dolly_forward"]}], "contained_actions": []},
                {"event_index": 1, "time_range_sec": "1.5 to 3", "event_description": "Still walking, the view turns smoothly left toward the living-room arch: the staircase slides out of the right side of the frame and the sofa, fireplace and olive tree come to the center, framed by the curve of the arch as it draws closer.", "camera_motion": [{"time_range_sec": "1.5 to 3", "camera_motion": ["dolly_forward", "pan_left"]}], "contained_actions": []},
            ],
            "quality": {"video_quality": "Sharp, stable architecture with straight verticals and real walking parallax.", "text_logos_overlays": None, "artifacts": None},
        },
    },
    "glass": {
        "duration_sec": 2.0,
        "prompt": {
            "video_style": "Photorealistic close-up product-style footage, one uninterrupted continuous shot.",
            "lighting": "Late-morning window light from the left, soft shadows on the table, bright highlights on the glass and water.",
            "perspective_view": "Static side view at table height, medium close-up of a wooden kitchen table.",
            "subjects_lifecycle": [
                {"subject_id": "<|Subject1|>", "display_name": "hand", "subject_attribute": "An adult right hand with a light skin tone and a rolled-up grey cotton sleeve."},
                {"subject_id": "<|Subject2|>", "display_name": "glass of water", "subject_attribute": "A tall clear glass tumbler three-quarters full of still water."},
            ],
            "init_frame_visual_description": "A tall clear glass three-quarters full of water stands near the front edge of an oiled oak kitchen table, centered in frame. A right hand in a grey sleeve rests flat on the table just to the right of the glass. Behind, out of focus, a white tiled wall and a window frame catch the light.",
            "events": [
                {"event_index": 0, "time_range_sec": "0 to 1", "event_description": "The hand slides left and the fingers flick the side of the glass near its top; the glass tips over to the left.", "camera_motion": [{"time_range_sec": "0 to 1", "camera_motion": ["static"]}], "contained_actions": [{"time_range_sec": "0 to 0.8", "action": "<|Subject1|> flicks the upper side of <|Subject2|> to the left."}, {"time_range_sec": "0.5 to 1", "action": "<|Subject2|> tips over and lands on its side."}]},
                {"event_index": 1, "time_range_sec": "1 to 2", "event_description": "Water pours out of the fallen glass and spreads in a thin sheet across the oak, running toward the table edge while the glass rolls slightly; the hand pulls back.", "camera_motion": [{"time_range_sec": "1 to 2", "camera_motion": ["static"]}], "contained_actions": [{"time_range_sec": "1 to 2", "action": "<|Subject2|> spills its water across the table."}, {"time_range_sec": "1.2 to 2", "action": "<|Subject1|> pulls back to the right."}]},
            ],
            "quality": {"video_quality": "Crisp water and glass edges, natural wood grain.", "text_logos_overlays": None, "artifacts": None},
        },
    },
}


def _log(message: str) -> None:
    print(f"[pan2-backend] {message}", file=sys.stderr, flush=True)


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
    """One generation running in the background (a clip takes longer than one request may wait)."""

    def __init__(self, clip_id: str, params: dict) -> None:
        self.id = clip_id
        self.params = params
        self.started = time.time()
        self.done = threading.Event()
        self.result: dict | None = None
        self.error: str | None = None
        self.path: Path | None = None
        self.sha256 = ""


class PAN2Backend(CustomBackend):
    """PAN 2 world model: a structured world-state caption (and optionally a first frame) in, a video out."""

    name = "pan2-world-model"
    version = "0.1.2"
    task_type = "text2image"
    requirements = "requirements.txt"
    requires_python = ">=3.12,<3.13"

    # ------------------------------------------------------------------ setup

    def setup(self, ctx: BackendContext) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        if os.environ.get("PAN2_WEIGHTS_DIR"):
            os.environ["HF_HOME"] = os.environ["PAN2_WEIGHTS_DIR"]
        t0 = time.time()

        import torch
        from huggingface_hub import snapshot_download

        if not torch.cuda.is_available():
            raise RuntimeError("PAN 2 needs a CUDA GPU")
        self.torch = torch
        self.device = torch.device("cuda", 0)
        self.card = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        self.total_gib = total / 2**30
        hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        Path(hf_home).mkdir(parents=True, exist_ok=True)
        disk = shutil.disk_usage(hf_home)
        _log(f"card {self.card} {self.total_gib:.1f} GiB, weights cache {hf_home} "
             f"({disk.free / 1e9:.0f} GB free of {disk.total / 1e9:.0f} GB)")
        if hf_home.startswith("/workspace"):
            _log("warning: /workspace is a network volume on many clouds; loading 86 GB from it is slow")

        local = snapshot_download(PAN2_REPO, revision=PAN2_REVISION, allow_patterns=WEIGHT_PATTERNS)
        t_download = time.time()
        size = sum(f.stat().st_size for f in Path(local).rglob("*") if f.is_file())
        _log(f"downloads {t_download - t0:.1f}s ({size / 1e9:.1f} GB, {size / 1e9 / max(t_download - t0, 1e-3):.2f} GB/s "
             f"including any cache hits)")

        from diffusers import ModularPipeline

        mode = (os.environ.get("PAN2_TEXT_ENCODER") or "auto").lower()
        if mode not in ("auto", "gpu", "offload", "cpu"):
            raise ValueError("PAN2_TEXT_ENCODER must be auto, gpu, offload or cpu")
        # auto: all on the GPU from 90 GiB up; below that the 63 GiB transformer and the
        # 15 GiB text encoder do not fit together, so the text encoder runs on the CPU.
        self.te_default = ("gpu" if self.total_gib >= 90 else "cpu") if mode == "auto" else mode
        self.te_offload_default = self.te_default != "gpu"
        te_device = "cuda" if self.te_default == "gpu" else "cpu"
        pipe = ModularPipeline.from_pretrained(PAN2_REPO, revision=PAN2_REVISION)
        pipe.load_components(
            names=COMPONENTS, revision=PAN2_REVISION, dtype=torch.bfloat16,
            device_map={"transformer": "cuda", "vae": "cuda", "text_encoder": te_device},
        )
        # The pipeline picks its execution device from its first model, which is the
        # text encoder; when that one waits in system RAM the rest must still run on the GPU.
        gpu = self.device
        pipe.__class__ = type("PAN2OnGPU", (type(pipe),), {"_execution_device": property(lambda _self: gpu)})
        if (os.environ.get("PAN2_GUARDRAIL") or "0") == "1":
            pipe.enable_safety_checker()
        else:
            pipe.disable_safety_checker()
        self.pipe = pipe
        self.te_on_gpu = te_device == "cuda"
        self.te_offload = self.te_offload_default
        self.te_cpu = self.te_default == "cpu"
        self._install_hooks()
        torch.cuda.synchronize()
        t_load = time.time()
        self.weights_gib = torch.cuda.memory_allocated(0) / 2**30

        self.lock = threading.Lock()
        self.clips: "collections.OrderedDict[str, _Clip]" = collections.OrderedDict()
        self.results = Path(tempfile.mkdtemp(prefix="pan2-clips-"))
        self.examples: dict[str, object] = {}
        self.vram = _VramSampler()
        self.setup_timings = {"download_s": round(t_download - t0, 1), "load_s": round(t_load - t_download, 1),
                              "weights_gib_on_gpu": round(self.weights_gib, 2)}
        _log(f"load {t_load - t_download:.1f}s, weights on the GPU {self.weights_gib:.2f} GiB, text encoder "
             f"{self.te_default}, guardrail "
             f"{'on' if pipe.requires_safety_checker else 'off'}, attention "
             f"{getattr(pipe.transformer, '_attention_backend', None) or 'native (SDPA)'}")
        if (os.environ.get("PAN2_WARMUP") or "0") == "1":
            self._generate(self._params({"preset": "glass", "num_frames": 9, "steps": 2, "height": 352,
                                         "width": 624}))
        _log(f"ready: setup {time.time() - t0:.1f}s")

    def _install_hooks(self) -> None:
        """Time the three stages, and move the text encoder to the GPU only while it runs when offload is on."""
        torch = self.torch
        te, tr, vae = self.pipe.text_encoder, self.pipe.transformer, self.pipe.vae
        self.stage = {"text_encoder_s": 0.0, "denoise_s": 0.0, "steps": 0, "decode_s": 0.0, "te_moves_s": 0.0}
        marks: dict[str, float] = {}

        def te_pre(module, args, kwargs):
            if self.te_cpu:  # run Qwen3.5-9B on the CPU: the GPU never holds it
                marks["te"] = time.time()
                return args, {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in kwargs.items()}
            if self.te_offload and not self.te_on_gpu:
                t = time.time()
                module.to(self.device)
                self.te_on_gpu = True
                self.stage["te_moves_s"] += time.time() - t
            marks["te"] = time.time()
            return args, {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in kwargs.items()}

        def te_post(module, args, kwargs, output):
            torch.cuda.synchronize()
            self.stage["text_encoder_s"] += time.time() - marks["te"]
            if self.te_cpu:
                hs = getattr(output, "hidden_states", None)
                if hs is not None:
                    output.hidden_states = tuple(h.to(self.device) for h in hs)
                if getattr(output, "last_hidden_state", None) is not None:
                    output.last_hidden_state = output.last_hidden_state.to(self.device)
                return output
            if self.te_offload and self.te_on_gpu:
                t = time.time()
                module.to("cpu")
                self.te_on_gpu = False
                torch.cuda.empty_cache()
                self.stage["te_moves_s"] += time.time() - t
            return output

        te.register_forward_pre_hook(te_pre, with_kwargs=True)
        te.register_forward_hook(te_post, with_kwargs=True)

        def tr_pre(module, args, kwargs):
            if "tr0" not in marks:
                marks["tr0"] = time.time()

        def tr_post(module, args, kwargs, output):
            self.stage["steps"] += 1
            return output

        tr.register_forward_pre_hook(tr_pre, with_kwargs=True)
        tr.register_forward_hook(tr_post, with_kwargs=True)

        decode = vae.decode

        def timed_decode(*args, **kwargs):
            torch.cuda.synchronize()
            if "tr0" in marks:
                self.stage["denoise_s"] = time.time() - marks.pop("tr0")
            t = time.time()
            out = decode(*args, **kwargs)
            torch.cuda.synchronize()
            self.stage["decode_s"] += time.time() - t
            return out

        vae.decode = timed_decode
        self._marks = marks

    # ---------------------------------------------------------------- inputs

    def _example(self, name: str):
        from PIL import Image

        if name not in EXAMPLE_IMAGES:
            raise InvalidInput(f"unknown example {name!r}; known: {sorted(EXAMPLE_IMAGES)}")
        if name not in self.examples:
            req = urllib.request.Request(EXAMPLE_IMAGES[name], headers={"User-Agent": "inferencekey-pan2-backend"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                self.examples[name] = Image.open(io.BytesIO(resp.read())).convert("RGB")
        return self.examples[name]

    def _params(self, src: dict) -> dict:
        words = str(src.get("prompt") or "").strip().lower().split() if isinstance(src.get("prompt"), str) else []
        preset_name = src.get("preset") or (words[0] if words and words[0] in PRESETS else None)
        p: dict = {}
        if preset_name:
            if preset_name not in PRESETS:
                raise InvalidInput(f"unknown preset {preset_name!r}; known: {sorted(PRESETS)}")
            p.update(PRESETS[preset_name])
            for w in words[1:]:  # dashboard words: "glass 1s 8steps"
                if w.endswith("steps") and w[:-5].isdigit():
                    p["steps"] = int(w[:-5])
                elif w.endswith("s") and w[:-1].replace(".", "", 1).isdigit():
                    p["duration_sec"] = float(w[:-1])
        else:
            if not src.get("prompt"):
                raise InvalidInput("send `prompt` (a structured PAN 2 caption) or `preset` (house or glass)")
            p["prompt"] = src["prompt"]
        for key in ("example", "image", "negative_prompt", "duration_sec", "num_frames", "steps", "guidance_scale",
                    "seed", "height", "width", "vram_limit_gib", "text_encoder", "text_encoder_offload", "vae_tiling"):
            if src.get(key) is not None and not (preset_name and key == "prompt"):
                p[key] = src[key]
        if isinstance(p["prompt"], (dict, list)):
            p["prompt"] = json.dumps(p["prompt"], ensure_ascii=False, separators=(",", ":"))
        if p.get("num_frames") is None:
            dur = float(p.get("duration_sec") or 3.0)
            p["num_frames"] = 4 * -(-int(round(dur * FPS)) // 4) + 1
        p["num_frames"] = int(p["num_frames"])
        if p["num_frames"] % 4 != 1 or not 5 <= p["num_frames"] <= 481:
            raise InvalidInput("`num_frames` must be 4n+1 between 5 and 481 (20 s at 24 fps)")
        p["steps"] = int(p.get("steps") or 20)
        if not 1 <= p["steps"] <= 100:
            raise InvalidInput("`steps` must be between 1 and 100")
        p["guidance_scale"] = float(p.get("guidance_scale") if p.get("guidance_scale") is not None else 2.5)
        p["seed"] = int(p["seed"]) if p.get("seed") is not None else 123
        p["height"], p["width"] = int(p.get("height") or 704), int(p.get("width") or 1248)
        if p["height"] % 32 or p["width"] % 32:
            raise InvalidInput("`height` and `width` must be multiples of 32")
        p["vram_limit_gib"] = float(p.get("vram_limit_gib") or 0)
        default_te = getattr(self, "te_default", "gpu")
        if p.get("text_encoder") is None and p.get("text_encoder_offload") is not None:
            p["text_encoder"] = "offload" if p["text_encoder_offload"] else "gpu"
        p["text_encoder"] = str(p.get("text_encoder") or default_te).lower()
        if p["text_encoder"] not in ("gpu", "offload", "cpu"):
            raise InvalidInput("`text_encoder` must be gpu, offload (GPU only while it runs) or cpu")
        p["text_encoder_offload"] = p["text_encoder"] != "gpu"
        p["vae_tiling"] = bool(p.get("vae_tiling")) or (os.environ.get("PAN2_VAE_TILING") or "0") == "1"
        return p

    def _first_frame(self, p: dict):
        from PIL import Image

        if p.get("image"):
            raw = str(p["image"])
            if raw.startswith("data:"):
                raw = raw.split(",", 1)[-1]
            try:
                img = Image.open(io.BytesIO(base64.b64decode(raw)))
                img.load()
            except Exception:
                raise InvalidInput("`image` is not a readable base64 image")
            return img.convert("RGB")
        if p.get("example"):
            return self._example(str(p["example"]))
        return None

    # --------------------------------------------------------------- running

    def _generate(self, p: dict) -> dict:
        torch = self.torch
        image = self._first_frame(p)
        if p["vram_limit_gib"] > 0:
            torch.cuda.set_per_process_memory_fraction(min(1.0, p["vram_limit_gib"] / self.total_gib), 0)
        self.te_cpu = p["text_encoder"] == "cpu"
        self.te_offload = p["text_encoder"] in ("offload", "cpu")
        if self.te_offload and self.te_on_gpu:
            self.pipe.text_encoder.to("cpu")
            self.te_on_gpu = False
        elif not self.te_offload and not self.te_on_gpu:
            self.pipe.text_encoder.to(self.device)
            self.te_on_gpu = True
        if p["vae_tiling"]:
            self.pipe.vae.enable_tiling()
        else:
            self.pipe.vae.disable_tiling()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        self.vram.reset()
        for k in self.stage:
            self.stage[k] = 0 if k == "steps" else 0.0
        self._marks.clear()
        tok = self.pipe.tokenizer(p["prompt"], add_special_tokens=False)
        prompt_tokens = len(tok.input_ids)
        t0 = time.time()
        try:
            kwargs = dict(prompt=p["prompt"], num_frames=p["num_frames"], num_inference_steps=p["steps"],
                          height=p["height"], width=p["width"], fps=float(FPS),
                          generator=torch.Generator().manual_seed(p["seed"]), output="videos")
            if p.get("negative_prompt"):
                kwargs["negative_prompt"] = p["negative_prompt"]
            if image is not None:
                kwargs["image"] = image
            if abs(p["guidance_scale"] - 2.5) > 1e-6:
                self.pipe.update_components(guider=self.pipe.guider.new(guidance_scale=p["guidance_scale"]))
            with torch.inference_mode():
                videos = self.pipe(**kwargs)
            torch.cuda.synchronize()
        finally:
            if p["vram_limit_gib"] > 0:
                torch.cuda.set_per_process_memory_fraction(1.0, 0)
            if abs(p["guidance_scale"] - 2.5) > 1e-6:
                self.pipe.update_components(guider=self.pipe.guider.new(guidance_scale=2.5))
        model_s = time.time() - t0
        video = videos[0]
        stage = dict(self.stage)
        steps = max(1, p["steps"])
        return {
            "video": video, "model_s": round(model_s, 2), "prompt_tokens": prompt_tokens,
            "timings": {"model_s": round(model_s, 2), "text_encoder_s": round(stage["text_encoder_s"], 2),
                        "text_encoder_moves_s": round(stage["te_moves_s"], 2),
                        "denoise_s": round(stage["denoise_s"], 2),
                        "transformer_calls": stage["steps"],
                        "seconds_per_step": round(stage["denoise_s"] / steps, 2),
                        "decode_s": round(stage["decode_s"], 2)},
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 2**20),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 2**20),
            "card_peak_mib": self.vram.peak,
        }

    @staticmethod
    def _encode_mp4(video, path: Path) -> None:
        import av
        import numpy as np

        frames = np.asarray(video)
        if frames.dtype != np.uint8:
            frames = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)
        with av.open(str(path), mode="w", format="mp4") as out:
            s = out.add_stream("libx264", rate=FPS)
            s.height, s.width = frames.shape[1], frames.shape[2]
            s.pix_fmt = "yuv420p"
            s.options = {"crf": "18", "preset": "medium"}
            for fr in frames:
                for packet in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="rgb24")):
                    out.mux(packet)
            for packet in s.encode():
                out.mux(packet)

    def _run_clip(self, clip: _Clip) -> None:
        try:
            with self.lock:  # one clip at a time on the GPU
                _log(f"clip {clip.id}: {clip.params['num_frames']} frames, {clip.params['steps']} steps, "
                     f"{clip.params['height']}x{clip.params['width']}, "
                     f"{'i2v' if clip.params.get('image') or clip.params.get('example') else 't2v'}, "
                     f"cap {clip.params['vram_limit_gib'] or 'none'}, text encoder {clip.params['text_encoder']}, "
                     f"vae tiling {clip.params['vae_tiling']}")
                out = self._generate(clip.params)
            t = time.time()
            path = self.results / f"{clip.id}.mp4"
            self._encode_mp4(out["video"], path)
            out["timings"]["mp4_encode_s"] = round(time.time() - t, 2)
            import numpy as np
            from PIL import Image

            frames = np.asarray(out.pop("video"))
            mid = frames[len(frames) // 2]
            if mid.dtype != np.uint8:
                mid = (np.clip(mid, 0, 1) * 255).round().astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(mid).save(buf, format="JPEG", quality=88)
            clip.path = path
            clip.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            p = clip.params
            clip.result = {
                "created": int(time.time()), "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")}],
                "output_format": "jpeg", "model": PAN2_REPO, "status": "done", "clip_id": clip.id,
                "task": "i2v" if (p.get("image") or p.get("example")) else "t2v",
                "num_frames": p["num_frames"], "fps": FPS, "seconds_of_video": round((p["num_frames"] - 1) / FPS, 2),
                "height": p["height"], "width": p["width"], "steps": p["steps"],
                "guidance_scale": p["guidance_scale"], "seed": p["seed"], "prompt_tokens": out["prompt_tokens"],
                "vram_limit_gib": p["vram_limit_gib"] or None, "text_encoder": p["text_encoder"],
                "vae_tiling": p["vae_tiling"], "card": self.card, "timings": out["timings"],
                "peak_allocated_mib": out["peak_allocated_mib"], "peak_reserved_mib": out["peak_reserved_mib"],
                "card_peak_mib": out["card_peak_mib"], "wall_s": round(time.time() - clip.started, 2),
            }
            _log(f"clip {clip.id} done: {json.dumps({k: clip.result[k] for k in ('timings', 'peak_allocated_mib', 'peak_reserved_mib', 'card_peak_mib', 'prompt_tokens')})}")
        except Exception as e:  # noqa: BLE001 - reported to the caller on the next poll
            clip.error = f"{type(e).__name__}: {e}"
            _log(f"clip {clip.id} failed: {clip.error}")
            try:
                self.torch.cuda.empty_cache()
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

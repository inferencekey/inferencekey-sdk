"""Simulate the outcome of an action as video with **PAN 2** (MBZUAI IFM) on one private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`pan2-world-model`, published with publish.py) that runs the
authors' Diffusers pipeline, and a clip takes minutes, so the call returns a
`clip_id` and the script polls until the video is ready.

PAN 2 reads a *structured* world-state caption (style, lighting, viewpoint,
subjects, the first frame, and timed events with camera motion), written by
a "conceptual reasoner" LLM with the authors' rewriter policy
(github.com/ifm-ai/PAN2, inference/pan-prompt-rewriter). Two ready-made
captions live in the backend as presets:

* `house` (default): image-to-video from the authors' own first frame, a 3 s
  first-person walk into the hall that turns left toward the living room;
* `glass`: text-to-video, 2 s, a hand knocks a glass of water over.

Or bring your own: `python main.py --caption my_caption.json [--image first.jpg]`.

Output in ./outputs: `<name>.mp4` (704 x 1248, 24 fps, H.264) and the
timings and GPU memory the backend measured.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py [--preset house|glass] [--steps 20]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = "pan2"
BACKEND_SLUG = "pan2-world-model"  # the slug publish.py registered
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def simulate(url: str, key: str, body: dict, out: Path) -> dict:
    """Start one clip, poll until it is done, fetch every part of the MP4 and check it."""
    headers = {"Authorization": f"Bearer {key}"}
    t = time.time()
    # One request waits at most about 270 s (the platform's limit is 300 s);
    # a longer clip comes back as {"status": "running", "clip_id": ...}.
    r = requests.post(url, headers=headers, json={"model": "default", **body}, timeout=330)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    answer = r.json()
    while answer.get("status") == "running":
        print(f"  still simulating ({answer['elapsed_s']:.0f} s)...")
        r = requests.post(url, headers=headers, timeout=330,
                          json={"model": "default", "clip_id": answer["clip_id"]})
        r.raise_for_status()
        answer = r.json()
    if answer.get("status") != "done":
        raise RuntimeError(f"clip {answer.get('clip_id')} failed: {answer.get('error')}")
    generated = time.time() - t
    parts = [base64.b64decode(answer["mp4_b64"])]
    for n in range(1, answer["parts"]):
        r = requests.post(url, headers=headers, timeout=300,
                          json={"model": "default", "clip_id": answer["clip_id"], "part": n})
        r.raise_for_status()
        parts.append(base64.b64decode(r.json()["mp4_b64"]))
    mp4 = b"".join(parts)
    if hashlib.sha256(mp4).hexdigest() != answer["mp4_sha256"]:
        raise RuntimeError("the MP4 arrived corrupted (SHA-256 mismatch)")
    out.write_bytes(mp4)
    meta = {k: v for k, v in answer.items() if k not in ("mp4_b64", "data")}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    tm = answer["timings"]
    print(
        f"wrote {out.name}: {answer['task']}, {answer['seconds_of_video']} s of video "
        f"({answer['num_frames']} frames, {answer['width']}x{answer['height']}), {answer['steps']} steps; "
        f"model {tm['model_s']} s (text {tm['text_encoder_s']} s, denoise {tm['denoise_s']} s = "
        f"{tm['seconds_per_step']} s/step, decode {tm['decode_s']} s), round trip {generated:.0f} s; "
        f"peak VRAM {answer['peak_reserved_mib']} MiB reserved by PyTorch, {answer['card_peak_mib']} MiB on the card "
        f"({answer['card']})"
    )
    return answer


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="house", choices=["house", "glass"])
    ap.add_argument("--caption", help="a JSON file with your own structured caption (instead of a preset)")
    ap.add_argument("--image", help="first frame for image-to-video with --caption")
    ap.add_argument("--duration", type=float, help="seconds of video (default: the preset's, 3 s for --caption)")
    ap.add_argument("--steps", type=int, default=20, help="denoising steps (the authors use 50)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if args.caption:
        body = {"prompt": json.loads(Path(args.caption).read_text())}
        if args.image:
            body["image"] = base64.b64encode(Path(args.image).read_bytes()).decode("ascii")
        name = Path(args.caption).stem
    else:
        body = {"preset": args.preset}
        name = args.preset
    body.update({"steps": args.steps, "seed": args.seed})
    if args.duration:
        body["duration_sec"] = args.duration

    mgmt = ManagementClient.from_env()
    # Reaches the backend: env vars for its process, how long the worker waits
    # for setup() (venv + about 86 GB of weights + loading them) and how long
    # one request may take.
    keys = ("PAN2_TEXT_ENCODER", "PAN2_VAE_TILING", "PAN2_GUARDRAIL", "PAN2_WEIGHTS_DIR", "PAN2_WARMUP", "HF_TOKEN")
    backend_env = {k: os.environ[k] for k in keys if os.environ.get(k)}
    config = {"ready_timeout_secs": 3600, "request_timeout_secs": 330, "env": backend_env}
    spec = WorkloadSpec(
        name="PAN 2 world model",
        slug=SLUG,
        model="IFM/PAN2",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        # Must match the backend package; the clip is served on the images route.
        task_type="text2image",
        worker_id=os.environ["IK_WORKER_ID"],
        config=config,
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
    )
    base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
    project = os.environ["INFERENCEKEY_PROJECT"]
    key = os.environ["INFERENCEKEY_API_KEY"]
    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        print("waiting for the worker to become ready (the first start downloads about 86 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.time() - t0:.0f} s")

        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/images/generations"
        OUT_DIR.mkdir(exist_ok=True)
        simulate(url, key, body, OUT_DIR / f"{name}.mp4")
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (requests.RequestException, RuntimeError) as e:
        print(f"request failed: {e}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the
        # workload (useful to try more captions from the dashboard).
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Upscale a video 4x with **RelayVSR** on one private NVIDIA GPU, through a custom backend.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`relayvsr-video-upscale`, published with publish.py) because no
built-in backend serves RelayVSR, and the payload is a VIDEO in and a video out.

RelayVSR only runs its big Wan2.2-based generator on sparse reference frames
(every `window_size - 1` frames) and lets a small network carry the detail to
every frame in between. The script upscales the same clip twice to show the
trade-off:

  1. `window_size` 16: a reference every 15 frames (the authors' default),
  2. `window_size` 32: a reference every 31 frames (fewer big-model calls).

Input: your own clip (`python main.py my-clip.mp4`, MP4/WebM/MOV up to 5 MB and
960 x 544 per frame) or, with no argument, the authors' bundled demo clip
(624 x 360, 16 fps). Output: `outputs/x4-w16.mp4` and `outputs/x4-w32.mp4`,
exactly 4x the input size, no audio.

Transport: the custom backend answers the OpenAI chat route
(`/v1/chat/completions`); the user message is the request as JSON and the
message content of the answer is the backend's JSON (base64 MP4 + timings).
The script turns the workload's `chat` endpoint on once with your dashboard
session token (or flip "Chat" in Dashboard -> Endpoint settings).

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py [clip.mp4]
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = os.environ.get("RVSR_WORKLOAD_SLUG", "relayvsr")
BACKEND_SLUG = os.environ.get("RVSR_BACKEND_SLUG", "relayvsr-video-upscale")
OUT_DIR = Path(__file__).resolve().parent / "outputs"
MAX_FRAMES = int(os.environ.get("RVSR_CLIP_FRAMES", "72"))


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def enable_chat_route(base: str, project: str, slug: str) -> None:
    """Turn on the workload's `chat` endpoint (text2image workloads start with images + jobs)."""
    user_token = os.environ.get("INFERENCEKEY_USER_TOKEN")
    if not user_token:
        print("INFERENCEKEY_USER_TOKEN not set: turn on 'Chat' in Dashboard -> Endpoint settings for this workload.")
        return
    sdk = {"Authorization": f"Bearer {os.environ['INFERENCEKEY_SDK_TOKEN']}"}
    r = requests.get(f"{base}/api/projects/{project}/workloads", headers=sdk, timeout=30)
    r.raise_for_status()
    items = r.json()
    items = items.get("items", items) if isinstance(items, dict) else items
    wid = next(w["id"] for w in items if w.get("slug") == slug)
    r = requests.patch(f"{base}/api/workloads/{wid}/endpoint_settings/chat",
                       headers={"Authorization": f"Bearer {user_token}"}, json={"enabled": True}, timeout=30)
    r.raise_for_status()
    print("chat endpoint enabled")


def upscale(url: str, key: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """One clip: the request JSON goes as the user message, the answer content is JSON."""
    req = {"model": "default", "messages": [{"role": "user", "content": json.dumps(body)}]}
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=req, timeout=900)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return json.loads(r.json()["choices"][0]["message"]["content"])


def main() -> int:
    _load_dotenv()
    clip = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    source: Dict[str, Any] = (
        {"video_b64": base64.b64encode(clip.read_bytes()).decode()} if clip else {"demo": True}
    )
    mgmt = ManagementClient.from_env()
    # Reaches the backend: env vars for its process, and how long the worker waits
    # for setup() (venv with torch 2.10 + 10.2 GB of weights + kernel warm-up) and for one clip.
    keys = ("RVSR_VRAM_LIMIT_GIB", "RVSR_DTYPE", "RVSR_MAX_FRAMES", "RVSR_OUTPUT_DIR", "RVSR_WARMUP")
    backend_env = {k: os.environ[k] for k in keys if os.environ.get(k)}
    config = {"ready_timeout_secs": 2400, "request_timeout_secs": 900, "env": backend_env}
    spec = WorkloadSpec(
        name="RelayVSR video upscaler",
        slug=SLUG,
        model="kopper/RelayVSR",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        # Must match the backend package, which the SDK can only tag text2image today
        # (no video task type for custom backends yet); the output is still MP4 video.
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
        enable_chat_route(base, project, ref.workload_slug)
        print("waiting for the worker to become ready (first start installs torch 2.10 and downloads 10.2 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        print(f"ready after {time.time() - t0:.1f} s")

        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/chat/completions"
        OUT_DIR.mkdir(exist_ok=True)
        for window in (16, 32):
            body = {**source, "window_size": window, "max_frames": MAX_FRAMES, "seed": 42}
            t = time.time()
            out = upscale(url, key, body)
            path = OUT_DIR / f"x4-w{window}.mp4"
            path.write_bytes(base64.b64decode(out["video_b64"]))
            tm = out["timings"]
            print(
                f"window {window}: {path.name} {out['input_width']}x{out['input_height']} -> "
                f"{out['width']}x{out['height']}, {out['frames']} frames, {out['keyframes']} reference frames; "
                f"model {tm['model_s']} s ({tm['model_fps']} frames/s: Wan {tm['wan_ms']:.0f} ms, "
                f"small network {tm['flash_ms']:.0f} ms), first frame {tm['first_frame_ms']:.0f} ms, "
                f"peak VRAM {tm['peak_vram_mib']} MiB, CRF {out['crf']}, round trip {time.time() - t:.1f} s"
            )
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (requests.RequestException, RuntimeError, KeyError, ValueError) as e:
        print(f"request failed: {e}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the workload.
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

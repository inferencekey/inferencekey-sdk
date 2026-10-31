"""An interactive video world with **Hy-Nomas** on one private NVIDIA GPU, through a custom backend.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`hy-nomas-world`, published with publish.py) because no built-in
backend serves Hy-Nomas, and each call is ONE turn of a world:

  1. `start` opens a world from a structured description (style, subject,
     scene, action) and returns its first 5-second segment,
  2. `continue` steers the SAME world with a new instruction; the backend
     keeps the model's key/value cache between calls, so the fox, the forest
     and the light carry over,
  3. a third turn adds one more change; the script joins the three
     segments into one MP4 of the whole world (ffmpeg on your machine).

Each MP4 lands in ./outputs. One segment per call keeps every request under
the gateway's synchronous limit.

Transport: the custom backend answers the OpenAI chat route
(`/v1/chat/completions`); the user message is the turn as JSON and the
message content of the answer is the backend's JSON (base64 MP4 + timings).
The script turns the workload's `chat` endpoint on once with your dashboard
session token (or flip "Chat" in Dashboard -> Endpoint settings).

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = os.environ.get("NOMAS_WORKLOAD_SLUG", "hy-nomas")
BACKEND_SLUG = os.environ.get("NOMAS_BACKEND_SLUG", "hy-nomas-world")
OUT_DIR = Path(__file__).resolve().parent / "outputs"

# The authors' own example world (examples/prompts.json in the model repo), turn by turn.
TURNS = [
    {"step": "start", "style": "Cinematic fantasy", "perspective": "Third-person view",
     "subjects": [{"subject_id": "fox", "display_name": "a small orange fox",
                   "subject_attribute": "Orange fur, white chest, and a fluffy tail."}],
     "scene": "A small orange fox stands on a stone path in a luminous forest. Blue fireflies drift between the trees.",
     "action": "The fox walks slowly along the path.", "seed": 42},
    {"step": "continue", "change": "Glowing blue flowers open beside the path as the fox continues forward.",
     "action": "The fox passes the flowers and looks around."},
    {"step": "continue", "change": "Snow begins to fall and the forest floor slowly turns white.",
     "action": "The fox stops and looks up at the falling snow."},
]


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


def turn(url: str, key: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """One world turn: the turn JSON goes as the user message."""
    req = {"model": "default", "messages": [{"role": "user", "content": json.dumps(body)}]}
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=req, timeout=600)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return json.loads(r.json()["choices"][0]["message"]["content"])


def main() -> int:
    _load_dotenv()
    mgmt = ManagementClient.from_env()
    # Reaches the backend: env vars for its process, and how long the worker waits
    # for setup() (venv with torch 2.10 + 34 GB of weights + load) and for one segment.
    backend_env = {k: os.environ[k] for k in ("NOMAS_VRAM_LIMIT_GIB", "NOMAS_HEIGHT", "NOMAS_WIDTH",
                                              "NOMAS_EXTRA_OVERRIDES") if os.environ.get(k)}
    config = {"ready_timeout_secs": 3600, "request_timeout_secs": 600, "env": backend_env}
    spec = WorkloadSpec(
        name="Hy-Nomas world model",
        slug=SLUG,
        model="tencent/Hy-Nomas",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        # Must match the backend package, which the SDK can only tag text2image today
        # (no text2video custom backends yet); the output is still MP4 video.
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
        print("waiting for the worker to become ready (first start installs torch 2.10 and downloads ~34 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.time() - t0:.1f} s")

        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/chat/completions"
        OUT_DIR.mkdir(exist_ok=True)
        session = None
        for i, body in enumerate(TURNS, 1):
            if session:
                body = {**body, "session_id": session}
            t = time.time()
            out = turn(url, key, body)
            session = out["session_id"]
            name = OUT_DIR / f"segment-{i}.mp4"
            name.write_bytes(base64.b64decode(out["video_b64"]))
            tm = out["timings"]
            print(f"turn {i} ({body['step']}): {name.name} {out['width']}x{out['height']}, {out['frames']} frames "
                  f"({out['seconds']:.2f} s of video), GPU {tm['gpu_s']} s, peak VRAM {tm['peak_vram_mib']} MiB, "
                  f"round trip {time.time() - t:.1f} s")
        # One MP4 per turn; join them into the whole world on your side.
        world = OUT_DIR / "world.mp4"
        parts = sorted(OUT_DIR.glob("segment-*.mp4"))
        lst = OUT_DIR / "segments.txt"
        lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-c", "copy", str(world)], check=True)
            print(f"wrote {world.name}: the whole world, {out['world_seconds']} s")
        else:
            print(f"install ffmpeg to join the segments: ffmpeg -f concat -i {lst} -c copy {world}")
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

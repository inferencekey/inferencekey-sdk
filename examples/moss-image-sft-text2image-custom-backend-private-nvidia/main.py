"""Generate images with **MOSS-Image-SFT** on a private NVIDIA GPU, through a custom backend.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`moss-image-text2image`, published with publish.py) because no
built-in backend serves MOSS-Image's diffusers custom pipeline.

Three calls show what the model is for:
1. a poster with several lines of text, the release recipe (50 steps, CFG 3);
2. the same prompt in "turbo" mode (community 8-step DMD LoRA, no CFG);
3. a JSON layout prompt that places the text and the object with bounding boxes.

The data plane has no typed image helper yet, so the script POSTs an
OpenAI-style request to `/v1/images/generations` with the data-plane key
(`ik_live_`) and writes the PNGs to ./outputs.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py
"""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = "moss-image-sft"
BACKEND_SLUG = "moss-image-text2image"  # the slug publish.py registered
OUT_DIR = Path(__file__).resolve().parent / "outputs"

POSTER = (
    "A clean conference poster on a white wall. Large bold title at the top: "
    "\"OPEN WEIGHTS DAY\". Below it, three lines in a smaller dark font:\n"
    "\"10:00 Keynote\"\n\"12:30 Lunch\"\n\"15:00 Workshops\"\n"
    "A small orange logo in the bottom right corner."
)
# Layout prompt: bbox is [y_min, x_min, y_max, x_max] in pixels, origin at the
# canvas centre (the user guide's "yxyx" + "abs_centralized").
LAYOUT = {
    "summary": {
        "meta": ["poster"],
        "canvas": "single",
        "caption": "A minimalist travel poster with a cream background, a blue "
        "mountain illustration, and the title EXPLORE at the top.",
    },
    "items": [
        {
            "label": "text",
            "caption": "Large dark-blue uppercase title centered at the top.",
            "text": "EXPLORE",
            "bbox": [-416, -384, -288, 384],
        },
        {
            "label": "object",
            "caption": "A flat blue mountain illustration in the lower center.",
            "bbox": [-192, -288, 384, 288],
        },
    ],
    "bbox_coord_space": "abs_centralized",
    "bbox_format": "yxyx",
}
CALLS = [
    ("poster-sft", {"prompt": POSTER, "mode": "sft"}),
    ("poster-turbo", {"prompt": POSTER, "mode": "turbo"}),
    ("layout-sft", {"prompt": LAYOUT, "mode": "sft"}),
]


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    _load_dotenv()
    mgmt = ManagementClient.from_env()
    worker_id = os.environ["IK_WORKER_ID"]

    spec = WorkloadSpec(
        name="MOSS-Image-SFT text-to-image",
        slug=SLUG,
        model="OpenMOSS-Team/MOSS-Image-SFT",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2image",
        worker_id=worker_id,
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
    )

    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # First start: the worker builds the backend venv (torch, diffusers,
        # transformers), downloads about 28 GB (model + LoRA) and loads it.
        print("waiting for the worker to become ready (first start downloads ~28 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.time() - t0:.0f} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        project = os.environ["INFERENCEKEY_PROJECT"]
        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/images/generations"
        headers = {"Authorization": f"Bearer {os.environ['INFERENCEKEY_API_KEY']}"}

        OUT_DIR.mkdir(exist_ok=True)
        for name, extra in CALLS:
            body = {"model": "default", "size": "1024x1024", "seed": 42, "n": 1, **extra}
            t = time.time()
            resp = requests.post(url, headers=headers, json=body, timeout=900)
            resp.raise_for_status()
            out = resp.json()
            path = OUT_DIR / f"{name}.png"
            path.write_bytes(base64.b64decode(out["data"][0]["b64_json"]))
            print(
                f"wrote {path.name}: {out.get('steps')} steps, round trip {time.time() - t:.1f} s, "
                f"GPU {out.get('seconds')} s, VRAM peak {out.get('vram_peak_gib')} GiB"
            )
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"image request failed: {e}: {e.response.text[:200]}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the
        # workload (useful to try more prompts from the dashboard playground).
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

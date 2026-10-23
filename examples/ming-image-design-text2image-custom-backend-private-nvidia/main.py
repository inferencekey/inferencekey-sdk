"""Generate design images with **Ming-Image-0.1-Design** on a private NVIDIA GPU, through a custom backend.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`ming-image-design-text2image`, published with publish.py) because
no built-in backend serves Ming-Image's own inference code.

Three calls show what the model is for:
1. a landing-page UI mockup with real copy, at 1024 x 1024 (the fast bucket);
2. an infographic poster at 2048 x 2048 (the authors' recommended size);
3. a transparent-background asset: `"background": "transparent"` puts the
   authors' RGBA phrase in front of the prompt and the PNG comes back with
   alpha 0 around the subject.

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

SLUG = "ming-image-design"
BACKEND_SLUG = "ming-image-design-text2image"  # the slug publish.py registered
OUT_DIR = Path(__file__).resolve().parent / "outputs"

UI = (
    "Landing page for a productivity app called Flow. Clean minimal design, white background, "
    "generous whitespace. Top nav with the wordmark 'Flow' on the left and links Product, Pricing, "
    "Docs on the right. Centered hero with a bold headline 'Focus without the noise' and a subtle "
    "gray subheading. A single purple call-to-action button labeled 'Start free'. Below the hero, "
    "three feature cards with small icons, soft rounded corners and soft shadows. Modern sans-serif "
    "typography."
)
INFOGRAPHIC = (
    "An editorial infographic poster titled \"HOW A GPU SERVES AN IMAGE\" in bold condensed "
    "uppercase at the top. Four numbered steps in a vertical timeline with flat icons: "
    "\"1 Request arrives\", \"2 Text is encoded\", \"3 Twelve denoising steps\", \"4 PNG goes back\". "
    "A footer line reads \"One GPU. One model. No cloud API.\" Navy, coral and cream palette, "
    "clean grid, generous margins."
)
STICKER = (
    "A round flat-vector sticker of a smiling orange fox holding a paint brush, thick white "
    "outline, the word \"DESIGN\" on a small blue ribbon below it."
)
CALLS = [
    ("ui-1024", {"prompt": UI, "size": "1024x1024"}),
    ("infographic-2048", {"prompt": INFOGRAPHIC, "size": "2048x2048"}),
    ("sticker-rgba-1024", {"prompt": STICKER, "size": "1024x1024", "background": "transparent"}),
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
    # Reaches the backend: env vars for its process and how long the worker
    # waits for setup() (venv + 53 GB download + load can exceed the 5 min default).
    backend_env = {k: os.environ[k] for k in ("MING_MEMORY_MODE", "MING_VRAM_LIMIT_GIB") if os.environ.get(k)}
    config = {"ready_timeout_secs": 2400, "request_timeout_secs": 900, "env": backend_env}

    spec = WorkloadSpec(
        name="Ming-Image-0.1-Design",
        slug=SLUG,
        model="inclusionAI/Ming-Image-0.1-Design",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2image",
        worker_id=worker_id,
        config=config,
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
        execution_policy_config={"replicas": 1},
    )

    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # First start: the worker builds the backend venv (torch 2.5, transformers,
        # diffusers), downloads about 53 GB of weights and loads them.
        print("waiting for the worker to become ready (first start downloads ~53 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.time() - t0:.0f} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        project = os.environ["INFERENCEKEY_PROJECT"]
        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/images/generations"
        headers = {"Authorization": f"Bearer {os.environ['INFERENCEKEY_API_KEY']}"}

        OUT_DIR.mkdir(exist_ok=True)
        for name, extra in CALLS:
            body = {"model": "default", "seed": 42, "n": 1, **extra}
            t = time.time()
            resp = requests.post(url, headers=headers, json=body, timeout=900)
            resp.raise_for_status()
            out = resp.json()
            ext = "png" if out.get("output_format", "png") == "png" else "webp"
            path = OUT_DIR / f"{name}.{ext}"
            path.write_bytes(base64.b64decode(out["data"][0]["b64_json"]))
            print(
                f"wrote {path.name}: {out.get('width')}x{out.get('height')} {out.get('mode')}, "
                f"{out.get('steps')} steps, round trip {time.time() - t:.1f} s, GPU {out.get('seconds')} s, "
                f"VRAM peak {out.get('vram_peak_gib')} GiB, transparent pixels {out.get('transparent_fraction')}"
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

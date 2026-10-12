"""Generate images with **Iris-3B Turbo** on a private NVIDIA GPU, through a custom backend.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`iris3b-text2image`, published with publish.py) because no built-in
backend serves Iris-3B. It is a pixel-space model: no VAE, no latent, the
network paints every pixel.

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

VARIANT = os.environ.get("IRIS_VARIANT", "turbo")  # "turbo" (4 steps) or "base" (100 steps)
SLUG = f"iris3b-{VARIANT}"
# The backend slug publish.py registered for this variant.
BACKEND_SLUG = "iris3b-text2image" if VARIANT == "turbo" else "iris3b-base-text2image"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

PROMPTS = [
    (
        "Macro photograph of a dew-covered spider web at sunrise, every droplet "
        "refracting the orange sky, shallow depth of field",
        "1024x1024",
    ),
    (
        "A vintage enamel sign on a brick wall that reads \"OPEN LATE\" in bold "
        "red letters, afternoon light",
        "1344x752",
    ),
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
        name=f"Iris-3B {VARIANT.capitalize()} text-to-image",
        slug=SLUG,
        model="speridlabs/iris-3b",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2image",
        worker_id=worker_id,
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
        execution_policy_config={"replicas": 1},
    )

    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # First start: the worker installs the backend venv (torch, transformers),
        # downloads ~6 GB of Turbo weights plus the ~9 GB Qwen3-VL text encoder,
        # and loads both. Give it room.
        print("waiting for the worker to become ready (first start downloads ~15 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        print(f"ready after {time.time() - t0:.0f} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        project = os.environ["INFERENCEKEY_PROJECT"]
        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/images/generations"
        headers = {"Authorization": f"Bearer {os.environ['INFERENCEKEY_API_KEY']}"}

        OUT_DIR.mkdir(exist_ok=True)
        for i, (prompt, size) in enumerate(PROMPTS):
            body = {"model": "default", "prompt": prompt, "size": size, "seed": 7 + i, "n": 1}
            t = time.time()
            resp = requests.post(url, headers=headers, json=body, timeout=600)
            resp.raise_for_status()
            out = resp.json()
            png = base64.b64decode(out["data"][0]["b64_json"])
            path = OUT_DIR / f"{SLUG}-{i}.png"
            path.write_bytes(png)
            gpu_s = out.get("seconds")
            print(f"wrote {path.name}: {size}, round trip {time.time() - t:.2f} s, GPU {gpu_s} s")
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

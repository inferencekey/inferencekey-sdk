"""Generate images in **one or two steps** with NitroFusion2 on a private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`nitrofusion2-text2image`, published with publish.py) because no
built-in backend serves the Z-Image pipeline.

Three calls show what the model is for:
1. the authors' own sign prompt in ONE step;
2. a chalkboard menu with exact words and prices in ONE step;
3. the same menu in TWO steps with the authors' de-distillation LoRA, so you
   can compare both modes on the same seed.

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

SLUG = "nitrofusion2"
BACKEND_SLUG = "nitrofusion2-text2image"  # the slug publish.py registered
OUT_DIR = Path(__file__).resolve().parent / "outputs"

SIGN = "a cat holding a sign that says hello world"
MENU = (
    "A cafe chalkboard menu on a brick wall, hand-lettered in white chalk. Title \"DAILY BREW\". "
    "Three lines: \"FLAT WHITE 3.20\", \"CROISSANT 2.80\", \"SOURDOUGH 4.50\". "
    "A small chalk drawing of a coffee cup in the corner, warm morning light."
)

CALLS = [
    ("sign-1step", {"prompt": SIGN, "steps": 1}),
    ("menu-1step", {"prompt": MENU, "steps": 1}),
    ("menu-2step", {"prompt": MENU, "steps": 2}),
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
    # waits for setup() (venv + about 21 GB of downloads + load).
    keys = ("NF2_MEMORY_MODE", "NF2_TRANSFORMER", "NF2_VRAM_LIMIT_GIB")
    backend_env = {k: os.environ[k] for k in keys if os.environ.get(k)}
    config = {"ready_timeout_secs": 2400, "request_timeout_secs": 600, "env": backend_env}
    spec = WorkloadSpec(
        name="NitroFusion2 text-to-image",
        slug=SLUG,
        model="ChenDY/NitroFusion2",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2image",
        worker_id=worker_id,
        config=config,
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
    )

    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        # First start: the worker builds the backend venv (torch, diffusers),
        # downloads the NitroFusion2 transformer and LoRA (12.7 GB) plus the
        # Z-Image-Turbo text encoder and VAE (8.2 GB), and loads them.
        print("waiting for the worker to become ready (first start downloads about 21 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        print(f"ready after {time.time() - t0:.0f} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        project = os.environ["INFERENCEKEY_PROJECT"]
        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/images/generations"
        headers = {"Authorization": f"Bearer {os.environ['INFERENCEKEY_API_KEY']}"}

        OUT_DIR.mkdir(exist_ok=True)
        for name, extra in CALLS:
            body = {"model": "default", "seed": 42, "n": 1, "size": "1024x1024", **extra}
            t = time.time()
            resp = requests.post(url, headers=headers, json=body, timeout=600)
            resp.raise_for_status()
            out = resp.json()
            path = OUT_DIR / f"{name}.png"
            path.write_bytes(base64.b64decode(out["data"][0]["b64_json"]))
            print(
                f"wrote {path.name}: {out.get('width')}x{out.get('height')}, {out.get('steps')} step(s), "
                f"round trip {time.time() - t:.2f} s, GPU {out.get('seconds')} s "
                f"(text encoder {out.get('encode_seconds')} s, VAE {out.get('vae_seconds')} s), "
                f"VRAM peak {out.get('vram_peak_gib')} GiB"
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

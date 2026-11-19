"""Generate and edit images with **MiniMax-H3-Image** on one private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`minimax-h3-image-comfyui`, published with publish.py) that runs
ComfyUI headless with Comfy-Org's quantized repack of the model (4-bit
diffusion transformer, NVFP4 text encoder), because the official BF16
checkpoint needs about 135 GB.

Three calls show what the model is for:
1. the poster prompt of the official ComfyUI template at 2048 x 2048 (4 MP);
2. the same prompt and seed at 4096 x 4096 (16 MP, the size the authors tune for);
3. an edit: the first image goes back in as `<Picture 1>` with an instruction.

The data plane has no typed image helper yet, so the script POSTs an
OpenAI-style request to `/v1/images/generations` with the data-plane key
(`ik_live_`) and writes the images to ./outputs.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = "minimax-h3-image"
BACKEND_SLUG = "minimax-h3-image-comfyui"  # the slug publish.py registered
OUT_DIR = Path(__file__).resolve().parent / "outputs"

# The prompt of Comfy-Org's text-to-image template for this model. The model
# expects a dense English description of the whole image.
POSTER = (
    "The image is a cheerful toy-packaging poster built around a handmade plush fox floating above a summer "
    "meadow, drawn in a warm, fairly saturated palette: clear sky blue deepening toward the top, rich "
    "burnt-sienna and terracotta plush for the fox, warm cream, fresh sage green in the meadow, cornflower and "
    "cherry-red accents on the small props, and buttercup yellow in the lettering, with bright afternoon sun "
    "from the upper left and crisp short shadows. The fox is the hero, sewn from long-staple mohair-type plush "
    "in warm burnt sienna and hand-stuffed so its silhouette is slightly lumpy; the pile is long and irregular, "
    "single fibres catching the light along its outline as a bright hairy halo against the sky, with cream "
    "muzzle and sock paws. Across the top of the frame the word \"MINIMAX\" is inflatable and rounded in thick "
    "felted wool; directly beneath, smaller and centred, \"H3\" sits between two small dots. Along the bottom, "
    "a heavy bold sans-serif logotype reads \"TEXT TO IMAGE\" in saturated cornflower blue. All lettering is "
    "English only, with no Chinese characters."
)
EDIT = (
    "Replace the plush fox in <Picture 1> with a plush grey rabbit with long floppy ears, sewn from the same "
    "long mohair plush, in the same pose and place. Keep the meadow, the sky and all the lettering exactly as "
    "they are."
)


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def _reference(path: Path, side: int = 1024) -> str:
    """A generated image as a small base64 JPEG, to send back as <Picture 1>."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> int:
    _load_dotenv()
    mgmt = ManagementClient.from_env()
    worker_id = os.environ["IK_WORKER_ID"]

    # Reaches the backend: env vars for its process, how long the worker waits
    # for setup() (venv + ComfyUI + about 29 GB of downloads + a warm-up) and
    # how long one request may take (a 16 MP image is slow on a small card).
    keys = ("H3_VRAM_LIMIT_GIB", "H3_COMFY_ARGS", "H3_DIT", "H3_TEXT_ENCODER", "H3_WARMUP")
    backend_env = {k: os.environ[k] for k in keys if os.environ.get(k)}
    config = {"ready_timeout_secs": 3600, "request_timeout_secs": 900, "env": backend_env}
    spec = WorkloadSpec(
        name="MiniMax-H3-Image",
        slug=SLUG,
        model="MiniMaxAI/MiniMax-H3-Image",
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
        print("waiting for the worker to become ready (first start downloads about 29 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.time() - t0:.0f} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        project = os.environ["INFERENCEKEY_PROJECT"]
        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/images/generations"
        headers = {"Authorization": f"Bearer {os.environ['INFERENCEKEY_API_KEY']}"}
        OUT_DIR.mkdir(exist_ok=True)

        def call(name: str, extra: dict) -> Path:
            body = {"model": "default", "seed": 42, "n": 1, **extra}
            t = time.time()
            resp = requests.post(url, headers=headers, json=body, timeout=900)
            resp.raise_for_status()
            out = resp.json()
            path = OUT_DIR / f"{name}.{'jpg' if out.get('output_format') == 'jpeg' else out.get('output_format')}"
            path.write_bytes(base64.b64decode(out["data"][0]["b64_json"]))
            print(
                f"wrote {path.name}: {out.get('width')}x{out.get('height')}, round trip {time.time() - t:.1f} s, "
                f"graph {out.get('seconds')} s, stages {out.get('stage_seconds')}, "
                f"card peak {out.get('vram_peak_mib')} MiB, {out.get('output_format')} q{out.get('output_quality')}"
            )
            return path

        first = call("poster-4mp", {"prompt": POSTER, "size": "2048x2048"})
        call("poster-16mp", {"prompt": POSTER, "size": "4096x4096"})
        call("edit-4mp", {"prompt": EDIT, "size": "2048x2048", "image": [_reference(first)]})
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"image request failed: {e}: {e.response.text[:300]}", file=sys.stderr)
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

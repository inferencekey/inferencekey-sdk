"""Turn one picture of an object into a 3D Gaussian splat with **Idol** on one private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`idol-image-to-3d`, published with publish.py) that wraps the
authors' `IdolPipeline`, and the output is a 3D file, not text or pixels.

Two calls on the same picture show the trade-off of the model card:
1. Idol, 50 guided steps (the quality reference);
2. Idol Turbo, 5 steps without guidance (the distilled phase LoRAs).

Input: your own picture (`python main.py photo.jpg --real` for a photograph,
`python main.py render.png` for a render or a generated image) or, with no
argument, the authors' backpack example (a generated picture the Turbo LoRAs
never saw in training). Output in ./outputs: `<name>-base.ply` and
`<name>-turbo.ply` (standard 3DGS PLY: open them in any Gaussian splat viewer,
for example https://superspl.at/editor) and the image the model saw.

Transport: the workload answers on the OpenAI images route
(`/v1/images/generations`); the answer carries the image the model saw in
`data[0].b64_json` and the `.ply` in parts of 2 MB (`ply_b64`), because one
result is limited to an 8 MB message. The script fetches the other parts with
`{"ply_id": ..., "part": n}` and checks the SHA-256.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py [picture] [--real]
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import time
from pathlib import Path

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = "idol"
BACKEND_SLUG = "idol-image-to-3d"  # the slug publish.py registered
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def _picture(path: Path, side: int = 1536) -> str:
    """Your picture as a base64 JPEG (or PNG when it has a transparent background), at most `side` px."""
    from PIL import Image

    img = Image.open(path)
    img.thumbnail((side, side))
    buf = io.BytesIO()
    if img.mode == "RGBA":
        img.save(buf, format="PNG")
    else:
        img.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate(url: str, key: str, body: dict, out: Path) -> dict:
    """One object: generate, then fetch the remaining parts of the .ply and check it."""
    headers = {"Authorization": f"Bearer {key}"}
    t = time.time()
    r = requests.post(url, headers=headers, json={"model": "default", **body}, timeout=900)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    first = r.json()
    generated = time.time() - t
    parts = [base64.b64decode(first["ply_b64"])]
    for n in range(1, first["parts"]):
        r = requests.post(url, headers=headers, timeout=300,
                          json={"model": "default", "ply_id": first["ply_id"], "part": n})
        r.raise_for_status()
        parts.append(base64.b64decode(r.json()["ply_b64"]))
    ply = b"".join(parts)
    if hashlib.sha256(ply).hexdigest() != first["ply_sha256"]:
        raise RuntimeError("the .ply arrived corrupted (SHA-256 mismatch)")
    out.write_bytes(ply)
    out.with_name(out.stem + "-input.jpg").write_bytes(base64.b64decode(first["data"][0]["b64_json"]))
    print(
        f"wrote {out.name}: {first['num_gaussians']:,} Gaussians, {len(ply) / 1e6:.1f} MB; "
        f"{'turbo ' if first['turbo'] else ''}{first['steps']} steps; model {first['seconds']} s "
        f"{first['stage_seconds']}, round trip {generated:.1f} s + {first['parts'] - 1} more parts in "
        f"{time.time() - t - generated:.1f} s; peak VRAM {first['peak_reserved_mib']} MiB reserved by PyTorch, "
        f"{first['card_peak_mib']} MiB on the card"
    )
    return first


def main() -> int:
    _load_dotenv()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    real = "--real" in sys.argv
    if args:
        picture = Path(args[0])
        source = {"image": _picture(picture), "mode": "real" if real else "synthetic"}
        name = picture.stem
    else:
        source = {"example": "obj-backpack", "mode": "synthetic"}
        name = "backpack"

    mgmt = ManagementClient.from_env()
    # Reaches the backend: env vars for its process, how long the worker waits
    # for setup() (venv + about 5 GB of weights + compiling the sampler) and
    # how long one request may take.
    keys = ("IDOL_ENCODERS", "IDOL_TURBO", "IDOL_OPTIMIZE", "IDOL_VRAM_LIMIT_GIB", "IDOL_VAE_TILING",
            "IDOL_OFFLOAD_ENCODERS", "IDOL_WARMUP", "HF_TOKEN")
    backend_env = {k: os.environ[k] for k in keys if os.environ.get(k)}
    config = {"ready_timeout_secs": 2400, "request_timeout_secs": 900, "env": backend_env}
    spec = WorkloadSpec(
        name="Idol image to 3D",
        slug=SLUG,
        model="speridlabs/idol",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        # Must match the backend package; the output is a .ply, served on the images route.
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
        print("waiting for the worker to become ready (first start downloads about 5 GB and compiles)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        print(f"ready after {time.time() - t0:.0f} s")

        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/images/generations"
        OUT_DIR.mkdir(exist_ok=True)
        generate(url, key, {**source, "seed": 42}, OUT_DIR / f"{name}-base.ply")
        generate(url, key, {**source, "seed": 42, "turbo": True}, OUT_DIR / f"{name}-turbo.ply")
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (requests.RequestException, RuntimeError) as e:
        print(f"request failed: {e}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the
        # workload (useful to try more pictures from the dashboard).
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

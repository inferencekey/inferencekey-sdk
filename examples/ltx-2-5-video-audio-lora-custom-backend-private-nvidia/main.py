"""Generate a video WITH its soundtrack from one prompt with **LTX-2.5** (Lightricks) on one private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`ltx-2-5`, published with publish.py) that runs the authors' own
`DistilledPipeline` (github.com/Lightricks/LTX-2), and a clip takes minutes,
so the script polls a `clip_id` and then fetches the MP4 in 2 MB parts.

LTX-2.5 makes the picture and the sound (speech, foley, ambience) together, in
one pass: put the dialogue and the sounds in the prompt.

Output in ./outputs: `<name>.mp4` (H.264 + AAC) and a `.json` with the timings
and GPU memory the backend measured.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py                              # text-to-video, 1536 x 1024, 121 frames (5 s at 24 fps)
      python main.py --image shot.jpg --lora cinemagraph --prompt "CINEMAGRAPH_MOTION, ..."   # an official LoRA
      python main.py --width 768 --height 512 --frames 49     # faster preview
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

SLUG = os.environ.get("IK_WORKLOAD_SLUG", "ltx-2-5-video")
BACKEND_SLUG = os.environ.get("IK_BACKEND_SLUG", "ltx-2-5")  # the slug publish.py registered
OUT_DIR = Path(__file__).resolve().parent / "outputs"

PROMPT = (
    "A medium close-up of a barista in a small sunlit coffee shop, a woman in her thirties with dark curly hair "
    "tied back and a green canvas apron, standing behind a wooden counter. She pours steamed milk into a white "
    "ceramic cup, drawing a leaf pattern in the foam, then slides the cup toward the camera and smiles. She says "
    "in a warm, clear voice: 'Fresh roast, every morning.' We hear the hiss of the espresso machine, the clink "
    "of the cup on its saucer and soft cafe chatter in the background. Warm morning light comes through the "
    "window behind her, shallow depth of field, the camera slowly pushes in."
)


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def _post(url: str, key: str, body: dict) -> dict:
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json={"model": "default", **body}, timeout=330)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def generate(url: str, key: str, body: dict, out: Path) -> dict:
    """Start one clip, poll until it is done, fetch every part of the MP4 and check it."""
    t = time.time()
    # One request waits at most about 270 s (the platform's limit is 300 s);
    # a longer clip comes back as {"status": "running", "clip_id": ...}.
    answer = _post(url, key, body)
    while answer.get("status") == "running":
        print(f"  still generating ({answer['elapsed_s']:.0f} s)...")
        answer = _post(url, key, {"clip_id": answer["clip_id"]})
    if answer.get("status") != "done":
        raise RuntimeError(f"clip {answer.get('clip_id')} failed: {answer.get('error')}")
    elapsed = time.time() - t
    parts = [base64.b64decode(answer["mp4_b64"])]
    for n in range(1, answer["parts"]):
        parts.append(base64.b64decode(_post(url, key, {"clip_id": answer["clip_id"], "part": n})["mp4_b64"]))
    mp4 = b"".join(parts)
    if hashlib.sha256(mp4).hexdigest() != answer["mp4_sha256"]:
        raise RuntimeError("the MP4 arrived corrupted (SHA-256 mismatch)")
    out.write_bytes(mp4)
    meta = {k: v for k, v in answer.items() if k not in ("mp4_b64", "data")}
    meta["round_trip_s"] = round(elapsed, 2)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    audio = answer.get("audio") or {}
    print(
        f"wrote {out.name}: {answer.get('width')} x {answer.get('height')}, {answer.get('frames')} frames "
        f"({answer.get('seconds_of_video')} s at {answer.get('fps')} fps), audio {audio.get('codec')} "
        f"{audio.get('sample_rate')} Hz; generation {answer['wall_s']} s on the GPU (models load included), "
        f"round trip {elapsed:.0f} s; peak {answer.get('peak_reserved_mib')} MiB reserved by PyTorch, "
        f"{answer.get('card_peak_mib')} MiB on the card ({answer.get('card')})"
    )
    return answer


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--image", help="first frame (PNG/JPEG): image-to-video")
    ap.add_argument("--lora", help="an official LoRA the backend downloaded (cinemagraph)")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--frames", type=int, default=121, help="8k+1: 49, 97, 121...")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offload", default="none", choices=["none", "cpu"], help="cpu if your card runs out of memory")
    ap.add_argument("--name", default="ltx25")
    args = ap.parse_args()

    mgmt = ManagementClient.from_env()
    # Reaches the backend: env vars for its process, how long the worker waits
    # for setup() (venv + about 66 GiB of weights) and how long one request may take.
    keys = ("HF_TOKEN", "LTX_HOME", "LTX_SOURCE", "LTX_DASH_WIDTH", "LTX_DASH_HEIGHT", "LTX_DASH_FRAMES")
    backend_env = {k: os.environ[k] for k in keys if os.environ.get(k)}
    config = {"ready_timeout_secs": 3600, "request_timeout_secs": 330, "env": backend_env}
    spec = WorkloadSpec(
        name="LTX-2.5 video with audio",
        slug=SLUG,
        model="Lightricks/LTX-2.5",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        # Must match the backend package; clips are served on the images route.
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
        print("waiting for the worker to become ready (the first start downloads about 66 GiB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.time() - t0:.0f} s")

        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/images/generations"
        OUT_DIR.mkdir(exist_ok=True)
        body = {"prompt": args.prompt, "seed": args.seed, "width": args.width, "height": args.height,
                "num_frames": args.frames, "offload": args.offload}
        if args.image:
            raw = Path(args.image).read_bytes()
            if len(raw) > 1_200_000:
                # Request bodies are limited to 2 MB and base64 adds a third:
                # send a big PNG as a JPEG instead (needs Pillow).
                import io

                from PIL import Image

                buf = io.BytesIO()
                Image.open(io.BytesIO(raw)).convert("RGB").save(buf, format="JPEG", quality=92)
                raw = buf.getvalue()
            body["image_b64"] = base64.b64encode(raw).decode("ascii")
        if args.lora:
            body.update(lora=args.lora, lora_strength=args.lora_strength)
        generate(url, key, body, OUT_DIR / f"{args.name}.mp4")
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (requests.RequestException, RuntimeError, OSError) as e:
        print(f"request failed: {e}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the
        # workload (useful to try more clips from the dashboard).
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

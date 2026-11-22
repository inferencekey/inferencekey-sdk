"""Re-render a low-poly game video as detailed video with **Matrix-Game 4.0** (Skywork AI) on one private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`matrix-game-4`, published with publish.py) that runs the authors'
own inference source (not Diffusers), and the input is a video, so the script
uploads it in 1 MB parts and then asks for a render.

Matrix-Game 4.0 never invents gameplay: a real game engine runs the game and
draws plain low-poly frames; the model repaints that stream, chunk by chunk,
into detailed 480 x 864 video. This example serves the Medieval Castle scene.

Input: a clean low-poly RGB video of the castle scene, without HUD. Record one
with the authors' game (github.com/SkyworkAI/Matrix-Game, Matrix-Game-4/:
`bash scripts/build_game.sh && bash scripts/play.sh --scene castle`, F9 starts
and stops a take; `reference.mkv` in the take folder is the clean stream).

Output in ./outputs: `<name>-<preset>.mp4` (864 x 480, the input's frame rate)
and the timings and GPU memory the backend measured.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py --input reference.mkv [--preset realtime|quality|rtx5090|rtx5080] [--seconds 10]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = "matrix-game-castle"
BACKEND_SLUG = "matrix-game-4"  # the slug publish.py registered
OUT_DIR = Path(__file__).resolve().parent / "outputs"
PART = 1_000_000  # request bodies are limited to 2 MB, and base64 adds a third


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


def upload(url: str, key: str, video: Path) -> str:
    """Send the input video in 1 MB parts; the last part answers its input_id."""
    blob = video.read_bytes()
    parts = max(1, -(-len(blob) // PART))
    uid = uuid.uuid4().hex
    answer: dict = {}
    for n in range(parts):
        chunk = base64.b64encode(blob[n * PART:(n + 1) * PART]).decode("ascii")
        answer = _post(url, key, {"upload_id": uid, "part": n, "parts": parts, "chunk_b64": chunk})
    if answer.get("status") != "uploaded":
        raise RuntimeError(f"upload did not finish: {answer}")
    print(f"uploaded {video.name} ({len(blob) / 1e6:.1f} MB in {parts} parts): input_id {answer['input_id']}")
    return answer["input_id"]


def render(url: str, key: str, body: dict, out: Path) -> dict:
    """Start one render, poll until it is done, fetch every part of the MP4 and check it."""
    t = time.time()
    # One request waits at most about 270 s (the platform's limit is 300 s);
    # a longer render comes back as {"status": "running", "clip_id": ...}.
    answer = _post(url, key, body)
    while answer.get("status") == "running":
        print(f"  still rendering ({answer['elapsed_s']:.0f} s)...")
        answer = _post(url, key, {"clip_id": answer["clip_id"]})
    if answer.get("status") != "done":
        raise RuntimeError(f"render {answer.get('clip_id')} failed: {answer.get('error')}")
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
    tm = answer["timings"]
    print(
        f"wrote {out.name}: {answer['frames']} frames ({answer['seconds_of_video']} s at {answer['fps']:g} fps), "
        f"preset {answer['preset']}: model {tm['model_s']} s = {tm['model_fps']} fps "
        f"({tm['steady_model_fps']} fps after the first chunk, median chunk {tm['chunk_ms_median']} ms for 4 frames), "
        f"round trip {elapsed:.0f} s; peak VRAM {answer['peak_reserved_mib']} MiB reserved by PyTorch, "
        f"{answer['card_peak_mib']} MiB on the card ({answer['card']})"
    )
    return answer


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="clean low-poly RGB video of the castle scene (no HUD)")
    ap.add_argument("--preset", default="realtime", choices=["realtime", "quality", "rtx5090", "rtx5080", "rtx5070ti"])
    ap.add_argument("--seconds", type=float, help="render only the first N seconds")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mgmt = ManagementClient.from_env()
    # Reaches the backend: env vars for its process, how long the worker waits
    # for setup() (venv + about 25 GB of resources + loading and warming up the
    # engine) and how long one request may take.
    keys = ("MG4_SCENE", "MG4_PRESET", "MG4_HOME", "HF_TOKEN")
    backend_env = {k: os.environ[k] for k in keys if os.environ.get(k)}
    backend_env.setdefault("MG4_PRESET", args.preset)
    config = {"ready_timeout_secs": 3600, "request_timeout_secs": 330, "env": backend_env}
    spec = WorkloadSpec(
        name="Matrix-Game 4.0 castle renderer",
        slug=SLUG,
        model="Skywork/Matrix-Game-4.0",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        # Must match the backend package; renders are served on the images route.
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
        print("waiting for the worker to become ready (the first start downloads about 25 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.time() - t0:.0f} s")

        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/images/generations"
        OUT_DIR.mkdir(exist_ok=True)
        video = Path(args.input)
        input_id = upload(url, key, video)
        body = {"input_id": input_id, "preset": args.preset, "seed": args.seed}
        if args.seconds:
            body["seconds"] = args.seconds
        render(url, key, body, OUT_DIR / f"{video.stem}-{args.preset}.mp4")
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (requests.RequestException, RuntimeError, OSError) as e:
        print(f"request failed: {e}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the
        # workload (useful to try more renders from the dashboard).
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Detect, count, point, read and cut out objects with **WeVisOne-VL-4B** on a private NVIDIA GPU, through a custom backend.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`wevisone-vl-perception`, published with publish.py) because no
built-in backend runs WeVisOne-VL's 9.6M-parameter pixel decoder, the piece
that turns the model's answer into masks and alpha mattes.

One model, several tasks, each a single request:
detection, counting, referring (with a target that is NOT in the image, which
the model must reject), pointing, OCR, segmentation and alpha matting.

The data plane has no typed vision helper yet, so the script POSTs JSON to
`/v1/images/generations` with the data-plane key (`ik_live_`):
`{"task": ..., "query": ..., "image": <https URL or base64>}`. The answer is the
OpenAI images shape (data[0] = annotated image, data[1] = RGBA cutout for
segmentation/matting) plus the structured result (boxes, points, count,
rejected, masks, timings). Everything lands in ./outputs.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = "wevisone-vl-4b"
BACKEND_SLUG = "wevisone-vl-perception"  # the slug publish.py registered
OUT_DIR = Path(__file__).resolve().parent / "outputs"

COCO = "http://images.cocodataset.org/val2017/"
AUTHORS = "https://raw.githubusercontent.com/Tencent/WeVisOne-VL/40f735b8daed437ece982db9718e9c9856f52a79/demo/examples/images/"

# (name, task, query, image). `image` may also be a local file path.
CALLS = [
    ("detect-kitchen", "detection", ["person", "pot", "bottle", "bowl", "knife"], COCO + "000000397133.jpg"),
    ("count-tennis-kids", "count_box", "people", COCO + "000000001000.jpg"),
    ("refer-cat", "referring", "the cat on the right", COCO + "000000039769.jpg"),
    ("reject-dog", "referring", "the dog", COCO + "000000039769.jpg"),
    ("point-catcher", "point", ["catcher"], COCO + "000000046252.jpg"),
    ("ocr-street", "ocr", "", COCO + "000000252219.jpg"),
    ("segment-bicycle", "segmentation", "the blue bicycle", COCO + "000000174482.jpg"),
    ("matte-portrait", "matting", "person", AUTHORS + "seabreeze.jpg"),
]


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def _image_field(src: str) -> str:
    """An https URL travels as is; a local file goes as base64."""
    path = Path(src)
    if not src.startswith(("http://", "https://")) and path.is_file():
        return base64.b64encode(path.read_bytes()).decode("ascii")
    return src


def run_calls(endpoint_url: str, api_key: str, calls=CALLS, out_dir: Path = OUT_DIR) -> list[dict]:
    out_dir.mkdir(exist_ok=True)
    headers = {"Authorization": f"Bearer {api_key}"}
    summary = []
    for name, task, query, image in calls:
        body = {"model": "default", "task": task, "query": query, "image": _image_field(image)}
        t = time.time()
        resp = requests.post(endpoint_url, headers=headers, json=body, timeout=900)
        resp.raise_for_status()
        out = resp.json()
        rtt = time.time() - t
        for item in out.get("data", []):
            (out_dir / f"{name}-{item.get('kind', 'image')}.{'jpg' if item.get('format') == 'jpeg' else 'png'}").write_bytes(base64.b64decode(item["b64_json"]))
        for i, m in enumerate(out.get("masks") or []):
            (out_dir / f"{name}-matte-{i}.png").write_bytes(base64.b64decode(m))
        slim = {k: v for k, v in out.items() if k not in ("data", "masks")}
        slim["round_trip_s"] = round(rtt, 3)
        (out_dir / f"{name}.json").write_text(json.dumps(slim, indent=1, ensure_ascii=False))
        what = f"count {out['count']}" if out.get("count") is not None else f"{out.get('instances')} instance(s)"
        if out.get("rejected"):
            what = "rejected: not in the image"
        print(
            f"{name:18s} {task:13s} {what:28s} GPU {out.get('seconds')} s "
            f"(text {out.get('seconds_generate')} s + masks {out.get('seconds_decode')} s), "
            f"round trip {rtt:.1f} s"
        )
        summary.append(slim)
    return summary


def main() -> int:
    _load_dotenv()
    mgmt = ManagementClient.from_env()
    worker_id = os.environ["IK_WORKER_ID"]
    # Reaches the backend: env vars for its process and how long the worker
    # waits for setup() (venv with vLLM + 19 GB download + two loads).
    backend_env = {
        k: os.environ[k]
        for k in ("WEVIS_ENGINE", "WEVIS_MODEL", "WEVIS_GPU_MEMORY_UTILIZATION", "WEVIS_VRAM_LIMIT_GIB", "WEVIS_MAX_PIXELS",
                  "WEVIS_MAX_MODEL_LEN", "WEVIS_MAX_NUM_SEQS")
        if os.environ.get(k)
    }
    config = {"ready_timeout_secs": 2400, "request_timeout_secs": 600, "env": backend_env}

    spec = WorkloadSpec(
        name="WeVisOne-VL-4B",
        slug=SLUG,
        model=backend_env.get("WEVIS_MODEL", "tencent/WeVisOne-VL-4B"),
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
        print("waiting for the worker to become ready (first start installs vLLM and downloads ~19 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.time() - t0:.0f} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        project = os.environ["INFERENCEKEY_PROJECT"]
        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/images/generations"
        run_calls(url, os.environ["INFERENCEKEY_API_KEY"])
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"request failed: {e}: {e.response.text[:300]}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the
        # workload (useful to try more images from the dashboard).
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

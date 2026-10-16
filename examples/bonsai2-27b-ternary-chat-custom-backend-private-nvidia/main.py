"""A 27B reasoning and vision model in 6 GB: Ternary Bonsai 2 27B on a private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`ternary-bonsai-2-27b`, published with publish.py) that downloads
Prism ML's llama.cpp fork and runs its llama-server, because only that fork
reads the ternary PTQ1_0 / PQ2_0 GGUF types.

It sends one reasoning question (and, with --image, one question about an
image) through the OpenAI-compatible `/v1/chat/completions` route and saves the
answer, the reasoning trace and the server's timings to ./outputs/.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py [--prompt "..."] [--image photo.jpg] [--effort medium|xhigh]
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

HERE = Path(__file__).resolve().parent
SLUG = os.environ.get("BONSAI_WORKLOAD_SLUG", "ternary-bonsai-2-27b")
BACKEND_SLUG = os.environ.get("BONSAI_BACKEND_SLUG", "ternary-bonsai-2-27b")  # from publish.py
OUT_DIR = HERE / "outputs"
DEFAULT_PROMPT = (
    "A warehouse robot moves 3 m north, 4 m east, 3 m south and 1 m west, then "
    "returns to its start in a straight line. Its battery drains 2% per metre. "
    "How much battery does the whole trip use? Answer in one sentence."
)


def _load_dotenv() -> None:
    env = HERE / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def chat(url: str, key: str, content, effort: str, max_tokens: int) -> dict:
    """One chat completion. Returns the backend's output object plus the round trip."""
    body = {
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        # Non-OpenAI fields travel in extra_body; the backend forwards them to llama-server.
        "extra_body": {"reasoning_effort": effort},
    }
    t0 = time.time()
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=600)
    r.raise_for_status()
    message = r.json()["choices"][0]["message"]["content"]
    # A custom backend returns its own JSON object; the data plane carries it as
    # the message content, so decode it (plain text is kept as is).
    try:
        out = json.loads(message)
    except (TypeError, ValueError):
        out = {"text": message}
    out["round_trip_s"] = round(time.time() - t0, 2)
    return out


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--image", help="optional image to ask about (uses the vision projector)")
    ap.add_argument("--image-prompt", default="Describe this image in two sentences.")
    ap.add_argument("--effort", default="medium", choices=["medium", "xhigh", "none"])
    ap.add_argument("--max-tokens", type=int, default=16384)
    args = ap.parse_args()

    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="Ternary Bonsai 2 27B",
        slug=SLUG,
        model="prism-ml/Ternary-Bonsai-2-27B-gguf",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2text",
        worker_id=os.environ["IK_WORKER_ID"],
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
        execution_policy_config={"replicas": 1},
    )
    results: dict = {"slug": SLUG}
    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        # First start: the worker builds the backend venv (CUDA runtime wheels),
        # then the backend downloads the fork (about 200 MB) and 6.6 GB of GGUF.
        mgmt.wait_until_ready(ref.workload_slug, timeout=1800)
        results["ready_after_s"] = round(time.time() - t0, 1)
        print(f"ready after {results['ready_after_s']} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{os.environ['INFERENCEKEY_PROJECT']}/{ref.workload_slug}/v1/chat/completions"
        key = os.environ["INFERENCEKEY_API_KEY"]

        out = chat(url, key, args.prompt, args.effort, args.max_tokens)
        results["text"] = {"prompt": args.prompt, "effort": args.effort, **out}
        t = out.get("timings") or {}
        print(f"answer: {out.get('text')!r}")
        print(f"  {t.get('generated_tokens')} tokens at {t.get('generation_tokens_per_s')} tokens/s, "
              f"{len(out.get('reasoning') or '')} chars of reasoning, round trip {out['round_trip_s']} s")

        if args.image:
            img = Path(args.image)
            content = [
                {"type": "text", "text": args.image_prompt},
                {"type": "image_url", "image_url": {"url": _data_uri(img)}},
            ]
            out = chat(url, key, content, args.effort, args.max_tokens)
            results["image"] = {"file": img.name, "prompt": args.image_prompt, **out}
            print(f"image answer: {out.get('text')!r} ({out['round_trip_s']} s)")

        OUT_DIR.mkdir(exist_ok=True)
        (OUT_DIR / "answer.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
        print(f"wrote {OUT_DIR / 'answer.json'}")
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"request failed: {e}: {e.response.text[:300]}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the
        # workload (useful to try more prompts from the dashboard).
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

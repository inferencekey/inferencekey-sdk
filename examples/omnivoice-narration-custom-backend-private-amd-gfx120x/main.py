"""Narrate text with **OmniVoice** on a private AMD GPU, through a custom backend.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`omnivoice-narration`, published with publish.py) that ships its own
saved voices, so every request sounds like the same narrator.

Transport: the custom backend answers the OpenAI chat route
(`/v1/chat/completions`); the message content is the backend's JSON (base64
audio + timings). The script turns the workload's `chat` endpoint on once, with
your dashboard session token (or flip "Chat" in Dashboard -> Endpoint settings).

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py "Text to narrate"
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = os.environ.get("OMNIVOICE_WORKLOAD_SLUG", "omnivoice-narration")
BACKEND_SLUG = os.environ.get("OMNIVOICE_BACKEND_SLUG", "omnivoice-narration")
OUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_TEXT = ("En un mundo donde nada es lo que parece... un hombre lo arriesgará todo. "
                "Este verano... la verdad saldrá a la luz.")


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def enable_chat_route(base: str, project: str, slug: str) -> None:
    """Turn on the workload's `chat` endpoint (text2audio workloads start with audio_speech + jobs)."""
    user_token = os.environ.get("INFERENCEKEY_USER_TOKEN")
    if not user_token:
        print("INFERENCEKEY_USER_TOKEN not set: turn on 'Chat' in Dashboard -> Endpoint settings for this workload.")
        return
    sdk = {"Authorization": f"Bearer {os.environ['INFERENCEKEY_SDK_TOKEN']}"}
    r = requests.get(f"{base}/api/projects/{project}/workloads", headers=sdk, timeout=30)
    r.raise_for_status()
    items = r.json()
    items = items.get("items", items) if isinstance(items, dict) else items
    wid = next(w["id"] for w in items if w.get("slug") == slug)
    r = requests.patch(f"{base}/api/workloads/{wid}/endpoint_settings/chat",
                       headers={"Authorization": f"Bearer {user_token}"}, json={"enabled": True}, timeout=30)
    r.raise_for_status()
    print("chat endpoint enabled")


def narrate(url: str, key: str, text: str, **options: Any) -> Dict[str, Any]:
    """One narration: the text to say plus options (voice, instruct, speed, response_format, seed)."""
    body = {"model": "default", "messages": [{"role": "user", "content": text}], "extra_body": options}
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=600)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    # The custom backend's JSON arrives as the message content (a string).
    return json.loads(r.json()["choices"][0]["message"]["content"])


def main() -> int:
    _load_dotenv()
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="OmniVoice narration",
        slug=SLUG,
        model="k2-fsa/OmniVoice",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2audio",
        worker_id=os.environ["IK_WORKER_ID"],
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
        enable_chat_route(base, project, ref.workload_slug)
        # First start: the worker builds the backend venv (ROCm torch, omnivoice)
        # and the backend downloads about 3 GB of weights.
        print("waiting for the worker to become ready (first start builds a venv and downloads ~3 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        print(f"ready after {time.time() - t0:.1f} s")

        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/chat/completions"
        OUT_DIR.mkdir(exist_ok=True)
        t = time.time()
        out = narrate(url, key, text, response_format="mp3")
        path = OUT_DIR / f"narration.{out['format']}"
        path.write_bytes(base64.b64decode(out["audio_b64"]))
        print(f"wrote {path.name}: {out['seconds']} s of audio with voice {out['voice']}, "
              f"GPU {out['timings']['gpu_s']} s, round trip {time.time() - t:.2f} s")
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (requests.RequestException, RuntimeError, KeyError, ValueError) as e:
        print(f"request failed: {e}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the workload.
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

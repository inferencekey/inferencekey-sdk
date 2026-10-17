"""Generate AND edit speech with **AuK-Flash** on a private NVIDIA GPU, through a custom backend.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`auk-flash-speech`, published with publish.py) because no built-in
backend serves AuK. Every task is one natural-language instruction, with or
without an input recording.

The script runs a small chain that needs no audio of your own:
  1. voice design: speech from a text description of the voice (no reference),
  2. zero-shot cloning: a new sentence in the voice generated in step 1,
  3. content editing: replace two words inside the step 1 recording,
  4. speed editing: the step 1 recording at 1.25x.
The WAV files land in ./outputs.

Transport: the custom backend answers the OpenAI chat route
(`/v1/chat/completions`); the message content is the backend's JSON (base64
WAV + timings). The script turns the workload's `chat` endpoint on once, with
your dashboard session token (or flip "Chat" in Dashboard -> Endpoint settings).

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
from typing import Any, Dict, Optional

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = os.environ.get("AUK_WORKLOAD_SLUG", "auk-flash")
BACKEND_SLUG = os.environ.get("AUK_BACKEND_SLUG", "auk-flash-speech")
OUT_DIR = Path(__file__).resolve().parent / "outputs"

VOICE = ("A woman in her thirties with a warm, steady voice, speaking clearly at a calm pace, "
         "like a podcast host introducing an episode.")
LINE = "Welcome back to the show. Today we look at how small teams ship big features."


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


def auk(url: str, key: str, instruction: str, audio: Optional[Path] = None, **options: Any) -> Dict[str, Any]:
    """One AuK request: instruction text, optional input WAV, options (gen_seconds, gen_text, seed)."""
    content: Any = instruction
    if audio is not None:
        content = [
            {"type": "text", "text": instruction},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio.read_bytes()).decode(), "format": "wav"}},
        ]
    body = {"model": "default", "messages": [{"role": "user", "content": content}], "extra_body": options}
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=600)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    # The custom backend's JSON arrives as the message content (a string).
    return json.loads(r.json()["choices"][0]["message"]["content"])


def save(out: Dict[str, Any], name: str, started: float) -> Path:
    path = OUT_DIR / name
    path.write_bytes(base64.b64decode(out["audio_b64"]))
    t = out["timings"]
    print(f"wrote {name}: {out['seconds']} s of audio, GPU {t['gpu_s']} s "
          f"({t['realtime_factor']}x real time), peak VRAM {t['peak_vram_mib']} MiB, "
          f"round trip {time.time() - started:.2f} s")
    return path


def main() -> int:
    _load_dotenv()
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="AuK-Flash speech generation and editing",
        slug=SLUG,
        model="tencent/AuK-Flash",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2audio",
        worker_id=os.environ["IK_WORKER_ID"],
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
        execution_policy_config={"replicas": 1},
    )
    base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
    project = os.environ["INFERENCEKEY_PROJECT"]
    key = os.environ["INFERENCEKEY_API_KEY"]
    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        enable_chat_route(base, project, ref.workload_slug)
        # First start: the worker builds the backend venv (torch 2.7, transformers),
        # the backend downloads the AuK source, 6.8 GB of AuK-Flash and the 12 GB
        # Qwen2.5-Omni-3B encoder, and loads them. Give it room.
        print("waiting for the worker to become ready (first start downloads about 19 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        print(f"ready after {time.time() - t0:.1f} s")

        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/chat/completions"
        OUT_DIR.mkdir(exist_ok=True)

        t = time.time()
        # Voice design. This template (the one in AuK's CLI example) kept the
        # description out of the audio in our tests; the cookbook's "Generate speech
        # based on ... The content to speak is ..." often read part of it aloud.
        out = auk(url, key, f'Based on the following description: "{VOICE}", generate speech content "{LINE}".', seed=7)
        host = save(out, "1-voice-design.wav", t)

        t = time.time()
        out = auk(url, key, 'Say the following with the same voice: "Thanks for listening. '
                            'See you next week, and keep building."', audio=host,
                  gen_text="Thanks for listening. See you next week, and keep building.", ref_text=LINE, seed=7)
        save(out, "2-zero-shot-clone.wav", t)

        t = time.time()
        out = auk(url, key, "Replace 'small teams' with 'two engineers'.", audio=host, seed=7)
        save(out, "3-content-edit.wav", t)

        t = time.time()
        out = auk(url, key, "Adjust the speech speed to 1.25x.", audio=host, seed=7)
        save(out, "4-speed-1.25x.wav", t)
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

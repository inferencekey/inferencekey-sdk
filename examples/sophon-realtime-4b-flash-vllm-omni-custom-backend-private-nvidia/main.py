"""A spoken conversation turn with **Sophon-Realtime-4B-Flash** on a private NVIDIA GPU, served by vLLM-Omni.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`sophon-realtime-4b-flash`, published with publish.py) that runs the
authors' `flash/install.sh` and `flash/start.sh` next to it, because no
built-in backend serves this vLLM-Omni plugin.

The platform takes HTTP requests, so each call carries one recorded caller
turn; the backend plays it into the model's OpenAI Realtime-style WebSocket at
speaking speed, keeps the microphone open (silence) and lets the model decide
on its own when the caller has finished, as it would on a live line. The script
calls the workload's `/v1/chat/completions` route three times:
  1. realtime turn, built-in voice `chelsie`,
  2. realtime turn in a cloned voice, if you pass a 3 to 10 s clip of your own
     voice (only clone your own voice or one you have permission to use),
  3. the same question in `chat` mode (whole clip in, streamed reply out).
For each: the reply WAV (24 kHz), a two-channel conversation WAV (caller left,
model right, on one clock), and a JSON with the reply text, its sound tags and
the timings land in ./outputs.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py my-question.wav [my-voice.wav]
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

SLUG = os.environ.get("SOPHON_WORKLOAD_SLUG", "sophon-realtime-4b")
BACKEND_SLUG = os.environ.get("SOPHON_BACKEND_SLUG", "sophon-realtime-4b-flash")
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def _workload(base: str, project: str, slug: str) -> Dict[str, Any]:
    sdk = {"Authorization": f"Bearer {os.environ['INFERENCEKEY_SDK_TOKEN']}"}
    r = requests.get(f"{base}/api/projects/{project}/workloads", headers=sdk, timeout=30)
    r.raise_for_status()
    items = r.json()
    items = items.get("items", items) if isinstance(items, dict) else items
    return next(w for w in items if w.get("slug") == slug)


def enable_chat_route(base: str, project: str, slug: str) -> None:
    """Turn on the workload's `chat` endpoint (text2audio workloads start with audio_speech + jobs)."""
    user_token = os.environ.get("INFERENCEKEY_USER_TOKEN")
    if not user_token:
        print("INFERENCEKEY_USER_TOKEN not set: turn on 'Chat' in Dashboard -> Endpoint settings for this workload.")
        return
    wid = _workload(base, project, slug)["id"]
    r = requests.patch(f"{base}/api/workloads/{wid}/endpoint_settings/chat",
                       headers={"Authorization": f"Bearer {user_token}"}, json={"enabled": True}, timeout=30)
    r.raise_for_status()
    print("chat endpoint enabled")


def allow_parallel_requests(base: str, project: str, slug: str, worker_id: str, n: int = 8) -> None:
    """Let the worker run n conversations of this workload at once (the assignment starts at 1).

    The 4B config holds 8 realtime sessions; with the default of 1 the platform
    hands turns to the backend one by one. Needs the SDK token's assignment:write scope.
    """
    sdk = {"Authorization": f"Bearer {os.environ['INFERENCEKEY_SDK_TOKEN']}"}
    w = _workload(base, project, slug)
    r = requests.patch(f"{base}/api/projects/{w['project_id']}/assignments/{w['id']}/{worker_id}",
                       headers=sdk, json={"max_in_flight": n}, timeout=30)
    print(f"max_in_flight={n}: HTTP {r.status_code}" + ("" if r.ok else f" {r.text[:200]}"))


def turn(url: str, key: str, name: str, question: Path, **options: Any) -> Dict[str, Any]:
    """One caller turn; the backend's JSON arrives as the message content (a string)."""
    audio = {"type": "input_audio", "input_audio": {"data": base64.b64encode(question.read_bytes()).decode(),
                                                    "format": "wav"}}
    body = {"model": "default", "messages": [{"role": "user", "content": [audio]}], "extra_body": options}
    t = time.time()
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=600)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    out = json.loads(r.json()["choices"][0]["message"]["content"])
    rt = time.time() - t
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{name}.wav").write_bytes(base64.b64decode(out.pop("audio_b64")))
    if out.get("conversation_b64"):
        (OUT_DIR / f"{name}.conversation.wav").write_bytes(base64.b64decode(out.pop("conversation_b64")))
    out["round_trip_s"] = round(rt, 2)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    t = out["timings"]
    if out["mode"] == "realtime":
        print(f"{name}: first reply audio {t['turn_latency_s']} s after the caller stopped, "
              f"{out['seconds']} s of speech, round trip {rt:.1f} s")
    else:
        print(f"{name}: first audio {t['first_audio_s']} s after the request, {out['seconds']} s of speech")
    print(f"  reply: {out['text']}")
    return out


def main() -> int:
    _load_dotenv()
    if len(sys.argv) < 2:
        print("usage: python main.py question.wav [your-voice.wav]", file=sys.stderr)
        return 2
    question = Path(sys.argv[1])
    voice_clip: Optional[Path] = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        slug=SLUG,
        name="Sophon-Realtime 4B voice agent",
        model="adventists-ai/Sophon-Realtime-4B-Flash",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2audio",
        worker_id=os.environ["IK_WORKER_ID"],
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
        # The first start installs vLLM-Omni, downloads 13 GB and compiles kernels: give setup() time.
        # SOPHON_* variables (see README) pass through to the backend: e.g. the 16 GB card settings.
        config={"ready_timeout_secs": 3600, "request_timeout_secs": 300,
                "env": {k: v for k, v in os.environ.items() if k.startswith("SOPHON_") and k not in (
                    "SOPHON_WORKLOAD_SLUG", "SOPHON_BACKEND_SLUG")}},
    )
    base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
    project = os.environ["INFERENCEKEY_PROJECT"]
    key = os.environ["INFERENCEKEY_API_KEY"]
    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        print("waiting for the worker to become ready (first start installs vLLM-Omni and compiles kernels)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.time() - t0:.1f} s")
        enable_chat_route(base, project, ref.workload_slug)
        allow_parallel_requests(base, project, ref.workload_slug, os.environ["IK_WORKER_ID"])

        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/chat/completions"
        turn(url, key, "1-realtime-chelsie", question, mode="realtime", voice="chelsie", language="en")
        if voice_clip is not None:
            turn(url, key, "2-realtime-cloned-voice", question, mode="realtime", voice="myvoice", language="en",
                 voice_clip_b64=base64.b64encode(voice_clip.read_bytes()).decode())
        else:
            print("no voice clip given: skipping the cloned-voice turn (python main.py question.wav my-voice.wav)")
        turn(url, key, "3-chat-chelsie", question, mode="chat", voice="chelsie", language="en")
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

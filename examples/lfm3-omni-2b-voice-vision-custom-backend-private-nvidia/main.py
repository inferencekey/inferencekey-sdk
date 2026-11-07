"""Voice, vision and speech output with **LFM3-Omni-2B** on a private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`lfm3-omni-2b`, published with publish.py) because the built-in
backends serve this model's text output at most, and its point is that it
also talks back.

The script sends three requests to one workload:
  1. a spoken question (WAV) answered in text and speech, interleaved,
  2. a photo plus a spoken question about it, answered in text,
  3. text to speech (a female American English voice).
Inputs come from ./samples (or IK_* paths); the answers land in ./outputs.

Transport: the custom backend answers the OpenAI chat route
(`/v1/chat/completions`); the message content is the backend's JSON (text,
base64 WAV, timings). The script turns the workload's `chat` endpoint on once,
with your dashboard session token (or flip "Chat" in Dashboard -> Endpoint
settings).

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

HERE = Path(__file__).resolve().parent
SLUG = os.environ.get("LFM3_WORKLOAD_SLUG", "lfm3-omni")
BACKEND_SLUG = os.environ.get("LFM3_BACKEND_SLUG", "lfm3-omni-2b")
OUT_DIR = HERE / "outputs"
QUESTION_WAV = Path(os.environ.get("IK_QUESTION_WAV", HERE / "samples" / "question.wav"))
IMAGE = Path(os.environ.get("IK_IMAGE", HERE / "samples" / "flyer.jpg"))
IMAGE_QUESTION_WAV = Path(os.environ.get("IK_IMAGE_QUESTION_WAV", HERE / "samples" / "flyer-question.wav"))


def _load_dotenv() -> None:
    env = HERE / ".env"
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


def audio_part(path: Path) -> Dict[str, Any]:
    return {"type": "input_audio",
            "input_audio": {"data": base64.b64encode(path.read_bytes()).decode(), "format": path.suffix.lstrip(".")}}


def image_part(path: Path) -> Dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"}}


def ask(url: str, key: str, messages: List[Dict[str, Any]], **options: Any) -> Dict[str, Any]:
    """One request; the backend's JSON arrives as the message content (a string)."""
    body = {"model": "default", "messages": messages, "extra_body": options}
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=600)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return json.loads(r.json()["choices"][0]["message"]["content"])


def report(out: Dict[str, Any], name: str, started: float) -> None:
    t = out["timings"]
    line = f"{name}: GPU {t['generate_s']} s, round trip {time.time() - started:.2f} s, peak VRAM {t['peak_vram_mib']} MiB"
    if out.get("audio_b64"):
        path = OUT_DIR / f"{name}.wav"
        path.write_bytes(base64.b64decode(out["audio_b64"]))
        line += f", {out['audio_seconds']} s of speech -> {path.name} ({t['realtime_factor']}x real time)"
    print(line)
    if out.get("text"):
        print(f"  text: {out['text']}")
    (OUT_DIR / f"{name}.json").write_text(json.dumps({k: v for k, v in out.items() if k != "audio_b64"}, indent=1))


def main() -> int:
    _load_dotenv()
    for p in (QUESTION_WAV, IMAGE, IMAGE_QUESTION_WAV):
        if not p.is_file():
            print(f"input not found: {p} (see samples/ in the README)", file=sys.stderr)
            return 1
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="LFM3-Omni-2B voice and vision",
        slug=SLUG,
        model="LiquidAI/LFM3-Omni-2B",
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
        # First start: the worker builds the backend venv (torch, transformers 5),
        # the backend downloads 4.5 GB of weights and loads them.
        print("waiting for the worker to become ready (first start installs torch and downloads 4.5 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=1800)
        print(f"ready after {time.time() - t0:.1f} s")

        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/chat/completions"
        OUT_DIR.mkdir(exist_ok=True)

        t = time.time()
        out = ask(url, key, [
            {"role": "system", "content": "Respond with interleaved text and audio."},
            {"role": "user", "content": [audio_part(QUESTION_WAV)]},
        ], mode="interleaved", max_new_tokens=1024)
        report(out, "1-voice-answer", t)

        t = time.time()
        out = ask(url, key, [
            {"role": "system", "content": "Answer questions about the provided image. "
                                          "Base every answer only on what can be seen. Respond with text."},
            {"role": "user", "content": [image_part(IMAGE), audio_part(IMAGE_QUESTION_WAV)]},
        ], mode="text", max_new_tokens=256)
        report(out, "2-spoken-question-about-image", t)

        t = time.time()
        out = ask(url, key, [
            {"role": "system", "content": "Perform TTS in American English using a female voice."},
            {"role": "user", "content": "The garden stays open until sunset, as long as the weather holds."},
        ], mode="audio", max_new_tokens=512)
        report(out, "3-text-to-speech", t)
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

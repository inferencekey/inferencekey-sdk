"""Speech with **Ming-3.0-tts-0.5B** on a private NVIDIA GPU, through a custom backend.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`ming-3-tts-05b`, published with publish.py) because no built-in
backend serves Ming-3.0-tts.

The script runs a small chain that needs no audio of your own:
  1. a built-in voice (UK_George) reads an English line,
  2. voice design: a new voice from a text description, no reference audio,
  3. zero-shot cloning: the voice from step 2 says a new line with inline
     paralinguistic tags ([Laughter], [Sigh]),
  4. the same cloned voice speaks Spanish.
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

SLUG = os.environ.get("MING_WORKLOAD_SLUG", "ming-tts")
BACKEND_SLUG = os.environ.get("MING_BACKEND_SLUG", "ming-3-tts-05b")
OUT_DIR = Path(__file__).resolve().parent / "outputs"

DESIGNED_VOICE = ("A calm middle-aged male narrator with a deep, warm voice, "
                  "speaking slowly and clearly, like a nature documentary.")
DESIGNED_LINE = "Every spring, the river carries snow from the mountains down to the valley."


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


def speak(url: str, key: str, text: str, reference: Optional[Path] = None, **options: Any) -> Dict[str, Any]:
    """One request: the text to say, an optional reference WAV to clone, task options."""
    content: Any = text
    if reference is not None:
        content = [
            {"type": "text", "text": text},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(reference.read_bytes()).decode(), "format": "wav"}},
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
    print(f"wrote {name} ({out['task']}): {out['seconds']} s of audio, GPU {t['gpu_s']} s "
          f"({t['realtime_factor']}x real time), peak VRAM {t['peak_vram_mib']} MiB, "
          f"round trip {time.time() - started:.2f} s")
    return path


def main() -> int:
    _load_dotenv()
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="Ming-3.0-tts-0.5B speech",
        slug=SLUG,
        model="inclusionAI/Ming-3.0-tts-0.5B",
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
        # First start: the worker builds the backend venv (torch 2.6, transformers),
        # the backend downloads the Ming source and 3.0 GB of weights, and loads them.
        print("waiting for the worker to become ready (first start installs torch and downloads 3 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=1800)
        print(f"ready after {time.time() - t0:.1f} s")

        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/chat/completions"
        OUT_DIR.mkdir(exist_ok=True)

        t = time.time()
        out = speak(url, key, "The garden will remain open until sunset, provided the weather stays clear.",
                    task="IP", voice="UK_George", seed=1895)
        save(out, "1-built-in-voice.wav", t)

        t = time.time()
        out = speak(url, key, DESIGNED_LINE, task="VOICE_DESIGN", style=DESIGNED_VOICE, seed=1895)
        narrator = save(out, "2-voice-design.wav", t)

        t = time.time()
        out = speak(url, key, "[Laughter] I did not expect that at all. [Sigh] Well, let's try one more time.",
                    reference=narrator, prompt_text=DESIGNED_LINE, task="TTS", language_id="en", seed=1895)
        save(out, "3-clone-with-tags.wav", t)

        t = time.time()
        out = speak(url, key, "Cada primavera, el río baja la nieve de las montañas hasta el valle.",
                    reference=narrator, prompt_text=DESIGNED_LINE, task="MULTILINGUAL", language_id="es", seed=1895)
        save(out, "4-clone-spanish.wav", t)
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

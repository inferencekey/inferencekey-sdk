"""A spoken conversation with **Haru-4B** (full duplex, 9 languages) on a private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`haru-4b-full-duplex`, published with publish.py) that wraps the
authors' PyTorch `DuplexSession`, because no built-in backend serves this model.

Haru listens and speaks at the same time, one 133 ms frame at a time. The
platform takes HTTP requests, so a call carries one recorded caller clip: the
backend feeds it to the model frame by frame, exactly like a live line, keeps
the microphone open for a few seconds of silence, and lets the model decide by
itself when to talk, when to say "mm-hm" and when to stay quiet. Haru ships
without voices: you give it one with a 3 s to 3 min clip of ONE speaker (only
use a voice you have the right to use).

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py caller.wav voice.wav [--language es]

Output in ./outputs: the model's side (`<name>.wav`, 24 kHz), the whole call
(`<name>.conversation.wav`, caller left, model right, one clock) and a JSON with
what the model wrote, when, and how long each frame took on the GPU.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = os.environ.get("HARU_WORKLOAD_SLUG", "haru-4b")
BACKEND_SLUG = os.environ.get("HARU_BACKEND_SLUG", "haru-4b-full-duplex")
OUT_DIR = Path(__file__).resolve().parent / "outputs"

# A character card per language (the backend renders it with the authors' templates).
CHARACTERS: Dict[str, Dict[str, str]] = {
    "en": {
        "persona": "Leo is a 34-year-old travel writer who lives in Lisbon. He is warm, curious and laughs easily.",
        "name": "Leo",
        "gender": "male",
        "backchannel": "mid",
        "other_name": "Anna",
        "relationship": "Old friends from university.",
        "situation": "A phone call to catch up after a long time.",
    },
    "es": {
        "persona": "Marcos tiene 34 años, es escritor de viajes y vive en Madrid. Es cercano, curioso y se ríe con facilidad.",
        "name": "Marcos",
        "gender": "male",
        "backchannel": "mid",
        "other_name": "Lucía",
        "relationship": "Amigos desde la universidad.",
        "situation": "Una llamada para ponerse al día después de mucho tiempo.",
    },
}


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
    wid = next(w for w in items if w.get("slug") == slug)["id"]
    r = requests.patch(f"{base}/api/workloads/{wid}/endpoint_settings/chat",
                       headers={"Authorization": f"Bearer {user_token}"}, json={"enabled": True}, timeout=30)
    r.raise_for_status()
    print("chat endpoint enabled")


def converse(url: str, key: str, name: str, caller: Path, voice: Path, **options: Any) -> Dict[str, Any]:
    """One recorded caller clip in, the model's side of the call out (the JSON arrives as the message content)."""
    audio = {"type": "input_audio", "input_audio": {"data": base64.b64encode(caller.read_bytes()).decode(),
                                                    "format": "wav"}}
    extra = {"voice_clip_b64": base64.b64encode(voice.read_bytes()).decode(), **options}
    body = {"model": "default", "messages": [{"role": "user", "content": [audio]}], "extra_body": extra}
    t = time.time()
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=600)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    out = json.loads(r.json()["choices"][0]["message"]["content"])
    rt = time.time() - t
    OUT_DIR.mkdir(exist_ok=True)
    ext = out["format"]  # wav, or flac for long calls (results travel in one message of at most 8 MB)
    (OUT_DIR / f"{name}.{ext}").write_bytes(base64.b64decode(out.pop("audio_b64")))
    conversation = out.pop("conversation_b64")
    if conversation:
        (OUT_DIR / f"{name}.conversation.{ext}").write_bytes(base64.b64decode(conversation))
    out["round_trip_s"] = round(rt, 2)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    t = out["timings"]
    print(f"{name}: {t['frames']} frames, {t['frame_ms_median']} ms per 133 ms frame on the GPU "
          f"({t['realtime_factor']}x real time), model starts {t['model_starts_after_caller_s']} s after the "
          f"caller's last word, round trip {rt:.1f} s")
    print(f"  model said: {out['text']}")
    return out


def main() -> int:
    _load_dotenv()
    p = argparse.ArgumentParser(description="One call with Haru-4B on your own GPU")
    p.add_argument("caller", type=Path, help="recording of the caller (WAV, any rate)")
    p.add_argument("voice", type=Path, help="3 s to 3 min of ONE speaker: the voice Haru speaks with")
    p.add_argument("--language", default="en", choices=sorted(CHARACTERS) + ["pt", "de", "fr", "zh", "ja", "ru", "ko"])
    p.add_argument("--tail-seconds", type=float, default=8.0, help="open microphone after the clip")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        slug=SLUG,
        name="Haru-4B voice line",
        model="higurashi-ai/Haru-4B",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2audio",
        worker_id=os.environ["IK_WORKER_ID"],
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
        # The first start installs PyTorch and downloads 10.5 GB: give setup() time.
        # HARU_* variables (see README) pass through to the backend, e.g. HARU_GPU_MEMORY_GB.
        config={"ready_timeout_secs": 2400, "request_timeout_secs": 300,
                "env": {k: v for k, v in os.environ.items() if k.startswith("HARU_") and k not in (
                    "HARU_WORKLOAD_SLUG", "HARU_BACKEND_SLUG")}},
    )
    base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
    project = os.environ["INFERENCEKEY_PROJECT"]
    key = os.environ["INFERENCEKEY_API_KEY"]
    character = CHARACTERS.get(args.language, CHARACTERS["en"])
    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        print("waiting for the worker to become ready (first start installs PyTorch and downloads the weights)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        print(f"ready after {time.time() - t0:.1f} s")
        enable_chat_route(base, project, ref.workload_slug)
        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/chat/completions"
        converse(url, key, f"call-{args.language}", args.caller, args.voice, language=args.language,
                 character=character, tail_seconds=args.tail_seconds, seed=args.seed)
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

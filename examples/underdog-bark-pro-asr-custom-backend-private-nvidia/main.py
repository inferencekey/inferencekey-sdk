"""Transcribe English speech with **Underdog Bark 3.0 Pro** on a private NVIDIA GPU, through a custom backend.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`underdog-bark-pro`, published with publish.py) because no built-in
backend runs a Parakeet TDT model. Bark 3.0 Pro is 0.6B parameters, 1.25 GB of
bf16 weights, CC-BY-4.0, and loads as a stock Transformers model.

The endpoint is OpenAI-compatible: `POST .../v1/audio/transcriptions` with a
multipart `file`, the same call `openai.audio.transcriptions.create()` makes,
and the answer is `{"text": ...}` plus timings.

IK_AUDIO is a file or a folder of audio files. If the folder has a
manifest.json with reference transcripts ([{"id": "<file stem>", "ref": "..."}]),
the script also scores word error rate with the Whisper English normalizer
(the Open ASR Leaderboard's). Results land in ./outputs/bark-pro.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

from asr_eval import run
from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = "underdog-bark-pro"
BACKEND_SLUG = "underdog-bark-pro"  # the slug publish.py registered
MODEL = "ConwayResearch/Underdog-Bark-3.0-Pro"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "bark-pro"


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    _load_dotenv()
    audio = os.environ.get("IK_AUDIO", "sample.wav")
    if not Path(audio).exists():
        print(f"audio not found: {audio!r}. Set IK_AUDIO to a wav/flac/mp3 file or a folder of them.", file=sys.stderr)
        return 1
    mgmt = ManagementClient.from_env()
    worker_id = os.environ["IK_WORKER_ID"]
    # Reaches the backend process as environment variables (see backend.py).
    backend_env = {
        k: os.environ[k]
        for k in ("BARK_DEVICE", "BARK_DTYPE", "BARK_VRAM_LIMIT_GIB", "BARK_CHUNK_SECONDS", "BARK_CPU_THREADS")
        if os.environ.get(k)
    }
    config = {"ready_timeout_secs": 1800, "request_timeout_secs": 600, "env": backend_env}

    spec = WorkloadSpec(
        name="Underdog Bark 3.0 Pro",
        slug=SLUG,
        model=MODEL,
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="audio2text",
        worker_id=worker_id,
        config=config,
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
    )

    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        print("waiting for the worker to become ready (first start installs torch and downloads 1.25 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=1800)
        print(f"ready after {time.time() - t0:.0f} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        project = os.environ["INFERENCEKEY_PROJECT"]
        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/audio/transcriptions"
        run(url, os.environ["INFERENCEKEY_API_KEY"], audio, OUT_DIR)
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"request failed: {e}: {e.response.text[:300]}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the
        # workload (to transcribe more audio, or run compare_qwen3_asr.py next).
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

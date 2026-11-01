"""Transcribe a recording with **NVIDIA Nemotron 3.5 Transcribe** on one private 32 GB NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: one speech model, and the PROMPT
picks the transcript. The script sends the same recording three times:

  1. ``verbatim``  the words as spoken, fillers and repetitions included,
  2. ``clean``     fillers and false starts removed, wording otherwise kept,
  3. ``speakers``  speaker tags (<spk:0>, <spk:1>...) plus word timestamps
                   and diarization segments from the model's CTC head.

The backend is a **custom Python backend** (``nemotron-transcribe``, published
with publish.py): it installs vLLM 0.32.0 with the NeMo Speech plugin from the
NeMo Speech lock file and serves the NVFP4 checkpoint (21.6 GB, Blackwell GPUs).
The endpoint is OpenAI-compatible: ``POST .../v1/audio/transcriptions`` with a
multipart ``file``; the instruction travels in the standard ``prompt`` field
(a style name or your own instruction) and ``timestamp_granularities[]=word``
turns on word timestamps. The request body limit of that route is 2 MiB, so
send compressed audio (Opus at 32 kbps holds about 8 minutes in 2 MiB; cut
longer recordings).

Outputs land in ./outputs/<style>.json (text, timings, timestamps).

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = os.environ.get("NEMOTRON_WORKLOAD_SLUG", "nemotron-transcribe")
BACKEND_SLUG = "nemotron-transcribe"  # the slug publish.py registered
MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
STYLES = [("verbatim", False), ("clean", False), ("speakers", True)]


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def transcribe(url: str, api_key: str, path: Path, prompt: str, timestamps: bool = False, timeout: int = 1800) -> dict:
    """One multipart request, the same shape openai.audio.transcriptions.create() sends."""
    data = [("model", "default"), ("prompt", prompt)]
    if timestamps:
        data.append(("timestamp_granularities[]", "word"))
    t = time.time()
    with open(path, "rb") as f:
        resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}"},
                             files={"file": (path.name, f, "application/octet-stream")}, data=data, timeout=timeout)
    resp.raise_for_status()
    out = resp.json()
    out["round_trip_s"] = round(time.time() - t, 3)
    return out


def main() -> int:
    _load_dotenv()
    audio = Path(os.environ.get("IK_AUDIO", "sample.ogg"))
    if not audio.is_file():
        print(f"audio not found: {audio}. Set IK_AUDIO to a wav/flac/mp3/ogg file (under 2 MiB).", file=sys.stderr)
        return 1
    mgmt = ManagementClient.from_env()
    # Backend knobs reach the backend process as environment variables (see backend.py).
    backend_env = {k: v for k, v in os.environ.items() if k.startswith("NEMOTRON_") and k != "NEMOTRON_WORKLOAD_SLUG"}
    # The first start builds the vLLM + NeMo Speech environment and downloads 21.6 GB: allow 40 minutes.
    config = {"ready_timeout_secs": 2400, "request_timeout_secs": 1800, "env": backend_env}

    spec = WorkloadSpec(
        name="Nemotron 3.5 Transcribe",
        slug=SLUG,
        model=MODEL,
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="audio2text",
        worker_id=os.environ["IK_WORKER_ID"],
        config=config,
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
    )

    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        print("waiting for the worker (first start: vLLM + NeMo Speech install, 21.6 GB download, kernel compile)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        print(f"ready after {time.time() - t0:.0f} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{os.environ['INFERENCEKEY_PROJECT']}/{ref.workload_slug}/v1/audio/transcriptions"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for style, timestamps in STYLES:
            out = transcribe(url, os.environ["INFERENCEKEY_API_KEY"], audio, style, timestamps)
            (OUT_DIR / f"{style}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
            words = len((out.get("ctc_timestamps") or {}).get("words") or [])
            print(f"\n== {style}: {out.get('duration')} s of audio in {out.get('seconds')} s on the GPU "
                  f"({out.get('rtfx')}x real time), round trip {out['round_trip_s']} s"
                  + (f", {words} timed words" if timestamps else ""))
            print(out.get("text", "")[:600])
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"request failed: {e}: {e.response.text[:300]}", file=sys.stderr)
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

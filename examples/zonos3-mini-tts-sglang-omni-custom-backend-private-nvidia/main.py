"""Text to speech and voice cloning with **ZONOS3-Mini** on a private NVIDIA GPU, served by SGLang-Omni.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`zonos3-mini-sglang-omni`, published with publish.py) that runs
SGLang-Omni's `sgl-omni serve` next to it, because no built-in backend serves
ZONOS3-Mini.

The script calls the workload's OpenAI-compatible `/v1/audio/speech` route:
  1. the default voice reads an English line,
  2. a preset voice (one of the model's 27) reads a Spanish line,
  3. if you pass a reference clip (any length from 0.5 s, about 10 s works
     well; no transcript needed), the model clones it for a new English line,
  4. and the same cloned voice speaks Spanish.
The WAV files (44.1 kHz mono) land in ./outputs.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py [reference.wav]
"""

from __future__ import annotations

import base64
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any, Optional

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = os.environ.get("ZONOS3_WORKLOAD_SLUG", "zonos3-mini")
BACKEND_SLUG = os.environ.get("ZONOS3_BACKEND_SLUG", "zonos3-mini-sglang-omni")
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def allow_parallel_requests(base: str, project: str, slug: str, worker_id: str, n: int = 16) -> None:
    """Let the worker run n requests of this workload at once (the assignment starts at 1).

    SGLang-Omni batches concurrent requests on the GPU; with the default of 1 the
    platform hands them to the backend one by one. Needs the SDK token's
    assignment:write scope.
    """
    sdk = {"Authorization": f"Bearer {os.environ['INFERENCEKEY_SDK_TOKEN']}"}
    r = requests.get(f"{base}/api/projects/{project}/workloads", headers=sdk, timeout=30)
    r.raise_for_status()
    items = r.json()
    items = items.get("items", items) if isinstance(items, dict) else items
    w = next(w for w in items if w.get("slug") == slug)
    r = requests.patch(f"{base}/api/projects/{w['project_id']}/assignments/{w['id']}/{worker_id}",
                       headers=sdk, json={"max_in_flight": n}, timeout=30)
    print(f"max_in_flight={n}: HTTP {r.status_code}" + ("" if r.ok else f" {r.text[:200]}"))


def speak(url: str, key: str, name: str, text: str, reference: Optional[Path] = None, **options: Any) -> Path:
    """One /v1/audio/speech request; writes outputs/<name> and prints the timing."""
    body: dict = {"model": "default", "input": text, "response_format": "wav", **options}
    if reference is not None:
        # No transcript needed: ZONOS3-Mini clones from the audio alone.
        body["ref_audio"] = base64.b64encode(reference.read_bytes()).decode()
    t = time.time()
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=600)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    elapsed = time.time() - t
    path = OUT_DIR / name
    path.write_bytes(r.content)
    with wave.open(str(path)) as w:
        seconds = w.getnframes() / float(w.getframerate())
    print(f"wrote {name}: {seconds:.2f} s of audio in {elapsed:.2f} s round trip "
          f"({seconds / elapsed:.1f}x real time, network included)")
    return path


def main() -> int:
    _load_dotenv()
    reference = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if reference is not None and not reference.is_file():
        print(f"reference clip not found: {reference}", file=sys.stderr)
        return 2
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="ZONOS3-Mini speech",
        slug=SLUG,
        model="Zyphra/ZONOS3-Mini",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2audio",
        worker_id=os.environ["IK_WORKER_ID"],
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
        # The first start installs SGLang-Omni and downloads about 14.5 GB: give setup() time.
        # SGLang-Omni's default memory share (0.3) is sized for an 80 GB card; see README for small cards.
        config={"ready_timeout_secs": 2400, "request_timeout_secs": 900,
                "env": {"ZONOS3_ENGINE_GPU_MEMORY_FRACTION": os.environ.get("ZONOS3_ENGINE_GPU_MEMORY_FRACTION", "0.55")}},
    )
    base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
    project = os.environ["INFERENCEKEY_PROJECT"]
    key = os.environ["INFERENCEKEY_API_KEY"]
    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        print("waiting for the worker to become ready (first start installs SGLang-Omni and downloads the weights)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.time() - t0:.1f} s")
        allow_parallel_requests(base, project, ref.workload_slug, os.environ["IK_WORKER_ID"])

        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/audio/speech"
        OUT_DIR.mkdir(exist_ok=True)
        speak(url, key, "1-default-voice.wav",
              "The garden will remain open until sunset, provided the weather stays clear.", seed=7)
        speak(url, key, "2-preset-voice-spanish.wav",
              "El jardín permanecerá abierto hasta el atardecer, si el tiempo sigue despejado.",
              voice="Ximena_es_F_1", seed=7)
        if reference is not None:
            speak(url, key, "3-clone-english.wav",
                  "Your order has shipped. It should arrive on Thursday, and you can track it from your account page.",
                  reference=reference, seed=7)
            speak(url, key, "4-clone-spanish.wav",
                  "Tu pedido ya ha salido. Debería llegar el jueves, y puedes seguirlo desde tu cuenta.",
                  reference=reference, seed=7)
        else:
            print("no reference clip given: skipping the cloning calls (python main.py my-voice.wav)")
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (requests.RequestException, RuntimeError, KeyError, ValueError, wave.Error) as e:
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

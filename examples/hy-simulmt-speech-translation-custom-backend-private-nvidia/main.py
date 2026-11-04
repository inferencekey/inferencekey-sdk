"""Translate speech while it is being spoken with **Tencent Hy-SimulMT** on one private 48 GB NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: a SIMULTANEOUS translation model. It
does not wait for the end of the sentence: it listens in 1.6 s blocks and,
after each block, either waits or writes the next piece of the translation.
The backend streams each recording to the model at the speed it was spoken,
the way a microphone would, and returns every partial result with the moment
it arrived, so you can see how far behind the speaker the translation runs.

The backend is a **custom Python backend** (``hy-simulmt``, published with
publish.py): it installs vLLM 0.32.0 with Tencent's ``simulst`` plugin and
their SimulInfer streaming service, and serves the FP8 Interleave checkpoint
(32 GB; it writes the source transcript next to the translation). The endpoint
is ``POST .../v1/audio/transcriptions`` with a multipart ``file`` plus
``direction`` (``zh-en`` or ``en-zh``). The request body limit of that route
is 2 MiB: about one minute of 16 kHz WAV, several minutes of FLAC or Opus.

Outputs land in ./outputs/<clip>.json (translation, source text, every partial
result with its arrival time, delays).

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

SLUG = os.environ.get("HYSIMULMT_WORKLOAD_SLUG", "hy-simulmt")
BACKEND_SLUG = "hy-simulmt"  # the slug publish.py registered
MODEL = "tencent/Hy-SimulMT-30B-A3B-Interleave-FP8"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def translate(url: str, api_key: str, path: Path, direction: str, transport: str = "realtime",
              tts: bool = False, timeout: int = 1200) -> dict:
    """One multipart request: the recording plus the direction."""
    data = [("model", "default"), ("direction", direction), ("transport", transport)]
    if tts:
        data.append(("tts", "true"))
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
    # IK_CLIPS: comma-separated direction:path pairs, e.g. "zh-en:speech_zh.flac,en-zh:speech_en.flac".
    clips = []
    for item in os.environ.get("IK_CLIPS", "zh-en:sample_zh.flac").split(","):
        direction, _, path = item.strip().partition(":")
        clips.append((direction, Path(path)))
    missing = [str(p) for _, p in clips if not p.is_file()]
    if missing:
        print(f"audio not found: {missing}. Set IK_CLIPS to direction:path pairs (each file under 2 MiB).", file=sys.stderr)
        return 1
    tts = os.environ.get("HYSIMULMT_TTS") == "1"
    mgmt = ManagementClient.from_env()
    # Backend knobs reach the backend process as environment variables (see backend.py).
    backend_env = {k: v for k, v in os.environ.items() if k.startswith("HYSIMULMT_") and k != "HYSIMULMT_WORKLOAD_SLUG"}
    # The first start builds the vLLM environment and downloads 32 GB: allow 40 minutes.
    config = {"ready_timeout_secs": 2400, "request_timeout_secs": 1200, "env": backend_env}

    spec = WorkloadSpec(
        name="Hy-SimulMT live translation",
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
        print("waiting for the worker (first start: vLLM + simulst + SimulInfer install, 32 GB download)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        print(f"ready after {time.time() - t0:.0f} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{os.environ['INFERENCEKEY_PROJECT']}/{ref.workload_slug}/v1/audio/transcriptions"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for direction, path in clips:
            out = translate(url, os.environ["INFERENCEKEY_API_KEY"], path, direction, tts=tts)
            wav = (out.get("tts") or {}).pop("wav_b64", None)
            if wav:
                import base64

                (OUT_DIR / f"{path.stem}.translated.wav").write_bytes(base64.b64decode(wav))
            (OUT_DIR / f"{path.stem}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
            d = out.get("delay_ms") or {}
            print(f"\n== {path.name} ({direction}): {out.get('audio_s')} s of speech, streamed at speaking speed")
            print(f"   first words of translation after {out.get('first_text_ms')} ms; each 1.6 s block answered "
                  f"{d.get('median')} ms after it was spoken (p90 {d.get('p90')} ms); round trip {out['round_trip_s']} s")
            print(f"   source:      {out.get('source_text', '')[:300]}")
            print(f"   translation: {out.get('text', '')[:300]}")
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

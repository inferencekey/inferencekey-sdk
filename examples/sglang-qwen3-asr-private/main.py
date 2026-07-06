"""Serve Qwen3-ASR-1.7B (speech→text) with the sglang backend on a private GPU.

The canonical SDK shape — ensure() -> wait_until_ready() -> transcribe -> delete() —
varying one axis vs the other examples: the backend is `sglang` and the task is
**audio2text** (ASR). sglang exposes an OpenAI-compatible
`/v1/audio/transcriptions` endpoint, so a caller uploads an audio file and gets
back a plain `{text}` transcription — the standard OpenAI transcription shape.

Placement is private: the workload is pinned to YOUR registered worker via
`worker_id`. The GPU vendor is inferred from that worker — this example was
verified on an **NVIDIA RTX 4060 Ti**; see the README for the ROCm caveats.

Run:  cp .env.example .env  # fill in real values, then
      python main.py
"""

from __future__ import annotations

import os
import sys

import requests

from inferencekey import (
    ManagementClient,
    WorkloadSpec,
    Backend,
    ExecutionPolicy,
    InferenceKeyError,
)

# --- The model -------------------------------------------------------------
# Qwen3-ASR-1.7B: a 1.7B multilingual ASR model (52 languages), ~4.7 GB of
# safetensors. It fits comfortably in VRAM (verified on a 16 GB 4060 Ti: ~11 GB
# used with the KV budget below). The real sizing constraint is *host RAM*, not
# VRAM — see the README's "Host RAM" note.
MODEL = "Qwen/Qwen3-ASR-1.7B"

# sglang launch command, run verbatim by the `sglang` backend.
#   --trust-remote-code  -> required (the model ships custom code)
#   --mem-fraction-static -> KV cache budget as a fraction of free VRAM. 0.60
#       leaves headroom on a 16 GB card; raise it if the GPU is dedicated to
#       this workload. The worker injects the listen host/port — do not set them.
COMMAND = (
    "sglang serve"
    f" --model-path {MODEL}"
    " --served-model-name qwen3-asr"
    " --trust-remote-code"
    " --mem-fraction-static 0.60"
)

SLUG = "qwen3-asr-sglang"

# A short audio clip to transcribe once the workload is ready. Point this at any
# local wav/mp3/flac/ogg file.
AUDIO_PATH = os.environ.get("IK_AUDIO_PATH", "sample.wav")


def main() -> int:
    # Fail fast on a missing audio file — before provisioning a GPU workload we'd
    # only tear down again. Clear message beats a late FileNotFoundError traceback.
    if not os.path.isfile(AUDIO_PATH):
        print(
            f"audio file not found: {AUDIO_PATH!r}. Drop a wav/mp3/flac/ogg there "
            f"or set IK_AUDIO_PATH to a real file.",
            file=sys.stderr,
        )
        return 1

    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()

    # The worker id comes from the environment — never hard-code it. It's the
    # UUID of your registered worker (Manager UI → Workers → copy the id).
    worker_id = os.environ["IK_WORKER_ID"]

    spec = WorkloadSpec(
        name="Qwen3-ASR 1.7B (sglang)",
        slug=SLUG,
        model=MODEL,
        backend=Backend.SGLANG,          # OpenAI /v1/audio/transcriptions; audio2text
        command=COMMAND,
        task_type="audio2text",
        worker_id=worker_id,
        # Pin the sglang version — Qwen3-ASR support needs a recent one.
        config={"sglang_version": "0.5.14"},
        # Private + fixed: one always-on replica. Stays reserved (and billing)
        # until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
        execution_policy_config={"replicas": 1},
    )

    try:
        # 1) Provision / reconcile. Idempotent by slug.
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # 2) Wait for the cold worker to install sglang, pull the model, and warm
        #    up. The first boot is slow (sglang builds kernels), so give it a
        #    generous timeout.
        print("waiting for the worker to become ready (cold start can take a while)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=1800)

        # 3) Transcribe an audio file via the OpenAI-compatible endpoint. The SDK
        #    has no typed ASR helper yet, so we POST the multipart body directly
        #    with the data-plane key (ik_live_). This is the same call the OpenAI
        #    SDK's audio.transcriptions.create() makes.
        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com")
        project = os.environ["INFERENCEKEY_PROJECT"]
        api_key = os.environ["INFERENCEKEY_API_KEY"]
        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/audio/transcriptions"

        with open(AUDIO_PATH, "rb") as f:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": f},
                data={"model": "qwen3-asr"},
                timeout=120,
            )
        resp.raise_for_status()
        out = resp.json()
        print(f"\ntranscription: {out.get('text', out)}\n")
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"transcription request failed: {e} — {e.response.text[:200]}", file=sys.stderr)
        return 1
    finally:
        # 4) Tear down so the reserved GPU stops billing. Idempotent and safe.
        deleted = mgmt.delete(SLUG)
        print(f"deleted={deleted}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

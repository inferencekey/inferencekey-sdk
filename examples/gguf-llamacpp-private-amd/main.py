"""Serve Gemma 4 26B (GGUF) with the llama.cpp backend on a private AMD R9700.

The canonical SDK shape — ensure() -> wait_until_ready() -> call -> delete() —
varying one axis vs the other examples: the backend is `llamacpp` and the model
is a **GGUF** file served by `llama-server` (not safetensors via vLLM).

Placement is private: the workload is pinned to YOUR registered AMD ROCm worker
(`worker_id` + `gpu_resource_id`). The spec does not carry the GPU vendor/arch —
that is inferred from the worker you pin to (an R9700 / gfx120x / RDNA4 box).

Run:  cp .env.example .env  # fill in real values, then
      python main.py
"""

from __future__ import annotations

import os
import sys

from inferencekey import (
    ManagementClient,
    DataClient,
    WorkloadSpec,
    Backend,
    ExecutionPolicy,
    InferenceKeyError,
)

# --- The model -------------------------------------------------------------
# Public, no-token GGUF of Gemma 4 26B (QAT q4_0). ~14.4 GB — fits the R9700's
# 32 GB with room for a large KV cache. llama-server fetches it straight from
# the HF Hub via `-hf <repo> --hf-file <file>` (no manual download, no venv).
MODEL = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
GGUF_FILE = "gemma-4-26B_q4_0-it.gguf"

# llama-server launch command, run verbatim by the `llamacpp` backend:
#   -ngl 99  -> offload all layers to the GPU
#   -c 8192  -> context window (this q4 fits up to 256K on the R9700; kept
#               modest here so the example is quick — raise as you like)
# The worker injects the listen port; do not hard-code --port.
COMMAND = f"llama-server -hf {MODEL} --hf-file {GGUF_FILE} -ngl 99 -c 8192"

SLUG = "gemma4-26b-llamacpp-amd"


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()

    # The worker id comes from the environment — never hard-code it. It's the
    # UUID of your registered worker (Manager UI → Workers → copy the id).
    worker_id = os.environ["IK_WORKER_ID"]

    spec = WorkloadSpec(
        name="Gemma 4 26B (llama.cpp / GGUF) on R9700",
        slug=SLUG,
        model=MODEL,
        backend=Backend.LLAMACPP,        # GGUF via llama-server; text2text only
        command=COMMAND,
        task_type="text2text",
        # Pin to your AMD ROCm box. The vendor/arch (RDNA4) lives on the worker.
        # gpu_resource_id is optional — only needed to target a specific GPU on a
        # multi-GPU worker; the R9700 has one GPU, so we omit it.
        worker_id=worker_id,
        # Private + fixed: one always-on replica. It stays reserved (and billing)
        # until delete() runs — see "Cost & cleanup" in the README.
        execution_policy=ExecutionPolicy.FIXED,
        execution_policy_config={"replicas": 1},
    )

    try:
        # 1) Provision / reconcile. Idempotent by slug.
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # 2) Wait for the cold worker to pull the image and serve. The first
        #    boot pulls the ROCm image and warms up Gemma 4 (a slow graph build
        #    on llama.cpp), so give it a generous timeout.
        print("waiting for the worker to become ready (cold start can take a while)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=1800)

        # 3) Call the OpenAI-compatible endpoint (data plane, ik_live_).
        data = DataClient.from_env()
        ep = data.endpoint(ref.workload_slug)
        out = ep.generate_text(
            prompt="In one sentence, what makes GGUF a good fit for AMD GPUs?",
            temperature=0.2,
            max_tokens=120,
        )
        print(f"\nmodel: {out.model}\n{out.text}\n")
    except InferenceKeyError as e:
        # Don't print tokens; surface a clean message.
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        # 4) Tear down so the reserved GPU stops billing. Idempotent and safe
        #    to call even if ensure() never succeeded.
        deleted = mgmt.delete(SLUG)
        print(f"deleted={deleted}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Serve Qwen3-Embedding-8B (GGUF) with the llama.cpp backend on a private AMD R9700.

The canonical SDK shape — ensure() -> wait_until_ready() -> call -> delete() —
varying two axes vs the `gguf-llamacpp-private-amd` example: the modality is
`embedding` (not `text2text`) and the endpoint call is `embed()` (not
`generate_text()`). The backend is still `llamacpp` serving a **GGUF** file with
`llama-server`, but launched in embedding mode — the `command` carries
`--embedding --pooling last`, which puts llama-server on the `/v1/embeddings`
route the platform proxies to.

Why llama.cpp for embeddings here: on an AMD RDNA4 box (R9700 / gfx1201) sglang
is unsupported, so llamacpp is the only embedding runtime available on this
worker. The platform only allows `(embedding, llamacpp)` since the
`embedding_llamacpp_combo` migration.

Placement is private: the workload is pinned to YOUR registered AMD ROCm worker
(`worker_id`). The spec does not carry the GPU vendor/arch — that is inferred
from the worker you pin to (an R9700 / gfx1201 / RDNA4 box).

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
    TaskType,
    ExecutionPolicy,
    InferenceKeyError,
)

# --- The model -------------------------------------------------------------
# Public GGUF of Qwen3-Embedding-8B. The Q8_0 quant is ~8 GB — trivially fits
# the R9700's 32 GB, leaving plenty of headroom (embeddings are memory-light vs
# a chat KV cache). llama-server fetches it straight from the HF Hub via
# `-hf <repo> --hf-file <file>` (no manual download, no venv). Bump to
# `Qwen3-Embedding-8B-F16.gguf` (~15 GB) for maximum fidelity if you need it.
MODEL = "Qwen/Qwen3-Embedding-8B-GGUF"
GGUF_FILE = "Qwen3-Embedding-8B-Q8_0.gguf"

# llama-server launch command, run verbatim by the `llamacpp` backend:
#   --embedding      -> serve the /v1/embeddings route (embedding mode)
#   --pooling last   -> Qwen3-Embedding pools the LAST token (per the model card);
#                       using the wrong pooling silently degrades vector quality
#   -ub 8192         -> micro-batch size the model card recommends for the server
#   -ngl 99          -> offload all layers to the GPU
# The worker injects the listen port; do not hard-code --port.
COMMAND = (
    f"llama-server -hf {MODEL} --hf-file {GGUF_FILE} "
    "--embedding --pooling last -ub 8192 -ngl 99"
)

SLUG = "qwen3-embedding-8b-llamacpp-amd"


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()

    # The worker id comes from the environment — never hard-code it. It's the
    # UUID of your registered worker (Manager UI → Workers → copy the id).
    worker_id = os.environ["IK_WORKER_ID"]

    spec = WorkloadSpec(
        name="Qwen3-Embedding-8B (llama.cpp / GGUF) on R9700",
        slug=SLUG,
        model=MODEL,
        backend=Backend.LLAMACPP,        # GGUF via llama-server
        command=COMMAND,
        task_type=TaskType.EMBEDDING,    # /v1/embeddings, not chat
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
        #    boot pulls the ROCm image and loads the GGUF, so give it a
        #    generous timeout. Embeddings warm up faster than a chat model
        #    (no long graph build), but the image pull dominates on a cold box.
        print("waiting for the worker to become ready (cold start can take a while)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=1800)

        # 3) Call the OpenAI-compatible endpoint (data plane, ik_live_).
        data = DataClient.from_env()
        ep = data.endpoint(ref.workload_slug)
        res = ep.embed(
            input=[
                "The quick brown fox jumps over the lazy dog.",
                "A fast auburn fox leaps above a sleepy hound.",
                "InferenceKey serves embeddings on your own GPUs.",
            ]
        )
        # One vector per input; report shape + a cosine similarity to prove the
        # vectors are meaningful (the two paraphrases should be closer than the
        # unrelated sentence).
        dim = len(res.embeddings[0]) if res.embeddings else 0
        print(f"\nmodel: {res.model}")
        print(f"got {len(res.embeddings)} vectors of dimension {dim}")
        sim_paraphrase = _cosine(res.embeddings[0], res.embeddings[1])
        sim_unrelated = _cosine(res.embeddings[0], res.embeddings[2])
        print(f"cosine(paraphrase)  = {sim_paraphrase:.3f}")
        print(f"cosine(unrelated)   = {sim_unrelated:.3f}")
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


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors, pure-Python (no numpy dep)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


if __name__ == "__main__":
    raise SystemExit(main())

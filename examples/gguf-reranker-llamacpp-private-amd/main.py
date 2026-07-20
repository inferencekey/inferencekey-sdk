"""Serve Qwen3-Reranker-0.6B (GGUF) with the llama.cpp backend on a private AMD R9700.

The canonical SDK shape — ensure() -> wait_until_ready() -> call -> delete() —
for a **reranker**: given a query and a list of documents, the model scores how
relevant each document is to the query and returns them ranked highest-first.

The backend is `llamacpp` serving a GGUF with `llama-server`, launched in rerank
mode: the `command` carries `--reranking --pooling rank`, which puts the server
on the `/v1/rerank` route the platform proxies to. You call it with
`endpoint.rerank(query=..., documents=[...])`.

Why llama.cpp for reranking here: on an AMD RDNA4 box (R9700 / gfx1201) sglang is
unsupported, so llamacpp is the only reranking runtime available on this worker.
The platform allows `(reranker, llamacpp)` since the `reranker_llamacpp_combo`
migration.

Placement is private: the workload is pinned to YOUR registered AMD ROCm worker
(`worker_id`). The GPU vendor/arch (RDNA4) is inferred from the worker.

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
# Public GGUF of Qwen3-Reranker-0.6B, converted by the llama.cpp team (ggml-org).
# The Q8_0 quant is ~0.6 GB — tiny; it fits the R9700's 32 GB with room to spare
# and reranking is memory-light (no chat KV cache). llama-server fetches it from
# the HF Hub via `-hf <repo> --hf-file <file>` (no manual download, no venv).
MODEL = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF"
GGUF_FILE = "qwen3-reranker-0.6b-q8_0.gguf"

# llama-server launch command, run verbatim by the `llamacpp` backend:
#   --reranking    -> serve the /v1/rerank route (rerank mode)
#   --pooling rank -> reranker models score with the rank pooling head; using the
#                     wrong pooling silently degrades relevance scores
#   -ngl 99        -> offload all layers to the GPU
# The worker injects the listen port; do not hard-code --port.
COMMAND = (
    f"llama-server -hf {MODEL} --hf-file {GGUF_FILE} "
    "--reranking --pooling rank -ngl 99"
)

SLUG = "qwen3-reranker-0.6b-llamacpp-amd"


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()

    # The worker id comes from the environment — never hard-code it.
    worker_id = os.environ["IK_WORKER_ID"]

    spec = WorkloadSpec(
        name="Qwen3-Reranker-0.6B (llama.cpp / GGUF) on R9700",
        slug=SLUG,
        model=MODEL,
        backend=Backend.LLAMACPP,        # GGUF via llama-server
        command=COMMAND,
        task_type=TaskType.RERANKER,     # /v1/rerank, not chat/embeddings
        worker_id=worker_id,
        # Private + fixed: one always-on replica, reserved until delete().
        execution_policy=ExecutionPolicy.FIXED,
        execution_policy_config={"replicas": 1},
    )

    try:
        # 1) Provision / reconcile. Idempotent by slug.
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # 2) Wait for the cold worker to pull the image and load the GGUF.
        #    The image pull dominates a cold boot; the 0.6B model itself loads
        #    fast. Give it a generous timeout anyway.
        print("waiting for the worker to become ready (cold start can take a while)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=1800)

        # 3) Call the /v1/rerank endpoint (data plane, ik_live_).
        data = DataClient.from_env()
        ep = data.endpoint(ref.workload_slug)
        query = "What is the capital of France?"
        documents = [
            "The Eiffel Tower is a famous landmark.",
            "Paris is the capital and most populous city of France.",
            "Pandas are bears native to China.",
            "France is a country in Western Europe; its government sits in Paris.",
        ]
        res = ep.rerank(query=query, documents=documents)

        print(f"\nmodel: {res.model}")
        print(f"query: {query}\n")
        print("ranked documents (best first):")
        for rank, item in enumerate(res.results, start=1):
            print(f"  {rank}. score={item.relevance_score:.4f}  {documents[item.index]!r}")
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        # 4) Tear down so the reserved GPU stops billing. Idempotent.
        deleted = mgmt.delete(SLUG)
        print(f"\ndeleted={deleted}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# GGUF embeddings + llama.cpp on a private AMD R9700 — serve Qwen3-Embedding-8B

Serves **Qwen3-Embedding-8B (GGUF, q8)** with the **`llamacpp`** backend on a
private **AMD R9700 (gfx1201 / RDNA4)** worker, as an **`embedding`** workload.
Two axes vary vs [`gguf-llamacpp-private-amd`](../gguf-llamacpp-private-amd): the
modality is `embedding` (not `text2text`) and you call `embed()` (not
`generate_text()`). It's still a GGUF served by `llama-server` — the difference
is the `--embedding --pooling last` flags that put the server on the
`/v1/embeddings` route.

## Why llama.cpp for embeddings here
On an AMD **RDNA4** box (R9700 / gfx1201) **sglang is unsupported** and vLLM
can't serve these checkpoints, so `llamacpp` is the **only** embedding runtime
available on this worker — the same reason these nodes already run llamacpp for
text generation. The platform allows `(embedding, llamacpp)` since the
`embedding_llamacpp_combo` migration; on other hardware `sglang` and `ollama`
also serve embeddings.

## Compatibility
- SDK: local build in this repo (or `>= 0.1.0` once published)
- Language: Python >= 3.9
- Placement: private (AMD ROCm, R9700 / gfx1201 / RDNA4)
- Backend: `llamacpp`   Modality: `embedding`   Policy: `fixed`

## Prerequisites
- The tokens/ids this example needs — see
  [tokens & placement](../README.md#tokens). Specifically:
  `INFERENCEKEY_SDK_TOKEN`, `INFERENCEKEY_API_KEY`, `INFERENCEKEY_PROJECT`,
  and the **private-worker** id `IK_WORKER_ID` (the worker's UUID — Manager UI →
  Workers → copy the id). `gpu_resource_id` is optional and omitted here (it
  only targets a specific GPU on a multi-GPU worker; the R9700 has one).
- Python >= 3.9.
- A **registered AMD ROCm worker** with an **R9700 (gfx1201)** GPU, running the
  ROCm base image with `llama-server` available. See the worker's
  [ROCm wheels & image docs](../../../../docs/amd-rocm-wheels-vllm.md) and the
  [llama.cpp ROCm runbook](../../../../docs/llamacpp-gemma4-rocm-runbook.md).
- **VRAM:** the q8 GGUF is ~8 GB and fits the R9700's 32 GB with huge headroom
  (embeddings don't grow a chat KV cache). Bump `GGUF_FILE` to the F16 (~15 GB)
  for maximum fidelity if you want it.

## Run
```bash
cp .env.example .env        # fill in real values
# install the local SDK — see ../CONTRIBUTING.md#4-depending-on-the-sdk:
python -m venv .venv && . .venv/bin/activate
pip install maturin
( cd ../../bindings/python && maturin develop --release )
python main.py
```

## What it does
- **ensure()** — declares the workload (`backend=Backend.LLAMACPP`,
  `task_type=TaskType.EMBEDDING`,
  `command="llama-server -hf … --hf-file … --embedding --pooling last -ub 8192 -ngl 99"`)
  pinned to your R9700 via `worker_id`, `fixed` policy with 1 replica.
  Idempotent by `slug`.
- **wait_until_ready()** — blocks while the cold worker pulls the ROCm image and
  `llama-server` loads the GGUF. Timeout is **1800 s** to cover the image pull;
  the embedding model itself warms up fast (no long graph build).
- **call** — one `embed()` over three sentences against the OpenAI-compatible
  `/v1/embeddings` endpoint, then prints the vector dimension and two cosine
  similarities (the two paraphrases should score higher than the unrelated pair).
- **delete()** — tears the workload down on exit.

## The pooling flag matters
Qwen3-Embedding pools the **last** token (`--pooling last`, per the model card).
Serving it with the wrong pooling (`mean`/`cls`) produces vectors that look
valid — right shape, no error — but score poorly on similarity. If your
retrieval quality is off, check this flag first.

## Cost & cleanup
This is a **private worker with a `fixed` policy**: one replica stays **always on
and billing** until it is deleted — the GPU is reserved the whole time, it does
**not** scale to zero. `main.py` calls **`delete()` in a `finally`** so it tears
down on success, error, and Ctrl-C. If a run is killed before `finally` executes,
delete it manually: re-run the example, or call
`mgmt.delete("qwen3-embedding-8b-llamacpp-amd")`.

## Troubleshooting
- **`KeyError: 'IK_WORKER_ID'`** — you didn't set the worker id. Copy
  `.env.example` to `.env` and fill it from Manager UI → Workers (copy the id).
- **`task_type 'embedding' does not allow backend 'llamacpp'`** — your Manager
  predates the `embedding_llamacpp_combo` migration. Update/redeploy the Manager.
- **`worker_id must belong to the same project`** — the worker isn't assigned to
  `INFERENCEKEY_PROJECT`. Assign it to that project in the Manager first.
- **Readiness times out** — usually the cold image pull. The example already
  uses a 1800 s timeout; if your network is slow pulling the image, raise it.
- **`platform_unsupported` / backend won't start** — make sure the worker is the
  **AMD ROCm gfx1201** box. `sglang` is unsupported on RDNA4; this example uses
  `llamacpp`, which is supported.
- **Similarities look random** — wrong pooling. Confirm `--pooling last` is in
  the `command` (see "The pooling flag matters" above).

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

# GGUF reranking + llama.cpp on a private AMD R9700 — serve Qwen3-Reranker-0.6B

Serves **Qwen3-Reranker-0.6B (GGUF, q8)** with the **`llamacpp`** backend on a
private **AMD R9700 (gfx1201 / RDNA4)** worker, as a **`reranker`** workload.
Given a query and a list of documents, the model scores each document's relevance
to the query and returns them ranked highest-first. It's a GGUF served by
`llama-server --reranking --pooling rank` (the flags that put the server on the
`/v1/rerank` route), and you call it with `endpoint.rerank(query=…, documents=[…])`.

## Why llama.cpp for reranking here
On an AMD **RDNA4** box (R9700 / gfx1201) **sglang is unsupported**, so `llamacpp`
is the **only** reranking runtime available on this worker. The platform allows
`(reranker, llamacpp)` since the `reranker_llamacpp_combo` migration; on other
hardware `sglang` also serves reranking.

## Compatibility
- SDK: local build in this repo (or `>= 0.1.0` once published)
- Language: Python >= 3.9
- Placement: private (AMD ROCm, R9700 / gfx1201 / RDNA4)
- Backend: `llamacpp`   Modality: `reranker`   Policy: `fixed`

## Prerequisites
- The tokens/ids this example needs — see
  [tokens & placement](../README.md#tokens): `INFERENCEKEY_SDK_TOKEN`,
  `INFERENCEKEY_API_KEY`, `INFERENCEKEY_PROJECT`, and the **private-worker** id
  `IK_WORKER_ID` (the worker's UUID — Manager UI → Workers → copy the id).
- Python >= 3.9.
- A **registered AMD ROCm worker** with an **R9700 (gfx1201)** GPU running the
  ROCm base image with `llama-server` available. See the worker's
  [ROCm wheels & image docs](../../../../docs/amd-rocm-wheels-vllm.md) and the
  [llama.cpp ROCm runbook](../../../../docs/llamacpp-gemma4-rocm-runbook.md).
- **VRAM:** the q8 GGUF is only ~0.6 GB — a 0.6B model — so it fits with room to
  spare. Reranking is memory-light (no chat KV cache).

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
  `task_type=TaskType.RERANKER`,
  `command="llama-server -hf … --hf-file … --reranking --pooling rank -ngl 99"`)
  pinned to your R9700 via `worker_id`, `fixed` policy with 1 replica.
- **wait_until_ready()** — blocks while the cold worker pulls the ROCm image and
  loads the GGUF (1800 s timeout to cover the image pull; the model loads fast).
- **call** — one `rerank()` over a query + four documents against the
  OpenAI-style `/v1/rerank` route, then prints the documents ranked by relevance
  score (the two Paris/France documents should rank above the unrelated ones).
- **delete()** — tears the workload down on exit.

## The pooling flag matters
Reranker models score with a **rank** pooling head (`--pooling rank`), and the
server must be in rerank mode (`--reranking`). Serving with the wrong pooling
produces scores that look valid but rank poorly. If your ranking is off, check
these flags first.

## Cost & cleanup
This is a **private worker with a `fixed` policy**: one replica stays **always on
and billing** until it is deleted — the GPU is reserved the whole time. `main.py`
calls **`delete()` in a `finally`** so it tears down on success, error, and
Ctrl-C. If a run is killed before `finally` executes, delete it manually: re-run
the example, or call `mgmt.delete("qwen3-reranker-0.6b-llamacpp-amd")`.

## Troubleshooting
- **`KeyError: 'IK_WORKER_ID'`** — you didn't set the worker id. Copy
  `.env.example` to `.env` and fill it from Manager UI → Workers.
- **`task_type 'reranker' does not allow backend 'llamacpp'`** — your Manager
  predates the `reranker_llamacpp_combo` migration. Update/redeploy the Manager.
- **`worker_id must belong to the same project`** — assign the worker to
  `INFERENCEKEY_PROJECT` in the Manager first.
- **Readiness times out** — usually the cold image pull; raise the timeout if
  your network is slow.
- **`platform_unsupported`** — make sure the worker is the **AMD ROCm gfx1201**
  box. `sglang` is unsupported on RDNA4; this example uses `llamacpp`.
- **Ranking looks random** — wrong pooling. Confirm `--reranking --pooling rank`
  are in the `command` (see "The pooling flag matters" above).

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

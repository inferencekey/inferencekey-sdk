# GGUF + llama.cpp on a private AMD R9700 — serve Gemma 4 26B from a GGUF file

Serves **Gemma 4 26B (GGUF, q4)** with the **`llamacpp`** backend on a private
**AMD R9700 (gfx120x / RDNA4)** worker. The one axis this varies vs the other
examples: the backend is `llamacpp` and the model is a GGUF served by
`llama-server` — the `command` points at a GGUF repo/file, no venv, no
safetensors.

## Compatibility
- SDK: local build in this repo (or `>= 0.1.0` once published)
- Language: Python >= 3.9
- Placement: private (AMD ROCm, R9700 / gfx120x / RDNA4)
- Backend: `llamacpp`   Policy: `fixed`

## Prerequisites
- The tokens/ids this example needs — see
  [tokens & placement](../README.md#tokens). Specifically:
  `INFERENCEKEY_SDK_TOKEN`, `INFERENCEKEY_API_KEY`, `INFERENCEKEY_PROJECT`,
  and the **private-worker** id `IK_WORKER_ID` (the worker's UUID — Manager UI →
  Workers → copy the id). `gpu_resource_id` is optional and omitted here (it
  only targets a specific GPU on a multi-GPU worker; the R9700 has one).
- Python >= 3.9.
- A **registered AMD ROCm worker** with an **R9700 (gfx120x)** GPU, running the
  ROCm gfx120x base image. See the worker's
  [ROCm wheels & image docs](../../../../docs/amd-rocm-wheels-vllm.md) and the
  [llama.cpp Gemma 4 runbook](../../../../docs/llamacpp-gemma4-rocm-runbook.md).
- **VRAM:** the q4 GGUF is ~14.4 GB and fits the R9700's 32 GB with room for a
  large KV cache. (The 31B q4 needs more — keep to 26B here.)

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
  `command="llama-server -hf … --hf-file … -ngl 99 -c 8192"`) pinned to your
  R9700 via `worker_id` + `gpu_resource_id`, `fixed` policy with 1 replica.
  Idempotent by `slug`.
- **wait_until_ready()** — blocks while the cold worker pulls the ROCm image and
  `llama-server` warms up Gemma 4 (a slow graph build on llama.cpp). Timeout is
  raised to **1800 s** because that warmup is long on this model.
- **call** — one non-streaming completion against the OpenAI-compatible endpoint.
- **delete()** — tears the workload down on exit.

## Cost & cleanup
This is a **private worker with a `fixed` policy**: one replica stays **always on
and billing** until it is deleted — the GPU is reserved the whole time, it does
**not** scale to zero. `main.py` calls **`delete()` in a `finally`** so it tears
down on success, error, and Ctrl-C. If a run is killed before `finally` executes,
delete it manually: re-run the example, or call `mgmt.delete("gemma4-26b-llamacpp-amd")`.

## Troubleshooting
- **`KeyError: 'IK_WORKER_ID'`** — you didn't set the worker id. Copy
  `.env.example` to `.env` and fill it from Manager UI → Workers (copy the id).
- **`worker_id must belong to the same project`** — the worker isn't assigned to
  `INFERENCEKEY_PROJECT`. Assign it to that project in the Manager first.
- **Readiness times out** — Gemma 4's first warmup on llama.cpp is slow (the
  worker can sit "loading" for many minutes with the GPU at ~100%). The example
  already uses a 1800 s timeout; if your network is slow pulling the image,
  raise it further.
- **`platform_unsupported` / backend won't start** — make sure the worker is the
  **AMD ROCm gfx120x** box. `sglang` is unsupported on RDNA4; this example uses
  `llamacpp`, which is supported.
- **OOM on the GPU** — lower the context in `COMMAND` (`-c 8192` → `-c 4096`), or
  confirm nothing else is holding VRAM on the R9700.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

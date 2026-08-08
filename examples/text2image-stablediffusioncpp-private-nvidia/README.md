# stable-diffusion.cpp on a private NVIDIA GPU — generate an image with Z-Image Turbo

Generates a **1024×1024 image** with the **`stablediffusioncpp`** backend on a
private **NVIDIA** worker. The one axis this varies vs the other examples: the
backend is `stablediffusioncpp` and the task is **text2image**, served by
`sd-server`. Because stable-diffusion.cpp compiles its CUDA kernels for the local
GPU arch (like llama.cpp), it runs on **older GPUs that sglang does not support**
— e.g. Volta (Tesla V100, sm_70).

## Compatibility
- SDK: local build in this repo (or `>= 0.1.0` once published)
- Language: Python >= 3.9
- Placement: private (NVIDIA CUDA; validated on 2× Tesla V100 / sm_70)
- Backend: `stablediffusioncpp`   Policy: `fixed`

## Prerequisites
- The tokens/ids this example needs — see
  [tokens & placement](../README.md#tokens). Specifically:
  `INFERENCEKEY_SDK_TOKEN`, `INFERENCEKEY_API_KEY`, `INFERENCEKEY_PROJECT`,
  and the **private-worker** id `IK_WORKER_ID` (the worker's UUID — Manager UI →
  Workers → copy the id).
- Python >= 3.9.
- A **registered NVIDIA worker** with a CUDA GPU. On first boot the worker
  compiles stable-diffusion.cpp for the GPU's compute capability
  (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<arch>`) unless a prebuilt
  `sd-server` is already present; on low-RAM nodes that compile needs swap, and
  the runtime needs `libgomp1`.
- **The three model files** reachable inside the worker container, at the paths
  in the command (override via the `IK_SDCPP_*` env vars):
  - diffusion model — `z_image_turbo-Q8_0.gguf` (~6.6 GB)
  - VAE — the FLUX `ae.safetensors` (~335 MB)
  - text encoder — `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (~2.5 GB)
- **VRAM:** the weights total ~9.9 GB and the VAE decode at 1024×1024 needs
  ~6.6 GB of scratch. `--auto-fit` spreads the modules across the GPUs so this
  fits on 2× 16 GB V100s; on a single 16 GB GPU the VAE decode can OOM.

## Run
```bash
cp .env.example .env        # fill in real values
# install the local SDK — see ../CONTRIBUTING.md#4-depending-on-the-sdk:
python -m venv .venv && . .venv/bin/activate
pip install maturin requests
( cd ../../bindings/python && maturin develop --release )
python main.py
```

The script writes the generated image to `output.png` in this folder.

## What it does
- **ensure()** — declares the workload (`backend=Backend.STABLE_DIFFUSION_CPP`,
  `command="sd-server --diffusion-model … --vae … --llm … --cfg-scale 1.0
  --steps 8 --fa --auto-fit -H 1024 -W 1024"`, `task_type="text2image"`) pinned
  to your NVIDIA worker via `worker_id`, `fixed` policy with 1 replica.
  Idempotent by `slug`.
- **wait_until_ready()** — blocks while the cold worker builds sd.cpp (CUDA
  compile, unless prebuilt) and loads the three models. Timeout is raised to
  **1800 s** because that first boot is long.
- **call** — POSTs an OpenAI-style image request to the workload's
  `/v1/images/generations` endpoint (the SDK has no typed image helper yet, so
  we use `requests` directly — same pattern as the forecast example), then
  decodes the `b64_json` and writes `output.png`.
- **delete()** — tears the workload down on exit.

## Cost & cleanup
This is a **private worker with a `fixed` policy**: one replica stays **always on
and billing** until it is deleted — the GPU is reserved the whole time, it does
**not** scale to zero. `main.py` calls **`delete()` in a `finally`** so it tears
down on success, error, and Ctrl-C. If a run is killed before `finally` executes,
delete it manually: re-run the example, or call `mgmt.delete("sdcpp-zimage")`.

## Troubleshooting
- **`KeyError: 'IK_WORKER_ID'`** — you didn't set the worker id. Copy
  `.env.example` to `.env` and fill it from Manager UI → Workers (copy the id).
- **`worker_id must belong to the same project`** — the worker isn't assigned to
  `INFERENCEKEY_PROJECT`. Assign it to that project in the Manager first.
- **`unknown argument: --port`** — you pinned an old worker binary that injects
  `--port`; sd-server uses `--listen-port`. Update the worker to a build that
  supports the `stablediffusioncpp` backend.
- **Readiness times out** — the first CUDA compile of sd.cpp plus loading three
  models is slow (and slower still on nodes with a narrow PCIe link). The example
  already uses a 1800 s timeout; raise it further if your node is slow.
- **`cudaMalloc failed: out of memory` on the last step** — the VAE decode needs
  a large scratch buffer on top of the weights. Keep `--auto-fit` (it spreads the
  load across GPUs); a single 16 GB GPU may not fit 1024×1024 — drop `-H/-W` or
  add a second GPU.
- **Command rejected / does not start** — the command must start with
  `sd-server` (the HTTP server), not `sd-cli` (the one-shot CLI, no HTTP).

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

# Iris-3B text-to-image (custom backend, private NVIDIA): a pixel-space image model with no VAE

Serves [Iris-3B](https://huggingface.co/speridlabs/iris-3b) from Sperid Labs
(Apache-2.0), a 3B text-to-image transformer that paints every pixel directly:
no VAE, no latent space. No built-in backend serves it (it is not a diffusers
pipeline and not a GGUF), so this example ships a **custom Python backend** that
drives the authors' own PyTorch code, publishes it to your tenant and deploys it
on your NVIDIA worker. The axis it varies: `backend` is a custom backend slug,
and the default checkpoint is the **4-step Turbo** one.

## Compatibility
- SDK: local build in this repo (see [CONTRIBUTING](../CONTRIBUTING.md#4-depending-on-the-sdk))
- Language: Python >= 3.9 for the client scripts; the backend itself runs on
  Python >= 3.11 (the worker provisions it with `uv`)
- Placement: private (NVIDIA, CUDA 13 driver, about 22 GB of VRAM for Turbo)
- Backend: custom (`iris3b-text2image`, or `iris3b-base-text2image`)   Policy: `fixed`

## Prerequisites
- An SDK token (`ik_sdk_`) and an API key (`ik_live_`) for the project, see
  [Tokens](../README.md#tokens).
- A registered NVIDIA worker assigned to the project (`IK_WORKER_ID`).
- To publish the backend from the script: your tenant id and a dashboard
  session token. Or upload `dist/*.tar.gz` from Dashboard -> Custom backends.

## Run
```bash
cp .env.example .env        # fill in real values
python -m venv .venv && . .venv/bin/activate
pip install maturin requests
( cd ../../bindings/python && maturin develop --release )
python publish.py           # once: packages backend.py and uploads it
python main.py              # ensure -> wait until ready -> 2 images in ./outputs
```

Base model instead of Turbo: `python publish.py --variant base`, then
`IRIS_VARIANT=base python main.py`.

## What it does
1. `publish.py` packages `backend.py` + `requirements.txt` (5 KB: code only,
   no weights) and registers it as a custom backend of your tenant.
2. `main.py` calls `ensure()` with `backend="iris3b-text2image"` and
   `task_type="text2image"`, pinned to your worker with a `fixed` policy.
3. On first start the worker creates the backend venv (torch, transformers),
   the backend downloads the pinned upstream source (about 1 MB), the Turbo
   weights (about 6 GB) and the Qwen3-VL-4B text encoder (about 9 GB), and
   loads both on the GPU.
4. `wait_until_ready()` returns, and the script POSTs two OpenAI-style requests
   to `/v1/images/generations` and saves the PNGs.

Request fields: `prompt` (required), `size` (`"1024x1024"`, `"1344x752"`, ...;
Turbo snaps to its nearest training bucket), `seed`, `n` (1 to 4), and
`output_format` (`png` by default, `jpeg` or `webp`). The base variant also reads
`steps` (default 100), `guidance_scale` (default 3) and `negative_prompt`.

## Measured (RunPod NVIDIA A40 48 GB, 2026-10-12)

| | Turbo (4 steps) | Base (100 steps, CFG 3) |
| --- | --- | --- |
| `ensure()` to first image possible, cold pod | 130 s (venv 16 s, weights 12 s, text encoder download + load 100 s) | 60 s (text encoder already cached; weights 43 s, load 11 s) |
| GPU time, 1024x1024 | 1.96 s | 90 s |
| GPU time, 1344x896 | 2.2 to 2.8 s | 106 s |
| Batch of 4 at 1024x1024 (`output_format: jpeg`) | 7.0 s (1.75 s per image) | not measured |
| VRAM (`nvidia-smi`) | 20.4 GB loaded, 21.8 GB peak | about 20 GB more on the same GPU (both loaded: 40.5 GB, peak 44.9 GB) |

The pod cost 0.59 USD per hour; the whole session (both variants, more than 40
images and a demo recording) ran 26 minutes, 0.26 USD.

## Cost & cleanup
`fixed` keeps one replica on your worker until you delete the workload. The
script deletes it on exit (also on error and Ctrl-C). With `IK_KEEP_ALIVE=1` it
keeps it, so you can try more prompts from the dashboard image tester; then
delete it with `ManagementClient.from_env().delete("iris3b-turbo")`. If the GPU
is a rented pod, also stop the pod: deleting the workload does not.

## Troubleshooting
- **504 `inference_timeout` after 5 minutes with `n` > 1 and PNG**: the result
  went over the 8 MB message limit and the worker dropped it. This backend now
  answers 400 for that case; ask for `output_format: "jpeg"` or `"webp"`.
- **`wait_until_ready()` returns but the first request takes a minute**: with a
  second workload on the same worker, readiness can be reported early; the
  request waits in the queue until the backend is up, it is not lost.
- **Slow first start**: set `HF_TOKEN` on the worker for faster Hugging Face
  downloads; later starts reuse the cache (a re-published backend restarted in
  23 s).
- **CUDA errors at load**: torch from PyPI is the CUDA 13 build; the host driver
  must support CUDA 13 (driver 580 or newer).

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

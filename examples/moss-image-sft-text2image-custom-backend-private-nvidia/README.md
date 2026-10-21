# MOSS-Image-SFT text-to-image (custom backend, private NVIDIA): text, infographics and JSON layouts

Serves [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) from
OpenMOSS (Apache-2.0), a 5B single-stream diffusion transformer with a
Qwen3.5-9B text encoder and the FLUX.2 small VAE. It is good at putting words in
pictures (posters, menus, infographics, handwritten Chinese) and accepts a
**JSON layout** prompt with a bounding box per text block or object.

The authors ship it as a diffusers *custom pipeline*
(`custom_pipeline="pipeline_moss_image"`, `trust_remote_code=True`), which no
built-in backend loads, so this example ships a **custom Python backend**,
publishes it to your tenant and deploys it on your NVIDIA worker. One loaded
model answers two kinds of request:

- `"mode": "sft"` (default): the release recipe, 50 Euler steps, CFG 3, block 22
  skipped in the negative branch.
- `"mode": "turbo"`: the community
  [8-step DMD LoRA by lodestones](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd)
  (`lora_latest.safetensors`, strength 0.7) applied unmerged, 8 steps, no CFG.
  A request with no `mode`, `guidance_scale` 1 and 16 steps or fewer also runs
  in turbo, so the dashboard image tester can use it.

## Compatibility
- SDK: local build in this repo (see [CONTRIBUTING](../CONTRIBUTING.md#4-depending-on-the-sdk))
- Language: Python >= 3.9 for the client scripts; the backend runs on Python
  >= 3.10 (the worker provisions it with `uv`)
- Placement: private (NVIDIA, CUDA 13 driver, about 30 GB of VRAM: a 40 GB or
  48 GB card)
- Backend: custom (`moss-image-text2image`)   Policy: `fixed`

## Prerequisites
- An SDK token (`ik_sdk_`) and an API key (`ik_live_`) for the project, see
  [Tokens](../README.md#tokens).
- A registered NVIDIA worker assigned to the project (`IK_WORKER_ID`).
- To publish the backend from the script: your tenant id and a dashboard
  session token. Or upload `dist/*.tar.gz` from Dashboard -> Custom backends.
- The first start downloads about 28 GB and then loads it in `setup()`. The
  worker waits 300 s for a custom backend by default; on a slow link raise
  `INFERENCEKEY_CUSTOM_BACKEND_READY_TIMEOUT_SECS` in the worker's environment.

## Run
```bash
cp .env.example .env        # fill in real values
python -m venv .venv && . .venv/bin/activate
pip install maturin requests
( cd ../../bindings/python && maturin develop --release )
python publish.py           # once: packages backend.py and uploads it
python main.py              # ensure -> wait until ready -> 3 images in ./outputs
```

## What it does
1. `publish.py` packages `backend.py` + `requirements.txt` (about 6 KB: code
   only, no weights) and registers it as a custom backend of your tenant.
2. `main.py` calls `ensure()` with `backend="moss-image-text2image"` and
   `task_type="text2image"`, pinned to your worker with a `fixed` policy.
3. On first start the worker creates the backend venv (torch, diffusers 0.38,
   transformers 5.3); `setup()` downloads the pinned model revision (bf16
   transformer 10.2 GB, Qwen3.5-9B text encoder 16.8 GB, VAE 0.25 GB) and the
   1.2 GB LoRA, then loads everything on the GPU.
4. The script POSTs three OpenAI-style requests to `/v1/images/generations`: a
   poster with four lines of text (SFT), the same poster in turbo mode, and a
   JSON layout prompt. PNGs land in `./outputs`.

Request fields: `prompt` (required: a string, or a layout object that the
backend serializes with `ensure_ascii=False`), `size` (`"1024x1024"`,
`"1344x768"`, ...; multiples of 16, up to 4 megapixels), `seed`, `mode`,
`steps` or `num_inference_steps`, `guidance_scale`, `negative_prompt`,
`lora_strength` (turbo), `n` (1 for PNG, up to 4 with `output_format` `jpeg` or
`webp`: results travel through an 8 MB message limit). The response adds
`seconds` (GPU time) and `vram_peak_gib`.

Layout prompts follow the [user guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md#layout-prompts-experimental):
`summary.caption` plus `items`, each with `label`, `caption`, optional `text`
and `bbox` as `[y_min, x_min, y_max, x_max]` in pixels with the origin at the
canvas centre (`"bbox_format": "yxyx"`, `"bbox_coord_space": "abs_centralized"`).

## Measured (RunPod NVIDIA A40 48 GB, 2026-10-21)

| | SFT (50 steps, CFG 3) | Turbo (8 steps, no CFG) |
| --- | --- | --- |
| `ensure()` to ready, cold pod | 63 s in total: venv 24 s, download 28.6 s, load 8.5 s (one workload serves both modes) | same workload |
| GPU time, 1024x1024 | 78.8 to 81.1 s | 8.8 to 9.2 s |
| GPU time, 1344x768 | 77.7 to 78.1 s | 8.6 to 8.9 s |
| GPU time, 1440x1440 | not measured | 17.3 s |
| 4 steps, 1024x1024 | | 5.2 s (text breaks down, see below) |
| VRAM | 25.9 GiB of weights; 28.7 GiB peak allocated at 1024x1024 (`nvidia-smi`: 30,361 MiB) | 31.5 GiB peak at 1440x1440 (`nvidia-smi`: 33,741 MiB) |

Round trip through the platform adds about 0.6 to 1.4 s per image. Re-publishing
the backend restarted the running workload in 16 s (weights cached). The pod cost
0.59 USD per hour; the whole session (22 images, a demo recording) ran 27
minutes, 0.27 USD.

What we saw on our own prompts: SFT rendered every text we asked for (a
four-line chalkboard menu with decimals, a four-step infographic, a handwritten
Chinese note) and moved text and objects when we moved their boxes. Turbo was
right on the menu, the Chinese note and the layouts, but wrote "OPEN WEIGHT /
WEIGHTS DAY" for "OPEN WEIGHTS DAY" and "PNG bock" for "PNG back"; at 4 steps
the text fell apart.

## Cost & cleanup
`fixed` keeps one replica on your worker until you delete the workload. The
script deletes it on exit (also on error and Ctrl-C). With `IK_KEEP_ALIVE=1` it
keeps it, so you can try more prompts from the dashboard image tester (8 steps
and guidance 1 for turbo); then delete it with
`ManagementClient.from_env().delete("moss-image-sft")`. If the GPU is a rented
pod, also stop the pod: deleting the workload does not.

## Troubleshooting
- **The workload never gets ready on the first start**: the download plus load
  took over the worker's 300 s custom backend timeout. Raise
  `INFERENCEKEY_CUSTOM_BACKEND_READY_TIMEOUT_SECS` on the worker; later starts
  reuse the Hugging Face cache.
- **Out of memory on a 24 or 32 GB card**: the text encoder alone is 16.8 GB
  in bf16. Use a 40 GB+ card.
- **"The fast path is not available" in the backend log**: the Qwen3.5 text
  encoder falls back to the PyTorch path without `flash-linear-attention`; it
  works, prompts are short, so we did not add it.
- **CUDA errors at load**: torch from PyPI is the CUDA 13 build; the host driver
  must support CUDA 13 (driver 580 or newer).

---
Attribution: MOSS-Image by OpenMOSS / MOSI (Apache-2.0, arXiv:2610.23374); 8-step
DMD LoRA by lodestones (Apache-2.0).

See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

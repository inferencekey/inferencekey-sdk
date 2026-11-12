# NitroFusion2 text-to-image in one or two steps (custom backend, private NVIDIA)

Serves [NitroFusion2](https://huggingface.co/ChenDY/NitroFusion2) by Dar-Yen
Chen and Yi-Zhe Song (Apache-2.0, [arXiv 2611.05648](https://arxiv.org/abs/2611.05648)),
a 6B one- and two-step distillation of Tongyi-MAI's
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo). It keeps
Z-Image-Turbo's pipeline (Qwen3 text encoder, VAE, flow-matching scheduler)
and swaps in its own diffusion transformer. For two steps the authors add a
"de-distillation" LoRA at scale 0.5 on the first step only; this backend
applies their schedule from `inference.py`.

No built-in backend serves the Z-Image pipeline, so this example ships a small
**custom Python backend** (`backend.py`) around the diffusers code the authors
publish. It downloads only what the pipeline needs: the NitroFusion2
transformer (12.3 GB) and LoRA (0.38 GB), plus Z-Image-Turbo's text encoder,
tokenizer, VAE and scheduler (8.2 GB), not Z-Image-Turbo's own transformer.

## Compatibility
- SDK: local build in this repo (see [CONTRIBUTING](../CONTRIBUTING.md#4-depending-on-the-sdk))
- Language: Python >= 3.9 for the client scripts; the backend runs on Python
  3.11 or 3.12 (the worker provisions it with `uv`)
- Placement: private (NVIDIA). **24 GB** keeps the whole bf16 pipeline on the
  GPU; a **16 GB** card works with the authors' 4-bit GGUF transformer
  (measured on a real card, see below).
- Disk: about 21 GB of weights (26 GB with the 4-bit file as well), plus a
  5.5 GB venv.

## What it deploys
- Workload `nitrofusion2`, task `text2image`, backend
  `nitrofusion2-text2image` (custom, published by `publish.py`), weights
  pinned to NitroFusion2 `62d1a700` and Z-Image-Turbo `f332072a`.
- Execution policy `fixed`, one replica on `IK_WORKER_ID`.
- Workload `config`: `ready_timeout_secs: 2400`, `request_timeout_secs: 600`,
  and `env` for the backend (`NF2_TRANSFORMER`, `NF2_MEMORY_MODE`,
  `NF2_VRAM_LIMIT_GIB` if you set them before running `main.py`).

## Run it
```bash
cp .env.example .env    # fill in tokens, project, worker id
python publish.py       # packages backend.py and uploads it to your tenant
IK_KEEP_ALIVE=1 python main.py
```
`main.py` makes three calls to `POST /endpoint/<project>/nitrofusion2/v1/images/generations`
(the authors' sign prompt in one step, a chalkboard menu in one step and in
two steps, all with seed 42) and writes `outputs/sign-1step.png`,
`outputs/menu-1step.png` and `outputs/menu-2step.png`.

Request fields (OpenAI images style, read from the passthrough body): `prompt`
(required), `size` (`"1024x1024"`; sides 256 to 2048, multiples of 16),
`seed`, `steps` or `num_inference_steps` (**1 to 4**, default 2), `dedistill`
(default true: the authors' LoRA schedule for 2 to 4 steps), `output_format`
(`png`, `jpeg` or `webp`). Guidance is always 0, as the authors run it. The
response adds `seconds` (GPU time), `encode_seconds` (text encoder),
`vae_seconds`, `vram_peak_gib` and `memory_mode`.

From the dashboard image tester: set **Inference steps** to 1 or 2 (its
default of 20 is rejected with a 400) and **Guidance scale** to 0.

## What we measured (2026-11-12, 1024 x 1024, seed 42)

| Card | Configuration | Ready after `ensure()` | GPU time, 1 step | GPU time, 2 steps | Peak VRAM |
| --- | --- | --- | --- | --- | --- |
| RTX 4090 (48 GB variant, backend capped at 23.4 GiB) | bf16, everything on the GPU (`gpu`) | 67 s (venv 28 s, download 33.8 s, load 4.4 s) | 0.78 to 0.82 s (round trip 1.45 to 1.50 s) | 1.32 to 1.33 s (round trip 1.95 to 2.00 s) | 21.86 GiB (torch) |
| RTX 2000 Ada 16 GB (real card) | `NF2_TRANSFORMER=nf2-Q4_K_M.gguf`, `gpu` | 76 s (download 28.4 s, load 23.0 s) | 3.98 to 4.33 s | 7.19 to 7.42 s | 15.11 GiB (torch), 15,948 MiB on the card |

Of the 0.78 s on the RTX 4090, the text encoder takes about 0.055 s and the
VAE decode 0.149 s; each transformer step is about 0.55 s. Four steps took
2.42 s.

Smaller budgets, emulated on the 4090 with `NF2_VRAM_LIMIT_GIB` (same prompts
and seed):

| Memory cap | Configuration | Time per image | Peak | Result |
| --- | --- | --- | --- | --- |
| 15.2 GiB | bf16 + `NF2_MEMORY_MODE=model` (offload) | 14.2 to 19.4 s | 12.3 GiB | works, same images |
| 15.2 GiB | 4-bit GGUF, `gpu` | 0.95 to 1.71 s | 15.11 GiB | works, same text |
| 15.2 GiB | Q8_0 GGUF, `gpu` | | | out of memory at the first request |
| 11.2 GiB | 4-bit GGUF + `model` offload | 7.5 to 10.5 s | 7.69 GiB (8,366 MiB on the card) | works, same text |
| 7.2 GiB | 4-bit GGUF + `model` offload | | | out of memory (the bf16 text encoder alone is 7.5 GiB) |
| 7.2 GiB | 4-bit GGUF + `sequential` offload | | | fails at load: `KeyError: None` |

No real 12 GB card was in stock, so the 12 GB row is emulated only.

What the images showed: one step is fast but spells less reliably. With seed
42 the authors' sign prompt came out as "HELLO WORd" in one step on the
RTX 4090 and as "Hello World" in two. On our chalkboard menu every mode we
tried (1, 2 and 4 steps, with and without the LoRA) wrote "SOURDOAGH".
Large headlines ("LISBOA", "Tram 28 - every 12 minutes") were right in one
step. The same seed gives different pixels on a different GPU.

## Files
- `backend.py`: the custom backend (`NitroFusion2Backend`).
- `publish.py`: packages and uploads the backend.
- `main.py`: ensure, wait until ready, three image calls, delete.
- `requirements.txt`: the backend's pinned dependencies (torch 2.14.1,
  diffusers 0.42.1, transformers 5.21.0, peft, gguf).

## Attribution
NitroFusion2 by Dar-Yen Chen and Yi-Zhe Song, Apache-2.0. Z-Image-Turbo by
Tongyi-MAI (Alibaba), Apache-2.0.

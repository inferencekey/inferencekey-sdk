# Ming-Image-0.1-Design text-to-image (custom backend, private NVIDIA): UI mockups, posters, RGBA

Serves [Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design)
from inclusionAI (MIT), a text-to-image model for text-rich visual design: UI
screens, posters, infographics, slides. Its image head is a 6B diffusion
transformer that reads the prompt through the hidden states of a Ling MoE
multimodal LLM (via a small Qwen2 connector) and decodes with a 4-channel
Qwen-Image VAE, so every image comes back as **RGBA**. With the authors'
trigger phrase the background can come back transparent.

The authors ship their own Python inference code
([inclusionAI/Ming-Image](https://github.com/inclusionAI/Ming-Image), `infer.py`),
which no built-in backend runs, so this example ships a **custom Python
backend** that downloads that code at a pinned commit, loads the model once
with the authors' loader and calls their `run_generation` per request.

## Compatibility
- SDK: local build in this repo (see [CONTRIBUTING](../CONTRIBUTING.md#4-depending-on-the-sdk))
- Language: Python >= 3.9 for the client scripts; the backend runs on Python
  3.11 (the worker provisions it with `uv`)
- Placement: private (NVIDIA). The authors validated **one 80 GB GPU in BF16**;
  we measured 50.5 GiB allocated at 1024 x 1024 and 62.8 GiB at 2048 x 2048
  (65,935 MiB on `nvidia-smi`). The backend has offloaded modes for smaller
  cards (`MING_MEMORY_MODE=lowvram|minimal`), but **we did not get to measure
  them**: see [Smaller GPUs](#smaller-gpus-not-measured-yet).
- Disk: about 53 GB of weights in the Hugging Face cache (the 6B transformer is
  12.3 GB; the Ling MoE LLM is 34 GB; connector 6.2 GB; VAE 0.25 GB), plus a
  6 GB venv.

## What it deploys
- Workload `ming-image-design`, task `text2image`, backend
  `ming-image-design-text2image` (custom, published by `publish.py`), model
  `inclusionAI/Ming-Image-0.1-Design` at revision `c57c4968`.
- Execution policy `fixed`, one replica on `IK_WORKER_ID`.
- Workload `config`: `ready_timeout_secs: 2400` (venv + 53 GB download + load
  can pass the worker's 5 minute default), `request_timeout_secs: 900`, and
  `env` for the backend (`MING_MEMORY_MODE`, `MING_VRAM_LIMIT_GIB` if you set
  them before running `main.py`).

## Run it
```bash
cp .env.example .env    # fill in tokens, project, worker id
python publish.py       # packages backend.py and uploads it to your tenant
IK_KEEP_ALIVE=1 python main.py
```
`main.py` makes three calls to `POST /endpoint/<project>/ming-image-design/v1/images/generations`
and writes `outputs/ui-1024.png`, `outputs/infographic-2048.png` and
`outputs/sticker-rgba-1024.png`.

Request fields (OpenAI images style, read from the passthrough body): `prompt`
(required), `size` (`"1024x1024"`, `"2048x2048"`, `"1920x1080"`...; the model
renders in a 1024 or 2048 bucket and resizes), `seed`, `steps` or
`num_inference_steps` (default 12), `guidance_scale` (default 1.0),
`background` (`"transparent"` puts the authors' phrase
`RGBA, 4-channel, transparent background` **on its own first line**),
`output_format` (`png` or `webp`, both keep alpha). The response adds
`seconds` (GPU time), `vram_peak_gib`, `bucket`, `transparent_fraction`.

From the dashboard image tester: set **Inference steps 12** and **Guidance
scale 1**; its defaults (20 and 4.5) are not this model's settings.

## Measured (2026-10-23, one NVIDIA A100-SXM4-80GB)
| What | Value |
| --- | --- |
| Cold start, `ensure()` to ready | 157 s with the first package (fresh venv about 60 s, 50 GB download in 69 s, load 23 s); that package had the authors' torch 2.4.0 and failed at the first request. Swapping torch to 2.5.1 inside the existing venv took 6 min 46 s, then 30 s to load. A clean cold start with this package was not timed |
| Reload with weights cached | 29 s (hot recycle after a backend update) |
| 1024 x 1024, 12 steps | 6.2 to 7.4 s GPU time, 7.3 to 8.4 s round trip (14.7 s on the first call) |
| 1920 x 1080 (1024 bucket) | 5.8 s GPU time |
| 2048 x 2048, 12 steps | 35.8 s GPU time, 41.5 s round trip |
| VRAM | 46.3 GiB weights; peak 50.5 GiB allocated at 1024, 62.8 GiB at 2048 |

What we saw (one sample per prompt, our own prompts): quoted text came out
right in posters, a slide with a bar chart and an infographic (one glyph glitch,
"PNGe" for "PNG"); text we did not spell out (subheadings, card bodies, chart
axes) came out as plausible-looking gibberish; a 12-line menu got two prices
wrong and repeated a line. The transparent background worked when the phrase
was on its own line (83 % of pixels alpha 0) and not with a comma after it;
other phrasings and prompts we tried came back opaque.

## Smaller GPUs (not measured yet)
`MING_MEMORY_MODE=lowvram` keeps the 34 GB MoE LLM in CPU RAM and streams it
once per prompt; `minimal` also streams the connector and part of the
diffusion transformer and tiles the VAE. `MING_VRAM_LIMIT_GIB` caps the process
(`torch.cuda.set_per_process_memory_fraction`) to emulate a smaller card. These
modes are in the code but did not run in our test (the budget ran out, and
changing only `config.env` on a running workload did not recycle the backend).
Treat them as untested.

## Known limits
- One image per request (`n` = 1). A 2048 x 2048 image can exceed the 8 MB
  result limit even as WebP; v0.1.4 steps the WebP quality down until it fits
  (written after the test, not yet run on a GPU).
- Re-running `ensure()` + `wait_until_ready()` against an already-ready
  workload waited without a readiness event in our run; call the endpoint
  directly if the workload is already up.

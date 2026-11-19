# MiniMax-H3-Image at 16 megapixels on one card (custom backend around ComfyUI, private NVIDIA)

Serves [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) by
MiniMax: a 33.1B single-stream diffusion transformer that generates and edits
images (up to 16 reference pictures) at about 16 megapixels in 6 steps, with
the Qwen3-VL-32B text encoder of MiniMax-H3. The official BF16 checkpoint
needs about 135 GB with its encoder. This example runs
[Comfy-Org's repack](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)
instead: the pruned 4-bit transformer (`minimax_h3_image_pruned_w4a8`, 12.5 GB),
the NVFP4 text encoder (15.7 GB) and the FP16 VAE (1.2 GB).

Those files are in ComfyUI's format, so the **custom Python backend**
(`backend.py`) downloads ComfyUI v0.45.0 and the three files, starts ComfyUI
headless on a loopback port and turns each request into the graph of the
official template (6 steps, euler/simple, no guidance). ComfyUI's memory
manager decides what stays on the GPU.

License: the model is under the
[MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE).
Read section IV: providing it as a hosted service to third parties, or revenue
above 1 million USD a year, needs a separate written authorization from
MiniMax. ComfyUI is GPL-3.0 and is downloaded at run time, not shipped here.

## Compatibility
- SDK: local build in this repo (see [CONTRIBUTING](../CONTRIBUTING.md#4-depending-on-the-sdk))
- Language: Python >= 3.9 for the client scripts (`requests`, `Pillow`); the
  backend runs on Python 3.11 or 3.12 (the worker provisions it with `uv`)
- Placement: private (NVIDIA). Measured on one **A40 (48 GB)**, where it used
  up to 30.9 GB of the card. Under memory caps of 24, 16, 12 and 8 GB on that
  same card it produced the same images (see below); no real small card was
  measured.
- System memory: ComfyUI keeps the weights in RAM and streams them to the
  card (it pinned 31 GB in our run). The worker had a 48 GB memory limit.
- Disk: 29.4 GB of weights plus a 6.8 GB venv.

## What it deploys
- Workload `minimax-h3-image`, task `text2image`, backend
  `minimax-h3-image-comfyui` (custom, published by `publish.py`), weights
  pinned to Comfy-Org/MiniMax-H3-Image `89c63508`, ComfyUI `v0.45.0`.
- Execution policy `fixed`, one replica on `IK_WORKER_ID`.
- Workload `config`: `ready_timeout_secs: 3600`, `request_timeout_secs: 900`,
  and `env` for the backend (`H3_VRAM_LIMIT_GIB`, `H3_COMFY_ARGS`, `H3_DIT`,
  `H3_TEXT_ENCODER`, `H3_WARMUP` if you set them before running `main.py`).

## Run it
```bash
cp .env.example .env    # fill in tokens, project, worker id
python publish.py       # packages backend.py and uploads it to your tenant
IK_KEEP_ALIVE=1 python main.py
```
`main.py` makes three calls to `POST /endpoint/<project>/minimax-h3-image/v1/images/generations`,
all with seed 42: the poster prompt of the official ComfyUI template at
2048 x 2048, the same prompt at 4096 x 4096, and an edit that sends the first
image back as `<Picture 1>`. It writes `outputs/poster-4mp.jpg`,
`outputs/poster-16mp.jpg` and `outputs/edit-4mp.jpg`.

Request fields (OpenAI images style, read from the passthrough body):
`prompt` (required; the model expects a dense English description), `size`
(`"2048x2048"` by default; sides in multiples of 32, up to about 20
megapixels), `seed`, `image` (a base64 string or data URL, or a list of up to
16: the reference pictures of an edit, `<Picture 1>`, `<Picture 2>`, ... in
the prompt), `output_format` (`jpeg` by default, `png` or `webp`). Like the
authors' servers, anything other than 6 steps, guidance 1 and no negative
prompt is answered with a 400. The response adds `seconds`, `stage_seconds`
(text encoder, sampler, VAE decode), `vram_peak_mib` (the card's `nvidia-smi`
peak during the request), `output_format` and `output_quality`.

Three things to know:
- **Results come back as JPEG.** A 16-megapixel PNG is 20 to 30 MB and the
  result travels through an 8 MB message that ends up a bit over twice the
  size of the base64 image, so the backend keeps the encoded image under
  about 2.5 MB and lowers the JPEG quality until it fits (95 at 4 MP, 85 at
  4096 x 4096, 70 for our 5472 x 3072 image). A result over the limit is
  dropped by the worker and the caller gets a 504 after five minutes.
- **Changing only `config.env` does not restart the backend** (the worker
  answered `reload_backend ... unknown backend` in our test). Delete the
  workload and run `main.py` again to apply new `H3_*` values.
- **From the dashboard image tester**: set **Inference steps** to 6 and
  **Guidance scale** to 1 (its defaults, 20 and 4.5, are rejected with a
  400). Its size list stops at 1024 x 1024, below the range the model is
  tuned for; larger sizes go through the API.

## What we measured (2026-11-19, one A40, seed 42)

Ready 135 s after `ensure()` on a fresh worker: venv 31 s, weights (29.4 GB)
23.5 s, ComfyUI start 21 s, warm-up image 55 s (the first load of the text
encoder takes 26 s of it). A restart with everything cached took 68 to 72 s.
The A40 is not a Blackwell card: ComfyUI ran the NVFP4 text encoder with its
emulated path and the 4-bit transformer with a native one.

| Request | Whole graph | Sampler (6 steps) | Round trip | Card peak |
| --- | --- | --- | --- | --- |
| 1024 x 1024 | 4.4 s | 2.5 s | 4.7 s | 29.4 GB |
| 2048 x 2048 | 9.3 to 10.6 s | 7.4 to 7.7 s | 10.4 to 11.4 s | 29.2 to 29.4 GB |
| 4096 x 4096 | 44.4 to 45.2 s | 37.5 to 38.1 s | 46.3 to 46.8 s | 30.7 to 30.9 GB |
| 5472 x 3072 | 43.9 s | 36.9 s | 45.6 s | |
| Edit, 1 reference, 2048 x 2048 | 18.8 to 20.2 s | 11.7 to 12.0 s | 20.1 to 21.4 s | 29.6 to 30.1 GB |
| Edit, 2 references, 2048 x 2048 | 28.1 s | 16.7 s | 29.9 s | 30.5 GB |

Memory caps on the same A40 (`H3_VRAM_LIMIT_GIB`: a hard torch cap plus
ComfyUI's `--reserve-vram`), same three calls, images identical to the
uncapped run pixel for pixel:

| Cap | 2048 x 2048 | 4096 x 4096 | Edit | Card peak (`nvidia-smi`) |
| --- | --- | --- | --- | --- |
| none | 10.4 s | 44.7 s | 19.5 s | 29,593 MiB |
| 23.4 GiB (a 24 GB card) | 13.8 s | 46.8 s | 23.6 s | 24,089 MiB |
| 15.2 GiB (16 GB) | 17.4 s | 49.3 s | 26.7 s | 15,769 MiB |
| 11.2 GiB (12 GB) | 15.8 s | 48.7 s | 24.7 s | 11,771 MiB |
| 7.2 GiB (8 GB) | 15.0 s | 48.9 s | 24.0 s | 7,739 MiB |

These are caps on a 48 GB card with a fast host and 48 GB of system memory,
not runs on real small cards (none was in stock on the day). A smaller card
is also a slower card, so expect longer times than these.

What the images showed: lettering was right in every text sample we ran (the
template poster, a departures board with six rows and a small plaque, a
chalkboard menu with prices). At 4096 x 4096 the same seed gives a different
picture than at 2048 x 2048, and on the poster the bottom logotype ran into
both edges of the frame. The one-reference edit replaced the fox with a rabbit
and kept the lettering, but the rabbit kept the fox's tail. The two-reference
edit ("put the fox from Picture 1 next to the fisherman from Picture 2")
returned the fisherman with harsher contrast and no fox.

## Files
- `backend.py`: the custom backend (`MiniMaxH3ImageBackend`).
- `publish.py`: packages and uploads the backend.
- `main.py`: ensure, wait until ready, three image calls, delete.
- `requirements.txt`: the backend's dependencies (ComfyUI v0.45.0's own
  requirements with torch pinned to 2.14.1).

## Attribution
MiniMax-H3-Image by MiniMax, MiniMax-H3-Image Community License Agreement.
Quantized repack by Comfy-Org. ComfyUI by Comfy-Org, GPL-3.0.

# PAN 2: a world model that simulates actions as video (custom backend, private NVIDIA GPU)

[PAN 2](https://huggingface.co/IFM/PAN2) (Institute of Foundation Models, MBZUAI,
Apache-2.0) is a world model: given the current state of a scene (an optional
first frame plus text) and actions described in text, it simulates the outcome
as video, 704 x 1248 at 24 fps, up to 20 s. A language model (the "conceptual
reasoner") writes the next world state as a structured, timed caption; a frozen
Qwen3.5-9B encodes it; a 29B-parameter MMDiT predicts the video latents; the
Wan 2.2 VAE decodes them.

No built-in backend serves it, so this example ships a **custom Python backend**
(`backend.py`) that runs the authors' Diffusers integration
(`ModularPipeline.from_pretrained("IFM/PAN2")`, diffusers 0.43.0) and answers on
the workload's OpenAI images route. A clip takes minutes, so the call returns a
`clip_id` when it is not done within about 270 s and the client polls it.

## What it deploys

| | |
| --- | --- |
| Model | `IFM/PAN2` at revision `82f54c6c`: transformer 67.4 GB, text encoder (Qwen3.5-9B) 15.9 GB, VAE (Wan 2.2) 2.8 GB, all bf16, 86.1 GB in one repo |
| Backend | custom, `pan2-world-model` (`task_type` `text2image`) |
| Policy | `fixed`, one replica on your worker (`IK_WORKER_ID`) |
| Disk | about 86 GB of weights plus about 6 GB of Python packages: give the worker a local disk of at least 100 GB, not a network volume |
| Prompts | PAN 2 reads structured captions. Two are built in (`house`, `glass`); write your own with the authors' rewriter policy (`inference/pan-prompt-rewriter` in [ifm-ai/PAN2](https://github.com/ifm-ai/PAN2)) and any instruction-following LLM |

## What we measured (2026-11-21, one RTX PRO 6000 Blackwell Server Edition, 96 GB)

| Clip | Steps | Model time | Per step | VAE decode | Peak on the card |
| --- | --- | --- | --- | --- | --- |
| `house`: image to video, 3 s (73 frames, 1248 x 704) | 20 | 109.3 s | 5.1 s | 6.1 s | 94,461 MiB |
| `glass`: text to video, 2 s (49 frames) | 20 | 71.0 s | 3.3 s | 4.1 s | 94,461 MiB |
| `house` stretched to 10 s (241 frames) | 20 | 490.0 s | 23.4 s | 20.2 s | 95,941 MiB |
| `glass 1s 8steps` from the dashboard tester (25 frames) | 8 | 15.7 s | 1.7 s | 2.1 s | 93,745 MiB |
| `house` 3 s, memory capped at 78 GiB, `text_encoder: "cpu"` | 20 | 149.1 s (text encoding 40.4 s on the CPU) | 5.1 s | 6.2 s | 81,419 MiB (PyTorch reserved 78,406) |

- Ready **47 s** after `ensure()` on a fresh worker: the 86.1 GB download took
  18.3 s on that host and loading 13.1 s.
- Every step runs the transformer twice (classifier-free guidance, 2.5): 20
  steps are 40 transformer passes. The authors' default is 50 steps.
- **Smaller than 96 GB:** the transformer (63 GiB) and the text encoder (15 GiB)
  do not fit together under an 80 GB budget. With `text_encoder: "cpu"` the
  clip came out practically the same (mean pixel difference 3.2 of 255 at the
  last frame) and 40 s slower. That rung is a PyTorch memory cap on the 96 GB
  card, not a real 80 GB card. Under the same cap, moving the text encoder to
  the GPU only while it runs (`offload`) ran out of memory. Cards with 48 GB or
  less would need the transformer's blocks streamed from system RAM; we did not
  try that.
- The Qwen3.5 linear-attention layers use the PyTorch fallback
  (`flash-linear-attention` and `causal_conv1d` are not installed); the text
  encoder takes 0.2 to 0.5 s on the GPU anyway.
- The authors' guardrail (`pan2-guardrail`) is **off** in this example (see
  below).

## Run it

```bash
# the SDK: local build of this repo, see ../CONTRIBUTING.md; the client also needs requests
cp .env.example .env    # fill in tokens, project, worker id
python publish.py       # once: uploads backend.py to your tenant as a custom backend
python main.py                       # house preset: 3 s from the authors' first frame
python main.py --preset glass        # text to video, 2 s
python main.py --caption my.json --image first.jpg --duration 4
```

Outputs land in `./outputs`: `<name>.mp4` and `<name>.json` with the timings
and memory the backend measured.

## The request

`POST {base}/endpoint/{project}/pan2/v1/images/generations` with the data-plane key:

```json
{"model": "default", "prompt": {"video_style": "...", "events": ["..."]}, "image": "<base64 first frame>",
 "duration_sec": 3, "steps": 20, "seed": 123}
```

- `prompt`: the structured caption (object or JSON string), or `preset`
  (`house`, `glass`). `image` makes it image to video; `example: "house"` uses
  the authors' first frame.
- Optional: `negative_prompt` (default: PAN 2's own), `num_frames` (4n+1),
  `guidance_scale` (2.5), `height` / `width`, `text_encoder` (`gpu`, `offload`,
  `cpu`), `vram_limit_gib` (cap memory to emulate a smaller card),
  `vae_tiling`, `wait_s`.
- Answer: `{"status": "running", "clip_id"}` while it renders; poll with
  `{"clip_id": "..."}`. When done: `status: "done"`, the MP4 in 2 MB parts
  (`mp4_b64` is part 0, then `{"clip_id", "part": n}`; check `mp4_sha256`),
  `data[0].b64_json` (the middle frame, so the dashboard's image tester shows
  something), `timings`, `peak_reserved_mib`, `card_peak_mib`. One result is
  limited to an 8 MB message, hence the parts.
- From the dashboard's image tester, write a preset and optional words:
  `glass 1s 8steps`.

## Settings (`config.env` of the workload)

| Variable | Default | |
| --- | --- | --- |
| `PAN2_TEXT_ENCODER` | `auto` | `gpu` from 90 GiB of card memory up, else `cpu` |
| `PAN2_VAE_TILING` | `0` | decode in tiles |
| `PAN2_WEIGHTS_DIR` | `~/.cache/huggingface` | keep it on local disk |
| `PAN2_GUARDRAIL` | `0` | the authors' guardrail, see below |
| `PAN2_WARMUP` | `0` | a tiny clip during `setup()` |

## Safety

The authors strongly recommend running PAN 2 behind
[`pan2-guardrail`](https://github.com/ifm-ai/pan2-guardrail) (Qwen3Guard-Gen-4B on
the request, a frame classifier on the video). This example turns the check
off to keep the test small; we did not run the guardrail. To use it, uncomment
the line in `requirements.txt` and set `PAN2_GUARDRAIL=1`. Read the model
card's out-of-scope uses before you expose the endpoint to anyone.

## Notes

- One clip at a time per worker; a new request while one renders gets a 400
  "busy" with the running `clip_id`.
- Keep `IK_KEEP_ALIVE=1` while you try captions, then delete the workload: the
  GPU stays reserved while it exists.
- Changing only `config.env` does not restart a running custom backend:
  delete the workload, then `ensure()` again.

Attribution: PAN 2 by IFM, MBZUAI (Apache-2.0). Qwen3.5-9B by the Qwen team,
Wan 2.2 VAE by Wan-AI, Diffusers by Hugging Face (all Apache-2.0).

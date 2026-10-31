# Hy-Nomas interactive video world (custom backend, private NVIDIA)

Serves [Hy-Nomas](https://huggingface.co/tencent/Hy-Nomas) from Tencent Hunyuan
(Apache-2.0), a segment-autoregressive video **world model**: it generates a world
one 5-second segment at a time (720p class, 1280 x 704, 24 fps, 8 distilled
denoising steps) and keeps a key/value cache of what it already generated, so each
new segment can be steered with a new instruction while the world stays the same.
No built-in backend serves it, so this example ships a **custom Python backend**
that drives the authors' own resident generation service (the GPU side of their web
demo, `apps/seg_ar_web_server.GenerationService`, source fetched at a pinned
commit) with their **single-GPU low-memory preset**
(`configs/experiment/single_gpu_low_mem_inference.yaml`: INT8 DiT, INT8 KV cache,
FP8 PixelUMM decoder, CPU offload), publishes it to your tenant and deploys it on
your NVIDIA worker.

Each call is one turn of a world: `start` opens it and returns the first segment,
`continue` steers it and returns the next one. One segment per call keeps every
request well under the gateway's synchronous limit.

## Compatibility
- SDK: local build in this repo (see [CONTRIBUTING](../CONTRIBUTING.md#4-depending-on-the-sdk))
- Language: Python >= 3.9 for the client scripts; the backend runs on CPython 3.12
  (the worker provisions it with `uv`; torch 2.10.0 + CUDA 13.0, transformers 4.57.1)
- Placement: private NVIDIA **Ada (SM89) GPU with 24 GB**, e.g. a GeForce RTX 4090:
  the FP8 decoder kernel is compiled for SM89. Driver for CUDA 13.0+ (580+),
  `nvcc` at `/usr/local/cuda/bin` (the kernel is built on first use), 64 GB+ host RAM
  (models are offloaded to RAM between stages), about 60 GB of disk for weights,
  venv and caches
- Backend: custom (`hy-nomas-world`)   Policy: `fixed`

## What we measured (2026-10-31, one RTX 4090 24 GB, AMD EPYC 7513 host)
| | |
| --- | --- |
| Weights the preset reads | INT8 DiT 9.06 GB + FP8 decoder 9.13 GB + Qwen2.5-VL-7B text encoder ~16.6 GB |
| `setup()` with weights on disk | 18 to 19 s (pipeline load 12 to 13 s) |
| Ready after `ensure()`, venv and weights already cached | 19 to 31 s |
| First segment in a fresh process | 85 to 138 s (kernel compile and caches) |
| Every next 5 s segment, 1280 x 704, 120 frames | **88 to 94 s** (denoise ~31 to 39 s, cache write ~8 to 10 s, decode and MP4 encode ~30 to 45 s) |
| GPU memory, PyTorch peak | 17.4 to 18.3 GiB (grows slightly per turn) |
| PyTorch capped at 20 GiB | works, byte-identical MP4s |
| PyTorch capped at 16 GiB | fails: out of memory moving the bf16 text encoder to the GPU |

The same turns with the same seed gave byte-identical MP4s on every run. What
worked and what did not, in our own short test: the world stays very consistent
from turn to turn (same subject, place and light), and an added object came in on
cue (blue flowers beside the path). Scene-wide changes did not: snow (twice, the
second time worded more strongly) and a storm never appeared, and the subject kept
walking instead of stopping. We wrote the structured prompts by hand; the authors'
web demo first rewrites user instructions with a separate 27B prompt enhancer
(Nomas-PE) that this example does not run.

## Prerequisites
- An SDK token (`ik_sdk_`) and an API key (`ik_live_`) for the project, see
  [Tokens](../README.md#tokens).
- A dashboard session token and your tenant id, to publish the backend and to turn
  on the workload's chat route (or do both by hand in the dashboard).
- An NVIDIA worker as above, assigned to the project; its id goes in `IK_WORKER_ID`.
- `ffmpeg` on your machine to join the segments into one MP4 (optional).

## How to run
```bash
cp .env.example .env      # fill in real values
pip install requests      # plus the local SDK build
python publish.py         # once: packages backend.py and uploads it to your tenant
python main.py            # ensure -> ready -> three turns of one world -> outputs/
```
`main.py` plays the authors' own example world (a small orange fox in a luminous
forest) turn by turn and writes `outputs/segment-1.mp4` ... `segment-3.mp4` and
`outputs/world.mp4`. Set `IK_KEEP_ALIVE=1` to keep the workload afterwards.

## The request
Send the turn as JSON in the last user message of `/v1/chat/completions` (or as top-level fields):
```json
{"step": "start", "style": "Cinematic fantasy", "perspective": "Third-person view",
 "subjects": [{"subject_id": "fox", "display_name": "a small orange fox",
               "subject_attribute": "Orange fur, white chest, and a fluffy tail."}],
 "scene": "A small orange fox stands on a stone path in a luminous forest.",
 "action": "The fox walks slowly along the path.", "seed": 42}
```
```json
{"step": "continue", "session_id": "w50f42932fa5e",
 "change": "Glowing blue flowers open beside the path as the fox continues forward.",
 "action": "The fox passes the flowers and looks around."}
```
You can also pass the authors' structured JSON whole (`world` for the first segment,
`segment` for later ones). The answer's message content is JSON: `session_id`,
`segment`, `video_b64` (MP4 of this segment, about 1 to 3 MB), `width`, `height`,
`frames`, `timings` (`gpu_s`, `peak_vram_mib`). A world keeps the seed it started
with. The backend holds one world at a time (one KV cache on the GPU): starting a
new world ends the previous one.

Knobs (workload `config.env`): `NOMAS_VRAM_LIMIT_GIB` caps PyTorch's GPU memory,
`NOMAS_HEIGHT` / `NOMAS_WIDTH` change the resolution, `NOMAS_EXTRA_OVERRIDES` adds
Hydra overrides such as `inference.infer_steps_video=6`.

## Notes
- Package and workload task type are `text2image`: the SDK cannot tag a custom
  backend `text2video` yet, and the workload must match the backend. The output is
  still MP4 video on the chat route.
- The authors' online worker also turns on `inference.seg_ar_segment_vae_decode`
  and `model.vae.stable_prefix_streaming`; with this preset on 24 GB that decodes
  while the DiT is still on the GPU and runs out of memory, so this backend decodes
  each segment after denoising. Segments are joined on the client.
- Not covered: camera paths (`camera_trajectory`), the authors' prompt enhancer,
  undo/rollback of a turn.

## Files
- `backend.py`: the custom backend (setup downloads source, CUTLASS headers and weights)
- `publish.py`: packages and uploads the backend
- `main.py`: deploys and plays three turns of one world
- `requirements.txt`: the backend's venv (torch 2.10 cu130, prebuilt FlashAttention 2 wheel, the authors' pins)
- `.env.example`: placeholders only

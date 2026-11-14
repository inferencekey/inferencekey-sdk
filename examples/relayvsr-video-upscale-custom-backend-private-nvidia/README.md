# RelayVSR 4x video upscaling (custom backend, private NVIDIA)

Serves [RelayVSR](https://huggingface.co/kopper/RelayVSR) (Xijun Wang, Xin Li,
Zirui Lang, Suhang Yao, Haoran Li and Zhibo Chen; Apache-2.0,
[arXiv 2609.37850](https://arxiv.org/abs/2609.37850)), a streaming **4x video
super-resolution** model with a split design:

- a **reference generator** adapted from Wan2.2-TI2V-5B (`wan.safetensors`,
  10.1 GB bf16) draws a detailed latent only for sparse keyframes, one every
  `window_size - 1` frames (every 15 by default), in one denoising step;
- a **small VSR network** (`flash.safetensors`, 128 MB) upscales every frame,
  reading the references at both ends of its window and the previous frame.

No built-in backend serves it, so this example ships a **custom Python backend**
that runs the authors' own inference code (`refvsr_inference`, fetched from
[github.com/kopperx/RelayVSR](https://github.com/kopperx/RelayVSR) at a pinned
commit) exactly like their `inference.py` (bf16, one step, seed 42), publishes it
to your tenant and deploys it on your NVIDIA worker. One request = one clip in,
one clip out, exactly 4x the input size.

## Compatibility
- SDK: local build in this repo (see [CONTRIBUTING](../CONTRIBUTING.md#4-depending-on-the-sdk))
- Language: Python >= 3.9 for the client scripts; the backend runs on CPython 3.12
  (the worker provisions it with `uv`: torch 2.10.0 + CUDA 12.8 and the authors' locked versions)
- Placement: private NVIDIA GPU. **24 GB** for 480p in (1920p out; tested on an RTX 4090 and an RTX 3090); a **16 GB**
  budget only fits 270p in (1080p out), see the table. About 25 GB of disk
  (venv about 7 GB, weights 10.2 GB), `gcc` on the worker (Triton and
  FlexAttention compile kernels on first use)
- Backend: custom (`relayvsr-video-upscale`)   Policy: `fixed`

## What we measured (2026-11-14)
Clip: 3 s of *Tears of Steel* (CC BY 3.0, Blender Foundation), cut to 854 x 480
and compressed; the authors' bundled demo clip for 360p. "Model fps" is GPU time
of the two models only (the authors' measure); warm calls, seed 42.

| | RTX 4090 24 GB | RTX 3090 24 GB |
| --- | --- | --- |
| Ready after `ensure()` (weights already on the worker's disk, fresh venv) | 71 s | 166 s (slower host downloads) |
| 480p in, 3416 x 1920 out, reference every 15 frames (72 frames) | **7.3 fps** | 3.8 fps |
| same, reference every 31 frames | 9.1 fps | 4.6 fps |
| 360p in, 2496 x 1440 out, reference every 15 frames (48 frames) | 12.3 fps | 5.9 fps |
| 270p in, 1920 x 1080 out, reference every 15 frames | 23.9 fps (29.1 every 31) | 10.7 fps |
| Peak GPU memory (PyTorch) 1920p / 1440p / 1080p out | 20,880 / 15,887 / 13,364 MiB | same |
| First call for a new frame size | slower (kernel compile): 4.7 fps instead of 7.3 at 1920p | 2.6 instead of 3.8 |
| Round trip of a 72-frame 1920p clip | 26 to 41 s: CPU H.264 encoding of the 4x frames dominates | 38 to 42 s |

Smaller budgets (PyTorch capped on the 4090 with `RVSR_VRAM_LIMIT_GIB`; no 16 GB
card was in stock to confirm on real hardware):

| Cap | 480p in | 360p in | 270p in |
| --- | --- | --- | --- |
| 15.2 GiB (16 GB card) | out of memory | out of memory | **works**, 23.9 fps, same output |
| 11.2 GiB (12 GB card) | | | out of memory (the generator alone is 9.6 GiB) |

What we saw in the output: clearly sharper faces, hair and fabric than plain
bicubic upscaling. With a reference every 31 frames the run is about 25 %
faster at 1920p, but on fast motion the frames far from a reference go soft
(a turning head 15 frames from the nearest reference); every 15 frames kept
it sharp.

## Prerequisites
- An SDK token (`ik_sdk_`) and an API key (`ik_live_`) for the project, see
  [Tokens](../README.md#tokens).
- A dashboard session token and your tenant id, to publish the backend and to turn
  on the workload's chat route (or do both by hand in the dashboard).
- An NVIDIA worker as above, assigned to the project; its id goes in `IK_WORKER_ID`.

## How to run
```bash
cp .env.example .env      # fill in real values
pip install requests      # plus the local SDK build
python publish.py         # once: packages backend.py and uploads it to your tenant
python main.py my-clip.mp4   # or no argument: the authors' bundled 360p demo clip
```
`main.py` upscales the clip twice, with a reference every 15 frames and every
31, and writes `outputs/x4-w16.mp4` and `outputs/x4-w32.mp4`. Set
`IK_KEEP_ALIVE=1` to keep the workload afterwards.

## The request
Send it as JSON in the last user message of `/v1/chat/completions` (or as top-level fields):
```json
{"video_b64": "<base64 MP4, up to 5 MB>", "window_size": 16, "max_frames": 72, "seed": 42}
```
`video_url` (https) or `"demo": true` (the authors' demo clip) work instead of
`video_b64`. Frames up to 960 x 544 (`RVSR_MAX_PIXELS`), up to 241 per call
(`RVSR_MAX_FRAMES`). The answer's message content is JSON: `video_b64` (H.264
MP4, no audio), `width`, `height`, `frames`, `fps`, `keyframes`, `crf`, and
`timings` (`model_fps`, `wan_ms`, `flash_ms`, `first_frame_ms`, `wall_s`,
`peak_vram_mib`).

## Notes
- Package and workload task type are `text2image`: the SDK cannot tag a custom
  backend with a video task type yet. The output is still MP4 video on the chat route.
- The answer travels in one message of at most 8 MB: the backend starts at CRF 18
  and raises the CRF until the base64 fits (our 72-frame 1920p clips came back at
  CRF 22, about 4.9 MB). Set `RVSR_OUTPUT_DIR` to keep a CRF 18 copy on the worker.
  For long videos, send chunks of 2 to 3 seconds and join them on your side.
- Audio is not copied (as in the authors' script).
- Not covered: fp16, the authors' first-keyframe mode, real-time streaming input.

## Files
- `backend.py`: the custom backend (setup downloads the authors' source and the two checkpoints)
- `publish.py`: packages and uploads the backend
- `main.py`: deploys and upscales one clip with two reference intervals
- `requirements.txt`: the backend's venv (torch 2.10.0 cu128, the authors' locked versions)
- `.env.example`: placeholders only

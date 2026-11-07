# LFM3-Omni-2B on a private GPU — voice, images and video in; text and speech out

Serves Liquid AI's **LFM3-Omni-2B** (2.17B, BF16, LFM Open License v1.0) as a
**custom Python backend**. One model takes text, images, video with its
soundtrack and 16 kHz audio, and answers in text, 24 kHz speech, or both
interleaved. The axis this example varies: the speech output. The built-in
engines serve this model's text at most, so the backend drives the
checkpoint's own Transformers code and returns the answer as text plus a
base64 WAV on the OpenAI chat route.

`main.py` sends three requests to one workload: a spoken question answered in
text and speech, a photo plus a spoken question about it, and text to speech.

## Compatibility
- SDK: local build in this repo (or `>= 0.1.0` once published)
- Language: Python >= 3.9 (client); CPython 3.11 or 3.12 on the worker (transformers 5)
- Placement: private (NVIDIA, bf16-capable: Ampere or newer)
- Backend: custom (`lfm3-omni-2b`, `backend.py`)   Task: `text2audio` (chat route)   Policy: `fixed`

## Prerequisites
- The tokens/ids this example needs — see [tokens & placement](../README.md#tokens):
  `INFERENCEKEY_SDK_TOKEN`, `INFERENCEKEY_API_KEY`, `INFERENCEKEY_PROJECT`, `IK_WORKER_ID`.
- For `publish.py` and to turn on the chat route: `INFERENCEKEY_TENANT_ID` and a
  dashboard session token in `INFERENCEKEY_USER_TOKEN` (or upload `dist/*.tar.gz`
  in Dashboard -> Custom backends and flip "Chat" in Endpoint settings).
- A registered NVIDIA worker with about **5 GB of free GPU memory** (6 GB for a
  one-minute video), a driver for CUDA 12.8 or newer, and `ffmpeg` for video
  input (the backend installs it with `apt-get` when it runs as root).
- The inputs in `samples/`: `question.wav` and `flyer-question.wav` were made
  with a local text-to-speech voice; `flyer.jpg` is a made-up event flyer.

## Run
```bash
cp .env.example .env        # fill in real values
python -m venv .venv && . .venv/bin/activate
pip install requests              # requirements.txt is for the backend (the worker installs it)
pip install maturin && ( cd ../../bindings/python && maturin develop --release )
python publish.py           # once: packages backend.py and uploads it to your tenant
python main.py
```

## What it does
- **ensure()** declares the workload (`backend="lfm3-omni-2b"`, `task_type="text2audio"`)
  pinned to your worker, `fixed` policy, one replica. Idempotent by slug.
- **wait_until_ready()** blocks while the worker builds the backend venv
  (torch 2.15, transformers 5.22), the backend downloads the checkpoint at a
  pinned revision (4.5 GB) and loads it in bf16, then warms up text and speech.
- **Three calls** on `/v1/chat/completions`. The message content is the
  backend's JSON: `text`, `audio_b64` (WAV), `audio_seconds`, `tool_calls` and
  `timings` (first text token, first speech frame, GPU seconds, peak VRAM).
  Answers land in `./outputs`.
- **delete()** tears the workload down on exit (`IK_KEEP_ALIVE=1` keeps it).

Request options (in `extra_body`): `mode` (`text`, `audio`, `interleaved`),
`max_new_tokens` (a speech frame of 80 ms counts as one), `temperature`
(greedy by default, as the authors recommend for text), `audio_temperature`
(0.4) and `audio_repetition_penalty` (1.2), `use_audio_in_video` (default
true: the model hears the soundtrack), `fps`, and `tools` (OpenAI function
schemas; the model's `<tool_call>` blocks come back parsed in `tool_calls`).
Message parts: `text`, `input_audio` (`{"data": <base64>, "format": "wav"}`),
`image_url` and `video_url` as `data:` URLs (the backend does not fetch URLs).

## Measured (InferenceKey deploy test, 2026-11-07)

| | RTX A5000 24 GB | RTX 2000 Ada 16 GB |
| --- | --- | --- |
| Ready after `ensure()` | 46 s (warm uv cache) | 59.5 s (fresh worker) |
| Spoken question -> text + speech | 9.8-10.5 s of speech in 6.9-7.4 s, first sound 0.28-0.50 s | 10.1 s in 6.1-6.6 s, first sound 0.20 s |
| Photo + spoken question -> text | 0.84-0.88 s on the GPU | 0.89-0.93 s |
| Text to speech (en, es) | 1.6x real time, first frame 0.05 s | 1.8-1.9x |
| 9.7 s video with soundtrack | 2.6 s | 2.5 s |
| Text only | 37.6 tokens/s | 40.5 tokens/s |
| GPU memory, one process | 4,832 MiB (5,966 MiB for a 60 s video) | 4,288-4,434 MiB PyTorch peak |

Under an emulated cap (`LFM3_VRAM_LIMIT_GIB`) every check passed down to
4.45 GiB; at 4.35 GiB the 10 s video ran out of memory, and a 60 s video
needs 5.6 GiB. Text speed is set by Transformers' decode loop, not by the GPU:
the small 70 W card was as fast as the A5000.

## Cost & cleanup
One always-on replica on your worker (`fixed`) until `delete()` runs; the
script deletes it on exit unless `IK_KEEP_ALIVE=1`. On Ctrl-C, run
`python -c "from inferencekey import ManagementClient as M; print(M.from_env().delete('lfm3-omni'))"`.

## Troubleshooting
- **`Model '...' is not available on this endpoint`**: send `"model": "default"`.
- **Video requests fail**: the worker needs `ffmpeg` (FFmpeg libraries for torchcodec).
- **Request too large**: inline media count against the request size limit
  (about 2 MiB of JSON); send short clips (a 480p, 10 s MP4 is about 230 KB).
- **Faster text**: SGLang support for LFM3 is an open pull request
  (sgl-project/sglang#46858). Released sglang 0.5.23 falls back to
  Transformers and crashes at CUDA graph capture. Built from that PR, text ran
  at 158 tokens/s on the A5000 with the same answers, but text output only.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

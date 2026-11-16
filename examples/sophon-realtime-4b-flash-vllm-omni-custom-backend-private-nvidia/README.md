# Sophon-Realtime-4B-Flash full-duplex voice on vLLM-Omni (custom backend, private NVIDIA)

Serves [Sophon-Realtime-4B-Flash](https://huggingface.co/adventists-ai/Sophon-Realtime-4B-Flash)
from Adventists AI (Apache-2.0): a **full-duplex** voice model. It listens while
the caller talks, decides from the voice when the caller has finished, writes a
reply with sound tags (`[laughter]`, `[sigh]`, `<strong>`) and speaks while it
writes. One vLLM-Omni model in three stages on one GPU: a 4B "thinker" (bf16),
a talker (one forward pass per 80 ms frame) and a 24 kHz codec decoder. The
authors serve it with `flash/install.sh` and `flash/start.sh` from
[github.com/adventists-ai/sophon-realtime](https://github.com/adventists-ai/sophon-realtime);
the server answers an OpenAI Realtime-style WebSocket (`/v1/realtime`) and
`/v1/chat/completions`.

No built-in InferenceKey backend runs this vLLM-Omni plugin, and the platform
forwards HTTP requests, not WebSockets. So this example ships a **custom Python
backend** that runs the authors' scripts next to it and, for each request,
**plays one recorded caller turn into the local `/v1/realtime` WebSocket at
speaking speed**, keeps the microphone open (silence) and lets the model decide
on its own when the turn is over, as on a live line. A request is one turn, not
a live phone call: for a real microphone you would put a WebSocket relay in
front of the server.

## Compatibility
- SDK: local build in this repo (see [CONTRIBUTING](../CONTRIBUTING.md#4-depending-on-the-sdk))
- Language: Python >= 3.9 for the client scripts. The backend is a thin client;
  the authors' `install.sh` builds a **separate Python 3.12 venv** with vLLM 0.30,
  vLLM-Omni 0.30, `duplexjev-vllm` 0.3.0 and the plugin (repository pinned at
  commit `2c48aa54`, weights at revision `ae4a8591`)
- Placement: private NVIDIA. **24 GB card** as the authors ship it, or **a 16 GB
  card** with the settings below. About 40 GB of free local disk (13 GB of
  weights plus the venv). Keep the cache on the local disk, not on a network volume
- Backend: custom (`sophon-realtime-4b-flash`)   Policy: `fixed`

## What we measured (2026-11-16)
"Gap" = from the caller's last voiced sample to the model's first audio,
measured inside the backend on the GPU host (realtime mode). Caller questions
were synthetic speech (Pocket TTS).

| | RTX 4090 24 GB, as shipped | RTX 5080 16 GB, fp8 thinker |
| --- | --- | --- |
| Ready after `ensure()`, empty caches | 326 s (13 GB in about 30 s, install about 2 min, kernel compile and warm-up) | 750 s (first compile on this card) |
| Ready again, caches warm | 95 s (16 GB settings on the 4090) | - |
| Gap, one conversation (median, range) | 0.28 s (0.25 to 0.42) | 0.38 s (0.26 to 0.41) |
| Gap, 4 / 8 conversations at once | 0.33 / 0.34 s | 2 at once: 0.35 / 0.38 s |
| `chat` mode, request to first audio | 0.16 to 0.18 s | 0.19 to 0.20 s |
| Speech generated per second of wall time | about 7 to 10 s (1 conversation), 4.3 to 4.8 s (8) | about 6.6 to 9.3 s |
| GPU memory, card peak (nvidia-smi) | 19.5 GiB (20,009 MiB) | 12.7 GiB (13,049 MiB) |

What we saw in the replies (Whisper medium on every reply):
- The words match the written reply; sound tags are performed, not read aloud.
- A 1.5 s "um" pause mid-sentence: on the 4090 as shipped it drafted a reply in
  the pause, kept it silent, and answered 0.88 s after the end. With the fp8
  thinker it said "What kind of" into the pause before yielding.
- A question ending in "...what time does it arrive?": every run (both cards,
  both settings) answered about 1.3 to 1.5 s **before the caller finished**, and
  said it twice. The answer was wrong every time (3:55 for 2:15 plus 95 minutes;
  `chat` mode said 4:25).
- Facts: Canberra every time, with 440,000 to 500,000 people, and once "4.4
  million". The authors warn that the 4B brain states wrong facts with confidence.
- Cloning from a 6.7 s clip works with a new voice name in realtime mode (we
  did not try it in `chat` mode, and did not score the similarity).

## Run on a 16 GB card
The shares are fractions of the **card's** memory, so set them for the card you
use. These ran on a real 16 GB RTX 5080 (and as a 16 GB budget on the 4090:
`0.30,0.20,0.05`, 13.5 GiB peak):

```bash
export SOPHON_THINKER_QUANTIZATION=fp8          # the 4B thinker in fp8 (4.7 GiB instead of 7.9)
export SOPHON_STAGE_GPU_MEMORY=0.44,0.30,0.075  # thinker, talker, decoder share of a 16 GB card
export SOPHON_THINKER_MAX_MODEL_LEN=4096
export SOPHON_MAX_SESSIONS=2                    # a third conversation is refused
python main.py my-question.wav
```

In bf16 a 16 GB budget does not start: with the thinker at 0.40 the talker
stage fails with `To serve at least one request with the model's max seq len
(4096), (0.44 GiB KV cache is needed, which is larger than the available KV
cache memory (0.19 GiB)`.

## How to run
```bash
cp .env.example .env              # fill in real values
python publish.py                 # once: uploads the backend to your tenant
python main.py my-question.wav [my-voice.wav]
```
`main.py`: `ensure()` -> `wait_until_ready()` -> turns on the workload's chat
route and lets 8 conversations run at once (`max_in_flight`) -> three calls on
`/v1/chat/completions`: a realtime turn with the built-in voice `chelsie`, a
realtime turn in a cloned voice (only if you pass a 3 to 10 s clip; **only clone
your own voice or one you have permission to use**), and the same question in
`chat` mode. Each call writes `outputs/<name>.wav` (the reply, 24 kHz),
`<name>.conversation.wav` (two channels on one clock: caller left, model right)
and `<name>.json` (reply text with its sound tags, timings, server events).
`IK_KEEP_ALIVE=1` keeps the workload after the run.

Request options (in `extra_body`): `mode` (`realtime` default, or `chat`),
`voice`, `language` (`en`, `zh`, ...), `voice_clip_b64`, `pace` (1.0 = speaking
speed), `max_wait_s` (how long the microphone stays open after the caller stops,
default 25), `max_tokens`, `temperature`.

## Notes
- The first start downloads the weights, installs vLLM-Omni and compiles
  kernels; the workload's `ready_timeout_secs` is 3600. The image needs no CUDA
  toolkit for the bf16 or fp8 4B builds (ours had no `nvcc`).
- The backend stops the server with the authors' `stop.sh` when the workload is
  deleted, and a watchdog stops it if the backend process dies.
- The realtime turn returns when the model's reply has been generated, not
  played: speech comes out several times faster than real time.
- Streaming the reply back as it is spoken is not available through the
  platform for custom backends; the reply arrives as one WAV.

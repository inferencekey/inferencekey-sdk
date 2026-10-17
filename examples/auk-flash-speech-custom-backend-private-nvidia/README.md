# AuK-Flash speech generation and editing (custom backend, private NVIDIA)

Serves [AuK-Flash](https://huggingface.co/tencent/AuK-Flash) from Tencent Hunyuan
(MIT), the 4-step distilled version of AuK, a 1.5B diffusion transformer that
generates **and edits** speech from one natural-language instruction: voice
design from a text description, zero-shot voice cloning, replacing or inserting
words inside a recording, pitch / speed / volume, emotion, whisper, denoising and
speaker separation. No built-in backend serves it, so this example ships a
**custom Python backend** that drives the authors' own PyTorch code, publishes it
to your tenant and deploys it on your NVIDIA worker.

## Compatibility
- SDK: local build in this repo (see [CONTRIBUTING](../CONTRIBUTING.md#4-depending-on-the-sdk))
- Language: Python >= 3.9 for the client scripts; the backend runs on CPython
  3.10 to 3.12 (the worker provisions it with `uv`)
- Placement: private NVIDIA, driver for CUDA 12.6+. About 19 GiB of VRAM at peak
  (24 GB card), or about 10.5 GiB with the CPU-offload variant (should fit a 16 GB card, not tested on one; much slower)
- Backend: custom (`auk-flash-speech`, or `auk-flash-speech-offload`)   Policy: `fixed`

## Licence: read this before using it in a product
AuK's code and weights are MIT. The model also needs the **Qwen2.5-Omni-3B**
encoder, which is under the **Qwen RESEARCH licence (non-commercial use only)**;
the backend downloads it from Hugging Face at start. The AuK team says it plans a
variant on a permissively licensed encoder
([issue #13](https://github.com/Tencent-Hunyuan/AuK/issues/13)). Voice cloning:
only clone voices you have permission to use.

## Prerequisites
- An SDK token (`ik_sdk_`) and an API key (`ik_live_`) for the project, see
  [Tokens](../README.md#tokens).
- A registered NVIDIA worker assigned to the project (`IK_WORKER_ID`).
- Your tenant id and a dashboard session token: `publish.py` uploads the backend
  with them, and `main.py` uses the session token to turn on the workload's
  `chat` endpoint. Without them, upload `dist/*.tar.gz` from Dashboard -> Custom
  backends and flip "Chat" in the workload's Endpoint settings.

## Run
```bash
cp .env.example .env        # fill in real values
python -m venv .venv && . .venv/bin/activate
pip install maturin requests
( cd ../../bindings/python && maturin develop --release )
python publish.py           # once: packages backend.py and uploads it
python main.py              # ensure -> wait until ready -> 4 WAV files in ./outputs
```

`main.py` needs no audio of your own: it designs a voice from a text description,
clones that voice for a new sentence, replaces two words inside the first
recording and speeds it up to 1.25x.

CPU-offload variant: `python publish.py --variant offload`, then
`AUK_BACKEND_SLUG=auk-flash-speech-offload AUK_WORKLOAD_SLUG=auk-flash-offload python main.py`.

## Request shape
The backend answers the OpenAI chat route, `/v1/chat/completions`:

```json
{"model": "default",
 "messages": [{"role": "user", "content": [
   {"type": "text", "text": "Replace 'twelve percent' with 'eighteen percent'."},
   {"type": "input_audio", "input_audio": {"data": "<base64 WAV>", "format": "wav"}}]}],
 "extra_body": {"seed": 7}}
```

The message content of the answer is the backend's JSON (as a string):
`audio_b64` (16-bit WAV, 24 kHz mono), `seconds`, `input_seconds`, `gen_seconds`
and `timings` (`gpu_s`, `realtime_factor`, `peak_vram_mib`).

- Instruction templates: AuK's [cookbook](https://github.com/Tencent-Hunyuan/AuK/blob/main/docs/COOKBOOK.md).
  For voice design use `Based on the following description: "<voice>", generate
  speech content "<text>".` In our tests the cookbook's other English template
  (`Generate speech based on the following description: ... The content to speak
  is: ...`) often spoke a few words of the description before the text.
- Output length: `gen_seconds` sets it. Without it the backend estimates it from
  the quoted text (14 characters per second), from `gen_text` + `ref_text` for
  cloning (the AuK CLI's rule), from the speed factor, or keeps the input length.
- Limits: 30 s of output per request (class attribute `max_gen_seconds`); keep
  input recordings under about 15 s so the request stays under the platform's
  2 MB body limit.
- Languages: English and Chinese. A Spanish test sentence came out partly garbled.

## Measured (RunPod NVIDIA RTX A5000 24 GB, 2026-10-17)

| | |
| --- | --- |
| `ensure()` to ready, empty caches | 71.6 s: venv 25.8 s, weights 8.7 s (18.7 GB), imports and setup about 13 s, model load 22.1 s |
| VRAM (`nvidia-smi`) | 14,837 MiB loaded, 19,390 MiB peak; offload variant 1,571 MiB idle, 10,755 MiB peak |
| GPU time per request, warm | 0.41 s for 6 s of designed voice, 0.76 s for 18.5 s, 0.84 s for a 3.3 s clone, 0.66 to 0.77 s for edits of 4.3 to 5.8 s clips (4x to 24x faster than real time) |
| Offload variant | 12.7 to 15.2 s of GPU per request for the same `main.py` |
| Concurrency | one request at a time per replica (the backend serialises GPU work); 4 parallel edits took 5.8 s |

The pod cost 0.27 USD per hour; the whole session (both variants, about 60
requests and a demo recording) ran 22 minutes, 0.10 USD.

## Cost & cleanup
`fixed` keeps one replica on your worker until you delete the workload. The
script deletes it on exit (also on error and Ctrl-C). With `IK_KEEP_ALIVE=1` it
keeps it; then delete it with `ManagementClient.from_env().delete("auk-flash")`.
If the GPU is a rented pod, also stop the pod: deleting the workload does not.

## Troubleshooting
- **`/v1/audio/speech` answers 500 "worker returned no payload for passthrough
  route"**: the platform does not yet relay a custom backend's answer on the TTS
  route. Use `/v1/chat/completions` as above.
- **403 / `endpoint_kind_disabled` on the chat route**: a `text2audio` workload
  starts with only `audio_speech` and `jobs` on. Turn on `chat` (main.py does it).
- **Voice design reads part of the description aloud**: use the template above,
  or set `gen_seconds` close to the natural length of the text.
- **CUDA errors at load**: torch 2.7 from PyPI is the CUDA 12.6 build; the driver
  must support CUDA 12.6 or newer.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

# Qwen3-ASR + sglang on a private GPU — speech→text

Serves **Qwen3-ASR-1.7B** (multilingual ASR, 52 languages) with the **`sglang`**
backend on a private worker. The axis this varies vs the other examples: the
backend is `sglang` and the task is **audio2text** — sglang exposes an
OpenAI-compatible `/v1/audio/transcriptions` endpoint, so you upload an audio
file and get back a `{text}` transcription.

Verified end-to-end on an **NVIDIA RTX 4060 Ti (16 GB)**. It also runs on AMD
ROCm, but with sharp edges — see [ROCm caveats](#rocm-caveats).

## Compatibility
- SDK: local build in this repo (or `>= 0.1.0` once published)
- Language: Python >= 3.9
- Placement: private (any registered GPU worker; verified on NVIDIA)
- Backend: `sglang` (>= 0.5.11 for Qwen3-ASR)   Task: `audio2text`   Policy: `fixed`

## Prerequisites
- The tokens/ids this example needs — see
  [tokens & placement](../README.md#tokens): `INFERENCEKEY_SDK_TOKEN`,
  `INFERENCEKEY_API_KEY`, `INFERENCEKEY_PROJECT`, and the **private-worker** id
  `IK_WORKER_ID`.
- Python >= 3.9, plus `requests` (see `requirements.txt`).
- A local audio file (wav/mp3/flac/ogg) — set `IK_AUDIO_PATH` or drop a
  `sample.wav` next to `main.py`.
- A **registered GPU worker** with enough **host RAM** (see below).

## Run
```bash
cp .env.example .env        # fill in real values
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# install the local SDK — see ../CONTRIBUTING.md#4-depending-on-the-sdk:
pip install maturin && ( cd ../../bindings/python && maturin develop --release )
python main.py
```

## What it does
- **ensure()** — declares the workload (`backend=Backend.SGLANG`,
  `command="sglang serve --model-path Qwen/Qwen3-ASR-1.7B --served-model-name
  qwen3-asr --trust-remote-code --mem-fraction-static 0.60"`) pinned to your
  worker via `worker_id`, `fixed` policy with 1 replica. Idempotent by `slug`.
- **wait_until_ready()** — blocks while the cold worker installs sglang, pulls
  the model, and builds kernels (slow on the first boot).
- **transcribe** — POSTs the audio file as `multipart/form-data` to the
  OpenAI-compatible `/v1/audio/transcriptions` route and prints the text. Add
  `response_format=verbose_json` for `duration`/`language`/`segments`.
- **delete()** — tears the workload down on exit.

## Resource notes — read before deploying

### VRAM (the easy part)
The model is ~4.7 GB. With `--mem-fraction-static 0.60` the whole workload sits
in **~11 GB of VRAM** on a 16 GB card — comfortable. Lower the fraction to share
the GPU with other workloads; raise it if the GPU is dedicated.

### Host RAM (the real constraint)
**sglang is heavy on *system* RAM, not just VRAM.** It launches several Python
worker processes (scheduler, tokenizer, detokenizer, tp workers); the scheduler
alone needs **~3–4 GB of host RAM** before serving. On a small node this is the
thing that fails first — a 6 GB VM here got the scheduler **OOM-killed** during
init (`Rank 0 scheduler died during initialization (exit code: -9)`), while a
16 GB VM ran with room to spare. **Size host RAM for the sum of all the node's
workloads' footprints, not just their VRAM.**

### ROCm caveats
On AMD ROCm, sglang >= 0.5.11 pins `torch==2.11.0`, which resolves to the
`+rocm` wheel from the AMD staging index. sglang also imports **torchvision**
(`torchvision.io.decode_jpeg` on startup), and the ROCm torchvision wheel is
mismatched with the ROCm torch — importing it in the wrong order fails with
`RuntimeError: operator torchvision::nms does not exist` (a known ROCm-ecosystem
issue that also hits ComfyUI / SD-WebUI). Prefer an **NVIDIA** worker for this
backend, where the stock CUDA wheels install cleanly and none of the above
applies.

## Cost & cleanup
`fixed` + 1 replica keeps one always-on replica reserved (and billing) until
`delete()` runs. `main.py` calls `delete()` in a `finally`, so a normal run tears
the workload down. If you kill the process mid-run, delete it from the Manager UI
(or re-run and let the `finally` fire) so the GPU reservation is released.

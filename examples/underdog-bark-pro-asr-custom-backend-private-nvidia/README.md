# underdog-bark-pro-asr-custom-backend-private-nvidia

Conway Research's **Underdog Bark 3.0 Pro** (CC-BY-4.0) on your own NVIDIA GPU,
or on its CPU: a 0.6B English speech-to-text model, fine-tuned from NVIDIA
Parakeet TDT 0.6B v3, 1.25 GB of bf16 weights. It writes what was said,
fillers and repeats included. The example deploys it behind an
OpenAI-compatible `/v1/audio/transcriptions` endpoint, transcribes a file or a
folder, and scores word error rate when you give it reference transcripts.
`compare_qwen3_asr.py` runs the same clips through Qwen3-ASR-1.7B on the
built-in sglang backend for a side-by-side.

| | |
| --- | --- |
| Model | [`ConwayResearch/Underdog-Bark-3.0-Pro`](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Pro) at revision `4b5c51e0` (only the PyTorch files: 1.26 GB; the Core ML and MLX copies are skipped) |
| Backend | **custom backend** `underdog-bark-pro` (`backend.py`): stock Transformers `ParakeetForTDT`, the authors' preprocessing (mono 16 kHz, 0.25 s of leading silence), greedy decoding |
| API | `POST .../v1/audio/transcriptions`, multipart `file` + `model` (what `openai.audio.transcriptions.create` sends) or JSON `{"audio_b64"}`; answer `{"text"}` plus `duration`, `seconds`, `rtfx`, `vram_peak_gib` |
| Placement | Private, `fixed` policy, pinned to `IK_WORKER_ID` |
| Needs | NVIDIA driver for CUDA 12.8 (torch 2.8.0 wheels), Python 3.12 on the worker; about 2 GB of VRAM, or none with `BARK_DEVICE=cpu` |

## Why a custom backend

vLLM, SGLang and llama.cpp do not serve Parakeet TDT models (a FastConformer
encoder with a token-and-duration transducer decoder). Transformers 5.19 does,
with no custom code, so `backend.py` is under 200 lines: download, load,
resample, `generate()`, decode. Long audio is cut into `BARK_CHUNK_SECONDS`
pieces (default 600 s) because the encoder attends over the whole input and
its memory grows with length.

## What we measured (InferenceKey deploy test, 2026-10-27)

40 English clips from the Open ASR Leaderboard test sets (8 each from AMI,
Earnings-22, GigaSpeech, VoxPopuli and LibriSpeech test-other; 347.5 s, 924
words), plus one 5-minute file of 33 Earnings-22 segments. Word error rate
after the Whisper English normalizer. Same RTX 3090 for both models.

| | Bark 3.0 Pro (this example) | Qwen3-ASR-1.7B (sglang) |
| --- | --- | --- |
| Word error rate, 40 clips | **6.28 %** | 7.03 % |
| AMI meetings / Earnings-22 calls | 11.54 % / 8.15 % | 16.92 % / 10.87 % |
| VoxPopuli / LibriSpeech other / GigaSpeech | 8.04 % / 2.35 % / 3.54 % | 5.53 % / 1.88 % / 4.04 % |
| Round trip per clip (median) | 0.50 s | 0.73 s |
| 5-minute file in one request | 2.8 s on the GPU, whole transcript | 3.7 s, output stops early (454 of 658 words missing) |
| First start (venv + download + load) | 322 s (venv ~300 s, download 4 s, load 2.4 s) | 357 s |
| Warm restart | 7 to 8 s | not measured |

Qwen3-ASR drops fillers and repeated words ("you know", "on, on"); the
references keep them, so they count as errors. On LibriSpeech and VoxPopuli
Qwen3-ASR made fewer errors. 40 clips is a small sample: the authors' own
leaderboard-scale numbers are in the model card.

### Which hardware

| Where | Config | Peak memory | Speed (5-minute file) | Output |
| --- | --- | --- | --- | --- |
| RTX 3090 24 GB (real) | bf16, one request | 2,682 MiB on the card, 2.15 GiB allocated | 2.8 s (107x real time) | reference |
| 4 GiB cap (emulated) | bf16, one request | 2.15 GiB allocated | 2.6 s | identical |
| 2 GiB cap (emulated) | bf16, one request | over: `CUDA out of memory` on the 5-minute file; short clips fine (1.24 GiB) | - | fails on long audio |
| 2 GiB cap (emulated) | bf16, `BARK_CHUNK_SECONDS=60` | 1.37 GiB allocated | 2.5 s | identical on the 40 clips |
| CPU, 16 threads (AMD EPYC 7C13) | float32, 60 s chunks | no GPU | 14.1 s (21x) | identical on the 40 clips |
| CPU, 4 threads | float32, 60 s chunks | no GPU | 30.8 s (9.8x) | identical on the 40 clips |

Not tried on a real small card: every 6 to 16 GB GPU we asked for was out of
stock at test time.

## Things to know

- **Uploads are limited to 2 MiB per request** on `/v1/audio/transcriptions`:
  about 65 s of 16-bit 16 kHz WAV. Send long audio as Opus or MP3 (a 5-minute
  Opus file at 40 kbps is 1.1 MB) or split it.
- Re-running `main.py` against a workload that is already up can leave
  `wait_until_ready()` at "scheduling" while the endpoint keeps serving. Delete
  the workload first (or skip `ensure()`) when you only want to transcribe.
  Changing `BARK_*` variables also needs a delete and a fresh `ensure()`.
- English only. No word timings, no speaker labels.

## Run

```bash
cp .env.example .env              # fill in tokens, project, worker id, IK_AUDIO
python -m venv .venv && . .venv/bin/activate
pip install maturin requests jiwer whisper-normalizer
( cd ../../bindings/python && maturin develop --release )
python publish.py                 # once: uploads the backend to your tenant
python main.py                    # deploy, transcribe, score, delete
IK_KEEP_ALIVE=1 python main.py && python compare_qwen3_asr.py   # side-by-side
```

Example request (any OpenAI client works the same way):

```bash
curl -s "$INFERENCEKEY_BASE_URL/endpoint/$INFERENCEKEY_PROJECT/underdog-bark-pro/v1/audio/transcriptions" \
  -H "Authorization: Bearer $INFERENCEKEY_API_KEY" -F file=@call.wav -F model=default
```

Outputs land in `./outputs/bark-pro` and `./outputs/qwen3-asr` (one JSON per
clip and a `summary.json`).

## Attribution

Underdog Bark 3.0 Pro by Conway Research, CC-BY-4.0, derived from
nvidia/parakeet-tdt-0.6b-v3 (NVIDIA, CC-BY-4.0) and moondream/parakeet-ultra
(Moondream, CC-BY-4.0). Qwen3-ASR-1.7B by the Qwen team, Apache-2.0.

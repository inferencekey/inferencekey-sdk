# nemotron-transcribe-asr-custom-backend-private-nvidia

NVIDIA's **Nemotron 3.5 Transcribe** (OpenMDW-1.1, commercial use allowed) on
one private 32 GB NVIDIA Blackwell GPU. It is a speech LLM: an audio encoder
(ASR transformer plus a speaker encoder) in front of the Nemotron 3.5 Lightning
30B-A3B hybrid Mamba-Transformer mixture of experts, about 4.3B active
parameters. **The prompt picks the transcript**: verbatim with fillers,
cleaned, speaker-tagged with word timestamps, translated, or an answer about
the audio. 40 languages, recordings up to one hour.

`main.py` deploys it behind an OpenAI-compatible `/v1/audio/transcriptions`
endpoint and sends the same recording three times: `verbatim`, `clean` and
`speakers` (speaker tags, word timestamps and diarization segments).

| | |
| --- | --- |
| Model | [`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4) at revision `66c6adc4` (21.6 GB) |
| Backend | **custom backend** `nemotron-transcribe` (`backend.py`): builds vLLM 0.32.0 + the NeMo Speech vLLM plugin from the NeMo Speech lock file (commit `e5d79d5a`) with uv, starts `vllm serve` with the authors' flags sized for 32 GB, and forwards each request |
| API | `POST .../v1/audio/transcriptions`, multipart `file` + `model`, the instruction in `prompt` (a style name: `verbatim`, `clean`, `speakers`, `nearfield`, `translate`, or your own text), `timestamp_granularities[]=word` for word timestamps; or JSON `{"audio_b64", "style" or "prompt", "timestamps", "language"}` |
| Answer | `{"text", "style", "prompt", "duration", "seconds", "rtfx", "completion_tokens", "finish_reason", "ctc_timestamps"}` (`ctc_timestamps`: `words` with start/end/speaker, `diarization` segments, tag mapping, `error`) |
| Placement | Private, `fixed` policy, pinned to `IK_WORKER_ID` |
| Needs | A Blackwell GPU (NVFP4), NVIDIA driver for CUDA 13, about 50 GB of disk, 32 GB of VRAM for the defaults (24 GB with the knobs below) |

## Why a custom backend

The model runs on vLLM, but only with the NeMo Speech plugin
(`VLLM_PLUGINS=nemo_speechlm`), installed together with vLLM 0.32.0 from the
NeMo Speech repository's lock file. The built-in `vllm` backend installs vLLM
alone, so `backend.py` builds that environment itself (cached on the worker's
disk), downloads the checkpoint and runs `vllm serve` as a child process:

- **FlashInfer kernels.** On first start FlashInfer compiles NVFP4 GEMMs with
  `nvcc`. On our 62 GB container ninja ran one `nvcc` per core and the kernel
  killed them (`FAILED: [code=137] ... Killed`). The backend installs
  FlashInfer's prebuilt kernels for the card (`flashinfer-jit-cache` and
  `flashinfer-jit-cache-sm120f` from flashinfer.ai) and caps what still
  compiles with `MAX_JOBS=6`.
- **Clean shutdown.** The worker stops a custom backend with SIGKILL; the vLLM
  child gets `PR_SET_PDEATHSIG`, so it never outlives the backend and keeps the
  GPU.
- The authors' generation-loop guardrail is applied to every answer.

## What we measured (InferenceKey deploy test, 2026-11-01)

One RTX PRO 4500 Blackwell 32 GB (200 W), NVFP4, MTP speculative decoding
(8 tokens), CUDA graphs, `--max-model-len 65536`. Times are GPU time inside
the backend; round trips add about 0.3 to 0.6 s.

| | Result |
| --- | --- |
| First start, cold | environment 84 s, 21.6 GB download 18 to 39 s, vLLM start with kernel builds about 4 min |
| Warm start (environment, weights and kernels cached) | ready in 88 s from `ensure()` |
| VRAM | weights 20.7 GiB, KV cache 4.78 GiB (1.07 M tokens), 28.1 GB on the card |
| 3-minute two-speaker recording ([PriMock57](https://github.com/babylonhealth/primock57), CC BY 4.0) | verbatim 4.1 s, clean 4.1 s, speakers + word timestamps 4.4 s (561 timed words) |
| 5-minute earnings-call file, one request | 4.8 s (63x real time) |
| 40 English clips from the Open ASR Leaderboard sets (924 words) | 6.82 % word errors (Whisper normalizer), 13.0 s of GPU for 347.5 s of audio |

The `clean` prompt removed every filler (33 in those 3 minutes) but also added
speaker tags it was not asked for.

### Which GPU

| GPU | Configuration | Result |
| --- | --- | --- |
| RTX PRO 4500 Blackwell 32 GB | defaults (MTP 8, CUDA graphs, 65,536 tokens) | works, the numbers above |
| 24 GB hardware slice (RTX PRO 6000 Blackwell MIG 1g.24gb) | `NEMOTRON_GPU_MEMORY_UTILIZATION=0.93`, `NEMOTRON_MAX_MODEL_LEN=16384`, `NEMOTRON_MAX_NUM_SEQS=2`, `NEMOTRON_MTP=0`, `NEMOTRON_ENFORCE_EAGER=1` | works: same 6.82 % on the 40 clips, 3 minutes with speaker tags in 35 s (a quarter of a PRO 6000's compute) |
| same 24 GB slice | MTP 8 and CUDA graphs | fails at load (out of memory) |

NVFP4 needs Blackwell: an RTX 4090 or other Ada and older cards cannot run this
checkpoint (the FP8 checkpoint, 34.7 GB, needs a 48 GB card).

## Run it

```bash
cp .env.example .env          # fill in real values
pip install inferencekey requests   # client side (requirements.txt is the backend's)
python publish.py             # once: uploads backend.py to your tenant
python main.py                # ensure, wait, three prompts on IK_AUDIO, delete
```

`IK_KEEP_ALIVE=1` keeps the workload for more requests. Calling it with the
OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="https://cloud.inferencekey.com/endpoint/<project>/nemotron-transcribe/v1",
                api_key="ik_live_...")
out = client.audio.transcriptions.create(model="default", file=open("meeting.ogg", "rb"),
                                         prompt="speakers", timestamp_granularities=["word"])
```

The transcription route takes request bodies up to 2 MiB: send compressed
audio (Opus at 32 kbps holds about 8 minutes), or cut longer recordings.

## License

The model is under the [OpenMDW License 1.1](https://openmdw.ai/license/1-1/)
(NVIDIA); NeMo Speech and vLLM are Apache-2.0. This example is under the SDK's
license.

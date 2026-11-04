# hy-simulmt-speech-translation-custom-backend-private-nvidia

Tencent's **Hy-SimulMT-30B-A3B** (Apache-2.0) on one private 48 GB NVIDIA GPU.
It is a **simultaneous** speech translation model for Chinese and English: it
does not wait for the end of the sentence. It listens in 1.6 s blocks and,
after each block, either waits or writes the next piece of the translation,
so the listener reads it while the speaker is still talking. The Interleave
checkpoint used here also writes what it heard (the source transcript) next to
the translation. An optional Qwen3-TTS sidecar reads the translation aloud in
a clone of the speaker's voice.

`main.py` deploys it behind `/v1/audio/transcriptions` and streams each
recording to the model **at the speed it was spoken**, the way a microphone
would. The answer carries every partial result with the moment it arrived.

| | |
| --- | --- |
| Model | [`tencent/Hy-SimulMT-30B-A3B-Interleave-FP8`](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-Interleave-FP8) at revision `7f37cb76` (32 GB: experts in block FP8, attention in BF16) |
| Backend | **custom backend** `hy-simulmt` (`backend.py`): builds vLLM 0.32.0 + Tencent's `simulst` 0.1.0 plugin + their SimulInfer service ([Hy-SimulMT](https://github.com/Tencent-Hunyuan/Hy-SimulMT) commit `7efcb18c`) with uv, starts `vllm serve` with the authors' launcher and SimulInfer in front of it |
| API | `POST .../v1/audio/transcriptions`, multipart `file` + `direction` (`zh-en` or `en-zh`; or `language` = `zh`/`en`, the source), `transport` (`realtime`, the default: paced at speaking speed; or `file`: as fast as the GPU goes), `tts` (`true` with the voice sidecar on) |
| Answer | `{"text", "source_text", "direction", "transport", "audio_s", "wall_s", "first_text_ms", "delay_ms" {median, p90, max}, "events" [{t_ms, start_ms, end_ms, text, source_text}], "complete", "tts" {first_audio_ms, seconds, wav_b64}}` |
| Placement | Private, `fixed` policy, pinned to `IK_WORKER_ID` |
| Needs | An NVIDIA GPU with 40 GB or more for the defaults (48 GB with the voice), NVIDIA driver for CUDA 13, about 70 GB of disk. Each request body is limited to 2 MiB (about one minute of 16 kHz WAV, several minutes of FLAC or Opus) |

## Why a custom backend

The model runs on stock vLLM, but only with Tencent's `simulst` plugin, and the
simultaneous part (cutting audio into blocks, voice activity detection, the
conversation context, deciding when to speak) lives in their SimulInfer
service, not in the model server. `backend.py` builds both, caches them on the
worker's disk and runs them as child processes. Three things it fixes:

- **The FP8 checkpoint is not recognized by SimulInfer** (2026-11-04): it reads
  the output format from `export_manifest.json` or a known checkpoint name, and
  the FP8 repo has neither, so SimulInfer stops with "无法识别所选模型的输出格式".
  The backend writes the manifest (source + translation for Interleave).
- **Encoder profiling.** With the authors' `--max-num-batched-tokens 16384`,
  vLLM profiles the audio encoder with 3,276 dummy clips; on a 48 GB card that
  threw thousands of out-of-memory retries before it recovered. The backend
  uses 4096.
- **The voice sidecar** needs vLLM-Omni 0.31.0rc1 on vLLM 0.31.0: SimulInfer's
  voice-clone launcher imports `Qwen3TTSSpeakerEncoder` from a module that
  vLLM-Omni 0.32.0 and 0.32.0rc1 no longer have. The backend builds that pair in
  a second environment.

The worker stops a custom backend with SIGKILL; the servers get
`PR_SET_PDEATHSIG`, and a restarted backend stops any server an earlier process
left on the GPU.

## What we measured (InferenceKey deploy test, 2026-11-04)

One RTX 6000 Ada 48 GB (Ada, sm_89, 300 W). vLLM 0.32.0 runs the block-FP8
experts with its Marlin weight-only FP8 kernel on this card (its log: "Your
GPU does not have native support for FP8 computation"). Test audio: four
FLEURS clips (CC BY 4.0), 50 to 56 s each, five or six read sentences, two in
Chinese and two in English, scored against FLEURS' parallel sentences.

| | Result |
| --- | --- |
| First start, cold | environment 47 s, 32 GB download 26 s, vLLM start about 4.5 min (import, load 27.5 s, encoder profiling); ready after about 6 min |
| Warm start (environment and weights cached) | ready in 73 s from `ensure()` |
| VRAM | weights 30.05 GiB, KV cache 5.09 GiB (55,552 tokens) at `gpu_memory_utilization` 0.78; 38.3 GB on the card while serving |
| Streamed at speaking speed | each 1.6 s block answered 0.43 to 0.81 s (median per clip) after it was spoken, p90 1.5 to 1.8 s; first words 2.3 to 5.5 s after the clip started |
| As fast as the GPU goes (`transport=file`) | 50 to 56 s of speech in 25 to 27.5 s |
| Quality (chrF against FLEURS) | Chinese to English 60.6 and 51.7; English to Chinese 42.2 and 23.4 |
| With the voice (Qwen3-TTS-12Hz-0.6B-Base on the same card) | Chinese to English: first translated speech after 8.7 s, 33.7 s of English audio for a 49.8 s clip; ready in 376 s; 48.3 GB of the 49.1 GB card used. English to Chinese with voice stalled (no answer after 4 minutes) |

What went wrong: in one English clip the model skipped a whole sentence
("Inland waterways can be a good theme...", twice, in two separate runs),
reduced another to five characters and flipped "couldn't have included" into
"could include"; the answer came back with `complete: false`.

Smaller cards (emulated on the 48 GB card, not real cards):

| Budget | Configuration | Result |
| --- | --- | --- |
| 48 GB | defaults (util 0.78, 8,192-token context, 4 sessions) | reference |
| 32 GB cap | `HYSIMULMT_GPU_MEMORY_UTILIZATION=0.66`, `HYSIMULMT_MAX_MODEL_LEN=4096`, `HYSIMULMT_MAX_NUM_SEQS=1` | starts, 7,136 tokens of KV cache, 32.3 GB on the card (no headroom on a real 32 GB card); both clips translated, chrF 62.8 / 37.6 |
| 24 GB | the authors' GGUF Q4_K_M (19.7 GB, translation only) on llama.cpp PR #31198, `llama-mtmd-cli`, offline file | 20.9 GB on the card, about 16x faster than real time, all sentences, chrF 66.1 / 42.3. Not servable yet: llama-server support for this model is a follow-up of that PR |

The cap only limits what vLLM keeps after it starts: during startup its audio
encoder profiling briefly filled the whole 48 GB card in every run (it hit
out-of-memory, retried and recovered), so a real 32 GB card may fail at that
step. Not tried on a real 32 GB or 24 GB card.

## Run it

```bash
cp .env.example .env      # fill in the tokens, the worker id and your clips
pip install -r requirements.txt
python publish.py         # once: uploads backend.py as the custom backend hy-simulmt
python main.py
```

`main.py` prints, per clip, when the first words arrived, how far behind the
speaker each block was answered, the source transcript and the translation,
and saves everything in `outputs/<clip>.json` (and `<clip>.translated.wav`
with `HYSIMULMT_TTS=1`). It deletes the workload at the end unless
`IK_KEEP_ALIVE=1`.

## Credits

Hy-SimulMT by Tencent Hunyuan (Apache-2.0), with the SimulInfer service and the
`simulst` vLLM plugin (Apache-2.0). vLLM and vLLM-Omni (Apache-2.0). Qwen3-TTS
by the Qwen team (Apache-2.0). FLEURS by Google (CC BY 4.0).

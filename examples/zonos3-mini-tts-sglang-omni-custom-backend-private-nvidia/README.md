# ZONOS3-Mini speech on SGLang-Omni (custom backend, private NVIDIA)

Serves [ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini) from Zyphra
(Apache-2.0): text to speech in 43 languages with voice cloning from a short
clip and **no transcript**. The backbone is a small hybrid Gated DeltaNet /
attention mixture of experts (800M parameters in all, about 100M active per
token) that writes 15 Hz codec frames, rendered to 44.1 kHz audio as they are
produced. The server for NVIDIA GPUs that the authors point to is
[SGLang-Omni](https://github.com/sgl-project/sglang-omni) (`sgl-omni serve`),
which no built-in InferenceKey backend runs, so this example ships a **custom
Python backend** that starts SGLang-Omni next to it and answers the workload's
OpenAI-compatible **`/v1/audio/speech`** route.

## Compatibility
- SDK: local build in this repo (see [CONTRIBUTING](../CONTRIBUTING.md#4-depending-on-the-sdk))
- Language: Python >= 3.9 for the client scripts. The backend itself is a thin
  client; it builds a **separate Python 3.12 venv** for SGLang-Omni (pinned
  commit `bac76771`: torch 2.14.1, SGLang 0.5.23, CUDA 13 wheels), because that
  dependency set needs pre-release resolution
- Placement: private NVIDIA, **driver for CUDA 13.0 or newer (580+)**, **a 24 GB
  card** (see below), about 30 GB of free local disk (14.5 GB of weights plus
  the venv). Keep the cache on the local disk, not on a network volume
- Backend: custom (`zonos3-mini-sglang-omni`)   Policy: `fixed`

## What we measured (2026-11-15)
Median, warm, a 9 s English line; "direct" = streaming PCM from the SGLang-Omni
server on the GPU host, "API" = the whole WAV through `/v1/audio/speech`.

| | NVIDIA L4 24 GB | RTX 4090 24 GB |
| --- | --- | --- |
| Ready after `ensure()`, empty caches | first start hit the memory-share error below (venv 177 s, weights 72 s) | 215 s (venv 54 s, weights 13 s) |
| Ready after `ensure()`, venv and weights cached | 98 s | - |
| Time to first audio, direct, 1 / 4 / 8 streams | 70 / 104 / 188 ms | 50 / 96 / 106 ms |
| Cloning from a 9.4 s clip, first audio, 1 / 8 streams | 99 / 193 ms | 55 / 102 ms |
| Real-time factor, direct, 1 / 8 streams | 0.031 / 0.079 | 0.023 / 0.049 |
| Round trip through the API, 1 / 4 / 8 at once | 0.41 / 0.70 / 1.12 s | 0.42 / 0.57 / 0.78 s |
| GPU memory, card peak (nvidia-smi) | 17.6 GiB | 18.1 GiB |

Whisper heard every word of our English, Spanish, German and French clips;
cloned clips scored 0.89 to 0.97 speaker similarity to the reference
(WavLM-base-plus-sv), against 0.44 to 0.71 for the default voice. Numbers are
the weak spot (see below).

**Which GPU:** the model file is under 2 GB, but the running server holds more
than 16 GiB: the codec and the vocoder stage load next to the language model.
With smaller limits (8 running requests, CUDA graphs up to batch 8) it still
peaked at 16.4 GiB, so a 16 GB card will not fit this server as it ships (we
did not try one). Both 24 GB cards worked.

## Prerequisites
- An SDK token (`ik_sdk_`, with `assignment:write`) and an API key (`ik_live_`)
  for the project, see [Tokens](../README.md#tokens).
- A registered NVIDIA worker assigned to the project (`IK_WORKER_ID`).
- Your tenant id and a dashboard session token for `publish.py`, or upload
  `dist/*.tar.gz` from Dashboard -> Custom backends.

## Run
```bash
cp .env.example .env        # fill in real values
python -m venv .venv && . .venv/bin/activate
pip install maturin requests
( cd ../../bindings/python && maturin develop --release )
python publish.py           # once: packages backend.py and uploads it
python main.py my-voice.wav # ensure -> wait until ready -> 4 WAV files in ./outputs
```

The default voice reads an English line, a preset voice (`Ximena_es_F_1`) reads
a Spanish one, and if you pass a clip (0.5 s or more; about 10 s works well)
the model clones it for an English line and a Spanish one. Use a voice you have
permission to clone.

## Request shape
```bash
curl -s "$INFERENCEKEY_BASE_URL/endpoint/$INFERENCEKEY_PROJECT/zonos3-mini/v1/audio/speech" \
  -H "Authorization: Bearer $INFERENCEKEY_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "default", "input": "Your order has shipped.", "voice": "Imogen_en-British_F_1", "seed": 7}' \
  -o speech.wav
```

| Field | Meaning |
| --- | --- |
| `input` | the text to say (up to 3,000 characters) |
| `voice` | `default`, or one of the 27 preset voices (`Imogen_en-British_F_1`, `Rupert_en-British_M_1`, `Ximena_es_F_1`, `Ananya_hi_F_1`... the names of the clips in the model's `gguf/voices/`) |
| `ref_audio` | a clip to clone, base64 (WAV, MP3, FLAC or OGG) or a data URI; no transcript |
| `condition_values` | `{"speaking_rate": <UTF-8 bytes per second>, "loudness": <LUFS>, "fmax": <Hz>}`; we measured the same line at 5.07 s with `speaking_rate` 9, 3.93 s by default and 3.00 s with 22 |
| `seed`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_new_tokens` | sampling, passed to SGLang-Omni |

The answer is a WAV (44.1 kHz, 16-bit mono). Through the platform it comes back
whole: `"stream": true` is refused for custom backends. Inside, the backend
streams from SGLang-Omni and measures the time to first audio (`timings` in
the backend's JSON, which the async `/jobs` route keeps; we did not use that
route in this test).

## Things we hit
- **SGLang-Omni's default memory share is sized for an 80 GB card.** On a 24 GB
  card the server stops with `Colocated GPU memory budget leaves no KV-cache
  headroom: total=22.49GiB, fraction=0.300, budget=6.75GiB, process_used=7.32GiB`.
  `main.py` sets `ZONOS3_ENGINE_GPU_MEMORY_FRACTION=0.55`
  (`--tts_engine.gpu_memory_fraction`); 0.36 also works, 0.33 does not.
- **One request at a time by default.** A workload created by `ensure()` starts
  with `max_in_flight = 1` on its worker, so concurrent calls queued: about 17 s
  of audio per second at 1, 4 or 8 clients on the L4. `main.py` raises it to 16
  (`allow_parallel_requests`): 59 s of audio per second at 8 clients on the L4,
  63 to 80 on the 4090.
- **Numbers**: "Order 4521 ... $1,249.99 ... Call 555-0142" was heard as "451",
  "$1,049" and "5555-0142" (the date and the percentage came out right). Spell
  out numbers that matter.
- **The `default` voice is not one fixed voice**: it changes with the seed.
  Pass a preset `voice` or a `ref_audio` for a consistent voice.
- **Voice design (age, accent, whisper on a span) is not in this server.**
  Zyphra's own runtimes (CUDA, MLX, ggml, WebGPU) expose those controls; the
  SGLang-Omni path gives cloning, the preset voices and the three conditions.
- `sgl-omni serve --model-path Zyphra/ZONOS3-Mini` downloads the whole repository
  (GGUF and WebGPU shards included). The backend downloads only the 14.5 GB that
  SGLang-Omni reads. The release on PyPI (0.2.0) predates ZONOS3, hence the pinned
  commit.

## Licence and responsible use
Weights are Apache-2.0. Only clone voices you have permission to use, and label
synthetic speech as synthetic where it could be mistaken for a real recording.

# Examples

Runnable, self-contained examples of the **InferenceKey SDK** — declare an AI
workload in code, ensure it exists on the platform, wait for it to be ready, and
call its OpenAI-compatible endpoint. Each example is a folder you can copy, set a
few environment variables, and run.

This folder is the SDK's front door: if you want to see what the SDK *does*
before reading any reference docs, start here. Every example follows the same
three-step shape — **`ensure()` → wait until ready → call the endpoint** — and
varies one axis at a time (where the model runs, which backend, which hardware)
so you can find the one that matches your setup.

> **Status: early development.** The catalogue below grows over time. An entry
> with no link is planned, not shipped. The published packages
> (`inferencekey` on PyPI, `@inferencekey/sdk` on npm) are not out yet, so today
> every example depends on the **local SDK build in this repo** — see
> [CONTRIBUTING.md](./CONTRIBUTING.md#4-depending-on-the-sdk) for the one-time
> build step.

## How to read this

The only thing that really changes between examples is **placement** — *where*
the model runs:

- **Cloud** — the platform schedules the workload on shared capacity. You set no
  hardware ids. This is the simplest path and where you should start.
- **Private worker** — the workload is pinned to a GPU box *you* registered, via
  `worker_id` + `gpu_resource_id`. The vendor and GPU architecture (NVIDIA, or
  AMD ROCm) live on the worker, **not** in the spec — the same
  `vllm serve <model>` command runs on both; only the worker image and a few
  serve flags differ. Examples encode the target hardware in their folder name
  so you can tell at a glance which box they expect.

## Catalogue

| Example | Lang | Placement | Hardware | Backend | Policy | What it shows |
| --- | --- | --- | --- | --- | --- | --- |
| _(planned)_ `hello-world-cloud` | Py / TS | Cloud | platform-chosen | `vllm` | `autoscaling` | The smallest end-to-end run: ensure → ready → one completion. **Start here.** |
| _(planned)_ `chat-streaming-cloud` | Py / TS | Cloud | platform-chosen | `sglang` | `autoscaling` | Token-by-token streaming chat with a local history REPL. |
| _(planned)_ `chat-streaming-private-nvidia` | Py | Private | NVIDIA | `vllm` | `fixed` | Pin a chat workload to your NVIDIA worker + a specific GPU resource. |
| _(planned)_ `chat-streaming-private-amd-gfx120x` | Py | Private | AMD R9700 (gfx120x) | `vllm` | `fixed` | Same, on an AMD ROCm RDNA4 worker — note the ROCm-specific serve flags. |
| [`gguf-llamacpp-private-amd`](./gguf-llamacpp-private-amd) | Py | Private | AMD R9700 (gfx120x) | `llamacpp` | `fixed` | Serve a **GGUF** model (Gemma 4 26B) with `llama-server` on an AMD ROCm worker. The `command` points at a GGUF repo/file (`-hf <repo> --hf-file <file.gguf>`); no venv, no safetensors. |
| [`gguf-embedding-llamacpp-private-amd`](./gguf-embedding-llamacpp-private-amd) | Py | Private | AMD R9700 (gfx120x) | `llamacpp` | `fixed` | Serve a **GGUF embedding** model (Qwen3-Embedding-8B) with `llama-server --embedding --pooling last` and call `embed()`. Same GGUF/`llamacpp` path as above but modality `embedding` — the only embedding runtime on RDNA4, where `sglang` is unsupported. |
| [`gguf-reranker-llamacpp-private-amd`](./gguf-reranker-llamacpp-private-amd) | Py | Private | AMD R9700 (gfx120x) | `llamacpp` | `fixed` | Serve a **GGUF reranker** (Qwen3-Reranker-0.6B) with `llama-server --reranking --pooling rank` and call `rerank(query, documents)` → documents ranked by relevance. Modality `reranker` — again the only reranking runtime on RDNA4. |
| [`text2image-stablediffusioncpp-private-nvidia`](./text2image-stablediffusioncpp-private-nvidia) | Py | Private | NVIDIA (Volta sm_70+) | `stablediffusioncpp` | `fixed` | Generate a **1024×1024 image** (Z-Image Turbo, GGUF) with the `stablediffusioncpp` backend served by `sd-server`. Modality `text2image`; the `command` points at three model files (`--diffusion-model` / `--vae` / `--llm`) and `--auto-fit` spreads them across the GPUs. Runs on **older GPUs sglang can't** (compiles CUDA kernels for the local arch, like llama.cpp) — validated on 2× Tesla V100. POSTs to `/v1/images/generations` (no typed image helper yet). |
| [`astabrief8b-cited-report-vllm-private-nvidia`](./astabrief8b-cited-report-vllm-private-nvidia) | Py | Private | NVIDIA, 24 GB+ | `vllm` | `fixed` | Ai2's **AstaBrief-8B** (Apache-2.0 Qwen3-8B fine-tune) turns a research question plus **your own retrieved excerpts** into a sectioned report with a `TLDR` per section and inline `[ID]` citations, in one pass. Uses the exact prompt Ai2 trained it on (downloaded at run time), counts which supplied references the report cites, and documents the checks to run before showing it. Ready in 246 s on a cold RTX A5000; about 39 tokens/s per report, about 133 tokens/s with four at once. |
| [`nemotron35-bolt-74b-vision-agent-llamacpp-private-nvidia`](./nemotron35-bolt-74b-vision-agent-llamacpp-private-nvidia) | Py | Private | NVIDIA, 32 GB+ | `llamacpp` | `fixed` | NVIDIA's **Nemotron 3.5 Bolt 74B-A9B** (OpenMDW-1.1), a 74B vision MoE shipped in 2-bit (22.0 GB GGUF), with its **DFlash drafter** for speculative decoding, on NVIDIA's 28GB llama.cpp recipe (98,304 context, q8_0 KV). One `llama-server` command pulls the model, vision projector and drafter from the Hub; the example asks a reasoning question, reads a receipt image and runs a two-tool agent loop. Needs the worker on llama.cpp `v0.7.0`+ (`INFERENCEKEY_LLAMACPP_TAG`). On an A40: 25.2 GiB of VRAM, about 136 tokens/s per request with the drafter and 70 without. |
| [`flint-coding-agent-llamacpp-private-nvidia`](./flint-coding-agent-llamacpp-private-nvidia) | Py | Private | NVIDIA, 24 GB+ | `llamacpp` | `fixed` | Recursive's **Flint** (Apache-2.0), a 39.8B coding-agent MoE with about 2.5B active, from its official **NVFP4 GGUF** (22.9 GB) at 65,536 context. The example asks a coding question, then runs a small **coding-agent loop** (list, read, write files, run pytest) that fixes a bug in a throwaway repo. Needs the worker on a llama.cpp build with PR #30928 (`INFERENCEKEY_LLAMACPP_TAG`). Measured: RTX 5090, 23,872 MiB, about 210 tokens/s per request; RTX 4090 at 16K context, 22,316 MiB, about 167 tokens/s (32K ran out of memory). |
| [`deepseek-v41-flash-vllm-exl3-private-nvidia`](./deepseek-v41-flash-vllm-exl3-private-nvidia) | Py | Private | NVIDIA 2x RTX PRO 6000 (sm_120) | `vllm-exl3` | `fixed` | Serve **DeepSeek-V4.1-Flash** (763B MoE, 3-bit EXL3) with host-memory expert offload and chat with it through the gateway using the official `openai` client (`model="default"`). The `command` is only `vllm serve <pack> --revision <sha> --served-model-name ...`; the worker pins the stack. Long cold start (install, 309 GiB download, load), so it waits up to 3600 s. `IK_KEEP_ALIVE=1` keeps the endpoint after the script ends. |

There is a fuller, two-language streaming demo today at
[`docs/sdk-demo`](../../../docs/sdk-demo) in the parent repo; the examples here
are the canonical, per-scenario successors to it.

### Custom backends (bring your own PyTorch)

The catalogue above is about the **placement** flow (`ensure()` → ready → call
the endpoint). A different axis is writing your **own** inference backend in
PyTorch and serving it with the SDK's `CustomBackend` contract:

| Example | Lang | What it shows |
| --- | --- | --- |
| [`custom-backend-echo`](./custom-backend-echo) | Py | The smallest `CustomBackend`: load a trivial `nn.Module` **once** in `setup()`, process jobs in `process()`, served over the SDK's loopback HTTP runtime (`python -m inferencekey.backend.serve`). No weights downloaded. |
| [`custom-backend-openai-chat`](./custom-backend-openai-chat) | Py | A `CustomBackend` that implements `chat()`: the runtime serves OpenAI `/v1/chat/completions` with SSE streaming and tool calls, so the workload works with Claude Code over `/v1/messages`. Forwards to an upstream OpenAI-compatible server, or runs a scripted mode with no model. |
| [`custom-backend-deepinfer`](./custom-backend-deepinfer) | Py | DeepSeek-V4.1-Flash on DeepSeek's DeepInfer engine with OpenAI chat (streaming, DSML tool calls, usage) for Claude Code. One process per data-parallel rank with a least-loaded router. Fake engine for tests; untested on real B200s. |
| [`custom-backend-text-sentiment`](./custom-backend-text-sentiment) | Py | The **text** representative: a tiny in-memory sentiment classifier (`nn.Embedding` + `nn.Linear`) over a fixed vocabulary. Real text jobs (`input.text`/`prompt` → `output.label`), `GET /meta` metadata, and the `serve_backend` code helper. No weights downloaded. |
| [`custom-backend-music-caption`](./custom-backend-music-caption) | Py | A real-world **`audio2text`** backend wrapping **LP-MusicCaps**: given a musical clip (`input.audio_b64`) it returns a natural-language description of the song (`output.description`). Loads a BART captioning model **once** in `setup()`, downloading the weights from Hugging Face on first start (never shipped in the package). Vendors an inference-only subset of `lpmc/`. Needs `transformers==4.26.1` on CPython 3.9–3.11. |
| [`chronos2-forecast-private-amd-gfx120x`](./chronos2-forecast-private-amd-gfx120x) | Py | The **`forecast`** representative: a `CustomBackend` wrapping Amazon **Chronos-2** (`amazon/chronos-2`) that implements the SDK's typed `ForecastRequest → ForecastResult` contract (not the free-dict `process` shape) and serves `POST /forecast`. Given a series history (`target` + optional past/future covariates) it returns a `median` + `quantiles` forecast for the next `horizon` steps. Loads the ~120M-param model **once** in `setup()` (weights pulled from Hugging Face, never shipped). Pinned to a private AMD R9700 (gfx120x); needs CPython 3.10+. |
| [`timesfm-forecast-private-amd-gfx120x`](./timesfm-forecast-private-amd-gfx120x) | Py | A second **`forecast`** backend, wrapping Google **TimesFM 2.5** (`google/timesfm-2.5-200m-pytorch`, ~200M). Same typed `ForecastRequest → ForecastResult` contract and `POST /forecast`, but **univariate** — it forecasts from `target` alone and ignores covariates (use Chronos-2 for covariate-aware forecasts). Maps requested `quantile_levels` onto TimesFM's fixed quantile grid. Loads + compiles the model **once** in `setup()`. Pinned to a private AMD R9700 (gfx120x); needs CPython 3.10+. |
| [`moirai2-forecast-private-amd-gfx120x`](./moirai2-forecast-private-amd-gfx120x) | Py | A third **`forecast`** backend, wrapping Salesforce **Moirai 2.0** (`Salesforce/moirai-2.0-R-small`) via `uni2ts` (installed from GitHub) + GluonTS. Same typed contract and `POST /forecast`; **univariate**, quantile-based — `median` is Moirai's native q0.5 and requested `quantile_levels` are read off the `QuantileForecast` (interpolated between its native 0.1 … 0.9). Loads the module **once** in `setup()`. Pinned to a private AMD R9700 (gfx120x); needs CPython 3.10+. |
| [`kronos-forecast-private-amd-gfx120x`](./kronos-forecast-private-amd-gfx120x) | Py | A fourth **`forecast`** backend, wrapping **Kronos** for financial **OHLCV** candlesticks. The one that varies the *data shape*: `target` is the close price and `past_covariates` carry `{open, high, low, volume}`. Kronos isn't pip-installable, so its code is **vendored** in `kronos_model/` (MIT). Generative → quantiles come from `n_paths` sampled runs. Loads the model **once** in `setup()`. Pinned to a private AMD R9700 (gfx120x); needs CPython 3.10+. |
| [`iris3b-text2image-custom-backend-private-nvidia`](./iris3b-text2image-custom-backend-private-nvidia) | Py | The **`text2image`** custom backend for a model no built-in backend serves: **Iris-3B** (Sperid Labs, Apache-2.0), a pixel-space generator with no VAE. Drives the authors' PyTorch code (source fetched at a pinned commit, weights from Hugging Face, nothing heavy in the 5 KB package), answers the OpenAI images shape, and includes `publish.py` + `main.py` for the full publish -> ensure -> generate flow. Turbo: 4 steps, about 2 s per 1024x1024 image on an A40; the 100-step base model is a second entrypoint. Private NVIDIA; needs CPython 3.11+ on the worker. |
| [`moss-image-sft-text2image-custom-backend-private-nvidia`](./moss-image-sft-text2image-custom-backend-private-nvidia) | Py | A **`text2image`** custom backend for a diffusers *custom pipeline* (`trust_remote_code`): OpenMOSS's **MOSS-Image-SFT** (Apache-2.0, 5B DiT + Qwen3.5-9B text encoder + FLUX.2 small VAE), strong at text in images and **JSON layout prompts with bounding boxes**. One workload serves two modes per request: the 50-step SFT recipe (CFG 3) and a community **8-step DMD LoRA** applied unmerged (no CFG). Measured on an A40 48 GB: ready 63 s after `ensure()` on a cold pod, 79 to 81 s per 1024x1024 image in SFT, 9 s in turbo, about 30 GB of VRAM; on smaller cards it offloads by itself (under 12 GB at 160 s per SFT image, emulated cap) |
| [`ming-image-design-text2image-custom-backend-private-nvidia`](./ming-image-design-text2image-custom-backend-private-nvidia) | Py | A **`text2image`** custom backend that wraps the authors' own inference code (`infer.py`, pinned commit): inclusionAI's **Ming-Image-0.1-Design** (MIT, 6B DiT conditioned on a Ling MoE LLM, 4-channel VAE) for UI mockups, posters and infographics, with **RGBA output** (`"background": "transparent"` puts the authors' phrase on its own line). Measured on one A100 80GB in BF16: 53 GB of weights downloaded in 69 s and loaded in 23 s, 6.2 to 7.4 s per 1024x1024 image and 36 s per 2048x2048 image at 12 steps, 62.8 GiB peak allocated at 2048. Offloaded modes for smaller cards are in the code but untested |
| [`embeddinggemma2-multimodal-embedding-custom-backend-private-nvidia`](./embeddinggemma2-multimodal-embedding-custom-backend-private-nvidia) | Py | The **multimodal `embedding`** custom backend: Google DeepMind's **EmbeddingGemma 2** (Apache-2.0, 740M) puts text, code, images, video and audio in one 768-d space. Answers `/v1/embeddings` with OpenAI-style JSON plus media items (data URI or URL), interleaved inputs, Matryoshka `dimensions` and task prefixes; decodes audio and video with PyAV. `main.py` runs text search through `embed()` and cross-modal search over a media folder. About 1.5 GB of VRAM loaded; 0.11 s per image, 0.16 s per 8.6 s of speech on an RTX 2000 Ada. A text-only entrypoint skips the vision and audio encoders (0.5 GB). Private NVIDIA; needs CPython 3.11+ on the worker. |
| [`bonsai2-27b-ternary-chat-custom-backend-private-nvidia`](./bonsai2-27b-ternary-chat-custom-backend-private-nvidia) | Py | A **`text2text`** custom backend that brings its own runtime: Prism ML's **Ternary Bonsai 2 27B** (Apache-2.0, ternary weights, 5.95 GB GGUF) only loads in Prism's llama.cpp fork, so `setup()` downloads the fork's CUDA 12.8 build plus the GGUF and vision projector and runs its `llama-server`; `process()` forwards OpenAI chat requests, images and `reasoning_effort`. Ready 55 s after `ensure()` on a fresh RTX 4090, about 104 tokens/s per request, 11.4 GiB of VRAM at 64K context. Private NVIDIA (driver for CUDA 12.8+); needs CPython 3.11+ on the worker. |
| [`limite-1b-violetto-math-custom-backend-private-nvidia`](./limite-1b-violetto-math-custom-backend-private-nvidia) | Py | A **`text2text`** custom backend that speaks OpenAI chat with streaming: Paradigma's **Limite 1B - Violetto** (Apache-2.0, 1B dense, trained from scratch for competition math) needs Paradigma's out-of-tree vLLM plugin, so `requirements.txt` installs vLLM 0.26.0 plus the plugin and `setup()` runs `vllm serve` with `VLLM_PLUGINS=limite`. `main.py` sends AIME 2026 problems at once and grades the `\boxed{}` answers. Measured on an RTX 4090: about 211 tokens/s per request, 27 of 30 problems right on one try (2 wrong, 1 with no answer), still correct with vLLM capped at about 4 GB; on an RTX 2000 Ada 16 GB, 85 tokens/s and ready 300 s after `ensure()` on a fresh pod. Private NVIDIA (driver for CUDA 13.0+), Python 3.12 on the worker. |
| [`wevisone-vl-4b-perception-custom-backend-private-nvidia`](./wevisone-vl-4b-perception-custom-backend-private-nvidia) | Py | A custom backend for a **vision perception** model: Tencent's **WeVisOne-VL-4B** (Apache-2.0, Qwen3-VL plus a 9.6M-parameter pixel decoder) does detection, referring with rejection, counting, pointing, OCR, segmentation and alpha matting in one model. `backend.py` drives the authors' package (vLLM for the text, a Transformers copy teacher-forced for the masks); requests go to `/v1/images/generations` as `{"task", "query", "image"}` and come back as annotated image, RGBA cut-out and structured boxes/points/counts. Measured: on an RTX 4090 a 1620x1080 alpha matte in about 1 s and object-level answers in 0.1 to 3.7 s (vLLM engine, 23.5 GB peak); every task on a real RTX 5080 16 GB with the Transformers engine (13.1 GB peak). Private NVIDIA (driver for CUDA 13.0+), Python 3.12 on the worker. |
| [`auk-flash-speech-custom-backend-private-nvidia`](./auk-flash-speech-custom-backend-private-nvidia) | Py | A **`text2audio`** custom backend for speech generation **and editing**: Tencent Hunyuan's **AuK-Flash** (MIT, 1.5B, 4 diffusion steps) takes one natural-language instruction plus an optional recording and does voice design, zero-shot cloning, word replacement inside a recording, pitch / speed / emotion / whisper, denoising and speaker separation. Drives the authors' PyTorch code (source at a pinned commit, AuK-Flash + the Qwen2.5-Omni-3B encoder from Hugging Face, 18.7 GB). Answers on `/v1/chat/completions` with a base64 WAV. Ready 72 s after `ensure()` with empty caches on an RTX A5000, 0.4 to 0.8 s of GPU per request, about 19 GiB of VRAM at peak (a CPU-offload entrypoint needs about 10.5 GiB, about 13 s per request). The encoder is under the Qwen research licence (non-commercial). Private NVIDIA; CPython 3.10 to 3.12 on the worker. |
| [`omnivoice-narration-custom-backend-private-amd-gfx120x`](./omnivoice-narration-custom-backend-private-amd-gfx120x) | Py | A **`text2audio`** custom backend for **voice narration** on an AMD R9700 (gfx120x): k2-fsa's **OmniVoice** (Apache-2.0, ~0.8B, 600+ languages) narrates text with **saved voices shipped inside the package** (`voz1`, `voz2`: OmniVoice voice-clone prompts of a few KB), or designs a voice on the fly from `instruct`. Splits long texts into sentence chunks, returns WAV, FLAC or MP3 base64 on `/v1/chat/completions`, ~4 GB of VRAM. Measured on the R9700: about 2x real time once warm. |

## Hello world

The canonical entry point is **`hello-world-cloud`**: the fewest moving parts —
cloud placement, no hardware ids, three environment variables — so you can prove
your tokens work before touching anything else. Once it ships, running it will
look like this (Python):

```bash
cd examples/hello-world-cloud
cp .env.example .env          # then fill in your tokens + project
# ...follow the example's own README to install the local SDK and run
```

Until then, the [`docs/sdk-demo`](../../../docs/sdk-demo) is the closest runnable
thing.

## Prerequisites (all examples)

- An InferenceKey account and project at
  [cloud.inferencekey.com](https://cloud.inferencekey.com).
- A **control-plane** SDK token (`ik_sdk_…`) and a **data-plane** API key
  (`ik_live_…`). They are **not** interchangeable: the control token provisions
  workloads but cannot call inference; the data key calls inference but cannot
  provision. See [Tokens](#tokens) below.
- The runtime for the example's language: **Python ≥ 3.9** or **Node ≥ 18**.
- The **local SDK build** (until the packages are published) —
  [CONTRIBUTING.md → Depending on the SDK](./CONTRIBUTING.md#4-depending-on-the-sdk).

## Tokens

Every example reads its secrets from the environment (or a local `.env`); nothing
is hard-coded. The common variables:

| Env var | Value | Where to get it |
| --- | --- | --- |
| `INFERENCEKEY_SDK_TOKEN` | `ik_sdk_…` — control plane | Project → **Tokens** → create SDK token |
| `INFERENCEKEY_API_KEY` | `ik_live_…` — data plane | **API Keys** → mint a key for the workload |
| `INFERENCEKEY_PROJECT` | project slug (e.g. `acme`) | the project you opened |
| `INFERENCEKEY_BASE_URL` | _(optional)_ self-hosted/staging URL | defaults to the public API |
| `IK_WORKER_ID` | `wrk_…` — **private examples only** | **Workers** → your registered GPU box |
| `IK_GPU_RESOURCE_ID` | `gpu_…` — **private examples only** | **GPU Resources** → the GPU's opaque id |

Cloud examples need only the first three (plus the optional base URL). Private
examples additionally need `IK_WORKER_ID` and `IK_GPU_RESOURCE_ID`.

## A note on cost

Examples that run on a **private worker** with a `fixed` policy keep a replica
**always on** until you tear it down — that GPU is reserved (and billing) the
whole time. Cloud examples use `autoscaling` so the platform can scale to zero.
**Every example that provisions GPU ends by deleting its workload** (`delete()`),
and its README spells out the teardown. If you Ctrl-C mid-run, re-run the example
(or call `delete()` on the slug) to make sure nothing is left running.

## Contributing an example

Adding one? Read [CONTRIBUTING.md](./CONTRIBUTING.md). It defines the folder and
naming convention, the required files, how to depend on the local SDK, and the
cost/teardown and secret-hygiene rules every example must follow.

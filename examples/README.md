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
| [`custom-backend-text-sentiment`](./custom-backend-text-sentiment) | Py | The **text** representative: a tiny in-memory sentiment classifier (`nn.Embedding` + `nn.Linear`) over a fixed vocabulary. Real text jobs (`input.text`/`prompt` → `output.label`), `GET /meta` metadata, and the `serve_backend` code helper. No weights downloaded. |
| [`custom-backend-music-caption`](./custom-backend-music-caption) | Py | A real-world **`audio2text`** backend wrapping **LP-MusicCaps**: given a musical clip (`input.audio_b64`) it returns a natural-language description of the song (`output.description`). Loads a BART captioning model **once** in `setup()`, downloading the weights from Hugging Face on first start (never shipped in the package). Vendors an inference-only subset of `lpmc/`. Needs `transformers==4.26.1` on CPython 3.9–3.11. |
| [`chronos2-forecast-private-amd-gfx120x`](./chronos2-forecast-private-amd-gfx120x) | Py | The **`forecast`** representative: a `CustomBackend` wrapping Amazon **Chronos-2** (`amazon/chronos-2`) that implements the SDK's typed `ForecastRequest → ForecastResult` contract (not the free-dict `process` shape) and serves `POST /forecast`. Given a series history (`target` + optional past/future covariates) it returns a `median` + `quantiles` forecast for the next `horizon` steps. Loads the ~120M-param model **once** in `setup()` (weights pulled from Hugging Face, never shipped). Pinned to a private AMD R9700 (gfx120x); needs CPython 3.10+. |
| [`timesfm-forecast-private-amd-gfx120x`](./timesfm-forecast-private-amd-gfx120x) | Py | A second **`forecast`** backend, wrapping Google **TimesFM 2.5** (`google/timesfm-2.5-200m-pytorch`, ~200M). Same typed `ForecastRequest → ForecastResult` contract and `POST /forecast`, but **univariate** — it forecasts from `target` alone and ignores covariates (use Chronos-2 for covariate-aware forecasts). Maps requested `quantile_levels` onto TimesFM's fixed quantile grid. Loads + compiles the model **once** in `setup()`. Pinned to a private AMD R9700 (gfx120x); needs CPython 3.10+. |
| [`moirai2-forecast-private-amd-gfx120x`](./moirai2-forecast-private-amd-gfx120x) | Py | A third **`forecast`** backend, wrapping Salesforce **Moirai 2.0** (`Salesforce/moirai-2.0-R-small`) via `uni2ts` (installed from GitHub) + GluonTS. Same typed contract and `POST /forecast`; **univariate**, quantile-based — `median` is Moirai's native q0.5 and requested `quantile_levels` are read off the `QuantileForecast` (interpolated between its native 0.1 … 0.9). Loads the module **once** in `setup()`. Pinned to a private AMD R9700 (gfx120x); needs CPython 3.10+. |
| [`kronos-forecast-private-amd-gfx120x`](./kronos-forecast-private-amd-gfx120x) | Py | A fourth **`forecast`** backend, wrapping **Kronos** for financial **OHLCV** candlesticks. The one that varies the *data shape*: `target` is the close price and `past_covariates` carry `{open, high, low, volume}`. Kronos isn't pip-installable, so its code is **vendored** in `kronos_model/` (MIT). Generative → quantiles come from `n_paths` sampled runs. Loads the model **once** in `setup()`. Pinned to a private AMD R9700 (gfx120x); needs CPython 3.10+. |

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

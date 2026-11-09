# SenseNova-Loop-9B-Preview on a private GPU: a looped 9B that reads charts and calls tools

Serves SenseTime's **SenseNova-Loop-9B-Preview** (Apache-2.0, text and images
in, text out) as a **custom Python backend**. The model stores 32 decoder
layers but runs 56 per token: layers 4 to 27 run twice with the same weights,
and inside that loop two residual streams are mixed by learned mHC routers.
The axis this example varies: a model no released serving engine runs yet.
The SGLang and vLLM support is in open pull requests, so the backend drives
the checkpoint's own Transformers code (`trust_remote_code`), the path the
authors checked (Transformers 5.18.0, PyTorch 2.11.0).

`main.py` sends three requests to one workload: a word problem with thinking
on, a chart image with a question about it, and a tool call.

## Compatibility
- SDK: local build in this repo (or `>= 0.1.0` once published)
- Language: Python >= 3.9 (client); CPython 3.10 to 3.13 on the worker
- Placement: private (NVIDIA, bf16-capable: Ampere or newer), **20 GB of GPU memory or more**
- Backend: custom (`sensenova-loop-9b`, `backend.py`)   Task: `text2text` (chat route)   Policy: `fixed`

## Prerequisites
- The tokens/ids this example needs, see [tokens & placement](../README.md#tokens):
  `INFERENCEKEY_SDK_TOKEN`, `INFERENCEKEY_API_KEY`, `INFERENCEKEY_PROJECT`, `IK_WORKER_ID`.
- For `publish.py`: `INFERENCEKEY_TENANT_ID` and a dashboard session token in
  `INFERENCEKEY_USER_TOKEN` (or upload `dist/*.tar.gz` in Dashboard -> Custom backends).
- A registered NVIDIA worker with about **19 GB of free GPU memory** for short
  prompts (21 GB for a 12K-token prompt), about 40 GB of free disk (the
  checkpoint ships mostly FP32 shards) and a driver for CUDA 12.8 or newer.
- `samples/energy-chart.png` is a made-up bakery electricity chart drawn for this example.

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
- **ensure()** declares the workload (`backend="sensenova-loop-9b"`, `task_type="text2text"`)
  pinned to your worker, `fixed` policy, one replica, with `ready_timeout_secs`
  raised to 3600 for the first download. Idempotent by slug.
- **wait_until_ready()** blocks while the worker builds the backend venv
  (torch 2.11.0, transformers 5.18.0, flash-linear-attention), the backend
  downloads the checkpoint at a pinned revision (about 38 GB, the MTP head is
  skipped) and loads it in bf16 (about 17.5 GiB), then warms up.
- **Three calls** on `/v1/chat/completions`. The message content is the
  backend's JSON: `content` (the answer), `reasoning` (the thinking),
  `tool_calls` (OpenAI shape, parsed from the model's `<tool_call>` blocks) and
  `timings` (prompt and completion tokens, tokens/s, first token, peak VRAM,
  layer runs per token). Answers land in `./outputs`.
- **delete()** tears the workload down on exit (`IK_KEEP_ALIVE=1` keeps it).

Request options (in `extra_body`): `enable_thinking` (default true, as the
model card), `max_tokens`, `temperature` (greedy when 0 or missing), `top_p`,
`tools` (top level, OpenAI function schemas), and `loop_passes` (default 2,
the trained setting): `1` runs each stored layer once, an ablation to see
what the second pass buys (it is twice as fast and the answers fall apart).
Message parts: `text` and `image_url` as `data:` URLs (the backend does not fetch URLs).

## Measured (InferenceKey deploy test, 2026-11-09)

| | A40 48 GB | RTX 4000 Ada 20 GB |
| --- | --- | --- |
| Ready after `ensure()` | 57 s (weights already on disk) | 107 s (fresh worker, weights on disk) |
| Word problem, thinking on (306 tokens) | 101 s, 3.0 tokens/s, right | 139 s, 2.25 tokens/s, right (310 tokens) |
| Chart question, thinking on (455 tokens) | 157 s, 2.9 tokens/s, right | 204 s, 2.25 tokens/s, right |
| Tool call (73 tokens) | 30 s, `get_exchange_rate(USD, EUR)` | 34 s, same call |
| Long prompt, hidden code | 12,046 tokens: right, first token in 5.3 to 14.6 s | 3,046 tokens: right (first token 11.1 s); about 6K tokens: out of memory |
| GPU memory (PyTorch peak) | 18.1 to 18.3 GiB short prompts, 19.9 GiB at 12K tokens | 18.1 to 18.3 GiB; 18.2 GiB at 3K tokens |

Decode is slow on this path: about 2.3 to 3 tokens/s with thinking off or on,
on both cards. Transformers' decode loop is launch-bound (one CPU core at 100 %,
the GPU at about 40 to 50 %) and every token runs 56 layers, so a faster GPU
does not help much. Under an emulated cap (`LOOP_VRAM_LIMIT_GIB`) the chart,
the word problem and the 12K-token prompt passed at 22.5 GiB; at 18.5 GiB the
short prompts passed (the 20 GB card above confirms it) and the 12K-token prompt ran out of memory;
at 14.5 GiB the weights do not load (17.54 GiB in one allocation).

## Cost & cleanup
One always-on replica on your worker (`fixed`) until `delete()` runs; the
script deletes it on exit unless `IK_KEEP_ALIVE=1`. On Ctrl-C, run
`python -c "from inferencekey import ManagementClient as M; print(M.from_env().delete('sensenova-loop'))"`.

## Troubleshooting
- **`Model '...' is not available on this endpoint`**: send `"model": "default"`.
- **`cannot import name 'chat_completion'`**: older SDK runtimes on PyPI route chat to
  `process()`; this backend does that, so publish the version in this folder.
- **Slow answers**: expected on the Transformers path (see above). The SenseNova
  Loop support in SGLang is an open pull request (sgl-project/sglang#47281):
  built from it on the same A40, the same prompts gave the same answers at
  9.2 tokens/s. vLLM support (vllm-project/vllm#64832) is text only.
- **MTP**: the checkpoint ships an MTP head (`mtp.safetensors`), but the
  Transformers code ignores it, the vLLM pull request lists it as unsupported,
  and the SGLang pull request does not wire it (with a one-line patch the head
  loads, then the draft step crashes). This backend skips that file.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

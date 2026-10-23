# custom-backend-deepinfer

DeepSeek-V4.1-Flash served by DeepSeek's [DeepInfer](https://github.com/deepseek-ai/DeepInfer)
engine as an InferenceKey custom backend, with OpenAI chat: streaming, tool
calls and usage. The workload's `/v1/chat/completions` and Anthropic
`/v1/messages` endpoints work, so Claude Code can program against it.

> **Untested on real hardware.** Everything below the engine boundary was
> written from the DeepInfer source and has only run with the fake engine. The
> list of what is unverified is at the end. Read it before renting B200s.

## How it works

| File | Role |
| --- | --- |
| `deepinfer_backend.py` | `DeepInferBackend`: implements `chat()` / `chat_stream()`. Maps OpenAI messages and tools to the DeepSeek-V4.1 prompt, streams reasoning and text, parses DSML tool calls back into OpenAI `tool_calls`, reports usage. |
| `deepinfer_engines.py` | `DeepInferEngine` (one process per data-parallel rank, each with a DeepInfer `LLM`, plus a least-loaded router) and `FakeEngine` (deterministic, no GPU). |
| `dsv41_encoding.py` | The model's official prompt encoder and completion parser, vendored unmodified from the model repository (MIT, `LICENSE-dsv41-encoding.txt`). |

Per request: the OpenAI request becomes a DeepSeek-V4.1 prompt (tools on the
system message, tool results as `<tool_result>` blocks, `<think>` in thinking
mode); the engine streams the completion; reasoning before `</think>` goes out
as `reasoning_content`, text as `content`, and a `<｜DSML｜ calls>` block becomes
`tool_calls`, sent once complete. DeepInfer has no load balancer across
data-parallel replicas in its open release; the router here sends each request
to the rank with the fewest requests in flight.

## Requirements (engine `deepinfer`)

- NVIDIA SM100 GPUs (B200 or GB200). DeepInfer's README names SM100; B300
  (sm_103) is not confirmed.
- At least 4 B200 to load the weights (about 510 GB) with room for context; the
  DeepInfer examples run one 8-GPU node (`ep 8`).
- Linux, CUDA 13, Python 3.12, GCC 13 or newer.
- DeepInfer built from source with its `build.sh` (it is not on PyPI).
- A local copy of `deepseek-ai/DeepSeek-V4.1-Flash-1017` (about 510 GB). The
  backend never downloads it on its own.

### Getting DeepInfer into the backend's environment

DeepInfer's `build.sh` compiles in place (no installable wheel) and installs
torch, its vendored CUDA kernels and transformers into the active Python. The
worker gives every custom backend a fresh venv, so DeepInfer is built once into
its own venv on the node (for example `/workspace/di-venv` on a network volume,
with a `.pth` file pointing at the DeepInfer checkout) and the backend uses it
through `deepinfer_venv`: its site-packages are appended to `sys.path` and the
rank processes start with its interpreter. Both venvs must use the same Python
minor version (3.12). `requirements.txt` stays empty.

## Workload config

| Key | Default | Meaning |
| --- | --- | --- |
| `engine` | `deepinfer` | `deepinfer` or `fake`. |
| `env.CUDA_VISIBLE_DEVICES` | all | GPUs for this backend, e.g. `"0,1,2,3"`. Rank *r* gets `cp` consecutive entries. |
| `env.DEEPINFER_HF_MODEL_DIR` or `model_dir` | none | Local checkpoint directory (must hold `config.json`). |
| `ep`, `dp`, `cp` | `4`, `ep / cp`, `1` | Expert, data and context parallel sizes (`dp * cp` GPUs, a multiple of `ep`). |
| `dspark` | `true` | DSpark speculative decoding (`DEEPINFER_ENABLE_MTP_VARLEN`). |
| `nextn_decode` | `5` | Draft length (`DEEPINFER_USE_NEXTN_DECODE`, 0 disables). |
| `swa_bounded_replay`, `swa_keep_tokens` | `true`, `140` | SWA bounded replay (`DEEPINFER_YOCO_PREFILL_*`). |
| `eplb`, `extra_experts` | `true`, `32` | Expert load balancing and redundant experts. |
| `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization` | `262144`, `64`, `32768`, `0.85` | Engine limits. Untuned. |
| `deepinfer_venv` | none | Prebuilt venv with DeepInfer (see above), e.g. `/workspace/di-venv`. |
| `deepinfer_env`, `llm_kwargs` | none | Extra DeepInfer environment variables and `LLM(...)` arguments. |
| `thinking_mode`, `reasoning_effort` | `thinking`, `high` | Prompt mode (`thinking` or `chat`) and effort (1 to 100, or `low` / `high` / `max`). |
| `temperature`, `top_p`, `default_max_tokens` | `0.6`, `1.0`, `8192` | Sampling defaults when the request has none. |
| `served_model_name` | `deepseek-v4.1-flash` | `model` field of the answers. |
| `load_timeout_seconds` | `3600` | How long `setup()` waits for every rank to load. |
| `ready_timeout_secs` | worker default (300) | Platform readiness wait. Set it above the load time, e.g. `5400`. |
| `request_timeout_secs` | 300 | Longest request, a whole stream included. Raise it for long agent turns, e.g. `1800`. |

Example:

```json
{
  "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "DEEPINFER_HF_MODEL_DIR": "/workspace/v41flash"},
  "ep": 8,
  "ready_timeout_secs": 5400,
  "request_timeout_secs": 1800
}
```

## Package and publish

```bash
python -m inferencekey.backend.package \
  --src examples/custom-backend-deepinfer \
  --entrypoint deepinfer_backend:DeepInferBackend \
  --requirements examples/custom-backend-deepinfer/requirements.txt \
  --name deepseek-v41-flash-deepinfer --version 0.1.0 --task-type text2text \
  --requires-python ">=3.12,<3.13" --api openai-chat --out dist
```

Publish the artifact (SDK `publish_custom_backend` or the dashboard), create a
`text2text` workload with `backend="deepseek-v41-flash-deepinfer"` and the config
above, and point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL="https://<manager>/endpoint/<project>/<workload>"
export ANTHROPIC_API_KEY="ik_live_..."
export ANTHROPIC_MODEL="deepseek-v4.1-flash"
```

## Try it without a GPU

```bash
export PYTHONPATH="$PWD/bindings/python/python:$PWD/examples/custom-backend-deepinfer"
python -m inferencekey.backend.serve --port 8099 --backend deepinfer_backend:DeepInferBackend \
  --config-json '{"engine": "fake", "fake_tool_arguments": {"file_path": "/tmp/a.txt"}}'
```

Tests: `bindings/python/tests/test_example_deepinfer.py`.

Throughput of the parent process (router, result queue, detokenization, SSE)
with simulated ranks, no GPU: `bench/README.md`.

## What is unverified (check on the first B200 run)

1. DeepInfer builds and imports in the backend's venv (wheel from `build.sh`, `deep_cluster` included).
2. `LLM(...)` accepts the arguments used here (taken from `examples/_config.py` and `offline_inference.py`) and loads the checkpoint with `ep 4` on 4 B200 and `ep 8` on 8.
3. A long-lived lockstep loop works: idle ranks keep calling `has_unfinished_requests()` (an all-reduce) and `step()` with no local work, for hours.
4. Requests can be added between steps while others are in flight (the offline example adds all requests before the first step).
5. `flush_inflight_reqs()` after the last step returns the remaining outputs and stays in lockstep.
6. One `RequestOutput` per generated token (with DSpark on, accepted draft tokens arrive as separate outputs), and `prompt_len` is the prompt length.
7. Decoding with the checkpoint's tokenizer, `skip_special_tokens=False`, yields the DSML and `</think>` markup the parser expects.
8. The model actually emits DSML tool calls in the format of `dsv41_encoding.py` for Claude Code's tools, and Claude Code accepts the `reasoning_content` it does not use.
9. The context length (`max_model_len`), batch limits and memory settings that fit 4 and 8 B200, and the time to load (to size `ready_timeout_secs`).
10. Whether DeepInfer reuses prompt prefixes across requests (a cache). Without one, every agent turn reads its whole context again.

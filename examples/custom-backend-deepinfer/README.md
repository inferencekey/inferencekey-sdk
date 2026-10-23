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
| `deepinfer_engines.py` | `DeepInferEngine` (one process per data-parallel rank, each with a DeepInfer `LLM`, plus a session-affinity router) and `FakeEngine` (deterministic, no GPU). |
| `prefix_cache.py` | The GPU prefix cache of one rank: chain hashes of 128-token blocks, the index of cached blocks and its LRU eviction (pure Python, no torch). |
| `dsv41_encoding.py` | The model's official prompt encoder and completion parser, vendored unmodified from the model repository (MIT, `LICENSE-dsv41-encoding.txt`). |

Per request: the OpenAI request becomes a DeepSeek-V4.1 prompt (tools on the
system message, tool results as `<tool_result>` blocks, `<think>` in thinking
mode); the engine streams the completion; reasoning before `</think>` goes out
as `reasoning_content`, text as `content`, and a `<｜DSML｜ calls>` block becomes
`tool_calls`, sent once complete. DeepInfer has no load balancer across
data-parallel replicas in its open release; the router here sends each request
to the rank with the fewest requests in flight.

The backend process serves every HTTP stream under one GIL, so the heavy work
stays in the rank processes:

- A rank tokenizes the prompts it receives on a small thread pool, off its
  lockstep loop (the Hugging Face fast tokenizer releases the GIL while it
  encodes, so a 137K-token prompt does not stall the other ranks).
- A rank detokenizes its completions incrementally, per request. For
  DeepSeek-V4.1's byte-level BPE tokenizer that is a token-to-bytes table plus
  an incremental UTF-8 decoder, checked at startup against the tokenizer's own
  `decode`; any other tokenizer gets the generic prefix/read window of vLLM and
  TGI. A character split across tokens goes out once complete.
- A rank sends at most one message per engine step, with the new text and the
  token counts of every request that advanced (usage stays exact).
- In the backend process a reader thread hands each item to its request, and
  the request's thread sends everything that arrived since its last chunk as
  one SSE chunk, at most every `stream_interval_ms` (longer when the number of
  streams would exceed `stream_chunks_per_second` in total).
- A client that disconnects frees its slot, and its rank aborts the request.
  A disconnect is only seen when a write fails, so a stream that has had
  nothing to send for `stream_heartbeat_seconds` (queued, prefilling, or
  generating a tool call, which goes out only once complete) writes an SSE
  comment (`: keepalive`), which clients ignore.

What fails, and how:

- Prompts are tokenized without the tokenizer's own BOS
  (`add_special_tokens=False`): the encoded prompt already starts with it, and
  DeepInfer's release tokenizer forces `add_bos_token` on.
- A prompt with a token id outside the model vocabulary (a message that quotes
  a reserved special token such as `<|place_holder_mm_span_0000|>`) is a 400.
  DeepInfer would drop it without an error and the client would wait forever.
- A prompt the engine will not schedule (longer than `max_model_len` or than
  the KV cache) is a 400 with the OpenAI error code `context_length_exceeded`,
  so a client can compact and retry; when that is known only after the stream
  started, the stream ends with an error event. A request DeepInfer stops for
  lack of memory, or a structured-output or vision failure, ends with an
  error, never with the bogus token DeepInfer reports for it.
- The ranks run in lockstep, so when one rank dies the others block in the
  next collective. The backend then fails every open stream, answers new
  requests with a 500 at once and stops the remaining ranks; restart the
  backend. The same happens if no rank produces anything for
  `stall_timeout_seconds` while requests are in flight. A single request with
  no output for `request_stall_seconds` fails on its own.
- At unload the ranks get `close_timeout_seconds` to exit, then are terminated
  (and killed if needed): a clean lockstep exit is not guaranteed, and a rank
  left behind would keep its GPUs and the process alive. A failed `setup()`
  stops the ranks it started.

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

## Prefix cache

A coding agent resends its whole conversation on every turn, and almost all of
it is the previous turn. DeepInfer's open release frees every KV block when a
request finishes, so each turn would prefill its whole context again (a 1M-token
context is about 11 s of one rank). With `prefix_cache` on (the default):

- Each rank keeps the KV blocks of finished prompts, pinned in the engine's
  block manager and named by a chain hash of their tokens, and starts a new
  request from its longest cached prefix. Only prompt blocks are kept: with
  `drop_thinking`, the next turn renders the previous answer differently. The
  last cached block is recomputed with the new tokens (SWA bounded replay), so
  a hit is the prefix minus 128 tokens. Shared prefixes (one system prompt for
  many sessions) are stored once.
- Cached blocks are evicted least recently used first when new prompts need
  room. The cache keeps free the engine's prefill watermark (10 % of the pool)
  plus `prefix_cache_free_fraction`.
- Every turn of a conversation goes to the rank that holds its prefix: the
  session is the OpenAI `user` field, or else a hash of the messages up to the
  first user message. A new session goes to the rank with the least KV in use.
- The ranks tokenize a prompt piece by piece between chat-template markers and
  keep the ids of long pieces, so a warm turn only tokenizes its new pieces
  (the full encode of 1M tokens costs about 3 s of CPU).
- `usage.prompt_tokens_details.cached_tokens` reports the prompt tokens served
  from the cache; requests may set `ignore_eos`.

It needs DeepInfer with a small patch that adds `LLMEngine.enqueue_with_prefix`,
`Scheduler.add_sequence_with_prefix` and `Sequence.block_table`. The prebuilt
package [inferencekey/deepinfer-sm100-cu132](https://huggingface.co/inferencekey/deepinfer-sm100-cu132)
ships DeepInfer built with it (`patches/deepinfer-prefix-cache.patch`, and
`rebuild-deepinfer-C.sh` to apply it to another tree). Without it, set
`prefix_cache: false`.

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
| `tokenizer_threads` | `2` | Prompt tokenization threads per rank. |
| `prefix_cache` | `true` | GPU prefix cache and session routing (see "Prefix cache"). Needs the patched DeepInfer. |
| `prefix_cache_free_fraction` | `0.05` | KV pool kept free by eviction on top of the 10 % prefill watermark. |
| `session_migrate_inflight` | `4 * max_num_seqs` | A session moves to another rank (one cold turn) only when its rank has this many requests in flight. |
| `session_table_size` | `100000` | Sessions remembered by the router (LRU). |
| `tokenizer_cache_tokens` | `64000000` | Token ids of repeated prompt pieces kept per rank. `0` disables. |
| `stats_interval_seconds` | `2.0` | How often each rank reports its KV and cache counters to the router. |
| `log_rank_stats_seconds` | `30` | How often the backend logs one line per rank: sessions, requests in flight, KV used, cached blocks, hits, evictions, preemptions. `0` disables. |
| `stuck_request_seconds` | `120` | A rank logs, once, any request with no output for this long and where it sits. `0` disables. |
| `stream_interval_ms` | `50` | Shortest time between two chunks of one stream: tokens that arrive meanwhile go out together. `0` sends every engine step on its own (more CPU in the backend process). |
| `stream_chunks_per_second` | `8000` | Budget of SSE chunks per second over all streams: with more than 400 streams the interval grows (1,000 streams: 125 ms) so the backend process keeps up. `0` disables. |
| `abort_on_disconnect` | `true` | Abort the request in DeepInfer when its client disconnects. |
| `stream_heartbeat_seconds` | `1.0` | After this long with nothing to send, a stream writes an SSE keepalive comment, so a client that left is noticed (and its request aborted) while the request is queued, prefilling or generating a tool call. `0` disables. |
| `request_stall_seconds` | `1800` | A request with no output for this long fails and frees its slot (a backstop for a request the engine lost). `0` disables. |
| `stall_timeout_seconds` | `600` | With requests in flight and no output from any rank for this long, the engine is considered hung: every stream fails. `0` disables. |
| `close_timeout_seconds` | `30` | At unload, how long the ranks get to exit before they are terminated. |
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
6. One `RequestOutput` per generated token (with DSpark on, accepted draft tokens arrive as separate outputs), and `prompt_len` is the prompt length. Several outputs of one request in one step are handled.
7. Decoding with the checkpoint's tokenizer, `skip_special_tokens=False`, yields the DSML and `</think>` markup the parser expects. (The rank's byte table reproduces that `decode` exactly for the DeepSeek-V4.1 `tokenizer.json` with transformers 4.50; checked offline, and again by every rank at startup.)
8. The model actually emits DSML tool calls in the format of `dsv41_encoding.py` for Claude Code's tools, and Claude Code accepts the `reasoning_content` it does not use.
9. The context length (`max_model_len`), batch limits and memory settings that fit 4 and 8 B200, and the time to load (to size `ready_timeout_secs`).
10. Whether DeepInfer reuses prompt prefixes across requests (a cache). Without one, every agent turn reads its whole context again.
11. `llm_engine.abort_request()` between steps, while other requests run and with DSpark in-flight steps, leaves the engine and the lockstep healthy. Set `abort_on_disconnect` to `false` if it does not.
12. The tokenizer from `llm.get_tokenizer()` can encode on the rank's tokenizer threads while the loop steps (encode only; nothing changes its settings). With the Hugging Face fast tokenizer it releases the GIL, as measured offline.
13. Requests the scheduler ignores or stops (`FINISHED_IGNORED`, `FINISHED_PREEMPTED`, `FINISHED_SO_FAILED`, `FINISHED_VE_FAILED`) arrive as one finished `RequestOutput` with `finish_reason` set (`prompt_length_exceeded`, `insufficient_system_resource`, `xgrammar ...`, `ve_error`), as read in `outputs.hpp`. Not seen on hardware yet; send a prompt over `max_model_len` and check for the 400.
14. The backend process at 200 to 400 streams on the B200 host: per-user tok/s and its CPU. In the benchmark it sits at about 100 to 130 % CPU at 400 streams (per chunk, not per token), with no `/v1/messages` adapter or proxy in front. If it is near 100 % there, raise `stream_interval_ms`, lower `stream_chunks_per_second`, or split the HTTP layer across processes; treat about 400 streams as the limit of one backend process until then.
15. The rank's added CPU stays small next to a GPU step. On the dev VM, grouping and detokenizing one step of 128 requests takes about 0.24 ms (2 us per request), and building and checking the byte table takes about 0.5 s at startup. Not measured: `RequestOutput.clone` and the queue message on the real engine.

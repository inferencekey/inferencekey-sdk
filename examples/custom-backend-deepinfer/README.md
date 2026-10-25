# custom-backend-deepinfer

DeepSeek-V4.1-Flash served by DeepSeek's [DeepInfer](https://github.com/deepseek-ai/DeepInfer)
engine as an InferenceKey custom backend, with OpenAI chat: streaming, tool
calls and usage. The workload's `/v1/chat/completions` and Anthropic
`/v1/messages` endpoints work, so Claude Code can program against it.

> **Run on 4x B200 (2026-10-24 and 25).** Load, serving, the prefix cache,
> the L1 host cache, tool calls and a Claude Code session work on real
> hardware with the configuration in "Tested on 4x B200" at the end, which
> also lists the measured capacity and what is still unverified (8 GPUs,
> B300). Read it before renting GPUs.

## How it works

| File | Role |
| --- | --- |
| `deepinfer_backend.py` | `DeepInferBackend`: implements `chat()` / `chat_stream()`. Maps OpenAI messages and tools to the DeepSeek-V4.1 prompt, streams reasoning and text, parses DSML tool calls back into OpenAI `tool_calls`, reports usage. |
| `deepinfer_engines.py` | `DeepInferEngine` (one process per data-parallel rank, each with a DeepInfer `LLM`, plus a session-affinity router) and `FakeEngine` (deterministic, no GPU). |
| `prefix_cache.py` | The GPU prefix cache of one rank: chain hashes of 128-token blocks, the index of cached blocks and its LRU eviction (pure Python, no torch). |
| `l1_cache.py` | The host RAM tier of the prefix cache: chain hash to host slot, LRU, demotions in flight, quarantine (pure Python, no torch). |
| `kv_mover.py` | Copies KV blocks between the GPU and pinned host memory: the mover interface and `TorchKVMover` (PyTorch, side streams, events). |
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

- NVIDIA SM100 GPUs (B200 or GB200). Tested on 4x B200 with driver 595.91
  (CUDA 13.2); B300 (sm_103) is not confirmed.
- At least 4 B200 to load the weights with room for context; the DeepInfer
  examples run one 8-GPU node (`ep 8`), which we have not run.
- Linux, CUDA 13, Python 3.12. The prebuilt package below needs the system
  packages `zstd` and `libdw-dev`; building from source also needs GCC 13 or
  newer.
- DeepInfer with the prefix-cache patch (it is not on PyPI): the prebuilt
  package or a source build, see below.
- A copy of `deepseek-ai/DeepSeek-V4.1-Flash-1017` (about 510 GB, 476 GiB on
  disk) on **local disk**. The backend never downloads it on its own. DeepInfer
  reads the weights with `mmap`: from a network filesystem (a RunPod pod
  volume or network volume mounted on `/workspace` is MooseFS) a load takes
  about 13 min or stalls; from local NVMe the whole load takes about 4 min. On RunPod,
  create the pod without a volume (`volumeInGb: 0`) and a container disk of
  about 900 GB, so `/workspace` is local.
- Host RAM for the L1 host cache when it is on (`l1_host_gb` per rank, 128 GiB
  in our runs; the backend lowers it to what the container can spare).

### Getting DeepInfer into the backend's environment

The worker gives every custom backend a fresh venv, so DeepInfer lives in its
own venv on the node (for example `/workspace/di-venv`, on local disk) and the
backend uses it through `deepinfer_venv`: its site-packages are appended to
`sys.path` and the rank processes start with its interpreter. Both venvs must
use the same Python minor version (3.12). `requirements.txt` stays empty.

Two ways to get that venv:

- **Prebuilt (what we use):** download
  [inferencekey/deepinfer-sm100-cu132](https://huggingface.co/inferencekey/deepinfer-sm100-cu132)
  and run its installer, then unpack the JIT kernel cache for your GPU count
  (optional: without it the kernels compile on first use). About 2 to 3 min
  on a B200 pod:

  ```bash
  apt-get install -y zstd libdw-dev
  python3 -m venv /workspace/hf-venv && /workspace/hf-venv/bin/pip install "huggingface_hub[hf_transfer]"
  HF_HUB_ENABLE_HF_TRANSFER=1 /workspace/hf-venv/bin/python -c \
    'from huggingface_hub import snapshot_download as d; d("inferencekey/deepinfer-sm100-cu132", local_dir="/workspace/deepinfer-prebuilt")'
  bash /workspace/deepinfer-prebuilt/install.sh /workspace/di-venv /workspace
  zstd -dc /workspace/deepinfer-prebuilt/jit-cache/jit-cache-4xb200-ep4.tar.zst | tar -x -C /workspace
  ```

  Point DeepInfer's JIT caches at the unpacked directories in the workload's
  `env` (`DG_JIT_CACHE_DIR=/workspace/jit-cache/deep_gemm`,
  `EP_JIT_CACHE_DIR=/workspace/jit-cache/deep_ep`,
  `TILELANG_CACHE_DIR=/workspace/jit-cache/tilelang`,
  `TRITON_CACHE_DIR=/workspace/jit-cache/triton`).
- **From source:** DeepInfer's `build.sh` compiles in place (no installable
  wheel) and installs torch, its vendored CUDA kernels and transformers into
  the active Python; build it into its own venv with a `.pth` file pointing at
  the checkout, after applying the patches of the prebuilt package
  (`patches/`, see "Prefix cache").

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
  first user message. A new session goes to the rank whose active sessions
  (seen in the last `session_idle_seconds`) add up to the fewest tokens.
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

## L1 host cache

HBM does not hold the KV of hundreds of long agent sessions (400 sessions of
340K tokens need about 120 GB of KV, a 4x B200 node has about 100M tokens of
KV pool). When the GPU cache (L0) has to evict an idle session, the L1 host
cache keeps a copy of its blocks in pinned host RAM, and the session's next
turn copies them back (measured on B200: about 17 GB/s host to GPU and 47 GB/s
GPU to host, so about 60 ms for a 1M-token session) instead of recomputing them (seconds of prefill that stall every rank). It is off by
default; with `l1_cache: true`:

- All paged KV of a rank is one GPU tensor (`cache_engine.gpu_blocks`, one row
  per block, about 111 KiB per block for the full model), so a block moves as
  one row. Nothing else is saved: bounded replay recomputes the last cached
  block with a zeroed state slot, as for an L0 hit.
- Demotion: the blocks L0 evicts are copied to host slots on a side stream,
  and stay pinned on the GPU until the copy completed (events polled between
  engine steps). A block that is already in L1 is just unpinned (L1 is
  inclusive: a promoted block keeps its host copy).
- Promotion: when a prompt's L0 hit continues in L1, the rank allocates fresh
  GPU blocks, copies the host rows into them, puts the restored chain back
  into L0, and only then gives the request to the engine, reusing the whole
  hit. `usage.prompt_tokens_details.cached_tokens` counts L0 and L1 tokens.
  Continuations shorter than `l1_min_promote_blocks` are recomputed; a
  promotion that waits more than `l1_promote_wait_seconds` for free GPU
  blocks falls back to recompute.
- Failures never break the engine step: a copy that cannot be issued falls
  back to plain L0 behaviour (drop the block, or recompute); a copy that failed
  halfway or got stuck keeps its GPU blocks and host slots out of use until its
  event completed. `l1_verify_fraction` adds checksums of the bytes as the
  prefill wrote them, as they were demoted and as they were restored, each
  compared only with a sum of the same physical block (a recompute of the
  same prefix may give other bytes); a mismatch turns L1 off, and nothing
  more is restored from it.
- The host budget is computed once, after every rank is ready: the container's
  memory limit minus what cannot be reclaimed (`anon + shmem + unevictable`),
  a reserve of `max(32 GiB, 5 %)` and the Engram host storage, split over the
  ranks and capped at `l1_host_gb`. Each rank pins its share at startup and
  never pins during serving; the backend logs the budget line and the pinned
  size per rank.
- L1 turns itself off (logged, the backend serves with L0 only) when the engine
  lacks what it needs: `cp` above 1, a block manager without `allocate_nolock`,
  no reachable KV tensor, no room in the budget, or a failed pin.

The rank line logged every `log_rank_stats_seconds` gains an L1 part (blocks
held, hits and hit tokens, promoted and demoted blocks, copy latency p50/p99
and GB/s, drops, timeouts, fallbacks), and the rank stats carry every `l1_*`
counter.

Checked on GPU: on one B200 with the trimmed model and `l1_verify_fraction:
1.0`, a byte-exact demote, overwrite and promote of a 32K-token prompt's
blocks (7 block-indexed tensors, 0 mismatches) and the same greedy output as
an L0 hit; on 4x B200 with the full model, hundreds of 450K-token sessions
moving between GPU and host with no failed copy and no verify failure. The
mover warms its kernels after pinning (CUDA loads them lazily, which cost the
first promotion about 50 ms).

What L1 does and does not buy: it keeps idle sessions warm in host RAM, so they
come back without a prefill. It does not raise how many sessions can be active
at once: the blocks of every sequence the engine is running must be on the
GPU. With agents that pause only seconds between turns nearly every session is
active, and above the GPU pool (about 120 sessions of 450K tokens on 4x B200)
promotions wait for free blocks, time out and fall back to recompute. L1 pays
off with human-paced sessions (minutes between turns) and many more sessions
than the GPU holds.

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
| `l1_cache` | `false` | L1 host cache (see "L1 host cache"). Needs `prefix_cache` and `cp` 1; turns itself off when the engine lacks what it needs. |
| `l1_host_gb` | `64` | Pinned host memory per rank, GiB, capped at startup by the node budget. |
| `l1_prealloc_gb` | `l1_host_gb` | Pinned at startup (nothing is pinned while serving), so a smaller value only shrinks the pool. |
| `l1_engram_host_gb` | from the model config | Engram host storage of the node to keep out of the budget (tables plus buffer storage). `0` when it is already counted in the process memory. |
| `l1_segment_mb` | `1024` | Size of one registered host segment. |
| `l1_staging_mb` | `64` | GPU staging buffer per copy direction. |
| `l1_index_ring` | `65536` | Block indices in flight per copy direction (pinned ring). |
| `l1_min_promote_blocks` | `2` | Shorter L1 continuations are recomputed instead. |
| `l1_promote_wait_seconds` | `2.0` | Longest wait for free GPU blocks before a promotion falls back to recompute. |
| `l1_copy_timeout_seconds` | `10.0` | A copy not done by then is logged as stuck and its memory kept out of use; three stuck copies turn L1 off. |
| `l1_verify_fraction` | `0.0` | Fraction of L0 inserts, demotions and promotions that carry checksums (1.0 for the GPU check). |
| `session_migrate_inflight` | `4 * max_num_seqs` | A session moves to another rank (one cold turn) only when its rank has this many requests in flight. |
| `session_table_size` | `100000` | Sessions remembered by the router (LRU). |
| `session_idle_seconds` | `1800` | A session idle this long no longer counts when new sessions are balanced across ranks. |
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

Tests: `bindings/python/tests/test_example_deepinfer.py`, `test_example_deepinfer_cache.py` (prefix
cache) and `test_example_deepinfer_l1.py` (L1 host cache).

The simulated ranks of `bench/sim_server.py` model L1 too: `--l1-pcie-gbps 50`
gives each rank a simulated mover and turns `l1_cache` on (copies take
`--l1-copy-latency-us` plus the bytes over that bandwidth, and admission waits
for them), `--l1-host-blocks`
sets its host slots, and `--kv-blocks` smaller than the sessions forces
eviction. Content tags per block catch an early unpin or admission
(`kv_mismatch` in the rank stats file).

Throughput of the parent process (router, result queue, detokenization, SSE)
with simulated ranks, no GPU: `bench/README.md`.

## Tested on 4x B200 (2026-10-24 and 25)

RunPod, 4x B200 (183 GB each), driver 595.91, 2 to 3 TB host RAM, 900 GB
container disk, no pod volume. DeepInfer from the prebuilt package above,
weights on local disk. Setup (weights and DeepInfer in parallel) 4 to 11 min,
load about 4 min.

Workload config of the runs:

```json
{
  "engine": "deepinfer",
  "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "DEEPINFER_HF_MODEL_DIR": "/workspace/v41flash",
          "IK_BACKEND_HTTP": "native"},
  "ep": 4,
  "deepinfer_venv": "/workspace/di-venv",
  "eplb": false, "extra_experts": 0,
  "max_model_len": 1064960, "max_num_seqs": 128, "max_num_batched_tokens": 16384,
  "gpu_memory_utilization": 0.90, "load_timeout_seconds": 3600,
  "l1_cache": true, "l1_host_gb": 128,
  "deepinfer_env": {"DEEPINFER_ENGRAM_LOCAL": "1", "EP_DISABLE_GIN": "1", "NCCL_IB_DISABLE": "1",
                    "MASTER_PORT": "29511", "DEEPINFER_NCCL_NVLS_ENABLE": "0", "NCCL_NVLS_ENABLE": "0"},
  "ready_timeout_secs": 5400,
  "request_timeout_secs": 1800
}
```

With it each rank has 111,456 KV blocks (14.3M tokens, 57M on the node) and
128 GiB of L1 (154M tokens per rank, 617M on the node).

Quality with the cache: the same long prompt (16K to 200K tokens, one fact
hidden at 10 to 90 % depth) sent cold and then again: both answers find the
fact and are identical; the repeat is served from the cache (98 to 99 % of the
prompt) in 0.8 to 1.5 s instead of 1.3 to 20.6 s.

Capacity, simulated coding agents with 3 s between turns (turn sizes from 30
days of real Claude Code calls), one measured turn per user. Pass = time to
first token p95 at most 15 s, per-user tok/s p50 at least 20 and p10 at least
10, cache hit at least 90 %, errors under 1 %:

| Context | Users | TTFT p50 / p95 | tok/s p50 / p10 | Cache | Errors | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 128K | 300 | 5.6 / 14.8 s | 35 / 16 | 98 % | 0/521 | pass |
| 128K | 350 | 7.4 / 21.0 s | 22 / 8.4 | 98 % | 0/676 | fail |
| 128K | 400 | 7.3 / 16.9 s | 29 / 9.5 | 98 % | 0/652 | fail (just) |
| 500K | 50 | 2.5 / 4.4 s | 163 / 100 | 100 % | 0/75 | pass |
| 500K | 100 | 2.6 / 37 s | 135 / 39 | 99 % | 0/237 | fail (p95 only) |
| 500K | 200 | 151 / 161 s | 26 / 9 | 41 % | 0/17 | fail (active sessions exceed the GPU pool) |
| 1M | 48 | 3.6 / 28.9 s | 69 / 22 | 100 % | 0/100 | fail (p95 only) |

Through InferenceKey (manager, NATS and a tunnel to the pod) 128K at 300 users
gave the same median TTFT, speed, cache and errors, with a p95 of 22 s. A
Claude Code session against the endpoint finished a small coding task (14 tool
calls, tests passing) in 39 s.

Confirmed on hardware: the prebuilt venv imports and loads with `ep 4`; the
long-lived lockstep loop (more than an hour, idle ranks included); adding requests between
steps; `flush_inflight_reqs()`; per-token outputs with DSpark; the tokenizer
markup, DSML tool calls and Claude Code; aborts while other requests run; the
prefix cache (with DeepInfer's zero-overhead scheduling a finished sequence is
finalized one step after its last token, and the rank loop keeps its entry
until then); 400 streams on one backend process.

Still unverified: 8 GPUs (`ep 8`); B300; the `finish_reason` of requests the
scheduler ignores or stops (`prompt_length_exceeded` and the others: send a
prompt over `max_model_len` and check for the 400); the CPU of the backend process at 400 streams and the cost of
`RequestOutput.clone` on the real engine.

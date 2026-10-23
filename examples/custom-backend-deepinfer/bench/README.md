# Parent-process benchmark (no GPU, no DeepInfer)

On 4x B200 (2026-10-23, DeepSeek-V4.1-Flash, EP4/DP4, `max_num_seqs` 128) the
backend's parent process sat at about 109 % CPU with 200 to 400 concurrent
streams while the GPUs idled, and clients saw 1 to 2 tok/s each. This harness
reproduces that on a laptop: it runs the backend's real parent-side code with
simulated rank processes and drives it with N concurrent streaming clients.

## What is real and what is simulated

Real (production code, unmodified):

- `DeepInferBackend` (`chat_stream`, the stream splitter, the DSv4.1 prompt encoder),
- `DeepInferEngine`: the spawn rank processes, the least-loaded router, the
  per-rank request queues, the single result queue, the reader thread, the
  per-request mailboxes and the chunk pacing (`stream_interval_ms`,
  `stream_chunks_per_second`),
- the SDK runtime's HTTP server (`inferencekey.backend.serve`), one thread per
  stream,
- even the rank loop itself (`_rank_main` and `_rank_loop`): prompt
  tokenization on its thread pool, incremental detokenization, one message
  per step.

Simulated: only what `_rank_main` imports. `fake_modules/` holds a stand-in
`torch` (empty) and `deepinfer` (`LLM`, `RequestOutput`, `SamplingParams`).
The simulated engine (`fake_modules/deepinfer/entrypoints/llm.py`) admits up
to `max_num_seqs` requests, prefills at `--prefill-rate` tok/s per rank within
`max_num_batched_tokens` per step, emits one `RequestOutput` per token per
running request per step at `--step-rate` steps/s, and keeps the ranks in
lockstep with a barrier (an OR across ranks in `has_unfinished_requests()` and
a barrier at the end of each `step()`), like DeepInfer's all-reduce.

There is no hook in the production code. `sim_server.py` appends `fake_modules`
to the END of `sys.path` just before the ranks are spawned (they inherit it),
so the fakes are only found where no real torch or DeepInfer is installed; the
simulation knobs travel in the existing `llm_kwargs` config key, which the rank
loop already passes to `LLM(...)`.

Tokenizer: the real DeepSeek-V4.1 `tokenizer.json` through transformers 4.50
(the version DeepInfer pins) with `--tokenizer-dir`. Without one,
`--fake-tokenizer` uses `fake_modules/bench_fake_tokenizer.py`, which spins
for the measured cost of the real one (4.4 us per token to decode, 5.6 us per
token to encode, GIL held).

## Setup

The venv must live OUTSIDE `examples/custom-backend-deepinfer/`: the package
step ships that whole directory. (This `bench/` folder ships too; it is inert
there, `fake_modules` is never on `sys.path` in production.)

```bash
uv venv -p 3.12 ~/.venvs/deepinfer-bench
uv pip install -p ~/.venvs/deepinfer-bench/bin/python "transformers==4.50.0" tokenizers numpy \
  -e bindings/python
```

A tokenizer directory needs `tokenizer.json` and `tokenizer_config.json` of
`deepseek-ai/DeepSeek-V4.1-Flash-1017` (about 6 MB, no weights).

## Run

```bash
B=examples/custom-backend-deepinfer/bench
PY=~/.venvs/deepinfer-bench/bin/python
export BENCH_TOKENIZER_DIR=/path/to/tokenizer-dir

$PY $B/load.py --concurrency 50,200,400                       # burst, like production
$PY $B/load.py --concurrency 200,400 --ramp-s 20              # starts spread over 20 s
$PY $B/load.py --concurrency 50 --prompt-tokens 137000        # long prompts
$PY $B/load.py --fake-tokenizer --concurrency 50,200          # no tokenizer files
$PY $B/load.py --backend null --concurrency 50,200,400,800     # the runtime alone
$PY $B/load.py --backend null --transport native ...           # refused unless the backend serves native
```

`--backend null` serves `null_backend.py` through the real
`python -m inferencekey.backend.serve` CLI: a backend whose `chat_stream`
only yields `--max-tokens` one-token chunks at `--tok-s` (50) tokens per
second, with no engine, no tokenizer and no multiprocessing, so whatever limits
it is the runtime's HTTP/SSE layer (`--chunk-format str` makes it yield
pre-serialized chunks). `--transport stdlib|native` sets `IK_BACKEND_HTTP` for
the backend, and `load.py` refuses to report a level unless the backend's
`http transport: ...` startup line names that transport (a runtime that prints
no such line predates the native transport and counts as `stdlib`).

Each level starts a fresh `sim_server.py`, runs N streams of `--max-tokens`
(256) from `--client-procs` (4) client processes, prints one JSON line and, with
`--out FILE`, appends it there. Other knobs: `--step-rate` (70), `--tokens-per-step`
(1), `--max-num-seqs` (128), `--prefill-rate` (88000, 0 makes prefill free),
`--dp` (4), `--no-lockstep`, and `--backend-config JSON` to override backend
config keys (for example `'{"stream_interval_ms": 0, "stream_chunks_per_second": 0}'`).

Fields of a result line:

| Field | Meaning |
| --- | --- |
| `total_tok_s` | completion tokens of the successful streams / wall time |
| `steady_tok_s` | median of the per-second token counts in the middle 80 % of the run (a chunk may carry several tokens; each stream's tokens are spread over its chunks by their length, successful streams only). The baseline harness counted one token per chunk, which was true there, so steady rates before and after are only roughly comparable; compare on `total_tok_s`, the same formula in both. Runs shorter than a few seconds have no steady rate (0). |
| `per_stream_tok_s` | completion tokens / (last token - first token), p10, median, p90 |
| `ttft_s` | time to first content token |
| `added_ttft_ms` | null backend only: the TTFT the runtime adds over the backend's own first-yield latency, from wall-clock stamps the backend puts in its first token (`bench_t`): TTFT minus (first yield - first resume of the generator). `server_before_backend_p99` is the server's part (TCP connect done to generator first resumed), `client_connect_p99` the client's own (its event loop opening hundreds of connections at once), `backend_own_p99` the backend's first-yield latency, `delivery_*` first yield to client read |
| `cpu_pct_mean` | mean CPU % between the first and the last token: `parent` (the backend process, 100 = one core), each rank, the clients |
| `ranks_busy_tok_s`, `ranks_steps_per_s` | what the simulated ranks produced while busy, from their own counters (1 s resolution) |
| `ranks_done_s`, `clients_done_s` | when the ranks produced their last token and when the clients received it; the gap is backlog queued in the parent |
| `sim_ceiling_tok_s` | what the ranks allow: min(N, dp x max_num_seqs) x step_rate x tokens_per_step |
| `errors`, `error_samples` | failed streams (the baseline's connection resets came from the listen backlog of 5) |
| `error_kinds` | failed streams by kind: `reset`, `refused`, `timeout`, `eof` (closed before `[DONE]`), `status` (not 200), `error_event` |
| `done_pct` | streams that ended with `[DONE]` and no error event |
| `offered_tok_s`, `delivered_pct` | what the backend can produce with every stream open at once (null: N x `--tok-s`; DeepInfer: the sim ceiling) and `steady_tok_s` as a share of it. With `--ramp-s` the streams are not all open at once, so the share is arrival-bound there. |
| `per_stream_tok_s.p5`, `ttft_s.p99`, `gap_ms` | slowest streams, slowest first tokens, and the gap between consecutive content chunks of a stream (median, p99, max) |
| `healthz` | the worker-like side prober: GET `/healthz` every 500 ms with a 3 s timeout, like the worker; its `misses` (no answer in 3 s, a connection error, or a status other than 200 and 503) are what the worker would count. A burst level lasts only 5 to 7 s, so its 11 to 15 samples make p99 the maximum. |
| `healthz_dense` | a second prober, every 50 ms with a 3 s timeout, whose 100+ samples per level give a real p50 and p99 (the gate-1 latency). In both, percentiles and `max_ms` count a miss at 3 s (a lower bound of its cost), and `max_answered_ms` is the slowest answered probe |
| `busiest_thread` | the parent thread with the most CPU over the run, % of one core (from `/proc/<pid>/task`, what `pidstat -t` reads) |
| `python_user_cores` | user-mode CPU of the parent's Python threads, in cores: an upper bound of the GIL-held fraction |
| `gil_exact` | the exact fraction of time some parent thread held the GIL, between the first and the last token (`held_fraction`) and over the whole level: `gil_sampler.py` reads the GIL's `locked` flag from the parent's memory (`/proc/<pid>/mem`) about 1,700 times a second without stopping it. This is the gate's GIL-held fraction. |
| `gil` | `py-spy record --gil --nonblocking` on the parent during the run: `gil_held_fraction` (samples / rate x seconds; without pausing the process py-spy reads the GIL's last holder, so threads that just released it around a socket write or a sleep still count), `serving_leaf_share` (samples whose innermost frame is in `socketserver`, `http.server`, `serve.py`, `json` or `socket`), `serving_cores` (those samples / rate x seconds) and the top innermost frames. On the native transport the nonblocking sampler sees almost none of the GIL holders (0.02 to 0.06 against an exact 0.45 to 0.6), so its attribution there is a handful of samples: `--py-spy-blocking --py-spy-rate 20` pauses the parent at every sample for exact attribution, at a cost in throughput (read only the attribution of such a run). `--py-spy ''` turns it off. |
| `serving_share_of_parent_cpu` | `gil.serving_cores` / parent cores: gate 2's serving-layer share of parent CPU (meaningful with `--py-spy-blocking`) |
| `rss_mb_max` | peak RSS of the parent |
| `frames_per_chunk`, `debug_stats` | native transport only: `/debug/stats` frames written per chunk pushed (coalescing), and the flusher's `flush_batches` / `flush_wakes` |
| `server_cpus`, `client_cpus`, `env_knobs` | `--server-cpus 0-7` pins the backend (and its ranks) to those CPUs, `--client-cpus 8-11` the clients; `env_knobs` records any `IK_BACKEND_*` / `IK_SSE_*` setting passed through to the backend |

## Baseline (2026-10-23, before the fix)

Shared dev VM, 12 vCPU (AMD Ryzen 7 5700X), other jobs running (load average
2.5 to 8), real DeepSeek-V4.1 tokenizer, defaults (dp 4, 70 steps/s, 1 token per
step, `max_num_seqs` 128, lockstep, prefill 88K tok/s per rank), 256 new tokens
per stream unless noted. Raw lines: `results/baseline-2026-10-23.jsonl`.

| Run | N | OK | Total tok/s | Per stream tok/s (p10 / median / p90) | TTFT median / max (s) | Parent CPU % mean / max | Rank CPU % | Ranks done / clients done (s) | Sim ceiling tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burst | 1 | 1 | 70 | 70 | 0.02 | 5 / 8 | 3 | 2.8 / 3.7 | 70 |
| burst | 8 | 8 | 557 | 70 / 70 / 70 | 0.02 | 31 / 48 | 3 | 3.7 / 3.7 | 560 |
| burst | 50 | 50 | 1,695 | 54 / 55 / 66 | 1.3 / 3.8 | 90 / 112 | 3 | 6.7 / 7.6 | 3,500 |
| burst | 200 | 143 (57 reset) | 341 (steady 1,435) | 27 / 35 / 77 | 3.0 / 55 | 34 / 114 | 3 | 58.9 / 58.4 | 14,000 |
| burst | 400 | 134 (266 reset) | 320 (steady 947) | 24 / 26 / 76 | 1.2 / 54 | 33 / 116 | 3 | 58.0 / 57.5 | 28,000 |
| ramp 20 s | 50 | 50 | 550 (arrival-bound) | 70 / 70 / 70 | 0.03 | 32 / 40 | 3 | 22.8 / 23.3 | 3,500 |
| ramp 20 s | 200 | 200 | 1,892 | 52 / 57 / 67 | 1.8 / 4.7 | **101** / 112 | 3 | 24.1 / 27.1 | 14,000 |
| ramp 20 s | 400 | 400 | 1,923 | 25 / 28 / 37 | 11.4 / 28.0 | **107** / 112 | 3 | 23.8 / **53.3** | 28,000 |
| 2,048 new tokens, burst | 50 | 50 | **320** | **6.4 / 6.6 / 7.2** | 1.1 / 3.2 | **102** / 110 | 4 | **32.1 / 320.2** | 3,500 |
| 137K-token prompts, ramp 20 s | 50 | 50 | 339 | 9 / 16 / 34 | 11.8 / 12.3 | 20 / 110 | 11 to 12 | 36.0 / 37.8 | 3,500 |
| same, prefill free (`--prefill-rate 0`) | 50 | 50 | 542 | 17 / 23 / 54 | 0.4 / 0.8 | 30 / 110 | 17 to 18 | 22.9 / 23.6 | 3,500 |
| fake tokenizer, ramp 20 s | 200 | 200 | 1,345 | 36 / 40 / 50 | 5.5 / 14.9 | 102 / 110 | 3 | 23.8 / 38.1 | 14,000 |

What it shows:

1. **The parent is CPU-bound.** From 50 streams on, the backend process sits at
   100 to 116 % CPU (one core, the GIL) while each simulated rank uses 3 %.
   Delivered throughput flattens at about 1,900 tok/s with 256-token answers,
   7 to 15 times below what the ranks produce (14,000 and 28,000 tok/s ceilings;
   the ranks keep stepping at 68 to 69 steps/s and finish long before the
   clients: at N=400 the ranks are done at 24 s, the clients at 53 s).
2. **It gets worse with longer answers.** The reader thread decodes the whole
   completion on every token (about 4.4 us per token of completion, so the cost
   per token grows with the length): with 2,048-token answers, 50 streams get
   6.6 tok/s each and 320 tok/s in total, while the ranks finished their work in
   32 s and the clients needed 320 s. That is the shape seen on the B200s
   (about 480 tok/s in total, 1 to 2 tok/s per user, reasoning answers are
   thousands of tokens long).
3. **Burst connections are dropped.** With all streams opening at once, 57 of
   200 and 266 of 400 get `Connection reset by peer`: the runtime's
   `ThreadingHTTPServer` listens with a backlog of 5 and its accept loop
   competes for the GIL (production lost about 300 of 400).
4. **Long prompts stall the ranks, not the parent.** The prompt is tokenized
   inside the rank loop (`_rank_main`), not in the parent, and a 137K-token
   prompt takes about 1.3 s there. Because the ranks run in lockstep, every
   rank waits: even with free prefill, the ranks drop from 70 to 24 steps/s
   (about 23 tok/s per stream). With prefill at 88K tok/s per rank they drop to
   10 steps/s. This matches the B200 picture of ranks with running=50 and about
   25 completion tok/s each while the GPUs idle.

Not modelled: DeepInfer's own Python scheduler time in the ranks, prefix
caching, and DSpark accepting several draft tokens per step
(`--tokens-per-step` can approximate the last one). The numbers come from a
shared VM and move by about 10 % between runs.

## After the fix (2026-10-23)

The fix (commits after the baseline): the ranks tokenize prompts on a thread
pool, detokenize incrementally (a token-to-bytes table for the DeepSeek-V4.1
tokenizer) and send one message per engine step; the backend process hands
each item to its request and paces the SSE chunks (`stream_interval_ms` 50,
at most `stream_chunks_per_second` 8000 in total); the runtime listens with a
backlog of 1024. Same VM, same settings, same day (load average 3 to 5). Raw
lines: `results/after-2026-10-23.jsonl`. The parent CPU column adds its kernel
share in parentheses (socket writes and thread wake-ups).

| Run | N | OK | Total tok/s | Steady tok/s | Per stream tok/s (p10 / median / p90) | TTFT median / max (s) | Parent CPU % mean / max (kernel) | Rank CPU % | Ranks done / clients done (s) | Ranks steps/s | Sim ceiling tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burst | 1 | 1 | 70 | 71 | 70 / 70 / 70 | 0.03 / 0.03 | 1 / 2 (0) | 2 to 3 | 3.7 / 3.7 | 63.1 | 70 |
| burst | 8 | 8 | 554 | 563 | 70 / 70 / 70 | 0.05 / 0.05 | 3 / 4 (1) | 3 to 3 | 3.6 / 3.7 | 63.0 | 560 |
| burst | 50 | 50 | 3,438 | 3,515 | 70 / 70 / 70 | 0.06 / 0.07 | 14 / 18 (5) | 3 to 4 | 2.9 / 3.7 | 66.9 | 3,500 |
| burst | 200 | 200 | 13,395 | 14,038 | 70 / 70 / 70 | 0.11 / 0.17 | 49 / 66 (20) | 5 to 5 | 3.8 / 3.8 | 63.1 | 14,000 |
| burst | 400 | 400 | 24,914 | 28,034 | 69 / 70 / 70 | 0.19 / 0.45 | 102 / 120 (46) | 6 to 6 | 3.8 / 4.1 | 62.8 | 28,000 |
| ramp 20 s | 50 | 50 | 550 | 638 | 70 / 70 / 70 | 0.04 / 0.04 | 3 / 6 (1) | 3 to 3 | 22.8 / 23.3 | 69.5 | 3,500 |
| ramp 20 s | 200 | 200 | 2,171 | 2,556 | 70 / 70 / 70 | 0.04 / 0.05 | 10 / 14 (4) | 4 to 4 | 23.0 / 23.6 | 69.6 | 14,000 |
| ramp 20 s | 400 | 400 | 4,332 | 5,116 | 70 / 70 / 70 | 0.04 / 0.05 | 18 / 24 (7) | 4 to 4 | 22.9 / 23.6 | 69.1 | 28,000 |
| 2,048 new tokens, burst | 50 | 50 | 3,491 | 3,512 | 70 / 70 / 70 | 0.07 / 0.08 | 13 / 18 (5) | 4 to 4 | 29.0 / 29.3 | 69.1 | 3,500 |
| 2,048 new tokens, burst | 400 | 400 | 27,560 | 28,000 | 70 / 70 / 70 | 0.19 / 0.46 | 103 / 122 (46) | 6 to 6 | 29.0 / 29.7 | 69.1 | 28,000 |
| 137K-token prompts, ramp 20 s | 50 | 50 | 493 | 66 | 12 / 20 / 50 | 2.69 / 2.96 | 4 / 14 (2) | 12 to 13 | 25.6 / 26.0 | 12.2 | 3,500 |
| same, prefill free (`--prefill-rate 0`) | 50 | 50 | 543 | 642 | 70 / 70 / 70 | 0.35 / 0.43 | 4 / 8 (1) | 14 to 20 | 22.8 / 23.6 | 69.2 | 3,500 |
| fake tokenizer, ramp 20 s | 200 | 200 | 2,171 | 2,557 | 70 / 70 / 70 | 0.04 / 0.04 | 10 / 14 (4) | 4 to 4 | 23.1 / 23.6 | 69.8 | 14,000 |
| fake tokenizer, 1,024 new tokens, burst | 400 | 400 | 25,895 | 26,527 | 66 / 66 / 67 | 0.21 / 0.45 | 103 / 118 (46) | 17 to 17 | 15.1 / 15.8 | 65.4 | 28,000 |
| 3 tokens per step (DSpark-like), 1,024 new tokens, burst | 400 | 400 | 74,502 | 83,966 | 208 / 209 / 210 | 0.19 / 0.61 | 106 / 124 (47) | 8 to 8 | 4.8 / 5.5 | 63.8 | 84,000 |
| 1,024 new tokens, burst, pacing off (`stream_interval_ms` 0, `stream_chunks_per_second` 0) | 400 | 400 | 11,195 | 11,144 | 30 / 30 / 32 | 0.52 / 6.87 | 157 / 176 (84) | 5 to 5 | 18.9 / 36.6 | 67.4 | 28,000 |
| 1,024 new tokens, burst (full batch: 4 x 128) | 512 | 512 | 33,819 | 35,862 | 70 / 70 / 70 | 0.25 / 0.87 | 108 / 132 (48) | 7 to 7 | 14.8 / 15.5 | 68.1 | 35,840 |
| 1,024 new tokens, burst, `--dp 8` | 1024 | 1024 | 53,304 | 57,906 | 59 / 60 / 68 | 1.10 / 5.57 | 138 / 182 (62) | 7 to 7 | 19.6 / 19.7 | 68.2 | 71,680 |

Before and after, same scenario on both sides (`total_tok_s`):

| Run | N | Total tok/s before | after | Per stream (median) before | after | What limits the after run |
| --- | --- | --- | --- | --- | --- | --- |
| burst | 200 | 341 (57 reset) | 13,395 | 35 | 70 | the simulator (ceiling 14,000) |
| burst | 400 | 320 (266 reset) | 24,914 | 26 | 70 | the simulator (steady 28,034 of 28,000) |
| ramp 20 s | 200 | 1,892 | 2,171 | 57 | 70 | arrivals (200 x 256 tokens in about 23 s) |
| ramp 20 s | 400 | 1,923 | 4,332 | 28 | 70 | arrivals; TTFT median 11.4 s to 0.04 s |
| 2,048 new tokens, burst | 50 | 320 | 3,491 | 6.6 | 70 | the simulator; clients done at 29 s instead of 320 s |
| 137K-token prompts, prefill free | 50 | 542 | 543 | 23 | 70 | arrivals; ranks back at 69 steps/s (was 24) |

None of these ratios measures the parent. In the burst rows most of the gain
is the connections the baseline lost (its listen backlog of 5), the ramp rows
are bound by how fast streams arrive, and at the default 70 steps/s every
after-run lands on the simulator's ceiling, so the default harness measures the
simulated ranks. The capacity runs below raise the step rate to find the
parent's limit instead.

### Capacity (2026-10-23)

The baseline's capacity is its saturated rate: with the step rate raised it
still delivers about 2,100 tok/s with the parent at 103 to 108 % CPU (ramp
20 s, 256 new tokens; the burst runs lose connections). After the fix the same
step rates are run as bursts, with 1,024 new tokens so that the steady state
lasts several seconds. Raw lines: `results/baseline-capacity-2026-10-23.jsonl`
and `results/review-fixes-2026-10-23.jsonl` (these runs also include the
review fixes: stream keepalives, failure handling, single BOS).

| Run | N | Step rate | Total tok/s | Steady tok/s | Per stream tok/s (p10 / median / p90) | Parent CPU % mean / max (kernel) | Clients CPU % | Ranks done / clients done (s) | Sim ceiling tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before, ramp 20 s, 256 tokens | 200 | 200 | 2,100 | 2,117 | 153 / 165 / 189 | 103 / 110 | 17 | 22.0 / 24.4 | 40,000 |
| before, ramp 20 s, 256 tokens | 400 | 140 | 2,060 | 2,087 | 53 / 58 / 67 | 108 / 112 | 18 | 22.7 / **49.7** | 56,000 |
| after, burst, 1,024 tokens | 200 | 200 | 38,429 | 39,959 | 199 / 199 / 200 | 57 / 74 (21) | 34 | 4.8 / 5.3 | 40,000 |
| after, burst, 1,024 tokens | 400 | 140 | 50,046 | 56,025 | 139 / 139 / 140 | 117 / 132 (51) | 64 | 7.9 / 8.2 | 56,000 |
| after, burst, 256 tokens | 400 | 140 | 42,636 | 54,682 | 133 / 135 / 139 | 116 / 124 (49) | 60 | 1.8 / 2.4 | 56,000 |

Against the baseline's about 2,060 to 2,100 tok/s that is at least 18x at
200 streams (the after run is still on the simulator's ceiling, with the parent
at 57 %) and 24x at 400 streams (27x on the steady rate, which also sits on the
ceiling). The 400-stream runs are close to the parent's limit, see point 7.

1. **The parent is no longer the limiter at the default rates.** Every
   short-prompt run delivers what the simulated ranks produce: the steady rate
   equals the ceiling, every stream gets the full 70 tok/s, and the clients
   finish when the ranks do (within about a second; the rank counters tick once
   a second). The capacity runs above put the parent's own limit at about
   50,000 to 56,000 tok/s with 400 streams on this VM.
2. **Answer length no longer matters.** Detokenization is incremental in the
   ranks (about 2 us per request and step), so 2,048-token answers stream at
   70 tok/s like 256-token ones.
3. **The cost in the parent is per chunk, not per token.** With 3 tokens per
   step (a DSpark-like run) the same parent CPU carries 74,500 tok/s (ceiling
   84,000). Without pacing (`stream_interval_ms` 0, `stream_chunks_per_second`
   0, one chunk per stream per step) the parent saturates again at 11,200 tok/s
   with 400 streams: the pacing is what keeps it cheap.
4. **Headroom.** The full batch of 4 ranks (512 streams) delivers 33,800 tok/s
   (steady 35,862 of 35,840) with the parent at 108 % CPU, half of it kernel
   time. 8 ranks with 1,024 streams deliver 53,300 tok/s (steady 57,900 of
   71,680) on this shared 12-vCPU VM, which then also runs the 8 ranks and the
   clients; the chunk budget stretches the interval to 128 ms there (without it
   that run collapsed to 11,400 tok/s).
5. **Long prompts no longer stall the ranks.** Tokenization runs on the
   rank's thread pool and the fast tokenizer releases the GIL: with free
   prefill the ranks keep 69 steps/s (24 before) and every stream gets
   70 tok/s. With prefill at 88K tok/s per rank the simulated prefill itself
   bounds the run (12 steps/s), which is the engine, not the backend.
6. The ranks now do more work (detokenization): 4 to 8 % CPU each instead of
   3 %, 17 % with the fake tokenizer, which has no byte table and uses the
   generic window path at the measured cost of the real `decode`.
7. **400 streams is close to the limit of one parent process.** At 400
   streams the parent stays at about 103 to 117 % CPU on average (120 to 132 %
   peak, about half of it kernel time: socket writes and thread wake-ups) both
   at 28,000 and at 56,000 tok/s. The cost is per chunk and per stream, not per
   token: with 50 ms pacing that is about 8,000 chunk writes per second. It does
   not limit the simulator, but the margin is small, and production adds the
   `/v1/messages` adapter and proxy hops, which are not modelled. Above about
   400 streams the chunk budget stretches the interval, so per-stream latency
   rises. Check the parent's CPU on the B200 host with 200 to 400 streams; if it
   is near 100 %, raise `stream_interval_ms`, lower
   `stream_chunks_per_second`, or split the HTTP layer across processes.
   Keepalives (`stream_heartbeat_seconds`) only fire on streams that have had
   nothing to send for a second, so they add nothing while tokens flow.

## Serving layer baseline (2026-10-23, stdlib transport)

The baseline for the native HTTP/SSE transport (IK-537), taken with the
runtime's standard library server (`ThreadingHTTPServer`, backlog 1024, one
thread per connection) after the DeepInfer fix above. Same shared 12-vCPU VM
(load average 3 to 4.5), 256 new tokens per stream, one run per row. Raw lines:
`results/serving-baseline-2026-10-23.jsonl`. Parent CPU is the mean between
the first and the last token, kernel share in parentheses; healthz is
p50 / p99 / max in ms and misses / probes.

### Bench A: null backend (the runtime alone), 50 tok/s per stream

| Run | N | OK | Offered tok/s | Total tok/s | Steady tok/s (delivered) | Per stream tok/s p5 / median | TTFT median / p99 (s) | Gap p99 (ms) | Healthz p50 / p99 / max (misses) | Parent CPU % (kernel) | Busiest thread % | Python user cores | py-spy GIL / serving share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burst | 50 | 50 | 2,500 | 2,494 | 2,500 (100 %) | 50.2 / 50.2 | 0.012 / 0.024 | 20.7 | 0.9 / 1.9 / 1.9 (0 of 11) | 18 (12) | 0.4 | 0.06 | 0.10 / 0.33 |
| burst | 200 | 200 | 10,000 | 9,753 | 10,000 (100 %) | 50.2 / 50.2 | 0.056 / 0.134 | 21.5 | 1.3 / 4.6 / 4.6 (0 of 11) | 78 (43) | 0.4 | 0.35 | 0.29 / 0.18 |
| burst | 400 | 400 | 20,000 | 11,819 | 17,962 (**90 %**) | 49.8 / 50.2 | 0.165 / **3.48** | 71.3 | 1.3 / 127 / 127 (**1 of 13**) | 141 (93) | 1.0 | 0.48 | 0.35 / 0.53 |
| burst | 800 | 800 | 40,000 | 13,405 | 16,237 (**41 %**) | 49.6 / 50.2 | 2.94 / **9.88** | 69.0 | 1.5 / 809 / 809 (**3 of 15**) | 161 (103) | 1.4 | 0.58 | 0.41 / 0.57 |
| ramp 20 s | 50 | 50 | 2,500 | 518 | 638 (arrival-bound) | 50.2 / 50.2 | 0.001 / 0.003 | 20.8 | 0.8 / 2.7 / 2.7 (0 of 50) | 5 (3) | 0.1 | 0.02 | 0.02 / 0.75 |
| ramp 20 s | 200 | 200 | 10,000 | 2,048 | 2,560 (arrival-bound) | 50.2 / 50.2 | 0.002 / 0.005 | 22.2 | 1.1 / 3.8 / 3.8 (0 of 51) | 21 (11) | 0.2 | 0.10 | 0.06 / 0.63 |
| ramp 20 s | 400 | 400 | 20,000 | 4,087 | 5,120 (arrival-bound) | 50.2 / 50.2 | 0.002 / 0.005 | 22.1 | 1.0 / 4.8 / 4.8 (0 of 51) | 42 (25) | 0.4 | 0.17 | 0.11 / 0.50 |
| ramp 20 s | 800 | 800 | 40,000 | 8,167 | 10,239 (arrival-bound) | 50.2 / 50.2 | 0.002 / 0.007 | 22.5 | 1.5 / 10.1 / 10.1 (0 of 51) | 79 (47) | 0.8 | 0.31 | 0.22 / 0.43 |

Against gate 1 (section 9.5 of the design), the stdlib transport fails at
burst: 90 % delivered at 400 streams (target 95 %) and 41 % at 800 (target
90 %), first tokens up to 3.5 s and 9.9 s late, and `/healthz` taking up to
0.8 s with misses. No connection was reset (the backlog of 1024 holds). The
busiest thread stays near 1 % because every stream has its own thread; the
cost is the parent as a whole: about 1.4 to 1.6 cores, two thirds of it
kernel time (one `write` and one `flush` per chunk, thread wake-ups), behind
one GIL. Spread over 20 s the same 800 streams are cheap (0.8 cores), so the
burst, not the steady rate, is what breaks it.

### Bench B: DeepInfer with simulated ranks (after the DeepInfer fix)

Defaults (dp 4, 70 steps/s, 1 token per step, `max_num_seqs` 128, so the
ceiling is 512 streams), burst.

| Tokenizer | N | OK | Sim ceiling tok/s | Total tok/s | Steady tok/s (delivered) | Per stream tok/s p5 / median | TTFT median / p99 (s) | Gap p99 (ms) | Healthz p50 / p99 / max (misses) | Parent CPU % (kernel) | Busiest thread % | Python user cores | py-spy GIL / serving share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fake | 50 | 50 | 3,500 | 3,434 | 3,483 (100 %) | 70.2 / 70.3 | 0.049 / 0.078 | 51.4 | 0.9 / 2.1 / 2.1 (0 of 8) | 15 (6) | 1.7 | 0.10 | 0.04 / 0.62 |
| fake | 200 | 200 | 14,000 | 13,383 | 14,008 (100 %) | 69.7 / 70.1 | 0.104 / 0.163 | 52.3 | 0.9 / 4.6 / 4.6 (0 of 8) | 51 (20) | 3.7 | 0.31 | 0.09 / 0.66 |
| fake | 400 | 400 | 28,000 | 24,569 | 27,966 (100 %) | 66.2 / 67.2 | 0.224 / 0.509 | 59.3 | 1.5 / 19.4 / 19.4 (0 of 9) | 109 (48) | 5.9 | 0.60 | 0.17 / 0.80 |
| fake | 800 | 800 | 35,840 | 24,776 | 27,624 (77 %) | 59.7 / 61.2 | 0.463 / 4.60 | 106.2 | 1.0 / 961 / 961 (0 of 16) | 91 (41) | 5.7 | 0.50 | 0.13 / 0.78 |
| real | 50 | 50 | 3,500 | 3,435 | 3,507 (100 %) | 70.2 / 70.3 | 0.058 / 0.073 | 51.5 | 1.0 / 1.9 / 1.9 (0 of 8) | 16 (6) | 1.4 | 0.10 | 0.05 / 0.65 |
| real | 200 | 200 | 14,000 | 13,324 | 13,972 (100 %) | 69.9 / 70.1 | 0.114 / 0.188 | 52.3 | 1.0 / 1.7 / 1.7 (0 of 8) | 51 (21) | 3.1 | 0.29 | 0.06 / 0.65 |
| real | 400 | 400 | 28,000 | 24,882 | 28,006 (100 %) | 68.9 / 69.5 | 0.174 / 0.448 | 61.3 | 2.7 / 5.3 / 5.3 (0 of 9) | 105 (49) | 5.6 | 0.56 | 0.15 / 0.79 |
| real | 800 | 800 | 35,840 | 26,714 | 33,779 (94 %) | 68.3 / 69.5 | 0.622 / 4.03 | 103.4 | 1.0 / 1,166 / 1,166 (0 of 15) | 90 (39) | 5.7 | 0.51 | 0.14 / 0.76 |
| real, 140 steps/s, 1,024 tokens | 400 | 400 | 56,000 | 52,567 | 56,058 (100 %) | 138.5 / 139.2 | 0.169 / 0.460 | 58.5 | 1.7 / 30.7 / 30.7 (0 of 16) | 111 (47) | 9.9 | 0.65 | 0.17 / 0.76 |

With the DeepInfer fix the parent keeps up with the simulated ranks up to
400 streams (gate 2's throughput target holds on stdlib), but it spends about
1.1 cores doing it, almost half in the kernel, and at 400 streams three
quarters of its GIL samples sit in the serving layer (gate 2 wants under 5 %).
At 800 streams (past the 512-stream ceiling, so 288 wait in the engine) the
burst of connections pushes `/healthz` past one second and first tokens past
4 s. These are the numbers the native transport has to beat; the gates are
measured on the same harness with `--transport native`.

### `/healthz` with a dense prober (rerun of the burst rows)

The tables above come from the worker-like prober alone: 8 to 16 samples
per burst level, so p99 is the maximum, and its misses were left out of the
latency figures (the 809 ms "max" at 800 streams hid 3 probes of 3 s or
more). `load.py` now runs a second prober every 50 ms and counts every miss
at the 3 s timeout (`healthz_dense`, and `max_answered_ms` for the slowest
answer). The burst rows were rerun on stdlib with both probers; delivered
throughput and TTFT came out within run-to-run noise of the tables above.
Raw lines: `results/serving-baseline-healthz-2026-10-23.jsonl`. Latencies in
ms; misses / probes.

| Bench | N | Delivered | TTFT p99 (s) | Worker prober p50 / max answered (misses) | Dense prober p50 / p99 / max answered (misses) |
| --- | --- | --- | --- | --- | --- |
| A null | 50 | 100 % | 0.020 | 0.9 / 2.0 (0 of 11) | 0.8 / 1.8 / 1.8 (0 of 104) |
| A null | 200 | 100 % | 0.119 | 1.1 / 2.6 (0 of 11) | 1.0 / 3.1 / 94 (0 of 105) |
| A null | 400 | 89 % | 3.03 | 2.0 / 2,714 (0 of 13) | 1.0 / **3,000** / 153 (**1 of 99**) |
| A null | 800 | 43 % | 8.95 | 1.8 / 2,879 (**2 of 14**) | 0.9 / **3,000** / 324 (**3 of 104**) |
| B DeepInfer, real tokenizer | 400 | 100 % | 0.447 | 1.3 / 5.2 (0 of 9) | 1.5 / **375** / 375 (0 of 76) |
| B DeepInfer, real tokenizer | 800 | 93 % | 4.08 | 2.0 / 1,571 (0 of 14) | 1.0 / 44 / 2,020 (0 of 116) |

So on stdlib `/healthz` fails gate 1 (p99 under 50 ms, no probe over 3 s)
at 400 and 800 null-backend streams, and DeepInfer at 400 streams has a p99
of 375 ms that the worker-like prober never saw. The median stays near 1 ms:
most probes are fast, and the slow ones are those that arrive while the
burst of connections is being accepted. These dense-prober rows are the
`/healthz` baseline for T8 and T10.

Reading the py-spy column: it is an upper bound (see `gil` above), and on this
harness the serving share includes `json.dumps` of every chunk, which stays in
Python with the native transport too.

## Native transport gates (2026-10-23, IK-537 T8 and T10)

The same harness with `--transport native` (the Rust HTTP/SSE server of the
SDK, commit 07db64a, which wakes stream bodies in batches; see "Tuning"
below), next to the stdlib transport rerun with the same instrumentation.
One run per row unless it says repeat; same shared 12-vCPU VM (load average
3 to 5 from other work). Raw lines: `results/gate1-2026-10-23.jsonl` and
`results/gate2-2026-10-23.jsonl`. Two measurements are new since the baseline
tables: the GIL-held fraction is now exact (`gil_exact`, read from the
parent's memory; the py-spy figure is an upper bound on stdlib and close to
useless on native), and the null backend stamps its first token so the TTFT
the runtime adds can be separated from the backend's own latency
(`added_ttft_ms`).

### Gate 1: null backend, 256 tokens at 50 tok/s per stream

Design 9.5 asks for an 8 vCPU box: the backend is pinned to CPUs 0 to 7
(`--server-cpus 0-7`) and the clients to CPUs 8 to 11, except the row marked
unpinned. Native at 400 and 800 streams was run three times (burst, burst
repeat twice).

| Criterion (gate 1) | Target | Native, measured | Verdict |
| --- | --- | --- | --- |
| 400 streams delivered | >= 95 % (19,000 tok/s) | 100.0 / 99.6 / 99.9 % (19,994 / 19,921 / 19,973 tok/s) | PASS |
| 800 streams delivered | >= 90 % (36,000 tok/s) | 100.1 / 99.7 / 100.3 % (40,032 / 39,890 / 40,112 tok/s) | PASS |
| Connection errors at a burst of 800 | 0 | 0 in every run (stdlib: 0 too, the backlog of 1024 holds) | PASS |
| `/healthz` dense prober p99 | < 50 ms | 400: 3.7 / 13.7 / 7.5 ms; 800: 31.9 / 4.1 / 6.3 ms (slowest answer 80 ms) | PASS |
| `/healthz` misses, either prober | 0 | 0 in every native run (stdlib: 1 at 400, 3 + 3 at 800) | PASS |
| Added TTFT over the backend's first yield, p99 | < 20 ms | burst: 400: 77 / 86 / 110 ms, 800: 275 / 348 / 303 ms; ramp 20 s: 5 to 10 ms at every level | **FAIL at burst** (PASS with a 20 s ramp) |
| Busiest OS thread | < 85 % of a core | 9 to 12 % (`ik-http-io-N`) | PASS |
| GIL-held fraction | < 50 % | 400: 0.40 / 0.48 / 0.45; 800: 0.69 / 0.68 / 0.73 | PASS at 400 (margin 2 to 10 points), **FAIL at 800** |
| Every stream ends with `[DONE]` | 100 % | 100 % in every run | PASS |

Against the stdlib rows of the same table, native turns 90.9 % and 39.3 %
delivered (400 and 800 burst) into 100 %, first tokens 3.1 s and 9.6 s late
into 0.08 to 0.35 s, and `/healthz` misses into none.

**Why added TTFT fails at burst.** Nearly all of it is on the server between
the TCP connect and the backend's generator first running
(`server_before_backend_p99`: 65 to 80 ms at 400, 245 to 326 ms at 800); the
backend's own first-yield latency is under 1 ms and the delivery of the first
token is about 1 ms. Every request start runs Python (`prepare_chat`, the
`json.loads` of the body, the generator's first resume) under the GIL, so a
burst of N simultaneous starts is serialized: about 0.1 to 0.15 ms each when
the GIL is otherwise idle (a burst with a 2 tok/s backend, GIL 3 to 5 %, still
shows 54 ms at 400 and 100 ms at 800 server side; three bursts in a row on one
server, with the request threads already spawned, give the same 71 to 93 ms
at 800, so thread creation is not the cause). At 800 the 40,000 chunks a
second the earlier streams already push take most of the GIL, and the starts
wait longer. With arrivals spread over 20 s the same levels add 5 to 6 ms at
p99. A 20 ms p99 for a simultaneous burst of 400 would need under 50 us of
GIL time per request start, which no Python request path gets near.

**Why the GIL fails at 800.** 800 streams at 50 tok/s are 40,000 chunks a
second, and each chunk holds the GIL for about 17 to 20 us. A breakdown with
the same pacing loop in isolation (exact sampler, `push` vs no `push`):
about 4 us is the generator thread waking from `time.sleep` and taking the
GIL, about 12 us is building the chunk dict and `json.dumps` of it (1.9 us
in a hot loop, but every chunk runs on a different, cache-cold thread), and
`push` with its pyo3 call is 1 to 3 us after the tuning below (7 to 9 us
before). `--chunk-format str` (the backend serializes, the runtime passes the
string through) does not lower it: the `json.dumps` just moves into the
backend. What is left is per-chunk Python in the backend and in the
`drive_chat_stream` loop, which the native transport cannot remove; the
design's phase 2 (chunk helpers formatted in Rust, a batched push API) is the
lever for it.

| Run | N | OK / errors | Steady tok/s (delivered) | Added TTFT p50 / p99 (ms) [server side p99] | Gap p99 (ms) | Dense healthz p50 / p99 / max answered (misses) | Worker-like misses | Parent CPU % (kernel) | Busiest thread % | GIL held | Frames per chunk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native burst | 50 | 50 / 0 | 2,499 (100.0 %) | 5.7 / 12.8 [5.8] | 22.1 | 0.5 / 2.5 / 3.5 (0 of 103) | 0 of 11 | 17 (9) | 2.0 | 0.06 | 0.989 |
| native burst | 200 | 200 / 0 | 9,999 (100.0 %) | 20.6 / 35.6 [24.6] | 22.5 | 0.5 / 1.9 / 3.7 (0 of 103) | 0 of 11 | 59 (29) | 5.1 | 0.209 | 0.989 |
| native burst | 400 | 400 / 0 | 19,994 (100.0 %) | 40.2 / 76.5 [65.1] | 23.6 | 0.6 / 3.7 / 7.5 (0 of 105) | 0 of 11 | 123 (64) | 9.2 | 0.402 | 0.988 |
| native burst | 800 | 800 / 0 | 40,032 (100.1 %) | 104.8 / 274.6 [245.2] | 149.4 | 0.8 / 31.9 / 36.6 (0 of 113) | 0 of 12 | 277 (176) | 12.0 | 0.687 | 0.482 |
| native ramp | 50 | 50 / 0 | 638 (25.5 %) | 1.1 / 10.0 [3.4] | 22.1 | 0.5 / 2.0 / 3.4 (0 of 494) | 0 of 50 | 5 (3) | 0.8 | 0.016 | 0.991 |
| native ramp | 200 | 200 / 0 | 2,560 (25.6 %) | 1.3 / 4.9 [2.1] | 22.6 | 0.5 / 1.7 / 4.4 (0 of 500) | 0 of 51 | 16 (8) | 1.5 | 0.047 | 0.99 |
| native ramp | 400 | 400 / 0 | 5,120 (25.6 %) | 1.3 / 4.8 [2.0] | 22.4 | 0.5 / 2.1 / 9.4 (0 of 501) | 0 of 51 | 29 (15) | 2.4 | 0.091 | 0.99 |
| native ramp | 800 | 800 / 0 | 10,240 (25.6 %) | 1.5 / 6.1 [2.2] | 23.0 | 0.5 / 1.7 / 8.7 (0 of 501) | 0 of 51 | 56 (29) | 4.3 | 0.176 | 0.99 |
| stdlib burst | 50 | 50 / 0 | 2,500 (100.0 %) | 11.2 / 16.1 [14.9] | 20.7 | 0.8 / 4.2 / 4.5 (0 of 104) | 0 of 11 | 18 (11) | 0.4 | 0.077 | - |
| stdlib burst | 200 | 200 / 0 | 9,998 (100.0 %) | 63.4 / 141.6 [137.1] | 21.9 | 1.1 / 8.0 / 114.7 (0 of 105) | 0 of 11 | 85 (49) | 1.0 | 0.29 | - |
| stdlib burst | 400 | 400 / 0 | 18,180 (90.9 %) | 148.4 / 3109.2 [3094.4] | 72.5 | 1.2 / 3000.0 / 319.2 (1 of 95) | 0 of 13 | 140 (91) | 1.4 | 0.373 | - |
| stdlib burst | 800 | 800 / 0 | 15,737 (39.3 %) | 2880.2 / 9581.3 [9567.8] | 72.2 | 1.0 / 3000.0 / 878.3 (3 of 105) | 3 of 15 | 153 (99) | 1.5 | 0.412 | - |
| stdlib ramp | 50 | 50 / 0 | 638 (25.5 %) | 0.8 / 3.2 [1.5] | 20.9 | 0.9 / 2.7 / 8.4 (0 of 494) | 0 of 50 | 6 (2) | 0.5 | 0.028 | - |
| stdlib ramp | 200 | 200 / 0 | 2,560 (25.6 %) | 1.1 / 5.9 [4.4] | 22.5 | 1.0 / 4.2 / 5.6 (0 of 500) | 0 of 51 | 23 (14) | 0.8 | 0.076 | - |
| stdlib ramp | 400 | 400 / 0 | 5,120 (25.6 %) | 1.0 / 4.5 [3.0] | 22.2 | 0.9 / 3.3 / 7.7 (0 of 501) | 0 of 51 | 41 (24) | 0.9 | 0.128 | - |
| stdlib ramp | 800 | 800 / 0 | 10,240 (25.6 %) | 1.0 / 5.3 [4.0] | 22.6 | 1.1 / 4.8 / 11.3 (0 of 501) | 0 of 51 | 77 (47) | 1.2 | 0.235 | - |
| native burst repeat | 400 | 400 / 0 | 19,921 (99.6 %) | 33.7 / 85.7 [76.0] | 54.5 | 0.8 / 13.7 / 19.0 (0 of 105) | 0 of 11 | 148 (86) | 10.3 | 0.482 | 0.943 |
| native burst repeat | 800 | 800 / 0 | 39,890 (99.7 %) | 94.5 / 348.1 [325.8] | 139.3 | 0.7 / 4.1 / 53.8 (0 of 111) | 0 of 12 | 267 (164) | 11.7 | 0.675 | 0.519 |
| native burst repeat | 400 | 400 / 0 | 19,973 (99.9 %) | 63.0 / 109.7 [79.4] | 39.2 | 0.7 / 7.5 / 20.6 (0 of 105) | 0 of 11 | 143 (79) | 10.4 | 0.451 | 0.978 |
| native burst repeat | 800 | 800 / 0 | 40,112 (100.3 %) | 97.8 / 303.4 [285.2] | 168.1 | 0.9 / 6.3 / 80.4 (0 of 110) | 0 of 12 | 277 (178) | 11.7 | 0.727 | 0.435 |
| native burst flush tick 0 | 400 | 400 / 0 | 20,025 (100.1 %) | 41.9 / 104.9 [92.6] | 34.9 | 0.7 / 5.4 / 34.1 (0 of 106) | 0 of 11 | 180 (113) | 17.1 | 0.514 | 0.984 |
| native burst flush tick 0 | 800 | 800 / 0 | 40,156 (100.4 %) | 136.0 / 446.0 [422.0] | 199.7 | 0.8 / 8.7 / 100.8 (0 of 116) | 0 of 12 | 290 (198) | 15.6 | 0.732 | 0.458 |
| native burst str chunks | 400 | 400 / 0 | 19,983 (99.9 %) | 50.0 / 88.8 [69.5] | 33.3 | 0.7 / 6.4 / 16.9 (0 of 106) | 0 of 11 | 157 (90) | 10.1 | 0.496 | 0.983 |
| native burst str chunks | 800 | 800 / 0 | 40,048 (100.1 %) | 100.0 / 337.0 [303.5] | 164.0 | 0.8 / 17.9 / 93.8 (0 of 113) | 0 of 12 | 274 (176) | 10.7 | 0.708 | 0.438 |
| native burst unpinned | 400 | 400 / 0 | 20,006 (100.0 %) | 42.0 / 98.9 [87.0] | 73.7 | 0.7 / 7.9 / 26.8 (0 of 105) | 0 of 11 | 154 (83) | 9.9 | 0.485 | 0.907 |
| native burst unpinned | 800 | 800 / 0 | 39,858 (99.6 %) | 115.7 / 319.2 [299.2] | 155.7 | 0.9 / 10.7 / 63.8 (0 of 113) | 0 of 12 | 306 (200) | 11.0 | 0.705 | 0.453 |


### Tuning (T8)

- **Body wakes (changed).** Each `push` woke its stream's body through the
  channel; with the IO threads parked that wake is a futex syscall (an IPI on
  this VM) made while holding the GIL. The server now hands the body's waker to
  a flusher task that wakes an idle server at once and, while pushes keep
  coming, wakes the pending bodies in one batch per 1 ms tick from an IO
  thread (`IK_BACKEND_FLUSH_TICK_US`, default 1000; 0 restores a wake per
  push). Rows "flush tick 0" against the native burst rows: at 400 streams the
  GIL goes from 0.51 to 0.40 to 0.48, parent CPU from 180 % to 123 to 148 %,
  the busiest thread from 17 % to 10 %; at 800 the GIL from 0.73 to 0.68 to
  0.73 and parent CPU from 290 % to 267 to 277 %. The bytes on the wire are
  the same (a Python test compares them).
- **`io_threads` (not changed).** A sweep of 1, 2, 4 and 8 IO threads at 400
  and 800 streams (before the flusher, unpinned, not saved) left the GIL
  within noise (0.53 to 0.56 at 400, 0.72 to 0.73 at 800) and only moved the
  busiest thread: 51 %, 35 %, 18 %, 11 % of a core. The default
  `min(4, ncpu)` stays.
- **Channel size and coalesce cap (not changed).** Neither binds here: at
  50 tok/s a stream's 256-event channel only fills if its client stalls for
  about 5 s, and frames are 0.9 to 1.0 per chunk at 400 streams (each stream
  has one event per 20 ms, so there is nothing to coalesce) and 0.44 to 0.52 at
  800, far below the 64 KiB cap.

### Gate 2: DeepInfer with simulated ranks, native transport and the DeepInfer fix

Unpinned (the ranks need cores of their own), 256 tokens, burst, dp 4,
`max_num_seqs` 128, real tokenizer unless noted.

| Criterion (gate 2, 400 streams) | Target | Native, measured | Verdict |
| --- | --- | --- | --- |
| Throughput within 10 % of the simulated capacity (28,000 tok/s) | >= 25,200 tok/s | steady 28,023 / 28,026 / 27,875 (99.6 to 100.1 %); total_tok_s (whole wall time, ramp-up and tail included) 25,794 / 25,871 / 25,465 (90.9 to 92.4 %) | PASS (fake tokenizer: steady 90.6 % in one run, 100.1 % in the repeat) |
| Connection errors, healthz misses | 0, 0 | 0, 0 (dense p99 10 to 16 ms; stdlib 685 ms) | PASS |
| Parent busiest thread | < 85 % of a core | 5.9 / 6.4 / 6.1 % | PASS |
| Serving-layer share of parent CPU (py-spy) | < 5 % | `socketserver` / `http.server` / socket frames: 0 % (framing and writes are in Rust). The Python that remains per chunk, `serve.py`'s stream driver and `json.dumps` of each chunk, is 9.2 % of parent CPU (blocking py-spy at 20 Hz, 151 samples: driver 13 % and `json` 13 % of GIL samples, GIL held 0.51) | PASS for the frames the design names; **FAIL (9.2 %)** if the runtime's per-chunk Python counts |

The serving share comes from a separate attribution run with a blocking
py-spy (rows "attribution"): the nonblocking sampler, used everywhere else,
sees almost none of the GIL holders on native. A blocking sampler pauses the
parent, so read only the attribution of those two rows, not their throughput.
Per delivered token, the runtime's serving frames cost about 4 us on native
against 7 us on stdlib (0.088 cores at 21,800 tok/s against 0.099 cores at
13,500 tok/s).

**The four configurations** ({stdlib, native} x {DeepInfer fix off, on}, 400
streams): fix off (the example as of before c722f5c, run from a scratch
worktree on today's runtime) delivers 1,690 to 1,760 tok/s on stdlib and
1,540 to 1,625 tok/s on native, with the parent's reader thread at 75 to
83 % of a core and the GIL held 0.90 to 0.93 of the time, its top frames in
`transformers` decoding: the transport cannot help while the per-token
re-decode is there (native only removes the `/healthz` stalls: dense p99
2 to 5 ms against 41 to 52 ms, slowest answer 8 to 29 ms against up to 2.2 s,
and stdlib's one miss). Fix on, both
transports reach the ceiling at 400 streams; native does it with 94 % parent
CPU instead of 117 %, `/healthz` p99 16 ms instead of 685 ms and first tokens
p99 0.29 s instead of 0.72 s.

**Capacity** (1,024 tokens, burst): at 140 steps/s and 400 streams native
delivers steady 54,752 tok/s of 56,000 (97.8 %; stdlib 96.0 %, with
`/healthz` p99 149 ms and a 612 ms answer); at 200 steps/s and 200 streams
39,599 of 40,000 (99.0 %); at 210 steps/s and 400 streams 76,357 tok/s
(90.9 % of 84,000) against stdlib's 59,908 (71.3 %). In that last row the
simulated ranks themselves only reach 175 steps/s on this VM (about 73,000
tok/s possible), so native is close to what was produced; stdlib's parent
falls behind (per stream 156 tok/s against native's 190).

At 800 streams (past the 512-stream ceiling, 288 streams wait in the engine)
`total_tok_s` is the same on both transports (26,151 / 26,286 native, 26,696
/ 26,643 stdlib), and on native the clients finish when the ranks do. The
lower native steady figure (78 % against 89 to 95 %) is the steady window
(the middle 80 % of a two-phase run) rather than lost tokens; not a gate
level, recorded for completeness.


| Run | N | Step rate | OK / errors | Sim ceiling tok/s | Total tok/s (of ceiling) | Steady tok/s (delivered) | TTFT p99 (s) | Dense healthz p50 / p99 / max answered (misses) | Worker-like misses | Parent CPU % (kernel) | Busiest thread % | GIL held | Serving share of parent CPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native, fix on, real tokenizer | 50 | 70 | 50 / 0 | 3,500 | 3,443 (98.4 %) | 3,493 (99.8 %) | 0.067 | 0.5 / 6.3 / 6.3 (0 of 75) | 0 of 8 | 14 (4) | 1.4 | 0.072 | - |
| native, fix on, real tokenizer | 200 | 70 | 200 / 0 | 14,000 | 13,487 (96.3 %) | 13,945 (99.6 %) | 0.146 | 0.5 / 8.0 / 8.0 (0 of 77) | 0 of 8 | 51 (18) | 3.7 | 0.259 | - |
| native, fix on, real tokenizer | 400 | 70 | 400 / 0 | 28,000 | 25,794 (92.1 %) | 28,023 (100.1 %) | 0.29 | 0.6 / 15.9 / 15.9 (0 of 80) | 0 of 8 | 94 (35) | 5.9 | 0.491 | - |
| native, fix on, real tokenizer | 800 | 70 | 800 / 0 | 35,840 | 26,151 (73.0 %) | 27,890 (77.8 %) | 4.174 | 0.5 / 10.3 / 51.3 (0 of 157) | 0 of 16 | 74 (29) | 5.6 | 0.372 | - |
| native, fix on, fake tokenizer | 50 | 70 | 50 / 0 | 3,500 | 3,433 (98.1 %) | 3,513 (100.4 %) | 0.078 | 0.5 / 5.1 / 5.1 (0 of 76) | 0 of 8 | 15 (6) | 1.7 | 0.068 | - |
| native, fix on, fake tokenizer | 200 | 70 | 200 / 0 | 14,000 | 13,230 (94.5 %) | 14,073 (100.5 %) | 0.216 | 0.6 / 3.7 / 3.7 (0 of 78) | 0 of 8 | 50 (18) | 4.0 | 0.257 | - |
| native, fix on, fake tokenizer | 400 | 70 | 400 / 0 | 28,000 | 24,246 (86.6 %) | 25,379 (90.6 %) | 0.323 | 0.6 / 29.7 / 29.7 (0 of 86) | 0 of 9 | 94 (35) | 5.9 | 0.476 | - |
| stdlib, fix on, real tokenizer | 50 | 70 | 50 / 0 | 3,500 | 3,438 (98.2 %) | 3,524 (100.7 %) | 0.073 | 0.9 / 6.3 / 6.3 (0 of 76) | 0 of 8 | 16 (6) | 1.4 | 0.088 | - |
| stdlib, fix on, real tokenizer | 200 | 70 | 200 / 0 | 14,000 | 13,205 (94.3 %) | 14,045 (100.3 %) | 0.224 | 0.9 / 147.1 / 147.1 (0 of 77) | 0 of 8 | 56 (22) | 3.9 | 0.285 | - |
| stdlib, fix on, real tokenizer | 400 | 70 | 400 / 0 | 28,000 | 23,270 (83.1 %) | 27,961 (99.9 %) | 0.72 | 2.7 / 685.1 / 685.1 (0 of 76) | 0 of 9 | 117 (53) | 6.1 | 0.523 | - |
| stdlib, fix on, real tokenizer | 800 | 70 | 800 / 0 | 35,840 | 26,696 (74.5 %) | 31,988 (89.3 %) | 4.153 | 1.3 / 145.9 / 2,024.8 (0 of 112) | 0 of 14 | 103 (46) | 6.6 | 0.502 | - |
| native, fix on, real tokenizer, repeat | 400 | 70 | 400 / 0 | 28,000 | 25,871 (92.4 %) | 28,026 (100.1 %) | 0.298 | 0.6 / 14.9 / 14.9 (0 of 80) | 0 of 8 | 97 (38) | 6.4 | 0.502 | - |
| native, fix on, real tokenizer, repeat | 400 | 70 | 400 / 0 | 28,000 | 25,465 (90.9 %) | 27,875 (99.6 %) | 0.298 | 0.6 / 10.2 / 10.2 (0 of 82) | 0 of 9 | 93 (35) | 6.1 | 0.487 | - |
| native capacity 140 steps/s | 400 | 140 | 400 / 0 | 56,000 | 51,591 (92.1 %) | 54,752 (97.8 %) | 0.283 | 0.6 / 3.1 / 7.5 (0 of 160) | 0 of 16 | 105 (38) | 11.2 | 0.561 | - |
| stdlib capacity 140 steps/s | 400 | 140 | 400 / 0 | 56,000 | 48,317 (86.3 %) | 53,770 (96.0 %) | 0.752 | 4.9 / 149.3 / 612.0 (0 of 157) | 0 of 18 | 127 (55) | 11.5 | 0.6 | - |
| native capacity 200 steps/s | 200 | 200 | 200 / 0 | 40,000 | 37,493 (93.7 %) | 39,599 (99.0 %) | 0.158 | 0.5 / 7.0 / 7.6 (0 of 109) | 0 of 11 | 57 (19) | 8.7 | 0.312 | - |
| native capacity 210 steps/s | 400 | 210 | 400 / 0 | 84,000 | 72,860 (86.7 %) | 76,357 (90.9 %) | 0.222 | 0.6 / 8.5 / 14.7 (0 of 113) | 0 of 12 | 114 (41) | 14.8 | 0.623 | - |
| stdlib capacity 210 steps/s | 400 | 210 | 400 / 0 | 84,000 | 58,718 (69.9 %) | 59,908 (71.3 %) | 1.039 | 8.7 / 57.9 / 926.9 (0 of 122) | 0 of 14 | 131 (56) | 13.9 | 0.644 | - |
| stdlib, fix off, real tokenizer | 50 | 70 | 50 / 0 | 3,500 | 1,725 (49.3 %) | 1,727 (49.4 %) | 0.073 | 13.1 / 41.1 / 60.8 (0 of 148) | 0 of 15 | 104 (15) | 75.2 | - | - |
| stdlib, fix off, real tokenizer | 200 | 70 | 200 / 0 | 14,000 | 1,944 (13.9 %) | 1,623 (11.6 %) | 5.346 | 14.5 / 41.2 / 1,765.6 (0 of 492) | 0 of 52 | 110 (16) | 78.4 | - | - |
| stdlib, fix off, real tokenizer | 400 | 70 | 400 / 0 | 28,000 | 1,689 (6.0 %) | 1,690 (6.0 %) | 40.26 | 16.6 / 51.7 / 2,186.2 (1 of 1104) | 1 of 114 | 107 (17) | 75.0 | - | - |
| native, fix off, real tokenizer | 50 | 70 | 50 / 0 | 3,500 | 1,873 (53.5 %) | 1,548 (44.2 %) | 0.073 | 0.6 / 4.2 / 24.8 (0 of 137) | 0 of 14 | 105 (13) | 79.4 | - | - |
| native, fix off, real tokenizer | 200 | 70 | 200 / 0 | 14,000 | 2,002 (14.3 %) | 1,514 (10.8 %) | 0.173 | 0.5 / 2.5 / 7.5 (0 of 510) | 0 of 52 | 110 (14) | 83.1 | - | - |
| native, fix off, real tokenizer | 400 | 70 | 400 / 0 | 28,000 | 1,902 (6.8 %) | 1,538 (5.5 %) | 0.37 | 0.5 / 4.7 / 29.3 (0 of 1073) | 0 of 108 | 110 (14) | 82.1 | - | - |
| attribution: native, fix on, blocking py-spy 20 Hz | 400 | 70 | 400 / 0 | 28,000 | 18,499 (66.1 %) | 21,828 (78.0 %) | 0.404 | 5.1 / 150.3 / 296.2 (0 of 410) | 0 of 45 | 96 (52) | 4.7 | 0.508 | 9.2 % |
| attribution: stdlib, fix on, blocking py-spy 20 Hz | 400 | 70 | 400 / 0 | 28,000 | 21,402 (76.4 %) | 13,466 (48.1 %) | 3.265 | 33.0 / 208.8 / 1,635.7 (0 of 271) | 0 of 37 | 122 (72) | 5.9 | 0.554 | 8.1 % |
| stdlib, fix off, real tokenizer, GIL measured | 400 | 70 | 400 / 0 | 28,000 | 1,821 (6.5 %) | 1,760 (6.3 %) | 34.606 | 14.5 / 40.7 / 2,077.0 (1 of 1023) | 1 of 106 | 110 (17) | 77.9 | 0.897 | - |
| native, fix off, real tokenizer, GIL measured | 400 | 70 | 400 / 0 | 28,000 | 1,962 (7.0 %) | 1,625 (5.8 %) | 0.364 | 0.5 / 2.4 / 19.5 (0 of 1042) | 0 of 105 | 110 (14) | 82.8 | 0.929 | - |
| native, fix on, fake tokenizer, repeat | 400 | 70 | 400 / 0 | 28,000 | 25,388 (90.7 %) | 28,030 (100.1 %) | 0.348 | 0.5 / 7.4 / 7.4 (0 of 81) | 0 of 9 | 86 (31) | 5.9 | 0.453 | - |
| native, fix on, real tokenizer, repeat | 800 | 70 | 800 / 0 | 35,840 | 26,286 (73.3 %) | 28,254 (78.8 %) | 4.154 | 0.5 / 2.2 / 92.3 (0 of 156) | 0 of 16 | 70 (26) | 5.4 | 0.352 | - |
| stdlib, fix on, real tokenizer, repeat | 800 | 70 | 800 / 0 | 35,840 | 26,643 (74.3 %) | 33,896 (94.6 %) | 4.03 | 1.1 / 21.8 / 1,687.6 (0 of 122) | 0 of 14 | 95 (41) | 6.0 | 0.473 | - |


### What the simulation does not model

- DeepInfer's own Python scheduler time in the ranks and the GPU step time:
  one token per step at a fixed step rate. The rank CPU added by
  detokenization (3 % before, 6 to 7 % at 400 streams, 11 % at 140 to 200
  steps/s) is never charged against a GPU step.
- Separate machines: the clients run on the same 12-vCPU VM and use 60 to 65 %
  CPU at 400 streams, competing with the ranks for cores; the rank step rate
  drops to 62 to 65 steps/s in the burst runs.
- The real engine's `add_request` and `abort_request` paths, the
  `finish_reason` of requests the scheduler gives up on, and dead-rank
  behavior (the simulator never drops, ignores or preempts a request). These
  are covered by unit tests against fakes only; `abort_on_disconnect` can be
  turned off if aborting misbehaves on the real engine.
- Prefix caching and DSpark's accepted draft tokens (approximated with
  `--tokens-per-step`).

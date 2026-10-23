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
| `gil` | `py-spy record --gil --nonblocking` on the parent during the run: `gil_held_fraction` (samples / rate x seconds; without pausing the process py-spy reads the GIL's last holder, so threads that just released it around a socket write or a sleep still count: an upper bound), `serving_leaf_share` (samples whose innermost frame is in `socketserver`, `http.server`, `serve.py`, `json` or `socket`) and the top innermost frames. A blocking py-spy pauses all threads at every sample and halved the throughput at 400 streams, so it is not used. `--py-spy ''` turns it off. |
| `rss_mb_max` | peak RSS of the parent |
| `frames_per_chunk`, `debug_stats` | native transport only: `/debug/stats` frames written per chunk pushed (coalescing) |

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

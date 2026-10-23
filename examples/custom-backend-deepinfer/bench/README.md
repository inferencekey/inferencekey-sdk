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
```

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
| `steady_tok_s` | median of the per-second token counts in the middle 80 % of the run (a chunk may carry several tokens; each stream's tokens are spread over its chunks by their length) |
| `per_stream_tok_s` | completion tokens / (last token - first token), p10, median, p90 |
| `ttft_s` | time to first content token |
| `cpu_pct_mean` | mean CPU % between the first and the last token: `parent` (the backend process, 100 = one core), each rank, the clients |
| `ranks_busy_tok_s`, `ranks_steps_per_s` | what the simulated ranks produced while busy, from their own counters (1 s resolution) |
| `ranks_done_s`, `clients_done_s` | when the ranks produced their last token and when the clients received it; the gap is backlog queued in the parent |
| `sim_ceiling_tok_s` | what the ranks allow: min(N, dp x max_num_seqs) x step_rate x tokens_per_step |
| `errors`, `error_samples` | failed streams (the baseline's connection resets came from the listen backlog of 5) |

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

Before and after, at the same settings:

| Run | N | Total tok/s before | after | Per stream (median) before | after | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| burst | 200 | 341 (57 reset) | 13,395 | 35 | 70 | x39, no resets, ceiling 14,000 |
| burst | 400 | 320 (266 reset) | 24,914 | 26 | 70 | x78, no resets, steady 28,034 of 28,000 |
| ramp 20 s | 400 | 1,923 | 4,332 | 28 | 70 | now bound by arrivals, TTFT median 11.4 s to 0.04 s |
| 2,048 new tokens, burst | 50 | 320 | 3,491 | 6.6 | 70 | x10.9, clients done at 29 s instead of 320 s |
| 137K-token prompts, prefill free | 50 | 542 | 543 | 23 | 70 | ranks back at 69 steps/s (was 24) |

1. **The parent is no longer the limiter.** Every short-prompt run delivers
   what the simulated ranks produce: the steady rate equals the ceiling, every
   stream gets the full 70 tok/s, and the clients finish when the ranks do
   (within about a second; the rank counters tick once a second). At 400
   streams the best baseline (1,923 tok/s) becomes 24,900 to 27,600 tok/s (x13
   to x14); in the burst runs that dropped connections before, x39 to x78.
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

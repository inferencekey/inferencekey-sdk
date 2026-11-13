# Mellum2.1 Thinking with its MTP head (vLLM, speculative decoding)

**Mellum2.1 Thinking** from JetBrains is a thinking model for code, math and
tool use: a 12B mixture-of-experts (64 experts, 8 per token, about 2.5B
parameters active), 28 layers with a 1,024-token sliding window in 3 of every
4, a 131,072-token context and the **Apache-2.0** license.
[JetBrains/Mellum2.1-12B-A2.5B-Thinking-MTP](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking-MTP)
is the same weights plus its **multi-token prediction (MTP) head**, one extra
block (0.86 GB) that drafts the next token so vLLM can check it in the same
step (speculative decoding). The card says the output is unchanged and
decoding runs 1.4 to 1.8x faster, with about 90 % of the drafted tokens
accepted.

This example deploys it with the built-in **`vllm`** backend on one private
NVIDIA GPU, with the model card's tool-calling recipe, and then:

1. asks the card's own Quickstart question (a bug in `mean()`) through the
   SDK's `generate_text()`;
2. streams a coding task and prints first-token latency and tokens/s;
3. runs a small agent loop with a `get_weather` tool until the model answers.

Switch MTP off (`MELLUM_MTP=0`) and run it again to compare on your own card:
`ensure()` updates the same workload with the new command.

| File | What it does |
| --- | --- |
| `main.py` | `ensure()` a fixed workload on your worker, wait until ready, run the three calls, save `outputs/answers-<mtp\|plain>.json`, `delete()` (unless `IK_KEEP_ALIVE=1`). |
| `.env.example` | The variables `main.py` reads. Copy to `.env`. |
| `requirements.txt` | Client side: the local SDK build and `requests`. |

## Compatibility

- **vLLM 0.33 or newer** (`vllm_version="0.33.0"`): older releases refuse the
  MTP tensors. vLLM 0.33 serves Mellum through its Transformers modeling
  backend (`MellumForCausalLM`, draft `TransformersMoEForMTP`) and installs
  transformers 5.20 and torch 2.14 for **CUDA 13** (driver 580 or newer).
- **BF16 (default)**: 23.5 GiB of weights, so a **32 GB card or larger**.
- **FP8 (`MELLUM_FP8=1`)**: vLLM's online FP8 quantization, 12.2 GiB of
  weights, fits **16 GB** with `MELLUM_CONTEXT=16384`. Native on Ada and
  Blackwell; on Ampere vLLM falls back to weight-only FP8 (Marlin).

## Measured (2026-11-13)

Single requests are one stream; "8 at once" is the total of eight parallel
requests. Warm server, same prompt, direct to vLLM unless noted.

| Card | Weights | Plain | MTP, 1 draft token (card) | MTP, 2 draft tokens |
| --- | --- | --- | --- | --- |
| RTX A6000 48 GB, 1 request | BF16 | 106.8 tok/s | 118.9 (1.11x) | 140.2 (1.31x) |
| RTX A6000 48 GB, 8 at once | BF16 | 418.4 tok/s | 607.3 (1.45x) | 581.8 (1.39x) |
| RTX 5080 16 GB, 1 request | FP8, 16K context | 217.2 tok/s | 288.7 (1.33x) | 291.2 (1.34x) |
| RTX 5080 16 GB, 8 at once | FP8, 16K context | 896.7 tok/s | 1,276.5 (1.42x) | 1,297.4 (1.45x) |

| What | Value |
| --- | --- |
| Drafted tokens accepted | 87 % with 1 draft token, 67 % with 2, 49 % with 3 (vLLM metrics) |
| Time to ready, cold worker (A6000) | 488.9 s: vLLM install about 4 min, 25 GB already downloaded, load 12.3 s, engine warm-up 179.7 s |
| Restart with a new command (warm) | 101.2 s |
| GPU memory, A6000 BF16, util 0.90 | 41,147 MiB peak; KV cache 16.0 GiB = 1,036,154 tokens (7.9x the full window) |
| GPU memory, RTX 5080 FP8 + MTP, 16K context | 15,832 of 16,303 MiB; KV cache 25,967 tokens |
| Through the endpoint (A6000, greedy) | 120 to 130 tok/s with MTP, 108 to 110 without |
| Through the endpoint (RTX 5080, FP8, 2 draft tokens) | 298 to 344 tok/s; the coding task in 2.5 s |

Greedy outputs were **not** byte-identical between repeated runs (with and
without MTP), so do not expect token-for-token reproducibility from vLLM
here; the answers stayed right (bug fixed, `merge_intervals` correct, 400 for
the math check).

## Prerequisites

- A private NVIDIA worker registered in your InferenceKey project (32 GB for
  BF16, 16 GB for FP8; CUDA 13 driver). Copy its id into `IK_WORKER_ID`.
- An SDK token (`ik_sdk_`, scopes `workload:read`, `workload:write`) and an API
  key (`ik_live_`) whose `allowed_endpoints` includes this workload.

## Run

```bash
cp .env.example .env    # fill in the values
# install the local SDK, see ../CONTRIBUTING.md#4-depending-on-the-sdk
pip install -r requirements.txt
python main.py                                   # BF16, MTP on (card settings)
MELLUM_MTP=0 python main.py                      # same, MTP off
MELLUM_FP8=1 MELLUM_CONTEXT=16384 MELLUM_SPEC_TOKENS=2 python main.py   # 16 GB card
```

## What it does

`main.py` declares a `vllm` workload pinned to your worker
(`ExecutionPolicy.FIXED`) with this command (the worker adds host and port):

```
vllm serve JetBrains/Mellum2.1-12B-A2.5B-Thinking-MTP --max-model-len 131072 \
  --gpu-memory-utilization 0.90 --max-num-seqs 16 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

## Things to know

1. **The speed-up depends on the card.** On an RTX A6000 with BF16 weights,
   the card's setting gave 1.11x on one request; on an RTX 5080 with FP8
   weights, 1.33x. With eight requests at once both cards gained about 1.4x.
   Try `MELLUM_SPEC_TOKENS=2` (best single-request speed on the A6000); 3 was
   slower again.
2. **The first sampled request is slow.** vLLM compiles a sampling kernel on
   the first request with a new sampling setting (we saw 52 tok/s, then 106).
   Send one warm-up request before you measure.
3. **One city per turn.** With tools, the model sometimes calls `get_weather`
   for one city, waits for the result, then calls it for the other. The
   example loops up to four turns.
4. **It thinks first.** Leave `max_tokens` of 8,192 or more; the thinking
   arrives in `reasoning_content`.
5. **FP8 on Ampere loads through a peak.** On the A6000, FP8 under a 16 GB cap
   served at 15.8 GB but touched 21.9 GB for a moment while quantizing at
   load. On the real RTX 5080 (native FP8) the load stayed under 16 GB.
6. vLLM logs `Add 6 padding layers, may waste at most 27.27% KV cache memory`
   and `Using default MoE config`: harmless, but there may be speed left on
   the table once vLLM ships a native Mellum implementation.

## Cost & cleanup

A `fixed` private workload keeps the GPU reserved while it exists. `main.py`
deletes it at the end unless `IK_KEEP_ALIVE=1`; if you keep it, call
`ManagementClient.delete("mellum21-thinking")` (or delete it in the panel)
when done.

## Troubleshooting

- `ValueError: No available memory for the cache blocks`: BF16 on a 24 GB card
  (the weights alone are 23.5 GiB). Use `MELLUM_FP8=1`.
- 403 `endpoint_forbidden`: add `<project>/mellum21-thinking` (or
  `<project>/*`) to the API key's `allowed_endpoints`.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

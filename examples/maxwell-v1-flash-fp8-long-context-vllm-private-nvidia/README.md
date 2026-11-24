# Maxwell-V1.0-Flash on one 48 GB card (vLLM, official FP8, 128K context)

**Maxwell-V1.0-Flash** from the Maxwell team is a 35B-total, 3B-active
mixture-of-experts model (256 routed experts, 12 per token, 2 shared; 45
layers) trained from scratch on about 26T tokens, with a 256K context and the
**Apache-2.0** license. Its block-sparse attention (MBSA) lets each KV group
attend to at most 32 blocks of 128 tokens (8 prefix, 8 local, 16 picked by a
learned indexer), so the attention work per token stops growing with the
context. The KV cache still keeps every token: memory grows with the context,
compute does not.

This example deploys the official FP8 build
([maxwell-model/Maxwell-V1-Flash-FP8](https://huggingface.co/maxwell-model/Maxwell-V1-Flash-FP8),
38.6 GB) with the built-in **`vllm`** backend on a single 48 GB NVIDIA card,
with the model card's flags (`--trust-remote-code --block-size 128`, reasoning
parser `qwen3`, tool parser `qwen3_coder`), a 131,072-token window and an FP8
KV cache, and then:

1. asks the model card's own question through the SDK's `generate_text()`;
2. optionally reads a long document of yours in one request (`MAXWELL_DOCUMENT`);
3. makes a tool call and sends the tool result back for the final answer.

| File | What it does |
| --- | --- |
| `prepare_worker.sh` | One-time setup of a self-hosted worker: installs the authors' vLLM fork (upstream wheel + the fork's Python) where the worker looks for it. |
| `main.py` | `ensure()` a fixed workload on your worker, wait until ready, run the calls, save `outputs/answers.json`, `delete()` (unless `IK_KEEP_ALIVE=1`). |
| `.env.example` | The variables `main.py` reads. Copy to `.env`. |
| `requirements.txt` | Client side: the local SDK build and `requests`. |

## Compatibility

- **GPU**: NVIDIA with **48 GB** (L40S, RTX 6000 Ada, A6000-class) for the full
  speed and a 128K window. FP8 KV cache needs Ada (sm_89) or newer.
- **24 to 32 GB cards** run it with vLLM's `--cpu-offload-gb` (part of the
  weights stay in host RAM), about 10x slower: measured below on a real RTX
  4090 and under emulated 32 GB and 24 GB caps.
- **Driver**: CUDA 13 (driver 580 or newer): the wheel brings torch 2.13 cu130.
- **vLLM**: upstream vLLM does not know this model yet. The authors' fork
  does, but their Docker image (`maxwellmodel/vllm-openai:v0.30.0`) is built
  for sm_90 and sm_103 only: its core CUDA kernels are missing for Ada and
  Ampere. `prepare_worker.sh` installs the upstream nightly wheel of the
  commit the fork is based on (kernels for sm_75 to sm_120) plus the fork's
  Python files. The MBSA kernel library itself is for Hopper and Blackwell; on
  other cards the fork runs its Triton block-sparse path.

Step-by-step guide, self-hosted or on InferenceKey: [How to deploy
Maxwell-V1.0-Flash](https://docs.inferencekey.com/en/deploy/maxwell-v1-flash/).

## Measured on one L40S 48 GB (2026-11-24)

| What | Value |
| --- | --- |
| Setup on a fresh worker | FP8 weights 38.6 GB downloaded in 25 s; `prepare_worker.sh` steps (venv, wheel, overlay) 66 s |
| Time to ready (`ensure()` to serving) | 169.9 s on the second start (load 20.7 s, 34.68 GiB of weights; engine init 111 s incl. TileLang and Triton warm-up) |
| KV cache, `--gpu-memory-utilization 0.965`, FP8 KV | 7.49 GiB = 155,136 tokens (1.18x the 131,072 window). At 0.95 the window did not fit: vLLM asked for 6.33 GiB with 6.25 GiB free and offered 129,408 tokens |
| Decode, one request (512 tokens) | 122 tok/s short prompt, 118 at 4K, 116 at 32K, 107 to 108 at 125K |
| First token, cold | 0.9 s at 4K, 5.1 s at 32K, 19.3 s at 125K (about 6,500 prompt tokens/s) |
| 125K-token contract bundle, 3 questions | right three times out of three (values and clause numbers) |
| Tool call | `get_contract({"contract_id": "MSA-2291"})` in 1.3 s, final answer 1.0 s |

### Smaller cards (CPU offload)

| GPU or cap | Configuration | GPU memory peak | Decode | 32K review, first token | Answer |
| --- | --- | --- | --- | --- | --- |
| L40S 48 GB (real) | no offload, 131K window | 44,874 MiB (vLLM reserves 0.965) | 116 tok/s | 5.1 s | right |
| 32 GB cap on the L40S (emulated) | `--cpu-offload-gb 12`, util 0.69 | 31,546 MiB | 18.7 tok/s | 43.0 s | values only in the thinking, answer listed clause numbers |
| 24 GB cap on the L40S (emulated) | `--cpu-offload-gb 20`, util 0.52 | 23,625 MiB | 12.9 tok/s | 64.1 s | same as 32 GB |
| RTX 4090 24 GB (real) | `--cpu-offload-gb 20`, util 0.92 | 22,851 MiB | 12.1 tok/s | 69.0 s | right |

Offloading needs about 20 GB of free host RAM on a 24 GB card. To try it, set
`MAXWELL_GPU_UTIL=0.92`, `MAXWELL_CONTEXT=40960` and
`MAXWELL_EXTRA_ARGS=--cpu-offload-gb 20` in `.env`.

## Prerequisites

- A private NVIDIA worker registered in your InferenceKey project (48 GB,
  CUDA 13 driver, Python 3.12). Copy its id into `IK_WORKER_ID`.
- Run `prepare_worker.sh` once on that worker host (as the user the worker
  runs as).
- An SDK token (`ik_sdk_`, scopes `workload:read`, `workload:write`) and an API
  key (`ik_live_`) whose `allowed_endpoints` includes this workload.

## Run

```bash
# on the worker host, once
./prepare_worker.sh

# on your machine
cp .env.example .env    # fill in the values
# install the local SDK, see ../CONTRIBUTING.md#4-depending-on-the-sdk
pip install -r requirements.txt
python main.py
```

## What it does

`main.py` declares a `vllm` workload pinned to your worker
(`ExecutionPolicy.FIXED`) with this command (the worker adds host and port):

```
vllm serve maxwell-model/Maxwell-V1-Flash-FP8 --revision 6f91969de0d39c9fdbb98306e8826ba4f661125b \
  --trust-remote-code --block-size 128 --max-model-len 131072 --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.965 --max-num-seqs 4 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

and `vllm_version="0.30.1rc1.dev219+ga9eafde59"`, the version
`prepare_worker.sh` installs. 0.965 of the card is what the 131,072 window
needed on a 46,068 MiB L40S (see the table); nothing else may use that GPU.

## Things to know

1. **It thinks before every answer**, and the endpoint does not forward
   `chat_template_kwargs`, so `enable_thinking: false` has no effect through
   InferenceKey. Leave `max_tokens` room for the thinking.
2. **Leave room for the output in the window.** A 124,929-token prompt with
   `max_tokens` 6,144 is one token over 131,072 and vLLM refuses it (400).
3. **Format slips.** Asked for "three numbered lines, each with the clause
   number", it sometimes answered with the clause numbers only, values left in
   the thinking (1 of 3 runs at 4K on the L40S; also on the emulated 32 GB and
   24 GB caps). Ask explicitly for the values.
4. **The first start compiles kernels** (torch.compile, TileLang mHC, Triton);
   later starts reuse the caches in `~/.cache`.

## Cost & cleanup

A `fixed` private workload keeps the GPU reserved while it exists. `main.py`
deletes it at the end unless `IK_KEEP_ALIVE=1`; if you keep it, call
`ManagementClient.delete("maxwell-v1-flash")` (or delete it in the panel) when done.

## Troubleshooting

- Install step fails with `No matching distribution found for
  vllm==0.30.1rc1.dev219+ga9eafde59`: `prepare_worker.sh` has not run on this
  worker (or ran as another user): the worker found no vLLM in its venv and
  tried PyPI.
- `ValueError: To serve at least one request with the model's max seq len ...`
  at start: raise `MAXWELL_GPU_UTIL` a little or lower `MAXWELL_CONTEXT` (the
  error prints the largest length that fits).
- `FileNotFoundError: ... 'ninja'` at warm-up: `ninja` is not on the worker's
  PATH; `prepare_worker.sh` installs it into the venv and links it.
- 403 `endpoint_forbidden`: add `<project>/maxwell-v1-flash` (or `<project>/*`)
  to the API key's `allowed_endpoints`.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

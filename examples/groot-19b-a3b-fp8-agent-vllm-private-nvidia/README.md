# Groot-19B-A3B on one RTX 5090 (vLLM, official FP8, full 256K context)

**Groot-19B-A3B-261023** from GrootTeam is a reasoning model for tool-using
and agent work, post-trained from Qwen3.6-35B-A3B. It is a 19B
mixture-of-experts (128 experts, 8 per token, about 3B parameters active per
token) with a vision encoder, a 262,144-token native context, one
multi-token-prediction (MTP) layer, and the **Apache-2.0** license. 30 of its 40
layers use linear attention, so the KV cache of the full window is small.

The authors publish BF16, **FP8** (block-wise E4M3, 20.9 GB), NVFP4 and GGUF
builds and write that FP8 runs the full 256K context on one RTX 5090. This
example deploys the FP8 build
([GrootTeam/Groot-19B-A3B-261023-FP8](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023-FP8))
with the built-in **`vllm`** backend on a single 32 GB card, with the model
card's vLLM recipe (reasoning parser, tool-call parser, MTP), and then:

1. asks a plain question through the SDK's `generate_text()`;
2. streams the card's first example (`Type "I love Groot" backwards`) with
   the thinking in `reasoning_content` and the answer in `content`;
3. makes a tool call (the card's `get_weather` example) and sends the tool
   result back for the final answer;
4. reads an image: the authors' own size-vs-score chart.

| File | What it does |
| --- | --- |
| `main.py` | `ensure()` a fixed workload on your worker, wait until ready, run the four calls, save `outputs/answers.json`, `delete()` (unless `IK_KEEP_ALIVE=1`). |
| `.env.example` | The variables `main.py` reads. Copy to `.env`. |
| `requirements.txt` | Client side: the local SDK build and `requests`. |

## Compatibility

- **GPU**: NVIDIA with **32 GB or more** (RTX 5090, RTX PRO 4500/5000, L40S,
  A6000-class). FP8 runs natively on Ada (sm_89) and newer.
- **Driver**: CUDA 13 (driver 580 or newer): `vllm==0.32.0` brings torch 2.13
  built for CUDA 13. The card asks for vLLM >= 0.19.1 and Transformers >= 5.8.1;
  vLLM 0.32.0 installs Transformers 5.19.
- **24 GB cards**: the FP8 weights alone take 19.7 GiB; with vLLM capped at
  24 GB there is room for about 9,600 tokens of context (measured, see below),
  too little for an agent. Use the authors' GGUF build on llama.cpp instead.

## Measured on one RTX 5090 32 GB (2026-11-03)

| What | Value |
| --- | --- |
| Time to ready, cold worker | 611.5 s: vLLM install about 170 s, 20.9 GB download 27.8 s, load 51.7 s (19.67 GiB), engine warm-up 312.6 s (two CUDA-graph captures and FlashInfer JIT on first start) |
| Warm restart (caches on disk, MTP off) | engine warm-up 86.8 s |
| KV cache at `--gpu-memory-utilization 0.92` | 6.77 GiB = 307,024 tokens with MTP (1.17x the 262,144 window); 7.37 GiB = 376,253 tokens without |
| GPU memory | 30,226 MiB serving; 32,016 MiB of 32,607 peak during a 205,525-token prompt |
| Decode, one request, MTP on (through the endpoint) | 284 to 304 tok/s short answers, 262 to 266 tok/s on 4.5K to 5.2K-token answers |
| Decode, MTP off (direct to vLLM) | 170 to 187 tok/s short, 210 tok/s long |
| Long prompt | 205,525 tokens: needle found, first token after 34.6 s |
| Tool calls | `get_weather` with valid JSON arguments in 0.9 s; two parallel calls (Lisbon, Oslo) and the final answer in 2.9 s |
| Image | the authors' chart read right (9,279 prompt tokens, 4.1 s) |

## Prerequisites

- A private NVIDIA worker registered in your InferenceKey project (32 GB,
  CUDA 13 driver). Copy its id into `IK_WORKER_ID`.
- An SDK token (`ik_sdk_`, scopes `workload:read`, `workload:write`) and an API
  key (`ik_live_`) whose `allowed_endpoints` includes this workload.

## Run

```bash
cp .env.example .env    # fill in the values
# install the local SDK, see ../CONTRIBUTING.md#4-depending-on-the-sdk
pip install -r requirements.txt
python main.py
```

Options in `.env`: `GROOT_CONTEXT` (default 262144) and `GROOT_MTP` (1 = on).

## What it does

`main.py` declares a `vllm` workload pinned to your worker
(`ExecutionPolicy.FIXED`) with this command (the worker adds host and port):

```
vllm serve GrootTeam/Groot-19B-A3B-261023-FP8 --max-model-len 262144 \
  --gpu-memory-utilization 0.92 --max-num-seqs 8 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 2, "video": 0}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}'
```

and `vllm_version="0.32.0"`. vLLM 0.32 logs ``method `qwen3_next_mtp` is
deprecated and replaced with mtp`` and runs it as `mtp`: same thing.

## Things to know

1. **It thinks before every answer.** "Capital of Australia, one word" used
   about 115 tokens, a 300-word essay 2,400 to 7,200. Give it `max_tokens` of
   8,192 or more; stream if you want to see `reasoning_content`.
2. **Thinking cannot be switched off per request through the endpoint.** The
   card does it with `chat_template_kwargs={"enable_thinking": false}`; the
   InferenceKey endpoint does not forward `chat_template_kwargs` (nor `top_k`
   and `min_p`), so the request still thinks. To serve it without thinking, add
   `--default-chat-template-kwargs '{"enable_thinking": false}'` to the command.
   `top_k=20` comes from the model's `generation_config.json` anyway.
3. **The first start is slow** (10 minutes on a cold worker), mostly vLLM's
   warm-up with MTP. Later starts reuse the compile caches.
4. **The full window is at the edge of 32 GB.** A 205K-token prompt took the
   card to 32,016 of 32,607 MiB. Keep other processes off the GPU, or lower
   `GROOT_CONTEXT` to 131072.
5. **NVFP4 did not start in vLLM 0.32 on the RTX 5090**: vLLM reported no
   native FP4 support on this GPU and fell back to weight-only Marlin, then
   FlashInfer's JIT build of its sm_120 kernels failed
   (`Engine core initialization failed`). Use the FP8 build with vLLM.

## Cost & cleanup

A `fixed` private workload keeps the GPU reserved while it exists. `main.py`
deletes it at the end unless `IK_KEEP_ALIVE=1`; if you keep it, call
`ManagementClient.delete("groot-19b-a3b")` (or delete it in the panel) when done.

## Troubleshooting

- `ValueError: To serve at least one request with the model's max seq len ...`
  at start: the context does not fit the KV memory left on your card; lower
  `GROOT_CONTEXT` (the error prints the largest length that fits).
- 403 `endpoint_forbidden`: add `<project>/groot-19b-a3b` (or `<project>/*`)
  to the API key's `allowed_endpoints`.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

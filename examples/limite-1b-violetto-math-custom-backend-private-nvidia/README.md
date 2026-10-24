# limite-1b-violetto-math-custom-backend-private-nvidia

Paradigma's **Limite 1B - Violetto** (Apache-2.0), a 1-billion-parameter dense
model trained from scratch for competition math, on your own NVIDIA GPU. The
example deploys it, sends **AIME 2026** problems all at once and checks each
`\boxed{}` answer against the official one.

| | |
| --- | --- |
| Model | [`paradigma-inc/limite-1b-violetto`](https://huggingface.co/paradigma-inc/limite-1b-violetto), BF16, one 2.1 GB safetensors file, 131,072-token context |
| Backend | **custom backend** `limite-1b-violetto` (`backend.py`): vLLM 0.26.0 plus Paradigma's [Limite vLLM plugin](https://github.com/paradigma-inc/limite-violetto) (pinned commit `7ddad34f`), served with `VLLM_PLUGINS=limite` |
| API | OpenAI `/v1/chat/completions` with token streaming (the backend implements `chat()` / `chat_stream()`) |
| Placement | Private, `fixed` policy, pinned to `IK_WORKER_ID` |
| Needs | NVIDIA driver for CUDA 13.0 (580+), Python 3.12 on the worker, about 6 GB of free VRAM (see below) |

## Why a custom backend

Violetto is a new architecture (`LimiteForCausalLM`) that stock vLLM does not
know. Paradigma ships an out-of-tree vLLM plugin for it, and the built-in
`vllm` backend installs vLLM only. So `requirements.txt` installs vLLM,
Transformers 5.6.2, `ninja` and the plugin from GitHub, and `setup()` starts
`vllm serve` with `VLLM_PLUGINS=limite` on a loopback port. `chat()` forwards
requests to it. Streaming matters: a hard problem can take tens of thousands of
tokens (our longest answer was about 101,000 tokens and took 16 minutes), far
beyond any non-streamed timeout.

The chat template only accepts the model's own fixed system prompt, so the
backend drops any other system message. Ask one problem per request, without
history.

## What we measured (InferenceKey deploy test, 2026-10-24)

| | RTX 4090 24 GB | RTX 2000 Ada 16 GB |
| --- | --- | --- |
| Ready after `ensure()` | 31 s with warm caches; first start on that pod: install 23 s, weights download + load 16 s, `torch.compile` 53 s | **300 s** on a fresh pod: install 54 s, load 13 s, `torch.compile` 103 s |
| Decode, one request | about 211 tokens/s (2,913 tokens in 13.8 s) | about 85 tokens/s |
| vLLM KV cache at `gpu_memory_utilization 0.90` | 18.75 GiB, 1,465,844 tokens | 11.64 GiB, 909,801 tokens |

- **All 30 AIME 2026 problems at once** on the RTX 4090 (one try each,
  temperature 0.6, budget 126,976 tokens): 25 of the 26 answers that finished
  were right. Four were lost to a bug in our client script and rerun: 2 right,
  1 wrong, and 1 whose stream ended after about 111,000 tokens without an
  answer. In total **27 right, 2 wrong, 1 with no answer**. vLLM peaked at about 3,600 generated tokens/s with 30
  requests running; answers ranged from about 3,000 to 101,000 tokens, so the
  whole batch took 16 minutes, set by the longest answer. Paradigma reports
  94.01 % on AIME 2026 averaged over 32 tries per problem; one try per problem
  is a smaller, noisier test.
- **Smaller cards.** With vLLM capped at 0.25 and 0.17 of the 4090
  (`LIMITE_GPU_MEMORY_UTILIZATION`, about 6 and 4 GB), the same problems came
  back right at the same per-request speed; the card showed 7.1 GB and 5.2 GB
  in use (the CUDA context sits outside the cap). At 0.12 (about 3 GB) vLLM
  refuses to start: a full 131,072-token sequence needs 1.68 GiB of KV cache.
  Those caps were emulated on the 4090, not real cards.

## Known issue: SDK runtime with chat support

`chat()` support in the custom-backend runtime is newer than the `inferencekey`
package on PyPI (0.260725.739). With that runtime the worker gets 404 on
`/v1/chat/completions`. The worker must run an SDK runtime from this
repository's `bindings/python` (main) until a new release is published.

## Run it

```bash
cp .env.example .env          # fill in real values
python -m venv .venv && . .venv/bin/activate
pip install maturin requests pyarrow
( cd ../../bindings/python && maturin develop --release )
python publish.py             # once: packages backend.py and uploads it to your tenant
python main.py                # deploy, then AIME 2026 problem 1
python main.py --problems all # the 30 problems at once
```

`main.py` downloads the problems from
[`MathArena/aime_2026`](https://huggingface.co/datasets/MathArena/aime_2026)
(CC BY-NC-SA 4.0) at the revision Paradigma's evaluator pins, and writes
answers, tokens and timings to `outputs/aime26.json`. Set `IK_KEEP_ALIVE=1` to
keep the workload afterwards; otherwise it is deleted at the end.

Each new assignment starts with `max_in_flight` 1, so problems run one at a
time until you raise the assignment's concurrency (we used 32) in the panel or
with `PATCH /api/projects/<project>/assignments/<workload>/<worker>`.

# Nemotron 3.5 Bolt 74B-A9B on a private NVIDIA GPU (llama.cpp + DFlash)

NVIDIA's **Nemotron 3.5 Bolt 74B-A9B 24GB** is a vision-language MoE (74B
parameters, 9B active per token) compressed from Nemotron 3.5 Super VL and
shipped natively in 2-bit, so the main GGUF is **22.0 GB**. NVIDIA publishes an
official llama.cpp GGUF, a 1.73 GB vision projector and a separate 1.0 GB
**LiLiCorr DFlash drafter** for speculative decoding. Licence: OpenMDW-1.1.

This example deploys it with the built-in **`llamacpp`** backend, using
NVIDIA's "28GB recipe" from the model card (98,304 tokens of context, q8_0 KV
cache, flash attention, up to 6 draft tokens), and then calls it three ways:

1. a reasoning question through the SDK's `generate_text()`;
2. an image question (a receipt drawn by the script) through the
   OpenAI-compatible endpoint;
3. a two-tool agent loop (`get_order`, `issue_refund`) through the same
   endpoint.

| File | What it does |
| --- | --- |
| `main.py` | `ensure()` a fixed workload on your worker, wait until ready, run the three calls, save `outputs/answers.json` and `outputs/receipt.png`, `delete()` (unless `IK_KEEP_ALIVE=1`). |
| `.env.example` | The variables `main.py` reads. Copy to `.env`. |
| `requirements.txt` | Client side: the local SDK build, `requests`, `pillow`. |

## Before you run it: the worker's llama.cpp version

Nemotron 3.5 support landed in llama.cpp on 2026-10-15
([ggml-org/llama.cpp#30520](https://github.com/ggml-org/llama.cpp/pull/30520)),
first in tag **`v0.7.0`**. On NVIDIA Linux the InferenceKey worker builds
`llama-server` from source with CUDA at a pinned tag, and the default pin is
older. Set this in the **worker's** environment (not in the workload) and
restart the worker:

```bash
INFERENCEKEY_LLAMACPP_TAG=v0.7.0
CMAKE_BUILD_PARALLEL_LEVEL=<number of CPU cores>   # the build is serial otherwise
INFERENCEKEY_LLAMACPP_READY_TIMEOUT_SECS=3000      # first start = build + 24.7 GB download
```

The build needs the CUDA toolkit (`nvcc`) in the image; the RunPod
`runpod/pytorch` CUDA images have it under `/usr/local/cuda/bin`.

## Run it

```bash
cp .env.example .env    # fill in your tokens, project and worker id
pip install -r requirements.txt   # plus the local SDK build, see ../CONTRIBUTING.md
python main.py          # IK_KEEP_ALIVE=1 to keep the workload afterwards
```

## What we measured (2026-10-20)

One RunPod **NVIDIA A40 48 GB** (secure cloud, 0.59 USD/h, 9 vCPUs, driver
595.91, CUDA 12.8 image), llama.cpp `v0.7.0`, the command in `main.py`:

- **Ready 764.8 s after `ensure()`** on a fresh pod. Most of it is the
  worker compiling llama.cpp with CUDA (705 s on 9 vCPUs). `llama-server` then
  downloaded the three GGUF files (24.7 GB) in about 44 s and loaded them in
  about 7 s. A second workload on the same worker, with the binary and the
  weights cached, was ready in 15.6 s.
- **VRAM**: 25,816 MiB for the server with the drafter (25,825 MiB peak on the
  card); 23,516 MiB without the drafter. Both fit a 32 GB card.
- **Decode speed, one request**, `llama-server`'s own timing: median
  **136 tokens/s with the DFlash drafter** (25 requests, 110 to 162) against
  **70 tokens/s without it** (63 to 71). Draft acceptance 70 % overall.
- **Prompt reading**: 24,944 tokens at 1,265 tokens/s with the drafter, 1,667
  without it; 70,773 tokens at 1,216 tokens/s (with the drafter). A code hidden
  in the middle came back right both times.
- End to end through the endpoint: the reasoning question in 3.9 s (265
  tokens), the receipt in 4.4 s, the two-tool refund loop in 7.0 s.

Answers from this run (`main.py`):

- Robot arithmetic: "The robot travels a total of 14 metres ... so the whole
  trip uses 28% battery." Right.
- Receipt: all four items and prices read correctly, total 15.20 confirmed.
- Agent: called `get_order`, then `issue_refund` with 9.00 EUR for the two
  broken 4.50 EUR bulbs, and left the lamp alone.

## Things to know

- The model thinks before it answers. The reasoning tokens count in
  `usage.completion_tokens` and against `max_tokens`, but non-streaming
  responses only carry the final answer, so give `max_tokens` room (the
  example uses 8,192).
- `--parallel 1`: one request at a time, as in NVIDIA's recipe. Raise it (and
  the assignment concurrency) only if the card has memory to spare.
- The drafter speeds up decoding but makes prompt reading slower and costs
  about 2.3 GB of VRAM. For long-document work with short answers, try the
  workload without the three `--spec-*` flags.

# Recursive Flint on a private NVIDIA GPU (llama.cpp, NVFP4)

Recursive's **Flint** is a mixture-of-experts model built for coding agents:
39.8B parameters, about 2.5B active per token, 262,144 tokens of context,
trained from scratch and released under **Apache-2.0**. The authors publish an
official **NVFP4 GGUF of 22.9 GB** for llama.cpp
([RecursiveSI/flint-nvfp4-gguf](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf)).

This example deploys it with the built-in **`llamacpp`** backend, using the
model card's `llama-server` command with the context cut to **65,536 tokens**
(F16 KV cache, flash attention, `--jinja`), and then calls it two ways:

1. a coding question through the SDK's `generate_text()`;
2. a small **coding-agent loop** through the OpenAI-compatible endpoint: the
   script writes a throwaway repo with one bug, gives the model four tools
   (`list_files`, `read_file`, `write_file`, `run_tests`) and asks it to make
   the failing tests pass. The tools are confined to that temp folder and the
   only command they run is `pytest`.

| File | What it does |
| --- | --- |
| `main.py` | `ensure()` a fixed workload on your worker, wait until ready, run the two calls, save `outputs/answers.json`, `delete()` (unless `IK_KEEP_ALIVE=1`). |
| `.env.example` | The variables `main.py` reads. Copy to `.env`. |
| `requirements.txt` | Client side: the local SDK build, `requests`, `pytest`. |

## Before you run it: the worker's llama.cpp build

Flint support landed in llama.cpp on 2026-10-21
([ggml-org/llama.cpp#30928](https://github.com/ggml-org/llama.cpp/pull/30928),
merge commit `a37cc461`; the chat template also needs #30929, merged the same
day). On NVIDIA Linux the InferenceKey worker builds `llama-server` from source
with CUDA, at the git tag or branch named by `INFERENCEKEY_LLAMACPP_TAG`, and
the default pin is older. When we tested, no release tag contained the PR yet
(`v0.7.0` was 134 commits behind it), so use the first tag published after
2026-10-21, or `master`. Set this in the **worker's** environment (not in the
workload) and restart the worker:

```bash
INFERENCEKEY_LLAMACPP_TAG=<a tag that contains a37cc461, or master>
CMAKE_BUILD_PARALLEL_LEVEL=<number of CPU cores>   # the build is serial otherwise
INFERENCEKEY_LLAMACPP_READY_TIMEOUT_SECS=3000      # first start = build + 22.9 GB download
```

The build needs the CUDA toolkit (`nvcc`, 12.8 or newer for Blackwell) in the
image; the RunPod `runpod/pytorch` CUDA 13 images have it under
`/usr/local/cuda/bin`. For an RTX 50 series card llama.cpp compiles the
`120a` target by itself (native FP4 instructions).

## Run it

```bash
cp .env.example .env    # fill in your tokens, project and worker id
pip install -r requirements.txt   # plus the local SDK build, see ../CONTRIBUTING.md
python main.py          # IK_KEEP_ALIVE=1 to keep the workload afterwards
```

## What we measured (2026-10-22)

llama.cpp at commit `a37cc461`, the NVFP4 GGUF, one request at a time.

| Card | Context | VRAM (llama-server) | Decode, tokens/s | Result |
| --- | --- | --- | --- | --- |
| RTX 5090 32 GB | 65,536 | 23,872 MiB | 194 to 216 (llama-bench tg128: 240) | works |
| RTX 4090 24 GB (23,028 MiB usable) | 16,384 | 22,316 MiB | 154 to 170 (llama-bench tg128: 181) | works |
| RTX 4090 24 GB | 32,768 | n/a | n/a | out of memory allocating the KV cache at load |

- **Time to ready**: on a worker that already had llama.cpp built and the GGUF
  cached, 13.5 s after `ensure()`. On a fresh pod the 22.9 GB download took
  about 113 s at roughly 200 MB/s and the load 5 to 6 s; building llama.cpp
  with CUDA took 6 min 26 s on the RTX 4090 pod and 7 min 13 s on the RTX 5090
  pod, with parallel build jobs.
- **Prompt reading**: 7,352 tokens/s for a 17,557-token prompt on the 5090,
  6,230 tokens/s for 8,807 tokens on the 4090.
- **This example on the 5090**: the coding answer in 3.0 s (374 tokens); the
  agent loop in 5.9 s, five turns: list files, run the tests (2 failed), read
  both files, rewrite one line, run the tests again (3 passed), then explain the
  bug. On the 4090 the same loop took 5.7 and 6.3 s in two runs.
- **One slow host**: our first RTX 5090 pod (power limit 400 W, driver
  580.178) decoded at only 27 to 29 tokens/s with this same file and build, and
  29 with the official int4 Q4_1 GGUF, with the GPU at 100 % and its memory
  almost idle. A second RTX 5090 (575 W, driver 595.91) did 240 in llama-bench.
  If your numbers look like the first ones, check the card before the model.
- **Less than 24 GB**: keeping part of the experts in system RAM
  (`--n-cpu-moe 18`, 16,116 MiB on the GPU) dropped decoding to 4.6 tokens/s, and
  `--n-cpu-moe 22` (14,388 MiB) to about 5.9, too slow for an interactive agent
  and past the endpoint's 120 s synchronous timeout for a 500-token answer
  (memory cap emulated on the 5090, not a real 16 GB card).

## Things to know

- The model reasons before it answers, and in this build the reasoning often
  comes back inside `content`. The reasoning tokens count in
  `usage.completion_tokens` and against `max_tokens`, so give it room (the
  example uses 16,384). The authors ask to keep `reasoning_content` in the
  history when it is returned; the agent loop does that.
- `-np 1`: one request at a time. Raise it (and the assignment concurrency)
  only if the card has memory to spare.
- Sampling follows the authors: temperature 1.0, top-p 0.95, top-k 20.
  Answers vary from run to run: in one of our long-prompt tests the model found
  the hidden string but got a small arithmetic question wrong.

# Ripple-65B Preview on a private 48 GB NVIDIA GPU (llama.cpp, Q4_K_M)

**Ripple-65B Preview** is the first model from **Ricursive Intelligence**: a
**dense 65B** transformer trained from scratch on 12T tokens, post-trained for
agentic coding, with up to **1M tokens** of context and an **Apache-2.0**
license. It is thin and deep: 80 layers, of which 64 use sliding-window
attention (1,024 tokens) and 16 use full attention without positional encoding,
8 KV heads, gated attention and value residuals. Its authors report **80.5 on
Terminal-Bench 2.1**, ahead of the much larger US open-weight MoE models in
their table (their numbers, not reproduced here).

At BF16 the weights are 129.6 GB. This example runs the 4-bit **Q4_K_M** GGUF
from [AtomicChat/Ripple-65B-Preview-GGUF](https://huggingface.co/AtomicChat/Ripple-65B-Preview-GGUF)
(38.8 GB) with every layer on **one 48 GB card**, with the built-in
**`llamacpp`** backend on llama.cpp **b12167**, the first release with the
`ricursive` architecture ([PR #31692](https://github.com/ggml-org/llama.cpp/pull/31692)).
Then it:

1. asks for a small coding task through the SDK's `generate_text()`;
2. runs one agent step with the tool from the model's own chat template
   (`execute_bash`): the model calls it, a canned result goes back, the model
   answers. Everything is saved to `outputs/answers.json`.

| File | What it does |
| --- | --- |
| `main.py` | `ensure()` a fixed workload on your worker, wait until ready, run the two calls, save `outputs/`, `delete()` (unless `IK_KEEP_ALIVE=1`). |
| `.env.example` | The variables `main.py` reads. Copy to `.env`. |
| `requirements.txt` | Client side: the local SDK build and `requests`. |

## 1. Run it

```bash
cp .env.example .env    # fill in your tokens, project and worker id
pip install -r requirements.txt   # plus the local SDK build, see ../CONTRIBUTING.md
python main.py          # IK_KEEP_ALIVE=1 to keep the workload afterwards
```

The command pins `INFERENCEKEY_LLAMACPP_ASSET=ubuntu-cuda-13.4-x64`: the worker
installs the prebuilt CUDA build from the llama.cpp release (169 MB, about a
second) instead of compiling llama.cpp from source, its default on NVIDIA. The
prebuilt build needs a driver for CUDA 13 and the CUDA 13 runtime libraries
(`libcudart.so.13`, `libcublas.so.13`) on the worker, as in the PyTorch CUDA 13
images. Without them, set `RIPPLE_LLAMACPP_ASSET=` (empty) to build from source.
Keep the Hugging Face cache on the worker's local disk, not on a network mount.

### Context and smaller cards

Only the 16 global layers keep a cache for the whole context (the other 64
keep 1,024 tokens), so the cache is about **64 KiB per token at f16**: long
context is cheap on this model.

| Variable | Default | Notes |
| --- | --- | --- |
| `RIPPLE_CONTEXT` | `32768` | On 48 GB with Q4_K_M: `131072` with `q8_0`, `262144` with `q4_0` (both measured) |
| `RIPPLE_KV_TYPE` | `f16` | `q8_0` halves the cache, `q4_0` quarters it |
| `RIPPLE_GGUF_FILE` | `Ripple-65B-Preview-Q4_K_M.gguf` | `...-AD-IQ3_XXS.gguf` (28.0 GB) for **32 GB**, `...-AD-IQ2_XXS.gguf` (20.0 GB) for **24 GB**, each with `RIPPLE_KV_TYPE=q8_0` |

## What we measured (2026-11-08)

Through a local InferenceKey manager, llama.cpp b12167 (prebuilt CUDA 13.4),
temperature 0.6, top_p 0.95 (Ricursive publishes no default sampling).

| GPU | File, context, cache | Peak VRAM | Decode, one request | Checks |
| --- | --- | --- | --- | --- |
| RTX A6000 48 GB (real) | Q4_K_M, 32K, f16 | 40,593 MiB | 14 to 16 tok/s | ready in 19 s with the file cached; coding answers right; sandboxed bug fix with `execute_bash` in 6 turns, 35 s; hidden code found in a 25,930-token prompt |
| RTX A6000 48 GB (real) | Q4_K_M, 128K, q8_0 | 42,373 MiB | 13.5 to 14.3 tok/s | hidden code found in a 118,836-token prompt (292 s, 9.1 tok/s at that depth) |
| RTX A6000 48 GB (real) | Q4_K_M, 256K, q4_0 | 43,440 MiB | 13.5 to 14.3 tok/s | hidden code found in a 237,532-token prompt (760 s, 6.8 tok/s at that depth) |
| 32 GB budget (on the A6000) | AD-IQ3_XXS, 32K, q8_0 | 28,395 MiB | 16.4 to 17.8 tok/s | bug fix in 5 turns; one of two coding answers had a wrong test |
| 24 GB budget (on the A6000) | AD-IQ2_XXS, 16K, q8_0 | 20,249 MiB | 20.6 to 21.9 tok/s | bug fix in 6 turns; one coding answer did not sort its input |
| **RTX 3090 24 GB (real)** | AD-IQ2_XXS, 16K, q8_0 | 20,901 MiB | 23 to 31 tok/s | ready in 19 s; bug fix in 7 turns, 32 s; one of three coding answers had a wrong test |

The 4-bit file got every coding answer right (4 of 4). The 2-bit and 3-bit
files are faster per token (they read fewer bytes per token) and still finished
the agent task every time, but they slipped in the tests they wrote. A 16 GB
card does not hold any file fully (the smallest, AD-IQ1_M, is 18.5 GB): not tested.
Not tested either: the authors' NVFP4 checkpoint with the DFlash2 drafter
(their serving path, SGLang on Blackwell or datacenter GPUs) and their benchmarks.

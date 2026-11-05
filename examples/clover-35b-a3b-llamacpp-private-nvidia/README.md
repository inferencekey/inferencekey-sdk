# Clover-35B-A3B-Preview on a private 24 GB NVIDIA GPU (llama.cpp, Q4_0)

**Clover-35B-A3B-Preview** is the first model from **DiffuSpace** (the team behind
Dream 7B): a **block-diffusion** language model derived from Qwen3.6-35B-A3B, a
35B mixture-of-experts with about **3B parameters active per token**, **128K
context** and **Apache-2.0**. With its authors' engines it fills blocks of 32
tokens in parallel instead of one token at a time.

DiffuSpace publishes official GGUF files
([Diffuspace/Clover-35B-A3B-Preview-GGUF](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview-GGUF));
the **Q4_0 is 19.6 GB** and fits a 24 GB card. llama.cpp merged Clover support on
2026-11-02 ([#31306](https://github.com/ggml-org/llama.cpp/pull/31306)); the first
release with it is **v0.9.0**.

This example deploys it with the built-in **`llamacpp`** backend, pinned to
llama.cpp v0.9.0 (`INFERENCEKEY_LLAMACPP_TAG=v0.9.0` in the command), with
16,384 tokens of context, and then:

1. asks the model card's own question through the SDK's `generate_text()`;
2. sends a compound drone control message with five orders and gets them back
   as a JSON array (the idea of the authors' UAV dispatch demo);
3. asks for a whole interactive desktop (dock, draggable windows, notes,
   calculator, clock) in one HTML file, the authors' GenUI showcase, and saves it
   to `outputs/desktop.html`.

| File | What it does |
| --- | --- |
| `main.py` | `ensure()` a fixed workload on your worker, wait until ready, run the three calls, save `outputs/`, `delete()` (unless `IK_KEEP_ALIVE=1`). |
| `patch_gguf.py` | Makes the official GGUF load correctly in upstream llama.cpp (see below). Run it once on the worker host. |
| `.env.example` | The variables `main.py` reads. Copy to `.env`. |
| `requirements.txt` | Client side: the local SDK build and `requests`. |

## Read this first: what runs, and what does not

- **llama.cpp v0.9.0 runs Clover autoregressively (one token at a time).** The
  merged support is "AR-only"; the self-speculative / diffusion decoding the
  authors use for their speed numbers (680 tokens/s on a GB300, 515 on an H200,
  both BF16 on SGLang) is in their SGLang branch
  ([sgl-project/sglang#46927](https://github.com/sgl-project/sglang/pull/46927),
  open on 2026-11-05) and was split out of the llama.cpp change. The GGUF card's
  quick start (`llama-speculative-simple --spec-type draft-self`) does not work
  with upstream llama.cpp: there is no `draft-self`.
- **The official GGUF does not load as published.** It was written by
  DiffuSpace's own llama.cpp build, and upstream v0.9.0 expects three things to
  be different. As published, llama-server stops with
  `key not found in model: clover_moe.expert_shared_count`. Adding only that key,
  it loads and answers with garbage (600 times the letter `G` to "What is the
  capital of Australia?"). `patch_gguf.py` adds the key, drops the 120
  diffusion-only tensors the AR path never reads, and rewrites the 30 small
  `ssm_a` vectors from `A_log` to `-exp(A_log)` (what upstream's converter
  writes). Everything else is copied byte for byte, no re-quantization; it took
  17 s. After it: right answers.

## 1. Prepare the worker host (once)

On the machine that runs your InferenceKey worker:

```bash
pip install -U "huggingface_hub[hf_xet]" gguf numpy
hf download Diffuspace/Clover-35B-A3B-Preview-GGUF Clover-35B-A3B-Preview-Q4_0.gguf --local-dir /root/models
python patch_gguf.py /root/models/Clover-35B-A3B-Preview-Q4_0.gguf /root/models/Clover-35B-A3B-Preview-Q4_0-ar.gguf
```

Keep the weights on the local disk (not a network mount). `CLOVER_GGUF` points
`main.py` at another path.

On NVIDIA Linux the worker builds `llama-server` v0.9.0 with CUDA the first time
(4 min 50 s on a 32-core pod). If your image has `nvcc` under
`/usr/local/cuda/bin` but not on `PATH` (the PyTorch CUDA 13 images), the build
fails with `cmake configure failed`; set this in the **worker's** environment and
restart it:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc
PATH=/usr/local/cuda/bin:$PATH
CMAKE_BUILD_PARALLEL_LEVEL=<number of CPU cores>
```

## 2. Run it

```bash
cp .env.example .env    # fill in your tokens, project and worker id
pip install -r requirements.txt   # plus the local SDK build, see ../CONTRIBUTING.md
python main.py          # IK_KEEP_ALIVE=1 to keep the workload afterwards
```

`CLOVER_CONTEXT=131072` gives the full window; it fits a 24 GB card (below).

## What we measured (2026-11-05, RTX 4090 24 GB, 450 W)

llama.cpp v0.9.0, the patched Q4_0, temperature 0, through the InferenceKey
endpoint unless noted.

- **Time to ready**: 11.7 s after `ensure()` with llama.cpp already built and the
  patched file on disk (first build 4 min 50 s; the 19.6 GB download took 35 s
  on that pod).
- **Speed**: 202 to 211 tokens/s decode per request, first token in 0.13 to
  0.31 s. The model thinks first (short: 37 tokens for "Canberra"; a 400-word
  essay used 4,665 tokens in 22.7 s).
- **This example**: "What is diffusion?" 1.6 s; dispatch message 5 of 5 orders
  right as JSON in 2.1 to 2.3 s (393 to 403 tokens; 5 runs); desktop page 4,163
  tokens in 20.3 s, 11,880 characters. The page works (dock, windows that drag,
  7 x 6 = 42, live clock, notes); its calculator window is too short and the
  `=` key is half hidden. At temperature 0 it wrote the same page twice.
- **Tools**: "Is it warmer in Lisbon or in Oslo?" with a `get_weather` tool: two
  parallel calls with valid JSON arguments in 1.29 s.
- **Long context**: at `-c 131072`, a code hidden in an 89,367-token prompt was
  found in 13.5 s (prefill 7,363 tokens/s, then 143 tokens/s).

| Card or cap | Configuration | Peak VRAM | Decode, tokens/s | Result |
| --- | --- | --- | --- | --- |
| RTX 4090 24 GB | this example: 16K context | 19,572 MiB | 202 to 211 | works |
| RTX 4090 24 GB | full 128K context (`-c 131072`) | 21,924 MiB | 202 (short), 143 at 89K | works, needle found |
| 16 GB, emulated | 16K, `--n-cpu-moe 10` | 15,456 MiB | 102 | works, about half speed |
| 16 GB, emulated | 16K, `--n-cpu-moe 14` | 13,728 MiB | 83 | works |
| 12 GB, emulated | 16K, `--n-cpu-moe 22` | 10,270 MiB | 65 | works |
| 8 GB, emulated | 16K, `--cpu-moe` | 2,494 MiB | 43 | works |

"Works" = the same questions answered right (Canberra; 5 of 5 dispatch orders).
The emulated rows ran on the same 4090 with the experts of N layers in system RAM
(16 CPU threads); they measure the GPU memory a smaller card would need, not a
real 16, 12 or 8 GB card. We tried a real RTX 5080 16 GB twice and got a broken
host both times (`CUDA unknown error`), so no real 16 GB result.

## Notes

- `chat_template_kwargs` (for example `enable_thinking: false`) is not forwarded
  by the InferenceKey endpoint; the model always thinks a little first.
- The authors' generation defaults (`presence_penalty 0.9`, `repetition_penalty
  1.05`) are set as server defaults in the command; their diffusion-only length
  and think-end biases have no llama.cpp equivalent.
- Benchmarks in the authors' blog (73.6 SWE-bench Verified, 51.7 Terminal-Bench
  2.1) were measured with their own engine, not with this setup.

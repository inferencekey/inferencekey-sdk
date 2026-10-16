# Ternary Bonsai 2 27B on a private NVIDIA GPU (custom backend)

Prism ML's **Ternary Bonsai 2 27B** (Apache-2.0) is a ternary rebuild of
Qwen3.8-27B: every language weight is -1, 0 or +1 with one FP16 scale per 128
weights, so a 27B reasoning and vision model fits in a **5.95 GB** GGUF
(`PTQ1_0`, 1.75 bits per weight) or **7.21 GB** (`PQ2_0`, 2-bit slots).

Those two weight types only load in Prism ML's
[llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp). Stock llama.cpp,
and therefore the built-in `llamacpp` backend, rejects them as unknown types.
This example is a **custom backend** that brings the fork with it:

- `setup()` downloads the fork's prebuilt Linux CUDA 12.8 build
  (`prism-b10777-e3de715`, about 200 MB), the GGUF and the 0.63 GB vision
  projector in parallel, starts the fork's `llama-server` on a loopback port
  (`-ngl 99 -fa on -c 65536 --jinja`, the model card's sampling) and waits for
  `/health`.
- `process()` forwards each chat request (messages, images, sampling, and
  llama.cpp fields such as `reasoning_effort`, `top_k`, `min_p` from
  `extra_body`) and returns `{text, reasoning, finish_reason, usage, timings}`.
- The fork's tarball links `libcudart.so.12` and `libcublas.so.12` but does not
  ship them, so `requirements.txt` installs them as `nvidia-*-cu12` wheels.
  The backend runs on any image with a CUDA 12.8+ driver, whatever toolkit the
  image has.

| File | What it does |
| --- | --- |
| `backend.py` | The custom backend (`TernaryBonsai2Backend`, and `TernaryBonsai2PQ2Backend` for the PQ2_0 file). |
| `requirements.txt` | Backend dependencies, installed by the worker with uv. |
| `publish.py` | Packages `backend.py` (about 6 KB, no weights) and publishes it to your tenant. |
| `main.py` | `ensure()` a fixed workload on your worker, wait until ready, ask one reasoning question (and with `--image`, one image question), save `outputs/answer.json`, `delete()`. |

## Run it

```bash
cp .env.example .env            # fill in your tokens, project and worker id
python publish.py               # once; --variant pq2 for the PQ2_0 file
python main.py                  # IK_KEEP_ALIVE=1 to keep the workload afterwards
python main.py --image board.jpg --image-prompt "Read every price on this board."
```

The client side needs the local SDK build (see `../CONTRIBUTING.md`) and
`requests`.

## What we measured (2026-10-16)

One RunPod RTX 4090 24 GB (driver 570.158, CUDA 12.8), PTQ1_0, `-c 65536`:

- **Ready 54.6 s after `ensure()`** on a fresh pod: 16.6 s to build the backend
  venv (1.1 GB, mostly the CUDA wheels), 31.5 s to download the fork and
  6.6 GB of GGUF files, 4.1 s for `llama-server` to load.
- **About 104 tokens/s** generation for a single request through the
  InferenceKey endpoint (median of 12, 101 to 104). `llama-bench` on the same
  card: tg128 105.8 tokens/s and pp512 3,279 tokens/s for PTQ1_0; 94.5 and
  3,230 for PQ2_0.
- Four requests at once (assignment concurrency 8): 9.6 s wall, 68 to 78
  tokens/s each, about 159 tokens/s in total.
- A 33,840-token prompt was read at 2,843 tokens/s (12.5 s) and the code
  hidden in its middle came back right.
- **11,672 MiB of VRAM** (11.4 GiB) for the server with 65,536 tokens of context, 4 slots and
  the vision projector loaded.

## Things to know

- It thinks before it answers. `reasoning_effort` `medium` (the backend's
  default) keeps answers short; `xhigh` is the model's own default; `none`
  skips the reasoning. `high` returns HTTP 500 in this chat template, so the
  backend maps it to `xhigh`. Keep `max_tokens` large (16,384 default): the
  reasoning counts against it.
- The platform passes non-OpenAI fields in `extra_body`. With the OpenAI
  Python client, `extra_body=` merges them into the top level of the JSON, so
  send the raw JSON shown in `main.py` instead.
- A custom backend answers `/v1/chat/completions` with its own JSON object as
  the message `content`; `main.py` decodes it. Streaming (`"stream": true`)
  is not supported for custom backends yet, and the response `usage` is zero:
  the real token counts are inside the decoded object.
- A workload pinned with `ensure()` starts with concurrency 1. Raise it on the
  assignment (Workloads page, "Conc.") so `llama-server` can batch requests.

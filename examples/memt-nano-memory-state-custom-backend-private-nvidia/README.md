# memt-nano-memory-state-custom-backend-private-nvidia

MemTensor's **MemT-1.0-Nano** (Apache-2.0) on your own NVIDIA GPU: a 36B
mixture-of-experts model (3B active, Qwen3.6-35B-A3B backbone) that writes past
turns into a **fixed-size memory state** instead of keeping the conversation as
context. The example tells it some facts, saves that state to a file, closes
the session, loads the file into a new session and asks about the facts.

| | |
| --- | --- |
| Model | [`MemTensor/MemT-1.0-Nano`](https://huggingface.co/MemTensor/MemT-1.0-Nano), bf16 as published, 70.3 GB of safetensors, pinned to revision `30fe0b22` |
| Backend | **custom backend** `memt-1-0-nano` (`backend.py`): Transformers 5.12.1 with `trust_remote_code`, PyTorch 2.11 (CUDA 13), driving the authors' `MemTSession` API |
| API | OpenAI `/v1/chat/completions`; the memory operation travels in `extra_body.memt` |
| Placement | Private, `fixed` policy, pinned to `IK_WORKER_ID` |
| Needs | NVIDIA driver for CUDA 13 (580+), an 80 GB card (we measured 68.1 GB peak), about 90 GB of local disk, Python 3.11 to 3.13 on the worker |

## Why a custom backend

The memory block and the session API (write turns into memory, save the state,
load it later) exist only in MemTensor's own modeling code on the Hub. No
built-in backend runs it, so `backend.py` loads the model with Transformers and
keeps named sessions in the backend process. Operations:

| op | what it does |
| --- | --- |
| `remember` | answers the turn, then commits the plaintext history into the memory state and drops the text (`enable_memory=True`) |
| `ask` | answers from the memory state plus the question only |
| `save` | writes the session's memory state to `<name>.safetensors` on the worker (returns size and sha256) |
| `close` | closes the session: memory and history are gone from the process |
| `load` | opens a new session from a saved state file |
| `status` | open sessions, saved states, load timings |

The state file stays on the worker's disk: at 41.9 MB it is larger than the
8 MB result limit of the platform, so the API returns its size and hash, not
its bytes.

## What we measured (InferenceKey deploy test, 2026-10-26, one A100-SXM4 80 GB)

| | |
| --- | --- |
| First start | venv about 23 s, 70.3 GB downloaded in 235 s, load 14.8 s |
| Warm start (`ensure()` to ready) | 20 s |
| GPU memory | 65.4 GiB allocated by PyTorch, 68.1 GB peak on the card |
| Memory state file | 41,944,024 bytes, the same after a 64-token chat and after a 1,467-token note |
| `remember` | 1,467-token note in 4.0 s; short facts in 1.4 to 5.7 s (most of it is the reply) |
| `save` / `load` | 0.7 to 1.0 s / 1.4 to 2.0 s |
| `ask` from a loaded state | 1.6 to 2.5 s for short answers; about 9 to 10 tokens/s decode |

- After loading the file into a new session, the model answered every question
  about the facts correctly: 4 of 4 for the short chat (bakery and city,
  supplier code, a sesame-free bread, daughter and school date) and 5 of 5 for
  five facts buried in the 1,467-token note. A fresh session without the file
  did not know the supplier code.
- **8-bit does not help.** With `MEMT_LOAD_IN_8BIT=1` (bitsandbytes) the model
  still takes 63.7 GiB, because the fused MoE expert weights are not linear
  layers and stay in bf16, and the first `remember` fails with
  `expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != signed char`.
  Capped to a 48 GB card (`MEMT_GPU_CAP_GIB=44`) it does not even load. So
  this example needs an 80 GB card.
- Decoding runs on PyTorch reference kernels for the linear-attention layers
  (no flash-linear-attention installed), hence the modest tokens/s.

## Run it

```bash
cp .env.example .env          # fill in real values
python -m venv .venv && . .venv/bin/activate
pip install maturin requests
( cd ../../bindings/python && maturin develop --release )
python publish.py             # once: packages backend.py and uploads it to your tenant
python main.py                # deploy, remember, save, close, load, ask
```

`main.py` writes every request, answer, token count and timing to
`outputs/memory-test.json`. Set `IK_KEEP_ALIVE=1` to keep the workload
afterwards; otherwise it is deleted at the end.

Changing the workload's `config.env` (for example `MEMT_LOAD_IN_8BIT`) on a
running custom-backend workload did not restart the backend in our test
(`reload_backend ... unknown backend`): delete the workload and run `main.py`
again.

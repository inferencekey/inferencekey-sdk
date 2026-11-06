# Nanbeige4.3-33B on a private 24 GB NVIDIA GPU (llama.cpp, Q4_K_M)

**Nanbeige4.3-33B** is the agent and coding model from the **Nanbeige team**
(Kanzhun, the company behind BOSS Zhipin): a **dense 33B** transformer on the
Llama architecture (52 layers, 8 KV heads), **512K context**, thinking on by
default, tool calls in an XML format, and **Apache-2.0**. Its authors call it the
teacher of the Nanbeige4.3 series and report it ahead of Qwen3.8-27B,
Muse-Glimmer-30B and Gemma4-31B on most of their agentic coding and "cowork"
benchmarks (80.9 on Terminal-Bench 2.1, 60.3 on SWE-Bench Pro; their numbers,
not reproduced here).

At BF16 the weights are 66 GB. This example runs the 4-bit **Q4_K_M** GGUF from
[bartowski/Nanbeige_Nanbeige4.3-33B-GGUF](https://huggingface.co/bartowski/Nanbeige_Nanbeige4.3-33B-GGUF)
(20.8 GB) with every layer on **one 24 GB card**, with the built-in
**`llamacpp`** backend pinned to llama.cpp **v0.9.0** (the GGUF was made with
b11845; `INFERENCEKEY_LLAMACPP_TAG=v0.9.0` in the command). Then it:

1. asks the model card's own question ("Which number is bigger, 9.11 or 9.8?")
   through the SDK's `generate_text()`;
2. makes one tool call with the tool from the GGUF card's prompt example
   (`get_stock_price`);
3. runs an agent turn: two parallel `get_weather` calls, the tool results go
   back, the model answers. Everything is saved to `outputs/answers.json`.

| File | What it does |
| --- | --- |
| `main.py` | `ensure()` a fixed workload on your worker, wait until ready, run the three calls, save `outputs/`, `delete()` (unless `IK_KEEP_ALIVE=1`). |
| `.env.example` | The variables `main.py` reads. Copy to `.env`. |
| `requirements.txt` | Client side: the local SDK build and `requests`. |

## 1. Run it

```bash
cp .env.example .env    # fill in your tokens, project and worker id
pip install -r requirements.txt   # plus the local SDK build, see ../CONTRIBUTING.md
python main.py          # IK_KEEP_ALIVE=1 to keep the workload afterwards
```

On NVIDIA Linux the worker builds `llama-server` v0.9.0 with CUDA the first time
(about 5 minutes on a 32-job build). If your image has `nvcc` under
`/usr/local/cuda/bin` but not on `PATH` (the PyTorch CUDA 13 images), set this in
the **worker's** environment and restart it:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc
PATH=/usr/local/cuda/bin:$PATH
CMAKE_BUILD_PARALLEL_LEVEL=<number of CPU cores>
```

Keep the Hugging Face cache on the worker's local disk, not on a network mount.

### Context and smaller cards

| Variable | Default | Notes |
| --- | --- | --- |
| `NANBEIGE_CONTEXT` | `8192` | 16384 fits 24 GB with the f16 cache; 32768 with `q8_0`; 65536 with `q4_0` |
| `NANBEIGE_KV_TYPE` | `f16` | `q8_0` halves the cache (about 104 KiB per token instead of 208), `q4_0` quarters it |
| `NANBEIGE_GGUF_FILE` | `..._Q4_K_M.gguf` | `Nanbeige_Nanbeige4.3-33B-IQ3_XXS.gguf` (14.5 GB) for a **16 GB** card, with `NANBEIGE_KV_TYPE=q8_0` |

## What we measured (2026-11-06)

llama.cpp v0.9.0, the card's sampling (0.6 / 0.95 for chat, 1.0 / 0.95 for
tools), through the InferenceKey endpoint unless noted.

- **Time to ready**: 303.7 s from `ensure()` on a fresh RTX 4090 worker, almost
  all of it the llama.cpp build; the 20.8 GB download took 15 s on that host,
  and loading the file takes about 4 s.
- **Speed (RTX 4090, Q4_K_M)**: 42 to 44 tokens/s decode per request, first
  token in 0.28 to 0.50 s on short prompts. "Canberra" took 40 tokens (1.2 s),
  the card's 9.11 vs 9.8 question 154 to 183 tokens (3.6 to 4.6 s, right every
  time), a `merge_intervals` function with tests 1,026 tokens in 24.6 s (the
  tests pass).
- **Tools**: `get_stock_price({"symbol":"AAPL"})` in 1.8 to 2.3 s; two parallel
  `get_weather` calls (Lisbon, Oslo) in 2.8 to 3.0 s, final answer in 3.1 s.
- **Long context (RTX 4090)**: a code hidden in the prompt was found at every
  size we tried: 14,909 tokens (f16 cache, 7.9 s), 30,734 tokens (`q8_0`,
  18.8 s), 61,428 tokens (`q4_0`, 49.1 s). Decode slows with depth: 37, 25 and
  16 tokens/s.

| Card or budget | Configuration | Peak VRAM | Decode, tokens/s | Result |
| --- | --- | --- | --- | --- |
| RTX 4090 24 GB | this example: Q4_K_M, 8K, f16 cache | 21,508 MiB | 42 to 44 | works |
| RTX 4090 24 GB | Q4_K_M, 16K, f16 cache | 23,186 MiB | 42 (37 at 15K) | works |
| RTX 4090 24 GB | Q4_K_M, 32K, `q8_0` cache | 23,434 MiB | 41 (25 at 31K) | works |
| RTX 4090 24 GB | Q4_K_M, 64K, `q4_0` cache | 23,802 MiB | 41 (16 at 61K) | works |
| 16 GB budget, on the 4090 | IQ3_XXS, 8K, `q8_0` cache | 14,730 MiB | 56 | works |
| 16 GB budget, on the 4090 | Q4_K_M, 36 of 52 layers on GPU | 14,413 MiB | 1.6 | works, unusable |
| 12 GB budget, on the 4090 | IQ2_XXS, 8K, `q8_0` cache | 10,670 MiB | 75 | answers right, one explanation wrong |
| **RTX 2000 Ada 16 GB (real card)** | IQ3_XXS, 8K, `q8_0` cache, through the endpoint | 14,392 MiB | 13.6 to 14.4 | works (same checks right) |

"Works" = the same checks answered right: Canberra, 9.11 vs 9.8, the stock tool
call, and a `merge_intervals` function whose own asserts pass. The budget rows
ran on the 4090 and measure the GPU memory a smaller card needs; the smaller
files are faster there because there is less to read per token. The real 16 GB
card is a 70 W workstation card with much less memory bandwidth, hence 14
tokens/s; on that host the first llama.cpp build took 19 min 50 s. At IQ2_XXS the
9.11 vs 9.8 answer was right but its explanation put the 8 "in the hundredths
place".

## Known issue: answers that land in the thinking channel

The chat template opens `<think>` for the model. On some answers (every 400-word
essay we asked for, 3 of 3, and the agent's final answer after the tool
results in 2 of the 3 runs of this example's flow) the model
skips thinking and never writes `</think>`, so llama.cpp's
`--reasoning-format deepseek` returns the whole answer as `reasoning_content`
and `content` is empty. `main.py` flags it (`answer_text()`).

Turning thinking off fixes it (the template then closes an empty think block):
`chat_template_kwargs: {"enable_thinking": false}` sent straight to llama-server
put the essay in `content`. The InferenceKey endpoint does not forward
`chat_template_kwargs` yet, so to do it through the endpoint add
`--chat-template-kwargs '{"enable_thinking":false}'` to the command (thinking
off for every request).

## Notes

- With no system message, the template inserts a default one in Chinese ("you
  are Nanbeige, a large language model developed by BOSS Zhipin"). Send your own
  system message if that matters to you.
- Benchmarks in the model card were measured by the authors with their own
  harnesses at the highest thinking effort, not with this setup.

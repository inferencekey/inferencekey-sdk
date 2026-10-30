# Sage-A1.5 research agent on a private NVIDIA GPU (llama.cpp, Q4_K_M)

**Sage-A1.5** from Shanghai AI Laboratory (InternScience) is an agentic research
model: a fine-tune of Qwen3.6-35B-A3B and the successor of Agents-A1. It is a
35B mixture-of-experts with about 3B parameters active per token, 262,144 tokens
of context, and **Apache-2.0**. It has two opt-in modes that you switch on from
the request side:

- **Agentic Orchestration**: register a `spawn_subagent` tool and the model fans
  independent sub-questions out to sub-agents that run in parallel, each with its
  own context and tools, then merges what they bring back.
- **Verification Mode**: one flag makes the model stop mid-task inside its own
  reasoning, list what is established and what failed, and change approach.

The authors publish an official **Q4_K_M GGUF of 21.7 GB** (plus a 0.9 GB vision
projector) for llama.cpp
([InternScience/Sage-A1.5-Q4_K_M-GGUF](https://huggingface.co/InternScience/Sage-A1.5-Q4_K_M-GGUF)).

This example deploys it with the built-in **`llamacpp`** backend, text only, with
**32,768 tokens** of context shared by 2 slots, and then:

1. asks a plain question through the SDK's `generate_text()`;
2. runs **Agentic Orchestration**: a research lead that only has
   `spawn_subagent`, and sub-agents with two tools that run on your machine,
   `wiki_search` and `read_page` (the public Wikipedia API, no key);
3. runs the same small research task with **Verification Mode off and on**.

| File | What it does |
| --- | --- |
| `main.py` | `ensure()` a fixed workload on your worker, wait until ready, run the three parts, save `outputs/answers.json`, `delete()` (unless `IK_KEEP_ALIVE=1`). |
| `.env.example` | The variables `main.py` reads. Copy to `.env`. |
| `requirements.txt` | Client side: the local SDK build and `requests`. |

## Three things to know first

1. **`--reasoning-format deepthink` does not exist.** The model card's
   `llama-server` command uses it, and llama-server exits with
   `Unknown reasoning format: deepthink` (the values are `none`, `deepseek`,
   `deepseek-legacy` and `auto`, also in the newest llama.cpp tag on
   2026-10-30). This example uses `deepseek`, which puts the thinking in
   `reasoning_content` and the answer in `content`.
2. **Verification Mode through the endpoint.** The card turns it on with
   `chat_template_kwargs={"verification": true}`. The InferenceKey endpoint does
   not forward `chat_template_kwargs` to the backend (nor `top_k`/`min_p`), so
   `main.py` appends the template's own line, `"\n\nVerification protocol:
   enabled"`, to the system message. We checked with llama-server's
   `/apply-template` that both give a byte-identical prompt, with and without
   tools. The sampling the card recommends is set as server defaults in the
   command (`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5`).
3. **It thinks a lot.** A two-sentence question used 1,905 tokens; with
   `max_tokens=4096` the same question once ran out inside the thinking and the
   reply was the thinking text. Give it room (`max_tokens` 8,192 to 16,384), and
   read `reasoning_content` from a streaming call if you want the self-review:
   the non-streaming endpoint returns `content` only.

## Before you run it: the worker

llama.cpp has supported this architecture (`qwen35moe`) for months; the worker's
default pin (`b9748`) loads it. On NVIDIA Linux the worker builds `llama-server`
with CUDA, which took 10 min 33 s on a 24-core pod. If your image has `nvcc`
under `/usr/local/cuda/bin` but not on `PATH` (the PyTorch CUDA 13
images), the build fails with `cmake configure failed`; set this in the
**worker's** environment and restart it:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc
PATH=/usr/local/cuda/bin:$PATH
CMAKE_BUILD_PARALLEL_LEVEL=<number of CPU cores>
```

Sub-agents run in parallel only if the assignment lets the worker take more than
one job at a time: raise the assignment's `max_in_flight` to 2 (the command's
`-np 2`). With 1 they queue.

## Run it

```bash
cp .env.example .env    # fill in your tokens, project and worker id
pip install -r requirements.txt   # plus the local SDK build, see ../CONTRIBUTING.md
python main.py          # IK_KEEP_ALIVE=1 to keep the workload afterwards
```

`SAGE_CONTEXT=65536` or `131072` gives more context; both fit a 24 GB card in our
test (see below).

## What we measured (2026-10-30, RTX 4090 24 GB)

llama.cpp `b9748`, the Q4_K_M GGUF, through the InferenceKey endpoint unless noted.
The card had a 203 W power limit (450 W is stock).

- **Time to ready**: 104.1 s after `ensure()` on a worker with llama.cpp already
  built (21.7 GB download plus load); a warm restart took 4 to 7 s.
- **Speed**: 135 to 137 tokens/s decode per request, first token in 0.22 to
  0.26 s, the same as calling llama-server directly (134 to 135). A 92,662-token
  prompt read at 2,987 tokens/s (31 s), then 99.6 tokens/s.
- **This example**: orchestration in 72.0 s with `max_in_flight` 1, 49.7 and
  55.7 s with 2; one `spawn_subagent` call with 3 sub-agents each time, right
  answer every time (hafnium 1923, technetium 1937, 14 years). Tool-call
  arguments were valid JSON in every call we made.
- **Verification Mode** on three easy fact-check questions, two runs each: all
  12 answers right with the mode off and on; on, the runs took 25.0 s instead of
  11.2 s, 6.5 tool calls instead of 4.0 and 1,890 tokens instead of 934. The
  authors report the gain on hard multi-hop search (BrowseComp 72.5 to 82.9 %),
  which we did not reproduce.

| Card or cap | Configuration | Peak VRAM | Decode, tokens/s | Result |
| --- | --- | --- | --- | --- |
| RTX 4090 24 GB | this example: 32K, 2 slots, text only | 21,760 MiB | 135 to 137 | works |
| RTX 4090 24 GB | model card: 64K + vision projector | 23,684 MiB | 138 | works, read a chart correctly |
| RTX 4090 24 GB | 128K, text only | 23,920 MiB | 137 to 139 | works, needle found at 92,662 tokens |
| 16 GB, emulated | 32K, `--n-cpu-moe 16` | 14,708 MiB | 24 to 25 | works, about 5.6x slower |
| 12 GB, emulated | 32K, `--n-cpu-moe 24` | 11,052 MiB | 13 to 18 | works, slower still |
| 8 GB, emulated | 32K, `--cpu-moe` | 3,048 MiB | 14 to 15 | 1 of 2 agent runs ran out of turns |

The emulated rows ran on the same 4090 with the experts of N layers in system RAM
(24 CPU threads, AMD EPYC 7402); they measure the GPU memory a smaller card would
need, not a real 16 or 12 GB card. Expert offload needs the rest of the weights in
system RAM (about 17 GB for `--cpu-moe`).

## Notes

- `--spec-type draft-mtp` with the same GGUF as draft model loads the whole
  21.7 GB file twice and runs out of memory on 24 GB; the built-in MTP layer was
  not usable this way with `b9748`.
- `wait_until_ready()` on a fixed workload that is already serving can wait
  until its timeout on some platform versions; if a second run stalls at
  "checking readiness", the endpoint is usually already answering.

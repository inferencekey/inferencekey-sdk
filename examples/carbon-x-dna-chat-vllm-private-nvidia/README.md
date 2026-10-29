# Carbon-X DNA chat on vLLM (private NVIDIA): raw DNA in an ordinary chat, answers in English

[Carbon-X](https://huggingface.co/HuggingFaceBio/Carbon-X) is Hugging Face's
Apache-2.0, 14.8B-parameter fine-tune of Qwen3-14B trained on DNA and text
together. You paste A/C/G/T straight into a chat message and it answers in
English: is this variant benign or pathogenic, is this region an enhancer or a
promoter, which organism group is this from, what are the next bases. Its own
tokenizer reads DNA as 6-mers (one token per 6 bases). This example serves it
with the stock `vllm` backend on one private NVIDIA GPU and asks it the
authors' own model-card questions. The axis it varies vs the other examples:
a chat model whose prompt carries a second "language", DNA, through custom
tokenizer code.

## Compatibility
- SDK: local build in this repo
- Language: Python >= 3.9
- Placement: private (NVIDIA, 48 GB in BF16 or 24 GB with `CARBON_FP8=1`;
  CUDA 13 driver, 580 or newer, for vLLM 0.32)
- Backend: `vllm` (pinned `0.32.0`)   Policy: `fixed`

## Prerequisites
- An SDK token (`ik_sdk_`), an API key (`ik_live_`) scoped to the workload or
  `your-project/*`, and your project slug ([tokens](../README.md#tokens)).
- A registered NVIDIA worker (`IK_WORKER_ID`):
  - **48 GB** (A40, RTX A6000, L40S, RTX 6000 Ada) for the BF16 weights
    (29.6 GB on disk, 27.6 GiB in GPU memory) with a 16K-token context;
  - **24 GB** (RTX 4090, RTX 3090, L4) with `CARBON_FP8=1`: vLLM quantizes the
    weights to FP8 while loading (16.0 GiB) and the context is 8,192 tokens.
- About 40 GB of free disk on the worker (weights plus the vLLM venv).

## Run
```bash
cp .env.example .env        # fill in real values
# install the local SDK, see ../CONTRIBUTING.md#4-depending-on-the-sdk:
python -m venv .venv && . .venv/bin/activate
pip install maturin
( cd ../../bindings/python && maturin develop --release )
set -a; . ./.env; set +a
python main.py                  # 48 GB card, BF16
CARBON_FP8=1 python main.py     # 24 GB card, FP8
```

## What it does
1. `ensure()` a `vllm` workload pinned to your worker with `vllm_version="0.32.0"` and
   `vllm serve HuggingFaceBio/Carbon-X --trust-remote-code --dtype bfloat16 --max-model-len 16384 --default-chat-template-kwargs '{"enable_thinking": false}'`.
   - `--trust-remote-code` loads the hybrid DNA tokenizer (`tokenizer.py` in the
     model repo). On first load it also fetches the Qwen3 BPE vocabulary it
     builds on, so the worker needs Hugging Face Hub access.
   - Carbon-X needs Transformers v5. vLLM 0.32.0 pins transformers 5.16 to
     5.19 (5.19.0 was installed), which is why the version is pinned.
   - Thinking is turned off on the server: the authors trained and evaluated
     the model that way, and the SDK's `generate_text()` has no field for
     `chat_template_kwargs`.
   - vLLM applies the model's `generation_config.json` (temperature 0.6,
     top-p 0.95, top-k 20), the sampling the authors used for every result.
2. `wait_until_ready()`: a cold worker installs vLLM and downloads the weights.
3. Asks two questions from the model card (the sickle-cell variant of the
   first 55 bases of human HBB, and "what does this sequence encode?"),
   prints the answers with timings and saves `outputs/answers.md`.

### Measured (2026-10-29)
| | RTX A6000 48 GB, BF16 | RTX 4090 24 GB, `CARBON_FP8=1` |
| --- | --- | --- |
| `ensure()` to ready, cold worker | 356 s (vLLM install 138 s, 30 GB download 22 s, load 29.5 s, engine start 136 s) | 262.5 s (install 106 s, download 20 s, load 27.9 s, engine start 89 s) |
| GPU memory | 27.62 GiB weights + 10.96 GiB KV cache (71,824 tokens); 40.6 GB on the card | 15.97 GiB weights + 3.18 GiB KV cache (20,800 tokens); 21.4 GB on the card |
| Decode, one request | 19.5 tokens/s | 50.5 tokens/s |
| A 4,096-base reference + 4,096-base variant (1,445 prompt tokens), about 120-token answer | 7.1 to 7.5 s | 2.4 to 2.6 s |
| 120 test questions, 8 at a time | 101 to 119 s | 46 s |

8,192 bases of DNA came to 1,445 prompt tokens: the 6-mer tokenizer packs
about 5.7 bases per token, wrapper text included.

On 40 variant pairs from the BioReason coding test set and 60 GenomeQA
questions (public data, 10 per sub-task) the BF16 model scored 77.5 % and
65.0 % (the authors report 74.9 % balanced accuracy and 63.6 % on the full
sets); the FP8 run on the 4090 scored 75.0 % and 66.7 %.

## Read the answer, not the explanation
The label is the useful part. In our runs the free-text reasoning was often
wrong even when the label was right: it named the wrong position or the wrong
base change ("C to C, so ATG becomes ACG" for a T-to-C change), and it called
the HBB start "PKM2". The answer also varies with sampling: asked 20 times, the
authors' sickle-cell example came back pathogenic 16 times and benign 4 times
(benign with temperature 0). For a decision, sample several times and count,
and compute positions and base changes in code (the authors give the same
advice). The model is not clinically validated.

DNA in answers comes back wrapped in literal `<dna>...</dna>` tags (they are
not stripped by `skip_special_tokens`): strip them before you parse.

## Cost & cleanup
A `fixed` workload keeps one replica on your worker until it is deleted.
`main.py` calls `delete()` on exit unless `IK_KEEP_ALIVE=1`. If you stop it
with Ctrl-C, run `python -c "from inferencekey import ManagementClient as M;
print(M.from_env().delete('carbon-x'))"`.

## Troubleshooting
- **Parallel requests queue one behind another.** A workload pinned to a
  worker through `ensure()` starts with a concurrency of 1 on that worker;
  raise `max_in_flight` on the assignment (we used 8) to let vLLM batch.
- **`Using a slow tokenizer` warning in the vLLM log.** Expected: the DNA
  tokenizer is Python code. It did not limit throughput in our runs.
- **16 GB cards.** vLLM 0.32 no longer accepts `--quantization bitsandbytes`,
  FP8 plus `--cpu-offload-gb` crashed (`Cannot copy out of meta tensor`), and
  BF16 with 16 GB offloaded to CPU loaded but produced 0.21 tokens/s. Use a
  24 GB card.
- **llama.cpp / GGUF** is not an option: the tokenizer is custom Python code
  those runtimes cannot load (model card).

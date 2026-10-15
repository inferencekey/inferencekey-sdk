# AstaBrief-8B cited reports on vLLM (private NVIDIA): question + your excerpts in, sectioned report with inline citations out

[AstaBrief-8B](https://huggingface.co/allenai/AstaBrief_8B) is Ai2's Apache-2.0
fine-tune of Qwen3-8B that turns a research question plus the excerpts YOUR
retrieval step found into a multi-section report with a short `TLDR` per section
and an inline `[ID]` citation behind each claim, in one pass. This example serves
it with the built-in `vllm` backend on one 24 GB NVIDIA GPU and calls it with the
exact prompt Ai2 fine-tuned it on. The axis it varies vs the other examples: a
chat model used as a report writer, fed with your own sources.

## Compatibility
- SDK: local build in this repo
- Language: Python >= 3.9
- Placement: private (NVIDIA, 24 GB or more; CUDA 13 driver for vLLM 0.20)
- Backend: `vllm` (0.20.0, the worker default)   Policy: `fixed`

## Prerequisites
- An SDK token (`ik_sdk_`), an API key (`ik_live_`) scoped to the workload or
  `your-project/*`, and your project slug ([tokens](../README.md#tokens)).
- A registered NVIDIA worker with 24 GB of VRAM or more (`IK_WORKER_ID`).
  The 16.4 GB of BF16 weights plus a 16K-token KV cache fit an RTX A5000,
  RTX 4090, L4 or larger.

## Run
```bash
cp .env.example .env        # fill in real values
# install the local SDK, see ../CONTRIBUTING.md#4-depending-on-the-sdk:
python -m venv .venv && . .venv/bin/activate
pip install maturin
( cd ../../bindings/python && maturin develop --release )
set -a; . ./.env; set +a
python main.py                       # the illustrative sample_references.json
python main.py my_references.json    # your own excerpts, same JSON shape
```

`my_references.json` is `{"query": "...", "references": [{"id", "author",
"year", "citations", "quote"}, ...]}`: one object per excerpt your retrieval
returned. Prefer plain numeric ids: with OpenAlex-style ids such as
`W4402683901` the model often dropped the `W` when citing.

## What it does
1. `ensure()` a `vllm` workload pinned to your worker, with
   `vllm serve allenai/AstaBrief_8B --max-model-len 16384 --dtype bfloat16`.
   vLLM downloads only the `*.safetensors` shards; the repo also ships duplicate
   `pytorch_model-*.bin` files that would double the 16.4 GB download.
2. `wait_until_ready()`: on a cold worker the first start installs vLLM and
   downloads the weights.
3. Builds the official prompt (downloaded at run time from Ai2's
   `AstaBrief_prompts` dataset, CC-BY-NC-4.0, pinned revision), sends it with
   the model card's settings (temperature 0.7, up to 4096 tokens), prints the
   report, counts sections and cited references, and saves `outputs/report.md`.

### Measured (RTX A5000 24 GB, RunPod secure cloud, 2026-10-15)
| Step | Time |
| --- | --- |
| `ensure()` to ready, cold pod | 245.7 s (vLLM install 137 s, weights download 18 s, engine start with torch.compile 40 s) |
| GPU memory | 21.9 GB used by vLLM (15.3 GiB weights, 4.6 GiB KV cache = 33,472 tokens) |
| First token, 4,400 to 5,100 prompt tokens (15 excerpts) | 1.2 to 1.6 s |
| Generation, one report | about 39 tokens/s; a 1,500 to 2,900-token report takes 40 to 85 s |
| Four reports at once (concurrency 8) | about 34 tokens/s each, about 133 tokens/s in total |
| `main.py` with the sample (6 excerpts) | 40.9 s, 1,171 prompt and 1,563 report tokens |

## Check the citations
In our 11 test reports, 363 of 380 inline citations pointed to supplied
excerpts, but 8 reports also cited 1 to 3 keys that were not in the prompt
(papers from the model's memory, or a supplied paper with a mangled id).
Before you show a report, check every `[ID | ...]` against the ids you sent;
`main.py` prints how many of yours were cited. The model also starts every
answer with an empty `<think>\n\n</think>` block and often writes `TLDR`
without the `;` the prompt asks for: strip and parse leniently.

## Cost & cleanup
A `fixed` workload keeps one replica on your worker until it is deleted.
`main.py` calls `delete()` on exit unless `IK_KEEP_ALIVE=1`. If you stop it
with Ctrl-C, run `python -c "from inferencekey import ManagementClient as M;
print(M.from_env().delete('astabrief-8b'))"`.

## Troubleshooting
- **Parallel requests queue one behind another.** A workload pinned to a
  worker through `ensure()` starts with a concurrency of 1 on that worker.
  Raise it (Workloads, the worker chip, `CONC.`) so vLLM can batch: 4 reports
  took 85 s at concurrency 8 instead of 209 s one after another. With
  concurrency 1, a streamed request that waits more than 180 s for its first
  token is cut by the gateway.
- **The install fails with a CUDA or driver error.** vLLM 0.20 brings
  torch 2.11 built for CUDA 13; the host driver must be 580 or newer.
- **Out of memory at start.** 24 GB is the floor at `--max-model-len 16384`;
  on 16 GB cards use a quantized variant or a shorter context.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

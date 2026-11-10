# Qwen3.5-0.8B with a weight edit on vLLM (private NVIDIA): the same model, minus its thinking loops

[RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)
is [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) with 131,072
weights changed (64 output columns of `down_proj` in layers 15 and 19, 0.016 %
of the model). A frontier LLM made the edit by reading the weights in a Python
session, with no gradient descent and no training
([the authors' post](https://rikiyat.github.io/blog/weight-space-programming/)).
The base model, with thinking on, tends to loop until it runs out of tokens;
the edited one mostly finishes and answers. Apache-2.0, same architecture,
same 1.7 GB of bf16 weights, same serving command.

This example serves the edited model with the stock `vllm` backend on one
private NVIDIA GPU and, with `IK_WITH_BASELINE=1`, the unedited base model on
the same card, then asks both the same MATH-500 problem with the authors'
sampling. The axis it varies vs the other examples: two workloads that differ
only in their weights, side by side on one GPU, to measure a weight edit.

## Compatibility
- SDK: local build in this repo
- Language: Python >= 3.9
- Placement: private (NVIDIA; 6 to 8 GB for the edited model alone with the
  full 34,816-token context (measured under memory caps), 24 GB for both side by side; CUDA 13 driver, 580 or
  newer, for vLLM 0.33)
- Backend: `vllm` (pinned `0.33.0`)   Policy: `fixed`

## Prerequisites
- An SDK token (`ik_sdk_`), an API key (`ik_live_`) scoped to
  `your-project/*`, and your project slug ([tokens](../README.md#tokens)).
- A registered NVIDIA worker (`IK_WORKER_ID`). We ran it on an RTX 4090
  24 GB; see "Which GPU" below for smaller cards.
- About 15 GB of free disk on the worker (the vLLM venv plus 1.7 GB per model).

## Run
```bash
cp .env.example .env        # fill in real values
# install the local SDK, see ../CONTRIBUTING.md#4-depending-on-the-sdk:
python -m venv .venv && . .venv/bin/activate
pip install maturin
( cd ../../bindings/python && maturin develop --release )
set -a; . ./.env; set +a
python main.py                       # edited model only
IK_WITH_BASELINE=1 python main.py    # edited + base on the same GPU
```

## What it does
1. `ensure()` one `vllm` workload per model, pinned to your worker, with
   `vllm_version="0.33.0"` and
   `vllm serve <model> --dtype bfloat16 --max-model-len 34816 --gpu-memory-utilization <0.90, or 0.45 each with the baseline> --default-chat-template-kwargs '{"enable_thinking": true}' --override-generation-config '{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}'`.
   - Qwen3.5-0.8B answers without thinking by default; the chat template
     kwargs turn thinking on for every request, as the authors evaluated it.
   - Neither repo ships a `generation_config.json`, so the server sets Qwen's
     thinking-mode sampling (vLLM 0.33 accepts `presence_penalty` there).
2. `wait_until_ready()`: a cold worker installs vLLM and downloads the weights.
3. Streams one MATH-500 problem (test/number_theory/572, "How many positive
   whole-number divisors does 196 have?", answer 9) from each endpoint with
   up to 32,768 new tokens, prints the boxed answer, tokens, time and whether
   it hit the limit, and saves `outputs/<slug>.json`. The call is plain HTTP
   to the OpenAI-compatible route so it can carry `presence_penalty`.

### Measured (2026-11-10, RTX 4090 24 GB, both models on the card)
| | |
| --- | --- |
| `ensure()` to ready, cold worker, both workloads | 418 s (vLLM 0.33.0 install about 3 min, weights 0.3 s, engine start with profiling and CUDA graphs about 3 min) |
| GPU memory | 1.63 GiB of weights per model; 17.3 GB on the card at the peak with both engines at 0.45 |
| One request alone | about 300 to 380 tokens/s decode |
| 50 MATH-500 problems per model, all 50 at once, both models at once | edited 987 s, base 1,194 s (2.6 million tokens in total) |

| 50 MATH-500 problems (the first 50 in the authors' order), 1 sample each | right | hit the 32,768-token limit | mean tokens |
| --- | --- | --- | --- |
| Edited | **31 (62 %)** | 10 (20 %) | 20,444 |
| Base | 4 (8 %) | 48 (96 %) | 31,832 |

Graded like the authors (last `\boxed{}` of the whole output, math-verify
0.9.0, their `mathqa.py`). On the same 50 problems the authors' own re-run
averaged 54 % for the edited model (4 samples each); on all 500 they report
51.0 % edited and 7.9 % base. The 50-problem batch talked to the two vLLM
servers the platform launched, from the worker itself; `main.py` goes
through the endpoint.

In the recorded run through the endpoint (both workloads at once, same
problem), the edited model answered 9 in 14,518 tokens and 87 s; the base
model hit the 32,768-token limit after 138 s, repeating
"Wait, 14 × 14 = 196".

### Which GPU
The edited model alone, same command with a lower `--gpu-memory-utilization`
(emulated memory caps on the 4090, five problems the edited model solved):

| Memory cap | KV cache | Result |
| --- | --- | --- |
| 8 GB (0.333) | 4.3 GiB, 343,186 tokens | 5 of 5 right, identical outputs to 6 GB |
| 6 GB (0.245) | 2.23 GiB, 177,561 tokens | 5 of 5 right, same speed (the longest, 31,606 tokens, in 96 s) |
| 5 GB (0.204) | 1.74 GiB, 138,766 tokens | with the default flags it does not start ("max_num_seqs (256) exceeds available Mamba cache blocks (202)"); with `--max-num-seqs 32`, 5 of 5 right, longest 30,948 tokens in 97 s |
| 4 GB (0.163) | 0.29 GiB | does not start: a 34,816-token request needs 0.44 GiB of KV cache (vLLM suggests a 22,304-token maximum) |
| 3 GB (0.122) | -0.66 GiB | does not start: no memory for the cache blocks |

These are memory caps on a 24 GB card, not real small cards (none smaller
than 16 GB was available to rent that day); a real 8 GB card also has less
compute and memory bandwidth, so expect it to be slower.

## Notes
- One sample per problem: the edited model's 62 % on 50 problems is within
  the noise of the authors' 54 % on the same problems.
- The edit stops loops; it does not teach new math. 20 % of the edited
  model's answers still ran to the limit, and the authors report that a model
  that does not loop (Gemma 3 270M) did not improve with the same method.
- The checkpoint is the full multimodal release; text prompts do not use the
  vision tower.

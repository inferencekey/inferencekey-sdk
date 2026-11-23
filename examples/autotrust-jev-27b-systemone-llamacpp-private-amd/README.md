# AutoTrust-JEV-27B on a private AMD R9700 32 GB (llama.cpp, System One decisions)

**AutoTrust-JEV-27B v1.1** ("Jev") by **AutoTrust AI Lab** (Hugging Face org
[`autotrust`](https://huggingface.co/autotrust), Apache-2.0) is a 27B **decision
model**: it does not write text. You give it a state and one or more questions
with options, and it returns a probability for every option, in one forward
pass. Its NOTICE says it builds on d3 (Decision 3.0, vLLM Semantic Router Team),
which is built on Qwen/Qwen3.8-27B (a hybrid of linear and full attention
layers). It reads the whole prompt bidirectionally (not causally), as it was
trained. The model card lists `choice` and `yes/no` questions, text only.

The authors serve it with vLLM and a bidirectional plugin that needs
FlashAttention (CUDA). This example serves it on an **AMD R9700 (gfx120x,
RDNA4, 32 GB)** instead, with the built-in **`llamacpp`** backend: llama.cpp
**b11815** has a native System One API (`/v1/systemone`) and runs the model
non-causally from a GGUF whose metadata says so. InferenceKey proxies the route:

```
POST {base}/endpoint/{project}/{workload}/v1/systemone
```

There was **no public GGUF of v1.1**, so we made our own **Q6_K** (step 1).

| File | What it does |
| --- | --- |
| `publish.py` | `ensure()` a fixed `llamacpp` workload on your AMD worker with the exact `llama-server` command, then wait until ready. `--dry-run` plans only; `--delete` tears it down. |
| `decide.py` | Sends one real driving decision from our Crazy Taxi test to `/v1/systemone` and prints the chosen option, the probabilities and the latency. Saves `outputs/decide.json`. |
| `autotrust_jev.py` | The llama.cpp converter module we wrote to turn the checkpoint into a GGUF (step 1). |
| `convert_retry.py` | Runs llama.cpp's converter with retries and timeouts on every Hugging Face read (step 1). |
| `.env.example` | The variables both scripts read. Copy to `.env`. |
| `requirements.txt` | Client side: the local SDK build (for `publish.py`) and `requests`. |

The model repo was renamed: `autotrust/AutoTrust-JEV-27B-v1.1` now redirects to
[`autotrust/AutoTrust-Decider-Pro-Vision-27B-v1.1`](https://huggingface.co/autotrust/AutoTrust-Decider-Pro-Vision-27B-v1.1).
Its current revision, `ae156900e9f338e423042b89e25d0fd8093064d4`, is the one
we converted, and the card still calls the model AutoTrust-JEV-27B v1.1.

## Compatibility
- SDK: local build in this repo (or `>= 0.1.0` once published)
- Language: Python >= 3.9
- Placement: private (AMD ROCm, R9700 / gfx120x / RDNA4, 32 GB)
- Backend: `llamacpp` (llama.cpp b11815, ROCm 10.0 build)   Policy: `fixed`
- Task type: `classification`

## Prerequisites
- The tokens and ids: `INFERENCEKEY_SDK_TOKEN` (`publish.py`),
  `INFERENCEKEY_API_KEY` (`decide.py`), `INFERENCEKEY_PROJECT` and
  `IK_WORKER_ID`. See [tokens & placement](../README.md#tokens).
- A registered AMD ROCm worker with an R9700 32 GB, assigned to the project.
- **VRAM:** about 22.4 GB for the model (below). It fits the R9700's 32 GB
  with room for a small second workload.
- **Disk:** 22.1 GB for the Q6_K GGUF on the worker host. Making it (step 1)
  needs about 50 GB free on the machine that converts (27 GB Q8_0 plus the Q6_K).

## Read this first

- **The prompt must be read whole, alone, in one pass.** The GGUF carries
  `qwen35.attention.causal = false`. llama.cpp then reads each prompt
  bidirectionally, which only matches the model if the whole prompt is in one
  micro-batch and is not shared with another request. Hence `-ub 2048` (longer
  than any prompt you send; ours are under 500 tokens), `--parallel 1`,
  `--ctx-checkpoints 0` and `--cache-ram 0` in the command. If your prompts can
  be longer than 2048 tokens, raise `-b`, `-ub` and `-c` together.
- **One request at a time.** With `--parallel 1` a second request waits.
- **A Q6_K of our own, not the authors' weights as served by them.** We did not
  compare its probabilities against the authors' bf16 vLLM setup.

## 1. Make the GGUF (once)

These are the steps we ran on 2026-10-22 on an x86 Linux VM, CPU only. Upstream
llama.cpp b11815 does not know the checkpoint (`decision_config.json` says
`"prompt": "jev"`), so `autotrust_jev.py` maps it onto llama.cpp's `vllm-d3`
decision model, the model Jev builds on.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout b11815        # we converted at 7170cad, the commit tagged b11815
cp ../autotrust_jev.py conversion/autotrust_jev.py
python3 - <<'EOF'          # register the module next to vllm_d3
p = "conversion/__init__.py"; s = open(p).read()
s = s.replace('    "VllmD3Model": "vllm_d3",\n',
              '    "VllmD3Model": "vllm_d3",\n    "AutoTrustJevModel": "autotrust_jev",\n', 1)
open(p, "w").write(s)
EOF
python3 -m venv ../venv
../venv/bin/pip install -r requirements/requirements-convert_hf_to_gguf.txt huggingface_hub

# Q8_0 first, streaming the safetensors from Hugging Face (no 54 GB download):
../venv/bin/python ../convert_retry.py convert_hf_to_gguf.py --remote autotrust/AutoTrust-JEV-27B-v1.1 \
  --outtype q8_0 --outfile AutoTrust-JEV-27B-v1.1-Q8_0.gguf

# then Q6_K with llama-quantize (a CPU build is enough):
../venv/bin/pip install cmake ninja
cmake -B build-cpu -G Ninja -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j8 --target llama-quantize
build-cpu/bin/llama-quantize --allow-requantize AutoTrust-JEV-27B-v1.1-Q8_0.gguf \
  AutoTrust-JEV-27B-v1.1-Q6_K.gguf Q6_K 12
```

What we saw:

- The converter warns that Transformers does not recognise `qwen3_5` and falls
  back to `config.json`, and logs `Unknown RoPE type: default`. Both were
  harmless here.
- `--remote` reads the weights with plain HTTP range requests and no retry:
  one dropped connection killed our first try. `convert_retry.py` adds retries
  and a (20 s, 60 s) timeout; our run needed 7 retries and took 43 min 52 s.
- `llama-quantize` took 155 s: `quant size = 21049.06 MiB (6.57 BPW)`, a file
  of 22,082,531,424 bytes, sha256
  `6f5d0830eeaa4b87b6388691a65cd460e527345a78f419940ea769c41e2d462c`.
- The GGUF metadata reads back `general.architecture = qwen35`,
  `qwen35.decision.type = vllm-d3`, `qwen35.attention.causal = False`, a
  `systemone` chat template and `general.license = apache-2.0`.

Not confirmed by us: a conversion from a local download instead of `--remote`;
whether Q6_K made straight from bf16 (instead of from Q8_0, as we did) gives
different probabilities; and the conversion under the repo's new name (we
converted before the rename; `--remote` has no revision flag, so check you get
revision `ae156900`).

## 2. Put it on the worker host

Copy the Q6_K to a local disk the worker container can read, and point
`JEV_GGUF` at the path **as the worker sees it**. On our worker the host's
Hugging Face cache volume is mounted at `/models/huggingface`, so the file lives
at `/models/huggingface/jevtaxi/AutoTrust-JEV-27B-v1.1-Q6_K.gguf` (the default).

## 3. Deploy it

```bash
cp .env.example .env        # fill in real values
# install the local SDK, see ../CONTRIBUTING.md#4-depending-on-the-sdk:
python -m venv .venv && . .venv/bin/activate
pip install maturin -r requirements.txt
( cd ../../bindings/python && maturin develop --release )
python publish.py --dry-run   # optional: log what ensure() would do, change nothing
python publish.py             # ensure() + wait_until_ready()
```

`INFERENCEKEY_LLAMACPP_TAG=b11815` and `INFERENCEKEY_LLAMACPP_ASSET=ubuntu-rocm-10.0-x64`
in the command make the worker install the prebuilt ROCm 10.0 build of that
llama.cpp release (`llama-b11815-bin-ubuntu-rocm-10.0-x64.tar.gz`) instead of
its default one.

## 4. Ask it a decision

```bash
python decide.py                 # the game's own request: 2 options
python decide.py --options all   # all 5 full-throttle steering options
python decide.py --runs 5        # repeat, then min / median / max latency
```

## The request

`POST {base}/endpoint/{project}/{workload}/v1/systemone` with the `ik_live_`
key as `Authorization: Bearer ...`:

```json
{"state": {"passenger_onboard": true, "route_arrow": "slightly_right",
           "route_arrow_angle_deg": 27, "speed_yd_s": 37, "destination_distance_yd": 439,
           "previous_action": "accelerate_straight", "...": "..."},
 "questions": {"action": {"type": "choice",
                          "instructions": "You drive a taxi in Crazy Taxi. Choose the best driving action ...",
                          "criteria": {"accelerate_slight_right": "Full throttle with a gentle correction to the right.",
                                       "accelerate_straight": "Full throttle, wheel centred."}}}}
```

`state` is any JSON object: the model treats it as data, not as instructions.
`questions` maps your own question names to a question each: `type` (`choice`),
`instructions`, and `criteria`, whose keys are the option names you get back
and whose values describe each option to the model. The answer:

```json
{"model": "/models/huggingface/jevtaxi/AutoTrust-JEV-27B-v1.1-Q6_K.gguf",
 "answers": {"action": {"type": "choice", "choice": "accelerate_slight_right",
                        "probabilities": {"accelerate_slight_right": 0.879906667497815,
                                          "accelerate_straight": 0.12009333250218505},
                        "confidence": 0.4703709128081277}},
 "usage": {"input_tokens": 453, "output_tokens": 0}}
```

Every question in a request is a separate prompt evaluation: with a
bidirectional model, questions cannot share a prefix. Put what you can in one
question with more options.

## What we measured

On our ik-node-1 worker (AMD Radeon AI PRO R9700 32 GB, ROCm), llama.cpp
b11815, the Q6_K above, through the public InferenceKey endpoint. Latency is
measured on the client, end to end (it includes the network to the endpoint).

**`decide.py`, 2026-11-23, 5 runs each:**

| Request | Answer | Probabilities | Input tokens | Latency (min / median / max) |
| --- | --- | --- | --- | --- |
| game's 2 options | `accelerate_slight_right` | 0.880 / 0.120 (`accelerate_straight`) | 453 | 0.716 / 0.719 / 0.806 s |
| all 5 options (`--options all`) | `accelerate_slight_right` | 0.871, then `accelerate_straight` 0.092, `accelerate_right` 0.033, `accelerate_slight_left` 0.004, `accelerate_left` 0.001 | 495 | 0.732 / 0.766 / 0.855 s |

- The same request gave the same probabilities on every call, and exactly the
  values the game logged for this state during play (0.879906667497815).
- **In a whole game** (2026-11-23, 10 minutes of Crazy Taxi, 1,512 decisions,
  197 of them sent to the model, the rest settled by fixed rules), the model
  answered in 0.686 to 0.888 s end to end, median 0.76 s.
- **GPU memory:** with the model loaded, `rocm-smi` showed 24,713,990,144 bytes
  in use on the card, which also held another workload of about 2.3 GB: about
  22.4 GB for Jev (2026-10-22).
- **Cold start:** about 19 minutes from the worker starting `llama-server` to
  ready on ik-node-1 (2026-10-22). We did not look into why it was that slow;
  the host has 5.8 GB of system RAM.
- On 2026-10-22, the same command started by hand on the worker answered a
  manual test in about 510 ms per decision (no proxy, no network); through the
  endpoint it was about 600 ms then.

## Cost & cleanup

A **private worker with a `fixed` policy**: one replica stays always on and
holds about 22 GB of the R9700 until the workload is deleted; it does not scale
to zero. `publish.py` does not delete it at the end, because `decide.py` needs
it. Tear it down with:

```bash
python publish.py --delete
```

## Troubleshooting

- **`KeyError: 'IK_WORKER_ID'` or `'INFERENCEKEY_PROJECT'`**: fill in `.env`
  from `.env.example`.
- **The workload never gets a worker:** the worker must be assigned to
  `INFERENCEKEY_PROJECT` (Manager UI, Workers). Our own workload was created by
  hand with an explicit worker assignment; we have not checked that `ensure()`
  alone creates that assignment.
- **Readiness times out:** the first start downloads the llama.cpp release and
  loads 22 GB (19 minutes on our host). `publish.py` waits up to an hour; the
  worker has its own llama.cpp ready timeout (`INFERENCEKEY_LLAMACPP_READY_TIMEOUT_SECS`,
  2400 s on ours).
- **`unknown model architecture` or no `/v1/systemone`:** the worker is not on
  llama.cpp b11815; keep `INFERENCEKEY_LLAMACPP_TAG=b11815` in the command.
- **Probabilities that look wrong:** check that prompts fit in `-ub` and that
  `--parallel 1` is set. A prompt split across micro-batches, or batched with
  another one, is no longer read the way the model was trained.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

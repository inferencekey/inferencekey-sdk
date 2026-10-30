# strata on a private NVIDIA GPU: Qwen3.8-Flash-Next Coder 125B on one card

Serves **Qwen3.8-Flash-Next Coder 125B** (the 1-bit IQ1_M build) with the
**`strata`** backend on **one** private NVIDIA GPU, checks that a second
`ensure()` reports no drift, asks it for a function, and tears it down. The one
axis this varies vs the other examples: the backend is `strata`, which runs a
big Mixture-of-Experts model on a single consumer GPU by keeping the hot
experts in VRAM and running the rest from host RAM. The same workload serves
Claude Code through the Anthropic-compatible `/v1/messages` route.

## Compatibility
- SDK: local build in this repo (or `>= 0.1.0` once published)
- Language: Python >= 3.9
- Placement: private (NVIDIA CUDA, Linux x86_64, one GPU)
- Backend: `strata`   Task: `text2text`   Policy: `fixed`
- Manager and worker: a release that knows the `strata` backend (an older
  Manager answers `task_type 'text2text' does not allow backend 'strata'`).

## Prerequisites
- The tokens/ids this example needs, see
  [tokens & placement](../README.md#tokens). Specifically:
  `INFERENCEKEY_SDK_TOKEN`, `INFERENCEKEY_API_KEY`, `INFERENCEKEY_PROJECT`,
  and the **private-worker** id `IK_WORKER_ID` (the worker's UUID: Manager UI ->
  Workers -> copy the id).
- The SDK token needs the scopes **`workload:read`** and **`workload:write`**:
  `ensure()` lists the project's workloads and `wait_until_ready()` reads the
  readiness events. The token wizard does not tick `workload:read` by default.
- Python >= 3.9.
- A **registered NVIDIA worker** (Linux x86_64) with:
  - **GPU:** one NVIDIA GPU with **24-32 GB** of VRAM. The numbers below were
    measured on an RTX 5090 (32 GB, sm_120). AMD (HIP), macOS and multi-GPU
    setups are refused with a clear message, not ignored.
  - **Host RAM:** at least **64 GB**. The engine pins about 23 GB for Coder
    IQ1_M and 50 GB or more under load; plan for about **128 GB** to run the
    full 125B comfortably. The worker reads the container limit, not the host
    total.
  - **Disk:** about **70 GB** free on a fast NVMe for the Strata data dir
    (engine, model GGUFs, expert packs, MTP draft).
  - **CPU:** the engine runs the cold experts on the CPU. On a container with a
    CPU quota the worker sizes the engine's pool from the quota on its own.

## The command, flag by flag
The `command` is a Strata pseudo-CLI: the worker passes the setup flags to
Strata's installer, writes the engine config, and launches the server itself.
`main.py` has two recipes:

| Recipe | `command` |
| --- | --- |
| One user (default) | `strata --family coder --model IQ1_M --context 262144 --kv int8` |
| Eight users (`IK_STRATA_USERS=8`) | `strata --family coder --model IQ1_M --context 262144 --kv int8 --parallel 8 --vram-reserve-mib 1536` |

| Flag | What it does |
| --- | --- |
| `strata` | The head of the command. `vllm serve ...`, `llama-server ...` or `python -m serve.server ...` are refused. |
| `--family coder --model IQ1_M` | The Qwen3.8-Flash-Next Coder family, 1-bit IQ1_M build of the 125B model. |
| `--context 262144` | Max context per request (256K tokens). |
| `--kv int8` | KV cache in int8, about half of f16, so the long context fits next to the hot experts. |
| `--parallel 8` | Eight slots served together. It turns speculative decoding (the MTP draft) **off**, so a single user is faster without it. The worker never sets it for you. |
| `--vram-reserve-mib 1536` | VRAM kept free for the slots. The engine default (700 MiB) pages VRAM with 8 slots. The worker adds 1536 itself when `--parallel` is above 1 and no reserve is given. |
| `--pool-workers N` (not set) | The engine's CPU pool. Left out, the worker clamps it to the container CPU quota (`cpu.max`) minus a few cores; sized from the host cores on a quota-limited container, decode drops about 5x. |
| `--port`, `--host` (never set) | Owned by the worker, which leases the port; stripped if present. |

`IK_STRATA_VERSION=v0.1.40` pins a Strata release tag (`config.strata_version`);
unset keeps the platform's default tag. Never a branch name.

## Expected numbers
Measured on an **RTX 5090** with Coder 125B IQ1_M, `--context 262144 --kv int8`:

| Recipe | Throughput |
| --- | --- |
| One user (speculative decoding on) | about 144 tok/s |
| Eight users, `--parallel 8 --vram-reserve-mib 1536` | about 130 tok/s in aggregate, about 19-21 tok/s per client |
| Four users | about 125 tok/s in aggregate |

The aggregate stays about 125-145 tok/s from 1 to 8 clients: Strata does not
scale up like vLLM batching, so do not plan capacity that way. Other GPUs give
other numbers.

## Cold start (why the timeout is 3600 s)
The first start on a fresh node downloads and builds about 60-70 GB (the Strata
engine, the model GGUFs, the expert packs and the MTP draft) and then loads the
experts before the port answers: 20-40 minutes on a fast link, more on a slow
one. `main.py` waits up to **3600 s**; override it with `IK_READY_TIMEOUT`
(seconds). A second start on the same node reuses the downloads.

## Run
```bash
cp .env.example .env        # fill in real values
# install the local SDK, see ../CONTRIBUTING.md#4-depending-on-the-sdk:
python -m venv .venv && . .venv/bin/activate
pip install maturin
( cd ../../bindings/python && maturin develop --release )
pip install -r requirements.txt
set -a && . ./.env && set +a
python main.py
```

For eight users, or to keep the workload for Claude Code:

```bash
IK_STRATA_USERS=8 python main.py
IK_KEEP_ALIVE=1 python main.py
```

## What it does
- **ensure()**: declares the workload (`backend=Backend.STRATA`,
  `command="strata --family coder --model IQ1_M --context 262144 --kv int8"`,
  `task_type="text2text"`) pinned to your worker via `worker_id`, `fixed`
  policy (one pinned worker = one replica). The spec sets no
  `execution_policy_config`: the platform stores no replica count for a fixed
  policy, so passing one would make every `ensure()` re-PATCH the workload.
- **second ensure()**: the same spec with `on_drift=OnDrift.FAIL`, which raises
  if the stored workload differs from the spec. It prints `second ensure(): no
  drift`.
- **wait_until_ready()**: blocks through the cold start above.
- **call**: one `generate_text()` on the workload's endpoint with the
  data-plane key (`ik_live_`); the script prints the model and the answer.
- **delete()**: tears the workload down on exit, unless `IK_KEEP_ALIVE=1`.

## Claude Code over `/v1/messages`
Run with `IK_KEEP_ALIVE=1`, then point Claude Code at the workload. Claude Code
appends `/v1/messages` to the base URL, so the base stops at the workload slug,
and every model variable is the slug (the endpoint accepts only the slug, the
workload's model name or `default`):

```bash
export ANTHROPIC_BASE_URL="https://api.inferencekey.com/endpoint/<project_slug>/strata-qwen-coder"
export ANTHROPIC_AUTH_TOKEN="ik_live_xxxxxxxxxxxxxxxxxxxxxxxx"
export ANTHROPIC_MODEL="strata-qwen-coder"
export ANTHROPIC_DEFAULT_OPUS_MODEL="strata-qwen-coder"
export ANTHROPIC_DEFAULT_SONNET_MODEL="strata-qwen-coder"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="strata-qwen-coder"
export CLAUDE_CODE_SUBAGENT_MODEL="strata-qwen-coder"
claude
```

The gateway translates `/v1/messages` (tools included) to an OpenAI chat
request for the Strata server. Use the one-user recipe for one developer;
switch to the eight-user one for a small team, and keep the assignment's
concurrent requests at 8 so further requests queue instead of crowding the
slots. See the
[code assistant guide](https://docs.inferencekey.com/en/guides/deploy-your-own-code-assistant/)
for the `curl` checks and the error shapes.

## Cost & cleanup
This is a **private worker with a `fixed` policy**: one replica stays **always on
and billing** until it is deleted, and the GPU is reserved the whole time
(Strata fills the VRAM with its expert cache, so nothing else shares that GPU).
It does **not** scale to zero. `main.py` calls **`delete()` in a `finally`** so
it tears down on success, error, and Ctrl-C, except when you set
`IK_KEEP_ALIVE=1`. With `IK_KEEP_ALIVE=1`, or if a run is killed before
`finally` executes, delete it yourself: re-run the example without
`IK_KEEP_ALIVE`, or call `mgmt.delete("strata-qwen-coder")`. The downloaded
Strata data (about 70 GB) stays on the node for the next start; remove it by
hand if you no longer need it.

## Troubleshooting
- **`task_type 'text2text' does not allow backend 'strata'`** (or an unknown
  backend 400): the Manager is older than the `strata` backend. Update the
  Manager and the worker before running the example.
- **400 naming the expected head and the allowed settings**: the command does
  not start with `strata`, or it uses a refused flag (`--engine`, `--gpus`,
  `--layer-split`, `--update`, ...), or the config has a key other than
  `command` and `strata_version`. Use one of the two recipes.
- **`install_strata` failed: "needs Linux x86_64 with an NVIDIA GPU"**, or a RAM
  or disk shortfall with the numbers: the node does not meet the
  prerequisites above. These failures are not retried; move to a bigger node.
- **Readiness times out**: the first download and expert load are slow on a
  narrow link. Raise `IK_READY_TIMEOUT` and follow the worker drawer
  ("Loading Strata experts (can take several minutes)").
- **Slow decode on a rented container**: an explicit `--pool-workers` above the
  container CPU quota throttles the engine; drop it and let the worker size it.
- **`workload drifted from spec: ...` on the second ensure()**: the stored workload was edited
  (for example in the panel) after the first call. Re-run: the first
  `ensure()` reconciles it.
- **`KeyError: 'IK_WORKER_ID'`**: the worker id is not set. Fill it from
  Manager UI -> Workers.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

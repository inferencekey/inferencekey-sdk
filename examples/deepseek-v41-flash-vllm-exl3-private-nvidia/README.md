# vllm-exl3 on a private NVIDIA box: chat with DeepSeek-V4.1-Flash on 2x RTX PRO 6000

Serves **DeepSeek-V4.1-Flash** (763B parameters, Mixture-of-Experts) with the
**`vllm-exl3`** backend on a private **NVIDIA** worker with two **RTX PRO 6000
(96 GB, Blackwell)**, then chats with it through the gateway using the official
`openai` Python client. The one axis this varies vs the other examples: the
backend is `vllm-exl3`, vLLM with the ExLlamaV3 plugin serving Diffbot's 3-bit
EXL3 pack with **host-memory expert offload** (the experts that do not fit in
VRAM stay pinned in host RAM and are read in place over PCIe).

## Compatibility
- SDK: local build in this repo (or `>= 0.1.0` once published)
- Language: Python >= 3.9
- Placement: private (NVIDIA CUDA; 2x RTX PRO 6000 Blackwell, sm_120)
- Backend: `vllm-exl3`   Task: `text2text`   Policy: `fixed`

## Prerequisites
- The tokens/ids this example needs, see
  [tokens & placement](../README.md#tokens). Specifically:
  `INFERENCEKEY_SDK_TOKEN`, `INFERENCEKEY_API_KEY`, `INFERENCEKEY_PROJECT`,
  and the **private-worker** id `IK_WORKER_ID` (the worker's UUID: Manager UI ->
  Workers -> copy the id).
- The SDK token (`INFERENCEKEY_SDK_TOKEN`) needs the scopes **`workload:read`**
  and **`workload:write`**: `ensure()` lists the project's workloads and
  `wait_until_ready()` reads the readiness events. The token wizard in the
  Manager does not tick `workload:read` by default, so tick it when you mint the
  token.
- Python >= 3.9.
- A **registered NVIDIA worker** with:
  - **GPUs:** 2x RTX PRO 6000 96 GB (Blackwell, compute capability 12.0,
    sm_120). The kernels are built for that arch only; the worker refuses, before
    installing anything, a node without two 12.x GPUs of 90 GiB each.
  - **Host RAM** for the offloaded experts: the plan pins about 164 GiB of them
    in host memory; the measured pods had a 351 GiB memory limit. Plan for at
    least that.
  - **Disk:** at least about 400 GB free (the pack is 309 GiB, plus the venv,
    kernel caches and builds).
  - **Topology:** both GPUs on the **same NUMA node** (`nvidia-smi topo -m`
    shows `PIX`, `PXB` or `NODE` between them, not `SYS`). A host with the two
    GPUs on different sockets decodes noticeably slower; the worker warns about
    it in the drawer.

## Cold start (why the timeout is 3600 s)
The first boot on a fresh node does three long things, measured on two RunPod
machines:

| Step | Measured |
| --- | --- |
| Install of the pinned stack (vLLM, torch, ExLlamaV3, kernels) | about 5 to 8 min |
| Pack download (309 GiB) | 2.5 to 7.4 min |
| Load to ready | 4.5 to 18 min, depending on the machine |

That adds up to about 12 min on the fastest measured machine and well over half
an hour on a slow one, so `main.py` waits up to **3600 s** by default. Override
it with `IK_READY_TIMEOUT` (seconds).

## Run
```bash
cp .env.example .env        # fill in real values
# install the local SDK, see ../CONTRIBUTING.md#4-depending-on-the-sdk:
python -m venv .venv && . .venv/bin/activate
pip install maturin
( cd ../../bindings/python && maturin develop --release )
pip install -r requirements.txt
python main.py
```

To keep the workload running after the script ends (for example to call the
endpoint from another tool), run it with `IK_KEEP_ALIVE=1`:

```bash
IK_KEEP_ALIVE=1 python main.py
```

The script then skips `delete()` and prints the endpoint base URL and how to
delete the workload later.

## What it does
- **ensure()**: declares the workload (`backend=Backend.VLLM_EXL3`,
  `command="vllm serve diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000
  --revision a1cf1bd227a754aa7e2ca5c21652bfaf256ee8a5 --served-model-name
  deepseek-v4.1-flash"`, `task_type="text2text"`) pinned to your worker via
  `worker_id`, `fixed` policy (one pinned worker = one replica). The spec sets no
  `execution_policy_config`: the platform stores no replica count for a fixed
  policy, so passing one would make every `ensure()` re-PATCH the workload. The
  command is the one the Manager UI prefills; the worker adds the recipe
  defaults and pins the stack, so there is no `vllm_version` (this backend's
  config holds only the command). Idempotent by `slug`.
- **wait_until_ready()**: blocks through the cold start above, up to
  `IK_READY_TIMEOUT` seconds (3600 by default).
- **call**: one chat completion with the official `openai` client against
  `<base>/endpoint/<project>/deepseek-v41-flash/v1` and the data-plane key
  (`ik_live_`), with `model="default"`. The gateway accepts `"default"` (or the
  workload's model name) and refuses any other name; the `--served-model-name`
  is only seen on the pod. The script prints the answer.
- **delete()**: tears the workload down on exit, unless `IK_KEEP_ALIVE=1`.

## Using the same workload from the panel and from code
If you also create the workload in the Manager panel, give it the slug
`deepseek-v41-flash` (or change `SLUG` in `main.py` to the panel's slug), so
`ensure()` adopts it instead of creating a second one. `ensure()` then aligns
the fields that drifted from the spec (for example the name) with a PATCH.

## Cost & cleanup
This is a **private worker with a `fixed` policy**: one replica stays **always on
and billing** until it is deleted. The two GPUs are reserved the whole time; it
does **not** scale to zero. `main.py` calls **`delete()` in a `finally`** so it
tears down on success, error, and Ctrl-C, except when you set `IK_KEEP_ALIVE=1`.
With `IK_KEEP_ALIVE=1`, or if a run is killed before `finally` executes, delete
it yourself: re-run the example without `IK_KEEP_ALIVE`, or call
`mgmt.delete("deepseek-v41-flash")`. On a rented node the cold start alone
(install, download and load) is paid time, so keep the workload only while you
use it.

## Troubleshooting
- **`PermissionDenied: permission denied [forbidden]: scope_insufficient: this
  ik_sdk_ token is missing the 'workload:read' scope`**: the SDK token was
  minted with the wizard defaults, which leave out `workload:read`. Mint a new
  SDK token with `workload:read` and `workload:write` and put it in
  `INFERENCEKEY_SDK_TOKEN`.
- **`KeyError: 'IK_WORKER_ID'`**: you didn't set the worker id. Copy
  `.env.example` to `.env` and fill it from Manager UI -> Workers (copy the id).
- **`worker_id must belong to the same project`**: the worker isn't assigned to
  `INFERENCEKEY_PROJECT`. Assign it to that project in the Manager first.
- **400 on create (unknown backend or config)**: the Manager or the worker is
  older than the `vllm-exl3` backend. Update both before running the example.
- **The worker refuses the node before installing**: the node does not have two
  GPUs of compute capability 12.x with 90 GiB each. Use 2x RTX PRO 6000.
- **Readiness times out**: the install, the 309 GiB download and the load are
  slow on some machines (narrow network, GPUs on two sockets). Raise
  `IK_READY_TIMEOUT`.
- **`failed to become ready: bootstrap step failed`**: the reason is in the
  worker drawer and the worker log in the Manager UI. The usual causes are the
  worker refusing the node (not two GPUs of compute capability 12.x with 90 GiB
  each) or a failed install.
- **`model_not_available`**: send `model="default"` (or the workload's model
  name), not the `--served-model-name`.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).

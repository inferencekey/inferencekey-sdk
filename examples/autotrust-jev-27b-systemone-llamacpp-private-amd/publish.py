"""Deploy AutoTrust-JEV-27B v1.1 ("Jev", Q6_K GGUF) with the built-in llama.cpp
backend on a private AMD R9700 32 GB, and expose its System One decision API.

Jev is a decision model: it does not write text. Given a state and questions
with options, it returns a probability for every option. llama.cpp b11815
serves that natively at `/v1/systemone`, and InferenceKey proxies the route:

    POST {base}/endpoint/{project}/{workload}/v1/systemone

The canonical SDK shape: ensure() -> wait_until_ready(). The call itself lives
in decide.py. The axis this example varies: a non-causal (bidirectional)
decision model, which needs a pinned llama.cpp release and a few flags so every
prompt is read whole, in one pass, alone.

Run:  cp .env.example .env   # fill in real values, then
      python publish.py             # create or reconcile, then wait until ready
      python publish.py --dry-run   # plan only: log what ensure() would do
      python publish.py --delete    # tear the workload down
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from inferencekey import (
    Backend,
    ExecutionPolicy,
    InferenceKeyError,
    ManagementClient,
    OnDrift,
    TaskType,
    WorkloadSpec,
)

HERE = Path(__file__).resolve().parent
MODEL = "autotrust/AutoTrust-JEV-27B-v1.1"
# Our own Q6_K conversion (21,049 MiB, 6.57 bits per weight): there is no public
# GGUF of v1.1. README, step 1, says how it was made. The path is the one the
# worker sees (on our worker the host's model cache is mounted at /models).
GGUF = os.environ.get("JEV_GGUF", "/models/huggingface/jevtaxi/AutoTrust-JEV-27B-v1.1-Q6_K.gguf")
SLUG = os.environ.get("JEV_WORKLOAD", "autotrust-jev-27b-llamacpp-amd")

# llama-server launch command, run verbatim by the `llamacpp` backend.
#   INFERENCEKEY_LLAMACPP_TAG=b11815   -> the worker uses this llama.cpp release,
#     which has the native System One API (`/v1/systemone`) and the `vllm-d3`
#     decision readout Jev's GGUF declares.
#   INFERENCEKEY_LLAMACPP_ASSET=ubuntu-rocm-10.0-x64 -> its ROCm 10.0 build
#     (sees the R9700 / gfx120x).
#   -ngl 99                -> every layer on the GPU.
#   -c 4096                -> context; a decision prompt is a few hundred tokens.
#   -b 2048 -ub 2048       -> the micro-batch must hold the LONGEST prompt: the
#     GGUF says `qwen35.attention.causal = false`, so a prompt is read
#     bidirectionally and must be evaluated whole, in one micro-batch.
#   --parallel 1           -> one slot, one prompt at a time, never batched with
#     another request.
#   --ctx-checkpoints 0 --cache-ram 0 -> no context checkpoints and no prompt
#     cache: every request is read from scratch, as the model expects.
# The worker injects the listen port; do not hard-code --port.
COMMAND = (
    "INFERENCEKEY_LLAMACPP_TAG=b11815 INFERENCEKEY_LLAMACPP_ASSET=ubuntu-rocm-10.0-x64 llama-server "
    f"-m {GGUF} -ngl 99 -c 4096 -b 2048 -ub 2048 --parallel 1 --ctx-checkpoints 0 --cache-ram 0"
)


def _load_dotenv() -> None:
    env = HERE / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def build_spec() -> WorkloadSpec:
    return WorkloadSpec(
        name="AutoTrust JEV 27B v1.1 (llama.cpp, AMD)",
        slug=SLUG,
        model=MODEL,
        description="Decision model (System One API) for the Crazy Taxi experiment.",
        backend=Backend.LLAMACPP,
        command=COMMAND,
        # A decision model: options in, probabilities out.
        task_type=TaskType.CLASSIFICATION,
        # Pin to your AMD ROCm box. The vendor/arch (RDNA4) lives on the worker.
        worker_id=os.environ["IK_WORKER_ID"],
        # Private + fixed: one always-on replica. It keeps the GPU until deleted.
        execution_policy=ExecutionPolicy.FIXED,
    )


def main(argv: list) -> int:
    _load_dotenv()
    mgmt = ManagementClient.from_env()
    try:
        if "--delete" in argv:
            print(f"deleted={mgmt.delete(SLUG)}")
            return 0

        dry_run = "--dry-run" in argv
        ref = mgmt.ensure(build_spec(), on_drift=OnDrift.DRY_RUN if dry_run else OnDrift.RECONCILE)
        if dry_run:
            print(f"dry run: planned {ref.project_slug}/{ref.workload_slug}; nothing was changed")
            return 0
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # A cold start loads the 21 GB GGUF; on our R9700 host it took about 19
        # minutes (README). Give it an hour.
        t0 = time.monotonic()
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.monotonic() - t0:.1f} s")
        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        print(f"System One endpoint: {base}/endpoint/{ref.project_slug}/{ref.workload_slug}/v1/systemone")
        print("next: python decide.py")
    except InferenceKeyError as e:
        # Never print tokens; surface a clean message.
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

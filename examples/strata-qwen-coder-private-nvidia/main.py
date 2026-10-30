"""Serve **Qwen3.8-Flash-Next Coder 125B** with the **strata** backend on ONE private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete())
varying one axis vs the other examples: the backend is `strata`, which serves a
big Mixture-of-Experts model on a single consumer GPU by keeping the hot
experts in VRAM and running the rest from host RAM. The model is the 1-bit
IQ1_M build of the 125B coder, the one measured on an RTX 5090.

The `command` is not a binary to run: it is a Strata pseudo-CLI
(`strata --family ... --model ...`) that the worker turns into Strata's
installer flags and engine config. The worker owns the port, the host and the
GPU, so the command never carries `--port`.

After the first ensure() the script runs ensure() again with
`on_drift=OnDrift.FAIL`: it raises if the stored workload differs from the
spec, so a clean second call proves the spec converged (no PATCH on re-runs).

Placement is private: the workload is pinned to YOUR registered worker via
`worker_id`. The spec does not carry the GPU vendor/arch; that is inferred from
the worker you pin to.

Set IK_STRATA_USERS=8 for the eight-user recipe (parallel slots) instead of
the one-user recipe, and IK_KEEP_ALIVE=1 to keep the workload (and its
endpoint, for Claude Code) running after the script ends.

Run:  cp .env.example .env  # fill in real values, then
      python main.py
"""

from __future__ import annotations

import os
import sys

from inferencekey import (
    ManagementClient,
    DataClient,
    WorkloadSpec,
    Backend,
    ExecutionPolicy,
    OnDrift,
    InferenceKeyError,
)

# --- The model / command ---------------------------------------------------
# The workload's model name on the wire. Strata picks the weights from the
# command (--family/--model), not from a Hugging Face id; this is the name the
# gateway also accepts as the model.
MODEL = "qwen3.8-flash-next-coder-iq1_m"

# One user, fastest single stream (about 144 tok/s measured on an RTX 5090):
#   strata              the head; anything else ("vllm serve", "llama-server")
#                       is refused by the platform
#   --family coder      Strata's model family: the Qwen3.8-Flash-Next Coder line
#   --model IQ1_M       the 1-bit IQ1_M build of the 125B coder (about 58-70 GB
#                       of downloads with the expert packs and the MTP draft)
#   --context 262144    max context per request; the engine reserves KV for it
#   --kv int8           KV cache in int8: about half the memory of f16, which is
#                       what lets 262144 tokens fit next to the hot experts
# No --parallel: one slot keeps speculative decoding (the MTP draft) on, which
# is what makes a single client fast.
COMMAND_ONE_USER = "strata --family coder --model IQ1_M --context 262144 --kv int8"

# Eight users together (about 130 tok/s in aggregate, about 19-21 tok/s each):
#   --parallel 8              eight slots served together. Parallel slots turn
#                             speculative decoding OFF, so one user gets less
#                             speed than with the recipe above; the aggregate
#                             stays about 125-145 tok/s from 1 to 8 clients.
#   --vram-reserve-mib 1536   VRAM kept free for the slots' working buffers. The
#                             engine default (700 MiB) pages VRAM with 8 slots.
#                             The worker adds 1536 itself when --parallel > 1
#                             and no reserve is given; it is written out here so
#                             the command says what runs.
COMMAND_EIGHT_USERS = (
    "strata --family coder --model IQ1_M --context 262144 --kv int8 "
    "--parallel 8 --vram-reserve-mib 1536"
)
# Not set in either recipe, on purpose:
#   --pool-workers N   the engine's CPU worker pool. Left out, the worker sizes it
#                      from the container CPU quota (cgroup cpu.max) minus a few
#                      cores of headroom; sized from the host cores on a quota
#                      limited container, decode drops about 5x. Set it only to
#                      override that.
#   --port / --host    owned by the worker (it leases the port); stripped if given.

USERS = os.environ.get("IK_STRATA_USERS", "1")
COMMAND = COMMAND_EIGHT_USERS if USERS == "8" else COMMAND_ONE_USER

# Optional release pin of Strata (a tag such as "v0.1.40", never a branch).
# Unset keeps the platform's default tag. A new tag starts a new process.
STRATA_VERSION = os.environ.get("IK_STRATA_VERSION")

SLUG = "strata-qwen-coder"
PROMPT = "Write a Python function that returns the n-th Fibonacci number iteratively."

# The first start downloads and builds about 60-70 GB (engine, model, expert
# packs, MTP draft) and then loads the experts: 20-40 min on a fast link. One
# hour leaves margin.
READY_TIMEOUT = int(os.environ.get("IK_READY_TIMEOUT", "3600"))


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()

    # The worker id comes from the environment, never hard-code it. It's the id
    # of your registered NVIDIA worker (Manager UI -> Workers -> copy it).
    worker_id = os.environ["IK_WORKER_ID"]

    spec = WorkloadSpec(
        name="Qwen3.8-Flash-Next Coder 125B (Strata)",
        slug=SLUG,
        model=MODEL,
        backend=Backend.STRATA,  # big MoE on one NVIDIA GPU; text2text only
        command=COMMAND,
        config={"strata_version": STRATA_VERSION} if STRATA_VERSION else None,
        task_type="text2text",
        # Pin to your box. gpu_resource_id is omitted: Strata serves one GPU and
        # the worker picks it.
        worker_id=worker_id,
        # Private + fixed: one always-on replica. It stays reserved (and billing)
        # until delete() runs, see "Cost & cleanup" in the README.
        # No execution_policy_config: the platform stores no replica count for a
        # fixed policy (one pinned worker = one replica), so passing one would
        # make every ensure() re-PATCH the workload.
        execution_policy=ExecutionPolicy.FIXED,
    )

    base = os.environ.get("INFERENCEKEY_BASE_URL", "https://api.inferencekey.com")
    project = os.environ["INFERENCEKEY_PROJECT"]

    ref = None
    try:
        # 1) Provision / reconcile. Idempotent by slug.
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug} ({USERS} user recipe)")

        # 1b) The same spec again must not drift: OnDrift.FAIL raises instead of
        #     sending a PATCH if the stored workload differs from the spec.
        mgmt.ensure(spec, on_drift=OnDrift.FAIL)
        print("second ensure(): no drift")

        # 2) Wait for the cold worker to install Strata, download the model and
        #    load the experts. On a fresh node this takes tens of minutes.
        print(
            "waiting for the worker to become ready (cold start: about 60-70 GB of "
            f"downloads and the expert load, 20-40 min; timeout {READY_TIMEOUT} s)..."
        )
        mgmt.wait_until_ready(ref.workload_slug, timeout=READY_TIMEOUT)

        # 3) Call the OpenAI-compatible endpoint (data plane, ik_live_).
        data = DataClient.from_env()
        ep = data.endpoint(ref.workload_slug)
        out = ep.generate_text(prompt=PROMPT, temperature=0.2, max_tokens=300)
        print(f"\nmodel: {out.model}\n{out.text}\n")
    except InferenceKeyError as e:
        # Don't print tokens; surface a clean message.
        print(f"error: {e}", file=sys.stderr)
        return 1
    except TimeoutError as e:
        # wait_until_ready() gave up before the worker reported ready.
        print(
            f"error: {e}; raise IK_READY_TIMEOUT or check the worker in the Manager UI",
            file=sys.stderr,
        )
        return 1
    finally:
        # 4) Tear down so the reserved GPU stops billing, unless asked to keep
        #    the endpoint alive. Idempotent and safe: runs on success, error,
        #    and Ctrl-C.
        keep_alive = os.environ.get("IK_KEEP_ALIVE") == "1"
        if keep_alive and ref is None:
            # ensure() failed, so there is no workload to keep or point at.
            print("ensure() did not complete, nothing was created to keep (IK_KEEP_ALIVE=1)")
        elif keep_alive:
            print(f"kept workload {SLUG} (IK_KEEP_ALIVE=1); it keeps billing until deleted")
            print(f"Claude Code base URL: {base}/endpoint/{project}/{SLUG} (model \"{SLUG}\")")
            print(
                "to delete it, re-run without IK_KEEP_ALIVE, or call "
                f"ManagementClient.from_env().delete(\"{SLUG}\")"
            )
        else:
            deleted = mgmt.delete(SLUG)
            print(f"deleted={deleted}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

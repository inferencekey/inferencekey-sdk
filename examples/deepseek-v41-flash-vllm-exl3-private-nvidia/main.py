"""Chat with **DeepSeek-V4.1-Flash** on the **vllm-exl3** backend on a private NVIDIA box.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete())
varying one axis vs the other examples: the backend is `vllm-exl3`, vLLM with the
ExLlamaV3 plugin serving Diffbot's 3-bit EXL3 pack of DeepSeek-V4.1-Flash (763B
parameters, Mixture-of-Experts) on two RTX PRO 6000 (96 GB, Blackwell). The
experts that do not fit in VRAM stay pinned in host RAM and are read in place
over PCIe (host-memory expert offload).

The chat goes through the InferenceKey gateway with the official `openai`
Python client: the gateway exposes an OpenAI-compatible base URL per workload,
`<base>/endpoint/<project>/<slug>/v1`, and callers send `model="default"`.

Placement is private: the workload is pinned to YOUR registered worker via
`worker_id`. The spec does not carry the GPU vendor/arch; that is inferred from
the worker you pin to (a node with two sm_120 GPUs of 96 GB).

Set IK_KEEP_ALIVE=1 to keep the workload (and its endpoint) running after the
script ends instead of deleting it.

Run:  cp .env.example .env  # fill in real values, then
      python main.py
"""

from __future__ import annotations

import os
import sys

import openai
from openai import OpenAI

from inferencekey import (
    ManagementClient,
    WorkloadSpec,
    Backend,
    ExecutionPolicy,
    InferenceKeyError,
)

# --- The model / command ---------------------------------------------------
# Diffbot's 3-bit EXL3 pack (309 GiB), cut to fit two 96 GB GPUs with the
# experts that do not fit offloaded to host RAM. The `model` field is the
# workload's model name on the wire; the gateway also accepts it as the model.
MODEL = "diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000"

# The `vllm serve` command, exactly as the Manager UI prefills it. That is the
# whole command on purpose: the worker adds every recipe default the command
# does not set, and pins the vLLM / torch / ExLlamaV3 stack itself, so there is
# no vllm_version to choose (the platform accepts only `command` in the config
# of this backend).
#   --revision            pins the pack to one Hugging Face commit
#   --served-model-name   the name vLLM answers under on the pod; callers
#                         never see it, they send "default" to the gateway
# The worker injects the listen port; do NOT hard-code it.
COMMAND = (
    "vllm serve diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 "
    "--revision a1cf1bd227a754aa7e2ca5c21652bfaf256ee8a5 "
    "--served-model-name deepseek-v4.1-flash"
)

SLUG = "deepseek-v41-flash"
PROMPT = "In two sentences, what is a Mixture-of-Experts model?"

# The cold start is long: install about 5-8 min, the 309 GiB pack download
# 2.5-7.4 min, and the load to ready 4.5-18 min depending on the machine. One
# hour covers the slowest measured machine with margin.
READY_TIMEOUT = int(os.environ.get("IK_READY_TIMEOUT", "3600"))


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()

    # The worker id comes from the environment, never hard-code it. It's the id
    # of your registered NVIDIA worker (Manager UI -> Workers -> copy it).
    worker_id = os.environ["IK_WORKER_ID"]

    spec = WorkloadSpec(
        name="DeepSeek-V4.1-Flash (vllm-exl3) on 2x RTX PRO 6000",
        slug=SLUG,
        model=MODEL,
        backend=Backend.VLLM_EXL3,  # vLLM + ExLlamaV3 with expert offload; text2text only
        command=COMMAND,
        task_type="text2text",
        # Pin to your box with two RTX PRO 6000. gpu_resource_id is omitted: the
        # backend uses both GPUs of the worker (tensor parallel 2).
        worker_id=worker_id,
        # Private + fixed: one always-on replica. It stays reserved (and billing)
        # until delete() runs, see "Cost & cleanup" in the README.
        # No execution_policy_config: the platform stores no replica count for a
        # fixed policy (one pinned worker = one replica), so passing one would
        # make every ensure() re-PATCH the workload.
        execution_policy=ExecutionPolicy.FIXED,
    )

    base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com")
    project = os.environ["INFERENCEKEY_PROJECT"]
    endpoint_base = f"{base}/endpoint/{project}/{SLUG}/v1"

    ref = None
    try:
        # 1) Provision / reconcile. Idempotent by slug.
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # 2) Wait for the cold worker to install the pinned stack, download the
        #    309 GiB pack and load it. On a fresh node this takes from about 12
        #    minutes to well over half an hour, hence the long timeout.
        print(
            "waiting for the worker to become ready (cold start: install, 309 GiB "
            f"download and load, from about 12 min to over 30 min; timeout {READY_TIMEOUT} s)..."
        )
        mgmt.wait_until_ready(ref.workload_slug, timeout=READY_TIMEOUT)

        # 3) Chat through the gateway with the official OpenAI client and the
        #    data-plane key (ik_live_). The gateway answers under "default".
        client = OpenAI(base_url=endpoint_base, api_key=os.environ["INFERENCEKEY_API_KEY"])
        completion = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=200,
        )
        answer = completion.choices[0].message.content
        print(f"\n{answer}\n")
    except InferenceKeyError as e:
        # Don't print tokens; surface a clean message.
        print(f"error: {e}", file=sys.stderr)
        return 1
    except openai.APIError as e:
        print(f"chat request failed: {e.message}", file=sys.stderr)
        return 1
    except TimeoutError as e:
        # wait_until_ready() gave up before the worker reported ready.
        print(
            f"error: {e}; raise IK_READY_TIMEOUT or check the worker in the Manager UI",
            file=sys.stderr,
        )
        return 1
    finally:
        # 4) Tear down so the reserved GPUs stop billing, unless asked to keep
        #    the endpoint alive. Idempotent and safe: runs on success, error,
        #    and Ctrl-C.
        keep_alive = os.environ.get("IK_KEEP_ALIVE") == "1"
        if keep_alive and ref is None:
            # ensure() failed, so there is no workload to keep or point at.
            print("ensure() did not complete, nothing was created to keep (IK_KEEP_ALIVE=1)")
        elif keep_alive:
            print(f"kept workload {SLUG} (IK_KEEP_ALIVE=1); it keeps billing until deleted")
            print(f"endpoint: {endpoint_base} (model \"default\")")
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

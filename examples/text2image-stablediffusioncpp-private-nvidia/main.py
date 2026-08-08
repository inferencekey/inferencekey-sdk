"""Generate an image with the **stable-diffusion.cpp** backend on a private NVIDIA GPU.

The canonical SDK shape — ensure() -> wait_until_ready() -> call -> delete() —
varying one axis vs the other examples: the backend is `stablediffusioncpp` and
the task is **text2image**, served by `sd-server` (stable-diffusion.cpp's HTTP
server). The data plane has no typed image helper yet, so — exactly like the
forecast example POSTs to `/forecast` — this script POSTs an OpenAI-style image
request to the workload's `/v1/images/generations` endpoint with the data-plane
key (`ik_live_`) and writes the returned PNG to disk.

Why stable-diffusion.cpp: it compiles its CUDA kernels for the local GPU
architecture (like llama.cpp), so it runs on **older GPUs that sglang does not
support** — e.g. Volta (Tesla V100, sm_70), where sglang's prebuilt sgl_kernel
has no kernel image. This example targets exactly that box.

Placement is private: the workload is pinned to YOUR registered worker via
`worker_id`. The spec does not carry the GPU vendor/arch — that is inferred from
the worker you pin to (an NVIDIA CUDA box).

Run:  cp .env.example .env  # fill in real values, then
      python main.py
"""

from __future__ import annotations

import base64
import os
import sys

import requests

from inferencekey import (
    ManagementClient,
    WorkloadSpec,
    Backend,
    ExecutionPolicy,
    InferenceKeyError,
)

# --- The model / command ---------------------------------------------------
# Z-Image Turbo is a diffusion model published as GGUF. Unlike a single-file
# LLM, a diffusion model needs THREE pieces, and the sd-server command points at
# all three (paths are inside the worker container — adjust to where your worker
# has them):
#   --diffusion-model  the diffusion GGUF itself (Z-Image Turbo, Q8_0 ~6.6 GB)
#   --vae              the VAE that decodes the latent into pixels (FLUX "ae")
#   --llm              the text encoder that turns the prompt into a condition
#                      (Qwen3-4B-Instruct GGUF)
# The `model` field is the served-model name reported on the wire; the actual
# weights come from the paths in the command.
MODEL = "Tongyi-MAI/Z-Image-Turbo"

# sd-server launch command, run verbatim by the `stablediffusioncpp` backend.
# The flags are the ones validated for the Turbo variant on a Tesla V100:
#   --cfg-scale 1.0 --steps 8   -> Turbo's recommended low-step schedule
#   --fa                        -> flash attention
#   --auto-fit                  -> split diffusion/text-encoder/vae across the
#                                  available GPUs by module size and free VRAM.
#                                  On 2x 16 GB V100s this is what keeps the VAE
#                                  decode's ~6.6 GB scratch buffer from OOMing on
#                                  top of the ~9.9 GB of weights.
#   -H 1024 -W 1024             -> output resolution
# The worker injects the listen port (`--listen-port`); do NOT hard-code it.
# sd-server is the HTTP *server* binary; the one-shot CLI is `sd-cli` and does
# not serve HTTP — the command must start with `sd-server`.
DIFFUSION_MODEL = os.environ.get(
    "IK_SDCPP_DIFFUSION_MODEL", "/sdcpp-models/z_image_turbo-Q8_0.gguf"
)
VAE = os.environ.get("IK_SDCPP_VAE", "/sdcpp-models/ae.safetensors")
LLM = os.environ.get("IK_SDCPP_LLM", "/sdcpp-models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")

COMMAND = (
    f"sd-server --diffusion-model {DIFFUSION_MODEL} --vae {VAE} --llm {LLM} "
    "--cfg-scale 1.0 --steps 8 --fa --auto-fit -H 1024 -W 1024"
)

SLUG = "sdcpp-zimage"
PROMPT = "a serene mountain lake at sunrise, photorealistic, high detail"
OUT_PATH = "output.png"


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()

    # The worker id comes from the environment — never hard-code it. It's the id
    # of your registered NVIDIA worker (Manager UI -> Workers -> copy it).
    worker_id = os.environ["IK_WORKER_ID"]

    spec = WorkloadSpec(
        name="Z-Image Turbo (stable-diffusion.cpp) on NVIDIA",
        slug=SLUG,
        model=MODEL,
        backend=Backend.STABLE_DIFFUSION_CPP,  # sd-server; text2image
        command=COMMAND,
        task_type="text2image",
        # Pin to your NVIDIA box. The vendor/arch (e.g. Volta sm_70) is inferred
        # from the worker — the same command runs on any CUDA arch sd.cpp builds
        # for. gpu_resource_id is optional and omitted; --auto-fit uses all GPUs
        # the worker exposes (2x V100 here).
        worker_id=worker_id,
        # Private + fixed: one always-on replica. It stays reserved (and billing)
        # until delete() runs — see "Cost & cleanup" in the README.
        execution_policy=ExecutionPolicy.FIXED,
        execution_policy_config={"replicas": 1},
    )

    try:
        # 1) Provision / reconcile. Idempotent by slug.
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # 2) Wait for the cold worker to build sd.cpp (CUDA compile for the local
        #    arch, unless a prebuilt binary is already present) and load the three
        #    models. The first boot is slow — the compile and the model loads take
        #    minutes — so give it a generous timeout.
        print("waiting for the worker to become ready (cold start can take a while)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=1800)

        # 3) Generate the image via the workload's /v1/images/generations
        #    endpoint. The SDK has no typed image helper yet, so we POST the
        #    OpenAI-style request directly with the data-plane key (ik_live_).
        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com")
        project = os.environ["INFERENCEKEY_PROJECT"]
        api_key = os.environ["INFERENCEKEY_API_KEY"]
        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/images/generations"

        body = {
            "model": "default",  # served model; sd-server answers under its launch id
            "prompt": PROMPT,
            "n": 1,
            "size": "1024x1024",
        }
        # Diffusion is slower than text: the first generation loads the models
        # and samples — allow a few minutes rather than the default read timeout.
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=300,
        )
        resp.raise_for_status()
        out = resp.json()

        # OpenAI image shape: {"data": [{"b64_json": "..."}]}. Decode and save.
        item = out["data"][0]
        png = base64.b64decode(item["b64_json"])
        with open(OUT_PATH, "wb") as f:
            f.write(png)
        print(f"\nwrote {OUT_PATH} ({len(png)} bytes)\n")
    except InferenceKeyError as e:
        # Don't print tokens; surface a clean message.
        print(f"error: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"image request failed: {e} — {e.response.text[:200]}", file=sys.stderr)
        return 1
    finally:
        # 4) Tear down so the reserved GPU stops billing. Idempotent and safe —
        #    runs on success, error, and Ctrl-C.
        deleted = mgmt.delete(SLUG)
        print(f"deleted={deleted}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

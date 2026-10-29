"""Serve Hugging Face's Carbon-X with the stock vLLM backend on one private
NVIDIA GPU and ask it about raw DNA in an ordinary chat.

The canonical SDK shape: ensure() -> wait_until_ready() -> calls. The axis
this example varies: the prompt carries DNA. Carbon-X (Apache-2.0, 14.8B, a
Qwen3-14B fine-tune) reads A/C/G/T runs as 6-mer tokens through its own
tokenizer, so the backend needs `--trust-remote-code`, and it was trained
and evaluated with thinking off, so the server turns thinking off for every
request (`--default-chat-template-kwargs`).

29.6 GB of BF16 weights: one 48 GB card (A40, L40S, RTX A6000, RTX 6000 Ada).
See the README for the smaller-card variant (FP8 on a 24 GB card).

Run:  cp .env.example .env   # fill in real values, then
      python main.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from inferencekey import (
    Backend,
    DataClient,
    ExecutionPolicy,
    InferenceKeyError,
    ManagementClient,
    WorkloadSpec,
)

MODEL = "HuggingFaceBio/Carbon-X"

# vLLM launch command, run verbatim by the `vllm` backend (the worker injects
# host and port; do not hard-code --port).
#   --trust-remote-code      -> loads the hybrid DNA tokenizer (tokenizer.py in
#                               the model repo): BPE for text, 6-mers for DNA.
#   --max-model-len 16384    -> post-training examples were under 16K tokens
#                               (about 90 kbp of DNA at 6 bases per token).
#   --default-chat-template-kwargs -> thinking off, as the authors trained and
#                               evaluated it; requests need no extra field.
# vLLM applies generation_config.json (temperature 0.6, top_p 0.95, top_k 20),
# the sampling the authors used for every reported result.
COMMAND = (
    f"vllm serve {MODEL} --trust-remote-code --dtype bfloat16 "
    "--max-model-len 16384 "
    "--default-chat-template-kwargs '{\"enable_thinking\": false}'"
)
# 24 GB cards (RTX 4090, RTX 3090, L4): set CARBON_FP8=1. vLLM quantizes the
# BF16 weights to FP8 while loading (15.8 GiB instead of 27.6 GiB) and the
# context drops to 8,192 tokens so the KV cache fits next to them.
if os.environ.get("CARBON_FP8") == "1":
    COMMAND = COMMAND.replace("--max-model-len 16384", "--max-model-len 8192 --quantization fp8")

# Carbon-X needs Transformers v5; vLLM 0.32.0 pins transformers 5.16-5.19.
# Its torch wheel needs a CUDA 13 driver (580 or newer) on the worker.
VLLM_VERSION = "0.32.0"

SLUG = "carbon-x"

# The authors' own example from the model card: the first 55 bases of human
# HBB (beta-globin) and the same stretch with codon 6 GAG -> GTG, the
# sickle-cell mutation (Glu6Val). The model is not told the gene.
QUESTIONS = [
    (
        "Reference: ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGG\n"
        "Variant:   ATGGTGCATCTGACTCCTGTGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGG\n\n"
        "Is this variant likely to be benign or pathogenic? Explain your reasoning."
    ),
    "What does this sequence encode?\n\nATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTT",
]

HERE = Path(__file__).resolve().parent


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="Carbon-X DNA chat (vLLM)",
        slug=SLUG,
        model=MODEL,
        backend=Backend.VLLM,
        command=COMMAND,
        vllm_version=VLLM_VERSION,
        task_type="text2text",
        # Pin to your NVIDIA worker (48 GB, CUDA 13 driver). Never hard-code the id.
        worker_id=os.environ["IK_WORKER_ID"],
        execution_policy=ExecutionPolicy.FIXED,
    )

    try:
        # 1) Provision / reconcile. Idempotent by slug.
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # 2) Wait until vLLM serves. A cold worker installs vLLM and downloads
        #    29.6 GB of weights first, so give it time.
        t0 = time.monotonic()
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        print(f"ready after {time.monotonic() - t0:.1f} s")

        # 3) Ask about DNA through the OpenAI-compatible endpoint (ik_live_).
        ep = DataClient.from_env().endpoint(ref.workload_slug)
        lines = []
        for q in QUESTIONS:
            t1 = time.monotonic()
            out = ep.generate_text(
                messages=[{"role": "user", "content": q}], max_tokens=1024
            )
            seconds = time.monotonic() - t1
            usage = (out.raw or {}).get("usage", {})
            answer = out.text.strip()
            print(f"\n> {q}\n\n{answer}\n")
            print(
                f"{seconds:.1f} s, {usage.get('prompt_tokens')} prompt tokens, "
                f"{usage.get('completion_tokens')} answer tokens"
            )
            lines += [f"## Q\n\n```\n{q}\n```\n", f"## A ({seconds:.1f} s)\n\n{answer}\n"]
        outputs = HERE / "outputs"
        outputs.mkdir(exist_ok=True)
        (outputs / "answers.md").write_text("\n".join(lines))
        print(f"saved {outputs / 'answers.md'}")
    except InferenceKeyError as e:
        # Don't print tokens; surface a clean message.
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        # 4) Tear down so the reserved GPU stops billing, unless you want to
        #    keep calling it (IK_KEEP_ALIVE=1).
        if os.environ.get("IK_KEEP_ALIVE") != "1":
            print(f"deleted={mgmt.delete(SLUG)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

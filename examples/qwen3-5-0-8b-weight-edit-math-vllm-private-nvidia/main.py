"""Serve the weight-edited Qwen3.5-0.8B (and, optionally, the unedited base
model beside it) with the stock vLLM backend on one private NVIDIA GPU, and
ask both the same MATH-500 problem with thinking on.

The edited checkpoint is RikiyaT/Qwen3.5-0.8B-weight-space-programming: a
frontier LLM edited 64 output columns of `down_proj` in layers 15 and 19
(131,072 weights, 0.016 % of the model) with no training, to stop the base
model's thinking loops. Same architecture, same size, same serving command.

The canonical SDK shape: ensure() -> wait_until_ready() -> calls. The axis
this example varies: two workloads that differ only in their weights share
one GPU, so the effect of a weight edit can be measured side by side.

Run:  cp .env.example .env   # fill in real values, then
      python main.py                    # edited model only
      IK_WITH_BASELINE=1 python main.py # edited + base on the same GPU
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

from inferencekey import (
    Backend,
    ExecutionPolicy,
    InferenceKeyError,
    ManagementClient,
    WorkloadSpec,
)

EDITED = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
BASE = "Qwen/Qwen3.5-0.8B"
WITH_BASELINE = os.environ.get("IK_WITH_BASELINE") == "1"

# The authors' evaluation protocol (Qwen's thinking-mode sampling).
#   --default-chat-template-kwargs -> thinking on: Qwen3.5-0.8B answers without
#                                     thinking by default.
#   --override-generation-config   -> the repos ship no generation_config.json,
#                                     so set Qwen's thinking-mode sampling on the
#                                     server; requests still send it explicitly.
#   --max-model-len 34816          -> a 32,768-token answer plus the prompt.
#   --gpu-memory-utilization       -> 0.45 each when both share one card.
GPU_SHARE = "0.45" if WITH_BASELINE else "0.90"
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}


def command(model: str) -> str:
    return (
        f"vllm serve {model} --dtype bfloat16 --max-model-len 34816 "
        f"--gpu-memory-utilization {GPU_SHARE} "
        "--default-chat-template-kwargs '{\"enable_thinking\": true}' "
        f"--override-generation-config '{json.dumps(SAMPLING)}'"
    )


# Qwen3.5 needs a recent vLLM; the authors' re-run used 0.33.0. Its torch wheel
# needs a CUDA 13 driver (580 or newer) on the worker.
VLLM_VERSION = "0.33.0"

WORKLOADS = [("qwen35-0-8b-edited", "Qwen3.5-0.8B edited (weight-space programming)", EDITED)]
if WITH_BASELINE:
    WORKLOADS.append(("qwen35-0-8b-base", "Qwen3.5-0.8B base", BASE))

# A MATH-500 problem (test/number_theory/572.json, answer 9), with Qwen's math
# instruction, as in the authors' evaluation. In the authors' samples the base
# model got it right 0 times in 4, the edited model 4 times in 4.
PROBLEM = "How many positive whole-number divisors does 196 have?"
PROMPT = PROBLEM + "\nPlease reason step by step, and put your final answer within \\boxed{}."
MAX_TOKENS = int(os.environ.get("IK_MAX_TOKENS", "32768"))

HERE = Path(__file__).resolve().parent


def ask(slug: str) -> dict:
    """Stream one chat completion from the OpenAI-compatible endpoint (ik_live_).

    Plain HTTP so the request can carry presence_penalty; streaming because a
    looping answer can run for minutes."""
    base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
    url = f"{base}/endpoint/{os.environ['INFERENCEKEY_PROJECT']}/{slug}/v1/chat/completions"
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
        **{k: v for k, v in SAMPLING.items() if k != "top_k"},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {os.environ['INFERENCEKEY_API_KEY']}")
    req.add_header("Content-Type", "application/json")
    t0, first, parts, usage, finish = time.monotonic(), None, [], {}, None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                text = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if text and first is None:
                    first = time.monotonic() - t0
                parts.append(text)
                finish = choice.get("finish_reason") or finish
    seconds = time.monotonic() - t0
    text = "".join(parts)
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    tokens = usage.get("completion_tokens") or 0
    return {
        "slug": slug, "seconds": round(seconds, 1), "first_token_s": round(first or 0, 2),
        "completion_tokens": tokens, "tokens_per_s": round(tokens / seconds, 1) if seconds else None,
        "finish_reason": finish, "hit_limit": finish == "length",
        "answer": boxed[-1] if boxed else None, "text": text,
    }


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()
    worker = os.environ["IK_WORKER_ID"]  # your NVIDIA worker; never hard-code it
    try:
        # 1) Provision / reconcile. Idempotent by slug.
        for slug, name, model in WORKLOADS:
            mgmt.ensure(WorkloadSpec(
                name=name, slug=slug, model=model, backend=Backend.VLLM,
                command=command(model), vllm_version=VLLM_VERSION, task_type="text2text",
                worker_id=worker, execution_policy=ExecutionPolicy.FIXED,
            ))
            print(f"ensured {slug}")

        # 2) Wait until vLLM serves. A cold worker installs vLLM (a few
        #    minutes) and downloads 1.7 GB of weights per model.
        t0 = time.monotonic()
        for slug, _, _ in WORKLOADS:
            mgmt.wait_until_ready(slug, timeout=2400)
            print(f"{slug} ready after {time.monotonic() - t0:.1f} s")

        # 3) Same problem, same sampling, thinking on.
        outputs = HERE / "outputs"
        outputs.mkdir(exist_ok=True)
        for slug, _, _ in WORKLOADS:
            r = ask(slug)
            print(
                f"\n{slug}: answer={r['answer']!r} (expected 9), {r['completion_tokens']} tokens "
                f"in {r['seconds']} s ({r['tokens_per_s']} tokens/s), first token {r['first_token_s']} s, "
                f"hit the token limit: {r['hit_limit']}"
            )
            print("   ...", r["text"][-300:].replace("\n", " "))
            (outputs / f"{slug}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        print(f"saved {outputs}")
    except InferenceKeyError as e:
        # Don't print tokens; surface a clean message.
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        # 4) Tear down so the reserved GPU stops billing, unless you want to
        #    keep calling it (IK_KEEP_ALIVE=1).
        if os.environ.get("IK_KEEP_ALIVE") != "1":
            for slug, _, _ in WORKLOADS:
                print(f"deleted {slug}={mgmt.delete(slug)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

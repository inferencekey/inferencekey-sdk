"""Serve Clover-35B-A3B-Preview (DiffuSpace, official Q4_0 GGUF) with the
llama.cpp backend on one private 24 GB NVIDIA GPU, then ask it three things:
the model card's own question, a compound dispatch message to split into
structured tasks, and a whole interactive web page in a single HTML file.

The canonical SDK shape: ensure() -> wait_until_ready() -> calls. The axis this
example varies: a brand-new architecture that needs a pinned llama.cpp release
(`INFERENCEKEY_LLAMACPP_TAG=v0.9.0` in the command), served by one stock
llama-server process with an OpenAI-compatible endpoint.

Clover-35B-A3B-Preview: a block-diffusion language model derived from
Qwen3.6-35B-A3B, 35B mixture-of-experts with about 3B active per token, 128K
context, Apache-2.0. Important: llama.cpp v0.9.0 runs it in plain
autoregressive (AR) mode only. The diffusion and self-speculative decoding the
authors use for their speed numbers live in their SGLang branch (not merged)
and in a separate llama.cpp change (not merged). See README.md.

Run:  cp .env.example .env   # fill in real values, then
      python main.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

from inferencekey import (
    Backend,
    DataClient,
    ExecutionPolicy,
    InferenceKeyError,
    ManagementClient,
    WorkloadSpec,
)

REPO = "Diffuspace/Clover-35B-A3B-Preview-GGUF"
# The official Q4_0 GGUF (19.6 GB) does NOT load as published in upstream
# llama.cpp v0.9.0. Run patch_gguf.py on the worker host first (README, step 1):
# it writes this file, a byte-for-byte copy with three metadata/tensor fixes.
GGUF_PATH = os.environ.get("CLOVER_GGUF", "/root/models/Clover-35B-A3B-Preview-Q4_0-ar.gguf")
CONTEXT = int(os.environ.get("CLOVER_CONTEXT", "16384"))
LLAMACPP_TAG = os.environ.get("CLOVER_LLAMACPP_TAG", "v0.9.0")

# llama-server launch command, run verbatim by the `llamacpp` backend.
#   INFERENCEKEY_LLAMACPP_TAG=v0.9.0 -> the worker builds this llama.cpp release
#     (the first one that knows the `clover-moe` architecture; merged 2026-11-02).
#   -m GGUF_PATH      -> the patched Q4_0 GGUF on the worker's local disk.
#   -ngl 99 -fa on    -> every layer on the GPU, flash attention.
#   -c CONTEXT        -> 16,384 tokens by default; 131,072 also fits 24 GB (README).
#   --jinja --reasoning-format deepseek -> the model's own chat template; the
#     thinking goes to `reasoning_content`, the answer to `content`.
#   --presence-penalty 0.9 --repeat-penalty 1.05 -> the authors' repetition
#     controls from generation_config.json (their diffusion-only length and
#     think-end biases have no llama.cpp equivalent).
# The worker injects the listen port; do not hard-code --port.
COMMAND = (
    f"INFERENCEKEY_LLAMACPP_TAG={LLAMACPP_TAG} "
    f"llama-server -m {GGUF_PATH} --alias clover-35b-a3b "
    f"-ngl 99 -fa on -c {CONTEXT} --jinja --reasoning-format deepseek "
    "--presence-penalty 0.9 --repeat-penalty 1.05"
)

SLUG = "clover-35b-a3b"
HERE = Path(__file__).resolve().parent

# A compound instruction in the spirit of the authors' UAV dispatch demo:
# one message, five independent orders, to be split into structured tasks.
DISPATCH = (
    "Control message 14:05. Lift the no-fly restriction over Riverside Park. "
    "UAV3: change waypoint 2 to the north gate of the stadium. "
    "UAV5 takes the bridge mapping survey, not a delivery. "
    "Shrink the no-fly radius around the hospital helipad from 800 m to 500 m. "
    "Emergency: send the nearest idle drone to the warehouse fire on Dock Street."
)
DISPATCH_SYSTEM = (
    "You parse drone control messages. Reply with ONLY a JSON array, one object per "
    'order, each {"type": one of lift_no_fly | change_waypoint | assign_mission | '
    'resize_no_fly | emergency_dispatch, "target": string, "details": string}.'
)

# The authors' GenUI showcase, as a prompt: a whole desktop in one HTML file.
GENUI = (
    "Write a single self-contained HTML file (inline CSS and JavaScript, no external "
    "files) that simulates a small desktop: a dock at the bottom with icons for "
    "Notes, Calculator and Clock; clicking an icon opens that app in a window that "
    "can be dragged by its title bar and closed. Notes is a textarea, Calculator "
    "does + - x /, Clock shows the live time. Reply with only the HTML."
)


def chat(url: str, api_key: str, messages: list, max_tokens: int, **extra) -> dict:
    """Streaming call to the OpenAI-compatible endpoint, reassembled. Streaming
    keeps `reasoning_content` (the thinking) separate from `content`."""
    out = {"content": "", "reasoning_content": ""}
    usage, t0, first = {}, time.monotonic(), None
    body = {"model": "default", "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0, **extra}
    with requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=body,
                       stream=True, timeout=1800) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            line = raw.decode("utf-8")
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                for k in ("content", "reasoning_content"):
                    if delta.get(k):
                        first = first or time.monotonic()
                        out[k] += delta[k]
    total = time.monotonic() - t0
    toks = usage.get("completion_tokens", 0)
    out.update(seconds=round(total, 2), first_token_s=round((first or t0) - t0, 2),
               completion_tokens=toks,
               tokens_per_s=round(toks / max(total - ((first or t0) - t0), 1e-6), 1))
    return out


def strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()


def main() -> int:
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="Clover-35B-A3B-Preview (Q4_0, llama.cpp)",
        slug=SLUG,
        model=REPO,
        backend=Backend.LLAMACPP,
        command=COMMAND,
        task_type="text2text",
        worker_id=os.environ["IK_WORKER_ID"],
        execution_policy=ExecutionPolicy.FIXED,
    )
    outputs = HERE / "outputs"
    outputs.mkdir(exist_ok=True)
    results: dict = {}

    try:
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        t0 = time.monotonic()
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.monotonic() - t0:.1f} s")

        # 1. The SDK's plain call, with the model card's own question.
        ep = DataClient.from_env().endpoint(ref.workload_slug)
        t1 = time.monotonic()
        out = ep.generate_text(messages=[{"role": "user", "content": "What is diffusion? Answer in three sentences."}],
                               max_tokens=8192, temperature=0)
        print(f"\nwhat is diffusion ({time.monotonic() - t1:.1f} s):\n{out.text.strip()}")
        results["what_is_diffusion"] = out.text

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{ref.project_slug}/{ref.workload_slug}/v1/chat/completions"
        key = os.environ["INFERENCEKEY_API_KEY"]

        # 2. Compound dispatch message -> structured tasks.
        d = chat(url, key, [{"role": "system", "content": DISPATCH_SYSTEM},
                            {"role": "user", "content": DISPATCH}], max_tokens=8192)
        print(f"\ndispatch ({d['seconds']} s, {d['completion_tokens']} tokens, {d['tokens_per_s']} tok/s):")
        print(d["content"].strip())
        results["dispatch"] = d

        # 3. A whole interactive page in one HTML file.
        g = chat(url, key, [{"role": "user", "content": GENUI}], max_tokens=12288)
        html = strip_fence(g["content"])
        (outputs / "desktop.html").write_text(html + "\n")
        print(f"\ngenui ({g['seconds']} s, {g['completion_tokens']} tokens, {g['tokens_per_s']} tok/s): "
              f"{len(html):,} characters -> {outputs / 'desktop.html'}")
        results["genui"] = {k: v for k, v in g.items() if k != "content"}

        (outputs / "answers.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nsaved {outputs / 'answers.json'}")
    except (InferenceKeyError, requests.RequestException) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        if os.environ.get("IK_KEEP_ALIVE") != "1":
            print(f"deleted={mgmt.delete(SLUG)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

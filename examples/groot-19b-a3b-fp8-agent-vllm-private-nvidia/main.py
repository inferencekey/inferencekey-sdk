"""Serve Groot-19B-A3B (official FP8 build) with the stock vLLM backend on one
private NVIDIA GPU and use it the way its authors pitch it: a reasoning model
that calls tools and reads images, with its full 256K context on one card.

The canonical SDK shape: ensure() -> wait_until_ready() -> calls. The axis this
example varies: a hybrid-attention vision MoE served with the authors' own
vLLM recipe (reasoning parser, tool-call parser, multi-token prediction) on a
single 32 GB consumer card instead of the two 80 GB GPUs of the model card.

Groot-19B-A3B-261023: 19B mixture-of-experts (about 3B active per token, 128
experts, 8 per token), post-trained from Qwen3.6-35B-A3B, Apache-2.0. The FP8
build (block-wise E4M3) is 20.9 GB; 30 of its 40 layers use linear attention,
so the KV cache of the full 262,144-token window is about 5 GiB.

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

MODEL = "GrootTeam/Groot-19B-A3B-261023-FP8"
CONTEXT = int(os.environ.get("GROOT_CONTEXT", "262144"))
MTP = os.environ.get("GROOT_MTP", "1") == "1"

# vLLM launch command, run verbatim by the `vllm` backend (the worker injects
# host and port; do not hard-code --port). It is the model card's vLLM recipe
# for one GPU instead of two:
#   --max-model-len          -> CONTEXT tokens (262,144 = the full native window).
#   --gpu-memory-utilization -> 0.92 of the card for weights + KV cache.
#   --max-num-seqs 8         -> a private endpoint, a few requests at a time.
#   --limit-mm-per-prompt    -> up to 2 images, no video (smaller warm-up profile).
#   --reasoning-parser qwen3 -> thinking in `reasoning_content`, answer in `content`.
#   --enable-auto-tool-choice --tool-call-parser qwen3_coder -> OpenAI `tool_calls`.
#   --speculative-config     -> multi-token prediction with the model's MTP layer.
COMMAND = (
    f"vllm serve {MODEL} --max-model-len {CONTEXT} --gpu-memory-utilization 0.92 "
    "--max-num-seqs 8 --enable-prefix-caching "
    "--limit-mm-per-prompt '{\"image\": 2, \"video\": 0}' "
    "--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
)
if MTP:
    COMMAND += " --speculative-config '{\"method\": \"qwen3_next_mtp\", \"num_speculative_tokens\": 2}'"

# vLLM >= 0.19.1 serves the model (card). 0.32.0 pins transformers 5.16-5.19
# (the card needs >= 5.8.1) and torch 2.13 cu13: the worker needs a CUDA 13 driver.
VLLM_VERSION = "0.32.0"

SLUG = "groot-19b-a3b"
HERE = Path(__file__).resolve().parent

# The authors' recommended sampling for thinking mode (model card). top_k=20 is
# vLLM's default for this model (generation_config.json).
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5}

# The model card's tool-calling example.
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}]

# The authors' own size-vs-capability chart from the model card.
CHART = "https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/resolve/main/assets/scatter_19b.png"


def chat(url: str, api_key: str, body: dict) -> dict:
    """Streaming call to the OpenAI-compatible endpoint, reassembled into one
    message. Streaming keeps `reasoning_content` separate from `content`; tool
    calls arrive in pieces."""
    msg = {"content": "", "reasoning_content": "", "tool_calls": {}}
    usage, t0, first = {}, time.monotonic(), None
    with requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, stream=True, timeout=1800,
                       json={"model": MODEL, "stream": True, "stream_options": {"include_usage": True},
                             **SAMPLING, **body}) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            line = raw.decode("utf-8")
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if text and first is None:
                    first = time.monotonic() - t0
                msg["content"] += delta.get("content") or ""
                msg["reasoning_content"] += delta.get("reasoning_content") or ""
                for tc in delta.get("tool_calls") or []:
                    slot = msg["tool_calls"].setdefault(tc.get("index", 0), {
                        "id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    slot["id"] = tc.get("id") or slot["id"]
                    fn = tc.get("function") or {}
                    slot["function"]["name"] += fn.get("name") or ""
                    slot["function"]["arguments"] += fn.get("arguments") or ""
    msg["tool_calls"] = [msg["tool_calls"][i] for i in sorted(msg["tool_calls"])]
    msg["seconds"] = round(time.monotonic() - t0, 2)
    msg["first_token_s"] = round(first, 2) if first is not None else None
    msg["usage"] = usage
    done = (usage or {}).get("completion_tokens") or 0
    if first is not None and done and msg["seconds"] > first:
        msg["tokens_per_s"] = round(done / (msg["seconds"] - first), 1)
    return msg


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="Groot-19B-A3B (FP8, vLLM)",
        slug=SLUG,
        model=MODEL,
        backend=Backend.VLLM,
        command=COMMAND,
        vllm_version=VLLM_VERSION,
        task_type="text2text",
        # Pin to your NVIDIA worker (32 GB or more). Never hard-code the id.
        worker_id=os.environ["IK_WORKER_ID"],
        execution_policy=ExecutionPolicy.FIXED,
    )
    outputs = HERE / "outputs"
    outputs.mkdir(exist_ok=True)
    results: dict = {"command": COMMAND}

    try:
        # 1) Provision / reconcile. Idempotent by slug.
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # 2) Wait until vLLM serves. A cold worker installs vLLM and downloads
        #    the 20.9 GB of FP8 weights first, so give it time.
        t0 = time.monotonic()
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        results["ready_s"] = round(time.monotonic() - t0, 1)
        print(f"ready after {results['ready_s']} s")

        # 3a) A plain question through the SDK (ik_live_). The model thinks
        #     before it answers: leave room, or the reply is cut in the thinking.
        ep = DataClient.from_env().endpoint(ref.workload_slug)
        t1 = time.monotonic()
        out = ep.generate_text(
            messages=[{"role": "user", "content": "In two sentences: what is a mixture-of-experts language model?"}],
            max_tokens=8192, temperature=1.0,
        )
        print(f"\nanswer ({time.monotonic() - t1:.1f} s):\n{out.text.strip()}")
        results["plain_answer"] = out.text

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{ref.project_slug}/{ref.workload_slug}/v1/chat/completions"
        key = os.environ["INFERENCEKEY_API_KEY"]

        # 3b) The model card's first example, streamed: thinking and answer apart.
        r = chat(url, key, {"messages": [{"role": "user", "content": 'Type "I love Groot" backwards'}],
                            "max_tokens": 8192})
        print(f"\nbackwards ({r['seconds']} s, {r['usage'].get('completion_tokens')} tokens, "
              f"{r.get('tokens_per_s')} tok/s): {r['content'].strip()}")
        results["backwards"] = r

        # 3c) A tool call (the card's get_weather example), then the final answer.
        messages = [{"role": "user", "content": "What's the weather like in Paris right now?"}]
        r = chat(url, key, {"messages": messages, "tools": TOOLS, "max_tokens": 4096})
        print(f"\ntool call ({r['seconds']} s): {json.dumps(r['tool_calls'])}")
        results["tool_call"] = r
        if r["tool_calls"]:
            messages.append({"role": "assistant", "content": r["content"], "tool_calls": r["tool_calls"]})
            for c in r["tool_calls"]:  # your tool runs here, on your side
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": json.dumps({"city": "Paris", "temp_c": 14, "sky": "light rain"})})
            r = chat(url, key, {"messages": messages, "tools": TOOLS, "max_tokens": 4096})
            print(f"final ({r['seconds']} s): {r['content'].strip()}")
            results["tool_answer"] = r

        # 3d) Vision: read the authors' own chart.
        r = chat(url, key, {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": CHART}},
            {"type": "text", "text": "Which model is farthest to the left on this chart, and what average "
                                     "score does the chart give it? Answer in one sentence."}]}],
            "max_tokens": 8192})
        print(f"\nchart ({r['seconds']} s, {r['usage'].get('prompt_tokens')} prompt tokens): {r['content'].strip()}")
        results["chart"] = r

        (outputs / "answers.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nsaved {outputs / 'answers.json'}")
    except (InferenceKeyError, requests.RequestException) as e:
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

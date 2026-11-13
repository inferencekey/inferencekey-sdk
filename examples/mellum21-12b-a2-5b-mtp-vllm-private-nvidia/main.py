"""Serve Mellum2.1 Thinking with its multi-token prediction (MTP) head on one
private NVIDIA GPU with the stock vLLM backend, and use it the way JetBrains
pitches it: a thinking model for code, math and tool calls.

The canonical SDK shape: ensure() -> wait_until_ready() -> calls. The axis this
example varies: speculative decoding with the model's own MTP head
(`--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'`),
switched on or off with one environment variable so you can measure both on
your own card.

Mellum2.1-12B-A2.5B-Thinking-MTP: 12B mixture-of-experts (64 experts, 8 per
token, about 2.5B parameters active), 28 layers (3 of every 4 with a 1,024-token
sliding window), 131,072-token context, Apache-2.0. BF16 weights are 24.3 GB
plus a 0.86 GB MTP head; the weights are byte-identical to
JetBrains/Mellum2.1-12B-A2.5B-Thinking.

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

MODEL = "JetBrains/Mellum2.1-12B-A2.5B-Thinking-MTP"
CONTEXT = int(os.environ.get("MELLUM_CONTEXT", "131072"))
MTP = os.environ.get("MELLUM_MTP", "1") == "1"
# Draft tokens per step. The card uses 1; on our RTX A6000, 2 was the fastest
# (1.3x single stream instead of 1.1x), 3 was slower again.
SPEC_TOKENS = int(os.environ.get("MELLUM_SPEC_TOKENS", "1"))
# FP8 weights (vLLM's online quantization): halves the weights to about 12 GiB
# for 16 to 24 GB cards. BF16 (the default) needs a 32 GB card or larger.
FP8 = os.environ.get("MELLUM_FP8", "0") == "1"
GPU_UTIL = os.environ.get("MELLUM_GPU_UTIL", "0.90")

# vLLM launch command, run verbatim by the `vllm` backend (the worker injects
# host and port; do not hard-code --port). It is the model card's tool-calling
# recipe:
#   --max-model-len          -> CONTEXT tokens (131,072 = the full native window).
#   --gpu-memory-utilization -> share of the card for weights + KV cache.
#   --max-num-seqs 16        -> a private endpoint, a few requests at a time.
#   --reasoning-parser qwen3 -> thinking in `reasoning_content`, answer in `content`.
#   --enable-auto-tool-choice --tool-call-parser hermes -> OpenAI `tool_calls`.
#   --speculative-config     -> the MTP head drafts SPEC_TOKENS tokens, the model checks them.
COMMAND = (
    f"vllm serve {MODEL} --max-model-len {CONTEXT} --gpu-memory-utilization {GPU_UTIL} "
    "--max-num-seqs 16 --reasoning-parser qwen3 "
    "--enable-auto-tool-choice --tool-call-parser hermes"
)
if FP8:
    COMMAND += " --quantization fp8"
if MTP:
    COMMAND += f" --speculative-config '{{\"method\": \"mtp\", \"num_speculative_tokens\": {SPEC_TOKENS}}}'"

# The card asks for vLLM >= 0.33 and transformers >= 5.20 (older vLLM refuses
# the MTP tensors). vLLM 0.33.0 installs transformers 5.20 and torch 2.14 built
# for CUDA 13: the worker needs a CUDA 13 driver.
VLLM_VERSION = "0.33.0"

SLUG = "mellum21-thinking"
HERE = Path(__file__).resolve().parent

# The card's sampling for chat (Quickstart). top_k=20 cannot travel through the
# endpoint; temperature and top_p do.
SAMPLING = {"temperature": 0.6, "top_p": 0.95}

# The model card's Quickstart prompt.
CARD_PROMPT = "Find the bug in this function and explain the fix: def mean(xs): return sum(xs) / len(xs) - 1"

CODE_PROMPT = (
    "Write a Python function merge_intervals(intervals) that takes a list of [start, end] pairs and "
    "returns the merged, sorted list of non-overlapping intervals (touching intervals merge). "
    "Answer with only the code in one ```python block."
)

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}]


def chat(url: str, api_key: str, body: dict) -> dict:
    """Streaming call to the OpenAI-compatible endpoint, reassembled into one
    message. Streaming keeps `reasoning_content` apart from `content`, and the
    timing gives first-token latency and decode speed."""
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
                if (text or delta.get("tool_calls")) and first is None:
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
        name=f"Mellum2.1 Thinking ({'FP8' if FP8 else 'BF16'}, {'MTP' if MTP else 'no MTP'}, vLLM)",
        slug=SLUG,
        model=MODEL,
        backend=Backend.VLLM,
        command=COMMAND,
        vllm_version=VLLM_VERSION,
        task_type="text2text",
        # Pin to your NVIDIA worker. Never hard-code the id.
        worker_id=os.environ["IK_WORKER_ID"],
        execution_policy=ExecutionPolicy.FIXED,
    )
    outputs = HERE / "outputs"
    outputs.mkdir(exist_ok=True)
    results: dict = {"command": COMMAND}

    try:
        # 1) Provision / reconcile. Idempotent by slug: changing MELLUM_MTP or
        #    MELLUM_FP8 and running again restarts the server with the new command.
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug} (MTP {'on' if MTP else 'off'})")

        # 2) Wait until vLLM serves. A cold worker installs vLLM and downloads
        #    25 GB of weights first, so give it time.
        t0 = time.monotonic()
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        results["ready_s"] = round(time.monotonic() - t0, 1)
        print(f"ready after {results['ready_s']} s")

        # 3a) The card's own question through the SDK (ik_live_). The model
        #     thinks first: leave room, or the reply is cut in the thinking.
        ep = DataClient.from_env().endpoint(ref.workload_slug)
        t1 = time.monotonic()
        out = ep.generate_text(messages=[{"role": "user", "content": CARD_PROMPT}],
                               max_tokens=16384, temperature=0.6)
        print(f"\ncard question ({time.monotonic() - t1:.1f} s):\n{out.text.strip()}")
        results["card_answer"] = out.text

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{ref.project_slug}/{ref.workload_slug}/v1/chat/completions"
        key = os.environ["INFERENCEKEY_API_KEY"]

        # 3b) A coding task, streamed, with first-token latency and decode speed.
        r = chat(url, key, {"messages": [{"role": "user", "content": CODE_PROMPT}], "max_tokens": 16384})
        print(f"\ncode ({r['seconds']} s, {r['usage'].get('completion_tokens')} tokens, "
              f"{r.get('tokens_per_s')} tok/s):\n{r['content'].strip()}")
        results["code"] = r

        # 3c) A small agent loop (hermes tool-call format): the model calls the
        #     tool, gets the result, and decides whether to call it again. It
        #     tends to call one city per turn, so allow a few turns.
        messages = [{"role": "user", "content": "Is it warmer in Lisbon or in Oslo right now?"}]
        weather = {"Lisbon": {"temp_c": 19, "sky": "sunny"}, "Oslo": {"temp_c": 4, "sky": "cloudy"}}
        results["agent_turns"] = []
        for turn in range(1, 5):
            r = chat(url, key, {"messages": messages, "tools": TOOLS, "max_tokens": 8192})
            results["agent_turns"].append(r)
            if not r["tool_calls"]:
                print(f"final answer, turn {turn} ({r['seconds']} s): {r['content'].strip()}")
                break
            print(f"turn {turn} ({r['seconds']} s): {json.dumps([c['function'] for c in r['tool_calls']])}")
            messages.append({"role": "assistant", "content": r["content"], "tool_calls": r["tool_calls"]})
            for c in r["tool_calls"]:  # your tool runs here, on your side
                city = json.loads(c["function"]["arguments"] or "{}").get("city", "")
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": json.dumps({"city": city, **weather.get(city, {})})})

        (outputs / f"answers-{'mtp' if MTP else 'plain'}.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nsaved {outputs / ('answers-' + ('mtp' if MTP else 'plain') + '.json')}")
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

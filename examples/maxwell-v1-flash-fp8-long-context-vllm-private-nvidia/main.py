"""Serve Maxwell-V1.0-Flash (official FP8 build) with the stock vLLM backend on
one private 48 GB NVIDIA GPU and use it the way its authors pitch it: long
documents read in one pass, thinking before answering, and tool calls.

The canonical SDK shape: ensure() -> wait_until_ready() -> calls. The axis this
example varies: a model whose serving code lives in its authors' vLLM fork,
on a single Ada card (sm_89) with an FP8 KV cache, so a 128K-token prompt
fits next to the 38.6 GB of weights.

Maxwell-V1.0-Flash: 35B-total, 3B-active mixture-of-experts (256 routed
experts, 12 per token, 2 shared), 45 layers, trained from scratch on about 26T
tokens, Apache-2.0. Its block-sparse attention (MBSA) attends to at most 32
blocks of 128 tokens (4,096 tokens) per KV group, so the attention work per
generated token stops growing with the context. The KV cache still keeps
every token: memory grows with the context, compute does not.

Before the first run, prepare the worker once with `prepare_worker.sh` (see the
README): the fork is not on PyPI, so the script installs the upstream vLLM
wheel the fork is based on and overlays the fork's Python files.

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

MODEL = "maxwell-model/Maxwell-V1-Flash-FP8"
REVISION = "6f91969de0d39c9fdbb98306e8826ba4f661125b"
CONTEXT = int(os.environ.get("MAXWELL_CONTEXT", "131072"))
KV_DTYPE = os.environ.get("MAXWELL_KV_DTYPE", "fp8")
GPU_UTIL = os.environ.get("MAXWELL_GPU_UTIL", "0.965")
HF_HOME = os.environ.get("MAXWELL_HF_HOME", "")

# vLLM launch command, run verbatim by the `vllm` backend (the worker injects
# host and port; do not hard-code --port). It is the model card's command for
# one GPU, plus what a 48 GB card needs:
#   --trust-remote-code --block-size 128 -> as on the card (128-token sparse blocks).
#   --max-model-len          -> CONTEXT tokens (131,072 by default).
#   --kv-cache-dtype fp8     -> halves the KV cache so 128K fits next to the weights.
#   --gpu-memory-utilization -> share of the card for weights + KV cache.
#   --max-num-seqs 4         -> a private endpoint, a few requests at a time.
#   --reasoning-parser qwen3 -> thinking in `reasoning_content`, answer in `content`.
#   --enable-auto-tool-choice --tool-call-parser qwen3_coder -> OpenAI `tool_calls`.
COMMAND = (
    (f"HF_HOME={HF_HOME} " if HF_HOME else "")
    + f"vllm serve {MODEL} --revision {REVISION} --trust-remote-code --block-size 128 "
    f"--max-model-len {CONTEXT} --kv-cache-dtype {KV_DTYPE} --gpu-memory-utilization {GPU_UTIL} "
    "--max-num-seqs 4 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
)
if os.environ.get("MAXWELL_EXTRA_ARGS"):
    COMMAND += " " + os.environ["MAXWELL_EXTRA_ARGS"]

# The version prepare_worker.sh installs: upstream vLLM at the commit the
# authors' fork is based on (a9eafde5), with the fork's Python on top. The
# worker sees it already installed and skips its own pip install.
VLLM_VERSION = "0.30.1rc1.dev219+ga9eafde59"

SLUG = "maxwell-v1-flash"
HERE = Path(__file__).resolve().parent

# The authors' sampling for their evaluations (model card).
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

TOOLS = [{"type": "function", "function": {
    "name": "get_contract", "description": "Fetch a contract's metadata from the document system",
    "parameters": {"type": "object", "properties": {
        "contract_id": {"type": "string", "description": "Contract id, e.g. MSA-2291"}},
        "required": ["contract_id"]}}}]


def chat(url: str, api_key: str, body: dict) -> dict:
    """Streaming call to the OpenAI-compatible endpoint, reassembled into one
    message, with the time to the first token and the decode speed."""
    msg = {"content": "", "reasoning_content": "", "tool_calls": {}}
    usage, t0, first = {}, time.monotonic(), None
    with requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, stream=True, timeout=3600,
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
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or "") \
                    + (delta.get("reasoning") or "")
                if (text or delta.get("tool_calls")) and first is None:
                    first = time.monotonic() - t0
                msg["content"] += delta.get("content") or ""
                msg["reasoning_content"] += (delta.get("reasoning_content") or delta.get("reasoning") or "")
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
    if first is not None and done > 1 and msg["seconds"] > first:
        msg["tokens_per_s"] = round((done - 1) / (msg["seconds"] - first), 1)
    return msg


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="Maxwell-V1.0-Flash (FP8, vLLM)",
        slug=SLUG,
        model=MODEL,
        backend=Backend.VLLM,
        command=COMMAND,
        vllm_version=VLLM_VERSION,
        task_type="text2text",
        # Pin to your NVIDIA worker (48 GB). Never hard-code the id.
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

        # 2) Wait until vLLM serves (weights download, load, warm-up).
        t0 = time.monotonic()
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        results["ready_s"] = round(time.monotonic() - t0, 1)
        print(f"ready after {results['ready_s']} s")

        # 3a) The model card's own question, through the SDK (ik_live_).
        ep = DataClient.from_env().endpoint(ref.workload_slug)
        t1 = time.monotonic()
        out = ep.generate_text(
            messages=[{"role": "user", "content": "Why is it harder to keep a room cool when the door is open "
                                                  "on a hot day? Explain in 3-4 simple sentences."}],
            max_tokens=4096, temperature=0.6,
        )
        print(f"\nanswer ({time.monotonic() - t1:.1f} s):\n{out.text.strip()}")
        results["card_question"] = out.text

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{ref.project_slug}/{ref.workload_slug}/v1/chat/completions"
        key = os.environ["INFERENCEKEY_API_KEY"]

        # 3b) A long document in one pass. Point MAXWELL_DOCUMENT at a text file
        #     (a contract bundle, a filing, a codebase dump) of up to ~125K tokens.
        doc_path = os.environ.get("MAXWELL_DOCUMENT")
        if doc_path:
            doc = Path(doc_path).read_text()
            r = chat(url, key, {"messages": [{"role": "user", "content": doc}], "max_tokens": 4096})
            print(f"\ndocument ({r['usage'].get('prompt_tokens')} prompt tokens, first token after "
                  f"{r['first_token_s']} s, {r.get('tokens_per_s')} tok/s):\n{r['content'].strip()}")
            results["document"] = r

        # 3c) A tool call, then the final answer with the tool result.
        messages = [{"role": "user", "content": "Who signed contract MSA-2291 for the supplier, and when does it renew?"}]
        r = chat(url, key, {"messages": messages, "tools": TOOLS, "max_tokens": 4096})
        print(f"\ntool call ({r['seconds']} s): {json.dumps(r['tool_calls'])}")
        results["tool_call"] = r
        if r["tool_calls"]:
            # Pass the whole assistant turn back, thinking included (model card).
            messages.append({"role": "assistant", "content": r["content"],
                             "reasoning_content": r["reasoning_content"], "tool_calls": r["tool_calls"]})
            for c in r["tool_calls"]:  # your tool runs here, on your side
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(
                    {"contract_id": "MSA-2291", "supplier_signatory": "Dana Whitfield, COO",
                     "signed": "2025-03-14", "renews": "2027-03-14", "auto_renewal": True})})
            r = chat(url, key, {"messages": messages, "tools": TOOLS, "max_tokens": 4096})
            print(f"final ({r['seconds']} s): {r['content'].strip()}")
            results["tool_answer"] = r

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

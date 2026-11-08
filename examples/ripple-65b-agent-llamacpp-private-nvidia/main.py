"""Serve Ripple-65B Preview (Ricursive Intelligence, Apache-2.0, dense 65B
agentic coding model trained from scratch) from a 4-bit GGUF with the
llama.cpp backend on one private 48 GB NVIDIA GPU, then ask it two things: a
coding task, and one agent step with the tool from the model's own chat
template (`execute_bash`), whose result goes back to the model.

The canonical SDK shape: ensure() -> wait_until_ready() -> calls. The axis this
example varies: a dense 65B model on a single card, every one of its 80 layers
on the GPU, with a llama.cpp build new enough to know its architecture.

Ripple-65B Preview: 80 layers (64 sliding-window layers with a 1,024-token
window, 16 global layers without positional encoding), 8 KV heads, up to 1M
tokens of context per the authors. Only the 16 global layers keep a full KV
cache, so long context is cheap: about 64 KiB per token at f16. See README.md
for what we measured on an RTX A6000.

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

REPO = "AtomicChat/Ripple-65B-Preview-GGUF"
GGUF_FILE = os.environ.get("RIPPLE_GGUF_FILE", "Ripple-65B-Preview-Q4_K_M.gguf")
CONTEXT = int(os.environ.get("RIPPLE_CONTEXT", "32768"))
KV_TYPE = os.environ.get("RIPPLE_KV_TYPE", "f16")
LLAMACPP_TAG = os.environ.get("RIPPLE_LLAMACPP_TAG", "b12167")
# The prebuilt CUDA 13.4 build from the llama.cpp release, instead of the
# worker's default on NVIDIA (compiling llama.cpp from source, minutes). It needs
# a driver for CUDA 13 and the CUDA 13 runtime libraries (libcudart.so.13,
# libcublas.so.13) on the worker. Set RIPPLE_LLAMACPP_ASSET= (empty) to build
# from source instead.
LLAMACPP_ASSET = os.environ.get("RIPPLE_LLAMACPP_ASSET", "ubuntu-cuda-13.4-x64")

# llama-server launch command, run verbatim by the `llamacpp` backend.
#   INFERENCEKEY_LLAMACPP_TAG=b12167 -> the first llama.cpp release with the
#     `ricursive` architecture (PR #31692). Older builds refuse the file.
#   INFERENCEKEY_LLAMACPP_ASSET=ubuntu-cuda-13.4-x64 -> install that release
#     tarball instead of building (see LLAMACPP_ASSET above).
#   -hf / --hf-file   -> the 38.8 GB Q4_K_M GGUF (weights, tokenizer, template).
#   -ngl 99 -fa on    -> all 80 layers on the GPU, flash attention.
#   -c CONTEXT        -> 32,768 tokens by default (about 2 GiB of f16 KV cache).
#     With -ctk/-ctv q8_0 the cache halves (README: what fits on 48 GB).
#   --jinja --reasoning-format deepseek -> the model's own chat template, which
#     renders the tools; any thinking goes to `reasoning_content`.
# The worker injects the listen port; do not hard-code --port.
PINS = f"INFERENCEKEY_LLAMACPP_TAG={LLAMACPP_TAG} "
if LLAMACPP_ASSET:
    PINS += f"INFERENCEKEY_LLAMACPP_ASSET={LLAMACPP_ASSET} "
COMMAND = (
    PINS
    + f"llama-server -hf {REPO} --hf-file {GGUF_FILE} --alias ripple-65b-preview "
    f"-ngl 99 -fa on -c {CONTEXT} -ctk {KV_TYPE} -ctv {KV_TYPE} "
    "--jinja --reasoning-format deepseek"
)

SLUG = "ripple-65b"
HERE = Path(__file__).resolve().parent
# Ricursive publishes no default sampling for the Preview; these are ours.
SAMPLING = {"temperature": 0.6, "top_p": 0.95}

BASH_TOOL = {  # the tool in the model's own chat template example
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": "Run a bash command in the repository root and return its stdout, stderr and exit code.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string", "description": "The bash command to run"}},
                       "required": ["command"]},
    },
}
CODE_TASK = ("Write a Python function merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]] "
             "that merges overlapping closed intervals, plus three assert-based tests. Reply with only the code.")


def chat(url: str, api_key: str, messages: list, max_tokens: int, **extra) -> dict:
    """Streaming call to the OpenAI-compatible endpoint, reassembled (text,
    thinking and tool calls)."""
    out: dict = {"content": "", "reasoning_content": "", "tool_calls": []}
    calls: dict = {}
    usage, t0, first = {}, time.monotonic(), None
    body = {"model": "default", "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, **SAMPLING, **extra}
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
                for tc in delta.get("tool_calls") or []:
                    first = first or time.monotonic()
                    c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    c["id"] = tc.get("id") or c["id"]
                    fn = tc.get("function") or {}
                    c["name"] += fn.get("name") or ""
                    c["arguments"] += fn.get("arguments") or ""
    total = time.monotonic() - t0
    toks = usage.get("completion_tokens", 0)
    out["tool_calls"] = [calls[i] for i in sorted(calls)]
    out.update(seconds=round(total, 2), first_token_s=round((first or t0) - t0, 2),
               prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=toks,
               tokens_per_s=round(toks / max(total - ((first or t0) - t0), 1e-6), 1))
    return out


def main() -> int:
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name=f"Ripple-65B Preview ({GGUF_FILE.removeprefix('Ripple-65B-Preview-').removesuffix('.gguf')}, llama.cpp)",
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
    results: dict = {"command": COMMAND}

    try:
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # A cold worker installs llama.cpp and downloads 38.8 GB first.
        t0 = time.monotonic()
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        results["ready_s"] = round(time.monotonic() - t0, 1)
        print(f"ready after {results['ready_s']} s")

        # 1. The SDK's plain call: a small coding task.
        ep = DataClient.from_env().endpoint(ref.workload_slug)
        t1 = time.monotonic()
        out = ep.generate_text(messages=[{"role": "user", "content": CODE_TASK}],
                               max_tokens=4096, temperature=SAMPLING["temperature"])
        print(f"\ncoding task ({time.monotonic() - t1:.1f} s):\n{out.text.strip()}")
        results["code"] = {"seconds": round(time.monotonic() - t1, 2), "text": out.text}

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{ref.project_slug}/{ref.workload_slug}/v1/chat/completions"
        key = os.environ["INFERENCEKEY_API_KEY"]

        # 2. One agent step with the template's own tool. The tool result here
        #    is canned; a real agent would run the command in a sandbox.
        msgs = [{"role": "user", "content": "Which directory am I in? Use the tool."}]
        a = chat(url, key, msgs, max_tokens=2048, tools=[BASH_TOOL])
        print(f"\ntool call ({a['seconds']} s, {a['completion_tokens']} tokens): "
              f"{[(c['name'], c['arguments']) for c in a['tool_calls']]}")
        results["tool_call"] = a
        if a["tool_calls"]:
            msgs.append({"role": "assistant", "content": a["content"] or None,
                         "tool_calls": [{"id": c["id"] or f"call_{i}", "type": "function",
                                         "function": {"name": c["name"], "arguments": c["arguments"]}}
                                        for i, c in enumerate(a["tool_calls"])]})
            for i, c in enumerate(a["tool_calls"]):
                msgs.append({"role": "tool", "tool_call_id": c["id"] or f"call_{i}",
                             "content": "exit_code: 0\nstdout:\n/home/dev/inventory\n"})
            b = chat(url, key, msgs, max_tokens=2048, tools=[BASH_TOOL])
            print(f"answer ({b['seconds']} s, {b['tokens_per_s']} tok/s): {b['content'].strip()}")
            results["answer"] = b

        (outputs / "answers.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nsaved {outputs / 'answers.json'}")
    except (InferenceKeyError, requests.RequestException, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        if os.environ.get("IK_KEEP_ALIVE") != "1":
            print(f"deleted={mgmt.delete(SLUG)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

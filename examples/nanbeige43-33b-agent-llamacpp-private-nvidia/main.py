"""Serve Nanbeige4.3-33B (Nanbeige, Apache-2.0, dense 33B agent and coding
model) from a 4-bit GGUF with the llama.cpp backend on one private 24 GB
NVIDIA GPU, then ask it three things: the model card's own question, one tool
call (the tool from the GGUF card's prompt example), and a two-step agent turn
with parallel tool calls whose results go back to the model.

The canonical SDK shape: ensure() -> wait_until_ready() -> calls. The axis this
example varies: a dense 33B model that only fits a 24 GB card at 4 bits, with
every layer on the GPU and a quantized KV cache to buy context.

Nanbeige4.3-33B: Llama-architecture dense transformer, 52 layers, 8 KV heads,
512K native context, thinking on by default, XML tool calls (Qwen3-Coder
style, parsed by llama.cpp's --jinja templates). See README.md for what we
measured on an RTX 4090.

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

REPO = "bartowski/Nanbeige_Nanbeige4.3-33B-GGUF"
GGUF_FILE = os.environ.get("NANBEIGE_GGUF_FILE", "Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf")
CONTEXT = int(os.environ.get("NANBEIGE_CONTEXT", "8192"))
KV_TYPE = os.environ.get("NANBEIGE_KV_TYPE", "f16")
LLAMACPP_TAG = os.environ.get("NANBEIGE_LLAMACPP_TAG", "v0.9.0")

# llama-server launch command, run verbatim by the `llamacpp` backend.
#   INFERENCEKEY_LLAMACPP_TAG=v0.9.0 -> the worker builds this llama.cpp release
#     (the GGUF was quantized with b11845; v0.9.0 is newer).
#   -hf / --hf-file   -> the 20.8 GB Q4_K_M GGUF (weights, tokenizer, template).
#   -ngl 99 -fa on    -> all 52 layers on the GPU, flash attention.
#   -c CONTEXT        -> 8,192 tokens by default. The f16 KV cache costs about
#     208 KiB per token; with -ctk/-ctv q8_0 it halves (README: what fits).
#   --jinja --reasoning-format deepseek -> the model's own chat template, which
#     also renders tools; the thinking goes to `reasoning_content`.
# The worker injects the listen port; do not hard-code --port.
COMMAND = (
    f"INFERENCEKEY_LLAMACPP_TAG={LLAMACPP_TAG} "
    f"llama-server -hf {REPO} --hf-file {GGUF_FILE} --alias nanbeige4.3-33b "
    f"-ngl 99 -fa on -c {CONTEXT} -ctk {KV_TYPE} -ctv {KV_TYPE} "
    "--jinja --reasoning-format deepseek"
)

SLUG = "nanbeige43-33b"
HERE = Path(__file__).resolve().parent

# The card's recommended sampling: 1.0 / 0.95 for agentic and tool use,
# 0.6 / 0.95 for reasoning and chat.
AGENT = {"temperature": 1.0, "top_p": 0.95}
CHAT = {"temperature": 0.6, "top_p": 0.95}

STOCK_TOOL = {  # the tool in the GGUF card's prompt-format example
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the current stock price",
        "parameters": {"type": "object",
                       "properties": {"symbol": {"type": "string", "description": "The stock symbol, e.g. AAPL, GOOG"}},
                       "required": ["symbol"]},
    },
}
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}
FAKE_WEATHER = {"Lisbon": {"temp_c": 19, "sky": "sunny"}, "Oslo": {"temp_c": 4, "sky": "rain"}}


def chat(url: str, api_key: str, messages: list, max_tokens: int, **extra) -> dict:
    """Streaming call to the OpenAI-compatible endpoint, reassembled (text,
    thinking and tool calls). Streaming keeps `reasoning_content` separate."""
    out: dict = {"content": "", "reasoning_content": "", "tool_calls": []}
    calls: dict = {}
    usage, t0, first = {}, time.monotonic(), None
    body = {"model": "default", "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, **extra}
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


def answer_text(r: dict) -> str:
    """The answer, also when it lands in the thinking channel. With thinking on,
    the template opens <think> for the model; on some long or final answers the
    model skips thinking and never writes </think>, so llama.cpp returns the
    whole answer as `reasoning_content` and `content` stays empty (README,
    "Known issue"). Flag it instead of printing nothing."""
    if r["content"].strip():
        return r["content"].strip()
    return "[content empty, answer in reasoning_content] " + r["reasoning_content"].strip()


def as_message(r: dict) -> dict:
    """The assistant turn to send back with the tool results."""
    return {"role": "assistant", "content": r["content"] or None,
            "tool_calls": [{"id": c["id"] or f"call_{i}", "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]}}
                           for i, c in enumerate(r["tool_calls"])]}


def main() -> int:
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="Nanbeige4.3-33B (Q4_K_M, llama.cpp)",
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

        # A cold worker builds llama.cpp and downloads 20.8 GB first.
        t0 = time.monotonic()
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        results["ready_s"] = round(time.monotonic() - t0, 1)
        print(f"ready after {results['ready_s']} s")

        # 1. The SDK's plain call, with the model card's own question.
        ep = DataClient.from_env().endpoint(ref.workload_slug)
        t1 = time.monotonic()
        out = ep.generate_text(messages=[{"role": "user", "content": "Which number is bigger, 9.11 or 9.8?"}],
                               max_tokens=8192, temperature=CHAT["temperature"])
        print(f"\n9.11 or 9.8 ({time.monotonic() - t1:.1f} s):\n{out.text.strip()}")
        results["card_question"] = {"seconds": round(time.monotonic() - t1, 2), "text": out.text}

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{ref.project_slug}/{ref.workload_slug}/v1/chat/completions"
        key = os.environ["INFERENCEKEY_API_KEY"]

        # 2. One tool call: the tool from the GGUF card's prompt example.
        s = chat(url, key, [{"role": "user", "content": "What is Apple trading at right now?"}],
                 max_tokens=4096, tools=[STOCK_TOOL], **AGENT)
        print(f"\ntool call ({s['seconds']} s, {s['completion_tokens']} tokens): {s['tool_calls']}")
        results["tool_call"] = s

        # 3. Agent turn: parallel calls, results back, final answer.
        msgs = [{"role": "user", "content": "Is it warmer in Lisbon or in Oslo right now? Check both."}]
        a = chat(url, key, msgs, max_tokens=4096, tools=[WEATHER_TOOL], **AGENT)
        print(f"\nagent step 1 ({a['seconds']} s): {[(c['name'], c['arguments']) for c in a['tool_calls']]}")
        msgs.append(as_message(a))
        for i, c in enumerate(a["tool_calls"]):
            city = json.loads(c["arguments"] or "{}").get("city", "")
            msgs.append({"role": "tool", "tool_call_id": c["id"] or f"call_{i}",
                         "content": json.dumps(FAKE_WEATHER.get(city, {"error": "unknown city"}))})
        b = chat(url, key, msgs, max_tokens=4096, tools=[WEATHER_TOOL], **AGENT)
        print(f"agent step 2 ({b['seconds']} s, {b['tokens_per_s']} tok/s): {answer_text(b)}")
        results["agent"] = {"step1": a, "step2": b}

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

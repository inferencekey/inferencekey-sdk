"""Serve NVIDIA Nemotron 3.5 Bolt 74B-A9B (24 GB GGUF) with the llama.cpp
backend on a private NVIDIA GPU, with its DFlash drafter, and use it three
ways: a reasoning answer, a question about an image, and a tool loop.

The canonical SDK shape: ensure() -> wait_until_ready() -> calls. The axis this
example varies: one llama-server process serves a 74B vision-language MoE whose
weights NVIDIA ships natively in 2-bit (IQ2_XXS after quantization-aware
distillation), plus a separate small drafter GGUF for speculative decoding.

Nemotron 3.5 Bolt 74B-A9B: 74B parameters, 9B active per token. The main GGUF
is 22.0 GB, the vision projector 1.73 GB and the LiLiCorr DFlash drafter
1.0 GB. NVIDIA's "28GB recipe" below (98,304 tokens of context, q8_0 KV cache,
up to 6 draft tokens) is meant for 32 GB cards; a 48 GB card has room to spare.

Worker requirement: Nemotron 3.5 support landed in llama.cpp on 2026-10-15
(ggml-org/llama.cpp PR #30520, first in tag v0.7.0). The InferenceKey worker
builds llama.cpp from a pinned tag, so set INFERENCEKEY_LLAMACPP_TAG=v0.7.0 (or
newer) in the worker's environment before you run this. See the README.

Run:  cp .env.example .env   # fill in real values, then
      python main.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw

from inferencekey import (
    Backend,
    DataClient,
    ExecutionPolicy,
    InferenceKeyError,
    ManagementClient,
    WorkloadSpec,
)

REPO = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF"
GGUF_FILE = "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf"
DRAFTER_REPO = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF"

# llama-server launch command, run verbatim by the `llamacpp` backend. It is
# NVIDIA's 28GB recipe from the model card, with Hugging Face downloads instead
# of local paths:
#   -hf / --hf-file            -> the 2-bit main model; the repo's mmproj (vision
#                                 projector) is fetched with it automatically.
#   --spec-draft-hf            -> the LiLiCorr drafter (its repo has one GGUF).
#   --spec-type draft-dflash   -> DFlash speculative decoding, up to 6 draft
#   --spec-draft-n-max 6          tokens per step, drafter fully on the GPU.
#   --ctx-size 98304 --parallel 1
#                              -> one 98,304-token slot.
#   --cache-type-k/v q8_0      -> 8-bit KV cache (with --flash-attn on).
#   --temp 1.0 --top-p 0.95    -> NVIDIA's sampling defaults.
#   --jinja                    -> the model's own chat template (tools, thinking).
# The worker injects the listen port; do not hard-code --port.
COMMAND = (
    f"llama-server -hf {REPO} --hf-file {GGUF_FILE} "
    f"--spec-draft-hf {DRAFTER_REPO} "
    "--spec-type draft-dflash --spec-draft-n-max 6 --spec-draft-ngl 999 "
    "-ngl 999 --batch-size 2048 --ubatch-size 512 "
    "--ctx-size 98304 --parallel 1 "
    "--cache-type-k q8_0 --cache-type-v q8_0 "
    "--temp 1.0 --top-p 0.95 --flash-attn on --jinja"
)

SLUG = "nemotron-bolt-74b"
HERE = Path(__file__).resolve().parent

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order by its id: items, amount paid and status.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Refund part or all of an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount_eur": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id", "amount_eur", "reason"],
            },
        },
    },
]


def run_tool(name: str, args: dict) -> dict:
    """Your real functions go here. These two return canned data."""
    if name == "get_order":
        return {
            "order_id": args.get("order_id"),
            "status": "delivered",
            "items": [
                {"sku": "LAMP-01", "name": "Desk lamp", "qty": 1, "price_eur": 39.90},
                {"sku": "BULB-E27", "name": "LED bulb", "qty": 4, "price_eur": 4.50},
            ],
            "paid_eur": 57.90,
            "note": "customer reports 2 of the 4 bulbs arrived broken",
        }
    if name == "issue_refund":
        return {"ok": True, "order_id": args.get("order_id"), "refunded_eur": args.get("amount_eur")}
    return {"error": f"unknown tool {name}"}


def receipt_png() -> bytes:
    """A small synthetic receipt, drawn here so the example needs no files."""
    img = Image.new("RGB", (520, 360), "white")
    d = ImageDraw.Draw(img)
    lines = [
        "CAFE NORTE - TICKET 0417",
        "",
        "2 x Cappuccino        6.40",
        "1 x Croissant         2.10",
        "1 x Orange juice      3.50",
        "1 x Tostada tomate    3.20",
        "",
        "TOTAL                15.20",
        "Paid by card",
    ]
    for i, line in enumerate(lines):
        d.text((30, 30 + i * 34), line, fill="black")
    buf = io.BytesIO()
    img.resize((1040, 720)).save(buf, format="PNG")
    return buf.getvalue()


def chat(url: str, api_key: str, body: dict) -> dict:
    """Plain HTTP to the OpenAI-compatible endpoint, for what the SDK's
    generate_text() does not take yet (images, tools)."""
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "default", **body},
        timeout=900,
    )
    r.raise_for_status()
    return r.json()


def tool_loop(url: str, api_key: str, question: str, max_turns: int = 5) -> list:
    messages = [{"role": "user", "content": question}]
    for _ in range(max_turns):
        msg = chat(url, api_key, {"messages": messages, "tools": TOOLS, "max_tokens": 8192})["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            messages.append({"role": "assistant", "content": msg.get("content") or ""})
            return messages
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            args = json.loads(c["function"]["arguments"] or "{}")
            result = run_tool(c["function"]["name"], args)
            print(f"  tool {c['function']['name']}({args}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)})
    return messages


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="Nemotron 3.5 Bolt 74B-A9B (llama.cpp, DFlash)",
        slug=SLUG,
        model=REPO,
        backend=Backend.LLAMACPP,
        command=COMMAND,
        task_type="text2text",
        # Pin to your NVIDIA worker (32 GB or more). Never hard-code the id.
        worker_id=os.environ["IK_WORKER_ID"],
        execution_policy=ExecutionPolicy.FIXED,
        execution_policy_config={"replicas": 1},
    )
    outputs = HERE / "outputs"
    outputs.mkdir(exist_ok=True)

    try:
        # 1) Provision / reconcile. Idempotent by slug.
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # 2) Wait until llama-server serves. A cold worker builds llama.cpp
        #    with CUDA and downloads about 24.7 GB of GGUF files first.
        t0 = time.monotonic()
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.monotonic() - t0:.1f} s")

        # 3a) A reasoning answer through the SDK (ik_live_).
        ep = DataClient.from_env().endpoint(ref.workload_slug)
        t1 = time.monotonic()
        out = ep.generate_text(
            messages=[{"role": "user", "content": (
                "A warehouse robot moves 3 m north, 4 m east, 3 m south and 1 m west, then "
                "returns to its start in a straight line. Its battery drains 2% per metre. "
                "How much battery does the whole trip use? Answer in one sentence.")}],
            max_tokens=8192,
        )
        usage = (out.raw or {}).get("usage", {})
        print(f"\nanswer: {out.text.strip()}")
        print(f"{time.monotonic() - t1:.1f} s, {usage.get('completion_tokens')} tokens, "
              f"finish={out.finish_reason}")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{ref.project_slug}/{ref.workload_slug}/v1/chat/completions"
        key = os.environ["INFERENCEKEY_API_KEY"]

        # 3b) An image question: the same endpoint takes OpenAI image parts.
        png = receipt_png()
        (outputs / "receipt.png").write_bytes(png)
        t2 = time.monotonic()
        res = chat(url, key, {"max_tokens": 8192, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            {"type": "text", "text": "Read this receipt. List each item with its price, and check whether the total is right."},
        ]}]})
        vision = res["choices"][0]["message"].get("content") or ""
        print(f"\nreceipt: {vision.strip()}\n{time.monotonic() - t2:.1f} s")

        # 3c) A tool loop: the model calls your functions, then answers.
        print("\nagent:")
        t3 = time.monotonic()
        transcript = tool_loop(url, key, (
            "Order A-7731: the customer says two of the bulbs arrived broken. "
            "Check the order and refund what is fair."))
        print(f"  final: {transcript[-1]['content'].strip()}")
        print(f"  {time.monotonic() - t3:.1f} s for the whole loop")

        (outputs / "answers.json").write_text(json.dumps({
            "reasoning_answer": out.text, "receipt": vision, "agent_transcript": transcript,
        }, indent=2) + "\n")
        print(f"saved {outputs / 'answers.json'}")
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

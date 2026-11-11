"""Serve SenseNova 6.8 Code2Video (SenseTime, Apache-2.0, a Qwen3.8-27B
fine-tune that makes short videos by writing and rendering code) from a 4-bit
GGUF plus its vision projector with the llama.cpp backend on one private
24 GB NVIDIA GPU, then make one call through the Anthropic-compatible
`/v1/messages` route, the same route Claude Code uses.

The canonical SDK shape: ensure() -> wait_until_ready() -> call. The axis this
example varies: a coding model meant to be driven by Claude Code, so the call
goes to `/v1/messages` and the model answers with a complete animation page
whose every frame is a function of time (`window.seek(t)`), saved to
outputs/clip.html. `run_claude_code.sh` then points Claude Code with the
authors' `code-video` skill at the same endpoint to produce a real MP4.

Run:  cp .env.example .env   # fill in real values, then
      python main.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

from inferencekey import (
    Backend,
    ExecutionPolicy,
    InferenceKeyError,
    ManagementClient,
    WorkloadSpec,
)

REPO = "bartowski/sensenova_SenseNova-6.8-Code2Video-GGUF"
GGUF_FILE = os.environ.get("C2V_GGUF_FILE", "sensenova_SenseNova-6.8-Code2Video-Q4_K_M.gguf")
MMPROJ_URL = (f"https://huggingface.co/{REPO}/resolve/main/"
              "mmproj-sensenova_SenseNova-6.8-Code2Video-bf16.gguf")
CONTEXT = int(os.environ.get("C2V_CONTEXT", "98304"))
KV_TYPE = os.environ.get("C2V_KV_TYPE", "q8_0")
LLAMACPP_TAG = os.environ.get("C2V_LLAMACPP_TAG", "b12167")
LLAMACPP_ASSET = os.environ.get("C2V_LLAMACPP_ASSET", "ubuntu-cuda-13.4-x64")

# llama-server launch command, run verbatim by the `llamacpp` backend.
#   INFERENCEKEY_LLAMACPP_TAG / _ASSET -> the worker installs this llama.cpp
#     release from its prebuilt CUDA 13.4 tarball (no build from source). The
#     GGUF was made with b11969; b12167 is newer. Empty asset = build.
#   -hf / --hf-file   -> the 17.2 GB Q4_K_M GGUF.
#   --mmproj-url      -> the 0.93 GB bf16 vision projector, so the model can
#     look at the frames it renders (the code-video skill reviews stills).
#   -c / -ctk / -ctv  -> 98,304 tokens with a q8_0 KV cache. Only 16 of the 64
#     layers keep a KV cache (the rest are linear attention), about 32 KiB per
#     token at q8_0, so 96K costs about 3 GiB.
#   --jinja --reasoning-format deepseek -> the model's own chat template
#     (tools included); its thinking goes to `reasoning_content`.
#   --temp 1.0 --top-k 20 --top-p 0.95 -> the model's generation_config.json.
#   -np 1 --cache-reuse 256 -> one agent at a time, prompt prefix reused
#     between turns (Claude Code resends the whole conversation every turn).
# The worker injects the listen port; do not hard-code --port.
_ASSET = f"INFERENCEKEY_LLAMACPP_ASSET={LLAMACPP_ASSET} " if LLAMACPP_ASSET else ""
COMMAND = (
    f"INFERENCEKEY_LLAMACPP_TAG={LLAMACPP_TAG} {_ASSET}"
    f"llama-server -hf {REPO} --hf-file {GGUF_FILE} --mmproj-url {MMPROJ_URL} "
    f"--alias sensenova-code2video -ngl 99 -fa on -c {CONTEXT} -ctk {KV_TYPE} -ctv {KV_TYPE} "
    "--jinja --reasoning-format deepseek --temp 1.0 --top-k 20 --top-p 0.95 "
    "-np 1 --cache-reuse 256"
)

SLUG = "sensenova-code2video"
HERE = Path(__file__).resolve().parent

BRIEF = (
    "Write ONE self-contained HTML file (inline CSS and JS, no network) for a 6-second, "
    "1280x720 animation: five blue dots appear one by one in a row, then the row turns into "
    "the text `1 + 2 + 3 + 4 + 5 = 15`. Every frame must be a pure function of time: expose "
    "`window.seek = async (t) => {...}` that draws the exact state at t seconds on a canvas, "
    "and set `window.READY = true` once fonts are loaded. No requestAnimationFrame loops, no "
    "Date.now(), no CSS animations. Reply with the file in one ```html code block."
)


def messages_call(url: str, api_key: str, prompt: str, max_tokens: int) -> dict:
    """Streaming call to the Anthropic-compatible Messages route (the way Claude
    Code calls it). Streaming matters: the model thinks for thousands of tokens
    first, and a non-streaming call can outlast the worker's request timeout."""
    t0 = time.monotonic()
    text, usage, stop = "", {}, None
    with requests.post(url, headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                       json={"model": SLUG, "max_tokens": max_tokens, "stream": True,
                             "messages": [{"role": "user", "content": prompt}]},
                       stream=True, timeout=1800) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            line = raw.decode("utf-8")
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev.get("type") == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                text += ev["delta"]["text"]
            elif ev.get("type") == "message_delta":
                usage = ev.get("usage") or usage
                stop = (ev.get("delta") or {}).get("stop_reason") or stop
    return {"seconds": round(time.monotonic() - t0, 2), "usage": usage, "stop_reason": stop, "text": text}


def main() -> int:
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="SenseNova 6.8 Code2Video (Q4_K_M, llama.cpp)",
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

        # A cold worker installs llama.cpp and downloads 18.1 GB first.
        t0 = time.monotonic()
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        results["ready_s"] = round(time.monotonic() - t0, 1)
        print(f"ready after {results['ready_s']} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{ref.project_slug}/{ref.workload_slug}/v1/messages"
        out = messages_call(url, os.environ["INFERENCEKEY_API_KEY"], BRIEF, max_tokens=16384)
        print(f"\n/v1/messages: {out['seconds']} s, usage {out['usage']}, stop {out['stop_reason']}")
        m = re.search(r"```html\s*(.*?)```", out["text"], re.S)
        if m:
            (outputs / "clip.html").write_text(m.group(1))
            print(f"saved {outputs / 'clip.html'} ({len(m.group(1))} chars); open it in a browser "
                  "or capture it with the code-video skill's scripts/capture.mjs")
        else:
            print("no ```html block in the answer; full text saved to outputs/answer.json")
        results["messages_call"] = out
        (outputs / "answer.json").write_text(json.dumps(results, indent=2) + "\n")
    except (InferenceKeyError, requests.RequestException) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        if os.environ.get("IK_KEEP_ALIVE") != "1":
            print(f"deleted={mgmt.delete(SLUG)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

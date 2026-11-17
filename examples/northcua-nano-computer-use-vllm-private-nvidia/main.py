"""Serve NorthCUA-Nano (Beipo AI's 9B computer-use model) on one private
NVIDIA GPU with the stock vLLM backend, then ask it for the next GUI action on
a screenshot, the way the NorthCUA Harness does.

The canonical SDK shape: ensure() -> wait_until_ready() -> call. The call
sends the Harness's system prompt, the task, a 1280x800 screenshot of a mail
client (samples/mail.png) and the Harness's desktop tools; the model answers
with its reasoning, a short status line and a tool call such as
click(x=..., y=...) in the 0..1000 coordinate space. We map it to pixels and
draw it on the screenshot (outputs/click.png).

NorthCUA-Nano: Qwen3.5-9B architecture (32 layers, 3 linear-attention layers
for every full-attention layer, vision encoder), BF16, 18.8 GB of weights,
262,144-token context, Apache-2.0.

Run:  cp .env.example .env   # fill in real values, then
      python main.py
For a full multi-step run in a real browser see browser_agent.py.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw

from inferencekey import Backend, ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

import northcua_step as ncua

MODEL = "beipo-ai/NorthCUA-Nano"
REVISION = "317dd35f56527900d1c0fb4f344107f181598b07"  # pin the checkpoint you tested
CONTEXT = int(os.environ.get("NORTHCUA_CONTEXT", "32768"))
GPU_UTIL = os.environ.get("NORTHCUA_GPU_UTIL", "0.90")
# FP8 weights (vLLM online quantization, about 10 GiB instead of 18.8 GB):
# for 16 GB cards. BF16 (default) needs a 24 GB card or larger.
FP8 = os.environ.get("NORTHCUA_FP8", "0") == "1"

# vLLM launch command, run verbatim by the `vllm` backend (the worker injects
# host and port):
#   --max-model-len 32768        -> enough for the system prompt, tools and a few screenshots.
#   --limit-mm-per-prompt        -> up to 4 screenshots per request, no video.
#   --reasoning-parser qwen3     -> thinking in `reasoning_content`, status text in `content`.
#   --tool-call-parser qwen3_coder -> the model's <tool_call><function=...> format as OpenAI tool_calls.
COMMAND = (
    f"vllm serve {MODEL} --revision {REVISION} --max-model-len {CONTEXT} "
    f"--gpu-memory-utilization {GPU_UTIL} --max-num-seqs 8 "
    "--limit-mm-per-prompt '{\"image\": 4, \"video\": 0}' "
    "--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
)
if FP8:
    COMMAND += " --quantization fp8"

# vLLM 0.33.0 ships the Qwen3.5 vision-language model class and installs
# torch built for CUDA 13: the worker needs a CUDA 13 driver.
VLLM_VERSION = "0.33.0"
SLUG = os.environ.get("NORTHCUA_SLUG", "northcua-nano")
HERE = Path(__file__).resolve().parent
TASK = "Open the email from Priya Shah about the Q3 invoice."


def main() -> int:
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name=f"NorthCUA-Nano ({'FP8' if FP8 else 'BF16'}, vLLM)",
        slug=SLUG,
        model=MODEL,
        backend=Backend.VLLM,
        command=COMMAND,
        vllm_version=VLLM_VERSION,
        task_type="text2text",
        worker_id=os.environ["IK_WORKER_ID"],  # your NVIDIA worker; never hard-code it
        execution_policy=ExecutionPolicy.FIXED,
    )
    outputs = HERE / "outputs"
    outputs.mkdir(exist_ok=True)
    try:
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        t0 = time.monotonic()
        # A cold worker installs vLLM and downloads 18.8 GB of weights first.
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        ready_s = round(time.monotonic() - t0, 1)
        print(f"ready after {ready_s} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{ref.project_slug}/{ref.workload_slug}/v1/chat/completions"
        key = os.environ["INFERENCEKEY_API_KEY"]

        png = (HERE / "samples" / "mail.png").read_bytes()
        width, height = Image.open(HERE / "samples" / "mail.png").size
        messages = [
            {"role": "system", "content": ncua.system_prompt(width, height)},
            {"role": "user", "content": ncua.task_prompt(TASK)},
            ncua.observation(png),
        ]
        r = ncua.call(url, key, MODEL, messages, temperature=0.0)
        print(f"\n{r['seconds']} s, {r['usage'].get('completion_tokens')} tokens")
        print(f"thinking: {r['reasoning'].strip()[:600]}")
        print(f"status:   {r['content'].strip()}")
        img = Image.open(HERE / "samples" / "mail.png").convert("RGB")
        draw = ImageDraw.Draw(img)
        for tc in r["tool_calls"]:
            a = ncua.args(tc)
            print(f"tool:     {tc['function']['name']}({json.dumps(a)})")
            if "x" in a and "y" in a and a["x"] is not None:
                px, py = ncua.to_pixels(a["x"], a["y"], width, height)
                print(f"          -> pixel ({px}, {py}) on the {width}x{height} screenshot")
                draw.ellipse([px - 14, py - 14, px + 14, py + 14], outline=(220, 30, 30), width=4)
        img.save(outputs / "click.png")
        (outputs / "answer.json").write_text(json.dumps({"ready_s": ready_s, "command": COMMAND, **r}, indent=2) + "\n")
        print(f"\nsaved {outputs / 'answer.json'} and {outputs / 'click.png'}")
    except (InferenceKeyError, requests.RequestException) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        if os.environ.get("IK_KEEP_ALIVE") != "1":
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

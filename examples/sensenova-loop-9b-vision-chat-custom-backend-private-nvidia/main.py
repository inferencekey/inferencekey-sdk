"""Chat and chart reading with **SenseNova-Loop-9B-Preview** on a private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`sensenova-loop-9b`, published with publish.py) that runs the
checkpoint's own Transformers code, because no released serving engine runs
this looped model yet.

The script sends three requests to one workload, through the OpenAI chat route:
  1. a word problem with thinking on (the answer and the thinking come apart),
  2. a chart image (./samples/energy-chart.png) with a question about it,
  3. a tool call (an exchange-rate function).
The answers land in ./outputs as JSON, with the backend's timings.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

HERE = Path(__file__).resolve().parent
SLUG = os.environ.get("LOOP_WORKLOAD_SLUG", "sensenova-loop")
BACKEND_SLUG = os.environ.get("LOOP_BACKEND_SLUG", "sensenova-loop-9b")
OUT_DIR = HERE / "outputs"
IMAGE = Path(os.environ.get("IK_IMAGE", HERE / "samples" / "energy-chart.png"))

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_exchange_rate",
        "description": "Latest exchange rate between two currencies.",
        "parameters": {
            "type": "object",
            "properties": {
                "base": {"type": "string", "description": "ISO code of the currency to convert from"},
                "quote": {"type": "string", "description": "ISO code of the currency to convert to"},
            },
            "required": ["base", "quote"],
        },
    },
}]


def _load_dotenv() -> None:
    env = HERE / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def image_part(path: Path) -> Dict[str, Any]:
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"}}


def ask(url: str, key: str, messages: List[Dict[str, Any]], **options: Any) -> Dict[str, Any]:
    """One chat request; returns {"content", "reasoning", "tool_calls", "timings"}.

    Options the OpenAI schema lacks travel in extra_body. The backend answers the
    chat route with its JSON (answer, thinking, tool calls, timings) as the
    message content; a runtime that serves chat() natively returns a plain
    chat.completion instead, and both shapes are handled here.
    """
    body = {"model": "default", "messages": messages, "max_tokens": options.pop("max_tokens", 2048)}
    if "tools" in options:
        body["tools"] = options.pop("tools")
    body["extra_body"] = options
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=900)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    j = r.json()
    msg = j["choices"][0]["message"]
    content = msg.get("content") or ""
    if isinstance(content, str) and content.lstrip().startswith("{"):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    return {"content": content, "reasoning": msg.get("reasoning_content") or "",
            "tool_calls": msg.get("tool_calls") or [], "timings": j.get("timings") or {}}


def report(out: Dict[str, Any], name: str, started: float) -> None:
    t = out.get("timings") or {}
    print(f"{name}: round trip {time.time() - started:.1f} s, {t.get('completion_tokens')} tokens, "
          f"{t.get('tokens_per_s')} tokens/s, first token {t.get('first_token_s')} s, peak VRAM {t.get('peak_vram_mib')} MiB")
    if out.get("reasoning"):
        print(f"  thinking: {len(out['reasoning'])} characters")
    if out.get("content"):
        print(f"  answer: {out['content'][:600]}")
    for call in out.get("tool_calls") or []:
        print(f"  tool call: {call['function']['name']}({call['function']['arguments']})")
    (OUT_DIR / f"{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


def main() -> int:
    _load_dotenv()
    if not IMAGE.is_file():
        print(f"input not found: {IMAGE}", file=sys.stderr)
        return 1
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="SenseNova-Loop-9B Preview",
        slug=SLUG,
        model="sensenova/SenseNova-Loop-9B-Preview",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2text",
        worker_id=os.environ["IK_WORKER_ID"],
        # The first start downloads about 38 GB and loads it: give setup() time.
        config={"ready_timeout_secs": 3600, "request_timeout_secs": 1200},
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
    )
    base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
    project = os.environ["INFERENCEKEY_PROJECT"]
    key = os.environ["INFERENCEKEY_API_KEY"]
    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        print("waiting for the worker to become ready (first start installs torch and downloads 38 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.time() - t0:.1f} s")

        url = f"{base}/endpoint/{project}/{ref.workload_slug}/v1/chat/completions"
        OUT_DIR.mkdir(exist_ok=True)

        t = time.time()
        out = ask(url, key, [{"role": "user", "content":
                              "Three painters paint 2 fences in 5 hours. Working at the same rate, how many hours "
                              "do 6 painters need for 8 fences? End with 'Answer: <number>'."}],
                  max_tokens=4096)
        report(out, "1-word-problem-thinking", t)

        t = time.time()
        out = ask(url, key, [{"role": "user", "content": [
            image_part(IMAGE),
            {"type": "text", "text": "Which month used the least electricity, and how much? How many kWh did "
                                     "July to September save compared with April to June, and what is that "
                                     "in euros at the grid price shown? Answer briefly."},
        ]}], max_tokens=4096)
        report(out, "2-chart-question", t)

        t = time.time()
        out = ask(url, key, [{"role": "user", "content": "How much is 250 US dollars in euros right now?"}],
                  tools=TOOLS, enable_thinking=False, max_tokens=512)
        report(out, "3-tool-call", t)
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (requests.RequestException, RuntimeError, KeyError, ValueError) as e:
        print(f"request failed: {e}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the workload.
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

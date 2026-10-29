"""Serve Recursive Flint (NVFP4 GGUF) with the llama.cpp backend on a private
NVIDIA GPU and use it two ways: a coding answer, and a small coding-agent loop
that reads a repo, runs its tests, fixes the bug and runs the tests again.

The canonical SDK shape: ensure() -> wait_until_ready() -> calls. The axis this
example varies: a 39.8B mixture-of-experts model built for coding agents
(about 2.5B active per token) whose authors ship official NVFP4 weights, served
by one llama-server process with OpenAI-compatible tool calls.

Recursive Flint: 39.8B parameters, about 2.5B active, 262,144-token context,
Apache-2.0. The NVFP4 GGUF is 22.9 GB. This example cuts the context to 65,536
tokens with an F16 KV cache so it fits a 32 GB card.

Worker requirement: Flint support landed in llama.cpp on 2026-10-21
(ggml-org/llama.cpp PR #30928). The InferenceKey worker builds llama.cpp from a
pinned tag, so it needs a build that contains that PR. See the README.

Run:  cp .env.example .env   # fill in real values, then
      python main.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
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

REPO = "RecursiveSI/flint-nvfp4-gguf"
GGUF_FILE = "flint-nvfp4.gguf"

# llama-server launch command, run verbatim by the `llamacpp` backend. It is the
# model card's command with two changes: Hugging Face download instead of a
# local path, and 65,536 tokens of context instead of 262,144.
#   -hf / --hf-file      -> the 22.9 GB NVFP4 GGUF (weights, tokenizer, template).
#   -ngl all             -> every layer on the GPU.
#   -c 65536 -np 1       -> one 65,536-token slot (prompt and output share it).
#   -fa on -ctk/-ctv f16 -> flash attention, F16 KV cache (the card's choice).
#   --jinja              -> the model's own chat template (reasoning, tools).
#   --temp 1 --top-p 0.95 --top-k 20 --min-p 0 -> the authors' sampling.
# The worker injects the listen port; do not hard-code --port.
COMMAND = (
    f"llama-server -hf {REPO} --hf-file {GGUF_FILE} --alias flint "
    "-ngl all -c 65536 -np 1 -fa on -ctk f16 -ctv f16 --jinja "
    "--temp 1 --top-p 0.95 --top-k 20 --min-p 0"
)

SLUG = "flint-coder"
HERE = Path(__file__).resolve().parent

# A tiny repo with one bug (two failing tests), written to a temp folder. The agent can only touch
# files inside that folder, and the only command it can run is the test suite.
REPO_FILES = {
    "invoices.py": '''"""Invoice helpers."""


def line_total(qty: int, unit_price: float, discount_pct: float = 0.0) -> float:
    """Price of one invoice line after a percentage discount, rounded to cents."""
    gross = qty * unit_price
    return round(gross - gross * discount_pct, 2)


def invoice_total(lines: list[dict], vat_pct: float = 21.0) -> float:
    """Sum of the lines plus VAT, rounded to cents."""
    subtotal = sum(line_total(**line) for line in lines)
    return round(subtotal * (1 + vat_pct / 100), 2)
''',
    "test_invoices.py": '''from invoices import invoice_total, line_total


def test_line_without_discount():
    assert line_total(3, 2.50) == 7.50


def test_line_with_discount():
    # 10 % off 4 x 25.00
    assert line_total(4, 25.0, discount_pct=10) == 90.00


def test_invoice_with_vat():
    lines = [
        {"qty": 2, "unit_price": 10.0},
        {"qty": 1, "unit_price": 50.0, "discount_pct": 20},
    ]
    # (20 + 40) * 1.21
    assert invoice_total(lines) == 72.60
''',
}

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List the files in the repository.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file of the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Replace the whole content of a file of the repository.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the test suite with pytest and return its output.",
        "parameters": {"type": "object", "properties": {}}}},
]


def run_tool(root: Path, name: str, args: dict) -> str:
    """The agent's hands. Paths are confined to `root`."""
    def safe(p: str) -> Path:
        target = (root / p).resolve()
        if root.resolve() not in target.parents and target != root.resolve():
            raise ValueError("path outside the repository")
        return target
    try:
        if name == "list_files":
            return "\n".join(sorted(p.name for p in root.iterdir() if p.is_file()))
        if name == "read_file":
            return safe(args["path"]).read_text()
        if name == "write_file":
            safe(args["path"]).write_text(args["content"])
            return f"wrote {args['path']} ({len(args['content'])} bytes)"
        if name == "run_tests":
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
                               cwd=root, capture_output=True, text=True, timeout=120)
            return (r.stdout + r.stderr)[-4000:]
    except Exception as e:  # report tool errors back to the model
        return f"error: {e}"
    return f"error: unknown tool {name}"


def chat(url: str, api_key: str, body: dict) -> dict:
    """Plain HTTP to the OpenAI-compatible endpoint, for what the SDK's
    generate_text() does not take yet (tools)."""
    r = requests.post(url, headers={"Authorization": f"Bearer {api_key}"},
                      json={"model": "default", **body}, timeout=1800)
    r.raise_for_status()
    return r.json()


def agent(url: str, api_key: str, root: Path, task: str, max_turns: int = 12) -> list:
    messages = [
        {"role": "system", "content": "You are a coding agent working in a small Python repository. "
                                      "Use the tools. Keep fixes minimal and do not change the tests."},
        {"role": "user", "content": task},
    ]
    for turn in range(max_turns):
        t = time.monotonic()
        res = chat(url, api_key, {"messages": messages, "tools": TOOLS, "max_tokens": 16384})
        msg = res["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        print(f"  turn {turn + 1}: {time.monotonic() - t:.1f} s, "
              f"{res.get('usage', {}).get('completion_tokens')} tokens, {len(calls)} tool call(s)")
        if not calls:
            messages.append({"role": "assistant", "content": msg.get("content") or ""})
            return messages
        # Keep the reasoning when it is returned (the authors ask for it).
        assistant = {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}
        if msg.get("reasoning_content"):
            assistant["reasoning_content"] = msg["reasoning_content"]
        messages.append(assistant)
        for c in calls:
            args = json.loads(c["function"].get("arguments") or "{}")
            out = run_tool(root, c["function"]["name"], args)
            shown = {k: (v if len(str(v)) < 60 else f"<{len(str(v))} chars>") for k, v in args.items()}
            print(f"    {c['function']['name']}({shown}) -> {out.strip().splitlines()[-1][:100] if out.strip() else ''}")
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
    return messages


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="Recursive Flint 40B-A2.5B (NVFP4, llama.cpp)",
        slug=SLUG,
        model=REPO,
        backend=Backend.LLAMACPP,
        command=COMMAND,
        task_type="text2text",
        # Pin to your NVIDIA worker (32 GB or more). Never hard-code the id.
        worker_id=os.environ["IK_WORKER_ID"],
        execution_policy=ExecutionPolicy.FIXED,
    )
    outputs = HERE / "outputs"
    outputs.mkdir(exist_ok=True)

    try:
        # 1) Provision / reconcile. Idempotent by slug.
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # 2) Wait until llama-server serves. A cold worker builds llama.cpp
        #    with CUDA and downloads the 22.9 GB GGUF first.
        t0 = time.monotonic()
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.monotonic() - t0:.1f} s")

        # 3a) A coding answer through the SDK (ik_live_).
        ep = DataClient.from_env().endpoint(ref.workload_slug)
        t1 = time.monotonic()
        out = ep.generate_text(
            messages=[{"role": "user", "content": (
                "Write a Python function merge_intervals(intervals) that merges overlapping "
                "closed intervals given as (start, end) tuples and returns them sorted. "
                "Add three asserts. Code only.")}],
            max_tokens=16384,
        )
        usage = (out.raw or {}).get("usage", {})
        print(f"\nanswer:\n{out.text.strip()}")
        print(f"{time.monotonic() - t1:.1f} s, {usage.get('completion_tokens')} tokens, "
              f"finish={out.finish_reason}")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{ref.project_slug}/{ref.workload_slug}/v1/chat/completions"
        key = os.environ["INFERENCEKEY_API_KEY"]

        # 3b) A coding-agent loop on a throwaway repo with one bug and failing tests.
        root = Path(tempfile.mkdtemp(prefix="flint-repo-"))
        for name, text in REPO_FILES.items():
            (root / name).write_text(text)
        print(f"\nagent (repo in {root}):")
        t2 = time.monotonic()
        transcript = agent(url, key, root,
                           "Some tests fail. Run the tests, find the bug, fix it, and run the tests again.")
        final_tests = run_tool(root, "run_tests", {})
        print(f"  final answer: {transcript[-1].get('content', '').strip()[:400]}")
        print(f"  tests after the agent: {final_tests.strip().splitlines()[-1]}")
        print(f"  {time.monotonic() - t2:.1f} s for the whole loop")

        (outputs / "answers.json").write_text(json.dumps({
            "coding_answer": out.text,
            "agent_transcript": transcript,
            "fixed_invoices_py": (root / "invoices.py").read_text(),
            "tests_after_agent": final_tests,
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

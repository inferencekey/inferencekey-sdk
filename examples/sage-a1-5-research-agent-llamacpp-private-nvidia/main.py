"""Serve Sage-A1.5 (official Q4_K_M GGUF) with the llama.cpp backend on a private
NVIDIA GPU and use its two agentic modes: Agentic Orchestration (the model
fans a question out to parallel sub-agents through a `spawn_subagent` tool)
and Verification Mode (an opt-in self-review inside its own reasoning).

The canonical SDK shape: ensure() -> wait_until_ready() -> calls. The axis this
example varies: an agentic research model whose special modes are switched on
from the request side (a tool and a system-prompt flag), served by one stock
llama-server process with OpenAI-compatible tool calls and streaming.

Sage-A1.5: 35B mixture-of-experts (about 3B active per token, Qwen3.6-35B-A3B
architecture), Apache-2.0, from Shanghai AI Laboratory (InternScience). The
Q4_K_M GGUF is 21.7 GB. This example runs it text-only (no vision projector)
with a 32,768-token context so it fits a 24 GB card.

The sub-agents get two tools that run on YOUR machine: a Wikipedia search and a
Wikipedia page reader (public API, no key). Swap them for your own.

Run:  cp .env.example .env   # fill in real values, then
      python main.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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

REPO = "InternScience/Sage-A1.5-Q4_K_M-GGUF"
GGUF_FILE = "Sage-A1.5-Q4_K_M.gguf"
CONTEXT = int(os.environ.get("SAGE_CONTEXT", "32768"))

# llama-server launch command, run verbatim by the `llamacpp` backend. It is the
# model card's text-only command with three changes: Hugging Face download, a
# smaller context, and the authors' sampling as server defaults.
#   -hf / --hf-file   -> the 21.7 GB Q4_K_M GGUF (weights, tokenizer, template).
#   --no-mmproj       -> skip the 0.9 GB vision projector (text only).
#   -ngl 99 -fa on    -> every layer on the GPU, flash attention.
#   -c / -np 2        -> CONTEXT tokens shared by 2 slots (sub-agents run in parallel).
#   --jinja           -> the model's own chat template (tools, verification flag).
#   --reasoning-format deepseek -> thinking in `reasoning_content`, answer in `content`.
#     (The model card says `deepthink`; upstream llama.cpp has no such value and
#     llama-server exits with "Unknown reasoning format: deepthink".)
#   --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5 -> authors' sampling.
# The worker injects the listen port; do not hard-code --port.
COMMAND = (
    f"llama-server -hf {REPO} --hf-file {GGUF_FILE} --no-mmproj --alias sage-a1.5 "
    f"-ngl 99 -fa on -c {CONTEXT} -np 2 --kv-unified --jinja --reasoning-format deepseek "
    "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5"
)

SLUG = "sage-a1-5"
HERE = Path(__file__).resolve().parent

# Verification Mode: the chat template appends exactly this line to the system
# message when a request carries chat_template_kwargs={"verification": true}.
# The InferenceKey endpoint does not forward chat_template_kwargs to the
# backend, so we append the same line ourselves: the rendered prompt is the same.
VERIFY_TRIGGER = "\n\nVerification protocol: enabled"

WIKI = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "inferencekey-sdk-example/0.1 (https://github.com/inferencekey/inferencekey-sdk)"}

RESEARCH_TOOLS = [
    {"type": "function", "function": {
        "name": "wiki_search", "description": "Search English Wikipedia. Returns the top titles with a snippet.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_page", "description": "Read the plain text of an English Wikipedia page (first 6,000 characters).",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}}},
]

# The authors' schema (model card, "Agentic Orchestration"), with the tool
# names in its description changed to the ones our sub-agents really have.
SPAWN_SUBAGENT = {"type": "function", "function": {
    "name": "spawn_subagent",
    "description": (
        "Dispatch one or more sub-agents IN PARALLEL to investigate independent "
        "sub-questions. Each sub-agent has access to wiki_search and read_page "
        "but CANNOT see your conversation. You MUST give each sub-agent everything "
        "it needs via the `task`, `context`, and `expected_output` fields. Sub-agents "
        "return structured JSON: {answer, status, evidence, unresolved_issues}."),
    "parameters": {"type": "object", "properties": {"tasks": {
        "type": "array", "minItems": 1,
        "description": "List of independent sub-tasks to dispatch in parallel.",
        "items": {"type": "object", "properties": {
            "task": {"type": "string", "description": "Self-contained sub-question. The sub-agent has no other context."},
            "context": {"type": "string", "description": "Background facts and constraints the sub-agent needs."},
            "expected_output": {"type": "string", "description": "Form of the desired answer (e.g. 'a year YYYY', 'a person name')."}},
            "required": ["task", "context", "expected_output"]}}},
        "required": ["tasks"]}}}

SUBAGENT_SYSTEM = (
    "You are a research sub-agent. Use wiki_search and read_page to answer the task. "
    "When done, reply with ONLY one JSON object: "
    '{"answer": ..., "status": "ok" | "unknown" | "insufficient_evidence", '
    '"evidence": [short quotes with page titles], "unresolved_issues": [...]}. '
    'If you cannot resolve the task, answer "UNKNOWN" instead of guessing.')

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5}


def run_research_tool(name: str, args: dict) -> str:
    """Tools of the sub-agents. They run here, on the caller's machine."""
    try:
        if name == "wiki_search":
            r = requests.get(WIKI, params={"action": "query", "list": "search", "srsearch": args["query"],
                                           "srlimit": 5, "format": "json"}, headers=UA, timeout=30)
            hits = r.json()["query"]["search"]
            return "\n".join(f"- {h['title']}: {h['snippet'].replace('<span class=\"searchmatch\">', '').replace('</span>', '')}"
                             for h in hits) or "no results"
        if name == "read_page":
            r = requests.get(WIKI, params={"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                                           "titles": args["title"], "format": "json"}, headers=UA, timeout=30)
            page = next(iter(r.json()["query"]["pages"].values()))
            return page.get("extract", "")[:6000] or "page not found"
    except Exception as e:  # report tool errors back to the model
        return f"error: {e}"
    return f"error: unknown tool {name}"


def chat(url: str, api_key: str, body: dict) -> dict:
    """Streaming call to the OpenAI-compatible endpoint, reassembled into one
    message. Streaming keeps `reasoning_content` (where the verification
    self-review lives) separate from `content`; tool calls arrive in pieces."""
    msg = {"content": "", "reasoning_content": "", "tool_calls": {}}
    usage, t0 = {}, time.monotonic()
    with requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, stream=True, timeout=1800,
                       json={"model": "default", "stream": True, "stream_options": {"include_usage": True},
                             **SAMPLING, **body}) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            line = raw.decode("utf-8")
            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
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
    msg["seconds"], msg["usage"] = round(time.monotonic() - t0, 1), usage
    return msg


def agent(url: str, key: str, system: str, user: str, tools: list, handler, max_turns: int = 20,
          label: str = "agent") -> dict:
    """A plain tool loop. `handler(name, args)` runs one tool call."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    calls_made, reasoning, tokens = 0, [], 0
    for turn in range(max_turns):
        msg = chat(url, key, {"messages": messages, "tools": tools, "max_tokens": 8192})
        reasoning.append(msg["reasoning_content"])
        tokens += (msg["usage"] or {}).get("completion_tokens") or 0
        print(f"  [{label}] turn {turn + 1}: {msg['seconds']} s, "
              f"{(msg['usage'] or {}).get('completion_tokens')} tokens, {len(msg['tool_calls'])} tool call(s)")
        if not msg["tool_calls"]:
            return {"answer": msg["content"].strip(), "turns": turn + 1, "tool_calls": calls_made,
                    "completion_tokens": tokens, "reasoning": reasoning, "messages": messages}
        messages.append({"role": "assistant", "content": msg["content"], "tool_calls": msg["tool_calls"]})
        for c in msg["tool_calls"]:
            calls_made += 1
            args = json.loads(c["function"]["arguments"] or "{}")
            out = handler(c["function"]["name"], args)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
    return {"answer": "(turn budget exhausted)", "turns": max_turns, "tool_calls": calls_made,
            "completion_tokens": tokens, "reasoning": reasoning, "messages": messages}


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="Sage-A1.5 35B-A3B (Q4_K_M, llama.cpp)",
        slug=SLUG,
        model=REPO,
        backend=Backend.LLAMACPP,
        command=COMMAND,
        task_type="text2text",
        # Pin to your NVIDIA worker (24 GB or more). Never hard-code the id.
        worker_id=os.environ["IK_WORKER_ID"],
        execution_policy=ExecutionPolicy.FIXED,
    )
    outputs = HERE / "outputs"
    outputs.mkdir(exist_ok=True)
    results: dict = {}

    try:
        # 1) Provision / reconcile. Idempotent by slug.
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # 2) Wait until llama-server serves. A cold worker builds llama.cpp
        #    with CUDA and downloads the 21.7 GB GGUF first.
        t0 = time.monotonic()
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        print(f"ready after {time.monotonic() - t0:.1f} s")

        # 3a) A plain question through the SDK (ik_live_).
        ep = DataClient.from_env().endpoint(ref.workload_slug)
        t1 = time.monotonic()
        out = ep.generate_text(
            messages=[{"role": "user", "content": "In two sentences: what is a mixture-of-experts language model?"}],
            # The model thinks at length before it answers: leave room for it, or the
            # reply is cut inside the thinking and you get that text instead.
            max_tokens=16384, temperature=1.0,
        )
        print(f"\nanswer ({time.monotonic() - t1:.1f} s):\n{out.text.strip()}")
        results["plain_answer"] = out.text

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{ref.project_slug}/{ref.workload_slug}/v1/chat/completions"
        key = os.environ["INFERENCEKEY_API_KEY"]

        # 3b) Agentic Orchestration: the main agent only has spawn_subagent.
        def run_subagents(name: str, args: dict) -> str:
            if name != "spawn_subagent":
                return f"error: unknown tool {name}"
            tasks = args.get("tasks", [])[:5]  # the authors' default max_tasks_per_spawn
            print(f"  spawn_subagent: {len(tasks)} task(s) in parallel")

            def one(i_task):
                i, t = i_task
                prompt = f"Task: {t['task']}\nContext: {t['context']}\nExpected output: {t['expected_output']}"
                res = agent(url, key, SUBAGENT_SYSTEM, prompt, RESEARCH_TOOLS, run_research_tool,
                            max_turns=12, label=f"sub-{i + 1}")
                return res["answer"]
            with ThreadPoolExecutor(max_workers=len(tasks) or 1) as pool:
                answers = list(pool.map(one, enumerate(tasks)))
            return json.dumps(answers)

        question = ("Hafnium, rhenium and technetium were among the last stable or near-stable elements to be "
                    "found. Which of the three was discovered first, and how many years passed between the first "
                    "and the last of the three discoveries? Use sub-agents to look up each element.")
        print("\norchestration:")
        t2 = time.monotonic()
        orch = agent(url, key, "You are a research lead. Delegate look-ups to sub-agents, then answer.",
                     question, [SPAWN_SUBAGENT], run_subagents, max_turns=6, label="main")
        print(f"  answer ({time.monotonic() - t2:.1f} s): {orch['answer'][:600]}")
        results["orchestration"] = {k: v for k, v in orch.items() if k != "messages"}

        # 3c) Verification Mode on a direct question with search tools: compare the
        #     model's reasoning with the flag off and on.
        system = "You are a careful research assistant. Use the tools, then give a short final answer."
        task = ("Which element was discovered by a team that included a woman scientist in 1925, and in which "
                "country was it discovered? Check it.")
        for mode, sys_msg in (("off", system), ("on", system + VERIFY_TRIGGER)):
            print(f"\nverification {mode}:")
            t3 = time.monotonic()
            res = agent(url, key, sys_msg, task, RESEARCH_TOOLS, run_research_tool, max_turns=12,
                        label=f"verify-{mode}")
            print(f"  answer ({time.monotonic() - t3:.1f} s, {res['tool_calls']} tool calls): {res['answer'][:400]}")
            results[f"verification_{mode}"] = {k: v for k, v in res.items() if k != "messages"}

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

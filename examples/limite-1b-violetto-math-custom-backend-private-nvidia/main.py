"""A 1B model for competition math: Limite 1B - Violetto on a private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`limite-1b-violetto`, published with publish.py) that installs
vLLM 0.26 plus Paradigma's Limite vLLM plugin, because stock vLLM does not know
the `LimiteForCausalLM` architecture.

It then sends AIME 2026 problems (MathArena/aime_2026, the revision Paradigma's
own evaluator pins) all at once, each as one streamed request through the
OpenAI-compatible `/v1/chat/completions` route, takes the last \\boxed{} of
each answer and checks it against the official answer. Answers, tokens and
timings go to ./outputs/.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py                   # problem 1 only
      python main.py --problems all    # the 30 problems in parallel
      python main.py --problems 1,5,12 --samples 2
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

HERE = Path(__file__).resolve().parent
SLUG = os.environ.get("LIMITE_WORKLOAD_SLUG", "limite-1b-violetto")
BACKEND_SLUG = os.environ.get("LIMITE_BACKEND_SLUG", "limite-1b-violetto")  # from publish.py
OUT_DIR = HERE / "outputs"
# MathArena/aime_2026 at the revision limite-evals pins (CC BY-NC-SA 4.0:
# downloaded at run time, not redistributed here).
AIME_URL = ("https://huggingface.co/datasets/MathArena/aime_2026/resolve/"
            "10b4e45b7a503075d4da8a0d57916a4f06ce6bd2/data/train-00000-of-00001.parquet")
# The instruction Paradigma's evaluator prepends to every AIME problem.
AIME_INSTRUCTION = ("Solve the following math problem. Explain your reasoning and put the "
                    "final answer in \\boxed{}.\n\n")
# Native context 131072 minus the evaluator's 4096-token prompt reserve.
MAX_TOKENS = 126976


def _load_dotenv() -> None:
    env = HERE / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def load_aime() -> list:
    import pyarrow.parquet as pq

    raw = requests.get(AIME_URL, timeout=60).content
    return pq.read_table(io.BytesIO(raw)).to_pylist()


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} in the text (balanced braces)."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j:k].strip()
    return None


def as_int(ans: str | None) -> int | None:
    if ans is None:
        return None
    digits = re.sub(r"[^0-9-]", "", ans.replace("\\,", ""))
    try:
        return int(digits)
    except ValueError:
        return None


def solve(url: str, key: str, row: dict, sample: int) -> dict:
    """One streamed chat completion for one problem."""
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": AIME_INSTRUCTION + row["problem"]}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
    }
    t0 = time.time()
    first = None
    parts, usage, finish = [], None, None
    with requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body,
                       stream=True, timeout=(30, 600)) as r:
        r.raise_for_status()
        # Split on b"\n" only: decoding first would also split on Unicode
        # separators (U+2028, U+0085) that can appear inside a JSON token.
        for raw in r.iter_lines():
            line = raw.decode("utf-8")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.time() - t0
                    parts.append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    text = "".join(parts)
    answer = last_boxed(text)
    tokens = (usage or {}).get("completion_tokens")
    total = time.time() - t0
    return {
        "problem": row["problem_idx"], "sample": sample, "gold": int(row["answer"]),
        "answer": answer, "correct": as_int(answer) == int(row["answer"]),
        "finish_reason": finish, "completion_tokens": tokens,
        "first_token_s": round(first, 2) if first is not None else None,
        "total_s": round(total, 1),
        "tokens_per_s": round(tokens / total, 1) if tokens else None,
        "text": text,
    }


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="1", help="'all' or a comma list of AIME 2026 problem numbers")
    ap.add_argument("--samples", type=int, default=1, help="answers per problem (the paper uses 32)")
    args = ap.parse_args()

    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="Limite 1B Violetto",
        slug=SLUG,
        model="paradigma-inc/limite-1b-violetto",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2text",
        worker_id=os.environ["IK_WORKER_ID"],
        config={
            # First start installs vLLM + torch (about 3 GB of wheels) and
            # downloads 2.1 GB of weights; give setup() time.
            "ready_timeout_secs": 1800,
            # A hard problem can take tens of thousands of tokens.
            "request_timeout_secs": 3600,
            "max_model_len": 131072,
            "gpu_memory_utilization": float(os.environ.get("LIMITE_GPU_MEMORY_UTILIZATION", "0.90")),
        },
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
        execution_policy_config={"replicas": 1},
    )
    results: dict = {"slug": SLUG}
    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        results["ready_after_s"] = round(time.time() - t0, 1)
        print(f"ready after {results['ready_after_s']} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{os.environ['INFERENCEKEY_PROJECT']}/{ref.workload_slug}/v1/chat/completions"
        key = os.environ["INFERENCEKEY_API_KEY"]

        rows = load_aime()
        if args.problems != "all":
            wanted = {int(x) for x in args.problems.split(",")}
            rows = [r for r in rows if r["problem_idx"] in wanted]
        jobs = [(r, s) for r in rows for s in range(args.samples)]
        print(f"sending {len(jobs)} requests at once")
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            done = list(pool.map(lambda j: solve(url, key, *j), jobs))
        wall = time.time() - t1
        for d in sorted(done, key=lambda d: (d["problem"], d["sample"])):
            mark = "ok " if d["correct"] else "BAD"
            print(f"  {mark} #{d['problem']:>2} s{d['sample']} answer={d['answer']!r} gold={d['gold']} "
                  f"{d['completion_tokens']} tok in {d['total_s']} s")
        tokens = sum(d["completion_tokens"] or 0 for d in done)
        right = sum(d["correct"] for d in done)
        results.update({
            "requests": len(done), "correct": right, "accuracy": round(right / len(done), 4),
            "wall_s": round(wall, 1), "completion_tokens": tokens,
            "aggregate_tokens_per_s": round(tokens / wall, 1), "answers": done,
        })
        print(f"{right}/{len(done)} correct, {tokens} tokens in {wall:.1f} s "
              f"({tokens / wall:.0f} tokens/s across all requests)")
        OUT_DIR.mkdir(exist_ok=True)
        (OUT_DIR / "aime26.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
        print(f"wrote {OUT_DIR / 'aime26.json'}")
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"request failed: {e}: {e.response.text[:300]}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the
        # workload (useful to try more prompts from the dashboard).
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

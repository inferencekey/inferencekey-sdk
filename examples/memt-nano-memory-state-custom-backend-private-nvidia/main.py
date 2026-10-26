"""A model that remembers into a file: MemT-1.0-Nano on a private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> calls -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`memt-1-0-nano`, published with publish.py) that runs MemTensor's
own Transformers code, because the model's memory block and its session API
(write turns into memory, save the state to a file, load it later) exist only
there.

The memory test, all through `/v1/chat/completions` with the operation in
`extra_body.memt`:

1. tell the model facts in two turns with memory on (the plaintext turns are
   committed into the fixed-size memory state and dropped),
2. save the memory state to a file on the worker and close the session,
3. load the file into a brand-new session and ask about the facts,
4. control: ask the same question in a fresh session with no memory.

Answers, token counts, timings and the state file size go to ./outputs/.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

HERE = Path(__file__).resolve().parent
SLUG = os.environ.get("MEMT_WORKLOAD_SLUG", "memt-nano")
BACKEND_SLUG = os.environ.get("MEMT_BACKEND_SLUG", "memt-1-0-nano")  # from publish.py
OUT_DIR = HERE / "outputs"

FACTS = [
    "Hi, I'm Dana Okafor. I run a small bakery in Porto called Forno Azul. "
    "Our main oven is a 2019 Rational iCombi Pro, and our flour supplier's account code is BX-4471.",
    "Two more things for later: I'm allergic to sesame, so never suggest recipes with it, "
    "and my daughter Ines starts school on 9 September.",
]
QUESTIONS = [
    ("What is the name of my bakery, and in which city is it?", ["forno azul", "porto"]),
    ("What is my flour supplier's account code?", ["bx-4471"]),
    ("Suggest one bread I could bake this weekend, keeping my allergy in mind.", ["sesame"]),
    ("When does my daughter start school, and what is her name?", ["ines", "september", "9"]),
]


def _load_dotenv() -> None:
    env = HERE / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def memt(url: str, key: str, op: dict) -> dict:
    """One memory operation. Returns the backend's output plus the round trip."""
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": op.get("text") or op["op"]}],
        "extra_body": {"memt": op},
    }
    t0 = time.time()
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=600)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    # A custom backend returns its own JSON object; the data plane carries it
    # as the message content.
    try:
        out = json.loads(content)
    except (TypeError, ValueError):
        out = {"answer": content}
    out["round_trip_s"] = round(time.time() - t0, 2)
    return out


def show(label: str, out: dict) -> None:
    extra = {k: out.get(k) for k in ("input_tokens", "output_tokens", "seconds", "bytes",
                                     "plaintext_messages_kept", "peak_allocated_gib") if out.get(k) is not None}
    print(f"[{label}] {out.get('answer', '')[:300]!r}\n    {extra} round trip {out['round_trip_s']} s")


def main() -> int:
    _load_dotenv()
    mgmt = ManagementClient.from_env()
    # Reaches the backend process (MEMT_* settings) and how long the worker
    # waits for setup(): venv with torch, a 70 GB download and the load.
    backend_env = {k: os.environ[k] for k in ("MEMT_MAX_NEW_TOKENS", "MEMT_LOAD_IN_8BIT", "MEMT_GPU_CAP_GIB")
                   if os.environ.get(k)}
    spec = WorkloadSpec(
        name="MemT-1.0-Nano",
        slug=SLUG,
        model="MemTensor/MemT-1.0-Nano",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2text",
        worker_id=os.environ["IK_WORKER_ID"],
        config={"ready_timeout_secs": 3000, "request_timeout_secs": 600, "env": backend_env},
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
        execution_policy_config={"replicas": 1},
    )
    results: dict = {"slug": SLUG, "steps": []}
    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}; waiting (first start downloads 70 GB)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=3600)
        results["ready_after_s"] = round(time.time() - t0, 1)
        print(f"ready after {results['ready_after_s']} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{os.environ['INFERENCEKEY_PROJECT']}/{ref.workload_slug}/v1/chat/completions"
        key = os.environ["INFERENCEKEY_API_KEY"]

        def step(label: str, op: dict) -> dict:
            out = memt(url, key, op)
            show(label, out)
            results["steps"].append({"label": label, "request": op, "response": out})
            return out

        for i, fact in enumerate(FACTS, 1):
            step(f"remember {i}", {"op": "remember", "session": "dana", "text": fact})
        step("save", {"op": "save", "session": "dana", "name": "dana"})
        step("close", {"op": "close", "session": "dana"})
        step("load", {"op": "load", "session": "dana-restored", "name": "dana"})
        score = 0
        for q, expected in QUESTIONS:
            out = step("ask (restored)", {"op": "ask", "session": "dana-restored", "text": q})
            hit = all(e in out.get("answer", "").lower() for e in expected)
            out["expected"], out["all_expected_found"] = expected, hit
            score += hit
        results["restored_score"] = f"{score}/{len(QUESTIONS)}"
        control = step("ask (no memory)", {"op": "ask", "session": "control", "text": QUESTIONS[1][0]})
        results["control_knows_code"] = "bx-4471" in control.get("answer", "").lower()
        step("close control", {"op": "close", "session": "control"})
        print(f"restored session: {results['restored_score']} answers with every expected word; "
              f"control session knows the code: {results['control_knows_code']}")

        OUT_DIR.mkdir(exist_ok=True)
        (OUT_DIR / "memory-test.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
        print(f"wrote {OUT_DIR / 'memory-test.json'}")
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"request failed: {e}: {e.response.text[:300]}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the
        # workload (useful to try the memory from the dashboard playground).
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

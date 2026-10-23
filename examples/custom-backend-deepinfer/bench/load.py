"""Load driver: N concurrent streaming chats against the simulated DeepInfer backend.

For each concurrency level it starts a fresh ``sim_server.py`` (the backend
process under test, the "parent"), opens N streaming ``/v1/chat/completions``
requests at once from a few client processes (so the clients are never the
bottleneck), and reports:

* total output tok/s (completion tokens / wall time) and the steady rate (median
  of the per-second token counts in the middle 80 % of the run);
* per-stream tok/s (completion tokens / time from first to last token) and TTFT
  (from the moment that stream connects);
* CPU of the parent process, of each rank process and of the clients, sampled
  from /proc twice a second;
* what the simulated ranks produced (their own counters): tokens per second
  while they were busy, their step rate and when they finished, against when the
  clients got their last token (the gap is backlog queued in the parent); and
  the ceiling the ranks allow: min(N, dp * max_num_seqs) * step_rate * tokens_per_step.

    python load.py --tokenizer-dir /path/to/tokenizer --concurrency 50,200,400
    python load.py --tokenizer-dir /path/to/tokenizer --concurrency 50 --prompt-tokens 137000
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import multiprocessing as mp
import os
import pathlib
import random
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional

HERE = pathlib.Path(__file__).resolve().parent
CLK_TCK = os.sysconf("SC_CLK_TCK")
BUCKET_S = 1.0


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


def repository_context(chars: int, seed: int) -> str:
    """Deterministic pseudo source code, like the rehearsal's 03_concurrency.py context."""
    rng = random.Random(seed)
    words = ["def", "return", "self", "config", "request", "response", "user", "token",
             "cache", "worker", "result", "error", "value", "items", "async", "await"]
    lines, size, n = [], 0, 0
    while size < chars:
        n += 1
        line = f"    {rng.choice(words)}_{n} = {rng.choice(words)}({rng.choice(words)}, {n})  # line {n}"
        lines.append(line)
        size += len(line) + 1
    return "# File: service.py (repository context)\n" + "\n".join(lines)


def chars_per_token(tokenizer_dir: str) -> float:
    """Characters per token of the context text, measured with the real tokenizer if there is one."""
    path = os.path.join(tokenizer_dir, "tokenizer.json") if tokenizer_dir else ""
    if path and os.path.exists(path):
        from tokenizers import Tokenizer  # noqa: PLC0415

        sample = repository_context(200_000, 0)
        return len(sample) / len(Tokenizer.from_file(path).encode(sample).ids)
    return 4.0


def request_body(user: int, prompt_chars: int, max_tokens: int) -> bytes:
    question = f"Developer {user}: explain how the scheduler batches requests, in detail."
    messages: List[Dict[str, str]] = []
    if prompt_chars:
        messages.append({"role": "system", "content": repository_context(prompt_chars, user)})
    messages.append({"role": "user", "content": question})
    body = {"model": "deepseek-v4.1-flash", "messages": messages, "stream": True,
            "max_tokens": max_tokens, "temperature": 0.6}
    return json.dumps(body).encode()


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------


async def _stream(user: int, port: int, body: bytes, t0: float, buckets: Dict[int, float],
                  timeout: float) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"user": user, "ok": False, "chunks": 0, "completion_tokens": 0,
                           "prompt_tokens": 0, "first": None, "last": None, "error": ""}
    writer = None
    chars: Dict[int, int] = {}
    try:
        rec["sent"] = time.time() - t0
        async with asyncio.timeout(timeout):
            reader, writer = await asyncio.open_connection("127.0.0.1", port, limit=1 << 22)
            rec["connected"] = time.time() - t0
            head = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                    f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode()
            writer.write(head + body)
            await writer.drain()
            status = await reader.readline()
            if b" 200 " not in status:
                rec["error"] = status.decode(errors="replace").strip() or "no status line"
                return rec
            while (await reader.readline()) not in (b"\r\n", b""):
                pass
            while True:
                line = await reader.readline()
                if not line:
                    rec["error"] = rec["error"] or "stream closed before [DONE]"
                    break
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    rec["ok"] = not rec["error"]
                    break
                obj = json.loads(payload)
                if "error" in obj:
                    rec["error"] = str(obj["error"])
                    continue
                choices = obj.get("choices")
                if choices:
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content") or delta.get("reasoning_content")
                    if text:
                        now = time.time()
                        rec["chunks"] += 1
                        rec["first"] = rec["first"] or now - t0
                        rec["last"] = now - t0
                        key = int((now - t0) / BUCKET_S)
                        chars[key] = chars.get(key, 0) + len(text)
                elif obj.get("usage"):
                    rec["completion_tokens"] = obj["usage"]["completion_tokens"]
                    rec["prompt_tokens"] = obj["usage"]["prompt_tokens"]
    except (TimeoutError, OSError, ValueError) as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        rec["end"] = time.time() - t0
        if writer is not None:
            writer.close()
    # A chunk carries one or more tokens (the backend may batch them), so the
    # per-second token counts spread this stream's tokens over its text.
    total_chars = sum(chars.values())
    if rec["ok"] and total_chars:
        scale = rec["completion_tokens"] / total_chars
        for key, n in chars.items():
            buckets[key] = buckets.get(key, 0) + n * scale
    return rec


def _client_main(users: List[int], n: int, port: int, prompt_chars: int, max_tokens: int, t0: float,
                 ramp_s: float, timeout: float, out: Any) -> None:
    bodies = {u: request_body(u, prompt_chars, max_tokens) for u in users}

    async def run() -> Dict[str, Any]:
        buckets: Dict[int, float] = {}

        async def one(u: int, delay: float) -> Dict[str, Any]:
            await asyncio.sleep(max(t0 + delay - time.time(), 0))
            return await _stream(u, port, bodies[u], t0, buckets, timeout)

        recs = await asyncio.gather(*(one(u, ramp_s * u / n) for u in users))
        return {"records": list(recs), "buckets": buckets}

    out.put(asyncio.run(run()))


# --------------------------------------------------------------------------
# Server and CPU sampling
# --------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cpu_ticks(pid: int, system_only: bool = False) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as fh:
            fields = fh.read().rsplit(")", 1)[1].split()
        if system_only:
            return int(fields[12])  # stime: kernel work (socket writes, futexes)
        return int(fields[11]) + int(fields[12])  # utime + stime
    except (OSError, IndexError, ValueError):
        return None


def _rank_pids(parent: int) -> List[int]:
    pids = []
    for stat in glob.glob("/proc/[0-9]*/stat"):
        try:
            with open(stat, encoding="ascii") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
            if int(fields[1]) != parent:
                continue
            pid = int(stat.split("/")[2])
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                if b"spawn_main" in fh.read():
                    pids.append(pid)
        except (OSError, IndexError, ValueError):
            continue
    return sorted(pids)


class CpuSampler(threading.Thread):
    """Samples the CPU % of the parent, the ranks and the clients every 0.5 s."""

    def __init__(self, parent: int, ranks: List[int], clients: List[int]) -> None:
        super().__init__(daemon=True)
        self.groups = {"parent": [parent], **{f"rank{i}": [p] for i, p in enumerate(ranks)}, "clients": clients}
        self.samples: List[Dict[str, float]] = []
        self.stop = threading.Event()

    def _ticks(self) -> Dict[str, int]:
        ticks = {name: sum(_cpu_ticks(p) or 0 for p in pids) for name, pids in self.groups.items()}
        ticks["parent_sys"] = _cpu_ticks(self.groups["parent"][0], system_only=True) or 0
        return ticks

    def run(self) -> None:
        last_t, last = time.time(), self._ticks()
        while not self.stop.wait(0.5):
            now, cur = time.time(), self._ticks()
            dt = now - last_t
            sample = {"t": now}
            sample.update({k: 100.0 * (cur[k] - last[k]) / CLK_TCK / dt for k in cur})
            self.samples.append(sample)
            last_t, last = now, cur


def start_server(args: argparse.Namespace, port: int, stats_path: str) -> subprocess.Popen:
    cmd = [sys.executable, str(HERE / "sim_server.py"), "--port", str(port), "--dp", str(args.dp),
           "--step-rate", str(args.step_rate), "--tokens-per-step", str(args.tokens_per_step),
           "--max-num-seqs", str(args.max_num_seqs), "--prefill-rate", str(args.prefill_rate),
           "--stats-path", stats_path]
    cmd += ["--fake-tokenizer"] if args.fake_tokenizer else ["--tokenizer-dir", args.tokenizer_dir]
    if args.no_lockstep:
        cmd.append("--no-lockstep")
    if args.backend_config:
        cmd += ["--backend-config", args.backend_config]
    log = open(stats_path + ".server.log", "w", encoding="utf-8")  # noqa: SIM115
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"sim_server exited; see {stats_path}.server.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as resp:
                if resp.status == 200:
                    return proc
        except OSError:
            pass
        time.sleep(0.3)
    raise TimeoutError("sim_server not ready after 120 s")


def stop_server(proc: subprocess.Popen) -> None:
    # The ranks are non-daemon and wait on each other, so kill the whole group.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def rank_production(stats_path: str, t0: float) -> Dict[str, float]:
    """What the ranks report: tokens, when they produced them, and their step rate meanwhile.

    The counters are sampled once a second, so times are good to about a second.
    """
    total = 0
    first, last = float("inf"), 0.0
    step_rates = []
    for path in glob.glob(stats_path + ".rank*"):
        with open(path, encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        busy = [(a, b) for a, b in zip(lines, lines[1:]) if b["tokens"] > a["tokens"]]
        if not busy:
            continue
        total += lines[-1]["tokens"]
        first, last = min(first, busy[0][0]["t"]), max(last, busy[-1][1]["t"])
        span = busy[-1][1]["t"] - busy[0][0]["t"]
        step_rates.append((busy[-1][1]["steps"] - busy[0][0]["steps"]) / max(span, 1e-9))
    if not step_rates:
        return {"tokens": 0, "busy_tok_s": 0.0, "first_s": 0.0, "done_s": 0.0, "steps_per_s": 0.0}
    return {
        "tokens": total,
        "busy_tok_s": total / max(last - first, 1e-9),
        "first_s": first - t0,
        "done_s": last - t0,
        "steps_per_s": statistics.mean(step_rates),
    }


# --------------------------------------------------------------------------
# One level
# --------------------------------------------------------------------------


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(int(q * len(values)), len(values) - 1)]


def run_level(args: argparse.Namespace, n: int, prompt_chars: int) -> Dict[str, Any]:
    port = _free_port()
    workdir = tempfile.mkdtemp(prefix="deepinfer-bench-")
    stats_path = os.path.join(workdir, "stats")
    server = start_server(args, port, stats_path)
    try:
        ranks = []
        for _ in range(50):
            ranks = _rank_pids(server.pid)
            if len(ranks) >= args.dp:
                break
            time.sleep(0.1)
        ctx = mp.get_context("fork")
        out = ctx.Queue()
        n_clients = max(1, min(args.client_procs, n))
        t0 = time.time() + 1.0 + prompt_chars * n / 2e8  # time to build the bodies
        clients = [
            ctx.Process(target=_client_main,
                        args=(list(range(i, n, n_clients)), n, port, prompt_chars, args.max_tokens, t0,
                              args.ramp_s, args.timeout, out))
            for i in range(n_clients)
        ]
        for c in clients:
            c.start()
        sampler = CpuSampler(server.pid, ranks, [c.pid for c in clients])
        sampler.start()
        parts = [out.get() for _ in clients]
        for c in clients:
            c.join()
        sampler.stop.set()
        sampler.join()
        records = [r for p in parts for r in p["records"]]
        buckets: Dict[int, float] = {}
        for p in parts:
            for k, v in p["buckets"].items():
                buckets[k] = buckets.get(k, 0) + v
        produced = rank_production(stats_path, t0)
    finally:
        stop_server(server)

    ok = [r for r in records if r["ok"]]
    errors = [r["error"] for r in records if not r["ok"]]
    wall = max((r["end"] for r in records), default=0.0)
    total_tokens = sum(r["completion_tokens"] for r in ok)
    per_stream = [r["completion_tokens"] / (r["last"] - r["first"]) for r in ok
                  if r["first"] is not None and r["last"] > r["first"]]
    ttft = [r["first"] - r["sent"] for r in ok if r["first"] is not None]
    last_token = max((r["last"] or 0 for r in ok), default=0.0)
    keys = [k for k in sorted(buckets) if 0.1 * last_token <= k * BUCKET_S and (k + 1) * BUCKET_S <= 0.9 * last_token]
    steady = statistics.median([buckets[k] / BUCKET_S for k in keys]) if keys else 0.0
    t_lo, t_hi = t0 + min((r["first"] for r in ok if r["first"] is not None), default=0.0), t0 + last_token
    window = [s for s in sampler.samples if t_lo <= s["t"] <= t_hi] or sampler.samples
    cpu = {k: statistics.mean(s[k] for s in window) for k in window[0] if k != "t"} if window else {}
    ceiling = min(n, args.dp * args.max_num_seqs) * args.step_rate * args.tokens_per_step
    return {
        "concurrency": n,
        "prompt_tokens_target": args.prompt_tokens,
        "prompt_tokens_reported": statistics.median([r["prompt_tokens"] for r in ok]) if ok else 0,
        "max_tokens": args.max_tokens,
        "ok": len(ok),
        "errors": len(errors),
        "error_samples": sorted(set(errors))[:5],
        "wall_s": round(wall, 2),
        "completion_tokens": total_tokens,
        "total_tok_s": round(total_tokens / wall, 1) if wall else 0.0,
        "steady_tok_s": round(steady, 1),
        "per_stream_tok_s": {"p10": round(_pct(per_stream, 0.1), 1), "median": round(_pct(per_stream, 0.5), 1),
                             "p90": round(_pct(per_stream, 0.9), 1)},
        "ttft_s": {"median": round(_pct(ttft, 0.5), 2), "p90": round(_pct(ttft, 0.9), 2),
                   "max": round(max(ttft, default=0.0), 2)},
        "cpu_pct_mean": {k: round(v, 1) for k, v in cpu.items()},
        "cpu_pct_parent_max": round(max((s["parent"] for s in window), default=0.0), 1),
        "clients_done_s": round(last_token, 2),
        "ranks_done_s": round(produced["done_s"], 2),
        "ranks_tokens": produced["tokens"],
        "ranks_busy_tok_s": round(produced["busy_tok_s"], 1),
        "ranks_steps_per_s": round(produced["steps_per_s"], 1),
        "sim_ceiling_tok_s": ceiling,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--concurrency", default="50,200,400")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, default=0,
                        help="add a repository context of about this many tokens to every prompt")
    parser.add_argument("--tokenizer-dir", default=os.environ.get("BENCH_TOKENIZER_DIR", ""))
    parser.add_argument("--fake-tokenizer", action="store_true")
    parser.add_argument("--dp", type=int, default=4)
    parser.add_argument("--step-rate", type=float, default=70.0)
    parser.add_argument("--tokens-per-step", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--prefill-rate", type=float, default=88000.0)
    parser.add_argument("--no-lockstep", action="store_true")
    parser.add_argument("--client-procs", type=int, default=4)
    parser.add_argument("--backend-config", default="",
                        help="JSON object merged into the backend config (e.g. stream_interval_ms, tokenizer_threads)")
    parser.add_argument("--ramp-s", type=float, default=0.0, help="spread the stream starts over this long")
    parser.add_argument("--timeout", type=float, default=900.0, help="per stream, seconds")
    parser.add_argument("--out", default="", help="append the results as JSON lines to this file")
    args = parser.parse_args()
    if not args.fake_tokenizer and not args.tokenizer_dir:
        parser.error("pass --tokenizer-dir (or BENCH_TOKENIZER_DIR) or --fake-tokenizer")

    prompt_chars = int(args.prompt_tokens * chars_per_token("" if args.fake_tokenizer else args.tokenizer_dir))
    for n in [int(x) for x in args.concurrency.split(",") if x]:
        result = run_level(args, n, prompt_chars)
        result["tokenizer"] = "fake" if args.fake_tokenizer else "hf"
        result["step_rate"] = args.step_rate
        result["tokens_per_step"] = args.tokens_per_step
        result["prefill_rate"] = args.prefill_rate
        result["ramp_s"] = args.ramp_s
        result["lockstep"] = not args.no_lockstep
        if args.backend_config:
            result["backend_config"] = json.loads(args.backend_config)
        print(json.dumps(result), flush=True)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()

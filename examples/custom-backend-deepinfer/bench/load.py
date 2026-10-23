"""Load driver: N concurrent streaming chats against a custom backend.

For each concurrency level it starts a fresh backend process (the "parent"):
``--backend deepinfer`` runs ``sim_server.py`` (the DeepInfer backend with
simulated ranks), ``--backend null`` runs ``null_backend.py`` through the real
``python -m inferencekey.backend.serve`` CLI (a backend that only streams paced
tokens, so it measures the runtime's HTTP/SSE layer alone). ``--transport``
picks the runtime's HTTP transport (``IK_BACKEND_HTTP``) and the run is refused
unless the backend's ``http transport:`` startup line agrees (a runtime without
the line counts as ``stdlib``). It opens N streaming ``/v1/chat/completions``
requests at once from a few client processes (so the clients are never the
bottleneck), probes ``/healthz`` on the side like the worker does (every
500 ms, 3 s timeout), and reports:

* total output tok/s (completion tokens / wall time) and the steady rate (median
  of the per-second token counts in the middle 80 % of the run);
* per-stream tok/s (completion tokens / time from first to last token), TTFT
  (from the moment that stream connects) and the gap between consecutive
  content chunks;
* errors by kind (reset, timeout, refused, eof before ``[DONE]``, status);
* ``/healthz`` latency (p50, p99, max) and misses (no answer within 3 s);
* CPU of the parent process, of its busiest thread, of each rank process and of
  the clients, and the parent's peak RSS, sampled from /proc twice a second;
* with py-spy (``--py-spy``, on by default when it is installed), the fraction
  of time some parent thread held the GIL and where it was (innermost frame);
* the native transport's ``/debug/stats`` frames-to-chunks ratio, when served;
* what the simulated ranks produced (their own counters): tokens per second
  while they were busy, their step rate and when they finished, against when the
  clients got their last token (the gap is backlog queued in the parent); and
  the ceiling the ranks allow: min(N, dp * max_num_seqs) * step_rate * tokens_per_step.

    python load.py --backend null --concurrency 50,200,400,800 --out results/x.jsonl
    python load.py --tokenizer-dir /path/to/tokenizer --concurrency 50,200,400
    python load.py --tokenizer-dir /path/to/tokenizer --concurrency 50 --prompt-tokens 137000
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import http.client
import json
import multiprocessing as mp
import os
import pathlib
import random
import re
import shutil
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


def _error_kind(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionResetError):
        return "reset"
    if isinstance(exc, ConnectionRefusedError):
        return "refused"
    if isinstance(exc, OSError):
        return "oserror"
    return "protocol"


async def _stream(user: int, port: int, body: bytes, t0: float, buckets: Dict[int, float],
                  timeout: float) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"user": user, "ok": False, "chunks": 0, "completion_tokens": 0,
                           "prompt_tokens": 0, "first": None, "last": None, "error": "",
                           "error_kind": "", "gaps": []}
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
                rec["error_kind"] = "status" if status else "eof"
                return rec
            while (await reader.readline()) not in (b"\r\n", b""):
                pass
            while True:
                line = await reader.readline()
                if not line:
                    rec["error"] = rec["error"] or "stream closed before [DONE]"
                    rec["error_kind"] = rec["error_kind"] or "eof"
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
                    rec["error_kind"] = "error_event"
                    continue
                choices = obj.get("choices")
                if choices:
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content") or delta.get("reasoning_content")
                    if text:
                        now = time.time()
                        if rec["last"] is not None:
                            rec["gaps"].append(now - t0 - rec["last"])
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
        rec["error_kind"] = _error_kind(exc)
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


def _thread_ticks(pid: int) -> Dict[int, Any]:
    """(utime + stime, utime, name) of every thread of ``pid``."""
    out: Dict[int, Any] = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        try:
            with open(f"/proc/{pid}/task/{tid}/stat", encoding="ascii", errors="replace") as fh:
                raw = fh.read()
            name = raw[raw.index("(") + 1:raw.rindex(")")]
            fields = raw.rsplit(")", 1)[1].split()
            out[int(tid)] = (int(fields[11]) + int(fields[12]), int(fields[11]), name)
        except (OSError, IndexError, ValueError):
            continue
    return out


def _rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


#: Threads of the native transport (tokio IO workers) never run Python.
NATIVE_THREAD_PREFIXES = ("tokio", "ik-http-io")


class CpuSampler(threading.Thread):
    """Samples the CPU % of the parent, its threads, the ranks and the clients, and the parent's RSS, every 0.5 s.

    The per-thread numbers come from /proc/<pid>/task, the same source as
    ``pidstat -t``; ``busiest_thread`` is the thread with the most CPU time over
    a window, as a percentage of one core.
    """

    def __init__(self, parent: int, ranks: List[int], clients: List[int]) -> None:
        super().__init__(daemon=True)
        self.parent = parent
        self.groups = {"parent": [parent], **{f"rank{i}": [p] for i, p in enumerate(ranks)}, "clients": clients}
        self.samples: List[Dict[str, float]] = []
        self.thread_samples: List[Any] = []  # (t, dt, {tid: (delta ticks, name)})
        self.stop = threading.Event()

    def _ticks(self) -> Dict[str, int]:
        ticks = {name: sum(_cpu_ticks(p) or 0 for p in pids) for name, pids in self.groups.items()}
        ticks["parent_sys"] = _cpu_ticks(self.groups["parent"][0], system_only=True) or 0
        return ticks

    def run(self) -> None:
        last_t, last, last_threads = time.time(), self._ticks(), _thread_ticks(self.parent)
        while not self.stop.wait(0.5):
            now, cur, threads = time.time(), self._ticks(), _thread_ticks(self.parent)
            dt = now - last_t
            sample = {"t": now, "rss_mb": _rss_mb(self.parent)}
            sample.update({k: 100.0 * (cur[k] - last[k]) / CLK_TCK / dt for k in cur})
            self.samples.append(sample)
            deltas = {tid: (ticks - last_threads.get(tid, (0, 0, ""))[0], user - last_threads.get(tid, (0, 0, ""))[1],
                            name)
                      for tid, (ticks, user, name) in threads.items()}
            self.thread_samples.append((now, dt, {k: v for k, v in deltas.items() if v[0] > 0}))
            last_t, last, last_threads = now, cur, threads

    def busiest_thread(self, t_lo: float, t_hi: float) -> Dict[str, Any]:
        window = [x for x in self.thread_samples if t_lo <= x[0] <= t_hi] or self.thread_samples
        if not window:
            return {"pct": 0.0, "name": ""}
        span = sum(dt for _, dt, _ in window)
        totals: Dict[int, int] = {}
        names: Dict[int, str] = {}
        native_user = 0
        for _, _, deltas in window:
            for tid, (ticks, user, name) in deltas.items():
                totals[tid] = totals.get(tid, 0) + ticks
                names[tid] = name
                if name.startswith(NATIVE_THREAD_PREFIXES):
                    native_user += user
        if not totals:
            return {"pct": 0.0, "name": ""}
        tid = max(totals, key=totals.get)
        return {"pct": round(100.0 * totals[tid] / CLK_TCK / span, 1), "name": names[tid],
                "main": tid == self.parent, "threads_seen": len(totals),
                "native_user_cores": round(native_user / CLK_TCK / span, 3)}


class HealthProber(threading.Thread):
    """GET /healthz every 0.5 s with a 3 s timeout, like the worker's readiness and liveness checks.

    A miss is what the worker counts as one: no answer within 3 s, a connection
    error, or a status other than 200 and 503.
    """

    def __init__(self, port: int, interval: float = 0.5, timeout: float = 3.0) -> None:
        super().__init__(daemon=True)
        self.port, self.interval, self.timeout = port, interval, timeout
        self.latencies: List[float] = []
        self.misses = 0
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
                conn.request("GET", "/healthz")
                resp = conn.getresponse()
                resp.read()
                conn.close()
                elapsed = time.monotonic() - started
                if resp.status in (200, 503) and elapsed <= self.timeout:
                    self.latencies.append(elapsed)
                else:
                    self.misses += 1
            except (OSError, http.client.HTTPException):
                self.misses += 1
            self.stop.wait(max(self.interval - (time.monotonic() - started), 0.0))

    def summary(self) -> Dict[str, Any]:
        ms = [x * 1000.0 for x in self.latencies]
        return {"probes": len(ms) + self.misses, "misses": self.misses,
                "p50_ms": round(_pct(ms, 0.5), 1), "p99_ms": round(_pct(ms, 0.99), 1),
                "max_ms": round(max(ms, default=0.0), 1)}


TRANSPORT_LINE = re.compile(r"^http transport: (\w+)(.*)$", re.MULTILINE)


def transport_line(log_path: str) -> Optional[str]:
    """The backend's ``http transport: ...`` startup line, or None if it printed none."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            match = TRANSPORT_LINE.search(fh.read())
    except OSError:
        return None
    return match.group(0).strip() if match else None


def check_transport(requested: str, line: Optional[str]) -> None:
    """Refuse to report numbers for a transport other than the one asked for.

    A runtime that prints no transport line predates the native transport, so
    it can only be serving with the standard library.
    """
    served = TRANSPORT_LINE.match(line).group(1) if line else "stdlib"
    if served != requested:
        raise RuntimeError(f"asked for transport {requested!r} but the backend says: {line or '(no transport line)'}")


def server_command(args: argparse.Namespace, port: int, stats_path: str) -> List[str]:
    if args.backend == "null":
        config = {"tok_s": args.tok_s, "chunk_format": args.chunk_format}
        if args.backend_config:
            config.update(json.loads(args.backend_config))
        return [sys.executable, "-m", "inferencekey.backend.serve", "--port", str(port),
                "--backend", "null_backend:NullBackend", "--config-json", json.dumps(config)]
    cmd = [sys.executable, str(HERE / "sim_server.py"), "--port", str(port), "--dp", str(args.dp),
           "--step-rate", str(args.step_rate), "--tokens-per-step", str(args.tokens_per_step),
           "--max-num-seqs", str(args.max_num_seqs), "--prefill-rate", str(args.prefill_rate),
           "--stats-path", stats_path]
    cmd += ["--fake-tokenizer"] if args.fake_tokenizer else ["--tokenizer-dir", args.tokenizer_dir]
    if args.no_lockstep:
        cmd.append("--no-lockstep")
    if args.backend_config:
        cmd += ["--backend-config", args.backend_config]
    return cmd


def start_server(args: argparse.Namespace, port: int, stats_path: str) -> subprocess.Popen:
    env = dict(os.environ, IK_BACKEND_HTTP=args.transport)
    env["PYTHONPATH"] = os.pathsep.join(x for x in (str(HERE), env.get("PYTHONPATH", "")) if x)
    log = open(stats_path + ".server.log", "w", encoding="utf-8")  # noqa: SIM115
    proc = subprocess.Popen(server_command(args, port, stats_path), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, cwd=str(HERE), env=env)
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"backend exited; see {stats_path}.server.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as resp:
                if resp.status == 200:
                    return proc
        except OSError:
            pass
        time.sleep(0.3)
    raise TimeoutError("backend not ready after 120 s")


def find_py_spy() -> str:
    beside = pathlib.Path(sys.executable).with_name("py-spy")
    return str(beside) if beside.exists() else (shutil.which("py-spy") or "")


class GilProfile:
    """``py-spy record --gil --nonblocking`` on the parent while the streams run.

    py-spy keeps only the samples where a thread holds the GIL, so the number of
    samples over (rate x seconds) estimates the fraction of time the GIL was
    held. Without pausing the process (``--nonblocking``, which a blocking
    py-spy needs: pausing 400 threads 100 times a second halves the throughput)
    py-spy reads the GIL's last holder, so a thread that has just released the
    GIL around a socket write or a ``time.sleep`` still counts: read it as an
    upper bound, next to ``busiest_thread.python_user_cores``. The innermost
    frame of each sample says where the time went.
    """

    SERVING_FILES = ("socketserver.py", "http/server.py", "inferencekey/backend/serve.py",
                     "json/encoder.py", "json/__init__.py", "socket.py")

    def __init__(self, binary: str, pid: int, out_path: str, rate: int) -> None:
        self.out_path, self.rate = out_path, rate
        self.log = open(out_path + ".log", "w", encoding="utf-8")  # noqa: SIM115
        self.started = time.monotonic()
        self.proc = subprocess.Popen([binary, "record", "--gil", "--nonblocking", "-r", str(rate), "-p", str(pid),
                                      "-f", "raw", "-o", out_path], stdout=self.log, stderr=subprocess.STDOUT)

    def finish(self) -> Dict[str, Any]:
        elapsed = time.monotonic() - self.started
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()
        leaves: Dict[str, int] = {}
        samples = serving = 0
        try:
            with open(self.out_path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    stack, _, count = line.rstrip("\n").rpartition(" ")
                    if not stack or not count.isdigit():
                        continue
                    n = int(count)
                    leaf = stack.rsplit(";", 1)[-1]
                    samples += n
                    leaves[leaf] = leaves.get(leaf, 0) + n
                    if any(f in leaf for f in self.SERVING_FILES):
                        serving += n
        except OSError:
            return {"error": f"no py-spy output; see {self.out_path}.log"}
        top = sorted(leaves.items(), key=lambda kv: -kv[1])[:5]
        return {
            "gil_held_fraction": round(samples / max(self.rate * elapsed, 1e-9), 3),
            "samples": samples,
            "seconds": round(elapsed, 1),
            "serving_leaf_share": round(serving / samples, 3) if samples else 0.0,
            "top_leaves": [[leaf, round(n / samples, 3)] for leaf, n in top] if samples else [],
        }


def debug_stats(port: int) -> Optional[Dict[str, Any]]:
    """The native transport's ``/debug/stats`` counters, or None where the route does not exist."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/debug/stats", timeout=3) as resp:
            return json.loads(resp.read().decode())
    except (OSError, ValueError):
        return None


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
    workdir = tempfile.mkdtemp(prefix="ik-bench-")
    stats_path = os.path.join(workdir, "stats")
    server = start_server(args, port, stats_path)
    profile: Optional[GilProfile] = None
    try:
        line = transport_line(stats_path + ".server.log")
        check_transport(args.transport, line)
        ranks: List[int] = []
        if args.backend == "deepinfer":
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
        time.sleep(max(t0 - time.time(), 0))
        prober = HealthProber(port)
        prober.start()
        if args.py_spy:
            profile = GilProfile(args.py_spy, server.pid, stats_path + ".pyspy.txt", args.py_spy_rate)
        parts = [out.get() for _ in clients]
        gil = profile.finish() if profile else None
        profile = None
        prober.stop.set()
        for c in clients:
            c.join()
        sampler.stop.set()
        sampler.join()
        prober.join()
        stats = debug_stats(port)
        records = [r for p in parts for r in p["records"]]
        buckets: Dict[int, float] = {}
        for p in parts:
            for k, v in p["buckets"].items():
                buckets[k] = buckets.get(k, 0) + v
        produced = rank_production(stats_path, t0) if args.backend == "deepinfer" else None
    finally:
        if profile is not None:
            profile.finish()
        stop_server(server)

    ok = [r for r in records if r["ok"]]
    errors = [r["error"] for r in records if not r["ok"]]
    error_kinds: Dict[str, int] = {}
    for r in records:
        if not r["ok"]:
            kind = r["error_kind"] or "other"
            error_kinds[kind] = error_kinds.get(kind, 0) + 1
    wall = max((r["end"] for r in records), default=0.0)
    total_tokens = sum(r["completion_tokens"] for r in ok)
    per_stream = [r["completion_tokens"] / (r["last"] - r["first"]) for r in ok
                  if r["first"] is not None and r["last"] > r["first"]]
    ttft = [r["first"] - r["sent"] for r in ok if r["first"] is not None]
    gaps = [g for r in records for g in r["gaps"]]
    last_token = max((r["last"] or 0 for r in ok), default=0.0)
    keys = [k for k in sorted(buckets) if 0.1 * last_token <= k * BUCKET_S and (k + 1) * BUCKET_S <= 0.9 * last_token]
    steady = statistics.median([buckets[k] / BUCKET_S for k in keys]) if keys else 0.0
    t_lo, t_hi = t0 + min((r["first"] for r in ok if r["first"] is not None), default=0.0), t0 + last_token
    window = [s for s in sampler.samples if t_lo <= s["t"] <= t_hi] or sampler.samples
    cpu = {k: statistics.mean(s[k] for s in window) for k in window[0] if k not in ("t", "rss_mb")} if window else {}
    busiest = sampler.busiest_thread(t_lo, t_hi)
    if args.backend == "deepinfer":
        offered = min(n, args.dp * args.max_num_seqs) * args.step_rate * args.tokens_per_step
    else:
        offered = n * args.tok_s
    result: Dict[str, Any] = {
        "backend": args.backend,
        "transport": args.transport,
        "transport_line": line,
        "concurrency": n,
        "prompt_tokens_target": args.prompt_tokens,
        "prompt_tokens_reported": statistics.median([r["prompt_tokens"] for r in ok]) if ok else 0,
        "max_tokens": args.max_tokens,
        "ok": len(ok),
        "errors": len(errors),
        "error_kinds": error_kinds,
        "error_samples": sorted(set(errors))[:5],
        "done_pct": round(100.0 * len(ok) / n, 1) if n else 0.0,
        "wall_s": round(wall, 2),
        "completion_tokens": total_tokens,
        "total_tok_s": round(total_tokens / wall, 1) if wall else 0.0,
        "steady_tok_s": round(steady, 1),
        "offered_tok_s": offered,
        "delivered_pct": round(100.0 * steady / offered, 1) if offered else 0.0,
        "per_stream_tok_s": {"p5": round(_pct(per_stream, 0.05), 1), "p10": round(_pct(per_stream, 0.1), 1),
                             "median": round(_pct(per_stream, 0.5), 1), "p90": round(_pct(per_stream, 0.9), 1)},
        "ttft_s": {"median": round(_pct(ttft, 0.5), 3), "p90": round(_pct(ttft, 0.9), 3),
                   "p99": round(_pct(ttft, 0.99), 3), "max": round(max(ttft, default=0.0), 3)},
        "gap_ms": {"median": round(1000 * _pct(gaps, 0.5), 1), "p99": round(1000 * _pct(gaps, 0.99), 1),
                   "max": round(1000 * max(gaps, default=0.0), 1)},
        "healthz": prober.summary(),
        "cpu_pct_mean": {k: round(v, 1) for k, v in cpu.items()},
        "cpu_pct_parent_max": round(max((s["parent"] for s in window), default=0.0), 1),
        "busiest_thread": busiest,
        # User-mode CPU of the threads that run Python, in cores: an upper bound
        # of the GIL-held fraction (C code that releases the GIL, like a fast
        # tokenizer, also counts). Per-thread samples miss threads that live less
        # than 0.5 s, so this is the process's user time minus the native threads'.
        "python_user_cores": round(max(cpu.get("parent", 0.0) - cpu.get("parent_sys", 0.0), 0.0) / 100.0
                                   - busiest.get("native_user_cores", 0.0), 3),
        "rss_mb_max": round(max((s["rss_mb"] for s in sampler.samples), default=0.0), 1),
        "gil": gil,
        "clients_done_s": round(last_token, 2),
    }
    if stats is not None:
        chunks = stats.get("chunks_pushed") or 0
        result["debug_stats"] = stats
        result["frames_per_chunk"] = round(stats.get("frames_written", 0) / chunks, 3) if chunks else None
    if produced is not None:
        result.update({
            "ranks_done_s": round(produced["done_s"], 2),
            "ranks_tokens": produced["tokens"],
            "ranks_busy_tok_s": round(produced["busy_tok_s"], 1),
            "ranks_steps_per_s": round(produced["steps_per_s"], 1),
            "sim_ceiling_tok_s": offered,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=("deepinfer", "null"), default="deepinfer")
    parser.add_argument("--transport", choices=("stdlib", "native"), default="stdlib",
                        help="IK_BACKEND_HTTP for the backend; the run is refused if its startup line disagrees")
    parser.add_argument("--concurrency", default="50,200,400")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--tok-s", type=float, default=50.0, help="null backend: tokens per second per stream")
    parser.add_argument("--chunk-format", choices=("dict", "str"), default="dict",
                        help="null backend: yield dict chunks or pre-serialized str chunks")
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
    parser.add_argument("--py-spy", default=find_py_spy(),
                        help="py-spy binary for the GIL profile (default: beside python or on PATH); '' turns it off")
    parser.add_argument("--py-spy-rate", type=int, default=100, help="py-spy samples per second")
    parser.add_argument("--label", default="", help="free text stored in every result line (for example 'burst')")
    parser.add_argument("--out", default="", help="append the results as JSON lines to this file (e.g. results/x.jsonl)")
    args = parser.parse_args()
    if args.backend == "deepinfer" and not args.fake_tokenizer and not args.tokenizer_dir:
        parser.error("pass --tokenizer-dir (or BENCH_TOKENIZER_DIR) or --fake-tokenizer")

    use_tokenizer = args.backend == "deepinfer" and not args.fake_tokenizer
    prompt_chars = int(args.prompt_tokens * chars_per_token(args.tokenizer_dir if use_tokenizer else ""))
    for n in [int(x) for x in args.concurrency.split(",") if x]:
        result = run_level(args, n, prompt_chars)
        if args.label:
            result["label"] = args.label
        result["ramp_s"] = args.ramp_s
        if args.backend == "null":
            result["tok_s"] = args.tok_s
            result["chunk_format"] = args.chunk_format
        else:
            result["tokenizer"] = "fake" if args.fake_tokenizer else "hf"
            result["step_rate"] = args.step_rate
            result["tokens_per_step"] = args.tokens_per_step
            result["prefill_rate"] = args.prefill_rate
            result["lockstep"] = not args.no_lockstep
        if args.backend_config:
            result["backend_config"] = json.loads(args.backend_config)
        print(json.dumps(result), flush=True)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()

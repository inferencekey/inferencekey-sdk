"""MemT-1.0-Nano (MemTensor) as an InferenceKey custom backend.

MemT-1.0-Nano is Qwen3.6-35B-A3B (36B MoE, 3B active) plus a native memory
block: past turns are written into a fixed-size memory state (10 full-attention
layers x 16 heads x 256 x 256, FP32) instead of being kept as plaintext context
or KV cache. The state can be saved to a safetensors file and loaded into a new
session later. No built-in backend runs it: it needs the authors' modeling code
(``trust_remote_code``) and its ``MemTSession`` API, so this backend drives
Transformers directly.

``setup()`` (once per process)
    downloads the pinned checkpoint (70.3 GB of safetensors) to local disk and
    loads it with Transformers in bfloat16 on GPU 0, or in
    bitsandbytes 8-bit with ``MEMT_LOAD_IN_8BIT=1``. ``MEMT_GPU_CAP_GIB`` caps
    the memory this process may use on the GPU, to emulate a smaller card.

``process(job)`` (per request) keeps named sessions in memory and accepts
    either a structured operation in ``extra_body.memt`` (or a top-level
    ``memt`` field), or plain chat text (the dashboard playground):

    ============  =========================================  ===================
    op            structured form                            chat text form
    ============  =========================================  ===================
    remember      {"op":"remember","session":s,"text":t}     ``remember: <t>``
    ask           {"op":"ask","session":s,"text":t}          any other text
    save          {"op":"save","session":s,"name":n}         ``save: <n>``
    close         {"op":"close","session":s}                 ``forget``
    load          {"op":"load","session":s,"name":n}         ``load: <n>``
    status        {"op":"status"}                            ``status``
    ============  =========================================  ===================

    ``remember`` answers the turn and then commits the plaintext history into
    the memory state (the authors' ``enable_memory=True``): the conversation
    text is dropped. ``ask`` answers from the memory state plus the question
    only. Only the LAST user message of a chat request is used: the model never
    sees earlier turns as text, that is the point of the memory.

The output is a JSON object with ``answer`` first, then what the operation did
(tokens in and out, seconds, memory file size and sha256, peak VRAM).

Settings: ``MEMT_*`` environment variables or the workload config ``env``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

MODEL_ID = "MemTensor/MemT-1.0-Nano"
# Pinned revision (2026-10-20), so the remote code we run is the code we read.
MODEL_REVISION = "30fe0b22422c4cc4c87475f98db5c8baa5c1a902"
PANEL_SESSION = "panel"
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _log(msg: str) -> None:
    print(f"[memt-backend] {msg}", file=sys.stderr, flush=True)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(f"MEMT_{key}", default)


def _gib(n: float) -> float:
    return round(n / 2**30, 2)


class MemTNanoBackend(CustomBackend):
    name = "memt-1.0-nano"
    version = "0.1.1"
    task_type = "text2text"
    description = "MemT-1.0-Nano (36B MoE, 3B active) with a saveable fixed-size memory state"

    # ------------------------------------------------------------------ setup

    def setup(self, ctx: BackendContext) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.timings: Dict[str, float] = {}
        self.max_new_tokens = int(_env("MAX_NEW_TOKENS", "256"))
        self.state_dir = Path(_env("STATE_DIR", str(Path.home() / "memt-states")))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        repo = _env("MODEL", MODEL_ID)
        revision = _env("REVISION", MODEL_REVISION)

        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        t0 = time.time()
        # Local disk only (the default HF cache under $HOME): a network volume
        # makes a 70 GB load take many minutes.
        path = snapshot_download(
            repo_id=repo,
            revision=revision,
            allow_patterns=["*.json", "*.py", "*.safetensors", "*.jinja", "*.txt"],
        )
        self.timings["download_s"] = round(time.time() - t0, 1)
        _log(f"snapshot ready in {self.timings['download_s']} s: {path}")

        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
            # Whole model on GPU 0: with "auto", accelerate may quietly move
            # layers to CPU when the card is tight, which makes it very slow.
            "device_map": {"": 0},
            "attn_implementation": "sdpa",
        }
        cap = float(_env("GPU_CAP_GIB", "0") or 0)
        if cap > 0:
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / total), 0)
            _log(f"GPU memory capped at {cap} GiB of {_gib(total)} GiB")
        self.mode = "bf16"
        if _env("LOAD_IN_8BIT") == "1":
            from transformers import BitsAndBytesConfig

            # The memory block (memt), the router and the head stay in bf16.
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["lm_head", "memt", "gate", "shared_expert_gate"],
            )
            self.mode = "int8 (bitsandbytes)"

        t1 = time.time()
        # Load by repo id + revision, not by the snapshot path: with a local
        # path, Transformers resolves modeling_memt.py through its cache
        # symlink and then looks for configuration_memt.py next to the blob
        # (FileNotFoundError). By repo id it reads the same cached files.
        self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, **kwargs)
        self.model.eval()
        self.timings["load_s"] = round(time.time() - t1, 1)
        self.devices = sorted({str(p.device) for p in self.model.parameters()})
        _log(f"model loaded in {self.timings['load_s']} s ({self.mode}) on {self.devices}; "
             f"allocated {_gib(torch.cuda.memory_allocated())} GiB")
        self.sessions: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- helpers

    def _vram(self) -> Dict[str, Any]:
        torch = self.torch
        return {
            "allocated_gib": _gib(torch.cuda.memory_allocated()),
            "peak_allocated_gib": _gib(torch.cuda.max_memory_allocated()),
        }

    def _gen(self) -> Dict[str, Any]:
        return {"max_new_tokens": self.max_new_tokens, "do_sample": False}

    def _session(self, sid: str, create: bool = True):
        s = self.sessions.get(sid)
        if s is None and create:
            s = self.model.create_session(self.tokenizer)
            self.sessions[sid] = s
        return s

    def _state_path(self, name: str) -> Path:
        if not _NAME.match(name or ""):
            raise InvalidInput("state name: 1-64 characters, letters, digits, . _ -")
        return self.state_dir / f"{name}.safetensors"

    def _chat(self, sid: str, text: str, remember: bool) -> Dict[str, Any]:
        if not text.strip():
            raise InvalidInput("empty text")
        s = self._session(sid)
        t0 = time.time()
        r = s.chat(text, enable_memory=remember, enable_thinking=False, return_dict=True, **self._gen())
        took = time.time() - t0
        out_tok = r.get("output_tokens") or 0
        return {
            "answer": r.get("answer", ""),
            "op": "remember" if remember else "ask",
            "session": sid,
            "input_tokens": r.get("input_tokens"),
            "output_tokens": out_tok,
            "seconds": round(took, 2),
            "tokens_per_s": round(out_tok / took, 1) if took > 0 else None,
            "memory_updated": r.get("memory_updated"),
            "committed_turns": r.get("committed_turns"),
            "plaintext_messages_kept": len(s.messages),
            "finish_reason": r.get("finish_reason"),
        }

    # ---------------------------------------------------------------- process

    def _op_from_job(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        extra = inp.get("extra_body") if isinstance(inp.get("extra_body"), dict) else {}
        op = extra.get("memt") or inp.get("memt")
        if isinstance(op, dict):
            return op
        messages: List[Dict[str, Any]] = inp.get("messages") or []
        text = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                text = c if isinstance(c, str) else " ".join(
                    p.get("text", "") for p in (c or []) if isinstance(p, dict))
                break
        if not text:
            text = inp.get("prompt") or ""
        if not isinstance(text, str) or not text.strip():
            raise InvalidInput("expected extra_body.memt {op: ...} or a user message")
        t = text.strip()
        low = t.lower()
        for prefix, opname, key in (("remember:", "remember", "text"), ("save:", "save", "name"),
                                    ("load:", "load", "name")):
            if low.startswith(prefix):
                return {"op": opname, "session": PANEL_SESSION, key: t[len(prefix):].strip()}
        if low in ("forget", "close", "forget everything"):
            return {"op": "close", "session": PANEL_SESSION}
        if low == "status":
            return {"op": "status"}
        return {"op": "ask", "session": PANEL_SESSION, "text": t}

    def process(self, job: Job) -> Result:
        op = self._op_from_job(job.input or {})
        kind = op.get("op")
        sid = str(op.get("session") or PANEL_SESSION)
        with self._lock:
            self.torch.cuda.reset_peak_memory_stats()
            if kind in ("remember", "ask"):
                out = self._chat(sid, str(op.get("text") or ""), kind == "remember")
            elif kind == "save":
                s = self._session(sid, create=False)
                if s is None:
                    raise InvalidInput(f"no open session {sid!r}")
                path = self._state_path(str(op.get("name") or sid))
                t0 = time.time()
                s.save_state(str(path), overwrite=True)
                took = time.time() - t0
                data = path.read_bytes()
                out = {
                    "answer": f"Saved the memory of session '{sid}' to {path.name}: "
                              f"{len(data):,} bytes in {took:.2f} s.",
                    "op": "save", "session": sid, "file": path.name, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(), "seconds": round(took, 2),
                }
            elif kind == "close":
                s = self.sessions.pop(sid, None)
                if s is not None:
                    s.close()
                out = {"answer": f"Session '{sid}' closed: its memory and plaintext history are gone "
                                 f"from this process.", "op": "close", "session": sid,
                       "existed": s is not None}
            elif kind == "load":
                path = self._state_path(str(op.get("name") or ""))
                if not path.is_file():
                    raise InvalidInput(f"no saved state {path.name}")
                old = self.sessions.pop(sid, None)
                if old is not None:
                    old.close()
                t0 = time.time()
                s = self.model.create_session(self.tokenizer, state=str(path))
                took = time.time() - t0
                self.sessions[sid] = s
                out = {"answer": f"Loaded {path.name} into session '{sid}' in {took:.2f} s. "
                                 f"Plaintext messages in the session: {len(s.messages)}.",
                       "op": "load", "session": sid, "file": path.name,
                       "bytes": path.stat().st_size, "seconds": round(took, 2),
                       "plaintext_messages": len(s.messages)}
            elif kind == "status":
                states = {p.name: p.stat().st_size for p in sorted(self.state_dir.glob("*.safetensors"))}
                out = {"answer": f"{len(self.sessions)} open session(s), {len(states)} saved state(s).",
                       "op": "status", "sessions": {k: len(v.messages) for k, v in self.sessions.items()},
                       "states": states, "mode": self.mode, "devices": self.devices,
                       "timings": self.timings}
            else:
                raise InvalidInput(f"unknown op {kind!r}")
            out.update(self._vram())
        return Result(output=out)

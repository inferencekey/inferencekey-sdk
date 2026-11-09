"""SenseNova-Loop-9B-Preview (SenseTime) as an InferenceKey custom backend.

SenseNova-Loop-9B-Preview is a looped multimodal 9B model: it stores 32 decoder
layers but runs 56 per token, because layers 4 to 27 run twice with the same
weights, and inside that looped block two residual streams are mixed by learned
mHC (manifold-constrained hyper-connection) routers. Text and images in, text
out, Apache-2.0.

No released serving engine runs it yet (the authors point to open SGLang and
vLLM pull requests and to LightLLM), so this backend drives the checkpoint's
own Transformers code (``trust_remote_code``), the path the authors checked:

``setup()`` (once per process)
    1. downloads the checkpoint (about 38 GB, mostly FP32 shards) at a pinned
       revision, skipping the MTP head (Transformers does not load it),
    2. loads it in bf16 (about 19 GB on the GPU; the mHC routers stay FP32),
    3. runs a short warm-up.

``chat(request)`` (OpenAI ``/v1/chat/completions``)
    takes OpenAI messages (``text`` and ``image_url`` parts with data URLs),
    ``tools``, ``max_tokens``, ``temperature``, ``top_p`` and returns an OpenAI
    ``chat.completion`` with ``reasoning_content`` (the model's thinking),
    parsed ``tool_calls`` and usage, plus a ``timings`` object.

Options in the request body or in ``extra_body``:

* ``enable_thinking`` (default true, the model card's default): thinking
  mode. False makes the template close the think block at once.
* ``loop_passes`` (default 2, the trained setting): how many times the looped
  block runs. 1 runs each stored layer once (32 executions instead of 56), an
  ablation that shows what the second pass buys.

Settings are class attributes; override them with ``LOOP_*`` environment
variables or the workload config (keys in lower case). ``LOOP_VRAM_LIMIT_GIB``
caps the memory PyTorch may use on the GPU (to check a smaller card).
"""

from __future__ import annotations

import base64
import copy
import io
import json
import os
import re
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

try:  # newer SDK runtimes serve chat() as an OpenAI route; older ones route chat to process()
    from inferencekey.backend import chat_completion
except ImportError:  # pragma: no cover
    chat_completion = None

CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _log(msg: str) -> None:
    print(f"[sensenova-loop-backend] {msg}", file=sys.stderr, flush=True)


def parse_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """The chat template's tool format: `<tool_call><function=name><parameter=k>v</parameter>...`."""
    calls = []
    for name, body in CALL_RE.findall(text or ""):
        arguments: Dict[str, Any] = {}
        for key, value in PARAM_RE.findall(body):
            try:
                arguments[key.strip()] = json.loads(value)
            except json.JSONDecodeError:
                arguments[key.strip()] = value
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": name.strip(), "arguments": json.dumps(arguments)}})
    content = CALL_RE.sub("", text or "").strip() if calls else (text or "").strip()
    return content, calls


def split_thinking(text: str) -> Tuple[str, str]:
    """`...thinking...</think>answer` -> (thinking, answer). The prompt already opened <think>."""
    if "</think>" in text:
        thinking, _, answer = text.partition("</think>")
        return thinking.replace("<think>", "").strip(), answer.strip()
    return "", text.strip()


class _FirstTick:
    """A generate() streamer that records when the first new token arrives (skip=1 drops the prompt)."""

    def __init__(self, skip: int = 1):
        self.skip = skip
        self.count = 0
        self.first: Optional[float] = None

    def put(self, value) -> None:  # noqa: ANN001
        self.count += 1
        if self.count > self.skip and self.first is None:
            self.first = time.time()

    def end(self) -> None:
        pass


class SenseNovaLoopBackend(CustomBackend):
    name = "sensenova-loop-9b"
    version = "0.1.1"
    task_type = "text2text"
    description = "SenseNova-Loop-9B-Preview (SenseTime): looped multimodal 9B, text and images in, text out"

    # Pinned upstream: the Hugging Face revision this example was tested with.
    hf_repo = "sensenova/SenseNova-Loop-9B-Preview"
    hf_revision = "9937e91c7301b7dcbb5bf17e3e4e95c499a11e99"
    # 0 = no cap. Otherwise the GiB PyTorch may allocate on the GPU.
    vram_limit_gib = 0.0
    # Upper bound for max_tokens in one request.
    max_new_tokens_cap = 8192
    # Longest image side in pixels before the processor sees it (bounds vision tokens).
    max_image_side = 1536

    # ------------------------------------------------------------------ setup

    def _setting(self, ctx: BackendContext, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"LOOP_{key.upper()}")
        if raw is None:
            raw = (ctx.config or {}).get(key)
        if raw is None:
            return default
        return type(default)(raw)

    def setup(self, ctx: BackendContext) -> None:
        for key in ("hf_repo", "hf_revision", "vram_limit_gib", "max_new_tokens_cap", "max_image_side"):
            setattr(self, key, self._setting(ctx, key))
        self.timings: Dict[str, float] = {}
        t_setup = time.time()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if torch.cuda.is_available() and self.vram_limit_gib > 0:
            total = torch.cuda.get_device_properties(0).total_memory
            frac = min(1.0, self.vram_limit_gib * 2**30 / total)
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            _log(f"VRAM capped at {self.vram_limit_gib} GiB (fraction {frac:.3f} of {total / 2**30:.1f} GiB)")

        t0 = time.time()
        model_dir = snapshot_download(
            self.hf_repo, revision=self.hf_revision,
            ignore_patterns=["assets/*", "examples/*", "*.md", "mtp.safetensors"],
        )
        self.timings["download_s"] = round(time.time() - t0, 1)
        _log(f"weights in {self.timings['download_s']} s at {model_dir}")

        t0 = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=dtype, device_map={"": self.device}, trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.timings["load_s"] = round(time.time() - t0, 1)
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        self.timings["vram_loaded_mib"] = round(torch.cuda.memory_allocated() / 2**20) if torch.cuda.is_available() else 0
        self._lock = threading.Lock()

        # The loop plan lives on the text model; keep the trained one and build others on demand.
        self.text_model = self.model.model.language_model
        self._plans: Dict[int, Tuple[tuple, Any]] = {}
        loop = self.model.config.loop_settings
        self.loop_start, self.loop_end = int(loop["start"]), int(loop["end"])
        self.trained_passes = int(loop["eval_count"])
        self._plans[self.trained_passes] = (self.text_model._loop_plan, self.text_model._loop_cache_config)

        t0 = time.time()
        self._generate([{"role": "user", "content": "Say hi."}], {"max_tokens": 8, "enable_thinking": False})
        self.timings["warmup_s"] = round(time.time() - t0, 1)
        self.timings["setup_s"] = round(time.time() - t_setup, 1)
        _log(f"SenseNova-Loop-9B ready on {self.device_name}: {self.timings}")

    # ---------------------------------------------------------------- loop

    def _use_passes(self, passes: int) -> None:
        """Point the text model at the execution plan with `passes` runs of layers [start, end)."""
        if passes not in self._plans:
            n = self.text_model.config.num_hidden_layers
            plan = tuple(range(self.loop_start)) + tuple(range(self.loop_start, self.loop_end)) * passes \
                + tuple(range(self.loop_end, n))
            cfg = copy.deepcopy(self._plans[self.trained_passes][1])
            cfg.num_hidden_layers = len(plan)
            cfg.layer_types = [self.text_model.config.layer_types[i] for i in plan]
            self._plans[passes] = (plan, cfg)
        self.text_model._loop_plan, self.text_model._loop_cache_config = self._plans[passes]

    # ---------------------------------------------------------------- inputs

    @staticmethod
    def _merged(inp: Dict[str, Any]) -> Dict[str, Any]:
        """Top-level fields plus ``extra_body`` (the platform carries non-OpenAI fields there)."""
        merged = dict(inp)
        extra = inp.get("extra_body")
        if isinstance(extra, dict):
            merged.update(extra)
        return merged

    def _image(self, url: str):
        from PIL import Image

        if url.startswith(("http://", "https://")):
            raise InvalidInput("send images inline as base64 data: URLs; this backend does not fetch URLs")
        data = url.partition(",")[2] if url.startswith("data:") else url
        try:
            img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        except Exception as e:  # noqa: BLE001
            raise InvalidInput(f"image is not a valid base64 picture: {e}") from e
        side = max(img.size)
        if side > self.max_image_side:
            scale = self.max_image_side / side
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)
        return img

    def _conversation(self, messages: Any) -> Tuple[List[Dict[str, Any]], int]:
        """OpenAI chat messages -> the processor's conversation (images decoded to PIL)."""
        if not isinstance(messages, list) or not messages:
            raise InvalidInput("expected 'messages': an OpenAI-style chat")
        conv: List[Dict[str, Any]] = []
        images = 0
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") not in ("system", "user", "assistant", "tool"):
                raise InvalidInput("each message needs a role: system, user, assistant or tool")
            content = msg.get("content")
            out: Dict[str, Any] = {"role": msg["role"]}
            if msg.get("tool_calls"):
                calls = []
                for c in msg["tool_calls"]:
                    fn = dict(c.get("function", {}))
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"] or "{}")
                        except json.JSONDecodeError:
                            fn["arguments"] = {}
                    calls.append({"type": "function", "function": fn})
                out["tool_calls"] = calls
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    kind = part.get("type")
                    if kind == "text":
                        parts.append({"type": "text", "text": str(part.get("text", ""))})
                    elif kind in ("image_url", "image"):
                        url = (part.get("image_url") or {}).get("url") if kind == "image_url" else part.get("image")
                        parts.append({"type": "image", "image": self._image(str(url or ""))})
                        images += 1
                    else:
                        raise InvalidInput(f"unsupported content part {kind!r} (text and image_url only)")
                out["content"] = parts
            else:
                out["content"] = "" if content is None else str(content)
            conv.append(out)
        return conv, images

    # -------------------------------------------------------------- generate

    def _generate(self, messages: Any, f: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        conv, images = self._conversation(messages)
        max_new = int(f.get("max_tokens") or f.get("max_completion_tokens") or f.get("max_new_tokens") or 2048)
        max_new = max(1, min(max_new, int(self.max_new_tokens_cap)))
        thinking = f.get("enable_thinking")
        thinking = True if thinking is None else str(thinking).lower() in ("1", "true", "yes", "on")
        passes = int(f.get("loop_passes") or self.trained_passes)
        if passes < 1 or passes > 4:
            raise InvalidInput("loop_passes must be between 1 and 4")
        tools = f.get("tools") or None

        t0 = time.time()
        inputs = self.processor.apply_chat_template(
            conv, tools=tools, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=thinking,
        ).to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        gen: Dict[str, Any] = {"max_new_tokens": max_new, "use_cache": True}
        temp = f.get("temperature")
        if temp is not None and float(temp) > 0:
            gen.update(do_sample=True, temperature=float(temp), top_p=float(f.get("top_p") or 0.95))
        else:
            gen.update(do_sample=False, temperature=None, top_p=None)
        tick = _FirstTick()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t_gen = time.time()
        with torch.inference_mode():
            self._use_passes(passes)
            try:
                out = self.model.generate(**inputs, streamer=tick, **gen)
            finally:
                self._use_passes(self.trained_passes)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        gen_s = time.time() - t_gen
        ids = out[0, prompt_tokens:]
        completion_tokens = int(ids.shape[0])
        raw = self.processor.decode(ids, skip_special_tokens=True)
        hit_limit = completion_tokens >= max_new
        reasoning, answer = split_thinking(raw) if thinking else ("", raw.strip())
        if thinking and "</think>" not in raw:
            reasoning, answer = raw.strip(), ""  # ran out of tokens while thinking
        content, calls = parse_tool_calls(answer) if tools else (answer, [])
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
        decode_s = gen_s - ((tick.first - t_gen) if tick.first else 0)
        timings = {
            "prep_s": round(t_gen - t0, 3),
            "generate_s": round(gen_s, 3),
            "first_token_s": round(tick.first - t_gen, 3) if tick.first else None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_s": round((completion_tokens - 1) / decode_s, 2) if decode_s > 0 and completion_tokens > 1 else None,
            "peak_vram_mib": round(peak),
            "loop_passes": passes,
            "layer_executions": len(self._plans[passes][0]),
            "images": images,
        }
        _log(f"request: {timings}")
        return {"content": content, "reasoning": reasoning, "tool_calls": calls, "timings": timings,
                "finish_reason": "length" if hit_limit else ("tool_calls" if calls else "stop")}

    # ---------------------------------------------------------------- serving

    def chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if chat_completion is None:
            raise InvalidInput("this SDK runtime has no chat_completion helper: send the request as a job")
        f = self._merged(request)
        with self._lock:
            try:
                out = self._generate(f.get("messages"), f)
            except (ValueError, TypeError) as e:
                raise InvalidInput(str(e)[:500]) from e
        completion = chat_completion(
            out["content"],
            model=self.hf_repo,
            tool_calls=out["tool_calls"] or None,
            finish_reason=out["finish_reason"],
            prompt_tokens=out["timings"]["prompt_tokens"],
            completion_tokens=out["timings"]["completion_tokens"],
            reasoning_content=out["reasoning"] or None,
        )
        completion["timings"] = dict(out["timings"], device=self.device_name)
        return completion

    def process(self, job: Job) -> Result:
        """The platform's chat route sends the OpenAI body here; the answer comes back as JSON."""
        f = self._merged(job.input)
        with self._lock:
            try:
                out = self._generate(f.get("messages"), f)
            except (ValueError, TypeError) as e:
                raise InvalidInput(str(e)[:500]) from e
        out["device"] = self.device_name
        return Result(output=out)

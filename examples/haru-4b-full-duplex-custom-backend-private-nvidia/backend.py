"""Haru-4B (higurashi), full-duplex spoken dialogue, as an InferenceKey custom backend.

Haru-4B is an Apache-2.0 full-duplex speech model on a Qwen3.5-4B backbone: it
listens and speaks at the same time, the way people do on a phone call. Every
133 ms (7.5 Hz) it takes one frame of the caller's audio and produces one frame
of its own audio and text. A text "character card" sets who it is and a short
reference clip sets its voice. Nine main languages: English, Spanish,
Portuguese, German, French, Chinese, Japanese, Russian and Korean. It is a
pretrained base model: it chats naturally but it is not an assistant and not a
reliable source of facts (the authors say so).

This backend wraps the authors' PyTorch package (``haru-speech`` on PyPI,
github.com/higurashi-ai/Haru) and its ``DuplexSession``:

``setup()`` (once per process)
    downloads the single bf16 checkpoint (10.5 GB) at a pinned Hugging Face
    revision to the local disk, loads it on the GPU, optionally caps the memory
    PyTorch may use (to emulate a smaller card), and runs a short warm-up
    session (this also compiles the Triton codec kernels).

``process(job)`` (per request)
    plays one recorded caller clip into a fresh ``DuplexSession`` frame by
    frame, exactly as a live line would feed it, then keeps the "microphone"
    open with ``tail_seconds`` of silence so the model can answer. The model
    decides by itself when to speak, when to backchannel and when to stay
    quiet. Returns the model's audio, a two-channel conversation WAV (caller
    left, model right, one clock), the text it wrote with its timing, how long
    each 133 ms frame took to compute, and when the model started to talk
    relative to the caller's last word.

The platform forwards HTTP requests, not live audio streams, so a request is
one recorded caller turn (or several turns in one clip), not a live microphone.

Request shapes (options at the top level or in ``extra_body``):

* OpenAI chat: the last user message carries an ``input_audio`` part
  ``{"data": <base64 WAV>, "format": "wav"}`` (the caller's speech).
* Plain fields: ``audio_b64``.

Options: ``voice_clip_b64`` (3 s to 3 min of ONE speaker, the voice the model
speaks with; only use a voice you have the right to use), ``voice`` (a name to
keep that clip under, so later requests can send just the name),
``language`` (card language: en, es, pt, de, fr, zh, ja, ru, ko; default en),
``character`` (card fields: persona, name, gender, backchannel, other_name,
relationship, situation) or ``system_prompt`` (a ready card), ``tail_seconds``
(silence after the clip, default 8), ``speak_first``, ``seed``,
``temperature``, ``top_k``, ``guidance_scale`` (1 to 2), ``eop_bias``
(> 0 makes it start talking sooner), ``pace`` ("offline", as fast as the GPU
goes, default; or "realtime", one frame every 133 ms like a live call),
``audio_format`` (``wav`` default, or ``flac``). A result travels in one
platform message of at most 8 MB: past about 40 s of call the backend switches
to FLAC on its own, and past about 70 s it leaves out the two-channel copy.

Settings are class attributes; override them with ``HARU_*`` environment
variables or the workload config (keys in lower case). ``gpu_memory_gb`` caps
the memory PyTorch may allocate on the card (0 = no cap) to check that the
model fits a smaller GPU; ``codec_backend`` is ``auto`` (the authors' Triton
codec when usable), ``triton`` or ``torch``.
"""

from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

RATE = 24_000
FRAME = 3200  # samples per 133 ms frame at 24 kHz
FRAME_S = FRAME / RATE
LANGUAGES = ("en", "es", "pt", "de", "fr", "zh", "ja", "ru", "ko")


def _log(msg: str) -> None:
    print(f"[haru-backend] {msg}", file=sys.stderr, flush=True)


def _read_audio(raw: bytes) -> Tuple[np.ndarray, int]:
    """Mono float32 samples and their rate from a WAV / FLAC / OGG file."""
    try:
        with wave.open(io.BytesIO(raw)) as w:
            rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            data = w.readframes(w.getnframes())
        if width != 2:
            raise wave.Error("not 16-bit PCM")
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        return x, rate
    except (wave.Error, EOFError):
        import soundfile as sf  # IEEE float WAV, FLAC, OGG

        x, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        return x.mean(axis=1), int(rate)


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x.astype(np.float32)
    from scipy.signal import resample_poly

    g = np.gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)


def _wav_bytes(x: np.ndarray, rate: int = RATE) -> bytes:
    """16-bit PCM WAV; ``x`` is [n] (mono) or [n, 2] (stereo)."""
    x = np.asarray(x, dtype=np.float32)
    channels = 1 if x.ndim == 1 else x.shape[1]
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _encode(x: np.ndarray, fmt: str) -> bytes:
    if fmt == "wav":
        return _wav_bytes(x)
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0), RATE, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def _frame_db(x: np.ndarray) -> np.ndarray:
    """Level of each 133 ms frame in dBFS."""
    n = len(x) // FRAME
    if n == 0:
        return np.zeros(0)
    rms = np.sqrt(np.mean(x[: n * FRAME].reshape(n, FRAME) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def _voiced_end(x: np.ndarray, threshold_db: float = -40.0) -> Optional[float]:
    """Time (s) of the end of the last 20 ms window above ``threshold_db``."""
    win = RATE // 50
    n = len(x) // win
    if n == 0:
        return None
    rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
    voiced = np.nonzero(20 * np.log10(rms) > threshold_db)[0]
    return None if voiced.size == 0 else float((voiced[-1] + 1) * win / RATE)


def _gpu_used_mib() -> Optional[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


class HaruDuplexBackend(CustomBackend):
    name = "haru-4b-full-duplex"
    version = "0.1.1"
    task_type = "text2audio"
    description = "Haru-4B (Apache-2.0): full-duplex spoken dialogue in 9 languages, voice from a short clip"

    # Pinned upstream: the Hugging Face revision this example was tested with.
    hf_repo = "higurashi-ai/Haru-4B"
    hf_revision = "8d6e83bcd3701a76fc27a764965665da1df7da78"
    codec_backend = "auto"
    # 0 = no cap. A number caps the GiB PyTorch may allocate on the card (emulates a smaller GPU).
    gpu_memory_gb = 0.0
    max_input_seconds = 180.0
    max_tail_seconds = 30.0
    # Results travel in one platform message (8 MB); keep a margin for the JSON around the audio.
    max_result_bytes = 7_000_000

    # ------------------------------------------------------------------ setup

    def _setting(self, ctx: BackendContext, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"HARU_{key.upper()}")
        if raw is None:
            raw = (ctx.config or {}).get(key)
        if raw is None or raw == "":
            return default
        return type(default)(raw)

    def setup(self, ctx: BackendContext) -> None:
        for key in ("hf_repo", "hf_revision", "codec_backend", "gpu_memory_gb", "max_input_seconds"):
            setattr(self, key, self._setting(ctx, key))
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        import torch
        from huggingface_hub import snapshot_download

        if not torch.cuda.is_available():
            raise RuntimeError("Haru-4B needs a CUDA GPU (about 12 GB of memory)")
        self.timings: Dict[str, Any] = {}
        t_setup = time.time()
        self.peak_mib = 0
        threading.Thread(target=self._sample_gpu, daemon=True).start()

        # Keep the weights on the local disk: a network volume loads many times slower.
        cache = Path(os.environ.get("HARU_CACHE_DIR", Path.home() / ".cache" / "haru-backend"))
        cache.mkdir(parents=True, exist_ok=True)
        t = time.time()
        local = cache / f"weights-{self.hf_revision[:12]}"
        snapshot_download(self.hf_repo, revision=self.hf_revision, local_dir=str(local), max_workers=16,
                          allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"])
        self.timings["download_s"] = round(time.time() - t, 1)

        total = torch.cuda.get_device_properties(0).total_memory
        if self.gpu_memory_gb > 0:
            fraction = min(1.0, self.gpu_memory_gb * 2**30 / total)
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            _log(f"memory cap: {self.gpu_memory_gb} GiB of {total / 2**30:.1f} GiB (fraction {fraction:.3f})")

        from haru import __version__ as haru_version
        from haru.models import load_model
        from haru.prompt import card_templates

        self.haru_version = haru_version
        t = time.time()
        self.model = load_model(str(local), device="cuda")
        torch.cuda.synchronize()
        self.timings["load_s"] = round(time.time() - t, 1)
        self.timings["weights_gib_on_gpu"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        self.templates = card_templates()["templates"]
        self.voices: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()  # one conversation at a time on the GPU

        # Warm-up: a short session (compiles the Triton codec kernels on first use).
        t = time.time()
        self.codec_used = self._warm_up()
        self.timings["warmup_s"] = round(time.time() - t, 1)
        self.timings["torch_peak_reserved_mib"] = int(torch.cuda.max_memory_reserved() / 2**20)
        self.timings["gpu_used_mib_after_setup"] = _gpu_used_mib()
        self.timings["setup_s"] = round(time.time() - t_setup, 1)
        _log(f"Haru-4B ready (haru-speech {haru_version}, codec {self.codec_used}, "
             f"GPU {torch.cuda.get_device_name(0)}): {self.timings}")

    def _warm_up(self) -> str:
        from haru.session import DuplexConfig, DuplexSession

        backends = [self.codec_backend] if self.codec_backend != "auto" else ["triton", "torch"]
        last: Optional[Exception] = None
        for backend in backends:
            try:
                cfg = DuplexConfig(codec_backend=backend, seed=0, max_frames=40)
                ref = (0.05 * np.sin(np.linspace(0, 2400 * np.pi, 3 * RATE))).astype(np.float32)
                s = DuplexSession(self.model, "Warm-up.", ref, cfg)
                s.start()
                silence = np.zeros(FRAME, dtype=np.float32)
                for _ in range(30):
                    s.step(silence)
                return backend
            except Exception as e:  # noqa: BLE001  (Triton missing or unusable: fall back)
                last = e
                _log(f"codec backend {backend} failed in warm-up: {e!r}")
        raise RuntimeError(f"warm-up failed: {last!r}")

    def _sample_gpu(self) -> None:
        while True:
            used = _gpu_used_mib()
            if used is not None and used > self.peak_mib:
                self.peak_mib = used
            time.sleep(0.5)

    # ---------------------------------------------------------------- inputs

    @staticmethod
    def _audio_from_messages(messages: Any) -> Optional[str]:
        if not isinstance(messages, list):
            return None
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user" or isinstance(msg.get("content"), str):
                continue
            for part in msg.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "input_audio":
                    return (part.get("input_audio") or {}).get("data")
        return None

    @staticmethod
    def _b64_audio(value: str) -> np.ndarray:
        if value.startswith("data:"):
            value = value.split(",", 1)[-1]
        x, rate = _read_audio(base64.b64decode(value))
        return _resample(x, rate, RATE)

    def _parse(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        f: Dict[str, Any] = dict(inp)
        if isinstance(f.get("extra_body"), dict):
            f = {**f, **f["extra_body"]}
        audio = f.get("audio_b64") or self._audio_from_messages(f.get("messages"))
        if not isinstance(audio, str) or not audio:
            raise InvalidInput("expected the caller's speech: a user message with an input_audio part, or audio_b64")
        f["speech"] = self._b64_audio(audio)
        if f["speech"].size / RATE > self.max_input_seconds:
            raise InvalidInput(f"audio is {f['speech'].size / RATE:.1f} s; the limit is {self.max_input_seconds} s")
        lang = str(f.get("language") or "en").lower()[:2]
        if lang not in LANGUAGES:
            raise InvalidInput(f"language must be one of {', '.join(LANGUAGES)}")
        f["language"] = lang

        name = str(f.get("voice") or "").strip().lower()
        if name and not re.fullmatch(r"[a-z0-9_.-]{1,40}", name):
            raise InvalidInput("voice names are lower case: a-z 0-9 _ - .")
        reference = None
        if f.get("voice_clip_b64"):
            reference = self._b64_audio(str(f["voice_clip_b64"]))
            if not 3.0 <= reference.size / RATE <= 180.0:
                raise InvalidInput("the voice clip should be 3 s to 3 min of one speaker")
            from haru.audio import prepare_reference

            reference = np.asarray(prepare_reference(reference), dtype=np.float32)
            if name:
                self.voices[name] = reference
        elif name:
            if name not in self.voices:
                raise InvalidInput(f"unknown voice {name!r}: send voice_clip_b64 once with this name")
            reference = self.voices[name]
        f["reference"] = reference

        if f.get("system_prompt"):
            f["card"] = str(f["system_prompt"])
        else:
            from haru.prompt import render_card

            character = f.get("character") if isinstance(f.get("character"), dict) else {}
            f["card"] = render_card(character, self.templates[lang]) if character else ""
        tail = float(f.get("tail_seconds", 8.0))
        f["tail_seconds"] = max(0.0, min(tail, self.max_tail_seconds))
        f["pace"] = str(f.get("pace") or "offline").lower()
        if f["pace"] not in ("offline", "realtime"):
            raise InvalidInput("pace must be offline or realtime")
        return f

    # -------------------------------------------------------------- dialogue

    def _converse(self, f: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from haru.session import DuplexConfig, DuplexSession

        speech = f["speech"]
        caller = np.concatenate([speech, np.zeros(int(f["tail_seconds"] * RATE), dtype=np.float32)])
        count = max(1, -(-len(caller) // FRAME))
        caller = np.pad(caller, (0, count * FRAME - len(caller)))
        cfg = DuplexConfig(
            guidance_scale=float(f.get("guidance_scale", 1.5)),
            temperature=float(f.get("temperature", 0.7)),
            top_k=int(f.get("top_k", 25)),
            eop_bias=float(f.get("eop_bias", 0.0)),
            speak_first=bool(f.get("speak_first", False)),
            seed=None if f.get("seed") is None else int(f["seed"]),
            max_frames=count + 1,
            codec_backend=self.codec_used,
        )
        torch.cuda.reset_peak_memory_stats()
        session = DuplexSession(self.model, f["card"], f["reference"], cfg)
        t0 = time.perf_counter()
        frames = [session.start()]
        prompt_s = time.perf_counter() - t0
        late = 0
        t_line = time.perf_counter()
        # Model frame t plays while the caller speaks frame t; it was computed from caller frames < t.
        for k in range(count - 1):
            frames.append(session.step(caller[k * FRAME:(k + 1) * FRAME]))
            if f["pace"] == "realtime":
                due = t_line + (k + 1) * FRAME_S
                now = time.perf_counter()
                if now > due:
                    late += 1
                else:
                    time.sleep(due - now)
        compute = np.array([fr.seconds for fr in frames[1:]]) * 1000 if len(frames) > 1 else np.zeros(1)
        reply = np.concatenate([fr.audio for fr in frames]).astype(np.float32)
        words = [{"t": round(fr.index * FRAME_S, 3), "text": fr.text} for fr in frames if fr.text]
        text = "".join(fr.text for fr in frames).strip()

        voiced_end = _voiced_end(speech)
        model_db = _frame_db(reply)
        talking = model_db > -45.0
        first_after = None
        if voiced_end is not None:
            idx = [i for i in np.nonzero(talking)[0] if i * FRAME_S >= voiced_end - 1e-6]
            first_after = round(float(idx[0]) * FRAME_S - voiced_end, 3) if idx else None
        caller_db = _frame_db(caller)
        overlap = float(np.sum(talking & (caller_db[: len(talking)] > -40.0)) * FRAME_S)
        first_text = None
        if voiced_end is not None:
            after = [w["t"] for w in words if w["t"] >= voiced_end]
            first_text = round(after[0] - voiced_end, 3) if after else None
        return {
            "reply": reply,
            "conversation": np.stack([caller, reply[: len(caller)]], axis=1),
            "text": text,
            "words": words,
            "timings": {
                "prompt_s": round(prompt_s, 3),
                "frames": len(frames),
                "frame_ms_median": round(float(np.median(compute)), 2),
                "frame_ms_p95": round(float(np.percentile(compute, 95)), 2),
                "frame_ms_max": round(float(np.max(compute)), 2),
                "realtime_factor": round(FRAME_S * 1000 / float(np.median(compute)), 2),
                "frames_over_133ms": int(np.sum(compute > FRAME_S * 1000)),
                "late_frames_realtime": late if f["pace"] == "realtime" else None,
                "caller_voiced_end_s": None if voiced_end is None else round(voiced_end, 3),
                "model_starts_after_caller_s": first_after,
                "first_text_after_caller_s": first_text,
                "model_talks_while_caller_talks_s": round(overlap, 2),
                "model_voiced_s": round(float(np.sum(talking) * FRAME_S), 2),
                "torch_peak_reserved_mib": int(torch.cuda.max_memory_reserved() / 2**20),
            },
        }

    def process(self, job: Job) -> Result:
        t0 = time.time()
        f = self._parse(job.input)
        with self.lock:
            out = self._converse(f)
        # The platform carries a result in one message of at most 8 MB; a bigger one is dropped and the
        # caller sees a timeout. Long calls switch to FLAC, then leave out the two-channel copy.
        fmt = str(f.get("audio_format") or "wav").lower()
        if fmt not in ("wav", "flac"):
            raise InvalidInput("audio_format must be wav or flac")
        reply, conversation = _encode(out["reply"], fmt), _encode(out["conversation"], fmt)
        if fmt == "wav" and 4 * (len(reply) + len(conversation)) / 3 > self.max_result_bytes:
            fmt = "flac"
            reply, conversation = _encode(out["reply"], fmt), _encode(out["conversation"], fmt)
        omitted = None
        if 4 * (len(reply) + len(conversation)) / 3 > self.max_result_bytes:
            conversation, omitted = b"", "the result would pass the 8 MB message limit: send a shorter clip"
        if 4 * len(reply) / 3 > self.max_result_bytes:
            raise InvalidInput("the reply would pass the 8 MB message limit: send a shorter clip or less tail_seconds")
        result = {
            "audio_b64": base64.b64encode(reply).decode(),
            "conversation_b64": base64.b64encode(conversation).decode() if conversation else None,
            "conversation_omitted": omitted,
            "format": fmt,
            "sample_rate": RATE,
            "seconds": round(len(out["reply"]) / RATE, 2),
            "text": out["text"],
            "words": out["words"],
            "language": f["language"],
            "voice": f.get("voice") or ("clip" if f["reference"] is not None else "none"),
            "card": f["card"],
            "pace": f["pace"],
            "timings": {**out["timings"], "process_s": round(time.time() - t0, 3)},
            "setup_timings": self.timings,
            "gpu_used_mib_peak": self.peak_mib,
            "gpu_memory_cap_gb": self.gpu_memory_gb or None,
            "codec_backend": self.codec_used,
            "model": f"{self.hf_repo}@{self.hf_revision[:8]}",
            "runtime": f"haru-speech {self.haru_version}",
        }
        return Result(output=result)

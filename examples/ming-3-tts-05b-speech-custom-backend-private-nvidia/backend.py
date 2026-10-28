"""Ming-3.0-tts-0.5B (inclusionAI, Ant Group) as an InferenceKey custom backend.

Ming-3.0-tts-0.5B is a dense 0.5B text-to-speech model (Qwen2.5-0.5B backbone,
a DiT patch head and an audio VAE, 44.1 kHz output, MIT). One checkpoint does
zero-shot voice cloning, voice design from a text description, free-form
delivery instructions, built-in voices, emotion / speed / pitch / volume
controls, 14 inline paralinguistic tags ("[Laughter]", "[Sigh]", ...) and seven
languages. No built-in backend serves it, so this backend drives the authors'
own Python API (``ming_tts.MingTTS``):

``setup()`` (once per process)
    1. downloads the Ming-3.0-tts source at a pinned commit from GitHub (a
       small tarball; ``ming_tts.py`` and its modules are not on PyPI) and puts
       it on ``sys.path``,
    2. downloads the checkpoint (about 3.0 GB) at a pinned revision,
    3. builds ``MingTTS`` on the GPU (bf16) and runs one short warm-up.

``process(job)`` (per request)
    reads the text to say and the task options, plus an optional reference
    recording for cloning, runs ``MingTTS.synthesize`` and returns
    ``{"audio_b64", "format", "sample_rate", "seconds", "timings", ...}``
    (16-bit PCM WAV, 44.1 kHz mono).

Request shapes accepted (options can sit at the top level or in ``extra_body``):

* OpenAI chat (``/v1/chat/completions``): the last user message carries the
  text to say and, for cloning, the reference audio as an ``input_audio``
  part ``{"data": <base64 wav>, "format": "wav"}``.
* Plain fields: ``text`` and ``audio_b64``.
* OpenAI TTS body in ``audio_body``: ``input`` is the text, ``voice`` a
  built-in voice name (``UK_George``, ``US_Emma``...), ``instructions`` a
  free-form delivery instruction.

Options: ``task`` (``TTS``, ``FREEFORM``, ``VOICE_DESIGN``, ``MULTILINGUAL``,
``IP``, ``EMOTION``, ``BASIC``, ``DIALECT``; picked from the other fields when
missing), ``language_id`` (``en``, ``es``, ``de``, ``fr``, ``ja``, ``ko``,
``zh``), ``instruction`` (FREEFORM), ``style`` (VOICE_DESIGN), ``voice`` / ``ip``
(IP), ``emotion``, ``speed`` / ``pitch`` / ``volume``, ``dialect``,
``prompt_text`` (transcript of the reference audio: required with it),
``seed``, ``cfg``, ``sigma``, ``max_length``, ``max_decode_steps``.

Settings are class attributes; override them with ``MING_*`` environment
variables or the runtime's ``--config-json`` (keys in lower case).
``MING_VRAM_LIMIT_GIB`` caps the memory PyTorch may use on the GPU (to check
that the model fits a smaller card).
"""

from __future__ import annotations

import base64
import io
import os
import random
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

MING_REPO = "inclusionAI/Ming-3.0-tts"
TASKS = ("TTS", "FREEFORM", "VOICE_DESIGN", "MULTILINGUAL", "IP", "EMOTION", "BASIC", "DIALECT")
NON_ZH_EN = {"es", "spanish", "de", "german", "fr", "french", "ja", "japanese", "ko", "korean"}


def _log(msg: str) -> None:
    print(f"[ming-tts-backend] {msg}", file=sys.stderr, flush=True)


def _wav_bytes(samples, sample_rate: int) -> bytes:
    """float tensor in [-1, 1] -> 16-bit PCM mono WAV bytes."""
    import numpy as np

    arr = samples.detach().cpu().float().numpy() if hasattr(samples, "detach") else np.asarray(samples)
    arr = arr.reshape(-1)
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class MingTtsBackend(CustomBackend):
    name = "ming-3-tts-05b"
    version = "0.1.0"
    task_type = "text2audio"
    description = "Ming-3.0-tts-0.5B (inclusionAI, MIT): cloning, voice design, free-form control, 14 paralinguistic tags"

    # Pinned upstream: code commit and Hugging Face revision this example was tested with.
    ming_commit = "9332fb9443325d4726ce02cefd4ec1c211df4079"
    hf_repo = "inclusionAI/Ming-3.0-tts-0.5B"
    hf_revision = "7247f4c94a5b2e987404ed28a5cfec41b1a4ca61"
    # FST text normalization of numbers, dates and units (needs pynini).
    text_normalization = True
    # 0 = no cap. Otherwise the GiB PyTorch may allocate on the GPU.
    vram_limit_gib = 0.0
    # Input guard: one request synthesizes at most this many characters.
    max_chars = 600

    # ------------------------------------------------------------------ setup

    def _setting(self, ctx: BackendContext, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"MING_{key.upper()}")
        if raw is None:
            raw = (ctx.config or {}).get(key)
        if raw is None:
            return default
        if isinstance(default, bool):
            return str(raw).lower() in ("1", "true", "yes", "on")
        return type(default)(raw)

    def _fetch_source(self, cache: Path) -> Path:
        src = cache / f"ming-3.0-tts-{self.ming_commit[:12]}"
        if (src / "ming_tts.py").is_file():
            return src
        url = f"https://codeload.github.com/{MING_REPO}/tar.gz/{self.ming_commit}"
        cache.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=cache, prefix=".src-"))
        req = urllib.request.Request(url, headers={"User-Agent": "inferencekey-ming-tts-backend"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = resp.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            # The 0.5B runtime only: skip the tiny model's package, figures and docs.
            members = [m for m in tf.getmembers()
                       if not any(f"/{d}/" in m.name + "/" for d in ("ming_tiny", "figures", "docker", "cookbooks"))]
            try:
                tf.extractall(tmp, members=members, filter="data")
            except TypeError:  # Python without extraction filters
                tf.extractall(tmp, members=members)
        inner = next(p for p in tmp.iterdir() if p.is_dir())
        if src.exists():
            shutil.rmtree(src)
        inner.rename(src)
        tmp.rmdir()
        return src

    def setup(self, ctx: BackendContext) -> None:
        for key in ("ming_commit", "hf_repo", "hf_revision", "text_normalization", "vram_limit_gib", "max_chars"):
            setattr(self, key, self._setting(ctx, key))
        cache = Path(os.environ.get("MING_CACHE_DIR", Path.home() / ".cache" / "ming-tts-backend"))
        self.timings: Dict[str, float] = {}
        t_setup = time.time()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        import torch
        from huggingface_hub import snapshot_download

        if torch.cuda.is_available() and self.vram_limit_gib > 0:
            total = torch.cuda.get_device_properties(0).total_memory
            frac = min(1.0, self.vram_limit_gib * 2**30 / total)
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            _log(f"VRAM capped at {self.vram_limit_gib} GiB (fraction {frac:.3f} of {total / 2**30:.1f} GiB)")

        t0 = time.time()
        src = self._fetch_source(cache)
        self.timings["source_s"] = round(time.time() - t0, 1)
        t0 = time.time()
        model_dir = snapshot_download(
            self.hf_repo, revision=self.hf_revision,
            ignore_patterns=["*.pdf", "figures/*", "*.md", ".gitattributes"],
        )
        self.timings["download_s"] = round(time.time() - t0, 1)
        _log(f"source in {self.timings['source_s']} s, weights in {self.timings['download_s']} s")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        from ming_tts import MingTTS

        t_load = time.time()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.engine = MingTTS(model_dir, device=device, text_normalization=bool(self.text_normalization))
        self.sample_rate = int(self.engine.sample_rate)
        self.timings["load_s"] = round(time.time() - t_load, 1)
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        self.timings["vram_loaded_mib"] = round(torch.cuda.memory_allocated() / 2**20) if torch.cuda.is_available() else 0
        self._lock = threading.Lock()
        # Warm-up: the first call pays for kernels, the FST normalizer and the speaker encoder session.
        t0 = time.time()
        with self._lock:
            self.engine.synthesize(task="IP", text="Ready.", controls={"ip": "US_Emma"}, use_zero_spk_emb=True)
        self.timings["warmup_s"] = round(time.time() - t0, 1)
        self.timings["setup_s"] = round(time.time() - t_setup, 1)
        _log(f"Ming-3.0-tts-0.5B ready on {self.device_name}, {self.sample_rate} Hz: {self.timings}")

    # ---------------------------------------------------------------- inputs

    @staticmethod
    def _merged(inp: Dict[str, Any]) -> Dict[str, Any]:
        """Top-level fields plus ``extra_body`` (the platform carries non-OpenAI fields there)."""
        merged = dict(inp)
        extra = inp.get("extra_body")
        if isinstance(extra, dict):
            merged.update(extra)
        return merged

    @staticmethod
    def _from_messages(messages: Any) -> Tuple[str, Optional[str]]:
        """Text and base64 audio from the last user message."""
        if not isinstance(messages, list):
            return "", None
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content, None
            texts: List[str] = []
            audio: Optional[str] = None
            for part in content or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
                elif part.get("type") == "input_audio":
                    audio = (part.get("input_audio") or {}).get("data")
                elif part.get("type") == "audio_url":
                    url = (part.get("audio_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        audio = url.split(",", 1)[-1]
            return " ".join(t for t in texts if t).strip(), audio
        return "", None

    def _parse(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        f = self._merged(inp)
        text, audio_b64 = self._from_messages(f.get("messages"))
        speech = f.get("audio_body") if isinstance(f.get("audio_body"), dict) else None
        if speech:  # OpenAI /v1/audio/speech body
            f = {**speech, **{k: v for k, v in f.items() if k != "audio_body"}}
            text = str(speech.get("input") or text or "")
            if speech.get("instructions") and not f.get("instruction"):
                f["instruction"] = speech["instructions"]
            audio_b64 = speech.get("ref_audio") or audio_b64
        text = str(f.get("text") or text or "").strip()
        audio_b64 = f.get("audio_b64") or audio_b64
        if isinstance(audio_b64, str) and audio_b64.startswith("data:"):
            audio_b64 = audio_b64.split(",", 1)[-1]
        if not text:
            raise InvalidInput("expected the text to say: a user message (chat), 'text' or an audio_body 'input'")
        if len(text) > int(self.max_chars):
            raise InvalidInput(f"text has {len(text)} characters; this backend caps one request at {self.max_chars}")

        voice = f.get("ip") or f.get("voice")
        if voice in ("alloy", "default", None, ""):  # OpenAI voice names mean "no preference"
            voice = None
        task = str(f.get("task") or "").strip().upper().replace("-", "_")
        if not task:
            if f.get("instruction"):
                task = "FREEFORM"
            elif f.get("style"):
                task = "VOICE_DESIGN"
            elif voice and not audio_b64:
                task = "IP"
            elif f.get("emotion"):
                task = "EMOTION"
            elif f.get("dialect"):
                task = "DIALECT"
            elif any(f.get(k) for k in ("speed", "pitch", "volume")):
                task = "BASIC"
            elif str(f.get("language_id") or "").lower() in NON_ZH_EN:
                task = "MULTILINGUAL"
            else:
                task = "TTS"
        if task not in TASKS:
            raise InvalidInput(f"unknown task {task!r}; expected one of {', '.join(TASKS)}")

        controls: Optional[Dict[str, str]] = None
        if task == "VOICE_DESIGN":
            if not f.get("style"):
                raise InvalidInput("VOICE_DESIGN needs 'style': a description of the voice")
            controls = {"style": str(f["style"])}
        elif task == "IP":
            if not voice:
                raise InvalidInput("IP needs 'voice' (or 'ip'): a built-in voice such as UK_George or US_Emma")
            controls = {"ip": str(voice)}
        elif task == "EMOTION":
            controls = {"emotion": str(f.get("emotion"))}
        elif task == "DIALECT":
            controls = {"dialect": str(f.get("dialect"))}
        elif task == "BASIC":
            # The model was trained with all three attributes set; default the missing ones to medium.
            controls = {"speed": str(f.get("speed") or "中速"), "pitch": str(f.get("pitch") or "中"),
                        "volume": str(f.get("volume") or "中")}
        if audio_b64 and not f.get("prompt_text") and task in ("TTS", "MULTILINGUAL"):
            raise InvalidInput("cloning needs 'prompt_text': the transcript of the reference audio")
        return {"task": task, "text": text, "audio_b64": audio_b64, "controls": controls, "fields": f}

    # ---------------------------------------------------------------- process

    def process(self, job: Job) -> Result:
        import numpy as np
        import torch

        req = self._parse(job.input)
        f = req["fields"]
        t0 = time.time()
        tmpdir = tempfile.mkdtemp(prefix="ming-")
        try:
            prompt_path = None
            if req["audio_b64"]:
                try:
                    raw = base64.b64decode(req["audio_b64"], validate=False)
                except Exception as e:  # noqa: BLE001
                    raise InvalidInput(f"audio is not valid base64: {e}") from e
                prompt_path = os.path.join(tmpdir, "prompt.wav")
                with open(prompt_path, "wb") as fh:
                    fh.write(raw)
                try:
                    import soundfile as sf

                    info = sf.info(prompt_path)
                    ref_s = info.frames / float(info.samplerate)
                except Exception as e:  # noqa: BLE001
                    raise InvalidInput(f"could not read the reference audio (send WAV or FLAC): {e}") from e
                if ref_s > 30:
                    raise InvalidInput(f"reference audio is {ref_s:.1f} s; keep it under 30 s (3 to 15 s works best)")
            kwargs: Dict[str, Any] = {
                "task": req["task"],
                "text": req["text"],
                "language_id": f.get("language_id"),
                "controls": req["controls"],
                "cfg": float(f.get("cfg", 2.0)),
                "sigma": float(f.get("sigma", 0.25)),
                "max_length": int(f.get("max_length", 24)),
                "max_decode_steps": int(f.get("max_decode_steps", 1000)),
            }
            if req["task"] == "FREEFORM":
                kwargs["instruction"] = str(f.get("instruction") or "")
            if prompt_path:
                # A reference fixes the voice; its transcript lets the model continue it as a prompt.
                kwargs.update(prompt_wav_path=prompt_path, prompt_text=f.get("prompt_text"),
                              use_spk_emb=bool(f.get("use_spk_emb", True)))
            else:
                # No reference: the authors' examples pass a zero speaker embedding.
                kwargs["use_zero_spk_emb"] = bool(f.get("use_zero_spk_emb", True))
            seed = f.get("seed")
            with self._lock:
                if seed is not None:
                    random.seed(int(seed))
                    np.random.seed(int(seed))
                    torch.manual_seed(int(seed))
                    torch.cuda.manual_seed_all(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                t_gpu = time.time()
                try:
                    waveform = self.engine.synthesize(**kwargs)
                except ValueError as e:  # the authors' task contract rejected the request
                    raise InvalidInput(str(e)) from e
                finally:
                    # The reference cache is keyed by file path, and our temp paths are unique.
                    self.engine._prompt_cache.clear()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gpu_s = time.time() - t_gpu
                peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
            wav = _wav_bytes(waveform, self.sample_rate)
            out_s = waveform.shape[-1] / float(self.sample_rate)
            return Result(output={
                "audio_b64": base64.b64encode(wav).decode(),
                "format": "wav",
                "sample_rate": self.sample_rate,
                "seconds": round(out_s, 2),
                "task": req["task"],
                "text": req["text"],
                "timings": {
                    "gpu_s": round(gpu_s, 3),
                    "total_s": round(time.time() - t0, 3),
                    "realtime_factor": round(out_s / gpu_s, 2) if gpu_s > 0 else None,
                    "peak_vram_mib": round(peak),
                },
                "model": self.hf_repo,
                "device": self.device_name,
            })
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

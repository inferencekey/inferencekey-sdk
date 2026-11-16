"""Sophon-Realtime-4B-Flash (Adventists AI), full-duplex speech, served by vLLM-Omni, as an InferenceKey custom backend.

Sophon-Realtime-4B-Flash is an Apache-2.0 full-duplex voice model: it listens
while the user talks, decides from the voice when the user has finished,
writes a reply with sound tags (``[laughter]``, ``[sigh]``, ``<strong>``) and
speaks it while it writes. It ships as one vLLM-Omni model in three stages on
one GPU: a 4B "thinker" (bf16) that listens and writes, a talker that turns the
tagged text into codec frames (one forward pass per 80 ms frame) and a codec
decoder (24 kHz). The authors serve it with ``flash/install.sh`` and
``flash/start.sh`` from github.com/adventists-ai/sophon-realtime; the server
answers an OpenAI Realtime-style WebSocket (``/v1/realtime``) and
``/v1/chat/completions``.

The platform forwards HTTP requests to a backend, not WebSockets, so this
backend runs the authors' server next to it and plays each request into the
local WebSocket as a live caller would:

``setup()`` (once per process)
    1. downloads the repository at a pinned commit and, in parallel, the
       weights at a pinned Hugging Face revision (13 GB) to the local disk,
    2. runs the authors' ``flash/install.sh`` (Python 3.12 venv, vLLM 0.30,
       vLLM-Omni 0.30, the plugin),
    3. runs the authors' ``flash/start.sh`` (it picks the 4B config for one
       24 GB card, waits for ``/health`` and the warm-up) on a loopback port.

``process(job)`` (per request; up to ``max_sessions`` run at once)
    ``mode: "realtime"`` (default): opens a ``/v1/realtime`` session, streams
    the caller's audio in 80 ms frames at speaking speed, then keeps streaming
    silence, exactly like an open microphone. The model decides on its own when
    the caller has finished (no commit is sent). Returns the reply audio, the
    reply text with its sound tags, the time from the end of the caller's
    speech to the first reply audio, and a two-channel "conversation" WAV
    (caller left, model right, on one clock).
    ``mode: "chat"``: one ``/v1/chat/completions`` request (whole clip in,
    streamed text and audio out), the authors' ``client_example.py`` path.

Request shapes (options at the top level or in ``extra_body``):

* OpenAI chat: the last user message carries an ``input_audio`` part
  ``{"data": <base64 WAV>, "format": "wav"}`` (the caller's speech).
* Plain fields: ``audio_b64``.

Options: ``mode`` (realtime | chat), ``voice`` (``chelsie`` default, ``brand``, or
a cloned voice name), ``language`` (``en`` default, ``zh``), ``voice_clip_b64``
(3 to 10 s of one speaker, saved as ``voice`` for zero-shot cloning; only clone
your own voice or one you have permission to use), ``pace`` (1.0 = speaking
speed), ``max_wait_s`` (how long to keep the microphone open after the caller
stops, default 25), ``max_tokens``, ``temperature``.

Settings are class attributes; override them with ``SOPHON_*`` environment
variables or the workload config (keys in lower case). ``stage_gpu_memory``
("0.55,0.2,0.07" in the 4B config) sets each stage's share of the card;
``thinker_quantization`` ("fp8") loads the thinker in fp8 on cards that support
it; both are for squeezing the model onto a smaller card (see README).
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

IN_RATE = 16000  # what the server's VAD and thinker listen at
FRAME_S = 0.08  # 80 ms microphone frames, like the authors' warm-up client
TAG_RE = re.compile(r"\[(laughter|cough|sigh|breath)\]|</?(laughter|strong)>")


def _log(msg: str) -> None:
    print(f"[sophon-backend] {msg}", file=sys.stderr, flush=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _read_wav(raw: bytes) -> Tuple[np.ndarray, int]:
    """WAV bytes -> mono float32 in [-1, 1] and its sample rate (16/24/32-bit PCM or 32-bit float)."""
    try:
        with wave.open(io.BytesIO(raw)) as w:
            rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            data = w.readframes(w.getnframes())
        if width == 2:
            x = np.frombuffer(data, "<i2").astype(np.float32) / 32768.0
        elif width == 4:
            x = np.frombuffer(data, "<i4").astype(np.float32) / 2147483648.0
        elif width == 1:
            x = (np.frombuffer(data, np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise InvalidInput(f"unsupported WAV sample width {width}")
    except wave.Error:
        import soundfile as sf  # IEEE float WAV, FLAC, OGG

        x, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        ch = x.shape[1]
        x = x.reshape(-1)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), int(rate)


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x
    n = int(round(x.size * dst / src))
    return np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)


def _wav_bytes(x: np.ndarray, rate: int, channels: int = 1) -> bytes:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _gpu_used_mib() -> Optional[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


class SophonRealtimeFlashBackend(CustomBackend):
    name = "sophon-realtime-4b-flash"
    version = "0.1.0"
    task_type = "text2audio"
    description = "Sophon-Realtime-4B-Flash (Apache-2.0): full-duplex voice with sound tags, served by vLLM-Omni"

    # Pinned upstream: the repository commit and the Hugging Face revision this example was tested with.
    repo = "adventists-ai/sophon-realtime"
    repo_commit = "2c48aa543fe6a123468933819000f368e74454f5"
    hf_repo = "adventists-ai/Sophon-Realtime-4B-Flash"
    hf_revision = "ae4a8591d0ae4c3b3da0250a613828bc2435963b"
    # Empty = the config start.sh picks (sophon_realtime_flash_4b.yaml for the 4B model).
    config_name = ""
    # Squeezing knobs (empty = as the authors ship it): per-stage share of the card, comma separated
    # (thinker, talker, decoder), the thinker's context, the thinker's quantization (fp8), sessions at once.
    stage_gpu_memory = ""
    thinker_max_model_len = 0
    thinker_quantization = ""
    max_sessions = 0
    # 1 = let start.sh run the stages under CUDA MPS when the image has it (the authors' default).
    mps = True
    server_start_timeout_s = 2700
    max_input_seconds = 60.0

    # ------------------------------------------------------------------ setup

    def _setting(self, ctx: BackendContext, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"SOPHON_{key.upper()}")
        if raw is None:
            raw = (ctx.config or {}).get(key)
        if raw is None:
            return default
        if isinstance(default, bool):
            return str(raw).lower() in ("1", "true", "yes", "on")
        return type(default)(raw)

    def _fetch_repo(self, cache: Path) -> Path:
        root = cache / f"sophon-realtime-{self.repo_commit[:12]}"
        if (root / "flash" / "start.sh").is_file():
            return root
        url = f"https://github.com/{self.repo}/archive/{self.repo_commit}.tar.gz"
        tmp = cache / f"repo-{self.repo_commit[:12]}.tar.gz"
        urllib.request.urlretrieve(url, tmp)
        with tarfile.open(tmp) as t:
            top = t.getmembers()[0].name.split("/")[0]
            t.extractall(cache, filter="data")
        shutil.rmtree(root, ignore_errors=True)
        (cache / top).rename(root)
        tmp.unlink()
        return root

    def _download(self, cache: Path) -> Path:
        from huggingface_hub import snapshot_download

        local = cache / f"weights-{self.hf_revision[:12]}"
        snapshot_download(self.hf_repo, revision=self.hf_revision, local_dir=str(local), max_workers=16)
        return local

    def _install(self, flash: Path) -> None:
        marker = flash / ".venv" / ".ik-install-complete"
        if marker.is_file():
            return
        log = flash / "install.log"
        r = subprocess.run(["bash", "install.sh"], cwd=str(flash), stdout=log.open("ab"), stderr=subprocess.STDOUT,
                           env={k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")})
        if r.returncode != 0:
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-30:])
            raise RuntimeError(f"install.sh failed ({r.returncode}):\n{tail}")
        marker.write_text("ok\n")

    def _write_config(self, flash: Path, weights: Path) -> str:
        """The config start.sh would pick, with the squeezing knobs applied; returns its file name."""
        import yaml

        name = self.config_name
        if not name:
            is4b = '"hidden_size": 2560' in (weights / "thinker" / "config.json").read_text()
            name = "sophon_realtime_flash_4b.yaml" if is4b else "sophon_realtime_flash_1gpu.yaml"
        if not (self.stage_gpu_memory or self.thinker_max_model_len or self.thinker_quantization or self.max_sessions):
            return name
        cfg = yaml.safe_load((flash / "config" / name).read_text())
        if self.stage_gpu_memory:
            for stage, share in zip(cfg["stages"], self.stage_gpu_memory.split(",")):
                stage["gpu_memory_utilization"] = float(share)
        if self.thinker_max_model_len:
            cfg["stages"][0]["max_model_len"] = int(self.thinker_max_model_len)
        if self.thinker_quantization:
            cfg["stages"][0]["quantization"] = self.thinker_quantization
        if self.max_sessions:
            cfg["duplex_session"]["max_sessions"] = int(self.max_sessions)
            for stage in cfg["stages"]:
                stage["max_num_seqs"] = min(int(stage["max_num_seqs"]), max(2, 2 * int(self.max_sessions)))
        out = "ik_" + name
        (flash / "config" / out).write_text(yaml.safe_dump(cfg, sort_keys=False))
        return out

    def setup(self, ctx: BackendContext) -> None:
        for key in ("repo_commit", "hf_repo", "hf_revision", "config_name", "stage_gpu_memory", "thinker_max_model_len",
                    "thinker_quantization", "max_sessions", "mps", "server_start_timeout_s", "max_input_seconds"):
            setattr(self, key, self._setting(ctx, key))
        # Keep everything on the local disk: weights on a network volume load many times slower.
        cache = Path(os.environ.get("SOPHON_CACHE_DIR", Path.home() / ".cache" / "sophon-backend"))
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        self.timings: Dict[str, Any] = {}
        t_setup = time.time()

        result: Dict[str, Any] = {}

        def download() -> None:
            t = time.time()
            try:
                result["weights"] = self._download(cache)
            except Exception as e:  # noqa: BLE001
                result["download_error"] = e
            result["download_s"] = round(time.time() - t, 1)

        dl = threading.Thread(target=download, daemon=True)
        dl.start()
        t0 = time.time()
        root = self._fetch_repo(cache)
        self.flash = root / "flash"
        self._install(self.flash)
        self.timings["install_s"] = round(time.time() - t0, 1)
        dl.join()
        if "download_error" in result:
            raise RuntimeError(f"weights download failed: {result['download_error']}")
        weights: Path = result["weights"]
        self.timings["download_s"] = result["download_s"]
        self.weights = weights
        self.voices_dir = weights / "voices"
        model_link = self.flash / "model"
        if model_link.is_symlink() or model_link.exists():
            model_link.unlink()
        model_link.symlink_to(weights)
        self.system_prompt = json.loads((self.flash / "config" / "system_prompt.json").read_text())
        _log(f"install in {self.timings['install_s']} s, weights in {self.timings['download_s']} s ({weights})")

        self.port = _free_port()
        config = self._write_config(self.flash, weights)
        self.config_used = config
        env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
        env.update({"GPU": "0", "PORT": str(self.port), "SOPHON_BRIDGE_PORT": str(_free_port()), "CONFIG": config,
                    "MPS": "1" if self.mps else "0"})
        _log(f"start.sh PORT={self.port} CONFIG={config} MPS={env['MPS']}")
        t0 = time.time()
        # Watchdog: if this backend process dies (even with SIGKILL), stop the server it started.
        subprocess.Popen(["bash", "-c", f"while kill -0 {os.getpid()} 2>/dev/null; do sleep 2; done; "
                          f"cd {self.flash} && PORT={self.port} bash stop.sh"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        atexit.register(self.teardown)
        previous = signal.getsignal(signal.SIGTERM)

        def on_term(signum, frame):  # the worker stops the backend with SIGTERM
            self.teardown()
            if callable(previous):
                previous(signum, frame)
            else:
                raise SystemExit(0)

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, on_term)
        self.peak_mib = 0
        threading.Thread(target=self._sample_gpu, daemon=True).start()
        r = subprocess.run(["bash", "start.sh"], cwd=str(self.flash), env=env, capture_output=True, text=True,
                           timeout=self.server_start_timeout_s)
        if r.returncode != 0 or "READY" not in r.stdout:
            raise RuntimeError(f"start.sh failed ({r.returncode}):\n{r.stdout[-3000:]}\n{self._log_tail()}")
        self.timings["server_start_s"] = round(time.time() - t0, 1)
        self.timings["gpu_used_mib_after_start"] = _gpu_used_mib()
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=10) as resp:
            self.model_id = json.load(resp)["data"][0]["id"]
        self.timings["setup_s"] = round(time.time() - t_setup, 1)
        _log(f"Sophon-Realtime ready on 127.0.0.1:{self.port}: {self.timings}")

    def _sample_gpu(self) -> None:
        while True:
            used = _gpu_used_mib()
            if used is not None and used > self.peak_mib:
                self.peak_mib = used
            time.sleep(0.5)

    def _log_tail(self, n: int = 40) -> str:
        try:
            log = self.flash / "logs" / f"server_{self.port}.log"
            return "\n".join(log.read_text(errors="replace").splitlines()[-n:])
        except (OSError, AttributeError):
            return ""

    def teardown(self) -> None:
        """Stop the vLLM-Omni server and its stage processes with the authors' stop.sh."""
        if getattr(self, "_stopped", False) or not hasattr(self, "port"):
            return
        self._stopped = True
        subprocess.run(["bash", "stop.sh"], cwd=str(self.flash), env={**os.environ, "PORT": str(self.port)},
                       capture_output=True, timeout=60)

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
                if isinstance(part, dict) and part.get("type") == "audio_url":
                    return (part.get("audio_url") or {}).get("url")
        return None

    def _parse(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        f: Dict[str, Any] = dict(inp)
        if isinstance(f.get("extra_body"), dict):
            f = {**f, **f["extra_body"]}
        audio = f.get("audio_b64") or self._audio_from_messages(f.get("messages"))
        if not isinstance(audio, str) or not audio:
            raise InvalidInput("expected the caller's speech: a user message with an input_audio part, or audio_b64")
        if audio.startswith("data:"):
            audio = audio.split(",", 1)[-1]
        x, rate = _read_wav(base64.b64decode(audio))
        if x.size / rate > self.max_input_seconds:
            raise InvalidInput(f"audio is {x.size / rate:.1f} s; the limit is {self.max_input_seconds} s")
        f["speech"] = _resample(x, rate, IN_RATE)
        f["mode"] = str(f.get("mode") or "realtime").lower()
        if f["mode"] not in ("realtime", "chat"):
            raise InvalidInput("mode must be realtime or chat")
        f["language"] = str(f.get("language") or "en")[:2]
        voice = str(f.get("voice") or "chelsie").strip().lower()
        if not re.fullmatch(r"[a-z0-9_.-]{1,40}", voice):
            raise InvalidInput("voice names are lower case: a-z 0-9 _ - .")
        f["voice"] = voice
        if f.get("voice_clip_b64"):
            self._save_voice(voice, str(f["voice_clip_b64"]))
        return f

    def _save_voice(self, voice: str, clip_b64: str) -> None:
        """Zero-shot cloning: the server reads voices/<name>.wav on the next request that uses the name."""
        if voice in ("chelsie", "brand"):
            raise InvalidInput("pick a new name for a cloned voice")
        if clip_b64.startswith("data:"):
            clip_b64 = clip_b64.split(",", 1)[-1]
        x, rate = _read_wav(base64.b64decode(clip_b64))
        if not 1.0 <= x.size / rate <= 30.0:
            raise InvalidInput("the voice clip should be 3 to 10 s of one speaker")
        data = _wav_bytes(x, rate)
        path = self.voices_dir / f"{voice}.wav"
        if not path.is_file() or path.read_bytes() != data:
            path.write_bytes(data)

    # -------------------------------------------------------------- realtime

    async def _realtime(self, f: Dict[str, Any]) -> Dict[str, Any]:
        import websockets

        speech: np.ndarray = f["speech"]
        pace = float(f.get("pace") or 1.0)
        max_wait = float(f.get("max_wait_s") or 25.0)
        n = int(IN_RATE * FRAME_S)
        events: List[Tuple[float, str]] = []
        audio: List[Tuple[float, bytes]] = []  # (arrival time, PCM16)
        responses: List[Dict[str, Any]] = []
        out_rate = [24000]
        text: List[str] = []
        state: Dict[str, Any] = {"first_audio": None, "first_text": None, "created": None, "done": None,
                                 "speech_started": None, "speech_stopped": None, "error": None, "mic_done": False}
        url = f"ws://127.0.0.1:{self.port}/v1/realtime"
        t0 = time.monotonic()
        speech_end = len(speech) / IN_RATE / pace
        async with websockets.connect(url, max_size=None) as ws:
            await ws.send(json.dumps({"type": "session.update", "session": {
                "modalities": ["audio", "text"], "voice": f["voice"],
                "input_audio_format": "pcm_f32le", "output_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad"},
                "extra_body": {"auto_response": True, "language": f["language"]}}}))
            done = asyncio.Event()

            async def reader() -> None:
                async for m in ws:
                    try:
                        ev = json.loads(m)
                    except ValueError:
                        continue
                    t = time.monotonic() - t0
                    typ = str(ev.get("type", ""))
                    if typ != "response.output_audio.delta":
                        events.append((round(t, 3), typ))
                    if typ == "response.output_audio.delta":
                        chunk = ev.get("delta") or ev.get("audio")
                        if isinstance(chunk, str) and chunk:
                            audio.append((t, base64.b64decode(chunk)))
                            state["first_audio"] = state["first_audio"] or t
                            if responses and responses[-1].get("first_audio_s") is None:
                                responses[-1]["first_audio_s"] = round(t, 3)
                            if isinstance(ev.get("sample_rate_hz"), int):
                                out_rate[0] = ev["sample_rate_hz"]
                    elif typ in ("response.output_audio_transcript.delta", "response.text.delta",
                                 "response.output_text.delta"):
                        d = ev.get("delta")
                        if isinstance(d, str) and d:
                            text.append(d)
                            state["first_text"] = state["first_text"] or t
                    elif typ == "response.created":
                        state["created"] = state["created"] or t
                        responses.append({"created_s": round(t, 3), "first_audio_s": None, "done_s": None})
                    elif typ == "input_audio_buffer.speech_started":
                        state["speech_started"] = state["speech_started"] or t
                    elif typ == "input_audio_buffer.speech_stopped":
                        state["speech_stopped"] = t
                    elif typ == "error":
                        state["error"] = ev.get("error") or ev
                    elif typ == "response.done":
                        if responses:
                            responses[-1]["done_s"] = round(t, 3)
                        # A reply (or a back-channel) while the caller is still talking does not end the turn.
                        if audio and state["mic_done"]:
                            state["done"] = t
                            done.set()
                            return

            task = asyncio.ensure_future(reader())
            mic = np.concatenate([np.zeros(int(IN_RATE * 0.3), np.float32), speech]).astype("<f4")
            speech_end += 0.3 / pace
            k = 0
            t_send = time.monotonic()
            while not done.is_set():
                if k < mic.size:
                    frame = mic[k:k + n]
                else:
                    state["mic_done"] = True  # open microphone: silence until the model has answered
                    if time.monotonic() - t0 > speech_end + max_wait:
                        break
                    frame = np.zeros(n, "<f4")
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": base64.b64encode(frame.tobytes()).decode(),
                                          "format": "pcm_f32le", "sample_rate_hz": IN_RATE}))
                k += n
                t_send += FRAME_S / pace
                await asyncio.sleep(max(0.0, t_send - time.monotonic()))
            # Keep listening a little after response.done so the tail of the audio arrives.
            await asyncio.sleep(0.3)
            task.cancel()
            try:
                await ws.send(json.dumps({"type": "session.close"}))
            except Exception:  # noqa: BLE001
                pass
        rate = out_rate[0]
        parts = [np.frombuffer(b, "<i2").astype(np.float32) / 32768.0 for _, b in audio]
        pcm = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        # Two-channel conversation on one clock: caller (left) and model (right), at the model's rate. Each
        # chunk plays when it arrives, or right after the previous one, as a client's player would.
        conv = None
        if parts:
            caller = _resample(mic, IN_RATE, rate)
            placed, cursor = [], 0
            for (t, _), x in zip(audio, parts):
                start = max(cursor, int(t * pace * rate))
                placed.append((start, x))
                cursor = start + x.size
            length = max(caller.size, cursor)
            left = np.zeros(length, np.float32)
            left[:caller.size] = caller
            right = np.zeros(length, np.float32)
            for start, x in placed:
                right[start:start + x.size] = x
            conv = _wav_bytes(np.stack([left, right], axis=1).reshape(-1), rate, channels=2)
        reply = "".join(text)
        return {
            "reply_wav": _wav_bytes(pcm, rate) if pcm.size else None, "conversation_wav": conv, "rate": rate,
            "text": reply, "seconds": round(pcm.size / rate, 2),
            "timings": {
                "speech_end_s": round(speech_end, 3),
                "first_audio_s": round(state["first_audio"], 3) if state["first_audio"] else None,
                "first_text_s": round(state["first_text"], 3) if state["first_text"] else None,
                "response_created_s": state["created"],
                "turn_latency_s": round(state["first_audio"] - speech_end, 3) if state["first_audio"] else None,
                "response_done_s": state["done"],
            },
            "replied_before_caller_finished": bool(state["first_audio"] is not None and state["first_audio"] < speech_end),
            "responses": responses, "events": events, "error": state["error"],
        }

    # ------------------------------------------------------------------ chat

    def _chat(self, f: Dict[str, Any]) -> Dict[str, Any]:
        import requests

        clip = base64.b64encode(_wav_bytes(f["speech"], IN_RATE)).decode()
        lang = f["language"]
        body = {"model": self.model_id, "stream": True, "modalities": ["text", "audio"], "language": lang,
                "voice": f["voice"], "max_tokens": int(f.get("max_tokens") or 300),
                "temperature": float(f.get("temperature") or 0.7), "top_p": 0.9,
                "messages": [{"role": "system", "content": self.system_prompt.get(lang, self.system_prompt.get("en"))},
                             {"role": "user", "content": [{"type": "text", "text": "<|audio|>"},
                                                          {"type": "input_audio",
                                                           "input_audio": {"data": clip, "format": "wav"}}]}]}
        t0 = time.time()
        first_audio = first_text = None
        text: List[str] = []
        chunks: List[np.ndarray] = []
        rate = 24000
        with requests.post(f"http://127.0.0.1:{self.port}/v1/chat/completions", json=body, stream=True,
                           timeout=600) as r:
            if r.status_code >= 400:
                raise RuntimeError(f"server error {r.status_code}: {r.text[:500]}")
            for line in r.iter_lines():
                s = line.decode(errors="replace").strip()
                if not s.startswith("data:") or s[5:].strip() == "[DONE]":
                    continue
                d = json.loads(s[5:])
                for ch in d.get("choices", []):
                    c = (ch.get("delta") or {}).get("content")
                    if not c:
                        continue
                    if d.get("modality") == "audio":
                        x, rate = _read_wav(base64.b64decode(c))
                        chunks.append(x)
                        first_audio = first_audio or time.time() - t0
                    else:
                        text.append(c)
                        first_text = first_text or time.time() - t0
        pcm = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        total = time.time() - t0
        return {"reply_wav": _wav_bytes(pcm, rate) if pcm.size else None, "conversation_wav": None, "rate": rate,
                "text": "".join(text), "seconds": round(pcm.size / rate, 2),
                "timings": {"first_audio_s": round(first_audio, 3) if first_audio else None,
                            "first_text_s": round(first_text, 3) if first_text else None, "total_s": round(total, 3)}}

    # --------------------------------------------------------------- process

    def process(self, job: Job) -> Result:
        t0 = time.time()
        f = self._parse(job.input)
        out = asyncio.run(self._realtime(f)) if f["mode"] == "realtime" else self._chat(f)
        if out["reply_wav"] is None:
            raise RuntimeError(f"no reply audio within the wait; events: {out.get('events', [])[-12:]}; "
                               f"error: {out.get('error')}")
        reply = out["text"]
        result = {
            "audio_b64": base64.b64encode(out["reply_wav"]).decode(),
            "format": "wav",
            "sample_rate": out["rate"],
            "seconds": out["seconds"],
            "text": reply,
            "spoken_text": TAG_RE.sub("", reply).strip(),
            "sound_tags": [m.group(0) for m in TAG_RE.finditer(reply)],
            "mode": f["mode"],
            "voice": f["voice"],
            "language": f["language"],
            "timings": {**out["timings"], "process_s": round(time.time() - t0, 3)},
            "gpu_used_mib_peak": self.peak_mib,
            "model": self.hf_repo,
            "server": f"vllm-omni 0.30 + sophon-realtime@{self.repo_commit[:12]} ({self.config_used})",
        }
        if out.get("conversation_wav"):
            result["conversation_b64"] = base64.b64encode(out["conversation_wav"]).decode()
        if f["mode"] == "realtime":
            result["replied_before_caller_finished"] = out["replied_before_caller_finished"]
            result["responses"] = out["responses"]
            result["events"] = out["events"]
        return Result(output=result)
